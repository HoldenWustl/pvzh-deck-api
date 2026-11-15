"use strict";

/*
 * YouTube deck frames
 *
 * YouTube stores three real frames per video (about 25%, 50% and 75%
 * of the way through) as plain images on i.ytimg.com. For deck videos
 * that keep the in-game deck panel on screen (e.g. FryEmUp's layout),
 * one of those frames shows the whole deck. This module downloads the
 * frames and crops the deck panel so it looks like an ordinary deck
 * screenshot to split-deck.js / read-deck.js.
 *
 * Only used by /api/recognize-youtube. Normal uploads are never cropped.
 */

const sharp = require("sharp");

const VIDEO_ID_PATTERN = /^[A-Za-z0-9_-]{11}$/;

/*
 * Tried in order. maxres* are 1280x720 and exist for HD uploads;
 * sd* (640x480, letterboxed) are the fallback. Frame 2 is the
 * midpoint, which is the most likely to be mid-game.
 */
const FRAME_NAMES = [
    "maxres2",
    "maxres1",
    "maxres3",
    "sd2",
    "sd1",
    "sd3",
];

// YouTube answers missing sizes with a 120x90 grey placeholder.
const PLACEHOLDER_MAX_WIDTH = 130;

// Crops are scaled up to about phone-screenshot width so OCR has room.
const TARGET_CROP_WIDTH = 1000;

function isValidVideoId(videoId) {
    return VIDEO_ID_PATTERN.test(String(videoId || ""));
}

async function fetchFrame(videoId, frameName, { timeoutMs = 10000 } = {}) {
    const url = `https://i.ytimg.com/vi/${videoId}/${frameName}.jpg`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
        const response = await fetch(url, { signal: controller.signal });
        if (!response.ok) return null;

        const buffer = Buffer.from(await response.arrayBuffer());
        const metadata = await sharp(buffer).metadata();

        if (!metadata.width || metadata.width <= PLACEHOLDER_MAX_WIDTH) {
            return null;
        }

        return { name: frameName, url, buffer, width: metadata.width, height: metadata.height };
    } catch (error) {
        return null;
    } finally {
        clearTimeout(timer);
    }
}

/*
 * Same "cyan" rule split-deck.js uses for its divider.
 */
function isCyan(r, g, b) {
    return (
        g >= 105 &&
        b >= 105 &&
        g - r >= 25 &&
        b - r >= 25 &&
        Math.abs(g - b) <= 80
    );
}

/*
 * Longest run of cyan pixels in a row, tolerating tiny gaps from
 * JPEG compression.
 */
function longestCyanRun(data, width, channels, y, maxGap = 3) {
    let best = { start: 0, end: -1, length: 0 };
    let start = -1;
    let lastCyan = -1;

    for (let x = 0; x < width; x++) {
        const index = (y * width + x) * channels;
        const cyan = isCyan(data[index], data[index + 1], data[index + 2]);

        if (cyan) {
            if (start < 0 || x - lastCyan - 1 > maxGap) start = x;
            lastCyan = x;

            const length = lastCyan - start + 1;
            if (length > best.length) best = { start, end: lastCyan, length };
        }
    }

    return best;
}

/*
 * The deck panel's header ends in a long cyan divider that spans the
 * panel's full width. Find the lowest such divider; its run gives the
 * panel's left/right edges, and the header sits directly above it.
 */
async function findDeckPanel(frameBuffer) {
    const { data, info } = await sharp(frameBuffer)
        .removeAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });

    const { width, height, channels } = info;

    const minRun = width * 0.22;
    const maxRun = width * 0.62;
    const yStart = Math.floor(height * 0.12);
    const yEnd = Math.floor(height * 0.85);

    const bands = [];
    let current = null;

    for (let y = yStart; y < yEnd; y++) {
        const run = longestCyanRun(data, width, channels, y);
        const qualifies = run.length >= minRun && run.length <= maxRun;

        if (
            qualifies &&
            current &&
            y - current.bottom <= 2 &&
            Math.abs(run.start - current.start) <= 8 &&
            Math.abs(run.end - current.end) <= 8
        ) {
            current.bottom = y;
            continue;
        }

        if (current) bands.push(current);
        current = qualifies
            ? { top: y, bottom: y, start: run.start, end: run.end }
            : null;
    }

    if (current) bands.push(current);

    const dividers = bands.filter(band => {
        const thickness = band.bottom - band.top + 1;
        return thickness >= 1 && thickness <= Math.max(14, height * 0.03);
    });

    if (!dividers.length) return null;

    const divider = dividers[dividers.length - 1];
    const panelLeft = divider.start;
    const panelRight = divider.end + 1;
    const panelWidth = panelRight - panelLeft;

    // Header height is about 12% of the panel width; keep a little margin.
    const top = Math.max(0, Math.round(divider.top - panelWidth * 0.14));

    // Stop at the end of the picture (sd frames are letterboxed in black).
    let bottom = height;
    for (let y = height - 1; y > divider.bottom; y--) {
        let sum = 0;
        let count = 0;
        for (let x = panelLeft; x < panelRight; x += 4) {
            const index = (y * width + x) * channels;
            sum += data[index] + data[index + 1] + data[index + 2];
            count++;
        }
        if (sum / (count * 3) > 14) {
            bottom = y + 1;
            break;
        }
    }

    const cropHeight = bottom - top;
    if (panelWidth < 120 || cropHeight < panelWidth * 0.5) return null;

    return {
        left: panelLeft,
        top,
        width: panelWidth,
        height: cropHeight,
        dividerY: divider.top,
    };
}


/*
 * Streamers' deck panels pack the cards tighter than the game's own
 * deck screen, so each cost badge pokes past its card's cell and gets
 * clipped. Space the grid out with background-coloured gaps so every
 * badge sits inside its own cell, like a normal deck screenshot.
 * Same detection rules as split-deck.js. Returns null if no grid.
 */
function quantizedDominant(data, width, channels, yFrom, yTo) {
    const counts = new Map();
    for (let y = yFrom; y < yTo; y += 2) {
        for (let x = 0; x < width; x += 2) {
            const i = (y * width + x) * channels;
            const key = ((data[i] >> 2) << 12) | ((data[i + 1] >> 2) << 6) | (data[i + 2] >> 2);
            counts.set(key, (counts.get(key) || 0) + 1);
        }
    }
    let bestKey = 0;
    let bestCount = -1;
    for (const [key, count] of counts) {
        if (count > bestCount) {
            bestKey = key;
            bestCount = count;
        }
    }
    return {
        r: (((bestKey >> 12) & 63) << 2) + 2,
        g: (((bestKey >> 6) & 63) << 2) + 2,
        b: ((bestKey & 63) << 2) + 2,
    };
}

function runsOf(values, predicate, minLength, mergeGap) {
    const runs = [];
    let start = -1;
    for (let i = 0; i <= values.length; i++) {
        const on = i < values.length && predicate(values[i]);
        if (on && start < 0) start = i;
        if (!on && start >= 0) {
            const run = { start, end: i - 1 };
            const previous = runs[runs.length - 1];
            if (previous && run.start - previous.end - 1 <= mergeGap) previous.end = run.end;
            else runs.push(run);
            start = -1;
        }
    }
    return runs.filter(run => run.end - run.start + 1 >= minLength);
}

async function spreadDeckGrid(pngBuffer, { gapX = 0.02, gapY = 0 } = {}) {
    const { data, info } = await sharp(pngBuffer)
        .removeAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });
    const { width, height, channels } = info;

    // 1. Divider under the header (lowest cyan band in the top 70%)
    const maxY = Math.floor(height * 0.7);
    const cyanScores = new Float32Array(maxY);
    for (let y = 0; y < maxY; y++) {
        let total = 0;
        let run = 0;
        let longest = 0;
        for (let x = 0; x < width; x++) {
            const i = (y * width + x) * channels;
            if (isCyan(data[i], data[i + 1], data[i + 2])) {
                total++;
                run++;
                if (run > longest) longest = run;
            } else {
                run = 0;
            }
        }
        cyanScores[y] = Math.max(longest / width, (total / width) * 0.8);
    }
    const dividerBands = runsOf(cyanScores, v => v >= 0.32, 2, 2);
    if (!dividerBands.length) return null;
    const deckTop = dividerBands[dividerBands.length - 1].end + 1;

    // 2. Background colour and foreground rows/columns of the grid
    const background = quantizedDominant(data, width, channels, deckTop, height);
    const isForeground = (x, y) => {
        const i = (y * width + x) * channels;
        const dr = data[i] - background.r;
        const dg = data[i + 1] - background.g;
        const db = data[i + 2] - background.b;
        return dr * dr + dg * dg + db * db > 28 * 28;
    };

    const deckHeight = height - deckTop;
    const rowFractions = new Float32Array(deckHeight);
    for (let y = 0; y < deckHeight; y++) {
        let count = 0;
        for (let x = 0; x < width; x++) count += isForeground(x, deckTop + y) ? 1 : 0;
        rowFractions[y] = count / width;
    }
    const rowBands = runsOf(rowFractions, v => v > 0.5, Math.max(12, Math.floor(deckHeight * 0.03)), 2);
    if (!rowBands.length) return null;

    const first = rowBands[0];
    const bandHeight = first.end - first.start + 1;
    const innerTop = deckTop + first.start + Math.floor(bandHeight * 0.15);
    const innerBottom = deckTop + first.end - Math.floor(bandHeight * 0.15);
    const columnFractions = new Float32Array(width);
    for (let x = 0; x < width; x++) {
        let count = 0;
        for (let y = innerTop; y <= innerBottom; y++) count += isForeground(x, y) ? 1 : 0;
        columnFractions[x] = count / (innerBottom - innerTop + 1);
    }
    const columns = runsOf(columnFractions, v => v > 0.5, Math.max(20, Math.floor(width * 0.08)), 3);
    if (columns.length < 2) return null;

    // 3. Cut just before every card's left edge (the first column too, so
    //    the outer margin matches the gaps) and just below each row
    const gapW = Math.round(width * gapX);
    const gapH = Math.round(width * gapY);
    const xCuts = columns.map(column => column.start);
    const yCuts = [deckTop, ...rowBands.slice(1).map((band, index) => deckTop + rowBands[index].end + 1)];

    // Outer margins match the gap between cards, so the first and last
    // cells come out the same size as the others
    const cardGaps = columns.slice(1).map((column, index) => column.start - columns[index].end - 1);
    const typicalGap = cardGaps.sort((a, b) => a - b)[Math.floor(cardGaps.length / 2)] || 0;
    const leftPad = gapW + Math.max(0, typicalGap - columns[0].start);
    const rightPad = gapW + Math.max(0, typicalGap - (width - 1 - columns[columns.length - 1].end));

    const xEdges = [0, ...xCuts, width];
    const yEdges = [...yCuts, height];
    const xShift = index => (index === 0 ? 0 : leftPad + gapW * (index - 1));
    const newWidth = width + leftPad + gapW * (xCuts.length - 1) + rightPad;
    const newHeight = height + gapH * yCuts.length;

    const composites = [];

    // Header: stretched to the new width so the deck name and the 40/40
    // counter keep the same relative positions the OCR expects
    const header = await sharp(pngBuffer)
        .extract({ left: 0, top: 0, width, height: deckTop })
        .resize({ width: newWidth, height: deckTop, fit: "fill", kernel: "lanczos3" })
        .png()
        .toBuffer();
    composites.push({ input: header, left: 0, top: 0 });

    for (let row = 0; row < yEdges.length - 1; row++) {
        for (let column = 0; column < xEdges.length - 1; column++) {
            const tileWidth = xEdges[column + 1] - xEdges[column];
            const tileHeight = yEdges[row + 1] - yEdges[row];
            if (tileWidth <= 0 || tileHeight <= 0) continue;

            const tile = await sharp(pngBuffer)
                .extract({ left: xEdges[column], top: yEdges[row], width: tileWidth, height: tileHeight })
                .png()
                .toBuffer();

            composites.push({
                input: tile,
                left: xEdges[column] + xShift(column), // column 0 is the strip left of card 1
                top: yEdges[row] + gapH * (row + 1),
            });
        }
    }

    return sharp({
        create: {
            width: newWidth,
            height: newHeight,
            channels: 3,
            background,
        },
    })
        .composite(composites)
        .png({ compressionLevel: 1 })
        .toBuffer();
}

async function cropDeckPanel(frameBuffer) {
    const panel = await findDeckPanel(frameBuffer);
    if (!panel) return null;

    const scale = Math.max(1, TARGET_CROP_WIDTH / panel.width);

    const buffer = await sharp(frameBuffer)
        .extract({
            left: panel.left,
            top: panel.top,
            width: panel.width,
            height: panel.height,
        })
        .resize({
            width: Math.round(panel.width * scale),
            kernel: "lanczos3",
        })
        .png({ compressionLevel: 1 })
        .toBuffer();

    // Give each cost badge room inside its own card cell
    const spread = await spreadDeckGrid(buffer).catch(() => null);

    return { buffer: spread || buffer, panel, scale, spread: Boolean(spread) };
}

module.exports = {
    FRAME_NAMES,
    isValidVideoId,
    fetchFrame,
    findDeckPanel,
    cropDeckPanel,
    spreadDeckGrid,
};
