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

    return { buffer, panel, scale };
}

module.exports = {
    FRAME_NAMES,
    isValidVideoId,
    fetchFrame,
    findDeckPanel,
    cropDeckPanel,
};
