"use strict";

/*
 * Recognition for YouTube frames.
 *
 * Same stages as recognizeDeck() in server.js (split -> OCR -> identify),
 * run on the cropped deck panel, with two extra safety nets because
 * video frames are softer than real screenshots:
 *
 *   - A cost the OCR couldn't read is inferred from the deck's sort order
 *     (cards are listed by cost). If one value fits, it's used; if a few
 *     fit, identification is run for each option (max MAX_COST_OPTIONS)
 *     and the best-matching result is kept.
 *   - A single unreadable copy count is filled from the 40-card total.
 *   - If anything is still unread after the first OCR pass, a second pass
 *     runs on a median-filtered copy (same card grid) and the readings
 *     are combined, since each version reads some digits the other misses.
 *
 * Normal uploads never come through here.
 */

const fs = require("fs/promises");
const path = require("path");
const crypto = require("crypto");
const sharp = require("sharp");

const MAX_COST_OPTIONS = 4;
const HIGHEST_COST = 12;

function inferMissingCosts(costs) {
    const missing = costs
        .map((cost, index) => (Number.isInteger(cost) ? null : index))
        .filter(index => index !== null);

    if (!missing.length) return [costs.slice()];

    // Allowed range for each missing slot from the known neighbours
    const ranges = missing.map(index => {
        let low = 0;
        let high = HIGHEST_COST;
        for (let i = index - 1; i >= 0; i--) {
            if (Number.isInteger(costs[i])) { low = costs[i]; break; }
        }
        for (let i = index + 1; i < costs.length; i++) {
            if (Number.isInteger(costs[i])) { high = costs[i]; break; }
        }
        return { index, low, high };
    });

    // Enumerate non-decreasing fills, stopping once there are too many
    const options = [];
    const filled = costs.slice();

    (function walk(position, floor) {
        if (options.length > MAX_COST_OPTIONS) return;
        if (position === ranges.length) {
            options.push(filled.slice());
            return;
        }
        const { index, low, high } = ranges[position];
        for (let value = Math.max(low, floor); value <= high; value++) {
            filled[index] = value;
            const nextIsAdjacent = ranges[position + 1] && ranges[position + 1].index === index + 1;
            walk(position + 1, nextIsAdjacent ? value : 0);
        }
    })(0, 0);

    return options.length <= MAX_COST_OPTIONS ? options : [];
}

/*
 * In video frames a cost digit often touches the badge's dark outline,
 * so OCR isolates the digit with a trail of outline attached. A
 * morphological closing (max filter, then min filter) erases dark lines
 * thinner than the kernel, removing the outline but keeping the much
 * thicker digit strokes. Applied only to each card's cost area, using
 * the same area read-deck.js reads.
 */
async function cleanCostBadges(imagePath, cards, radius) {
    const { data, info } = await sharp(imagePath)
        .removeAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });
    const { width, height, channels } = info;
    const out = Buffer.from(data);

    const filter = (source, target, x0, y0, x1, y1, pick) => {
        for (let y = y0; y < y1; y++) {
            for (let x = x0; x < x1; x++) {
                for (let c = 0; c < 3; c++) {
                    let value = pick === "max" ? 0 : 255;
                    for (let dy = -radius; dy <= radius; dy++) {
                        const yy = Math.min(height - 1, Math.max(0, y + dy));
                        for (let dx = -radius; dx <= radius; dx++) {
                            const xx = Math.min(width - 1, Math.max(0, x + dx));
                            const v = source[(yy * width + xx) * channels + c];
                            value = pick === "max" ? Math.max(value, v) : Math.min(value, v);
                        }
                    }
                    target[(y * width + x) * channels + c] = value;
                }
            }
        }
    };

    for (const card of cards) {
        const x0 = Math.max(0, Math.floor(card.x + card.width * 0.66));
        const y0 = Math.max(0, Math.floor(card.y));
        const x1 = Math.min(width, Math.ceil(card.x + card.width));
        const y1 = Math.min(height, Math.ceil(card.y + card.height * 0.42));
        if (x1 <= x0 || y1 <= y0) continue;

        const dilated = Buffer.from(out);
        filter(out, dilated, x0, y0, x1, y1, "max");
        filter(dilated, out, x0, y0, x1, y1, "min");
    }

    return sharp(out, { raw: { width, height, channels } }).png({ compressionLevel: 1 }).toBuffer();
}

function fillSingleMissingCopy(copies) {
    const missing = copies.filter(copy => !Number.isInteger(copy)).length;
    if (missing !== 1) return copies;

    const known = copies.reduce((total, copy) => total + (Number.isInteger(copy) ? copy : 0), 0);
    const remainder = 40 - known;
    if (remainder < 1 || remainder > 4) return copies;

    return copies.map(copy => (Number.isInteger(copy) ? copy : remainder));
}

function createYouTubeRecognizer({
    runProcess,
    splitScript,
    readScript,
    identifyScript,
    cardDataPath,
    referenceIndexPath,
    pythonBin,
    tempRoot,
}) {
    return async function recognizeYouTubeCrop(imageBuffer, label) {
        const jobId = crypto.randomUUID();
        const jobDirectory = path.join(tempRoot, jobId);
        const cardsDirectory = path.join(jobDirectory, "cards");
        const imagePath = path.join(jobDirectory, "deck.png");

        await fs.mkdir(jobDirectory, { recursive: true });

        try {
            console.log(`[${jobId}] Received ${label}`);

            await sharp(imageBuffer).png({ compressionLevel: 1 }).toFile(imagePath);

            await runProcess(process.execPath, [splitScript, imagePath, cardsDirectory], {
                jobId,
                timeoutMs: 2 * 60 * 1000,
            });

            // OCR reads a copy whose cost badges have their outlines removed;
            // identification still uses the untouched card crops from split.
            const meta = JSON.parse(await fs.readFile(path.join(cardsDirectory, "cards.json"), "utf8"));
            const ocrImagePath = path.join(jobDirectory, "deck-ocr.png");
            const radius = Math.max(1, Math.round((meta.image?.width || 1000) / 1000 * 2));
            await fs.writeFile(ocrImagePath, await cleanCostBadges(imagePath, meta.cards, radius));

            await runProcess(process.execPath, [readScript, ocrImagePath, cardsDirectory], {
                jobId,
                timeoutMs: 5 * 60 * 1000,
            });

            const readPath = path.join(cardsDirectory, "deck-read.json");
            const deck = JSON.parse(await fs.readFile(readPath, "utf8"));

            const unread = deck.cards.some(card => !Number.isInteger(card.cost) || !Number.isInteger(card.copies));

            if (unread) {
                // Second OCR pass on a median-filtered copy, same card grid
                const altDirectory = path.join(jobDirectory, "cards-alt");
                const altImagePath = path.join(jobDirectory, "deck-alt.png");
                await fs.mkdir(altDirectory, { recursive: true });
                await fs.copyFile(path.join(cardsDirectory, "cards.json"), path.join(altDirectory, "cards.json"));
                await sharp(imagePath).median(5).png({ compressionLevel: 1 }).toFile(altImagePath);

                try {
                    await runProcess(process.execPath, [readScript, altImagePath, altDirectory], {
                        jobId,
                        timeoutMs: 5 * 60 * 1000,
                    });
                    const alt = JSON.parse(await fs.readFile(path.join(altDirectory, "deck-read.json"), "utf8"));
                    let filledFromAlt = 0;
                    deck.cards.forEach((card, i) => {
                        const altCard = alt.cards?.[i];
                        if (!altCard) return;
                        if (!Number.isInteger(card.cost) && Number.isInteger(altCard.cost)) { card.cost = altCard.cost; filledFromAlt++; }
                        if (!Number.isInteger(card.copies) && Number.isInteger(altCard.copies)) { card.copies = altCard.copies; filledFromAlt++; }
                    });
                    if (!deck.deckName && alt.deckName) deck.deckName = alt.deckName;
                    console.log(`[${jobId}] Second OCR pass filled ${filledFromAlt} reading(s)`);
                } catch (error) {
                    console.log(`[${jobId}] Second OCR pass failed: ${error.message.split("\n")[0]}`);
                }
            }

            const copies = fillSingleMissingCopy(deck.cards.map(card => card.copies));
            const costOptions = inferMissingCosts(deck.cards.map(card => card.cost));

            if (!costOptions.length) {
                throw new Error("Too many card costs were unreadable.");
            }

            if (costOptions.length > 1 || costOptions[0].some((cost, i) => cost !== deck.cards[i].cost)) {
                console.log(`[${jobId}] Inferred missing costs: ${costOptions.length} option(s)`);
            }

            let best = null;

            for (const costs of costOptions) {
                const patched = {
                    ...deck,
                    copies,
                    totalCopies: copies.reduce((total, copy) => total + (Number(copy) || 0), 0),
                    costs,
                    cards: deck.cards.map((card, i) => ({ ...card, copies: copies[i], cost: costs[i] })),
                };
                patched.copyTotalIs40 = patched.totalCopies === 40;

                await fs.writeFile(readPath, JSON.stringify(patched, null, 2));

                try {
                    await runProcess(pythonBin, [identifyScript, cardsDirectory, cardDataPath, referenceIndexPath], {
                        jobId,
                        timeoutMs: 10 * 60 * 1000,
                    });
                } catch (error) {
                    console.log(`[${jobId}] Identification failed for costs ${costs.join(",")}: ${error.message.split("\n")[0]}`);
                    continue;
                }

                const result = JSON.parse(
                    await fs.readFile(path.join(cardsDirectory, "deck-identified.json"), "utf8")
                );
                const score = (result.cards || []).reduce((total, card) => total + (Number(card.matchScore) || 0), 0);

                if (!best || score > best.score) {
                    best = { result, score };
                }
            }

            if (!best) {
                throw new Error("Card identification failed for every cost reading.");
            }

            // A wide header can let the "40/40" counter into the name reading
            if (typeof best.result.deckName === "string") {
                best.result.deckName = best.result.deckName
                    .replace(/\s*\d{1,2}\s*\/\s*\d[\d()]*\s*$/, "")
                    .replace(/\s+[:;,.|]+$/, "")
                    .trim() || best.result.deckName;
            }

            return best.result;
        } finally {
            await fs.rm(jobDirectory, { recursive: true, force: true }).catch(() => {});
        }
    };
}

module.exports = {
    createYouTubeRecognizer,
    inferMissingCosts,
    fillSingleMissingCopy,
};
