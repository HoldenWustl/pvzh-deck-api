"use strict";

const express = require("express");
const cors = require("cors");
const multer = require("multer");
const rateLimit = require("express-rate-limit");
const sharp = require("sharp");

const fs = require("fs/promises");
const path = require("path");
const os = require("os");
const crypto = require("crypto");
const { spawn } = require("child_process");
const { performance } = require("perf_hooks");

const youtubeFrames = require("./youtube-frames");

const app = express();

const PORT = Number(process.env.PORT) || 7860;
const PROJECT_ROOT = __dirname;

const SPLIT_SCRIPT = path.join(
    PROJECT_ROOT,
    "split-deck.js"
);

const READ_SCRIPT = path.join(
    PROJECT_ROOT,
    "read-deck.js"
);

const IDENTIFY_SCRIPT = path.join(
    PROJECT_ROOT,
    "identify-cards.py"
);

const CARD_DATA_PATH = path.join(
    PROJECT_ROOT,
    "card_data.json"
);

const REFERENCE_INDEX_PATH = path.join(
    PROJECT_ROOT,
    "reference_index"
);

const TEMP_ROOT = path.join(
    os.tmpdir(),
    "pvzh-deck-recognition"
);

const PYTHON_BIN =
    process.env.PYTHON_BIN || "python";

sharp.concurrency(1);

sharp.cache({
    memory: 32,
    files: 0,
    items: 50,
});

app.set("trust proxy", 1);

function isAllowedOrigin(origin) {
    if (!origin) {
        return true;
    }

    if (
        origin === "https://pvzhvault.com" ||
        origin === "https://www.pvzhvault.com"
    ) {
        return true;
    }

    if (
        /^https:\/\/[a-z0-9-]+\.github\.io$/i.test(origin)
    ) {
        return true;
    }

    if (
        /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/i.test(
            origin
        )
    ) {
        return true;
    }

    return false;
}

app.use(
    cors({
        origin(origin, callback) {
            if (isAllowedOrigin(origin)) {
                callback(null, true);
                return;
            }

            callback(
                new Error(
                    `Origin is not allowed: ${origin}`
                )
            );
        },

        methods: [
            "GET",
            "POST",
            "OPTIONS",
        ],
    })
);

const recognitionLimiter = rateLimit({
    windowMs: 15 * 60 * 1000,
    limit: 20,
    standardHeaders: true,
    legacyHeaders: false,

    handler(request, response) {
        response.status(429).json({
            error:
                "Too many deck-recognition requests. Please try again later.",
        });
    },
});

const upload = multer({
    storage: multer.memoryStorage(),

    limits: {
        files: 1,
        fileSize: 15 * 1024 * 1024,
    },

    fileFilter(request, file, callback) {
        if (
            !file.mimetype ||
            !file.mimetype.startsWith("image/")
        ) {
            callback(
                new Error(
                    "Only image files are accepted."
                )
            );

            return;
        }

        callback(null, true);
    },
});

/*
 * One job at a time is deliberate for a 512 MB instance.
 */
let queueTail = Promise.resolve();

function enqueueRecognition(task) {
    const result = queueTail.then(
        task,
        task
    );

    queueTail = result.catch(() => {});

    return result;
}

function runProcess(
    command,
    args,
    {
        jobId,
        timeoutMs = 10 * 60 * 1000,
    } = {}
) {
    return new Promise(
        (resolve, reject) => {
            const started = performance.now();

            console.log(
                `[${jobId}] Running:`,
                command,
                ...args
            );

            const child = spawn(
                command,
                args,
                {
                    cwd: PROJECT_ROOT,
                    shell: false,
                    env: process.env,
                }
            );

            let stdout = "";
            let stderr = "";
            let settled = false;

            const timer = setTimeout(
                () => {
                    if (settled) {
                        return;
                    }

                    settled = true;
                    child.kill("SIGKILL");

                    reject(
                        new Error(
                            `${path.basename(command)} timed out.`
                        )
                    );
                },
                timeoutMs
            );

            child.stdout.on(
                "data",
                chunk => {
                    const text = chunk.toString();

                    stdout += text;
                    process.stdout.write(
                        `[${jobId}] ${text}`
                    );
                }
            );

            child.stderr.on(
                "data",
                chunk => {
                    const text = chunk.toString();

                    stderr += text;
                    process.stderr.write(
                        `[${jobId}] ${text}`
                    );
                }
            );

            child.on(
                "error",
                error => {
                    if (settled) {
                        return;
                    }

                    settled = true;
                    clearTimeout(timer);

                    reject(
                        new Error(
                            `Could not start ${command}: ${error.message}`
                        )
                    );
                }
            );

            child.on(
                "close",
                code => {
                    if (settled) {
                        return;
                    }

                    settled = true;
                    clearTimeout(timer);

                    const elapsedSeconds =
                        (performance.now() - started) / 1000;

                    if (code !== 0) {
                        reject(
                            new Error(
                                [
                                    `${path.basename(command)} exited with code ${code}.`,
                                    stderr.trim(),
                                    stdout.trim(),
                                ]
                                    .filter(Boolean)
                                    .join("\n")
                            )
                        );

                        return;
                    }

                    resolve({
                        stdout,
                        stderr,
                        elapsedSeconds,
                    });
                }
            );
        }
    );
}

async function recognizeDeck(
    imageBuffer,
    originalFilename
) {
    const jobId =
        crypto.randomUUID();

    const jobStarted =
        performance.now();

    const jobDirectory = path.join(
        TEMP_ROOT,
        jobId
    );

    const cardsDirectory = path.join(
        jobDirectory,
        "cards"
    );

    const imagePath = path.join(
        jobDirectory,
        "deck.png"
    );

    await fs.mkdir(
        jobDirectory,
        {
            recursive: true,
        }
    );

    try {
        console.log(
            `[${jobId}] Received ${originalFilename}`
        );

        const metadata = await sharp(
            imageBuffer,
            {
                limitInputPixels:
                    50_000_000,
            }
        ).metadata();

        if (
            !metadata.width ||
            !metadata.height
        ) {
            throw new Error(
                "The uploaded file is not a readable image."
            );
        }

        if (
            metadata.width < 300 ||
            metadata.height < 300
        ) {
            throw new Error(
                "The uploaded image is too small."
            );
        }

        const normalizeStarted =
            performance.now();

        /*
         * PNG compression level changes size, not image quality.
         * Level 1 is much faster for this temporary local file.
         */
        await sharp(
            imageBuffer,
            {
                limitInputPixels:
                    50_000_000,
            }
        )
            .rotate()
            .png({
                compressionLevel: 1,
            })
            .toFile(imagePath);

        const normalizeSeconds =
            (performance.now() - normalizeStarted) / 1000;

        console.log(
            `[${jobId}] Normalized image in ${normalizeSeconds.toFixed(2)}s`
        );

        console.log(
            `[${jobId}] 1/3 Splitting cards`
        );

        const splitResult =
            await runProcess(
                process.execPath,
                [
                    SPLIT_SCRIPT,
                    imagePath,
                    cardsDirectory,
                ],
                {
                    jobId,
                    timeoutMs:
                        2 * 60 * 1000,
                }
            );

        console.log(
            `[${jobId}] Split stage: ${splitResult.elapsedSeconds.toFixed(2)}s`
        );

        console.log(
            `[${jobId}] 2/3 Reading name, copies and costs`
        );

        const readResult =
            await runProcess(
                process.execPath,
                [
                    READ_SCRIPT,
                    imagePath,
                    cardsDirectory,
                ],
                {
                    jobId,
                    timeoutMs:
                        5 * 60 * 1000,
                }
            );

        console.log(
            `[${jobId}] OCR stage: ${readResult.elapsedSeconds.toFixed(2)}s`
        );

        console.log(
            `[${jobId}] 3/3 Identifying cards`
        );

        const identifyResult =
            await runProcess(
                PYTHON_BIN,
                [
                    IDENTIFY_SCRIPT,
                    cardsDirectory,
                    CARD_DATA_PATH,
                    REFERENCE_INDEX_PATH,
                ],
                {
                    jobId,
                    timeoutMs:
                        10 * 60 * 1000,
                }
            );

        console.log(
            `[${jobId}] Identification stage: ${identifyResult.elapsedSeconds.toFixed(2)}s`
        );

        const resultPath = path.join(
            cardsDirectory,
            "deck-identified.json"
        );

        const resultText =
            await fs.readFile(
                resultPath,
                "utf8"
            );

        const result =
            JSON.parse(resultText);

        const totalSeconds =
            (performance.now() - jobStarted) / 1000;

        result.serverTiming = {
            normalizeSeconds:
                Number(normalizeSeconds.toFixed(3)),
            splitSeconds:
                Number(splitResult.elapsedSeconds.toFixed(3)),
            ocrSeconds:
                Number(readResult.elapsedSeconds.toFixed(3)),
            identifySeconds:
                Number(identifyResult.elapsedSeconds.toFixed(3)),
            totalSeconds:
                Number(totalSeconds.toFixed(3)),
        };

        console.log(
            `[${jobId}] Recognition complete: ${result.deckName} in ${totalSeconds.toFixed(2)}s`
        );

        return result;
    } finally {
        await fs.rm(
            jobDirectory,
            {
                recursive: true,
                force: true,
            }
        ).catch(error => {
            console.error(
                `[${jobId}] Could not clean temporary files:`,
                error
            );
        });
    }
}

app.get(
    "/",
    (request, response) => {
        response.json({
            service:
                "PvZH Deck Recognition API",
            status: "running",
        });
    }
);

app.get(
    "/health",
    (request, response) => {
        response.json({
            ok: true,
        });
    }
);

app.post(
    "/api/recognize-deck",
    recognitionLimiter,
    (request, response) => {
        upload.single("deckImage")(
            request,
            response,
            async uploadError => {
                if (uploadError) {
                    const tooLarge =
                        uploadError.code ===
                        "LIMIT_FILE_SIZE";

                    response
                        .status(400)
                        .json({
                            error: tooLarge
                                ? "The image is larger than 15 MB."
                                : uploadError.message,
                        });

                    return;
                }

                if (!request.file) {
                    response
                        .status(400)
                        .json({
                            error:
                                "No deck image was uploaded.",
                        });

                    return;
                }

                try {
                    const result =
                        await enqueueRecognition(
                            () =>
                                recognizeDeck(
                                    request.file.buffer,
                                    request.file.originalname
                                )
                        );

                    response.json(result);
                } catch (error) {
                    console.error(
                        "Recognition failed:",
                        error
                    );

                    response
                        .status(500)
                        .json({
                            error:
                                error?.message ||
                                "Deck recognition failed.",
                        });
                }
            }
        );
    }
);


/*
 * YouTube links
 *
 * Reads a deck from one of the three still frames YouTube keeps for
 * every video (about 25% / 50% / 75% through). The deck panel is
 * cropped out of the frame first, so the normal pipeline sees what
 * looks like an ordinary deck screenshot. Uploads are never cropped.
 */
const YOUTUBE_MAX_RECOGNITION_ATTEMPTS = 3;

function countCopies(result) {
    return (result?.cards || []).reduce(
        (total, card) => total + (Number(card.copies) || 0),
        0
    );
}

function summarizeAttempt(result) {
    const cards = (result?.cards || []).map(card => ({
        name: card.name || null,
        copies: card.copies ?? null,
        cost: card.cost ?? null,
        needsReview: Boolean(card.needsReview),
    }));

    return {
        deckName: result?.deckName ?? null,
        cardsFound: cards.length,
        totalCopies: countCopies(result),
        cards,
    };
}

async function recognizeYouTubeDeck(videoId) {
    let attempts = 0;
    let panelsFound = 0;
    let fetchedHd = false;
    const diagnostics = [];

    for (const frameName of youtubeFrames.FRAME_NAMES) {
        if (attempts >= YOUTUBE_MAX_RECOGNITION_ATTEMPTS) {
            break;
        }

        // sd frames are the same moments at lower resolution: only
        // fall back to them when the video has no HD frames at all.
        if (fetchedHd && frameName.startsWith("sd")) {
            break;
        }

        const frame = await youtubeFrames.fetchFrame(
            videoId,
            frameName
        );

        if (!frame) {
            diagnostics.push({ frame: frameName, available: false });
            continue;
        }

        if (frameName.startsWith("maxres")) {
            fetchedHd = true;
        }

        const crop = await youtubeFrames.cropDeckPanel(
            frame.buffer
        );

        const entry = {
            frame: frameName,
            available: true,
            frameSize: `${frame.width}x${frame.height}`,
            panel: crop ? crop.panel : null,
        };
        diagnostics.push(entry);

        if (!crop) {
            console.log(
                `[youtube ${videoId}] No deck panel in ${frameName}`
            );
            continue;
        }

        panelsFound++;
        attempts++;

        try {
            const result = await enqueueRecognition(
                () =>
                    recognizeDeck(
                        crop.buffer,
                        `youtube-${videoId}-${frameName}.png`
                    )
            );

            const total = countCopies(result);
            Object.assign(entry, summarizeAttempt(result));

            if (total === 40) {
                result.source = {
                    type: "youtube",
                    videoId,
                    frame: frameName,
                    diagnostics,
                };

                return result;
            }

            console.log(
                `[youtube ${videoId}] ${frameName} read ${total} cards, trying the next frame`
            );
        } catch (error) {
            entry.error = String(error?.message || error).slice(0, 600);
            console.log(
                `[youtube ${videoId}] ${frameName} failed: ${error.message}`
            );
        }
    }

    const error = new Error(
        panelsFound
            ? "Found a deck list in that video, but couldn't read all 40 cards."
            : "Couldn't find a deck list in that video's preview frames."
    );

    error.statusCode = 422;
    error.code = panelsFound
        ? "DECK_UNREADABLE"
        : "NO_DECK_PANEL";
    error.diagnostics = diagnostics;

    throw error;
}

app.get(
    "/api/recognize-youtube",
    recognitionLimiter,
    async (request, response) => {
        const videoId = String(
            request.query.v || ""
        );

        if (
            !youtubeFrames.isValidVideoId(videoId)
        ) {
            response.status(400).json({
                error:
                    "That isn't a valid YouTube video ID.",
            });

            return;
        }

        try {
            const result =
                await recognizeYouTubeDeck(videoId);

            response.json(result);
        } catch (error) {
            console.error(
                `YouTube recognition failed for ${videoId}:`,
                error.message
            );

            response
                .status(error.statusCode || 500)
                .json({
                    error:
                        error?.message ||
                        "Deck recognition failed.",
                    code: error.code || null,
                    diagnostics: error.diagnostics || null,
                });
        }
    }
);


/*
 * Debug: shows the exact image the YouTube path feeds the reader.
 *   /api/youtube-frame?v=VIDEO_ID&frame=maxres2          cropped panel
 *   /api/youtube-frame?v=VIDEO_ID&frame=maxres2&crop=0   whole frame
 */
const debugFrameLimiter = rateLimit({
    windowMs: 15 * 60 * 1000,
    limit: 60,
    standardHeaders: true,
    legacyHeaders: false,
});

app.get(
    "/api/youtube-frame",
    debugFrameLimiter,
    async (request, response) => {
        const videoId = String(request.query.v || "");
        const frameName = String(request.query.frame || "maxres2");
        const wantCrop = request.query.crop !== "0";

        if (
            !youtubeFrames.isValidVideoId(videoId) ||
            !youtubeFrames.FRAME_NAMES.includes(frameName)
        ) {
            response.status(400).json({
                error: `Use ?v=VIDEO_ID&frame=${youtubeFrames.FRAME_NAMES.join("|")}`,
            });
            return;
        }

        try {
            const frame = await youtubeFrames.fetchFrame(videoId, frameName);

            if (!frame) {
                response.status(404).json({ error: `YouTube has no ${frameName} frame for this video.` });
                return;
            }

            if (!wantCrop) {
                response.type("jpeg").send(frame.buffer);
                return;
            }

            const crop = await youtubeFrames.cropDeckPanel(frame.buffer);

            if (!crop) {
                response.status(404).json({ error: `No deck panel found in ${frameName}.` });
                return;
            }

            response.set("X-Deck-Panel", JSON.stringify(crop.panel));
            response.type("png").send(crop.buffer);
        } catch (error) {
            response.status(500).json({ error: error?.message || "Could not build the frame." });
        }
    }
);

app.use(
    (
        error,
        request,
        response,
        next
    ) => {
        console.error(
            "Unhandled server error:",
            error
        );

        response.status(500).json({
            error:
                error?.message ||
                "Unexpected server error.",
        });
    }
);

app.listen(
    PORT,
    "0.0.0.0",
    () => {
        console.log(
            `PvZH recognition API listening on port ${PORT}`
        );

        console.log(
            `Reference index: ${REFERENCE_INDEX_PATH}`
        );
    }
);
