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
