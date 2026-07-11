const sharp = require("sharp");
const fs = require("fs/promises");
const path = require("path");
const { createWorker, PSM } = require("tesseract.js");
const englishData = require("@tesseract.js-data/eng");

const INPUT = process.argv[2] || "deck.png";
const CARDS_DIR = process.argv[3] || "cards";

const META_FILE = path.join(CARDS_DIR, "cards.json");
const OUTPUT_FILE = path.join(CARDS_DIR, "deck-read.json");
const DEBUG_DIR = path.join(CARDS_DIR, "ocr-debug");
const DEBUG_OUTPUT = process.env.PVZH_DEBUG === "1";

const clamp = (number, minimum, maximum) =>
  Math.max(minimum, Math.min(maximum, number));

function safeRect(rect, imageWidth, imageHeight) {
  const left = clamp(
    Math.round(rect.left),
    0,
    imageWidth - 1
  );

  const top = clamp(
    Math.round(rect.top),
    0,
    imageHeight - 1
  );

  const right = clamp(
    Math.round(rect.left + rect.width),
    left + 1,
    imageWidth
  );

  const bottom = clamp(
    Math.round(rect.top + rect.height),
    top + 1,
    imageHeight
  );

  return {
    left,
    top,
    width: right - left,
    height: bottom - top,
  };
}

function cleanName(text) {
  return String(text || "")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^[^A-Za-z0-9]+/, "")
    .replace(/[^A-Za-z0-9'&:+!?().\- ]+$/, "")
    .replace(/\s+40(?:\s*\/\s*40)?$/, "")
    .trim();
}

function parseCopies(text) {
  const matches = String(text || "").match(/[1-4]/g);

  if (!matches) {
    return null;
  }

  return Number(matches[matches.length - 1]);
}

function parseCost(text) {
  const match = String(text || "").match(/\d{1,2}/);

  if (!match) {
    return null;
  }

  const value = Number(match[0]);

  if (
    !Number.isInteger(value) ||
    value < 0 ||
    value > 20
  ) {
    return null;
  }

  return value;
}

function sharpFromRawSource(source) {
  return sharp(source.data, {
    raw: source.raw,
  });
}

async function preprocessName(
  source,
  rect,
  threshold
) {
  return sharpFromRawSource(source)
    .extract(rect)
    .grayscale()
    .threshold(threshold)
    .resize({
      width: rect.width * 2,
      kernel: sharp.kernel.cubic,
    })
    .png({ compressionLevel: 1 })
    .toBuffer();
}

async function preprocessCopies(
  source,
  rect,
  threshold
) {
  return sharpFromRawSource(source)
    .extract(rect)
    .grayscale()
    .threshold(threshold)
    .negate()
    .resize({
      width: rect.width * 4,
      kernel: sharp.kernel.cubic,
    })
    .extend({
      top: 20,
      bottom: 20,
      left: 20,
      right: 20,
      background: {
        r: 255,
        g: 255,
        b: 255,
        alpha: 1,
      },
    })
    .png({ compressionLevel: 1 })
    .toBuffer();
}

/*
 * Finds connected groups of dark pixels.
 *
 * This is used to isolate the black number inside
 * the sun or brain cost icon while ignoring the
 * surrounding card artwork.
 */
function findComponents(
  binary,
  width,
  height
) {
  const seen = new Uint8Array(binary.length);
  const queue = new Int32Array(binary.length);
  const result = [];

  for (
    let start = 0;
    start < binary.length;
    start++
  ) {
    if (!binary[start] || seen[start]) {
      continue;
    }

    let head = 0;
    let tail = 0;

    queue[tail++] = start;
    seen[start] = 1;

    let minX = width;
    let minY = height;
    let maxX = -1;
    let maxY = -1;
    let sumX = 0;
    let sumY = 0;

    const pixels = [];

    while (head < tail) {
      const index = queue[head++];
      const x = index % width;
      const y = Math.floor(index / width);

      pixels.push(index);

      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);

      sumX += x;
      sumY += y;

      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) {
            continue;
          }

          const nextX = x + dx;
          const nextY = y + dy;

          if (
            nextX < 0 ||
            nextX >= width ||
            nextY < 0 ||
            nextY >= height
          ) {
            continue;
          }

          const next =
            nextY * width + nextX;

          if (
            binary[next] &&
            !seen[next]
          ) {
            seen[next] = 1;
            queue[tail++] = next;
          }
        }
      }
    }

    const area = pixels.length;

    result.push({
      pixels,
      area,
      x: minX,
      y: minY,
      width: maxX - minX + 1,
      height: maxY - minY + 1,
      cx: sumX / area,
      cy: sumY / area,
    });
  }

  return result;
}

/*
 * Isolates the cost number from its sun or brain icon.
 *
 * Several thresholds are attempted later because
 * antialiasing and screenshot compression vary.
 */
async function isolateCostDigits(
  source,
  rect,
  threshold
) {
  const {
    data,
    info,
  } = await sharpFromRawSource(source)
    .extract(rect)
    .raw()
    .toBuffer({
      resolveWithObject: true,
    });

  const {
    width,
    height,
    channels,
  } = info;

  const binary = new Uint8Array(
    width * height
  );

  for (
    let index = 0;
    index < binary.length;
    index++
  ) {
    const offset = index * channels;

    const red = data[offset];
    const green = data[offset + 1];
    const blue = data[offset + 2];

    const grayscale =
      0.299 * red +
      0.587 * green +
      0.114 * blue;

    binary[index] =
      grayscale < threshold ? 1 : 0;
  }

  const minimumArea = Math.max(
    12,
    Math.round(
      width * height * 0.004
    )
  );

  const candidates = findComponents(
    binary,
    width,
    height
  )
    .filter(component => {
      const centerX =
        component.cx / width;

      const centerY =
        component.cy / height;

      return (
        component.area >= minimumArea &&
        component.height >= height * 0.25 &&
        component.height <= height * 0.95 &&
        component.width <= width * 0.8 &&
        centerX >= 0.12 &&
        centerX <= 0.88 &&
        centerY >= 0.08 &&
        centerY <= 0.82
      );
    })
    .map(component => {
      const centrality =
        Math.abs(
          component.cx / width - 0.5
        ) +
        Math.abs(
          component.cy / height - 0.48
        );

      return {
        ...component,

        score:
          component.area -
          width *
            height *
            0.09 *
            centrality,
      };
    })
    .sort(
      (first, second) =>
        second.score - first.score
    );

  if (candidates.length === 0) {
    return null;
  }

  const primary = candidates[0];
  const selected = [primary];

  /*
   * A cost can theoretically have two digits,
   * such as 10 or 11.
   *
   * Include a nearby component if it looks like
   * a second digit of approximately the same size.
   */
  const second = candidates
    .slice(1)
    .find(component => {
      const verticalDistance =
        Math.abs(
          component.cy - primary.cy
        );

      const primaryRight =
        primary.x + primary.width;

      const componentRight =
        component.x + component.width;

      const horizontalGap = Math.max(
        0,
        Math.max(
          primary.x,
          component.x
        ) -
          Math.min(
            primaryRight,
            componentRight
          )
      );

      return (
        component.height >=
          primary.height * 0.65 &&
        component.height <=
          primary.height * 1.35 &&
        component.area >=
          primary.area * 0.25 &&
        verticalDistance <=
          height * 0.15 &&
        horizontalGap <=
          width * 0.18
      );
    });

  if (second) {
    selected.push(second);
  }

  const minimumX = Math.min(
    ...selected.map(
      component => component.x
    )
  );

  const minimumY = Math.min(
    ...selected.map(
      component => component.y
    )
  );

  const maximumX = Math.max(
    ...selected.map(
      component =>
        component.x +
        component.width -
        1
    )
  );

  const maximumY = Math.max(
    ...selected.map(
      component =>
        component.y +
        component.height -
        1
    )
  );

  const padding = 10;

  const canvasWidth =
    maximumX -
    minimumX +
    1 +
    padding * 2;

  const canvasHeight =
    maximumY -
    minimumY +
    1 +
    padding * 2;

  /*
   * Start with a white image and draw only
   * the selected digit pixels in black.
   */
  const canvas = Buffer.alloc(
    canvasWidth * canvasHeight,
    255
  );

  for (const component of selected) {
    for (
      const pixelIndex
      of component.pixels
    ) {
      const x =
        pixelIndex % width;

      const y =
        Math.floor(
          pixelIndex / width
        );

      const outputX =
        x - minimumX + padding;

      const outputY =
        y - minimumY + padding;

      canvas[
        outputY * canvasWidth +
        outputX
      ] = 0;
    }
  }

  return sharp(canvas, {
    raw: {
      width: canvasWidth,
      height: canvasHeight,
      channels: 1,
    },
  })
    .resize({
      width: 90,
      height: 130,
      fit: "contain",
      kernel: sharp.kernel.nearest,
    })
    .png({ compressionLevel: 1 })
    .toBuffer();
}

function mostCommon(values) {
  const counts = new Map();

  for (const value of values) {
    counts.set(
      value,
      (counts.get(value) || 0) + 1
    );
  }

  return (
    [...counts.entries()]
      .sort(
        (first, second) =>
          second[1] - first[1]
      )[0]?.[0] ?? null
  );
}

async function main() {
  const metadata = JSON.parse(
    await fs.readFile(
      META_FILE,
      "utf8"
    )
  );

  /*
   * Decode the uploaded image exactly once. Every OCR crop is then
   * taken from this in-memory raw image instead of decoding deck.png
   * again for every Tesseract attempt.
   */
  const {
    data: sourceData,
    info: sourceInfo,
  } = await sharp(INPUT)
    .removeAlpha()
    .raw()
    .toBuffer({
      resolveWithObject: true,
    });

  const imageWidth = sourceInfo.width;
  const imageHeight = sourceInfo.height;

  if (!imageWidth || !imageHeight) {
    throw new Error(
      "Could not read the image size."
    );
  }

  const source = {
    data: sourceData,
    raw: {
      width: sourceInfo.width,
      height: sourceInfo.height,
      channels: sourceInfo.channels,
    },
  };

  if (DEBUG_OUTPUT) {
    await fs.mkdir(DEBUG_DIR, {
      recursive: true,
    });
  }

  /*
   * Use locally installed English training data.
   * This avoids downloading the OCR model whenever
   * the script is run on a new machine.
   */
  const worker = await createWorker(
    "eng",
    1,
    {
      langPath:
        englishData.langPath,

      gzip:
        englishData.gzip,
    }
  );

  try {
    /*
     * --------------------------------------------------
     * 1. DECK NAME
     * --------------------------------------------------
     */

    const nameRect = safeRect(
      {
        left:
          imageWidth * 0.10,

        top:
          metadata.deckTop -
          imageHeight * 0.12,

        width:
          imageWidth * 0.76,

        height:
          imageHeight * 0.105,
      },
      imageWidth,
      imageHeight
    );

    await worker.setParameters({
      tessedit_pageseg_mode:
        PSM.SINGLE_LINE,

      tessedit_char_whitelist:
        "",

      user_defined_dpi:
        "300",
    });

    let deckName = null;
    let nameDebugImage = null;

    for (
      const threshold
      of [170, 200]
    ) {
      const buffer =
        await preprocessName(
          source,
          nameRect,
          threshold
        );

      const result =
        await worker.recognize(
          buffer
        );

      const candidate =
        cleanName(
          result.data.text
        );

      if (candidate.length >= 2) {
        deckName = candidate;
        nameDebugImage = buffer;
        break;
      }
    }

    if (DEBUG_OUTPUT && nameDebugImage) {
      await fs.writeFile(
        path.join(
          DEBUG_DIR,
          "deck-name.png"
        ),
        nameDebugImage
      );
    }

    /*
     * --------------------------------------------------
     * 2. COPY COUNTS
     * --------------------------------------------------
     */

    await worker.setParameters({
      tessedit_pageseg_mode:
        PSM.SINGLE_WORD,

      tessedit_char_whitelist:
        "xX1234",

      user_defined_dpi:
        "300",
    });

    const copies = [];

    for (
      let cardIndex = 0;
      cardIndex <
      metadata.cards.length;
      cardIndex++
    ) {
      const card =
        metadata.cards[cardIndex];

      let value = null;
      let debugBuffer = null;

      /*
       * Two slightly different crops are attempted.
       *
       * One works better for most cards, while the
       * narrower version avoids nearby artwork or
       * stat bubbles on certain cards.
       */
      const variants = [
        {
          width: 0.28,
          height: 0.34,
          threshold: 200,
        },
        {
          width: 0.24,
          height: 0.32,
          threshold: 185,
        },
      ];

      for (
        const variant
        of variants
      ) {
        const rect = safeRect(
          {
            left:
              card.x +
              card.width * 0.02,

            top:
              card.y +
              card.height * 0.60,

            width:
              card.width *
              variant.width,

            height:
              card.height *
              variant.height,
          },
          imageWidth,
          imageHeight
        );

        const buffer =
          await preprocessCopies(
            source,
            rect,
            variant.threshold
          );

        const result =
          await worker.recognize(
            buffer
          );

        value = parseCopies(
          result.data.text
        );

        if (!debugBuffer) {
          debugBuffer = buffer;
        }

        if (value !== null) {
          debugBuffer = buffer;
          break;
        }
      }

      copies.push(value);

      if (DEBUG_OUTPUT && debugBuffer) {
        await fs.writeFile(
          path.join(
            DEBUG_DIR,
            `copies_${String(
              cardIndex + 1
            ).padStart(
              2,
              "0"
            )}.png`
          ),
          debugBuffer
        );
      }
    }

    /*
     * --------------------------------------------------
     * 3. CARD COSTS
     * --------------------------------------------------
     */

    await worker.setParameters({
      tessedit_pageseg_mode:
        PSM.SINGLE_CHAR,

      tessedit_char_whitelist:
        "0123456789",

      user_defined_dpi:
        "300",
    });

    const costs = [];

    for (
      let cardIndex = 0;
      cardIndex <
      metadata.cards.length;
      cardIndex++
    ) {
      const card =
        metadata.cards[cardIndex];

      /*
       * The sun or brain cost bubble sits in the
       * upper-right portion of each detected card.
       */
      const rect = safeRect(
        {
          left:
            card.x +
            card.width * 0.68,

          top:
            card.y,

          width:
            card.width * 0.30,

          height:
            card.height * 0.40,
        },
        imageWidth,
        imageHeight
      );

      const readings = [];
      const debugImages =
        new Map();

      /*
       * Read the isolated digit using several
       * darkness thresholds.
       *
       * Stop when the same value has been seen twice.
       */
      for (
        const threshold
        of [70, 85, 100, 115, 130]
      ) {
        const buffer =
          await isolateCostDigits(
            source,
            rect,
            threshold
          );

        if (!buffer) {
          continue;
        }

        const result =
          await worker.recognize(
            buffer
          );

        const value =
          parseCost(
            result.data.text
          );

        if (value === null) {
          continue;
        }

        readings.push(value);

        if (
          !debugImages.has(value)
        ) {
          debugImages.set(
            value,
            buffer
          );
        }

        const count =
          readings.filter(
            reading =>
              reading === value
          ).length;

        if (count >= 2) {
          break;
        }
      }

      const value =
        mostCommon(readings);

      costs.push(value);

      const debugImage =
        debugImages.get(value);

      if (DEBUG_OUTPUT && debugImage) {
        await fs.writeFile(
          path.join(
            DEBUG_DIR,
            `cost_${String(
              cardIndex + 1
            ).padStart(
              2,
              "0"
            )}.png`
          ),
          debugImage
        );
      }
    }

    /*
     * --------------------------------------------------
     * 4. VALIDATION
     * --------------------------------------------------
     */

    const allCopiesRead =
      copies.every(
        Number.isInteger
      );

    const allCostsRead =
      costs.every(
        Number.isInteger
      );

    const totalCopies =
      allCopiesRead
        ? copies.reduce(
            (
              total,
              value
            ) =>
              total + value,
            0
          )
        : null;

    const costsNonDecreasing =
      allCostsRead &&
      costs.every(
        (
          cost,
          index
        ) =>
          index === 0 ||
          cost >=
            costs[index - 1]
      );

    const cards =
      metadata.cards.map(
        (
          card,
          index
        ) => ({
          index:
            index + 1,

          row:
            card.row,

          column:
            card.column,

          filename:
            card.filename,

          copies:
            copies[index],

          cost:
            costs[index],
        })
      );

    const output = {
      input:
        path.basename(INPUT),

      deckName,

      distinctCards:
        cards.length,

      copies,

      totalCopies,

      copyTotalIs40:
        totalCopies === 40,

      costs,

      costsNonDecreasing,

      cards,
    };

    await fs.writeFile(
      OUTPUT_FILE,
      JSON.stringify(
        output,
        null,
        2
      )
    );

    console.log(
      `Deck name: ${
        deckName ||
        "(not read)"
      }`
    );

    console.log(
      `Copies:    ${copies.join(
        ", "
      )}`
    );

    console.log(
      `Total:     ${
        totalCopies ??
        "(incomplete)"
      } ${
        totalCopies === 40
          ? "✓"
          : "✗"
      }`
    );

    console.log(
      `Costs:     ${costs.join(
        ", "
      )}`
    );

    console.log(
      `Sorted:    ${
        costsNonDecreasing
          ? "yes ✓"
          : "no ✗"
      }`
    );

    console.log(
      `Saved:     ${OUTPUT_FILE}`
    );
  } finally {
    await worker.terminate();
  }
}

main().catch(error => {
  console.error(
    `\nDeck reading failed: ${error.message}`
  );

  process.exitCode = 1;
});