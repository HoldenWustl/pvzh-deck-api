const sharp = require("sharp");
const fs = require("fs/promises");
const path = require("path");
const {
  createWorker,
  PSM
} = require("tesseract.js");

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
    height: bottom - top
  };
}

function cleanName(text) {
  return String(text || "")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^[^A-Za-z0-9]+/, "")
    .replace(/[^A-Za-z0-9'&:+!?().\- ]+$/, "")
    .replace(/\s+40(?:\s*\/\s*40)?$/, "")
    .replace(/\s+[()]+$/, "")
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
    raw: source.raw
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
      kernel: sharp.kernel.cubic
    })
    .png({
      compressionLevel: 1
    })
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
      kernel: sharp.kernel.cubic
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
        alpha: 1
      }
    })
    .png({
      compressionLevel: 1
    })
    .toBuffer();
}

async function preprocessTightDigit(
  source,
  rect,
  threshold,
  invert
) {
  let pipeline = sharpFromRawSource(source)
    .extract(rect)
    .grayscale()
    .threshold(threshold);

  if (invert) {
    pipeline = pipeline.negate();
  }

  return pipeline
    .resize({
      width: Math.max(
        80,
        rect.width * 8
      ),
      kernel: sharp.kernel.cubic
    })
    .extend({
      top: 24,
      bottom: 24,
      left: 24,
      right: 24,
      background: {
        r: 255,
        g: 255,
        b: 255,
        alpha: 1
      }
    })
    .png({
      compressionLevel: 1
    })
    .toBuffer();
}
async function isolateCopyDigit(
  source,
  rect,
  threshold
) {
  const {
    data,
    info
  } = await sharpFromRawSource(source)
    .extract(rect)
    .raw()
    .toBuffer({
      resolveWithObject: true
    });

  const {
    width,
    height,
    channels
  } = info;

  /*
   * The copy badge contains white text on a dark background.
   * Mark bright pixels as foreground.
   */
  const binary =
    new Uint8Array(width * height);

  for (
    let index = 0;
    index < binary.length;
    index++
  ) {
    const offset =
      index * channels;

    const red = data[offset];
    const green = data[offset + 1];
    const blue = data[offset + 2];

    const grayscale =
      0.299 * red +
      0.587 * green +
      0.114 * blue;

    binary[index] =
      grayscale > threshold
        ? 1
        : 0;
  }

  const minimumArea = Math.max(
    5,
    Math.round(
      width * height * 0.002
    )
  );

  /*
   * The x is on the left of the badge.
   * The number is the large white component on the right.
   */
  const digitComponents = findComponents(
    binary,
    width,
    height
  ).filter(component => {
    const centerX =
      component.cx / width;

    const centerY =
      component.cy / height;

    return (
      component.area >= minimumArea &&
      component.height >= height * 0.25 &&
      component.height <= height * 0.90 &&
      component.width <= width * 0.55 &&
      centerX >= 0.43 &&
      centerX <= 0.94 &&
      centerY >= 0.12 &&
      centerY <= 0.90
    );
  });

  if (digitComponents.length === 0) {
    return null;
  }

  /*
   * Usually this contains one component. Keeping all plausible
   * right-side components also handles a digit broken into pieces
   * by antialiasing or screenshot compression.
   */
  const minimumX = Math.min(
    ...digitComponents.map(
      component => component.x
    )
  );

  const minimumY = Math.min(
    ...digitComponents.map(
      component => component.y
    )
  );

  const maximumX = Math.max(
    ...digitComponents.map(
      component =>
        component.x +
        component.width -
        1
    )
  );

  const maximumY = Math.max(
    ...digitComponents.map(
      component =>
        component.y +
        component.height -
        1
    )
  );

  const digitWidth =
    maximumX - minimumX + 1;

  const digitHeight =
    maximumY - minimumY + 1;

  const aspectRatio =
    digitWidth / digitHeight;

  const padding = 12;

  const canvasWidth =
    digitWidth + padding * 2;

  const canvasHeight =
    digitHeight + padding * 2;

  /*
   * Tesseract works best with a black digit
   * on a clean white background.
   */
  const canvas = Buffer.alloc(
    canvasWidth * canvasHeight,
    255
  );

  for (
    const component
    of digitComponents
  ) {
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
        outputY *
        canvasWidth +
        outputX
      ] = 0;
    }
  }

  const buffer = await sharp(
    canvas,
    {
      raw: {
        width: canvasWidth,
        height: canvasHeight,
        channels: 1
      }
    }
  )
    .resize({
      width: 90,
      height: 130,
      fit: "contain",
      kernel: sharp.kernel.nearest
    })
    .png({
      compressionLevel: 1
    })
    .toBuffer();

  return {
    buffer,
    aspectRatio
  };
}
function ocrConfidence(result) {
  const confidence = Number(
    result?.data?.confidence
  );

  return Number.isFinite(confidence)
    ? confidence
    : 0;
}

function collapseCandidates(
  candidates,
  minimumValue,
  maximumValue
) {
  const groups = new Map();

  for (const candidate of candidates) {
    if (
      !Number.isInteger(candidate.value) ||
      candidate.value < minimumValue ||
      candidate.value > maximumValue
    ) {
      continue;
    }

    let group = groups.get(candidate.value);

    if (!group) {
      group = {
        votes: 0,
        best: candidate
      };

      groups.set(candidate.value, group);
    }

    group.votes++;

    if (candidate.score > group.best.score) {
      group.best = candidate;
    }
  }

  return [...groups.values()]
    .map(group => ({
      ...group.best,

      /*
       * Repeated agreement across different thresholds/crops
       * is useful evidence, but confidence still matters most.
       */
      score:
        group.best.score +
        Math.min(group.votes - 1, 4) * 6,

      votes: group.votes
    }))
    .sort(
      (first, second) =>
        second.score - first.score
    );
}

function bestCandidate(
  candidates,
  minimumValue,
  maximumValue
) {
  return (
    collapseCandidates(
      candidates,
      minimumValue,
      maximumValue
    )[0] || null
  );
}

function chooseCopiesTotalingForty(
  candidateLists
) {
  let states = new Map();

  states.set(0, {
    score: 0,
    values: [],
    buffers: [],
    selected: []
  });

  for (const candidates of candidateLists) {
    const choices = collapseCandidates(
      candidates,
      1,
      4
    );

    if (choices.length === 0) {
      return null;
    }

    const nextStates = new Map();

    for (const [total, state] of states) {
      for (const candidate of choices) {
        const nextTotal =
          total + candidate.value;

        if (nextTotal > 40) {
          continue;
        }

        const nextScore =
          state.score + candidate.score;

        const existing =
          nextStates.get(nextTotal);

        if (
          !existing ||
          nextScore > existing.score
        ) {
          nextStates.set(
            nextTotal,
            {
              score: nextScore,

              values: [
                ...state.values,
                candidate.value
              ],

              buffers: [
                ...state.buffers,
                candidate.buffer
              ],

              selected: [
                ...state.selected,
                candidate
              ]
            }
          );
        }
      }
    }

    states = nextStates;
  }

  return states.get(40) || null;
}

function valuesAreNonDecreasing(values) {
  return (
    values.every(Number.isInteger) &&
    values.every(
      (value, index) =>
        index === 0 ||
        value >= values[index - 1]
    )
  );
}

function chooseNonDecreasingCosts(
  candidateLists
) {
  let states = new Map();

  states.set(-1, {
    score: 0,
    values: [],
    buffers: [],
    selected: []
  });

  for (const candidates of candidateLists) {
    const choices = collapseCandidates(
      candidates,
      0,
      20
    );

    if (choices.length === 0) {
      return null;
    }

    const nextStates = new Map();

    for (const candidate of choices) {
      let bestPrevious = null;

      for (const [lastValue, state] of states) {
        if (lastValue > candidate.value) {
          continue;
        }

        const nextScore =
          state.score + candidate.score;

        if (
          !bestPrevious ||
          nextScore > bestPrevious.score
        ) {
          bestPrevious = {
            score: nextScore,

            values: [
              ...state.values,
              candidate.value
            ],

            buffers: [
              ...state.buffers,
              candidate.buffer
            ],

            selected: [
              ...state.selected,
              candidate
            ]
          };
        }
      }

      if (!bestPrevious) {
        continue;
      }

      const existing =
        nextStates.get(candidate.value);

      if (
        !existing ||
        bestPrevious.score > existing.score
      ) {
        nextStates.set(
          candidate.value,
          bestPrevious
        );
      }
    }

    states = nextStates;

    if (states.size === 0) {
      return null;
    }
  }

  return (
    [...states.values()].sort(
      (first, second) =>
        second.score - first.score
    )[0] || null
  );
}

function findComponents(
  binary,
  width,
  height
) {
  const seen =
    new Uint8Array(binary.length);

  const queue =
    new Int32Array(binary.length);

  const result = [];

  for (
    let start = 0;
    start < binary.length;
    start++
  ) {
    if (
      !binary[start] ||
      seen[start]
    ) {
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
      const index =
        queue[head++];

      const x =
        index % width;

      const y =
        Math.floor(index / width);

      pixels.push(index);

      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);

      sumX += x;
      sumY += y;

      for (
        let dy = -1;
        dy <= 1;
        dy++
      ) {
        for (
          let dx = -1;
          dx <= 1;
          dx++
        ) {
          if (
            dx === 0 &&
            dy === 0
          ) {
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
      width:
        maxX - minX + 1,
      height:
        maxY - minY + 1,
      cx:
        sumX / area,
      cy:
        sumY / area
    });
  }

  return result;
}

async function isolateCostDigits(
  source,
  rect,
  threshold
) {
  const {
    data,
    info
  } = await sharpFromRawSource(source)
    .extract(rect)
    .raw()
    .toBuffer({
      resolveWithObject: true
    });

  const {
    width,
    height,
    channels
  } = info;

  const binary =
    new Uint8Array(width * height);

  for (
    let index = 0;
    index < binary.length;
    index++
  ) {
    const offset =
      index * channels;

    const red = data[offset];
    const green = data[offset + 1];
    const blue = data[offset + 2];

    const grayscale =
      0.299 * red +
      0.587 * green +
      0.114 * blue;

    binary[index] =
      grayscale < threshold
        ? 1
        : 0;
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
            centrality
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

  const second = candidates
    .slice(1)
    .find(component => {
      const verticalDistance =
        Math.abs(
          component.cy -
          primary.cy
        );

      const primaryRight =
        primary.x +
        primary.width;

      const componentRight =
        component.x +
        component.width;

      const horizontalGap =
        Math.max(
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
        x -
        minimumX +
        padding;

      const outputY =
        y -
        minimumY +
        padding;

      canvas[
        outputY *
        canvasWidth +
        outputX
      ] = 0;
    }
  }

  return sharp(canvas, {
    raw: {
      width: canvasWidth,
      height: canvasHeight,
      channels: 1
    }
  })
    .resize({
      width: 90,
      height: 130,
      fit: "contain",
      kernel: sharp.kernel.nearest
    })
    .png({
      compressionLevel: 1
    })
    .toBuffer();
}

async function main() {
  const metadata = JSON.parse(
    await fs.readFile(
      META_FILE,
      "utf8"
    )
  );

  const {
    data: sourceData,
    info: sourceInfo
  } = await sharp(INPUT)
    .removeAlpha()
    .raw()
    .toBuffer({
      resolveWithObject: true
    });

  const imageWidth =
    sourceInfo.width;

  const imageHeight =
    sourceInfo.height;

  if (
    !imageWidth ||
    !imageHeight
  ) {
    throw new Error(
      "Could not read the image size."
    );
  }

  const source = {
    data: sourceData,

    raw: {
      width:
        sourceInfo.width,

      height:
        sourceInfo.height,

      channels:
        sourceInfo.channels
    }
  };

  if (DEBUG_OUTPUT) {
    await fs.mkdir(
      DEBUG_DIR,
      {
        recursive: true
      }
    );
  }

  const worker = await createWorker(
    "eng",
    1,
    {
      langPath:
        englishData.langPath,

      gzip:
        englishData.gzip
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
          imageHeight * 0.105
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
        "300"
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

    if (
      DEBUG_OUTPUT &&
      nameDebugImage
    ) {
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
        "300"
    });

    const copyCandidates = [];
    const copies = [];
    const copyDebugBuffers = [];

    for (
      let cardIndex = 0;
      cardIndex <
      metadata.cards.length;
      cardIndex++
    ) {
      const card =
        metadata.cards[cardIndex];

      const candidates = [];
      let firstDebugBuffer = null;

      /*
       * Do not stop at the first valid OCR result.
       * A valid-looking result can still be wrong, as with x3 -> 1.
       */
      const variants = [
        {
          width: 0.28,
          height: 0.34,
          threshold: 200
        },
        {
          width: 0.24,
          height: 0.32,
          threshold: 185
        }
      ];

      for (const variant of variants) {
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
              variant.height
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

        if (!firstDebugBuffer) {
          firstDebugBuffer = buffer;
        }

        const result =
          await worker.recognize(
            buffer
          );

        const value =
          parseCopies(
            result.data.text
          );

        if (value !== null) {
          candidates.push({
            value,
            score: ocrConfidence(result),
            buffer,
            source: "badge"
          });
        }
      }

      const initial = bestCandidate(
        candidates,
        1,
        4
      );

      copyCandidates.push(candidates);
      copies.push(initial?.value ?? null);
      copyDebugBuffers.push(
        initial?.buffer ?? firstDebugBuffer
      );
    }

    const initialCopyTotal =
  copies.every(Number.isInteger)
    ? copies.reduce(
        (total, value) =>
          total + value,
        0
      )
    : null;

let repairedCopies = null;

/*
 * Whenever the initial OCR does not total 40,
 * always gather tight digit readings before attempting
 * to repair the deck.
 *
 * Do not accept an arbitrary 40-card combination from
 * the weaker broad OCR candidates.
 */
if (initialCopyTotal !== 40) {
  await worker.setParameters({
    tessedit_pageseg_mode:
      PSM.SINGLE_CHAR,

    tessedit_char_whitelist:
      "1234",

    user_defined_dpi:
      "300"
  });

  for (
    let cardIndex = 0;
    cardIndex <
    metadata.cards.length;
    cardIndex++
  ) {
    const card =
      metadata.cards[cardIndex];

    /*
     * Capture the whole xN badge. isolateCopyDigit()
     * finds the digit inside it, so exact digit coordinates
     * are no longer hard-coded.
     */
    const badgeRect = safeRect(
      {
        left:
          card.x +
          card.width * 0.005,

        top:
          card.y +
          card.height * 0.585,

        width:
          card.width * 0.30,

        height:
          card.height * 0.385
      },
      imageWidth,
      imageHeight
    );

    const isolatedReadings = [];

    for (
      const threshold
      of [145, 165, 185, 205, 225]
    ) {
      const isolated =
        await isolateCopyDigit(
          source,
          badgeRect,
          threshold
        );

      if (!isolated) {
        continue;
      }

      const result =
        await worker.recognize(
          isolated.buffer
        );

      const value =
        parseCopies(
          result.data.text
        );

      if (value === null) {
        continue;
      }

      /*
       * A real 1 is narrow. If Tesseract calls a visibly
       * wide component "1", reject that reading. This
       * specifically protects against clear 3/4 glyphs
       * being mistaken for 1.
       */
      if (
        value === 1 &&
        isolated.aspectRatio > 0.46
      ) {
        continue;
      }

      isolatedReadings.push({
        value,

        score:
          ocrConfidence(result) + 35,

        buffer:
          isolated.buffer,

        source:
          "isolated-digit"
      });
    }

    copyCandidates[
      cardIndex
    ].push(
      ...isolatedReadings
    );

    if (DEBUG_OUTPUT) {
      const summarized =
        collapseCandidates(
          copyCandidates[
            cardIndex
          ],
          1,
          4
        ).map(candidate => ({
          value:
            candidate.value,

          score:
            Math.round(
              candidate.score
            ),

          votes:
            candidate.votes,

          source:
            candidate.source
        }));

      console.log(
        `Copy ${cardIndex + 1} candidates:`,
        summarized
      );
    }
  }

  repairedCopies =
    chooseCopiesTotalingForty(
      copyCandidates
    );
}

    if (repairedCopies) {
      const changedPositions = [];

      repairedCopies.values.forEach(
        (value, index) => {
          if (copies[index] !== value) {
            changedPositions.push(
              `${index + 1}: ${copies[index]} -> ${value}`
            );
          }
        }
      );

      copies.splice(
        0,
        copies.length,
        ...repairedCopies.values
      );

      copyDebugBuffers.splice(
        0,
        copyDebugBuffers.length,
        ...repairedCopies.buffers
      );

      if (changedPositions.length > 0) {
        console.log(
          "Copy OCR repaired using multiple readings and the 40-card total: " +
          changedPositions.join(", ")
        );
      }
    }

    if (DEBUG_OUTPUT) {
      for (
        let cardIndex = 0;
        cardIndex <
        copyDebugBuffers.length;
        cardIndex++
      ) {
        const buffer =
          copyDebugBuffers[cardIndex];

        if (!buffer) {
          continue;
        }

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
          buffer
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
        "300"
    });

    const costCandidates = [];
    const costs = [];
    const costDebugBuffers = [];

    for (
      let cardIndex = 0;
      cardIndex <
      metadata.cards.length;
      cardIndex++
    ) {
      const card =
        metadata.cards[cardIndex];

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
            card.height * 0.40
        },
        imageWidth,
        imageHeight
      );

      const candidates = [];
      let firstAttemptBuffer = null;

      /*
       * Keep all readings. Two repeated wrong readings must not
       * permanently win before the sorted-cost rule is considered.
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

        if (!firstAttemptBuffer) {
          firstAttemptBuffer = buffer;
        }

        const result =
          await worker.recognize(
            buffer
          );

        const value =
          parseCost(
            result.data.text
          );

        if (value !== null) {
          candidates.push({
            value,
            score: ocrConfidence(result),
            buffer,
            source: "component"
          });
        }
      }

      const initial = bestCandidate(
        candidates,
        0,
        20
      );

      costCandidates.push(candidates);
      costs.push(initial?.value ?? null);
      costDebugBuffers.push(
        initial?.buffer ?? firstAttemptBuffer
      );
    }

    const initialCostsWereSorted =
      valuesAreNonDecreasing(costs);

    if (!initialCostsWereSorted) {
      const suspiciousIndices = new Set();

      for (
        let index = 0;
        index < costs.length;
        index++
      ) {
        const value = costs[index];

        if (!Number.isInteger(value)) {
          suspiciousIndices.add(index);
          continue;
        }

        if (
          index > 0 &&
          Number.isInteger(costs[index - 1]) &&
          value < costs[index - 1]
        ) {
          suspiciousIndices.add(index - 1);
          suspiciousIndices.add(index);
        }

        if (
          index + 1 < costs.length &&
          Number.isInteger(costs[index + 1]) &&
          value > costs[index + 1]
        ) {
          suspiciousIndices.add(index);
          suspiciousIndices.add(index + 1);
        }
      }

      /*
       * Challenge any impossible/non-monotone reading with a direct,
       * tighter crop. Try both SINGLE_CHAR and SINGLE_WORD because
       * Tesseract occasionally calls this game's blocky 4 a 0.
       */
      for (const pageSegMode of [
        PSM.SINGLE_CHAR,
        PSM.SINGLE_WORD
      ]) {
        await worker.setParameters({
          tessedit_pageseg_mode:
            pageSegMode,

          tessedit_char_whitelist:
            "0123456789",

          user_defined_dpi:
            "300"
        });

        for (const cardIndex of suspiciousIndices) {
          const card =
            metadata.cards[cardIndex];

          const tightRect = safeRect(
            {
              left:
                card.x +
                card.width * 0.745,

              top:
                card.y +
                card.height * 0.015,

              width:
                card.width * 0.23,

              height:
                card.height * 0.34
            },
            imageWidth,
            imageHeight
          );

          const votes = new Map();

          for (
            const threshold
            of [90, 110, 130, 150, 170, 190, 210]
          ) {
            const buffer =
              await preprocessTightDigit(
                source,
                tightRect,
                threshold,
                false
              );

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

            costCandidates[
              cardIndex
            ].push({
              value,
              score:
                ocrConfidence(result) + 3,
              buffer,
              source:
                pageSegMode === PSM.SINGLE_CHAR
                  ? "tight-char"
                  : "tight-word"
            });

            const count =
              (votes.get(value) || 0) + 1;

            votes.set(value, count);

            if (count >= 2) {
              break;
            }
          }
        }
      }
    }

    const repairedCosts =
      chooseNonDecreasingCosts(
        costCandidates
      );

    if (repairedCosts) {
      const changedPositions = [];

      repairedCosts.values.forEach(
        (value, index) => {
          if (costs[index] !== value) {
            changedPositions.push(
              `${index + 1}: ${costs[index]} -> ${value}`
            );
          }
        }
      );

      costs.splice(
        0,
        costs.length,
        ...repairedCosts.values
      );

      costDebugBuffers.splice(
        0,
        costDebugBuffers.length,
        ...repairedCosts.buffers
      );

      if (changedPositions.length > 0) {
        console.log(
          "Cost OCR repaired using multiple readings and sorted deck order: " +
          changedPositions.join(", ")
        );
      }
    }

    if (DEBUG_OUTPUT) {
      for (
        let cardIndex = 0;
        cardIndex <
        costDebugBuffers.length;
        cardIndex++
      ) {
        const buffer =
          costDebugBuffers[cardIndex];

        if (!buffer) {
          continue;
        }

        const suffix =
          Number.isInteger(costs[cardIndex])
            ? ""
            : "_FAILED";

        await fs.writeFile(
          path.join(
            DEBUG_DIR,
            `cost_${String(
              cardIndex + 1
            ).padStart(
              2,
              "0"
            )}${suffix}.png`
          ),
          buffer
        );
      }
    }

    /*
     * --------------------------------------------------
     * 4. VALIDATION AND OUTPUT
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
            (total, value) =>
              total + value,
            0
          )
        : null;

    const costsNonDecreasing =
      allCostsRead &&
      costs.every(
        (cost, index) =>
          index === 0 ||
          cost >=
            costs[index - 1]
      );

    const cards =
      metadata.cards.map(
        (card, index) => ({
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
            costs[index]
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

      cards
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