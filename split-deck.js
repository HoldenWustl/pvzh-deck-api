const sharp = require("sharp");
const fs = require("fs/promises");
const path = require("path");

const INPUT = process.argv[2] || "deck.png";
const OUTPUT_DIR = process.argv[3] || "cards";
const DEBUG_OUTPUT = process.env.PVZH_DEBUG === "1";

const clamp = (value, min, max) =>
  Math.max(min, Math.min(max, value));

function median(values) {
  if (values.length === 0) return 0;

  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);

  if (sorted.length % 2 === 1) {
    return sorted[middle];
  }

  return (sorted[middle - 1] + sorted[middle]) / 2;
}

function findRuns(values, predicate, minLength = 1, mergeGap = 0) {
  const runs = [];

  let start = -1;
  let last = -1;

  for (let i = 0; i < values.length; i++) {
    if (predicate(values[i], i)) {
      if (start < 0) start = i;
      last = i;
    } else if (start >= 0) {
      runs.push({
        start,
        end: last,
      });

      start = -1;
      last = -1;
    }
  }

  if (start >= 0) {
    runs.push({
      start,
      end: last,
    });
  }

  // Merge small breaks caused by image compression or anti-aliasing.
  const merged = [];

  for (const run of runs) {
    const previous = merged[merged.length - 1];

    if (
      previous &&
      run.start - previous.end - 1 <= mergeGap
    ) {
      previous.end = run.end;
    } else {
      merged.push({ ...run });
    }
  }

  return merged.filter(
    run => run.end - run.start + 1 >= minLength
  );
}

function findDeckDivider(data, width, height, channels) {
  /*
   * PvZ Heroes has a bright cyan horizontal separator immediately
   * above the card area.
   *
   * We search for the lowest long cyan row in the upper 70%
   * of the screenshot.
   */

  let bestY = -1;
  const maxY = Math.floor(height * 0.7);

  for (let y = 0; y < maxY; y++) {
    let cyanPixelCount = 0;

    for (let x = 0; x < width; x++) {
      const index = (y * width + x) * channels;

      const r = data[index];
      const g = data[index + 1];
      const b = data[index + 2];

      const looksCyan =
        r < 100 &&
        g > 140 &&
        b > 140 &&
        Math.abs(g - b) < 100;

      if (looksCyan) {
        cyanPixelCount++;
      }
    }

    if (cyanPixelCount >= width * 0.35) {
      bestY = y;
    }
  }

  if (bestY < 0) {
    throw new Error(
      "Could not find the cyan divider above the card grid."
    );
  }

  return bestY + 1;
}

function dominantBackground(
  data,
  width,
  height,
  channels,
  deckTop
) {
  /*
   * The empty deck area is mostly one blue background color.
   *
   * We quantize colors into buckets and choose the most common
   * color below the cyan divider.
   */

  const counts = new Map();
  const sampleStep = 2;

  for (let y = deckTop; y < height; y += sampleStep) {
    for (let x = 0; x < width; x += sampleStep) {
      const index = (y * width + x) * channels;

      const quantizedR = data[index] >> 2;
      const quantizedG = data[index + 1] >> 2;
      const quantizedB = data[index + 2] >> 2;

      const key =
        (quantizedR << 12) |
        (quantizedG << 6) |
        quantizedB;

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

function buildMasks(
  data,
  width,
  height,
  channels,
  deckTop,
  background
) {
  const deckHeight = height - deckTop;

  const foreground = new Uint8Array(width * deckHeight);
  const edges = new Uint8Array(width * deckHeight);
  const grayscale = new Float32Array(width * deckHeight);

  const colorDistanceThresholdSquared = 28 * 28;

  for (let y = 0; y < deckHeight; y++) {
    for (let x = 0; x < width; x++) {
      const sourceIndex =
        ((deckTop + y) * width + x) * channels;

      const destinationIndex = y * width + x;

      const r = data[sourceIndex];
      const g = data[sourceIndex + 1];
      const b = data[sourceIndex + 2];

      const differenceR = r - background.r;
      const differenceG = g - background.g;
      const differenceB = b - background.b;

      const colorDistanceSquared =
        differenceR * differenceR +
        differenceG * differenceG +
        differenceB * differenceB;

      foreground[destinationIndex] =
        colorDistanceSquared >
        colorDistanceThresholdSquared
          ? 1
          : 0;

      grayscale[destinationIndex] =
        0.299 * r +
        0.587 * g +
        0.114 * b;
    }
  }

  /*
   * Cards contain lots of visual edges.
   * The empty blue background contains very few.
   */

  for (let y = 1; y < deckHeight; y++) {
    for (let x = 1; x < width; x++) {
      const index = y * width + x;

      const difference =
        Math.abs(
          grayscale[index] - grayscale[index - 1]
        ) +
        Math.abs(
          grayscale[index] - grayscale[index - width]
        );

      edges[index] = difference > 30 ? 1 : 0;
    }
  }

  return {
    foreground,
    edges,
    deckHeight,
  };
}

function inferGrid(
  foreground,
  edges,
  width,
  deckHeight
) {
  /*
   * First, detect horizontal bands occupied by cards.
   */

  const rowScores = new Float32Array(deckHeight);

  for (let y = 0; y < deckHeight; y++) {
    let foregroundPixels = 0;
    const offset = y * width;

    for (let x = 0; x < width; x++) {
      foregroundPixels += foreground[offset + x];
    }

    rowScores[y] = foregroundPixels / width;
  }

  const rowBands = findRuns(
    rowScores,
    score => score > 0.5,
    Math.max(12, Math.floor(deckHeight * 0.03)),
    2
  );

  if (rowBands.length < 2) {
    throw new Error(
      `Only found ${rowBands.length} solid card row(s). ` +
      "At least two are needed to infer the grid."
    );
  }

  /*
   * Use the first full row to infer the columns.
   */

  const firstBand = rowBands[0];
  const firstBandHeight =
    firstBand.end - firstBand.start + 1;

  const innerTop =
    firstBand.start +
    Math.floor(firstBandHeight * 0.15);

  const innerBottom =
    firstBand.end -
    Math.floor(firstBandHeight * 0.15);

  const columnScores = new Float32Array(width);

  for (let x = 0; x < width; x++) {
    let foregroundPixels = 0;
    let pixelCount = 0;

    for (let y = innerTop; y <= innerBottom; y++) {
      foregroundPixels += foreground[y * width + x];
      pixelCount++;
    }

    columnScores[x] =
      foregroundPixels / pixelCount;
  }

  const columnRuns = findRuns(
    columnScores,
    score => score > 0.5,
    Math.max(20, Math.floor(width * 0.08)),
    3
  );

  if (columnRuns.length < 2) {
    throw new Error(
      `Only found ${columnRuns.length} card column(s).`
    );
  }

  /*
   * Put each crop boundary halfway through the gaps
   * between adjacent card columns.
   */

  const columnGaps = [];

  for (let i = 0; i < columnRuns.length - 1; i++) {
    columnGaps.push(
      columnRuns[i + 1].start -
      columnRuns[i].end -
      1
    );
  }

  const typicalGap = Math.max(
    2,
    Math.round(median(columnGaps))
  );

  const xBoundaries = [
    clamp(
      columnRuns[0].start -
      Math.floor(typicalGap / 2),
      0,
      width
    ),
  ];

  for (let i = 0; i < columnRuns.length - 1; i++) {
    const boundary = Math.floor(
      (
        columnRuns[i].end +
        columnRuns[i + 1].start +
        1
      ) / 2
    );

    xBoundaries.push(boundary);
  }

  xBoundaries.push(
    clamp(
      columnRuns[columnRuns.length - 1].end +
      Math.ceil(typicalGap / 2) +
      1,
      0,
      width
    )
  );

  /*
   * Infer the vertical distance between rows.
   */

  const rowStarts = rowBands.map(row => row.start);

  const rowPitches = rowStarts
    .slice(1)
    .map(
      (rowStart, index) =>
        rowStart - rowStarts[index]
    );

  const rowPitch = Math.round(
    median(rowPitches)
  );

  const visibleRowHeight = median(
    rowBands.map(
      row => row.end - row.start + 1
    )
  );

  const verticalGap = Math.max(
    2,
    rowPitch - visibleRowHeight
  );

  const firstRowTop = clamp(
    Math.round(
      firstBand.start - verticalGap * 0.75
    ),
    0,
    deckHeight - 1
  );

  const cards = [];
  let foundPreviousRow = false;

  /*
   * Check every possible grid cell.
   *
   * A cell is considered occupied when it contains enough
   * edge detail. Empty blue cells contain almost none.
   */

  for (
    let row = 0, y1 = firstRowTop;
    y1 < deckHeight;
    row++, y1 += rowPitch
  ) {
    const y2 = Math.min(
      deckHeight,
      y1 + rowPitch
    );

    if (y2 - y1 < rowPitch * 0.55) {
      break;
    }

    let cardsInThisRow = 0;

    for (
      let column = 0;
      column < xBoundaries.length - 1;
      column++
    ) {
      const x1 = xBoundaries[column];
      const x2 = xBoundaries[column + 1];

      const paddingX = Math.max(
        2,
        Math.floor((x2 - x1) * 0.05)
      );

      const paddingY = Math.max(
        2,
        Math.floor((y2 - y1) * 0.05)
      );

      let edgePixels = 0;
      let pixelCount = 0;

      for (
        let y = y1 + paddingY;
        y < y2 - paddingY;
        y++
      ) {
        const offset = y * width;

        for (
          let x = x1 + paddingX;
          x < x2 - paddingX;
          x++
        ) {
          edgePixels += edges[offset + x];
          pixelCount++;
        }
      }

      const edgeDensity =
        pixelCount > 0
          ? edgePixels / pixelCount
          : 0;

      const occupied = edgeDensity > 0.06;

      if (!occupied) {
        continue;
      }

      cardsInThisRow++;

      cards.push({
        row: row + 1,
        column: column + 1,
        x: x1,
        y: y1,
        width: x2 - x1,
        height: y2 - y1,
        edgeDensity: Number(
          edgeDensity.toFixed(4)
        ),
      });
    }

    if (cardsInThisRow > 0) {
      foundPreviousRow = true;
    } else if (foundPreviousRow) {
      break;
    }
  }

  if (cards.length === 0) {
    throw new Error(
      "No occupied card cells were found."
    );
  }

  return {
    cards,
    rowBands,
    columnRuns,
    xBoundaries,
    rowPitch,
    firstRowTop,
  };
}

function escapeXml(value) {
  return String(value).replace(
    /[<>&'"]/g,
    character => ({
      "<": "&lt;",
      ">": "&gt;",
      "&": "&amp;",
      "'": "&apos;",
      '"': "&quot;",
    })[character]
  );
}

async function main() {
  await fs.rm(OUTPUT_DIR, {
    recursive: true,
    force: true,
  });

  await fs.mkdir(OUTPUT_DIR, {
    recursive: true,
  });

  const image = sharp(INPUT).removeAlpha();

  const {
    data,
    info,
  } = await image.raw().toBuffer({
    resolveWithObject: true,
  });

  const {
    width,
    height,
    channels,
  } = info;

  const deckTop = findDeckDivider(
    data,
    width,
    height,
    channels
  );

  const background = dominantBackground(
    data,
    width,
    height,
    channels,
    deckTop
  );

  const {
    foreground,
    edges,
    deckHeight,
  } = buildMasks(
    data,
    width,
    height,
    channels,
    deckTop,
    background
  );

  const grid = inferGrid(
    foreground,
    edges,
    width,
    deckHeight
  );

  /*
   * Save each detected card as an individual PNG.
   */

  for (
    let index = 0;
    index < grid.cards.length;
    index++
  ) {
    const card = grid.cards[index];

    const filename =
      `card_${String(index + 1).padStart(2, "0")}` +
      `_r${card.row}_c${card.column}.png`;

    await sharp(INPUT)
      .extract({
        left: card.x,
        top: deckTop + card.y,
        width: card.width,
        height: card.height,
      })
      .png({ compressionLevel: 1 })
      .toFile(
        path.join(OUTPUT_DIR, filename)
      );

    card.filename = filename;
  }

  /*
   * Save machine-readable information about every crop.
   */

  const metadata = {
    input: path.basename(INPUT),

    image: {
      width,
      height,
    },

    deckTop,
    detectedColumns: grid.columnRuns.length,
    rowPitch: grid.rowPitch,
    background,
    cardCount: grid.cards.length,

    cards: grid.cards.map(card => ({
      ...card,
      y: deckTop + card.y,
    })),
  };

  await fs.writeFile(
    path.join(OUTPUT_DIR, "cards.json"),
    JSON.stringify(metadata, null, 2)
  );

  if (DEBUG_OUTPUT) {
    /*
     * Create a debug image with green boxes around each card.
     */

    const rectangles = grid.cards
      .map((card, index) => {
        const y = deckTop + card.y;

        return `
          <rect
            x="${card.x}"
            y="${y}"
            width="${card.width}"
            height="${card.height}"
            fill="none"
            stroke="#00ff66"
            stroke-width="4"
          />

          <rect
            x="${card.x + 4}"
            y="${y + 4}"
            width="38"
            height="28"
            fill="#000000"
            fill-opacity="0.75"
          />

          <text
            x="${card.x + 23}"
            y="${y + 25}"
            text-anchor="middle"
            font-family="Arial"
            font-size="20"
            font-weight="bold"
            fill="#ffffff"
          >
            ${index + 1}
          </text>
        `;
      })
      .join("");

    const svg = Buffer.from(`
      <svg
        width="${width}"
        height="${height}"
        xmlns="http://www.w3.org/2000/svg"
      >
        ${rectangles}

        <text
          x="12"
          y="30"
          font-family="Arial"
          font-size="22"
          font-weight="bold"
          fill="#ffffff"
          stroke="#000000"
          stroke-width="3"
          paint-order="stroke"
        >
          ${escapeXml(
            `Detected ${grid.cards.length} cards`
          )}
        </text>
      </svg>
    `);

    await sharp(INPUT)
      .composite([
        {
          input: svg,
          top: 0,
          left: 0,
        },
      ])
      .png({ compressionLevel: 1 })
      .toFile(
        path.join(OUTPUT_DIR, "debug-grid.png")
      );
  }

  console.log(
    `Detected ${grid.cards.length} cards.`
  );

  console.log(
    `Saved crops to: ${path.resolve(OUTPUT_DIR)}`
  );

  if (DEBUG_OUTPUT) {
    console.log(
      `Check ${path.join(
        OUTPUT_DIR,
        "debug-grid.png"
      )} to verify the boxes.`
    );
  }
}

main().catch(error => {
  console.error(
    `\nDeck splitting failed: ${error.message}`
  );

  process.exitCode = 1;
});