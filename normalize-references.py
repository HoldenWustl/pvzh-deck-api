"""
Normalize reference card images into the canonical 200x150 frame.

    python normalize-references.py [SOURCE_DIR] [OUTPUT_DIR] [--rebuild-template]

Defaults: SOURCE_DIR=reference_cards_original, OUTPUT_DIR=reference_cards

What it does:
  1. (Re)builds card_template.npy from the source images if requested or
     missing. The template is the average edge map of the shared deck UI.
  2. Aligns every source image against the template and writes a
     canonical PNG with the same base name into OUTPUT_DIR.
  3. Writes OUTPUT_DIR/../reference_review/ with:
       report.json   alignment score, scale and art coverage per card
       flagged.txt   cards worth re-capturing, with the reason
       sheet_XX.jpg  contact sheets of every normalized card, with the
                     art mask outlined, for a quick visual check

Re-run it whenever you add or replace a source image.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

import card_vision as cv


SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

args = [a for a in sys.argv[1:] if not a.startswith("--")]
REBUILD = "--rebuild-template" in sys.argv
SOURCE_DIR = Path(args[0]) if len(args) > 0 else Path("reference_cards_original")
OUTPUT_DIR = Path(args[1]) if len(args) > 1 else Path("reference_cards")
REVIEW_DIR = OUTPUT_DIR.parent / "reference_review"

# Most source captures were 200x130 crops from the deck screen; they
# seed the template at this nominal position on the canvas.
SEED_SIZE = (200, 130)
SEED_ORIGIN = (30, 25)

# Review thresholds.
LOW_ALIGN_SCORE = 0.42
LOW_ART_COVERAGE = 0.97
HEAVY_UPSCALE = 1.35


def load(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read {path}")
    return image


def build_template(images: dict[str, np.ndarray]) -> np.ndarray:
    seeds = {
        name: image for name, image in images.items()
        if (image.shape[1], image.shape[0]) == SEED_SIZE
    }
    if len(seeds) < 50:
        raise ValueError("Not enough uniformly sized captures to seed the template.")

    edges = {name: cv.edge_map(image) for name, image in seeds.items()}
    positions = {name: SEED_ORIGIN for name in seeds}

    template = None
    for iteration in range(4):
        total = np.zeros((cv.CANVAS_H, cv.CANVAS_W), np.float32)
        count = np.zeros_like(total)

        for name, (x, y) in positions.items():
            e = edges[name]
            h, w = e.shape
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(cv.CANVAS_W, x + w), min(cv.CANVAS_H, y + h)
            total[y0:y1, x0:x1] += e[y0 - y:y1 - y, x0 - x:x1 - x]
            count[y0:y1, x0:x1] += 1

        template = total / np.maximum(count, 1)
        template[count < 30] = 0

        matcher = cv._Template(template)
        moved = 0
        new_positions = {}
        for name, e in edges.items():
            _, _, x, y = matcher.best(e, [1.0], 0.75)
            new_positions[name] = (int(x), int(y))
            moved += new_positions[name] != positions[name]
        positions = new_positions
        print(f"  template pass {iteration + 1}: {moved} captures re-positioned")

    return template.astype(np.float32)


def contact_sheets(rows: list[dict], canon: dict[str, np.ndarray]) -> None:
    cols, per_sheet = 8, 64
    contours, _ = cv2.findContours(cv.ART_MASK, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for sheet_index in range(0, len(rows), per_sheet):
        chunk = rows[sheet_index:sheet_index + per_sheet]
        tiles = []
        for row in chunk:
            tile = canon[row["file"]].copy()
            cv2.drawContours(tile, contours, -1, (0, 255, 0), 1)
            label = np.full((18, cv.CROP_W, 3), 255, np.uint8)
            colour = (0, 0, 200) if row["flags"] else (0, 0, 0)
            cv2.putText(label, row["name"][:30], (2, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, colour, 1, cv2.LINE_AA)
            tiles.append(np.vstack([tile, label]))
        while len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 255))
        grid = np.vstack([
            np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)
        ])
        out = REVIEW_DIR / f"sheet_{sheet_index // per_sheet + 1:02d}.jpg"
        cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> int:
    if not SOURCE_DIR.exists():
        print(f"Missing source directory {SOURCE_DIR}")
        return 1

    paths = sorted(p for p in SOURCE_DIR.iterdir() if p.suffix.lower() in SUPPORTED)
    images = {p.name: load(p) for p in paths}
    print(f"Loaded {len(images)} source images from {SOURCE_DIR}")

    if REBUILD or not cv.TEMPLATE_PATH.exists():
        print("Building card template...")
        np.save(cv.TEMPLATE_PATH, build_template(images), allow_pickle=False)
        cv._TEMPLATES.clear()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    canon: dict[str, np.ndarray] = {}
    art_pixels = float((cv.ART_MASK > 0).sum())

    for number, path in enumerate(paths, 1):
        image = images[path.name]
        canonical, coverage, alignment = cv.align_card(image)

        art_coverage = float(
            ((coverage > 0) & (cv.ART_MASK > 0)).sum() / art_pixels
        )

        flags = []
        if alignment.fallback:
            flags.append("alignment failed (centred instead)")
        elif alignment.score < LOW_ALIGN_SCORE:
            flags.append(f"weak alignment ({alignment.score:.2f})")
        if art_coverage < LOW_ART_COVERAGE:
            flags.append(f"art clipped ({art_coverage:.0%} of art visible)")
        if alignment.scale > HEAVY_UPSCALE:
            flags.append(
                f"low resolution (upscaled {alignment.scale:.2f}x; "
                f"source {image.shape[1]}x{image.shape[0]})"
            )

        # Alpha = which pixels came from the source image (the rest is
        # filled with deck background). The index builder masks on it.
        out_path = OUTPUT_DIR / f"{path.stem}.webp"
        cv2.imwrite(
            str(out_path),
            np.dstack([canonical, coverage]),
            [cv2.IMWRITE_WEBP_QUALITY, 95],  # alpha stays lossless
        )
        canon[path.name] = canonical

        rows.append({
            "file": path.name,
            "name": path.stem,
            "output": out_path.name,
            "source_size": [image.shape[1], image.shape[0]],
            "align_score": round(alignment.score, 3),
            "scale": round(alignment.scale, 3),
            "offset": [round(alignment.x, 1), round(alignment.y, 1)],
            "art_coverage": round(art_coverage, 3),
            "flags": flags,
        })

        if number % 50 == 0:
            print(f"  aligned {number}/{len(paths)}")

    with (REVIEW_DIR / "report.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)

    flagged = [r for r in rows if r["flags"]]
    with (REVIEW_DIR / "flagged.txt").open("w", encoding="utf-8") as f:
        for r in sorted(flagged, key=lambda r: r["name"].lower()):
            f.write(f"{r['file']}: {'; '.join(r['flags'])}\n")

    contact_sheets(rows, canon)

    print(f"Wrote {len(rows)} normalized cards to {OUTPUT_DIR}")
    print(f"{len(flagged)} flagged for review -> {REVIEW_DIR / 'flagged.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
