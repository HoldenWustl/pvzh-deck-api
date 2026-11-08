from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

import card_vision as cv


CARD_DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("card_data.json")
REFERENCE_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("reference_cards")
OUTPUT_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("reference_index")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Explicit aliases for the two filename typos shown in the deployment log.
FILENAME_ALIASES = {
    "cukoozombie": "Cuckoo_Zombie",
    "intergalaticwarlord": "Intergalactic_Warlord",
}


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def display_name(value: str) -> str:
    return str(value).replace("_", " ").strip()


def main() -> None:
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)

    if not CARD_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {CARD_DATA_PATH}")

    if not REFERENCE_DIR.exists():
        raise FileNotFoundError(f"Missing {REFERENCE_DIR}")

    with CARD_DATA_PATH.open("r", encoding="utf-8") as file:
        card_data = json.load(file)

    if not isinstance(card_data, dict):
        raise ValueError("card_data.json must be an object keyed by card name.")

    name_lookup: dict[str, str] = {}

    for card_id, metadata in card_data.items():
        for possible_name in (
            str(card_id),
            str(metadata.get("Name", "")),
        ):
            normalized = normalize_name(possible_name)

            if normalized:
                name_lookup[normalized] = card_id

    for bad_name, correct_id in FILENAME_ALIASES.items():
        if correct_id in card_data:
            name_lookup[bad_name] = correct_id

    image_paths = sorted(
        path
        for path in REFERENCE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(f"No reference images found in {REFERENCE_DIR}")

    features_dir = OUTPUT_DIR / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    fingerprints: list[np.ndarray] = []
    ignored: list[str] = []

    total = len(image_paths)

    for image_number, image_path in enumerate(image_paths, start=1):
        normalized_stem = normalize_name(image_path.stem)
        card_id = name_lookup.get(normalized_stem)

        if card_id is None:
            ignored.append(image_path.name)
            print(f"[{image_number}/{total}] Ignored: {image_path.name}")
            continue

        metadata = card_data[card_id]

        try:
            cost = int(float(metadata["Cost"]))
        except (KeyError, TypeError, ValueError):
            ignored.append(image_path.name)
            print(f"[{image_number}/{total}] Ignored invalid cost: {image_path.name}")
            continue

        card_class = str(metadata.get("Class", "")).strip()

        if not card_class:
            ignored.append(image_path.name)
            print(f"[{image_number}/{total}] Ignored missing class: {image_path.name}")
            continue

        print(f"[{image_number}/{total}] {card_id}")

        canonical, coverage, aligned = cv.load_reference(image_path)
        features = cv.extract_features(canonical, coverage)

        if not aligned:
            print(f"    WARNING: could not align {image_path.name}")

        feature_filename = f"{len(entries):04d}.npz"
        feature_path = features_dir / feature_filename

        np.savez_compressed(
            feature_path,
            kp_0=features.keypoints_xy,
            desc_0=features.descriptors,
            hist_0=features.histogram,
        )

        fingerprints.append(
            cv.make_fingerprint(canonical, coverage)[np.newaxis, :]
        )

        entries.append(
            {
                "cardId": card_id,
                "name": display_name(metadata.get("Name", card_id)),
                "class": card_class,
                "cost": cost,
                "referenceFilename": image_path.name,
                "featureFile": f"features/{feature_filename}",
                "aligned": aligned,
            }
        )

    if not entries:
        raise ValueError("No usable reference cards were indexed.")

    fingerprint_array = np.stack(fingerprints, axis=0).astype(np.float32)

    np.save(
        OUTPUT_DIR / "fingerprints.npy",
        fingerprint_array,
        allow_pickle=False,
    )

    index = {
        "version": 2,
        "views": 1,
        "sift": cv.SIFT_PARAMS,
        "canonicalSize": [cv.CROP_W, cv.CROP_H],
        "entries": entries,
        "ignoredFiles": ignored,
    }

    with (OUTPUT_DIR / "index.json").open("w", encoding="utf-8") as file:
        json.dump(index, file, indent=2, ensure_ascii=False)

    print()
    print(f"Indexed {len(entries)} cards.")
    print(f"Ignored {len(ignored)} files.")
    print(f"Saved index to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
