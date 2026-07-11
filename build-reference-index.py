from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np


CARD_DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("card_data.json")
REFERENCE_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("reference_cards")
OUTPUT_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("reference_index")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

SIFT = cv2.SIFT_create(
    nfeatures=1000,
    contrastThreshold=0.02,
    edgeThreshold=12,
)

# Explicit aliases for the two filename typos shown in the deployment log.
FILENAME_ALIASES = {
    "cukoozombie": "Cuckoo_Zombie",
    "intergalaticwarlord": "Intergalactic_Warlord",
}


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def display_name(value: str) -> str:
    return str(value).replace("_", " ").strip()


def load_image(path: Path) -> np.ndarray:
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return image


def safe_crop(
    image: np.ndarray,
    left_fraction: float,
    top_fraction: float,
    right_fraction: float,
    bottom_fraction: float,
) -> np.ndarray:
    height, width = image.shape[:2]

    x1 = max(0, min(width - 1, round(width * left_fraction)))
    y1 = max(0, min(height - 1, round(height * top_fraction)))
    x2 = max(x1 + 1, min(width, round(width * right_fraction)))
    y2 = max(y1 + 1, min(height, round(height * bottom_fraction)))

    return image[y1:y2, x1:x2]


def resize_for_features(
    image: np.ndarray,
    maximum_dimension: int = 650,
) -> np.ndarray:
    height, width = image.shape[:2]
    largest = max(width, height)

    if largest <= maximum_dimension:
        return image

    scale = maximum_dimension / largest

    return cv2.resize(
        image,
        (
            max(1, round(width * scale)),
            max(1, round(height * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )


def create_views(image: np.ndarray) -> list[np.ndarray]:
    return [
        image,
        safe_crop(
            image,
            left_fraction=0.03,
            top_fraction=0.04,
            right_fraction=0.90,
            bottom_fraction=0.84,
        ),
        safe_crop(
            image,
            left_fraction=0.09,
            top_fraction=0.10,
            right_fraction=0.82,
            bottom_fraction=0.78,
        ),
    ]


def make_fingerprint(image: np.ndarray) -> np.ndarray:
    """
    A cheap, illumination-tolerant visual signature used only for
    shortlisting. Final decisions still use the original SIFT score.
    """

    small = cv2.resize(
        image,
        (32, 32),
        interpolation=cv2.INTER_AREA,
    )

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)

    channels = []
    for channel_index in range(3):
        channel = lab[:, :, channel_index]
        channel = channel - float(channel.mean())
        std = float(channel.std())

        if std > 1e-6:
            channel = channel / std

        channels.append(channel.reshape(-1))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobel_x, sobel_y)

    edge_mean = float(edges.mean())
    edge_std = float(edges.std())
    edges = edges - edge_mean

    if edge_std > 1e-6:
        edges = edges / edge_std

    vector = np.concatenate(
        [
            *channels,
            edges.reshape(-1),
        ]
    ).astype(np.float32)

    norm = float(np.linalg.norm(vector))

    if norm > 1e-8:
        vector /= norm

    return vector


def extract_reference_view(image: np.ndarray) -> dict[str, np.ndarray]:
    image = resize_for_features(image)

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    grayscale = clahe.apply(grayscale)

    keypoints, descriptors = SIFT.detectAndCompute(
        grayscale,
        None,
    )

    if keypoints:
        keypoint_xy = np.asarray(
            [keypoint.pt for keypoint in keypoints],
            dtype=np.float32,
        )
    else:
        keypoint_xy = np.empty((0, 2), dtype=np.float32)

    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)
    else:
        descriptors = np.asarray(descriptors, dtype=np.float32)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [16, 16],
        [0, 180, 0, 256],
    )

    cv2.normalize(
        histogram,
        histogram,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
    )

    return {
        "keypoints": keypoint_xy,
        "descriptors": descriptors,
        "histogram": histogram.astype(np.float32),
        "fingerprint": make_fingerprint(image),
    }


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

        image = load_image(image_path)
        view_data = [
            extract_reference_view(view)
            for view in create_views(image)
        ]

        feature_filename = f"{len(entries):04d}.npz"
        feature_path = features_dir / feature_filename

        np.savez_compressed(
            feature_path,
            kp_0=view_data[0]["keypoints"],
            desc_0=view_data[0]["descriptors"],
            hist_0=view_data[0]["histogram"],
            kp_1=view_data[1]["keypoints"],
            desc_1=view_data[1]["descriptors"],
            hist_1=view_data[1]["histogram"],
            kp_2=view_data[2]["keypoints"],
            desc_2=view_data[2]["descriptors"],
            hist_2=view_data[2]["histogram"],
        )

        fingerprints.append(
            np.stack(
                [
                    view_data[0]["fingerprint"],
                    view_data[1]["fingerprint"],
                    view_data[2]["fingerprint"],
                ],
                axis=0,
            )
        )

        entries.append(
            {
                "cardId": card_id,
                "name": display_name(metadata.get("Name", card_id)),
                "class": card_class,
                "cost": cost,
                "referenceFilename": image_path.name,
                "featureFile": f"features/{feature_filename}",
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
        "version": 1,
        "sift": {
            "nfeatures": 1000,
            "contrastThreshold": 0.02,
            "edgeThreshold": 12,
        },
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
