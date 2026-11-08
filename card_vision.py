"""
Shared card alignment and feature extraction.

Both build-reference-index.py (reference cards) and identify-cards.py
(cards cropped from a user's screenshot) run every image through the
same steps, so the two sides are compared like-for-like:

1. align_card(): find where the card sits in the image by matching it
   against card_template.npy, an edge template of the UI that every
   deck-view card shares (frame, cost badge, copy-count box, stat
   badges). The card is then warped into a fixed 200x150 canonical
   frame, so the frame, badges and art always land on the same pixels
   no matter how the image was cropped or scaled.

2. ART_MASK: the part of the canonical frame that differs from card to
   card. The cost badge, copy-count box and attack/health badges are
   excluded, because they look nearly identical on every card of the
   same cost and otherwise produce "matches" against the wrong card.

3. extract_features(): SIFT keypoints restricted to the art mask, a
   masked colour histogram and a small fingerprint for shortlisting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------
# Canonical geometry
# ---------------------------------------------------------

# The template canvas. Coordinates below are in canvas pixels.
CANVAS_W = 260
CANVAS_H = 200

# The canonical card crop inside the canvas (card frame plus the
# badges that overhang it, with a few pixels of breathing room).
CROP_X = 30
CROP_Y = 14
CROP_W = 200
CROP_H = 150

# Deck-screen background colour (BGR). Used to fill any part of the
# canonical frame that a source image doesn't cover.
BACKGROUND_BGR = (98, 72, 55)

# Registration below this score falls back to a simple centred resize.
MIN_ALIGN_SCORE = 0.30

TEMPLATE_PATH = Path(__file__).with_name("card_template.npy")


def _build_art_mask() -> np.ndarray:
    mask = np.zeros((CROP_H, CROP_W), np.uint8)

    def rect(x1: int, y1: int, x2: int, y2: int, value: int) -> None:
        mask[
            y1 - CROP_Y:y2 - CROP_Y,
            x1 - CROP_X:x2 - CROP_X,
        ] = value

    rect(44, 36, 216, 153, 255)   # inside the card frame
    rect(30, 100, 96, 164, 0)     # "x4" copy-count box
    rect(116, 100, 230, 164, 0)   # attack and health badges
    rect(160, 14, 230, 86, 0)     # cost badge

    return mask


ART_MASK = _build_art_mask()


# ---------------------------------------------------------
# Template registration
# ---------------------------------------------------------

def edge_map(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
    magnitude = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1),
    )
    return np.minimum(magnitude, 120.0) / 120.0


class _Template:
    """Pre-computed arrays for overlap-aware normalised correlation."""

    PAD = 60

    def __init__(self, template: np.ndarray, factor: float = 1.0):
        if factor != 1.0:
            template = cv2.resize(
                template, None, fx=factor, fy=factor,
                interpolation=cv2.INTER_AREA,
            )

        valid = (template > 0).astype(np.float32)
        mean = float(template[valid > 0].mean())
        centred = np.where(valid > 0, template - mean, 0).astype(np.float32)

        pad = max(8, round(self.PAD * factor))
        self.pad = pad
        self.factor = factor

        def padded(array: np.ndarray) -> np.ndarray:
            out = np.zeros(
                (array.shape[0] + 2 * pad, array.shape[1] + 2 * pad),
                np.float32,
            )
            out[pad:-pad, pad:-pad] = array
            return out

        self.t = padded(centred)
        self.t2 = padded(centred * centred)
        self.valid = padded(valid)

    def best(
        self,
        edges: np.ndarray,
        scales,
        min_coverage: float,
    ) -> tuple[float, float, float, float]:
        best = (-9.0, 1.0, 0.0, 0.0)

        for scale in scales:
            scaled = cv2.resize(
                edges, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
            )
            h, w = scaled.shape

            if (
                h < 8 or w < 8
                or h >= self.t.shape[0]
                or w >= self.t.shape[1]
            ):
                continue

            centred = scaled - scaled.mean()
            norm = float(np.sqrt((centred * centred).sum())) + 1e-6
            ones = np.ones_like(scaled)

            numerator = cv2.matchTemplate(self.t, centred, cv2.TM_CCORR)
            energy = cv2.matchTemplate(self.t2, ones, cv2.TM_CCORR)
            coverage = cv2.matchTemplate(self.valid, ones, cv2.TM_CCORR) / (h * w)

            score = numerator / (np.sqrt(np.maximum(energy, 0)) * norm + 1e-6)
            score = np.where(coverage >= min_coverage, score, -9.0)

            _, value, _, location = cv2.minMaxLoc(score)

            if value > best[0]:
                best = (
                    float(value),
                    float(scale),
                    float(location[0] - self.pad),
                    float(location[1] - self.pad),
                )

        return best


_TEMPLATES: dict[float, _Template] = {}


def _template(factor: float) -> _Template:
    if factor not in _TEMPLATES:
        if not TEMPLATE_PATH.exists():
            raise FileNotFoundError(
                f"Missing {TEMPLATE_PATH}. Run normalize-references.py."
            )
        _TEMPLATES[factor] = _Template(
            np.load(TEMPLATE_PATH, allow_pickle=False), factor
        )
    return _TEMPLATES[factor]


@dataclass
class Alignment:
    score: float
    scale: float
    x: float
    y: float
    fallback: bool


def register_card(
    image: np.ndarray,
    scale_hint: float | None = None,
    min_coverage: float = 0.60,
) -> Alignment:
    """
    Find the similarity transform (uniform scale + translation) that
    maps the image onto the template canvas.
    """

    height, width = image.shape[:2]
    edges = edge_map(image)

    if scale_hint is None:
        low, high = 140.0 / width, 260.0 / width
    else:
        low, high = scale_hint * 0.88, scale_hint * 1.12

    # Coarse pass at half resolution.
    coarse = _template(0.5)
    coarse_scales = np.exp(np.linspace(np.log(low), np.log(high), 18))
    half_edges = cv2.resize(edges, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    _, coarse_scale, _, _ = coarse.best(half_edges, coarse_scales, min_coverage)

    # Fine pass around the coarse scale.
    fine = _template(1.0)
    fine_scales = coarse_scale * np.exp(np.linspace(-0.035, 0.035, 7))
    score, scale, x, y = fine.best(edges, fine_scales, min_coverage)

    if score < MIN_ALIGN_SCORE:
        # Assume the crop is roughly the card and centre it.
        scale = CROP_W / width
        x = CROP_X + (CROP_W - width * scale) / 2
        y = CROP_Y + (CROP_H - height * scale) / 2
        return Alignment(score, scale, x, y, True)

    return Alignment(score, scale, x, y, False)


def warp_to_canonical(
    image: np.ndarray,
    alignment: Alignment,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the 200x150 canonical card and a mask of the pixels that
    actually came from the source image.
    """

    s = alignment.scale
    matrix = np.float32([
        [s, 0, alignment.x - CROP_X],
        [0, s, alignment.y - CROP_Y],
    ])

    canonical = cv2.warpAffine(
        image, matrix, (CROP_W, CROP_H),
        flags=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=BACKGROUND_BGR,
    )

    coverage = cv2.warpAffine(
        np.full(image.shape[:2], 255, np.uint8), matrix, (CROP_W, CROP_H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return canonical, coverage


def align_card(
    image: np.ndarray,
    scale_hint: float | None = None,
) -> tuple[np.ndarray, np.ndarray, Alignment]:
    alignment = register_card(image, scale_hint=scale_hint)
    canonical, coverage = warp_to_canonical(image, alignment)
    return canonical, coverage, alignment


def load_reference(path: Path) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Returns (canonical BGR, coverage mask, aligned_ok).

    Images written by normalize-references.py are already canonical
    (200x150 with coverage in the alpha channel) and are used as-is.
    Anything else is aligned on the fly.
    """

    raw = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)

    if raw is None:
        raise ValueError(f"Could not read image: {path}")

    if raw.ndim == 3 and raw.shape[2] == 4 and raw.shape[:2] == (CROP_H, CROP_W):
        return np.ascontiguousarray(raw[:, :, :3]), np.ascontiguousarray(raw[:, :, 3]), True

    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    elif raw.shape[2] == 4:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

    canonical, coverage, alignment = align_card(raw)
    return canonical, coverage, not alignment.fallback


# ---------------------------------------------------------
# Features
# ---------------------------------------------------------

# Canonical images are small, so SIFT runs on a 2x upscale.
FEATURE_UPSCALE = 2

SIFT_PARAMS = dict(nfeatures=600, contrastThreshold=0.02, edgeThreshold=12)
SIFT = cv2.SIFT_create(**SIFT_PARAMS)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


@dataclass
class FeatureSet:
    keypoints_xy: np.ndarray   # in canonical pixel coordinates
    descriptors: np.ndarray
    histogram: np.ndarray


def usable_mask(coverage: np.ndarray) -> np.ndarray:
    covered = cv2.erode(coverage, np.ones((5, 5), np.uint8))
    return cv2.bitwise_and(ART_MASK, covered)


def extract_features(canonical: np.ndarray, coverage: np.ndarray) -> FeatureSet:
    mask = usable_mask(coverage)

    big = cv2.resize(
        canonical, None, fx=FEATURE_UPSCALE, fy=FEATURE_UPSCALE,
        interpolation=cv2.INTER_CUBIC,
    )
    big_mask = cv2.resize(
        mask, None, fx=FEATURE_UPSCALE, fy=FEATURE_UPSCALE,
        interpolation=cv2.INTER_NEAREST,
    )

    gray = CLAHE.apply(cv2.cvtColor(big, cv2.COLOR_BGR2GRAY))
    keypoints, descriptors = SIFT.detectAndCompute(gray, big_mask)

    if keypoints:
        xy = np.asarray([k.pt for k in keypoints], np.float32) / FEATURE_UPSCALE
    else:
        xy = np.empty((0, 2), np.float32)

    if descriptors is None:
        descriptors = np.empty((0, 128), np.float32)

    hsv = cv2.cvtColor(canonical, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], mask, [16, 16], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, 0, 1, cv2.NORM_MINMAX)

    return FeatureSet(
        keypoints_xy=xy,
        descriptors=np.asarray(descriptors, np.float32),
        histogram=histogram.astype(np.float32),
    )


def make_fingerprint(canonical: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """
    Cheap, illumination-tolerant signature of the art region, used only
    to shortlist candidates. Chrome and uncovered pixels are replaced by
    the art's mean colour so they contribute nothing.
    """

    mask = usable_mask(coverage) > 0
    image = canonical.copy()

    if mask.any():
        image[~mask] = image[mask].mean(axis=0).astype(np.uint8)

    small = cv2.resize(image, (40, 30), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(
        mask.astype(np.uint8), (40, 30), interpolation=cv2.INTER_NEAREST
    ) > 0

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    parts = []

    for channel_index in range(3):
        channel = lab[:, :, channel_index]
        values = channel[small_mask] if small_mask.any() else channel.ravel()
        channel = (channel - values.mean()) / (values.std() + 1e-6)
        parts.append(np.where(small_mask, channel, 0).ravel())

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edges = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    values = edges[small_mask] if small_mask.any() else edges.ravel()
    edges = (edges - values.mean()) / (values.std() + 1e-6)
    parts.append(np.where(small_mask, edges, 0).ravel())

    vector = np.concatenate(parts).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else vector


# ---------------------------------------------------------
# Similarity
# ---------------------------------------------------------

SIFT_RATIO = 0.75
MATCHER = cv2.BFMatcher(cv2.NORM_L2)

# Inlier counts that map to a full SIFT score. Set high enough that a
# card matched on its whole art region clearly outscores one that only
# shares a few incidental features.
INLIER_SATURATION = 45.0
GOOD_SATURATION = 60.0


def sift_similarity(
    query: FeatureSet,
    reference: FeatureSet,
    aligned: bool = True,
) -> float:
    if len(query.descriptors) < 2 or len(reference.descriptors) < 2:
        return 0.0

    try:
        pairs = MATCHER.knnMatch(query.descriptors, reference.descriptors, k=2)
    except cv2.error:
        return 0.0

    good = [
        pair[0] for pair in pairs
        if len(pair) == 2 and pair[0].distance < SIFT_RATIO * pair[1].distance
    ]

    good_score = min(len(good) / GOOD_SATURATION, 1.0)

    if len(good) < 4:
        return 0.1 * good_score

    source = np.float32([query.keypoints_xy[m.queryIdx] for m in good])
    destination = np.float32([reference.keypoints_xy[m.trainIdx] for m in good])

    try:
        transform, inliers = cv2.estimateAffinePartial2D(
            source, destination,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=500,
        )
    except cv2.error:
        transform, inliers = None, None

    if transform is None or inliers is None:
        return 0.2 * good_score

    inlier_count = int(inliers.ravel().sum())

    if aligned:
        # Both images are in canonical coordinates, so the true match is
        # close to the identity transform. Anything far from it is a
        # coincidental alignment of unrelated features.
        scale = float(np.hypot(transform[0, 0], transform[1, 0]))
        shift = float(np.hypot(transform[0, 2], transform[1, 2]))

        if not (0.85 <= scale <= 1.18) or shift > 25:
            inlier_count = min(inlier_count, 3)

    inlier_score = min(inlier_count / INLIER_SATURATION, 1.0)
    inlier_ratio = inlier_count / len(good)

    return 0.62 * inlier_score + 0.23 * inlier_ratio + 0.15 * good_score


def histogram_similarity(first: FeatureSet, second: FeatureSet) -> float:
    correlation = cv2.compareHist(
        first.histogram, second.histogram, cv2.HISTCMP_CORREL
    )
    return max(0.0, min(1.0, (correlation + 1.0) / 2.0))


def visual_similarity(
    query: FeatureSet,
    reference: FeatureSet,
    aligned: bool = True,
) -> float:
    return float(
        0.88 * sift_similarity(query, reference, aligned)
        + 0.12 * histogram_similarity(query, reference)
    )
