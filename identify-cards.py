from __future__ import annotations

import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

CARDS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cards")
CARD_DATA_PATH = (
    Path(sys.argv[2]) if len(sys.argv) > 2 else Path("card_data.json")
)
REFERENCE_INDEX_DIR = (
    Path(sys.argv[3]) if len(sys.argv) > 3 else Path("reference_index")
)

DECK_READ_PATH = CARDS_DIR / "deck-read.json"
OUTPUT_PATH = CARDS_DIR / "deck-identified.json"
INDEX_JSON_PATH = REFERENCE_INDEX_DIR / "index.json"
FINGERPRINTS_PATH = REFERENCE_INDEX_DIR / "fingerprints.npy"


# ---------------------------------------------------------
# PvZ Heroes classes
# ---------------------------------------------------------

PLANT_CLASSES = {
    "Guardian",
    "Kabloom",
    "Mega-Grow",
    "Smarty",
    "Solar",
}

ZOMBIE_CLASSES = {
    "Beastly",
    "Brainy",
    "Crazy",
    "Hearty",
    "Sneaky",
}


# ---------------------------------------------------------
# Recognition tuning
# ---------------------------------------------------------

SIFT_RATIO = 0.74

LOW_SCORE_THRESHOLD = 0.28
LOW_MARGIN_THRESHOLD = 0.035

# Fast first attempt.
INITIAL_CLASS_GROUPS = 3
INITIAL_PER_CARD = 18
INITIAL_PER_CLASS = 4

# Wider second attempt before the exact full fallback.
EXPANDED_CLASS_GROUPS = 8
EXPANDED_PER_CARD = 40
EXPANDED_PER_CLASS = 8

# Clean direct screenshots usually score around 0.85-1.00.
# Below these values, the script widens the candidate search.
FAST_ACCEPT_SCORE = 0.82
FAST_ACCEPT_MARGIN = 0.015
FAST_ACCEPT_FINGERPRINT = 0.62

ENABLE_FULL_FALLBACK = (
    os.environ.get("PVZH_FULL_FALLBACK", "1") != "0"
)


@dataclass
class FeatureSet:
    keypoints_xy: np.ndarray
    descriptors: np.ndarray
    histogram: np.ndarray


@dataclass
class ReferenceCard:
    card_id: str
    display_name: str
    card_class: str
    cost: int
    reference_filename: str
    feature_path: Path


SIFT = cv2.SIFT_create(
    nfeatures=1000,
    contrastThreshold=0.02,
    edgeThreshold=12,
)

MATCHER = cv2.BFMatcher(cv2.NORM_L2)


# ---------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------

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


def create_query_views(image: np.ndarray) -> list[np.ndarray]:
    return [
        image,
        safe_crop(
            image,
            left_fraction=0.03,
            top_fraction=0.05,
            right_fraction=0.82,
            bottom_fraction=0.80,
        ),
        safe_crop(
            image,
            left_fraction=0.10,
            top_fraction=0.12,
            right_fraction=0.75,
            bottom_fraction=0.72,
        ),
    ]


def make_fingerprint(image: np.ndarray) -> np.ndarray:
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


# ---------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------

def extract_features(image: np.ndarray) -> FeatureSet:
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
        keypoints_xy = np.asarray(
            [keypoint.pt for keypoint in keypoints],
            dtype=np.float32,
        )
    else:
        keypoints_xy = np.empty((0, 2), dtype=np.float32)

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

    return FeatureSet(
        keypoints_xy=keypoints_xy,
        descriptors=descriptors,
        histogram=histogram.astype(np.float32),
    )


def extract_all_query_data(
    image: np.ndarray,
) -> tuple[list[FeatureSet], np.ndarray]:
    views = create_query_views(image)

    features = [
        extract_features(view)
        for view in views
    ]

    fingerprints = np.stack(
        [
            make_fingerprint(
                resize_for_features(view)
            )
            for view in views
        ],
        axis=0,
    ).astype(np.float32)

    return features, fingerprints


# ---------------------------------------------------------
# Exact image similarity
# ---------------------------------------------------------

def sift_similarity(
    query: FeatureSet,
    reference: FeatureSet,
) -> float:
    query_descriptors = query.descriptors
    reference_descriptors = reference.descriptors

    if (
        len(query_descriptors) < 2
        or len(reference_descriptors) < 2
    ):
        return 0.0

    try:
        pairs = MATCHER.knnMatch(
            query_descriptors,
            reference_descriptors,
            k=2,
        )
    except cv2.error:
        return 0.0

    good_matches = []

    for pair in pairs:
        if len(pair) < 2:
            continue

        best, second_best = pair

        if best.distance < SIFT_RATIO * second_best.distance:
            good_matches.append(best)

    if not good_matches:
        return 0.0

    good_score = min(len(good_matches) / 28.0, 1.0)

    if len(good_matches) < 4:
        return 0.12 * good_score

    source_points = np.float32(
        [
            query.keypoints_xy[match.queryIdx]
            for match in good_matches
        ]
    ).reshape(-1, 1, 2)

    destination_points = np.float32(
        [
            reference.keypoints_xy[match.trainIdx]
            for match in good_matches
        ]
    ).reshape(-1, 1, 2)

    try:
        _, inlier_mask = cv2.findHomography(
            source_points,
            destination_points,
            cv2.RANSAC,
            5.0,
        )
    except cv2.error:
        inlier_mask = None

    if inlier_mask is None:
        return 0.20 * good_score

    inlier_count = int(inlier_mask.ravel().sum())
    inlier_ratio = inlier_count / len(good_matches)
    inlier_score = min(inlier_count / 18.0, 1.0)

    return (
        0.52 * inlier_score
        + 0.30 * inlier_ratio
        + 0.18 * good_score
    )


def histogram_similarity(
    first: FeatureSet,
    second: FeatureSet,
) -> float:
    correlation = cv2.compareHist(
        first.histogram,
        second.histogram,
        cv2.HISTCMP_CORREL,
    )

    return max(
        0.0,
        min(1.0, (correlation + 1.0) / 2.0),
    )


def visual_similarity(
    query_features: list[FeatureSet],
    reference_features: list[FeatureSet],
) -> float:
    """
    This is the same 3x3 exact comparison used by the original script.
    The optimization is that it runs only after a cheap shortlist.
    """

    best_score = 0.0

    for query in query_features:
        for reference in reference_features:
            feature_score = sift_similarity(
                query,
                reference,
            )

            color_score = histogram_similarity(
                query,
                reference,
            )

            combined_score = (
                0.88 * feature_score
                + 0.12 * color_score
            )

            best_score = max(
                best_score,
                combined_score,
            )

    return float(best_score)


# ---------------------------------------------------------
# Index loading
# ---------------------------------------------------------

def load_reference_index() -> tuple[list[ReferenceCard], np.ndarray]:
    if not INDEX_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Missing {INDEX_JSON_PATH}. "
            "Run build-reference-index.py during the Docker build."
        )

    if not FINGERPRINTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FINGERPRINTS_PATH}."
        )

    with INDEX_JSON_PATH.open("r", encoding="utf-8") as file:
        index = json.load(file)

    references = [
        ReferenceCard(
            card_id=str(entry["cardId"]),
            display_name=str(entry["name"]),
            card_class=str(entry["class"]),
            cost=int(entry["cost"]),
            reference_filename=str(entry["referenceFilename"]),
            feature_path=(
                REFERENCE_INDEX_DIR
                / str(entry["featureFile"])
            ),
        )
        for entry in index["entries"]
    ]

    fingerprints = np.load(
        FINGERPRINTS_PATH,
        allow_pickle=False,
        mmap_mode="r",
    )

    if fingerprints.shape[0] != len(references):
        raise ValueError(
            "Reference metadata and fingerprint counts do not match."
        )

    return references, fingerprints


def load_reference_feature_file(
    reference: ReferenceCard,
) -> list[FeatureSet]:
    with np.load(
        reference.feature_path,
        allow_pickle=False,
    ) as archive:
        return [
            FeatureSet(
                keypoints_xy=np.asarray(
                    archive[f"kp_{view_index}"],
                    dtype=np.float32,
                ),
                descriptors=np.asarray(
                    archive[f"desc_{view_index}"],
                    dtype=np.float32,
                ),
                histogram=np.asarray(
                    archive[f"hist_{view_index}"],
                    dtype=np.float32,
                ),
            )
            for view_index in range(3)
        ]


# ---------------------------------------------------------
# Fast shortlist
# ---------------------------------------------------------

def calculate_quick_scores(
    query_fingerprints: list[np.ndarray],
    reference_fingerprints: np.ndarray,
) -> np.ndarray:
    """
    Returns one score per uploaded card and reference card.

    Both sides are L2-normalized. Matrix multiplication therefore
    gives cosine similarity. We retain the best of the 3x3 view pairs.
    """

    card_count = len(query_fingerprints)
    reference_count = reference_fingerprints.shape[0]

    scores = np.full(
        (card_count, reference_count),
        -1.0,
        dtype=np.float32,
    )

    flattened_references = np.asarray(
        reference_fingerprints,
        dtype=np.float32,
    ).reshape(
        reference_count * 3,
        -1,
    )

    for card_index, query_views in enumerate(query_fingerprints):
        similarities = (
            np.asarray(query_views, dtype=np.float32)
            @ flattened_references.T
        )

        similarities = similarities.reshape(
            3,
            reference_count,
            3,
        )

        best = similarities.max(axis=(0, 2))

        # Convert cosine similarity from roughly [-1, 1] to [0, 1].
        scores[card_index] = np.clip(
            (best + 1.0) / 2.0,
            0.0,
            1.0,
        )

    return scores


def possible_class_groups(
    references: list[ReferenceCard],
) -> list[tuple[str, ...]]:
    available = {
        reference.card_class
        for reference in references
    }

    groups: list[tuple[str, ...]] = []

    for side_classes in (
        PLANT_CLASSES,
        ZOMBIE_CLASSES,
    ):
        present = sorted(
            side_classes.intersection(available)
        )

        groups.extend(
            (card_class,)
            for card_class in present
        )

        groups.extend(
            itertools.combinations(
                present,
                2,
            )
        )

    return groups


def rank_class_groups(
    quick_scores: np.ndarray,
    references: list[ReferenceCard],
    recognized_costs: list[int],
) -> list[tuple[tuple[str, ...], float]]:
    groups = possible_class_groups(references)
    ranked: list[tuple[tuple[str, ...], float]] = []

    for group in groups:
        group_set = set(group)
        total = 0.0
        valid = True

        for card_index, recognized_cost in enumerate(recognized_costs):
            candidates = [
                quick_scores[
                    card_index,
                    reference_index,
                ]
                for reference_index, reference
                in enumerate(references)
                if (
                    reference.cost == recognized_cost
                    and reference.card_class in group_set
                )
            ]

            if not candidates:
                valid = False
                break

            total += float(max(candidates))

        if not valid:
            continue

        if len(group) == 2:
            total -= 0.002

        ranked.append((group, total))

    ranked.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return ranked


def build_candidate_sets(
    quick_scores: np.ndarray,
    references: list[ReferenceCard],
    recognized_costs: list[int],
    ranked_groups: list[tuple[tuple[str, ...], float]],
    group_count: int,
    per_card: int,
    per_class: int,
) -> list[set[int]]:
    selected_groups = [
        group
        for group, _ in ranked_groups[:group_count]
    ]

    selected_classes = set(
        itertools.chain.from_iterable(
            selected_groups
        )
    )

    result: list[set[int]] = []

    for card_index, recognized_cost in enumerate(recognized_costs):
        compatible = [
            reference_index
            for reference_index, reference
            in enumerate(references)
            if (
                reference.cost == recognized_cost
                and reference.card_class in selected_classes
            )
        ]

        compatible.sort(
            key=lambda reference_index:
                float(
                    quick_scores[
                        card_index,
                        reference_index,
                    ]
                ),
            reverse=True,
        )

        chosen = set(
            compatible[:per_card]
        )

        for card_class in selected_classes:
            same_class = [
                reference_index
                for reference_index in compatible
                if (
                    references[
                        reference_index
                    ].card_class
                    == card_class
                )
            ]

            chosen.update(
                same_class[:per_class]
            )

        result.append(chosen)

    return result


# ---------------------------------------------------------
# Assignment
# ---------------------------------------------------------

def hungarian_minimize(costs: np.ndarray) -> dict[int, int]:
    """
    Rectangular Hungarian algorithm for rows <= columns.
    This replaces SciPy's linear_sum_assignment, removing a large
    import and memory cost without changing the optimization problem.
    """

    n, m = costs.shape

    if n > m:
        raise ValueError(
            "Hungarian assignment requires at least as many columns as rows."
        )

    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(m + 1, dtype=np.float64)
    p = np.zeros(m + 1, dtype=np.int32)
    way = np.zeros(m + 1, dtype=np.int32)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0

        minv = np.full(
            m + 1,
            np.inf,
            dtype=np.float64,
        )

        used = np.zeros(
            m + 1,
            dtype=bool,
        )

        while True:
            used[j0] = True
            i0 = int(p[j0])

            delta = np.inf
            j1 = 0

            for j in range(1, m + 1):
                if used[j]:
                    continue

                current = (
                    costs[i0 - 1, j - 1]
                    - u[i0]
                    - v[j]
                )

                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0

                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1

            if p[j0] == 0:
                break

        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1

            if j0 == 0:
                break

    assignment: dict[int, int] = {}

    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[int(p[j]) - 1] = j - 1

    return assignment


def choose_class_group(
    score_matrix: np.ndarray,
    references: list[ReferenceCard],
    recognized_costs: list[int],
    allowed_groups: list[tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    groups = (
        allowed_groups
        if allowed_groups is not None
        else possible_class_groups(references)
    )

    best_group: tuple[str, ...] | None = None
    best_total = -float("inf")

    for group in groups:
        group_set = set(group)
        total = 0.0
        valid = True

        for card_index, recognized_cost in enumerate(recognized_costs):
            candidates = [
                score_matrix[
                    card_index,
                    reference_index,
                ]
                for reference_index, reference
                in enumerate(references)
                if (
                    reference.cost == recognized_cost
                    and reference.card_class in group_set
                    and score_matrix[
                        card_index,
                        reference_index,
                    ] >= 0
                )
            ]

            if not candidates:
                valid = False
                break

            total += max(candidates)

        if not valid:
            continue

        if len(group) == 2:
            total -= 0.015

        if total > best_total:
            best_total = total
            best_group = group

    if best_group is None:
        raise ValueError(
            "No one- or two-class combination could explain every card."
        )

    return best_group


def assign_unique_cards(
    score_matrix: np.ndarray,
    references: list[ReferenceCard],
    recognized_costs: list[int],
    class_group: tuple[str, ...],
) -> dict[int, int]:
    allowed_classes = set(class_group)

    usable_reference_indices = [
        index
        for index, reference
        in enumerate(references)
        if reference.card_class in allowed_classes
    ]

    if len(usable_reference_indices) < len(recognized_costs):
        raise ValueError(
            "There are fewer usable reference cards than extracted cards."
        )

    impossible = 10_000.0

    assignment_costs = np.full(
        (
            len(recognized_costs),
            len(usable_reference_indices),
        ),
        impossible,
        dtype=np.float64,
    )

    for card_index, recognized_cost in enumerate(recognized_costs):
        for local_reference_index, reference_index in enumerate(
            usable_reference_indices
        ):
            reference = references[reference_index]
            score = score_matrix[
                card_index,
                reference_index,
            ]

            if (
                reference.cost != recognized_cost
                or score < 0
            ):
                continue

            assignment_costs[
                card_index,
                local_reference_index,
            ] = -float(score)

    local_assignment = hungarian_minimize(
        assignment_costs
    )

    result: dict[int, int] = {}

    for row_index, local_reference_index in local_assignment.items():
        if (
            assignment_costs[
                row_index,
                local_reference_index,
            ]
            >= impossible
        ):
            raise ValueError(
                f"Could not assign a cost-{recognized_costs[row_index]} "
                f"card at position {row_index + 1}."
            )

        result[row_index] = (
            usable_reference_indices[
                local_reference_index
            ]
        )

    return result


# ---------------------------------------------------------
# Exact scoring stages
# ---------------------------------------------------------

def exact_score_candidates(
    query_features: list[list[FeatureSet]],
    candidate_sets: list[set[int]],
    references: list[ReferenceCard],
    score_matrix: np.ndarray,
    feature_cache: dict[int, list[FeatureSet]],
) -> int:
    comparisons = 0

    union = sorted(
        set().union(*candidate_sets)
    )

    for reference_index in union:
        if reference_index not in feature_cache:
            feature_cache[reference_index] = (
                load_reference_feature_file(
                    references[reference_index]
                )
            )

    for card_index, candidates in enumerate(candidate_sets):
        for reference_index in candidates:
            if score_matrix[
                card_index,
                reference_index,
            ] >= 0:
                continue

            score_matrix[
                card_index,
                reference_index,
            ] = visual_similarity(
                query_features[card_index],
                feature_cache[reference_index],
            )

            comparisons += 1

    return comparisons


def build_full_candidate_sets(
    references: list[ReferenceCard],
    recognized_costs: list[int],
) -> list[set[int]]:
    return [
        {
            reference_index
            for reference_index, reference
            in enumerate(references)
            if reference.cost == recognized_cost
        }
        for recognized_cost in recognized_costs
    ]


def evaluate_assignments(
    score_matrix: np.ndarray,
    quick_scores: np.ndarray,
    references: list[ReferenceCard],
    recognized_costs: list[int],
    allowed_groups: list[tuple[str, ...]] | None,
) -> tuple[
    tuple[str, ...],
    dict[int, int],
    list[dict],
    bool,
]:
    selected_classes = choose_class_group(
        score_matrix=score_matrix,
        references=references,
        recognized_costs=recognized_costs,
        allowed_groups=allowed_groups,
    )

    assignments = assign_unique_cards(
        score_matrix=score_matrix,
        references=references,
        recognized_costs=recognized_costs,
        class_group=selected_classes,
    )

    diagnostics: list[dict] = []
    confident = True

    selected_class_set = set(
        selected_classes
    )

    for card_index, assigned_reference_index in assignments.items():
        assigned_score = float(
            score_matrix[
                card_index,
                assigned_reference_index,
            ]
        )

        competing_scores = sorted(
            [
                float(
                    score_matrix[
                        card_index,
                        reference_index,
                    ]
                )
                for reference_index, reference
                in enumerate(references)
                if (
                    reference.cost
                    == recognized_costs[card_index]
                    and reference.card_class
                    in selected_class_set
                    and reference_index
                    != assigned_reference_index
                    and score_matrix[
                        card_index,
                        reference_index,
                    ] >= 0
                )
            ],
            reverse=True,
        )

        second_best_score = (
            competing_scores[0]
            if competing_scores
            else 0.0
        )

        margin = (
            assigned_score
            - second_best_score
        )

        fingerprint_score = float(
            quick_scores[
                card_index,
                assigned_reference_index,
            ]
        )

        same_cost_quick_scores = sorted(
            [
                float(
                    quick_scores[
                        card_index,
                        reference_index,
                    ]
                )
                for reference_index, reference
                in enumerate(references)
                if (
                    reference.cost
                    == recognized_costs[card_index]
                )
            ],
            reverse=True,
        )

        fingerprint_rank = (
            same_cost_quick_scores.index(
                fingerprint_score
            )
            + 1
        )

        card_confident = (
    assigned_score >= 0.82
    and (
        margin >= 0.005
        or assigned_score >= 0.90
    )
)

        diagnostics.append(
            {
                "cardIndex": card_index,
                "referenceIndex":
                    assigned_reference_index,
                "score": assigned_score,
                "margin": margin,
                "fingerprintScore":
                    fingerprint_score,
                "fingerprintRank":
                    fingerprint_rank,
                "confident": card_confident,
            }
        )

        if not card_confident:
            confident = False

    return (
        selected_classes,
        assignments,
        diagnostics,
        confident,
    )


# ---------------------------------------------------------
# Output
# ---------------------------------------------------------

def make_output(
    deck: dict,
    score_matrix: np.ndarray,
    references: list[ReferenceCard],
    recognized_costs: list[int],
    selected_classes: tuple[str, ...],
    assignments: dict[int, int],
    search_stage: str,
    exact_comparisons: int,
    elapsed_seconds: float,
) -> dict:
    output_cards = []
    actually_used_classes = set()
    selected_class_set = set(selected_classes)

    for card_index, deck_card in enumerate(deck["cards"]):
        assigned_reference_index = assignments[card_index]
        assigned_reference = references[
            assigned_reference_index
        ]

        actually_used_classes.add(
            assigned_reference.card_class
        )

        allowed_alternatives = [
            (
                reference,
                float(
                    score_matrix[
                        card_index,
                        reference_index,
                    ]
                ),
            )
            for reference_index, reference
            in enumerate(references)
            if (
                reference.cost
                == recognized_costs[card_index]
                and reference.card_class
                in selected_class_set
                and score_matrix[
                    card_index,
                    reference_index,
                ] >= 0
            )
        ]

        allowed_alternatives.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        assigned_score = float(
            score_matrix[
                card_index,
                assigned_reference_index,
            ]
        )

        competing_scores = [
            score
            for reference, score
            in allowed_alternatives
            if (
                reference.card_id
                != assigned_reference.card_id
            )
        ]

        second_best_score = (
            competing_scores[0]
            if competing_scores
            else 0.0
        )

        margin = (
            assigned_score
            - second_best_score
        )

        needs_review = (
            assigned_score < LOW_SCORE_THRESHOLD
            or margin < LOW_MARGIN_THRESHOLD
        )

        alternatives = [
            {
                "cardId": reference.card_id,
                "name": reference.display_name,
                "class": reference.card_class,
                "cost": reference.cost,
                "score": round(score, 4),
            }
            for reference, score
            in allowed_alternatives[:3]
        ]

        output_cards.append(
            {
                **deck_card,
                "cardId":
                    assigned_reference.card_id,
                "name":
                    assigned_reference.display_name,
                "class":
                    assigned_reference.card_class,
                "matchedReference":
                    assigned_reference.reference_filename,
                "matchScore":
                    round(assigned_score, 4),
                "matchMargin":
                    round(margin, 4),
                "costVerified":
                    assigned_reference.cost
                    == recognized_costs[card_index],
                "needsReview":
                    needs_review,
                "alternatives":
                    alternatives,
            }
        )

    return {
        **deck,
        "identifiedClasses":
            sorted(actually_used_classes),
        "selectedClassSearchGroup":
            list(selected_classes),
        "allCardsIdentified":
            all(
                not card["needsReview"]
                for card in output_cards
            ),
        "recognitionDiagnostics": {
            "searchStage": search_stage,
            "exactComparisons":
                exact_comparisons,
            "elapsedSeconds":
                round(elapsed_seconds, 3),
            "fullFallbackEnabled":
                ENABLE_FULL_FALLBACK,
        },
        "cards":
            output_cards,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    started = time.perf_counter()

    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)

    if not DECK_READ_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DECK_READ_PATH}. Run read-deck.js first."
        )

    if not CARD_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CARD_DATA_PATH}."
        )

    with DECK_READ_PATH.open("r", encoding="utf-8") as file:
        deck = json.load(file)

    deck_cards = deck.get("cards", [])

    if not deck_cards:
        raise ValueError(
            "deck-read.json does not contain any cards."
        )

    recognized_costs: list[int] = []

    for index, card in enumerate(deck_cards):
        cost = card.get("cost")

        if not isinstance(cost, int):
            raise ValueError(
                f"Card {index + 1} has no recognized integer cost."
            )

        recognized_costs.append(cost)

    print("Loading precomputed reference index...")

    references, reference_fingerprints = (
        load_reference_index()
    )

    print(
        f"Loaded metadata for {len(references)} reference cards."
    )

    query_features: list[list[FeatureSet]] = []
    query_fingerprints: list[np.ndarray] = []

    for index, card in enumerate(deck_cards):
        filename = card.get("filename")

        if not filename:
            raise ValueError(
                f"Card {index + 1} has no filename."
            )

        image_path = CARDS_DIR / filename
        image = load_image(image_path)

        features, fingerprints = (
            extract_all_query_data(image)
        )

        query_features.append(features)
        query_fingerprints.append(fingerprints)

    print("Running fast reference shortlist...")

    quick_scores = calculate_quick_scores(
        query_fingerprints,
        reference_fingerprints,
    )

    ranked_groups = rank_class_groups(
        quick_scores,
        references,
        recognized_costs,
    )

    if not ranked_groups:
        raise ValueError(
            "No valid class groups were found."
        )

    score_matrix = np.full(
        (
            len(deck_cards),
            len(references),
        ),
        -1.0,
        dtype=np.float64,
    )

    feature_cache: dict[
        int,
        list[FeatureSet],
    ] = {}

    exact_comparisons = 0
    search_stage = "initial-shortlist"

    initial_candidates = build_candidate_sets(
        quick_scores=quick_scores,
        references=references,
        recognized_costs=recognized_costs,
        ranked_groups=ranked_groups,
        group_count=INITIAL_CLASS_GROUPS,
        per_card=INITIAL_PER_CARD,
        per_class=INITIAL_PER_CLASS,
    )

    exact_comparisons += exact_score_candidates(
        query_features=query_features,
        candidate_sets=initial_candidates,
        references=references,
        score_matrix=score_matrix,
        feature_cache=feature_cache,
    )

    allowed_groups = [
        group
        for group, _
        in ranked_groups[:INITIAL_CLASS_GROUPS]
    ]

    try:
        (
            selected_classes,
            assignments,
            diagnostics,
            confident,
        ) = evaluate_assignments(
            score_matrix=score_matrix,
            quick_scores=quick_scores,
            references=references,
            recognized_costs=recognized_costs,
            allowed_groups=allowed_groups,
        )
    except ValueError:
        confident = False
        selected_classes = tuple()
        assignments = {}
        diagnostics = []

    if not confident:
        print(
            "Initial shortlist was uncertain; expanding exact search..."
        )

        search_stage = "expanded-shortlist"

        expanded_candidates = build_candidate_sets(
            quick_scores=quick_scores,
            references=references,
            recognized_costs=recognized_costs,
            ranked_groups=ranked_groups,
            group_count=EXPANDED_CLASS_GROUPS,
            per_card=EXPANDED_PER_CARD,
            per_class=EXPANDED_PER_CLASS,
        )

        exact_comparisons += exact_score_candidates(
            query_features=query_features,
            candidate_sets=expanded_candidates,
            references=references,
            score_matrix=score_matrix,
            feature_cache=feature_cache,
        )

        allowed_groups = [
            group
            for group, _
            in ranked_groups[:EXPANDED_CLASS_GROUPS]
        ]

        try:
            (
                selected_classes,
                assignments,
                diagnostics,
                confident,
            ) = evaluate_assignments(
                score_matrix=score_matrix,
                quick_scores=quick_scores,
                references=references,
                recognized_costs=recognized_costs,
                allowed_groups=allowed_groups,
            )
        except ValueError:
            confident = False
            selected_classes = tuple()
            assignments = {}
            diagnostics = []

    if (
        not confident
        and ENABLE_FULL_FALLBACK
    ):
        print(
            "Expanded search was uncertain; running exact full fallback..."
        )

        search_stage = "full-exact-fallback"

        selected_class_set = set(selected_classes)

        full_candidates = [
            {
                reference_index
                for reference_index, reference
                in enumerate(references)
                if (
                    reference.cost == recognized_cost
                    and reference.card_class
                    in selected_class_set
                )
            }
            for recognized_cost in recognized_costs
        ]

        exact_comparisons += exact_score_candidates(
            query_features=query_features,
            candidate_sets=full_candidates,
            references=references,
            score_matrix=score_matrix,
            feature_cache=feature_cache,
        )

        (
            selected_classes,
            assignments,
            diagnostics,
            confident,
        ) = evaluate_assignments(
            score_matrix=score_matrix,
            quick_scores=quick_scores,
            references=references,
            recognized_costs=recognized_costs,
            allowed_groups=None,
        )

    if not assignments:
        raise ValueError(
            "No complete card assignment could be produced."
        )

    elapsed = (
        time.perf_counter()
        - started
    )

    output = make_output(
        deck=deck,
        score_matrix=score_matrix,
        references=references,
        recognized_costs=recognized_costs,
        selected_classes=selected_classes,
        assignments=assignments,
        search_stage=search_stage,
        exact_comparisons=exact_comparisons,
        elapsed_seconds=elapsed,
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Deck: {deck.get('deckName') or '(unnamed)'}"
    )

    print(
        "Classes: "
        + " + ".join(
            output["identifiedClasses"]
        )
    )

    print(
        f"Copies: {deck.get('totalCopies')} "
        f"{'✓' if deck.get('copyTotalIs40') else '✗'}"
    )

    print(
        f"Search: {search_stage}; "
        f"{exact_comparisons} exact comparisons; "
        f"{elapsed:.2f}s"
    )

    print()

    for index, card in enumerate(
        output["cards"],
        start=1,
    ):
        review_marker = (
            "  REVIEW"
            if card["needsReview"]
            else ""
        )

        print(
            f"{index:02d}. "
            f"x{card.get('copies')} "
            f"cost {card.get('cost'):>2}  "
            f"{card['name']} "
            f"[{card['class']}] "
            f"score={card['matchScore']:.3f}"
            f"{review_marker}"
        )

    print()
    print(f"Saved: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print(
            f"Card identification failed: {error}"
        )
        sys.exit(1)
