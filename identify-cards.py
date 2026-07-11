from __future__ import annotations

import itertools
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

CARDS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cards")
CARD_DATA_PATH = (
    Path(sys.argv[2]) if len(sys.argv) > 2 else Path("card_data.json")
)
REFERENCE_DIR = (
    Path(sys.argv[3]) if len(sys.argv) > 3 else Path("reference_cards")
)

DECK_READ_PATH = CARDS_DIR / "deck-read.json"
OUTPUT_PATH = CARDS_DIR / "deck-identified.json"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


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

# Lowe ratio used for SIFT feature matching.
SIFT_RATIO = 0.74

# A result below this should usually be reviewed manually.
LOW_SCORE_THRESHOLD = 0.28

# A small difference between the best and second-best results
# means the match is ambiguous.
LOW_MARGIN_THRESHOLD = 0.035


@dataclass
class FeatureSet:
    keypoints: list
    descriptors: np.ndarray | None
    histogram: np.ndarray


@dataclass
class ReferenceCard:
    card_id: str
    display_name: str
    card_class: str
    cost: int
    image_path: Path
    features: list[FeatureSet]


# SIFT is much better than plain pixel comparison for this task.
SIFT = cv2.SIFT_create(
    nfeatures=1000,
    contrastThreshold=0.02,
    edgeThreshold=12,
)

MATCHER = cv2.BFMatcher(cv2.NORM_L2)


# ---------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------

def normalize_name(value: str) -> str:
    """
    Converts names such as:

        Alien_Ooze
        alien ooze
        Alien-Ooze.jpg

    into the same normalized key.
    """

    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def display_name(value: str) -> str:
    return str(value).replace("_", " ").strip()


def load_image(path: Path) -> np.ndarray:
    """
    cv2.imread can have trouble with some Unicode Windows paths.
    imdecode handles those paths more reliably.
    """

    raw = np.fromfile(path, dtype=np.uint8)
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


# ---------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------

def create_views(
    image: np.ndarray,
    is_extracted_card: bool,
) -> list[np.ndarray]:
    """
    We compare multiple parts of each image.

    The full image is useful when reference_cards contains
    deck-style thumbnails.

    The smaller crops are useful when the reference image is
    a fuller card image and the deck screenshot contains UI
    overlays around the edges.
    """

    views = [image]

    if is_extracted_card:
        # Removes most of the cost bubble and bottom stat icons.
        views.append(
            safe_crop(
                image,
                left_fraction=0.03,
                top_fraction=0.05,
                right_fraction=0.82,
                bottom_fraction=0.80,
            )
        )

        # Mostly pure central artwork.
        views.append(
            safe_crop(
                image,
                left_fraction=0.10,
                top_fraction=0.12,
                right_fraction=0.75,
                bottom_fraction=0.72,
            )
        )

    else:
        # Reference images may contain borders or additional UI.
        views.append(
            safe_crop(
                image,
                left_fraction=0.03,
                top_fraction=0.04,
                right_fraction=0.90,
                bottom_fraction=0.84,
            )
        )

        views.append(
            safe_crop(
                image,
                left_fraction=0.09,
                top_fraction=0.10,
                right_fraction=0.82,
                bottom_fraction=0.78,
            )
        )

    return views


def extract_features(image: np.ndarray) -> FeatureSet:
    image = resize_for_features(image)

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improves feature detection in dark or compressed screenshots.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    grayscale = clahe.apply(grayscale)

    keypoints, descriptors = SIFT.detectAndCompute(
        grayscale,
        None,
    )

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
        keypoints=keypoints,
        descriptors=descriptors,
        histogram=histogram,
    )


def extract_all_features(
    image: np.ndarray,
    is_extracted_card: bool,
) -> list[FeatureSet]:
    return [
        extract_features(view)
        for view in create_views(
            image,
            is_extracted_card=is_extracted_card,
        )
    ]


# ---------------------------------------------------------
# Image similarity
# ---------------------------------------------------------

def sift_similarity(
    query: FeatureSet,
    reference: FeatureSet,
) -> float:
    query_descriptors = query.descriptors
    reference_descriptors = reference.descriptors

    if (
        query_descriptors is None
        or reference_descriptors is None
        or len(query_descriptors) < 2
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
            query.keypoints[match.queryIdx].pt
            for match in good_matches
        ]
    ).reshape(-1, 1, 2)

    destination_points = np.float32(
        [
            reference.keypoints[match.trainIdx].pt
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

    # Converts the usual -1 through 1 correlation range to 0 through 1.
    return max(0.0, min(1.0, (correlation + 1.0) / 2.0))


def visual_similarity(
    query_features: list[FeatureSet],
    reference_features: list[FeatureSet],
) -> float:
    """
    Compares every query crop to every reference crop and keeps
    the best result.

    SIFT provides most of the score. Color similarity is a small
    tie-breaker for cards with limited feature detail.
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
# Loading card metadata and references
# ---------------------------------------------------------

def load_card_data() -> tuple[dict, dict[str, str]]:
    with CARD_DATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "card_data.json must contain an object keyed by card name."
        )

    name_lookup: dict[str, str] = {}

    for card_id, metadata in data.items():
        possible_names = {
            str(card_id),
            str(metadata.get("Name", "")),
        }

        for name in possible_names:
            normalized = normalize_name(name)

            if normalized:
                name_lookup[normalized] = card_id

    return data, name_lookup


def load_reference_cards(
    card_data: dict,
    name_lookup: dict[str, str],
    wanted_costs: set[int],
) -> list[ReferenceCard]:
    references: list[ReferenceCard] = []
    ignored_files: list[str] = []

    image_paths = sorted(
        path
        for path in REFERENCE_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(
            f"No reference images found in {REFERENCE_DIR.resolve()}"
        )

    for image_path in image_paths:
        normalized_stem = normalize_name(image_path.stem)
        card_id = name_lookup.get(normalized_stem)

        if card_id is None:
            ignored_files.append(image_path.name)
            continue

        metadata = card_data[card_id]

        try:
            cost = int(float(metadata["Cost"]))
        except (KeyError, TypeError, ValueError):
            ignored_files.append(image_path.name)
            continue

        # There is no reason to calculate image features for costs
        # that do not appear anywhere in this deck.
        if cost not in wanted_costs:
            continue

        card_class = str(metadata.get("Class", "")).strip()

        if not card_class:
            ignored_files.append(image_path.name)
            continue

        image = load_image(image_path)

        references.append(
            ReferenceCard(
                card_id=card_id,
                display_name=display_name(
                    metadata.get("Name", card_id)
                ),
                card_class=card_class,
                cost=cost,
                image_path=image_path,
                features=extract_all_features(
                    image,
                    is_extracted_card=False,
                ),
            )
        )

    if ignored_files:
        print(
            f"Warning: ignored {len(ignored_files)} reference file(s) "
            "that could not be matched to card_data.json."
        )

        for filename in ignored_files[:10]:
            print(f"  - {filename}")

        if len(ignored_files) > 10:
            print(
                f"  ...and {len(ignored_files) - 10} more"
            )

    if not references:
        raise ValueError(
            "No usable reference cards matched card_data.json."
        )

    return references


# ---------------------------------------------------------
# Class-pair selection
# ---------------------------------------------------------

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

        groups.extend((card_class,) for card_class in present)

        groups.extend(
            itertools.combinations(
                present,
                2,
            )
        )

    return groups


def choose_class_group(
    score_matrix: np.ndarray,
    references: list[ReferenceCard],
    recognized_costs: list[int],
) -> tuple[str, ...]:
    groups = possible_class_groups(references)

    if not groups:
        raise ValueError(
            "No recognized PvZ Heroes classes were found in card_data.json."
        )

    best_group: tuple[str, ...] | None = None
    best_total = -float("inf")

    for group in groups:
        group_set = set(group)
        total = 0.0
        valid = True

        for card_index, recognized_cost in enumerate(recognized_costs):
            candidates = [
                score_matrix[card_index, reference_index]
                for reference_index, reference in enumerate(references)
                if (
                    reference.cost == recognized_cost
                    and reference.card_class in group_set
                )
            ]

            if not candidates:
                valid = False
                break

            total += max(candidates)

        if not valid:
            continue

        # A tiny preference for a single class when it fits equally well.
        # This does not materially override image evidence.
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


# ---------------------------------------------------------
# Unique card assignment
# ---------------------------------------------------------

def assign_unique_cards(
    score_matrix: np.ndarray,
    references: list[ReferenceCard],
    recognized_costs: list[int],
    class_group: tuple[str, ...],
) -> dict[int, int]:
    """
    Each distinct card should appear only once in the deck-list grid.

    The Hungarian assignment algorithm finds the best combined
    identification while preventing duplicate card names.
    """

    allowed_classes = set(class_group)

    usable_reference_indices = [
        index
        for index, reference in enumerate(references)
        if reference.card_class in allowed_classes
    ]

    card_count = len(recognized_costs)
    reference_count = len(usable_reference_indices)

    if reference_count < card_count:
        raise ValueError(
            "There are fewer usable reference cards than extracted cards."
        )

    impossible = 10_000.0

    assignment_costs = np.full(
        (card_count, reference_count),
        impossible,
        dtype=np.float64,
    )

    for card_index, recognized_cost in enumerate(recognized_costs):
        for local_reference_index, reference_index in enumerate(
            usable_reference_indices
        ):
            reference = references[reference_index]

            if reference.cost != recognized_cost:
                continue

            # linear_sum_assignment minimizes, so negate similarity.
            assignment_costs[
                card_index,
                local_reference_index,
            ] = -score_matrix[
                card_index,
                reference_index,
            ]

    row_indices, column_indices = linear_sum_assignment(
        assignment_costs
    )

    result: dict[int, int] = {}

    for row_index, column_index in zip(
        row_indices,
        column_indices,
    ):
        if assignment_costs[row_index, column_index] >= impossible:
            raise ValueError(
                f"Could not assign a cost-{recognized_costs[row_index]} "
                f"card at position {row_index + 1}."
            )

        result[row_index] = usable_reference_indices[
            column_index
        ]

    return result


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    if not DECK_READ_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DECK_READ_PATH}. Run read-deck.js first."
        )

    if not CARD_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CARD_DATA_PATH}."
        )

    if not REFERENCE_DIR.exists():
        raise FileNotFoundError(
            f"Missing reference folder {REFERENCE_DIR}."
        )

    with DECK_READ_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
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

    card_data, name_lookup = load_card_data()

    print("Loading reference card images...")

    references = load_reference_cards(
        card_data=card_data,
        name_lookup=name_lookup,
        wanted_costs=set(recognized_costs),
    )

    print(
        f"Loaded {len(references)} cost-compatible reference cards."
    )

    query_features: list[list[FeatureSet]] = []

    for index, card in enumerate(deck_cards):
        filename = card.get("filename")

        if not filename:
            raise ValueError(
                f"Card {index + 1} has no filename."
            )

        image_path = CARDS_DIR / filename
        image = load_image(image_path)

        query_features.append(
            extract_all_features(
                image,
                is_extracted_card=True,
            )
        )

    score_matrix = np.full(
        (
            len(deck_cards),
            len(references),
        ),
        -1.0,
        dtype=np.float64,
    )

    print("Comparing extracted cards with references...")

    for card_index, recognized_cost in enumerate(recognized_costs):
        compatible_indices = [
            reference_index
            for reference_index, reference in enumerate(references)
            if reference.cost == recognized_cost
        ]

        print(
            f"  Card {card_index + 1}/{len(deck_cards)}: "
            f"cost {recognized_cost}, "
            f"{len(compatible_indices)} possible references"
        )

        for reference_index in compatible_indices:
            score_matrix[
                card_index,
                reference_index,
            ] = visual_similarity(
                query_features[card_index],
                references[reference_index].features,
            )

    selected_classes = choose_class_group(
        score_matrix=score_matrix,
        references=references,
        recognized_costs=recognized_costs,
    )

    assignments = assign_unique_cards(
        score_matrix=score_matrix,
        references=references,
        recognized_costs=recognized_costs,
        class_group=selected_classes,
    )

    output_cards = []
    actually_used_classes = set()

    for card_index, deck_card in enumerate(deck_cards):
        assigned_reference_index = assignments[card_index]
        assigned_reference = references[assigned_reference_index]

        actually_used_classes.add(
            assigned_reference.card_class
        )

        allowed_alternatives = [
            (
                reference,
                float(score_matrix[card_index, reference_index]),
            )
            for reference_index, reference in enumerate(references)
            if (
                reference.cost == recognized_costs[card_index]
                and reference.card_class in set(selected_classes)
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
            for reference, score in allowed_alternatives
            if reference.card_id != assigned_reference.card_id
        ]

        second_best_score = (
            competing_scores[0]
            if competing_scores
            else 0.0
        )

        margin = assigned_score - second_best_score

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
            for reference, score in allowed_alternatives[:3]
        ]

        output_cards.append(
            {
                **deck_card,
                "cardId": assigned_reference.card_id,
                "name": assigned_reference.display_name,
                "class": assigned_reference.card_class,
                "matchedReference": assigned_reference.image_path.name,
                "matchScore": round(assigned_score, 4),
                "matchMargin": round(margin, 4),
                "costVerified": (
                    assigned_reference.cost
                    == recognized_costs[card_index]
                ),
                "needsReview": needs_review,
                "alternatives": alternatives,
            }
        )

    output = {
        **deck,
        "identifiedClasses": sorted(
            actually_used_classes
        ),
        "selectedClassSearchGroup": list(
            selected_classes
        ),
        "allCardsIdentified": all(
            not card["needsReview"]
            for card in output_cards
        ),
        "cards": output_cards,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(f"Deck: {deck.get('deckName') or '(unnamed)'}")
    print(
        "Classes: "
        + " + ".join(
            sorted(actually_used_classes)
        )
    )
    print(
        f"Copies: {deck.get('totalCopies')} "
        f"{'✓' if deck.get('copyTotalIs40') else '✗'}"
    )
    print()

    for index, card in enumerate(output_cards, start=1):
        review_marker = "  REVIEW" if card["needsReview"] else ""

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
        print(f"Card identification failed: {error}")
        sys.exit(1)