from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


CARD_DATA_PATH = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path("card_data.json")
)

REFERENCE_DIR = (
    Path(sys.argv[2])
    if len(sys.argv) > 2
    else Path("reference_cards")
)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


def normalize_name(value: object) -> str:
    """
    These all normalize to the same value:

        Galacta_Cactus
        Galacta Cactus
        galacta-cactus
        Galacta_Cactus.jpg
    """

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower(),
    )


def print_section(
    title: str,
    items: list[str],
) -> None:
    print()
    print(title)

    if not items:
        print("  None")
        return

    for item in items:
        print(f"  - {item}")


def main() -> int:
    if not CARD_DATA_PATH.exists():
        print(
            f"ERROR: Could not find {CARD_DATA_PATH.resolve()}"
        )
        return 1

    if not REFERENCE_DIR.exists():
        print(
            f"ERROR: Could not find {REFERENCE_DIR.resolve()}"
        )
        return 1

    try:
        with CARD_DATA_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            card_data = json.load(file)
    except json.JSONDecodeError as error:
        print(
            f"ERROR: card_data.json is invalid JSON: {error}"
        )
        return 1

    if not isinstance(card_data, dict):
        print(
            "ERROR: card_data.json must contain an object "
            "keyed by card ID."
        )
        return 1

    # Maps normalized names to every card ID that uses that name.
    normalized_lookup: dict[str, set[str]] = defaultdict(set)

    for card_id, metadata in card_data.items():
        possible_names = [card_id]

        if isinstance(metadata, dict):
            possible_names.append(
                metadata.get("Name", "")
            )

        for possible_name in possible_names:
            normalized = normalize_name(
                possible_name
            )

            if normalized:
                normalized_lookup[
                    normalized
                ].add(card_id)

    image_paths = sorted(
        path
        for path in REFERENCE_DIR.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    )

    matched_images_by_card: dict[
        str,
        list[Path],
    ] = defaultdict(list)

    unmatched_images: list[Path] = []
    ambiguous_images: list[
        tuple[Path, list[str]]
    ] = []

    for image_path in image_paths:
        normalized_stem = normalize_name(
            image_path.stem
        )

        matching_card_ids = sorted(
            normalized_lookup.get(
                normalized_stem,
                set(),
            )
        )

        if len(matching_card_ids) == 0:
            unmatched_images.append(
                image_path
            )

        elif len(matching_card_ids) > 1:
            ambiguous_images.append(
                (
                    image_path,
                    matching_card_ids,
                )
            )

        else:
            matched_images_by_card[
                matching_card_ids[0]
            ].append(image_path)

    missing_cards = [
        card_id
        for card_id in card_data
        if not matched_images_by_card.get(
            card_id
        )
    ]

    duplicate_references = {
        card_id: paths
        for card_id, paths
        in matched_images_by_card.items()
        if len(paths) > 1
    }

    ambiguous_card_names = {
        normalized: sorted(card_ids)
        for normalized, card_ids
        in normalized_lookup.items()
        if len(card_ids) > 1
    }

    matched_image_count = sum(
        len(paths)
        for paths
        in matched_images_by_card.values()
    )

    print("=" * 64)
    print("CARD DATA / REFERENCE IMAGE CHECK")
    print("=" * 64)

    print(
        f"card_data entries:          {len(card_data)}"
    )

    print(
        f"Reference images:           {len(image_paths)}"
    )

    print(
        f"Matched reference images:   {matched_image_count}"
    )

    print(
        f"Cards with a reference:     "
        f"{len(matched_images_by_card)}"
    )

    print(
        f"Unmatched reference images: "
        f"{len(unmatched_images)}"
    )

    print(
        f"Cards missing references:   "
        f"{len(missing_cards)}"
    )

    print(
        f"Cards with duplicate refs:  "
        f"{len(duplicate_references)}"
    )

    print(
        f"Ambiguous reference names:  "
        f"{len(ambiguous_images)}"
    )

    print(
        f"Ambiguous card-data names:  "
        f"{len(ambiguous_card_names)}"
    )

    print_section(
        "REFERENCE IMAGES NOT FOUND IN card_data.json:",
        [
            str(
                path.relative_to(
                    REFERENCE_DIR
                )
            )
            for path in unmatched_images
        ],
    )

    print_section(
        "CARDS IN card_data.json WITHOUT A REFERENCE IMAGE:",
        [
            (
                f"{card_id} "
                f"({card_data[card_id].get('Name', card_id)})"
                if isinstance(
                    card_data[card_id],
                    dict,
                )
                else card_id
            )
            for card_id in missing_cards
        ],
    )

    print_section(
        "CARDS WITH MULTIPLE REFERENCE IMAGES:",
        [
            (
                f"{card_id}: "
                + ", ".join(
                    str(
                        path.relative_to(
                            REFERENCE_DIR
                        )
                    )
                    for path in paths
                )
            )
            for card_id, paths
            in sorted(
                duplicate_references.items()
            )
        ],
    )

    print_section(
        "REFERENCE IMAGES THAT MATCH MULTIPLE CARD IDs:",
        [
            (
                f"{path.relative_to(REFERENCE_DIR)} "
                f"could match: "
                f"{', '.join(card_ids)}"
            )
            for path, card_ids
            in ambiguous_images
        ],
    )

    print_section(
        "CARD-DATA NAMES THAT NORMALIZE TO THE SAME VALUE:",
        [
            (
                f"{normalized}: "
                f"{', '.join(card_ids)}"
            )
            for normalized, card_ids
            in sorted(
                ambiguous_card_names.items()
            )
        ],
    )

    every_image_has_card = (
        not unmatched_images
        and not ambiguous_images
    )

    every_card_has_image = (
        not missing_cards
    )

    exactly_one_image_per_card = (
        not duplicate_references
    )

    no_name_collisions = (
        not ambiguous_card_names
    )

    print()
    print("=" * 64)

    print(
        "A) Every reference image exists in card_data.json: "
        + (
            "PASS ✓"
            if every_image_has_card
            else "FAIL ✗"
        )
    )

    print(
        "B) Every card_data card has a reference image:    "
        + (
            "PASS ✓"
            if every_card_has_image
            else "FAIL ✗"
        )
    )

    print(
        "C) Exactly one reference image per card:          "
        + (
            "PASS ✓"
            if exactly_one_image_per_card
            else "FAIL ✗"
        )
    )

    print(
        "D) No normalized-name collisions:                "
        + (
            "PASS ✓"
            if no_name_collisions
            else "FAIL ✗"
        )
    )

    print("=" * 64)

    all_good = (
        every_image_has_card
        and every_card_has_image
        and exactly_one_image_per_card
        and no_name_collisions
    )

    if all_good:
        print(
            "\nEverything matches perfectly. ✓"
        )
        return 0

    print(
        "\nProblems were found. See the lists above."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())