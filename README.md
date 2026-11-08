# PvZH Deck Recognition API

Image-recognition backend for [PvZH Vault](https://pvzhvault.com).

The API accepts a screenshot of a completed PvZ Heroes deck, extracts the cards, reads the deck information with OCR, identifies each card against a reference library, and returns the completed deck as JSON.

## What it recognizes

- Deck name
- Card copy counts
- Card costs
- Individual cards
- Plant or zombie classes
- The most likely hero class combination
- Cards that may require manual review

## How it works

The recognition pipeline has three main stages:

1. **Deck splitting**

   `split-deck.js` detects the card grid and saves each card as an individual crop.

2. **OCR**

   `read-deck.js` uses Tesseract.js to read:

   - The deck name
   - The copy count for each card
   - The cost of each card

3. **Card identification**

   `identify-cards.py` compares the extracted cards against a precomputed reference index using OpenCV image features.

The final result is returned from:

```text
POST /api/recognize-deck

## Reference cards

Recognition compares each uploaded card against `reference_cards/`. Both sides go through `card_vision.py` first:

1. **Alignment.** Every card is matched against `card_template.npy` (an edge template of the deck-view UI shared by all cards) and warped into the same 200×150 frame, regardless of how it was cropped or scaled.
2. **Art-only matching.** The cost badge, copy-count box and attack/health badges are masked out. They look the same on every card of a given cost, and used to produce near-ties between different cards (e.g. Evolutionary Leap vs Lurch for Lunch).

### Adding or replacing a reference

`reference_cards_original/` holds the raw captures; `reference_cards/` is generated from it. To add or replace a card:

1. Crop the card from a deck screenshot (a crop from `split-deck.js` works well) and save it to `reference_cards_original/` named after the card, e.g. `Sow_Magic_Beans.png`.
2. Run `python normalize-references.py`.
3. Check `reference_review/flagged.txt` and the `sheet_XX.jpg` contact sheets.

Use `--rebuild-template` only if the game's card UI changes.

### Captures worth replacing

These still match correctly but are the weakest references: clipped (Poison Mushroom, Pair of Pears, Sow Magic Beans), low resolution (Pear Pal, Vanilla, Pea-Nut, Button Mushroom, Copter Commando) or captured greyed-out (Gargantuar-Throwing Imp, Turquoise Skull Zombie, Planet of the Grapes, Pecanolith).
