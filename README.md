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