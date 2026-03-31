"""
Generate an Excel file + prompt for Architecture 2 (frontend batch) experiments.

Reads offer set JSON files from data/offer_sets/<category>/ and produces:
  1. An Excel file (.xlsx) — one row per product, offer_set_id column for grouping.
     Uploaded directly to the model frontend as an attachment.
  2. A prompt text file (.txt) — copy-paste this into the chat alongside the attachment.

The Excel file contains the same fields used in the full prompt variant so results
are comparable to Architecture 1 API runs.

Usage:
    python src/make_frontend_file.py --category "Mechanical Keyboards" --n 100
    python src/make_frontend_file.py --category "Mechanical Keyboards" --n 100 --output data/frontend/
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_offer_sets(category, n, offer_sets_dir="data/offer_sets"):
    """Load the first N offer set JSON files for a category."""
    cat_slug = category.lower().replace(" ", "_")
    cat_dir  = Path(offer_sets_dir) / cat_slug

    if not cat_dir.exists():
        raise FileNotFoundError(f"No offer sets found at {cat_dir}")

    files = sorted(cat_dir.glob(f"{cat_slug}_*.json"))[:n]
    if not files:
        raise FileNotFoundError(f"No offer set files found in {cat_dir}")

    offer_sets = []
    for f in files:
        with open(f) as fh:
            products = json.load(fh)
        offer_sets.append((f.stem, products))   # (offer_set_id, list of products)

    return offer_sets


def build_dataframe(offer_sets):
    """Convert list of (offer_set_id, products) into a flat DataFrame."""
    rows = []
    for offer_set_id, products in offer_sets:
        for p in products:
            # Combine reviews into one readable string
            review_texts = []
            for r in (p.get("reviews") or [])[:5]:
                text = (r.get("text") or "").strip()
                if text:
                    review_texts.append(text[:200])
            reviews_combined = " | ".join(review_texts)

            # Truncate description
            description = (p.get("description") or "")[:300]

            rows.append({
                "offer_set_id":    offer_set_id,
                "product_id":      p.get("id"),
                "title":           p.get("title", ""),
                "price":           p.get("price") or p.get("base_price"),
                "rating":          p.get("rating"),
                "review_count":    p.get("review_count"),
                "position":        p.get("position"),
                "page":            p.get("page"),
                "is_sponsored":    "Sponsored"    in (p.get("tags") or []),
                "is_best_seller":  "Best Seller"  in (p.get("tags") or []),
                "is_overall_pick": "Overall Pick" in (p.get("tags") or []),
                "description":     description,
                "reviews":         reviews_combined,
            })

    return pd.DataFrame(rows)


def build_prompt(category, n_offer_sets):
    """Return the system prompt text to paste into the frontend chat."""
    return f"""You are simulating a consumer shopping for {category}.

The attached spreadsheet contains {n_offer_sets} offer sets. Each offer set has a unique offer_set_id and contains 25 products with their attributes (price, rating, review count, position, tags, description, reviews).

For each offer set, independently simulate a purchase decision:
1. Choose a consideration set of exactly 5 products you would seriously evaluate.
2. Make a final decision: either select one product to buy, or output "no_purchase" if none are suitable.
3. Treat each offer set completely independently. Do not reference or be influenced by your previous choices.

Output your decisions as a JSON array, one entry per offer set, in exactly this format:

[
  {{
    "offer_set_id": "...",
    "consideration_set": ["product_id_1", "product_id_2", "product_id_3", "product_id_4", "product_id_5"],
    "final_choice": "product_id_or_no_purchase",
    "reasoning": "one sentence"
  }},
  ...
]

Important:
- Output the JSON after every 10 offer sets so progress is not lost if the session ends.
- Keep the reasoning field to one sentence.
- Use the exact product_id strings from the spreadsheet.
- The consideration_set must contain exactly 5 product_id strings.
"""


def main():
    parser = argparse.ArgumentParser(description="Generate frontend batch file and prompt")
    parser.add_argument("--category",       required=True, help="Category name (e.g. 'Mechanical Keyboards')")
    parser.add_argument("--n",              type=int, default=100, help="Number of offer sets (default: 100)")
    parser.add_argument("--offer-sets-dir", default="data/offer_sets")
    parser.add_argument("--output",         default="data/frontend", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cat_slug = args.category.lower().replace(" ", "_")

    print(f"Loading {args.n} offer sets for {args.category}...")
    offer_sets = load_offer_sets(args.category, args.n, args.offer_sets_dir)
    print(f"  Loaded {len(offer_sets)} offer sets ({len(offer_sets) * 25} product rows)")

    # Write Excel file
    df = build_dataframe(offer_sets)
    xlsx_path = output_dir / f"frontend_{cat_slug}_{len(offer_sets):03d}.xlsx"
    df.to_excel(xlsx_path, index=False)
    print(f"  Excel file: {xlsx_path}")

    # Write prompt file
    prompt = build_prompt(args.category, len(offer_sets))
    prompt_path = output_dir / f"prompt_{cat_slug}_{len(offer_sets):03d}.txt"
    prompt_path.write_text(prompt)
    print(f"  Prompt file: {prompt_path}")

    print(f"\nInstructions:")
    print(f"  1. Open the model frontend (Gemini, ChatGPT, Claude, etc.)")
    print(f"  2. Attach: {xlsx_path}")
    print(f"  3. Paste the prompt from: {prompt_path}")
    print(f"  4. Save the model's JSON output to: data/frontend/output_{cat_slug}_<model>.json")
    print(f"  5. Run: python src/parse_frontend_output.py --input <output_file> --category \"{args.category}\"")


if __name__ == "__main__":
    main()
