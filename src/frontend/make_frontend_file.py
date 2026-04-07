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


def load_offer_sets(category, n, offset=0, offer_sets_dir="data/offer_sets"):
    """Load N offer set JSON files for a category, starting at offset."""
    cat_slug = category.lower().replace(" ", "_")
    cat_dir  = Path(offer_sets_dir) / cat_slug

    if not cat_dir.exists():
        raise FileNotFoundError(f"No offer sets found at {cat_dir}")

    files = sorted(cat_dir.glob(f"{cat_slug}_*.json"))[offset:offset + n]
    if not files:
        raise FileNotFoundError(f"No offer set files found in {cat_dir}")

    offer_sets = []
    for f in files:
        with open(f) as fh:
            products = json.load(fh)
        offer_sets.append((f.stem, products))

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
2. Make a final decision: either select one product to buy, or output "no_purchase" if nothing is compelling enough.

Target: approximately 30% of your decisions should be no_purchase. You can see your prior decisions in this session — track your running no_purchase rate and adjust your selectivity accordingly. If you have been purchasing too readily, be more demanding. If you have made too many no_purchase decisions, be more willing to commit.

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
- Keep the reasoning field to one sentence.
- Use the exact product_id strings from the spreadsheet.
- The consideration_set must contain exactly 5 product_id strings.
"""


def main():
    parser = argparse.ArgumentParser(description="Generate frontend batch file and prompt")
    parser.add_argument("--category",       required=True)
    parser.add_argument("--n",              type=int, default=100, help="Number of offer sets per batch")
    parser.add_argument("--offset",         type=int, default=0,   help="Skip first N offer sets (for batching)")
    parser.add_argument("--offer-sets-dir", default="data/offer_sets")
    parser.add_argument("--output",         default="data/frontend")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cat_slug = args.category.lower().replace(" ", "_")

    print(f"Loading {args.n} offer sets for {args.category} (offset={args.offset})...")
    offer_sets = load_offer_sets(args.category, args.n, offset=args.offset,
                                 offer_sets_dir=args.offer_sets_dir)
    print(f"  Loaded {len(offer_sets)} offer sets ({len(offer_sets) * 25} product rows)")

    batch_label = f"{args.offset + 1:03d}-{args.offset + len(offer_sets):03d}"

    df = build_dataframe(offer_sets)
    xlsx_path = output_dir / f"frontend_{cat_slug}_{batch_label}.xlsx"
    df.to_excel(xlsx_path, index=False)
    print(f"  Excel file: {xlsx_path}")

    # One prompt file per category — all equal-sized batches share the same prompt
    prompt_path = output_dir / f"prompt_{cat_slug}.txt"
    prompt_path.write_text(build_prompt(args.category, args.n))
    print(f"  Prompt file: {prompt_path}  (shared across all {args.n}-offer-set batches)")

    print(f"\nInstructions:")
    print(f"  1. Open Gemini frontend")
    print(f"  2. Attach: {xlsx_path}")
    print(f"  3. Paste the prompt from: {prompt_path}")
    print(f"  4. Save output to: data/frontend/output_{cat_slug}_{batch_label}_gemini.json")
    print(f"  5. Run: python src/frontend/parse_frontend_output.py --input <output_file> --category \"{args.category}\"")


if __name__ == "__main__":
    main()
