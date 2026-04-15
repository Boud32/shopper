"""
Generate an Excel file + prompt for Architecture 2 (frontend batch) experiments.

Reads offer set JSON files from data/offer_sets/<category>/ and produces:
  1. An Excel file (.xlsx) — one row per product, offer_set_id column for grouping.
     Uploaded directly to the model frontend as an attachment.
  2. A prompt text file (.txt) — copy-paste this into the chat alongside the attachment.

The Excel file contains the same fields used in the full prompt variant so results
are comparable to Architecture 1 API runs.

Row order within each offer set is shuffled (seeded by offer_set_id) so the model
cannot shortcut by taking the first N rows. The 'position' column still reflects
search rank and the model must read it explicitly.

Usage:
    python src/frontend/make_frontend_file.py --category "Backpacks" --n 500
    python src/frontend/make_frontend_file.py --category "Backpacks" --n 100 --offset 0
"""

import argparse
import hashlib
import json
import random
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


def _human_scan_order(products, rng):
    """
    Return products in a human-like scan order seeded by rng.

    Three modes are assigned randomly per offer set:
      linear   (40%) — roughly top-to-bottom with small noise: [1,3,2,5,4,8,...]
      snake    (35%) — scan forward to ~65% depth, then reverse: [1,2,4,3,8,...,25,22,19,...]
      accordion(25%) — alternate front/back chunks, snaking in both directions:
                        [1,2,3,25,24,4,5,23,22,6,7,21,...]

    All modes start near position 1 (lowest positions always surface first).
    The accordion pattern produces the most aggressive end-jumping behavior.
    """
    by_pos = sorted(products, key=lambda p: p.get("position", 99))
    n = len(by_pos)
    sigma = 1.5   # within-segment noise

    def noisy_sort(items, reverse=False):
        return sorted(items,
                      key=lambda p: p.get("position", 0) + rng.gauss(0, sigma),
                      reverse=reverse)

    roll = rng.random()

    if roll < 0.40:
        # Linear: small noise, monotonically increasing on average
        return noisy_sort(by_pos, reverse=False)

    elif roll < 0.75:
        # Snake: forward to pivot, then reverse the tail
        pivot = rng.randint(int(n * 0.55), int(n * 0.80))
        return noisy_sort(by_pos[:pivot]) + noisy_sort(by_pos[pivot:], reverse=True)

    else:
        # Accordion: alternate front and back chunks, snaking inward
        result = []
        lo, hi = 0, n - 1
        take_front = True
        while lo <= hi:
            k = rng.randint(2, 4)
            if take_front:
                chunk = by_pos[lo : min(lo + k, hi + 1)]
                result.extend(noisy_sort(chunk))
                lo += len(chunk)
            else:
                chunk = by_pos[max(hi - k + 1, lo) : hi + 1]
                result.extend(noisy_sort(chunk, reverse=True))
                hi -= len(chunk)
            take_front = not take_front
        return result


def build_dataframe(offer_sets):
    """
    Convert list of (offer_set_id, products) into a flat DataFrame.

    Row order within each offer set simulates a human scan pattern (seeded by
    offer_set_id for reproducibility). The model cannot shortcut by taking the
    first N rows — it must read the 'position' column as an explicit attribute.
    """
    rows = []
    for offer_set_id, products in offer_sets:
        seed = int(hashlib.md5(offer_set_id.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        shuffled = _human_scan_order(products, rng)

        for p in shuffled:
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

I will attach the spreadsheet in the next message. Please read these instructions first and confirm you understand before I send the data.

The spreadsheet contains {n_offer_sets} offer sets. Each offer set has a unique offer_set_id and contains 25 products with their attributes (price, rating, review count, position, tags, description, reviews).

Important: rows within each offer set are in RANDOM order — do not use row order as a proxy for quality or relevance. The 'position' column contains the actual search rank (1 = top of page) and should be treated as one attribute among many, not as a sorting key.

Do not write or execute any code. For each offer set, act as a human consumer: read the product listings and make each decision using your own judgment.

For each offer set, independently simulate a purchase decision:
1. Choose a consideration set of exactly 5 products you would seriously evaluate, based on their attributes.
2. Make a final decision: either select one product to buy, or output "no_purchase" if nothing is compelling enough.

No_purchase is the right choice whenever the selection is uninspiring, the price feels too high for the quality on offer, ratings are mediocre, or nothing stands out clearly. A browsing session that ends without buying is completely normal — target approximately 30% no_purchase overall. Track your running rate and stay honest: if you have been buying too readily, be more demanding.

Output your decisions as a JSON array, one entry per offer set:

[
  {{
    "offer_set_id": "...",
    "consideration_set": ["product_id_1", "product_id_2", "product_id_3", "product_id_4", "product_id_5"],
    "final_choice": "product_id_or_no_purchase",
    "reasoning": "one sentence"
  }},
  ...
]

Rules:
- Use the exact product_id strings from the spreadsheet.
- consideration_set must contain exactly 5 product_id strings.
- reasoning must be one sentence.

Please confirm you understand, then I will send the spreadsheet.
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

    print(f"\nNext steps:")
    print(f"  1. Open your model frontend (Claude, ChatGPT, Gemini, etc.)")
    print(f"  2. Attach:     {xlsx_path}")
    print(f"  3. Paste prompt from: {prompt_path}")
    print(f"  4. Save the JSON output to, e.g.:")
    print(f"       data/frontend/output_{cat_slug}_{batch_label}_<model>.json")
    print(f"  5. Parse + export:")
    print(f"       python src/frontend/parse_frontend_output.py \\")
    print(f"           --input data/frontend/output_{cat_slug}_{batch_label}_<model>.json \\")
    print(f"           --category \"{args.category}\" --provider <model-name>")


if __name__ == "__main__":
    main()
