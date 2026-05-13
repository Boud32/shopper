"""
Export long-format experiment data to Excel for independent MNL verification.

Each row is one product in one offer set. The file contains everything needed
to re-fit an MNL from scratch: product attributes, tags, and the model's choices.
Features are exported in raw (un-normalized) units so the recipient can apply
their own normalization or transformation.

Output columns:
    offer_set_id       — identifier shared across all 25 products in an offer set
                         (e.g. "backpacks_001")
    product_id         — unique product identifier
    title              — product name (for readability / sanity checking)
    position           — search rank (1-indexed)
    price              — product price ($)
    rating             — star rating (0–5)
    review_count       — raw review count
    log_review_count   — log(1 + review_count)
    is_sponsored       — 1/0
    is_best_seller     — 1/0
    is_overall_pick    — 1/0
    in_consideration   — 1 if model shortlisted this product
    chosen             — 1 if model selected this product as final choice
                         (all zeros for no_purchase experiments)
    no_purchase        — 1 for every row in an experiment that ended in no_purchase

Usage:
    python src/export_for_analysis.py --architecture a2
    python src/export_for_analysis.py --architecture a1
    python src/export_for_analysis.py --architecture a2 --category "Backpacks"
"""

import argparse
import json
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, ".")
from src.analysis_helper import load_results_to_dataframe


ARCH_DIRS = {
    "a1": "data/architecture1/results",
    "a2": "data/architecture2/results",
}


def _load_titles(offer_sets_dir, df):
    """
    Build a (offer_set_id, product_id) → title lookup from offer set JSON files.

    offer_set_id is expected to be in the form "{category_slug}_{number}" (e.g.
    "backpacks_001"), which maps directly to the filename under offer_sets_dir.
    """
    lookup = {}
    pairs = df[["offer_set_id", "category"]].drop_duplicates()
    for _, row in pairs.iterrows():
        offer_set_id = row["offer_set_id"]
        category = row["category"]
        if not category:
            continue
        slug = category.lower().replace(" ", "_")
        path = Path(offer_sets_dir) / slug / f"{offer_set_id}.json"
        if not path.exists():
            continue
        with open(path) as f:
            products = json.load(f)
        for p in products:
            lookup[(offer_set_id, p["id"])] = p.get("title", "")
    return lookup


def build_export_df(df, offer_sets_dir="data/offer_sets"):
    """
    Reshape the long-format DataFrame into the export format.
    Strips provider/model/variant columns not needed for MNL verification.
    """
    df = df.copy()

    # Derive offer_set_id from experiment_id by stripping the result_<provider>_ prefix.
    # The first regex handles both architectures:
    #   A1: result_groq_backpacks_001         → backpacks_001
    #   A2: result_gemini-frontend_backpacks_001 → backpacks_001
    # (hyphens in the provider token are not underscores, so [^_]+ captures the whole token)
    df["offer_set_id"] = df["experiment_id"].str.replace(
        r"^result_[^_]+_", "", regex=True
    )

    # Load product titles from offer set files
    title_lookup = _load_titles(offer_sets_dir, df)
    df["title"] = df.apply(
        lambda r: title_lookup.get((r["offer_set_id"], r["product_id"]), ""), axis=1
    )

    df["log_review_count"] = np.log1p(df["review_count"].astype(float))
    df["in_consideration"] = df["in_consideration"].astype(int)
    df["chosen"]           = df["chosen"].astype(int)
    df["is_sponsored"]     = df["is_sponsored"].astype(int)
    df["is_best_seller"]   = df["is_best_seller"].astype(int)
    df["is_overall_pick"]  = df["is_overall_pick"].astype(int)

    # Explicit no_purchase flag: 1 for all rows in an experiment where no product was chosen
    no_purchase_exps = set(
        df.groupby("offer_set_id")
        .filter(lambda g: g["chosen"].sum() == 0)["offer_set_id"]
    )
    df["no_purchase"] = df["offer_set_id"].isin(no_purchase_exps).astype(int)

    cols = [
        "offer_set_id", "product_id", "title",
        "position", "price", "rating", "review_count", "log_review_count",
        "is_sponsored", "is_best_seller", "is_overall_pick",
        "in_consideration", "chosen", "no_purchase",
    ]
    return df[cols + ["category"]].sort_values(["offer_set_id", "position"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", default="a2", choices=["a1", "a2"],
                        help="Which architecture's results to export (default: a2)")
    parser.add_argument("--category",     default=None,
                        help="Export a single category only (default: all)")
    parser.add_argument("--output",       default="data/architecture2",
                        help="Output directory (default: data/architecture2)")
    args = parser.parse_args()

    results_dir = ARCH_DIRS[args.architecture]
    print(f"Loading results from {results_dir}...")
    df = load_results_to_dataframe(results_dir=results_dir, include_ablation=False)
    print(f"  {df['experiment_id'].nunique()} experiments loaded")

    export = build_export_df(df)

    if args.category:
        export = export[export["category"] == args.category]

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    categories = export["category"].dropna().unique()
    for cat in sorted(categories):
        cat_df   = export[export["category"] == cat].drop(columns=["category"])
        slug     = cat.lower().replace(" ", "_")
        out_file = output_path / f"mnl_data_{args.architecture}_{slug}.xlsx"
        cat_df.to_excel(out_file, index=False)
        n_exp        = cat_df["offer_set_id"].nunique()
        n_no_purchase = cat_df.groupby("offer_set_id")["no_purchase"].first().sum()
        print(f"  {cat}: {n_exp} offer sets, {int(n_no_purchase)} no_purchase → {out_file}")

    print("Done.")


if __name__ == "__main__":
    main()
