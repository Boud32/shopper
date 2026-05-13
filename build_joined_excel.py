"""
Join the per-category MNL Excel exports into one file per provider, so the
full dataset is glanceable in Excel without opening 6+ files.

Output:
  data/architecture2/mnl_exports/joined/joined_gemini-3-flash-preview.xlsx
  data/architecture2/mnl_exports/joined/joined_claude-opus-4-7.xlsx

Each file has one sheet per category plus a 'README' sheet describing the
schema. The category sheets share an identical column layout.
"""
import glob
import re
from pathlib import Path
import pandas as pd


def category_from_filename(path, pattern):
    slug = re.search(pattern, path).group(1)
    return slug.replace("_", " ").title()


def join_provider(files, pattern, out_path, provider_label, source_dir_label):
    pieces = []
    for f in sorted(files):
        cat = category_from_filename(f, pattern)
        sub = pd.read_excel(f)
        sub.insert(0, "category", cat)
        pieces.append((cat, sub))

    summary_rows = []
    for cat, sub in pieces:
        n_offer = sub["offer_set_id"].nunique()
        n_np    = sub.groupby("offer_set_id")["no_purchase"].first().sum()
        n_chosen = sub["chosen"].sum()
        summary_rows.append({
            "category"        : cat,
            "offer_sets"      : n_offer,
            "rows"            : len(sub),
            "no_purchase"     : int(n_np),
            "no_purchase_pct" : f"{100*n_np/n_offer:.1f}%",
            "purchases"       : int(n_chosen),
        })

    readme = pd.DataFrame([
        ["provider",        provider_label],
        ["source_dir",      source_dir_label],
        ["row_unit",        "one product within one offer set"],
        ["offer_set_size",  "25 products per offer set"],
        ["chosen",          "1 = model selected this product as final purchase"],
        ["no_purchase",     "1 = experiment ended without any purchase (1 for all 25 rows)"],
        ["in_consideration","1 = model shortlisted this product"],
        ["features (raw)",  "position, price, rating, review_count, log_review_count, "
                            "is_sponsored, is_best_seller, is_overall_pick"],
        ["MNL z-scoring",   "applied INSIDE fit_mnl on continuous features only; "
                            "raw values are kept here so the recipient can re-fit independently"],
    ], columns=["field", "value"])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        full = pd.concat([sub for _, sub in pieces], ignore_index=True)
        full.to_excel(writer, sheet_name="all_categories", index=False)
        for cat, sub in pieces:
            sheet = cat[:31]
            sub.to_excel(writer, sheet_name=sheet, index=False)

    print(f"  → {out_path}")
    for r in summary_rows:
        print(f"      {r['category']:25s}  offer_sets={r['offer_sets']:>4d}  "
              f"rows={r['rows']:>6d}  NP={r['no_purchase']:>4d} ({r['no_purchase_pct']:>5s})")


def main():
    print("Joining Gemini-3 Flash Preview xlsx files...")
    join_provider(
        files            = glob.glob("data/architecture2/mnl_exports/gemini-3-flash-preview/mnl_data_a2_*.xlsx"),
        pattern          = r"mnl_data_a2_(.+)\.xlsx",
        out_path         = "data/architecture2/mnl_exports/joined/joined_gemini-3-flash-preview.xlsx",
        provider_label   = "gemini-3-flash-preview",
        source_dir_label = "data/architecture2/mnl_exports/gemini-3-flash-preview/",
    )

    print("\nJoining Claude Opus 4.7 xlsx files...")
    join_provider(
        files            = glob.glob("data/architecture2/mnl_exports/opus4-7/mnl_data_04-20_*.xlsx"),
        pattern          = r"mnl_data_04-20_(.+)\.xlsx",
        out_path         = "data/architecture2/mnl_exports/joined/joined_claude-opus-4-7.xlsx",
        provider_label   = "claude-opus-4-7",
        source_dir_label = "data/architecture2/mnl_exports/opus4-7/",
    )


if __name__ == "__main__":
    main()
