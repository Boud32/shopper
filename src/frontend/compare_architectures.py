"""
Compare Architecture 1 (API) vs Architecture 2 (frontend) results.

Computes agreement metrics per offer set and plots agreement rate vs. session
position to test the context-degradation hypothesis: does agreement decline
as the model processes more offer sets in a single session?

Usage:
    python src/compare_architectures.py \\
        --api-results     data/results/ \\
        --frontend-results data/results/frontend/ \\
        --category        "Mechanical Keyboards" \\
        --api-provider    groq \\
        --frontend-provider gemini-frontend
"""

import argparse
from pathlib import Path

import pandas as pd


def jaccard(set_a, set_b):
    """Jaccard similarity between two sets."""
    a = set(set_a)
    b = set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def load_decisions(results_dir, provider, category, offer_sets_dir="data/offer_sets"):
    """
    Load raw decision data (consideration set, final choice, session_position)
    directly from result JSON files for a given provider and category.

    Returns a DataFrame with one row per offer set:
        offer_set_id, final_choice, consideration_set (list), session_position
    """
    import json

    results_path = Path(results_dir)
    pattern      = f"result_{provider}_*.json"
    files        = sorted(results_path.glob(pattern))

    rows = []
    for f in files:
        with open(f) as fh:
            result = json.load(fh)

        meta     = result.get("metadata", {})
        decision = result.get("decision", {})
        source   = meta.get("source_batch", "")

        # Infer offer_set_id from source_batch filename
        offer_set_id = Path(source).stem if source else f.stem.replace(f"result_{provider}_", "")

        # Filter by category via offer_set_id prefix
        cat_slug = category.lower().replace(" ", "_")
        if not offer_set_id.startswith(cat_slug):
            continue

        rows.append({
            "offer_set_id":     offer_set_id,
            "final_choice":     decision.get("final_choice", "no_purchase"),
            "consideration_set": decision.get("consideration_set") or [],
            "session_position": meta.get("session_position"),   # None for Architecture 1
        })

    return pd.DataFrame(rows)


def compare(api_dir, frontend_dir, category, api_provider, frontend_provider, offer_sets_dir):
    """
    Merge Architecture 1 and Architecture 2 decisions by offer_set_id,
    compute per-offer-set agreement metrics.

    Returns a DataFrame with columns:
        offer_set_id, session_position,
        choice_agree (bool), jaccard (float)
    """
    api_df      = load_decisions(api_dir,      api_provider,      category, offer_sets_dir)
    frontend_df = load_decisions(frontend_dir, frontend_provider, category, offer_sets_dir)

    if api_df.empty:
        raise ValueError(f"No results found for provider '{api_provider}' in {api_dir}")
    if frontend_df.empty:
        raise ValueError(f"No results found for provider '{frontend_provider}' in {frontend_dir}")

    merged = pd.merge(api_df, frontend_df, on="offer_set_id", suffixes=("_api", "_fe"))
    print(f"Matched offer sets: {len(merged)} "
          f"(API: {len(api_df)}, Frontend: {len(frontend_df)})")

    merged["choice_agree"] = merged["final_choice_api"] == merged["final_choice_fe"]
    merged["jaccard"]      = merged.apply(
        lambda r: jaccard(r["consideration_set_api"], r["consideration_set_fe"]), axis=1
    )
    merged["session_position"] = merged["session_position_fe"]

    return merged[["offer_set_id", "session_position", "choice_agree", "jaccard"]]


def print_summary(results):
    n     = len(results)
    agree = results["choice_agree"].mean()
    jac   = results["jaccard"].mean()
    print("=" * 50)
    print(f"Architecture comparison — {n} matched offer sets")
    print("=" * 50)
    print(f"  Choice agreement : {agree:.1%}")
    print(f"  Mean Jaccard     : {jac:.3f}")
    print()

    # Break into quartiles by session position if available
    if results["session_position"].notna().any():
        results = results.dropna(subset=["session_position"]).copy()
        results["session_position"] = results["session_position"].astype(int)
        results["quartile"] = pd.qcut(results["session_position"], q=4,
                                      labels=["Q1 (early)", "Q2", "Q3", "Q4 (late)"])
        print("  Agreement by session position quartile:")
        for q, grp in results.groupby("quartile", observed=True):
            print(f"    {q}: {grp['choice_agree'].mean():.1%}  "
                  f"(Jaccard {grp['jaccard'].mean():.3f}, n={len(grp)})")
        print()
        print("  If agreement declines from Q1 to Q4, context degradation is confirmed.")


def main():
    parser = argparse.ArgumentParser(description="Compare API vs. frontend architecture results")
    parser.add_argument("--api-results",       default="data/results")
    parser.add_argument("--frontend-results",  default="data/results/frontend")
    parser.add_argument("--category",          required=True)
    parser.add_argument("--api-provider",      default="groq")
    parser.add_argument("--frontend-provider", default="gemini-frontend")
    parser.add_argument("--offer-sets-dir",    default="data/offer_sets")
    args = parser.parse_args()

    results = compare(
        api_dir          = args.api_results,
        frontend_dir     = args.frontend_results,
        category         = args.category,
        api_provider     = args.api_provider,
        frontend_provider = args.frontend_provider,
        offer_sets_dir   = args.offer_sets_dir,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
