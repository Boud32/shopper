"""
Analysis utilities for the Shopper experiment engine.

The core output is a long-format pandas DataFrame with one row per product per
experiment — the standard structure for Multinomial Logit (MNL) estimation.

Each row captures:
  - Experiment context  (id, model, timestamp)
  - Product attributes  (price, rating, review_count, position, page, tags)
  - Choice outcomes     (in_consideration, chosen)

Usage:
    from src.analysis import load_results_to_dataframe
    df = load_results_to_dataframe()
    print(df.groupby("provider")["chosen"].value_counts())
"""

import json
import os
from pathlib import Path

import pandas as pd


def load_results_to_dataframe(
    results_dir="data/results",
    experiments_dir="data/experiments",
):
    """
    Load all result files and their corresponding batch files into a long-format DataFrame.

    Each result file contains a model's decision (consideration set + final choice)
    and a pointer to the batch file it was run on. This function joins them so every
    product in a batch gets a row, with boolean flags for whether it was considered
    and/or chosen.

    Args:
        results_dir:     Directory containing result_*.json files.
        experiments_dir: Fallback directory for batch files if the path in metadata
                         is not found on disk (e.g. after moving files).

    Returns:
        pd.DataFrame with columns:
            experiment_id   str    Stem of the result filename (e.g. "result_gemini_...")
            provider        str    Model provider key (e.g. "gemini")
            model           str    Full model ID (e.g. "gemini-2.5-flash")
            timestamp       int    Unix timestamp of the run
            product_id      str    Product ID from the batch
            category        str    Product category
            price           float  Price at experiment time (post-mutation)
            rating          float  Product rating
            review_count    int    Number of reviews
            position        int    Search result position (1-indexed); None if not set
            page            int    Page number (1-indexed); None if not set
            is_sponsored    bool   True if "Sponsored" in product tags
            is_best_seller  bool   True if "Best Seller" in product tags
            is_overall_pick bool   True if "Overall Pick" in product tags
            in_consideration bool  True if product was in the model's consideration set
            chosen          bool   True if product was the model's final choice
    """
    rows = []

    for result_file in sorted(Path(results_dir).glob("result_*.json")):
        with open(result_file) as f:
            result = json.load(f)

        metadata = result.get("metadata", {})
        source_batch = metadata.get("source_batch", "")

        # Resolve batch file path — fall back to experiments_dir if needed
        if not os.path.exists(source_batch):
            batch_filename = os.path.basename(source_batch)
            source_batch = os.path.join(experiments_dir, batch_filename)

        if not os.path.exists(source_batch):
            print(f"  Warning: batch file not found for {result_file.name}, skipping.")
            continue

        with open(source_batch) as f:
            products = json.load(f)

        decision = result.get("decision", {})
        consideration_set = set(decision.get("consideration_set", []))
        final_choice = decision.get("final_choice")
        experiment_id = result_file.stem

        for product in products:
            product_id = product.get("id")
            tags = product.get("tags") or []

            rows.append({
                "experiment_id":   experiment_id,
                "provider":        metadata.get("provider"),
                "model":           metadata.get("model"),
                "timestamp":       metadata.get("timestamp"),
                "product_id":      product_id,
                "category":        product.get("category"),
                "price":           product.get("price"),
                "rating":          product.get("rating"),
                "review_count":    product.get("review_count"),
                "position":        product.get("position"),
                "page":            product.get("page"),
                "is_sponsored":    "Sponsored" in tags,
                "is_best_seller":  "Best Seller" in tags,
                "is_overall_pick": "Overall Pick" in tags,
                "in_consideration": product_id in consideration_set,
                "chosen":          product_id == final_choice,
            })

    df = pd.DataFrame(rows)

    if df.empty:
        print("No results found.")
        return df

    # Coerce types
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").astype("Int64")
    df["position"] = pd.to_numeric(df["position"], errors="coerce").astype("Int64")
    df["page"] = pd.to_numeric(df["page"], errors="coerce").astype("Int64")

    return df


def summary(df):
    """Print a quick summary of the experiment DataFrame."""
    if df.empty:
        print("DataFrame is empty.")
        return

    print(f"Experiments: {df['experiment_id'].nunique()}")
    print(f"Providers:   {sorted(df['provider'].dropna().unique())}")
    print(f"Categories:  {sorted(df['category'].dropna().unique())}")
    print(f"Total rows:  {len(df)}\n")

    print("Choices by provider:")
    chosen = df[df["chosen"]].groupby("provider")["product_id"].count()
    print(chosen.to_string(), "\n")

    print("Consideration set size by provider (mean):")
    considered = df[df["in_consideration"]].groupby(
        ["experiment_id", "provider"]
    ).size().reset_index(name="set_size")
    print(considered.groupby("provider")["set_size"].mean().round(1).to_string())


if __name__ == "__main__":
    df = load_results_to_dataframe()
    summary(df)
    print("\nSample rows:")
    print(df[df["chosen"]].to_string(index=False))
