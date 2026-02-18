"""
Ingest product data from the McAuley Lab Amazon Reviews 2023 dataset on Hugging Face.

Streams metadata + reviews, filters to target product types, joins reviews to products
by parent_asin, and writes to data/seed_catalog.json.

Each unique HF source file is streamed only once, even when shared across categories.

Usage:
    python src/ingest_kaggle.py
    python src/ingest_kaggle.py --categories "Wireless Headphones" "Gaming Mice"
    python src/ingest_kaggle.py --products-per-category 50 --reviews-per-product 3
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

HF_BASE = "hf://datasets/McAuley-Lab/Amazon-Reviews-2023"


def _parquet(category_dir):
    return {"fmt": "parquet", "path": f"{HF_BASE}/{category_dir}/full-*.parquet"}

def _jsonl_meta(name):
    return {"fmt": "json", "path": f"{HF_BASE}/raw/meta_categories/meta_{name}.jsonl"}

def _jsonl_review(name):
    return {"fmt": "json", "path": f"{HF_BASE}/raw/review_categories/{name}.jsonl"}


CATEGORY_CONFIGS = {
    "Wireless Headphones": {
        "meta_sources":   [_parquet("raw_meta_Electronics")],
        "review_sources": [_jsonl_review("Electronics")],
        # Require "wireless" or "bluetooth" qualifier to exclude wired headphones,
        # replacement ear tips, and other accessories that match "headphone"/"earbuds" alone.
        "keywords": ["wireless headphone", "bluetooth headphone", "wireless earbud",
                     "bluetooth earbud", "true wireless"],
    },
    "Smartwatches": {
        "meta_sources":   [_parquet("raw_meta_Electronics"),
                           _parquet("raw_meta_Cell_Phones_and_Accessories")],
        "review_sources": [_jsonl_review("Electronics"),
                           _jsonl_review("Cell_Phones_and_Accessories")],
        "keywords": ["smartwatch", "smart watch", "fitness tracker"],
    },
    "Mechanical Keyboards": {
        "meta_sources":   [_parquet("raw_meta_Electronics")],
        "review_sources": [_jsonl_review("Electronics")],
        "keywords": ["mechanical keyboard", "gaming keyboard"],
    },
    "Gaming Mice": {
        "meta_sources":   [_parquet("raw_meta_Electronics")],
        "review_sources": [_jsonl_review("Electronics")],
        # Drop "wireless mouse" — too broad, matches generic office mice.
        # "gaming mice" catches plural titles; "esports mouse" catches competitive peripherals.
        "keywords": ["gaming mouse", "gaming mice", "esports mouse"],
    },
    "Toothbrushes": {
        "meta_sources":   [_jsonl_meta("Health_and_Household")],
        "review_sources": [_jsonl_review("Health_and_Household")],
        "keywords": ["electric toothbrush", "toothbrush"],
    },
    "Running Shoes": {
        "meta_sources":   [_jsonl_meta("Sports_and_Outdoors"),
                           _jsonl_meta("Clothing_Shoes_and_Jewelry")],
        "review_sources": [_jsonl_review("Sports_and_Outdoors"),
                           _jsonl_review("Clothing_Shoes_and_Jewelry")],
        "keywords": ["running shoe", "running shoes"],
    },
}


def parse_price(price_str):
    if not price_str or price_str == "None":
        return None
    cleaned = re.sub(r"[^\d.]", "", str(price_str))
    if not cleaned:
        return None
    try:
        value = float(cleaned)
        return value if value > 0 else None
    except ValueError:
        return None


def matches_keywords(item, keywords):
    title = (item.get("title") or "").lower()
    categories = item.get("categories") or []
    cat_text = " ".join(str(c) for c in categories).lower() if isinstance(categories, list) else str(categories).lower()
    return any(kw.lower() in title or kw.lower() in cat_text for kw in keywords)


def _stream_jsonl(path):
    """Stream a HuggingFace JSONL file line-by-line, yielding dicts.

    Using fsspec directly avoids datasets' schema inference, which fails when
    a column changes type mid-file (e.g. price as int then string).
    """
    import fsspec
    with fsspec.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _open_stream(source):
    fmt, path = source["fmt"], source["path"]
    if fmt == "parquet":
        return load_dataset("parquet", data_files={"full": path}, split="full", streaming=True)
    return _stream_jsonl(path)


def stream_metadata_multi(source, category_configs, category_limits):
    """
    Stream one metadata source file once, collecting products for all categories that use it.

    category_configs: {cat_name: {"keywords": [...]}}
    category_limits:  {cat_name: max_products_to_collect}
    Returns:          {cat_name: {parent_asin: item}}
    """
    path_label = source["path"].split("/")[-1]
    cats = list(category_configs.keys())
    print(f"  Streaming metadata from {path_label} ({', '.join(cats)})...")
    ds = _open_stream(source)

    results = {cat: {} for cat in category_configs}
    full_categories = set()
    scanned = 0

    for item in ds:
        scanned += 1
        if scanned % 50_000 == 0:
            counts = ", ".join(f"{cat}: {len(results[cat])}" for cat in cats)
            print(f"    Scanned {scanned:,} | {counts}")

        title = item.get("title")
        if not title or not title.strip():
            continue
        if parse_price(item.get("price")) is None:
            continue
        parent_asin = item.get("parent_asin")
        if not parent_asin:
            continue

        for cat, cfg in category_configs.items():
            if cat in full_categories or parent_asin in results[cat]:
                continue
            if matches_keywords(item, cfg["keywords"]):
                results[cat][parent_asin] = item
                if len(results[cat]) >= category_limits[cat]:
                    full_categories.add(cat)

        if len(full_categories) >= len(category_configs):
            break

    print(f"    Done. Scanned {scanned:,} items.")
    for cat in cats:
        print(f"      {cat}: {len(results[cat])} products")
    return results


def fetch_reviews_multi(source, asin_set, per_product, max_scan=5_000_000):
    """
    Stream one review source file once, collecting reviews for all requested ASINs.

    Stops early once all products are satisfied OR max_scan rows are read —
    whichever comes first. Products with fewer than per_product reviews are
    still returned; the caller filters out those with zero reviews.

    asin_set:    set of parent_asins to collect reviews for
    per_product: max reviews to keep per product (takes verified + most helpful)
    max_scan:    hard limit on rows read (default 5M) to avoid full-file scans
    Returns:     {parent_asin: [review_dicts]}
    """
    if not asin_set:
        return {}

    path_label = source["path"].split("/")[-1]
    print(f"  Streaming reviews from {path_label} ({len(asin_set)} products, max {max_scan:,} rows)...")
    ds = _open_stream(source)

    candidates = defaultdict(list)
    satisfied = set()
    scanned = 0

    for item in ds:
        scanned += 1
        if scanned % 500_000 == 0:
            print(f"    Scanned {scanned:,} reviews, satisfied {len(satisfied)}/{len(asin_set)}...")

        asin = item.get("parent_asin")
        if asin not in asin_set or asin in satisfied:
            continue

        candidates[asin].append(item)
        if len(candidates[asin]) >= per_product:
            satisfied.add(asin)

        if len(satisfied) >= len(asin_set):
            break

        if scanned >= max_scan:
            print(f"    Hit max_scan limit ({max_scan:,}). Stopping with {len(satisfied)}/{len(asin_set)} fully satisfied.")
            break

    reviews = {}
    for asin, revs in candidates.items():
        revs.sort(
            key=lambda r: (1 if r.get("verified_purchase") else 0, r.get("helpful_vote") or 0),
            reverse=True,
        )
        reviews[asin] = revs[:per_product]

    print(f"    Done. Scanned {scanned:,} reviews, got reviews for {len(reviews)} products.")
    return reviews


def transform_product(meta, reviews, category, idx):
    desc_raw = meta.get("description") or []
    description = "\n".join(str(s) for s in desc_raw if s).strip() if isinstance(desc_raw, list) else str(desc_raw).strip()

    features_raw = meta.get("features") or []
    features = [str(f).strip() for f in features_raw if f and str(f).strip()] if isinstance(features_raw, list) else []

    rating = meta.get("average_rating") or 0.0
    cat_slug = category.lower().replace(" ", "_")

    return {
        "id": f"prod_{cat_slug}_{idx:03d}",
        "parent_asin": meta.get("parent_asin", ""),
        "category": category,
        "title": meta.get("title", ""),
        "description": description,
        "base_price": parse_price(meta.get("price")),
        "rating": round(float(rating), 1),
        "review_count": int(meta.get("rating_number") or 0),
        "features": features,
        "reviews": [
            {
                "rating": r.get("rating", 0),
                "title": r.get("title", ""),
                "text": r.get("text", ""),
                "verified": bool(r.get("verified_purchase")),
                "helpful_votes": r.get("helpful_vote") or 0,
            }
            for r in reviews
        ],
        "store": meta.get("store") or "",
        "tags": [],
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest Amazon Reviews 2023 data into seed catalog")
    parser.add_argument("--categories", nargs="+", default=list(CATEGORY_CONFIGS.keys()),
                        choices=list(CATEGORY_CONFIGS.keys()))
    parser.add_argument("--products-per-category", type=int, default=200)
    parser.add_argument("--reviews-per-product", type=int, default=5)
    parser.add_argument("--output", type=str, default="data/seed_catalog.json")
    args = parser.parse_args()

    selected = {cat: CATEGORY_CONFIGS[cat] for cat in args.categories}

    # ── Phase 1: Metadata — one pass per unique source file ──────────────────
    print("\n" + "=" * 60)
    print("PHASE 1: Streaming metadata")
    print("=" * 60)

    # Map source path → {cat: cfg} for categories that need it, preserving source dict
    meta_sources_by_path = {}   # path → source dict
    meta_cats_by_path = defaultdict(dict)  # path → {cat: cfg}
    for cat, cfg in selected.items():
        for src in cfg["meta_sources"]:
            meta_sources_by_path[src["path"]] = src
            meta_cats_by_path[src["path"]][cat] = cfg

    all_products = {cat: {} for cat in selected}  # cat → {asin: meta}

    for src_path, src in meta_sources_by_path.items():
        # Only include categories that still need more products from this source
        active_cats = {
            cat: cfg for cat, cfg in meta_cats_by_path[src_path].items()
            if len(all_products[cat]) < args.products_per_category
        }
        if not active_cats:
            continue
        limits = {cat: args.products_per_category - len(all_products[cat]) for cat in active_cats}
        batch = stream_metadata_multi(src, active_cats, limits)
        for cat, products in batch.items():
            remaining = args.products_per_category - len(all_products[cat])
            for asin, item in list(products.items())[:remaining]:
                all_products[cat][asin] = item

    # ── Phase 2: Reviews — one pass per unique source file ───────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: Streaming reviews")
    print("=" * 60)

    # Map each review source path to the set of ASINs that need reviews from it.
    # An ASIN is added to the FIRST review source for its category; if that source
    # doesn't have it, the second source will pick it up since the ASIN won't be
    # in all_reviews yet when that source is processed.
    review_sources_by_path = {}   # path → source dict
    review_asins_by_path = defaultdict(set)  # path → set of asins
    for cat, cfg in selected.items():
        asins = set(all_products[cat].keys())
        for src in cfg["review_sources"]:
            review_sources_by_path[src["path"]] = src
            review_asins_by_path[src["path"]].update(asins)

    all_reviews = {}  # asin → [reviews]

    for src_path, src in review_sources_by_path.items():
        needed = review_asins_by_path[src_path] - all_reviews.keys()
        if not needed:
            continue
        batch = fetch_reviews_multi(src, needed, args.reviews_per_product)
        all_reviews.update(batch)

    # ── Phase 3: Transform and write ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3: Transforming output")
    print("=" * 60)

    full_catalog = []
    for cat in args.categories:
        count_before = len(full_catalog)
        for idx, (asin, meta) in enumerate(all_products[cat].items(), start=1):
            reviews = all_reviews.get(asin, [])
            if not reviews:
                continue
            full_catalog.append(transform_product(meta, reviews, cat, idx))
        print(f"  {cat}: {len(full_catalog) - count_before} products")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(full_catalog, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done! Wrote {len(full_catalog)} products to {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
