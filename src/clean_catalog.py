"""
Clean data/seed_catalog.json by applying per-category title filters and price floors.

The ingest pipeline uses keyword matching against both product titles AND Amazon
category metadata, which is loose — accessories and adjacent products slip in because
Amazon assigns them to broad category trees. This script re-filters using title-only
matching to keep only genuine category products.

Strategy per category:
  - Positive filter: product title must contain at least one core keyword
  - Price floor:    remove very cheap items (accessories, single-use trial packs)
  - Negative filter: explicit exclusions for known junk patterns

The original catalog is backed up as data/seed_catalog_raw.json before overwriting.

Usage:
    python src/clean_catalog.py           # dry-run: shows counts, no write
    python src/clean_catalog.py --apply   # write cleaned catalog to data/seed_catalog.json
"""

import argparse
import json
import re
import shutil
from pathlib import Path


# Per-category configuration.
# 'require':  product title must contain at least one of these strings (case-insensitive)
# 'exclude':  product title must NOT contain any of these strings (applied after require)
# 'price_min': products below this price are dropped
CATEGORY_FILTERS = {
    "Toilet Paper": {
        "require": [
            "toilet paper", "bath tissue", "bathroom tissue",
            "toilet tissue", "loo roll", "toilet roll",
        ],
        "exclude": [],
        "price_min": 4.0,
    },
    "Bluetooth Speakers": {
        "require": [
            "bluetooth speaker", "wireless speaker", "portable speaker",
            "bt speaker", "waterproof speaker",
        ],
        "exclude": [
            "case for", "case compatible", "charging cord", "charging cable",
            "power cord", "micro charging", "replacement cord", "replacement cable",
        ],
        "price_min": 12.0,
    },
    "Yoga Mats": {
        "require": [
            "yoga mat", "exercise mat", "fitness mat", "pilates mat",
            "yoga and exercise mat", "non-slip mat", "workout mat",
        ],
        "exclude": [
            "balance beam", "foam tile", "foam flooring", "interlocking",
            "martial arts", "puzzle mat", "floor tile",
        ],
        "price_min": 12.0,
    },
    "Backpacks": {
        "require": [
            # Note: checked with word-boundary logic in is_clean() — "backpack" will
            # not match "backpacking" or "backpack beach chair".
            "backpack", "backpacks", "daypack", "rucksack", "school bag", "bookbag",
            "book bag", "hiking pack", "laptop bag",
        ],
        "exclude": [
            # Non-backpack products that mention "backpack" incidentally
            "backpack rain cover",   # rain covers for backpacks (not backpacks with rain cover)
            "molle accessories", "attachment kit", "attachments for",
            "webbing strap", "nylon webbing",
            "water bottle",          # standalone water bottles
            "sleeping pad", "sleeping mat", "sleeping bag",
            "bike helmet", "bicycle helmet", "cycling helmet",
            "paddle board", "paddleboard", "sup board", "stand up paddle",
            "ski boot", "snowboard boot",
            "telescoping stool", "folding stool",
            "drill bit", "auger bit",
            "beach chair", "camp chair", "camping chair",
            "backpacking tent", "backpacking stove",
            # Accessories / cosmetic patches
            "patch", "embroidery", "keychain", "key chain",
            "tent stake", "tent peg", "hand warmer", "lantern",
            " shirt", "dog tank",
        ],
        "price_min": 12.0,
    },
    "Coffee Makers": {
        "require": [
            "coffee maker", "coffeemaker", "coffee machine", "coffee brewer",
            "drip coffee", "single serve coffee", "pod coffee", "k-cup",
            "coffee pot", "percolator", "french press", "pour over",
            "moka pot", "espresso maker", "aeropress", "cold brew coffee",
        ],
        "exclude": [
            "cord organizer", "cord wrap", "cord holder", "cord manager",
            "appliance slider", "appliance mover",
            "silicone gasket", " gasket", "rubber gasket",
            "replacement part", "cleaning tablet", "descaling",
        ],
        "price_min": 10.0,
    },
    "Protein Powder": {
        "require": [
            "protein powder", "whey protein", "protein shake",
            "protein supplement", "protein blend", "plant protein",
            "vegan protein", "casein protein", "mass gainer",
        ],
        "exclude": [
            "detox program", "weight loss kit", "ready-to-drink shot",
            "cleanse program",
        ],
        "price_min": 12.0,
    },
}


def _price(product):
    return product.get("price") or product.get("base_price") or 0.0


def _kw_match(title_lower, keyword):
    """
    Match keyword against title. For single-word keywords ending in common
    suffixes that could be substrings of longer words (e.g., "backpack" inside
    "backpacking"), use a word-boundary check. Multi-word keywords use plain
    substring match since they're already specific enough.
    """
    if " " in keyword:
        return keyword in title_lower
    return bool(re.search(r"\b" + re.escape(keyword) + r"\b", title_lower))


def is_clean(product, category):
    """Return True if the product passes all filters for its category."""
    cfg = CATEGORY_FILTERS.get(category)
    if cfg is None:
        return True  # unknown category: pass through

    title_lower = (product.get("title") or "").lower()
    price = _price(product)

    # Price floor
    if price < cfg["price_min"]:
        return False

    # Must match at least one required keyword in title
    if not any(_kw_match(title_lower, kw) for kw in cfg["require"]):
        return False

    # Must not match any exclusion keyword in title
    if any(_kw_match(title_lower, kw) for kw in cfg["exclude"]):
        return False

    return True


def clean(catalog):
    """Return (clean_catalog, removed_catalog) lists."""
    clean_products = []
    removed = []
    for product in catalog:
        cat = product.get("category", "")
        if is_clean(product, cat):
            clean_products.append(product)
        else:
            removed.append(product)
    return clean_products, removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog",  default="data/seed_catalog.json")
    parser.add_argument("--apply",    action="store_true",
                        help="Write cleaned catalog (default: dry run only)")
    parser.add_argument("--show-removed", action="store_true",
                        help="Print removed product titles")
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    with open(catalog_path) as f:
        catalog = json.load(f)

    clean_products, removed = clean(catalog)

    # Summary
    by_cat_before = {}
    by_cat_after  = {}
    by_cat_removed = {}
    for p in catalog:
        by_cat_before.setdefault(p.get("category", "unknown"), []).append(p)
    for p in clean_products:
        by_cat_after.setdefault(p.get("category", "unknown"), []).append(p)
    for p in removed:
        by_cat_removed.setdefault(p.get("category", "unknown"), []).append(p)

    print(f"{'Category':<22}  {'Before':>6}  {'Removed':>7}  {'After':>5}")
    print("-" * 46)
    for cat in sorted(by_cat_before):
        n_before  = len(by_cat_before.get(cat, []))
        n_removed = len(by_cat_removed.get(cat, []))
        n_after   = len(by_cat_after.get(cat, []))
        print(f"{cat:<22}  {n_before:>6}  {n_removed:>7}  {n_after:>5}")
    print("-" * 46)
    print(f"{'TOTAL':<22}  {len(catalog):>6}  {len(removed):>7}  {len(clean_products):>5}")

    if args.show_removed:
        print()
        for cat in sorted(by_cat_removed):
            print(f"\n--- Removed from {cat} ---")
            for p in sorted(by_cat_removed[cat], key=_price):
                reason = []
                cfg = CATEGORY_FILTERS.get(cat, {})
                title_lower = (p.get("title") or "").lower()
                if _price(p) < cfg.get("price_min", 0):
                    reason.append(f"price ${_price(p):.2f} < floor")
                if not any(kw in title_lower for kw in cfg.get("require", [])):
                    reason.append("no required keyword")
                if any(kw in title_lower for kw in cfg.get("exclude", [])):
                    reason.append("exclusion keyword")
                print(f"  ${_price(p):>6.2f}  [{', '.join(reason)}]  {(p.get('title') or '')[:75]}")

    if args.apply:
        backup_path = catalog_path.with_name("seed_catalog_raw.json")
        shutil.copy(catalog_path, backup_path)
        print(f"\nBacked up original → {backup_path}")
        with open(catalog_path, "w") as f:
            json.dump(clean_products, f, indent=2)
        print(f"Wrote {len(clean_products)} products → {catalog_path}")
    else:
        print("\nDry run — pass --apply to write changes.")


if __name__ == "__main__":
    main()
