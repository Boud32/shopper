"""
Generates a universe of static offer sets for large-scale experiments.

Each offer set is saved as its own numbered batch JSON file with positions
and tags assigned once. The same universe is run through multiple models to
enable valid cross-model comparison.

Usage:
    python src/generate_universe.py --category "Running Shoes" --n 500
    python src/generate_universe.py --category "Running Shoes" --n 600  # top up safely

Output files: data/offer_sets/running_shoes/running_shoes_001.json, ...002.json, etc.

Re-running is safe — existing files are skipped, so you can increase --n to
add more offer sets without regenerating completed ones. Each file is seeded
with (random_seed + i) before generation, so output is fully deterministic.
"""

import argparse
import copy
import json
import os
import random
import sys


POSITION_MODES = ["random", "price_asc", "price_desc", "rating_desc"]


def _load_seed(seed_path):
    try:
        with open(seed_path) as f:
            data = json.load(f)
        print(f"Loaded {len(data)} products from {seed_path}")
        return data
    except FileNotFoundError:
        print(f"Error: Seed file not found at {seed_path}")
        return []


def create_batch(seed_data, size=25, category=None):
    """Randomly samples `size` products from seed_data, optionally filtered by category."""
    pool = seed_data
    if category:
        pool = [p for p in seed_data if p.get("category") == category]
        if not pool:
            print(f"No products found for category: {category}")
            return []

    if size <= len(pool):
        return random.sample(pool, size)
    else:
        return random.choices(pool, k=size)


def mutate(batch, price_multiplier=1.0, target_mutations=None):
    """
    Renames base_price → price and applies a global price multiplier.
    Optional target_mutations: list of {"id": ..., "field": ..., "value": ...} overrides.
    """
    mutated = []
    for item in batch:
        new_item = copy.deepcopy(item)
        if "base_price" in new_item:
            price = new_item.pop("base_price")
            new_item["price"] = round(price * price_multiplier, 2)
        mutated.append(new_item)

    if target_mutations:
        item_map = {item["id"]: item for item in mutated}
        for m in target_mutations:
            if m.get("id") in item_map:
                item_map[m["id"]][m["field"]] = m["value"]

    return mutated


def assign_positions(batch, mode="random", page_size=10):
    """
    Assigns position and page fields to each product.

    mode: "random" | "price_asc" | "price_desc" | "rating_desc"
    page_size: results per page (default 10, mimics a standard SERP).
    """
    if mode not in POSITION_MODES:
        raise ValueError(f"mode must be one of {POSITION_MODES}, got '{mode}'")

    ordered = [copy.deepcopy(p) for p in batch]

    if mode == "random":
        random.shuffle(ordered)
    elif mode == "price_asc":
        ordered.sort(key=lambda p: p.get("price") or p.get("base_price") or 0)
    elif mode == "price_desc":
        ordered.sort(key=lambda p: p.get("price") or p.get("base_price") or 0, reverse=True)
    elif mode == "rating_desc":
        ordered.sort(key=lambda p: p.get("rating") or 0, reverse=True)

    for i, product in enumerate(ordered):
        product["position"] = i + 1
        product["page"] = (i // page_size) + 1

    return ordered


def inject_tags(batch, n_sponsored=(3, 5), n_best_seller=(0, 1), n_overall_pick=(0, 1)):
    """
    Randomly assigns commercial tags (Sponsored, Best Seller, Overall Pick).

    Calibrated to observed Amazon search pages (~17% sponsored, ~6% Best Seller).
    Tags are injected here — not sourced from Amazon data — so they are fully
    controlled experimental variables.
    """
    batch = [copy.deepcopy(p) for p in batch]
    commercial = {"Sponsored", "Best Seller", "Overall Pick"}
    for p in batch:
        p["tags"] = [t for t in (p.get("tags") or []) if t not in commercial]

    def _assign(tag, n_range):
        n = min(random.randint(*n_range), len(batch))
        for p in random.sample(batch, n):
            if tag not in p["tags"]:
                p["tags"].append(tag)

    _assign("Sponsored", n_sponsored)
    _assign("Best Seller", n_best_seller)
    _assign("Overall Pick", n_overall_pick)

    return batch


def generate_universe(
    category,
    n=500,
    batch_size=25,
    position_mode="random",
    offer_sets_dir="data/offer_sets",
    seed_path="data/seed_catalog.json",
    random_seed=42,
):
    seed_data = _load_seed(seed_path)
    if not seed_data:
        return

    available = sorted(set(p.get("category") for p in seed_data if p.get("category")))
    if category not in available:
        print(f"Error: Category '{category}' not found.")
        print(f"Available: {available}")
        return

    slug = category.lower().replace(" ", "_")
    output_dir = os.path.join(offer_sets_dir, slug)
    os.makedirs(output_dir, exist_ok=True)

    created = 0
    skipped = 0

    for i in range(1, n + 1):
        filename = f"{slug}_{i:03d}.json"
        output_path = os.path.join(output_dir, filename)

        if os.path.exists(output_path):
            skipped += 1
            continue

        random.seed(random_seed + i)
        batch = create_batch(seed_data, size=batch_size, category=category)
        batch = mutate(batch)
        batch = assign_positions(batch, mode=position_mode)
        batch = inject_tags(batch)

        with open(output_path, "w") as f:
            json.dump(batch, f, indent=2)

        created += 1
        if created % 50 == 0:
            print(f"  Created {created} new offer sets...")

    print(f"\nDone. Created: {created}, Skipped (already existed): {skipped}")
    print(f"Offer sets in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a universe of static offer sets")
    parser.add_argument("--category", required=True, help="Product category (e.g. 'Running Shoes')")
    parser.add_argument("--n", type=int, default=500, help="Number of offer sets (default: 500)")
    parser.add_argument("--batch-size", type=int, default=25, help="Products per offer set (default: 25)")
    parser.add_argument(
        "--position-mode", default="random",
        choices=POSITION_MODES,
    )
    parser.add_argument("--offer-sets-dir", default="data/offer_sets")
    parser.add_argument("--seed", default="data/seed_catalog.json")
    parser.add_argument("--random-seed", type=int, default=42, help="Base random seed (default: 42)")

    args = parser.parse_args()
    generate_universe(
        category=args.category,
        n=args.n,
        batch_size=args.batch_size,
        position_mode=args.position_mode,
        offer_sets_dir=args.offer_sets_dir,
        seed_path=args.seed,
        random_seed=args.random_seed,
    )
