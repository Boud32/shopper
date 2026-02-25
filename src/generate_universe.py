"""
Generates a universe of static offer sets for large-scale experiments.

Each offer set is saved as its own numbered batch JSON file with positions
and tags assigned once. The same universe is run through multiple models to
enable valid cross-model comparison.

Usage:
    python src/generate_universe.py --category "Running Shoes" --n 500
    python src/generate_universe.py --category "Running Shoes" --n 600  # top up safely

Output files: data/experiments/running_shoes/running_shoes_001.json, ...002.json, etc.

Re-running is safe — existing files are skipped, so you can increase --n to
add more offer sets without regenerating completed ones.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, ".")
from src.generator import ExperimentGenerator


def generate_universe(
    category,
    n=500,
    batch_size=25,
    position_mode="random",
    experiments_dir="data/experiments",
    seed_path="data/seed_catalog.json",
):
    gen = ExperimentGenerator(seed_path=seed_path)

    if not gen.seed_data:
        print("Error: No seed data loaded.")
        return

    available = sorted(set(p.get("category") for p in gen.seed_data if p.get("category")))
    if category not in available:
        print(f"Error: Category '{category}' not found.")
        print(f"Available: {available}")
        return

    slug = category.lower().replace(" ", "_")
    output_dir = os.path.join(experiments_dir, slug)
    os.makedirs(output_dir, exist_ok=True)

    created = 0
    skipped = 0

    for i in range(1, n + 1):
        filename = f"{slug}_{i:03d}.json"
        output_path = os.path.join(output_dir, filename)

        if os.path.exists(output_path):
            skipped += 1
            continue

        batch = gen.create_batch(size=batch_size, category=category)
        batch = gen.mutate(batch)
        batch = gen.assign_positions(batch, mode=position_mode)
        batch = gen.inject_tags(batch)

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
        choices=["random", "price_asc", "price_desc", "rating_desc"],
    )
    parser.add_argument("--experiments-dir", default="data/experiments")
    parser.add_argument("--seed", default="data/seed_catalog.json")

    args = parser.parse_args()
    generate_universe(
        category=args.category,
        n=args.n,
        batch_size=args.batch_size,
        position_mode=args.position_mode,
        experiments_dir=args.experiments_dir,
        seed_path=args.seed,
    )
