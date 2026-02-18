import copy
import json
import random
import time
import os


POSITION_MODES = ["random", "price_asc", "price_desc", "rating_desc"]


class ExperimentGenerator:
    def __init__(self, seed_path="data/seed_catalog.json", output_dir="data/experiments"):
        self.seed_path = seed_path
        self.output_dir = output_dir
        self.seed_data = []
        self._load_seed()

    def _load_seed(self):
        """Loads the seed catalog from disk."""
        try:
            with open(self.seed_path, 'r') as f:
                self.seed_data = json.load(f)
            print(f"Loaded {len(self.seed_data)} products from {self.seed_path}")
        except FileNotFoundError:
            print(f"Error: Seed file not found at {self.seed_path}")
            self.seed_data = []

    def create_batch(self, size=25, category=None):
        """Randomly samples 'size' products from the seed, optionally filtered by category."""
        if not self.seed_data:
            print("No seed data available.")
            return []

        pool = self.seed_data
        if category:
            pool = [p for p in self.seed_data if p.get("category") == category]
            if not pool:
                print(f"No products found for category: {category}")
                return []
            print(f"Filtered pool size for '{category}': {len(pool)}")

        if size <= len(pool):
            return random.sample(pool, size)
        else:
            return random.choices(pool, k=size)

    def mutate(self, batch, price_multiplier=1.0, target_mutations=None):
        """
        Applies mutations to a batch.

        - Renames base_price → price (seed uses base_price; experiments use price).
        - Applies a global price_multiplier to all products.
        - Applies targeted field overrides via target_mutations.

        Args:
            batch:             List of product dicts from create_batch().
            price_multiplier:  e.g. 1.1 raises all prices by 10%.
            target_mutations:  List of {"id": ..., "field": ..., "value": ...} overrides.
                               Example: [{"id": "prod_001", "field": "rating", "value": 3.0}]
        """
        mutated_batch = []
        for item in batch:
            new_item = copy.deepcopy(item)
            if "base_price" in new_item:
                price = new_item.pop("base_price")
                new_item["price"] = round(price * price_multiplier, 2)
            mutated_batch.append(new_item)

        if target_mutations:
            item_map = {item["id"]: item for item in mutated_batch}
            for mutation in target_mutations:
                target_id = mutation.get("id")
                field = mutation.get("field")
                value = mutation.get("value")
                if target_id in item_map:
                    item_map[target_id][field] = value

        return mutated_batch

    def assign_positions(self, batch, mode="random", page_size=10):
        """
        Assigns position and page fields to each product in a batch.

        Position reflects where a product appears in a search results page — a key
        experimental variable for studying order/presentation bias in agent decisions.
        This is injected here rather than sourced from Amazon data, which has no
        meaningful position signal for our controlled experiments.

        Args:
            batch:      List of product dicts (output of mutate()).
            mode:       Ordering before position assignment. One of:
                          "random"      — shuffle (default; baseline for detecting position bias)
                          "price_asc"   — cheapest first
                          "price_desc"  — most expensive first
                          "rating_desc" — highest rated first
            page_size:  Number of results per page (default 10, mimics a standard SERP).

        Returns:
            New list with integer "position" (1-indexed, global rank) and
            "page" (1-indexed page number) added to each product dict.
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

    def inject_tags(
        self,
        batch,
        n_sponsored=(3, 5),
        n_best_seller=(0, 1),
        n_overall_pick=(0, 1),
    ):
        """
        Randomly assigns commercial tags to products in a batch.

        Tags are injected here — the Amazon source data carries no sponsored/placement
        signals relevant to our controlled experiments.

        Calibration (observed on Amazon toothbrushes, 64 products, page 1):
            Sponsored:    11/64 ≈ 17%  → (3, 5) per 25-product batch
            Best Seller:   4/64 ≈  6%  → typically 0 or 1 per query
            Overall Pick:  0/64         → typically 0 or 1 per query

        In practice you see at most 1 Best Seller and 1 Overall Pick per search,
        and only occasionally both at the same time. Tags are drawn independently
        so co-existence is possible but rare at (0, 1) ranges.
        TODO: discuss with professor whether Overall Pick and Best Seller
        should be mutually exclusive.

        Args:
            batch:           List of product dicts (output of assign_positions()).
            n_sponsored:     (min, max) number of Sponsored tags to assign.
            n_best_seller:   (min, max) number of Best Seller tags to assign.
            n_overall_pick:  (min, max) number of Overall Pick tags to assign.

        Returns:
            New list with "tags" fields populated.
        """
        import copy
        batch = [copy.deepcopy(p) for p in batch]

        # Clear any pre-existing commercial tags so inject is idempotent
        commercial = {"Sponsored", "Best Seller", "Overall Pick"}
        for p in batch:
            p["tags"] = [t for t in (p.get("tags") or []) if t not in commercial]

        def _assign(tag, n_range):
            n = random.randint(*n_range)
            n = min(n, len(batch))
            for p in random.sample(batch, n):
                if tag not in p["tags"]:
                    p["tags"].append(tag)

        _assign("Sponsored", n_sponsored)
        _assign("Best Seller", n_best_seller)
        _assign("Overall Pick", n_overall_pick)

        return batch

    def validate_batch(self, batch, n_sponsored=(2, 8), n_best_seller=(0, 2), n_overall_pick=(0, 2)):
        """
        Checks that tag counts in a batch are within expected ranges.
        Prints a warning for each violation. Does not raise — caller decides whether to proceed.

        Returns:
            True if all checks pass, False if any violation found.
        """
        counts = {"Sponsored": 0, "Best Seller": 0, "Overall Pick": 0}
        for p in batch:
            for tag in (p.get("tags") or []):
                if tag in counts:
                    counts[tag] += 1

        limits = {
            "Sponsored":    n_sponsored,
            "Best Seller":  n_best_seller,
            "Overall Pick": n_overall_pick,
        }
        valid = True
        for tag, (lo, hi) in limits.items():
            n = counts[tag]
            if not (lo <= n <= hi):
                print(f"  WARNING: '{tag}' count is {n}, expected [{lo}, {hi}]")
                valid = False

        if valid:
            print(f"  Batch valid — tags: {counts}")
        return valid

    def serialize(self, batch, filename=None):
        """Saves the batch to JSON and returns the output path."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"batch_{timestamp}.json"

        output_path = os.path.join(self.output_dir, filename)
        os.makedirs(self.output_dir, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(batch, f, indent=2)

        print(f"Saved batch to {output_path}")
        return output_path


if __name__ == "__main__":
    generator = ExperimentGenerator()

    print("\n--- Generating Batches for All Categories ---")
    if generator.seed_data:
        categories = sorted(set(p.get("category") for p in generator.seed_data if p.get("category")))
        for cat in categories:
            slug = cat.lower().replace(" ", "_")
            print(f"Generating for category: {cat}")
            batch = generator.create_batch(size=25, category=cat)
            if batch:
                batch = generator.mutate(batch)
                batch = generator.assign_positions(batch, mode="random")
                batch = generator.inject_tags(batch)
                if generator.validate_batch(batch):
                    generator.serialize(batch, f"batch_{slug}.json")
