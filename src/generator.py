import json
import random
import time
import os
from pathlib import Path

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

    def create_batch(self, size=5, category=None):
        """Randomly selects 'size' items from the seed catalog, optionally filtered by category."""
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
            # If we want all items from the category regardless of size if size > pool
            # The user mentioned "total catalog of products of a given product type stays constant"
            # So if size > pool, maybe just return the pool? 
            # But adhering to 'size' with replacement is safer to match previous logic, 
            # OR we change semantics to "get max 'size' items".
            # Let's stick to sampling logic for now, but if size > pool, return pool + samples?
            # Or just replacement.
            return random.choices(pool, k=size)

    def mutate(self, batch, price_multiplier=1.0, target_mutations=None):
        """
        Applies mutations to the batch.
        
        Args:
            batch: List of product dicts.
            price_multiplier: Float to multiply all prices by.
            target_mutations: List of dicts, e.g. [{"id": "prod_001", "field": "rating", "value": 3.0}]
        """
        mutated_batch = []
        # Create a deep copy to avoid modifying the seed cache if we were reusing objects directly
        # (though json.load creates fresh dicts, good practice)
        import copy
        
        for item in batch:
            # Deep copy the item so we don't mutate shared references if any
            new_item = copy.deepcopy(item)
            
            # Apply global mutations
            if "base_price" in new_item:
                # Rename base_price to price for the experiment feed as per spec?
                # Spec says: "base_price" in seed, "price" in experiment.
                price = new_item.pop("base_price")
                new_item["price"] = round(price * price_multiplier, 2)
            
            mutated_batch.append(new_item)

        # Apply target specific mutations
        if target_mutations:
            item_map = {item['id']: item for item in mutated_batch}
            for mutation in target_mutations:
                target_id = mutation.get("id")
                field = mutation.get("field")
                value = mutation.get("value")
                
                if target_id in item_map:
                    item_map[target_id][field] = value
        
        return mutated_batch

    def serialize(self, batch, filename=None):
        """Saves the batch to JSON."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"batch_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # Ensure output dir exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(batch, f, indent=2)
        
        print(f"Saved batch to {output_path}")
        return output_path

if __name__ == "__main__":
    # Demonstration when running script directly
    generator = ExperimentGenerator()
    

    
    # 3. Create a batch for EACH category available in the seed
    print("\n--- Generating Batches for All Categories ---")
    if generator.seed_data:
        categories = set(p.get("category") for p in generator.seed_data if p.get("category"))
        for cat in categories:
            # Create a filename safe string
            cat_filename = cat.lower().replace(" ", "_")
            print(f"Generating for category: {cat}")
            
            # Create batch of 20 (or max available)
            batch = generator.create_batch(size=20, category=cat)
            if batch:
                generator.serialize(batch, f"batch_{cat_filename}.json")
