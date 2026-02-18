Project Spec: Agentic Shopping Data Generator
1. Project Overview
We are building a "Headless Shopping" experiment engine to study the sensitivity of AI agents to product features when making purchase decisions. Instead of using Visual Language Models (VLM) on screenshots, we feed agents raw JSON product data (simulating a web scraper) and measure how their purchasing decisions change based on manipulated attributes (Price, Rating, Sponsored status, Position, etc.).

The end goal is to estimate a Multinomial Logit (MNL) model treating each AI agent as a "consumer" with utility coefficients for product features (price, reviews, sponsored placement, etc.), and to compare these coefficients across models.

Core Architecture: "Seed and Mutate"

Seed: A realistic catalog of product data sourced from real ecommerce datasets (Amazon Reviews 2023, Kaggle).

Mutate: A procedural Python engine that samples products and mathematically alters attributes (Price, Rating, Position, Tags) to create unique experimental conditions. Key concept: "Same product, different universe" — the same product appearing in different competitive assortments.

Transport: The data is serialized to JSON to be sent to the LLM.

2. Models Under Study
Each experiment batch is run against multiple LLMs to compare decision-making behavior:

- Gemini (Google) — gemini-flash
- Claude (Anthropic) — claude-sonnet
- ChatGPT (OpenAI) — gpt-4o-mini
- DeepSeek — deepseek-chat

The same prompt and catalog are sent to each model. Results are saved with model metadata for cross-model comparison.

3. Environment Setup

Conda environment: shopper (Python 3.11)

```bash
conda activate shopper
pip install google-genai python-dotenv openai anthropic
```

API keys are stored in `.env` at the project root:
```
GEMINI_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
DEEPSEEK_API_KEY=...
```

4. File Structure
The project should be organized as follows:

/data
  ├── seed_catalog.json       # Realistic product catalog (sourced from Kaggle + LLM expansion)
  ├── experiments/            # Output folder for generated experiment batches
  └── results/                # Output folder for Agent decisions (tagged by model)
/src
  ├── ingest_kaggle.py        # Data ingestion: streams Amazon Reviews 2023 from HF into seed catalog
  ├── generator.py            # The "Mixer": Loads seed, mutates, saves experiment JSON
  ├── agent_runner.py         # The "Runner": Multi-model client, sends JSON to LLMs, parses responses
  ├── create_mock_catalog.py  # Legacy mock catalog generator
  └── analysis.py             # MNL estimation and cross-model comparison

5. Data Schemas
A. The Product Object (Input)

This is the structure of a single item in the product catalog. Fields marked with * are new additions to support richer agent reasoning and MNL feature extraction.

JSON
{
  "id": "prod_headphones_001",
  "parent_asin": "B0C8PSQF7C",                                      // * Amazon ASIN for traceability
  "category": "Wireless Headphones",
  "title": "Sony WH-1000XM5 Wireless Noise Canceling Headphones",
  "description": "Industry-leading noise cancellation...",           // * Full product description
  "base_price": 348.00,       // "base_price" in seed, "price" in experiment
  "rating": 4.8,
  "review_count": 12405,
  "features": ["30-hour battery life", "Bluetooth 5.2", ...],       // * Real bullet-point features from listings
  "reviews": [                                                       // * Sample review text (up to 5 per product)
    {"rating": 5, "title": "Best headphones ever", "text": "...", "verified": true, "helpful_votes": 12},
    {"rating": 3, "title": "Good but heavy", "text": "...", "verified": true, "helpful_votes": 3}
  ],
  "store": "Sony",                                                   // * Store/brand from metadata
  "tags": [],                                                        // Populated by mutation engine (Sponsored, Overall Pick, Best Seller)
  "position": 3,                                                     // Injected by assign_positions() in generator — NOT from Amazon data
  "page": 1                                                          // Page number derived from position and page_size (default 10 per page)
}

B. The Agent Response (Output)

The LLM must be instructed to return strictly this JSON format. Includes a "no_purchase" option for the outside good in MNL estimation.

JSON
{
  "experiment_id": "exp_2026_01_26_A",
  "decision": {
    "consideration_set": ["prod_002", "prod_005", "prod_009"],
    "final_choice": "prod_002",          // Can be "no_purchase" if none are suitable
    "reasoning_trace": "Selected prod_002 because..."
  }
}

6. Research Questions
1. What are the sensitivities of different AI agents to product features (price, rating, reviews, sponsored status, position)?
2. How do consideration sets differ across models for the same catalog?
3. Does the competitive assortment change agent behavior? ("Same product, different universe")
4. How does prompt framing affect choices? (top-k selection vs. utility threshold)
5. Can we estimate stable MNL coefficients for each model?

7. Data Strategy
Phase 1 (Current): Use Amazon Reviews 2023 dataset (McAuley Lab, Hugging Face) as seed data. This provides real titles, prices, descriptions, features, and review text. Ingested via `src/ingest_kaggle.py` which streams data from Hugging Face without downloading the full 22GB dataset.

Target categories: Wireless Headphones, Smartwatches, Mechanical Keyboards, Gaming Mice, Toothbrushes, Running Shoes. ~500 products per category with 5 reviews each.

```bash
# Full ingestion (all categories, ~500 products each, 5 reviews per product)
python src/ingest_kaggle.py

# Specific categories or smaller batches
python src/ingest_kaggle.py --categories "Wireless Headphones" "Gaming Mice"
python src/ingest_kaggle.py --products-per-category 100 --reviews-per-product 3
```

Phase 2: LLM-expand seed data to fill gaps — generate realistic product descriptions and reviews for items missing detail.

Phase 3: Consider live scraping for dynamic catalog experiments (features and catalogs that change over time).

8. Implementation Requirements
Step 1: generator.py (The Mixer)

ExperimentGenerator class with the following pipeline:

create_batch(size=25, category=None): Randomly samples products from the seed catalog, optionally filtered by category.

mutate(batch, price_multiplier=1.0, target_mutations=None):
- Renames base_price → price (seed uses base_price; experiments use price).
- Applies a global price_multiplier (e.g. 1.1 = +10%).
- Applies targeted field overrides, e.g. [{"id": "prod_001", "field": "rating", "value": 3.0}] to "nerf" a specific product.
- Tags (Sponsored, Overall Pick, Best Seller) are set via target_mutations on the "tags" field.

assign_positions(batch, mode="random", page_size=10):
- Injects "position" (1-indexed global rank) and "page" (1-indexed page number) into each product.
- Position is a key experimental variable: it is NOT sourced from Amazon data (which has no meaningful signal for controlled experiments) — it is assigned here.
- Modes:
  - "random"      — shuffle before assigning (default; baseline for detecting position bias)
  - "price_asc"   — cheapest product at position 1
  - "price_desc"  — most expensive product at position 1
  - "rating_desc" — highest-rated product at position 1
- page_size defaults to 10 (mimics a standard Amazon SERP page).

serialize(batch, filename=None): Saves the batch as data/experiments/batch_{timestamp}.json.

Full pipeline example:
```python
batch = generator.create_batch(size=25, category="Gaming Mice")
batch = generator.mutate(batch, price_multiplier=1.1)
batch = generator.assign_positions(batch, mode="random")
generator.serialize(batch)
```

Step 2: agent_runner.py (The Multi-Model Client)

AgentClient class:

- Ingest: Reads a batch_{timestamp}.json file.
- Prompt: Wraps the JSON in a system prompt that includes a category hint ("A customer is looking to buy X"). The prompt is a variable under study — start with a baseline and iterate.
- Call: Sends to all configured models (Gemini, Claude, ChatGPT, DeepSeek) via a unified provider interface.
- Parse: Validates that each response is valid JSON and matches the Output Schema.
- Save: Results are tagged with model name, prompt version, and timestamp. metadata.source_batch records the batch file path for joining at analysis time.

Step 3: analysis.py (MNL Estimation)

load_results_to_dataframe(results_dir, experiments_dir): Joins all result files with their source batch files and returns a long-format pandas DataFrame for MNL estimation.

DataFrame schema (one row per product per experiment):

| Column           | Type   | Description                                      |
|------------------|--------|--------------------------------------------------|
| experiment_id    | str    | Result filename stem                             |
| provider         | str    | Model provider key (e.g. "gemini")               |
| model            | str    | Full model ID                                    |
| timestamp        | int    | Unix timestamp of the run                        |
| product_id       | str    | Product ID                                       |
| category         | str    | Product category                                 |
| price            | float  | Price at experiment time (post-mutation)         |
| rating           | float  | Product rating                                   |
| review_count     | int    | Number of reviews                                |
| position         | int    | Search result position (1-indexed)               |
| page             | int    | Page number (1-indexed)                          |
| is_sponsored     | bool   | "Sponsored" tag present                          |
| is_best_seller   | bool   | "Best Seller" tag present                        |
| is_overall_pick  | bool   | "Overall Pick" tag present                       |
| in_consideration | bool   | Product was in the model's consideration set     |
| chosen           | bool   | Product was the model's final choice             |

MNL estimation and cross-model coefficient comparison to be implemented in a future phase.
