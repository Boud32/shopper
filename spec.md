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
  ├── generator.py            # The "Mixer": Loads seed, mutates, saves experiment JSON
  ├── agent_runner.py         # The "Runner": Multi-model client, sends JSON to LLMs, parses responses
  ├── create_mock_catalog.py  # Legacy mock catalog generator
  └── analysis.py             # MNL estimation and cross-model comparison

5. Data Schemas
A. The Product Object (Input)

This is the structure of a single item in the product catalog. Fields marked with * are new additions to support richer agent reasoning and MNL feature extraction.

JSON
{
  "id": "prod_001",
  "title": "Sony WH-1000XM5 Wireless Noise Canceling Headphones",
  "description": "Industry-leading noise cancellation...",           // * Full product description
  "base_price": 348.00,       // "base_price" in seed, "price" in experiment
  "rating": 4.8,
  "review_count": 12405,
  "reviews": [                                                       // * Sample review text
    {"rating": 5, "title": "Best headphones ever", "text": "..."},
    {"rating": 3, "title": "Good but heavy", "text": "..."}
  ],
  "attributes": {
    "battery_life": "30 hours",
    "weight": "250g",
    "connection": "Bluetooth 5.2"
  },
  "tags": ["Best Seller", "Over-Ear", "Sponsored", "Overall Pick"], // * Sponsored + Overall Pick tags
  "position": 3                                                      // * Search result position (page rank)
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
Phase 1 (Current): Use Amazon Reviews 2023 dataset (McAuley Lab, Kaggle) as seed data. This provides real titles, prices, descriptions, and review text.

Phase 2: LLM-expand seed data to fill gaps — generate realistic product descriptions and reviews for items missing detail.

Phase 3: Consider live scraping for dynamic catalog experiments (features and catalogs that change over time).

8. Implementation Requirements
Step 1: generator.py (The Mixer)

Create a class ExperimentGenerator that performs the following:

Load: Read data/seed_catalog.json.

Sample: Function create_batch(size=10) that randomly selects size items from the seed.

Mutate:
- Implement a price_multiplier argument. If set to 1.1, increase all prices by 10%.
- Implement a target_mutation argument. Example: mutate_product(id="prod_001", field="rating", value=3.0). This allows us to specifically "nerf" a popular product to see if the Agent abandons it.
- Support adding/removing tags (Sponsored, Overall Pick) per product.
- Support setting position vectors.

Serialize: Save the resulting batch as data/experiments/batch_{timestamp}.json.

Step 2: agent_runner.py (The Multi-Model Client)

Create a class AgentClient that:

Ingest: Reads a batch_{timestamp}.json file.

Prompt: Wraps the JSON in a system prompt. The prompt is a variable under study — start with a baseline and iterate.

Call: Sends to all configured models (Gemini, Claude, ChatGPT, DeepSeek) via a unified provider interface.

Parse: Validates that each response is valid JSON and matches the Output Schema.

Save: Results are tagged with model name, prompt version, and timestamp.

Step 3: analysis.py (MNL Estimation)

Aggregate results across models and experiment conditions. Estimate MNL coefficients per model. Compare feature sensitivities cross-model.
