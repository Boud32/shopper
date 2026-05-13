# Shopper

Headless shopping experiment engine. Studies AI agent sensitivity to product attributes (price, rating, position, sponsored tags) via Multinomial Logit (MNL) estimation. Real Amazon products (734 products across 6 categories, v3 catalog), real reviews, agents choose 5-product consideration sets and a final product from 25-product offer sets.

## Quick start

```bash
uv sync
source .venv/bin/activate
```

The seed catalog (`data/seed_catalog.json`) is checked in — you do not need to re-run ingestion to reproduce results. Re-running `ingest_kaggle.py` + `clean_catalog.py` will produce a *similar* catalog but not a byte-identical one: an interactive LLM-assisted cleaning pass was applied on top of the rule-based filters during catalog curation, and that pass is not scripted. Treat the committed catalog as the canonical source of truth.

Offer sets (`data/offer_sets/`) are gitignored because they are fully reproducible from the seed catalog via `python -m src.generate_universe` (seeded with `random_seed + i`, so output is byte-identical given the committed catalog).

## Pipeline

```
ingest_kaggle.py  →  seed_catalog.json
                       │
                       ▼
generate_universe.py → data/offer_sets/{cat}/{cat}_NNN.json  (500 per cat × 6 cats)
                       │
        ┌──────────────┴──────────────┐
        ▼ Architecture 1               ▼ Architecture 2 (primary)
src/architecture1/                    src/architecture2/
  run_universe.py                       make_frontend_file.py  (→ xlsx + prompt)
  + providers.py                        [paste into chat frontend by hand]
  (stateless API loop)                  parse_frontend_output.py
        │                              │
        │                              │
        ▼ data/architecture1/results/   ▼ data/architecture2/results/
        └──────────────┬───────────────┘
                       ▼
              src/mnl.py  +  comparison.ipynb / analysis.ipynb
```

## Key files

- `src/mnl.py` — MNL estimation (L-BFGS-B). `fit_mnl(df, category=..., outside_good=True)` is the entry point.
- `src/analysis_helper.py` — `load_results_to_dataframe()` turns the result JSONs into a long-format DataFrame.
- `comparison.ipynb` — side-by-side v2/Gemini vs.\ v3/Claude fits.
- `analysis.ipynb` — exploratory: tag influence, position distributions, ablation tables.
- `src/architecture2/build_joined_excel.py` — joins per-category Excel exports into one workbook per provider.
- `notes/progress_report.tex` — the report.

## Architectures

- **Architecture 1 (API):** one stateless API call per offer set. Used for the v1 Groq runs (~5–7 min per 50 offers).
- **Architecture 2 (frontend):** all offer sets uploaded as a spreadsheet to a chat frontend (Claude.ai, Gemini), processed in a single stateful session (~30 s per 500 offers). Free on any model with a chat frontend. **Primary pipeline.**

## v3 catalog

Six everyday categories, 84–160 products each: Protein Powder, Laundry Detergent, Conditioner, Shampoo, Yoga Mats, Bluetooth Speakers. See `src/ingest_kaggle.py::CATEGORY_CONFIGS` for keyword filters.

## First three things for a new contributor

1. Run `uv sync` and open `comparison.ipynb`. The notebook reproduces the v2/Gemini and v3/Claude MNL fits end-to-end and walks through one offer set's $U_j \to V_j \to P(j)$ computation.
2. Read the Challenges section of `notes/progress_report.tex` for the lessons learned (catalog iteration, synthetic-generation discovery on Yoga Mats, no-purchase identification).
3. The first useful experiment is the Gemini-batch=50 rerun on Yoga Mats — see Next Steps in the report.
