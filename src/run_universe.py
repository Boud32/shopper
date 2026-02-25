"""
Runs all offer sets in a universe directory through a model, with resumability.

Completion is tracked by result file existence: a result named
result_{model}_{offer_set_stem}.json means that offer set is done.
Re-running is safe — completed offer sets are skipped automatically.

Usage:
    python src/run_universe.py --category-dir data/experiments/running_shoes --model gemini-flash
    python src/run_universe.py --category-dir data/experiments/running_shoes --model gemini-flash --k 5

To run multiple models over the same universe, just run again with a different --model.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from src.agent_runner import PROVIDERS
from dotenv import load_dotenv

load_dotenv()


def call_provider(prompt, provider_name, max_retries=3):
    """Calls a provider with retry logic for transient rate limits."""
    if provider_name not in PROVIDERS:
        print(f"Error: Unknown provider '{provider_name}'. Available: {list(PROVIDERS.keys())}")
        return None

    call_fn, display_name, _ = PROVIDERS[provider_name]

    for attempt in range(1, max_retries + 1):
        try:
            return call_fn(prompt)
        except Exception as e:
            err = str(e)

            if "PerDay" in err or "per_day" in err.lower():
                print(f"\nDaily quota exhausted for {display_name}. Stopping.")
                return "QUOTA_EXHAUSTED"

            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                delay = 60
                match = re.search(r"retryDelay.*?(\d+(?:\.\d+)?)s", err)
                if match:
                    delay = float(match.group(1)) + 2
                if attempt < max_retries:
                    print(f"  Rate limited. Waiting {delay:.0f}s (attempt {attempt}/{max_retries})...")
                    time.sleep(delay)
                    continue

            print(f"  Error: {e}")
            return None

    return None


def parse_response(response_str):
    if not response_str:
        return None
    try:
        if "```json" in response_str:
            response_str = response_str.split("```json")[1].split("```")[0].strip()
        elif "```" in response_str:
            response_str = response_str.split("```")[1].split("```")[0].strip()
        return json.loads(response_str)
    except json.JSONDecodeError:
        print(f"  Failed to parse response: {response_str[:200]}")
        return None


def slim_product(p, desc_chars=300, review_chars=200):
    """Truncate verbose text fields to keep prompt tokens manageable."""
    s = dict(p)
    if s.get("description"):
        s["description"] = s["description"][:desc_chars]
    if s.get("reviews"):
        trimmed = []
        for r in s["reviews"]:
            r2 = dict(r)
            if r2.get("text"):
                r2["text"] = r2["text"][:review_chars]
            trimmed.append(r2)
        s["reviews"] = trimmed
    return s


def slim_product(p, desc_chars=300, review_chars=200):
    """
    Trim verbose fields before building the prompt (~8k tokens vs ~15k unmodified).

    features: dropped. They are low-signal bullet points (e.g. "Rubber sole",
    "Imported") that are almost always restated in the description. Dropping them
    saves ~4,000 tokens per 25-product batch with no meaningful loss of information.

    description: truncated to desc_chars. Median is ~160 chars so most products are
    unaffected; outliers (up to 1,061 chars) are cut at a natural reading boundary.

    reviews: text truncated to review_chars each. Five reviews at up to 454 chars
    each is the primary token driver; the sentiment is clear well before that limit.
    """
    s = dict(p)
    s.pop("features", None)
    if s.get("description"):
        s["description"] = s["description"][:desc_chars]
    if s.get("reviews"):
        s["reviews"] = [
            dict(r, text=r["text"][:review_chars]) if r.get("text") else dict(r)
            for r in s["reviews"]
        ]
    return s


def build_prompt(products, category, k):
    category_line = f"A customer is looking to buy {category}."
    slimmed = [slim_product(p) for p in products]
    products_json = json.dumps(slimmed, indent=2)
    return f"""You are a shopping assistant. {category_line} Review the following JSON product feed and select a consideration set of {k} products that best meet the customer's needs.

You may also choose "no_purchase" as your final choice if none of the products are suitable.

Product Feed:
{products_json}

You MUST respond with ONLY valid JSON in this exact format, no other text:
{{
  "experiment_id": "exp_placeholder",
  "decision": {{
    "consideration_set": ["prod_id_1", "prod_id_2"],
    "final_choice": "prod_id_1",
    "reasoning_trace": "Your reasoning here..."
  }}
}}"""


def run_universe(category_dir, model, k=5, results_dir="data/results"):
    category_dir = Path(category_dir)
    if not category_dir.exists():
        print(f"Error: Directory not found: {category_dir}")
        return

    offer_set_files = sorted(category_dir.glob("*.json"))
    if not offer_set_files:
        print(f"No offer set files found in {category_dir}")
        return

    _, display_name, model_id = PROVIDERS[model]

    # Infer category from the first file
    with open(offer_set_files[0]) as f:
        first_batch = json.load(f)
    category = first_batch[0].get("category", "Unknown") if first_batch else "Unknown"

    n_total = len(offer_set_files)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    print(f"Category:   {category}")
    print(f"Model:      {display_name} ({model_id})")
    print(f"Offer sets: {n_total}")
    print(f"Results in: {results_dir}\n")

    completed = 0
    run = 0
    failed = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 5

    for offer_set_file in offer_set_files:
        stem = offer_set_file.stem  # e.g. running_shoes_001
        result_filename = f"result_{model}_{stem}.json"
        result_path = results_path / result_filename

        if result_path.exists():
            completed += 1
            continue

        with open(offer_set_file) as f:
            products = json.load(f)

        done_so_far = completed + run
        print(f"[{done_so_far + 1}/{n_total}] {stem}...", end=" ", flush=True)

        prompt = build_prompt(products, category, k)
        response_str = call_provider(prompt, model)

        if response_str == "QUOTA_EXHAUSTED":
            print(f"\nStopped after {run} new inferences ({completed} already done).")
            return

        result = parse_response(response_str)

        if result:
            result["metadata"] = {
                "source_batch": str(offer_set_file),
                "offer_set_id": stem,
                "timestamp": int(time.time()),
                "provider": model,
                "model": model_id,
                "k": k,
            }
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            print("done")
            run += 1
            consecutive_failures = 0
        else:
            print("failed")
            failed += 1
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\nStopping: {consecutive_failures} consecutive failures — check model name or API key.")
                return

    total_done = completed + run
    print(f"\nFinished. {run} new, {completed} already done, {failed} failed. Total: {total_done}/{n_total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a universe of offer sets through a model")
    parser.add_argument("--category-dir", required=True, help="Directory of offer set batch files")
    parser.add_argument(
        "--model", default="gemini-flash",
        choices=list(PROVIDERS.keys()),
        help=f"Model to use. Choices: {list(PROVIDERS.keys())}",
    )
    parser.add_argument("--k", type=int, default=5, help="Consideration set size (default: 5)")
    parser.add_argument("--results-dir", default="data/results")

    args = parser.parse_args()
    run_universe(
        category_dir=args.category_dir,
        model=args.model,
        k=args.k,
        results_dir=args.results_dir,
    )
