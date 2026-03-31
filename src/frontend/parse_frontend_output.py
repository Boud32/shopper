"""
Parse a model's JSON output from an Architecture 2 (frontend batch) session
into the same result file format used by Architecture 1 (run_universe.py).

This lets analysis_helper.py and mnl.py work on both architectures identically.

The model output file should be a JSON array like:
[
  {
    "offer_set_id": "mechanical_keyboards_001",
    "consideration_set": ["prod_x", "prod_y", "prod_z", "prod_a", "prod_b"],
    "final_choice": "prod_x",
    "reasoning": "..."
  },
  ...
]

Each entry becomes one result_<provider>_<offer_set_id>.json file,
matching the format written by run_universe.py.

Usage:
    python src/parse_frontend_output.py \\
        --input data/frontend/output_mechanical_keyboards_gemini.json \\
        --category "Mechanical Keyboards" \\
        --provider gemini-frontend \\
        --model gemini-2.5-flash \\
        --offer-sets-dir data/offer_sets \\
        --output data/results/frontend/
"""

import argparse
import json
import re
import time
from pathlib import Path


def extract_json_array(text):
    """
    Extract a JSON array from model output that may contain surrounding text.

    Models sometimes wrap the JSON in markdown code blocks or add commentary.
    This strips the wrapper and returns the parsed array.
    """
    # Try parsing directly first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the outermost [...] block
    start = text.find("[")
    end   = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not extract a valid JSON array from the model output.")


def load_offer_set(offer_set_id, category, offer_sets_dir):
    """Load the offer set JSON file matching the given offer_set_id."""
    cat_slug = category.lower().replace(" ", "_")
    path = Path(offer_sets_dir) / cat_slug / f"{offer_set_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Offer set not found: {path}")
    with open(path) as f:
        return json.load(f)


def parse_and_write(input_path, category, provider, model, offer_sets_dir, output_dir):
    """
    Read model JSON output, validate each entry, write one result file per offer set.

    Returns (n_written, n_skipped).
    """
    with open(input_path) as f:
        raw = f.read()

    entries = extract_json_array(raw)
    print(f"Parsed {len(entries)} entries from {input_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_written  = 0
    n_skipped  = 0
    timestamp  = int(time.time())

    for i, entry in enumerate(entries):
        offer_set_id     = entry.get("offer_set_id", "").strip()
        consideration    = entry.get("consideration_set") or []
        final_choice     = (entry.get("final_choice") or "no_purchase").strip()
        reasoning        = entry.get("reasoning", "")
        session_position = i + 1   # 1-indexed position within the session

        if not offer_set_id:
            print(f"  [{i+1}] Skipping entry with missing offer_set_id")
            n_skipped += 1
            continue

        # Validate against actual offer set
        try:
            products = load_offer_set(offer_set_id, category, offer_sets_dir)
        except FileNotFoundError as e:
            print(f"  [{i+1}] Skipping {offer_set_id}: {e}")
            n_skipped += 1
            continue

        product_ids = {p["id"] for p in products}

        # Validate consideration set
        valid_consideration = [pid for pid in consideration if pid in product_ids]
        invalid_consideration = [pid for pid in consideration if pid not in product_ids]
        if invalid_consideration:
            print(f"  [{i+1}] {offer_set_id}: {len(invalid_consideration)} invalid product IDs in consideration set (dropped)")

        # Validate final choice
        if final_choice != "no_purchase" and final_choice not in product_ids:
            print(f"  [{i+1}] {offer_set_id}: final_choice '{final_choice}' not in offer set — treating as no_purchase")
            final_choice = "no_purchase"

        result = {
            "metadata": {
                "provider":         provider,
                "model":            model,
                "architecture":     "frontend",
                "session_position": session_position,   # for degradation analysis
                "timestamp":        timestamp,
                "source_batch":     str(Path(offer_sets_dir) / category.lower().replace(" ", "_") / f"{offer_set_id}.json"),
                "variant":          "full",
            },
            "decision": {
                "consideration_set": valid_consideration,
                "final_choice":      final_choice,
                "reasoning":         reasoning,
            },
        }

        out_file = output_path / f"result_{provider}_{offer_set_id}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        n_written += 1

    print(f"\nDone: {n_written} written, {n_skipped} skipped → {output_path}")
    return n_written, n_skipped


def main():
    parser = argparse.ArgumentParser(description="Parse frontend model output into result files")
    parser.add_argument("--input",          required=True, help="Path to model JSON output file")
    parser.add_argument("--category",       required=True, help="Category name (e.g. 'Mechanical Keyboards')")
    parser.add_argument("--provider",       default="gemini-frontend",
                        help="Provider label for result filenames (default: gemini-frontend)")
    parser.add_argument("--model",          default="gemini-2.5-flash",
                        help="Model ID (default: gemini-2.5-flash)")
    parser.add_argument("--offer-sets-dir", default="data/offer_sets")
    parser.add_argument("--output",         default="data/results/frontend",
                        help="Output directory for result files (default: data/results/frontend)")
    args = parser.parse_args()

    parse_and_write(
        input_path     = args.input,
        category       = args.category,
        provider       = args.provider,
        model          = args.model,
        offer_sets_dir = args.offer_sets_dir,
        output_dir     = args.output,
    )


if __name__ == "__main__":
    main()
