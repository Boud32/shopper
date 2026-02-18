import json
import os
import time
import argparse
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def call_gemini(prompt):
    """Calls Google Gemini API."""
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )
    return response.text


def call_openai(prompt):
    """Calls OpenAI (ChatGPT) API."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def call_claude(prompt):
    """Calls Anthropic Claude API."""
    from anthropic import Anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in environment")

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def call_deepseek(prompt):
    """Calls DeepSeek API (OpenAI-compatible)."""
    from openai import OpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set in environment")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


# Registry: name -> (call function, display name, model id for metadata)
PROVIDERS = {
    "gemini":   (call_gemini,   "Gemini",   "gemini-2.5-flash"),
    "openai":   (call_openai,   "ChatGPT",  "gpt-4o-mini"),
    "claude":   (call_claude,   "Claude",   "claude-sonnet-4-20250514"),
    "deepseek": (call_deepseek, "DeepSeek", "deepseek-chat"),
}

# ---------------------------------------------------------------------------
# Agent Client
# ---------------------------------------------------------------------------

class AgentClient:
    def __init__(self, experiment_file, k=5, models=None):
        self.experiment_file = experiment_file
        self.k = k
        self.models = models or ["gemini"]
        self.products = self._load_experiment()

    def _load_experiment(self):
        """Loads the experiment JSON file."""
        try:
            with open(self.experiment_file, 'r') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} products from {self.experiment_file}")
            return data
        except FileNotFoundError:
            print(f"Error: Experiment file not found at {self.experiment_file}")
            return []

    def create_prompt(self):
        """Wraps the product data in a shopping-assistant prompt."""
        # Infer category from the batch for a realistic task framing
        category = None
        if self.products:
            category = self.products[0].get("category")
        category_line = f"A customer is looking to buy {category}." if category else "A customer has asked you to find the best product."

        products_json = json.dumps(self.products, indent=2)
        prompt = f"""You are a shopping assistant. {category_line} Review the following JSON product feed and select a consideration set of {self.k} products that best meet the customer's needs.

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
        return prompt

    def call_llm(self, prompt, provider_name):
        """Calls a specific LLM provider."""
        if provider_name not in PROVIDERS:
            print(f"Error: Unknown provider '{provider_name}'. Available: {list(PROVIDERS.keys())}")
            return None

        call_fn, display_name, _ = PROVIDERS[provider_name]
        print(f"\n--- Sending prompt to {display_name} ---")

        try:
            response = call_fn(prompt)
            print(f"Received response from {display_name}.")
            return response
        except Exception as e:
            print(f"Error calling {display_name}: {e}")
            return None

    def parse_response(self, response_str):
        """Parses and validates the LLM response."""
        if not response_str:
            return None
        try:
            if "```json" in response_str:
                response_str = response_str.split("```json")[1].split("```")[0].strip()
            elif "```" in response_str:
                response_str = response_str.split("```")[1].split("```")[0].strip()

            return json.loads(response_str)
        except json.JSONDecodeError:
            print(f"Error: Failed to parse LLM response. Raw: {response_str[:500]}")
            return None

    def save_result(self, result, provider_name):
        """Saves the result to the results directory, tagged with model info."""
        output_dir = "data/results"
        os.makedirs(output_dir, exist_ok=True)

        _, _, model_id = PROVIDERS[provider_name]
        timestamp = int(time.time())
        filename = f"result_{provider_name}_{timestamp}.json"

        if "metadata" not in result:
            result["metadata"] = {}
        result["metadata"].update({
            "source_batch": self.experiment_file,
            "timestamp": timestamp,
            "provider": provider_name,
            "model": model_id,
            "k": self.k,
        })

        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved result to {output_path}")
        return output_path

    def run(self):
        """Runs the experiment against all configured models."""
        prompt = self.create_prompt()
        results = {}

        for provider_name in self.models:
            response_str = self.call_llm(prompt, provider_name)
            result = self.parse_response(response_str)

            if result:
                print(f"\n--- {PROVIDERS[provider_name][1]} Decision ---")
                print(json.dumps(result, indent=2))
                self.save_result(result, provider_name)
                results[provider_name] = result
            else:
                print(f"\n--- {provider_name}: No valid result ---")

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Shopper Agent across multiple LLMs")
    parser.add_argument("--file", type=str, help="Path to the experiment batch JSON file")
    parser.add_argument("--k", type=int, default=5, help="Size of consideration set (default: 5)")
    parser.add_argument(
        "--models", type=str, nargs="+",
        default=["gemini"],
        choices=list(PROVIDERS.keys()),
        help=f"Which models to run. Choices: {list(PROVIDERS.keys())}",
    )

    args = parser.parse_args()

    # Find the most recent batch file if not provided
    if args.file:
        experiment_file = args.file
    else:
        exp_dir = "data/experiments"
        try:
            files = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if f.startswith("batch_")]
            if files:
                experiment_file = max(files, key=os.path.getctime)
            else:
                print("No experiment batches found. Run generator.py first.")
                exit(1)
        except FileNotFoundError:
            print("Experiment directory not found.")
            exit(1)

    client = AgentClient(experiment_file, k=args.k, models=args.models)
    client.run()
