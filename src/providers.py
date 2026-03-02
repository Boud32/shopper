"""
LLM provider registry for the Shopper experiment engine.

Each provider is a simple function that takes a prompt string and returns
a response string. The PROVIDERS dict maps short provider keys to a tuple of:
    (call_fn, display_name, model_id)

model_id is stored in result metadata so the DataFrame can filter by model.

Free-tier limits (as of Mar 2026):
    gemini       — gemini-2.5-flash,  ~25 req/day
    gemini-flash — gemini-2.0-flash,  requires billing (limit: 0 on free tier)
    groq         — llama-3.3-70b,     ~14,400 req/day but 100k tokens/day
                   (~8 effective requests at our ~8k token prompt size)
"""

import os
from dotenv import load_dotenv

load_dotenv()


def call_gemini(prompt, model="gemini-2.5-flash"):
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return response.text


def call_openai(prompt):
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


def call_groq(prompt, model="llama-3.3-70b-versatile"):
    from openai import OpenAI

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in environment")

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


PROVIDERS = {
    "gemini":       (lambda p: call_gemini(p, "gemini-2.5-flash"), "Gemini 2.5 Flash",      "gemini-2.5-flash"),
    "gemini-flash": (lambda p: call_gemini(p, "gemini-2.0-flash"), "Gemini 2.0 Flash",      "gemini-2.0-flash"),
    "groq":         (call_groq,     "Groq Llama 3.3 70B", "llama-3.3-70b-versatile"),
    "openai":       (call_openai,   "ChatGPT",             "gpt-4o-mini"),
    "claude":       (call_claude,   "Claude",              "claude-sonnet-4-20250514"),
    "deepseek":     (call_deepseek, "DeepSeek",            "deepseek-chat"),
}
