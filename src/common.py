import os
import re
from openai import OpenAI

# Three small, cheap, non-reasoning chat models from different labs.
# Check the exact IDs on openrouter.ai/models before running and edit if needed.
MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-nemo",
    "qwen/qwen-2.5-7b-instruct",
]
# Larger model for the robustness checks (king replay and big-council run). Check the ID on openrouter.ai/models.
BIG_MODEL = "meta-llama/llama-3.3-70b-instruct"

# Load KEY=value lines from the repo's .env file into the environment.
_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")  # repo root
if os.path.exists(_env):
    for line in open(_env):
        k, _, v = line.partition("=")
        if k.strip() and v.strip():
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPEN_ROUTER_API"],
                max_retries=5)


def ask(model, system, user, temperature=0.7, max_tokens=150):
    r = client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return r.choices[0].message.content or ""


def letter(text, tag=None):
    """Extract A-D. With a tag (e.g. 'VOTE'), only accept the letter right after 'TAG:'."""
    if tag:
        m = re.search(tag + r"[\s:*\-]*\(?([ABCD])\b", text, re.I)
        return m.group(1).upper() if m else None
    m = re.search(r"\b([ABCD])\b", text.strip())
    return m.group(1) if m else None


def format_q(q):
    return q["question"] + "\n" + "\n".join(f"{L}. {c}" for L, c in zip("ABCD", q["choices"]))
