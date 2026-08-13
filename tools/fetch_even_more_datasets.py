"""
Dataset Expansion Script Part 2 — Fetching additional Math, Science, Code, and Logic datasets
from Hugging Face directly into data/data/trainingdata/.

Datasets:
  1. camel-ai/math (Camel AI Mathematical Reasoning)
  2. math_qa (Step-by-step MathQA problem solving)
  3. mbpp (Mostly Basic Python Problems)
  4. fka/awesome-chatgpt-prompts (Prompt engineering & persona instructions)
  5. HuggingFaceH4/deita-10k-v0.8 (High quality instruction tuning dataset)
  6. m-a-p/CodeFeedback-Single-Turn-Instruction (Programming & algorithm reasoning)

Usage:
    python tools/fetch_even_more_datasets.py
"""
from __future__ import annotations

import json
import ssl
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path("data/data/trainingdata")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

MORE_TARGETS = [
    ("camel_math", "https://datasets-server.huggingface.co/rows?dataset=camel-ai/math&config=default&split=train&offset=0&limit=1500"),
    ("math_qa", "https://datasets-server.huggingface.co/rows?dataset=math_qa&config=default&split=train&offset=0&limit=1500"),
    ("mbpp_code", "https://datasets-server.huggingface.co/rows?dataset=google-research-datasets/mbpp&config=full&split=train&offset=0&limit=1000"),
    ("chatgpt_prompts", "https://datasets-server.huggingface.co/rows?dataset=fka/awesome-chatgpt-prompts&config=default&split=train&offset=0&limit=1000"),
    ("deita_10k", "https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/deita-10k-v0.8&config=default&split=train&offset=0&limit=1000"),
]

SEP = "═" * 80


def fetch_and_save(name: str, url: str):
    out_file = OUT_DIR / f"huggingface_{name}.json"
    print(f"Fetching '{name}' directly from Hugging Face API...")
    req = urllib.request.Request(url, headers={"User-Agent": "Antigravity/1.0"})

    try:
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            rows = data.get("rows", [])
            examples = []

            for r in rows:
                row = r.get("row", {})
                # Math QA format
                if "Problem" in row and "Rationale" in row:
                    examples.append({
                        "prompt": f"Human: {row['Problem']}\n\nAssistant: <think>\n{row['Rationale']}\n</think>\nAnswer: {row.get('correct', '')}",
                        "response": str(row.get("Rationale", ""))
                    })
                # Camel Math format
                elif "message_1" in row and "message_2" in row:
                    examples.append({
                        "prompt": f"Human: {row['message_1']}\n\nAssistant: <think>\n{row['message_2']}\n</think>",
                        "response": str(row['message_2'])
                    })
                # MBPP Code format
                elif "text" in row and "code" in row:
                    examples.append({
                        "prompt": f"Human: Write a python function for: {row['text']}\n\nAssistant: ```python\n{row['code']}\n```",
                        "response": str(row['code'])
                    })
                # ChatGPT Prompts format
                elif "act" in row and "prompt" in row:
                    examples.append({
                        "prompt": f"Human: Act as a {row['act']}\n\nAssistant: Understood. I will act as a {row['act']}. {row['prompt']}",
                        "response": str(row['prompt'])
                    })
                # Generic fallback
                elif "instruction" in row or "prompt" in row:
                    p = row.get("instruction") or row.get("prompt")
                    r_val = row.get("output") or row.get("response") or ""
                    examples.append({"prompt": f"Human: {p}\n\nAssistant: {r_val}", "response": str(r_val)})
                else:
                    examples.append({"prompt": json.dumps(row, ensure_ascii=False), "response": ""})

            if examples:
                out_data = {"metadata": {"dataset": name, "total_examples": len(examples)}, "examples": examples}
                out_file.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"  [OK] Saved {len(examples):,} formatted examples -> {out_file.name}")
            else:
                print(f"  [Notice] No valid rows parsed for {name}")

    except Exception as e:
        print(f"  [Notice] Could not fetch {name}: {e}")


def main():
    print(SEP)
    print(" HUGGING FACE DATASET EXPANSION (PART 2)")
    print(SEP)

    for name, url in MORE_TARGETS:
        fetch_and_save(name, url)

    files = list(OUT_DIR.glob("*.json"))
    print(f"\n{SEP}")
    print(f" 🎉 TOTAL EXPANDED DATASETS: {len(files)} JSON files in {OUT_DIR}")
    print(SEP)


if __name__ == "__main__":
    main()
