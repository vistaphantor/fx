"""
Automated Hugging Face Search & Fetcher for High-Quality Math/Reasoning/Code Datasets.

Searches HF Hub for reasoning, math, and STEM datasets, then fetches high-value samples
directly into data/data/trainingdata/ without requiring pyarrow.

Usage:
    python tools/search_and_fetch_more.py
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

# High-quality open reasoning / math / code datasets on Hugging Face
DATASETS_TO_FETCH = [
    ("meta_math_qa", "https://datasets-server.huggingface.co/rows?dataset=meta-math/MetaMathQA&config=default&split=train&offset=0&limit=1500"),
    ("open_thoughts_r1", "https://datasets-server.huggingface.co/rows?dataset=open-thoughts/OpenThoughts-114k&config=default&split=train&offset=0&limit=1000"),
    ("math_dataset", "https://datasets-server.huggingface.co/rows?dataset=lighteval/MATH&config=all&split=train&offset=0&limit=1000"),
    ("code_feedback", "https://datasets-server.huggingface.co/rows?dataset=m-a-p/CodeFeedback-Filtered-Instruction&config=default&split=train&offset=0&limit=1000"),
    ("wizard_math", "https://datasets-server.huggingface.co/rows?dataset=WizardLM/WizardLM_evol_instruct_70k&config=default&split=train&offset=0&limit=1000"),
    ("alpaca_cleaned", "https://datasets-server.huggingface.co/rows?dataset=yahma/alpaca-cleaned&config=default&split=train&offset=0&limit=1500"),
]

SEP = "═" * 80


def fetch_dataset(name: str, url: str):
    out_file = OUT_DIR / f"huggingface_{name}.json"
    print(f"Fetching '{name}' from Hugging Face...")
    req = urllib.request.Request(url, headers={"User-Agent": "Antigravity/1.0"})

    try:
        with urllib.request.urlopen(req, context=ctx, timeout=40) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            rows = data.get("rows", [])
            examples = []

            for r in rows:
                row = r.get("row", {})
                # Extract prompt/response fields across different dataset schemas
                q = row.get("query") or row.get("question") or row.get("instruction") or row.get("prompt") or ""
                a = row.get("response") or row.get("answer") or row.get("output") or row.get("solution") or ""

                if q:
                    if "solution" in row or "thought" in row or "<think>" in str(a):
                        full_text = f"Human: {q}\n\nAssistant: <think>\n{a}\n</think>"
                    else:
                        full_text = f"Human: {q}\n\nAssistant: {a}" if a else str(q)
                    examples.append({"prompt": full_text, "response": str(a)})

            if examples:
                out_data = {"metadata": {"dataset": name, "total_examples": len(examples)}, "examples": examples}
                out_file.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"  [OK] Saved {len(examples):,} examples -> {out_file.name}")
            else:
                print(f"  [Notice] No valid rows parsed for {name}")

    except Exception as e:
        print(f"  [Notice] Could not fetch {name}: {e}")


def main():
    print(SEP)
    print(" HUGGING FACE REASONING & MATH DATASET SEARCH & INGGESTION")
    print(SEP)

    for name, url in DATASETS_TO_FETCH:
        fetch_dataset(name, url)

    # Count total datasets in folder
    json_files = list(OUT_DIR.glob("*.json"))
    print(f"\n{SEP}")
    print(f" 🎉 SUCCESS! Training folder now contains {len(json_files)} datasets in {OUT_DIR}")
    print(SEP)


if __name__ == "__main__":
    main()
