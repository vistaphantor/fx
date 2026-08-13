"""
Direct Hugging Face Data Downloader — No pyarrow or datasets package required!

Downloads GSM8K, Kimi K2.5, GLM 5.2, and reasoning datasets via direct HF API/Parquet/JSON
and formats them into data/data/trainingdata/.

Usage:
    python tools/fetch_hf_direct.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import ssl
from pathlib import Path

# Ensure UTF-8 output on Windows console
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path("data/data/trainingdata")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HF_URLS = [
    ("gsm8k_main", "https://datasets-server.huggingface.co/rows?dataset=openai/gsm8k&config=main&split=train&offset=0&limit=1000"),
    ("gsm8k_socratic", "https://datasets-server.huggingface.co/rows?dataset=openai/gsm8k&config=socratic&split=train&offset=0&limit=1000"),
    ("kimi_math", "https://datasets-server.huggingface.co/rows?dataset=Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned&config=General-Math&split=train&offset=0&limit=1000"),
    ("glm_conv", "https://datasets-server.huggingface.co/rows?dataset=ianncity/GLM-5.2-Conversation&config=default&split=train&offset=0&limit=1000"),
]

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def download_hf_dataset(name: str, url: str):
    out_file = OUT_DIR / f"huggingface_{name}.json"
    print(f"Downloading '{name}' directly from Hugging Face API...")
    req = urllib.request.Request(url, headers={"User-Agent": "Antigravity/1.0"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            rows = data.get("rows", [])
            examples = []
            for r in rows:
                row_data = r.get("row", {})
                if "question" in row_data and "answer" in row_data:
                    q = row_data["question"]
                    a = row_data["answer"]
                    examples.append({"prompt": f"Human: {q}\n\nAssistant: <think>\n{a}\n</think>", "response": a})
                elif "instruction" in row_data or "prompt" in row_data:
                    p = row_data.get("instruction") or row_data.get("prompt")
                    resp_val = row_data.get("output") or row_data.get("response") or ""
                    examples.append({"prompt": f"Human: {p}\n\nAssistant: {resp_val}", "response": str(resp_val)})
                else:
                    examples.append({"prompt": json.dumps(row_data), "response": ""})

            out_data = {"metadata": {"dataset": name, "total_examples": len(examples)}, "examples": examples}
            out_file.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [OK] Saved {len(examples):,} formatted examples to {out_file}")

    except Exception as e:
        print(f"  [Error] Failed downloading {name}: {e}")


def main():
    print("=" * 80)
    print(" DIRECT HUGGING FACE DATASET DOWNLOADER (No pyarrow required)")
    print("=" * 80)

    for name, url in HF_URLS:
        download_hf_dataset(name, url)

    print("\n[Done] Datasets downloaded and ready in data/data/trainingdata/")


if __name__ == "__main__":
    main()
