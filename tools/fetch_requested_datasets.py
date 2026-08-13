"""
Dataset Ingestion Script — Downloads requested reasoning, math, and conversational datasets
from Hugging Face and saves them formatted in data/data/trainingdata/.

Datasets:
  1. Manusagents/GPT-5.5-Gemini-3.1-Pro-Grok-4-Claude-Fable-5-Mythos-5-Qwen-3.7-Max-and-more-Distillation-Dataset
  2. Qyrou/reasoning-corpus-4K-5M-v1
  3. ianncity/GLM-5.2-Conversation
  4. Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned (General-Distillation)
  5. Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned (General-Math)
  6. Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned (MultilingualSTEM)
  7. openai/gsm8k (main)
  8. openai/gsm8k (socratic)

Usage:
    python tools/fetch_requested_datasets.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure datasets package is importable
try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not installed yet. Run: pip install datasets")
    sys.exit(1)

OUTPUT_DIR = Path("data/data/trainingdata")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    ("Manusagents/GPT-5.5-Gemini-3.1-Pro-Grok-4-Claude-Fable-5-Mythos-5-Qwen-3.7-Max-and-more-Distillation-Dataset", None, "manusagents_distillation"),
    ("Qyrou/reasoning-corpus-4K-5M-v1", None, "qyrou_reasoning_4k_5m"),
    ("ianncity/GLM-5.2-Conversation", None, "ianncity_glm_conversation"),
    ("Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned", "General-Distillation", "kimi_general_distillation"),
    ("Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned", "General-Math", "kimi_general_math"),
    ("Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned", "MultilingualSTEM", "kimi_multilingual_stem"),
    ("openai/gsm8k", "main", "gsm8k_main"),
    ("openai/gsm8k", "socratic", "gsm8k_socratic"),
]

SEP = "═" * 90


def save_dataset_as_json(ds, name: str, max_rows: int = 5000) -> Path:
    """Extract and format rows into standard training format."""
    out_file = OUTPUT_DIR / f"huggingface_{name}.json"
    examples = []

    # Get available split (train, test, or default)
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    split_data = ds[split_name]

    print(f"  Processing '{name}' ({len(split_data):,} rows available, taking up to {max_rows:,})...")

    for i, row in enumerate(split_data):
        if i >= max_rows:
            break

        # Case 1: GSM8K (question + answer)
        if "question" in row and "answer" in row:
            examples.append({
                "prompt": f"Human: {row['question']}\n\nAssistant: <think>\n{row['answer']}\n</think>",
                "response": row["answer"],
                "metadata": {"source": name, "row_id": i}
            })
        # Case 2: Conversations / Messages list
        elif "messages" in row or "conversations" in row:
            msgs = row.get("messages") or row.get("conversations")
            turns = []
            for m in msgs:
                role = m.get("role") or m.get("from") or "user"
                val  = m.get("content") or m.get("value") or ""
                if role in ("user", "human"):
                    turns.append(f"Human: {val}")
                elif role in ("assistant", "gpt", "bot"):
                    turns.append(f"Assistant: {val}")
            if turns:
                examples.append({
                    "prompt": "\n\n".join(turns),
                    "response": "",
                    "metadata": {"source": name, "row_id": i}
                })
        # Case 3: Prompt / Response or Instruction / Output
        elif "instruction" in row or "prompt" in row or "text" in row:
            p = row.get("instruction") or row.get("prompt") or row.get("text") or ""
            r = row.get("output") or row.get("response") or row.get("completion") or ""
            if p:
                full_text = f"Human: {p}\n\nAssistant: {r}" if r else str(p)
                examples.append({
                    "prompt": full_text,
                    "response": str(r),
                    "metadata": {"source": name, "row_id": i}
                })
        # Generic fallback: stringify row
        else:
            examples.append({
                "prompt": json.dumps(row, ensure_ascii=False),
                "response": "",
                "metadata": {"source": name, "row_id": i}
            })

    output_data = {
        "metadata": {
            "dataset": name,
            "total_examples": len(examples),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "examples": examples,
    }

    out_file.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [Saved] {out_file}  ({len(examples):,} formatted examples)")
    return out_file


def main():
    print(SEP)
    print(" HUGGING FACE REASONING & MATH DATASET DOWNLOADER")
    print(SEP)

    for repo_id, config_name, save_name in TARGETS:
        print(f"\nFetching: {repo_id} (config={config_name})...")
        try:
            if config_name:
                ds = load_dataset(repo_id, config_name)
            else:
                ds = load_dataset(repo_id)

            save_dataset_as_json(ds, save_name)
        except Exception as e:
            print(f"  [Warning/Error] Failed to fetch {repo_id}: {e}")

    print(f"\n{SEP}")
    print(" [Complete] All available datasets saved to data/data/trainingdata/")
    print(SEP)


if __name__ == "__main__":
    main()
