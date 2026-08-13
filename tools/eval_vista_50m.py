"""
Vista 50M Reasoning Model — Evaluation & Benchmark Suite ("Exam")

Runs a standardized benchmark across 4 domains:
  1. Math Reasoning & Word Problems
  2. Logical Reasoning & CoT
  3. Python Code Generation & Logic
  4. General Knowledge & Instruction Following

Usage:
  python tools/eval_vista_50m.py
  python tools/eval_vista_50m.py --model data/models/language_50m/vista_50m_best.pt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.tokenizer import BPETokenizer

# ── Standard Exam Questions ──────────────────────────────────────────────────
EXAM_SUITE = [
    # ── Category 1: Math ─────────────────────────────────────────────────────
    {
        "id": "MATH_01",
        "category": "Math",
        "prompt": "Human: What is 15% of 200?\n\nAssistant: <think>",
        "expected_keywords": ["30"],
    },
    {
        "id": "MATH_02",
        "category": "Math",
        "prompt": "Human: If a train travels 60 miles per hour for 2.5 hours, how many miles does it cover?\n\nAssistant: <think>",
        "expected_keywords": ["150"],
    },
    {
        "id": "MATH_03",
        "category": "Math",
        "prompt": "Human: Solve for x: 3x + 9 = 24.\n\nAssistant: <think>",
        "expected_keywords": ["5"],
    },
    {
        "id": "MATH_04",
        "category": "Math",
        "prompt": "Human: A store gives a 20% discount on a $50 jacket. What is the final price?\n\nAssistant: <think>",
        "expected_keywords": ["40"],
    },

    # ── Category 2: Logic & CoT Reasoning ────────────────────────────────────
    {
        "id": "LOGIC_01",
        "category": "Logic",
        "prompt": "Human: All cats are mammals. All mammals breathe air. Does a cat breathe air?\n\nAssistant: <think>",
        "expected_keywords": ["yes", "breathe", "mammal"],
    },
    {
        "id": "LOGIC_02",
        "category": "Logic",
        "prompt": "Human: If Alice is older than Bob, and Bob is older than Charlie, who is the youngest?\n\nAssistant: <think>",
        "expected_keywords": ["charlie"],
    },

    # ── Category 3: Python Coding ─────────────────────────────────────────────
    {
        "id": "CODE_01",
        "category": "Coding",
        "prompt": "Human: Write a Python function `is_even(n)` that returns True if n is even.\n\nAssistant: <think>",
        "expected_keywords": ["def", "% 2 == 0", "return"],
    },
    {
        "id": "CODE_02",
        "category": "Coding",
        "prompt": "Human: Write a Python line to calculate the sum of numbers from 1 to 10.\n\nAssistant: <think>",
        "expected_keywords": ["sum", "range"],
    },

    # ── Category 4: Instruction & General QA ──────────────────────────────────
    {
        "id": "QA_01",
        "category": "General QA",
        "prompt": "Human: What is the capital of France?\n\nAssistant: <think>",
        "expected_keywords": ["paris"],
    },
    {
        "id": "QA_02",
        "category": "General QA",
        "prompt": "Human: What is the primary purpose of an AI language model?\n\nAssistant: <think>",
        "expected_keywords": ["text", "understand", "generate", "language", "process"],
    },
]


def load_model_and_tokenizer(model_path_str: str | None = None):
    # Paths
    default_model = "data/models/language_50m/vista_50m_best.pt"
    model_path = Path(model_path_str) if model_path_str else Path(default_model)

    if not model_path.exists():
        alt = Path("data/models/language_50m/vista_50m.pt")
        if alt.exists():
            model_path = alt
        else:
            print(f"❌ Error: Model checkpoint not found at {model_path}")
            print("   Ensure you downloaded vista_50m_best.pt from Kaggle to data/models/language_50m/")
            return None, None

    tok_path = Path("data/models/language_50m/tokenizer.json")
    if not tok_path.exists():
        tok_path = Path("data/models/language/tokenizer.json")

    if not tok_path.exists():
        print(f"❌ Error: Tokenizer file not found at {tok_path}")
        return None, None

    print(f"📦 Loading Tokenizer: {tok_path}")
    tok = BPETokenizer.load(tok_path)

    print(f"🧠 Loading Model Checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    cfg = checkpoint.get("config", {
        "vocab_size": tok.vocab_size,
        "d_model": 512, "n_layers": 16, "n_heads": 16,
        "ffn_dim": 2048, "max_seq_len": 512, "dropout": 0.0
    })

    model = VistaReasoningGPT(
        vocab_size  = tok.vocab_size,
        d_model     = cfg.get("d_model", 512),
        n_layers    = cfg.get("n_layers", 16),
        n_heads     = cfg.get("n_heads", 16),
        ffn_dim     = cfg.get("ffn_dim", 2048),
        max_seq_len = cfg.get("max_seq_len", 512),
        dropout     = 0.0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"✅ Model loaded successfully! ({model.get_num_params()/1e6:.1f}M Parameters)")
    return model, tok


def run_exam(model: VistaReasoningGPT, tok: BPETokenizer):
    print("\n" + "=" * 70)
    print(" 🎓 VISTA-50M REASONING MODEL — AUTOMATED EXAM BENCHMARK ")
    print("=" * 70 + "\n")

    results = []
    category_scores = {}

    start_time = time.time()

    for idx, test in enumerate(EXAM_SUITE, 1):
        q_id = test["id"]
        category = test["category"]
        prompt = test["prompt"]
        expected = test["expected_keywords"]

        prompt_ids = tok.encode(prompt, add_bos=True, add_eos=False)
        inp_tensor = torch.tensor([prompt_ids], dtype=torch.long)

        # Generate response
        with torch.no_grad():
            out_tensor = model.generate(
                inp_tensor,
                max_new_tokens=250,
                temperature=0.6,
                top_k=40,
                top_p=0.9,
                eos_id=tok.eos_id(),
            )

        full_output = tok.decode(out_tensor[0].tolist())
        
        # Clean response string
        gen_text = full_output[len(prompt):].strip()
        has_think_tag = "<think>" in full_output or "</think>" in gen_text or "<think>" in prompt

        # Evaluation criteria check
        gen_lower = gen_text.lower()
        matched_keywords = [kw for kw in expected if kw.lower() in gen_lower]
        passed = len(matched_keywords) > 0

        if category not in category_scores:
            category_scores[category] = {"passed": 0, "total": 0}
        category_scores[category]["total"] += 1
        if passed:
            category_scores[category]["passed"] += 1

        status_symbol = "✅ PASS" if passed else "❌ FAIL"

        print(f"[{idx}/{len(EXAM_SUITE)}] {q_id} ({category}) ── {status_symbol}")
        print(f"   Prompt:   {prompt.splitlines()[0]}")
        print(f"   Response: {gen_text[:120]}..." if len(gen_text) > 120 else f"   Response: {gen_text}")
        print(f"   Matched:  {matched_keywords} / Expected: {expected}\n")

        results.append({
            "id": q_id,
            "category": category,
            "passed": passed,
            "matched_keywords": matched_keywords,
            "expected_keywords": expected,
            "response": gen_text,
            "has_think_tag": has_think_tag
        })

    total_time = time.time() - start_time
    total_passed = sum(1 for r in results if r["passed"])
    overall_accuracy = (total_passed / len(EXAM_SUITE)) * 100

    # ── Report Summary ────────────────────────────────────────────────────────
    print("=" * 70)
    print(" 📊 EXAM RESULTS SUMMARY")
    print("=" * 70)
    print(f" Overall Score: {total_passed}/{len(EXAM_SUITE)} ({overall_accuracy:.1f}%)")
    print(f" Total Evaluation Time: {total_time:.2f} seconds\n")

    print(" Domain Breakdown:")
    for cat, sc in category_scores.items():
        acc = (sc['passed'] / sc['total']) * 100
        print(f"   • {cat:<12}: {sc['passed']}/{sc['total']} ({acc:.1f}%)")

    # Save Exam Report JSON
    out_report = Path("data/models/language_50m/exam_report.json")
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_accuracy": overall_accuracy,
            "total_passed": total_passed,
            "total_questions": len(EXAM_SUITE),
            "category_scores": category_scores,
            "detailed_results": results
        }, f, indent=2)

    print(f"\n💾 Saved full exam report to: {out_report}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Path to vista_50m_best.pt")
    args = parser.parse_args()

    model, tok = load_model_and_tokenizer(args.model)
    if model is None:
        return
    run_exam(model, tok)


if __name__ == "__main__":
    main()
