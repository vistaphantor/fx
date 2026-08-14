from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.language.model_bundle import load_model_bundle
from src.language.protocol import (
    build_chat_prompt,
    extract_assistant_response,
    generation_stop_ids,
)


EXAM_SUITE = [
    ("MATH_01", "Math", "What is 15% of 200?", ["30"]),
    ("MATH_02", "Math", "If a train travels 60 miles per hour for 2.5 hours, how many miles does it cover?", ["150"]),
    ("MATH_03", "Math", "Solve for x: 3x + 9 = 24.", ["5"]),
    ("MATH_04", "Math", "A store gives a 20% discount on a $50 jacket. What is the final price?", ["40"]),
    ("LOGIC_01", "Logic", "All cats are mammals. All mammals breathe air. Does a cat breathe air?", ["yes", "breathe", "mammal"]),
    ("LOGIC_02", "Logic", "If Alice is older than Bob, and Bob is older than Charlie, who is the youngest?", ["charlie"]),
    ("CODE_01", "Coding", "Write a Python function is_even(n) that returns True if n is even.", ["def", "% 2", "return"]),
    ("CODE_02", "Coding", "Write a Python line to calculate the sum of numbers from 1 to 10.", ["sum", "range"]),
    ("QA_01", "General QA", "What is the capital of France?", ["paris"]),
    ("QA_02", "General QA", "What is the primary purpose of an AI language model?", ["language", "text", "generate"]),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="data/models/trading_language/15m")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    model, tokenizer, manifest = load_model_bundle(args.bundle, device="cpu")
    stops = generation_stop_ids(tokenizer)
    print(
        f"Evaluating {manifest.architecture} {manifest.parameter_count / 1e6:.2f}M "
        f"stage={manifest.training_stage} context={model.max_seq_len}"
    )

    passed = 0
    for index, (case_id, category, question, keywords) in enumerate(EXAM_SUITE, 1):
        prompt = build_chat_prompt([("user", question)])
        ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        if len(ids) > model.max_seq_len:
            raise RuntimeError(f"exam_prompt_exceeds_trained_context:{case_id}")

        output = model.generate(
            torch.tensor([ids], dtype=torch.long),
            max_new_tokens=args.max_new_tokens,
            temperature=0.35,
            top_k=20,
            top_p=0.85,
            stop_ids=stops,
        )
        decoded = tokenizer.decode(output[0].tolist(), skip_special=False)
        response = extract_assistant_response(decoded)
        lower = response.casefold()
        matched = [keyword for keyword in keywords if keyword.casefold() in lower]
        ok = bool(matched)
        passed += int(ok)
        print(
            f"[{index:02d}/{len(EXAM_SUITE)}] {'PASS' if ok else 'FAIL'} "
            f"{case_id} {category}: {response[:180]!r}"
        )

    score = passed / len(EXAM_SUITE)
    print(f"score={passed}/{len(EXAM_SUITE)} ({score:.1%})")


if __name__ == "__main__":
    main()
