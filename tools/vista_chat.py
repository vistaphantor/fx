from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.language.model_bundle import load_model_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        default="data/models/trading_language/15m",
    )
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    args = parser.parse_args()

    model, tok, manifest = load_model_bundle(args.bundle, device="cpu")
    print(
        f"Loaded {manifest.architecture} "
        f"{manifest.parameter_count/1e6:.2f}M "
        f"stage={manifest.training_stage}"
    )

    history: list[str] = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input in {"/quit", "/exit"}:
            break
        if user_input == "/reset":
            history.clear()
            continue

        history.append(f"<user>{user_input}</user>")
        history = history[-6:]
        prompt = "\n".join(history) + "\n<assistant>"

        ids = tok.encode(prompt, add_bos=True, add_eos=False)
        if len(ids) > model.max_seq_len:
            ids = ids[-model.max_seq_len:]

        inp = torch.tensor([ids], dtype=torch.long)

        out = model.generate(
            inp,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_id=tok.eos_id(),
        )

        generated = tok.decode(out[0].tolist(), skip_special=False)

        response = generated.split("<assistant>")[-1]
        if "</assistant>" in response:
            response = response.split("</assistant>", 1)[0]
        if "<user>" in response:
            response = response.split("<user>", 1)[0]

        response = response.strip()
        print(f"\nVista: {response}")
        history.append(f"<assistant>{response}</assistant>")


if __name__ == "__main__":
    main()
