from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.language.source_mix_audit import audit_supervised_source_mix, format_supervised_source_mix
from src.language.streaming_sources import load_hf_source_config
from src.language.tokenizer import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit actual supervised-token mix by authoritative training source.")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--hf-config", default="config/hf_sources.json")
    parser.add_argument("--training-stage", default="foundation")
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer_path = Path(args.bundle) / ".training" / "tokenizer.json"
    if not tokenizer_path.exists():
        raise SystemExit(f"Tokenizer not found: {tokenizer_path}")
    tokenizer = BPETokenizer.load(tokenizer_path)
    specs = load_hf_source_config(args.hf_config)
    stats = audit_supervised_source_mix(
        specs=specs,
        stage=args.training_stage,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        seed=args.seed,
        max_examples=args.examples,
    )
    print(format_supervised_source_mix(stats))
    for row in stats:
        print(
            f"  {row.source_id:<24} examples={row.examples:>5,} "
            f"supervised_tokens={row.supervised_tokens:>9,} share={row.percent:>6.2f}%"
        )


if __name__ == "__main__":
    main()
