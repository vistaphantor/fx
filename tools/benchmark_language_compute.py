from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.language.compute_budget import benchmark_training_throughput
from src.language.curriculum import select_curriculum
from src.language.data_pipeline import load_all_training_text
from src.language.tokenizer import BPETokenizer
from src.language.training_pipeline import build_example_sequences, split_by_prompt_family
from tools.train_language_reasoner import (
    PROFILES,
    SEED,
    _model_config,
    _remove_exam_families,
    _stable_unique,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure real CPU training throughput before an expensive Vista language run."
    )
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--data-root", default="data/data/trainingdata")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--wall-clock-hours", type=float, default=4.0)
    parser.add_argument("--threads", type=int, default=max(1, min(4, os.cpu_count() or 2)))
    args = parser.parse_args()

    if args.steps <= 0 or args.wall_clock_hours <= 0:
        raise ValueError("steps and wall-clock-hours must be positive")

    bundle_dir = Path(args.bundle_dir)
    work_dir = bundle_dir / ".training"
    preflight_path = work_dir / "preflight.json"
    tokenizer_path = work_dir / "tokenizer.json"
    if not preflight_path.exists() or not tokenizer_path.exists():
        raise RuntimeError("run_train_language_reasoner_--preflight-only_first")

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    profile = str(preflight["profile"])
    stage = str(preflight["training_stage"])
    if profile not in PROFILES:
        raise RuntimeError(f"unknown_profile_in_preflight:{profile}")
    cfg = dict(PROFILES[profile])
    cfg.update(dropout=0.10, lr_min=1e-5, weight_decay=0.01, grad_clip=1.0, val_split=0.05)

    torch.set_num_threads(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    local = load_all_training_text(
        Path(args.data_root),
        cfg["max_examples"],
        False,
        SEED,
    )
    hf_sample_path = work_dir / "hf_preflight_sample.json"
    hf_sample: list[str] = []
    if hf_sample_path.exists():
        payload = json.loads(hf_sample_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise RuntimeError("hf_preflight_sample_invalid")
        hf_sample = [str(value) for value in payload]

    texts = _stable_unique(local + hf_sample)
    texts, _ = _remove_exam_families(texts, stage)
    selected = select_curriculum(texts, stage=stage, seed=SEED).texts
    train_texts, _ = split_by_prompt_family(
        selected,
        val_fraction=cfg["val_split"],
        seed=SEED,
    )

    tokenizer = BPETokenizer.load(tokenizer_path)
    sequences = build_example_sequences(
        train_texts,
        tokenizer,
        seq_len=cfg["seq_len"],
    )
    report = benchmark_training_throughput(
        model_config=_model_config(cfg, tokenizer),
        tokenizer=tokenizer,
        sequences=sequences,
        batch_size=cfg["batch_size"],
        steps=args.steps,
        wall_clock_hours=args.wall_clock_hours,
    )

    output_path = work_dir / "compute_probe.json"
    output_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("[ComputeProbe] PASS")
    print(f"  params                         : {report.parameter_count:,}")
    print(f"  measured useful tokens/sec     : {report.useful_tokens_per_second:,.1f}")
    print(f"  projected useful tokens/{report.wall_clock_hours:g}h : {report.projected_useful_tokens:,}")
    print(f"  projected tokens/parameter     : {report.projected_tokens_per_parameter:.2f}")
    print(f"  reference target tokens        : {report.reference_target_tokens:,}")
    print(f"  projected hours to reference   : {report.projected_hours_to_reference_target:.2f}")
    print(f"  required tokens/sec in window  : {report.required_tokens_per_second_for_reference_in_window:,.1f}")
    print(f"  report                         : {output_path}")


if __name__ == "__main__":
    main()
