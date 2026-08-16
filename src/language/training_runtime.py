from __future__ import annotations

import argparse
import os
import random

import torch

from src.language.training_budget import cap_prediction_targets, set_token_scheduled_lr
from src.language.training_loop import run_training
from src.language.training_prepare import prepare_training
from src.language.training_profiles import PROFILES
from src.language.training_session_contract import (
    PREFLIGHT_MANIFEST_VERSION,
    SEED,
    TRAINER_VERSION,
    TRAINING_STATE_VERSION,
)
from src.language.training_support import normalize_stage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="2m")
    parser.add_argument("--bundle-dir", default=None)
    parser.add_argument("--training-stage", default="foundation")
    parser.add_argument("--init-bundle", default=None)
    parser.add_argument("--hf-config", required=True)
    parser.add_argument("--hf-sample-examples", type=int, default=None)
    parser.add_argument("--threads", type=int, default=max(1, min(4, os.cpu_count() or 2)))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--curriculum-upgrade", action="store_true")
    parser.add_argument("--session-hours", type=float, default=None)
    parser.add_argument("--target-tokens-per-parameter", type=float, default=None)
    parser.add_argument("--compute-probe-steps", type=int, default=3)
    parser.add_argument("--source-audit-rows", type=int, default=500)
    parser.add_argument("--min-source-serialization-rate", type=float, default=0.80)
    parser.add_argument("--warmup-fraction", type=float, default=0.02)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.curriculum_upgrade and not args.resume:
        raise ValueError("--curriculum-upgrade requires --resume")
    if args.session_hours is not None and args.session_hours <= 0:
        raise ValueError("--session-hours must be positive when supplied")
    args.training_stage = normalize_stage(args.training_stage)

    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    prepared = prepare_training(args)
    if prepared is not None:
        run_training(prepared, args)


__all__ = [
    "TRAINER_VERSION", "TRAINING_STATE_VERSION", "PREFLIGHT_MANIFEST_VERSION",
    "cap_prediction_targets", "main", "set_token_scheduled_lr",
]
