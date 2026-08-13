"""Deep Residual Reasoning Model Trainer.

Trains a 32-layer deep residual neural network on XAUUSD features and logs
every forward/backward propagation step with full chain-of-thought reasoning
to `data/models/training_steps.jsonl` — including epoch-level reasoning sentences,
fold-level evaluations, and an overall walk-forward reasoning summary.

These reasoning sentences are designed to serve as training targets when the
model is later extended with language generation capabilities.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.strategy.intelligence_pipeline import run_pipeline


def main():
    sep = "=" * 100
    print(sep)
    print(" DEEP RESIDUAL REASONING MODEL — 32-LAYER TRAINING & PROPAGATION LOG")
    print(sep)

    steps_log_file = Path("data/models/training_steps.jsonl")
    steps_log_file.parent.mkdir(parents=True, exist_ok=True)
    if steps_log_file.exists():
        steps_log_file.unlink()

    args = SimpleNamespace(
        features="data/features.jsonl",
        diagnostics="data/live_diagnostics.jsonl",
        paper_trades="data/paper_trades.jsonl",
        candles=None,
        candidate_model="data/models/local_edge_model.candidate.npz",
        live_model="data/models/local_edge_model.npz",
        report="data/models/training_report_reasoning.json",
        horizon_bars=12,
        folds=5,
        epochs=400,
        learning_rate=0.003,
        hidden_size=48,
        num_layers=16,
        threshold=0.55,
        fixed_threshold=False,
        min_abs_return=0.0,
        promote=True,
        min_promotion_samples=100,
        min_promotion_precision=0.52,
        min_promotion_profit_factor=1.05,
    )

    print(f"\n[Architecture]  {args.num_layers} layers  |  hidden_size={args.hidden_size}")
    print(f"[Topology]      74 inputs -> {args.hidden_size} projection -> "
          f"{(args.num_layers - 2) // 2} residual blocks x 2 -> 1 output")
    print(f"[Training]      epochs={args.epochs}  lr={args.learning_rate}  folds={args.folds}")
    print(f"[Promotion]     precision>={args.min_promotion_precision}  "
          f"PF>={args.min_promotion_profit_factor}")
    print("\n[1/3] Running walk-forward evaluation with full reasoning logging...\n")

    report = run_pipeline(args)

    # ── Extract and write step logs ───────────────────────────────────────────
    candidate = report.get("candidate", {}) or {}
    step_logs = candidate.get("step_logs", [])

    print(f"[2/3] Writing {len(step_logs)} epoch step logs to {steps_log_file}...")
    with open(steps_log_file, "w", encoding="utf-8") as f:
        for step in step_logs:
            f.write(json.dumps(step) + "\n")

    # ── Print fold-level reasoning ────────────────────────────────────────────
    walk_fwd = report.get("walk_forward", {})
    folds = walk_fwd.get("folds", [])

    print("\n[3/3] Walk-Forward Fold Reasoning:")
    print("-" * 100)
    for fold_report in folds:
        fold_reasoning = fold_report.get("reasoning", "")
        if fold_reasoning:
            print(f"\n  {fold_reasoning}")
            prec = fold_report.get("precision", 0.0)
            rec = fold_report.get("recall", 0.0)
            pf = fold_report.get("profit_factor", 0.0)
            rate = fold_report.get("allowed_rate", 0.0)
            print(f"    ↳ Metrics: precision={prec:.3f}  recall={rec:.3f}  "
                  f"profit_factor={pf:.2f}  allowed_rate={rate:.1%}")

    # ── Overall reasoning ─────────────────────────────────────────────────────
    overall_reasoning = walk_fwd.get("reasoning", "")
    avg = walk_fwd.get("average", {})
    promo = report.get("promotion", {}) or {}

    print("\n" + sep)
    print(" OVERALL MODEL REASONING")
    print(sep)
    if overall_reasoning:
        print(f"\n  {overall_reasoning}\n")

    print(sep)
    print(" TRAINING SUMMARY")
    print(sep)
    print(f"  Architecture : {args.num_layers}-layer Deep Residual MLP")
    print(f"  Total Rows   : {report.get('rows', 0):,}")
    print(f"  Step Logs    : {len(step_logs)} epochs logged to {steps_log_file}")
    print(f"  Train Acc    : {candidate.get('train_accuracy', 0.0):.4f}")
    print(f"  Val Acc      : {candidate.get('validation_accuracy', 0.0):.4f}")
    print(f"  Avg Precision: {avg.get('precision', 0.0):.4f}  (gate: {args.min_promotion_precision})")
    print(f"  Avg Recall   : {avg.get('recall', 0.0):.4f}")
    print(f"  Avg PF       : {avg.get('profit_factor', 0.0):.4f}  (gate: {args.min_promotion_profit_factor})")
    print(f"  Promoted     : {promo.get('promoted', False)}")
    if not promo.get("promoted"):
        print(f"  Reason       : {promo.get('reason', 'unknown')}")

    # ── Sample reasoning sentences ────────────────────────────────────────────
    if step_logs:
        print(f"\n  [First Epoch Reasoning]\n  {step_logs[0].get('reasoning', '')}")
        if len(step_logs) > 1:
            print(f"\n  [Final Epoch Reasoning]\n  {step_logs[-1].get('reasoning', '')}")

    print(sep)


if __name__ == "__main__":
    main()
