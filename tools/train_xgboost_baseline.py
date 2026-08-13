"""XGBoost Baseline Trainer + Label Economics Audit.

As recommended by the quant: before tuning a 16-layer ResNet further,
establish a tree-based baseline. If XGBoost cannot find an edge in the
cleaned 18-feature dataset, a neural network definitely won't.

Also audits label economics:
  - Positive rate (win rate)
  - avg_win / avg_loss ratio
  - Kelly fraction alignment
  - Mathematical break-even check

Usage:
    python tools/train_xgboost_baseline.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.strategy.local_edge_model import (
    FEATURE_NAMES,
    build_dataset,
    load_feature_rows,
    vectorize_features,
)
from src.strategy.intelligence_pipeline import build_labeled_rows

# ── scikit-learn tree-based baselines (always available) ─────────────────────
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler


SEP = "=" * 100


def _precision_recall_pf(probs, labels, realized, threshold):
    preds = (probs >= threshold).astype(float)
    tp = float(np.sum((preds == 1) & (labels == 1)))
    fp = float(np.sum((preds == 1) & (labels == 0)))
    fn = float(np.sum((preds == 0) & (labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # Profit factor
    pos_mask = preds == 1
    wins  = realized[pos_mask & (labels == 1)]
    losses = realized[pos_mask & (labels == 0)]
    total_win  = float(np.sum(np.abs(wins))) if len(wins) else 0.0
    total_loss = float(np.sum(np.abs(losses))) if len(losses) else 0.0
    pf = min(total_win / total_loss, 10.0) if total_loss > 1e-9 else (999.0 if total_win > 0 else 0.0)
    allowed_rate = float(np.mean(preds))
    return precision, recall, pf, allowed_rate, int(tp), int(fp), int(fn)


def label_economics_audit(rows: list[dict]) -> dict:
    """Audit the label economics — win rate, avg_win, avg_loss, break-even check."""
    realized = np.asarray([float(r.get("realized_return", 0.0) or 0.0) for r in rows if "label" in r])
    labels   = np.asarray([float(r["label"]) for r in rows if "label" in r])

    if len(labels) == 0:
        return {"error": "no labeled rows"}

    pos_rate = float(labels.mean())
    win_returns  = realized[labels == 1]
    loss_returns = realized[labels == 0]

    avg_win  = float(np.mean(win_returns))  if len(win_returns)  > 0 else 0.0
    avg_loss = float(np.mean(loss_returns)) if len(loss_returns) > 0 else 0.0
    avg_loss_abs = abs(avg_loss)

    # Minimum required avg_win for break-even at this win rate:
    # Expected value = pos_rate * avg_win + (1-pos_rate) * avg_loss >= 0
    # avg_win >= avg_loss_abs * (1-pos_rate) / pos_rate
    break_even_rr = (avg_loss_abs * (1.0 - pos_rate) / max(pos_rate, 1e-9)) if pos_rate > 0 else 999.0
    actual_rr     = avg_win / avg_loss_abs if avg_loss_abs > 1e-9 else 0.0
    ev = pos_rate * avg_win + (1.0 - pos_rate) * avg_loss

    # Kelly fraction (theoretical)
    # f* = (b*p - q) / b  where b = avg_win/avg_loss, p = win_rate, q = 1-p
    b = actual_rr
    kelly = ((b * pos_rate) - (1.0 - pos_rate)) / b if b > 1e-9 else -1.0

    viable = ev > 0 and actual_rr >= break_even_rr

    return {
        "total_labeled": int(len(labels)),
        "positive_rate": round(pos_rate, 4),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "actual_rr": round(actual_rr, 4),
        "break_even_rr": round(break_even_rr, 4),
        "expected_value_per_trade": round(ev, 6),
        "kelly_fraction": round(kelly, 4),
        "mathematically_viable": viable,
        "verdict": (
            f"WIN RATE={pos_rate:.1%}, RR={actual_rr:.2f}x (need >={break_even_rr:.2f}x to break even). "
            f"EV per trade = {ev:.5f}. "
            f"{'MATHEMATICALLY VIABLE - labels support a profitable edge.' if viable else 'MATHEMATICALLY IMPOSSIBLE - labels cannot produce a profitable strategy at this win rate and RR.'}"
        ),
    }


def walk_forward_tree(model_factory, x, y, realized, folds=5, label="Model"):
    """5-fold walk-forward tree-based model validation."""
    results = []
    fold_size = len(x) // (folds + 1)

    for fold in range(1, folds + 1):
        train_end = fold_size * fold
        test_end  = fold_size * (fold + 1) if fold < folds else len(x)
        if train_end < 20 or test_end <= train_end:
            continue

        x_tr, y_tr = x[:train_end], y[:train_end]
        x_te, y_te = x[train_end:test_end], y[train_end:test_end]
        real_te    = realized[train_end:test_end]

        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(x_tr)
        x_te_s = scaler.transform(x_te)

        model = model_factory(fold)
        model.fit(x_tr_s, y_tr)
        probs = model.predict_proba(x_te_s)[:, 1]

        # Search best threshold
        best = None
        for thr in np.arange(0.30, 0.91, 0.02):
            p, r, pf, rate, tp, fp, fn = _precision_recall_pf(probs, y_te, real_te, thr)
            if rate < 0.005 or rate > 0.45 or r < 0.03 or p < 0.10:
                score = -1.0
            else:
                fbeta = (1.25 * p * r) / max(0.25 * p + r, 1e-9)
                prec_boost = 0.30 if p >= 0.52 else (0.15 if p >= 0.45 else 0.0)
                score = 0.50 * fbeta + 0.20 * min(pf / 3.0, 1.0) + prec_boost
            if best is None or score > best["score"]:
                best = dict(threshold=float(thr), score=score, precision=p, recall=r,
                            profit_factor=pf, allowed_rate=rate, tp=tp, fp=fp, fn=fn)

        if best:
            results.append({"fold": fold, "train": train_end, "test": test_end - train_end, **best})

    return results


def _print_walk_forward(results, label, gate_prec=0.52, gate_pf=1.05):
    passing = 0
    for res in results:
        p = res["precision"]
        gate = "PASS" if p >= gate_prec else "FAIL"
        if p >= gate_prec:
            passing += 1
        print(f"  Fold {res['fold']}/5 [{gate}]: "
              f"precision={p:.3f}  recall={res['recall']:.3f}  "
              f"PF={res['profit_factor']:.2f}  "
              f"rate={res['allowed_rate']:.1%}  "
              f"thr={res['threshold']:.2f}  "
              f"TP={res['tp']} FP={res['fp']} FN={res['fn']}")

    avg_prec = float(np.mean([r["precision"] for r in results])) if results else 0.0
    avg_pf   = float(np.mean([r["profit_factor"] for r in results])) if results else 0.0
    avg_rec  = float(np.mean([r["recall"] for r in results])) if results else 0.0

    print(f"\n  {label} Summary: avg_precision={avg_prec:.3f}  avg_recall={avg_rec:.3f}  "
          f"avg_PF={avg_pf:.2f}  folds_passing={passing}/{len(results)}")

    if avg_prec >= gate_prec and avg_pf >= gate_pf:
        verdict = "PASSES gate. Features contain a real edge. Neural net should match this."
    elif avg_prec >= 0.42:
        verdict = "Finds partial signal. Features have promise. Neural net may improve with tuning."
    else:
        verdict = "CANNOT find an edge. Root cause is FEATURES/LABELS — not the model architecture."

    print(f"  VERDICT: {label} {verdict}")
    return avg_prec, avg_pf


def main():
    print(SEP)
    print(" TREE BASELINE + LABEL ECONOMICS AUDIT")
    print(" (scikit-learn GradientBoosting + RandomForest)")
    print(SEP)
    print(f"\n Features ({len(FEATURE_NAMES)} total — pruned from 74):")
    for i, f in enumerate(FEATURE_NAMES):
        print(f"   {i+1:2d}. {f}")

    # ── Load data ─────────────────────────────────────────────────────────────
    rows = build_labeled_rows(
        feature_path="data/features.jsonl",
        diagnostics_path="data/live_diagnostics.jsonl",
        paper_trades_path="data/paper_trades.jsonl",
        horizon_bars=12,
    )
    rows = [r for r in rows if "label" in r]
    print(f"\n Total labeled rows: {len(rows):,}")

    if len(rows) < 30:
        print("  Not enough rows to evaluate. Check data/features.jsonl.")
        return

    # ── Label Economics Audit ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(" LABEL ECONOMICS AUDIT")
    print(SEP)
    econ = label_economics_audit(rows)
    for k, v in econ.items():
        if k != "verdict":
            print(f"  {k:35s}: {v}")
    print(f"\n  {econ.get('verdict', '')}\n")

    if not econ.get("mathematically_viable"):
        print("  *** WARNING: Labels are NOT economically viable.")
        print("  *** The avg_win/avg_loss ratio does not support a 21% win rate.")
        print("  *** Fix TP/SL ratio in MT5 before any further training.\n")

    # ── Feature correlation check ─────────────────────────────────────────────
    x_all = np.vstack([vectorize_features(r) for r in rows])
    y_all = np.asarray([float(r["label"]) for r in rows])
    real_all = np.asarray([float(r.get("realized_return", 0.0) or 0.0) for r in rows])

    print(SEP)
    print(" FEATURE CORRELATION CHECK (top 10 pairs)")
    print(SEP)
    corr = np.corrcoef(x_all.T)
    pairs = []
    n = len(FEATURE_NAMES)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((abs(corr[i, j]), FEATURE_NAMES[i], FEATURE_NAMES[j]))
    pairs.sort(reverse=True)
    for r_val, fa, fb in pairs[:10]:
        flag = " *** DROP ONE" if r_val > 0.85 else ""
        print(f"  {fa:30s} <-> {fb:30s}  r={r_val:.3f}{flag}")

    # ── Gradient Boosting Walk-Forward ────────────────────────────────────────
    print(f"\n{SEP}")
    print(" GRADIENT BOOSTING CLASSIFIER — 5-Fold Walk-Forward")
    print(SEP)
    n_pos = int(y_all.sum())
    n_neg = int(len(y_all) - n_pos)
    pw = n_neg / max(n_pos, 1)

    def gbm_factory(fold):
        return GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42 + fold,
        )

    gbm_results = walk_forward_tree(gbm_factory, x_all, y_all, real_all, label="GradientBoosting")
    gbm_prec, gbm_pf = _print_walk_forward(gbm_results, "GradientBoosting")

    # ── Random Forest Walk-Forward ────────────────────────────────────────────
    print(f"\n{SEP}")
    print(" RANDOM FOREST CLASSIFIER — 5-Fold Walk-Forward")
    print(SEP)

    def rf_factory(fold):
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42 + fold,
            n_jobs=-1,
        )

    rf_results = walk_forward_tree(rf_factory, x_all, y_all, real_all, label="RandomForest")
    rf_prec, rf_pf = _print_walk_forward(rf_results, "RandomForest")

    # ── Final Diagnosis ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(" FINAL DIAGNOSIS")
    print(SEP)
    best_prec = max(gbm_prec, rf_prec)
    if best_prec >= 0.52:
        print("  GOOD NEWS: Tree models find the edge. ResNet can match this with the 18-feature set.")
        print("  ACTION: Run train_reasoning_model.py with current settings.")
    elif best_prec >= 0.42:
        print("  PARTIAL SIGNAL: Tree models find weak edge. Further feature engineering needed.")
        print("  ACTION: Review top correlated pairs above and add MT5-specific features.")
    else:
        print("  NO SIGNAL: Tree models cannot find edge. Root cause is labels/features.")
        print("  ACTION:")
        print("    1. Fix TP/SL ratio — ensure avg_win >= break_even_rr * avg_loss")
        print("    2. Add higher-quality features: VWAP, order flow imbalance, tick volume")
        print("    3. Increase min_abs_return filter to only train on high-conviction setups")
    print(SEP)

    # Save audit to JSON
    audit_path = Path("data/models/label_economics_audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({
        "economics": econ,
        "gradient_boosting": {"avg_precision": gbm_prec, "avg_pf": gbm_pf,
                               "folds": gbm_results},
        "random_forest":     {"avg_precision": rf_prec,  "avg_pf": rf_pf,
                               "folds": rf_results},
    }, indent=2), encoding="utf-8")
    print(f"\n  Audit saved to {audit_path}")


if __name__ == "__main__":
    main()

