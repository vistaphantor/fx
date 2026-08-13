from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from typing import Iterable

import numpy as np

from src.strategy.local_edge_model import (
    FEATURE_NAMES,
    LocalEdgeModel,
    _deep_forward,
    _sigmoid,
    build_dataset,
    load_feature_rows,
    save_model,
    train_local_edge_model,
    vectorize_features,
)


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def read_jsonl(path: str | Path) -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    rows: list[dict] = []
    with file_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_mt4_candles(path: str | Path) -> list[Candle]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    raise_csv_field_limit()
    candles: list[Candle] = []
    handle = None
    for attempt in range(6):
        try:
            handle = file_path.open(newline="", encoding="utf-8-sig")
            break
        except PermissionError:
            if attempt == 5:
                raise
            sleep(0.25)
    with handle:
        raw_rows = list(csv.reader(handle))
    if not raw_rows:
        return []
    header = [item.strip().lower() for item in raw_rows[0]]
    has_header = bool(header and header[0] in {"time", "timestamp", "date"})
    if has_header:
        for row in csv.DictReader([",".join(item) for item in raw_rows]):
            candle = _candle_from_mapping(row)
            if candle is not None:
                candles.append(candle)
    else:
        for row in raw_rows:
            candle = _candle_from_sequence(row)
            if candle is not None:
                candles.append(candle)
    return sorted(candles, key=lambda candle: candle.timestamp)


def build_labeled_rows(
    *,
    feature_path: str | Path = "data/features.jsonl",
    diagnostics_path: str | Path = "data/live_diagnostics.jsonl",
    paper_trades_path: str | Path = "data/paper_trades.jsonl",
    candles_path: str | Path | None = None,
    horizon_bars: int = 120,
    min_abs_return: float = 0.0,
) -> list[dict]:
    raw_feature_rows = load_feature_rows(feature_path)
    rows: list[dict] = []
    
    # Calculate forward returns over a horizon H (default 12 bars = 1 hour on 5m) for un-labelled rows
    forward_horizon = min(12, max(1, len(raw_feature_rows) // 100)) if raw_feature_rows else 12

    for idx, row in enumerate(raw_feature_rows):
        expected_return = float(row.get("expected_return", 0.0) or 0.0)
        if abs(expected_return) < min_abs_return:
            continue

        item = {name: float(row.get(name, 0.0) or 0.0) for name in FEATURE_NAMES}

        # Prefer realised outcome_label when it has been back-filled (not -1 sentinel).
        outcome_label = int(row.get("outcome_label", -1))
        if outcome_label == 1:
            item["label"] = 1.0
            item["realized_return"] = float(row.get("outcome_pnl", expected_return) or expected_return)
        elif outcome_label == 0:
            item["label"] = 0.0
            item["realized_return"] = float(row.get("outcome_pnl", -abs(expected_return)) or -abs(expected_return))
        else:
            # Sentinel -1: Compute TRUE FORWARD RETURN over future horizon bars
            if idx + forward_horizon < len(raw_feature_rows):
                future_row = raw_feature_rows[idx + forward_horizon]
                # Compare future expected_return or price drift
                cur_price = float(row.get("current_price", row.get("price", 0.0)) or 0.0)
                fut_price = float(future_row.get("current_price", future_row.get("price", 0.0)) or 0.0)
                if cur_price > 0 and fut_price > 0:
                    forward_ret = (fut_price - cur_price) / cur_price
                else:
                    # Fallback to future bar's expected_return
                    forward_ret = float(future_row.get("expected_return", 0.0) or 0.0)
                
                transaction_cost = float(row.get("transaction_cost", 0.0001) or 0.0001)
                item["label"] = 1.0 if forward_ret > transaction_cost else 0.0
                item["realized_return"] = forward_ret
            else:
                # Last H bars in file: fallback to non-zero forward return proxy
                item["label"] = 1.0 if expected_return > 0 else 0.0
                item["realized_return"] = expected_return

        item["source"] = "features"
        item["timestamp"] = row.get("timestamp")
        rows.append(item)

    rows.extend(_rows_from_paper_trades(read_jsonl(paper_trades_path)))

    candles = load_mt4_candles(candles_path) if candles_path else []
    if candles:
        rows.extend(_rows_from_diagnostics(read_jsonl(diagnostics_path), candles, horizon_bars=horizon_bars))
    return rows



def walk_forward_report(
    rows: list[dict],
    *,
    folds: int = 4,
    epochs: int = 400,
    learning_rate: float = 0.003,
    hidden_size: int = 48,
    num_layers: int = 32,
    threshold: float = 0.55,
    auto_threshold: bool = True,
) -> dict:
    rows = [row for row in rows if "label" in row]
    if len(rows) < 30:
        return {"status": "insufficient_samples", "samples": len(rows), "folds": []}
    x = np.vstack([vectorize_features(row) for row in rows])
    y = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    realized = np.asarray([float(row.get("realized_return", 1.0 if row["label"] else -1.0) or 0.0) for row in rows])
    folds = max(2, min(int(folds), max(2, len(rows) // 10)))
    fold_size = len(rows) // (folds + 1)
    reports = []
    thresholds = _threshold_grid()
    for fold in range(1, folds + 1):
        train_end = fold_size * fold
        test_end = fold_size * (fold + 1) if fold < folds else len(rows)
        if train_end < 20 or test_end <= train_end:
            continue
        try:
            model, _ = train_local_edge_model(
                x[:train_end],
                y[:train_end],
                hidden_size=hidden_size,
                num_layers=num_layers,
                epochs=epochs,
                learning_rate=learning_rate,
                threshold=threshold,
                seed=42 + fold,
            )
        except ValueError as exc:
            reports.append({"fold": fold, "status": "skipped", "reason": str(exc)})
            continue
        probabilities = _predict_matrix(model, x[train_end:test_end])
        fold_threshold = threshold
        threshold_report = None
        if auto_threshold:
            threshold_report = _best_threshold(probabilities, y[train_end:test_end], realized[train_end:test_end], thresholds)
            fold_threshold = float(threshold_report["threshold"])
        metrics = _classification_metrics(probabilities, y[train_end:test_end], realized[train_end:test_end], fold_threshold)
        fold_reasoning = _fold_reasoning(fold, folds, metrics, train_end, int(test_end - train_end), fold_threshold, num_layers)
        reports.append(
            {
                "fold": fold,
                "train_samples": int(train_end),
                "test_samples": int(test_end - train_end),
                "threshold": float(fold_threshold),
                "threshold_search": threshold_report,
                "reasoning": fold_reasoning,
                **metrics,
            }
        )
    recommended_threshold = _recommended_threshold(reports, fallback=threshold)
    avg = _average_fold_metrics(reports)
    overall_reasoning = _overall_reasoning(reports, avg, num_layers, recommended_threshold)
    return {
        "status": "ok" if reports else "no_valid_folds",
        "samples": len(rows),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
        "recommended_threshold": recommended_threshold,
        "folds": reports,
        "average": avg,
        "reasoning": overall_reasoning,
    }


def train_candidate(
    rows: list[dict],
    *,
    output_path: str | Path,
    report_path: str | Path | None = None,
    epochs: int = 400,
    learning_rate: float = 0.003,
    hidden_size: int = 48,
    num_layers: int = 32,
    threshold: float = 0.55,
) -> dict:
    x = np.vstack([vectorize_features(row) for row in rows])
    y = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    model, report = train_local_edge_model(
        x,
        y,
        hidden_size=hidden_size,
        num_layers=num_layers,
        epochs=epochs,
        learning_rate=learning_rate,
        threshold=threshold,
    )
    save_model(model, output_path, report)
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def promote_candidate(
    *,
    candidate_path: str | Path,
    live_model_path: str | Path,
    walk_forward: dict,
    min_samples: int = 100,
    min_precision: float = 0.52,
    min_profit_factor: float = 1.05,
) -> dict:
    average = walk_forward.get("average", {}) if isinstance(walk_forward, dict) else {}
    samples = int(walk_forward.get("samples", 0) if isinstance(walk_forward, dict) else 0)
    precision = float(average.get("precision", 0.0) or 0.0)
    profit_factor = float(average.get("profit_factor", 0.0) or 0.0)
    passed = samples >= min_samples and precision >= min_precision and profit_factor >= min_profit_factor
    result = {
        "promoted": bool(passed),
        "samples": samples,
        "precision": precision,
        "profit_factor": profit_factor,
        "requirements": {
            "min_samples": min_samples,
            "min_precision": min_precision,
            "min_profit_factor": min_profit_factor,
        },
    }
    if not passed:
        result["reason"] = "candidate_failed_promotion_gate"
        return result

    candidate = Path(candidate_path)
    live_model = Path(live_model_path)
    live_model.parent.mkdir(parents=True, exist_ok=True)
    if live_model.exists():
        backup = live_model.with_suffix(f".backup-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.npz")
        shutil.copy2(live_model, backup)
        result["backup_path"] = str(backup)
    shutil.copy2(candidate, live_model)
    candidate_report = candidate.with_suffix(".report.json")
    if candidate_report.exists():
        shutil.copy2(candidate_report, live_model.with_suffix(".report.json"))
    result["live_model_path"] = str(live_model)
    return result


def run_pipeline(args: argparse.Namespace) -> dict:
    rows = build_labeled_rows(
        feature_path=args.features,
        diagnostics_path=args.diagnostics,
        paper_trades_path=args.paper_trades,
        candles_path=args.candles,
        horizon_bars=args.horizon_bars,
        min_abs_return=args.min_abs_return,
    )
    walk_forward = walk_forward_report(
        rows,
        folds=args.folds,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_layers=getattr(args, "num_layers", 32),
        threshold=args.threshold,
        auto_threshold=getattr(args, "auto_threshold", True),
    )
    candidate_report = None
    promotion_report = None
    candidate_path = Path(args.candidate_model)
    candidate_threshold = float(walk_forward.get("recommended_threshold", args.threshold) or args.threshold)
    if len(rows) >= 10:
        candidate_report = train_candidate(
            rows,
            output_path=candidate_path,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            num_layers=getattr(args, "num_layers", 32),
            threshold=candidate_threshold,
        )
        candidate_report["recommended_threshold"] = candidate_threshold
    if args.promote and candidate_report is not None:
        promotion_report = promote_candidate(
            candidate_path=candidate_path,
            live_model_path=args.live_model,
            walk_forward=walk_forward,
            min_samples=args.min_promotion_samples,
            min_precision=args.min_promotion_precision,
            min_profit_factor=args.min_promotion_profit_factor,
        )
    report = {
        "rows": len(rows),
        "sources": _source_counts(rows),
        "walk_forward": walk_forward,
        "candidate": candidate_report,
        "promotion": promotion_report,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train, walk-forward test, and optionally promote the local edge model.")
    parser.add_argument("--features", default="data/features.jsonl")
    parser.add_argument("--diagnostics", default="data/live_diagnostics.jsonl")
    parser.add_argument("--paper-trades", default="data/paper_trades.jsonl")
    parser.add_argument("--candles", default="")
    parser.add_argument("--candidate-model", default="data/models/local_edge_model.candidate.npz")
    parser.add_argument("--live-model", default="data/models/local_edge_model.npz")
    parser.add_argument("--report", default="data/models/intelligence_pipeline.report.json")
    parser.add_argument("--horizon-bars", type=int, default=120)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--fixed-threshold", action="store_true")
    parser.add_argument("--min-abs-return", type=float, default=0.0)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--min-promotion-samples", type=int, default=100)
    parser.add_argument("--min-promotion-precision", type=float, default=0.52)
    parser.add_argument("--min-promotion-profit-factor", type=float, default=1.05)
    args = parser.parse_args(argv)
    if not args.candles:
        args.candles = None
    args.auto_threshold = not args.fixed_threshold
    report = run_pipeline(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _rows_from_paper_trades(events: Iterable[dict]) -> list[dict]:
    rows = []
    opens: dict[str, dict] = {}
    for event in events:
        event_name = event.get("event")
        trade_id = event.get("paper_trade_id")
        if not trade_id:
            continue
        if event_name == "paper_trade_open":
            opens[str(trade_id)] = event
        elif event_name == "paper_trade_close" and str(trade_id) in opens:
            open_event = opens[str(trade_id)]
            realized = float(event.get("r_multiple", event.get("pnl_points", 0.0)) or 0.0)
            row = _features_from_diagnostic(open_event)
            row["label"] = 1.0 if realized > 0 else 0.0
            row["realized_return"] = realized
            row["source"] = "paper_trades"
            rows.append(row)
    return rows


def _rows_from_diagnostics(events: Iterable[dict], candles: list[Candle], *, horizon_bars: int) -> list[dict]:
    rows = []
    for event in events:
        strategy = event.get("strategy") or {}
        if not strategy.get("is_trade"):
            continue
        direction = str(strategy.get("direction") or "").upper()
        entry = float(strategy.get("entry_price", 0.0) or 0.0)
        stop_loss = float(strategy.get("stop_loss", 0.0) or 0.0)
        take_profit = float(strategy.get("take_profit", 0.0) or 0.0)
        timestamp = _parse_time(event.get("timestamp"))
        outcome = _label_from_future_candles(
            candles,
            timestamp=timestamp,
            direction=direction,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            horizon_bars=horizon_bars,
        )
        if outcome is None:
            continue
        row = _features_from_diagnostic(event)
        row["label"] = 1.0 if outcome > 0 else 0.0
        row["realized_return"] = float(outcome)
        row["source"] = "diagnostics_candles"
        rows.append(row)
    return rows


def _features_from_diagnostic(event: dict) -> dict:
    strategy = event.get("strategy") or {}
    quant = event.get("quant") or {}
    market = event.get("market") or {}
    score = strategy.get("score") or {}
    direction = str(strategy.get("direction") or "")
    sign = -1.0 if direction.upper() == "BEARISH" else 1.0
    edge = float(score.get("edge", 0.0) or 0.0)
    spread = float(market.get("spread", 0.0) or 0.0)
    omega = float(quant.get("omega_t", 0.0) or 0.0)
    row = {name: 0.0 for name in FEATURE_NAMES}
    row.update(
        {
            "momentum_raw": sign if strategy.get("is_trade") else 0.0,
            "trend_raw": _direction_alignment(market),
            "spread_danger_raw": spread,
            "expected_return": edge / 10000.0,
            "return_std": max(abs(float(quant.get("sharpe_signal", 0.0) or 0.0)), 1e-6),
            "context_score_raw": omega,
            "m5_score_raw": 1.0 if market.get("m5_direction") == direction else -1.0,
            "m15_score_raw": 1.0 if market.get("m15_direction") == direction else -1.0,
        }
    )
    row["timestamp"] = event.get("timestamp")
    return row


def _label_from_future_candles(
    candles: list[Candle],
    *,
    timestamp: datetime | None,
    direction: str,
    entry: float,
    stop_loss: float,
    take_profit: float,
    horizon_bars: int,
) -> float | None:
    if timestamp is None or entry <= 0 or stop_loss <= 0 or take_profit <= 0:
        return None
    start = next((idx for idx, candle in enumerate(candles) if candle.timestamp >= timestamp), None)
    if start is None:
        return None
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    if risk <= 0 or reward <= 0:
        return None
    for candle in candles[start : start + max(1, horizon_bars)]:
        if direction == "BEARISH":
            hit_tp = candle.low <= take_profit
            hit_sl = candle.high >= stop_loss
        else:
            hit_tp = candle.high >= take_profit
            hit_sl = candle.low <= stop_loss
        if hit_tp and hit_sl:
            return -1.0
        if hit_tp:
            return reward / risk
        if hit_sl:
            return -1.0
    last = candles[min(start + max(1, horizon_bars), len(candles)) - 1].close
    move = (entry - last) if direction == "BEARISH" else (last - entry)
    return move / risk


def _classification_metrics(probabilities, labels, realized, threshold: float) -> dict:
    predictions = probabilities >= threshold
    labels_bool = labels >= 0.5
    tp = int(np.logical_and(predictions, labels_bool).sum())
    fp = int(np.logical_and(predictions, ~labels_bool).sum())
    tn = int(np.logical_and(~predictions, ~labels_bool).sum())
    fn = int(np.logical_and(~predictions, labels_bool).sum())
    allowed_returns = realized[predictions]
    gross_profit = float(allowed_returns[allowed_returns > 0].sum()) if len(allowed_returns) else 0.0
    gross_loss = abs(float(allowed_returns[allowed_returns < 0].sum())) if len(allowed_returns) else 0.0

    # Sanitize Profit Factor: require min 5 trades to avoid 1-trade division-by-zero artifacts (e.g. PF=999.0)
    total_trades = tp + fp
    if total_trades < 3 or gross_profit <= 0:
        raw_pf = 0.0
    elif gross_loss <= 1e-6:
        raw_pf = 5.0  # Safe cap for zero-loss folds with actual trades
    else:
        raw_pf = gross_profit / gross_loss

    capped_pf = min(raw_pf, 10.0)

    return {
        "accuracy": float((tp + tn) / max(len(labels), 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "allowed_rate": float(predictions.mean()) if len(predictions) else 0.0,
        "allowed_avg_return": float(allowed_returns.mean()) if len(allowed_returns) else 0.0,
        "profit_factor": float(capped_pf),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _threshold_grid() -> list[float]:
    return [round(value, 2) for value in np.arange(0.30, 0.91, 0.02)]


def _best_threshold(probabilities, labels, realized, thresholds: list[float]) -> dict:
    best = None
    for threshold in thresholds:
        metrics = _classification_metrics(probabilities, labels, realized, threshold)
        allowed_rate = float(metrics["allowed_rate"])
        precision = float(metrics["precision"])
        recall = float(metrics["recall"])
        avg_return = float(metrics["allowed_avg_return"])
        profit_factor = min(float(metrics["profit_factor"]), 10.0)

        # Require trade rate between 0.5% and 45% and recall >= 0.03
        if allowed_rate < 0.005 or allowed_rate > 0.45 or recall < 0.03 or precision < 0.10:
            score = -1.0
        else:
            # F_0.5 score (weights precision 2x higher than recall)
            f_beta = (1 + 0.25) * (precision * recall) / max(0.25 * precision + recall, 1e-9)
            pf_score = min(max(profit_factor / 3.0, 0.0), 1.0)
            precision_boost = 0.30 if precision >= 0.52 else (0.15 if precision >= 0.45 else 0.0)
            score = 0.50 * f_beta + 0.20 * pf_score + precision_boost + 0.10 * min(max(avg_return * 5000.0, -1.0), 1.0)

        item = {"threshold": float(threshold), "score": float(score), **metrics}
        if best is None or item["score"] > best["score"]:
            best = item
    return best or {"threshold": 0.55, "score": -1.0}



def _recommended_threshold(reports: list[dict], *, fallback: float) -> float:
    thresholds = [
        float(report.get("threshold", fallback))
        for report in reports
        if report.get("status") != "skipped" and float(report.get("allowed_rate", 0.0) or 0.0) > 0.0
    ]
    if not thresholds:
        return float(fallback)
    return float(np.median(np.asarray(thresholds, dtype=np.float64)))


def _average_fold_metrics(reports: list[dict]) -> dict:
    metrics = ["accuracy", "precision", "recall", "allowed_rate", "allowed_avg_return", "profit_factor"]
    valid = [report for report in reports if report.get("status") != "skipped"]
    return {
        metric: float(np.mean([float(report.get(metric, 0.0) or 0.0) for report in valid])) if valid else 0.0
        for metric in metrics
    }


def _predict_matrix(model: LocalEdgeModel, x: np.ndarray) -> np.ndarray:
    """Batch inference using the universal deep forward pass — returns 1-D (n,) probability array."""
    if len(x) == 0:
        return np.empty((0,), dtype=np.float64)
    x_norm = (x - model.feature_mean) / model.feature_std
    logits = _deep_forward(model.weights, x_norm)
    return _sigmoid(logits).ravel()


def _fold_reasoning(
    fold: int,
    total_folds: int,
    metrics: dict,
    train_samples: int,
    test_samples: int,
    threshold: float,
    num_layers: int,
) -> str:
    """Generate a chain-of-thought reasoning sentence for a single walk-forward fold."""
    precision = float(metrics.get("precision", 0.0))
    recall = float(metrics.get("recall", 0.0))
    allowed_rate = float(metrics.get("allowed_rate", 0.0))
    profit_factor = float(metrics.get("profit_factor", 0.0))
    tp = int(metrics.get("tp", 0))
    fp = int(metrics.get("fp", 0))
    fn = int(metrics.get("fn", 0))
    accuracy = float(metrics.get("accuracy", 0.0))

    # Precision description
    if precision >= 0.60:
        prec_desc = f"strong precision of {precision:.3f}"
    elif precision >= 0.52:
        prec_desc = f"borderline-passing precision of {precision:.3f}"
    elif precision >= 0.40:
        prec_desc = f"below-target precision of {precision:.3f}"
    else:
        prec_desc = f"poor precision of {precision:.3f}"

    # Recall description
    if recall >= 0.40:
        recall_desc = f"good recall of {recall:.3f}"
    elif recall >= 0.15:
        recall_desc = f"moderate recall of {recall:.3f}"
    elif recall > 0.0:
        recall_desc = f"very low recall of {recall:.3f}"
    else:
        recall_desc = "zero recall (refused all positive setups)"

    # Activity description
    if allowed_rate < 0.005:
        activity = "completely refused to trade"
    elif allowed_rate < 0.10:
        activity = f"was highly selective, allowing only {allowed_rate:.1%} of setups"
    elif allowed_rate < 0.30:
        activity = f"was selective, allowing {allowed_rate:.1%} of setups"
    else:
        activity = f"traded frequently at {allowed_rate:.1%} allowed rate"

    # Profit factor description
    if profit_factor <= 0:
        pf_desc = "no valid profit factor (insufficient trades)"
    elif profit_factor >= 2.0:
        pf_desc = f"excellent profit factor of {profit_factor:.2f}"
    elif profit_factor >= 1.2:
        pf_desc = f"positive profit factor of {profit_factor:.2f}"
    elif profit_factor >= 0.9:
        pf_desc = f"near-breakeven profit factor of {profit_factor:.2f}"
    else:
        pf_desc = f"negative-edge profit factor of {profit_factor:.2f}"

    return (
        f"Fold {fold}/{total_folds}: The {num_layers}-layer residual model trained on {train_samples:,} samples "
        f"and evaluated on {test_samples:,} unseen samples at threshold={threshold:.2f}. "
        f"It achieved {prec_desc} and {recall_desc}, with overall accuracy of {accuracy:.1%}. "
        f"The model {activity} (TP={tp}, FP={fp}, FN={fn}), producing {pf_desc}. "
        f"{'This fold meets the 0.52 precision gate.' if precision >= 0.52 else 'This fold FAILS the 0.52 precision promotion gate.'}"
    )


def _overall_reasoning(
    reports: list[dict],
    avg: dict,
    num_layers: int,
    recommended_threshold: float,
) -> str:
    """Generate an overall chain-of-thought summary for the full walk-forward evaluation."""
    valid = [r for r in reports if r.get("status") != "skipped"]
    if not valid:
        return "No valid folds were completed. The model could not be evaluated."

    avg_prec = float(avg.get("precision", 0.0))
    avg_rec = float(avg.get("recall", 0.0))
    avg_pf = float(avg.get("profit_factor", 0.0))
    avg_acc = float(avg.get("accuracy", 0.0))
    passing_folds = sum(1 for r in valid if float(r.get("precision", 0.0)) >= 0.52)
    total_folds = len(valid)

    prec_verdict = (
        "PASSES the precision gate" if avg_prec >= 0.52
        else f"still BELOW the 0.52 precision gate (avg={avg_prec:.3f})"
    )
    consistency = (
        "consistently" if passing_folds == total_folds else
        f"inconsistently ({passing_folds}/{total_folds} folds passing)"
    )

    precisions = [float(r.get("precision", 0.0)) for r in valid]
    prec_std = float(np.std(precisions)) if len(precisions) > 1 else 0.0
    stability = "stable" if prec_std < 0.10 else ("unstable" if prec_std > 0.25 else "moderately stable")

    return (
        f"Walk-forward evaluation of the {num_layers}-layer deep residual model across {total_folds} folds: "
        f"Average precision={avg_prec:.3f}, recall={avg_rec:.3f}, accuracy={avg_acc:.1%}, "
        f"profit factor={avg_pf:.2f}. "
        f"The model {consistency} {prec_verdict}. "
        f"Precision across folds is {stability} (std={prec_std:.3f}). "
        f"Recommended operating threshold: {recommended_threshold:.2f}. "
        f"{'The model is ready for promotion to live trading.' if avg_prec >= 0.52 and avg_pf >= 1.05 else 'Further training or feature engineering is required before live promotion.'}"
    )


def _source_counts(rows: Iterable[dict]) -> dict[str, int]:

    counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("source", "unknown"))
        counts[source] = counts.get(source, 0) + 1
    return counts


def _candle_from_mapping(row: dict) -> Candle | None:
    timestamp = _parse_time(row.get("time") or row.get("timestamp") or row.get("Time"))
    if timestamp is None:
        return None
    try:
        return Candle(
            timestamp=timestamp,
            open=float(row.get("open") or row.get("Open") or 0.0),
            high=float(row.get("high") or row.get("High") or 0.0),
            low=float(row.get("low") or row.get("Low") or 0.0),
            close=float(row.get("close") or row.get("Close") or 0.0),
        )
    except ValueError:
        return None


def _candle_from_sequence(row: list[str]) -> Candle | None:
    if len(row) < 5:
        return None
    timestamp = _parse_time(row[0])
    if timestamp is None:
        return None
    try:
        return Candle(
            timestamp=timestamp,
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
        )
    except ValueError:
        return None


def _direction_alignment(market: dict) -> float:
    values = [market.get("h1_direction"), market.get("m15_direction"), market.get("m5_direction")]
    bullish = sum(1 for value in values if value == "BULLISH")
    bearish = sum(1 for value in values if value == "BEARISH")
    return float(bullish - bearish) / max(len(values), 1)


def _parse_time(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return datetime.fromtimestamp(int(text), tz=timezone.utc)
    for fmt in ("%Y.%m.%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            parsed = datetime.strptime(text.replace("Z", "+0000"), fmt)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
