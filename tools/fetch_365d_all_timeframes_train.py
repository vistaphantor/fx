"""Causal multi-asset market-data builder and local-edge model trainer.

Downloads liquid Gold/FX histories, constructs only information fully observable at
each historical decision time, labels a pre-gate directional candidate from later
same-symbol prices net of execution cost, globally time-sorts the experiment, and
runs the repo's authoritative local-edge pipeline with an untouched fixed-threshold
walk-forward evaluation.

This is research/model training. It does not connect to a broker or place trades.
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import yfinance as yf

from src.strategy.equity_tracker import EquityTracker
from src.strategy.features import FeatureExtractor, append_enriched_snapshot_to_file
from src.strategy.historical_training import (
    aggregate_consecutive_bars,
    average_true_range_proxy,
    closed_candles_at,
    realized_directional_outcome,
)
from src.strategy.quant_engine import QuantParams, evaluate_master_equation
from src.strategy.session_engine import SessionEngine


LABEL_HORIZON_M15_BARS = 4
MAX_TRAINING_ROWS_PER_SYMBOL = 700
SYMBOLS = [
    "GC=F",
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "AUDUSD=X",
    "USDCAD=X",
    "USDCHF=X",
]


def convert_df_to_candles(df):
    candles = []
    if df.empty:
        return candles
    for idx, row in df.iterrows():
        ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
        candles.append(
            SimpleNamespace(
                timestamp=ts,
                time=int(ts.timestamp()),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row.get("Volume", 100) or 0.0),
            )
        )
    return sorted(candles, key=lambda candle: candle.timestamp)


def _download_symbol_history(symbol: str) -> dict[str, list]:
    ticker = yf.Ticker(symbol)
    d1 = convert_df_to_candles(ticker.history(period="1y", interval="1d"))
    h1 = convert_df_to_candles(ticker.history(period="730d", interval="1h"))
    m30 = convert_df_to_candles(ticker.history(period="60d", interval="30m"))
    m15 = convert_df_to_candles(ticker.history(period="60d", interval="15m"))
    m5 = convert_df_to_candles(ticker.history(period="60d", interval="5m"))
    h4 = aggregate_consecutive_bars(h1, group_size=4)
    return {"D1": d1, "H4": h4, "H1": h1, "M30": m30, "M15": m15, "M5": m5}


def _causal_history(history: dict[str, list], decision_time) -> SimpleNamespace | None:
    d1 = closed_candles_at(history["D1"], decision_time, bar_duration=timedelta(days=1), limit=20)
    h4 = closed_candles_at(history["H4"], decision_time, bar_duration=timedelta(hours=4), limit=20)
    h1 = closed_candles_at(history["H1"], decision_time, bar_duration=timedelta(hours=1), limit=50)
    m30 = closed_candles_at(history["M30"], decision_time, bar_duration=timedelta(minutes=30), limit=50)
    m15 = closed_candles_at(history["M15"], decision_time, bar_duration=timedelta(minutes=15), limit=50)
    m5 = closed_candles_at(history["M5"], decision_time, bar_duration=timedelta(minutes=5), limit=20)

    if len(m15) < 20 or len(m5) < 5:
        return None
    if not d1:
        d1 = m15
    if not h4:
        h4 = h1 if h1 else m15
    if not h1:
        h1 = m15
    if not m30:
        m30 = m15

    return SimpleNamespace(
        d1_candles=d1,
        h4_candles=h4,
        h1_candles=h1,
        m30_candles=m30,
        m15_candles=m15,
        m5_candles=m5,
    )


def _absolute_spread(symbol: str) -> float:
    if symbol == "GC=F":
        return 0.25
    if symbol == "USDJPY=X":
        return 0.015
    return 0.00015


def _candidate_direction(quant_decision, expected_return: float) -> int:
    """Recover the directional proposal before TRADE/SKIP execution gates.

    The local-edge model is itself a TRADE/SKIP gate, so restricting its training
    set to trades already approved by the quant gate creates selection bias. The
    certainty-equivalent scores retain both directional proposals even when the
    quant engine ultimately returns action=0.
    """
    scores = (getattr(quant_decision, "metadata", {}) or {}).get("ce_scores") or {}
    long_score = float(scores.get(1, float("-inf")))
    short_score = float(scores.get(-1, float("-inf")))
    if long_score != short_score:
        return 1 if long_score > short_score else -1
    if expected_return > 0:
        return 1
    if expected_return < 0:
        return -1
    return 0


def _sort_feature_log_by_time(path: Path) -> None:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict) and row.get("timestamp"):
            rows.append(row)
    rows.sort(key=lambda row: str(row["timestamp"]))
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _strict_validation_gate(walk_forward: dict) -> tuple[bool, dict]:
    folds = [fold for fold in walk_forward.get("folds", []) if fold.get("status") != "skipped"]
    average = walk_forward.get("average", {}) or {}
    recent = folds[-2:]
    requirements = {
        "min_samples": 1500,
        "min_average_precision": 0.52,
        "min_average_profit_factor": 1.10,
        "min_recent_fold_precision": 0.45,
        "min_recent_fold_profit_factor": 1.00,
        "min_recent_fold_allowed_trades": 5,
        "required_recent_folds": 2,
    }
    recent_checks = []
    for fold in recent:
        allowed_trades = int(fold.get("tp", 0) or 0) + int(fold.get("fp", 0) or 0)
        recent_checks.append(
            {
                "fold": fold.get("fold"),
                "precision": float(fold.get("precision", 0.0) or 0.0),
                "profit_factor": float(fold.get("profit_factor", 0.0) or 0.0),
                "allowed_trades": allowed_trades,
                "passed": (
                    float(fold.get("precision", 0.0) or 0.0) >= requirements["min_recent_fold_precision"]
                    and float(fold.get("profit_factor", 0.0) or 0.0) >= requirements["min_recent_fold_profit_factor"]
                    and allowed_trades >= requirements["min_recent_fold_allowed_trades"]
                ),
            }
        )
    passed = (
        int(walk_forward.get("samples", 0) or 0) >= requirements["min_samples"]
        and float(average.get("precision", 0.0) or 0.0) >= requirements["min_average_precision"]
        and float(average.get("profit_factor", 0.0) or 0.0) >= requirements["min_average_profit_factor"]
        and len(recent_checks) == requirements["required_recent_folds"]
        and all(item["passed"] for item in recent_checks)
    )
    return passed, {"passed": passed, "requirements": requirements, "recent_folds": recent_checks}


def main():
    print("=" * 100)
    print(" STRICT CAUSAL MULTI-ASSET LOCAL-EDGE MARKET MODEL TRAINING")
    print("=" * 100)

    features_file = Path("data/features.jsonl")
    features_file.parent.mkdir(parents=True, exist_ok=True)
    features_file.write_text("", encoding="utf-8")

    equity_tracker = EquityTracker(dd_max=0.20, initial_equity=10000.0)
    quant_params = QuantParams(position_r_max=0.02)
    session_engine = SessionEngine()

    total_added = 0
    source_counts: dict[str, int] = {}

    print(f"\n[1/3] Downloading causal histories for {SYMBOLS}...")
    for symbol in SYMBOLS:
        history = _download_symbol_history(symbol)
        m15_candles = history["M15"]
        print(
            f"      {symbol}: D1={len(history['D1'])}, H4={len(history['H4'])}, "
            f"H1={len(history['H1'])}, M30={len(history['M30'])}, "
            f"M15={len(m15_candles)}, M5={len(history['M5'])}"
        )
        if len(m15_candles) <= 50 + LABEL_HORIZON_M15_BARS:
            print(f"      {symbol}: skipped (insufficient M15 history)")
            continue

        # Normalization state is market-local. Reusing one rolling z-score buffer
        # across symbols would let the previous symbol change the next one's inputs.
        feature_extractor = FeatureExtractor(window=100)
        step = max(1, (len(m15_candles) - 50 - LABEL_HORIZON_M15_BARS) // MAX_TRAINING_ROWS_PER_SYMBOL)
        symbol_added = 0
        for index in range(50, len(m15_candles) - LABEL_HORIZON_M15_BARS, step):
            current = m15_candles[index]
            decision_time = current.timestamp + timedelta(minutes=15)
            causal = _causal_history(history, decision_time)
            if causal is None:
                continue

            absolute_spread = _absolute_spread(symbol)
            current_price = float(current.close)
            if current_price <= 0:
                continue
            transaction_cost_ratio = max(absolute_spread / current_price, 1e-8)
            causal.spread = absolute_spread

            dummy_strategy = SimpleNamespace(
                is_trade=True,
                entry_price=current_price,
                stop_loss=current_price * 0.995,
                take_profit=current_price * 1.010,
                metadata={"volatility_state": None, "h4_context": None},
            )

            from src.live_trade_loop import _extract_features_from_strategy

            snapshot, _expected_return, _return_std = _extract_features_from_strategy(
                live_input=causal,
                strategy_result=dummy_strategy,
                feature_extractor=feature_extractor,
                spread=absolute_spread,
            )

            session_score = session_engine.compute_session_score(current.timestamp)
            recent = causal.m15_candles[-20:]
            recent_returns = [
                (later.close - earlier.close) / max(earlier.close, 1e-9)
                for earlier, later in zip(recent[:-1], recent[1:])
            ] or [0.0]

            quant_decision = evaluate_master_equation(
                features=snapshot,
                params=quant_params,
                equity=10000.0,
                drawdown_ratio=equity_tracker.drawdown_ratio,
                recent_returns=recent_returns,
                transaction_cost=transaction_cost_ratio,
                win_rate=0.5,
                avg_win=1.0,
                avg_loss=1.0,
                session_score=session_score,
                dxy_trend=0.0,
            )

            direction = _candidate_direction(quant_decision, float(snapshot.expected_return))
            if direction not in {-1, 1}:
                continue

            outcome = realized_directional_outcome(
                m15_candles,
                index,
                direction=direction,
                horizon_bars=LABEL_HORIZON_M15_BARS,
                transaction_cost_ratio=transaction_cost_ratio,
            )
            if outcome is None:
                continue

            metadata = quant_decision.metadata
            append_enriched_snapshot_to_file(
                snapshot,
                features_file,
                quant_is_trade=bool(quant_decision.is_trade),
                quant_action=direction,
                omega_t=quant_decision.omega_t,
                kelly_fraction=float(metadata.get("kelly_fraction", 0.0)),
                ce_score_trade=float((metadata.get("ce_scores") or {}).get(direction, 0.0)),
                ce_score_flat=float((metadata.get("ce_scores") or {}).get(0, 0.0)),
                sharpe_signal=quant_decision.sharpe_signal,
                drawdown_dampener=quant_decision.drawdown_dampener,
                lot_multiplier=quant_decision.lot_multiplier,
                transaction_cost=transaction_cost_ratio,
                win_rate=0.5,
                avg_win=1.0,
                avg_loss=1.0,
                session_score=session_score,
                dxy_trend=0.0,
                drawdown_ratio=0.0,
                spread=absolute_spread,
                atr=average_true_range_proxy(causal.m15_candles),
                lot_requested=0.01,
                commission_per_lot=6.0,
                current_equity=10000.0,
                outcome_label=outcome.label,
                outcome_pnl=outcome.net_return,
                outcome_r_multiple=0.0,
            )
            symbol_added += 1
            total_added += 1

        source_counts[symbol] = symbol_added
        print(f"      {symbol}: {symbol_added} realized direction-aware candidate rows")

    _sort_feature_log_by_time(features_file)
    print(f"\n[2/3] Built {total_added} causal rows, globally chronological: {source_counts}")
    if total_added < 1500:
        raise RuntimeError(f"insufficient_causal_training_rows:{total_added}")

    print("\n[3/3] Running fixed-threshold untouched walk-forward evaluation...")
    from src.strategy.intelligence_pipeline import promote_candidate, run_pipeline

    args = SimpleNamespace(
        features=str(features_file),
        diagnostics="data/live_diagnostics.jsonl",
        paper_trades="data/paper_trades.jsonl",
        candles=None,
        candidate_model="data/models/local_edge_model.strict_candidate.npz",
        live_model="data/models/local_edge_model.npz",
        report="data/models/training_report_strict.json",
        horizon_bars=LABEL_HORIZON_M15_BARS,
        folds=5,
        epochs=800,
        learning_rate=0.025,
        hidden_size=32,
        num_layers=32,
        threshold=0.55,
        fixed_threshold=True,
        auto_threshold=False,
        min_abs_return=0.0,
        promote=False,
        min_promotion_samples=1500,
        min_promotion_precision=0.52,
        min_promotion_profit_factor=1.10,
    )

    report = run_pipeline(args)
    walk_forward = report.get("walk_forward") or {}
    strict_passed, strict_report = _strict_validation_gate(walk_forward)
    report["strict_validation"] = strict_report

    promotion = {
        "promoted": False,
        "reason": "strict_validation_failed",
        "samples": int(walk_forward.get("samples", 0) or 0),
    }
    if strict_passed:
        promotion = promote_candidate(
            candidate_path=args.candidate_model,
            live_model_path=args.live_model,
            walk_forward=walk_forward,
            min_samples=args.min_promotion_samples,
            min_precision=args.min_promotion_precision,
            min_profit_factor=args.min_promotion_profit_factor,
        )
    report["promotion"] = promotion
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    average = walk_forward.get("average") or {}
    print("\n" + "=" * 100)
    print(" STRICT CAUSAL MULTI-ASSET TRAINING COMPLETE")
    print(f" Total Dataset Rows: {report.get('rows')}")
    print(f" Strict Validation Passed: {strict_passed}")
    print(f" Candidate Promoted: {promotion.get('promoted')}")
    print(f" Walk-Forward Precision: {float(average.get('precision', 0.0) or 0.0):.4f}")
    print(f" Walk-Forward Profit Factor: {float(average.get('profit_factor', 0.0) or 0.0):.4f}")
    print(f" Live Model Path: {promotion.get('live_model_path')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
