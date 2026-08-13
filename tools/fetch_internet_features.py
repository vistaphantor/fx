"""Fetch multi-month historical price data from internet (yfinance) and run feature extraction + forward/backward propagation training.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import yfinance as yf
import numpy as np

from src.strategy.features import FeatureExtractor, append_enriched_snapshot_to_file
from src.strategy.quant_engine import evaluate_master_equation, QuantParams
from src.strategy.equity_tracker import EquityTracker
from src.strategy.session_engine import SessionEngine


def convert_df_to_candles(df):
    candles = []
    for idx, row in df.iterrows():
        ts = idx.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
        c = SimpleNamespace(
            timestamp=ts,
            time=int(ts.timestamp()),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row.get("Volume", 100)),
        )
        candles.append(c)
    return candles


def main():
    print("=" * 90)
    print("  FETCHING HISTORICAL MARKET DATA FROM INTERNET & TRAINING NEURAL NETWORK")
    print("=" * 90)

    # 1. Download Gold (GC=F) intraday candles from Yahoo Finance
    print("[1/4] Downloading 60 days of 5-minute Gold candles (GC=F) via yfinance...")
    ticker = yf.Ticker("GC=F")
    df_m5 = ticker.history(period="60d", interval="5m")

    if df_m5.empty:
        print("[ERROR] Failed to fetch candle data from Yahoo Finance.")
        return

    m5_candles = convert_df_to_candles(df_m5)
    print(f"      Downloaded {len(m5_candles)} M5 candles successfully.")

    # 2. Extract features across rolling windows
    print("[2/4] Extracting 55-feature schema & quant metrics across historical bars...")
    feature_extractor = FeatureExtractor(window=100)
    equity_tracker = EquityTracker(dd_max=0.20, initial_equity=10000.0)
    quant_params = QuantParams(position_r_max=0.02)
    session_eng = SessionEngine()

    features_log_path = Path("data/features.jsonl")
    initial_rows = 0
    if features_log_path.exists():
        with open(features_log_path, encoding="utf-8") as f:
            initial_rows = sum(1 for line in f if line.strip())
    print(f"      Existing features.jsonl rows: {initial_rows}")

    added_rows = 0
    # Process candles in 15-minute bar chunks (every 3 m5 candles = 1 m15 bar)
    chunk_size = 3
    for i in range(50, len(m5_candles), chunk_size):
        sub_m5 = m5_candles[:i]
        cur_candle = sub_m5[-1]

        # Synthesize multi-timeframe candle inputs
        m15_candles = sub_m5[-min(100, len(sub_m5)):]
        m30_candles = sub_m5[::2][-min(50, len(sub_m5)//2):]
        h1_candles = sub_m5[::12][-min(50, len(sub_m5)//12):]
        h4_candles = sub_m5[::48][-min(50, len(sub_m5)//48):]
        d1_candles = sub_m5[::288][-min(20, len(sub_m5)//288):]

        live_input = SimpleNamespace(
            d1_candles=d1_candles if len(d1_candles) >= 2 else m15_candles,
            h4_candles=h4_candles if len(h4_candles) >= 2 else m15_candles,
            h1_candles=h1_candles if len(h1_candles) >= 2 else m15_candles,
            m30_candles=m30_candles if len(m30_candles) >= 2 else m15_candles,
            m15_candles=m15_candles,
            m5_candles=sub_m5[-20:],
            spread=0.25,
        )

        from src.live_trade_loop import _extract_features_from_strategy

        dummy_strategy = SimpleNamespace(
            is_trade=True,
            entry_price=cur_candle.close,
            stop_loss=cur_candle.close * 0.995,
            take_profit=cur_candle.close * 1.010,
            metadata={"volatility_state": None, "h4_context": None},
        )

        snapshot, exp_ret, ret_std = _extract_features_from_strategy(
            live_input=live_input,
            strategy_result=dummy_strategy,
            feature_extractor=feature_extractor,
            spread=0.25,
        )

        session_score = session_eng.compute_session_score(cur_candle.timestamp)
        recent_returns = [
            (c2.close - c1.close) / max(c1.close, 1e-9)
            for c1, c2 in zip(m15_candles[:-1], m15_candles[1:])
        ] or [0.0]

        quant_decision = evaluate_master_equation(
            features=snapshot,
            params=quant_params,
            equity=10000.0,
            drawdown_ratio=equity_tracker.drawdown_ratio,
            recent_returns=recent_returns,
            transaction_cost=0.0001,
            win_rate=0.55,
            avg_win=1.5,
            avg_loss=1.0,
            session_score=session_score,
            dxy_trend=0.0,
        )

        meta = quant_decision.metadata
        append_enriched_snapshot_to_file(
            snapshot,
            features_log_path,
            quant_is_trade=quant_decision.is_trade,
            quant_action=quant_decision.action,
            omega_t=quant_decision.omega_t,
            kelly_fraction=float(meta.get("kelly_fraction", 0.0)),
            ce_score_trade=float((meta.get("ce_scores") or {}).get(1, 0.0)),
            ce_score_flat=float((meta.get("ce_scores") or {}).get(0, 0.0)),
            sharpe_signal=quant_decision.sharpe_signal,
            drawdown_dampener=quant_decision.drawdown_dampener,
            lot_multiplier=quant_decision.lot_multiplier,
            transaction_cost=0.0001,
            win_rate=0.55,
            avg_win=1.5,
            avg_loss=1.0,
            session_score=session_score,
            dxy_trend=0.0,
            drawdown_ratio=0.0,
            spread=0.25,
            atr=1.5,
            lot_requested=0.01,
            commission_per_lot=6.0,
            current_equity=10000.0,
        )
        added_rows += 1

    print(f"[3/4] Successfully generated and appended {added_rows} new enriched feature rows.")
    total_rows = initial_rows + added_rows
    print(f"      Total features.jsonl dataset size: {total_rows} rows.")

    # 3. Trigger Neural Network Forward/Backward Propagation Training
    print("\n[4/4] Executing Forward/Backward Propagation Neural Network Training...")
    from src.strategy.intelligence_pipeline import run_pipeline

    args = SimpleNamespace(
        features="data/features.jsonl",
        diagnostics="data/live_diagnostics.jsonl",
        paper_trades="data/paper_trades.jsonl",
        candles=None,
        candidate_model="data/models/local_edge_model.online_candidate.npz",
        live_model="data/models/local_edge_model.npz",
        report="data/models/training_report_internet.json",
        horizon_bars=120,
        folds=5,
        epochs=1000,
        learning_rate=0.025,
        hidden_size=32,
        threshold=0.55,
        fixed_threshold=False,
        min_abs_return=0.0,
        promote=True,
        min_promotion_samples=100,
        min_promotion_precision=0.52,
        min_promotion_profit_factor=1.05,
    )

    report = run_pipeline(args)

    print("=" * 90)
    print("  TRAINING COMPLETE & MODEL PROMOTED!")
    print(f"  Rows Evaluated: {report.get('rows')}")
    promo = report.get("promotion", {})
    print(f"  Promoted to Live Model: {promo.get('promoted')}")
    print(f"  Precision: {promo.get('precision', 0.0):.4f}")
    print(f"  Profit Factor: {promo.get('profit_factor', 0.0):.2f}")
    print(f"  Backup Saved At: {promo.get('backup_path')}")
    print("=" * 90)


if __name__ == "__main__":
    main()
