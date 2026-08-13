from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import sleep

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_settings
from src.mt4_bridge import Mt4BridgeModule
from src.strategy.intelligence_pipeline import promote_candidate, walk_forward_report
from src.strategy.local_edge_model import FEATURE_NAMES, save_model, train_local_edge_model


TIMEFRAME_FILES = {
    "M1": "fx_bridge_rates_{symbol}_M1.csv",
    "M5": "fx_bridge_rates_{symbol}_M5.csv",
    "M15": "fx_bridge_rates_{symbol}_M15.csv",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train local edge only from live broker candles written by MT4.")
    parser.add_argument("--symbol", default="")
    parser.add_argument("--common-files-dir", default="")
    parser.add_argument("--timeframe", default="M1", choices=sorted(TIMEFRAME_FILES))
    parser.add_argument("--horizon-bars", type=int, default=8)
    parser.add_argument("--min-move", type=float, default=0.02)
    parser.add_argument("--candidate-model", default="data/models/local_edge_model.live_market_candidate.npz")
    parser.add_argument("--live-model", default="data/models/local_edge_model.npz")
    parser.add_argument("--report", default="data/models/live_market_edge.report.json")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--min-promotion-samples", type=int, default=300)
    parser.add_argument("--min-promotion-precision", type=float, default=0.55)
    parser.add_argument("--min-promotion-profit-factor", type=float, default=1.10)
    args = parser.parse_args(argv)

    settings = load_settings()
    symbol = args.symbol or settings.mt4_chart_symbol or settings.trading_symbol
    common_dir = Path(args.common_files_dir) if args.common_files_dir else _default_common_files_dir()
    safe_symbol = Mt4BridgeModule(common_dir)._safe_symbol(symbol)
    candles = _read_bridge_rates(common_dir / TIMEFRAME_FILES[args.timeframe].format(symbol=safe_symbol))
    rows = build_rows_from_candles(candles, horizon_bars=args.horizon_bars, min_move=args.min_move)

    report = {
        "symbol": symbol,
        "timeframe": args.timeframe,
        "common_files_dir": str(common_dir),
        "candles": len(candles),
        "rows": len(rows),
        "source": "mt4_live_broker_candles_only",
    }
    if len(rows) < 30:
        report["status"] = "insufficient_live_market_candles"
        _write_report(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    walk_forward = walk_forward_report(
        rows,
        folds=4,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        threshold=args.threshold,
        auto_threshold=True,
    )
    candidate_threshold = float(walk_forward.get("recommended_threshold", args.threshold) or args.threshold)
    x = np.vstack([_vector(row) for row in rows])
    y = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    model, candidate_report = train_local_edge_model(
        x,
        y,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        threshold=candidate_threshold,
    )
    save_model(model, args.candidate_model, candidate_report)

    promotion = None
    if args.promote:
        promotion = promote_candidate(
            candidate_path=args.candidate_model,
            live_model_path=args.live_model,
            walk_forward=walk_forward,
            min_samples=args.min_promotion_samples,
            min_precision=args.min_promotion_precision,
            min_profit_factor=args.min_promotion_profit_factor,
        )

    report.update(
        {
            "status": "ok",
            "positive_rate": sum(float(row["label"]) for row in rows) / len(rows),
            "walk_forward": walk_forward,
            "candidate": candidate_report,
            "promotion": promotion,
        }
    )
    _write_report(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_rows_from_candles(candles: list[dict], *, horizon_bars: int, min_move: float) -> list[dict]:
    rows: list[dict] = []
    horizon = max(1, int(horizon_bars))
    for idx in range(30, max(30, len(candles) - horizon)):
        window = candles[max(0, idx - 30) : idx + 1]
        current = candles[idx]
        future = candles[idx + horizon]
        close = float(current["close"])
        if close <= 0.0:
            continue
        realized_move = float(future["close"]) - close
        if abs(realized_move) < float(min_move):
            continue
        row = _features_from_window(window)
        row["label"] = 1.0 if realized_move > 0.0 else 0.0
        row["realized_return"] = realized_move / close
        row["timestamp"] = current["time"]
        rows.append(row)
    return rows


def _features_from_window(window: list[dict]) -> dict:
    closes = np.asarray([float(row["close"]) for row in window], dtype=np.float64)
    highs = np.asarray([float(row["high"]) for row in window], dtype=np.float64)
    lows = np.asarray([float(row["low"]) for row in window], dtype=np.float64)
    volumes = np.asarray([float(row.get("tick_volume", 0.0) or 0.0) for row in window], dtype=np.float64)
    returns = np.diff(closes) / np.maximum(closes[:-1], 1e-9)
    short_ret = (closes[-1] - closes[-4]) / max(closes[-4], 1e-9) if len(closes) >= 4 else 0.0
    long_ret = (closes[-1] - closes[0]) / max(closes[0], 1e-9)
    ranges = (highs - lows) / np.maximum(closes, 1e-9)
    atr = float(np.mean(ranges[-14:])) if len(ranges) else 0.0
    vol = float(np.std(returns[-14:])) if len(returns) else 0.0
    volume_z = _zscore(volumes[-1], volumes)
    momentum_z = _zscore(short_ret, returns[-14:] if len(returns) else np.asarray([0.0]))
    trend_z = _zscore(long_ret, returns if len(returns) else np.asarray([0.0]))
    range_position = (closes[-1] - lows[-20:].min()) / max(highs[-20:].max() - lows[-20:].min(), 1e-9)

    row = {name: 0.0 for name in FEATURE_NAMES}
    row.update(
        {
            "momentum_z": momentum_z,
            "trend_z": trend_z,
            "volume_z": volume_z,
            "volatility_risk_z": _zscore(vol, ranges[-20:]),
            "entry_distance_z": _zscore(range_position, np.linspace(0.0, 1.0, num=20)),
            "spread_danger_z": 0.0,
            "m5_score_z": np.sign(short_ret),
            "m15_score_z": np.sign(long_ret),
            "context_score_z": np.sign(long_ret) * min(abs(long_ret) / max(atr, 1e-9), 3.0),
            "structure_score_z": (range_position - 0.5) * 2.0,
            "statistical_score_z": _zscore(returns[-1] if len(returns) else 0.0, returns[-20:] if len(returns) else np.asarray([0.0])),
            "omega_t": max(0.0, min(2.0, abs(long_ret) / max(vol, 1e-9))) if vol > 0 else 0.0,
            "kelly_fraction": max(0.0, min(0.25, abs(short_ret) / max(vol * 10.0, 1e-9))) if vol > 0 else 0.0,
            "session_score": 1.0,
            "spread": 0.0,
            "atr": atr,
        }
    )
    return row


def _vector(row: dict) -> np.ndarray:
    return np.asarray([float(row.get(name, 0.0) or 0.0) for name in FEATURE_NAMES], dtype=np.float64)


def _zscore(value: float, sample) -> float:
    arr = np.asarray(sample, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    std = float(arr.std())
    if std <= 1e-12:
        return 0.0
    return float((float(value) - float(arr.mean())) / std)


def _read_bridge_rates(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    text = ""
    for attempt in range(10):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                break
        except PermissionError:
            if attempt == 9:
                raise
        sleep(0.1)
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            rows.append(
                {
                    "time": int(float(parts[0])),
                    "open": float(parts[1]),
                    "high": float(parts[2]),
                    "low": float(parts[3]),
                    "close": float(parts[4]),
                    "tick_volume": float(parts[5] or 0.0),
                }
            )
        except ValueError:
            continue
    return sorted(rows, key=lambda row: row["time"])


def _default_common_files_dir() -> Path:
    import os

    return Path(os.environ.get("APPDATA", "")) / "MetaQuotes" / "Terminal" / "Common" / "Files"


def _write_report(path: str | Path, report: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
