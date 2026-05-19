from __future__ import annotations

from src.strategy.math_engine import resolve_quant_metrics
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from time import sleep

from src.market_data import fetch_candles
from src.strategy.breakout import BreakoutDirection
from src.strategy.structural import detect_fair_value_gaps
import logging

QUICK_COMMENT_PREFIX = "quick-scalp"
ML_TRAINING_FILE = "ml_training_data.csv"
ML_TRAINING_SCHEMA_VERSION = 2
ML_TRAINING_FIELDS = [
    "schema_version",
    "strategy_mode",
    "candle_timeframe",
    "timestamp",
    "iso_year",
    "iso_week",
    "training_week_id",
    "training_version",
    "day_of_week",
    "hour_utc",
    "symbol",
    "account_balance",
    "account_equity",
    "account_profit",
    "account_currency",
    "positions_count",
    "trade_status",
    "is_tradeable",
    "decision_reason",
    "failed_node",
    "planned_direction",
    "entry_price",
    "stop_loss",
    "take_profit",
    "spread",
    "target_value",
    "target_progress",
    "tick_dir",
    "m1_dir",
    "fib_dir",
    "fib_zone",
    "rsi",
    "sar_dir",
    "mtf_m1_dir",
    "mtf_m5_dir",
    "mtf_m15_dir",
    "mtf_h1_dir",
    "fib_ok",
    "sar_ok",
    "rsi_ok",
    "confluence_score",
    "quant_hurst",
    "quant_reversion",
    "quant_ofi",
    "quant_kelly_lot",
    "quant_z_score",
    "quant_smoothness",
    "m1_open",
    "m1_high",
    "m1_low",
    "m1_close",
    "m1_range",
    "m1_body",
    "m1_body_ratio",
    "tick_count",
    "tick_first_mid",
    "tick_last_mid",
    "tick_net_move",
    "tick_up_moves",
    "tick_down_moves",
    "tick_directional_consistency",
    "label_outcome",
    "label_tp_before_sl",
    "label_max_favorable",
    "label_max_adverse",
    "label_seconds_to_outcome",
]
TRADE_HISTORY_LIMIT = 50
trade_history = [] # Global list to track recent trades


def save_training_snapshot(state: dict):
    """Save a schema-versioned feature snapshot for future ML training."""
    try:
        row = build_training_snapshot_row(state)
        output_path = _training_output_path(ML_TRAINING_FILE)
        file_exists = os.path.isfile(output_path)
        with open(output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ML_TRAINING_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        pass


def build_training_snapshot_row(state: dict) -> dict[str, object]:
    timestamp = _parse_snapshot_timestamp(str(state.get("timestamp", "")))
    iso_year, iso_week, _ = timestamp.isocalendar()
    signals = state.get("signals", {})
    account = state.get("account", {})
    trading = state.get("trading", {})
    quant = signals.get("quant", {}) or {}
    mtf = signals.get("mtf", {}) or {}
    confluence = signals.get("confluence", {}) or {}
    candle_stats = _latest_candle_training_stats(state.get("market_data", {}).get("m1_candles", []))
    tick_stats = _tick_training_stats(state.get("market_data", {}).get("ticks", []))
    confluence_score = sum(1 for key in ("fib_ok", "sar_ok", "rsi_ok") if bool(confluence.get(key, False)))
    training_week_id = f"{iso_year}-W{iso_week:02d}"
    strategy_mode = str(trading.get("strategy_mode", "quick_scalp") or "quick_scalp")
    version_prefix = strategy_mode.replace("_", "-")

    return {
        "schema_version": ML_TRAINING_SCHEMA_VERSION,
        "strategy_mode": strategy_mode,
        "candle_timeframe": trading.get("candle_timeframe", "M1"),
        "timestamp": timestamp.isoformat(),
        "iso_year": iso_year,
        "iso_week": iso_week,
        "training_week_id": training_week_id,
        "training_version": f"{version_prefix}-{training_week_id}",
        "day_of_week": timestamp.weekday(),
        "hour_utc": timestamp.hour,
        "symbol": trading.get("symbol", ""),
        "account_balance": _fmt_float(account.get("balance")),
        "account_equity": _fmt_float(account.get("equity")),
        "account_profit": _fmt_float(account.get("profit")),
        "account_currency": account.get("currency", ""),
        "positions_count": trading.get("positions_count", 0),
        "trade_status": trading.get("status", ""),
        "is_tradeable": int(bool(trading.get("is_tradeable", False))),
        "decision_reason": trading.get("decision_reason", trading.get("status", "")),
        "failed_node": trading.get("failed_node", ""),
        "planned_direction": trading.get("planned_direction", signals.get("tick_dir", "None")),
        "entry_price": _fmt_float(trading.get("entry_price")),
        "stop_loss": _fmt_float(trading.get("stop_loss")),
        "take_profit": _fmt_float(trading.get("take_profit")),
        "spread": _fmt_float(trading.get("spread")),
        "target_value": _fmt_float(trading.get("target_value")),
        "target_progress": _fmt_float(trading.get("target_progress")),
        "tick_dir": signals.get("tick_dir", "None"),
        "m1_dir": signals.get("m1_dir", mtf.get("m1", "None")),
        "fib_dir": signals.get("fib_dir", "None"),
        "fib_zone": signals.get("fib_zone", "None"),
        "rsi": _fmt_float(signals.get("rsi")),
        "sar_dir": signals.get("sar_dir", "None"),
        "mtf_m1_dir": mtf.get("m1", signals.get("m1_dir", "None")),
        "mtf_m5_dir": mtf.get("m5", "None"),
        "mtf_m15_dir": mtf.get("m15", signals.get("fib_dir", "None")),
        "mtf_h1_dir": mtf.get("h1", "None"),
        "fib_ok": int(bool(confluence.get("fib_ok", False))),
        "sar_ok": int(bool(confluence.get("sar_ok", False))),
        "rsi_ok": int(bool(confluence.get("rsi_ok", False))),
        "confluence_score": confluence_score,
        "quant_hurst": _fmt_float(quant.get("hurst")),
        "quant_reversion": _fmt_float(quant.get("reversion")),
        "quant_ofi": _fmt_float(quant.get("ofi")),
        "quant_kelly_lot": _fmt_float(quant.get("kelly_lot")),
        "quant_z_score": _fmt_float(quant.get("z_score")),
        "quant_smoothness": _fmt_float(quant.get("smoothness")),
        **candle_stats,
        **tick_stats,
        "label_outcome": "",
        "label_tp_before_sl": "",
        "label_max_favorable": "",
        "label_max_adverse": "",
        "label_seconds_to_outcome": "",
    }


def _training_output_path(path: str) -> str:
    if not os.path.isfile(path):
        return path
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
    except Exception:
        header = []
    if header == ML_TRAINING_FIELDS:
        return path

    root, ext = os.path.splitext(path)
    versioned_path = f"{root}_schema_v{ML_TRAINING_SCHEMA_VERSION}{ext or '.csv'}"
    return versioned_path


def _parse_snapshot_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _latest_candle_training_stats(candles) -> dict[str, str]:
    latest = candles[-1] if candles else {}
    open_price = _dict_float(latest, "open")
    high = _dict_float(latest, "high")
    low = _dict_float(latest, "low")
    close = _dict_float(latest, "close")
    candle_range = max(high - low, 0.0)
    body = abs(close - open_price)
    body_ratio = body / candle_range if candle_range > 0.0 else 0.0
    return {
        "m1_open": _fmt_float(open_price),
        "m1_high": _fmt_float(high),
        "m1_low": _fmt_float(low),
        "m1_close": _fmt_float(close),
        "m1_range": _fmt_float(candle_range),
        "m1_body": _fmt_float(body),
        "m1_body_ratio": _fmt_float(body_ratio),
    }


def _tick_training_stats(ticks) -> dict[str, str | int]:
    mids = []
    for tick in ticks or []:
        bid = _dict_float(tick, "bid")
        ask = _dict_float(tick, "ask")
        if bid > 0.0 and ask > 0.0:
            mids.append((bid + ask) / 2.0)
        else:
            mids.append(max(bid, ask))
    mids = [mid for mid in mids if mid > 0.0]
    up_moves = 0
    down_moves = 0
    for previous_mid, current_mid in zip(mids, mids[1:]):
        if current_mid > previous_mid:
            up_moves += 1
        elif current_mid < previous_mid:
            down_moves += 1
    total_moves = up_moves + down_moves
    consistency = max(up_moves, down_moves) / total_moves if total_moves else 0.0
    first_mid = mids[0] if mids else 0.0
    last_mid = mids[-1] if mids else 0.0
    return {
        "tick_count": len(mids),
        "tick_first_mid": _fmt_float(first_mid),
        "tick_last_mid": _fmt_float(last_mid),
        "tick_net_move": _fmt_float(last_mid - first_mid if mids else 0.0),
        "tick_up_moves": up_moves,
        "tick_down_moves": down_moves,
        "tick_directional_consistency": _fmt_float(consistency),
    }


def _dict_float(mapping, key: str) -> float:
    try:
        if isinstance(mapping, dict):
            return float(mapping.get(key, 0.0) or 0.0)
        return float(getattr(mapping, key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _fmt_float(value) -> str:
    try:
        return f"{float(value or 0.0):.6f}"
    except (TypeError, ValueError):
        return "0.000000"


def save_bot_state(state: dict):
    """Save bot state to a JSON file for the dashboard."""
    try:
        temp_file = "bot_state.json.tmp"
        with open(temp_file, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(temp_file, "bot_state.json")
    except Exception:
        pass


def fetch_m1_candles(mt5_module, symbol: str, count: int = 2):
    return fetch_candles(mt5_module, symbol, "TIMEFRAME_M1", count, minimum=1)


def fetch_m15_candles(mt5_module, symbol: str, count: int = 50):
    return fetch_candles(mt5_module, symbol, "TIMEFRAME_M15", count, minimum=10)


def fetch_h1_candles(mt5_module, symbol: str, count: int = 50):
    return fetch_candles(mt5_module, symbol, "TIMEFRAME_H1", count, minimum=10)


def fetch_m5_candles(mt5_module, symbol: str, count: int = 50):
    return fetch_candles(mt5_module, symbol, "TIMEFRAME_M5", count, minimum=10)


def resolve_m1_direction(candles) -> BreakoutDirection | None:
    if not candles:
        return None
    latest = candles[-1]
    open_price = float(getattr(latest, "open"))
    close_price = float(getattr(latest, "close"))
    if close_price > open_price:
        return BreakoutDirection.BULLISH
    if close_price < open_price:
        return BreakoutDirection.BEARISH
    return None


def fetch_recent_ticks(mt5_module, symbol: str, count: int = 100):
    # Try multiple methods to get ticks (some servers prefer different APIs)
    flags_all = getattr(mt5_module, "COPY_TICKS_ALL", 0)
    flags_info = getattr(mt5_module, "COPY_TICKS_INFO", 1) # Fallback to bid/ask only
    
    # Method 1: copy_ticks_from_pos (most efficient)
    if hasattr(mt5_module, "copy_ticks_from_pos"):
        ticks = mt5_module.copy_ticks_from_pos(symbol, 0, count, flags_all)
        if ticks is not None and len(ticks) > 0:
            return list(ticks)
        # Try info flags if all flags failed
        ticks = mt5_module.copy_ticks_from_pos(symbol, 0, count, flags_info)
        if ticks is not None and len(ticks) > 0:
            return list(ticks)
            
    # Method 2: copy_ticks_from (timestamp based)
    if hasattr(mt5_module, "copy_ticks_from"):
        from datetime import timedelta
        # Get last 10 minutes of ticks
        start_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        ticks = mt5_module.copy_ticks_from(symbol, start_time, count, flags_all)
        if ticks is not None and len(ticks) > 0:
            return list(ticks)
            
    return []


def resolve_tick_direction(ticks, *, point: float = 0.0):
    if ticks is None:
        return QuickTickGuidance(None, "tick_api_unavailable", 0, 0.0, 0, 0)
    if len(ticks) < 3:
        return QuickTickGuidance(None, "insufficient_ticks", len(ticks), 0.0, 0, 0)

    mids = [_tick_mid_price(tick) for tick in ticks]
    mids = [mid for mid in mids if mid > 0.0]
    if len(mids) < 3:
        return QuickTickGuidance(None, "invalid_ticks", len(ticks), 0.0, 0, 0)

    up_moves = 0
    down_moves = 0
    for previous_mid, current_mid in zip(mids, mids[1:]):
        if current_mid > previous_mid:
            up_moves += 1
        elif current_mid < previous_mid:
            down_moves += 1

    net_move = mids[-1] - mids[0]
    # Tightened minimum move: require at least 1 pip net move (not micro-noise).
    # Previously 0.000005 * price on XAUUSD@2400 = 0.012 pts — accepted random noise.
    # Now: 1 pip = max(point*10, price*0.00004) which is ~0.10 for XAUUSD, 0.001 for forex.
    pip = max(float(point or 0.0) * 10.0, abs(mids[-1]) * 0.00004)
    minimum_move = max(pip, float(point or 0.0) * 2.0)
    total_moves = up_moves + down_moves
    # Require at least 60% directional consistency in addition to net move
    if total_moves > 0 and net_move >= minimum_move and up_moves / total_moves >= 0.60:
        return QuickTickGuidance(BreakoutDirection.BULLISH, "tick_momentum", len(mids), net_move, up_moves, down_moves)
    if total_moves > 0 and net_move <= -minimum_move and down_moves / total_moves >= 0.60:
        return QuickTickGuidance(BreakoutDirection.BEARISH, "tick_momentum", len(mids), net_move, up_moves, down_moves)
    return QuickTickGuidance(None, "tick_chop", len(mids), net_move, up_moves, down_moves)


def _tick_mid_price(tick) -> float:
    bid = _tick_value(tick, "bid")
    ask = _tick_value(tick, "ask")
    if bid > 0.0 and ask > 0.0:
        return (bid + ask) / 2.0
    return max(bid, ask)


def _tick_value(tick, name: str) -> float:
    if hasattr(tick, name):
        return float(getattr(tick, name) or 0.0)
    try:
        return float(tick[name] or 0.0)
    except (KeyError, IndexError, TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class QuickFibonacciGuidance:
    direction: BreakoutDirection | None
    zone: str
    current_price: float
    swing_low: float
    swing_high: float

    def allows(self, direction: BreakoutDirection) -> bool:
        return self.direction is direction and self.zone in {
            "in_market_mover",
            "towards_market_mover",
            "golden_zone",
        }


@dataclass(frozen=True)
class QuickIndicatorGuidance:
    rsi: float
    sar: float
    sar_direction: BreakoutDirection | None

    def allows(self, direction: BreakoutDirection) -> bool:
        if self.sar_direction is not direction:
            return False
        if direction is BreakoutDirection.BULLISH:
            return self.rsi < 70.0
        return self.rsi > 30.0


@dataclass(frozen=True)
class QuickFvgGuidance:
    direction: BreakoutDirection | None
    top: float
    bottom: float
    bars_since: int
    current_price: float
    reason: str

    def allows(self, direction: BreakoutDirection) -> bool:
        return self.direction is direction and self.reason == "matching_fvg"


@dataclass(frozen=True)
class QuickTickGuidance:
    direction: BreakoutDirection | None
    reason: str
    tick_count: int
    net_move: float
    up_moves: int
    down_moves: int


@dataclass(frozen=True)
class QuickSpreadState:
    allowed: bool
    reason: str
    spread: float
    spread_pips: float
    max_spread_pips: float


@dataclass(frozen=True)
class QuickGridPermission:
    allowed: bool
    reason: str
    max_new_entries: int
    min_spacing: float
    current_price: float


@dataclass(frozen=True)
class QuickGuidanceDecision:
    allowed: bool
    reason: str
    max_new_entries_override: int | None
    fib_allows: bool
    sar_allows: bool
    rsi_allows: bool


@dataclass(frozen=True)
class QuickEntryQualityGate:
    allowed: bool
    reason: str
    body_ratio: float
    max_fvg_bars: int
    price_distance: float
    max_price_distance: float


def resolve_quick_fibonacci_guidance(candles) -> QuickFibonacciGuidance:
    if len(candles) < 2:
        return QuickFibonacciGuidance(None, "unavailable", 0.0, 0.0, 0.0)

    swing_low_index, swing_low_candle = min(enumerate(candles), key=lambda item: float(getattr(item[1], "low")))
    swing_high_index, swing_high_candle = max(enumerate(candles), key=lambda item: float(getattr(item[1], "high")))
    swing_low = float(getattr(swing_low_candle, "low"))
    swing_high = float(getattr(swing_high_candle, "high"))
    current_price = float(getattr(candles[-1], "close"))
    impulse_range = swing_high - swing_low
    if impulse_range <= 0:
        return QuickFibonacciGuidance(None, "flat", current_price, swing_low, swing_high)

    if swing_low_index < swing_high_index:
        zone = _classify_bullish_fibonacci_zone(current_price, swing_low, swing_high)
        return QuickFibonacciGuidance(BreakoutDirection.BULLISH, zone, current_price, swing_low, swing_high)

    zone = _classify_bearish_fibonacci_zone(current_price, swing_low, swing_high)
    return QuickFibonacciGuidance(BreakoutDirection.BEARISH, zone, current_price, swing_low, swing_high)


def resolve_quick_fvg_guidance(
    candles,
    *,
    direction: BreakoutDirection,
    current_price: float,
    max_bars_since: int = 20,
) -> QuickFvgGuidance:
    direction_value = 1 if direction is BreakoutDirection.BULLISH else -1
    matching_gaps = [
        gap
        for gap in detect_fair_value_gaps(candles, lookback=max_bars_since)
        if int(getattr(gap, "direction", 0)) == direction_value
        and int(getattr(gap, "bars_since", max_bars_since + 1)) <= int(max_bars_since)
        and _fvg_is_on_chase_side(gap, direction, current_price)
    ]
    if not matching_gaps:
        return QuickFvgGuidance(None, 0.0, 0.0, 0, float(current_price), "no_matching_fvg")

    nearest_gap = min(
        matching_gaps,
        key=lambda gap: abs(float(current_price) - ((float(gap.top) + float(gap.bottom)) / 2.0)),
    )
    return QuickFvgGuidance(
        direction,
        float(nearest_gap.top),
        float(nearest_gap.bottom),
        int(getattr(nearest_gap, "bars_since", 0)),
        float(current_price),
        "matching_fvg",
    )


def _fvg_is_on_chase_side(fvg, direction: BreakoutDirection, current_price: float) -> bool:
    top = float(getattr(fvg, "top"))
    bottom = float(getattr(fvg, "bottom"))
    if direction is BreakoutDirection.BULLISH:
        return float(current_price) >= bottom
    return float(current_price) <= top


def _classify_bullish_fibonacci_zone(current_price: float, swing_low: float, swing_high: float) -> str:
    impulse_range = swing_high - swing_low
    retrace_382 = swing_high - (impulse_range * 0.382)
    retrace_618 = swing_high - (impulse_range * 0.618)
    retrace_786 = swing_high - (impulse_range * 0.786)
    if current_price >= retrace_382:
        return "in_market_mover"
    if current_price >= retrace_618:
        return "towards_market_mover"
    if current_price >= retrace_786:
        return "golden_zone"
    return "outside"


def _classify_bearish_fibonacci_zone(current_price: float, swing_low: float, swing_high: float) -> str:
    impulse_range = swing_high - swing_low
    retrace_382 = swing_low + (impulse_range * 0.382)
    retrace_618 = swing_low + (impulse_range * 0.618)
    retrace_786 = swing_low + (impulse_range * 0.786)
    if current_price <= retrace_382:
        return "in_market_mover"
    if current_price <= retrace_618:
        return "towards_market_mover"
    if current_price <= retrace_786:
        return "golden_zone"
    return "outside"


def resolve_quick_indicator_guidance(candles) -> QuickIndicatorGuidance:
    if len(candles) < 15:
        return QuickIndicatorGuidance(50.0, 0.0, None)

    rsi = _calculate_rsi([float(getattr(candle, "close")) for candle in candles], period=14)
    sar = _calculate_parabolic_sar(candles)
    latest_close = float(getattr(candles[-1], "close"))
    if latest_close > sar:
        sar_direction = BreakoutDirection.BULLISH
    elif latest_close < sar:
        sar_direction = BreakoutDirection.BEARISH
    else:
        sar_direction = None
    return QuickIndicatorGuidance(rsi, sar, sar_direction)


def resolve_quick_guidance_decision(
    *,
    direction: BreakoutDirection,
    fibonacci_guidance: QuickFibonacciGuidance,
    indicator_guidance: QuickIndicatorGuidance,
) -> QuickGuidanceDecision:
    fib_allows = fibonacci_guidance.allows(direction)
    sar_allows = indicator_guidance.sar_direction is direction
    rsi_allows = _rsi_allows_direction(indicator_guidance.rsi, direction)
    if _rsi_is_extreme_exhaustion(indicator_guidance.rsi, direction):
        return QuickGuidanceDecision(False, "indicator_filter", None, fib_allows, sar_allows, rsi_allows)
    if not rsi_allows:
        return QuickGuidanceDecision(True, "structure_override", 1, fib_allows, sar_allows, rsi_allows)
    if fib_allows and sar_allows:
        return QuickGuidanceDecision(True, "full_guidance", None, fib_allows, sar_allows, rsi_allows)
    if fib_allows or sar_allows:
        return QuickGuidanceDecision(True, "mixed_guidance", 1, fib_allows, sar_allows, rsi_allows)
    return QuickGuidanceDecision(False, "signal_override", None, fib_allows, sar_allows, rsi_allows)


def _rsi_allows_direction(rsi: float, direction: BreakoutDirection) -> bool:
    if direction is BreakoutDirection.BULLISH:
        return rsi < 70.0
    return rsi > 30.0


def _rsi_is_extreme_exhaustion(rsi: float, direction: BreakoutDirection) -> bool:
    if direction is BreakoutDirection.BULLISH:
        return rsi >= 85.0
    return rsi <= 15.0


def _calculate_rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) <= period:
        return 50.0

    changes = [closes[index] - closes[index - 1] for index in range(1, len(closes))]
    recent_changes = changes[-period:]
    gains = [max(change, 0.0) for change in recent_changes]
    losses = [abs(min(change, 0.0)) for change in recent_changes]
    average_gain = sum(gains) / period
    average_loss = sum(losses) / period
    if average_loss == 0:
        return 100.0 if average_gain > 0 else 50.0
    relative_strength = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def _calculate_parabolic_sar(candles, step: float = 0.02, maximum: float = 0.2) -> float:
    highs = [float(getattr(candle, "high")) for candle in candles]
    lows = [float(getattr(candle, "low")) for candle in candles]
    if len(candles) < 2:
        return lows[-1] if lows else 0.0

    bullish = float(getattr(candles[1], "close")) >= float(getattr(candles[0], "close"))
    sar = lows[0] if bullish else highs[0]
    extreme_point = highs[0] if bullish else lows[0]
    acceleration = step

    for index in range(1, len(candles)):
        previous_sar = sar
        sar = previous_sar + acceleration * (extreme_point - previous_sar)

        if bullish:
            if lows[index] < sar:
                bullish = False
                sar = extreme_point
                extreme_point = lows[index]
                acceleration = step
            else:
                sar = min(sar, lows[index - 1], lows[index])
                if highs[index] > extreme_point:
                    extreme_point = highs[index]
                    acceleration = min(acceleration + step, maximum)
        else:
            if highs[index] > sar:
                bullish = True
                sar = extreme_point
                extreme_point = highs[index]
                acceleration = step
            else:
                sar = max(sar, highs[index - 1], highs[index])
                if lows[index] < extreme_point:
                    extreme_point = lows[index]
                    acceleration = min(acceleration + step, maximum)

    return sar


def resolve_quick_grid_permission(
    *,
    candles,
    positions,
    direction: BreakoutDirection,
    fibonacci_guidance: QuickFibonacciGuidance,
    mt5_module,
    tick,
) -> QuickGridPermission:
    current_price = _grid_current_price(direction, tick)
    atr = _calculate_average_true_range(candles[-14:]) if len(candles) >= 2 else 0.0
    # Grid spacing: 1.2 × ATR so consecutive entries are far enough apart to each
    # have independent R/R rather than stacking into the same noise band.
    min_spacing = max(atr * 1.2, abs(current_price) * 0.0003)
    max_new_entries = _max_new_entries_for_zone(fibonacci_guidance.zone)

    if not _latest_candle_has_tradeable_body(candles):
        return QuickGridPermission(False, "weak_candle", max_new_entries, min_spacing, current_price)

    same_side_positions = [
        position
        for position in positions
        if _position_direction(position, mt5_module) is direction
    ]
    for position in same_side_positions:
        open_price = float(getattr(position, "price_open", getattr(position, "price", current_price)) or current_price)
        if abs(current_price - open_price) < min_spacing:
            return QuickGridPermission(False, "grid_spacing", max_new_entries, min_spacing, current_price)

    return QuickGridPermission(True, "ok", max_new_entries, min_spacing, current_price)


def _grid_current_price(direction: BreakoutDirection, tick) -> float:
    if direction is BreakoutDirection.BULLISH:
        return float(getattr(tick, "ask", getattr(tick, "bid", 0.0)) or 0.0)
    return float(getattr(tick, "bid", getattr(tick, "ask", 0.0)) or 0.0)


def start_dashboard():
    """Start the dashboard server in a separate process."""
    def run_server():
        subprocess.Popen([sys.executable, "dashboard_server.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    logging.info("DASHBOARD SERVER STARTED at http://localhost:8000")


def _max_new_entries_for_zone(zone: str) -> int:
    if zone == "golden_zone":
        return 3
    if zone == "towards_market_mover":
        return 2
    if zone == "in_market_mover":
        return 1
    return 0


def _latest_candle_has_tradeable_body(candles) -> bool:
    if not candles:
        return False
    latest = candles[-1]
    high = float(getattr(latest, "high"))
    low = float(getattr(latest, "low"))
    open_price = float(getattr(latest, "open", getattr(latest, "close")))
    close_price = float(getattr(latest, "close"))
    candle_range = high - low
    if candle_range <= 0:
        return False
    return abs(close_price - open_price) / candle_range >= 0.35


def _latest_candle_body_ratio(candles) -> float:
    if not candles:
        return 0.0
    latest = candles[-1]
    high = float(getattr(latest, "high"))
    low = float(getattr(latest, "low"))
    open_price = float(getattr(latest, "open", getattr(latest, "close")))
    close_price = float(getattr(latest, "close"))
    candle_range = high - low
    if candle_range <= 0:
        return 0.0
    return abs(close_price - open_price) / candle_range


def resolve_quick_mixed_guidance_quality_gate(
    *,
    candles,
    fvg_guidance: QuickFvgGuidance,
    direction: BreakoutDirection,
    current_price: float,
    minimum_body_ratio: float = 0.50,
    max_fvg_bars: int = 8,
    max_price_distance_atr: float = 1.0,
) -> QuickEntryQualityGate:
    if int(fvg_guidance.bars_since) > int(max_fvg_bars):
        return QuickEntryQualityGate(False, "mixed_guidance_stale_fvg", 0.0, max_fvg_bars, 0.0, 0.0)

    body_ratio = _latest_candle_body_ratio(candles)
    if body_ratio < float(minimum_body_ratio):
        return QuickEntryQualityGate(False, "mixed_guidance_weak_body", body_ratio, max_fvg_bars, 0.0, 0.0)

    atr = _calculate_average_true_range(candles[-14:]) if len(candles) >= 2 else 0.0
    midpoint = (float(fvg_guidance.top) + float(fvg_guidance.bottom)) / 2.0
    price_distance = abs(float(current_price) - midpoint)
    max_price_distance = float(atr) * float(max_price_distance_atr)
    if atr > 0.0 and price_distance > max_price_distance:
        return QuickEntryQualityGate(
            False,
            "mixed_guidance_far_from_fvg",
            body_ratio,
            max_fvg_bars,
            price_distance,
            max_price_distance,
        )

    return QuickEntryQualityGate(True, "ok", body_ratio, max_fvg_bars, price_distance, max_price_distance)


def _position_direction(position, mt5_module) -> BreakoutDirection | None:
    position_type = getattr(position, "type", None)
    if position_type is None:
        return None
    if int(position_type) == int(getattr(mt5_module, "ORDER_TYPE_BUY", 0)):
        return BreakoutDirection.BULLISH
    if int(position_type) == int(getattr(mt5_module, "ORDER_TYPE_SELL", 1)):
        return BreakoutDirection.BEARISH
    return None


def _calculate_average_true_range(candles) -> float:
    if len(candles) < 2:
        return 0.0

    true_ranges = []
    previous_close = float(getattr(candles[0], "close"))
    for candle in candles[1:]:
        high = float(getattr(candle, "high"))
        low = float(getattr(candle, "low"))
        true_ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
        previous_close = float(getattr(candle, "close"))
    if not true_ranges:
        return 0.0
    return sum(true_ranges) / len(true_ranges)


def resolve_spread_state(*, mt5_module, symbol: str, max_spread_pips: float = 3.5) -> QuickSpreadState:
    tick = mt5_module.symbol_info_tick(symbol) if hasattr(mt5_module, "symbol_info_tick") else None
    if tick is None:
        return QuickSpreadState(False, "no_tick_for_spread", 0.0, 0.0, float(max_spread_pips))
    spread = _tick_value(tick, "ask") - _tick_value(tick, "bid")
    symbol_info = mt5_module.symbol_info(symbol) if hasattr(mt5_module, "symbol_info") else None
    point = float(getattr(symbol_info, "point", 0.0) or 0.0)
    if point <= 0.0:
        return QuickSpreadState(True, "spread_point_unavailable", spread, 0.0, float(max_spread_pips))
    pip_size = _symbol_pip_size(mt5_module, symbol, tick=tick)
    spread_pips = spread / pip_size if pip_size > 0.0 else 0.0
    if spread <= 0.0:
        return QuickSpreadState(False, "invalid_spread", spread, spread_pips, float(max_spread_pips))
    if spread_pips > float(max_spread_pips):
        return QuickSpreadState(False, "spread_too_wide", spread, spread_pips, float(max_spread_pips))
    return QuickSpreadState(True, "ok", spread, spread_pips, float(max_spread_pips))


def _symbol_pip_size(mt5_module, symbol: str, *, tick=None) -> float:
    symbol_info = mt5_module.symbol_info(symbol) if hasattr(mt5_module, "symbol_info") else None
    point = float(getattr(symbol_info, "point", 0.0) or 0.0)
    if point <= 0:
        price = _tick_value(tick, "ask") if tick is not None else 1.0
        point = max(abs(price) * 0.00001, 0.00001)
    return point * 10.0


def _broker_minimum_stop_distance(mt5_module, symbol: str) -> float:
    symbol_info = mt5_module.symbol_info(symbol) if hasattr(mt5_module, "symbol_info") else None
    point = float(getattr(symbol_info, "point", 0.0) or 0.0)
    stops_level = float(getattr(symbol_info, "trade_stops_level", 0.0) or 0.0)
    freeze_level = float(getattr(symbol_info, "trade_freeze_level", 0.0) or 0.0)
    return max(stops_level, freeze_level) * point


def build_quick_trade_levels(
    *,
    mt5_module,
    symbol: str,
    direction: BreakoutDirection,
    atr: float = 0.0,
    atr_sl_multiplier: float = 0.8,
    atr_tp_multiplier: float = 2.0,
    stop_loss_pips: float = 80.0,
    take_profit_pips: float = 240.0,
) -> tuple[float, float]:
    """Compute ATR-based SL and TP, falling back to pip-based sizing.

    ATR-based sizing (preferred):
        SL = atr * atr_sl_multiplier  (default 0.8 × ATR — tight enough to cut losers quickly)
        TP = atr * atr_tp_multiplier  (default 2.0 × ATR — positive EV with realistic win rate)

    Pip fallback (when atr=0 or symbol info unavailable):
        Uses the legacy stop_loss_pips / take_profit_pips values.
    """
    tick = mt5_module.symbol_info_tick(symbol) if hasattr(mt5_module, "symbol_info_tick") else None
    if tick is None:
        raise RuntimeError(f"No tick data available for symbol: {symbol}")

    ask = _tick_value(tick, "ask")
    bid = _tick_value(tick, "bid")
    spread = max(ask - bid, 0.0)
    pip_size = _symbol_pip_size(mt5_module, symbol, tick=tick)
    broker_minimum_distance = _broker_minimum_stop_distance(mt5_module, symbol)

    if float(atr) > 0.0:
        stop_distance = max(float(atr) * float(atr_sl_multiplier), broker_minimum_distance)
        take_profit_distance = max(float(atr) * float(atr_tp_multiplier), broker_minimum_distance) + spread
    else:
        stop_distance = max(float(stop_loss_pips) * pip_size, broker_minimum_distance)
        take_profit_distance = max(float(take_profit_pips) * pip_size, broker_minimum_distance) + spread

    if direction is BreakoutDirection.BULLISH:
        entry = ask
        return entry - stop_distance, entry + take_profit_distance

    entry = bid
    return entry + stop_distance, entry - take_profit_distance


def compute_quick_expected_value(
    *,
    stop_distance: float,
    reward_distance: float,
    spread: float,
    win_rate: float = 0.50,
) -> float:
    """Compute expected value per unit risk after spread cost.

    EV = win_rate * reward_distance - (1 - win_rate) * stop_distance - spread

    A positive EV is required before opening.  Callers pass the absolute
    price distances (same units) and the one-way spread.
    """
    safe_stop = max(float(stop_distance), 1e-9)
    safe_reward = max(float(reward_distance), 1e-9)
    safe_spread = max(float(spread), 0.0)
    p = max(0.0, min(1.0, float(win_rate)))
    return p * safe_reward - (1.0 - p) * safe_stop - safe_spread


def has_margin_for_quick_order(
    *,
    mt5_module,
    symbol: str,
    direction: BreakoutDirection,
    lot: float,
    min_free_margin: float,
) -> bool:
    if not all(hasattr(mt5_module, name) for name in ("account_info", "symbol_info_tick", "order_calc_margin")):
        return True

    account_info = mt5_module.account_info()
    if account_info is None:
        return True

    margin_free = float(
        getattr(account_info, "margin_free", getattr(account_info, "free_margin", 0.0)) or 0.0
    )
    if margin_free <= float(min_free_margin):
        return False

    tick = mt5_module.symbol_info_tick(symbol)
    if tick is None:
        return False

    if direction is BreakoutDirection.BULLISH:
        order_type = getattr(mt5_module, "ORDER_TYPE_BUY", 0)
        price = float(getattr(tick, "ask"))
    else:
        order_type = getattr(mt5_module, "ORDER_TYPE_SELL", 1)
        price = float(getattr(tick, "bid"))

    required_margin = mt5_module.order_calc_margin(order_type, symbol, float(lot), price)
    if required_margin is None:
        return True
    return margin_free - float(required_margin) >= float(min_free_margin)


def close_profitable_quick_positions(
    *,
    executor,
    symbol: str,
    profit_target: float,
    max_loss: float | None = None,
    tick_direction: BreakoutDirection | None = None,
    log_fn=print,
) -> int:
    closed = 0
    positions = executor.list_bot_positions(symbol, comment_prefix=QUICK_COMMENT_PREFIX)
    mt5_module = getattr(executor, "mt5_module", None)
    best_position = None
    best_profit = None
    worst_position = None
    worst_profit = None
    positive_count = 0
    negative_count = 0
    flat_count = 0
    net_profit = 0.0
    for position in positions:
        # Scale targets by lot size (base 0.01)
        lot_ratio = float(getattr(position, "volume", 0.01)) / 0.01
        scaled_target = profit_target * lot_ratio
        scaled_max_loss = max_loss * lot_ratio
        
        profit = float(getattr(position, "profit", 0.0) or 0.0)
        net_profit += profit
        if profit > 0:
            positive_count += 1
        elif profit < 0:
            negative_count += 1
        else:
            flat_count += 1
        if best_profit is None or profit > best_profit:
            best_position = position
            best_profit = profit
        if worst_profit is None or profit < worst_profit:
            worst_position = position
            worst_profit = profit
        position_direction = _position_direction(position, mt5_module) if mt5_module is not None else None
        exit_reason = None
        # Min profit floor for tick_turn: 15% of target
        tick_turn_floor = scaled_target * 0.15
        if (
            profit >= tick_turn_floor
            and tick_direction is not None
            and position_direction is not None
            and tick_direction is not position_direction
        ):
            exit_reason = "tick_turn"
        elif tick_direction is None and profit >= scaled_target:
            exit_reason = "profit_target"
        elif scaled_max_loss > 0.0 and profit <= -abs(scaled_max_loss):
            exit_reason = "max_loss"

        if exit_reason is not None:
            if exit_reason == "tick_turn":
                exit_comment = f"{QUICK_COMMENT_PREFIX}-tick-turn-profit-exit"
            elif exit_reason == "max_loss":
                exit_comment = f"{QUICK_COMMENT_PREFIX}-loss-exit"
            else:
                exit_comment = f"{QUICK_COMMENT_PREFIX}-profit-exit"
            try:
                executor.close_position(position, comment=exit_comment)
            except Exception as exc:
                log_fn(
                    f"QUICK PROFIT EXIT REJECTED {symbol} "
                    f"ticket={getattr(position, 'ticket', 'unknown')} "
                    f"profit={profit:.2f} reason={exc} exit_reason={exit_reason}"
                )
                continue
            closed += 1
            log_fn(
                f"QUICK PROFIT EXIT {symbol} ticket={getattr(position, 'ticket', 'unknown')} "
                f"profit={profit:.2f} reason={exit_reason}"
            )
            
            # Add to history for dashboard
            trade_history.append({
                "time": datetime.now(timezone.utc).isoformat(),
                "ticket": getattr(position, "ticket", "unknown"),
                "profit": round(profit, 2),
                "reason": exit_reason
            })
            if len(trade_history) > TRADE_HISTORY_LIMIT:
                trade_history.pop(0)
    if closed == 0 and best_position is not None:
        log_fn(
            f"QUICK PROFIT WAIT {symbol} positions={len(positions)} "
            f"positive={positive_count} negative={negative_count} flat={flat_count} "
            f"net_profit={net_profit:.2f} "
            f"best_ticket={getattr(best_position, 'ticket', 'unknown')} "
            f"best_profit={float(best_profit):.2f} "
            f"worst_ticket={getattr(worst_position, 'ticket', 'unknown')} "
            f"worst_profit={float(worst_profit):.2f} "
            f"target={float(profit_target):.2f} "
            f"tick_direction={getattr(tick_direction, 'value', 'NONE')}"
        )
    return closed


def save_bot_state(state: dict):
    """Save bot state to a JSON file for the dashboard."""
    try:
        temp_file = "bot_state.json.tmp"
        with open(temp_file, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(temp_file, "bot_state.json")
    except Exception:
        pass


def run_quick_scalp_loop(
    *,
    mt5_module,
    executor,
    symbol: str,
    lot: float,
    max_positions: int,
    profit_target: float,
    poll_seconds: int,
    max_loss: float = 0.0,
    max_spread_pips: float = 4.0,
    min_free_margin: float = 0.0,
    max_loops: int | None = None,
    atr_sl_multiplier: float = 0.8,
    atr_tp_multiplier: float = 2.0,
    reload_check_fn=None,
    sleep_fn=sleep,
    log_fn=print,
):
    loop_count = 0
    insufficient_margin_logged = False
    last_hold_log = None
    def log_hold_once(message: str) -> None:
        nonlocal last_hold_log
        if message != last_hold_log:
            log_fn(message)
            last_hold_log = message

    while max_loops is None or loop_count < max_loops:
        # Emergency Panic Check
        if os.path.exists("panic.signal"):
            log_fn("!!! PANIC SIGNAL DETECTED - CLOSING ALL POSITIONS !!!")
            all_pos = executor.list_bot_positions(symbol, comment_prefix=QUICK_COMMENT_PREFIX)
            for p in all_pos:
                try: executor.close_position(p, comment="PANIC-EXIT")
                except: pass
            try: os.remove("panic.signal")
            except: pass
            log_fn("Panic exit complete. Bot paused for 30s.")
            sleep_fn(30)
            continue

        account_info = mt5_module.account_info() if hasattr(mt5_module, "account_info") else None

        # 1. Hard Stop Check (Global Account Protection)
        if max_loss > 0 and account_info and getattr(account_info, "profit", 0) <= -abs(max_loss):
            log_fn(f"!!! HARD STOP TRIGGERED !!! Account floating loss exceeded ${max_loss}")
            all_pos = executor.list_bot_positions(symbol, comment_prefix=QUICK_COMMENT_PREFIX)
            for p in all_pos:
                try: executor.close_position(p, comment="HARD-STOP-EXIT")
                except: pass
            log_fn("Hard stop complete. Bot shutting down to protect capital.")
            return "hard_stop_triggered"

        # 2. Dynamic Lot Calculation
        balance = getattr(account_info, "balance", 0.0)
        current_lot = lot
        if os.getenv("QUICK_DYNAMIC_LOT_ENABLED", "false").lower() == "true":
            # For KES accounts: 13,000 KES is approx $100 USD
            compound_unit = float(os.getenv("QUICK_LOT_PER_100_USD", "13000.0"))
            max_cap = float(os.getenv("QUICK_MAX_LOT_CAP", "0.02"))
            
            # Target 0.02 lots per 13,000 KES
            calculated_lot = (balance / compound_unit) * 0.02
            current_lot = max(0.01, min(max_cap, round(calculated_lot, 2)))
            
            # Margin Check: If free margin is too low, scale down
            # (This uses the fresh balance)
            min_margin = float(os.getenv("QUICK_MIN_FREE_MARGIN", "500.0"))
            if account_info is not None:
                free_margin = getattr(account_info, "margin_free", 0.0)
                if free_margin < min_margin:
                    current_lot = 0.01 # Fallback to safety
        
        # Ensure symbol is active and subscribed for ticks
        if hasattr(mt5_module, "symbol_select"):
            mt5_module.symbol_select(symbol, True)
        
        candles = fetch_m1_candles(mt5_module, symbol, count=30)
        m1_direction = resolve_m1_direction(candles)
        current_tick = mt5_module.symbol_info_tick(symbol) if hasattr(mt5_module, "symbol_info_tick") else None
        point = _symbol_pip_size(mt5_module, symbol, tick=current_tick) / 10.0
        recent_ticks = fetch_recent_ticks(mt5_module, symbol, count=100)
        tick_guidance = resolve_tick_direction(recent_ticks, point=point)
        direction = tick_guidance.direction

        positions = executor.list_bot_positions(symbol, comment_prefix=QUICK_COMMENT_PREFIX)

        # Unified Dashboard State Update (at start of each loop)
        # Fresh account info for Dashboard (Direct MT5 Sync)
        _live_info = mt5_module.account_info()
        _acc_dict = _live_info._asdict() if _live_info and hasattr(_live_info, "_asdict") else {}
        
        _bal = float(_acc_dict.get("balance", 0.0))
        _eq = float(_acc_dict.get("equity", 0.0))
        _prof = float(_acc_dict.get("profit", 0.0))
        _curr = str(_acc_dict.get("currency", "KES"))
        
        logging.info(f"[SYNC] Account Pulse | Bal: {_bal} | Eq: {_eq} | Prof: {_prof} {_curr}")

        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "account": {
                "balance": _bal,
                "equity": _eq,
                "profit": _prof,
                "currency": _curr,
            },
            "signals": {
                "tick_dir": direction.value if direction else "None",
                "m1_dir": m1_direction.value if m1_direction else "None",
                "fib_dir": "None",
                "fib_zone": "None",
                "rsi": 0.0,
                "sar_dir": "None",
                "confluence": {"fib_ok": False, "sar_ok": False, "rsi_ok": False}
            },
            "market_data": {
                "m1_candles": [
                    {"time": int(c.timestamp.timestamp()), "open": c.open, "high": c.high, "low": c.low, "close": c.close}
                    for c in candles[-50:]
                ] if candles else [],
                "ticks": [
                    {"time": int(getattr(t, "time", 0)), "bid": getattr(t, "bid", 0), "ask": getattr(t, "ask", 0)}
                    for t in (recent_ticks[-60:] if recent_ticks else [])
                ]
            },
            "trading": {
                "symbol": symbol,
                "strategy_mode": "quick_scalp",
                "candle_timeframe": "M1",
                "positions_count": len(positions),
                "status": "Scanning/Waiting" if direction is None else "Signal Found",
                "is_tradeable": False,
                "history": trade_history,
                "target_value": profit_target
            }
        }
        save_bot_state(state)

        close_profitable_quick_positions(
            executor=executor,
            symbol=symbol,
            profit_target=profit_target,
            max_loss=max_loss,
            tick_direction=direction,
            log_fn=lambda message: log_hold_once(message) if "QUICK PROFIT WAIT" in message else log_fn(message),
        )

        positions = executor.list_bot_positions(symbol, comment_prefix=QUICK_COMMENT_PREFIX)
        if direction is None:
            log_hold_once(
                f"QUICK HOLD {symbol} reason={tick_guidance.reason} "
                f"m1_signal={getattr(m1_direction, 'value', 'NONE')} "
                f"ticks={tick_guidance.tick_count} net_move={tick_guidance.net_move:.2f}"
            )
        elif m1_direction is None:
            log_hold_once(
                f"QUICK HOLD {symbol} reason=m1_signal_unavailable "
                f"tick_direction={direction.value} "
                f"tick_reason={tick_guidance.reason} "
                f"ticks={tick_guidance.tick_count} net_move={tick_guidance.net_move:.2f}"
            )
        else:
            # Tick is the primary signal (highest frequency signal).
            # When m1 agrees: trade in that direction (best confluence).
            # When m1 disagrees: use tick direction but cap to 1 entry (reduced conviction).
            trade_direction = direction
            m1_agrees = m1_direction is direction
            indicator_guidance = resolve_quick_indicator_guidance(candles)
            try:
                m15_candles = fetch_m15_candles(mt5_module, symbol, count=50)
                fibonacci_guidance = resolve_quick_fibonacci_guidance(m15_candles)
                fvg_guidance = resolve_quick_fvg_guidance(
                    m15_candles,
                    direction=trade_direction,
                    current_price=fibonacci_guidance.current_price,
                )
            except Exception as exc:
                log_hold_once(f"QUICK HOLD {symbol} reason=structure_unavailable detail={exc}")
                fibonacci_guidance = None
                fvg_guidance = None
            if fibonacci_guidance is None:
                loop_count += 1
                if reload_check_fn is not None and reload_check_fn():
                    log_fn(f"CODE CHANGE DETECTED {symbol} reloading bot")
                    return "reload_requested"
                if max_loops is not None and loop_count >= max_loops:
                    break
                sleep_fn(poll_seconds)
                continue
            guidance_decision = resolve_quick_guidance_decision(
                direction=trade_direction,
                fibonacci_guidance=fibonacci_guidance,
                indicator_guidance=indicator_guidance,
            )
            # When m1 disagrees cap entries to 1 regardless of guidance
            if not m1_agrees and guidance_decision.allowed:
                from dataclasses import replace as _dc_replace
                override = min(guidance_decision.max_new_entries_override or 1, 1)
                guidance_decision = QuickGuidanceDecision(
                    guidance_decision.allowed,
                    guidance_decision.reason,
                    override,
                    guidance_decision.fib_allows,
                    guidance_decision.sar_allows,
                    guidance_decision.rsi_allows,
                )
            if not guidance_decision.allowed:
                log_hold_once(
                    f"QUICK HOLD {symbol} reason={guidance_decision.reason} signal={direction.value} "
                    f"m1_signal={getattr(m1_direction, 'value', 'NONE')} "
                    f"tick_reason={tick_guidance.reason} "
                    f"fib_direction={getattr(fibonacci_guidance.direction, 'value', 'NONE')} "
                    f"fib_zone={fibonacci_guidance.zone} "
                    f"rsi={indicator_guidance.rsi:.2f} "
                    f"sar_direction={getattr(indicator_guidance.sar_direction, 'value', 'NONE')} "
                    f"sar={indicator_guidance.sar:.2f} "
                    f"fvg_direction={getattr(fvg_guidance.direction, 'value', 'NONE')} "
                    f"fvg_top={fvg_guidance.top:.2f} fvg_bottom={fvg_guidance.bottom:.2f} "
                    f"fvg_bars_since={fvg_guidance.bars_since}"
                )
            else:
                fvg_matches = fvg_guidance is not None and fvg_guidance.allows(trade_direction)
                if not fvg_matches:
                    if guidance_decision.reason == "mixed_guidance":
                        log_hold_once(
                            f"QUICK HOLD {symbol} reason=no_matching_fvg_mixed_guidance signal={direction.value} "
                            f"m1_signal={getattr(m1_direction, 'value', 'NONE')} "
                            f"direction={trade_direction.value} "
                            f"fib_direction={getattr(fibonacci_guidance.direction, 'value', 'NONE')} "
                            f"fib_zone={fibonacci_guidance.zone} "
                            f"swing_low={fibonacci_guidance.swing_low:.2f} "
                            f"swing_high={fibonacci_guidance.swing_high:.2f} "
                            f"current_price={fibonacci_guidance.current_price:.2f}"
                        )
                        loop_count += 1
                        if reload_check_fn is not None and reload_check_fn():
                            log_fn(f"CODE CHANGE DETECTED {symbol} reloading bot")
                            return "reload_requested"
                        if max_loops is not None and loop_count >= max_loops:
                            break
                        sleep_fn(poll_seconds)
                        continue
                elif guidance_decision.reason == "mixed_guidance":
                    quality_gate = resolve_quick_mixed_guidance_quality_gate(
                        candles=candles,
                        fvg_guidance=fvg_guidance,
                        direction=trade_direction,
                        current_price=_grid_current_price(trade_direction, current_tick) if current_tick is not None else fibonacci_guidance.current_price,
                    )
                    if not quality_gate.allowed:
                        log_hold_once(
                            f"QUICK HOLD {symbol} reason={quality_gate.reason} signal={direction.value} "
                            f"m1_signal={getattr(m1_direction, 'value', 'NONE')} "
                            f"direction={trade_direction.value} "
                            f"fib_zone={fibonacci_guidance.zone} "
                            f"fvg_top={fvg_guidance.top:.2f} fvg_bottom={fvg_guidance.bottom:.2f} "
                            f"fvg_bars_since={fvg_guidance.bars_since} "
                            f"body_ratio={quality_gate.body_ratio:.2f} "
                            f"price_distance={quality_gate.price_distance:.2f} "
                            f"max_price_distance={quality_gate.max_price_distance:.2f}"
                        )

                # Compute Target Progress
                best_profit = 0.0
                if positions:
                    best_profit = max([float(getattr(p, "profit", 0)) for p in positions])
                target_pct = min(100, max(0, (best_profit / profit_target) * 100)) if profit_target > 0 else 0

                # Compute MTF Trends
                try:
                    m5_c = fetch_m5_candles(mt5_module, symbol, count=2)
                    h1_c = fetch_h1_candles(mt5_module, symbol, count=2)
                    m5_dir = resolve_m1_direction(m5_c).value if m5_c and resolve_m1_direction(m5_c) else "Neutral"
                    h1_dir = resolve_m1_direction(h1_c).value if h1_c and resolve_m1_direction(h1_c) else "Neutral"
                except:
                    m5_dir = h1_dir = "Error"

                # Compute Quant Math Metrics
                quant_m = resolve_quant_metrics(
                    prices=[c.close for c in candles[-30:]] if candles else [],
                    ticks=state["market_data"]["ticks"],
                    balance=float(getattr(account_info, "balance", 0.0) or 0.0)
                )

                # Detailed Dashboard Signal Update (after indicators resolved)
                state["signals"].update({
                    "fib_dir": getattr(fibonacci_guidance.direction, "value", "None") if fibonacci_guidance else "None",
                    "fib_zone": getattr(fibonacci_guidance, "zone", "None") if fibonacci_guidance else "None",
                    "rsi": round(indicator_guidance.rsi, 2) if indicator_guidance else 0.0,
                    "sar_dir": getattr(indicator_guidance.sar_direction, "value", "None") if indicator_guidance else "None",
                    "quant": {
                        "hurst": quant_m.hurst_exponent,
                        "reversion": quant_m.ou_reversion_speed,
                        "ofi": quant_m.order_flow_imbalance,
                        "kelly_lot": quant_m.kelly_suggested_lot,
                        "z_score": quant_m.z_score,
                        "smoothness": quant_m.autocorrelation
                    },
                    "mtf": {
                        "m1": m1_direction.value if m1_direction else "None",
                        "m5": m5_dir,
                        "m15": getattr(fibonacci_guidance.direction, "value", "None") if fibonacci_guidance else "None",
                        "h1": h1_dir
                    },
                    "confluence": {
                        "fib_ok": guidance_decision.fib_allows if guidance_decision else False,
                        "sar_ok": guidance_decision.sar_allows if guidance_decision else False,
                        "rsi_ok": guidance_decision.rsi_allows if guidance_decision else False,
                    }
                })
                state["trading"]["target_progress"] = round(target_pct, 1)
                state["trading"]["status"] = guidance_decision.reason if guidance_decision else "Scanning"
                state["trading"]["is_tradeable"] = guidance_decision.allowed if guidance_decision else False
                save_bot_state(state)
                save_training_snapshot(state)

                spread_state = resolve_spread_state(mt5_module=mt5_module, symbol=symbol, max_spread_pips=max_spread_pips)
                if not spread_state.allowed:
                    log_hold_once(
                        f"QUICK HOLD {symbol} reason={spread_state.reason} "
                        f"spread_pips={spread_state.spread_pips:.2f} "
                        f"max_spread_pips={spread_state.max_spread_pips:.2f}"
                    )
                    tick = None
                    grid_permission = QuickGridPermission(False, spread_state.reason, 0, 0.0, 0.0)
                else:
                    tick = current_tick
                if tick is None:
                    if spread_state.allowed:
                        log_hold_once(f"QUICK HOLD {symbol} reason=no_tick_for_grid")
                        grid_permission = QuickGridPermission(False, "no_tick_for_grid", 0, 0.0, 0.0)
                elif spread_state.allowed:
                    grid_permission = resolve_quick_grid_permission(
                        candles=candles,
                        positions=positions,
                        direction=trade_direction,
                        fibonacci_guidance=fibonacci_guidance,
                        mt5_module=mt5_module,
                        tick=tick,
                    )
                    if not grid_permission.allowed:
                        log_hold_once(
                            f"QUICK HOLD {symbol} reason={grid_permission.reason} "
                            f"direction={trade_direction.value} price={grid_permission.current_price:.2f} "
                            f"spacing={grid_permission.min_spacing:.2f}"
                        )

                # Compute ATR for sizing (M1 candles, last 14 bars)
                m1_atr = _calculate_average_true_range(candles[-14:]) if len(candles) >= 2 else 0.0

                new_entries = 0
                max_new_entries = grid_permission.max_new_entries
                if guidance_decision.max_new_entries_override is not None:
                    max_new_entries = min(max_new_entries, guidance_decision.max_new_entries_override)
                while (
                    grid_permission.allowed
                    and len(positions) < int(max_positions)
                    and new_entries < int(max_new_entries)
                ):
                    if not has_margin_for_quick_order(
                        mt5_module=mt5_module,
                        symbol=symbol,
                        direction=trade_direction,
                        lot=current_lot,
                        min_free_margin=min_free_margin,
                    ):
                        if not insufficient_margin_logged:
                            log_fn(f"QUICK HOLD {symbol} reason=insufficient_free_margin positions={len(positions)}")
                            insufficient_margin_logged = True
                        break

                    try:
                        stop_loss, take_profit = build_quick_trade_levels(
                            mt5_module=mt5_module,
                            symbol=symbol,
                            direction=trade_direction,
                            atr=m1_atr,
                            atr_sl_multiplier=atr_sl_multiplier,
                            atr_tp_multiplier=atr_tp_multiplier,
                        )
                    except Exception as exc:
                        log_fn(f"QUICK ORDER REJECTED {symbol} reason={exc} positions={len(positions)}")
                        break

                    # --- Expected-Value gate ---
                    # Require positive EV before opening.  A 50% win rate is used as a
                    # conservative baseline; the TP must exceed the SL + spread cost.
                    _cur_tick = mt5_module.symbol_info_tick(symbol) if hasattr(mt5_module, "symbol_info_tick") else None
                    _ask = _tick_value(_cur_tick, "ask") if _cur_tick is not None else 0.0
                    _bid = _tick_value(_cur_tick, "bid") if _cur_tick is not None else 0.0
                    _spread = max(_ask - _bid, 0.0)
                    if trade_direction is BreakoutDirection.BULLISH:
                        _entry = _ask
                        _sl_dist = abs(_entry - stop_loss)
                        _tp_dist = abs(take_profit - _entry)
                    else:
                        _entry = _bid
                        _sl_dist = abs(stop_loss - _entry)
                        _tp_dist = abs(_entry - take_profit)
                    _ev = compute_quick_expected_value(
                        stop_distance=_sl_dist,
                        reward_distance=_tp_dist,
                        spread=_spread,
                        win_rate=0.50,
                    )
                    if _ev <= 0.0:
                        log_hold_once(
                            f"QUICK HOLD {symbol} reason=negative_ev "
                            f"direction={trade_direction.value} "
                            f"sl_dist={_sl_dist:.5f} tp_dist={_tp_dist:.5f} "
                            f"spread={_spread:.5f} ev={_ev:.6f}"
                        )
                        break

                    try:
                        position = executor.open_strategy_trade(
                            symbol=symbol,
                            direction=trade_direction,
                            lot=current_lot,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            comment=QUICK_COMMENT_PREFIX,
                        )
                    except Exception as exc:
                        log_fn(f"QUICK ORDER REJECTED {symbol} reason={exc} positions={len(positions)}")
                        break

                    positions.append(position)
                    new_entries += 1
                    insufficient_margin_logged = False
                    last_hold_log = None
                    log_fn(
                        f"QUICK TRADE OPENED {symbol} ticket={getattr(position, 'ticket', 'unknown')} "
                        f"signal={direction.value} m1_signal={getattr(m1_direction, 'value', 'NONE')} "
                        f"m1_agrees={m1_agrees} "
                        f"tick_reason={tick_guidance.reason} direction={trade_direction.value} "
                        f"guidance={guidance_decision.reason} fib_zone={fibonacci_guidance.zone} "
                        f"swing_low={fibonacci_guidance.swing_low:.2f} "
                        f"swing_high={fibonacci_guidance.swing_high:.2f} "
                        f"fvg_top={getattr(fvg_guidance, 'top', 0.0):.2f} fvg_bottom={getattr(fvg_guidance, 'bottom', 0.0):.2f} "
                        f"fvg_direction={getattr(getattr(fvg_guidance, 'direction', None), 'value', 'NONE')} "
                        f"fvg_bars_since={getattr(fvg_guidance, 'bars_since', 0)} "
                        f"rsi={indicator_guidance.rsi:.2f} "
                        f"sar={indicator_guidance.sar:.2f} spacing={grid_permission.min_spacing:.2f} "
                        f"atr={m1_atr:.5f} ev={_ev:.6f} "
                        f"lot={lot} positions={len(positions)}"
                    )
                    grid_permission = resolve_quick_grid_permission(
                        candles=candles,
                        positions=positions,
                        direction=trade_direction,
                        fibonacci_guidance=fibonacci_guidance,
                        mt5_module=mt5_module,
                        tick=tick,
                    )

        loop_count += 1
        if reload_check_fn is not None and reload_check_fn():
            log_fn(f"CODE CHANGE DETECTED {symbol} reloading bot")
            return "reload_requested"
        if max_loops is not None and loop_count >= max_loops:
            break
        sleep_fn(poll_seconds)
