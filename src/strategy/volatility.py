from __future__ import annotations

from dataclasses import dataclass

from src.market_data import Candle


@dataclass(frozen=True, slots=True)
class VolatilityState:
    short_atr: float
    medium_atr: float
    realized_range: float
    body_efficiency: float
    range_expansion_ratio: float


def compute_atr(candles: list[Candle], *, period: int) -> float:
    if period <= 0:
        raise ValueError("period must be greater than 0")
    if len(candles) < period:
        raise ValueError(f"ATR requires at least {period} candles")

    sample = candles[-period:]
    true_ranges = [_true_range(sample[index], sample[index - 1] if index > 0 else None) for index in range(len(sample))]
    return sum(true_ranges) / len(true_ranges)


def build_volatility_state(*, candles: list[Candle]) -> VolatilityState:
    if len(candles) < 3:
        raise ValueError("volatility state requires at least 3 candles")

    short_period = min(3, len(candles))
    medium_period = min(6, len(candles))
    short_atr = compute_atr(candles, period=short_period)
    medium_atr = compute_atr(candles, period=medium_period)
    realized_range = max(float(candle.high) for candle in candles) - min(float(candle.low) for candle in candles)
    body_total = sum(abs(float(candle.close) - float(candle.open)) for candle in candles)
    range_total = sum(max(float(candle.high) - float(candle.low), 0.0) for candle in candles)
    body_efficiency = (body_total / range_total) if range_total else 0.0
    range_expansion_ratio = (short_atr / medium_atr) if medium_atr else 0.0

    return VolatilityState(
        short_atr=short_atr,
        medium_atr=medium_atr,
        realized_range=realized_range,
        body_efficiency=body_efficiency,
        range_expansion_ratio=range_expansion_ratio,
    )


def _true_range(candle: Candle, previous_candle: Candle | None) -> float:
    high = float(candle.high)
    low = float(candle.low)
    intrabar_range = high - low
    if previous_candle is None:
        return intrabar_range

    previous_close = float(previous_candle.close)
    return max(
        intrabar_range,
        abs(high - previous_close),
        abs(low - previous_close),
    )
