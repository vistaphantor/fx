from __future__ import annotations

from collections.abc import Sequence

from src.market_data import Candle
from src.strategy.breakout import BreakoutDirection
from src.strategy.engine import NoTradeResult, TradePlan
from src.strategy.risk import build_trade_levels


def evaluate_top_down_strategy(
    *,
    d1_candles: Sequence[Candle],
    h4_candles: Sequence[Candle],
    h1_candles: Sequence[Candle],
    m30_candles: Sequence[Candle],
    m15_candles: Sequence[Candle],
    risk_buffer: float,
) -> TradePlan | NoTradeResult:
    _require_timeframe(d1_candles, "D1", minimum=1)
    _require_timeframe(h4_candles, "H4", minimum=4)
    _require_timeframe(h1_candles, "H1", minimum=4)
    _require_timeframe(m30_candles, "M30", minimum=4)
    _require_timeframe(m15_candles, "M15", minimum=5)

    h1_direction = _velocity_direction(h1_candles, lookback=4)
    m30_direction = _velocity_direction(m30_candles, lookback=4)
    if h1_direction is None or m30_direction is None or h1_direction is not m30_direction:
        return _no_trade("h1_m30_direction_not_aligned", h1_direction=h1_direction, m30_direction=m30_direction)

    m15_direction = _velocity_direction(m15_candles, lookback=2)
    if m15_direction is not h1_direction:
        return _no_trade(
            "m15_velocity_disagrees_with_top_down_bias",
            top_down_direction=h1_direction,
            m15_direction=m15_direction,
        )

    entry_price = float(m15_candles[-1].close)
    previous_daily = d1_candles[-1]
    h4_range = h4_candles[-4:]
    m15_structure = m15_candles[-5:]

    if h1_direction is BreakoutDirection.BULLISH:
        objective = float(previous_daily.high)
        if entry_price >= objective:
            return _no_trade("daily_high_objective_already_hit", daily_objective=objective, entry_price=entry_price)
        structure_low = min(candle.low for candle in m15_structure)
        structure_high = max(candle.high for candle in m15_structure)
    else:
        objective = float(previous_daily.low)
        if entry_price <= objective:
            return _no_trade("daily_low_objective_already_hit", daily_objective=objective, entry_price=entry_price)
        structure_low = min(candle.low for candle in m15_structure)
        structure_high = max(candle.high for candle in m15_structure)

    levels = build_trade_levels(
        entry_price=entry_price,
        direction=h1_direction,
        retest_structure_low=structure_low,
        retest_structure_high=structure_high,
        buffer=risk_buffer,
        candle_timestamp=m15_candles[-1].timestamp,
    )
    return TradePlan(
        is_trade=True,
        direction=levels.direction,
        entry_price=levels.entry_price,
        stop_loss=levels.stop_loss,
        take_profit=levels.take_profit,
        reason="top_down_trade_plan_ready",
        metadata={
            "stage": "top_down_strategy",
            "daily_objective": objective,
            "h1_direction": h1_direction.value,
            "m30_direction": m30_direction.value,
            "m15_direction": m15_direction.value,
        },
    )


def _velocity_direction(candles: Sequence[Candle], *, lookback: int) -> BreakoutDirection | None:
    if len(candles) < lookback:
        return None
    first_close = candles[-lookback].close
    last_close = candles[-1].close
    if last_close > first_close:
        return BreakoutDirection.BULLISH
    if last_close < first_close:
        return BreakoutDirection.BEARISH
    return None


def _require_timeframe(candles: Sequence[Candle], timeframe: str, *, minimum: int) -> None:
    if len(candles) < minimum:
        raise ValueError(f"{timeframe} strategy input requires at least {minimum} candles")
    if any(candle.timeframe != timeframe for candle in candles):
        raise ValueError(f"{timeframe} strategy input received a different timeframe")


def _no_trade(reason: str, **metadata) -> NoTradeResult:
    return NoTradeResult(is_trade=False, reason=reason, metadata={"stage": "top_down_strategy", **metadata})
