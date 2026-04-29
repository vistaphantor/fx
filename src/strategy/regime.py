from __future__ import annotations

from dataclasses import dataclass

from src.market_data import Candle
from src.strategy.gap import GapDecision
from src.strategy.volatility import VolatilityState


@dataclass(frozen=True, slots=True)
class RegimeState:
    name: str
    tradable: bool
    continuation_bias: float
    reversion_bias: float
    confidence: float


def classify_regime(
    *,
    h1_candles: list[Candle],
    m30_candles: list[Candle],
    m15_candles: list[Candle],
    volatility_state: VolatilityState,
    gap_decision: GapDecision | None,
) -> RegimeState:
    _require_candles(h1_candles, "H1", minimum=2)
    _require_candles(m30_candles, "M30", minimum=2)
    _require_candles(m15_candles, "M15", minimum=2)

    if gap_decision is not None and gap_decision.has_gap:
        if gap_decision.fill_preferred:
            return RegimeState(
                name="gap_reversion",
                tradable=True,
                continuation_bias=0.8,
                reversion_bias=1.15,
                confidence=0.72,
            )
        return RegimeState(
            name="gap_acceptance",
            tradable=True,
            continuation_bias=1.2,
            reversion_bias=0.7,
            confidence=0.8,
        )

    h1_trend = float(h1_candles[-1].close) - float(h1_candles[0].close)
    m15_impulse = float(m15_candles[-1].close) - float(m15_candles[0].close)

    if (
        volatility_state.range_expansion_ratio >= 1.15
        and volatility_state.body_efficiency >= 0.65
        and abs(h1_trend) >= max(volatility_state.short_atr * 0.5, 0.1)
        and abs(m15_impulse) >= max(volatility_state.short_atr * 0.4, 0.1)
    ):
        return RegimeState(
            name="expansion",
            tradable=True,
            continuation_bias=1.25,
            reversion_bias=0.55,
            confidence=0.85,
        )

    if volatility_state.range_expansion_ratio <= 0.8 and volatility_state.body_efficiency <= 0.4:
        return RegimeState(
            name="compression",
            tradable=False,
            continuation_bias=0.5,
            reversion_bias=0.85,
            confidence=0.75,
        )

    if abs(h1_trend) >= max(volatility_state.medium_atr * 0.35, 0.1):
        return RegimeState(
            name="trend",
            tradable=True,
            continuation_bias=1.1,
            reversion_bias=0.6,
            confidence=0.78,
        )

    return RegimeState(
        name="pullback",
        tradable=True,
        continuation_bias=0.95,
        reversion_bias=0.8,
        confidence=0.62,
    )


def _require_candles(candles: list[Candle], timeframe: str, *, minimum: int) -> None:
    if len(candles) < minimum:
        raise ValueError(f"{timeframe} regime input requires at least {minimum} candles")
    if any(candle.timeframe != timeframe for candle in candles):
        raise ValueError(f"{timeframe} regime input received a different timeframe")
