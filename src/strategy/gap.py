from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from src.market_data import Candle
from src.strategy.breakout import BreakoutDirection


@dataclass(frozen=True, slots=True)
class GapDecision:
    has_gap: bool
    gap_size: float
    size_class: str
    gap_direction: BreakoutDirection | None
    preferred_trade_direction: BreakoutDirection | None
    fill_preferred: bool
    reason: str
    metadata: dict[str, object]


def evaluate_gap_context(*, h4_candles: list[Candle], m15_candles: list[Candle]) -> GapDecision:
    _require_timeframe(h4_candles, "H4", minimum=2)
    _require_timeframe(m15_candles, "M15", minimum=2)

    boundary = _latest_session_boundary(m15_candles)
    if boundary is None:
        return GapDecision(
            has_gap=False,
            gap_size=0.0,
            size_class="none",
            gap_direction=None,
            preferred_trade_direction=None,
            fill_preferred=False,
            reason="no_session_gap_detected",
            metadata={},
        )

    previous_candle, current_candle, gap_minutes = boundary
    reference_close = float(previous_candle.close)
    session_open = float(current_candle.open)
    current_close = float(m15_candles[-1].close)
    gap_delta = session_open - reference_close
    gap_size = abs(gap_delta)
    average_h4_range = _average_range(h4_candles[-min(5, len(h4_candles)) :])
    minimum_gap = max(average_h4_range * 0.15, 0.25)

    if gap_size < minimum_gap:
        return GapDecision(
            has_gap=False,
            gap_size=gap_size,
            size_class="none",
            gap_direction=None,
            preferred_trade_direction=None,
            fill_preferred=False,
            reason="gap_below_threshold",
            metadata={
                "reference_close": reference_close,
                "session_open": session_open,
                "current_close": current_close,
                "gap_minutes": gap_minutes,
                "average_h4_range": average_h4_range,
            },
        )

    gap_ratio = gap_size / average_h4_range if average_h4_range else gap_size
    size_class = _classify_gap(gap_ratio)
    gap_direction = BreakoutDirection.BULLISH if gap_delta > 0 else BreakoutDirection.BEARISH
    fill_direction = (
        BreakoutDirection.BEARISH if gap_direction is BreakoutDirection.BULLISH else BreakoutDirection.BULLISH
    )
    fill_completed = (
        current_close <= reference_close if gap_direction is BreakoutDirection.BULLISH else current_close >= reference_close
    )
    if fill_completed:
        overfill_amount = (
            reference_close - current_close if gap_direction is BreakoutDirection.BULLISH else current_close - reference_close
        )
        return GapDecision(
            has_gap=True,
            gap_size=gap_size,
            size_class=size_class,
            gap_direction=gap_direction,
            preferred_trade_direction=None,
            fill_preferred=False,
            reason="gap_fill_completed",
            metadata={
                "reference_close": reference_close,
                "session_open": session_open,
                "current_close": current_close,
                "gap_minutes": gap_minutes,
                "gap_ratio": gap_ratio,
                "average_h4_range": average_h4_range,
                "fill_completed": True,
                "overfill_amount": overfill_amount,
            },
        )

    continuation_extension = (
        current_close - session_open if gap_direction is BreakoutDirection.BULLISH else session_open - current_close
    )
    continuation_threshold = max(gap_size * 0.25, average_h4_range * 0.25, 0.25)
    accepted_away = continuation_extension >= continuation_threshold

    fill_preferred = not (size_class == "large" and accepted_away)
    preferred_trade_direction = fill_direction if fill_preferred else gap_direction
    reason = "gap_fill_preferred" if fill_preferred else "gap_continuation_preferred"

    return GapDecision(
        has_gap=True,
        gap_size=gap_size,
        size_class=size_class,
        gap_direction=gap_direction,
        preferred_trade_direction=preferred_trade_direction,
        fill_preferred=fill_preferred,
        reason=reason,
        metadata={
            "reference_close": reference_close,
            "session_open": session_open,
            "current_close": current_close,
            "gap_minutes": gap_minutes,
            "gap_ratio": gap_ratio,
            "average_h4_range": average_h4_range,
            "continuation_extension": continuation_extension,
            "continuation_threshold": continuation_threshold,
            "accepted_away": accepted_away,
        },
    )


def _latest_session_boundary(m15_candles: list[Candle]) -> tuple[Candle, Candle, float] | None:
    for index in range(len(m15_candles) - 1, 0, -1):
        previous_candle = m15_candles[index - 1]
        current_candle = m15_candles[index]
        delta_minutes = (current_candle.timestamp - previous_candle.timestamp).total_seconds() / 60.0
        if delta_minutes > 30.0:
            return previous_candle, current_candle, delta_minutes
    return None


def _average_range(candles: list[Candle]) -> float:
    return float(fmean(float(candle.high) - float(candle.low) for candle in candles))


def _classify_gap(gap_ratio: float) -> str:
    if gap_ratio < 0.35:
        return "small"
    if gap_ratio < 0.9:
        return "moderate"
    return "large"


def _require_timeframe(candles: list[Candle], timeframe: str, *, minimum: int) -> None:
    if len(candles) < minimum:
        raise ValueError(f"{timeframe} gap input requires at least {minimum} candles")
    if any(candle.timeframe != timeframe for candle in candles):
        raise ValueError(f"{timeframe} gap input received a different timeframe")
