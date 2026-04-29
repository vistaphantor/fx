"""First-retest tracking for breakout-follow-through setups."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from src.market_data import Candle
from src.strategy.breakout import BreakoutDirection, BreakoutResult


M5_MINUTES = 5


@dataclass(frozen=True, slots=True)
class RetestState:
    breakout: BreakoutResult
    zone: dict[str, Any]
    max_candles_since_breakout: int
    touch_count: int = 0
    first_touch_timestamp: datetime | None = None
    is_invalidated: bool = False
    invalidation_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RetestOutcome:
    state: RetestState
    is_first_touch: bool
    is_invalidated: bool
    reason: str


def _zone_kind(zone: dict[str, Any]) -> str:
    kind = zone.get("kind")
    if hasattr(kind, "value"):
        return str(kind.value)
    return str(kind)


def _refinement_high(zone: dict[str, Any]) -> float:
    return float(zone["refinement_high"])


def _refinement_low(zone: dict[str, Any]) -> float:
    return float(zone["refinement_low"])


def _candle_age_in_m5_candles(breakout_timestamp: datetime, candle_timestamp: datetime) -> int:
    delta = candle_timestamp - breakout_timestamp
    return int(delta.total_seconds() // (M5_MINUTES * 60))


def _validate_state(breakout: BreakoutResult, zone: dict[str, Any]) -> None:
    zone_kind = _zone_kind(zone)
    if breakout.direction is BreakoutDirection.BULLISH and zone_kind != "SUPPLY":
        raise ValueError("bullish breakout requires a supply zone")
    if breakout.direction is BreakoutDirection.BEARISH and zone_kind != "DEMAND":
        raise ValueError("bearish breakout requires a demand zone")


def _touches_retest_zone(state: RetestState, candle: Candle) -> bool:
    if state.breakout.direction not in (BreakoutDirection.BULLISH, BreakoutDirection.BEARISH):
        raise ValueError("RetestState breakout must have a direction")

    return candle.high >= _refinement_low(state.zone) and candle.low <= _refinement_high(state.zone)


def _is_degraded_retest(state: RetestState, candle: Candle) -> bool:
    if state.breakout.direction is BreakoutDirection.BULLISH:
        return candle.low < _refinement_low(state.zone)

    if state.breakout.direction is BreakoutDirection.BEARISH:
        return candle.high > _refinement_high(state.zone)

    raise ValueError("RetestState breakout must have a direction")


def start_retest_tracking(
    breakout: BreakoutResult,
    zone: dict[str, Any],
    *,
    max_candles_since_breakout: int,
) -> RetestState:
    """Create a retest tracker for a confirmed breakout."""

    if not breakout.is_breakout or breakout.direction is None:
        raise ValueError("start_retest_tracking requires a confirmed breakout")
    if max_candles_since_breakout < 1:
        raise ValueError("max_candles_since_breakout must be at least 1")
    _validate_state(breakout, zone)

    return RetestState(
        breakout=breakout,
        zone=zone,
        max_candles_since_breakout=max_candles_since_breakout,
    )


def observe_retest(state: RetestState, candle: Candle) -> RetestOutcome:
    """Update retest state for a new M5 candle."""

    if state.is_invalidated:
        return RetestOutcome(
            state=state,
            is_first_touch=False,
            is_invalidated=True,
            reason=state.invalidation_reason or "retest_invalidated",
        )

    _validate_state(state.breakout, state.zone)

    if candle.timeframe != "M5":
        raise ValueError("observe_retest requires an M5 candle")

    candle_age = _candle_age_in_m5_candles(state.breakout.candle_timestamp, candle.timestamp)
    if candle_age >= state.max_candles_since_breakout:
        invalidated = replace(state, is_invalidated=True, invalidation_reason="late_retest")
        return RetestOutcome(
            state=invalidated,
            is_first_touch=False,
            is_invalidated=True,
            reason="late_retest",
        )

    if state.touch_count > 0:
        invalidated = replace(state, is_invalidated=True, invalidation_reason="retest_already_used")
        return RetestOutcome(
            state=invalidated,
            is_first_touch=False,
            is_invalidated=True,
            reason="retest_already_used",
        )

    if not _touches_retest_zone(state, candle):
        return RetestOutcome(
            state=state,
            is_first_touch=False,
            is_invalidated=False,
            reason="no_retest",
        )

    if _is_degraded_retest(state, candle):
        invalidated = replace(state, is_invalidated=True, invalidation_reason="degraded_retest")
        return RetestOutcome(
            state=invalidated,
            is_first_touch=False,
            is_invalidated=True,
            reason="degraded_retest",
        )

    touched = replace(
        state,
        touch_count=1,
        first_touch_timestamp=candle.timestamp,
    )
    return RetestOutcome(
        state=touched,
        is_first_touch=True,
        is_invalidated=False,
        reason="first_touch",
    )
