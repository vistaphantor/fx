from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.market_data import Candle
from src.strategy.breakout import BreakoutDirection
from src.strategy.direction import DirectionDecision

if TYPE_CHECKING:
    from src.strategy.setup import SetupDecision


@dataclass(frozen=True, slots=True)
class TriggerDecision:
    is_ready: bool
    reason: str
    entry_price: float | None
    invalidation_price: float | None
    metadata: dict[str, object]
    quality_score: float = 0.0
    expected_move_multiple: float = 0.0


def evaluate_m15_trigger(
    *,
    m15_candles: list[Candle],
    setup_decision: SetupDecision,
    direction_decision: DirectionDecision,
) -> TriggerDecision:
    _require_timeframe(m15_candles, "M15", minimum=3)
    if not setup_decision.is_ready:
        return TriggerDecision(
            is_ready=False,
            reason="m15_setup_not_ready",
            entry_price=None,
            invalidation_price=None,
            metadata={},
            quality_score=0.0,
            expected_move_multiple=0.0,
        )
    if not direction_decision.is_valid or direction_decision.direction is None:
        return TriggerDecision(
            is_ready=False,
            reason="m15_direction_missing",
            entry_price=None,
            invalidation_price=None,
            metadata={},
            quality_score=0.0,
            expected_move_multiple=0.0,
        )

    newest_index = len(m15_candles) - 1
    oldest_candidate_index = max(1, len(m15_candles) - 3)
    for index in range(newest_index, oldest_candidate_index - 1, -1):
        candidate = m15_candles[index]
        previous = m15_candles[index - 1]
        candidate_trigger = _evaluate_m15_trigger_candidate(
            candidate=candidate,
            previous=previous,
            direction=direction_decision.direction,
        )
        if candidate_trigger is None:
            continue
        if _trigger_invalidated(
            direction=direction_decision.direction,
            invalidation_price=candidate_trigger.invalidation_price,
            later_candles=m15_candles[index + 1 :],
        ):
            continue

        freshness_penalty = max(0, newest_index - index)
        quality_score = max(candidate_trigger.quality_score - (freshness_penalty * 0.1), 0.6)
        reason = "m15_trigger_ready" if index == newest_index else "m15_trigger_still_valid"
        metadata = dict(candidate_trigger.metadata)
        metadata.update({"trigger_candle_index": index, "trigger_candle_timestamp": candidate.timestamp})
        return TriggerDecision(
            is_ready=True,
            reason=reason,
            entry_price=candidate_trigger.entry_price,
            invalidation_price=candidate_trigger.invalidation_price,
            metadata=metadata,
            quality_score=quality_score,
            expected_move_multiple=candidate_trigger.expected_move_multiple,
        )

    return TriggerDecision(
        is_ready=False,
        reason="m15_velocity_missing",
        entry_price=None,
        invalidation_price=None,
        metadata={},
        quality_score=0.0,
        expected_move_multiple=0.0,
    )


def evaluate_m5_trigger(
    *,
    m5_candles: list[Candle],
    setup_decision: SetupDecision,
    direction_decision: DirectionDecision,
    confirmation_decision: TriggerDecision | None = None,
) -> TriggerDecision:
    _require_timeframe(m5_candles, "M5", minimum=3)
    if confirmation_decision is not None and not confirmation_decision.is_ready:
        return TriggerDecision(
            is_ready=False,
            reason="m5_confirmation_missing",
            entry_price=None,
            invalidation_price=None,
            metadata={},
            quality_score=0.0,
            expected_move_multiple=0.0,
        )
    if not setup_decision.is_ready:
        return TriggerDecision(
            is_ready=False,
            reason="m5_setup_not_ready",
            entry_price=None,
            invalidation_price=None,
            metadata={},
            quality_score=0.0,
            expected_move_multiple=0.0,
        )
    if not direction_decision.is_valid or direction_decision.direction is None:
        return TriggerDecision(
            is_ready=False,
            reason="m5_direction_missing",
            entry_price=None,
            invalidation_price=None,
            metadata={},
            quality_score=0.0,
            expected_move_multiple=0.0,
        )

    latest = m5_candles[-1]
    previous = m5_candles[-2]
    midpoint = (float(previous.high) + float(previous.low)) / 2.0
    if direction_decision.direction is BreakoutDirection.BULLISH:
        if latest.close > midpoint:
            invalidation_price = float(latest.low)
            risk = max(float(latest.close) - invalidation_price, 1e-9)
            expected_move_multiple = max((float(latest.high) - float(previous.low)) / risk, 0.0)
            return TriggerDecision(
                is_ready=True,
                reason="m5_trigger_ready",
                entry_price=float(latest.close),
                invalidation_price=invalidation_price,
                metadata={"trigger_kind": "bullish_execution"},
                quality_score=1.0,
                expected_move_multiple=expected_move_multiple,
            )
        continuation_decision = _evaluate_m5_continuation(
            latest=latest,
            previous=previous,
            direction=direction_decision.direction,
            confirmation_decision=confirmation_decision,
        )
        if continuation_decision is not None and continuation_decision.is_ready:
            return continuation_decision
    else:
        if latest.close < midpoint:
            invalidation_price = float(latest.high)
            risk = max(invalidation_price - float(latest.close), 1e-9)
            expected_move_multiple = max((float(previous.high) - float(latest.low)) / risk, 0.0)
            return TriggerDecision(
                is_ready=True,
                reason="m5_trigger_ready",
                entry_price=float(latest.close),
                invalidation_price=invalidation_price,
                metadata={"trigger_kind": "bearish_execution"},
                quality_score=1.0,
                expected_move_multiple=expected_move_multiple,
            )
        continuation_decision = _evaluate_m5_continuation(
            latest=latest,
            previous=previous,
            direction=direction_decision.direction,
            confirmation_decision=confirmation_decision,
        )
        if continuation_decision is not None and continuation_decision.is_ready:
            return continuation_decision

    return TriggerDecision(
        is_ready=False,
        reason="m5_velocity_missing",
        entry_price=None,
        invalidation_price=None,
        metadata={} if continuation_decision is None else dict(continuation_decision.metadata),
        quality_score=0.0,
        expected_move_multiple=0.0,
    )


def _require_timeframe(candles: list[Candle], timeframe: str, *, minimum: int) -> None:
    if len(candles) < minimum:
        raise ValueError(f"{timeframe} trigger input requires at least {minimum} candles")
    if any(candle.timeframe != timeframe for candle in candles):
        raise ValueError(f"{timeframe} trigger input received a different timeframe")


def _evaluate_m15_trigger_candidate(
    *,
    candidate: Candle,
    previous: Candle,
    direction: BreakoutDirection,
) -> TriggerDecision | None:
    midpoint = (float(previous.high) + float(previous.low)) / 2.0
    if direction is BreakoutDirection.BULLISH:
        if candidate.close > midpoint:
            invalidation_price = float(candidate.low)
            risk = max(float(candidate.close) - invalidation_price, 1e-9)
            expected_move_multiple = max((float(candidate.high) - float(previous.low)) / risk, 0.0)
            return TriggerDecision(
                is_ready=True,
                reason="m15_trigger_ready",
                entry_price=float(candidate.close),
                invalidation_price=invalidation_price,
                metadata={"trigger_kind": "bullish_aggressive"},
                quality_score=1.0,
                expected_move_multiple=expected_move_multiple,
            )
        return None

    if candidate.close < midpoint:
        invalidation_price = float(candidate.high)
        risk = max(invalidation_price - float(candidate.close), 1e-9)
        expected_move_multiple = max((float(previous.high) - float(candidate.low)) / risk, 0.0)
        return TriggerDecision(
            is_ready=True,
            reason="m15_trigger_ready",
            entry_price=float(candidate.close),
            invalidation_price=invalidation_price,
            metadata={"trigger_kind": "bearish_aggressive"},
            quality_score=1.0,
            expected_move_multiple=expected_move_multiple,
        )
    return None


def _trigger_invalidated(
    *,
    direction: BreakoutDirection,
    invalidation_price: float | None,
    later_candles: list[Candle],
) -> bool:
    if invalidation_price is None:
        return True
    if direction is BreakoutDirection.BULLISH:
        return any(float(candle.low) <= float(invalidation_price) for candle in later_candles)
    return any(float(candle.high) >= float(invalidation_price) for candle in later_candles)


def _continuation_retrace_fraction(*, direction: BreakoutDirection, latest_close: float, confirmation_decision: TriggerDecision | None) -> float | None:
    if confirmation_decision is None or confirmation_decision.entry_price is None or confirmation_decision.invalidation_price is None:
        return None
    entry_price = float(confirmation_decision.entry_price)
    invalidation_price = float(confirmation_decision.invalidation_price)
    risk = abs(invalidation_price - entry_price)
    if risk <= 1e-9:
        return None
    if direction is BreakoutDirection.BULLISH:
        if latest_close < entry_price:
            return (entry_price - latest_close) / risk
        return 0.0
    if latest_close > entry_price:
        return (latest_close - entry_price) / risk
    return 0.0


def _evaluate_m5_continuation(
    *,
    latest: Candle,
    previous: Candle,
    direction: BreakoutDirection,
    confirmation_decision: TriggerDecision | None,
) -> TriggerDecision | None:
    if confirmation_decision is None or confirmation_decision.entry_price is None or confirmation_decision.invalidation_price is None:
        return None

    entry_price = float(confirmation_decision.entry_price)
    invalidation_price = float(confirmation_decision.invalidation_price)
    risk = abs(invalidation_price - entry_price)
    if risk <= 1e-9:
        return None

    candle_range = max(float(latest.high) - float(latest.low), 1e-9)
    continuation_retrace = _continuation_retrace_fraction(
        direction=direction,
        latest_close=float(latest.close),
        confirmation_decision=confirmation_decision,
    )
    if continuation_retrace is None:
        return None

    structure_intact = True
    close_position_score = 0.0
    rejection_wick_score = 0.0
    directional_body_score = 0.0
    reclaim_score = 0.0
    trigger_kind = "continuation"
    if direction is BreakoutDirection.BULLISH:
        structure_intact = float(latest.low) > invalidation_price
        close_position_score = max(0.0, min((float(latest.close) - float(latest.low)) / candle_range, 1.0))
        rejection_wick_score = max(0.0, min((min(float(latest.open), float(latest.close)) - float(latest.low)) / candle_range, 1.0))
        directional_body_score = max(0.0, min((float(latest.close) - float(latest.open)) / candle_range, 1.0))
        reclaim_score = max(0.0, 1.0 - max(float(previous.close) - float(latest.close), 0.0) / risk)
        trigger_kind = "bullish_continuation"
    else:
        structure_intact = float(latest.high) < invalidation_price
        close_position_score = max(0.0, min((float(latest.high) - float(latest.close)) / candle_range, 1.0))
        rejection_wick_score = max(0.0, min((float(latest.high) - max(float(latest.open), float(latest.close))) / candle_range, 1.0))
        directional_body_score = max(0.0, min((float(latest.open) - float(latest.close)) / candle_range, 1.0))
        reclaim_score = max(0.0, 1.0 - max(float(latest.close) - float(previous.close), 0.0) / risk)
        trigger_kind = "bearish_continuation"

    entry_distance_score = max(0.0, 1.0 - abs(float(latest.close) - entry_price) / max(risk * 0.90, 1e-9))
    retrace_score = max(0.0, 1.0 - continuation_retrace / 0.80)
    continuation_score = (
        (retrace_score * 0.30)
        + (close_position_score * 0.24)
        + (directional_body_score * 0.18)
        + (rejection_wick_score * 0.12)
        + (reclaim_score * 0.10)
        + (entry_distance_score * 0.06)
    )

    metadata = {
        "trigger_kind": trigger_kind,
        "continuation_retrace": continuation_retrace,
        "continuation_score": continuation_score,
        "close_position_score": close_position_score,
        "directional_body_score": directional_body_score,
        "rejection_wick_score": rejection_wick_score,
        "reclaim_score": reclaim_score,
        "entry_distance_score": entry_distance_score,
        "structure_intact": structure_intact,
    }
    is_ready = structure_intact and continuation_retrace <= 0.85 and continuation_score >= 0.48
    return TriggerDecision(
        is_ready=is_ready,
        reason="m5_continuation_ready" if is_ready else "m5_velocity_missing",
        entry_price=float(latest.close) if is_ready else None,
        invalidation_price=invalidation_price if is_ready else None,
        metadata=metadata,
        quality_score=max(0.0, min(continuation_score, 1.0)) if is_ready else 0.0,
        expected_move_multiple=max(float(confirmation_decision.expected_move_multiple), 1.0) if is_ready else 0.0,
    )
