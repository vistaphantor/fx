from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.market_data import Candle
from src.strategy.breakout import BreakoutDirection
from src.strategy.context import H4Context
from src.strategy.direction import DirectionDecision
from src.strategy.tradingview_confluence import TradingViewConfluence

if TYPE_CHECKING:
    from src.strategy.trigger import TriggerDecision


@dataclass(frozen=True, slots=True)
class SetupDecision:
    is_ready: bool
    reason: str
    setup_state: str
    metadata: dict[str, object]
    quality_score: float = 0.0


def evaluate_m30_setup(
    *,
    m30_candles: list[Candle],
    direction_decision: DirectionDecision,
    h4_context: H4Context,
    tradingview_confluence: TradingViewConfluence | None = None,
) -> SetupDecision:
    _require_timeframe(m30_candles, "M30", minimum=3)
    if not direction_decision.is_valid or direction_decision.direction is None:
        return SetupDecision(
            is_ready=False,
            reason="m30_missing_direction_bias",
            setup_state="blocked",
            metadata={},
            quality_score=0.0,
        )

    latest = m30_candles[-1]
    previous = m30_candles[-2]
    current_price = float(direction_decision.metadata.get("current_price", float(latest.close)))
    h1_bias_strength = _m30_bias_strength(direction_decision)

    if direction_decision.direction is BreakoutDirection.BULLISH:
        demand_upper = max(zone[1] for zone in h4_context.demand_zones)
        supply_lower = min(zone[0] for zone in h4_context.supply_zones)
        progress = _range_progress(current_price, demand_upper, supply_lower)
        state_scores, continuation_metadata = _evaluate_m30_state_scores(
            latest=latest,
            previous=previous,
            direction=direction_decision.direction,
            current_price=current_price,
            progress=progress,
            near_zone_boundary=demand_upper,
            opposing_zone_boundary=supply_lower,
        )
        best_state, best_state_score = _best_m30_state(state_scores)
        state_metadata = {
            "range_progress": progress,
            "h1_bias_strength": h1_bias_strength,
            "zone_rejection_score": state_scores["zone_rejection"],
            "breakaway_acceptance_score": state_scores["breakaway_acceptance"],
            "continuation_reacceptance_score": state_scores["continuation_reacceptance"],
            "best_state": best_state,
            "best_state_score": best_state_score,
            **continuation_metadata,
        }
        if latest.close > latest.open and latest.low <= demand_upper:
            return SetupDecision(
                is_ready=True,
                reason="m30_rejection_ready",
                setup_state="rejecting_level",
                metadata={**state_metadata, "demand_upper": demand_upper},
                quality_score=1.0,
            )
        if (
            state_scores["continuation_reacceptance"] >= 0.56
            and h1_bias_strength >= 0.45
            and bool(continuation_metadata.get("structure_intact", False))
        ):
            return SetupDecision(
                is_ready=True,
                reason="m30_continuation_reacceptance_ready",
                setup_state="continuation_reacceptance",
                metadata=state_metadata,
                quality_score=max(0.56, min(state_scores["continuation_reacceptance"], 1.0)),
            )
        if 0.35 <= progress <= 0.65:
            if state_scores["continuation_reacceptance"] >= 0.56 and h1_bias_strength >= 0.10:
                return SetupDecision(
                    is_ready=False,
                    reason="m30_setup_not_ready",
                    setup_state="continuation_reacceptance",
                    metadata=state_metadata,
                    quality_score=min(state_scores["continuation_reacceptance"], 0.45),
                )
            if (
                tradingview_confluence is not None
                and tradingview_confluence.is_active
                and tradingview_confluence.preferred_direction is BreakoutDirection.BULLISH
                and tradingview_confluence.setup_bonus > 0
                and progress <= 0.45
            ):
                return SetupDecision(
                    is_ready=True,
                    reason="m30_tradingview_confluence_ready",
                    setup_state="confluence_override",
                    metadata=state_metadata,
                    quality_score=0.8,
                )
            return SetupDecision(
                is_ready=False,
                reason="m30_midrange_noise",
                setup_state="midrange",
                metadata=state_metadata,
                quality_score=0.25,
            )
        if current_price > float(previous.high):
            return SetupDecision(
                is_ready=True,
                reason="m30_breakaway_ready",
                setup_state="breaking_away",
                metadata={**state_metadata, "supply_lower": supply_lower},
                quality_score=0.9,
            )
        if (
            tradingview_confluence is not None
            and tradingview_confluence.is_active
            and tradingview_confluence.preferred_direction is BreakoutDirection.BULLISH
            and tradingview_confluence.setup_bonus > 0
            and progress <= 0.30
            and latest.close >= previous.close
        ):
            return SetupDecision(
                is_ready=True,
                reason="m30_tradingview_confluence_ready",
                setup_state="confluence_override",
                metadata=state_metadata,
                quality_score=0.75,
            )
        if state_scores["continuation_reacceptance"] >= 0.50 and h1_bias_strength >= 0.10:
            return SetupDecision(
                is_ready=False,
                reason="m30_setup_not_ready",
                setup_state="continuation_reacceptance",
                metadata=state_metadata,
                quality_score=min(state_scores["continuation_reacceptance"], 0.45),
            )
    else:
        supply_lower = min(zone[0] for zone in h4_context.supply_zones)
        demand_upper = max(zone[1] for zone in h4_context.demand_zones)
        progress = _range_progress(current_price, demand_upper, supply_lower)
        state_scores, continuation_metadata = _evaluate_m30_state_scores(
            latest=latest,
            previous=previous,
            direction=direction_decision.direction,
            current_price=current_price,
            progress=progress,
            near_zone_boundary=supply_lower,
            opposing_zone_boundary=demand_upper,
        )
        best_state, best_state_score = _best_m30_state(state_scores)
        state_metadata = {
            "range_progress": progress,
            "h1_bias_strength": h1_bias_strength,
            "zone_rejection_score": state_scores["zone_rejection"],
            "breakaway_acceptance_score": state_scores["breakaway_acceptance"],
            "continuation_reacceptance_score": state_scores["continuation_reacceptance"],
            "best_state": best_state,
            "best_state_score": best_state_score,
            **continuation_metadata,
        }
        if latest.close < latest.open and latest.high >= supply_lower:
            return SetupDecision(
                is_ready=True,
                reason="m30_rejection_ready",
                setup_state="rejecting_level",
                metadata={**state_metadata, "supply_lower": supply_lower},
                quality_score=1.0,
            )
        if (
            state_scores["continuation_reacceptance"] >= 0.56
            and h1_bias_strength >= 0.45
            and bool(continuation_metadata.get("structure_intact", False))
        ):
            return SetupDecision(
                is_ready=True,
                reason="m30_continuation_reacceptance_ready",
                setup_state="continuation_reacceptance",
                metadata=state_metadata,
                quality_score=max(0.56, min(state_scores["continuation_reacceptance"], 1.0)),
            )
        if 0.35 <= progress <= 0.65:
            if state_scores["continuation_reacceptance"] >= 0.56 and h1_bias_strength >= 0.10:
                return SetupDecision(
                    is_ready=False,
                    reason="m30_setup_not_ready",
                    setup_state="continuation_reacceptance",
                    metadata=state_metadata,
                    quality_score=min(state_scores["continuation_reacceptance"], 0.45),
                )
            if (
                tradingview_confluence is not None
                and tradingview_confluence.is_active
                and tradingview_confluence.preferred_direction is BreakoutDirection.BEARISH
                and tradingview_confluence.setup_bonus > 0
                and progress >= 0.55
            ):
                return SetupDecision(
                    is_ready=True,
                    reason="m30_tradingview_confluence_ready",
                    setup_state="confluence_override",
                    metadata=state_metadata,
                    quality_score=0.8,
                )
            return SetupDecision(
                is_ready=False,
                reason="m30_midrange_noise",
                setup_state="midrange",
                metadata=state_metadata,
                quality_score=0.25,
            )
        if current_price < float(previous.low):
            return SetupDecision(
                is_ready=True,
                reason="m30_breakaway_ready",
                setup_state="breaking_away",
                metadata={**state_metadata, "demand_upper": demand_upper},
                quality_score=0.9,
            )
        if (
            tradingview_confluence is not None
            and tradingview_confluence.is_active
            and tradingview_confluence.preferred_direction is BreakoutDirection.BEARISH
            and tradingview_confluence.setup_bonus > 0
            and progress >= 0.70
            and latest.close <= previous.close
        ):
            return SetupDecision(
                is_ready=True,
                reason="m30_tradingview_confluence_ready",
                setup_state="confluence_override",
                metadata=state_metadata,
                quality_score=0.75,
            )
        if state_scores["continuation_reacceptance"] >= 0.50 and h1_bias_strength >= 0.10:
            return SetupDecision(
                is_ready=False,
                reason="m30_setup_not_ready",
                setup_state="continuation_reacceptance",
                metadata=state_metadata,
                quality_score=min(state_scores["continuation_reacceptance"], 0.45),
            )

    return SetupDecision(
        is_ready=False,
        reason="m30_setup_not_ready",
        setup_state="approaching",
        metadata={"current_price": current_price, "h1_bias_strength": h1_bias_strength},
        quality_score=0.1,
    )


def evaluate_m10_setup(
    *,
    m10_candles: list[Candle],
    direction_decision: DirectionDecision,
    h4_context: H4Context,
    confirmation_decision: TriggerDecision | None = None,
    tradingview_confluence: TradingViewConfluence | None = None,
) -> SetupDecision:
    _require_timeframe(m10_candles, "M10", minimum=3)
    if confirmation_decision is not None and not confirmation_decision.is_ready:
        return SetupDecision(
            is_ready=False,
            reason="m10_confirmation_missing",
            setup_state="blocked",
            metadata={},
            quality_score=0.0,
        )
    if not direction_decision.is_valid or direction_decision.direction is None:
        return SetupDecision(
            is_ready=False,
            reason="m10_missing_direction_bias",
            setup_state="blocked",
            metadata={},
            quality_score=0.0,
        )

    latest = m10_candles[-1]
    previous = m10_candles[-2]
    current_price = float(latest.close)
    average_range = sum(float(candle.high) - float(candle.low) for candle in m10_candles[-3:]) / 3.0
    average_range = max(average_range, 1e-9)
    continuation_feedback: dict[str, object] = {}

    if direction_decision.direction is BreakoutDirection.BULLISH:
        demand_upper = max(zone[1] for zone in h4_context.demand_zones)
        supply_lower = min(zone[0] for zone in h4_context.supply_zones)
        progress = _range_progress(current_price, demand_upper, supply_lower)
        if latest.close > previous.close and latest.low <= float(previous.low) + (average_range * 0.60):
            return SetupDecision(
                is_ready=True,
                reason="m10_refinement_ready",
                setup_state="refining",
                metadata={"range_progress": progress, "average_range": average_range},
                quality_score=0.85,
            )
        continuation_retrace = _continuation_retrace_fraction(
            direction=direction_decision.direction,
            latest_close=float(latest.close),
            confirmation_decision=confirmation_decision,
        )
        if continuation_retrace is not None and continuation_retrace <= 0.35:
            return SetupDecision(
                is_ready=True,
                reason="m10_continuation_ready",
                setup_state="refining",
                metadata={
                    "range_progress": progress,
                    "average_range": average_range,
                    "continuation_retrace": continuation_retrace,
                },
                quality_score=0.8,
            )
        continuation_decision = _evaluate_m10_continuation(
            latest=latest,
            previous=previous,
            direction=direction_decision.direction,
            confirmation_decision=confirmation_decision,
            progress=progress,
        )
        if continuation_decision is not None:
            continuation_feedback = dict(continuation_decision.metadata)
            if continuation_decision.is_ready:
                return continuation_decision
        if (
            tradingview_confluence is not None
            and tradingview_confluence.is_active
            and tradingview_confluence.preferred_direction is BreakoutDirection.BULLISH
            and tradingview_confluence.setup_bonus > 0
            and latest.close >= previous.close
        ):
            return SetupDecision(
                is_ready=True,
                reason="m10_tradingview_refinement_ready",
                setup_state="confluence_override",
                metadata={"range_progress": progress, "average_range": average_range},
                quality_score=0.75,
            )
    else:
        demand_upper = max(zone[1] for zone in h4_context.demand_zones)
        supply_lower = min(zone[0] for zone in h4_context.supply_zones)
        progress = _range_progress(current_price, demand_upper, supply_lower)
        if latest.close < previous.close and latest.high >= float(previous.high) - (average_range * 0.60):
            return SetupDecision(
                is_ready=True,
                reason="m10_refinement_ready",
                setup_state="refining",
                metadata={"range_progress": progress, "average_range": average_range},
                quality_score=0.85,
            )
        continuation_retrace = _continuation_retrace_fraction(
            direction=direction_decision.direction,
            latest_close=float(latest.close),
            confirmation_decision=confirmation_decision,
        )
        if continuation_retrace is not None and continuation_retrace <= 0.35:
            return SetupDecision(
                is_ready=True,
                reason="m10_continuation_ready",
                setup_state="refining",
                metadata={
                    "range_progress": progress,
                    "average_range": average_range,
                    "continuation_retrace": continuation_retrace,
                },
                quality_score=0.8,
            )
        continuation_decision = _evaluate_m10_continuation(
            latest=latest,
            previous=previous,
            direction=direction_decision.direction,
            confirmation_decision=confirmation_decision,
            progress=progress,
        )
        if continuation_decision is not None:
            continuation_feedback = dict(continuation_decision.metadata)
            if continuation_decision.is_ready:
                return continuation_decision
        if (
            tradingview_confluence is not None
            and tradingview_confluence.is_active
            and tradingview_confluence.preferred_direction is BreakoutDirection.BEARISH
            and tradingview_confluence.setup_bonus > 0
            and latest.close <= previous.close
        ):
            return SetupDecision(
                is_ready=True,
                reason="m10_tradingview_refinement_ready",
                setup_state="confluence_override",
                metadata={"range_progress": progress, "average_range": average_range},
                quality_score=0.75,
            )

    return SetupDecision(
        is_ready=False,
        reason="m10_setup_not_ready",
        setup_state="refining",
        metadata={"current_price": current_price, "average_range": average_range, **continuation_feedback},
        quality_score=0.2,
    )


def _range_progress(current_price: float, demand_upper: float, supply_lower: float) -> float:
    if supply_lower <= demand_upper:
        return 0.5
    return (current_price - demand_upper) / (supply_lower - demand_upper)


def _m30_bias_strength(direction_decision: DirectionDecision) -> float:
    preferred = (
        direction_decision.bullish_contribution
        if direction_decision.direction is BreakoutDirection.BULLISH
        else direction_decision.bearish_contribution
    )
    opposing = (
        direction_decision.bearish_contribution
        if direction_decision.direction is BreakoutDirection.BULLISH
        else direction_decision.bullish_contribution
    )
    total = preferred + opposing
    if total <= 1e-9:
        return 0.0
    return max(preferred - opposing, 0.0) / total


def _best_m30_state(state_scores: dict[str, float]) -> tuple[str, float]:
    best_state = max(state_scores, key=state_scores.get)
    return best_state, state_scores[best_state]


def _evaluate_m30_state_scores(
    *,
    latest: Candle,
    previous: Candle,
    direction: BreakoutDirection,
    current_price: float,
    progress: float,
    near_zone_boundary: float,
    opposing_zone_boundary: float,
) -> tuple[dict[str, float], dict[str, object]]:
    candle_range = max(float(latest.high) - float(latest.low), 1e-9)
    previous_range = max(float(previous.high) - float(previous.low), 1e-9)
    zone_span = max(abs(opposing_zone_boundary - near_zone_boundary), 1e-9)
    zone_distance = abs(current_price - near_zone_boundary)
    zone_proximity = max(0.0, 1.0 - min(zone_distance / max(zone_span * 0.60, 1e-9), 1.0))

    if direction is BreakoutDirection.BULLISH:
        close_position = max(0.0, min((float(latest.close) - float(latest.low)) / candle_range, 1.0))
        rejection_wick = max(0.0, min((min(float(latest.open), float(latest.close)) - float(latest.low)) / candle_range, 1.0))
        directional_body = max(0.0, min((float(latest.close) - float(latest.open)) / candle_range, 1.0))
        breakaway_acceptance = max(0.0, min(1.0 - max(float(previous.high) - current_price, 0.0) / previous_range, 1.0))
        continuation_retrace = max(float(previous.close) - float(latest.close), 0.0) / previous_range
        pullback_damage = max(float(previous.close) - float(latest.low), 0.0) / max(previous_range * 1.20, 1e-9)
        structure_intact = float(latest.low) > float(previous.low)
        setup_kind = "bullish_continuation"
    else:
        close_position = max(0.0, min((float(latest.high) - float(latest.close)) / candle_range, 1.0))
        rejection_wick = max(0.0, min((float(latest.high) - max(float(latest.open), float(latest.close))) / candle_range, 1.0))
        directional_body = max(0.0, min((float(latest.open) - float(latest.close)) / candle_range, 1.0))
        breakaway_acceptance = max(0.0, min(1.0 - max(current_price - float(previous.low), 0.0) / previous_range, 1.0))
        continuation_retrace = max(float(latest.close) - float(previous.close), 0.0) / previous_range
        pullback_damage = max(float(latest.high) - float(previous.close), 0.0) / max(previous_range * 1.20, 1e-9)
        structure_intact = float(latest.high) < float(previous.high)
        setup_kind = "bearish_continuation"

    retrace_quality = max(0.0, 1.0 - min(continuation_retrace / 0.60, 1.0))
    reacceptance_strength = (close_position * 0.60) + (rejection_wick * 0.40)
    expansion_persistence = max(0.0, 1.0 - max(previous_range - candle_range, 0.0) / previous_range)
    retrace_damage = max(0.0, min((pullback_damage - 0.55) / 0.45, 1.0))

    continuation_score = (
        (retrace_quality * 0.28)
        + ((1.0 if structure_intact else 0.0) * 0.24)
        + (reacceptance_strength * 0.18)
        + (expansion_persistence * 0.12)
        + (breakaway_acceptance * 0.10)
        + (zone_proximity * 0.08)
        - (retrace_damage * 0.12)
    )
    continuation_score = max(0.0, min(continuation_score, 1.0))

    zone_rejection_score = max(
        0.0,
        min(
            (zone_proximity * 0.40)
            + (rejection_wick * 0.25)
            + (directional_body * 0.20)
            + (close_position * 0.15),
            1.0,
        ),
    )
    breakaway_score = max(
        0.0,
        min(
            (breakaway_acceptance * 0.45)
            + (close_position * 0.20)
            + (directional_body * 0.15)
            + (expansion_persistence * 0.10)
            + (max(progress if direction is BreakoutDirection.BULLISH else (1.0 - progress), 0.0) * 0.10),
            1.0,
        ),
    )

    return (
        {
            "zone_rejection": zone_rejection_score,
            "breakaway_acceptance": breakaway_score,
            "continuation_reacceptance": continuation_score,
        },
        {
            "continuation_retrace": continuation_retrace,
            "retrace_quality": retrace_quality,
            "reacceptance_strength": reacceptance_strength,
            "breakaway_acceptance": breakaway_acceptance,
            "zone_proximity": zone_proximity,
            "expansion_persistence": expansion_persistence,
            "retrace_damage": retrace_damage,
            "structure_intact": structure_intact,
            "close_position_score": close_position,
            "directional_body_score": directional_body,
            "rejection_wick_score": rejection_wick,
            "setup_kind": setup_kind,
        },
    )


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


def _evaluate_m10_continuation(
    *,
    latest: Candle,
    previous: Candle,
    direction: BreakoutDirection,
    confirmation_decision: TriggerDecision | None,
    progress: float,
) -> SetupDecision | None:
    if confirmation_decision is None or confirmation_decision.entry_price is None or confirmation_decision.invalidation_price is None:
        return None

    entry_price = float(confirmation_decision.entry_price)
    invalidation_price = float(confirmation_decision.invalidation_price)
    risk = abs(invalidation_price - entry_price)
    if risk <= 1e-9:
        return None

    candle_range = max(float(latest.high) - float(latest.low), 1e-9)
    previous_range = max(float(previous.high) - float(previous.low), 1e-9)
    continuation_retrace = _continuation_retrace_fraction(
        direction=direction,
        latest_close=float(latest.close),
        confirmation_decision=confirmation_decision,
    )
    if continuation_retrace is None:
        return None

    if direction is BreakoutDirection.BULLISH:
        structure_intact = float(latest.low) > invalidation_price
        close_position_score = max(0.0, min((float(latest.close) - float(latest.low)) / candle_range, 1.0))
        rejection_wick_score = max(0.0, min((min(float(latest.open), float(latest.close)) - float(latest.low)) / candle_range, 1.0))
        directional_body_score = max(0.0, min((float(latest.close) - float(latest.open)) / candle_range, 1.0))
        reclaim_score = max(0.0, 1.0 - max(float(previous.close) - float(latest.close), 0.0) / risk)
        setup_kind = "bullish_continuation"
    else:
        structure_intact = float(latest.high) < invalidation_price
        close_position_score = max(0.0, min((float(latest.high) - float(latest.close)) / candle_range, 1.0))
        rejection_wick_score = max(0.0, min((float(latest.high) - max(float(latest.open), float(latest.close))) / candle_range, 1.0))
        directional_body_score = max(0.0, min((float(latest.open) - float(latest.close)) / candle_range, 1.0))
        reclaim_score = max(0.0, 1.0 - max(float(latest.close) - float(previous.close), 0.0) / risk)
        setup_kind = "bearish_continuation"

    entry_distance_score = max(0.0, 1.0 - abs(float(latest.close) - entry_price) / max(risk * 0.95, 1e-9))
    retrace_quality = max(0.0, 1.0 - continuation_retrace / 0.90)
    structure_integrity = 1.0 if structure_intact else 0.0
    slope_persistence = max(0.0, 1.0 - abs(float(previous.close) - float(latest.close)) / max(risk * 1.5, 1e-9))
    expansion_decay = max(0.0, min((previous_range - candle_range) / previous_range, 1.0))
    expansion_persistence = 1.0 - expansion_decay
    retrace_damage = max(0.0, min((continuation_retrace - 0.55) / 0.45, 1.0))
    continuation_score = (
        (retrace_quality * 0.24)
        + (structure_integrity * 0.22)
        + (reclaim_score * 0.18)
        + (slope_persistence * 0.16)
        + (expansion_persistence * 0.12)
        + (entry_distance_score * 0.08)
        - (retrace_damage * 0.18)
    )
    metadata = {
        "range_progress": progress,
        "continuation_retrace": continuation_retrace,
        "continuation_score": continuation_score,
        "retrace_quality": retrace_quality,
        "structure_integrity": structure_integrity,
        "close_position_score": close_position_score,
        "directional_body_score": directional_body_score,
        "rejection_wick_score": rejection_wick_score,
        "reclaim_score": reclaim_score,
        "entry_distance_score": entry_distance_score,
        "slope_persistence": slope_persistence,
        "expansion_decay": expansion_decay,
        "expansion_persistence": expansion_persistence,
        "retrace_damage": retrace_damage,
        "structure_intact": structure_intact,
        "setup_kind": setup_kind,
    }
    is_ready = structure_intact and continuation_retrace <= 0.90 and continuation_score >= 0.46
    return SetupDecision(
        is_ready=is_ready,
        reason="m10_continuation_ready" if is_ready else "m10_setup_not_ready",
        setup_state="refining",
        metadata=metadata,
        quality_score=max(0.0, min(continuation_score, 1.0)) if is_ready else 0.2,
    )


def _require_timeframe(candles: list[Candle], timeframe: str, *, minimum: int) -> None:
    if len(candles) < minimum:
        raise ValueError(f"{timeframe} setup input requires at least {minimum} candles")
    if any(candle.timeframe != timeframe for candle in candles):
        raise ValueError(f"{timeframe} setup input received a different timeframe")
