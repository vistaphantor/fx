from __future__ import annotations

from dataclasses import dataclass

from src.config import SymbolStrategyProfile
from src.market_data import Candle
from src.strategy.breakout import BreakoutDirection
from src.strategy.context import build_daily_context, build_h4_context
from src.strategy.direction import determine_h1_bias
from src.strategy.gap import evaluate_gap_context
from src.strategy.patterns import detect_three_drives
from src.strategy.regime import classify_regime
from src.strategy.risk import build_trade_levels
from src.strategy.scoring import score_market_sides
from src.strategy.setup import evaluate_m10_setup, evaluate_m30_setup
from src.strategy.tradingview_confluence import build_tradingview_confluence
from src.strategy.trigger import evaluate_m15_trigger, evaluate_m5_trigger
from src.strategy.volatility import build_volatility_state
from src.tradingview import TradingViewAlert


@dataclass(frozen=True, slots=True)
class TopDownTradePlan:
    is_trade: bool
    direction: BreakoutDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    objective_price: float
    reason: str
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class TopDownNoTrade:
    is_trade: bool
    reason: str
    failed_node: str
    metadata: dict[str, object]


def evaluate_top_down_decision_tree(
    *,
    symbol: str = "",
    d1_candles: list[Candle],
    h4_candles: list[Candle],
    h1_candles: list[Candle],
    m30_candles: list[Candle],
    m15_candles: list[Candle],
    m10_candles: list[Candle] | None = None,
    m5_candles: list[Candle] | None = None,
    risk_buffer: float,
    tradingview_alert: TradingViewAlert | None = None,
    strategy_profile: SymbolStrategyProfile | None = None,
) -> TopDownTradePlan | TopDownNoTrade:
    current_price = float((m5_candles or m15_candles)[-1].close)
    profile = strategy_profile or _default_strategy_profile(symbol)
    daily_context = build_daily_context(d1_candles, current_price=current_price)
    h4_context = build_h4_context(h4_candles)
    gap_decision = evaluate_gap_context(h4_candles=h4_candles, m15_candles=m15_candles)
    incoming_tradingview_confluence = build_tradingview_confluence(
        symbol=symbol,
        direction=tradingview_alert.direction if tradingview_alert and tradingview_alert.direction else BreakoutDirection.BULLISH,
        gap_decision=gap_decision,
        alert=tradingview_alert,
    )

    direction_decision = determine_h1_bias(
        h1_candles=h1_candles,
        daily_context=daily_context,
        h4_context=h4_context,
        gap_decision=gap_decision,
        tradingview_confluence=incoming_tradingview_confluence,
    )
    if not direction_decision.is_valid or direction_decision.direction is None:
        return _no_trade(direction_decision.reason, "h1_direction", **direction_decision.metadata)
    tradingview_confluence = build_tradingview_confluence(
        symbol=symbol,
        direction=direction_decision.direction,
        gap_decision=gap_decision,
        alert=tradingview_alert,
    )

    setup_decision = evaluate_m30_setup(
        m30_candles=m30_candles,
        direction_decision=direction_decision,
        h4_context=h4_context,
        tradingview_confluence=tradingview_confluence,
    )
    if not setup_decision.is_ready:
        return _no_trade(setup_decision.reason, "m30_setup", **setup_decision.metadata)

    trigger_decision = evaluate_m15_trigger(
        m15_candles=m15_candles,
        setup_decision=setup_decision,
        direction_decision=direction_decision,
    )
    if not trigger_decision.is_ready or trigger_decision.entry_price is None or trigger_decision.invalidation_price is None:
        return _no_trade(trigger_decision.reason, "m15_trigger", **trigger_decision.metadata)
    if not m10_candles:
        return _no_trade("m10_data_missing", "m10_setup")

    refinement_decision = evaluate_m10_setup(
        m10_candles=m10_candles,
        direction_decision=direction_decision,
        h4_context=h4_context,
        confirmation_decision=trigger_decision,
        tradingview_confluence=tradingview_confluence,
    )
    if not refinement_decision.is_ready:
        return _no_trade(refinement_decision.reason, "m10_setup", **refinement_decision.metadata)
    if not m5_candles:
        return _no_trade("m5_data_missing", "m5_trigger")

    execution_decision = evaluate_m5_trigger(
        m5_candles=m5_candles,
        setup_decision=refinement_decision,
        direction_decision=direction_decision,
        confirmation_decision=trigger_decision,
    )
    if not execution_decision.is_ready or execution_decision.entry_price is None or execution_decision.invalidation_price is None:
        return _no_trade(execution_decision.reason, "m5_trigger", **execution_decision.metadata)

    volatility_state = build_volatility_state(candles=m15_candles[-min(12, len(m15_candles)) :])
    regime_state = classify_regime(
        h1_candles=h1_candles,
        m30_candles=m30_candles,
        m15_candles=m15_candles,
        volatility_state=volatility_state,
        gap_decision=gap_decision,
    )

    score_decision = score_market_sides(
        bullish_inputs=_bullish_inputs(
            daily_context=daily_context,
            h4_context=h4_context,
            direction_decision=direction_decision,
            setup_decision=setup_decision,
            trigger_decision=execution_decision,
            gap_decision=gap_decision,
            tradingview_confluence=tradingview_confluence,
            current_price=current_price,
        ),
        bearish_inputs=_bearish_inputs(
            daily_context=daily_context,
            h4_context=h4_context,
            direction_decision=direction_decision,
            setup_decision=setup_decision,
            trigger_decision=execution_decision,
            gap_decision=gap_decision,
            tradingview_confluence=tradingview_confluence,
            current_price=current_price,
        ),
        uncertainty_inputs=_uncertainty_inputs(
            direction_decision=direction_decision,
            volatility_state=volatility_state,
            tradingview_confluence=tradingview_confluence,
        ),
        expected_move_multiple=max(trigger_decision.expected_move_multiple, profile.minimum_expected_move_multiple),
        min_expected_move_multiple=profile.minimum_expected_move_multiple,
        base_threshold=profile.min_edge_threshold,
        max_uncertainty_threshold=profile.max_uncertainty_threshold,
        preferred_direction=direction_decision.direction,
        regime_state=regime_state,
    )
    if not score_decision.is_tradeable:
        return _no_trade(
            "quant_edge_insufficient",
            "scoring",
            regime_state=regime_state,
            volatility_state=volatility_state,
            score_decision=score_decision,
            symbol_profile=profile,
        )

    if _daily_objective_exhausted(
        direction_decision.direction,
        daily_context,
        volatility_state=volatility_state,
        setup_quality=setup_decision.quality_score,
        trigger_quality=execution_decision.quality_score,
    ):
        return _no_trade(
            "d1_objective_exhausted",
            "d1_context",
            current_price=daily_context.current_price,
            objective_high=daily_context.objective_high,
            objective_low=daily_context.objective_low,
            hybrid_objective_high=_hybrid_objective_high(daily_context),
            hybrid_objective_low=_hybrid_objective_low(daily_context),
            chosen_direction=direction_decision.direction,
            daily_expansion_ratio=daily_context.daily_expansion_ratio,
            remaining_objective_distance=_remaining_objective_distance(direction_decision.direction, daily_context),
        )

    pattern_decision = detect_three_drives(
        candles=m15_candles,
        reference_levels=tuple(h4_context.volume_profile_levels) + (h4_context.previous_session_low, h4_context.previous_session_high),
    )

    execution_candles = m5_candles
    structure_low = min(candle.low for candle in execution_candles[-3:])
    structure_high = max(candle.high for candle in execution_candles[-3:])
    levels = build_trade_levels(
        entry_price=execution_decision.entry_price,
        direction=direction_decision.direction,
        retest_structure_low=structure_low,
        retest_structure_high=structure_high,
        buffer=risk_buffer,
        candle_timestamp=execution_candles[-1].timestamp,
    )

    objective_price = _directional_objective_price(direction_decision.direction, daily_context)
    is_continuation_setup = _is_continuation_setup(
        setup_decision=setup_decision,
        trigger_decision=trigger_decision,
        refinement_decision=refinement_decision,
        execution_decision=execution_decision,
    )

    return TopDownTradePlan(
        is_trade=True,
        direction=levels.direction,
        entry_price=levels.entry_price,
        stop_loss=levels.stop_loss,
        take_profit=levels.take_profit,
        objective_price=objective_price,
        reason="top_down_trade_plan_ready",
        metadata={
            "daily_context": daily_context,
            "h4_context": h4_context,
            "gap_decision": gap_decision,
            "tradingview_confluence": tradingview_confluence,
            "direction_reason": direction_decision.reason,
            "setup_reason": setup_decision.reason,
            "trigger_reason": trigger_decision.reason,
            "refinement_reason": refinement_decision.reason,
            "execution_reason": execution_decision.reason,
            "pattern_reason": pattern_decision.reason,
            "pattern_confluence_score": pattern_decision.confluence_score,
            "volatility_state": volatility_state,
            "regime_state": regime_state,
            "score_decision": score_decision,
            "symbol_profile": profile,
            "is_continuation_setup": is_continuation_setup,
            "m15_quality": trigger_decision.quality_score,
            "m10_quality": refinement_decision.quality_score,
            "m5_quality": execution_decision.quality_score,
            "range_expansion_ratio": volatility_state.range_expansion_ratio,
            "body_efficiency": volatility_state.body_efficiency,
            "regime_confidence": regime_state.confidence,
        },
    )


def _no_trade(reason: str, failed_node: str, **metadata) -> TopDownNoTrade:
    return TopDownNoTrade(
        is_trade=False,
        reason=reason,
        failed_node=failed_node,
        metadata=metadata,
    )


def _is_continuation_setup(*, setup_decision, trigger_decision, refinement_decision, execution_decision) -> bool:
    continuation_setup_states = {"continuation_reacceptance"}
    continuation_reasons = {
        "m15_trigger_still_valid",
        "m10_continuation_ready",
        "m5_continuation_ready",
    }
    if getattr(setup_decision, "setup_state", None) in continuation_setup_states:
        return True
    if getattr(trigger_decision, "reason", None) in continuation_reasons:
        return True
    if getattr(refinement_decision, "reason", None) in continuation_reasons:
        return True
    if getattr(execution_decision, "reason", None) in continuation_reasons:
        return True
    return False


def _default_strategy_profile(symbol: str) -> SymbolStrategyProfile:
    normalized_symbol = symbol.strip().upper() if symbol else "XAUUSD"
    if normalized_symbol == "EURJPY":
        return SymbolStrategyProfile(
            symbol="EURJPY",
            min_edge_threshold=1.8,
            max_uncertainty_threshold=1.0,
            minimum_expected_move_multiple=1.6,
            add_on_edge_multiplier=1.35,
            trend_regime_weight=1.0,
            compression_regime_weight=0.8,
            breakeven_distance=0.15,
            campaign_base_add_trigger_r=1.5,
            campaign_add_trigger_floor_r=1.25,
            campaign_add_trigger_ceiling_r=1.75,
        )
    return SymbolStrategyProfile(
        symbol="XAUUSD",
        min_edge_threshold=0.0,
        max_uncertainty_threshold=10.0,
        minimum_expected_move_multiple=1.0,
        add_on_edge_multiplier=1.25,
        trend_regime_weight=1.1,
        compression_regime_weight=0.75,
        breakeven_distance=1.5,
        campaign_base_add_trigger_r=1.5,
        campaign_add_trigger_floor_r=1.25,
        campaign_add_trigger_ceiling_r=1.75,
    )


def _bullish_inputs(
    *,
    daily_context,
    h4_context,
    direction_decision,
    setup_decision,
    trigger_decision,
    gap_decision,
    tradingview_confluence,
    current_price: float,
) -> dict[str, float]:
    demand_distance = _nearest_distance(current_price, h4_context.demand_zones, side="demand")
    supply_distance = _nearest_distance(current_price, h4_context.supply_zones, side="supply")
    demand_score = 1.2 if demand_distance is None else max(0.0, 1.2 - demand_distance / max(h4_context.session_range * 0.2, 1.0))
    supply_penalty = 0.8 if supply_distance is not None and supply_distance < max(h4_context.session_range * 0.1, 1.0) else 0.0
    return {
        "location": max(0.0, (1.0 - daily_context.range_position) * 2.0 + demand_score - supply_penalty),
        "momentum": max(direction_decision.bullish_contribution * 0.35, 0.0),
        "setup": setup_decision.quality_score if direction_decision.direction is BreakoutDirection.BULLISH else 0.15,
        "trigger": trigger_decision.quality_score if direction_decision.direction is BreakoutDirection.BULLISH else 0.1,
        "gap": 0.8 if gap_decision.preferred_trade_direction is BreakoutDirection.BULLISH else 0.1,
        "external": float(getattr(tradingview_confluence, "direction_bonus", 0)) * 0.25
        + float(getattr(tradingview_confluence, "setup_bonus", 0)) * 0.2,
    }


def _bearish_inputs(
    *,
    daily_context,
    h4_context,
    direction_decision,
    setup_decision,
    trigger_decision,
    gap_decision,
    tradingview_confluence,
    current_price: float,
) -> dict[str, float]:
    demand_distance = _nearest_distance(current_price, h4_context.demand_zones, side="demand")
    supply_distance = _nearest_distance(current_price, h4_context.supply_zones, side="supply")
    supply_score = 1.2 if supply_distance is None else max(0.0, 1.2 - supply_distance / max(h4_context.session_range * 0.2, 1.0))
    demand_penalty = 0.8 if demand_distance is not None and demand_distance < max(h4_context.session_range * 0.1, 1.0) else 0.0
    return {
        "location": max(0.0, daily_context.range_position * 2.0 + supply_score - demand_penalty),
        "momentum": max(direction_decision.bearish_contribution * 0.35, 0.0),
        "setup": setup_decision.quality_score if direction_decision.direction is BreakoutDirection.BEARISH else 0.15,
        "trigger": trigger_decision.quality_score if direction_decision.direction is BreakoutDirection.BEARISH else 0.1,
        "gap": 0.8 if gap_decision.preferred_trade_direction is BreakoutDirection.BEARISH else 0.1,
        "external": float(getattr(tradingview_confluence, "direction_bonus", 0)) * 0.25
        if getattr(tradingview_confluence, "preferred_direction", None) is BreakoutDirection.BEARISH
        else 0.0,
    }


def _uncertainty_inputs(*, direction_decision, volatility_state, tradingview_confluence) -> dict[str, float]:
    contribution_gap = abs(direction_decision.bullish_contribution - direction_decision.bearish_contribution)
    timeframe_conflict = 1.0 / (1.0 + contribution_gap)
    volatility_instability = abs(volatility_state.range_expansion_ratio - 1.0) * 0.25
    external_conflict = float(getattr(tradingview_confluence, "direction_penalty", 0)) * 0.2
    return {
        "timeframe_conflict": timeframe_conflict,
        "volatility_instability": volatility_instability,
        "external_conflict": external_conflict,
    }


def _nearest_distance(current_price: float, zones: tuple[tuple[float, float], ...], *, side: str) -> float | None:
    distances = []
    for lower, upper in zones:
        if lower <= current_price <= upper:
            return 0.0
        if side == "demand" and current_price >= upper:
            distances.append(current_price - upper)
        if side == "supply" and current_price <= lower:
            distances.append(lower - current_price)
    return min(distances) if distances else None


def _daily_objective_exhausted(
    direction: BreakoutDirection,
    daily_context,
    *,
    volatility_state,
    setup_quality: float,
    trigger_quality: float,
) -> bool:
    continuation_weakening = (
        float(trigger_quality) < 0.55
        or float(setup_quality) < 0.55
        or float(volatility_state.body_efficiency) < 0.45
        or float(volatility_state.range_expansion_ratio) < 0.90
    )
    if not continuation_weakening:
        return False

    overextended = float(daily_context.daily_expansion_ratio) >= 1.10
    if not overextended:
        return False

    if direction is BreakoutDirection.BULLISH:
        return float(daily_context.current_price) >= _hybrid_objective_high(daily_context)
    return float(daily_context.current_price) <= _hybrid_objective_low(daily_context)


def _remaining_objective_distance(direction: BreakoutDirection, daily_context) -> float:
    if direction is BreakoutDirection.BULLISH:
        return _hybrid_objective_high(daily_context) - float(daily_context.current_price)
    return float(daily_context.current_price) - _hybrid_objective_low(daily_context)


def _directional_objective_price(direction: BreakoutDirection, daily_context) -> float:
    objective_extension = max(float(daily_context.adr) * 0.25, 1.0)
    if direction is BreakoutDirection.BULLISH:
        base_objective = _hybrid_objective_high(daily_context)
        if float(daily_context.current_price) > base_objective:
            return base_objective + objective_extension
        return base_objective
    base_objective = _hybrid_objective_low(daily_context)
    if float(daily_context.current_price) < base_objective:
        return base_objective - objective_extension
    return base_objective


def _hybrid_objective_high(daily_context) -> float:
    return max(float(daily_context.previous_day_high), float(daily_context.current_day_projection_high))


def _hybrid_objective_low(daily_context) -> float:
    return min(float(daily_context.previous_day_low), float(daily_context.current_day_projection_low))
