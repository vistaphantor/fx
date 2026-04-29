from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Mapping

from src.strategy.breakout import BreakoutDirection
from src.strategy.volatility import VolatilityState


@dataclass(frozen=True, slots=True)
class ExecutionAssessment:
    is_tradeable: bool
    reason: str
    effective_entry: float
    effective_stop_distance: float
    effective_reward_distance: float
    effective_rr: float
    expected_slippage: float
    spread_pressure: float
    slippage_pressure: float
    stop_degradation: float
    capacity_penalty: float
    execution_penalty: float
    normalized_transaction_cost: float
    recommended_lot_multiplier: float
    effective_gain_remaining: float = 0.0
    effective_loss_if_failed: float = 0.0
    directional_tail_proxy: float = 0.0
    continuation_probability: float = 0.0
    continuation_mu: float = 0.0
    continuation_ev: float = 0.0
    dynamic_rr_floor: float = 1.5


def assess_market_order_execution(
    *,
    direction: BreakoutDirection,
    planned_entry: float,
    stop_loss: float,
    take_profit: float,
    current_bid: float,
    current_ask: float,
    spread: float,
    volatility_state: VolatilityState,
    requested_lot: float,
    campaign_exposure_pct: float = 0.0,
    continuation_context: Mapping[str, float | bool] | None = None,
) -> ExecutionAssessment:
    stop_distance = abs(float(planned_entry) - float(stop_loss))
    reward_distance = abs(float(take_profit) - float(planned_entry))
    if stop_distance <= 0 or reward_distance <= 0:
        return ExecutionAssessment(
            is_tradeable=False,
            reason="invalid_trade_levels",
            effective_entry=float(planned_entry),
            effective_stop_distance=0.0,
            effective_reward_distance=0.0,
            effective_rr=0.0,
            expected_slippage=0.0,
            spread_pressure=0.0,
            slippage_pressure=0.0,
            stop_degradation=0.0,
            capacity_penalty=0.0,
            execution_penalty=1.0,
            normalized_transaction_cost=0.0,
            recommended_lot_multiplier=0.0,
        )

    market_price = float(current_ask if direction is BreakoutDirection.BULLISH else current_bid)
    safe_spread = max(float(spread), 0.0)
    expected_slippage = max(safe_spread * 0.5, float(volatility_state.short_atr) * 0.12, 0.0)
    if direction is BreakoutDirection.BULLISH:
        effective_entry = max(float(planned_entry), market_price) + expected_slippage
    else:
        effective_entry = min(float(planned_entry), market_price) - expected_slippage

    effective_stop_distance = abs(effective_entry - float(stop_loss))
    effective_reward_distance = max(abs(float(take_profit) - effective_entry), 0.0)
    effective_rr = effective_reward_distance / max(effective_stop_distance, 1e-9)

    spread_pressure = safe_spread / max(stop_distance, 1e-9)
    slippage_pressure = expected_slippage / max(stop_distance, 1e-9)
    stop_degradation = max(0.0, (effective_stop_distance - stop_distance) / max(stop_distance, 1e-9))
    capacity_penalty = min(max(float(campaign_exposure_pct), 0.0) / 100.0, 1.0) * 0.5
    if float(requested_lot) > 0.01:
        capacity_penalty += min((float(requested_lot) - 0.01) / 10.0, 0.5)

    execution_penalty = (
        (spread_pressure * 0.45)
        + (slippage_pressure * 0.35)
        + (stop_degradation * 0.15)
        + (capacity_penalty * 0.45)
    )

    normalized_transaction_cost = (safe_spread + expected_slippage) / max(abs(market_price), 1e-9)
    effective_gain_remaining = effective_reward_distance / max(abs(market_price), 1e-9)
    effective_loss_if_failed = effective_stop_distance / max(abs(market_price), 1e-9)
    continuation_probability = 0.0
    continuation_mu = 0.0
    continuation_ev = 0.0
    dynamic_rr_floor = 1.5
    directional_tail_proxy = 0.0

    is_continuation_setup = bool(continuation_context and continuation_context.get("is_continuation_setup"))
    if is_continuation_setup:
        continuation_probability = _continuation_probability(
            continuation_context=continuation_context,
            execution_penalty=execution_penalty,
        )
        dynamic_rr_floor = _dynamic_rr_floor(
            continuation_probability=continuation_probability,
            trend_confidence=float(continuation_context.get("regime_confidence", 0.5) or 0.5),
            execution_penalty=execution_penalty,
        )
        directional_tail_proxy = _directional_tail_proxy(
            direction=direction,
            spread_pressure=spread_pressure,
            slippage_pressure=slippage_pressure,
            volatility_state=volatility_state,
        )
        continuation_mu = (
            continuation_probability * effective_gain_remaining
            - (1.0 - continuation_probability) * effective_loss_if_failed
            - normalized_transaction_cost
        )
        continuation_ev = (
            continuation_probability * effective_rr
            - (1.0 - continuation_probability)
            - execution_penalty
            - directional_tail_proxy
        )

    if spread_pressure > 0.33:
        return _reject(
            reason="spread_pressure_too_high",
            effective_entry=effective_entry,
            effective_stop_distance=effective_stop_distance,
            effective_reward_distance=effective_reward_distance,
            effective_rr=effective_rr,
            expected_slippage=expected_slippage,
            spread_pressure=spread_pressure,
            slippage_pressure=slippage_pressure,
            stop_degradation=stop_degradation,
            capacity_penalty=capacity_penalty,
            execution_penalty=execution_penalty,
            normalized_transaction_cost=normalized_transaction_cost,
            effective_gain_remaining=effective_gain_remaining,
            effective_loss_if_failed=effective_loss_if_failed,
            directional_tail_proxy=directional_tail_proxy,
            continuation_probability=continuation_probability,
            continuation_mu=continuation_mu,
            continuation_ev=continuation_ev,
            dynamic_rr_floor=dynamic_rr_floor,
        )

    if is_continuation_setup and effective_rr < 0.60:
        return _reject(
            reason="continuation_rr_below_disaster_floor",
            effective_entry=effective_entry,
            effective_stop_distance=effective_stop_distance,
            effective_reward_distance=effective_reward_distance,
            effective_rr=effective_rr,
            expected_slippage=expected_slippage,
            spread_pressure=spread_pressure,
            slippage_pressure=slippage_pressure,
            stop_degradation=stop_degradation,
            capacity_penalty=capacity_penalty,
            execution_penalty=execution_penalty,
            normalized_transaction_cost=normalized_transaction_cost,
            effective_gain_remaining=effective_gain_remaining,
            effective_loss_if_failed=effective_loss_if_failed,
            directional_tail_proxy=directional_tail_proxy,
            continuation_probability=continuation_probability,
            continuation_mu=continuation_mu,
            continuation_ev=continuation_ev,
            dynamic_rr_floor=dynamic_rr_floor,
        )

    if not is_continuation_setup and effective_rr < 1.5:
        return _reject(
            reason="execution_rr_degraded",
            effective_entry=effective_entry,
            effective_stop_distance=effective_stop_distance,
            effective_reward_distance=effective_reward_distance,
            effective_rr=effective_rr,
            expected_slippage=expected_slippage,
            spread_pressure=spread_pressure,
            slippage_pressure=slippage_pressure,
            stop_degradation=stop_degradation,
            capacity_penalty=capacity_penalty,
            execution_penalty=execution_penalty,
            normalized_transaction_cost=normalized_transaction_cost,
            effective_gain_remaining=effective_gain_remaining,
            effective_loss_if_failed=effective_loss_if_failed,
            directional_tail_proxy=directional_tail_proxy,
            continuation_probability=continuation_probability,
            continuation_mu=continuation_mu,
            continuation_ev=continuation_ev,
            dynamic_rr_floor=dynamic_rr_floor,
        )

    if is_continuation_setup and effective_rr < dynamic_rr_floor and continuation_ev <= 0.0:
        return _reject(
            reason="continuation_ev_negative",
            effective_entry=effective_entry,
            effective_stop_distance=effective_stop_distance,
            effective_reward_distance=effective_reward_distance,
            effective_rr=effective_rr,
            expected_slippage=expected_slippage,
            spread_pressure=spread_pressure,
            slippage_pressure=slippage_pressure,
            stop_degradation=stop_degradation,
            capacity_penalty=capacity_penalty,
            execution_penalty=execution_penalty,
            normalized_transaction_cost=normalized_transaction_cost,
            effective_gain_remaining=effective_gain_remaining,
            effective_loss_if_failed=effective_loss_if_failed,
            directional_tail_proxy=directional_tail_proxy,
            continuation_probability=continuation_probability,
            continuation_mu=continuation_mu,
            continuation_ev=continuation_ev,
            dynamic_rr_floor=dynamic_rr_floor,
        )

    if execution_penalty > 0.65:
        return _reject(
            reason="execution_penalty_too_high",
            effective_entry=effective_entry,
            effective_stop_distance=effective_stop_distance,
            effective_reward_distance=effective_reward_distance,
            effective_rr=effective_rr,
            expected_slippage=expected_slippage,
            spread_pressure=spread_pressure,
            slippage_pressure=slippage_pressure,
            stop_degradation=stop_degradation,
            capacity_penalty=capacity_penalty,
            execution_penalty=execution_penalty,
            normalized_transaction_cost=normalized_transaction_cost,
            effective_gain_remaining=effective_gain_remaining,
            effective_loss_if_failed=effective_loss_if_failed,
            directional_tail_proxy=directional_tail_proxy,
            continuation_probability=continuation_probability,
            continuation_mu=continuation_mu,
            continuation_ev=continuation_ev,
            dynamic_rr_floor=dynamic_rr_floor,
        )

    recommended_lot_multiplier = 1.0
    if execution_penalty > 0.35:
        recommended_lot_multiplier = 0.5
    elif execution_penalty > 0.18:
        recommended_lot_multiplier = 0.75

    return ExecutionAssessment(
        is_tradeable=True,
        reason="execution_approved",
        effective_entry=effective_entry,
        effective_stop_distance=effective_stop_distance,
        effective_reward_distance=effective_reward_distance,
        effective_rr=effective_rr,
        expected_slippage=expected_slippage,
        spread_pressure=spread_pressure,
        slippage_pressure=slippage_pressure,
        stop_degradation=stop_degradation,
        capacity_penalty=capacity_penalty,
        execution_penalty=execution_penalty,
        normalized_transaction_cost=normalized_transaction_cost,
        recommended_lot_multiplier=recommended_lot_multiplier,
        effective_gain_remaining=effective_gain_remaining,
        effective_loss_if_failed=effective_loss_if_failed,
        directional_tail_proxy=directional_tail_proxy,
        continuation_probability=continuation_probability,
        continuation_mu=continuation_mu,
        continuation_ev=continuation_ev,
        dynamic_rr_floor=dynamic_rr_floor,
    )


def _reject(
    *,
    reason: str,
    effective_entry: float,
    effective_stop_distance: float,
    effective_reward_distance: float,
    effective_rr: float,
    expected_slippage: float,
    spread_pressure: float,
    slippage_pressure: float,
    stop_degradation: float,
    capacity_penalty: float,
    execution_penalty: float,
    normalized_transaction_cost: float,
    effective_gain_remaining: float,
    effective_loss_if_failed: float,
    directional_tail_proxy: float,
    continuation_probability: float,
    continuation_mu: float,
    continuation_ev: float,
    dynamic_rr_floor: float,
) -> ExecutionAssessment:
    return ExecutionAssessment(
        is_tradeable=False,
        reason=reason,
        effective_entry=effective_entry,
        effective_stop_distance=effective_stop_distance,
        effective_reward_distance=effective_reward_distance,
        effective_rr=effective_rr,
        expected_slippage=expected_slippage,
        spread_pressure=spread_pressure,
        slippage_pressure=slippage_pressure,
        stop_degradation=stop_degradation,
        capacity_penalty=capacity_penalty,
        execution_penalty=execution_penalty,
        normalized_transaction_cost=normalized_transaction_cost,
        recommended_lot_multiplier=0.0,
        effective_gain_remaining=effective_gain_remaining,
        effective_loss_if_failed=effective_loss_if_failed,
        directional_tail_proxy=directional_tail_proxy,
        continuation_probability=continuation_probability,
        continuation_mu=continuation_mu,
        continuation_ev=continuation_ev,
        dynamic_rr_floor=dynamic_rr_floor,
    )


def _continuation_probability(
    *,
    continuation_context: Mapping[str, float | bool],
    execution_penalty: float,
) -> float:
    m15_quality = float(continuation_context.get("m15_quality", 0.0) or 0.0)
    m10_quality = float(continuation_context.get("m10_quality", 0.0) or 0.0)
    m5_quality = float(continuation_context.get("m5_quality", 0.0) or 0.0)
    range_expansion_ratio = float(continuation_context.get("range_expansion_ratio", 1.0) or 1.0)
    body_efficiency = float(continuation_context.get("body_efficiency", 0.5) or 0.5)
    regime_confidence = float(continuation_context.get("regime_confidence", 0.5) or 0.5)

    raw_score = (
        2.2 * (m15_quality - 0.5)
        + 2.0 * (m10_quality - 0.5)
        + 1.8 * (m5_quality - 0.5)
        + 1.0 * (range_expansion_ratio - 1.0)
        + 1.2 * (body_efficiency - 0.5)
        + 1.4 * (regime_confidence - 0.5)
        - 1.5 * execution_penalty
    )
    return _sigmoid(raw_score)


def _dynamic_rr_floor(
    *,
    continuation_probability: float,
    trend_confidence: float,
    execution_penalty: float,
) -> float:
    return _clamp(
        1.20
        - (0.45 * continuation_probability)
        - (0.20 * max(min(trend_confidence, 1.0), 0.0))
        + (0.25 * execution_penalty),
        0.60,
        1.50,
    )


def _directional_tail_proxy(
    *,
    direction: BreakoutDirection,
    spread_pressure: float,
    slippage_pressure: float,
    volatility_state: VolatilityState,
) -> float:
    shared_tail = (
        max(float(volatility_state.range_expansion_ratio) - 1.0, 0.0) * 0.006
        + max(1.0 - float(volatility_state.body_efficiency), 0.0) * 0.005
    )
    if direction is BreakoutDirection.BULLISH:
        return shared_tail + (slippage_pressure * 0.003)
    return shared_tail + (spread_pressure * 0.003)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    exp_value = exp(value)
    return exp_value / (1.0 + exp_value)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
