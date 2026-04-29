from __future__ import annotations

from dataclasses import dataclass

from src.config import SymbolStrategyProfile
from src.market_data import Candle
from src.strategy.breakout import BreakoutDirection
from src.strategy.context import build_h4_context
from src.strategy.direction import DirectionDecision
from src.strategy.risk import build_trade_levels
from src.strategy.setup import SetupDecision, evaluate_m10_setup
from src.strategy.trigger import TriggerDecision, evaluate_m15_trigger, evaluate_m5_trigger


@dataclass(frozen=True, slots=True)
class CampaignAddDecision:
    is_ready: bool
    reason: str
    direction: BreakoutDirection | None
    entry_price: float | None
    stop_loss: float | None
    take_profit: float | None
    metadata: dict[str, object]
    quality_score: float = 0.0
    lot_multiplier: float = 1.0
    quant_state: str = "neutral"


def evaluate_campaign_add(
    *,
    symbol: str,
    live_input,
    direction: BreakoutDirection,
    risk_buffer: float,
    latest_trade_r_multiple: float,
    continuation_edge: float | None = None,
    continuation_threshold: float | None = None,
    strategy_profile: SymbolStrategyProfile | None = None,
    quant_decision=None,
) -> CampaignAddDecision:
    h4_candles = getattr(live_input, "h4_candles", None) or []
    m15_candles = getattr(live_input, "m15_candles", None) or []
    m10_candles = getattr(live_input, "m10_candles", None) or []
    m5_candles = getattr(live_input, "m5_candles", None) or []

    if len(m15_candles) < 3:
        return CampaignAddDecision(
            is_ready=False,
            reason="campaign_add_m15_data_missing",
            direction=direction,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            metadata={},
        )
    if len(m10_candles) < 3:
        return CampaignAddDecision(
            is_ready=False,
            reason="campaign_add_m10_data_missing",
            direction=direction,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            metadata={},
        )
    if len(m5_candles) < 3:
        return CampaignAddDecision(
            is_ready=False,
            reason="campaign_add_m5_data_missing",
            direction=direction,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            metadata={},
        )

    current_price = float(m5_candles[-1].close)
    synthetic_direction = DirectionDecision(
        is_valid=True,
        direction=direction,
        reason="campaign_direction_carryover",
        metadata={"current_price": current_price, "symbol": symbol},
        bullish_contribution=1.0 if direction is BreakoutDirection.BULLISH else 0.0,
        bearish_contribution=1.0 if direction is BreakoutDirection.BEARISH else 0.0,
    )
    synthetic_setup = SetupDecision(
        is_ready=True,
        reason="campaign_context_ready",
        setup_state="campaign_continuation",
        metadata={"current_price": current_price},
        quality_score=0.75,
    )

    m15_decision = evaluate_m15_trigger(
        m15_candles=m15_candles,
        setup_decision=synthetic_setup,
        direction_decision=synthetic_direction,
    )
    if not m15_decision.is_ready:
        return CampaignAddDecision(
            is_ready=False,
            reason="campaign_add_m15_not_ready",
            direction=direction,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            metadata=dict(m15_decision.metadata),
        )

    try:
        h4_context = build_h4_context(h4_candles)
    except ValueError:
        return CampaignAddDecision(
            is_ready=False,
            reason="campaign_add_h4_context_missing",
            direction=direction,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            metadata={},
        )

    m10_decision = evaluate_m10_setup(
        m10_candles=m10_candles,
        direction_decision=synthetic_direction,
        h4_context=h4_context,
        confirmation_decision=m15_decision,
        tradingview_confluence=None,
    )
    if not m10_decision.is_ready:
        return CampaignAddDecision(
            is_ready=False,
            reason="campaign_add_m10_not_ready",
            direction=direction,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            metadata=dict(m10_decision.metadata),
        )

    m5_decision = evaluate_m5_trigger(
        m5_candles=m5_candles,
        setup_decision=m10_decision,
        direction_decision=synthetic_direction,
        confirmation_decision=m15_decision,
    )
    if not m5_decision.is_ready or m5_decision.entry_price is None or m5_decision.invalidation_price is None:
        return CampaignAddDecision(
            is_ready=False,
            reason="campaign_add_m5_not_ready",
            direction=direction,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            metadata=dict(m5_decision.metadata),
        )

    add_score = _campaign_add_score(
        m15_quality=m15_decision.quality_score,
        m10_quality=m10_decision.quality_score,
        m5_quality=m5_decision.quality_score,
        latest_trade_r_multiple=latest_trade_r_multiple,
        continuation_edge=continuation_edge,
        continuation_threshold=continuation_threshold,
        trigger_r=_campaign_base_trigger_r(strategy_profile),
    )
    acceleration_score = _momentum_acceleration_score(
        m10_quality=m10_decision.quality_score,
        m5_quality=m5_decision.quality_score,
        range_expansion=_range_expansion_score(m15_decision.expected_move_multiple),
        body_efficiency=_body_efficiency_score(
            m10_metadata=m10_decision.metadata,
            m5_metadata=m5_decision.metadata,
        ),
    )
    acceleration_bonus = min(acceleration_score * 0.25, 0.25)
    execution_penalty = _execution_penalty(
        m10_metadata=m10_decision.metadata,
        m5_metadata=m5_decision.metadata,
    )
    volatility_penalty = _volatility_penalty(m15_quality=m15_decision.quality_score)
    adaptive_trigger_r = _adaptive_add_trigger_r(
        base_trigger_r=_campaign_base_trigger_r(strategy_profile),
        accel_bonus=acceleration_bonus,
        execution_penalty=execution_penalty,
        volatility_penalty=volatility_penalty,
        floor_r=_campaign_trigger_floor_r(strategy_profile),
        ceiling_r=_campaign_trigger_ceiling_r(strategy_profile),
    )
    if latest_trade_r_multiple < adaptive_trigger_r:
        return CampaignAddDecision(
            is_ready=False,
            reason="campaign_add_progress_below_trigger",
            direction=direction,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            metadata={
                "latest_trade_r_multiple": latest_trade_r_multiple,
                "adaptive_add_trigger_r": adaptive_trigger_r,
                "acceleration_score": acceleration_score,
                "execution_penalty": execution_penalty,
                "volatility_penalty": volatility_penalty,
            },
        )

    threshold = _adaptive_add_threshold(
        base_threshold=_campaign_add_threshold(strategy_profile),
        protected_bonus=_protected_campaign_bonus(latest_trade_r_multiple),
        accel_bonus=min(acceleration_score * 0.06, 0.06),
        execution_penalty=min(execution_penalty * 0.5, 0.08),
        quant_penalty=0.0,
        lower_bound=0.45,
        upper_bound=0.75,
    )
    if add_score < threshold:
        return CampaignAddDecision(
            is_ready=False,
            reason="campaign_add_edge_too_weak",
            direction=direction,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            metadata={
                "add_score": add_score,
                "add_threshold": threshold,
                "m15_quality": m15_decision.quality_score,
                "m10_quality": m10_decision.quality_score,
                "m5_quality": m5_decision.quality_score,
                "latest_trade_r_multiple": latest_trade_r_multiple,
                "continuation_edge": continuation_edge,
                "continuation_threshold": continuation_threshold,
                "adaptive_add_trigger_r": adaptive_trigger_r,
                "acceleration_score": acceleration_score,
                "execution_penalty": execution_penalty,
                "volatility_penalty": volatility_penalty,
            },
        )

    quant_policy = _campaign_quant_policy(
        direction=direction,
        quant_decision=quant_decision,
        latest_trade_r_multiple=latest_trade_r_multiple,
        add_score=add_score,
        add_threshold=threshold,
    )
    if quant_policy["hard_block"]:
        return CampaignAddDecision(
            is_ready=False,
            reason="campaign_add_quant_blocked",
            direction=direction,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            metadata={
                "quant_action": quant_policy["quant_action"],
                "quant_cvar": quant_policy["cvar"],
                "quant_drawdown_dampener": quant_policy["drawdown_dampener"],
            },
            quant_state="hard_block",
        )

    structure_low = min(float(candle.low) for candle in m5_candles[-3:])
    structure_high = max(float(candle.high) for candle in m5_candles[-3:])
    levels = build_trade_levels(
        entry_price=float(m5_decision.entry_price),
        direction=direction,
        retest_structure_low=structure_low,
        retest_structure_high=structure_high,
        buffer=risk_buffer,
        candle_timestamp=m5_candles[-1].timestamp,
        omega_t=add_score,
    )
    return CampaignAddDecision(
        is_ready=True,
        reason="campaign_add_quant_reduced" if quant_policy["lot_multiplier"] < 1.0 else "campaign_add_ready",
        direction=levels.direction,
        entry_price=levels.entry_price,
        stop_loss=levels.stop_loss,
        take_profit=levels.take_profit,
        metadata={
            "add_score": add_score,
            "add_threshold": threshold,
            "m15_reason": m15_decision.reason,
            "m10_reason": m10_decision.reason,
            "m5_reason": m5_decision.reason,
            "m15_quality": m15_decision.quality_score,
            "m10_quality": m10_decision.quality_score,
            "m5_quality": m5_decision.quality_score,
            "latest_trade_r_multiple": latest_trade_r_multiple,
            "continuation_edge": continuation_edge,
            "continuation_threshold": continuation_threshold,
            "adaptive_add_trigger_r": adaptive_trigger_r,
            "acceleration_score": acceleration_score,
            "execution_penalty": execution_penalty,
            "volatility_penalty": volatility_penalty,
            "quant_action": quant_policy["quant_action"],
            "quant_cvar": quant_policy["cvar"],
            "quant_drawdown_dampener": quant_policy["drawdown_dampener"],
        },
        quality_score=add_score,
        lot_multiplier=quant_policy["lot_multiplier"],
        quant_state=quant_policy["quant_state"],
    )


def _campaign_add_score(
    *,
    m15_quality: float,
    m10_quality: float,
    m5_quality: float,
    latest_trade_r_multiple: float,
    continuation_edge: float | None,
    continuation_threshold: float | None,
    trigger_r: float,
) -> float:
    progress_bonus = max(min((float(latest_trade_r_multiple) - float(trigger_r)) / max(float(trigger_r), 1e-9), 1.0), 0.0)
    continuation_bonus = 0.0
    if continuation_edge is not None and continuation_threshold is not None and continuation_threshold > 0:
        continuation_bonus = max(min(float(continuation_edge) / float(continuation_threshold), 1.5), 0.0) / 1.5
    return (
        (max(float(m15_quality), 0.0) * 0.24)
        + (max(float(m10_quality), 0.0) * 0.34)
        + (max(float(m5_quality), 0.0) * 0.32)
        + (progress_bonus * 0.05)
        + (continuation_bonus * 0.05)
    )


def _campaign_add_threshold(strategy_profile: SymbolStrategyProfile | None) -> float:
    if strategy_profile is None:
        return 0.56
    multiplier = max(float(strategy_profile.add_on_edge_multiplier), 1.0)
    return 0.50 + min((multiplier - 1.0) * 0.25, 0.10)


def _campaign_base_trigger_r(strategy_profile: SymbolStrategyProfile | None) -> float:
    if strategy_profile is None:
        return 1.5
    return float(strategy_profile.campaign_base_add_trigger_r)


def _campaign_trigger_floor_r(strategy_profile: SymbolStrategyProfile | None) -> float:
    if strategy_profile is None:
        return 1.25
    return float(strategy_profile.campaign_add_trigger_floor_r)


def _campaign_trigger_ceiling_r(strategy_profile: SymbolStrategyProfile | None) -> float:
    if strategy_profile is None:
        return 1.75
    return float(strategy_profile.campaign_add_trigger_ceiling_r)


def _adaptive_add_trigger_r(
    *,
    base_trigger_r: float,
    accel_bonus: float,
    execution_penalty: float,
    volatility_penalty: float,
    floor_r: float,
    ceiling_r: float,
) -> float:
    return round(
        _clamp(
            float(base_trigger_r) - float(accel_bonus) + float(execution_penalty) + float(volatility_penalty),
            float(floor_r),
            float(ceiling_r),
        ),
        2,
    )


def _adaptive_add_threshold(
    *,
    base_threshold: float,
    protected_bonus: float,
    accel_bonus: float,
    execution_penalty: float,
    quant_penalty: float,
    lower_bound: float,
    upper_bound: float,
) -> float:
    return round(
        _clamp(
            float(base_threshold)
            - float(protected_bonus)
            - float(accel_bonus)
            + float(execution_penalty)
            + float(quant_penalty),
            float(lower_bound),
            float(upper_bound),
        ),
        4,
    )


def _momentum_acceleration_score(*, m10_quality: float, m5_quality: float, range_expansion: float, body_efficiency: float) -> float:
    m10_delta = _clamp((float(m10_quality) - 0.5) / 0.5, 0.0, 1.0)
    m5_delta = _clamp((float(m5_quality) - 0.5) / 0.5, 0.0, 1.0)
    return _clamp(
        (0.35 * m10_delta)
        + (0.30 * m5_delta)
        + (0.20 * float(range_expansion))
        + (0.15 * float(body_efficiency)),
        0.0,
        1.0,
    )


def _range_expansion_score(expected_move_multiple: float) -> float:
    return _clamp((float(expected_move_multiple) - 1.0) / 2.0, 0.0, 1.0)


def _body_efficiency_score(*, m10_metadata: dict[str, object], m5_metadata: dict[str, object]) -> float:
    m10_body = float(m10_metadata.get("directional_body_score", 0.5) or 0.5)
    m5_body = float(m5_metadata.get("directional_body_score", 0.5) or 0.5)
    return _clamp((m10_body + m5_body) / 2.0, 0.0, 1.0)


def _execution_penalty(*, m10_metadata: dict[str, object], m5_metadata: dict[str, object]) -> float:
    m10_retrace = float(m10_metadata.get("continuation_retrace", 0.0) or 0.0)
    m5_retrace = float(m5_metadata.get("continuation_retrace", 0.0) or 0.0)
    m10_entry_distance = 1.0 - float(m10_metadata.get("entry_distance_score", 0.7) or 0.7)
    m5_entry_distance = 1.0 - float(m5_metadata.get("entry_distance_score", 0.7) or 0.7)
    penalty = (
        (max(m10_retrace - 0.25, 0.0) * 0.18)
        + (max(m5_retrace - 0.25, 0.0) * 0.18)
        + (max(m10_entry_distance, 0.0) * 0.08)
        + (max(m5_entry_distance, 0.0) * 0.08)
    )
    return _clamp(penalty, 0.0, 0.25)


def _volatility_penalty(*, m15_quality: float) -> float:
    return _clamp((1.0 - float(m15_quality)) * 0.10, 0.0, 0.10)


def _protected_campaign_bonus(latest_trade_r_multiple: float) -> float:
    if latest_trade_r_multiple >= 2.0:
        return 0.08
    if latest_trade_r_multiple >= 1.25:
        return 0.04
    return 0.0


def _clamp(value: float, lower_bound: float, upper_bound: float) -> float:
    return max(float(lower_bound), min(float(upper_bound), float(value)))


def _campaign_quant_policy(*, direction: BreakoutDirection, quant_decision, latest_trade_r_multiple: float, add_score: float, add_threshold: float) -> dict[str, object]:
    if quant_decision is None:
        return {
            "hard_block": False,
            "lot_multiplier": 1.0,
            "quant_state": "neutral",
            "quant_action": None,
            "cvar": 0.0,
            "drawdown_dampener": 1.0,
        }

    quant_action = int(getattr(quant_decision, "action", 0) or 0)
    cvar = float(getattr(quant_decision, "cvar", 0.0) or 0.0)
    drawdown_dampener = float(getattr(quant_decision, "drawdown_dampener", 1.0) or 1.0)
    direction_flip = (
        (direction is BreakoutDirection.BULLISH and quant_action == -1)
        or (direction is BreakoutDirection.BEARISH and quant_action == 1)
    )
    if direction_flip or cvar > 0.01 or drawdown_dampener < 0.35:
        return {
            "hard_block": True,
            "lot_multiplier": 0.0,
            "quant_state": "hard_block",
            "quant_action": quant_action,
            "cvar": cvar,
            "drawdown_dampener": drawdown_dampener,
        }

    if quant_action == 0 and latest_trade_r_multiple >= 2.0 and add_score >= add_threshold:
        return {
            "hard_block": False,
            "lot_multiplier": 0.5,
            "quant_state": "soft_reduce",
            "quant_action": quant_action,
            "cvar": cvar,
            "drawdown_dampener": drawdown_dampener,
        }

    return {
        "hard_block": False,
        "lot_multiplier": 1.0,
        "quant_state": "aligned" if quant_action != 0 else "neutral",
        "quant_action": quant_action,
        "cvar": cvar,
        "drawdown_dampener": drawdown_dampener,
    }
