from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.strategy.breakout import BreakoutDirection


@dataclass(frozen=True, slots=True)
class FusionDecision:
    action: str
    score: float
    confidence: float
    lot_multiplier: float
    reason: str
    components: dict[str, float] = field(default_factory=dict)
    hard_block: bool = False

    @property
    def is_trade(self) -> bool:
        return self.action in {"BUY", "SELL"} and not self.hard_block


def fuse_decision(
    *,
    strategy_result,
    live_input,
    quant_decision=None,
    local_edge_probability: float | None = None,
    orderflow_signal=None,
    features=None,
    base_lot_multiplier: float = 1.0,
    settings=None,
) -> FusionDecision:
    strategy_trade = bool(getattr(strategy_result, "is_trade", False))
    direction = getattr(strategy_result, "direction", None)
    direction_sign = _direction_sign(direction)

    local_edge = _probability_component(local_edge_probability)
    quant = _quant_component(quant_decision, direction_sign)
    orderflow = _orderflow_component(orderflow_signal, direction)
    regime = _regime_component(strategy_result)
    alignment = _alignment_component(live_input, direction)
    spread_penalty = _spread_penalty(live_input)
    drawdown_penalty = _drawdown_penalty(quant_decision)
    indicator_math = _indicator_math_component(features)
    risk_math_penalty = _risk_math_penalty(features)
    if strategy_trade:
        strategy_component = 1.0
    else:
        # Near-misses (e.g. setup was good but missed velocity trigger) get a smaller penalty
        failed_node = getattr(strategy_result, "failed_node", "")
        # Hierarchy: h1_direction (severe) -> m30_setup -> m15_setup -> m15_trigger -> m5_trigger (minor)
        hierarchy_penalties = {
            "h1_direction": -0.80,
            "m30_setup": -0.65,
            "m15_setup": -0.50,
            "m15_trigger": -0.35,
            "m5_trigger": -0.20,
        }
        strategy_component = hierarchy_penalties.get(failed_node, -0.55)

    metadata = getattr(strategy_result, "metadata", {}) or {}
    regime_state = metadata.get("regime_state")
    regime_name = getattr(regime_state, "name", "pullback")
    
    # --- Dynamic Adaptive Weighting System ---
    # Base allocations adapt to sub-engine availability
    base_weights = {
        "edge": 0.22 if local_edge_probability is not None else 0.05,
        "quant": 0.20 if quant_decision is not None else 0.10,
        "orderflow": 0.12 if orderflow_signal is not None else 0.05,
        "regime": 0.10,
        "alignment": 0.10,
        "ind": 0.16 if features is not None else 0.08,
        "strat": 0.16,
    }
    # Dynamic conviction scaling: components showing high conviction get boosted up to 1.3x
    raw_convictions = {
        "edge": abs(local_edge),
        "quant": abs(quant),
        "orderflow": abs(orderflow),
        "regime": abs(regime),
        "alignment": abs(alignment),
        "ind": abs(indicator_math),
        "strat": abs(strategy_component),
    }
    dynamic_weights = {
        k: base_weights[k] * (0.80 + 0.50 * raw_convictions[k])
        for k in base_weights
    }
    total_w = sum(dynamic_weights.values()) or 1.0
    w_edge = dynamic_weights["edge"] / total_w
    w_quant = dynamic_weights["quant"] / total_w
    w_orderflow = dynamic_weights["orderflow"] / total_w
    w_regime = dynamic_weights["regime"] / total_w
    w_align = dynamic_weights["alignment"] / total_w
    w_ind = dynamic_weights["ind"] / total_w
    w_strat = dynamic_weights["strat"] / total_w

    threshold = float(getattr(settings, "fusion_trade_threshold", 0.62) or 0.62)
    confluence, support_count, conflict_count = _confluence_component(
        strategy_component=strategy_component,
        local_edge=local_edge,
        local_edge_probability=local_edge_probability,
        local_edge_threshold=float(getattr(settings, "local_edge_threshold", 0.55) or 0.55),
        quant=quant,
        orderflow=orderflow,
        regime=regime,
        alignment=alignment,
        indicator_math=indicator_math,
    )
    adjusted_threshold = _adjusted_threshold(threshold, confluence, support_count, conflict_count)

    score = (
        w_edge * local_edge
        + w_quant * quant
        + w_orderflow * orderflow
        + w_regime * regime
        + w_align * alignment
        + w_ind * indicator_math
        + w_strat * strategy_component
        + 0.12 * confluence
        - 0.07 * spread_penalty
        - 0.08 * drawdown_penalty
        - 0.07 * risk_math_penalty
    )
    score = _clamp(score, -1.0, 1.0)
    confidence = _clamp((score + 1.0) / 2.0, 0.0, 1.0)

    hard_block = _hard_block(strategy_result, quant_decision, local_edge_probability, settings)
    if hard_block:
        action = "WAIT"
    elif strategy_trade and confidence >= adjusted_threshold and direction is BreakoutDirection.BULLISH:
        action = "BUY"
    elif strategy_trade and confidence >= adjusted_threshold and direction is BreakoutDirection.BEARISH:
        action = "SELL"
    else:
        action = "WAIT"

    lot_multiplier = _lot_multiplier(confidence, base_lot_multiplier, action)
    reason = _reason(
        action=action,
        confidence=confidence,
        threshold=adjusted_threshold,
        strategy_result=strategy_result,
        quant_decision=quant_decision,
        local_edge_probability=local_edge_probability,
        hard_block=hard_block,
    )
    return FusionDecision(
        action=action,
        score=score,
        confidence=confidence,
        lot_multiplier=lot_multiplier,
        reason=reason,
        hard_block=hard_block,
        components={
            "local_edge": local_edge,
            "quant": quant,
            "orderflow": orderflow,
            "regime": regime,
            "alignment": alignment,
            "indicator_math": indicator_math,
            "strategy": strategy_component,
            "confluence": confluence,
            "support_count": float(support_count),
            "conflict_count": float(conflict_count),
            "base_threshold": threshold,
            "adjusted_threshold": adjusted_threshold,
            "spread_penalty": spread_penalty,
            "drawdown_penalty": drawdown_penalty,
            "risk_math_penalty": risk_math_penalty,
        },
    )


def _probability_component(probability: float | None) -> float:
    if probability is None:
        return 0.0
    return _clamp((float(probability) - 0.5) * 2.0, -1.0, 1.0)


def _quant_component(quant_decision, direction_sign: int) -> float:
    if quant_decision is None:
        return 0.0
    omega = _clamp(float(getattr(quant_decision, "omega_t", 0.0) or 0.0), 0.0, 1.0)
    action = int(getattr(quant_decision, "action", 0) or 0)
    direction_alignment = 0.0
    if direction_sign and action:
        direction_alignment = 1.0 if action == direction_sign else -1.0
    trade_bias = 0.35 if bool(getattr(quant_decision, "is_trade", False)) else -0.15
    return _clamp((0.75 * omega) + (0.35 * direction_alignment) + trade_bias, -1.0, 1.0)


def _orderflow_component(orderflow_signal, direction) -> float:
    if orderflow_signal is None or direction is None:
        return 0.0
    try:
        from src.strategy.orderflow import score_orderflow_for_direction

        return _clamp(score_orderflow_for_direction(orderflow_signal, direction).alignment_score, -1.0, 1.0)
    except Exception:
        return 0.0


def _regime_component(strategy_result) -> float:
    metadata = getattr(strategy_result, "metadata", {}) or {}
    regime_state = metadata.get("regime_state")
    confidence = float(getattr(regime_state, "confidence", metadata.get("regime_confidence", 0.5)) or 0.5)
    continuation = 0.15 if metadata.get("is_continuation_setup") else 0.0
    return _clamp(((confidence - 0.5) * 2.0) + continuation, -1.0, 1.0)


def _alignment_component(live_input, direction) -> float:
    if direction is None or not live_input:
        return 0.0
    wanted = "BULLISH" if direction is BreakoutDirection.BULLISH else "BEARISH"
    groups = [
        getattr(live_input, "h1_candles", []),
        getattr(live_input, "m30_candles", []),
        getattr(live_input, "m15_candles", []),
        getattr(live_input, "m5_candles", []),
    ]
    votes = 0
    total = 0
    for candles in groups:
        if len(candles) < 1:
            continue
        candle = candles[-1]
        if isinstance(candle, str) or not hasattr(candle, "close"):
            continue
        total += 1
        open_val = float(getattr(candle, "open", candle.close))
        actual = "BULLISH" if float(candle.close) >= open_val else "BEARISH"
        votes += 1 if actual == wanted else -1
    return _clamp(votes / max(total, 1) if total > 0 else 0.0, -1.0, 1.0)


def _spread_penalty(live_input) -> float:
    spread = abs(float(getattr(live_input, "spread", 0.0) or 0.0))
    candles = getattr(live_input, "m15_candles", []) or []
    if not candles:
        return 0.0
    ranges = []
    for candle in candles[-10:]:
        if isinstance(candle, str) or not hasattr(candle, "high") or not hasattr(candle, "low"):
            continue
        ranges.append(abs(float(candle.high) - float(candle.low)))
    if not ranges:
        return 0.0
    avg_range = sum(ranges) / len(ranges)
    if avg_range <= 0:
        return 0.0
    return _clamp(spread / avg_range, 0.0, 1.0)


def _drawdown_penalty(quant_decision) -> float:
    if quant_decision is None:
        return 0.0
    return _clamp(float(getattr(quant_decision, "drawdown_ratio", 0.0) or 0.0), 0.0, 1.0)


def _indicator_math_component(features) -> float:
    if features is None:
        return 0.0
    values = [
        float(getattr(features, "structure_score_raw", 0.0) or 0.0),
        float(getattr(features, "volatility_score_raw", 0.0) or 0.0),
        float(getattr(features, "momentum_indicator_raw", 0.0) or 0.0),
        float(getattr(features, "trend_indicator_raw", 0.0) or 0.0),
        float(getattr(features, "orderflow_volume_raw", 0.0) or 0.0),
        float(getattr(features, "statistical_score_raw", 0.0) or 0.0),
    ]
    return _clamp(sum(values) / max(len(values), 1), -1.0, 1.0)


def _risk_math_penalty(features) -> float:
    if features is None:
        return 0.0
    return _clamp(float(getattr(features, "risk_math_raw", 0.0) or 0.0), 0.0, 1.0)


def _confluence_component(
    *,
    strategy_component: float,
    local_edge: float,
    local_edge_probability: float | None,
    local_edge_threshold: float,
    quant: float,
    orderflow: float,
    regime: float,
    alignment: float,
    indicator_math: float,
) -> tuple[float, int, int]:
    signals = {
        "strategy": strategy_component,
        "quant": quant,
        "orderflow": orderflow,
        "regime": regime,
        "alignment": alignment,
        "indicator_math": indicator_math,
    }
    if local_edge_probability is not None:
        edge_margin = float(local_edge_probability) - float(local_edge_threshold)
        signals["local_edge"] = _clamp(edge_margin / 0.20, -1.0, 1.0)
    else:
        signals["local_edge"] = local_edge

    support_count = sum(1 for value in signals.values() if value >= 0.25)
    conflict_count = sum(1 for value in signals.values() if value <= -0.25)
    weighted_balance = sum(_clamp(value, -1.0, 1.0) for value in signals.values()) / max(len(signals), 1)
    count_balance = (support_count - conflict_count) / max(support_count + conflict_count, 1)
    return _clamp((0.60 * weighted_balance) + (0.40 * count_balance), -1.0, 1.0), support_count, conflict_count


def _adjusted_threshold(base_threshold: float, confluence: float, support_count: int, conflict_count: int) -> float:
    threshold = float(base_threshold)
    if support_count >= 5 and confluence >= 0.45:
        threshold -= 0.04
    elif support_count >= 4 and confluence >= 0.30:
        threshold -= 0.02
    if conflict_count >= 2:
        threshold += 0.05
    if conflict_count >= 3 or confluence <= -0.40:
        threshold += 0.08
    return _clamp(threshold, 0.52, 0.78)


def _hard_block(strategy_result, quant_decision, local_edge_probability, settings) -> bool:
    if not bool(getattr(strategy_result, "is_trade", False)):
        return False
    min_probability = float(getattr(settings, "fusion_hard_min_probability", 0.45) or 0.45)
    if local_edge_probability is not None and float(local_edge_probability) < min_probability:
        return True
    if quant_decision is not None:
        drawdown = float(getattr(quant_decision, "drawdown_ratio", 0.0) or 0.0)
        if drawdown >= 0.95:
            return True
    return False


def _lot_multiplier(confidence: float, base_lot_multiplier: float, action: str) -> float:
    if action == "WAIT":
        return 0.0
    if confidence < 0.70:
        confidence_multiplier = 0.35 + ((confidence - 0.62) / 0.08) * 0.40
    elif confidence < 0.85:
        confidence_multiplier = 0.75 + ((confidence - 0.70) / 0.15) * 0.35
    else:
        confidence_multiplier = 1.10 + ((confidence - 0.85) / 0.15) * 0.20
    return _clamp(float(base_lot_multiplier) * confidence_multiplier, 0.01, 1.30)


def _reason(
    *,
    action: str,
    confidence: float,
    threshold: float,
    strategy_result,
    quant_decision,
    local_edge_probability,
    hard_block: bool,
) -> str:
    if hard_block:
        strategy_dir = getattr(strategy_result, "direction", None)
        quant_action = getattr(quant_decision, "action", 0) if quant_decision is not None else 0
        if strategy_dir is BreakoutDirection.BULLISH and quant_action == -1:
            return "quant_direction_mismatch"
        if strategy_dir is BreakoutDirection.BEARISH and quant_action == 1:
            return "quant_direction_mismatch"
        return f"fusion_hard_block_pwin_{float(local_edge_probability or 0.0):.3f}"
    if action == "WAIT":
        strategy_reason = str(getattr(strategy_result, "reason", "") or "")
        quant_reason = str(getattr(quant_decision, "reason", "") or "") if quant_decision is not None else ""
        return f"fusion_score_below_threshold_{confidence:.3f}_{threshold:.3f}:{strategy_reason or quant_reason or 'wait'}"
    return f"fusion_{action.lower()}_confidence_{confidence:.3f}"


def _direction_sign(direction) -> int:
    if direction is BreakoutDirection.BULLISH:
        return 1
    if direction is BreakoutDirection.BEARISH:
        return -1
    return 0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
