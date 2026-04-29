from __future__ import annotations

from dataclasses import dataclass

from src.strategy.breakout import BreakoutDirection
from src.strategy.regime import RegimeState


@dataclass(frozen=True, slots=True)
class SideScore:
    location_score: float
    momentum_score: float
    setup_score: float
    trigger_score: float
    gap_score: float
    external_confluence_score: float
    expected_move_score: float
    total: float


@dataclass(frozen=True, slots=True)
class ScoreDecision:
    bullish: SideScore
    bearish: SideScore
    uncertainty_penalty: float
    edge: float
    threshold: float
    expected_move_multiple: float
    is_tradeable: bool
    z_score_normalized_edge: float = 0.0


def score_market_sides(
    *,
    bullish_inputs: dict[str, float],
    bearish_inputs: dict[str, float],
    uncertainty_inputs: dict[str, float],
    expected_move_multiple: float,
    min_expected_move_multiple: float,
    base_threshold: float,
    max_uncertainty_threshold: float,
    preferred_direction: BreakoutDirection,
    regime_state: RegimeState,
    campaign_active: bool = False,
    add_on_edge_multiplier: float = 1.0,
) -> ScoreDecision:
    bullish = _build_side_score(
        inputs=bullish_inputs,
        expected_move_multiple=expected_move_multiple,
        regime_state=regime_state,
        side=BreakoutDirection.BULLISH,
        preferred_direction=preferred_direction,
    )
    bearish = _build_side_score(
        inputs=bearish_inputs,
        expected_move_multiple=expected_move_multiple,
        regime_state=regime_state,
        side=BreakoutDirection.BEARISH,
        preferred_direction=preferred_direction,
    )
    uncertainty_penalty = sum(float(value) for value in uncertainty_inputs.values())
    threshold = float(base_threshold)
    if campaign_active:
        threshold *= max(float(add_on_edge_multiplier), 1.0)

    edge = bullish.total - bearish.total
    is_tradeable = (
        regime_state.tradable
        and expected_move_multiple >= min_expected_move_multiple
        and uncertainty_penalty <= max_uncertainty_threshold
        and abs(edge) >= threshold
    )

    # Z-score normalized edge: edge relative to threshold
    z_norm = (edge / threshold) if threshold > 0 else 0.0

    return ScoreDecision(
        bullish=bullish,
        bearish=bearish,
        uncertainty_penalty=uncertainty_penalty,
        edge=edge,
        threshold=threshold,
        expected_move_multiple=expected_move_multiple,
        is_tradeable=is_tradeable,
        z_score_normalized_edge=z_norm,
    )


def _build_side_score(
    *,
    inputs: dict[str, float],
    expected_move_multiple: float,
    regime_state: RegimeState,
    side: BreakoutDirection,
    preferred_direction: BreakoutDirection,
) -> SideScore:
    location = float(inputs.get("location", 0.0))
    momentum = float(inputs.get("momentum", 0.0))
    setup = float(inputs.get("setup", 0.0))
    trigger = float(inputs.get("trigger", 0.0))
    gap = float(inputs.get("gap", 0.0))
    external = float(inputs.get("external", 0.0))

    base_total = location + momentum + setup + trigger + gap + external
    regime_bias = regime_state.continuation_bias if side is preferred_direction else regime_state.reversion_bias
    confidence_bias = max(regime_state.confidence, 0.1)
    expected_move_score = expected_move_multiple * 0.25
    total = (base_total * regime_bias * confidence_bias) + expected_move_score

    return SideScore(
        location_score=location,
        momentum_score=momentum,
        setup_score=setup,
        trigger_score=trigger,
        gap_score=gap,
        external_confluence_score=external,
        expected_move_score=expected_move_score,
        total=total,
    )
