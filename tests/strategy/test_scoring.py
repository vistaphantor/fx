import pytest

from src.strategy.breakout import BreakoutDirection
from src.strategy.regime import RegimeState
from src.strategy.scoring import score_market_sides


def test_score_sides_returns_bullish_edge_when_location_and_trigger_dominate():
    decision = score_market_sides(
        bullish_inputs={
            "location": 2.4,
            "momentum": 1.5,
            "setup": 1.2,
            "trigger": 1.6,
            "gap": 0.7,
            "external": 0.8,
        },
        bearish_inputs={
            "location": 0.8,
            "momentum": 0.4,
            "setup": 0.3,
            "trigger": 0.2,
            "gap": 0.0,
            "external": 0.1,
        },
        uncertainty_inputs={
            "timeframe_conflict": 0.2,
            "volatility_instability": 0.1,
        },
        expected_move_multiple=2.8,
        min_expected_move_multiple=2.0,
        base_threshold=2.0,
        max_uncertainty_threshold=1.0,
        preferred_direction=BreakoutDirection.BULLISH,
        regime_state=RegimeState(
            name="trend",
            tradable=True,
            continuation_bias=1.2,
            reversion_bias=0.5,
            confidence=0.85,
        ),
    )

    assert decision.bullish.total > decision.bearish.total
    assert decision.edge > decision.threshold
    assert decision.is_tradeable is True


def test_apply_thresholds_blocks_trade_when_uncertainty_exceeds_limit():
    decision = score_market_sides(
        bullish_inputs={
            "location": 2.2,
            "momentum": 1.2,
            "setup": 1.0,
            "trigger": 1.1,
            "gap": 0.4,
            "external": 0.3,
        },
        bearish_inputs={
            "location": 0.7,
            "momentum": 0.6,
            "setup": 0.4,
            "trigger": 0.3,
            "gap": 0.0,
            "external": 0.0,
        },
        uncertainty_inputs={
            "timeframe_conflict": 0.9,
            "volatility_instability": 0.8,
        },
        expected_move_multiple=2.4,
        min_expected_move_multiple=2.0,
        base_threshold=2.0,
        max_uncertainty_threshold=1.0,
        preferred_direction=BreakoutDirection.BULLISH,
        regime_state=RegimeState(
            name="trend",
            tradable=True,
            continuation_bias=1.1,
            reversion_bias=0.4,
            confidence=0.7,
        ),
    )

    assert decision.uncertainty_penalty == pytest.approx(1.7)
    assert decision.is_tradeable is False
