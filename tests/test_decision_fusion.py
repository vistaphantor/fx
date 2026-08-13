from types import SimpleNamespace

from src.strategy.breakout import BreakoutDirection
from src.strategy.decision_fusion import fuse_decision
from src.strategy.decision_tree import TopDownTradePlan


def _trade_plan(direction=BreakoutDirection.BULLISH):
    return TopDownTradePlan(
        is_trade=True,
        direction=direction,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=103.0,
        objective_price=104.0,
        reason="top_down_trade_plan_ready",
        metadata={"regime_state": SimpleNamespace(name="trend", confidence=0.85)},
    )


def _live_input():
    bullish = SimpleNamespace(open=100.0, high=101.0, low=99.5, close=100.8)
    return SimpleNamespace(
        h1_candles=[bullish],
        m30_candles=[bullish],
        m15_candles=[bullish],
        m5_candles=[bullish],
        spread=0.05,
    )


def _conflicting_live_input():
    bearish = SimpleNamespace(open=100.8, high=101.0, low=99.5, close=100.0)
    return SimpleNamespace(
        h1_candles=[bearish],
        m30_candles=[bearish],
        m15_candles=[bearish],
        m5_candles=[bearish],
        spread=0.05,
    )


def test_fusion_rewards_multi_source_confluence_with_lower_threshold():
    decision = fuse_decision(
        strategy_result=_trade_plan(),
        live_input=_live_input(),
        quant_decision=SimpleNamespace(is_trade=True, omega_t=0.9, action=1, drawdown_ratio=0.0),
        local_edge_probability=0.82,
        orderflow_signal=SimpleNamespace(
            delta_bias=0.8,
            cvd_slope=0.8,
            imbalance_score=0.6,
            absorption_score=0.2,
            profile_location_score=0.4,
            vwap_alignment=0.5,
            liquidity_obstacle_score=0.0,
        ),
        features=SimpleNamespace(
            structure_score_raw=0.8,
            volatility_score_raw=0.6,
            momentum_indicator_raw=0.7,
            trend_indicator_raw=0.7,
            orderflow_volume_raw=0.6,
            statistical_score_raw=0.7,
            risk_math_raw=0.0,
        ),
        settings=SimpleNamespace(
            fusion_trade_threshold=0.62,
            fusion_hard_min_probability=0.35,
            local_edge_threshold=0.55,
        ),
    )

    assert decision.is_trade
    assert decision.components["support_count"] >= 5
    assert decision.components["confluence"] > 0.45
    assert decision.components["adjusted_threshold"] < decision.components["base_threshold"]


def test_fusion_penalizes_conflicting_confluence_with_higher_threshold():
    decision = fuse_decision(
        strategy_result=_trade_plan(),
        live_input=_conflicting_live_input(),
        quant_decision=SimpleNamespace(is_trade=False, omega_t=0.8, action=-1, drawdown_ratio=0.0),
        local_edge_probability=0.40,
        orderflow_signal=SimpleNamespace(
            delta_bias=-0.8,
            cvd_slope=-0.8,
            imbalance_score=-0.6,
            absorption_score=0.0,
            profile_location_score=-0.4,
            vwap_alignment=-0.5,
            liquidity_obstacle_score=0.2,
        ),
        features=SimpleNamespace(
            structure_score_raw=-0.7,
            volatility_score_raw=-0.6,
            momentum_indicator_raw=-0.7,
            trend_indicator_raw=-0.7,
            orderflow_volume_raw=-0.6,
            statistical_score_raw=-0.7,
            risk_math_raw=0.1,
        ),
        settings=SimpleNamespace(
            fusion_trade_threshold=0.62,
            fusion_hard_min_probability=0.35,
            local_edge_threshold=0.55,
        ),
    )

    assert not decision.is_trade
    assert decision.components["conflict_count"] >= 3
    assert decision.components["confluence"] < 0.0
    assert decision.components["adjusted_threshold"] > decision.components["base_threshold"]
