from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.market_data import Candle
from src.strategy.breakout import BreakoutDirection
from src.strategy.decision_tree import evaluate_top_down_decision_tree
from src.strategy.direction import DirectionDecision
from src.strategy.regime import RegimeState
from src.strategy.scoring import ScoreDecision, SideScore
from src.strategy.setup import SetupDecision
from src.strategy.trigger import TriggerDecision
from src.strategy.volatility import VolatilityState


def _candles(timeframe: str, rows: list[tuple[float, float, float, float, int]]) -> list[Candle]:
    start = datetime(2026, 4, 6, 0, 0, tzinfo=timezone.utc)
    minutes = {"D1": 1440, "H4": 240, "H1": 60, "M30": 30, "M15": 15, "M10": 10, "M5": 5}[timeframe]
    candles = []
    for index, (open_price, high, low, close, volume) in enumerate(rows):
        candles.append(
            Candle(
                timestamp=start + timedelta(minutes=minutes * index),
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                timeframe=timeframe,
            )
        )
    return candles


def test_decision_tree_returns_trade_plan_when_all_nodes_align():
    result = evaluate_top_down_decision_tree(
        d1_candles=_candles(
            "D1",
            [
                (101.0, 108.0, 99.0, 105.0, 1000),
                (105.0, 112.0, 100.0, 109.0, 1200),
            ],
        ),
        h4_candles=_candles(
            "H4",
            [
                (100.0, 104.0, 98.0, 103.0, 80),
                (103.0, 110.0, 96.0, 108.0, 95),
                (108.0, 109.0, 101.0, 102.0, 70),
                (102.0, 117.0, 100.0, 116.0, 150),
                (116.0, 131.0, 114.0, 128.0, 210),
                (128.0, 129.0, 120.0, 123.0, 130),
                (123.0, 126.0, 119.0, 124.0, 90),
            ],
        ),
        h1_candles=_candles(
            "H1",
            [
                (99.0, 101.0, 98.0, 100.0, 100),
                (100.0, 102.0, 99.5, 101.5, 101),
                (101.5, 104.0, 101.0, 103.0, 102),
                (103.0, 105.0, 102.5, 104.0, 103),
                (104.0, 107.0, 103.5, 106.0, 104),
            ],
        ),
        m30_candles=_candles(
            "M30",
            [
                (104.5, 105.0, 103.4, 104.0, 100),
                (104.0, 104.5, 103.2, 103.9, 101),
                (103.9, 106.4, 103.0, 106.0, 102),
            ],
        ),
        m15_candles=_candles(
            "M15",
            [
                (104.0, 104.5, 103.8, 104.1, 100),
                (104.1, 104.4, 103.9, 104.0, 101),
                (104.0, 106.2, 103.95, 105.9, 102),
                (105.9, 106.3, 104.8, 105.5, 103),
                (105.5, 106.0, 104.7, 105.7, 104),
                (105.7, 106.4, 104.5, 106.1, 105),
            ],
        ),
        m10_candles=_candles(
            "M10",
            [
                (105.4, 105.7, 105.1, 105.5, 100),
                (105.5, 105.9, 105.2, 105.7, 101),
                (105.7, 106.1, 105.5, 105.95, 102),
            ],
        ),
        m5_candles=_candles(
            "M5",
            [
                (105.9, 106.0, 105.7, 105.95, 100),
                (105.95, 106.2, 105.8, 106.05, 101),
                (106.05, 106.8, 105.9, 106.4, 102),
            ],
        ),
        risk_buffer=0.05,
    )

    assert result.is_trade is True
    assert result.direction is BreakoutDirection.BULLISH
    assert result.reason == "top_down_trade_plan_ready"
    assert result.entry_price == 106.4
    assert result.stop_loss < result.entry_price
    assert result.take_profit > result.entry_price
    assert result.metadata["daily_context"].current_price == 106.4
    assert result.metadata["regime_state"].tradable is True
    assert result.metadata["score_decision"].is_tradeable is True
    assert result.metadata["is_continuation_setup"] is False
    assert result.metadata["m15_quality"] > 0.0
    assert result.metadata["m10_quality"] > 0.0
    assert result.metadata["m5_quality"] > 0.0
    assert result.metadata["range_expansion_ratio"] > 0.0
    assert result.metadata["body_efficiency"] > 0.0
    assert result.metadata["regime_confidence"] > 0.0


def test_decision_tree_marks_true_continuation_trade_plans_as_continuations(monkeypatch):
    from src.strategy import decision_tree as decision_tree_module

    monkeypatch.setattr(
        decision_tree_module,
        "evaluate_m30_setup",
        lambda **kwargs: SetupDecision(
            is_ready=True,
            reason="m30_continuation_reacceptance_ready",
            setup_state="continuation_reacceptance",
            metadata={},
            quality_score=0.88,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "evaluate_m15_trigger",
        lambda **kwargs: TriggerDecision(
            is_ready=True,
            reason="m15_trigger_still_valid",
            entry_price=106.0,
            invalidation_price=104.8,
            metadata={},
            quality_score=0.9,
            expected_move_multiple=2.0,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "evaluate_m10_setup",
        lambda **kwargs: SetupDecision(
            is_ready=True,
            reason="m10_continuation_ready",
            setup_state="refining",
            metadata={},
            quality_score=0.82,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "evaluate_m5_trigger",
        lambda **kwargs: TriggerDecision(
            is_ready=True,
            reason="m5_continuation_ready",
            entry_price=106.4,
            invalidation_price=105.6,
            metadata={},
            quality_score=0.84,
            expected_move_multiple=1.9,
        ),
    )

    result = evaluate_top_down_decision_tree(
        d1_candles=_candles(
            "D1",
            [
                (101.0, 108.0, 99.0, 105.0, 1000),
                (105.0, 112.0, 100.0, 109.0, 1200),
            ],
        ),
        h4_candles=_candles(
            "H4",
            [
                (100.0, 104.0, 98.0, 103.0, 80),
                (103.0, 110.0, 96.0, 108.0, 95),
                (108.0, 109.0, 101.0, 102.0, 70),
                (102.0, 117.0, 100.0, 116.0, 150),
                (116.0, 131.0, 114.0, 128.0, 210),
                (128.0, 129.0, 120.0, 123.0, 130),
                (123.0, 126.0, 119.0, 124.0, 90),
            ],
        ),
        h1_candles=_candles(
            "H1",
            [
                (99.0, 101.0, 98.0, 100.0, 100),
                (100.0, 102.0, 99.5, 101.5, 101),
                (101.5, 104.0, 101.0, 103.0, 102),
                (103.0, 105.0, 102.5, 104.0, 103),
                (104.0, 107.0, 103.5, 106.0, 104),
            ],
        ),
        m30_candles=_candles(
            "M30",
            [
                (104.5, 105.0, 103.4, 104.0, 100),
                (104.0, 104.5, 103.2, 103.9, 101),
                (103.9, 106.4, 103.0, 106.0, 102),
            ],
        ),
        m15_candles=_candles(
            "M15",
            [
                (104.0, 104.5, 103.8, 104.1, 100),
                (104.1, 104.4, 103.9, 104.0, 101),
                (104.0, 106.2, 103.95, 105.9, 102),
                (105.9, 106.3, 104.8, 105.5, 103),
                (105.5, 106.0, 104.7, 105.7, 104),
                (105.7, 106.4, 104.5, 106.1, 105),
            ],
        ),
        m10_candles=_candles(
            "M10",
            [
                (105.4, 105.7, 105.1, 105.5, 100),
                (105.5, 105.9, 105.2, 105.7, 101),
                (105.7, 106.1, 105.5, 105.95, 102),
            ],
        ),
        m5_candles=_candles(
            "M5",
            [
                (105.9, 106.0, 105.7, 105.95, 100),
                (105.95, 106.2, 105.8, 106.05, 101),
                (106.05, 106.8, 105.9, 106.4, 102),
            ],
        ),
        risk_buffer=0.05,
    )

    assert result.is_trade is True
    assert result.metadata["is_continuation_setup"] is True


def test_decision_tree_returns_exact_failed_node_reason():
    result = evaluate_top_down_decision_tree(
        d1_candles=_candles(
            "D1",
            [
                (101.0, 108.0, 99.0, 105.0, 1000),
                (105.0, 112.0, 100.0, 109.0, 1200),
            ],
        ),
        h4_candles=_candles(
            "H4",
            [
                (100.0, 104.0, 98.0, 103.0, 80),
                (103.0, 110.0, 96.0, 108.0, 95),
                (108.0, 109.0, 101.0, 102.0, 70),
                (102.0, 117.0, 100.0, 116.0, 150),
                (116.0, 131.0, 114.0, 128.0, 210),
                (128.0, 129.0, 120.0, 123.0, 130),
                (123.0, 126.0, 119.0, 124.0, 90),
            ],
        ),
        h1_candles=_candles(
            "H1",
            [
                (99.0, 101.0, 98.0, 100.0, 100),
                (100.0, 102.0, 99.5, 101.5, 101),
                (101.5, 104.0, 101.0, 103.0, 102),
                (103.0, 105.0, 102.5, 104.0, 103),
                (104.0, 107.0, 103.5, 106.0, 104),
            ],
        ),
        m30_candles=_candles(
            "M30",
            [
                (105.4, 106.1, 105.0, 105.7, 100),
                (105.7, 106.2, 105.2, 105.9, 101),
                (105.9, 106.05, 104.7, 105.3, 102),
            ],
        ),
        m15_candles=_candles(
            "M15",
            [
                (104.0, 104.5, 103.8, 104.1, 100),
                (104.1, 104.4, 103.9, 104.0, 101),
                (104.0, 106.2, 103.95, 105.9, 102),
            ],
        ),
        m10_candles=_candles(
            "M10",
            [
                (105.8, 106.0, 105.6, 105.9, 100),
                (105.9, 106.1, 105.7, 106.0, 101),
                (106.0, 106.2, 105.8, 106.1, 102),
            ],
        ),
        m5_candles=_candles(
            "M5",
            [
                (106.0, 106.1, 105.9, 106.0, 100),
                (106.0, 106.15, 105.95, 106.02, 101),
                (106.02, 106.18, 105.98, 106.03, 102),
            ],
        ),
        risk_buffer=0.05,
    )

    assert result.is_trade is False
    assert result.reason == "m30_midrange_noise"
    assert result.failed_node == "m30_setup"


def test_decision_tree_returns_no_trade_when_quant_edge_is_too_close(monkeypatch):
    from src.strategy import decision_tree as decision_tree_module
    from src.strategy.volatility import VolatilityState

    monkeypatch.setattr(
        decision_tree_module,
        "build_volatility_state",
        lambda **kwargs: VolatilityState(
            short_atr=1.2,
            medium_atr=1.1,
            realized_range=8.0,
            body_efficiency=0.7,
            range_expansion_ratio=1.09,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "classify_regime",
        lambda **kwargs: RegimeState(
            name="trend",
            tradable=True,
            continuation_bias=1.1,
            reversion_bias=0.6,
            confidence=0.8,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "score_market_sides",
        lambda **kwargs: ScoreDecision(
            bullish=SideScore(1.0, 1.0, 1.0, 1.0, 0.2, 0.0, 0.7, 4.2),
            bearish=SideScore(1.0, 0.9, 0.9, 0.8, 0.2, 0.0, 0.7, 4.0),
            uncertainty_penalty=0.3,
            edge=0.2,
            threshold=2.0,
            expected_move_multiple=2.1,
            is_tradeable=False,
        ),
    )

    result = evaluate_top_down_decision_tree(
        d1_candles=_candles(
            "D1",
            [
                (101.0, 108.0, 99.0, 105.0, 1000),
                (105.0, 112.0, 100.0, 109.0, 1200),
            ],
        ),
        h4_candles=_candles(
            "H4",
            [
                (100.0, 104.0, 98.0, 103.0, 80),
                (103.0, 110.0, 96.0, 108.0, 95),
                (108.0, 109.0, 101.0, 102.0, 70),
                (102.0, 117.0, 100.0, 116.0, 150),
                (116.0, 131.0, 114.0, 128.0, 210),
                (128.0, 129.0, 120.0, 123.0, 130),
                (123.0, 126.0, 119.0, 124.0, 90),
            ],
        ),
        h1_candles=_candles(
            "H1",
            [
                (99.0, 101.0, 98.0, 100.0, 100),
                (100.0, 102.0, 99.5, 101.5, 101),
                (101.5, 104.0, 101.0, 103.0, 102),
                (103.0, 105.0, 102.5, 104.0, 103),
                (104.0, 107.0, 103.5, 106.0, 104),
            ],
        ),
        m30_candles=_candles(
            "M30",
            [
                (104.5, 105.0, 103.4, 104.0, 100),
                (104.0, 104.5, 103.2, 103.9, 101),
                (103.9, 106.4, 103.0, 106.0, 102),
            ],
        ),
        m15_candles=_candles(
            "M15",
            [
                (104.0, 104.5, 103.8, 104.1, 100),
                (104.1, 104.4, 103.9, 104.0, 101),
                (104.0, 106.2, 103.95, 105.9, 102),
                (105.9, 106.3, 104.8, 105.5, 103),
                (105.5, 106.0, 104.7, 105.7, 104),
                (105.7, 106.4, 104.5, 106.1, 105),
            ],
        ),
        m10_candles=_candles(
            "M10",
            [
                (105.4, 105.7, 105.1, 105.5, 100),
                (105.5, 105.9, 105.2, 105.7, 101),
                (105.7, 106.1, 105.5, 105.95, 102),
            ],
        ),
        m5_candles=_candles(
            "M5",
            [
                (105.9, 106.0, 105.7, 105.95, 100),
                (105.95, 106.2, 105.8, 106.05, 101),
                (106.05, 106.8, 105.9, 106.4, 102),
            ],
        ),
        risk_buffer=0.05,
    )

    assert result.is_trade is False
    assert result.reason == "quant_edge_insufficient"
    assert result.failed_node == "scoring"


def test_decision_tree_allows_bullish_reversal_below_daily_low_when_direction_points_up(monkeypatch):
    from src.strategy import decision_tree as decision_tree_module

    monkeypatch.setattr(
        decision_tree_module,
        "determine_h1_bias",
        lambda **kwargs: DirectionDecision(
            is_valid=True,
            direction=BreakoutDirection.BULLISH,
            reason="h1_reversal_context_bullish",
            metadata={},
            bullish_contribution=4.0,
            bearish_contribution=3.0,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "evaluate_m30_setup",
        lambda **kwargs: SetupDecision(
            is_ready=True,
            reason="m30_rejection_ready",
            setup_state="rejecting_level",
            metadata={},
            quality_score=1.0,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "evaluate_m15_trigger",
        lambda **kwargs: TriggerDecision(
            is_ready=True,
            reason="m15_trigger_ready",
            entry_price=99.6,
            invalidation_price=99.1,
            metadata={},
            quality_score=1.0,
            expected_move_multiple=2.0,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "evaluate_m10_setup",
        lambda **kwargs: SetupDecision(
            is_ready=True,
            reason="m10_refinement_ready",
            setup_state="refining",
            metadata={},
            quality_score=0.9,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "evaluate_m5_trigger",
        lambda **kwargs: TriggerDecision(
            is_ready=True,
            reason="m5_trigger_ready",
            entry_price=99.5,
            invalidation_price=99.0,
            metadata={},
            quality_score=1.0,
            expected_move_multiple=2.5,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "build_volatility_state",
        lambda **kwargs: VolatilityState(
            short_atr=1.0,
            medium_atr=1.0,
            realized_range=8.0,
            body_efficiency=0.8,
            range_expansion_ratio=1.05,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "classify_regime",
        lambda **kwargs: RegimeState(
            name="trend",
            tradable=True,
            continuation_bias=1.0,
            reversion_bias=0.9,
            confidence=0.8,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "score_market_sides",
        lambda **kwargs: ScoreDecision(
            bullish=SideScore(1.0, 1.0, 1.0, 1.0, 0.2, 0.0, 0.7, 5.0),
            bearish=SideScore(0.5, 0.4, 0.4, 0.3, 0.1, 0.0, 0.7, 2.4),
            uncertainty_penalty=0.2,
            edge=2.6,
            threshold=1.0,
            expected_move_multiple=2.5,
            is_tradeable=True,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "detect_three_drives",
        lambda **kwargs: SimpleNamespace(reason="pattern_optional", confluence_score=0.0),
    )

    result = evaluate_top_down_decision_tree(
        d1_candles=_candles(
            "D1",
            [
                (101.0, 108.0, 99.0, 105.0, 1000),
                (105.0, 110.0, 100.0, 104.0, 1200),
            ],
        ),
        h4_candles=_candles(
            "H4",
            [
                (99.5, 101.0, 98.0, 100.0, 80),
                (100.0, 101.5, 98.5, 99.0, 90),
                (99.0, 100.0, 97.5, 98.4, 85),
                (98.4, 100.2, 97.8, 99.6, 92),
                (99.6, 101.2, 99.1, 100.4, 95),
                (100.4, 101.1, 99.2, 99.8, 88),
                (99.8, 100.3, 98.7, 99.5, 91),
            ],
        ),
        h1_candles=_candles(
            "H1",
            [
                (101.0, 101.5, 100.5, 101.0, 100),
                (101.0, 101.2, 100.0, 100.4, 101),
                (100.4, 100.8, 99.2, 99.6, 102),
                (99.6, 100.1, 98.9, 99.3, 103),
                (99.3, 99.8, 98.7, 99.1, 104),
            ],
        ),
        m30_candles=_candles(
            "M30",
            [
                (99.8, 100.0, 99.2, 99.4, 100),
                (99.4, 99.7, 99.0, 99.3, 101),
                (99.3, 99.8, 98.9, 99.5, 102),
            ],
        ),
        m15_candles=_candles(
            "M15",
            [
                (99.7, 99.9, 99.2, 99.4, 100),
                (99.4, 99.6, 98.9, 99.2, 101),
                (99.2, 99.8, 98.8, 99.5, 102),
            ],
        ),
        m10_candles=_candles(
            "M10",
            [
                (99.4, 99.5, 99.1, 99.3, 100),
                (99.3, 99.6, 99.0, 99.4, 101),
                (99.4, 99.7, 99.2, 99.5, 102),
            ],
        ),
        m5_candles=_candles(
            "M5",
            [
                (99.3, 99.5, 99.1, 99.2, 100),
                (99.2, 99.4, 99.0, 99.3, 101),
                (99.3, 99.7, 98.9, 99.5, 102),
            ],
        ),
        risk_buffer=0.05,
    )

    assert result.is_trade is True
    assert result.direction is BreakoutDirection.BULLISH
    assert result.reason == "top_down_trade_plan_ready"


def test_decision_tree_blocks_bullish_trade_when_daily_high_is_already_spent(monkeypatch):
    from src.strategy import decision_tree as decision_tree_module

    monkeypatch.setattr(
        decision_tree_module,
        "determine_h1_bias",
        lambda **kwargs: DirectionDecision(
            is_valid=True,
            direction=BreakoutDirection.BULLISH,
            reason="h1_bias_bullish",
            metadata={},
            bullish_contribution=5.0,
            bearish_contribution=1.0,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "evaluate_m30_setup",
        lambda **kwargs: SetupDecision(
            is_ready=True,
            reason="m30_breakaway_ready",
            setup_state="breaking_away",
            metadata={},
            quality_score=0.45,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "evaluate_m15_trigger",
        lambda **kwargs: TriggerDecision(
            is_ready=True,
            reason="m15_trigger_ready",
            entry_price=121.0,
            invalidation_price=119.8,
            metadata={},
            quality_score=0.45,
            expected_move_multiple=2.0,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "evaluate_m10_setup",
        lambda **kwargs: SetupDecision(
            is_ready=True,
            reason="m10_refinement_ready",
            setup_state="refining",
            metadata={},
            quality_score=0.45,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "evaluate_m5_trigger",
        lambda **kwargs: TriggerDecision(
            is_ready=True,
            reason="m5_trigger_ready",
            entry_price=121.0,
            invalidation_price=119.7,
            metadata={},
            quality_score=0.45,
            expected_move_multiple=2.0,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "build_volatility_state",
        lambda **kwargs: VolatilityState(
            short_atr=0.8,
            medium_atr=1.0,
            realized_range=3.2,
            body_efficiency=0.35,
            range_expansion_ratio=0.82,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "classify_regime",
        lambda **kwargs: RegimeState(
            name="trend",
            tradable=True,
            continuation_bias=1.0,
            reversion_bias=0.4,
            confidence=0.7,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "score_market_sides",
        lambda **kwargs: ScoreDecision(
            bullish=SideScore(1.2, 1.0, 0.7, 0.6, 0.1, 0.0, 0.7, 4.3),
            bearish=SideScore(0.6, 0.5, 0.4, 0.4, 0.1, 0.0, 0.7, 2.7),
            uncertainty_penalty=0.2,
            edge=1.6,
            threshold=1.0,
            expected_move_multiple=2.0,
            is_tradeable=True,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "detect_three_drives",
        lambda **kwargs: SimpleNamespace(reason="pattern_optional", confluence_score=0.0),
    )

    result = evaluate_top_down_decision_tree(
        d1_candles=_candles(
            "D1",
            [
                (101.0, 108.0, 99.0, 105.0, 1000),
                (105.0, 110.0, 100.0, 109.0, 1200),
            ],
        ),
        h4_candles=_candles(
            "H4",
            [
                (100.0, 104.0, 98.0, 103.0, 80),
                (103.0, 110.0, 96.0, 108.0, 95),
                (108.0, 109.0, 101.0, 102.0, 70),
                (102.0, 117.0, 100.0, 116.0, 150),
                (116.0, 131.0, 114.0, 128.0, 210),
                (128.0, 129.0, 120.0, 123.0, 130),
                (123.0, 126.0, 119.0, 124.0, 90),
            ],
        ),
        h1_candles=_candles(
            "H1",
            [
                (107.0, 109.0, 106.5, 108.0, 100),
                (108.0, 110.0, 107.5, 109.0, 101),
                (109.0, 111.0, 108.5, 110.0, 102),
                (110.0, 112.0, 109.5, 111.0, 103),
                (111.0, 113.0, 110.5, 112.0, 104),
            ],
        ),
        m30_candles=_candles(
            "M30",
            [
                (109.8, 110.1, 109.4, 109.9, 100),
                (109.9, 110.3, 109.6, 110.0, 101),
                (110.0, 110.5, 109.8, 110.2, 102),
            ],
        ),
        m15_candles=_candles(
            "M15",
            [
                (109.9, 110.2, 109.7, 110.0, 100),
                (110.0, 110.4, 109.8, 110.1, 101),
                (110.1, 110.6, 109.9, 110.4, 102),
            ],
        ),
        m10_candles=_candles(
            "M10",
            [
                (110.0, 110.2, 109.9, 110.1, 100),
                (110.1, 110.3, 110.0, 110.2, 101),
                (110.2, 110.4, 110.1, 110.3, 102),
            ],
        ),
        m5_candles=_candles(
            "M5",
            [
                (120.4, 120.6, 120.2, 120.5, 100),
                (120.5, 120.8, 120.4, 120.7, 101),
                (120.7, 121.2, 120.6, 121.0, 102),
            ],
        ),
        risk_buffer=0.05,
    )

    assert result.is_trade is False
    assert result.reason == "d1_objective_exhausted"
    assert result.failed_node == "d1_context"


def test_decision_tree_allows_early_bullish_sweep_when_adr_is_not_exhausted(monkeypatch):
    from src.strategy import decision_tree as decision_tree_module

    monkeypatch.setattr(
        decision_tree_module,
        "determine_h1_bias",
        lambda **kwargs: DirectionDecision(
            is_valid=True,
            direction=BreakoutDirection.BULLISH,
            reason="h1_bias_bullish",
            metadata={},
            bullish_contribution=5.0,
            bearish_contribution=1.0,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "evaluate_m30_setup",
        lambda **kwargs: SetupDecision(
            is_ready=True,
            reason="m30_breakaway_ready",
            setup_state="breaking_away",
            metadata={},
            quality_score=0.9,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "evaluate_m15_trigger",
        lambda **kwargs: TriggerDecision(
            is_ready=True,
            reason="m15_trigger_ready",
            entry_price=114.0,
            invalidation_price=112.8,
            metadata={},
            quality_score=0.95,
            expected_move_multiple=2.2,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "evaluate_m10_setup",
        lambda **kwargs: SetupDecision(
            is_ready=True,
            reason="m10_refinement_ready",
            setup_state="refining",
            metadata={},
            quality_score=0.85,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "evaluate_m5_trigger",
        lambda **kwargs: TriggerDecision(
            is_ready=True,
            reason="m5_trigger_ready",
            entry_price=114.1,
            invalidation_price=112.9,
            metadata={},
            quality_score=0.95,
            expected_move_multiple=2.4,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "build_volatility_state",
        lambda **kwargs: VolatilityState(
            short_atr=0.8,
            medium_atr=1.0,
            realized_range=3.2,
            body_efficiency=0.75,
            range_expansion_ratio=1.05,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "classify_regime",
        lambda **kwargs: RegimeState(
            name="trend",
            tradable=True,
            continuation_bias=1.1,
            reversion_bias=0.5,
            confidence=0.82,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "score_market_sides",
        lambda **kwargs: ScoreDecision(
            bullish=SideScore(1.2, 1.1, 1.0, 1.0, 0.2, 0.0, 0.7, 5.2),
            bearish=SideScore(0.6, 0.5, 0.4, 0.4, 0.1, 0.0, 0.7, 2.7),
            uncertainty_penalty=0.15,
            edge=2.5,
            threshold=1.2,
            expected_move_multiple=2.4,
            is_tradeable=True,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "detect_three_drives",
        lambda **kwargs: SimpleNamespace(reason="pattern_optional", confluence_score=0.0),
    )

    result = evaluate_top_down_decision_tree(
        d1_candles=_candles(
            "D1",
            [
                (100.0, 108.0, 99.0, 104.0, 1000),
                (104.0, 111.0, 101.0, 109.0, 1200),
                (109.0, 113.0, 105.0, 112.0, 1400),
            ],
        ),
        h4_candles=_candles(
            "H4",
            [
                (108.0, 110.0, 106.0, 109.0, 80),
                (109.0, 112.0, 107.0, 111.0, 90),
                (111.0, 113.5, 108.5, 112.5, 95),
                (112.5, 114.5, 111.5, 113.8, 100),
                (113.8, 115.0, 112.6, 114.4, 110),
                (114.4, 114.8, 113.2, 113.9, 90),
                (113.9, 114.5, 113.4, 114.2, 85),
            ],
        ),
        h1_candles=_candles(
            "H1",
            [
                (110.0, 110.8, 109.5, 110.4, 100),
                (110.4, 111.0, 109.9, 110.8, 101),
                (110.8, 112.0, 110.5, 111.7, 102),
                (111.7, 113.2, 111.3, 112.8, 103),
                (112.8, 114.4, 112.2, 114.0, 104),
            ],
        ),
        m30_candles=_candles(
            "M30",
            [
                (112.6, 113.1, 112.2, 112.9, 100),
                (112.9, 113.6, 112.6, 113.3, 101),
                (113.3, 114.3, 113.0, 114.0, 102),
            ],
        ),
        m15_candles=_candles(
            "M15",
            [
                (112.8, 113.1, 112.5, 112.9, 100),
                (112.9, 113.6, 112.7, 113.4, 101),
                (113.4, 114.2, 113.1, 113.9, 102),
                (113.9, 114.4, 113.6, 114.1, 103),
            ],
        ),
        m10_candles=_candles(
            "M10",
            [
                (113.5, 113.8, 113.3, 113.6, 100),
                (113.6, 114.0, 113.5, 113.8, 101),
                (113.8, 114.2, 113.7, 114.0, 102),
            ],
        ),
        m5_candles=_candles(
            "M5",
            [
                (113.8, 114.0, 113.7, 113.9, 100),
                (113.9, 114.2, 113.8, 114.0, 101),
                (114.0, 114.4, 113.9, 114.1, 102),
            ],
        ),
        risk_buffer=0.05,
    )

    assert result.is_trade is True
    assert result.direction is BreakoutDirection.BULLISH


def test_decision_tree_allows_bearish_continuation_slightly_below_daily_low(monkeypatch):
    from src.strategy import decision_tree as decision_tree_module

    monkeypatch.setattr(
        decision_tree_module,
        "determine_h1_bias",
        lambda **kwargs: DirectionDecision(
            is_valid=True,
            direction=BreakoutDirection.BEARISH,
            reason="h1_bias_bearish",
            metadata={},
            bullish_contribution=2.0,
            bearish_contribution=5.0,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "evaluate_m30_setup",
        lambda **kwargs: SetupDecision(
            is_ready=True,
            reason="m30_breakaway_ready",
            setup_state="breaking_away",
            metadata={},
            quality_score=1.0,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "evaluate_m15_trigger",
        lambda **kwargs: TriggerDecision(
            is_ready=True,
            reason="m15_trigger_ready",
            entry_price=98.5,
            invalidation_price=99.4,
            metadata={},
            quality_score=1.0,
            expected_move_multiple=2.0,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "evaluate_m10_setup",
        lambda **kwargs: SetupDecision(
            is_ready=True,
            reason="m10_refinement_ready",
            setup_state="refining",
            metadata={},
            quality_score=0.9,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "evaluate_m5_trigger",
        lambda **kwargs: TriggerDecision(
            is_ready=True,
            reason="m5_trigger_ready",
            entry_price=98.4,
            invalidation_price=99.0,
            metadata={},
            quality_score=1.0,
            expected_move_multiple=2.4,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "build_volatility_state",
        lambda **kwargs: VolatilityState(
            short_atr=1.0,
            medium_atr=1.0,
            realized_range=8.0,
            body_efficiency=0.8,
            range_expansion_ratio=1.05,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "classify_regime",
        lambda **kwargs: RegimeState(
            name="trend",
            tradable=True,
            continuation_bias=1.1,
            reversion_bias=0.6,
            confidence=0.8,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "score_market_sides",
        lambda **kwargs: ScoreDecision(
            bullish=SideScore(0.5, 0.4, 0.4, 0.3, 0.1, 0.0, 0.7, 2.4),
            bearish=SideScore(1.0, 1.0, 1.0, 1.0, 0.2, 0.0, 0.7, 5.0),
            uncertainty_penalty=0.2,
            edge=-2.6,
            threshold=1.0,
            expected_move_multiple=2.5,
            is_tradeable=True,
        ),
    )
    monkeypatch.setattr(
        decision_tree_module,
        "detect_three_drives",
        lambda **kwargs: SimpleNamespace(reason="pattern_optional", confluence_score=0.0),
    )

    result = evaluate_top_down_decision_tree(
        d1_candles=_candles(
            "D1",
            [
                (101.0, 108.0, 99.0, 105.0, 1000),
                (105.0, 110.0, 100.0, 104.0, 1200),
            ],
        ),
        h4_candles=_candles(
            "H4",
            [
                (99.5, 101.0, 98.0, 100.0, 80),
                (100.0, 101.5, 98.5, 99.0, 90),
                (99.0, 100.0, 97.5, 98.4, 85),
                (98.4, 100.2, 97.8, 99.6, 92),
                (99.6, 101.2, 99.1, 100.4, 95),
                (100.4, 101.1, 99.2, 99.8, 88),
                (99.8, 100.3, 98.7, 99.5, 91),
            ],
        ),
        h1_candles=_candles(
            "H1",
            [
                (101.0, 101.5, 100.5, 101.0, 100),
                (101.0, 101.2, 100.0, 100.4, 101),
                (100.4, 100.8, 99.2, 99.6, 102),
                (99.6, 100.1, 98.9, 99.3, 103),
                (99.3, 99.8, 98.7, 99.1, 104),
            ],
        ),
        m30_candles=_candles(
            "M30",
            [
                (99.8, 100.0, 99.2, 99.4, 100),
                (99.4, 99.7, 99.0, 99.3, 101),
                (99.3, 99.5, 98.7, 98.9, 102),
            ],
        ),
        m15_candles=_candles(
            "M15",
            [
                (99.7, 99.9, 99.2, 99.4, 100),
                (99.4, 99.6, 98.9, 99.2, 101),
                (99.2, 99.4, 98.4, 98.6, 102),
            ],
        ),
        m10_candles=_candles(
            "M10",
            [
                (99.2, 99.3, 98.8, 99.0, 100),
                (99.0, 99.1, 98.6, 98.8, 101),
                (98.8, 98.9, 98.2, 98.5, 102),
            ],
        ),
        m5_candles=_candles(
            "M5",
            [
                (98.9, 99.0, 98.7, 98.8, 100),
                (98.8, 98.9, 98.4, 98.6, 101),
                (98.6, 98.7, 98.1, 98.4, 102),
            ],
        ),
        risk_buffer=0.05,
    )

    assert result.is_trade is True
    assert result.direction is BreakoutDirection.BEARISH


def test_decision_tree_requires_true_m10_and_m5_inputs():
    result = evaluate_top_down_decision_tree(
        d1_candles=_candles(
            "D1",
            [
                (101.0, 108.0, 99.0, 105.0, 1000),
                (105.0, 112.0, 100.0, 109.0, 1200),
            ],
        ),
        h4_candles=_candles(
            "H4",
            [
                (100.0, 104.0, 98.0, 103.0, 80),
                (103.0, 110.0, 96.0, 108.0, 95),
                (108.0, 109.0, 101.0, 102.0, 70),
                (102.0, 117.0, 100.0, 116.0, 150),
                (116.0, 131.0, 114.0, 128.0, 210),
                (128.0, 129.0, 120.0, 123.0, 130),
                (123.0, 126.0, 119.0, 124.0, 90),
            ],
        ),
        h1_candles=_candles(
            "H1",
            [
                (99.0, 101.0, 98.0, 100.0, 100),
                (100.0, 102.0, 99.5, 101.5, 101),
                (101.5, 104.0, 101.0, 103.0, 102),
                (103.0, 105.0, 102.5, 104.0, 103),
                (104.0, 107.0, 103.5, 106.0, 104),
            ],
        ),
        m30_candles=_candles(
            "M30",
            [
                (104.5, 105.0, 103.4, 104.0, 100),
                (104.0, 104.5, 103.2, 103.9, 101),
                (103.9, 106.4, 103.0, 106.0, 102),
            ],
        ),
        m15_candles=_candles(
            "M15",
            [
                (104.0, 104.5, 103.8, 104.1, 100),
                (104.1, 104.4, 103.9, 104.0, 101),
                (104.0, 106.2, 103.95, 105.9, 102),
                (105.9, 106.3, 104.8, 105.5, 103),
                (105.5, 106.0, 104.7, 105.7, 104),
                (105.7, 106.4, 104.5, 106.1, 105),
            ],
        ),
        risk_buffer=0.05,
    )

    assert result.is_trade is False
    assert result.reason == "m10_data_missing"
    assert result.failed_node == "m10_setup"


def test_decision_tree_uses_asset_specific_risk_buffer():
    from src.config import SymbolStrategyProfile

    profile = SymbolStrategyProfile(
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
        risk_buffer=5.5,
        trail_distance=15.0,
    )

    result = evaluate_top_down_decision_tree(
        d1_candles=_candles(
            "D1",
            [
                (101.0, 108.0, 99.0, 105.0, 1000),
                (105.0, 112.0, 100.0, 109.0, 1200),
            ],
        ),
        h4_candles=_candles(
            "H4",
            [
                (100.0, 104.0, 98.0, 103.0, 80),
                (103.0, 110.0, 96.0, 108.0, 95),
                (108.0, 109.0, 101.0, 102.0, 70),
                (102.0, 117.0, 100.0, 116.0, 150),
                (116.0, 131.0, 114.0, 128.0, 210),
                (128.0, 129.0, 120.0, 123.0, 130),
                (123.0, 126.0, 119.0, 124.0, 90),
            ],
        ),
        h1_candles=_candles(
            "H1",
            [
                (99.0, 101.0, 98.0, 100.0, 100),
                (100.0, 102.0, 99.5, 101.5, 101),
                (101.5, 104.0, 101.0, 103.0, 102),
                (103.0, 105.0, 102.5, 104.0, 103),
                (104.0, 107.0, 103.5, 106.0, 104),
            ],
        ),
        m30_candles=_candles(
            "M30",
            [
                (104.5, 105.0, 103.4, 104.0, 100),
                (104.0, 104.5, 103.2, 103.9, 101),
                (103.9, 106.4, 103.0, 106.0, 102),
            ],
        ),
        m15_candles=_candles(
            "M15",
            [
                (104.0, 104.5, 103.8, 104.1, 100),
                (104.1, 104.4, 103.9, 104.0, 101),
                (104.0, 106.2, 103.95, 105.9, 102),
                (105.9, 106.3, 104.8, 105.5, 103),
                (105.5, 106.0, 104.7, 105.7, 104),
                (105.7, 106.4, 104.5, 106.1, 105),
            ],
        ),
        m10_candles=_candles(
            "M10",
            [
                (105.4, 105.7, 105.1, 105.5, 100),
                (105.5, 105.9, 105.2, 105.7, 101),
                (105.7, 106.1, 105.5, 105.95, 102),
            ],
        ),
        m5_candles=_candles(
            "M5",
            [
                (105.9, 106.0, 105.7, 105.95, 100),
                (105.95, 106.2, 105.8, 106.05, 101),
                (106.05, 106.8, 105.9, 106.4, 102),
            ],
        ),
        risk_buffer=0.05,
        strategy_profile=profile,
    )

    # Let's check the trade plan stop loss matches our 5.5 buffer
    import pytest
    assert result.is_trade is True
    # The structure low is 105.7 on M5. With 5.5 buffer, stop loss should be 105.7 - 5.5 = 100.2
    assert result.stop_loss == pytest.approx(100.2)

    from src.strategy.decision_tree import _default_strategy_profile
    xau_def = _default_strategy_profile("XAUUSD")
    assert xau_def.risk_buffer == 2.0
    assert xau_def.trail_distance == 10.0
