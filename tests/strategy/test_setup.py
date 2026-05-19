from datetime import datetime, timedelta, timezone

from src.market_data import Candle
from src.strategy.breakout import BreakoutDirection
from src.strategy.context import H4Context
from src.strategy.direction import DirectionDecision
from src.strategy.orderflow import parse_orderflow_payload
from src.strategy.setup import evaluate_m10_setup, evaluate_m30_setup
from src.strategy.trigger import TriggerDecision
from src.strategy.tradingview_confluence import TradingViewConfluence


def _candles(rows: list[tuple[float, float, float, float]]) -> list[Candle]:
    start = datetime(2026, 4, 6, 8, 0, tzinfo=timezone.utc)
    candles = []
    for index, (open_price, high, low, close) in enumerate(rows):
        candles.append(
            Candle(
                timestamp=start + timedelta(minutes=30 * index),
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=100 + index,
                timeframe="M30",
            )
        )
    return candles


def test_evaluate_m30_setup_marks_rejection_as_ready():
    direction_decision = DirectionDecision(
        is_valid=True,
        direction=BreakoutDirection.BULLISH,
        reason="h1_bias_bullish",
        metadata={},
    )
    h4_context = H4Context(
        previous_session_high=110.0,
        previous_session_low=96.0,
        demand_zones=((101.0, 104.0),),
        supply_zones=((112.0, 116.0),),
        volume_profile_levels=(109.0, 106.0, 103.0),
    )

    result = evaluate_m30_setup(
        m30_candles=_candles(
            [
                (104.5, 105.0, 103.8, 104.0),
                (104.0, 104.5, 103.6, 103.9),
                (103.9, 106.4, 103.7, 106.0),
            ]
        ),
        direction_decision=direction_decision,
        h4_context=h4_context,
    )

    assert result.is_ready is True
    assert result.reason == "m30_rejection_ready"
    assert result.setup_state == "rejecting_level"
    assert result.quality_score > 0


def test_evaluate_m30_setup_blocks_ready_rejection_when_orderflow_conflicts():
    direction_decision = DirectionDecision(
        is_valid=True,
        direction=BreakoutDirection.BULLISH,
        reason="h1_bias_bullish",
        metadata={},
    )
    h4_context = H4Context(
        previous_session_high=110.0,
        previous_session_low=96.0,
        demand_zones=((101.0, 104.0),),
        supply_zones=((112.0, 116.0),),
        volume_profile_levels=(109.0, 106.0, 103.0),
    )
    orderflow = parse_orderflow_payload(
        {
            "symbol": "XAUUSD",
            "delta": -900,
            "buyvolume": 200,
            "sellvolume": 1100,
            "cvd_slope": -0.7,
            "imbalance": "sell_stacked",
            "vwap_bias": "below",
        }
    )

    result = evaluate_m30_setup(
        m30_candles=_candles(
            [
                (104.5, 105.0, 103.8, 104.0),
                (104.0, 104.5, 103.6, 103.9),
                (103.9, 106.4, 103.7, 106.0),
            ]
        ),
        direction_decision=direction_decision,
        h4_context=h4_context,
        orderflow_signal=orderflow,
    )

    assert result.is_ready is False
    assert result.reason == "m30_orderflow_conflict"
    assert result.metadata["orderflow_alignment"] < -0.35


def test_evaluate_m30_setup_rejects_dirty_midrange_location():
    direction_decision = DirectionDecision(
        is_valid=True,
        direction=BreakoutDirection.BULLISH,
        reason="h1_bias_bullish",
        metadata={},
    )
    h4_context = H4Context(
        previous_session_high=110.0,
        previous_session_low=96.0,
        demand_zones=((100.0, 102.0),),
        supply_zones=((114.0, 116.0),),
        volume_profile_levels=(109.0, 107.0, 103.0),
    )

    result = evaluate_m30_setup(
        m30_candles=_candles(
            [
                (107.0, 108.0, 106.0, 107.5),
                (107.5, 108.5, 107.0, 108.0),
                (108.0, 109.0, 107.5, 108.4),
            ]
        ),
        direction_decision=direction_decision,
        h4_context=h4_context,
    )

    assert result.is_ready is False
    assert result.reason == "m30_midrange_noise"
    assert result.setup_state == "midrange"
    assert result.quality_score < 0.5


def test_evaluate_m30_setup_can_promote_borderline_case_with_tradingview_confluence():
    direction_decision = DirectionDecision(
        is_valid=True,
        direction=BreakoutDirection.BULLISH,
        reason="h1_bias_bullish",
        metadata={},
    )
    confluence = TradingViewConfluence(
        is_active=True,
        reason="tradingview_gap_fill_alignment",
        direction_bonus=2,
        direction_penalty=0,
        setup_bonus=1,
        trigger_bonus=0,
        preferred_direction=BreakoutDirection.BULLISH,
        metadata={},
    )
    h4_context = H4Context(
        previous_session_high=110.0,
        previous_session_low=96.0,
        demand_zones=((100.0, 102.0),),
        supply_zones=((114.0, 116.0),),
        volume_profile_levels=(109.0, 107.0, 103.0),
    )

    result = evaluate_m30_setup(
        m30_candles=_candles(
            [
                (104.0, 104.7, 103.6, 104.5),
                (104.5, 105.2, 104.2, 104.9),
                (104.9, 106.0, 104.7, 105.2),
            ]
        ),
        direction_decision=direction_decision,
        h4_context=h4_context,
        tradingview_confluence=confluence,
    )

    assert result.is_ready is True
    assert result.reason == "m30_tradingview_confluence_ready"
    assert result.quality_score > 0


def test_evaluate_m30_setup_uses_live_current_price_for_bearish_breakaway():
    direction_decision = DirectionDecision(
        is_valid=True,
        direction=BreakoutDirection.BEARISH,
        reason="h1_bias_bearish",
        metadata={"current_price": 98.4},
    )
    h4_context = H4Context(
        previous_session_high=110.0,
        previous_session_low=96.0,
        demand_zones=((100.0, 102.0),),
        supply_zones=((114.0, 116.0),),
        volume_profile_levels=(109.0, 107.0, 103.0),
    )

    result = evaluate_m30_setup(
        m30_candles=_candles(
            [
                (101.0, 101.5, 100.6, 101.2),
                (101.2, 101.4, 100.8, 101.0),
                (101.0, 101.1, 100.7, 100.9),
            ]
        ),
        direction_decision=direction_decision,
        h4_context=h4_context,
    )

    assert result.is_ready is True
    assert result.reason == "m30_breakaway_ready"


def test_evaluate_m30_setup_accepts_bullish_continuation_reacceptance_with_strong_h1_bias():
    direction_decision = DirectionDecision(
        is_valid=True,
        direction=BreakoutDirection.BULLISH,
        reason="h1_bias_bullish",
        metadata={},
        bullish_contribution=8.0,
        bearish_contribution=1.0,
    )
    h4_context = H4Context(
        previous_session_high=110.0,
        previous_session_low=96.0,
        demand_zones=((100.0, 102.0),),
        supply_zones=((112.0, 116.0),),
        volume_profile_levels=(109.0, 107.0, 103.0),
    )

    result = evaluate_m30_setup(
        m30_candles=_candles(
            [
                (103.2, 106.8, 103.0, 106.2),
                (106.2, 107.4, 105.6, 107.0),
                (106.9, 107.2, 105.9, 106.6),
            ]
        ),
        direction_decision=direction_decision,
        h4_context=h4_context,
    )

    assert result.is_ready is True
    assert result.reason == "m30_continuation_reacceptance_ready"
    assert result.setup_state == "continuation_reacceptance"
    assert result.metadata["continuation_reacceptance_score"] >= 0.5
    assert result.metadata["h1_bias_strength"] > 0.5


def test_evaluate_m30_setup_rejects_same_continuation_when_h1_bias_is_weak():
    direction_decision = DirectionDecision(
        is_valid=True,
        direction=BreakoutDirection.BULLISH,
        reason="h1_bias_bullish",
        metadata={},
        bullish_contribution=4.0,
        bearish_contribution=3.0,
    )
    h4_context = H4Context(
        previous_session_high=110.0,
        previous_session_low=96.0,
        demand_zones=((100.0, 102.0),),
        supply_zones=((112.0, 116.0),),
        volume_profile_levels=(109.0, 107.0, 103.0),
    )

    result = evaluate_m30_setup(
        m30_candles=_candles(
            [
                (103.2, 106.8, 103.0, 106.2),
                (106.2, 107.4, 105.6, 107.0),
                (106.9, 107.2, 105.9, 106.6),
            ]
        ),
        direction_decision=direction_decision,
        h4_context=h4_context,
    )

    assert result.is_ready is False
    assert result.reason == "m30_setup_not_ready"
    assert result.setup_state == "continuation_reacceptance"
    assert result.metadata["continuation_reacceptance_score"] >= 0.5
    assert result.metadata["h1_bias_strength"] < 0.5


def test_evaluate_m10_setup_marks_refinement_ready_after_m15_confirmation():
    direction_decision = DirectionDecision(
        is_valid=True,
        direction=BreakoutDirection.BULLISH,
        reason="h1_bias_bullish",
        metadata={},
    )
    confirmation_decision = TriggerDecision(
        is_ready=True,
        reason="m15_trigger_ready",
        entry_price=105.9,
        invalidation_price=103.95,
        metadata={},
        quality_score=1.0,
        expected_move_multiple=1.8,
    )
    h4_context = H4Context(
        previous_session_high=110.0,
        previous_session_low=96.0,
        demand_zones=((101.0, 104.0),),
        supply_zones=((112.0, 116.0),),
        volume_profile_levels=(109.0, 106.0, 103.0),
    )

    result = evaluate_m10_setup(
        m10_candles=[
            Candle(timestamp=datetime(2026, 4, 6, 8, 0, tzinfo=timezone.utc), open=105.2, high=105.4, low=105.0, close=105.3, volume=100, timeframe="M10"),
            Candle(timestamp=datetime(2026, 4, 6, 8, 10, tzinfo=timezone.utc), open=105.3, high=105.6, low=105.1, close=105.5, volume=101, timeframe="M10"),
            Candle(timestamp=datetime(2026, 4, 6, 8, 20, tzinfo=timezone.utc), open=105.5, high=105.95, low=105.35, close=105.8, volume=102, timeframe="M10"),
        ],
        direction_decision=direction_decision,
        h4_context=h4_context,
        confirmation_decision=confirmation_decision,
    )

    assert result.is_ready is True
    assert result.reason == "m10_refinement_ready"


def test_evaluate_m10_setup_accepts_shallow_bearish_continuation_retrace():
    direction_decision = DirectionDecision(
        is_valid=True,
        direction=BreakoutDirection.BEARISH,
        reason="h1_bias_bearish",
        metadata={},
    )
    confirmation_decision = TriggerDecision(
        is_ready=True,
        reason="m15_trigger_ready",
        entry_price=4652.22,
        invalidation_price=4667.89,
        metadata={},
        quality_score=1.0,
        expected_move_multiple=1.3,
    )
    h4_context = H4Context(
        previous_session_high=4725.39,
        previous_session_low=4666.35,
        demand_zones=((4667.11, 4676.89),),
        supply_zones=((4690.17, 4701.25),),
        volume_profile_levels=(4682.44, 4712.48, 4702.78),
    )

    result = evaluate_m10_setup(
        m10_candles=[
            Candle(timestamp=datetime(2026, 4, 28, 7, 50, tzinfo=timezone.utc), open=4667.58, high=4669.6, low=4667.25, close=4667.58, volume=100, timeframe="M10"),
            Candle(timestamp=datetime(2026, 4, 28, 8, 0, tzinfo=timezone.utc), open=4667.57, high=4667.89, low=4648.53, close=4654.01, volume=101, timeframe="M10"),
            Candle(timestamp=datetime(2026, 4, 28, 8, 10, tzinfo=timezone.utc), open=4654.0, high=4655.31, low=4648.86, close=4655.07, volume=102, timeframe="M10"),
        ],
        direction_decision=direction_decision,
        h4_context=h4_context,
        confirmation_decision=confirmation_decision,
    )

    assert result.is_ready is True
    assert result.reason == "m10_continuation_ready"


def test_evaluate_m10_setup_accepts_scored_bearish_continuation_pullback_after_m15_impulse():
    direction_decision = DirectionDecision(
        is_valid=True,
        direction=BreakoutDirection.BEARISH,
        reason="h1_bias_bearish",
        metadata={},
    )
    confirmation_decision = TriggerDecision(
        is_ready=True,
        reason="m15_trigger_ready",
        entry_price=4631.44,
        invalidation_price=4657.73,
        metadata={},
        quality_score=0.8,
        expected_move_multiple=1.08,
    )
    h4_context = H4Context(
        previous_session_high=4725.39,
        previous_session_low=4666.35,
        demand_zones=((4667.11, 4676.89),),
        supply_zones=((4690.17, 4701.25),),
        volume_profile_levels=(4682.44, 4712.48, 4702.78),
    )

    result = evaluate_m10_setup(
        m10_candles=[
            Candle(timestamp=datetime(2026, 4, 28, 8, 30, tzinfo=timezone.utc), open=4657.08, high=4657.73, low=4637.22, close=4637.34, volume=100, timeframe="M10"),
            Candle(timestamp=datetime(2026, 4, 28, 8, 40, tzinfo=timezone.utc), open=4637.28, high=4642.79, low=4631.40, close=4641.05, volume=101, timeframe="M10"),
            Candle(timestamp=datetime(2026, 4, 28, 8, 50, tzinfo=timezone.utc), open=4646.10, high=4646.60, low=4638.89, close=4641.72, volume=102, timeframe="M10"),
        ],
        direction_decision=direction_decision,
        h4_context=h4_context,
        confirmation_decision=confirmation_decision,
    )

    assert result.is_ready is True
    assert result.reason == "m10_continuation_ready"
    assert result.quality_score >= 0.48
    assert result.metadata["setup_kind"] == "bearish_continuation"
    assert float(result.metadata["continuation_retrace"]) > 0.35


def test_evaluate_m10_setup_rejects_deep_bearish_continuation_pullback_after_m15_impulse():
    direction_decision = DirectionDecision(
        is_valid=True,
        direction=BreakoutDirection.BEARISH,
        reason="h1_bias_bearish",
        metadata={},
    )
    confirmation_decision = TriggerDecision(
        is_ready=True,
        reason="m15_trigger_ready",
        entry_price=4631.44,
        invalidation_price=4657.73,
        metadata={},
        quality_score=0.8,
        expected_move_multiple=1.08,
    )
    h4_context = H4Context(
        previous_session_high=4725.39,
        previous_session_low=4666.35,
        demand_zones=((4667.11, 4676.89),),
        supply_zones=((4690.17, 4701.25),),
        volume_profile_levels=(4682.44, 4712.48, 4702.78),
    )

    result = evaluate_m10_setup(
        m10_candles=[
            Candle(timestamp=datetime(2026, 4, 28, 8, 30, tzinfo=timezone.utc), open=4657.08, high=4657.73, low=4637.22, close=4637.34, volume=100, timeframe="M10"),
            Candle(timestamp=datetime(2026, 4, 28, 8, 40, tzinfo=timezone.utc), open=4637.28, high=4654.60, low=4631.40, close=4653.40, volume=101, timeframe="M10"),
            Candle(timestamp=datetime(2026, 4, 28, 8, 50, tzinfo=timezone.utc), open=4656.80, high=4657.10, low=4652.89, close=4654.95, volume=102, timeframe="M10"),
        ],
        direction_decision=direction_decision,
        h4_context=h4_context,
        confirmation_decision=confirmation_decision,
    )

    assert result.is_ready is False
    assert result.reason == "m10_setup_not_ready"
    assert "continuation_score" in result.metadata


def test_evaluate_m10_setup_rewards_orderly_bullish_continuation_with_richer_scores():
    direction_decision = DirectionDecision(
        is_valid=True,
        direction=BreakoutDirection.BULLISH,
        reason="h1_bias_bullish",
        metadata={},
    )
    confirmation_decision = TriggerDecision(
        is_ready=True,
        reason="m15_trigger_ready",
        entry_price=2320.0,
        invalidation_price=2310.0,
        metadata={},
        quality_score=0.9,
        expected_move_multiple=1.6,
    )
    h4_context = H4Context(
        previous_session_high=2355.0,
        previous_session_low=2290.0,
        demand_zones=((2310.0, 2318.0),),
        supply_zones=((2340.0, 2350.0),),
        volume_profile_levels=(2334.0, 2325.0, 2316.0),
    )

    result = evaluate_m10_setup(
        m10_candles=[
            Candle(timestamp=datetime(2026, 4, 28, 10, 0, tzinfo=timezone.utc), open=2318.0, high=2328.0, low=2316.0, close=2327.0, volume=100, timeframe="M10"),
            Candle(timestamp=datetime(2026, 4, 28, 10, 10, tzinfo=timezone.utc), open=2327.0, high=2330.0, low=2322.5, close=2324.2, volume=101, timeframe="M10"),
            Candle(timestamp=datetime(2026, 4, 28, 10, 20, tzinfo=timezone.utc), open=2314.9, high=2317.2, low=2314.5, close=2316.2, volume=102, timeframe="M10"),
        ],
        direction_decision=direction_decision,
        h4_context=h4_context,
        confirmation_decision=confirmation_decision,
    )

    assert result.is_ready is True
    assert result.reason == "m10_continuation_ready"
    assert result.metadata["continuation_score"] >= 0.55
    assert result.metadata["structure_intact"] is True


def test_evaluate_m10_setup_exposes_decay_terms_when_continuation_is_not_ready():
    direction_decision = DirectionDecision(
        is_valid=True,
        direction=BreakoutDirection.BULLISH,
        reason="h1_bias_bullish",
        metadata={},
    )
    confirmation_decision = TriggerDecision(
        is_ready=True,
        reason="m15_trigger_ready",
        entry_price=2320.0,
        invalidation_price=2310.0,
        metadata={},
        quality_score=0.9,
        expected_move_multiple=1.6,
    )
    h4_context = H4Context(
        previous_session_high=2355.0,
        previous_session_low=2290.0,
        demand_zones=((2310.0, 2318.0),),
        supply_zones=((2340.0, 2350.0),),
        volume_profile_levels=(2334.0, 2325.0, 2316.0),
    )

    result = evaluate_m10_setup(
        m10_candles=[
            Candle(timestamp=datetime(2026, 4, 28, 11, 0, tzinfo=timezone.utc), open=2318.0, high=2328.0, low=2316.0, close=2327.0, volume=100, timeframe="M10"),
            Candle(timestamp=datetime(2026, 4, 28, 11, 10, tzinfo=timezone.utc), open=2327.0, high=2328.0, low=2318.5, close=2320.5, volume=101, timeframe="M10"),
            Candle(timestamp=datetime(2026, 4, 28, 11, 20, tzinfo=timezone.utc), open=2320.4, high=2321.0, low=2310.6, close=2312.0, volume=102, timeframe="M10"),
        ],
        direction_decision=direction_decision,
        h4_context=h4_context,
        confirmation_decision=confirmation_decision,
    )

    assert result.is_ready is False
    assert result.reason == "m10_setup_not_ready"
    assert "slope_persistence" in result.metadata
    assert "expansion_decay" in result.metadata
    assert "retrace_damage" in result.metadata
