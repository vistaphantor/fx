from datetime import datetime, timedelta, timezone

from src.market_data import Candle
from src.strategy.breakout import BreakoutDirection
from src.strategy.direction import DirectionDecision
from src.strategy.setup import SetupDecision
from src.strategy.trigger import TriggerDecision, evaluate_m15_trigger, evaluate_m5_trigger


def _candles(rows: list[tuple[float, float, float, float]]) -> list[Candle]:
    start = datetime(2026, 4, 6, 8, 0, tzinfo=timezone.utc)
    candles = []
    for index, (open_price, high, low, close) in enumerate(rows):
        candles.append(
            Candle(
                timestamp=start + timedelta(minutes=15 * index),
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=100 + index,
                timeframe="M15",
            )
        )
    return candles


def test_evaluate_m15_trigger_accepts_velocity_and_candle_intent():
    setup_decision = SetupDecision(
        is_ready=True,
        reason="m30_rejection_ready",
        setup_state="rejecting_level",
        metadata={},
    )
    direction_decision = DirectionDecision(
        is_valid=True,
        direction=BreakoutDirection.BULLISH,
        reason="h1_bias_bullish",
        metadata={},
    )

    result = evaluate_m15_trigger(
        m15_candles=_candles(
            [
                (104.0, 104.5, 103.8, 104.1),
                (104.1, 104.4, 103.9, 104.0),
                (104.0, 106.2, 103.95, 105.9),
            ]
        ),
        setup_decision=setup_decision,
        direction_decision=direction_decision,
    )

    assert result.is_ready is True
    assert result.reason == "m15_trigger_ready"
    assert result.entry_price == 105.9
    assert result.invalidation_price == 103.95
    assert result.quality_score > 0
    assert result.expected_move_multiple > 1.0


def test_evaluate_m15_trigger_blocks_when_velocity_is_missing():
    setup_decision = SetupDecision(
        is_ready=True,
        reason="m30_rejection_ready",
        setup_state="rejecting_level",
        metadata={},
    )
    direction_decision = DirectionDecision(
        is_valid=True,
        direction=BreakoutDirection.BULLISH,
        reason="h1_bias_bullish",
        metadata={},
    )

    result = evaluate_m15_trigger(
        m15_candles=_candles(
            [
                (104.0, 104.3, 103.9, 104.1),
                (104.1, 104.2, 103.95, 104.0),
                (104.0, 104.2, 103.98, 104.05),
            ]
        ),
        setup_decision=setup_decision,
        direction_decision=direction_decision,
    )

    assert result.is_ready is False
    assert result.reason == "m15_velocity_missing"
    assert result.entry_price is None
    assert result.quality_score == 0.0


def test_evaluate_m15_trigger_keeps_recent_confirmation_alive_until_invalidated():
    setup_decision = SetupDecision(
        is_ready=True,
        reason="m30_rejection_ready",
        setup_state="rejecting_level",
        metadata={},
    )
    direction_decision = DirectionDecision(
        is_valid=True,
        direction=BreakoutDirection.BULLISH,
        reason="h1_bias_bullish",
        metadata={},
    )

    result = evaluate_m15_trigger(
        m15_candles=_candles(
            [
                (104.0, 104.5, 103.8, 104.1),
                (104.1, 104.4, 103.9, 104.0),
                (104.0, 106.2, 103.95, 105.9),
                (105.9, 106.0, 104.3, 104.7),
            ]
        ),
        setup_decision=setup_decision,
        direction_decision=direction_decision,
    )

    assert result.is_ready is True
    assert result.reason == "m15_trigger_still_valid"
    assert result.entry_price == 105.9
    assert result.invalidation_price == 103.95
    assert result.quality_score >= 0.8


def test_evaluate_m15_trigger_rejects_old_confirmation_after_invalidation():
    setup_decision = SetupDecision(
        is_ready=True,
        reason="m30_rejection_ready",
        setup_state="rejecting_level",
        metadata={},
    )
    direction_decision = DirectionDecision(
        is_valid=True,
        direction=BreakoutDirection.BULLISH,
        reason="h1_bias_bullish",
        metadata={},
    )

    result = evaluate_m15_trigger(
        m15_candles=_candles(
            [
                (104.0, 104.5, 103.8, 104.1),
                (104.1, 104.4, 103.9, 104.0),
                (104.0, 106.2, 103.95, 105.9),
                (105.9, 106.0, 103.8, 104.1),
            ]
        ),
        setup_decision=setup_decision,
        direction_decision=direction_decision,
    )

    assert result.is_ready is False
    assert result.reason == "m15_velocity_missing"


def test_evaluate_m5_trigger_accepts_execution_after_refinement():
    setup_decision = SetupDecision(
        is_ready=True,
        reason="m10_refinement_ready",
        setup_state="refining",
        metadata={},
        quality_score=0.8,
    )
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

    result = evaluate_m5_trigger(
        m5_candles=[
            Candle(timestamp=datetime(2026, 4, 6, 8, 0, tzinfo=timezone.utc), open=105.7, high=105.9, low=105.6, close=105.8, volume=100, timeframe="M5"),
            Candle(timestamp=datetime(2026, 4, 6, 8, 5, tzinfo=timezone.utc), open=105.8, high=106.0, low=105.7, close=105.9, volume=101, timeframe="M5"),
            Candle(timestamp=datetime(2026, 4, 6, 8, 10, tzinfo=timezone.utc), open=105.9, high=106.4, low=105.85, close=106.2, volume=102, timeframe="M5"),
        ],
        setup_decision=setup_decision,
        direction_decision=direction_decision,
        confirmation_decision=confirmation_decision,
    )

    assert result.is_ready is True
    assert result.reason == "m5_trigger_ready"
    assert result.entry_price == 106.2


def test_evaluate_m5_trigger_accepts_shallow_bearish_continuation_retrace():
    setup_decision = SetupDecision(
        is_ready=True,
        reason="m10_continuation_ready",
        setup_state="refining",
        metadata={},
        quality_score=0.8,
    )
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
        expected_move_multiple=1.36,
    )

    result = evaluate_m5_trigger(
        m5_candles=[
            Candle(timestamp=datetime(2026, 4, 28, 8, 5, tzinfo=timezone.utc), open=4663.09, high=4663.34, low=4648.53, close=4654.01, volume=100, timeframe="M5"),
            Candle(timestamp=datetime(2026, 4, 28, 8, 10, tzinfo=timezone.utc), open=4654.0, high=4654.83, low=4648.86, close=4652.22, volume=101, timeframe="M5"),
            Candle(timestamp=datetime(2026, 4, 28, 8, 15, tzinfo=timezone.utc), open=4655.10, high=4656.00, low=4652.40, close=4653.02, volume=102, timeframe="M5"),
        ],
        setup_decision=setup_decision,
        direction_decision=direction_decision,
        confirmation_decision=confirmation_decision,
    )

    assert result.is_ready is True
    assert result.reason == "m5_continuation_ready"


def test_evaluate_m5_trigger_accepts_scored_bearish_continuation_pullback_after_impulse():
    setup_decision = SetupDecision(
        is_ready=True,
        reason="m10_continuation_ready",
        setup_state="refining",
        metadata={},
        quality_score=0.82,
    )
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
        invalidation_price=4634.18,
        metadata={},
        quality_score=1.0,
        expected_move_multiple=12.38,
    )

    result = evaluate_m5_trigger(
        m5_candles=[
            Candle(timestamp=datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc), open=4631.46, high=4634.04, low=4625.73, close=4633.99, volume=2900, timeframe="M5"),
            Candle(timestamp=datetime(2026, 4, 28, 9, 5, tzinfo=timezone.utc), open=4633.98, high=4634.20, low=4631.40, close=4633.30, volume=1833, timeframe="M5"),
            Candle(timestamp=datetime(2026, 4, 28, 9, 10, tzinfo=timezone.utc), open=4633.62, high=4634.12, low=4632.05, close=4632.82, volume=1920, timeframe="M5"),
        ],
        setup_decision=setup_decision,
        direction_decision=direction_decision,
        confirmation_decision=confirmation_decision,
    )

    assert result.is_ready is True
    assert result.reason == "m5_continuation_ready"
    assert result.quality_score >= 0.48
    assert result.metadata["trigger_kind"] == "bearish_continuation"
    assert float(result.metadata["continuation_retrace"]) > 0.45


def test_evaluate_m5_trigger_rejects_deep_bearish_continuation_pullback_even_after_impulse():
    setup_decision = SetupDecision(
        is_ready=True,
        reason="m10_continuation_ready",
        setup_state="refining",
        metadata={},
        quality_score=0.82,
    )
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
        invalidation_price=4634.18,
        metadata={},
        quality_score=1.0,
        expected_move_multiple=12.38,
    )

    result = evaluate_m5_trigger(
        m5_candles=[
            Candle(timestamp=datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc), open=4631.46, high=4634.04, low=4625.73, close=4633.99, volume=2900, timeframe="M5"),
            Candle(timestamp=datetime(2026, 4, 28, 9, 5, tzinfo=timezone.utc), open=4633.98, high=4634.20, low=4631.40, close=4633.30, volume=1833, timeframe="M5"),
            Candle(timestamp=datetime(2026, 4, 28, 9, 10, tzinfo=timezone.utc), open=4634.10, high=4634.17, low=4632.80, close=4634.05, volume=1920, timeframe="M5"),
        ],
        setup_decision=setup_decision,
        direction_decision=direction_decision,
        confirmation_decision=confirmation_decision,
    )

    assert result.is_ready is False
    assert result.reason == "m5_velocity_missing"
    assert "continuation_score" in result.metadata
