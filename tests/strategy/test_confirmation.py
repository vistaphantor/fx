from datetime import datetime, timezone

import pytest

from src.market_data import Candle


def _supply_zone() -> dict[str, float | str]:
    return {
        "kind": "SUPPLY",
        "major_low": 2344.5,
        "major_high": 2351.25,
        "refinement_low": 2347.0,
        "refinement_high": 2351.25,
    }


def _demand_zone() -> dict[str, float | str]:
    return {
        "kind": "DEMAND",
        "major_low": 2323.75,
        "major_high": 2331.0,
        "refinement_low": 2323.75,
        "refinement_high": 2328.0,
    }


def test_detect_rejection_candle_confirmation_returns_bullish_trigger_for_supply_retest():
    from src.strategy.breakout import BreakoutDirection, BreakoutResult
    from src.strategy.confirmation import ConfirmationTriggerType, detect_rejection_candle_confirmation

    breakout = BreakoutResult(
        is_breakout=True,
        direction=BreakoutDirection.BULLISH,
        reason="close_beyond_zone",
        breakout_price=2351.5,
        zone_high=2351.25,
        zone_low=None,
        candle_timestamp=datetime(2026, 4, 4, 8, 5, tzinfo=timezone.utc),
    )
    candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 10, tzinfo=timezone.utc),
        open=2351.1,
        high=2351.9,
        low=2347.8,
        close=2351.45,
        volume=104,
        timeframe="M5",
    )

    result = detect_rejection_candle_confirmation(candle, breakout, _supply_zone())

    assert result.is_confirmed is True
    assert result.trigger_type is ConfirmationTriggerType.REJECTION_CANDLE
    assert result.reason == "rejection_candle_confirmed"
    assert result.confirmation_price == 2351.45
    assert result.zone_high == 2351.25
    assert result.breakout_direction is BreakoutDirection.BULLISH


def test_detect_rejection_candle_confirmation_returns_bearish_trigger_for_demand_retest():
    from src.strategy.breakout import BreakoutDirection, BreakoutResult
    from src.strategy.confirmation import ConfirmationTriggerType, detect_rejection_candle_confirmation

    breakout = BreakoutResult(
        is_breakout=True,
        direction=BreakoutDirection.BEARISH,
        reason="close_beyond_zone",
        breakout_price=2322.75,
        zone_high=None,
        zone_low=2323.75,
        candle_timestamp=datetime(2026, 4, 4, 13, 20, tzinfo=timezone.utc),
    )
    candle = Candle(
        timestamp=datetime(2026, 4, 4, 13, 25, tzinfo=timezone.utc),
        open=2323.1,
        high=2328.1,
        low=2323.0,
        close=2323.25,
        volume=99,
        timeframe="M5",
    )

    result = detect_rejection_candle_confirmation(candle, breakout, _demand_zone())

    assert result.is_confirmed is True
    assert result.trigger_type is ConfirmationTriggerType.REJECTION_CANDLE
    assert result.reason == "rejection_candle_confirmed"
    assert result.confirmation_price == 2323.25
    assert result.zone_low == 2323.75
    assert result.breakout_direction is BreakoutDirection.BEARISH


def test_detect_rejection_candle_confirmation_rejects_wick_only_probe():
    from src.strategy.breakout import BreakoutDirection, BreakoutResult
    from src.strategy.confirmation import ConfirmationTriggerType, detect_rejection_candle_confirmation

    breakout = BreakoutResult(
        is_breakout=True,
        direction=BreakoutDirection.BULLISH,
        reason="close_beyond_zone",
        breakout_price=2351.5,
        zone_high=2351.25,
        zone_low=None,
        candle_timestamp=datetime(2026, 4, 4, 8, 5, tzinfo=timezone.utc),
    )
    candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 10, tzinfo=timezone.utc),
        open=2350.9,
        high=2352.0,
        low=2348.4,
        close=2351.0,
        volume=97,
        timeframe="M5",
    )

    result = detect_rejection_candle_confirmation(candle, breakout, _supply_zone())

    assert result.is_confirmed is False
    assert result.trigger_type is ConfirmationTriggerType.REJECTION_CANDLE
    assert result.reason == "no_rejection_confirmation"
    assert result.confirmation_price is None


@pytest.mark.parametrize(
    ("breakout_direction", "zone_factory", "close_price"),
    [
        ("BULLISH", _supply_zone, 2351.25),
        ("BEARISH", _demand_zone, 2323.75),
    ],
)
def test_detect_rejection_candle_confirmation_rejects_exact_boundary_closes(
    breakout_direction,
    zone_factory,
    close_price,
):
    from src.strategy.breakout import BreakoutDirection, BreakoutResult
    from src.strategy.confirmation import ConfirmationTriggerType, detect_rejection_candle_confirmation

    direction = getattr(BreakoutDirection, breakout_direction)
    zone = zone_factory()
    breakout = BreakoutResult(
        is_breakout=True,
        direction=direction,
        reason="close_beyond_zone",
        breakout_price=close_price,
        zone_high=zone.get("major_high") if direction is BreakoutDirection.BULLISH else None,
        zone_low=zone.get("major_low") if direction is BreakoutDirection.BEARISH else None,
        candle_timestamp=datetime(2026, 4, 4, 8, 5, tzinfo=timezone.utc),
    )
    candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 10, tzinfo=timezone.utc),
        open=close_price,
        high=close_price,
        low=close_price,
        close=close_price,
        volume=100,
        timeframe="M5",
    )

    result = detect_rejection_candle_confirmation(candle, breakout, zone)

    assert result.is_confirmed is False
    assert result.trigger_type is ConfirmationTriggerType.REJECTION_CANDLE
    assert result.reason == "no_rejection_confirmation"
    assert result.confirmation_price is None


def test_detect_micro_structure_break_confirmation_returns_bullish_trigger():
    from src.strategy.breakout import BreakoutDirection, BreakoutResult
    from src.strategy.confirmation import ConfirmationTriggerType, detect_micro_structure_break_confirmation

    breakout = BreakoutResult(
        is_breakout=True,
        direction=BreakoutDirection.BULLISH,
        reason="close_beyond_zone",
        breakout_price=2351.5,
        zone_high=2351.25,
        zone_low=None,
        candle_timestamp=datetime(2026, 4, 4, 8, 5, tzinfo=timezone.utc),
    )
    candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 15, tzinfo=timezone.utc),
        open=2351.0,
        high=2352.0,
        low=2350.6,
        close=2351.8,
        volume=113,
        timeframe="M5",
    )

    result = detect_micro_structure_break_confirmation(
        candle,
        breakout,
        micro_structure_high=2351.6,
        micro_structure_low=2350.4,
    )

    assert result.is_confirmed is True
    assert result.trigger_type is ConfirmationTriggerType.MICRO_STRUCTURE_BREAK
    assert result.reason == "micro_structure_break_confirmed"
    assert result.confirmation_price == 2351.8
    assert result.breakout_direction is BreakoutDirection.BULLISH


def test_detect_micro_structure_break_confirmation_returns_bearish_trigger():
    from src.strategy.breakout import BreakoutDirection, BreakoutResult
    from src.strategy.confirmation import ConfirmationTriggerType, detect_micro_structure_break_confirmation

    breakout = BreakoutResult(
        is_breakout=True,
        direction=BreakoutDirection.BEARISH,
        reason="close_beyond_zone",
        breakout_price=2322.75,
        zone_high=None,
        zone_low=2323.75,
        candle_timestamp=datetime(2026, 4, 4, 13, 20, tzinfo=timezone.utc),
    )
    candle = Candle(
        timestamp=datetime(2026, 4, 4, 13, 30, tzinfo=timezone.utc),
        open=2323.2,
        high=2323.5,
        low=2322.4,
        close=2322.55,
        volume=122,
        timeframe="M5",
    )

    result = detect_micro_structure_break_confirmation(
        candle,
        breakout,
        micro_structure_high=2323.4,
        micro_structure_low=2322.7,
    )

    assert result.is_confirmed is True
    assert result.trigger_type is ConfirmationTriggerType.MICRO_STRUCTURE_BREAK
    assert result.reason == "micro_structure_break_confirmed"
    assert result.confirmation_price == 2322.55
    assert result.breakout_direction is BreakoutDirection.BEARISH


def test_detect_micro_structure_break_confirmation_rejects_wick_only_break():
    from src.strategy.breakout import BreakoutDirection, BreakoutResult
    from src.strategy.confirmation import ConfirmationTriggerType, detect_micro_structure_break_confirmation

    breakout = BreakoutResult(
        is_breakout=True,
        direction=BreakoutDirection.BULLISH,
        reason="close_beyond_zone",
        breakout_price=2351.5,
        zone_high=2351.25,
        zone_low=None,
        candle_timestamp=datetime(2026, 4, 4, 8, 5, tzinfo=timezone.utc),
    )
    candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 15, tzinfo=timezone.utc),
        open=2351.0,
        high=2351.9,
        low=2350.8,
        close=2351.45,
        volume=113,
        timeframe="M5",
    )

    result = detect_micro_structure_break_confirmation(
        candle,
        breakout,
        micro_structure_high=2351.6,
        micro_structure_low=2350.4,
    )

    assert result.is_confirmed is False
    assert result.trigger_type is ConfirmationTriggerType.MICRO_STRUCTURE_BREAK
    assert result.reason == "no_micro_structure_break"
    assert result.confirmation_price is None


@pytest.mark.parametrize(
    ("breakout_direction", "close_price", "micro_structure_high", "micro_structure_low"),
    [
        ("BULLISH", 2351.6, 2351.6, 2350.4),
        ("BEARISH", 2322.7, 2323.4, 2322.7),
    ],
)
def test_detect_micro_structure_break_confirmation_rejects_exact_boundary_closes(
    breakout_direction,
    close_price,
    micro_structure_high,
    micro_structure_low,
):
    from src.strategy.breakout import BreakoutDirection, BreakoutResult
    from src.strategy.confirmation import ConfirmationTriggerType, detect_micro_structure_break_confirmation

    direction = getattr(BreakoutDirection, breakout_direction)
    breakout = BreakoutResult(
        is_breakout=True,
        direction=direction,
        reason="close_beyond_zone",
        breakout_price=close_price,
        zone_high=2351.25 if direction is BreakoutDirection.BULLISH else None,
        zone_low=2323.75 if direction is BreakoutDirection.BEARISH else None,
        candle_timestamp=datetime(2026, 4, 4, 8, 5, tzinfo=timezone.utc),
    )
    candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 15, tzinfo=timezone.utc),
        open=close_price,
        high=close_price,
        low=close_price,
        close=close_price,
        volume=113,
        timeframe="M5",
    )

    result = detect_micro_structure_break_confirmation(
        candle,
        breakout,
        micro_structure_high=micro_structure_high,
        micro_structure_low=micro_structure_low,
    )

    assert result.is_confirmed is False
    assert result.trigger_type is ConfirmationTriggerType.MICRO_STRUCTURE_BREAK
    assert result.reason == "no_micro_structure_break"
    assert result.confirmation_price is None


def test_confirmation_helpers_reject_non_m5_candles():
    from src.strategy.breakout import BreakoutDirection, BreakoutResult
    from src.strategy.confirmation import detect_micro_structure_break_confirmation, detect_rejection_candle_confirmation

    breakout = BreakoutResult(
        is_breakout=True,
        direction=BreakoutDirection.BULLISH,
        reason="close_beyond_zone",
        breakout_price=2351.5,
        zone_high=2351.25,
        zone_low=None,
        candle_timestamp=datetime(2026, 4, 4, 8, 5, tzinfo=timezone.utc),
    )
    candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 10, tzinfo=timezone.utc),
        open=2351.1,
        high=2351.9,
        low=2347.8,
        close=2351.45,
        volume=104,
        timeframe="M15",
    )

    try:
        detect_rejection_candle_confirmation(candle, breakout, _supply_zone())
    except ValueError as exc:
        assert str(exc) == "confirmation helpers require an M5 candle"
    else:
        raise AssertionError("detect_rejection_candle_confirmation should reject non-M5 candles")

    try:
        detect_micro_structure_break_confirmation(
            candle,
            breakout,
            micro_structure_high=2351.6,
            micro_structure_low=2350.4,
        )
    except ValueError as exc:
        assert str(exc) == "confirmation helpers require an M5 candle"
    else:
        raise AssertionError("detect_micro_structure_break_confirmation should reject non-M5 candles")


def test_confirmation_helpers_reject_malformed_breakout_state():
    from src.strategy.breakout import BreakoutDirection, BreakoutResult
    from src.strategy.confirmation import detect_micro_structure_break_confirmation, detect_rejection_candle_confirmation

    malformed_breakout = BreakoutResult(
        is_breakout=False,
        direction=BreakoutDirection.BULLISH,
        reason="wick_only_probe",
        breakout_price=None,
        zone_high=2351.25,
        zone_low=None,
        candle_timestamp=datetime(2026, 4, 4, 8, 5, tzinfo=timezone.utc),
    )
    candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 10, tzinfo=timezone.utc),
        open=2351.1,
        high=2351.9,
        low=2347.8,
        close=2351.45,
        volume=104,
        timeframe="M5",
    )

    try:
        detect_rejection_candle_confirmation(candle, malformed_breakout, _supply_zone())
    except ValueError as exc:
        assert str(exc) == "confirmation helpers require a confirmed breakout"
    else:
        raise AssertionError("detect_rejection_candle_confirmation should reject malformed breakout state")

    try:
        detect_micro_structure_break_confirmation(
            candle,
            malformed_breakout,
            micro_structure_high=2351.6,
            micro_structure_low=2350.4,
        )
    except ValueError as exc:
        assert str(exc) == "confirmation helpers require a confirmed breakout"
    else:
        raise AssertionError("detect_micro_structure_break_confirmation should reject malformed breakout state")
