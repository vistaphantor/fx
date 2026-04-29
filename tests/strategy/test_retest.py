from dataclasses import replace
from datetime import datetime, timezone

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


def test_retest_tracker_allows_only_first_touch_after_breakout():
    from src.strategy.breakout import detect_breakout
    from src.strategy.retest import observe_retest, start_retest_tracking

    breakout_candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 5, tzinfo=timezone.utc),
        open=2348.5,
        high=2352.5,
        low=2347.25,
        close=2351.5,
        volume=120,
        timeframe="M5",
    )
    breakout = detect_breakout(breakout_candle, _supply_zone())
    state = start_retest_tracking(breakout, _supply_zone(), max_candles_since_breakout=3)

    retest_candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 10, tzinfo=timezone.utc),
        open=2351.4,
        high=2351.9,
        low=2350.75,
        close=2351.6,
        volume=104,
        timeframe="M5",
    )

    outcome = observe_retest(state, retest_candle)

    assert outcome.is_first_touch is True
    assert outcome.is_invalidated is False
    assert outcome.reason == "first_touch"
    assert outcome.state.first_touch_timestamp == retest_candle.timestamp
    assert outcome.state.touch_count == 1


def test_retest_tracker_invalidates_second_touch_after_first_retest():
    from src.strategy.breakout import detect_breakout
    from src.strategy.retest import observe_retest, start_retest_tracking

    breakout_candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 5, tzinfo=timezone.utc),
        open=2348.5,
        high=2352.5,
        low=2347.25,
        close=2351.5,
        volume=120,
        timeframe="M5",
    )
    breakout = detect_breakout(breakout_candle, _supply_zone())
    state = start_retest_tracking(breakout, _supply_zone(), max_candles_since_breakout=3)

    first_touch = Candle(
        timestamp=datetime(2026, 4, 4, 8, 10, tzinfo=timezone.utc),
        open=2351.4,
        high=2351.9,
        low=2350.75,
        close=2351.6,
        volume=104,
        timeframe="M5",
    )
    touched = observe_retest(state, first_touch)

    second_touch = Candle(
        timestamp=datetime(2026, 4, 4, 8, 15, tzinfo=timezone.utc),
        open=2351.3,
        high=2352.0,
        low=2350.85,
        close=2351.7,
        volume=109,
        timeframe="M5",
    )
    rejected = observe_retest(touched.state, second_touch)

    assert rejected.is_first_touch is False
    assert rejected.is_invalidated is True
    assert rejected.reason == "retest_already_used"
    assert rejected.state.is_invalidated is True


def test_retest_tracker_invalidates_late_retest_after_window_expires():
    from src.strategy.breakout import detect_breakout
    from src.strategy.retest import observe_retest, start_retest_tracking

    breakout_candle = Candle(
        timestamp=datetime(2026, 4, 4, 13, 20, tzinfo=timezone.utc),
        open=2327.75,
        high=2328.0,
        low=2322.5,
        close=2322.75,
        volume=131,
        timeframe="M5",
    )
    breakout = detect_breakout(breakout_candle, _demand_zone())
    state = start_retest_tracking(breakout, _demand_zone(), max_candles_since_breakout=1)

    late_candle = Candle(
        timestamp=datetime(2026, 4, 4, 13, 25, tzinfo=timezone.utc),
        open=2324.25,
        high=2327.5,
        low=2324.0,
        close=2326.0,
        volume=95,
        timeframe="M5",
    )
    outcome = observe_retest(state, late_candle)

    assert outcome.is_first_touch is False
    assert outcome.is_invalidated is True
    assert outcome.reason == "late_retest"
    assert outcome.state.is_invalidated is True


def test_retest_tracker_invalidates_degraded_first_touch():
    from src.strategy.breakout import detect_breakout
    from src.strategy.retest import observe_retest, start_retest_tracking

    breakout_candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 5, tzinfo=timezone.utc),
        open=2348.5,
        high=2352.5,
        low=2347.25,
        close=2351.5,
        volume=120,
        timeframe="M5",
    )
    breakout = detect_breakout(breakout_candle, _supply_zone())
    state = start_retest_tracking(breakout, _supply_zone(), max_candles_since_breakout=3)

    degraded_retest = Candle(
        timestamp=datetime(2026, 4, 4, 8, 10, tzinfo=timezone.utc),
        open=2350.8,
        high=2351.4,
        low=2346.6,
        close=2347.1,
        volume=111,
        timeframe="M5",
    )
    outcome = observe_retest(state, degraded_retest)

    assert outcome.is_first_touch is False
    assert outcome.is_invalidated is True
    assert outcome.reason == "degraded_retest"
    assert outcome.state.is_invalidated is True


def test_retest_tracker_returns_no_retest_when_price_never_touches_zone():
    from src.strategy.breakout import detect_breakout
    from src.strategy.retest import observe_retest, start_retest_tracking

    breakout_candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 5, tzinfo=timezone.utc),
        open=2348.5,
        high=2352.5,
        low=2347.25,
        close=2351.5,
        volume=120,
        timeframe="M5",
    )
    breakout = detect_breakout(breakout_candle, _supply_zone())
    state = start_retest_tracking(breakout, _supply_zone(), max_candles_since_breakout=3)

    untouched_candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 10, tzinfo=timezone.utc),
        open=2352.25,
        high=2352.75,
        low=2352.0,
        close=2352.5,
        volume=90,
        timeframe="M5",
    )

    outcome = observe_retest(state, untouched_candle)

    assert outcome.is_first_touch is False
    assert outcome.is_invalidated is False
    assert outcome.reason == "no_retest"
    assert outcome.state.touch_count == 0


def test_retest_tracker_passes_through_already_invalidated_state():
    from src.strategy.breakout import detect_breakout
    from src.strategy.retest import observe_retest, start_retest_tracking

    breakout_candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 5, tzinfo=timezone.utc),
        open=2348.5,
        high=2352.5,
        low=2347.25,
        close=2351.5,
        volume=120,
        timeframe="M5",
    )
    breakout = detect_breakout(breakout_candle, _supply_zone())
    state = start_retest_tracking(breakout, _supply_zone(), max_candles_since_breakout=3)
    invalidated_state = replace(state, is_invalidated=True, invalidation_reason="late_retest")

    candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 15, tzinfo=timezone.utc),
        open=2351.3,
        high=2352.0,
        low=2350.85,
        close=2351.7,
        volume=109,
        timeframe="M5",
    )

    outcome = observe_retest(invalidated_state, candle)

    assert outcome.is_first_touch is False
    assert outcome.is_invalidated is True
    assert outcome.reason == "late_retest"
    assert outcome.state.is_invalidated is True


def test_retest_tracker_rejects_non_m5_candle():
    from src.strategy.breakout import detect_breakout
    from src.strategy.retest import observe_retest, start_retest_tracking

    breakout_candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 5, tzinfo=timezone.utc),
        open=2348.5,
        high=2352.5,
        low=2347.25,
        close=2351.5,
        volume=120,
        timeframe="M5",
    )
    breakout = detect_breakout(breakout_candle, _supply_zone())
    state = start_retest_tracking(breakout, _supply_zone(), max_candles_since_breakout=3)

    non_m5_candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 10, tzinfo=timezone.utc),
        open=2351.4,
        high=2351.9,
        low=2350.75,
        close=2351.6,
        volume=104,
        timeframe="M15",
    )

    try:
        observe_retest(state, non_m5_candle)
    except ValueError as exc:
        assert str(exc) == "observe_retest requires an M5 candle"
    else:
        raise AssertionError("observe_retest should reject non-M5 candles")


def test_retest_tracker_rejects_inconsistent_breakout_and_zone():
    from src.strategy.breakout import detect_breakout
    from src.strategy.retest import start_retest_tracking

    breakout_candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 5, tzinfo=timezone.utc),
        open=2348.5,
        high=2352.5,
        low=2347.25,
        close=2351.5,
        volume=120,
        timeframe="M5",
    )
    breakout = detect_breakout(breakout_candle, _supply_zone())

    try:
        start_retest_tracking(breakout, _demand_zone(), max_candles_since_breakout=3)
    except ValueError as exc:
        assert str(exc) == "bullish breakout requires a supply zone"
    else:
        raise AssertionError("start_retest_tracking should reject mismatched zones")


def test_start_retest_tracking_rejects_non_breakout_input():
    from src.strategy.breakout import BreakoutResult
    from src.strategy.retest import start_retest_tracking

    breakout = BreakoutResult(
        is_breakout=False,
        direction=None,
        reason="no_breakout",
        breakout_price=None,
        zone_high=2351.25,
        zone_low=None,
        candle_timestamp=datetime(2026, 4, 4, 8, 5, tzinfo=timezone.utc),
    )

    try:
        start_retest_tracking(breakout, _supply_zone(), max_candles_since_breakout=3)
    except ValueError as exc:
        assert str(exc) == "start_retest_tracking requires a confirmed breakout"
    else:
        raise AssertionError("start_retest_tracking should reject non-breakout input")


def test_start_retest_tracking_rejects_zero_or_negative_window():
    from src.strategy.breakout import detect_breakout
    from src.strategy.retest import start_retest_tracking

    breakout_candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 5, tzinfo=timezone.utc),
        open=2348.5,
        high=2352.5,
        low=2347.25,
        close=2351.5,
        volume=120,
        timeframe="M5",
    )
    breakout = detect_breakout(breakout_candle, _supply_zone())

    try:
        start_retest_tracking(breakout, _supply_zone(), max_candles_since_breakout=0)
    except ValueError as exc:
        assert str(exc) == "max_candles_since_breakout must be at least 1"
    else:
        raise AssertionError("start_retest_tracking should reject zero window")


def test_retest_tracker_invalidates_bearish_degraded_retest():
    from src.strategy.breakout import detect_breakout
    from src.strategy.retest import observe_retest, start_retest_tracking

    breakout_candle = Candle(
        timestamp=datetime(2026, 4, 4, 13, 20, tzinfo=timezone.utc),
        open=2327.75,
        high=2328.0,
        low=2322.5,
        close=2322.75,
        volume=131,
        timeframe="M5",
    )
    breakout = detect_breakout(breakout_candle, _demand_zone())
    state = start_retest_tracking(breakout, _demand_zone(), max_candles_since_breakout=3)

    degraded_retest = Candle(
        timestamp=datetime(2026, 4, 4, 13, 25, tzinfo=timezone.utc),
        open=2323.25,
        high=2328.4,
        low=2323.0,
        close=2325.0,
        volume=99,
        timeframe="M5",
    )

    outcome = observe_retest(state, degraded_retest)

    assert outcome.is_first_touch is False
    assert outcome.is_invalidated is True
    assert outcome.reason == "degraded_retest"
    assert outcome.state.is_invalidated is True
