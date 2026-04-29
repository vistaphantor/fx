from datetime import datetime, timezone

from src.market_data import Candle


def test_detect_breakout_requires_m5_close_beyond_supply_zone():
    from src.strategy.breakout import BreakoutDirection, detect_breakout

    candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 5, tzinfo=timezone.utc),
        open=2348.5,
        high=2352.5,
        low=2347.25,
        close=2351.5,
        volume=120,
        timeframe="M5",
    )
    zone = {
        "kind": "SUPPLY",
        "major_low": 2344.5,
        "major_high": 2351.25,
        "refinement_low": 2347.0,
        "refinement_high": 2351.25,
    }

    result = detect_breakout(candle, zone)

    assert result.is_breakout is True
    assert result.direction is BreakoutDirection.BULLISH
    assert result.reason == "close_beyond_zone"
    assert result.breakout_price == 2351.5
    assert result.zone_high == 2351.25
    assert result.candle_timestamp == datetime(2026, 4, 4, 8, 5, tzinfo=timezone.utc)


def test_detect_breakout_rejects_wick_only_probe_through_supply_zone():
    from src.strategy.breakout import detect_breakout

    candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 10, tzinfo=timezone.utc),
        open=2349.0,
        high=2352.0,
        low=2348.25,
        close=2350.75,
        volume=98,
        timeframe="M5",
    )
    zone = {
        "kind": "SUPPLY",
        "major_low": 2344.5,
        "major_high": 2351.25,
        "refinement_low": 2347.0,
        "refinement_high": 2351.25,
    }

    result = detect_breakout(candle, zone)

    assert result.is_breakout is False
    assert result.reason == "wick_only_probe"
    assert result.direction is None
    assert result.breakout_price is None


def test_detect_breakout_rejects_wick_only_probe_through_demand_zone():
    from src.strategy.breakout import detect_breakout

    candle = Candle(
        timestamp=datetime(2026, 4, 4, 13, 10, tzinfo=timezone.utc),
        open=2325.75,
        high=2326.5,
        low=2322.75,
        close=2324.0,
        volume=105,
        timeframe="M5",
    )
    zone = {
        "kind": "DEMAND",
        "major_low": 2323.75,
        "major_high": 2331.0,
        "refinement_low": 2323.75,
        "refinement_high": 2328.0,
    }

    result = detect_breakout(candle, zone)

    assert result.is_breakout is False
    assert result.reason == "wick_only_probe"
    assert result.direction is None
    assert result.breakout_price is None


def test_detect_breakout_detects_bearish_breakthrough_demand_zone():
    from src.strategy.breakout import BreakoutDirection, detect_breakout

    candle = Candle(
        timestamp=datetime(2026, 4, 4, 13, 20, tzinfo=timezone.utc),
        open=2327.75,
        high=2328.0,
        low=2322.5,
        close=2322.75,
        volume=131,
        timeframe="M5",
    )
    zone = {
        "kind": "DEMAND",
        "major_low": 2323.75,
        "major_high": 2331.0,
        "refinement_low": 2323.75,
        "refinement_high": 2328.0,
    }

    result = detect_breakout(candle, zone)

    assert result.is_breakout is True
    assert result.direction is BreakoutDirection.BEARISH
    assert result.reason == "close_beyond_zone"
    assert result.breakout_price == 2322.75
    assert result.zone_low == 2323.75


def test_detect_breakout_rejects_default_no_breakout_path():
    from src.strategy.breakout import detect_breakout

    candle = Candle(
        timestamp=datetime(2026, 4, 4, 9, 0, tzinfo=timezone.utc),
        open=2349.25,
        high=2351.0,
        low=2348.75,
        close=2350.5,
        volume=87,
        timeframe="M5",
    )
    zone = {
        "kind": "SUPPLY",
        "major_low": 2344.5,
        "major_high": 2351.25,
        "refinement_low": 2347.0,
        "refinement_high": 2351.25,
    }

    result = detect_breakout(candle, zone)

    assert result.is_breakout is False
    assert result.reason == "no_breakout"
    assert result.direction is None
    assert result.breakout_price is None


def test_detect_breakout_treats_exact_boundaries_as_no_breakout():
    from src.strategy.breakout import detect_breakout

    supply_boundary = Candle(
        timestamp=datetime(2026, 4, 4, 8, 20, tzinfo=timezone.utc),
        open=2350.5,
        high=2351.25,
        low=2349.5,
        close=2351.25,
        volume=101,
        timeframe="M5",
    )
    demand_boundary = Candle(
        timestamp=datetime(2026, 4, 4, 13, 40, tzinfo=timezone.utc),
        open=2324.5,
        high=2325.25,
        low=2323.75,
        close=2323.75,
        volume=102,
        timeframe="M5",
    )
    supply_zone = {
        "kind": "SUPPLY",
        "major_low": 2344.5,
        "major_high": 2351.25,
        "refinement_low": 2347.0,
        "refinement_high": 2351.25,
    }
    demand_zone = {
        "kind": "DEMAND",
        "major_low": 2323.75,
        "major_high": 2331.0,
        "refinement_low": 2323.75,
        "refinement_high": 2328.0,
    }

    supply_result = detect_breakout(supply_boundary, supply_zone)
    demand_result = detect_breakout(demand_boundary, demand_zone)

    assert supply_result.is_breakout is False
    assert supply_result.reason == "no_breakout"
    assert demand_result.is_breakout is False
    assert demand_result.reason == "no_breakout"


def test_detect_breakout_rejects_non_m5_candle():
    from src.strategy.breakout import detect_breakout

    candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 5, tzinfo=timezone.utc),
        open=2348.5,
        high=2352.5,
        low=2347.25,
        close=2351.5,
        volume=120,
        timeframe="M15",
    )
    zone = {
        "kind": "SUPPLY",
        "major_low": 2344.5,
        "major_high": 2351.25,
        "refinement_low": 2347.0,
        "refinement_high": 2351.25,
    }

    try:
        detect_breakout(candle, zone)
    except ValueError as exc:
        assert str(exc) == "detect_breakout requires an M5 candle"
    else:
        raise AssertionError("detect_breakout should reject non-M5 candles")
