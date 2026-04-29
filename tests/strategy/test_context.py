from datetime import datetime, timedelta, timezone

import pytest

from src.market_data import Candle
from src.strategy.context import build_daily_context, build_h4_context


def _candles(timeframe: str, rows: list[tuple[float, float, float, float, int]]) -> list[Candle]:
    start = datetime(2026, 4, 6, 0, 0, tzinfo=timezone.utc)
    minutes = {"D1": 1440, "H4": 240}[timeframe]
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


def test_build_daily_context_returns_high_low_and_range_position():
    d1_candles = _candles(
        "D1",
        [
            (101.0, 108.0, 99.0, 105.0, 1000),
            (105.0, 112.0, 100.0, 109.0, 1200),
        ],
    )

    context = build_daily_context(d1_candles, current_price=106.0)

    assert context.daily_high == pytest.approx(112.0)
    assert context.daily_low == pytest.approx(100.0)
    assert context.current_price == pytest.approx(106.0)
    assert context.range_position == pytest.approx(0.5)
    assert context.objective_high == pytest.approx(112.0)
    assert context.objective_low == pytest.approx(100.0)


def test_build_daily_context_tracks_previous_day_and_adr_projection():
    d1_candles = _candles(
        "D1",
        [
            (100.0, 108.0, 99.0, 104.0, 1000),
            (104.0, 111.0, 101.0, 109.0, 1200),
            (109.0, 113.0, 105.0, 112.0, 1400),
        ],
    )

    context = build_daily_context(d1_candles, current_price=114.0)

    assert context.previous_day_high == pytest.approx(113.0)
    assert context.previous_day_low == pytest.approx(105.0)
    assert context.current_day_open == pytest.approx(112.0)
    assert context.adr > 0.0
    assert context.daily_expansion == pytest.approx(2.0)
    assert context.daily_expansion_ratio < 1.0
    assert context.current_day_projection_high > context.previous_day_high


def test_build_h4_context_returns_previous_session_levels_and_volume_profile_markers():
    h4_candles = _candles(
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
    )

    context = build_h4_context(h4_candles)

    assert context.previous_session_high == pytest.approx(131.0)
    assert context.previous_session_low == pytest.approx(96.0)
    assert any(zone[0] == pytest.approx(96.0) for zone in context.demand_zones)
    assert any(zone[1] == pytest.approx(131.0) for zone in context.supply_zones)
    assert context.volume_profile_levels[0] == pytest.approx((131.0 + 114.0 + 128.0) / 3.0)
    assert context.normalized_demand_zones[0][0] == pytest.approx(0.0)
    assert 0.0 <= context.normalized_demand_zones[0][1] <= 1.0
    assert 0.0 <= context.normalized_supply_zones[0][0] <= 1.0
    assert context.normalized_supply_zones[0][1] == pytest.approx(1.0)
