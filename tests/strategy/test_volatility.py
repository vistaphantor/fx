from datetime import datetime, timedelta, timezone

import pytest

from src.market_data import Candle
from src.strategy.volatility import build_volatility_state, compute_atr


def _candles(rows: list[tuple[float, float, float, float, int]]) -> list[Candle]:
    start = datetime(2026, 4, 28, 0, 0, tzinfo=timezone.utc)
    candles = []
    for index, (open_price, high, low, close, volume) in enumerate(rows):
        candles.append(
            Candle(
                timestamp=start + timedelta(minutes=15 * index),
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                timeframe="M15",
            )
        )
    return candles


def test_compute_atr_returns_expected_value_for_known_candles():
    candles = _candles(
        [
            (100.0, 105.0, 99.0, 103.0, 100),
            (103.0, 110.0, 103.0, 109.0, 120),
            (109.0, 114.0, 107.0, 112.0, 140),
        ]
    )

    atr = compute_atr(candles, period=3)

    assert atr == pytest.approx((6.0 + 7.0 + 7.0) / 3.0)


def test_build_volatility_state_normalizes_body_efficiency_and_range():
    candles = _candles(
        [
            (100.0, 103.0, 99.0, 102.0, 100),
            (102.0, 106.0, 101.0, 105.0, 110),
            (105.0, 110.0, 104.0, 109.0, 120),
            (109.0, 115.0, 108.0, 114.0, 130),
            (114.0, 121.0, 113.0, 120.0, 140),
            (120.0, 128.0, 119.0, 127.0, 150),
        ]
    )

    state = build_volatility_state(candles=candles)

    assert state.short_atr > 0
    assert state.medium_atr > 0
    assert state.realized_range == pytest.approx(29.0)
    assert 0.0 < state.body_efficiency <= 1.0
    assert state.range_expansion_ratio > 1.0
