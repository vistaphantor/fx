from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.strategy.historical_training import (
    aggregate_consecutive_bars,
    average_true_range_proxy,
    closed_candles_at,
    realized_forward_outcome,
)


def _candle(minute: int, close: float, *, high: float | None = None, low: float | None = None, volume: float = 1.0):
    return SimpleNamespace(
        timestamp=datetime(2026, 1, 1, 10, minute, tzinfo=timezone.utc),
        open=close - 0.2,
        high=close + 0.5 if high is None else high,
        low=close - 0.5 if low is None else low,
        close=close,
        volume=volume,
    )


def test_closed_candles_excludes_still_forming_and_future_bars():
    candles = [_candle(0, 100.0), _candle(15, 101.0), _candle(30, 102.0), _candle(45, 103.0)]

    visible = closed_candles_at(
        candles,
        datetime(2026, 1, 1, 10, 30, tzinfo=timezone.utc),
        bar_duration=timedelta(minutes=15),
    )

    assert [c.timestamp.minute for c in visible] == [0, 15]


def test_closed_candles_limit_keeps_most_recent_observable_history():
    candles = [_candle(0, 100.0), _candle(15, 101.0), _candle(30, 102.0)]

    visible = closed_candles_at(
        candles,
        datetime(2026, 1, 1, 11, 0, tzinfo=timezone.utc),
        bar_duration=timedelta(minutes=15),
        limit=2,
    )

    assert [c.close for c in visible] == [101.0, 102.0]


def test_realized_forward_outcome_uses_actual_later_close_and_cost_gate():
    candles = [_candle(0, 100.0), _candle(15, 100.2), _candle(30, 101.0)]

    profitable = realized_forward_outcome(candles, 0, horizon_bars=2, transaction_cost_ratio=0.005)
    too_expensive = realized_forward_outcome(candles, 0, horizon_bars=2, transaction_cost_ratio=0.02)

    assert profitable is not None
    assert profitable.label == 1
    assert profitable.current_price == 100.0
    assert profitable.future_price == 101.0
    assert profitable.realized_return == pytest.approx(0.01)
    assert too_expensive is not None
    assert too_expensive.label == 0


def test_realized_forward_outcome_fails_closed_without_future_bar():
    candles = [_candle(0, 100.0), _candle(15, 100.2)]
    assert realized_forward_outcome(candles, 1, horizon_bars=1, transaction_cost_ratio=0.0) is None


def test_aggregate_consecutive_bars_preserves_ohlcv_and_drops_partial_group():
    candles = [
        _candle(0, 100.2, high=101.0, low=99.0, volume=2),
        _candle(15, 101.2, high=102.0, low=100.0, volume=3),
        _candle(30, 100.8, high=101.5, low=100.5, volume=4),
    ]

    aggregated = aggregate_consecutive_bars(candles, group_size=2)

    assert len(aggregated) == 1
    assert aggregated[0].open == pytest.approx(100.0)
    assert aggregated[0].high == pytest.approx(102.0)
    assert aggregated[0].low == pytest.approx(99.0)
    assert aggregated[0].close == pytest.approx(101.2)
    assert aggregated[0].volume == pytest.approx(5.0)


def test_average_true_range_proxy_is_causal_range_mean():
    candles = [
        _candle(0, 100.0, high=101.0, low=99.0),
        _candle(15, 101.0, high=103.0, low=100.0),
    ]
    assert average_true_range_proxy(candles, lookback=2) == pytest.approx(2.5)
