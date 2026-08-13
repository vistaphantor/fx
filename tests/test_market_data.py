from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import numpy as np

from src.market_data import Candle, build_live_strategy_input, candle_from_mt5_rate, timeframe_label


def test_timeframe_label_maps_supported_mt5_timeframes():
    assert timeframe_label("TIMEFRAME_M5") == "M5"
    assert timeframe_label("TIMEFRAME_M10") == "M10"
    assert timeframe_label("TIMEFRAME_M15") == "M15"
    assert timeframe_label("TIMEFRAME_H1") == "H1"


def test_timeframe_label_rejects_unsupported_timeframes():
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        timeframe_label("TIMEFRAME_M999")


@pytest.mark.parametrize("timeframe", ["M5", "H1", "", "timeframe_m5", None])
def test_timeframe_label_rejects_non_explicit_inputs(timeframe):
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        timeframe_label(timeframe)


def test_candle_from_mt5_rate_maps_required_fields():
    rate = SimpleNamespace(
        time=1712224200,
        open=2334.5,
        high=2338,
        low=2332.25,
        close=2336.75,
        tick_volume=17,
    )

    candle = candle_from_mt5_rate(rate, timeframe="TIMEFRAME_M5")

    assert candle == Candle(
        timestamp=datetime(2024, 4, 4, 9, 50, tzinfo=timezone.utc),
        open=2334.5,
        high=2338.0,
        low=2332.25,
        close=2336.75,
        volume=17,
        timeframe="M5",
    )


@pytest.mark.parametrize("volume_field", ["volume", "real_volume"])
def test_candle_from_mt5_rate_uses_fallback_volume_fields(volume_field):
    rate = {
        "time": datetime(2024, 4, 4, 12, 30, tzinfo=timezone.utc),
        "open": 2334.5,
        "high": 2338,
        "low": 2332.25,
        "close": 2336.75,
        volume_field: 19,
    }

    candle = candle_from_mt5_rate(rate, timeframe="TIMEFRAME_H1")

    assert candle.volume == 19


def test_candle_from_mt5_rate_rejects_missing_volume():
    rate = {
        "time": datetime(2024, 4, 4, 12, 30, tzinfo=timezone.utc),
        "open": 2334.5,
        "high": 2338,
        "low": 2332.25,
        "close": 2336.75,
    }

    with pytest.raises(ValueError, match="missing a volume field"):
        candle_from_mt5_rate(rate, timeframe="H1")


def test_candle_from_mt5_rate_normalizes_datetime_and_volume_fields():
    rate = {
        "time": datetime(2024, 4, 4, 12, 30, tzinfo=timezone.utc),
        "open": "2334.5",
        "high": 2338,
        "low": 2332.25,
        "close": 2336.75,
        "tick_volume": "17",
    }

    candle = candle_from_mt5_rate(rate, timeframe="TIMEFRAME_H1")

    assert candle.timestamp == datetime(2024, 4, 4, 12, 30, tzinfo=timezone.utc)
    assert candle.open == 2334.5
    assert candle.high == 2338.0
    assert candle.low == 2332.25
    assert candle.close == 2336.75
    assert candle.volume == 17
    assert candle.timeframe == "H1"


def test_candle_from_mt5_rate_reads_numpy_record_style_rows():
    class RecordLikeRate:
        def __init__(self):
            self.values = {
                "time": 1712224200,
                "open": 2334.5,
                "high": 2338,
                "low": 2332.25,
                "close": 2336.75,
                "tick_volume": 17,
            }

        def __getitem__(self, name):
            return self.values[name]

    candle = candle_from_mt5_rate(RecordLikeRate(), timeframe="TIMEFRAME_M5")

    assert candle.open == 2334.5
    assert candle.volume == 17
    assert candle.timeframe == "M5"


def test_candle_from_mt5_rate_accepts_numpy_integer_timestamp():
    rate = {
        "time": np.int64(1712224200),
        "open": 2334.5,
        "high": 2338,
        "low": 2332.25,
        "close": 2336.75,
        "tick_volume": 17,
    }

    candle = candle_from_mt5_rate(rate, timeframe="TIMEFRAME_M5")

    assert candle.timestamp == datetime(2024, 4, 4, 9, 50, tzinfo=timezone.utc)


def test_candle_from_mt5_rate_rejects_naive_datetime():
    rate = SimpleNamespace(
        time=datetime(2024, 4, 4, 10, 10),
        open=1,
        high=2,
        low=0.5,
        close=1.5,
        tick_volume=1,
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        candle_from_mt5_rate(rate, timeframe="TIMEFRAME_M5")


def test_build_live_strategy_input_fetches_mt5_candles():
    class FakeMt5:
        TIMEFRAME_D1 = "d1"
        TIMEFRAME_H4 = "h4"
        TIMEFRAME_H1 = "h1"
        TIMEFRAME_M30 = "m30"
        TIMEFRAME_M10 = "m10"
        TIMEFRAME_M15 = "m15"
        TIMEFRAME_M5 = "m5"

        def __init__(self):
            self.calls = []

        def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
            self.calls.append((symbol, timeframe, start_pos, count))
            if timeframe == self.TIMEFRAME_D1:
                return [
                    {
                        "time": 1712220000 + (index * 86400),
                        "open": 2330 + index,
                        "high": 2360 + index,
                        "low": 2320 + index,
                        "close": 2349 + index,
                        "tick_volume": 1,
                    }
                    for index in range(count)
                ]
            if timeframe == self.TIMEFRAME_H4:
                return [
                    {
                        "time": 1712220000 + (index * 14400),
                        "open": 2340 + index,
                        "high": 2351.25 + index,
                        "low": 2344.5 + index,
                        "close": 2349 + index,
                        "tick_volume": 1,
                    }
                    for index in range(count)
                ]
            if timeframe == self.TIMEFRAME_H1:
                return [
                    {
                        "time": 1712220000 + (index * 3600),
                        "open": 2340 + index,
                        "high": 2351.25 + index,
                        "low": 2344.5 + index,
                        "close": 2349 + index,
                        "tick_volume": 1,
                    }
                    for index in range(count)
                ]
            if timeframe == self.TIMEFRAME_M30:
                return [
                    {
                        "time": 1712220000 + (index * 1800),
                        "open": 2346 + index,
                        "high": 2351.25 + index,
                        "low": 2347.0 + index,
                        "close": 2349 + index,
                        "tick_volume": 1,
                    }
                    for index in range(count)
                ]
            if timeframe == self.TIMEFRAME_M15:
                return [
                    {
                        "time": 1712220000 + (index * 900),
                        "open": 2346 + index,
                        "high": 2351.25 + index,
                        "low": 2347.0 + index,
                        "close": 2349 + index,
                        "tick_volume": 1,
                    }
                    for index in range(count)
                ]
            if timeframe == self.TIMEFRAME_M10:
                return [
                    {
                        "time": 1712220000 + (index * 600),
                        "open": 2347 + index,
                        "high": 2352.0 + index,
                        "low": 2346.8 + index,
                        "close": 2350 + index,
                        "tick_volume": 1,
                    }
                    for index in range(count)
                ]
            return [
                {
                    "time": 1712223900 + (index * 300),
                    "open": 2348.5 + (index * 0.1),
                    "high": 2352.5 + (index * 0.1),
                    "low": 2347.25 + (index * 0.1),
                    "close": 2351.5 + (index * 0.1),
                    "tick_volume": 1,
                }
                for index in range(count)
            ]

    fake_mt5 = FakeMt5()
    live_input = build_live_strategy_input(fake_mt5, "XAUUSD")

    assert live_input.symbol == "XAUUSD"
    assert len(live_input.d1_candles) == 10
    assert len(live_input.h4_candles) == 14
    assert len(live_input.h1_candles) == 24
    assert len(live_input.m30_candles) == 20
    assert len(live_input.m10_candles) == 8
    assert len(live_input.m5_candles) == 60
    assert live_input.breakout_candle.timeframe == "M5"
    assert live_input.retest_candle.timeframe == "M5"
    assert len(live_input.m15_candles) == 100
    assert len(fake_mt5.calls) == 7
