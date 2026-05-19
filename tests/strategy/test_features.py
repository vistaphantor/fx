"""Tests for feature extraction and z-score normalization."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.strategy.features import (
    FeatureExtractor,
    FeatureSnapshot,
    RollingBuffer,
    extract_entry_distance,
    extract_expected_return,
    extract_momentum,
    extract_spread_danger,
    extract_volume,
)
from src.strategy.orderflow import parse_orderflow_payload


class TestRollingBuffer:
    def test_capacity_enforced(self):
        buf = RollingBuffer(capacity=3)
        for i in range(10):
            buf.push(float(i))
        assert len(buf) == 3

    def test_mean(self):
        buf = RollingBuffer(capacity=100)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            buf.push(v)
        assert buf.mean() == pytest.approx(3.0)

    def test_std(self):
        buf = RollingBuffer(capacity=100)
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
            buf.push(v)
        assert buf.std() > 0

    def test_std_single_element_returns_1(self):
        buf = RollingBuffer(capacity=100)
        buf.push(5.0)
        assert buf.std() == 1.0

    def test_z_score_center_value(self):
        buf = RollingBuffer(capacity=100)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            buf.push(v)
        z = buf.z_score(3.0)
        assert abs(z) < 0.01  # Center of the distribution

    def test_z_score_extreme_value(self):
        buf = RollingBuffer(capacity=100)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            buf.push(v)
        z = buf.z_score(10.0)
        assert z > 2.0  # Well above mean

    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError):
            RollingBuffer(capacity=0)


class TestFeatureExtractor:
    def test_first_snapshot_works(self):
        extractor = FeatureExtractor(window=10)
        snapshot = extractor.update(
            momentum_raw=1.0, trend_raw=1.0, volume_raw=1.0,
            order_block_raw=0.5, volatility_risk_raw=1.2,
            entry_distance_raw=0.3, spread_danger_raw=0.1,
            expected_return=0.01, return_std=0.02,
        )
        assert isinstance(snapshot, FeatureSnapshot)
        assert extractor.snapshot_count == 1

    def test_z_scores_evolve_over_time(self):
        extractor = FeatureExtractor(window=10)
        z_scores = []
        for i in range(20):
            snapshot = extractor.update(
                momentum_raw=float(i),
                trend_raw=1.0,
                volume_raw=1.0,
                order_block_raw=0.5,
                volatility_risk_raw=1.0,
                entry_distance_raw=0.3,
                spread_danger_raw=0.1,
            )
            z_scores.append(snapshot.momentum_z)
        # Z-scores should not all be identical
        assert len(set(z_scores)) > 1

    def test_window_respected(self):
        extractor = FeatureExtractor(window=5)
        assert extractor.window == 5

    def test_same_timestamp_only_pushes_one_bar(self):
        extractor = FeatureExtractor(window=10)
        ts = datetime(2026, 4, 28, 6, 15, tzinfo=timezone.utc)

        extractor.update(
            momentum_raw=1.0, trend_raw=1.0, volume_raw=1.0,
            order_block_raw=1.0, volatility_risk_raw=1.0,
            entry_distance_raw=1.0, spread_danger_raw=1.0,
            timestamp=ts,
        )
        extractor.update(
            momentum_raw=2.0, trend_raw=2.0, volume_raw=2.0,
            order_block_raw=2.0, volatility_risk_raw=2.0,
            entry_distance_raw=2.0, spread_danger_raw=2.0,
            timestamp=ts,
        )

        assert extractor.snapshot_count == 1
        assert len(extractor._buffers["momentum"]) == 1

    def test_orderflow_raw_becomes_z_scored_feature(self):
        extractor = FeatureExtractor(window=10)
        signal = parse_orderflow_payload(
            {
                "symbol": "XAUUSD",
                "delta": 1200,
                "buyvolume": 1400,
                "sellvolume": 200,
                "cvd_slope": 0.8,
                "imbalance": "buy_stacked",
                "vwap_bias": "above",
            }
        )

        snapshot = extractor.update(
            momentum_raw=1.0, trend_raw=1.0, volume_raw=1.0,
            order_block_raw=0.5, volatility_risk_raw=1.0,
            entry_distance_raw=0.3, spread_danger_raw=0.1,
            orderflow_raw=signal,
        )

        assert snapshot.orderflow_raw > 0.5
        assert snapshot.orderflow_z == 0.0


class TestExtractMomentum:
    def test_basic_momentum(self):
        class FakeCandle:
            def __init__(self, close, open_p=100, high=101, low=99, volume=100):
                self.close = close
                self.open = open_p
                self.high = high
                self.low = low
                self.volume = volume

        candles = [FakeCandle(100 + i) for i in range(10)]
        momentum = extract_momentum(candles, candles, candles, candles, candles)
        assert momentum != 0.0

    def test_not_enough_candles(self):
        class FakeCandle:
            def __init__(self, close, open_p=100, high=101, low=99, volume=100):
                self.close = close
                self.open = open_p
                self.high = high
                self.low = low
                self.volume = volume

        candles = [FakeCandle(100)]
        assert extract_momentum(candles, candles, candles, candles, candles) == 0.0


class TestExtractVolume:
    def test_normal_volume(self):
        class FakeCandle:
            def __init__(self, volume, open_p=100, high=105, low=95, close=102):
                self.volume = volume
                self.open = open_p
                self.high = high
                self.low = low
                self.close = close

        candles = [FakeCandle(100)] * 19 + [FakeCandle(200)]
        ratio = extract_volume(candles, lookback=20)
        # Last candle has 200, avg of all is (19*100 + 200)/20 = 105
        # body efficiency = |102-100| / (105-95) = 2/10 = 0.2
        # ratio = (200/105) * 0.2 = 1.9 * 0.2 = 0.38
        assert ratio > 0

    def test_empty_candles(self):
        assert extract_volume([], lookback=20) == 1.0


class TestExtractEntryDistance:
    def test_inside_zone(self):
        distance = extract_entry_distance(
            current_price=100.0,
            demand_zones=((99.0, 101.0),),
            supply_zones=((110.0, 115.0),),
        )
        assert distance == 0.0

    def test_between_zones(self):
        distance = extract_entry_distance(
            current_price=105.0,
            demand_zones=((95.0, 100.0),),
            supply_zones=((110.0, 115.0),),
        )
        assert distance > 0.0


class TestExtractSpreadDanger:
    def test_normal_spread(self):
        danger = extract_spread_danger(spread=0.5, atr=10.0)
        assert danger == pytest.approx(0.05)

    def test_zero_atr(self):
        assert extract_spread_danger(spread=0.5, atr=0.0) == 0.0


class TestExtractExpectedReturn:
    def test_positive_trend(self):
        class FakeCandle:
            def __init__(self, close):
                self.close = close

        candles = [FakeCandle(100 + i) for i in range(21)]
        mean_ret, std_ret = extract_expected_return(candles, lookback=20)
        assert mean_ret > 0

    def test_not_enough_candles(self):
        class FakeCandle:
            def __init__(self, close):
                self.close = close

        mean_ret, std_ret = extract_expected_return([FakeCandle(100)], lookback=20)
        assert mean_ret == 0.0

    def test_ewma_weights_recent_returns_more_than_old_outlier(self):
        class FakeCandle:
            def __init__(self, close):
                self.close = close

        candles = [FakeCandle(100.0), FakeCandle(90.0)]
        price = 90.0
        for _ in range(19):
            price *= 1.002
            candles.append(FakeCandle(price))

        mean_ret, _ = extract_expected_return(candles, lookback=20)
        arithmetic_mean = sum(
            (candles[i].close - candles[i - 1].close) / candles[i - 1].close
            for i in range(1, len(candles))
        ) / 20

        assert mean_ret > arithmetic_mean
        assert mean_ret > 0.0
