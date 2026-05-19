from datetime import datetime, timezone

import pytest

from src.strategy.breakout import BreakoutDirection
from src.strategy.orderflow import (
    OrderflowSignalStore,
    parse_orderflow_payload,
    score_orderflow_for_direction,
)


def test_parse_orderflow_payload_normalizes_gocharting_fields():
    signal = parse_orderflow_payload(
        {
            "symbol": "GC",
            "target_symbol": "XAUUSD",
            "timeframe": "M5",
            "timestamp": "2026-05-13T09:05:00+00:00",
            "delta": -1840,
            "buyvolume": 620,
            "sellvolume": 2460,
            "cvd_slope": -0.72,
            "imbalance": "sell_stacked",
            "vwap_bias": "below",
            "absorption": True,
            "profile_location": "below_value_area",
            "liquidity_obstacle": 0.15,
        }
    )

    assert signal.symbol == "XAUUSD"
    assert signal.source_symbol == "GC"
    assert signal.timeframe == "M5"
    assert signal.delta_bias < 0
    assert signal.cvd_slope == pytest.approx(-0.72)
    assert signal.imbalance_score < -0.6
    assert signal.vwap_alignment < 0
    assert signal.absorption_score == 1.0


def test_score_orderflow_rewards_matching_bearish_context():
    signal = parse_orderflow_payload(
        {
            "symbol": "XAUUSD",
            "delta": -1500,
            "buyvolume": 400,
            "sellvolume": 1800,
            "cvd_slope": -0.8,
            "imbalance": "sell_stacked",
            "vwap_bias": "below",
            "profile_location": "below_value_area",
        }
    )

    score = score_orderflow_for_direction(signal, BreakoutDirection.BEARISH)

    assert score.alignment_score > 0.65
    assert score.is_supportive is True
    assert score.is_conflicting is False


def test_score_orderflow_flags_conflicting_bullish_pressure_for_short():
    signal = parse_orderflow_payload(
        {
            "symbol": "XAUUSD",
            "delta": 1200,
            "buyvolume": 1800,
            "sellvolume": 500,
            "cvd_slope": 0.75,
            "imbalance": "buy_stacked",
            "vwap_bias": "above",
        }
    )

    score = score_orderflow_for_direction(signal, BreakoutDirection.BEARISH)

    assert score.alignment_score < -0.45
    assert score.is_conflicting is True


def test_orderflow_store_round_trips_latest_signal(tmp_path):
    store = OrderflowSignalStore(tmp_path / "orderflow_state.json")
    signal = parse_orderflow_payload(
        {
            "symbol": "GC",
            "target_symbol": "XAUUSD",
            "timestamp": datetime(2026, 5, 13, 9, 5, tzinfo=timezone.utc).isoformat(),
            "delta": -500,
        }
    )

    store.record(signal)

    now = datetime(2026, 5, 13, 9, 6, tzinfo=timezone.utc)
    assert store.latest_for("XAUUSD", now=now).source_symbol == "GC"
    assert store.latest_for("xauusd", now=now).delta_bias < 0
