"""Tests for equity tracking, drawdown computation, and Kelly position sizing."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.strategy.equity_tracker import (
    EquityTracker,
    compute_kelly_fraction,
    compute_position_size,
    compute_price_risk_per_lot,
)


class TestEquityTracker:
    def test_initial_state(self):
        tracker = EquityTracker(dd_max=0.20, initial_equity=10000.0)
        assert tracker.current_equity == 10000.0
        assert tracker.peak_equity == 10000.0
        assert tracker.current_drawdown == 0.0
        assert tracker.drawdown_ratio == 0.0
        assert tracker.drawdown_dampener == 1.0

    def test_new_high_updates_peak(self):
        tracker = EquityTracker(dd_max=0.20, initial_equity=10000.0)
        tracker.update(11000.0)
        assert tracker.peak_equity == 11000.0
        assert tracker.current_drawdown == 0.0

    def test_drawdown_computation(self):
        tracker = EquityTracker(dd_max=0.20, initial_equity=10000.0)
        tracker.update(10000.0)
        tracker.update(9000.0)  # 10% drawdown
        assert tracker.current_drawdown == pytest.approx(1000.0)
        assert tracker.current_drawdown_pct == pytest.approx(0.10)
        assert tracker.drawdown_ratio == pytest.approx(0.50)  # 10% / 20% max
        assert tracker.drawdown_dampener == pytest.approx(0.50)

    def test_full_drawdown(self):
        tracker = EquityTracker(dd_max=0.20, initial_equity=10000.0)
        tracker.update(10000.0)
        tracker.update(8000.0)  # 20% drawdown = DD_max
        assert tracker.drawdown_ratio == pytest.approx(1.0)
        assert tracker.drawdown_dampener == pytest.approx(0.0)

    def test_beyond_max_drawdown_clamped(self):
        tracker = EquityTracker(dd_max=0.20, initial_equity=10000.0)
        tracker.update(10000.0)
        tracker.update(7000.0)  # 30% drawdown > DD_max
        assert tracker.drawdown_ratio == 1.0
        assert tracker.drawdown_dampener == 0.0

    def test_recovery_after_drawdown(self):
        tracker = EquityTracker(dd_max=0.20, initial_equity=10000.0)
        tracker.update(10000.0)
        tracker.update(9000.0)
        tracker.update(10500.0)
        assert tracker.peak_equity == 10500.0
        assert tracker.current_drawdown == 0.0

    def test_history_tracked(self):
        tracker = EquityTracker(dd_max=0.20, initial_equity=10000.0)
        tracker.update(10000.0)
        tracker.update(9500.0)
        tracker.update(10200.0)
        assert len(tracker.history) == 3

    def test_invalid_dd_max(self):
        with pytest.raises(ValueError):
            EquityTracker(dd_max=0.0)
        with pytest.raises(ValueError):
            EquityTracker(dd_max=-0.1)


class TestKellyFraction:
    def test_fifty_fifty_even_odds(self):
        # p=0.5, avg_win=avg_loss → kelly = (0.5*1 - 0.5)/1 = 0
        kelly = compute_kelly_fraction(0.5, 1.0, 1.0)
        assert kelly == pytest.approx(0.0)

    def test_edge(self):
        # p=0.6, b = avg_win/avg_loss = 1.5 → kelly = (0.6*1.5 - 0.4)/1.5 = 0.333
        kelly = compute_kelly_fraction(0.6, 1.5, 1.0)
        assert kelly > 0

    def test_negative_edge_clamped_to_zero(self):
        kelly = compute_kelly_fraction(0.3, 0.5, 1.0)
        assert kelly == 0.0

    def test_invalid_inputs(self):
        assert compute_kelly_fraction(0.5, 0.0, 1.0) == 0.0
        assert compute_kelly_fraction(0.5, 1.0, 0.0) == 0.0


class TestPositionSize:
    def test_minimum_lot_floor(self):
        size = compute_position_size(
            equity=100.0,
            win_rate=0.5,
            avg_win=1.0,
            avg_loss=1.0,
            omega_t=0.9,
            r_max=0.02,
            volume_min=0.01,
            price_per_lot=100000.0,
        )
        assert size == 0.01  # Should be at minimum

    def test_positive_sizing(self):
        size = compute_position_size(
            equity=100000.0,
            win_rate=0.6,
            avg_win=2.0,
            avg_loss=1.0,
            omega_t=0.9,
            r_max=0.02,
            volume_min=0.01,
            volume_step=0.01,
            price_per_lot=100.0,
        )
        assert size >= 0.01

    def test_zero_equity(self):
        size = compute_position_size(
            equity=0.0,
            win_rate=0.6,
            avg_win=2.0,
            avg_loss=1.0,
            omega_t=0.9,
            r_max=0.02,
            volume_min=0.01,
        )
        assert size == 0.01  # Falls to minimum

    def test_omega_zero_gives_minimum(self):
        size = compute_position_size(
            equity=100000.0,
            win_rate=0.6,
            avg_win=2.0,
            avg_loss=1.0,
            omega_t=0.0,
            r_max=0.02,
            volume_min=0.01,
            price_per_lot=100.0,
        )
        assert size == 0.01


class TestPriceRiskPerLot:
    def test_uses_tick_value_and_tick_size_for_stop_distance(self):
        symbol_info = type(
            "SymbolInfo",
            (),
            {
                "trade_tick_value": 1.0,
                "trade_tick_size": 0.01,
                "trade_contract_size": 100.0,
            },
        )()

        risk = compute_price_risk_per_lot(
            entry_price=4700.0,
            stop_loss=4696.0,
            symbol_info=symbol_info,
        )

        assert risk == pytest.approx(400.0)
