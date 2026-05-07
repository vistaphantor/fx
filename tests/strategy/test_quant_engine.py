"""Tests for the quant engine: master equation, Ω_t, CVaR, execution rules."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from src.strategy.features import FeatureSnapshot
from src.strategy.quant_engine import (
    OmegaWeights,
    QuantDecision,
    QuantParams,
    apply_execution_rules,
    compute_cara_utility,
    compute_certainty_equivalent,
    compute_cvar,
    compute_omega_signal,
    compute_omega_t,
    evaluate_master_equation,
)


def _make_features(
    *,
    momentum_z=0.0,
    trend_z=0.0,
    volume_z=0.0,
    order_block_z=0.0,
    volatility_risk_z=0.0,
    entry_distance_z=0.0,
    spread_danger_z=0.0,
    expected_return=0.01,
    return_std=0.05,
) -> FeatureSnapshot:
    return FeatureSnapshot(
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        momentum_raw=1.0,
        trend_raw=1.0,
        volume_raw=1.0,
        order_block_raw=1.0,
        volatility_risk_raw=1.0,
        entry_distance_raw=0.5,
        spread_danger_raw=0.1,
        momentum_z=momentum_z,
        trend_z=trend_z,
        volume_z=volume_z,
        order_block_z=order_block_z,
        volatility_risk_z=volatility_risk_z,
        entry_distance_z=entry_distance_z,
        spread_danger_z=spread_danger_z,
        expected_return=expected_return,
        return_std=return_std,
    )


class TestOmegaSignal:
    def test_all_positive_z_scores_gives_high_signal(self):
        features = _make_features(
            momentum_z=2.0, trend_z=2.0, volume_z=2.0, order_block_z=2.0,
            volatility_risk_z=-1.0, entry_distance_z=-1.0, spread_danger_z=-1.0,
        )
        signal = compute_omega_signal(features, OmegaWeights())
        assert signal > 0.9

    def test_all_negative_z_scores_gives_low_signal(self):
        features = _make_features(
            momentum_z=-2.0, trend_z=-2.0, volume_z=-2.0, order_block_z=-2.0,
            volatility_risk_z=2.0, entry_distance_z=2.0, spread_danger_z=2.0,
        )
        signal = compute_omega_signal(features, OmegaWeights())
        assert signal < 0.1

    def test_zero_z_scores_gives_approximately_0_5(self):
        features = _make_features()
        signal = compute_omega_signal(features, OmegaWeights())
        assert 0.45 <= signal <= 0.55

    def test_signal_is_bounded_0_1(self):
        for extreme in [-100, -10, 0, 10, 100]:
            features = _make_features(momentum_z=extreme)
            signal = compute_omega_signal(features, OmegaWeights())
            assert 0.0 <= signal <= 1.0


class TestOmegaT:
    def test_positive_sharpe_and_no_drawdown(self):
        features = _make_features(
            momentum_z=1.5, trend_z=1.5, expected_return=0.02, return_std=0.01,
        )
        omega = compute_omega_t(features, OmegaWeights(), drawdown_dampener=1.0)
        assert omega > 0

    def test_negative_signed_return_does_not_zero_trade_quality(self):
        features = _make_features(expected_return=-0.05, return_std=0.01)
        omega = compute_omega_t(features, OmegaWeights(), drawdown_dampener=1.0)
        # Signed bearish expectancy should still count as quality for a short setup.
        assert omega > 0.4

    def test_full_drawdown_kills_omega(self):
        features = _make_features(expected_return=0.02, return_std=0.01)
        omega = compute_omega_t(features, OmegaWeights(), drawdown_dampener=0.0)
        assert omega == 0.0

    def test_half_drawdown_reduces_omega(self):
        features = _make_features(
            momentum_z=1.0, trend_z=1.0, expected_return=0.02, return_std=0.01,
        )
        omega_full = compute_omega_t(features, OmegaWeights(), drawdown_dampener=1.0)
        omega_half = compute_omega_t(features, OmegaWeights(), drawdown_dampener=0.5)
        assert omega_half < omega_full
        assert omega_half == pytest.approx(omega_full * 0.5, abs=0.01)

    def test_strong_sharpe_does_not_boost_omega_above_raw_signal(self):
        features = _make_features(expected_return=0.02, return_std=0.01)
        omega = compute_omega_t(features, OmegaWeights(), drawdown_dampener=1.0)
        signal = compute_omega_signal(features, OmegaWeights())

        assert omega == pytest.approx(signal)

    def test_poor_sharpe_penalizes_omega(self):
        features = _make_features(expected_return=0.00001, return_std=0.01)
        omega = compute_omega_t(features, OmegaWeights(), drawdown_dampener=1.0)
        signal = compute_omega_signal(features, OmegaWeights())

        assert omega < signal


class TestCVaR:
    def test_empty_returns(self):
        assert compute_cvar([], 0.05) == 0.0

    def test_all_positive_returns(self):
        returns = [0.01, 0.02, 0.03, 0.04, 0.05]
        cvar = compute_cvar(returns, 0.05)
        # Tail is just the worst return, which is positive → CVaR = max(-0.01, 0) = 0 as its loss
        assert cvar == pytest.approx(0.0, abs=0.02)

    def test_negative_tail(self):
        returns = [-0.10, -0.05, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
                   0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        cvar = compute_cvar(returns, 0.05)
        assert cvar > 0  # Should capture the -0.10 tail

    def test_all_negative_returns(self):
        returns = [-0.05, -0.04, -0.03, -0.02, -0.01]
        cvar = compute_cvar(returns, 0.20)
        assert cvar > 0


class TestCARAUtility:
    def test_flat_action_no_cost(self):
        utility = compute_cara_utility(
            gamma=2.0, wealth=1.0, action=0,
            position_fraction=0.01, expected_return=0.01,
            omega_t=0.8, transaction_cost=0.001,
            cvar=0.05, cvar_eta=1.5, drawdown_pct=0.0, dd_rho=0.5,
        )
        # Flat: no trade PnL, no cost, just base wealth
        assert utility < 0  # CARA utility is always negative

    def test_long_with_positive_return_better_than_flat(self):
        utility_flat = compute_cara_utility(
            gamma=2.0, wealth=1.0, action=0,
            position_fraction=0.1, expected_return=0.10,
            omega_t=0.9, transaction_cost=0.0001,
            cvar=0.0001, cvar_eta=0.1, drawdown_pct=0.0, dd_rho=0.5,
        )
        utility_long = compute_cara_utility(
            gamma=2.0, wealth=1.0, action=1,
            position_fraction=0.1, expected_return=0.10,
            omega_t=0.9, transaction_cost=0.0001,
            cvar=0.0001, cvar_eta=0.1, drawdown_pct=0.0, dd_rho=0.5,
        )
        assert utility_long > utility_flat

    def test_short_with_negative_return_better_than_long(self):
        utility_long = compute_cara_utility(
            gamma=2.0, wealth=1.0, action=1,
            position_fraction=0.01, expected_return=-0.05,
            omega_t=0.9, transaction_cost=0.0001,
            cvar=0.001, cvar_eta=1.0, drawdown_pct=0.0, dd_rho=0.5,
        )
        utility_short = compute_cara_utility(
            gamma=2.0, wealth=1.0, action=-1,
            position_fraction=0.01, expected_return=-0.05,
            omega_t=0.9, transaction_cost=0.0001,
            cvar=0.001, cvar_eta=1.0, drawdown_pct=0.0, dd_rho=0.5,
        )
        assert utility_short > utility_long

    def test_trade_pnl_scale_can_overcome_realistic_cost_and_tail(self):
        utility_flat = compute_cara_utility(
            gamma=2.0, wealth=1.0, action=0,
            position_fraction=0.02, expected_return=0.006,
            omega_t=0.8, transaction_cost=0.0002,
            cvar=0.001, cvar_eta=1.5, drawdown_pct=0.0, dd_rho=0.5,
        )
        utility_long = compute_cara_utility(
            gamma=2.0, wealth=1.0, action=1,
            position_fraction=0.02, expected_return=0.006,
            omega_t=0.8, transaction_cost=0.0002,
            cvar=0.001, cvar_eta=1.5, drawdown_pct=0.0, dd_rho=0.5,
        )

        assert utility_long > utility_flat


class TestCertaintyEquivalent:
    def test_positive_continuation_mu_can_outweigh_cost_and_tail(self):
        ce_trade = compute_certainty_equivalent(
            action=1,
            position_fraction=0.02,
            continuation_mu=0.012,
            transaction_cost=0.00012,
            cvar_dir=0.002,
            cvar_eta=1.5,
            drawdown_pct=0.0,
            dd_rho=0.5,
        )
        ce_flat = compute_certainty_equivalent(
            action=0,
            position_fraction=0.02,
            continuation_mu=0.012,
            transaction_cost=0.00012,
            cvar_dir=0.002,
            cvar_eta=1.5,
            drawdown_pct=0.0,
            dd_rho=0.5,
        )

        assert ce_trade > ce_flat

    def test_large_directional_tail_can_make_flat_preferable(self):
        ce_trade = compute_certainty_equivalent(
            action=1,
            position_fraction=0.02,
            continuation_mu=0.006,
            transaction_cost=0.00012,
            cvar_dir=0.020,
            cvar_eta=1.5,
            drawdown_pct=0.0,
            dd_rho=0.5,
        )
        ce_flat = compute_certainty_equivalent(
            action=0,
            position_fraction=0.02,
            continuation_mu=0.006,
            transaction_cost=0.00012,
            cvar_dir=0.020,
            cvar_eta=1.5,
            drawdown_pct=0.0,
            dd_rho=0.5,
        )

        assert ce_trade < ce_flat


class TestExecutionRules:
    def test_flat_action_rejected(self):
        is_trade, reason = apply_execution_rules(
            action=0, omega_t=0.9, expected_return=0.01,
            transaction_cost=0.001, cvar=0.001, cvar_eta=1.5,
            omega_threshold=0.75,
        )
        assert not is_trade
        assert "flat" in reason

    def test_omega_below_threshold_rejected(self):
        is_trade, reason = apply_execution_rules(
            action=1, omega_t=0.5, expected_return=0.01,
            transaction_cost=0.001, cvar=0.001, cvar_eta=1.5,
            omega_threshold=0.75,
        )
        assert not is_trade
        assert "omega_below" in reason

    def test_return_below_cost_plus_cvar_rejected(self):
        is_trade, reason = apply_execution_rules(
            action=1, omega_t=0.9, expected_return=0.001,
            transaction_cost=0.01, cvar=0.05, cvar_eta=1.5,
            omega_threshold=0.75,
        )
        assert not is_trade
        assert "expected_return_below" in reason

    def test_valid_long_approved(self):
        is_trade, reason = apply_execution_rules(
            action=1, omega_t=0.9, expected_return=0.10,
            transaction_cost=0.001, cvar=0.001, cvar_eta=1.5,
            omega_threshold=0.75,
        )
        assert is_trade
        assert "long_approved" in reason

    def test_valid_short_approved(self):
        is_trade, reason = apply_execution_rules(
            action=-1, omega_t=0.85, expected_return=-0.08,
            transaction_cost=0.001, cvar=0.001, cvar_eta=1.0,
            omega_threshold=0.75,
        )
        assert is_trade
        assert "short_approved" in reason


class TestMasterEquation:
    def test_returns_quant_decision(self):
        features = _make_features(
            momentum_z=1.0, trend_z=1.0, volume_z=0.5,
            expected_return=0.02, return_std=0.01,
        )
        decision = evaluate_master_equation(
            features=features,
            params=QuantParams(),
            equity=10000.0,
            drawdown_ratio=0.0,
            recent_returns=[0.01, -0.005, 0.015, 0.02, -0.01],
            transaction_cost=0.0001,
        )
        assert isinstance(decision, QuantDecision)
        assert decision.action in (-1, 0, 1)
        assert 0.0 <= decision.omega_t
        assert -1 in decision.utility_scores
        assert 0 in decision.utility_scores
        assert 1 in decision.utility_scores

    def test_high_drawdown_prevents_trade(self):
        features = _make_features(expected_return=0.02, return_std=0.01)
        decision = evaluate_master_equation(
            features=features,
            params=QuantParams(),
            equity=10000.0,
            drawdown_ratio=1.0,  # Full drawdown
            recent_returns=[0.01, -0.005],
            transaction_cost=0.0001,
        )
        # With full drawdown, Ω_t → 0, so trade should be blocked
        assert not decision.is_trade or decision.omega_t < 0.01

    def test_negative_expected_return_favors_short_or_flat(self):
        features = _make_features(
            momentum_z=-2.0, trend_z=-2.0,
            expected_return=-0.03, return_std=0.01,
        )
        decision = evaluate_master_equation(
            features=features,
            params=QuantParams(),
            equity=10000.0,
            drawdown_ratio=0.0,
            recent_returns=[-0.01, -0.02, -0.005, -0.015, -0.01],
            transaction_cost=0.0001,
        )
        assert decision.action <= 0  # Short or flat

    def test_continuation_quant_can_approve_positive_ev_trade(self):
        features = _make_features(
            momentum_z=0.4,
            trend_z=0.6,
            volume_z=0.3,
            expected_return=0.0005,
            return_std=0.01,
        )
        decision = evaluate_master_equation(
            features=features,
            params=QuantParams(),
            equity=10000.0,
            drawdown_ratio=0.0,
            recent_returns=[0.004, -0.002, 0.003, -0.001, 0.002],
            transaction_cost=0.0001,
            win_rate=0.64,
            avg_win=0.0024,
            avg_loss=0.0009,
            continuation_context={
                "is_continuation_setup": True,
                "continuation_probability": 0.82,
                "mu_cont": 0.012,
                "cvar_dir": 0.002,
            },
        )

        assert decision.is_trade is True
        assert decision.reason == "continuation_quant_approved"
        assert decision.action == 1
        assert decision.metadata["ce_scores"][1] > decision.metadata["ce_scores"][0]

    def test_continuation_quant_can_reduce_borderline_trade(self):
        features = _make_features(
            momentum_z=0.3,
            trend_z=0.4,
            volume_z=0.2,
            expected_return=0.0003,
            return_std=0.01,
        )
        decision = evaluate_master_equation(
            features=features,
            params=QuantParams(),
            equity=10000.0,
            drawdown_ratio=0.0,
            recent_returns=[0.003, -0.002, 0.002, -0.001, 0.001],
            transaction_cost=0.0001,
            win_rate=0.62,
            avg_win=0.0020,
            avg_loss=0.0010,
            continuation_context={
                "is_continuation_setup": True,
                "continuation_probability": 0.62,
                "mu_cont": 0.006,
                "cvar_dir": 0.0032,
            },
        )

        assert decision.is_trade is True
        assert decision.reason == "continuation_quant_reduced"
        assert decision.metadata["lot_multiplier"] < 1.0

    def test_continuation_quant_blocks_when_directional_tail_overwhelms_edge(self):
        features = _make_features(
            momentum_z=0.3,
            trend_z=0.4,
            volume_z=0.2,
            expected_return=0.0003,
            return_std=0.01,
        )
        decision = evaluate_master_equation(
            features=features,
            params=QuantParams(),
            equity=10000.0,
            drawdown_ratio=0.0,
            recent_returns=[0.003, -0.002, 0.002, -0.001, 0.001],
            transaction_cost=0.0001,
            win_rate=0.62,
            avg_win=0.0020,
            avg_loss=0.0010,
            continuation_context={
                "is_continuation_setup": True,
                "continuation_probability": 0.62,
                "mu_cont": 0.005,
                "cvar_dir": 0.020,
            },
        )

        assert decision.is_trade is False
        assert decision.reason == "continuation_quant_blocked"
        assert decision.action == 0

    def test_fresh_entry_quant_remains_flat_without_continuation_context(self):
        features = _make_features(
            momentum_z=0.3,
            trend_z=0.4,
            volume_z=0.2,
            expected_return=0.0003,
            return_std=0.01,
        )
        decision = evaluate_master_equation(
            features=features,
            params=QuantParams(),
            equity=10000.0,
            drawdown_ratio=0.0,
            recent_returns=[0.003, -0.002, 0.002, -0.001, 0.001],
            transaction_cost=0.0001,
        )

        assert decision.is_trade is False
