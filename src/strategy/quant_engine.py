"""Quant decision engine implementing the master equation.

The action selector optimizes incremental certainty-equivalent trade value
instead of applying CARA utility directly to normalized wealth. This keeps
the decision sensitive to actual trade economics, especially for continuation
entries where raw expected return terms are small.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.strategy.features import FeatureSnapshot


@dataclass(frozen=True, slots=True)
class OmegaWeights:
    """Weights for the Omega z-score combination."""

    momentum: float = 1.0
    trend: float = 1.0
    volume: float = 0.8
    order_block: float = 0.9
    volatility_risk: float = 0.7
    entry_distance: float = 0.6
    spread_danger: float = 0.5


@dataclass(frozen=True, slots=True)
class QuantParams:
    """Parameters for the quant decision engine."""

    gamma: float = 2.0
    cvar_alpha: float = 0.05
    cvar_eta: float = 1.5
    dd_rho: float = 0.5
    dd_max: float = 0.20
    omega_threshold: float = 0.75
    position_r_max: float = 0.02
    transaction_lambda: float = 1.0
    omega_weights: OmegaWeights = OmegaWeights()


@dataclass(frozen=True, slots=True)
class QuantDecision:
    """Output of the quant decision engine."""

    action: int
    omega_t: float
    position_size_fraction: float
    expected_return: float
    cvar: float
    drawdown_ratio: float
    drawdown_dampener: float
    utility_scores: dict[int, float]
    sharpe_signal: float
    reason: str
    is_trade: bool
    lot_multiplier: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def compute_omega_signal(features: FeatureSnapshot, weights: OmegaWeights) -> float:
    """Compute the raw sigmoid signal from weighted z-scores."""
    linear = (
        weights.momentum * features.momentum_z
        + weights.trend * features.trend_z
        + weights.volume * features.volume_z
        + weights.order_block * features.order_block_z
        - weights.volatility_risk * features.volatility_risk_z
        - weights.entry_distance * features.entry_distance_z
        - weights.spread_danger * features.spread_danger_z
    )
    return _sigmoid(linear)


def compute_omega_t(
    features: FeatureSnapshot,
    weights: OmegaWeights,
    drawdown_dampener: float,
    bars_per_day: float = 96.0,
) -> float:
    """Compute the full trade-quality multiplier Omega_t."""
    signal = compute_omega_signal(features, weights)
    return_std = max(features.return_std, 1e-9)
    raw_sharpe = abs(features.expected_return) / return_std
    annualized_sharpe = raw_sharpe * math.sqrt(max(bars_per_day, 1.0))
    sharpe_quality = _sigmoid(annualized_sharpe)
    dd_factor = max(0.0, min(1.0, drawdown_dampener))
    return signal * sharpe_quality * dd_factor


def compute_cvar(returns: list[float], alpha: float = 0.05) -> float:
    """Compute Conditional Value at Risk (Expected Shortfall)."""
    if not returns or alpha <= 0 or alpha >= 1:
        return 0.0

    sorted_returns = sorted(returns)
    cutoff_index = max(1, int(math.ceil(len(sorted_returns) * alpha)))
    tail = sorted_returns[:cutoff_index]
    cvar = -sum(tail) / len(tail)
    return max(cvar, 0.0)


def compute_cara_utility(
    *,
    gamma: float,
    wealth: float,
    action: int,
    position_fraction: float,
    expected_return: float,
    omega_t: float,
    transaction_cost: float,
    cvar: float,
    cvar_eta: float,
    drawdown_pct: float,
    dd_rho: float,
) -> float:
    """Legacy CARA utility helper retained for direct tests and compatibility."""
    trade_pnl = action * position_fraction * expected_return * omega_t
    cost_penalty = abs(action) * transaction_cost
    tail_risk = cvar_eta * cvar * abs(action)
    dd_penalty = dd_rho * drawdown_pct
    exponent_arg = wealth + trade_pnl - cost_penalty - tail_risk - dd_penalty
    clamped = max(-500.0, min(500.0, -gamma * exponent_arg))
    return -math.exp(clamped)


def compute_certainty_equivalent(
    *,
    action: int,
    position_fraction: float,
    continuation_mu: float,
    transaction_cost: float,
    cvar_dir: float,
    cvar_eta: float,
    drawdown_pct: float,
    dd_rho: float,
) -> float:
    """Compute incremental certainty-equivalent trade value."""
    trade_edge = action * position_fraction * continuation_mu
    action_penalty = abs(action) * position_fraction * (transaction_cost + (cvar_eta * cvar_dir))
    drawdown_penalty = dd_rho * drawdown_pct
    return trade_edge - action_penalty - drawdown_penalty


def _certainty_equivalent_utility(*, gamma: float, certainty_equivalent: float) -> float:
    """Optional CARA-style transform of incremental trade value for logging."""
    clamped = max(-500.0, min(500.0, -gamma * certainty_equivalent))
    return -math.exp(clamped)


def evaluate_master_equation(
    *,
    features: FeatureSnapshot,
    params: QuantParams,
    equity: float,
    drawdown_ratio: float,
    recent_returns: list[float],
    transaction_cost: float,
    win_rate: float = 0.5,
    avg_win: float = 1.0,
    avg_loss: float = 1.0,
    continuation_context: dict[str, Any] | None = None,
) -> QuantDecision:
    """Evaluate the master equation and return the optimal action."""
    del equity  # Quant ranking is scale-invariant here; sizing happens downstream.

    drawdown_dampener = max(1.0 - min(drawdown_ratio, 1.0), 0.0)
    omega_t = compute_omega_t(features, params.omega_weights, drawdown_dampener)

    return_std = max(features.return_std, 1e-9)
    sharpe_signal = abs(features.expected_return) / return_std
    cvar = compute_cvar(recent_returns, alpha=params.cvar_alpha)

    from src.strategy.equity_tracker import compute_kelly_fraction

    kelly = compute_kelly_fraction(win_rate, avg_win, avg_loss)
    position_fraction = min(params.position_r_max, kelly) * omega_t

    is_continuation_setup = bool(continuation_context and continuation_context.get("is_continuation_setup"))
    continuation_mu = float(features.expected_return)
    cvar_dir = float(cvar)
    if is_continuation_setup:
        continuation_mu = float(continuation_context.get("mu_cont", features.expected_return) or features.expected_return)
        cvar_dir = float(
            continuation_context.get(
                "cvar_dir",
                continuation_context.get("directional_tail_proxy", cvar),
            )
            or cvar
        )

    scaled_transaction_cost = params.transaction_lambda * transaction_cost
    ce_scores: dict[int, float] = {}
    utility_scores: dict[int, float] = {}
    for action in (-1, 0, 1):
        ce_scores[action] = compute_certainty_equivalent(
            action=action,
            position_fraction=position_fraction,
            continuation_mu=continuation_mu,
            transaction_cost=scaled_transaction_cost,
            cvar_dir=cvar_dir,
            cvar_eta=params.cvar_eta,
            drawdown_pct=drawdown_ratio,
            dd_rho=params.dd_rho,
        )
        utility_scores[action] = _certainty_equivalent_utility(
            gamma=params.gamma,
            certainty_equivalent=ce_scores[action],
        )

    best_action = max(ce_scores, key=ce_scores.get)
    lot_multiplier = 1.0

    if is_continuation_setup:
        continuation_action = best_action
        if continuation_action == 0 and continuation_mu != 0.0:
            continuation_action = 1 if continuation_mu > 0 else -1
        is_trade, reason, lot_multiplier = _apply_continuation_execution_rules(
            action=continuation_action,
            ce_scores=ce_scores,
            omega_t=omega_t,
            continuation_mu=continuation_mu,
            cvar_dir=cvar_dir,
            omega_threshold=params.omega_threshold,
            continuation_context=continuation_context or {},
        )
        best_action = continuation_action if is_trade else 0
    else:
        is_trade, reason = apply_execution_rules(
            action=best_action,
            omega_t=omega_t,
            expected_return=features.expected_return,
            transaction_cost=scaled_transaction_cost,
            cvar=cvar,
            cvar_eta=params.cvar_eta,
            omega_threshold=params.omega_threshold,
        )

    if not is_trade:
        best_action = 0

    return QuantDecision(
        action=best_action,
        omega_t=omega_t,
        position_size_fraction=position_fraction * lot_multiplier,
        expected_return=continuation_mu if is_continuation_setup else features.expected_return,
        cvar=cvar,
        drawdown_ratio=drawdown_ratio,
        drawdown_dampener=drawdown_dampener,
        utility_scores=utility_scores,
        sharpe_signal=sharpe_signal,
        reason=reason,
        is_trade=is_trade,
        lot_multiplier=lot_multiplier,
        metadata={
            "kelly_fraction": kelly,
            "omega_signal": compute_omega_signal(features, params.omega_weights),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "lot_multiplier": lot_multiplier,
            "mu_cont": continuation_mu,
            "cvar_dir": cvar_dir,
            "ce_scores": ce_scores,
        },
    )


def apply_execution_rules(
    *,
    action: int,
    omega_t: float,
    expected_return: float,
    transaction_cost: float,
    cvar: float,
    cvar_eta: float,
    omega_threshold: float,
) -> tuple[bool, str]:
    """Apply the fresh-entry execution gate."""
    if action == 0:
        return False, "master_equation_flat"

    if omega_t <= omega_threshold:
        return False, f"omega_below_threshold_{omega_t:.4f}_vs_{omega_threshold}"

    directional_return = action * expected_return
    cost_plus_tail = transaction_cost + cvar_eta * cvar
    if directional_return <= cost_plus_tail:
        return (
            False,
            f"expected_return_below_cost_plus_cvar_{directional_return:.6f}_vs_{cost_plus_tail:.6f}",
        )

    direction_label = "long" if action == 1 else "short"
    return True, f"master_equation_{direction_label}_approved"


def _apply_continuation_execution_rules(
    *,
    action: int,
    ce_scores: dict[int, float],
    omega_t: float,
    continuation_mu: float,
    cvar_dir: float,
    omega_threshold: float,
    continuation_context: dict[str, Any],
) -> tuple[bool, str, float]:
    if action == 0:
        return False, "continuation_quant_blocked", 0.0

    relaxed_omega_floor = max(omega_threshold * 0.35, 0.15)
    if omega_t <= relaxed_omega_floor:
        return False, "continuation_quant_blocked", 0.0

    directional_return = action * continuation_mu
    if directional_return <= 0:
        return False, "continuation_quant_blocked", 0.0

    trade_ce = ce_scores.get(action, float("-inf"))
    flat_ce = ce_scores.get(0, float("-inf"))
    ce_edge = trade_ce - flat_ce
    continuation_probability = float(continuation_context.get("continuation_probability", 0.5) or 0.5)

    if trade_ce <= flat_ce:
        return False, "continuation_quant_blocked", 0.0
    if trade_ce <= 0.0 and continuation_probability < 0.70:
        return False, "continuation_quant_blocked", 0.0
    if cvar_dir >= 0.015 and continuation_probability < 0.75:
        return False, "continuation_quant_blocked", 0.0

    if ce_edge >= 2e-5:
        return True, "continuation_quant_approved", 1.0
    if ce_edge >= 5e-6:
        return True, "continuation_quant_reduced", 0.5
    return False, "continuation_quant_blocked", 0.0
