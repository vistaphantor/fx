"""Trade-level risk helpers for breakout retest setups."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.strategy.breakout import BreakoutDirection


@dataclass(frozen=True, slots=True)
class TradeLevels:
    direction: BreakoutDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    risk: float
    reward: float
    risk_reward_ratio: float
    candle_timestamp: datetime
    transaction_cost: float = 0.0
    breakeven_probability: float = 0.0


def _validate_buffer(buffer: float) -> None:
    if buffer <= 0:
        raise ValueError("buffer must be positive")


def build_trade_levels(
    *,
    entry_price: float,
    direction: BreakoutDirection,
    retest_structure_low: float,
    retest_structure_high: float,
    buffer: float,
    candle_timestamp: datetime,
    omega_t: float = 0.0,
    spread: float = 0.0,
) -> TradeLevels:
    """Build trade levels with stop loss beyond the retest structure.

    The reward multiplier scales with omega_t:
    - Base: 3R
    - High quality (omega_t ≥ 0.9): up to 5R
    - Low quality: stays at 3R
    """

    _validate_buffer(buffer)

    entry_price = float(entry_price)
    retest_structure_low = float(retest_structure_low)
    retest_structure_high = float(retest_structure_high)
    buffer = float(buffer)

    # Dynamic R:R based on trade quality.
    # Base 2.5R ensures a positive-expectancy floor; bonus kicks in above omega_t=0.60.
    omega_clamped = max(0.0, min(1.0, omega_t))
    reward_multiplier = 2.5 + 2.5 * max(0.0, (omega_clamped - 0.60) / 0.40)
    reward_multiplier = min(reward_multiplier, 5.0)

    if direction is BreakoutDirection.BULLISH:
        stop_loss = retest_structure_low - buffer
        risk = entry_price - stop_loss
        take_profit = entry_price + (risk * reward_multiplier)
    elif direction is BreakoutDirection.BEARISH:
        stop_loss = retest_structure_high + buffer
        risk = stop_loss - entry_price
        take_profit = entry_price - (risk * reward_multiplier)
    else:
        raise ValueError("direction must be a BreakoutDirection")

    if risk <= 0:
        raise ValueError("entry_price must be beyond the retest structure")

    reward = risk * reward_multiplier
    risk_reward_ratio = reward / risk

    # Transaction cost: spread (round-trip) + 10% slippage buffer on top.
    # Commission is accounted for separately in the live execution layer.
    transaction_cost = max(float(spread), 0.0) * 1.10

    # Breakeven probability: minimum win rate needed to break even
    breakeven_probability = transaction_cost / (reward + transaction_cost) if reward > 0 else 1.0

    return TradeLevels(
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk=risk,
        reward=reward,
        risk_reward_ratio=risk_reward_ratio,
        candle_timestamp=candle_timestamp,
        transaction_cost=transaction_cost,
        breakeven_probability=breakeven_probability,
    )
