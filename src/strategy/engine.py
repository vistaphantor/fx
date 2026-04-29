"""Thin coordinator for the break-and-retest strategy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.market_data import Candle
from src.strategy.breakout import BreakoutDirection, BreakoutResult, detect_breakout
from src.strategy.confirmation import detect_rejection_candle_confirmation
from src.strategy.retest import observe_retest, start_retest_tracking
from src.strategy.risk import build_trade_levels
from src.strategy.session_filter import is_allowed_session


@dataclass(frozen=True, slots=True)
class TradePlan:
    is_trade: bool
    direction: BreakoutDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class NoTradeResult:
    is_trade: bool
    reason: str
    metadata: dict[str, Any]


def _no_trade(reason: str, *, stage: str, **metadata: Any) -> NoTradeResult:
    payload = {"stage": stage, **metadata}
    return NoTradeResult(is_trade=False, reason=reason, metadata=payload)


def _trade_plan(levels, *, confirmation_reason: str) -> TradePlan:
    return TradePlan(
        is_trade=True,
        direction=levels.direction,
        entry_price=levels.entry_price,
        stop_loss=levels.stop_loss,
        take_profit=levels.take_profit,
        reason="trade_plan_ready",
        metadata={"confirmation_reason": confirmation_reason},
    )


def evaluate_break_and_retest_setup(
    *,
    session_timestamp: datetime,
    breakout_candle: Candle,
    retest_candle: Candle,
    zone: dict[str, Any],
    risk_buffer: float,
    max_candles_since_breakout: int,
) -> TradePlan | NoTradeResult:
    """Coordinate the break-and-retest strategy stages."""

    if not is_allowed_session(session_timestamp):
        return _no_trade("session_not_allowed", stage="session_filter", timestamp=session_timestamp)

    breakout = detect_breakout(breakout_candle, zone)
    if not breakout.is_breakout or breakout.direction is None:
        return _no_trade(
            breakout.reason,
            stage="breakout",
            breakout_reason=breakout.reason,
            breakout_timestamp=breakout.candle_timestamp,
        )

    retest_state = start_retest_tracking(
        breakout,
        zone,
        max_candles_since_breakout=max_candles_since_breakout,
    )
    retest_outcome = observe_retest(retest_state, retest_candle)
    if retest_outcome.is_invalidated or not retest_outcome.is_first_touch:
        return _no_trade(
            retest_outcome.reason,
            stage="retest",
            retest_reason=retest_outcome.reason,
            retest_timestamp=retest_outcome.state.breakout.candle_timestamp,
        )

    confirmation = detect_rejection_candle_confirmation(retest_candle, breakout, zone)
    if not confirmation.is_confirmed:
        return _no_trade(
            confirmation.reason,
            stage="confirmation",
            confirmation_reason=confirmation.reason,
            confirmation_timestamp=confirmation.candle_timestamp,
        )

    levels = build_trade_levels(
        entry_price=confirmation.confirmation_price,
        direction=breakout.direction,
        retest_structure_low=zone["refinement_low"],
        retest_structure_high=zone["refinement_high"],
        buffer=risk_buffer,
        candle_timestamp=retest_candle.timestamp,
    )
    return _trade_plan(levels, confirmation_reason=confirmation.reason)
