"""Entry confirmation helpers for breakout retest setups."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from src.market_data import Candle
from src.strategy.breakout import BreakoutDirection, BreakoutResult


class ConfirmationTriggerType(str, Enum):
    REJECTION_CANDLE = "REJECTION_CANDLE"
    MICRO_STRUCTURE_BREAK = "MICRO_STRUCTURE_BREAK"


@dataclass(frozen=True, slots=True)
class ConfirmationResult:
    is_confirmed: bool
    trigger_type: ConfirmationTriggerType
    reason: str
    confirmation_price: float | None
    breakout_direction: BreakoutDirection | None
    zone_high: float | None
    zone_low: float | None
    candle_timestamp: datetime


def _zone_kind(zone: dict[str, Any]) -> str:
    kind = zone.get("kind")
    if hasattr(kind, "value"):
        return str(kind.value)
    return str(kind)


def _zone_bounds(zone: dict[str, Any]) -> tuple[float, float]:
    return (
        float(min(zone["major_low"], zone["refinement_low"])),
        float(max(zone["major_high"], zone["refinement_high"])),
    )


def _require_m5_candle(candle: Candle) -> None:
    if candle.timeframe != "M5":
        raise ValueError("confirmation helpers require an M5 candle")


def _require_confirmed_breakout(breakout: BreakoutResult) -> None:
    if not breakout.is_breakout or breakout.direction is None:
        raise ValueError("confirmation helpers require a confirmed breakout")


def _confirm_rejection_candle(
    candle: Candle,
    breakout: BreakoutResult,
    zone: dict[str, Any],
) -> ConfirmationResult:
    zone_kind = _zone_kind(zone)
    zone_low, zone_high = _zone_bounds(zone)

    if breakout.direction is BreakoutDirection.BULLISH and zone_kind == "SUPPLY":
        if candle.close > zone_high:
            return ConfirmationResult(
                is_confirmed=True,
                trigger_type=ConfirmationTriggerType.REJECTION_CANDLE,
                reason="rejection_candle_confirmed",
                confirmation_price=candle.close,
                breakout_direction=breakout.direction,
                zone_high=zone_high,
                zone_low=None,
                candle_timestamp=candle.timestamp,
            )

    if breakout.direction is BreakoutDirection.BEARISH and zone_kind == "DEMAND":
        if candle.close < zone_low:
            return ConfirmationResult(
                is_confirmed=True,
                trigger_type=ConfirmationTriggerType.REJECTION_CANDLE,
                reason="rejection_candle_confirmed",
                confirmation_price=candle.close,
                breakout_direction=breakout.direction,
                zone_high=None,
                zone_low=zone_low,
                candle_timestamp=candle.timestamp,
            )

    return ConfirmationResult(
        is_confirmed=False,
        trigger_type=ConfirmationTriggerType.REJECTION_CANDLE,
        reason="no_rejection_confirmation",
        confirmation_price=None,
        breakout_direction=breakout.direction,
        zone_high=zone_high if zone_kind == "SUPPLY" else None,
        zone_low=zone_low if zone_kind == "DEMAND" else None,
        candle_timestamp=candle.timestamp,
    )


def detect_rejection_candle_confirmation(
    candle: Candle,
    breakout: BreakoutResult,
    zone: dict[str, Any],
) -> ConfirmationResult:
    """Confirm a breakout retest using a candle close beyond the zone boundary."""

    _require_m5_candle(candle)
    _require_confirmed_breakout(breakout)
    return _confirm_rejection_candle(candle, breakout, zone)


def detect_micro_structure_break_confirmation(
    candle: Candle,
    breakout: BreakoutResult,
    *,
    micro_structure_high: float,
    micro_structure_low: float,
) -> ConfirmationResult:
    """Confirm a breakout retest using a micro-structure close break."""

    _require_m5_candle(candle)
    _require_confirmed_breakout(breakout)

    if breakout.direction is BreakoutDirection.BULLISH:
        if candle.close > micro_structure_high:
            return ConfirmationResult(
                is_confirmed=True,
                trigger_type=ConfirmationTriggerType.MICRO_STRUCTURE_BREAK,
                reason="micro_structure_break_confirmed",
                confirmation_price=candle.close,
                breakout_direction=breakout.direction,
                zone_high=None,
                zone_low=None,
                candle_timestamp=candle.timestamp,
            )
    elif breakout.direction is BreakoutDirection.BEARISH:
        if candle.close < micro_structure_low:
            return ConfirmationResult(
                is_confirmed=True,
                trigger_type=ConfirmationTriggerType.MICRO_STRUCTURE_BREAK,
                reason="micro_structure_break_confirmed",
                confirmation_price=candle.close,
                breakout_direction=breakout.direction,
                zone_high=None,
                zone_low=None,
                candle_timestamp=candle.timestamp,
            )

    return ConfirmationResult(
        is_confirmed=False,
        trigger_type=ConfirmationTriggerType.MICRO_STRUCTURE_BREAK,
        reason="no_micro_structure_break",
        confirmation_price=None,
        breakout_direction=breakout.direction,
        zone_high=None,
        zone_low=None,
        candle_timestamp=candle.timestamp,
    )
