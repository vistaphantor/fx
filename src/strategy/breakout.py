"""M5 breakout detection helpers.

The breakout rule is intentionally strict:
- a valid breakout requires the candle close to move beyond the zone boundary
- a wick that pierces the zone without the close confirming is rejected

The detector returns a structured result so later strategy stages can inspect
the direction, price, and zone boundary that was broken.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from src.market_data import Candle


class BreakoutDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


@dataclass(frozen=True, slots=True)
class BreakoutResult:
    is_breakout: bool
    direction: BreakoutDirection | None
    reason: str
    breakout_price: float | None
    zone_high: float | None
    zone_low: float | None
    candle_timestamp: datetime


def _zone_kind(zone: dict[str, Any]) -> str:
    kind = zone.get("kind")
    if hasattr(kind, "value"):
        return str(kind.value)
    return str(kind)


def _zone_high(zone: dict[str, Any]) -> float:
    return float(max(zone["major_high"], zone["refinement_high"]))


def _zone_low(zone: dict[str, Any]) -> float:
    return float(min(zone["major_low"], zone["refinement_low"]))


def detect_breakout(candle: Candle, zone: dict[str, Any]) -> BreakoutResult:
    """Detect whether ``candle`` breaks ``zone`` on close.

    The detector only accepts ``M5`` candles.
    Supply zones break bullish when price closes above the zone high.
    Demand zones break bearish when price closes below the zone low.
    """

    if candle.timeframe != "M5":
        raise ValueError("detect_breakout requires an M5 candle")

    zone_kind = _zone_kind(zone)
    zone_high = _zone_high(zone)
    zone_low = _zone_low(zone)

    if zone_kind == "SUPPLY":
        if candle.close > zone_high:
            return BreakoutResult(
                is_breakout=True,
                direction=BreakoutDirection.BULLISH,
                reason="close_beyond_zone",
                breakout_price=candle.close,
                zone_high=zone_high,
                zone_low=None,
                candle_timestamp=candle.timestamp,
            )
        if candle.high > zone_high:
            return BreakoutResult(
                is_breakout=False,
                direction=None,
                reason="wick_only_probe",
                breakout_price=None,
                zone_high=zone_high,
                zone_low=None,
                candle_timestamp=candle.timestamp,
            )

    if zone_kind == "DEMAND":
        if candle.close < zone_low:
            return BreakoutResult(
                is_breakout=True,
                direction=BreakoutDirection.BEARISH,
                reason="close_beyond_zone",
                breakout_price=candle.close,
                zone_high=None,
                zone_low=zone_low,
                candle_timestamp=candle.timestamp,
            )
        if candle.low < zone_low:
            return BreakoutResult(
                is_breakout=False,
                direction=None,
                reason="wick_only_probe",
                breakout_price=None,
                zone_high=None,
                zone_low=zone_low,
                candle_timestamp=candle.timestamp,
            )

    return BreakoutResult(
        is_breakout=False,
        direction=None,
        reason="no_breakout",
        breakout_price=None,
        zone_high=zone_high if zone_kind == "SUPPLY" else None,
        zone_low=zone_low if zone_kind == "DEMAND" else None,
        candle_timestamp=candle.timestamp,
    )
