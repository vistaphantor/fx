"""Deterministic H1/M15 zone construction helpers.

The strategy tests use small synthetic structures rather than live market
data.  This module keeps the zone-building rules explicit and easy to verify:

- ``SwingPoint`` captures the H1 pivot that defines the zone direction.
- ``M15RefinementBounds`` captures the supporting lower/upper bounds.
- ``build_zones`` pairs swings with refinement bounds in timestamp order and
  returns plain dictionaries to keep assertions straightforward.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class ZoneKind(str, Enum):
    SUPPLY = "SUPPLY"
    DEMAND = "DEMAND"


@dataclass(frozen=True, slots=True)
class SwingPoint:
    timestamp: datetime
    price: float
    kind: ZoneKind


@dataclass(frozen=True, slots=True)
class M15RefinementBounds:
    timestamp: datetime
    low: float
    high: float


def _refinement_buffer() -> float:
    return 3.0


def _build_zone(swing: SwingPoint, bounds: M15RefinementBounds) -> dict[str, Any]:
    major_low = min(bounds.low, bounds.high)
    major_high = max(bounds.low, bounds.high)
    buffer = _refinement_buffer()

    if swing.kind is ZoneKind.SUPPLY:
        refinement_low = max(major_low, swing.price - buffer)
        refinement_high = major_high
    else:
        refinement_low = major_low
        refinement_high = min(major_high, swing.price + buffer)

    return {
        "kind": swing.kind,
        "swing_price": swing.price,
        "major_low": major_low,
        "major_high": major_high,
        "refinement_low": refinement_low,
        "refinement_high": refinement_high,
    }


def build_zones(
    h1_swings: list[SwingPoint],
    m15_bounds: list[M15RefinementBounds],
) -> list[dict[str, Any]]:
    """Build ordered zone dictionaries from H1 swings and M15 bounds.

    The implementation is intentionally deterministic for synthetic test data:
    both inputs are sorted by timestamp and paired by position.
    """

    if len(h1_swings) != len(m15_bounds):
        raise ValueError("h1_swings and m15_bounds must contain the same number of items")

    sorted_swings = sorted(h1_swings, key=lambda item: item.timestamp)
    sorted_bounds = sorted(m15_bounds, key=lambda item: item.timestamp)
    return [_build_zone(swing, bounds) for swing, bounds in zip(sorted_swings, sorted_bounds)]
