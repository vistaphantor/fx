from datetime import datetime, timezone

import pytest

from src.strategy.zones import M15RefinementBounds, SwingPoint, ZoneKind, build_zones


def test_build_zones_sorts_inputs_before_pairing():
    h1_swings = [
        SwingPoint(timestamp=datetime(2026, 4, 4, 7, 0, tzinfo=timezone.utc), price=2325.0, kind=ZoneKind.DEMAND),
        SwingPoint(timestamp=datetime(2026, 4, 4, 6, 0, tzinfo=timezone.utc), price=2350.0, kind=ZoneKind.SUPPLY),
    ]
    m15_bounds = [
        M15RefinementBounds(timestamp=datetime(2026, 4, 4, 7, 15, tzinfo=timezone.utc), low=2323.75, high=2331.0),
        M15RefinementBounds(timestamp=datetime(2026, 4, 4, 6, 15, tzinfo=timezone.utc), low=2344.5, high=2351.25),
    ]

    zones = build_zones(h1_swings, m15_bounds)

    assert zones == [
        {
            "kind": ZoneKind.SUPPLY,
            "swing_price": 2350.0,
            "major_low": 2344.5,
            "major_high": 2351.25,
            "refinement_low": 2347.0,
            "refinement_high": 2351.25,
        },
        {
            "kind": ZoneKind.DEMAND,
            "swing_price": 2325.0,
            "major_low": 2323.75,
            "major_high": 2331.0,
            "refinement_low": 2323.75,
            "refinement_high": 2328.0,
        },
    ]


def test_build_zones_rejects_unequal_input_lengths():
    h1_swings = [
        SwingPoint(timestamp=datetime(2026, 4, 4, 6, 0, tzinfo=timezone.utc), price=2350.0, kind=ZoneKind.SUPPLY),
    ]
    m15_bounds = [
        M15RefinementBounds(timestamp=datetime(2026, 4, 4, 6, 15, tzinfo=timezone.utc), low=2344.5, high=2351.25),
        M15RefinementBounds(timestamp=datetime(2026, 4, 4, 7, 15, tzinfo=timezone.utc), low=2323.75, high=2331.0),
    ]

    with pytest.raises(ValueError, match="same number of items"):
        build_zones(h1_swings, m15_bounds)
