from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.market_data import Candle


@dataclass(frozen=True, slots=True)
class DailyContext:
    daily_high: float
    daily_low: float
    current_price: float
    range_position: float
    objective_high: float
    objective_low: float
    previous_day_high: float = 0.0
    previous_day_low: float = 0.0
    previous_day_close: float = 0.0
    current_day_open: float = 0.0
    adr: float = 0.0
    current_day_projection_high: float = 0.0
    current_day_projection_low: float = 0.0
    daily_expansion: float = 0.0
    daily_expansion_ratio: float = 0.0


@dataclass(frozen=True, slots=True)
class H4Context:
    previous_session_high: float
    previous_session_low: float
    demand_zones: tuple[tuple[float, float], ...]
    supply_zones: tuple[tuple[float, float], ...]
    volume_profile_levels: tuple[float, ...]
    session_range: float = 0.0
    normalized_demand_zones: tuple[tuple[float, float], ...] = ()
    normalized_supply_zones: tuple[tuple[float, float], ...] = ()


def build_daily_context(d1_candles: list[Candle], *, current_price: float | None = None) -> DailyContext:
    _require_timeframe(d1_candles, "D1", minimum=1)
    latest = d1_candles[-1]
    price = float(latest.close if current_price is None else current_price)
    day_range = latest.high - latest.low
    range_position = 0.5 if day_range == 0 else (price - latest.low) / day_range
    range_position = max(0.0, min(1.0, range_position))
    adr_sample = d1_candles[-min(5, len(d1_candles)) :]
    adr = sum(float(candle.high) - float(candle.low) for candle in adr_sample) / len(adr_sample)
    current_day_open = float(latest.close)
    daily_expansion = abs(price - current_day_open)
    daily_expansion_ratio = daily_expansion / max(adr, 1e-9)
    return DailyContext(
        daily_high=float(latest.high),
        daily_low=float(latest.low),
        current_price=price,
        range_position=range_position,
        objective_high=float(latest.high),
        objective_low=float(latest.low),
        previous_day_high=float(latest.high),
        previous_day_low=float(latest.low),
        previous_day_close=float(latest.close),
        current_day_open=current_day_open,
        adr=adr,
        current_day_projection_high=current_day_open + adr,
        current_day_projection_low=current_day_open - adr,
        daily_expansion=daily_expansion,
        daily_expansion_ratio=daily_expansion_ratio,
    )


def build_h4_context(h4_candles: list[Candle], *, session_candle_count: int = 6) -> H4Context:
    _require_timeframe(h4_candles, "H4", minimum=session_candle_count)
    previous_session = _select_previous_session(h4_candles, session_candle_count=session_candle_count)
    previous_session_high = max(candle.high for candle in previous_session)
    previous_session_low = min(candle.low for candle in previous_session)
    demand_zones = _swing_low_zones(previous_session)
    supply_zones = _swing_high_zones(previous_session)
    if not demand_zones:
        demand_zones = ((previous_session_low, previous_session[-1].close),)
    if not supply_zones:
        supply_zones = ((previous_session[-1].close, previous_session_high),)
    demand_zones = tuple(sorted(demand_zones, key=lambda zone: (zone[0], zone[1])))
    supply_zones = tuple(sorted(supply_zones, key=lambda zone: (zone[1], zone[0]), reverse=True))
    volume_profile_levels = _volume_profile_levels(previous_session)
    session_range = max(float(previous_session_high) - float(previous_session_low), 0.0)
    return H4Context(
        previous_session_high=float(previous_session_high),
        previous_session_low=float(previous_session_low),
        demand_zones=tuple(demand_zones),
        supply_zones=tuple(supply_zones),
        volume_profile_levels=volume_profile_levels,
        session_range=session_range,
        normalized_demand_zones=_normalize_zones(demand_zones, previous_session_low, previous_session_high),
        normalized_supply_zones=_normalize_zones(supply_zones, previous_session_low, previous_session_high),
    )


def _select_previous_session(h4_candles: list[Candle], *, session_candle_count: int) -> list[Candle]:
    if len(h4_candles) >= session_candle_count + 1:
        return h4_candles[-(session_candle_count + 1) : -1]
    return h4_candles[-session_candle_count:]


def _swing_low_zones(candles: list[Candle]) -> tuple[tuple[float, float], ...]:
    zones = []
    for previous_candle, candle, next_candle in _triples(candles):
        if candle.low < previous_candle.low and candle.low < next_candle.low:
            upper_bound = min(candle.open, candle.close)
            zones.append((float(candle.low), float(upper_bound)))
    return tuple(zones)


def _swing_high_zones(candles: list[Candle]) -> tuple[tuple[float, float], ...]:
    zones = []
    for previous_candle, candle, next_candle in _triples(candles):
        if candle.high > previous_candle.high and candle.high > next_candle.high:
            lower_bound = max(candle.open, candle.close)
            zones.append((float(lower_bound), float(candle.high)))
    return tuple(zones)


def _volume_profile_levels(candles: list[Candle]) -> tuple[float, ...]:
    ordered = sorted(candles, key=lambda candle: candle.volume, reverse=True)
    levels = []
    for candle in ordered[:3]:
        typical_price = (candle.high + candle.low + candle.close) / 3.0
        levels.append(float(typical_price))
    return tuple(levels)


def _normalize_zones(
    zones: tuple[tuple[float, float], ...] | list[tuple[float, float]],
    session_low: float,
    session_high: float,
) -> tuple[tuple[float, float], ...]:
    session_range = float(session_high) - float(session_low)
    if session_range <= 0:
        return tuple((0.5, 0.5) for _ in zones)
    normalized = []
    for lower, upper in zones:
        normalized.append(
            (
                max(0.0, min(1.0, (float(lower) - float(session_low)) / session_range)),
                max(0.0, min(1.0, (float(upper) - float(session_low)) / session_range)),
            )
        )
    return tuple(normalized)


def _triples(candles: Iterable[Candle]) -> Iterable[tuple[Candle, Candle, Candle]]:
    candle_list = list(candles)
    for index in range(1, len(candle_list) - 1):
        yield candle_list[index - 1], candle_list[index], candle_list[index + 1]


def _require_timeframe(candles: list[Candle], timeframe: str, *, minimum: int) -> None:
    if len(candles) < minimum:
        raise ValueError(f"{timeframe} context requires at least {minimum} candles")
    if any(candle.timeframe != timeframe for candle in candles):
        raise ValueError(f"{timeframe} context received a different timeframe")
