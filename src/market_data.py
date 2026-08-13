from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from typing import Any

from src.strategy.zones import M15RefinementBounds, SwingPoint, ZoneKind, build_zones


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str


@dataclass(frozen=True, slots=True)
class LiveStrategyInput:
    symbol: str
    session_timestamp: object
    d1_candles: list[Candle]
    h4_candles: list[Candle]
    h1_candles: list[Candle]
    m30_candles: list[Candle]
    m10_candles: list[Candle]
    breakout_candle: Candle
    retest_candle: Candle
    m5_candles: list[Candle]
    m15_candles: list[Candle]
    zone: dict
    spread: float = 0.0
    tick_data: dict = None


_TIMEFRAME_LABELS = {
    "TIMEFRAME_M1": "M1",
    "TIMEFRAME_M2": "M2",
    "TIMEFRAME_M3": "M3",
    "TIMEFRAME_M4": "M4",
    "TIMEFRAME_M5": "M5",
    "TIMEFRAME_M6": "M6",
    "TIMEFRAME_M10": "M10",
    "TIMEFRAME_M12": "M12",
    "TIMEFRAME_M15": "M15",
    "TIMEFRAME_M20": "M20",
    "TIMEFRAME_M30": "M30",
    "TIMEFRAME_H1": "H1",
    "TIMEFRAME_H2": "H2",
    "TIMEFRAME_H3": "H3",
    "TIMEFRAME_H4": "H4",
    "TIMEFRAME_H6": "H6",
    "TIMEFRAME_H8": "H8",
    "TIMEFRAME_H12": "H12",
    "TIMEFRAME_D1": "D1",
    "TIMEFRAME_W1": "W1",
    "TIMEFRAME_MN1": "MN1",
}


def timeframe_label(timeframe: Any) -> str:
    if timeframe in _TIMEFRAME_LABELS:
        return _TIMEFRAME_LABELS[timeframe]

    raise ValueError(f"Unsupported timeframe: {timeframe!r}")


def _get_rate_value(rate: Any, name: str) -> Any:
    if isinstance(rate, dict):
        return rate[name]
    try:
        return rate[name]
    except (KeyError, IndexError, TypeError, ValueError):
        return getattr(rate, name)


def _normalize_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("MT5 rate timestamp must be timezone-aware")
        return value.astimezone(timezone.utc)

    if isinstance(value, Real):
        return datetime.fromtimestamp(value, tz=timezone.utc)

    raise TypeError(f"Unsupported MT5 rate timestamp type: {type(value)!r}")


def candle_from_mt5_rate(rate: Any, timeframe: Any) -> Candle:
    timestamp = _normalize_timestamp(_get_rate_value(rate, "time"))
    open_price = float(_get_rate_value(rate, "open"))
    high_price = float(_get_rate_value(rate, "high"))
    low_price = float(_get_rate_value(rate, "low"))
    close_price = float(_get_rate_value(rate, "close"))

    volume_value = None
    for field_name in ("tick_volume", "volume", "real_volume"):
        try:
            volume_value = _get_rate_value(rate, field_name)
        except (KeyError, AttributeError):
            continue
        if volume_value is not None:
            break

    if volume_value is None:
        raise ValueError("MT5 rate is missing a volume field")

    return Candle(
        timestamp=timestamp,
        open=open_price,
        high=high_price,
        low=low_price,
        close=close_price,
        volume=int(volume_value),
        timeframe=timeframe_label(timeframe),
    )


def fetch_candles(mt5_module, symbol: str, timeframe_attr: str, count: int, minimum: int | None = None) -> list[Candle]:
    timeframe_value = getattr(mt5_module, timeframe_attr)
    rates = mt5_module.copy_rates_from_pos(symbol, timeframe_value, 1, count)
    minimum_required = count if minimum is None else minimum
    if rates is None or len(rates) < minimum_required:
        raise RuntimeError(f"Not enough {timeframe_attr} candle data for {symbol}")
    return [candle_from_mt5_rate(rate, timeframe_attr) for rate in rates]


def fetch_optional_candles(mt5_module, symbol: str, timeframe_attr: str, count: int) -> list[Candle]:
    try:
        return fetch_candles(mt5_module, symbol, timeframe_attr, count)
    except RuntimeError:
        return []


def build_live_strategy_input(mt5_module, symbol: str) -> LiveStrategyInput:
    d1_candles = fetch_candles(mt5_module, symbol, "TIMEFRAME_D1", 10, minimum=1)
    # Fetch 14 H4 candles: build_h4_context needs session_candle_count=6 *completed* bars
    # plus the live bar, so 7 minimum — use 14 for richer zone history
    h4_candles = fetch_candles(mt5_module, symbol, "TIMEFRAME_H4", 14)
    h1_candles = fetch_candles(mt5_module, symbol, "TIMEFRAME_H1", 24)
    m30_candles = fetch_candles(mt5_module, symbol, "TIMEFRAME_M30", 20)
    m15_candles = fetch_candles(mt5_module, symbol, "TIMEFRAME_M15", 100)
    m5_candles = fetch_candles(mt5_module, symbol, "TIMEFRAME_M5", 60)
    m10_candles = fetch_optional_candles(mt5_module, symbol, "TIMEFRAME_M10", 8)
    if len(m10_candles) < 8:
        m10_candles = _aggregate_m5_to_m10(m5_candles, count=8)
    if len(m10_candles) < 8:
        m10_candles = _derive_m10_from_m15(m15_candles, count=8)
    breakout_candle = m5_candles[-2]
    retest_candle = m5_candles[-1]

    # Fetch spread data for quant engine
    spread = 0.0
    tick_data = {}
    tick_getter = getattr(mt5_module, "symbol_info_tick", None)
    if tick_getter is not None:
        tick = tick_getter(symbol)
        if tick is not None:
            ask = float(getattr(tick, "ask", 0.0) or 0.0)
            bid = float(getattr(tick, "bid", 0.0) or 0.0)
            spread = ask - bid
            tick_data = {"ask": ask, "bid": bid, "spread": spread}

    return LiveStrategyInput(
        symbol=symbol,
        session_timestamp=retest_candle.timestamp.astimezone(timezone.utc),
        d1_candles=d1_candles,
        h4_candles=h4_candles,
        h1_candles=h1_candles,
        m30_candles=m30_candles,
        m10_candles=m10_candles,
        breakout_candle=breakout_candle,
        retest_candle=retest_candle,
        m5_candles=m5_candles,
        m15_candles=m15_candles,
        zone=_build_supply_zone(h1_candles[-1], m15_candles[-1]),
        spread=spread,
        tick_data=tick_data,
    )


def _build_supply_zone(h1_candle: Candle, m15_candle: Candle) -> dict:
    swing = SwingPoint(timestamp=h1_candle.timestamp, price=h1_candle.high, kind=ZoneKind.SUPPLY)
    bounds = M15RefinementBounds(timestamp=m15_candle.timestamp, low=m15_candle.low, high=m15_candle.high)
    return build_zones([swing], [bounds])[0]


def _aggregate_m5_to_m10(m5_candles: list[Candle], *, count: int) -> list[Candle]:
    if len(m5_candles) < 2:
        return []
    pairs = []
    usable = m5_candles[-(count * 2):]
    if len(usable) % 2:
        usable = usable[1:]
    for index in range(0, len(usable), 2):
        chunk = usable[index : index + 2]
        if len(chunk) < 2:
            continue
        pairs.append(_combine_candles(chunk, "TIMEFRAME_M10"))
    return pairs[-count:]


def _derive_m10_from_m15(m15_candles: list[Candle], *, count: int) -> list[Candle]:
    return [
        Candle(
            timestamp=candle.timestamp,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            timeframe="M10",
        )
        for candle in m15_candles[-count:]
    ]


def _combine_candles(candles: list[Candle], timeframe: str) -> Candle:
    return Candle(
        timestamp=candles[-1].timestamp,
        open=candles[0].open,
        high=max(candle.high for candle in candles),
        low=min(candle.low for candle in candles),
        close=candles[-1].close,
        volume=sum(candle.volume for candle in candles),
        timeframe=timeframe_label(timeframe),
    )
