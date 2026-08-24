from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class ForwardOutcome:
    label: int
    realized_return: float
    current_price: float
    future_price: float


def _utc_timestamp(value: Any) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("candle timestamp must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("candle timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def closed_candles_at(
    candles: Sequence[Any],
    decision_time: datetime,
    *,
    bar_duration: timedelta,
    limit: int | None = None,
) -> list[Any]:
    """Return only candles whose complete bar was observable by decision_time.

    Historical providers commonly stamp a candle at bar-open. Comparing only the
    raw timestamp leaks the still-forming bar into a past decision. This authority
    therefore admits a candle only when ``open_timestamp + bar_duration`` is no
    later than the decision timestamp.
    """
    decision = _utc_timestamp(decision_time)
    if bar_duration <= timedelta(0):
        raise ValueError("bar_duration must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")

    eligible = [
        candle
        for candle in candles
        if _utc_timestamp(getattr(candle, "timestamp")) + bar_duration <= decision
    ]
    return eligible[-limit:] if limit is not None else eligible


def aggregate_consecutive_bars(candles: Sequence[Any], *, group_size: int) -> list[Any]:
    """Aggregate sorted equal-duration candles without inventing intermediate data."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if not candles:
        return []

    ordered = sorted(candles, key=lambda candle: _utc_timestamp(getattr(candle, "timestamp")))
    aggregated: list[Any] = []
    for start in range(0, len(ordered), group_size):
        chunk = ordered[start : start + group_size]
        if len(chunk) != group_size:
            break
        aggregated.append(
            SimpleNamespace(
                timestamp=_utc_timestamp(getattr(chunk[0], "timestamp")),
                time=int(_utc_timestamp(getattr(chunk[0], "timestamp")).timestamp()),
                open=float(getattr(chunk[0], "open")),
                high=max(float(getattr(candle, "high")) for candle in chunk),
                low=min(float(getattr(candle, "low")) for candle in chunk),
                close=float(getattr(chunk[-1], "close")),
                volume=sum(float(getattr(candle, "volume", 0.0) or 0.0) for candle in chunk),
            )
        )
    return aggregated


def realized_forward_outcome(
    candles: Sequence[Any],
    index: int,
    *,
    horizon_bars: int,
    transaction_cost_ratio: float,
) -> ForwardOutcome | None:
    """Build a supervised label from an actually later close on the same series."""
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    if transaction_cost_ratio < 0:
        raise ValueError("transaction_cost_ratio cannot be negative")
    future_index = index + horizon_bars
    if index < 0 or future_index >= len(candles):
        return None

    current_price = float(getattr(candles[index], "close"))
    future_price = float(getattr(candles[future_index], "close"))
    if current_price <= 0 or future_price <= 0:
        return None

    realized_return = (future_price - current_price) / current_price
    return ForwardOutcome(
        label=1 if realized_return > transaction_cost_ratio else 0,
        realized_return=realized_return,
        current_price=current_price,
        future_price=future_price,
    )


def average_true_range_proxy(candles: Sequence[Any], *, lookback: int = 14) -> float:
    """Return a causal high-low range proxy for execution-cost normalization."""
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    recent = list(candles)[-lookback:]
    if not recent:
        return 0.0
    ranges = [max(float(getattr(candle, "high")) - float(getattr(candle, "low")), 0.0) for candle in recent]
    return sum(ranges) / len(ranges)
