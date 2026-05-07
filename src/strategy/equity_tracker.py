"""Equity curve tracking, drawdown computation, and Kelly-fraction position sizing.

Tracks the account equity over time to provide:
- Current drawdown ratio (DD_t / DD_max)
- Kelly-criterion position sizing adjusted by Ω_t
- Equity history persistence for analysis
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EquitySnapshot:
    """A single equity observation."""

    timestamp: datetime
    equity: float
    peak_equity: float
    drawdown: float
    drawdown_ratio: float


class EquityTracker:
    """Tracks equity, peak equity, and drawdown over time.

    Parameters
    ----------
    dd_max : float
        Maximum drawdown threshold as a fraction (e.g. 0.20 = 20%).
        Used to compute the drawdown ratio ``DD_t / DD_max``.
    initial_equity : float, optional
        Starting equity. If not provided, the first ``update()`` call
        sets the initial value.
    """

    def __init__(self, dd_max: float = 0.20, initial_equity: float | None = None) -> None:
        if dd_max <= 0:
            raise ValueError("dd_max must be positive")
        self._dd_max = dd_max
        self._peak_equity: float = initial_equity or 0.0
        self._current_equity: float = initial_equity or 0.0
        self._history: list[EquitySnapshot] = []

    @property
    def dd_max(self) -> float:
        return self._dd_max

    @property
    def current_equity(self) -> float:
        return self._current_equity

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    @property
    def current_drawdown(self) -> float:
        """Absolute drawdown from peak."""
        return max(self._peak_equity - self._current_equity, 0.0)

    @property
    def current_drawdown_pct(self) -> float:
        """Drawdown as a fraction of peak equity."""
        if self._peak_equity <= 0:
            return 0.0
        return self.current_drawdown / self._peak_equity

    @property
    def drawdown_ratio(self) -> float:
        """DD_t / DD_max, clamped to [0, 1]."""
        return min(self.current_drawdown_pct / self._dd_max, 1.0)

    @property
    def drawdown_dampener(self) -> float:
        """Multiplicative drawdown dampener: (1 - DD_t / DD_max)."""
        return max(1.0 - self.drawdown_ratio, 0.0)

    @property
    def history(self) -> list[EquitySnapshot]:
        return list(self._history)

    def update(self, equity: float, timestamp: datetime | None = None) -> EquitySnapshot:
        """Record a new equity observation.

        Updates peak equity and appends to history.
        """
        ts = timestamp or datetime.now(tz=timezone.utc)
        self._current_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

        snapshot = EquitySnapshot(
            timestamp=ts,
            equity=equity,
            peak_equity=self._peak_equity,
            drawdown=self.current_drawdown,
            drawdown_ratio=self.drawdown_ratio,
        )
        self._history.append(snapshot)
        return snapshot


def compute_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """Compute the Kelly fraction: (p * b - q) / b.

    Parameters
    ----------
    win_rate : float
        Probability of winning (p), in [0, 1].
    avg_win : float
        Average win amount (b), must be positive.
    avg_loss : float
        Average loss amount (used to derive q = 1 - p), must be positive.

    Returns
    -------
    float
        The Kelly fraction, clamped to [0, 1].
    """
    if avg_win <= 0 or avg_loss <= 0:
        return 0.0
    p = max(0.0, min(1.0, win_rate))
    q = 1.0 - p
    b = avg_win / avg_loss  # Odds ratio
    kelly = (p * b - q) / b
    return max(0.0, min(1.0, kelly))


def compute_position_size(
    *,
    equity: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    omega_t: float,
    r_max: float = 0.02,
    volume_min: float = 0.01,
    volume_step: float = 0.01,
    price_per_lot: float = 1.0,
) -> float:
    """Compute position size using the Kelly-fraction formula.

    w_t = Equity * min(r_max, Kelly) * Ω_t

    The result is rounded down to the nearest ``volume_step`` and
    clamped to ``[volume_min, ∞)``.

    Parameters
    ----------
    equity : float
        Current account equity.
    win_rate : float
        Historical win probability.
    avg_win : float
        Average winning trade amount.
    avg_loss : float
        Average losing trade amount.
    omega_t : float
        Trade quality multiplier from the quant engine.
    r_max : float
        Maximum risk fraction (default 2% of equity).
    volume_min : float
        Broker minimum lot size.
    volume_step : float
        Broker lot step size.
    price_per_lot : float
        Approximate value of 1 lot in account currency (for sizing).

    Returns
    -------
    float
        Position size in lots.
    """
    if equity <= 0 or price_per_lot <= 0:
        return volume_min

    kelly = compute_kelly_fraction(win_rate, avg_win, avg_loss)
    risk_fraction = min(r_max, kelly) * max(0.0, min(1.0, omega_t))
    risk_amount = equity * risk_fraction

    # Convert risk amount to lots
    raw_lots = risk_amount / price_per_lot
    if raw_lots < volume_min:
        return volume_min

    # Round down to nearest volume_step
    if volume_step > 0:
        steps = math.floor((raw_lots - volume_min) / volume_step)
        normalized = volume_min + (steps * volume_step)
    else:
        normalized = raw_lots

    return round(max(normalized, volume_min), 8)


def compute_price_risk_per_lot(*, entry_price: float, stop_loss: float, symbol_info) -> float:
    """Estimate account-currency risk for one lot from entry to stop."""
    stop_distance = abs(float(entry_price) - float(stop_loss))
    if stop_distance <= 0:
        return 0.0

    tick_value = _first_positive_attr(
        symbol_info,
        "trade_tick_value_loss",
        "trade_tick_value",
        "trade_tick_value_profit",
    )
    tick_size = _first_positive_attr(symbol_info, "trade_tick_size", "point")
    if tick_value is not None and tick_size is not None:
        return (stop_distance / tick_size) * tick_value

    contract_size = _first_positive_attr(symbol_info, "trade_contract_size")
    if contract_size is not None:
        return stop_distance * contract_size

    return 0.0


def _first_positive_attr(source, *names: str) -> float | None:
    for name in names:
        value = getattr(source, name, None)
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_equity_history(history: list[EquitySnapshot], path: str | Path) -> None:
    """Save equity history as JSONL."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as fh:
        for snapshot in history:
            fh.write(json.dumps(_snapshot_to_jsonable(snapshot)) + "\n")


def append_equity_snapshot_to_file(snapshot: EquitySnapshot, path: str | Path) -> None:
    """Append one equity snapshot as JSONL."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(_snapshot_to_jsonable(snapshot)) + "\n")


def _snapshot_to_jsonable(snapshot: EquitySnapshot) -> dict[str, float | str]:
    return {
        "timestamp": snapshot.timestamp.isoformat(),
        "equity": snapshot.equity,
        "peak_equity": snapshot.peak_equity,
        "drawdown": snapshot.drawdown,
        "drawdown_ratio": snapshot.drawdown_ratio,
    }


def load_equity_history(path: str | Path) -> list[EquitySnapshot]:
    """Load equity history from JSONL."""
    file_path = Path(path)
    if not file_path.exists():
        return []
    snapshots: list[EquitySnapshot] = []
    with open(file_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            snapshots.append(
                EquitySnapshot(
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    equity=float(data["equity"]),
                    peak_equity=float(data["peak_equity"]),
                    drawdown=float(data["drawdown"]),
                    drawdown_ratio=float(data["drawdown_ratio"]),
                )
            )
    return snapshots
