"""Feature extraction and z-score normalization for the quant engine.

Computes rolling z-scores of 7 market features used to calculate the
trade-quality multiplier Ω_t:

    Z(M_t)  — MTF Momentum (Weighted D1→M15 directional bias)
    Z(T_t)  — Trend Alignment (Agreement across all timeframes)
    Z(V_t)  — Volume Conviction (Volume ratio × body efficiency)
    Z(O_t)  — Structural Quality (Order blocks + FVG + S/D confluence)
    Z(R_t)  — Volatility risk (ATR expansion ratio, inverted)
    Z(D_t)  — Entry distance from nearest S/D zone
    Z(C_t)  — Spread / correlation danger
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.strategy.breakout import BreakoutDirection


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    """A single observation of all raw and z-scored features."""

    timestamp: datetime

    # Raw values
    momentum_raw: float
    trend_raw: float
    volume_raw: float
    order_block_raw: float
    volatility_risk_raw: float
    entry_distance_raw: float
    spread_danger_raw: float

    # Z-scored values
    momentum_z: float
    trend_z: float
    volume_z: float
    order_block_z: float
    volatility_risk_z: float
    entry_distance_z: float
    spread_danger_z: float

    # Derived
    expected_return: float = 0.0
    return_std: float = 1.0
    orderflow_raw: float = 0.0
    orderflow_z: float = 0.0
    m5_score_raw: float = 0.0
    m5_score_z: float = 0.0
    m15_score_raw: float = 0.0
    m15_score_z: float = 0.0
    context_score_raw: float = 0.0
    context_score_z: float = 0.0
    structure_score_raw: float = 0.0
    structure_score_z: float = 0.0
    volatility_score_raw: float = 0.0
    volatility_score_z: float = 0.0
    momentum_indicator_raw: float = 0.0
    momentum_indicator_z: float = 0.0
    trend_indicator_raw: float = 0.0
    trend_indicator_z: float = 0.0
    orderflow_volume_raw: float = 0.0
    orderflow_volume_z: float = 0.0
    risk_math_raw: float = 0.0
    risk_math_z: float = 0.0
    statistical_score_raw: float = 0.0
    statistical_score_z: float = 0.0


class RollingBuffer:
    """Fixed-capacity ring buffer that tracks running mean and variance."""

    def __init__(self, capacity: int = 100) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._buffer: deque[float] = deque(maxlen=capacity)
        self._sum: float = 0.0
        self._sum_sq: float = 0.0

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._buffer)

    def push(self, value: float) -> None:
        if len(self._buffer) == self._capacity:
            evicted = self._buffer[0]
            self._sum -= evicted
            self._sum_sq -= evicted * evicted
        self._buffer.append(value)
        self._sum += value
        self._sum_sq += value * value

    def mean(self) -> float:
        count = len(self._buffer)
        if count == 0:
            return 0.0
        return self._sum / count

    def std(self) -> float:
        count = len(self._buffer)
        if count < 2:
            return 1.0  # Avoid division by zero; z-score = 0 when std=1
        mean = self._sum / count
        variance = (self._sum_sq / count) - (mean * mean)
        # Clamp to avoid negative variance from floating point drift
        return math.sqrt(max(variance, 0.0)) or 1.0

    def z_score(self, value: float) -> float:
        return (value - self.mean()) / self.std()


class FeatureExtractor:
    """Extracts and z-normalizes the 7 market features over a rolling window.

    Usage:
        extractor = FeatureExtractor(window=100)
        snapshot = extractor.update(
            momentum_raw=..., trend_raw=..., volume_raw=...,
            order_block_raw=..., volatility_risk_raw=...,
            entry_distance_raw=..., spread_danger_raw=...,
            expected_return=..., return_std=...,
            timestamp=datetime.now(tz=timezone.utc),
        )
    """

    FEATURE_NAMES = (
        "momentum",
        "trend",
        "volume",
        "order_block",
        "volatility_risk",
        "entry_distance",
        "spread_danger",
        "orderflow",
        "m5_score",
        "m15_score",
        "context_score",
        "structure_score",
        "volatility_score",
        "momentum_indicator",
        "trend_indicator",
        "orderflow_volume",
        "risk_math",
        "statistical_score",
    )

    def __init__(self, window: int = 100) -> None:
        self._window = window
        self._buffers: dict[str, RollingBuffer] = {
            name: RollingBuffer(capacity=window) for name in self.FEATURE_NAMES
        }
        self._snapshot_count = 0
        self._last_timestamp: datetime | None = None

    @property
    def window(self) -> int:
        return self._window

    @property
    def snapshot_count(self) -> int:
        return self._snapshot_count

    def update(
        self,
        *,
        momentum_raw: float,
        trend_raw: float,
        volume_raw: float,
        order_block_raw: float,
        volatility_risk_raw: float,
        entry_distance_raw: float,
        spread_danger_raw: float,
        orderflow_raw: Any = 0.0,
        m5_score_raw: float = 0.0,
        m15_score_raw: float = 0.0,
        context_score_raw: float = 0.0,
        structure_score_raw: float = 0.0,
        volatility_score_raw: float = 0.0,
        momentum_indicator_raw: float = 0.0,
        trend_indicator_raw: float = 0.0,
        orderflow_volume_raw: float = 0.0,
        risk_math_raw: float = 0.0,
        statistical_score_raw: float = 0.0,
        expected_return: float = 0.0,
        return_std: float = 1.0,
        timestamp: datetime | None = None,
    ) -> FeatureSnapshot:
        ts = timestamp or datetime.now(tz=timezone.utc)

        raw_values = {
            "momentum": momentum_raw,
            "trend": trend_raw,
            "volume": volume_raw,
            "order_block": order_block_raw,
            "volatility_risk": volatility_risk_raw,
            "entry_distance": entry_distance_raw,
            "spread_danger": spread_danger_raw,
            "orderflow": _coerce_orderflow_raw(orderflow_raw),
            "m5_score": float(m5_score_raw),
            "m15_score": float(m15_score_raw),
            "context_score": float(context_score_raw),
            "structure_score": float(structure_score_raw),
            "volatility_score": float(volatility_score_raw),
            "momentum_indicator": float(momentum_indicator_raw),
            "trend_indicator": float(trend_indicator_raw),
            "orderflow_volume": float(orderflow_volume_raw),
            "risk_math": float(risk_math_raw),
            "statistical_score": float(statistical_score_raw),
        }

        should_push = self._last_timestamp is None or ts != self._last_timestamp

        # Push one complete feature row per bar timestamp, then compute z-scores.
        z_scores: dict[str, float] = {}
        for name, raw in raw_values.items():
            if should_push:
                self._buffers[name].push(raw)

            z_scores[name] = self._buffers[name].z_score(raw)

        if should_push:
            self._snapshot_count += 1
            self._last_timestamp = ts

        return FeatureSnapshot(
            timestamp=ts,
            momentum_raw=momentum_raw,
            trend_raw=trend_raw,
            volume_raw=volume_raw,
            order_block_raw=order_block_raw,
            volatility_risk_raw=volatility_risk_raw,
            entry_distance_raw=entry_distance_raw,
            spread_danger_raw=spread_danger_raw,
            orderflow_raw=raw_values["orderflow"],
            momentum_z=z_scores["momentum"],
            trend_z=z_scores["trend"],
            volume_z=z_scores["volume"],
            order_block_z=z_scores["order_block"],
            volatility_risk_z=z_scores["volatility_risk"],
            entry_distance_z=z_scores["entry_distance"],
            spread_danger_z=z_scores["spread_danger"],
            orderflow_z=z_scores["orderflow"],
            m5_score_raw=raw_values["m5_score"],
            m5_score_z=z_scores["m5_score"],
            m15_score_raw=raw_values["m15_score"],
            m15_score_z=z_scores["m15_score"],
            context_score_raw=raw_values["context_score"],
            context_score_z=z_scores["context_score"],
            structure_score_raw=raw_values["structure_score"],
            structure_score_z=z_scores["structure_score"],
            volatility_score_raw=raw_values["volatility_score"],
            volatility_score_z=z_scores["volatility_score"],
            momentum_indicator_raw=raw_values["momentum_indicator"],
            momentum_indicator_z=z_scores["momentum_indicator"],
            trend_indicator_raw=raw_values["trend_indicator"],
            trend_indicator_z=z_scores["trend_indicator"],
            orderflow_volume_raw=raw_values["orderflow_volume"],
            orderflow_volume_z=z_scores["orderflow_volume"],
            risk_math_raw=raw_values["risk_math"],
            risk_math_z=z_scores["risk_math"],
            statistical_score_raw=raw_values["statistical_score"],
            statistical_score_z=z_scores["statistical_score"],
            expected_return=expected_return,
            return_std=max(return_std, 1e-9),
        )


def _coerce_orderflow_raw(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return max(-1.0, min(1.0, float(value)))
    try:
        from src.strategy.orderflow import score_orderflow_for_direction

        bullish = score_orderflow_for_direction(value, BreakoutDirection.BULLISH).alignment_score
        bearish = score_orderflow_for_direction(value, BreakoutDirection.BEARISH).alignment_score
        return max(-1.0, min(1.0, bullish if abs(bullish) >= abs(bearish) else -bearish))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Helpers to extract raw feature values from existing strategy outputs
# ---------------------------------------------------------------------------

def compute_mtf_directional_bias(
    d1_candles: list[Any],
    h4_candles: list[Any],
    h1_candles: list[Any],
    m30_candles: list[Any],
    m15_candles: list[Any],
) -> float:
    """Weighted multi-timeframe directional score [-1, 1]."""
    def _score_tf(candles: list[Any], lookback: int) -> float:
        if len(candles) < lookback: return 0.0
        atr = sum(max(c.high - c.low, 1e-9) for c in candles[-10:]) / 10.0
        delta = float(candles[-1].close) - float(candles[-lookback].close)
        return max(-1.0, min(1.0, delta / (atr * 2) if atr > 0 else 0.0))

    scores = [
        _score_tf(d1_candles, 2) * 0.30,
        _score_tf(h4_candles, 2) * 0.25,
        _score_tf(h1_candles, 4) * 0.20,
        _score_tf(m30_candles, 3) * 0.15,
        _score_tf(m15_candles, 4) * 0.10,
    ]
    base_score = sum(scores)
    
    # Alignment bonus
    directions = [s > 0 for s in scores if abs(s) > 0.01]
    if len(directions) >= 3 and (all(directions) or not any(directions)):
        base_score *= (1.0 + 0.2 * len(directions))
        
    return max(-1.0, min(1.0, base_score))


def compute_trend_alignment(
    d1_candles: list[Any],
    h4_candles: list[Any],
    h1_candles: list[Any],
    m30_candles: list[Any],
    m15_candles: list[Any],
) -> float:
    """Measures how well all timeframes align in direction."""
    def _dir(candles: list[Any]) -> int:
        if len(candles) < 2: return 0
        return 1 if candles[-1].close > candles[-2].close else -1

    dirs = [_dir(d1_candles), _dir(h4_candles), _dir(h1_candles), _dir(m30_candles), _dir(m15_candles)]
    return sum(dirs) / 5.0


def compute_enhanced_volume(m15_candles: list[Any], lookback: int = 20) -> float:
    """Volume ratio weighted by body efficiency."""
    if not m15_candles: return 1.0
    recent = m15_candles[-min(lookback, len(m15_candles)):]
    avg_vol = sum(float(c.volume) for c in recent) / max(len(recent), 1)
    current = m15_candles[-1]
    vol_ratio = float(current.volume) / avg_vol if avg_vol > 0 else 1.0
    
    body = abs(float(current.close) - float(current.open))
    candle_range = max(float(current.high) - float(current.low), 1e-9)
    efficiency = body / candle_range
    
    return vol_ratio * efficiency


def extract_momentum(
    d1_candles: list[Any],
    h4_candles: list[Any],
    h1_candles: list[Any],
    m30_candles: list[Any],
    m15_candles: list[Any],
) -> float:
    """Redirects to MTF directional bias."""
    return compute_mtf_directional_bias(d1_candles, h4_candles, h1_candles, m30_candles, m15_candles)


def extract_trend(
    d1_candles: list[Any],
    h4_candles: list[Any],
    h1_candles: list[Any],
    m30_candles: list[Any],
    m15_candles: list[Any],
) -> float:
    """Redirects to trend alignment score."""
    return compute_trend_alignment(d1_candles, h4_candles, h1_candles, m30_candles, m15_candles)


def extract_volume(m15_candles: list[Any], lookback: int = 20) -> float:
    """Redirects to enhanced volume."""
    return compute_enhanced_volume(m15_candles, lookback)


def extract_order_block_quality(
    price: float,
    candles: list[Any],
    demand_zones: tuple[tuple[float, float], ...],
    supply_zones: tuple[tuple[float, float], ...],
    atr: float
) -> float:
    """Integrated structural score."""
    from src.strategy.structural import score_structural_context
    return score_structural_context(price, candles, demand_zones, supply_zones, atr)


def extract_volatility_risk(volatility_state: Any) -> float:
    """ATR expansion ratio (higher = more volatile = more risk)."""
    return float(getattr(volatility_state, "range_expansion_ratio", 1.0))


def extract_entry_distance(
    current_price: float,
    demand_zones: tuple[tuple[float, float], ...],
    supply_zones: tuple[tuple[float, float], ...],
) -> float:
    """Distance to nearest supply/demand zone (normalized by session range)."""
    distances: list[float] = []
    for lower, upper in demand_zones:
        if lower <= current_price <= upper:
            return 0.0
        distances.append(abs(current_price - upper))
    for lower, upper in supply_zones:
        if lower <= current_price <= upper:
            return 0.0
        distances.append(abs(current_price - lower))
    return min(distances) if distances else 0.0


def extract_spread_danger(spread: float, atr: float) -> float:
    """Spread as fraction of ATR — high spread relative to ATR = danger."""
    if atr <= 0:
        return 0.0
    return spread / atr


def extract_expected_return(m15_candles: list[Any], lookback: int = 20) -> tuple[float, float]:
    """Compute EWMA mean and std of recent M15 returns.

    Returns (expected_return, return_std).
    """
    candles = m15_candles[-min(lookback + 1, len(m15_candles)):]
    if len(candles) < 2:
        return 0.0, 1.0

    returns: list[float] = []
    for i in range(1, len(candles)):
        prev_close = float(candles[i - 1].close)
        cur_close = float(candles[i].close)
        if prev_close > 0:
            returns.append((cur_close - prev_close) / prev_close)

    if not returns:
        return 0.0, 1.0

    alpha = 2.0 / (len(returns) + 1.0)
    weights = [(1.0 - alpha) ** (len(returns) - 1 - index) for index in range(len(returns))]
    weight_sum = sum(weights)
    mean_ret = sum(weight * ret for weight, ret in zip(weights, returns)) / weight_sum
    if len(returns) < 2:
        return mean_ret, 1.0

    variance = sum(weight * ((ret - mean_ret) ** 2) for weight, ret in zip(weights, returns)) / weight_sum
    std_ret = math.sqrt(max(variance, 0.0)) or 1.0
    return mean_ret, std_ret


def _coerce_orderflow_raw(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return max(-1.0, min(1.0, float(value)))
    try:
        from src.strategy.orderflow import score_orderflow_for_direction

        bullish = score_orderflow_for_direction(value, BreakoutDirection.BULLISH).alignment_score
        bearish = score_orderflow_for_direction(value, BreakoutDirection.BEARISH).alignment_score
        return max(-1.0, min(1.0, bullish if abs(bullish) >= abs(bearish) else -bearish))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Helpers to extract raw feature values from existing strategy outputs
# ---------------------------------------------------------------------------

def compute_mtf_directional_bias(
    d1_candles: list[Any],
    h4_candles: list[Any],
    h1_candles: list[Any],
    m30_candles: list[Any],
    m15_candles: list[Any],
) -> float:
    """Weighted multi-timeframe directional score [-1, 1]."""
    def _score_tf(candles: list[Any], lookback: int) -> float:
        if len(candles) < lookback: return 0.0
        atr = sum(max(c.high - c.low, 1e-9) for c in candles[-10:]) / 10.0
        delta = float(candles[-1].close) - float(candles[-lookback].close)
        return max(-1.0, min(1.0, delta / (atr * 2) if atr > 0 else 0.0))

    scores = [
        _score_tf(d1_candles, 2) * 0.30,
        _score_tf(h4_candles, 2) * 0.25,
        _score_tf(h1_candles, 4) * 0.20,
        _score_tf(m30_candles, 3) * 0.15,
        _score_tf(m15_candles, 4) * 0.10,
    ]
    base_score = sum(scores)
    
    # Alignment bonus
    directions = [s > 0 for s in scores if abs(s) > 0.01]
    if len(directions) >= 3 and (all(directions) or not any(directions)):
        base_score *= (1.0 + 0.2 * len(directions))
        
    return max(-1.0, min(1.0, base_score))


def compute_trend_alignment(
    d1_candles: list[Any],
    h4_candles: list[Any],
    h1_candles: list[Any],
    m30_candles: list[Any],
    m15_candles: list[Any],
) -> float:
    """Measures how well all timeframes align in direction."""
    def _dir(candles: list[Any]) -> int:
        if len(candles) < 2: return 0
        return 1 if candles[-1].close > candles[-2].close else -1

    dirs = [_dir(d1_candles), _dir(h4_candles), _dir(h1_candles), _dir(m30_candles), _dir(m15_candles)]
    return sum(dirs) / 5.0


def compute_enhanced_volume(m15_candles: list[Any], lookback: int = 20) -> float:
    """Volume ratio weighted by body efficiency."""
    if not m15_candles: return 1.0
    recent = m15_candles[-min(lookback, len(m15_candles)):]
    avg_vol = sum(float(c.volume) for c in recent) / max(len(recent), 1)
    current = m15_candles[-1]
    vol_ratio = float(current.volume) / avg_vol if avg_vol > 0 else 1.0
    
    body = abs(float(current.close) - float(current.open))
    candle_range = max(float(current.high) - float(current.low), 1e-9)
    efficiency = body / candle_range
    
    return vol_ratio * efficiency


def extract_momentum(
    d1_candles: list[Any],
    h4_candles: list[Any],
    h1_candles: list[Any],
    m30_candles: list[Any],
    m15_candles: list[Any],
) -> float:
    """Redirects to MTF directional bias."""
    return compute_mtf_directional_bias(d1_candles, h4_candles, h1_candles, m30_candles, m15_candles)


def extract_trend(
    d1_candles: list[Any],
    h4_candles: list[Any],
    h1_candles: list[Any],
    m30_candles: list[Any],
    m15_candles: list[Any],
) -> float:
    """Redirects to trend alignment score."""
    return compute_trend_alignment(d1_candles, h4_candles, h1_candles, m30_candles, m15_candles)


def extract_volume(m15_candles: list[Any], lookback: int = 20) -> float:
    """Redirects to enhanced volume."""
    return compute_enhanced_volume(m15_candles, lookback)


def extract_order_block_quality(
    price: float,
    candles: list[Any],
    demand_zones: tuple[tuple[float, float], ...],
    supply_zones: tuple[tuple[float, float], ...],
    atr: float
) -> float:
    """Integrated structural score."""
    from src.strategy.structural import score_structural_context
    return score_structural_context(price, candles, demand_zones, supply_zones, atr)


def extract_volatility_risk(volatility_state: Any) -> float:
    """ATR expansion ratio (higher = more volatile = more risk)."""
    return float(getattr(volatility_state, "range_expansion_ratio", 1.0))


def extract_entry_distance(
    current_price: float,
    demand_zones: tuple[tuple[float, float], ...],
    supply_zones: tuple[tuple[float, float], ...],
) -> float:
    """Distance to nearest supply/demand zone (normalized by session range)."""
    distances: list[float] = []
    for lower, upper in demand_zones:
        if lower <= current_price <= upper:
            return 0.0
        distances.append(abs(current_price - upper))
    for lower, upper in supply_zones:
        if lower <= current_price <= upper:
            return 0.0
        distances.append(abs(current_price - lower))
    return min(distances) if distances else 0.0


def extract_spread_danger(spread: float, atr: float) -> float:
    """Spread as fraction of ATR — high spread relative to ATR = danger."""
    if atr <= 0:
        return 0.0
    return spread / atr


def extract_expected_return(m15_candles: list[Any], lookback: int = 20) -> tuple[float, float]:
    """Compute EWMA mean and std of recent M15 returns.

    Returns (expected_return, return_std).
    """
    candles = m15_candles[-min(lookback + 1, len(m15_candles)):]
    if len(candles) < 2:
        return 0.0, 1.0

    returns: list[float] = []
    for i in range(1, len(candles)):
        prev_close = float(candles[i - 1].close)
        cur_close = float(candles[i].close)
        if prev_close > 0:
            returns.append((cur_close - prev_close) / prev_close)

    if not returns:
        return 0.0, 1.0

    alpha = 2.0 / (len(returns) + 1.0)
    weights = [(1.0 - alpha) ** (len(returns) - 1 - index) for index in range(len(returns))]
    weight_sum = sum(weights)
    mean_ret = sum(weight * ret for weight, ret in zip(weights, returns)) / weight_sum
    if len(returns) < 2:
        return mean_ret, 1.0

    variance = sum(weight * ((ret - mean_ret) ** 2) for weight, ret in zip(weights, returns)) / weight_sum
    std_ret = math.sqrt(max(variance, 0.0)) or 1.0
    return mean_ret, std_ret


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def append_snapshot_to_file(snapshot: FeatureSnapshot, path: str | Path) -> None:
    """Append a feature snapshot as a JSON line to the given file."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(snapshot)
    data["timestamp"] = snapshot.timestamp.isoformat()
    with open(file_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(data) + "\n")


def append_enriched_snapshot_to_file(
    snapshot: FeatureSnapshot,
    path: str | Path,
    *,
    # Quant decision outputs
    quant_is_trade: bool = False,
    quant_action: int = 0,
    omega_t: float = 0.0,
    kelly_fraction: float = 0.0,
    ce_score_trade: float = 0.0,
    ce_score_flat: float = 0.0,
    sharpe_signal: float = 0.0,
    drawdown_dampener: float = 1.0,
    lot_multiplier: float = 1.0,
    # Trade economics
    transaction_cost: float = 0.0,
    win_rate: float = 0.5,
    avg_win: float = 1.0,
    avg_loss: float = 1.0,
    # Session & macro context
    session_score: float = 1.0,
    dxy_trend: float = 0.0,
    drawdown_ratio: float = 0.0,
    # Execution context
    spread: float = 0.0,
    atr: float = 1.0,
    lot_requested: float = 0.01,
    commission_per_lot: float = 6.0,
    current_equity: float = 0.0,
    # Outcome label — filled in at trade-close time via patch_outcome_label()
    # 1=win, 0=loss, -1=no-trade (not yet filled)
    outcome_label: int = -1,
    outcome_pnl: float = 0.0,
    outcome_r_multiple: float = 0.0,
) -> None:
    """Append a fully-enriched ML training row to the feature log.

    Every column needed by the local-edge model is captured here so that the
    offline trainer can do proper supervised learning without needing to join
    separate log files.  The ``outcome_label`` / ``outcome_pnl`` /
    ``outcome_r_multiple`` fields start at sentinel values and are updated in
    place when the trade closes (see ``patch_outcome_label``).
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(snapshot)
    data["timestamp"] = snapshot.timestamp.isoformat()

    # ── Quant decision columns ────────────────────────────────────────────────
    data["quant_is_trade"] = int(quant_is_trade)
    data["quant_action"] = quant_action
    data["omega_t"] = round(omega_t, 6)
    data["kelly_fraction"] = round(kelly_fraction, 6)
    data["ce_score_trade"] = round(ce_score_trade, 6)
    data["ce_score_flat"] = round(ce_score_flat, 6)
    data["sharpe_signal"] = round(sharpe_signal, 6)
    data["drawdown_dampener"] = round(drawdown_dampener, 6)
    data["lot_multiplier"] = round(lot_multiplier, 6)

    # ── Trade economics ───────────────────────────────────────────────────────
    data["transaction_cost"] = round(transaction_cost, 8)
    data["win_rate"] = round(win_rate, 6)
    data["avg_win"] = round(avg_win, 6)
    data["avg_loss"] = round(avg_loss, 6)

    # ── Session & macro context ───────────────────────────────────────────────
    data["session_score"] = round(session_score, 6)
    data["dxy_trend"] = round(dxy_trend, 6)
    data["drawdown_ratio"] = round(drawdown_ratio, 6)

    # ── Execution context ─────────────────────────────────────────────────────
    data["spread"] = round(spread, 6)
    data["atr"] = round(atr, 6)
    data["lot_requested"] = round(lot_requested, 6)
    data["commission_per_lot"] = round(commission_per_lot, 4)
    data["current_equity"] = round(current_equity, 2)

    # ── Deferred outcome labels ───────────────────────────────────────────────
    data["outcome_label"] = outcome_label       # sentinel -1 until trade closes
    data["outcome_pnl"] = round(outcome_pnl, 6)
    data["outcome_r_multiple"] = round(outcome_r_multiple, 6)

    with open(file_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(data) + "\n")


def patch_outcome_label(
    path: str | Path,
    ts_iso: str,
    outcome_label: int,
    outcome_pnl: float,
    outcome_r_multiple: float,
) -> bool:
    """Back-fill the outcome columns for a specific timestamp row in the log.

    Reads the whole file, updates the matching row in memory, and rewrites it.
    Intended to be called once per trade close with the realised P&L.

    Returns True if a matching row was found and patched.
    """
    file_path = Path(path)
    if not file_path.exists():
        return False

    lines = file_path.read_text(encoding="utf-8").splitlines()
    patched = False
    new_lines: list[str] = []
    for line in lines:
        if not line.strip():
            new_lines.append(line)
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            new_lines.append(line)
            continue
        if row.get("timestamp") == ts_iso and row.get("outcome_label", -1) == -1:
            row["outcome_label"] = outcome_label
            row["outcome_pnl"] = round(outcome_pnl, 6)
            row["outcome_r_multiple"] = round(outcome_r_multiple, 6)
            patched = True
            new_lines.append(json.dumps(row))
        else:
            new_lines.append(line)

    if patched:
        file_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return patched
