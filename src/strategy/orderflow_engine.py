"""
orderflow_engine.py
-------------------
Computes real-time orderflow + volume profile metrics directly from MT5 tick data.

No external service (GoCharting, webhook, etc.) required.
All metrics are derived from bid/ask/last/volume fields from MT5.

Metrics produced
----------------
Core orderflow
  delta_bias          : (buy_vol - sell_vol) / total_vol          [-1, 1]
  cvd_slope           : linear slope of cumulative delta           [-1, 1]
  imbalance_score     : bid/ask size imbalance per tick, averaged  [-1, 1]
  absorption_score    : high-vol ticks where price didn't move     [0,  1]
  vwap_alignment      : (price - VWAP) / VWAP, normalised         [-1, 1]
  liquidity_obstacle  : spread-width proxy                         [0,  1]

Volume Profile (tick-based)
  poc_bias            : (price - POC) / value_area_width           [-1, 1]
  value_area_position : +1 above VA, 0 inside VA, -1 below VA
  profile_oscillator  : composite [-1,1], positive = bullish profile bias
  hvn_proximity       : distance to nearest High Volume Node       [0,  1]
  lvn_proximity       : distance to nearest Low Volume Node        [0,  1]

Volume oscillators
  vwap_oscillator     : VWAP vs rolling mid price, normalised      [-1, 1]
  volume_momentum     : recent vol vs baseline vol (buy-side bias) [-1, 1]
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VolumeProfile:
    """Tick-based volume profile for the current window."""
    poc: float              # Point of Control — price with max volume
    vah: float              # Value Area High  (70 % of volume)
    val: float              # Value Area Low   (70 % of volume)
    value_area_width: float # VAH - VAL
    bins: list[tuple[float, float]]  # (price_mid, volume) sorted by price


@dataclass(frozen=True)
class LiveOrderflowSnapshot:
    symbol: str
    timestamp: datetime

    # ── Core orderflow ──────────────────────────────────────────────
    delta_bias: float             # [-1, 1]
    cvd_slope: float              # [-1, 1]
    imbalance_score: float        # [-1, 1]
    absorption_score: float       # [0,  1]
    vwap_alignment: float         # [-1, 1]
    liquidity_obstacle_score: float  # [0, 1]

    # ── Volume profile ──────────────────────────────────────────────
    poc_bias: float               # [-1, 1]  +ve = price above POC (bullish)
    value_area_position: float    # +1 above VA, 0 inside VA, -1 below VA
    profile_oscillator: float     # [-1, 1]  composite bullish/bearish bias
    hvn_proximity: float          # [0, 1]   0 = at HVN, 1 = far from HVN
    lvn_proximity: float          # [0, 1]   0 = at LVN, 1 = far from LVN

    # ── Volume oscillators ──────────────────────────────────────────
    vwap_oscillator: float        # [-1, 1]
    volume_momentum: float        # [-1, 1]

    # ── Aggregates ──────────────────────────────────────────────────
    tick_count: int
    buy_volume: float
    sell_volume: float
    net_delta: float
    vwap: float
    poc: float
    vah: float
    val: float


# ---------------------------------------------------------------------------
# Volume Profile builder
# ---------------------------------------------------------------------------

class VolumeProfileBuilder:
    """
    Builds a price-volume histogram from ticks.
    Uses a fixed number of price bins across the session's price range.
    """

    def __init__(self, n_bins: int = 100, value_area_pct: float = 0.70) -> None:
        self.n_bins = n_bins
        self.value_area_pct = value_area_pct

    def build(self, ticks: list[dict]) -> VolumeProfile | None:
        prices = [t["last"] or (t["ask"] + t["bid"]) / 2.0 for t in ticks
                  if (t["last"] > 0 or (t["ask"] > 0 and t["bid"] > 0))]
        volumes = [t["volume"] for t in ticks
                   if (t["last"] > 0 or (t["ask"] > 0 and t["bid"] > 0))]

        if len(prices) < 5:
            return None

        lo, hi = min(prices), max(prices)
        if hi <= lo:
            return None

        bin_width = (hi - lo) / self.n_bins
        buckets: list[float] = [0.0] * self.n_bins

        for price, vol in zip(prices, volumes):
            idx = min(int((price - lo) / bin_width), self.n_bins - 1)
            buckets[idx] += vol

        # POC = highest volume bin
        poc_idx = buckets.index(max(buckets))
        poc_price = lo + (poc_idx + 0.5) * bin_width

        # Value Area: expand outward from POC until 70% of total volume captured
        total_vol = sum(buckets)
        target = total_vol * self.value_area_pct
        accumulated = buckets[poc_idx]
        low_idx = poc_idx
        high_idx = poc_idx

        while accumulated < target:
            can_go_low = low_idx > 0
            can_go_high = high_idx < self.n_bins - 1
            if not can_go_low and not can_go_high:
                break
            vol_below = buckets[low_idx - 1] if can_go_low else -1.0
            vol_above = buckets[high_idx + 1] if can_go_high else -1.0
            if vol_above >= vol_below:
                high_idx += 1
                accumulated += buckets[high_idx]
            else:
                low_idx -= 1
                accumulated += buckets[low_idx]

        vah = lo + (high_idx + 1) * bin_width
        val = lo + low_idx * bin_width
        bins = [(lo + (i + 0.5) * bin_width, buckets[i]) for i in range(self.n_bins)]

        return VolumeProfile(
            poc=poc_price,
            vah=vah,
            val=val,
            value_area_width=max(vah - val, bin_width),
            bins=bins,
        )

    def find_hvn(self, profile: VolumeProfile, current_price: float, top_n: int = 3) -> list[float]:
        """Return prices of top-N highest volume nodes."""
        sorted_bins = sorted(profile.bins, key=lambda b: b[1], reverse=True)
        return [price for price, _ in sorted_bins[:top_n]]

    def find_lvn(self, profile: VolumeProfile, current_price: float, top_n: int = 3) -> list[float]:
        """Return prices of top-N lowest volume nodes (excluding zero-volume gaps)."""
        nonzero = [(price, vol) for price, vol in profile.bins if vol > 0]
        sorted_bins = sorted(nonzero, key=lambda b: b[1])
        return [price for price, _ in sorted_bins[:top_n]]


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class LiveOrderflowEngine:
    """
    Accumulates MT5 ticks and computes orderflow + volume profile metrics.

    Usage:
        engine = LiveOrderflowEngine("XAUUSD", window=500)
        engine.ingest(ticks)          # call each poll cycle
        snap = engine.snapshot()      # get latest metrics
    """

    def __init__(
        self,
        symbol: str,
        window: int = 500,
        cvd_slope_lookback: int = 50,
        max_spread_reference: float = 2.0,
        profile_bins: int = 100,
    ) -> None:
        self.symbol = symbol.upper()
        self.window = window
        self.cvd_slope_lookback = cvd_slope_lookback
        self.max_spread_reference = float(max_spread_reference)
        self._profile_builder = VolumeProfileBuilder(n_bins=profile_bins)

        self._ticks: deque[dict[str, float]] = deque(maxlen=window)
        self._cvd_series: deque[float] = deque(maxlen=window)
        self._cumulative_delta: float = 0.0
        self._last_seen_time: float = 0.0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, ticks: Sequence[Any]) -> int:
        """Feed raw MT5 ticks. Returns number of new ticks added."""
        added = 0
        for tick in ticks:
            t = _extract_tick(tick)
            if t is None:
                continue
            if t["time"] <= self._last_seen_time:
                continue
            self._last_seen_time = t["time"]
            self._accumulate(t)
            added += 1
        return added

    def _accumulate(self, t: dict[str, float]) -> None:
        bid = t["bid"]
        ask = t["ask"]
        last = t.get("last") or 0.0
        vol = t.get("volume") or t.get("volume_real") or 1.0
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0

        if last > 0:
            # Trade-price classification (when last price is available)
            if ask > 0 and last >= ask:
                buy_vol, sell_vol = vol, 0.0
            elif bid > 0 and last <= bid:
                buy_vol, sell_vol = 0.0, vol
            else:
                buy_vol = sell_vol = vol / 2.0
        elif mid > 0 and self._ticks:
            # Uptick/downtick rule for bid/ask-only ticks (XAUUSD, FX pairs)
            prev_mid = (self._ticks[-1]["bid"] + self._ticks[-1]["ask"]) / 2.0
            if mid > prev_mid:
                buy_vol, sell_vol = vol, 0.0
            elif mid < prev_mid:
                buy_vol, sell_vol = 0.0, vol
            else:
                buy_vol = sell_vol = vol / 2.0
        else:
            buy_vol = sell_vol = vol / 2.0

        # Use mid as the effective price when no last trade price exists
        effective_last = last if last > 0 else mid

        delta = buy_vol - sell_vol
        self._cumulative_delta += delta

        self._ticks.append({
            "bid": bid, "ask": ask, "last": effective_last,
            "volume": vol, "buy_vol": buy_vol, "sell_vol": sell_vol,
            "delta": delta, "time": t["time"],
        })
        self._cvd_series.append(self._cumulative_delta)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> LiveOrderflowSnapshot | None:
        if len(self._ticks) < 10:
            return None

        ticks = list(self._ticks)
        buy_volume = sum(t["buy_vol"] for t in ticks)
        sell_volume = sum(t["sell_vol"] for t in ticks)
        total_volume = max(buy_volume + sell_volume, 1.0)
        net_delta = buy_volume - sell_volume
        delta_bias = _clamp(net_delta / total_volume, -1.0, 1.0)

        cvd_slope        = self._compute_cvd_slope()
        imbalance_score  = self._compute_imbalance(ticks)
        absorption_score = self._compute_absorption(ticks)
        vwap, vwap_align = self._compute_vwap(ticks)
        liq_obstacle     = self._compute_liquidity_obstacle(ticks)

        # ── Volume profile ──────────────────────────────────────────
        profile = self._profile_builder.build(ticks)
        current_price = ticks[-1]["last"] or ticks[-1]["ask"] or ticks[-1]["bid"]

        poc_bias, va_position, profile_osc, hvn_prox, lvn_prox, poc, vah, val = (
            self._compute_profile_metrics(profile, current_price, ticks)
        )

        # ── Volume oscillators ──────────────────────────────────────
        vwap_osc   = self._compute_vwap_oscillator(ticks, vwap)
        vol_momentum = self._compute_volume_momentum(ticks)

        return LiveOrderflowSnapshot(
            symbol=self.symbol,
            timestamp=datetime.now(timezone.utc),
            delta_bias=delta_bias,
            cvd_slope=cvd_slope,
            imbalance_score=imbalance_score,
            absorption_score=absorption_score,
            vwap_alignment=vwap_align,
            liquidity_obstacle_score=liq_obstacle,
            poc_bias=poc_bias,
            value_area_position=va_position,
            profile_oscillator=profile_osc,
            hvn_proximity=hvn_prox,
            lvn_proximity=lvn_prox,
            vwap_oscillator=vwap_osc,
            volume_momentum=vol_momentum,
            tick_count=len(ticks),
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            net_delta=net_delta,
            vwap=vwap,
            poc=poc,
            vah=vah,
            val=val,
        )

    # ------------------------------------------------------------------
    # Core orderflow helpers
    # ------------------------------------------------------------------

    def _compute_cvd_slope(self) -> float:
        series = list(self._cvd_series)[-self.cvd_slope_lookback:]
        n = len(series)
        if n < 3:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(series) / n
        num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(series))
        den = sum((i - x_mean) ** 2 for i in range(n))
        if den == 0:
            return 0.0
        slope = num / den
        norm = max(abs(y_mean), 1.0)
        return _clamp(slope / norm, -1.0, 1.0)

    def _compute_imbalance(self, ticks: list[dict]) -> float:
        scores = [(t["buy_vol"] - t["sell_vol"]) / max(t["buy_vol"] + t["sell_vol"], 1.0)
                  for t in ticks]
        return _clamp(sum(scores) / len(scores), -1.0, 1.0)

    def _compute_absorption(self, ticks: list[dict]) -> float:
        if len(ticks) < 2:
            return 0.0
        mean_vol = sum(t["volume"] for t in ticks) / len(ticks)
        absorbed = sum(
            1 for prev, curr in zip(ticks, ticks[1:])
            if abs(curr["last"] - prev["last"]) < 1e-8 and curr["volume"] >= mean_vol * 1.5
        )
        return min(absorbed / max(len(ticks) - 1, 1), 1.0)

    def _compute_vwap(self, ticks: list[dict]) -> tuple[float, float]:
        tp_vol   = sum(t["last"] * t["volume"] for t in ticks if t["last"] > 0)
        total_vol = sum(t["volume"] for t in ticks)
        if total_vol <= 0:
            return 0.0, 0.0
        vwap = tp_vol / total_vol
        last_price = ticks[-1]["last"] or ticks[-1]["ask"] or ticks[-1]["bid"]
        if vwap <= 0 or last_price <= 0:
            return vwap, 0.0
        deviation = (last_price - vwap) / vwap
        return vwap, _clamp(deviation / 0.005, -1.0, 1.0)

    def _compute_liquidity_obstacle(self, ticks: list[dict]) -> float:
        spreads = [t["ask"] - t["bid"] for t in ticks if t["ask"] > 0 and t["bid"] > 0]
        if not spreads:
            return 0.0
        avg_spread = sum(spreads) / len(spreads)
        mid = (ticks[-1]["ask"] + ticks[-1]["bid"]) / 2.0 if ticks else 1.0
        pip = mid * 0.00001 if mid > 0 else 0.0001
        ref = self.max_spread_reference * pip * 10.0
        return _clamp(avg_spread / ref, 0.0, 1.0) if ref > 0 else 0.0

    # ------------------------------------------------------------------
    # Volume profile helpers
    # ------------------------------------------------------------------

    def _compute_profile_metrics(
        self,
        profile: VolumeProfile | None,
        current_price: float,
        ticks: list[dict],
    ) -> tuple[float, float, float, float, float, float, float, float]:
        """
        Returns:
            poc_bias, va_position, profile_oscillator,
            hvn_proximity, lvn_proximity, poc, vah, val
        """
        if profile is None or profile.value_area_width <= 0:
            return 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0

        poc = profile.poc
        vah = profile.vah
        val = profile.val
        va_width = profile.value_area_width

        # POC bias: how far above/below POC is current price, relative to VA width
        poc_bias = _clamp((current_price - poc) / va_width, -1.0, 1.0)

        # Value area position
        if current_price > vah:
            va_position = 1.0   # above value area — bullish breakout territory
        elif current_price < val:
            va_position = -1.0  # below value area — bearish breakdown territory
        else:
            # Normalised position inside VA: 0 = at VAL, 1 = at VAH, centre = 0.5
            inner = (current_price - val) / va_width
            va_position = (inner - 0.5) * 2.0  # map [0,1] → [-1,1]

        # HVN proximity: 0 = sitting right on a HVN, 1 = far from any HVN
        hvn_prices = self._profile_builder.find_hvn(profile, current_price)
        hvn_prox = _node_proximity(current_price, hvn_prices, va_width)

        # LVN proximity: 0 = sitting on a thin node (fast move area), 1 = far from any LVN
        lvn_prices = self._profile_builder.find_lvn(profile, current_price)
        lvn_prox = _node_proximity(current_price, lvn_prices, va_width)

        # Profile oscillator: composite
        # + price above POC, inside/above VA → bullish
        # + near HVN (support/resistance) → context for the move
        # + LVN below price → price passed through thin air upward → bullish
        profile_osc = _clamp(
            poc_bias * 0.40
            + va_position * 0.35
            + (1.0 - hvn_prox) * 0.15 * (1.0 if poc_bias >= 0 else -1.0)
            + (1.0 - lvn_prox) * 0.10,
            -1.0, 1.0,
        )

        return poc_bias, va_position, profile_osc, hvn_prox, lvn_prox, poc, vah, val

    # ------------------------------------------------------------------
    # Volume oscillators
    # ------------------------------------------------------------------

    def _compute_vwap_oscillator(self, ticks: list[dict], vwap: float) -> float:
        """
        Rolling mid-price vs VWAP oscillator.
        Positive = price consistently trading above VWAP (bullish).
        """
        if vwap <= 0:
            return 0.0
        mids = [(t["ask"] + t["bid"]) / 2.0 for t in ticks if t["ask"] > 0 and t["bid"] > 0]
        if not mids:
            return 0.0
        deviations = [(m - vwap) / vwap for m in mids]
        avg_dev = sum(deviations) / len(deviations)
        # ±0.3% average deviation → clip to ±1
        return _clamp(avg_dev / 0.003, -1.0, 1.0)

    def _compute_volume_momentum(self, ticks: list[dict]) -> float:
        """
        Compares buy-side volume in recent half vs earlier half of window.
        +1 = accelerating buying, -1 = accelerating selling.
        """
        n = len(ticks)
        if n < 4:
            return 0.0
        half = n // 2
        early = ticks[:half]
        recent = ticks[half:]

        def _net_bias(chunk: list[dict]) -> float:
            buy = sum(t["buy_vol"] for t in chunk)
            sell = sum(t["sell_vol"] for t in chunk)
            total = max(buy + sell, 1.0)
            return (buy - sell) / total

        early_bias  = _net_bias(early)
        recent_bias = _net_bias(recent)
        return _clamp(recent_bias - early_bias, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Serialise → OrderflowSignalStore-compatible payload
    # ------------------------------------------------------------------

    def to_signal_payload(self) -> dict[str, Any] | None:
        snap = self.snapshot()
        if snap is None:
            return None
        return {
            "symbol": snap.symbol,
            "target_symbol": snap.symbol,
            "timeframe": "TICK",
            "timestamp": snap.timestamp.isoformat(),
            # Core
            "buy_volume": snap.buy_volume,
            "sell_volume": snap.sell_volume,
            "delta": snap.net_delta,
            "cvd_slope": snap.cvd_slope,
            "imbalance": snap.imbalance_score,
            "absorption": snap.absorption_score,
            "vwap_bias": snap.vwap_alignment,
            "liquidity_obstacle": snap.liquidity_obstacle_score,
            # Volume profile
            "profile_location": snap.profile_oscillator,
            "poc_bias": snap.poc_bias,
            "value_area_position": snap.value_area_position,
            "hvn_proximity": snap.hvn_proximity,
            "lvn_proximity": snap.lvn_proximity,
            # Volume oscillators
            "vwap_oscillator": snap.vwap_oscillator,
            "volume_momentum": snap.volume_momentum,
            # Raw levels for logging/dashboard
            "poc": snap.poc,
            "vah": snap.vah,
            "val": snap.val,
            "vwap": snap.vwap,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_proximity(price: float, node_prices: list[float], normaliser: float) -> float:
    """0 = right on a node, 1 = far from all nodes (distance > normaliser)."""
    if not node_prices or normaliser <= 0:
        return 1.0
    min_dist = min(abs(price - p) for p in node_prices)
    return _clamp(min_dist / normaliser, 0.0, 1.0)


def _extract_tick(tick: Any) -> dict[str, float] | None:
    """Safely extract bid/ask/last/volume/time from an MT5 tick."""
    def _get(name: str, default: float = 0.0) -> float:
        if hasattr(tick, name):
            v = getattr(tick, name)
        else:
            try:
                v = tick[name]
            except (KeyError, IndexError, TypeError, ValueError):
                return default
        try:
            return float(v or 0.0)
        except (TypeError, ValueError):
            return default

    bid  = _get("bid")
    ask  = _get("ask")
    last = _get("last") or _get("last_deal")
    volume = _get("volume") or _get("volume_real") or 1.0
    time_val = _get("time") or _get("time_msc", 0.0) / 1000.0

    if bid <= 0 and ask <= 0:
        return None

    return {
        "bid": bid, "ask": ask, "last": last,
        "volume": volume, "volume_real": _get("volume_real"),
        "time": time_val,
    }


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
