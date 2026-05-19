"""
m15_engine.py
-------------
Computes structural, liquidity, and volatility signals from M15 candles.
Focuses on 'setup-timeframe' context like sweeps, squeezes, and swing retracements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence
from src.market_data import Candle
from src.strategy.breakout import BreakoutDirection


@dataclass(frozen=True)
class M15SignalSnapshot:
    liquidity_sweep: float    # -1 (bearish sweep) to 1 (bullish sweep)
    volatility_squeeze: float # 0 (expansion) to 1 (max squeeze)
    retracement_level: float  # 0 to 1 (distance in current swing)
    trend_intensity: float    # -1 to 1 (organized directional movement)
    wick_rejection: float     # -1 (rejection from above) to 1 (rejection from below)
    composite_score: float    # -1 to 1 (overall M15 structural bias)


class M15SignalEngine:
    def __init__(self, swing_lookback: int = 50, squeeze_period: int = 20):
        self.swing_lookback = swing_lookback
        self.squeeze_period = squeeze_period

    def compute(self, candles: Sequence[Candle]) -> M15SignalSnapshot | None:
        if len(candles) < self.swing_lookback:
            return None

        closes = [float(c.close) for c in candles]
        highs = [float(c.high) for c in candles]
        lows = [float(c.low) for c in candles]
        opens = [float(c.open) for c in candles]

        # 1. Liquidity Sweep Detection
        sweep_score = 0.0
        # Bullish sweep: low < min(previous 10 lows) but close > min(previous 10 lows)
        prev_lows = lows[-11:-1]
        prev_highs = highs[-11:-1]
        min_low = min(prev_lows)
        max_high = max(prev_highs)
        
        if lows[-1] < min_low and closes[-1] > min_low:
            sweep_score = 1.0 # Bullish sweep
        elif highs[-1] > max_high and closes[-1] < max_high:
            sweep_score = -1.0 # Bearish sweep

        # 2. Volatility Squeeze (BB vs KC proxy)
        # Simplified: StdDev / ATR
        std = self._calculate_std(closes[-self.squeeze_period:])
        atr = self._calculate_atr(highs[-self.squeeze_period:], lows[-self.squeeze_period:], closes[-self.squeeze_period-1:])
        # Low ratio means Bollinger Bands are inside/near Keltner Channels
        squeeze_ratio = std / atr if atr > 0 else 1.0
        volatility_squeeze = max(0.0, 1.0 - squeeze_ratio)

        # 3. Swing Retracement Level
        swing_hi = max(highs[-self.swing_lookback:])
        swing_lo = min(lows[-self.swing_lookback:])
        swing_range = max(swing_hi - swing_lo, 1e-9)
        retracement_level = (closes[-1] - swing_lo) / swing_range

        # 4. Trend Intensity (ADX-style)
        plus_dm = sum(max(0, highs[i] - highs[i-1]) for i in range(-14, 0))
        minus_dm = sum(max(0, lows[i-1] - lows[i]) for i in range(-14, 0))
        tr_sum = sum(max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])) for i in range(-14, 0))
        
        di_plus = (plus_dm / tr_sum) if tr_sum > 0 else 0
        di_minus = (minus_dm / tr_sum) if tr_sum > 0 else 0
        trend_intensity = (di_plus - di_minus) / (di_plus + di_minus) if (di_plus + di_minus) > 0 else 0

        # 5. Multi-bar Wick Rejection
        # Average wick size over last 5 bars
        upper_wicks = [h - max(o, c) for h, o, c in zip(highs[-5:], opens[-5:], closes[-5:])]
        lower_wicks = [min(o, c) - l for l, o, c in zip(lows[-5:], opens[-5:], closes[-5:])]
        avg_range = sum(h-l for h, l in zip(highs[-5:], lows[-5:])) / 5.0
        
        wick_rej_score = (sum(lower_wicks) - sum(upper_wicks)) / (avg_range * 5.0) if avg_range > 0 else 0.0
        wick_rej_score = max(-1.0, min(1.0, wick_rej_score))

        # Composite Score
        # Weighting structural factors
        composite = (
            trend_intensity * 0.35 +
            sweep_score * 0.25 +
            wick_rej_score * 0.20 +
            (1.0 if volatility_squeeze > 0.5 else 0.0) * trend_intensity * 0.10 +
            (0.5 - abs(retracement_level - 0.5)) * 0.10 # Bonus for being at mid-retracement
        )
        composite = max(-1.0, min(1.0, composite))

        return M15SignalSnapshot(
            liquidity_sweep=sweep_score,
            volatility_squeeze=volatility_squeeze,
            retracement_level=retracement_level,
            trend_intensity=trend_intensity,
            wick_rejection=wick_rej_score,
            composite_score=composite
        )

    def _calculate_std(self, data: list[float]) -> float:
        if not data: return 0.0
        mean = sum(data) / len(data)
        var = sum((x - mean)**2 for x in data) / len(data)
        return math.sqrt(var)

    def _calculate_atr(self, highs: list[float], lows: list[float], prev_closes: list[float]) -> float:
        tr = []
        for i in range(len(highs)):
            h, l = highs[i], lows[i]
            pc = prev_closes[i]
            tr.append(max(h - l, abs(h - pc), abs(l - pc)))
        return sum(tr) / len(tr) if tr else 0.0
