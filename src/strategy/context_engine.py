"""
context_engine.py
-----------------
Computes high-level contextual signals from M30 and H1 candles.
Focuses on 'bias-timeframe' context like H1 structure, M30 volume nodes, and PDH/PDL levels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence
from src.market_data import Candle
from src.strategy.breakout import BreakoutDirection


@dataclass(frozen=True)
class ContextSignalSnapshot:
    h1_trend_bias: float      # -1 (bearish structure) to 1 (bullish structure)
    m30_volume_bias: float    # -1 to 1 (volume-weighted directional bias)
    pd_proximity: float       # -1 (at PDL) to 1 (at PDH), 0 in middle
    fvg_proximity: float      # distance to nearest H1 FVG
    composite_context: float  # -1 to 1 (overall context bias)


class ContextEngine:
    def __init__(self, h1_lookback: int = 48, m30_lookback: int = 48):
        self.h1_lookback = h1_lookback
        self.m30_lookback = m30_lookback

    def compute(self, h1_candles: Sequence[Candle], m30_candles: Sequence[Candle], d1_candles: Sequence[Candle]) -> ContextSignalSnapshot | None:
        if len(h1_candles) < 10 or len(m30_candles) < 10:
            return None

        h1_closes = [float(c.close) for c in h1_candles]
        m30_closes = [float(c.close) for c in m30_candles]
        m30_vols = [float(c.volume) for c in m30_candles]
        
        # 1. H1 Trend Bias (EMA 20/50 alignment)
        h1_ema20 = self._calculate_ema(h1_closes, 20)
        h1_ema50 = self._calculate_ema(h1_closes, 50)
        h1_trend = 1.0 if h1_ema20 > h1_ema50 else -1.0
        # Refine with slope
        h1_slope = (h1_closes[-1] - h1_closes[-5]) / 5.0
        h1_trend_bias = h1_trend * (1.2 if (h1_slope > 0 and h1_trend > 0) or (h1_slope < 0 and h1_trend < 0) else 0.8)
        h1_trend_bias = max(-1.0, min(1.0, h1_trend_bias))

        # 2. M30 Volume-Weighted Bias
        # Does high volume happen on up or down bars?
        m30_vbias = 0.0
        total_v = sum(m30_vols[-10:])
        if total_v > 0:
            for i in range(-10, 0):
                c = m30_candles[i]
                direction = 1 if float(c.close) > float(c.open) else -1
                m30_vbias += (float(c.volume) / total_v) * direction
        m30_volume_bias = max(-1.0, min(1.0, m30_vbias))

        # 3. PDH/PDL Proximity (using D1 candles)
        pd_proximity = 0.0
        if len(d1_candles) >= 2:
            prev_day = d1_candles[-2]
            pdh = float(prev_day.high)
            pdl = float(prev_day.low)
            curr_price = h1_closes[-1]
            if pdh > pdl:
                # Map price to [-1, 1] relative to prev day range
                pd_proximity = ((curr_price - pdl) / (pdh - pdl)) * 2.0 - 1.0
        pd_proximity = max(-1.0, min(1.0, pd_proximity))

        # 4. H1 Fair Value Gap (FVG) proximity
        # Simple FVG: gap between candle[i-2].high and candle[i].low (bearish) or vice versa
        fvg_prox = 0.0
        for i in range(len(h1_candles)-1, len(h1_candles)-10, -1):
            c0 = h1_candles[i]
            c2 = h1_candles[i-2]
            # Bullish FVG
            if float(c2.high) < float(c0.low):
                gap_mid = (float(c2.high) + float(c0.low)) / 2.0
                dist = (h1_closes[-1] - gap_mid)
                fvg_prox = max(fvg_prox, 1.0 - abs(dist)/(h1_closes[-1]*0.01))
            # Bearish FVG
            elif float(c2.low) > float(c0.high):
                gap_mid = (float(c2.low) + float(c0.high)) / 2.0
                dist = (h1_closes[-1] - gap_mid)
                fvg_prox = min(fvg_prox, -(1.0 - abs(dist)/(h1_closes[-1]*0.01)))
        fvg_proximity = max(-1.0, min(1.0, fvg_prox))

        # Composite Context
        composite = (
            h1_trend_bias * 0.40 +
            m30_volume_bias * 0.30 +
            pd_proximity * 0.15 +
            fvg_proximity * 0.15
        )
        composite = max(-1.0, min(1.0, composite))

        return ContextSignalSnapshot(
            h1_trend_bias=h1_trend_bias,
            m30_volume_bias=m30_volume_bias,
            pd_proximity=pd_proximity,
            fvg_proximity=fvg_proximity,
            composite_context=composite
        )

    def _calculate_ema(self, prices: list[float], period: int) -> float:
        if not prices: return 0.0
        ema = prices[0]
        multiplier = 2 / (period + 1)
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        return ema
