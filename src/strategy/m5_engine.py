"""
m5_engine.py
------------
Computes high-fidelity micro-structure and momentum signals from M5 candles.
These signals provide the 'execution-timeframe' context that filters or 
boosts the higher-timeframe setups.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence
from src.market_data import Candle
from src.strategy.breakout import BreakoutDirection


@dataclass(frozen=True)
class M5SignalSnapshot:
    rsi: float                # 0-100
    ema_stack_score: float    # -1 to 1 (direction and alignment)
    atr_regime: float         # > 1 expansion, < 1 contraction
    velocity: float           # -1 to 1 (consecutive directional closes)
    vol_ratio: float          # current bar vol / avg vol
    swing_recency: float      # 0 to 1 (1 = just happened)
    body_ratio: float         # 0 to 1 (avg body ratio)
    structure_score: float    # -1 to 1 (HH/HL vs LH/LL)
    ofi: float                # -1 to 1 (order flow imbalance proxy)
    composite_score: float    # -1 to 1 (overall bullish/bearish alignment)


class M5SignalEngine:
    def __init__(self, rsi_period: int = 14, ema_fast: int = 8, ema_mid: int = 21, ema_slow: int = 50):
        self.rsi_period = rsi_period
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow

    def compute(self, candles: Sequence[Candle]) -> M5SignalSnapshot | None:
        if len(candles) < max(self.rsi_period, self.ema_slow) + 5:
            return None

        closes = [float(c.close) for c in candles]
        highs = [float(c.high) for c in candles]
        lows = [float(c.low) for c in candles]
        vols = [float(c.volume) for c in candles]

        # 1. RSI
        rsi = self._calculate_rsi(closes)

        # 2. EMA Stack
        ema8 = self._calculate_ema(closes, self.ema_fast)
        ema21 = self._calculate_ema(closes, self.ema_mid)
        ema50 = self._calculate_ema(closes, self.ema_slow)
        
        ema_score = 0.0
        if ema8 > ema21 > ema50:
            ema_score = 1.0 if closes[-1] > ema8 else 0.7
        elif ema8 < ema21 < ema50:
            ema_score = -1.0 if closes[-1] < ema8 else -0.7
        
        # 3. ATR Regime
        atr_short = self._calculate_atr(highs[-5:], lows[-5:], closes[-6:])
        atr_long = self._calculate_atr(highs[-20:], lows[-20:], closes[-21:])
        atr_regime = atr_short / atr_long if atr_long > 0 else 1.0

        # 4. Velocity (Consecutive closes)
        velocity = 0.0
        last_dir = 0
        count = 0
        for i in range(len(closes)-1, len(closes)-6, -1):
            curr_dir = 1 if closes[i] > closes[i-1] else -1 if closes[i] < closes[i-1] else 0
            if count == 0:
                last_dir = curr_dir
                count = 1
            elif curr_dir == last_dir and curr_dir != 0:
                count += 1
            else:
                break
        velocity = (count / 5.0) * last_dir

        # 5. Vol Ratio
        avg_vol = sum(vols[-20:]) / 20.0
        vol_ratio = vols[-1] / avg_vol if avg_vol > 0 else 1.0

        # 6. Body Ratio
        bodies = [abs(float(c.close) - float(c.open)) for c in candles[-10:]]
        ranges = [max(float(c.high) - float(c.low), 1e-9) for c in candles[-10:]]
        body_ratio = sum(b/r for b, r in zip(bodies, ranges)) / 10.0

        # 7. Structure (HH/HL)
        structure_score = 0.0
        hh = highs[-1] > max(highs[-6:-1])
        hl = lows[-1] > min(lows[-6:-1])
        ll = lows[-1] < min(lows[-6:-1])
        lh = highs[-1] < max(highs[-6:-1])
        
        if hh and hl: structure_score = 1.0
        elif ll and lh: structure_score = -1.0
        elif hh: structure_score = 0.5
        elif ll: structure_score = -0.5

        # 8. OFI Proxy
        # OFI = (BidVol - AskVol) -> here we use price action proxy
        # If close > open and close near high -> bullish imbalance
        ofi = 0.0
        for i in range(len(candles)-5, len(candles)):
            c = candles[i]
            body = float(c.close) - float(c.open)
            full_range = max(float(c.high) - float(c.low), 1e-9)
            ofi += (body / full_range) * (float(c.volume) / avg_vol)
        ofi = max(-1.0, min(1.0, ofi / 5.0))

        # 9. Swing Recency
        # Just use distance from last 20-bar high/low
        dist_h = (max(highs[-20:]) - closes[-1]) / max(atr_long, 1e-9)
        dist_l = (closes[-1] - min(lows[-20:])) / max(atr_long, 1e-9)
        swing_recency = 1.0 - min(dist_h, dist_l) / 5.0
        swing_recency = max(0.0, min(1.0, swing_recency))

        # Composite Score
        composite = (
            ema_score * 0.25 +
            structure_score * 0.20 +
            velocity * 0.15 +
            ofi * 0.15 +
            (1.0 if rsi > 50 else -1.0) * 0.10 +
            (vol_ratio - 1.0) * 0.10 * (1.0 if closes[-1] > closes[-2] else -1.0) +
            (atr_regime - 1.0) * 0.05
        )
        composite = max(-1.0, min(1.0, composite))

        return M5SignalSnapshot(
            rsi=rsi,
            ema_stack_score=ema_score,
            atr_regime=atr_regime,
            velocity=velocity,
            vol_ratio=vol_ratio,
            swing_recency=swing_recency,
            body_ratio=body_ratio,
            structure_score=structure_score,
            ofi=ofi,
            composite_score=composite
        )

    def _calculate_rsi(self, prices: list[float], period: int = 14) -> float:
        if len(prices) < period + 1: return 50.0
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0: return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _calculate_ema(self, prices: list[float], period: int) -> float:
        ema = prices[0]
        multiplier = 2 / (period + 1)
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def _calculate_atr(self, highs: list[float], lows: list[float], prev_closes: list[float]) -> float:
        tr = []
        for h, l, pc in zip(highs, lows, prev_closes):
            tr.append(max(h - l, abs(h - pc), abs(l - pc)))
        return sum(tr) / len(tr) if tr else 0.0
