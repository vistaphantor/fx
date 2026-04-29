from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from src.market_data import Candle

@dataclass(frozen=True, slots=True)
class OrderBlock:
    price_low: float
    price_high: float
    direction: int  # 1 for Demand (Bullish OB), -1 for Supply (Bearish OB)
    strength: float # Body-to-range ratio
    imbalance: float # Volume ratio
    timestamp: Any
    bars_since: int = 0
    is_fresh: bool = True

@dataclass(frozen=True, slots=True)
class FairValueGap:
    top: float
    bottom: float
    direction: int # 1 for Bullish gap, -1 for Bearish gap
    timestamp: Any
    bars_since: int = 0
    fill_ratio: float = 0.0

def detect_order_blocks(candles: list[Candle], lookback: int = 50) -> list[OrderBlock]:
    """
    Detects Order Blocks (OB). 
    A Bullish OB is typically the last bearish candle before a strong bullish move.
    A Bearish OB is typically the last bullish candle before a strong bearish move.
    """
    if len(candles) < 3:
        return []

    obs = []
    # Simplified logic for detection:
    # Bullish OB: Bearish candle followed by a significant bullish engulfing or displacement
    # Bearish OB: Bullish candle followed by a significant bearish engulfing or displacement
    
    for i in range(len(candles) - 2, len(candles) - lookback - 2, -1):
        if i < 0: break
        
        c = candles[i]
        c_next = candles[i+1]
        
        # Bullish OB (Demand)
        if c.close < c.open: # Bearish candle
            if c_next.close > c_next.open and (c_next.close - c_next.open) > (c.open - c.close) * 1.5:
                strength = abs(c.close - c.open) / max(c.high - c.low, 1e-9)
                obs.append(OrderBlock(
                    price_low=float(c.low),
                    price_high=float(c.high),
                    direction=1,
                    strength=strength,
                    imbalance=float(c_next.volume / c.volume) if c.volume > 0 else 1.0,
                    timestamp=c.timestamp,
                    bars_since=len(candles) - 1 - i
                ))
        
        # Bearish OB (Supply)
        elif c.close > c.open: # Bullish candle
            if c_next.close < c_next.open and (c_next.open - c_next.close) > (c.close - c.open) * 1.5:
                strength = abs(c.close - c.open) / max(c.high - c.low, 1e-9)
                obs.append(OrderBlock(
                    price_low=float(c.low),
                    price_high=float(c.high),
                    direction=-1,
                    strength=strength,
                    imbalance=float(c_next.volume / c.volume) if c.volume > 0 else 1.0,
                    timestamp=c.timestamp,
                    bars_since=len(candles) - 1 - i
                ))
                
    return obs

def detect_fair_value_gaps(candles: list[Candle], lookback: int = 20) -> list[FairValueGap]:
    """
    Detects Fair Value Gaps (FVG).
    Bullish FVG: low of candle i+1 is higher than high of candle i-1.
    Bearish FVG: high of candle i+1 is lower than low of candle i-1.
    """
    fvgs = []
    if len(candles) < 3:
        return []
        
    for i in range(len(candles) - 2, len(candles) - lookback - 2, -1):
        if i < 1: break
        
        prev = candles[i-1]
        mid = candles[i]
        post = candles[i+1]
        
        # Bullish FVG
        if post.low > prev.high:
            fvgs.append(FairValueGap(
                top=float(post.low),
                bottom=float(prev.high),
                direction=1,
                timestamp=mid.timestamp,
                bars_since=len(candles) - 1 - i
            ))
            
        # Bearish FVG
        elif post.high < prev.low:
            fvgs.append(FairValueGap(
                top=float(prev.low),
                bottom=float(post.high),
                direction=-1,
                timestamp=mid.timestamp,
                bars_since=len(candles) - 1 - i
            ))
            
    return fvgs

def score_structural_context(
    price: float, 
    candles: list[Candle], 
    demand_zones: tuple[tuple[float, float], ...], 
    supply_zones: tuple[tuple[float, float], ...],
    atr: float
) -> float:
    """
    Computes a composite structural quality score [0, 1].
    """
    if atr <= 0: return 0.5
    
    obs = detect_order_blocks(candles)
    fvgs = detect_fair_value_gaps(candles)
    
    ob_score = 0.0
    fvg_score = 0.0
    sd_score = 0.0
    
    # 1. OB Scoring
    for ob in obs:
        freshness = math.exp(-0.1 * ob.bars_since)
        proximity = max(0, 1 - abs(price - (ob.price_high + ob.price_low)/2) / (atr * 5))
        ob_score += ob.strength * ob.imbalance * freshness * proximity
    
    ob_score = min(ob_score, 1.0)
    
    # 2. FVG Scoring
    for fvg in fvgs:
        freshness = math.exp(-0.1 * fvg.bars_since)
        proximity = max(0, 1 - abs(price - (fvg.top + fvg.bottom)/2) / (atr * 5))
        fvg_score += freshness * proximity
    
    fvg_score = min(fvg_score, 1.0)
    
    # 3. S/D Confluence
    for lower, upper in demand_zones:
        if lower <= price <= upper:
            sd_score += 1.0
        else:
            dist = min(abs(price - lower), abs(price - upper))
            sd_score += max(0, 1 - dist / atr)
            
    for lower, upper in supply_zones:
        if lower <= price <= upper:
            sd_score += 1.0
        else:
            dist = min(abs(price - lower), abs(price - upper))
            sd_score += max(0, 1 - dist / atr)

    sd_score = min(sd_score / 4.0, 1.0) # Normalize
    
    # Composite
    return 0.4 * ob_score + 0.3 * fvg_score + 0.3 * sd_score
