"""
backtest_multi_tf.py
--------------------
Advanced backtest script that utilizes the new M5, M15, and Context engines.
Simulates the 'Master Equation' logic over a 96-hour window.
"""

from __future__ import annotations

import os
import sys
import math
import itertools
import multiprocessing
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.market_data import Candle
from src.strategy.m5_engine import M5SignalEngine
from src.strategy.m15_engine import M15SignalEngine
from src.strategy.context_engine import ContextEngine

# Simulated Quant Engine Logic
def compute_omega(m5_score: float, m15_score: float, context_score: float) -> float:
    linear = (
        0.90 * m5_score +
        0.85 * m15_score +
        0.80 * context_score
    )
    return 1.0 / (1.0 + math.exp(-linear))

@dataclass
class Trade:
    direction: str
    entry: float
    sl: float
    tp: float
    omega: float

def simulate_strategy(df: pd.DataFrame, omega_threshold: float = 0.55) -> dict:
    closes = df['Close'].tolist()
    highs = df['High'].tolist()
    lows = df['Low'].tolist()
    opens = df['Open'].tolist()
    volumes = df['Volume'].tolist()
    times = df.index.tolist()
    
    candles = [Candle(times[i], opens[i], highs[i], lows[i], closes[i], int(volumes[i]), "M1") for i in range(len(df))]
    
    m5_eng = M5SignalEngine()
    m15_eng = M15SignalEngine()
    ctx_eng = ContextEngine()
    
    trades = []
    closed_pnls = []
    start_idx = 3000 
    active_trade: Trade | None = None
    max_omega = 0.5
    
    for i in range(start_idx, len(candles) - 1):
        curr_candle = candles[i]
        next_candle = candles[i+1]
        
        if active_trade:
            if active_trade.direction == "BUY":
                if next_candle.low <= active_trade.sl:
                    closed_pnls.append(active_trade.sl - active_trade.entry)
                    active_trade = None
                elif next_candle.high >= active_trade.tp:
                    closed_pnls.append(active_trade.tp - active_trade.entry)
                    active_trade = None
            else:
                if next_candle.high >= active_trade.sl:
                    closed_pnls.append(active_trade.entry - active_trade.sl)
                    active_trade = None
                elif next_candle.low <= active_trade.tp:
                    closed_pnls.append(active_trade.entry - active_trade.tp)
                    active_trade = None
        
        if active_trade: continue
        
        m5_candles = candles[i-500:i+1:5]
        m15_candles = candles[i-1500:i+1:15]
        m30_candles = candles[i-3000:i+1:30]
        h1_candles = candles[i-6000:i+1:60]
        d1_candles = candles[i-4320:i+1:1440]
        
        m5_snap = m5_eng.compute(m5_candles)
        m15_snap = m15_eng.compute(m15_candles)
        ctx_snap = ctx_eng.compute(h1_candles, m30_candles, d1_candles)
        
        if not (m5_snap and m15_snap and ctx_snap): continue
        
        omega = compute_omega(m5_snap.composite_score, m15_snap.composite_score, ctx_snap.composite_context)
        max_omega = max(max_omega, omega)
        
        if omega >= omega_threshold:
            atr = (max(highs[i-14:i+1]) - min(lows[i-14:i+1])) / 2.0
            direction = "BUY"
            entry = next_candle.open
            sl = entry - atr * 1.5
            tp = entry + atr * 3.0
            active_trade = Trade(direction, entry, sl, tp, omega)
        elif omega <= (1.0 - omega_threshold):
            atr = (max(highs[i-14:i+1]) - min(lows[i-14:i+1])) / 2.0
            direction = "SELL"
            entry = next_candle.open
            sl = entry + atr * 1.5
            tp = entry - atr * 3.0
            active_trade = Trade(direction, entry, sl, tp, omega)

    print(f"  [debug] Max Omega observed: {max_omega:.4f}")
    
    if not closed_pnls:
        return {"net_pnl": 0, "trades": 0, "win_rate": 0}
        
    net_pnl = sum(closed_pnls)
    wins = sum(1 for p in closed_pnls if p > 0)
    
    return {
        "net_pnl": round(net_pnl, 4),
        "trades": len(closed_pnls),
        "win_rate": round(wins / len(closed_pnls) * 100, 2)
    }

if __name__ == "__main__":
    print("[data] Downloading 120h of data for MTF backtest...")
    df = yf.download("GC=F", period="5d", interval="1m", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    print(f"[data] {len(df)} M1 bars ready.")
    
    print("\n[backtest] Running Multi-TF Engine Simulation...")
    results = simulate_strategy(df)
    
    print("\n" + "="*40)
    print("  96H MULTI-TF BACKTEST RESULTS")
    print("="*40)
    print(f"  Trades Executed : {results['trades']}")
    print(f"  Net P&L (pts)   : {results['net_pnl']}")
    print(f"  Win Rate        : {results['win_rate']}%")
    print("="*40)
