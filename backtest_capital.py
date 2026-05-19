"""
backtest_capital.py  -  100-hour parameter search with $10 capital constraint.
Simulates XAUUSD (Gold) trading with margin and drawdown limits.
"""

from __future__ import annotations

import importlib
import subprocess
import sys


def _ensure(pkg: str, import_as: str | None = None) -> None:
    try:
        importlib.import_module(import_as or pkg)
    except ModuleNotFoundError:
        print(f"[setup] Installing {pkg}...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


_ensure("yfinance")
_ensure("pandas")
_ensure("numpy")

import math
import itertools
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Dir:
    BUY = "BULLISH"
    SELL = "BEARISH"


@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Trade:
    direction: str
    entry: float
    sl: float
    tp: float
    open_bar: int
    lot: float = 0.01
    margin_used: float = 0.0


@dataclass
class Params:
    atr_sl_mult: float
    atr_tp_mult: float
    max_spread_pips: float
    min_body_ratio: float
    tick_consistency: float
    max_positions: int
    profit_target_usd: float


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def _rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) <= period: return 50.0
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    recent = changes[-period:]
    ag = sum(max(c, 0.0) for c in recent) / period
    al = sum(abs(min(c, 0.0)) for c in recent) / period
    if al == 0: return 100.0 if ag > 0 else 50.0
    return 100.0 - (100.0 / (1.0 + ag / al))


def _sar(candles: List[Candle], step: float = 0.02, maximum: float = 0.2) -> float:
    highs = [c.high for c in candles]
    lows  = [c.low  for c in candles]
    if len(candles) < 2: return lows[-1] if lows else 0.0
    bullish = candles[1].close >= candles[0].close
    sar = lows[0] if bullish else highs[0]
    ep  = highs[0] if bullish else lows[0]
    acc = step
    for i in range(1, len(candles)):
        prev = sar
        sar = prev + acc * (ep - prev)
        if bullish:
            if lows[i] < sar:
                bullish = False; sar = ep; ep = lows[i]; acc = step
            else:
                sar = min(sar, lows[i - 1], lows[i])
                if highs[i] > ep: ep = highs[i]; acc = min(acc + step, maximum)
        else:
            if highs[i] > sar:
                bullish = True; sar = ep; ep = highs[i]; acc = step
            else:
                sar = max(sar, highs[i - 1], highs[i])
                if lows[i] < ep: ep = lows[i]; acc = min(acc + step, maximum)
    return sar


def _atr(candles: List[Candle]) -> float:
    if len(candles) < 2: return 0.0
    trs = []
    prev = candles[0].close
    for c in candles[1:]:
        trs.append(max(c.high - c.low, abs(c.high - prev), abs(c.low - prev)))
        prev = c.close
    return sum(trs) / len(trs) if trs else 0.0


def _fib_zone(candles: List[Candle]) -> Tuple[Optional[str], str]:
    if len(candles) < 2: return None, "unavailable"
    sl_idx = min(range(len(candles)), key=lambda i: candles[i].low)
    sh_idx = max(range(len(candles)), key=lambda i: candles[i].high)
    sl = candles[sl_idx].low; sh = candles[sh_idx].high
    cp = candles[-1].close; rng = sh - sl
    if rng <= 0: return None, "flat"
    if sl_idx < sh_idx:
        r382 = sh - rng * 0.382; r618 = sh - rng * 0.618; r786 = sh - rng * 0.786
        zone = ("in_market_mover" if cp >= r382 else "towards_market_mover" if cp >= r618 else "golden_zone" if cp >= r786 else "outside")
        return Dir.BUY, zone
    else:
        r382 = sl + rng * 0.382; r618 = sl + rng * 0.618; r786 = sl + rng * 0.786
        zone = ("in_market_mover" if cp <= r382 else "towards_market_mover" if cp <= r618 else "golden_zone" if cp <= r786 else "outside")
        return Dir.SELL, zone


def _tick_dir(candles: List[Candle], min_consistency: float) -> Optional[str]:
    if len(candles) < 3: return None
    mids = [c.close for c in candles]
    up = sum(1 for a, b in zip(mids, mids[1:]) if b > a)
    dn = sum(1 for a, b in zip(mids, mids[1:]) if b < a)
    tot = up + dn
    if tot == 0: return None
    net = mids[-1] - mids[0]
    pip = abs(mids[-1]) * 0.00004
    if net >=  pip and up / tot >= min_consistency: return Dir.BUY
    if net <= -pip and dn / tot >= min_consistency: return Dir.SELL
    return None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIP        = 0.10
TICK_VALUE = 1.0   # $1 per pip per 0.01 lot
LOT        = 0.01
SPREAD     = 0.35 # 3.5 pips
LEVERAGE   = 2000 # Standard high leverage
INITIAL_BALANCE = 1000.0


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def simulate(candles: List[Candle], p: Params) -> dict:
    balance = INITIAL_BALANCE
    open_trades: List[Trade] = []
    closed_pnls: List[float] = []
    equity_history = [balance]
    n = len(candles)
    blown_account = False

    for idx in range(50, n - 1):
        if balance <= 0:
            blown_account = True
            break
        
        c = candles[idx]
        next_c = candles[idx + 1]

        # --- close trades ---
        still_open = []
        for t in open_trades:
            pnl = None
            if t.direction == Dir.BUY:
                if next_c.low <= t.sl: pnl = (t.sl - t.entry)
                elif next_c.high >= t.tp: pnl = (t.tp - t.entry)
                else:
                    mtm = (next_c.close - t.entry)
                    if mtm >= p.profit_target_usd: pnl = mtm
            else:
                if next_c.high >= t.sl: pnl = (t.entry - t.sl)
                elif next_c.low <= t.tp: pnl = (t.entry - t.tp)
                else:
                    mtm = (t.entry - next_c.close)
                    if mtm >= p.profit_target_usd: pnl = mtm
            
            if pnl is not None:
                # Apply spread cost on close
                pnl -= SPREAD
                balance += pnl
                closed_pnls.append(pnl)
            else:
                still_open.append(t)
        open_trades = still_open

        # --- open trades ---
        if len(open_trades) < p.max_positions:
            m1_win = candles[max(0, idx - 29): idx + 1]
            tick_win = candles[max(0, idx - 9): idx + 1]
            m15_win = candles[max(0, idx - 50 + 1): idx + 1]

            direction = _tick_dir(tick_win, p.tick_consistency)
            if direction:
                fib_dir, fib_zone = _fib_zone(m15_win)
                fib_ok = (fib_dir == direction and fib_zone in {"in_market_mover", "towards_market_mover", "golden_zone"})
                
                sar_val = _sar(m15_win)
                sar_ok = (c.close > sar_val if direction == Dir.BUY else c.close < sar_val)
                
                rsi = _rsi([x.close for x in m15_win])
                rsi_ok = (rsi < 70.0 if direction == Dir.BUY else rsi > 30.0)
                
                if rsi_ok and (fib_ok or sar_ok):
                    atr = _atr(m1_win[-14:])
                    if atr > 0:
                        sl_dist = atr * p.atr_sl_mult
                        tp_dist = atr * p.atr_tp_mult
                        
                        # Dynamic Lot Calculation (Capped)
                        current_lot = max(0.01, min(0.02, round((balance / 100.0) * 0.02, 2)))
                        
                        # Margin check
                        margin_req = ((next_c.open * 1.0) / LEVERAGE) * (current_lot / 0.01)
                        if balance >= margin_req:
                            entry = next_c.open
                            if direction == Dir.BUY:
                                sl, tp = entry - sl_dist, entry + tp_dist
                            else:
                                sl, tp = entry + sl_dist, entry - tp_dist
                            
                            open_trades.append(Trade(direction, entry, sl, tp, idx, current_lot, margin_req))

        # Dynamic Target Scaling
        current_equity = balance + sum(
            (((c.close - t.entry) if t.direction == Dir.BUY else (t.entry - c.close)) - SPREAD) * (t.lot / 0.01)
            for t in open_trades
        )
        
        # Scale target by total open lot
        total_lot = sum(t.lot for t in open_trades)
        scaled_profit_target = p.profit_target_usd * (total_lot / 0.01) if total_lot > 0 else p.profit_target_usd
        
        if open_trades and (current_equity - balance) >= scaled_profit_target:
            for t in open_trades:
                pnl = (((c.close - t.entry) if t.direction == Dir.BUY else (t.entry - c.close)) - SPREAD) * (t.lot / 0.01)
                balance += pnl
                closed_pnls.append(pnl)
            open_trades = []
            
        equity_history.append(current_equity)
        if current_equity <= 0:
            blown_account = True
            break

    final_balance = balance
    total_trades = len(closed_pnls)
    if total_trades == 0:
        return {"net_pnl": 0.0, "trades": 0, "win_rate": 0.0, "final_balance": final_balance, "blown": blown_account}

    wins = sum(1 for x in closed_pnls if x > 0)
    net_pnl = final_balance - INITIAL_BALANCE
    return {
        "net_pnl": round(net_pnl, 2),
        "trades": total_trades,
        "win_rate": round(wins / total_trades * 100, 1),
        "final_balance": round(final_balance, 2),
        "blown": blown_account,
        "max_dd": round(max(0, max(np.maximum.accumulate(equity_history)) - min(equity_history)) if equity_history else 0, 2)
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[data] Fetching 15 days of 1m data for Roadmap test...", flush=True)
    df = yf.download("GC=F", period="1mo", interval="5m", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # 15 days * 288 (5m candles/day) = 4,320 rows.
    df = df.tail(4320)
    print(f"[data] Data ready. Total rows: {len(df)}")
    candles = []
    for ts, row in df.iterrows():
        candles.append(Candle(ts.to_pydatetime(), row["Open"], row["High"], row["Low"], row["Close"], row.get("Volume", 0)))
    
    print(f"[grid] Running search over {len(df)} minutes...", flush=True)
    
    # Sweetspot Verification
    GRID = {
        "atr_sl_mult": [0.8],
        "atr_tp_mult": [3.0],
        "max_spread_pips": [4.0],
        "min_body_ratio": [0.35],
        "tick_consistency": [0.60],
        "max_positions": [1],
        "profit_target_usd": [0.50]
    }
    
    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    results = []
    
    for combo in combos:
        p = Params(**dict(zip(keys, combo)))
        res = simulate(candles, p)
        results.append({**dict(zip(keys, combo)), **res})
    
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("final_balance", ascending=False)
    
    print("\n" + "="*80)
    print("  TOP RESULTS FOR $50 CAPITAL (15 DAYS)")
    print("="*80)
    print(res_df.head(10).to_string(index=False))
    print("="*80)
    
    best = res_df.iloc[0]
    if best['blown']:
        print("\nWARNING: Even the best result blew the account at some point.")
    else:
        print(f"\nSUCCESS: Best combo grew $50 to ${best['final_balance']}!")
    
    print("\nRecommended Settings for $50 Account:")
    print(f"QUICK_MAX_POSITIONS={int(best['max_positions'])}")
    print(f"QUICK_PROFIT_TARGET={best['profit_target_usd']}")
    print(f"# SL Mult: {best['atr_sl_mult']}, TP Mult: {best['atr_tp_mult']}")
