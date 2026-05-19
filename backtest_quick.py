"""
backtest_quick.py  -  96-hour parameter grid search for the quick-scalp strategy.
Mirrors the logic in src/quick_scalp_loop.py (no MT5 required).
Results saved to backtest_results.csv and printed to console.

Run:  python backtest_quick.py
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
from dataclasses import dataclass, field
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
class FVG:
    top: float
    bottom: float
    direction: int   # 1=bull, -1=bear
    bars_since: int


@dataclass
class Trade:
    direction: str
    entry: float
    sl: float
    tp: float
    open_bar: int
    lot: float = 0.01


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
# Indicators  (faithful copies from quick_scalp_loop.py)
# ---------------------------------------------------------------------------

def _rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) <= period:
        return 50.0
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    recent = changes[-period:]
    ag = sum(max(c, 0.0) for c in recent) / period
    al = sum(abs(min(c, 0.0)) for c in recent) / period
    if al == 0:
        return 100.0 if ag > 0 else 50.0
    return 100.0 - (100.0 / (1.0 + ag / al))


def _sar(candles: List[Candle], step: float = 0.02, maximum: float = 0.2) -> float:
    highs = [c.high for c in candles]
    lows  = [c.low  for c in candles]
    if len(candles) < 2:
        return lows[-1] if lows else 0.0
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
                if highs[i] > ep:
                    ep = highs[i]; acc = min(acc + step, maximum)
        else:
            if highs[i] > sar:
                bullish = True; sar = ep; ep = highs[i]; acc = step
            else:
                sar = max(sar, highs[i - 1], highs[i])
                if lows[i] < ep:
                    ep = lows[i]; acc = min(acc + step, maximum)
    return sar


def _atr(candles: List[Candle]) -> float:
    if len(candles) < 2:
        return 0.0
    trs = []
    prev = candles[0].close
    for c in candles[1:]:
        trs.append(max(c.high - c.low, abs(c.high - prev), abs(c.low - prev)))
        prev = c.close
    return sum(trs) / len(trs) if trs else 0.0


def _detect_fvgs(candles: List[Candle], lookback: int = 20) -> List[FVG]:
    fvgs: List[FVG] = []
    n = len(candles)
    for i in range(n - 2, max(1, n - lookback - 2), -1):
        prev = candles[i - 1]; post = candles[i + 1]
        bs = n - 1 - i
        if post.low > prev.high:
            fvgs.append(FVG(post.low, prev.high, 1, bs))
        elif post.high < prev.low:
            fvgs.append(FVG(prev.low, post.high, -1, bs))
    return fvgs


def _fib_zone(candles: List[Candle]) -> Tuple[Optional[str], str]:
    if len(candles) < 2:
        return None, "unavailable"
    sl_idx = min(range(len(candles)), key=lambda i: candles[i].low)
    sh_idx = max(range(len(candles)), key=lambda i: candles[i].high)
    sl = candles[sl_idx].low; sh = candles[sh_idx].high
    cp = candles[-1].close; rng = sh - sl
    if rng <= 0:
        return None, "flat"
    if sl_idx < sh_idx:
        r382 = sh - rng * 0.382; r618 = sh - rng * 0.618; r786 = sh - rng * 0.786
        zone = ("in_market_mover" if cp >= r382 else
                "towards_market_mover" if cp >= r618 else
                "golden_zone" if cp >= r786 else "outside")
        return Dir.BUY, zone
    else:
        r382 = sl + rng * 0.382; r618 = sl + rng * 0.618; r786 = sl + rng * 0.786
        zone = ("in_market_mover" if cp <= r382 else
                "towards_market_mover" if cp <= r618 else
                "golden_zone" if cp <= r786 else "outside")
        return Dir.SELL, zone


def _fib_allows(fib_dir: Optional[str], zone: str, direction: str) -> bool:
    return fib_dir == direction and zone in {"in_market_mover", "towards_market_mover", "golden_zone"}


def _tick_dir(candles: List[Candle], min_consistency: float) -> Optional[str]:
    if len(candles) < 3:
        return None
    mids = [c.close for c in candles]
    up = sum(1 for a, b in zip(mids, mids[1:]) if b > a)
    dn = sum(1 for a, b in zip(mids, mids[1:]) if b < a)
    tot = up + dn
    if tot == 0:
        return None
    net = mids[-1] - mids[0]
    pip = abs(mids[-1]) * 0.00004
    if net >=  pip and up / tot >= min_consistency:
        return Dir.BUY
    if net <= -pip and dn / tot >= min_consistency:
        return Dir.SELL
    return None


def _body_ratio(c: Candle) -> float:
    rng = c.high - c.low
    return abs(c.close - c.open) / rng if rng > 0 else 0.0


def _ev(sl_dist: float, tp_dist: float, spread: float) -> float:
    return 0.5 * max(tp_dist, 1e-9) - 0.5 * max(sl_dist, 1e-9) - max(spread, 0.0)


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def fetch_ohlcv(symbol: str = "GC=F", hours: int = 96) -> pd.DataFrame:
    print(f"[data] Downloading {hours}h of 1-min data for {symbol}...", flush=True)
    end   = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours + 3)
    df = yf.download(
        symbol, start=start, end=end,
        interval="1m", auto_adjust=True,
        progress=False, threads=False,
    )
    if df.empty:
        raise RuntimeError(f"No data returned for {symbol}.")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    cutoff = end - timedelta(hours=hours)
    df = df[df.index >= cutoff]
    # Flatten multi-level columns (yfinance v0.2+)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    print(f"[data] {len(df)} M1 bars  "
          f"({df.index[0].strftime('%Y-%m-%d %H:%M')} UTC  ->  "
          f"{df.index[-1].strftime('%Y-%m-%d %H:%M')} UTC)", flush=True)
    return df


def df_to_candles(df: pd.DataFrame) -> List[Candle]:
    rows = []
    for ts, row in df.iterrows():
        rows.append(Candle(
            timestamp=ts.to_pydatetime(),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row.get("Volume", 1) or 1),
        ))
    return rows


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

PIP        = 0.10   # 1 pip for XAUUSD
TICK_VALUE = 1.0    # USD per pip per 0.01 lot (approximate)
LOT        = 0.01
SPREAD     = 0.20 * PIP   # simulated spread
WARMUP     = 50
M15_WIN    = 50     # M1 bars used as M15 proxy


def simulate(candles: List[Candle], p: Params) -> dict:
    open_trades: List[Trade] = []
    closed_pnls: List[float] = []
    rejected = 0
    n = len(candles)

    for idx in range(WARMUP, n - 1):
        c      = candles[idx]
        next_c = candles[idx + 1]

        # --- close open trades ---
        still_open: List[Trade] = []
        for t in open_trades:
            pnl: Optional[float] = None
            if t.direction == Dir.BUY:
                if next_c.low  <= t.sl: pnl = (t.sl - t.entry) / PIP * TICK_VALUE
                elif next_c.high >= t.tp: pnl = (t.tp - t.entry) / PIP * TICK_VALUE
                else:
                    mtm = (next_c.close - t.entry) / PIP * TICK_VALUE
                    if mtm >= p.profit_target_usd: pnl = mtm
            else:
                if next_c.high >= t.sl: pnl = (t.entry - t.sl) / PIP * TICK_VALUE
                elif next_c.low  <= t.tp: pnl = (t.entry - t.tp) / PIP * TICK_VALUE
                else:
                    mtm = (t.entry - next_c.close) / PIP * TICK_VALUE
                    if mtm >= p.profit_target_usd: pnl = mtm
            if pnl is not None:
                closed_pnls.append(pnl)
            else:
                still_open.append(t)
        open_trades = still_open

        if len(open_trades) >= p.max_positions:
            continue

        # --- signals ---
        m1_win   = candles[max(0, idx - 29): idx + 1]
        tick_win = candles[max(0, idx - 9):  idx + 1]
        m15_win  = candles[max(0, idx - M15_WIN + 1): idx + 1]

        direction = _tick_dir(tick_win, p.tick_consistency)
        if direction is None:
            continue

        m1_dir = (Dir.BUY if c.close > c.open else
                  Dir.SELL if c.close < c.open else None)

        fib_dir, fib_zone = _fib_zone(m15_win)
        fib_ok  = _fib_allows(fib_dir, fib_zone, direction)

        closes  = [x.close for x in m15_win]
        rsi     = _rsi(closes)
        sar_val = _sar(m15_win)
        sar_dir = (Dir.BUY if c.close > sar_val else
                   Dir.SELL if c.close < sar_val else None)
        sar_ok  = sar_dir == direction
        rsi_ok  = (rsi < 70.0 if direction == Dir.BUY else rsi > 30.0)
        rsi_ext = (rsi >= 85.0 if direction == Dir.BUY else rsi <= 15.0)

        if rsi_ext:
            continue
        if not rsi_ok:
            max_entries = 1; guidance = "structure_override"
        elif fib_ok and sar_ok:
            max_entries = 3; guidance = "full_guidance"
        elif fib_ok or sar_ok:
            max_entries = 1; guidance = "mixed_guidance"
        else:
            continue

        if m1_dir is not None and m1_dir != direction:
            max_entries = min(max_entries, 1)

        if guidance == "mixed_guidance":
            fvgs = _detect_fvgs(m15_win, lookback=20)
            dv   = 1 if direction == Dir.BUY else -1
            if not any(f.direction == dv and f.bars_since <= 8 for f in fvgs):
                continue
            if _body_ratio(c) < p.min_body_ratio:
                continue

        # spread gate
        if SPREAD / PIP > p.max_spread_pips:
            rejected += 1; continue

        # ATR sizing
        atr = _atr(m1_win[-14:]) if len(m1_win) >= 2 else 0.0
        if atr <= 0:
            continue
        sl_dist = atr * p.atr_sl_mult
        tp_dist = atr * p.atr_tp_mult

        if _ev(sl_dist, tp_dist, SPREAD) <= 0:
            rejected += 1; continue

        # open trade at next bar open
        entry = next_c.open
        if direction == Dir.BUY:
            sl = entry - sl_dist; tp = entry + tp_dist
        else:
            sl = entry + sl_dist; tp = entry - tp_dist

        new_cnt = 0
        while len(open_trades) < p.max_positions and new_cnt < max_entries:
            open_trades.append(Trade(direction, entry, sl, tp, idx, LOT))
            new_cnt += 1

    # force-close remaining at last price
    if candles:
        last = candles[-1]
        for t in open_trades:
            pnl = ((last.close - t.entry) if t.direction == Dir.BUY
                   else (t.entry - last.close)) / PIP * TICK_VALUE
            closed_pnls.append(pnl)

    total = len(closed_pnls)
    if total == 0:
        return dict(trades=0, net_pnl=0.0, win_rate=0.0,
                    sharpe=0.0, max_dd=0.0, avg_pnl=0.0, rejected=rejected)

    wins = sum(1 for x in closed_pnls if x > 0)
    net  = sum(closed_pnls)
    avg  = net / total
    std  = float(np.std(closed_pnls)) if total > 1 else 1e-9
    sharpe = (avg / std) * math.sqrt(total) if std > 0 else 0.0

    equity  = np.cumsum(closed_pnls)
    run_max = np.maximum.accumulate(equity)
    max_dd  = float(np.max(run_max - equity))

    return dict(
        trades=total,
        net_pnl=round(net, 4),
        win_rate=round(wins / total * 100, 1),
        sharpe=round(sharpe, 4),
        max_dd=round(max_dd, 4),
        avg_pnl=round(avg, 4),
        rejected=rejected,
    )


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

GRID = {
    "atr_sl_mult":       [0.5, 0.8, 1.2],
    "atr_tp_mult":       [1.5, 2.0, 2.5, 3.0],
    "max_spread_pips":   [2.0, 3.5, 5.0],
    "min_body_ratio":    [0.35, 0.50, 0.65],
    "tick_consistency":  [0.55, 0.60, 0.70],
    "max_positions":     [1, 2, 3],
    "profit_target_usd": [0.10, 0.20, 0.50],
}


import multiprocessing

def _run_single_sim(args):
    keys, combo, candles = args
    p = Params(**dict(zip(keys, combo)))
    m = simulate(candles, p)
    return {**dict(zip(keys, combo)), **m}

def run_grid_search(candles: List[Candle]) -> pd.DataFrame:
    keys   = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    total  = len(combos)
    print(f"\n[grid] {total} combos to test using {multiprocessing.cpu_count()} cores...\n", flush=True)
    
    args = [(keys, combo, candles) for combo in combos]
    
    rows = []
    with multiprocessing.Pool() as pool:
        for i, row in enumerate(pool.imap_unordered(_run_single_sim, args), 1):
            rows.append(row)
            if i == 1 or i % 100 == 0 or i == total:
                print(f"  [{i}/{total}]  {i/total*100:.0f}%", flush=True)
                
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

SEP = "=" * 130


def display_results(df: pd.DataFrame, top_n: int = 15) -> pd.Series:
    filt = df[(df["trades"] >= 10) & (df["net_pnl"] > 0)].copy()
    if filt.empty:
        print("\n[!] No combos met >=10 trades + positive PnL. Showing top by PnL.\n")
        filt = df.nlargest(top_n, "net_pnl").copy()

    ranked = filt.sort_values(["sharpe", "net_pnl"], ascending=False).head(top_n)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)

    print(f"\n{SEP}")
    print(f"  TOP {top_n} WINNING COMBOS  (ranked: Sharpe > Net P&L)")
    print(SEP)
    print(ranked.to_string(index=False))
    print(SEP)

    best = ranked.iloc[0]
    print(f"\n*** BEST COMBO ***")
    print(f"  atr_sl_mult      = {best['atr_sl_mult']}")
    print(f"  atr_tp_mult      = {best['atr_tp_mult']}")
    print(f"  max_spread_pips  = {best['max_spread_pips']}")
    print(f"  min_body_ratio   = {best['min_body_ratio']}")
    print(f"  tick_consistency = {best['tick_consistency']}")
    print(f"  max_positions    = {int(best['max_positions'])}")
    print(f"  profit_target    = {best['profit_target_usd']}")
    print(f"\n  Trades    : {int(best['trades'])}")
    print(f"  Win rate  : {best['win_rate']}%")
    print(f"  Net P&L   : ${best['net_pnl']:.4f}")
    print(f"  Sharpe    : {best['sharpe']:.4f}")
    print(f"  Max DD    : ${best['max_dd']:.4f}")
    print(f"  Avg/trade : ${best['avg_pnl']:.4f}\n")
    return best


def suggest_settings(best: pd.Series) -> None:
    print("--- Suggested .env updates ---")
    print(f"QUICK_MAX_POSITIONS={int(best['max_positions'])}")
    print(f"QUICK_PROFIT_TARGET={best['profit_target_usd']}")
    print(f"# atr_sl_mult={best['atr_sl_mult']}  atr_tp_mult={best['atr_tp_mult']}")
    print(f"# max_spread_pips={best['max_spread_pips']}  min_body_ratio={best['min_body_ratio']}")
    print(f"# tick_consistency={best['tick_consistency']}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df_raw   = fetch_ohlcv("GC=F", hours=96)
    candles  = df_to_candles(df_raw)
    print(f"[data] {len(candles)} candles ready.\n", flush=True)

    results  = run_grid_search(candles)

    csv_path = "backtest_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"\n[output] Full results -> {csv_path}\n", flush=True)

    best = display_results(results)
    suggest_settings(best)
