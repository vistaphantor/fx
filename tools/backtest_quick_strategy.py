from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import sys
from types import SimpleNamespace

import MetaTrader5 as mt5
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.quick_scalp_loop import run_quick_scalp_loop
from src.mt5_client import Mt5Session

UTC = timezone.utc

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the quick.py strategy path on MT5 historical ticks.")
    parser.add_argument("--env", default=".env", help="Path to the .env file")
    parser.add_argument("--symbol", default=None, help="Override trading symbol")
    parser.add_argument("--hours", type=float, default=2.0, help="Backtest duration in hours when --start is omitted")
    parser.add_argument("--start", default=None, help="UTC ISO timestamp, e.g. 2026-05-07T00:00:00+00:00")
    parser.add_argument("--end", default=None, help="UTC ISO timestamp, defaults to latest available tick")
    parser.add_argument("--initial-equity", type=float, default=5000.0, help="Starting account equity")
    return parser.parse_args()

def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Timestamp must be timezone-aware: {value}")
    return parsed.astimezone(UTC)

def _rate_to_dict(rate) -> dict[str, float]:
    return {
        "time": int(rate["time"]),
        "open": float(rate["open"]),
        "high": float(rate["high"]),
        "low": float(rate["low"]),
        "close": float(rate["close"]),
        "tick_volume": int(rate["tick_volume"]),
        "spread": int(rate["spread"]) if "spread" in rate.dtype.names else 0,
        "real_volume": int(rate["real_volume"]) if "real_volume" in rate.dtype.names else 0,
    }

def _fetch_rates(symbol: str, timeframe_value: int, start: datetime, end: datetime) -> list[dict[str, float]]:
    chunk_days = 10
    cursor_end = end
    collected: list[dict[str, float]] = []
    seen_times: set[int] = set()

    while cursor_end > start:
        cursor_start = max(start, cursor_end - timedelta(days=chunk_days))
        rates = mt5.copy_rates_range(symbol, timeframe_value, cursor_start, cursor_end)
        if rates is None or len(rates) == 0:
            break
        chunk = [_rate_to_dict(rate) for rate in rates]
        new_rows = 0
        for row in reversed(chunk):
            stamp = int(row["time"])
            if stamp in seen_times:
                continue
            seen_times.add(stamp)
            collected.append(row)
            new_rows += 1
        if new_rows == 0:
            break
        earliest_chunk_time = min(int(row["time"]) for row in chunk)
        cursor_end = datetime.fromtimestamp(earliest_chunk_time, tz=UTC) - timedelta(seconds=1)

    collected.sort(key=lambda row: int(row["time"]))
    return collected

def _tick_to_dict(tick) -> dict[str, float]:
    return {
        "time": int(tick["time"]),
        "time_msc": int(tick["time_msc"]),
        "bid": float(tick["bid"]),
        "ask": float(tick["ask"]),
        "last": float(tick["last"]),
        "volume": float(tick["volume"]),
        "flags": int(tick["flags"]),
    }

def _fetch_ticks(symbol: str, start: datetime, end: datetime) -> list[dict[str, float]]:
    # Download ticks in 1-day chunks
    cursor_start = start
    collected = []
    seen_msc: set[int] = set()

    while cursor_start < end:
        cursor_end = min(end, cursor_start + timedelta(days=1))
        ticks = mt5.copy_ticks_range(symbol, cursor_start, cursor_end, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            cursor_start = cursor_end
            continue
        chunk = [_tick_to_dict(t) for t in ticks]
        for row in chunk:
            msc = row["time_msc"]
            if msc in seen_msc:
                continue
            seen_msc.add(msc)
            collected.append(row)
        cursor_start = cursor_end

    collected.sort(key=lambda row: row["time_msc"])
    return collected


def _latest_available_candle_time(symbol: str) -> datetime | None:
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 1, 1)
    if rates is None or len(rates) == 0:
        return None
    return datetime.fromtimestamp(int(rates[-1]["time"]), tz=UTC)


def _session_bucket(timestamp: datetime) -> str:
    hour = timestamp.astimezone(UTC).hour
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 13:
        return "london"
    if 13 <= hour < 20:
        return "new_york"
    return "late_us"


def _summarize_group(trades: list[ClosedTrade]) -> dict[str, object]:
    wins = sum(1 for trade in trades if trade.profit > 0)
    losses = len(trades) - wins
    gross_profit = sum(trade.profit for trade in trades if trade.profit > 0)
    gross_loss = sum(abs(trade.profit) for trade in trades if trade.profit <= 0)
    sl_count = sum(1 for trade in trades if trade.reason == "sl")
    tick_turn_count = sum(1 for trade in trades if trade.reason == "quick-scalp-tick-turn-profit-exit")
    tp_count = sum(1 for trade in trades if trade.reason == "tp")
    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "net": gross_profit - gross_loss,
        "win_rate": (wins / len(trades) * 100.0) if trades else 0.0,
        "sl": sl_count,
        "tick_turn": tick_turn_count,
        "tp": tp_count,
    }


def _print_breakdowns(closed_trades: list[ClosedTrade]) -> None:
    by_day: dict[str, list[ClosedTrade]] = {}
    by_session: dict[str, list[ClosedTrade]] = {}

    for trade in closed_trades:
        day_key = trade.closed_at.astimezone(UTC).strftime("%Y-%m-%d")
        by_day.setdefault(day_key, []).append(trade)
        session_key = _session_bucket(trade.closed_at)
        by_session.setdefault(session_key, []).append(trade)

    print("\nby_day:")
    for day_key in sorted(by_day):
        summary = _summarize_group(by_day[day_key])
        print(
            f"  {day_key} trades={summary['trades']} net={summary['net']:.2f} "
            f"win_rate={summary['win_rate']:.2f}% sl={summary['sl']} "
            f"tick_turn={summary['tick_turn']} tp={summary['tp']}"
        )

    print("\nby_session:")
    for session_key in ("asia", "london", "new_york", "late_us"):
        trades = by_session.get(session_key, [])
        summary = _summarize_group(trades)
        print(
            f"  {session_key} trades={summary['trades']} net={summary['net']:.2f} "
            f"win_rate={summary['win_rate']:.2f}% sl={summary['sl']} "
            f"tick_turn={summary['tick_turn']} tp={summary['tp']}"
        )

@dataclass
class ClosedTrade:
    ticket: int
    direction: str
    volume: float
    entry_price: float
    exit_price: float
    profit: float
    opened_at: datetime
    closed_at: datetime
    reason: str

class ReplayQuickMt5Module:
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    COPY_TICKS_ALL = mt5.COPY_TICKS_ALL
    TIMEFRAME_M1 = mt5.TIMEFRAME_M1
    TIMEFRAME_M5 = mt5.TIMEFRAME_M5
    TIMEFRAME_M15 = mt5.TIMEFRAME_M15

    def __init__(
        self,
        *,
        symbol: str,
        ticks: list[dict[str, float]],
        m1_rates: list[dict[str, float]],
        m15_rates: list[dict[str, float]],
        point: float,
        initial_equity: float,
    ) -> None:
        self.symbol = symbol
        self.ticks = ticks
        self.m1_rates = m1_rates
        self.m15_rates = m15_rates
        self.point = point
        self.current_tick_index = 0
        self.equity = float(initial_equity)
        self.margin_free = self.equity

    def account_info(self):
        return SimpleNamespace(equity=self.equity, margin_free=self.margin_free, free_margin=self.margin_free)

    def symbol_info(self, symbol: str):
        if symbol != self.symbol:
            return None
        info = mt5.symbol_info(symbol)
        return info

    def symbol_info_tick(self, symbol: str):
        if symbol != self.symbol:
            return None
        return SimpleNamespace(**self.ticks[self.current_tick_index])

    def order_calc_margin(self, order_type, symbol: str, volume: float, price: float):
        return mt5.order_calc_margin(order_type, symbol, volume, price)

    def copy_rates_from_pos(self, symbol: str, timeframe_value, start_pos: int, count: int):
        if symbol != self.symbol:
            return []
        current_time = self.ticks[self.current_tick_index]["time"]
        if timeframe_value == mt5.TIMEFRAME_M1:
            completed = [rate for rate in self.m1_rates if int(rate["time"]) <= current_time]
            return completed[-count:] if completed else []
        elif timeframe_value == mt5.TIMEFRAME_M15:
            completed = [rate for rate in self.m15_rates if int(rate["time"]) <= current_time]
            return completed[-count:] if completed else []
        return []
        
    def copy_ticks_from_pos(self, symbol: str, start_pos: int, count: int, flags: int):
        if symbol != self.symbol:
            return []
        start_idx = max(0, self.current_tick_index - count + 1)
        # return dicts that act like ticks
        chunk = self.ticks[start_idx : self.current_tick_index + 1]
        return chunk

class ReplayQuickExecutor:
    def __init__(self, *, mt5_module: ReplayQuickMt5Module, symbol: str, contract_size: float) -> None:
        self.mt5_module = mt5_module
        self.symbol = symbol
        self.contract_size = float(contract_size)
        self.next_ticket = 1
        self.open_positions: list[SimpleNamespace] = []
        self.closed_trades: list[ClosedTrade] = []
        self.open_reasons = Counter()
        self.close_reasons = Counter()

    def list_bot_positions(self, symbol, comment_prefix=""):
        return [
            position
            for position in self.open_positions
            if position.symbol == symbol and str(getattr(position, "comment", "")).startswith(comment_prefix)
        ]

    def open_strategy_trade(self, symbol, direction, lot, stop_loss, take_profit, comment):
        tick = self.mt5_module.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No replay tick available for {symbol}")
        order_type = self.mt5_module.ORDER_TYPE_BUY if direction.value == "BULLISH" else self.mt5_module.ORDER_TYPE_SELL
        fill_price = float(tick.ask if order_type == self.mt5_module.ORDER_TYPE_BUY else tick.bid)
        position = SimpleNamespace(
            ticket=self.next_ticket,
            symbol=symbol,
            type=order_type,
            volume=float(lot),
            price_open=fill_price,
            entry_price=fill_price,
            sl=float(stop_loss),
            stop_loss=float(stop_loss),
            tp=float(take_profit),
            comment=comment,
            initial_stop_loss=float(stop_loss),
            opened_at=self.current_time,
            profit=0.0
        )
        self.next_ticket += 1
        self.open_positions.append(position)
        self.open_reasons["entry"] += 1
        # subtract margin
        margin = self.mt5_module.order_calc_margin(order_type, symbol, float(lot), fill_price)
        if margin:
            self.mt5_module.margin_free -= float(margin)
        return position

    def close_position(self, position, comment):
        tick = self.mt5_module.symbol_info_tick(position.symbol)
        if tick is None:
            raise RuntimeError(f"No replay tick available for {position.symbol}")
        exit_price = float(tick.bid if int(position.type) == self.mt5_module.ORDER_TYPE_BUY else tick.ask)
        self._finalize_position(position, exit_price=exit_price, reason=comment, closed_at=self.current_time)
        return SimpleNamespace(retcode=0)

    @property
    def current_time(self) -> datetime:
        return datetime.fromtimestamp(
            self.mt5_module.ticks[self.mt5_module.current_tick_index]["time"],
            tz=UTC,
        )

    def update_profits(self):
        tick = self.mt5_module.symbol_info_tick(self.symbol)
        if not tick:
            return
        for position in self.open_positions:
            exit_price = float(tick.bid if int(position.type) == self.mt5_module.ORDER_TYPE_BUY else tick.ask)
            position.profit = self._profit_for_position(position, exit_price)
            
    def apply_tick_exits(self):
        tick = self.mt5_module.symbol_info_tick(self.symbol)
        if not tick:
            return
        survivors: list[SimpleNamespace] = []
        for position in self.open_positions:
            exit_price = float(tick.bid if int(position.type) == self.mt5_module.ORDER_TYPE_BUY else tick.ask)
            reason = self._tick_exit(position, exit_price)
            if reason is None:
                survivors.append(position)
                continue
            self._finalize_position(position, exit_price=exit_price, reason=reason, closed_at=self.current_time)
        self.open_positions = survivors

    def _tick_exit(self, position, exit_price: float) -> str | None:
        stop_loss = float(getattr(position, "sl", 0.0) or 0.0)
        take_profit = float(getattr(position, "tp", 0.0) or 0.0)
        is_buy = int(position.type) == self.mt5_module.ORDER_TYPE_BUY

        if stop_loss > 0.0:
            if is_buy and exit_price <= stop_loss:
                return "sl"
            if not is_buy and exit_price >= stop_loss:
                return "sl"
                
        if take_profit > 0.0:
            if is_buy and exit_price >= take_profit:
                return "tp"
            if not is_buy and exit_price <= take_profit:
                return "tp"
        return None

    def _finalize_position(self, position, *, exit_price: float, reason: str, closed_at: datetime) -> None:
        profit = self._profit_for_position(position, exit_price)
        self.mt5_module.equity += profit
        
        # restore margin (approximate)
        margin = self.mt5_module.order_calc_margin(position.type, position.symbol, float(position.volume), position.price_open)
        if margin:
            self.mt5_module.margin_free += float(margin) + profit
            
        self.close_reasons[reason] += 1
        self.closed_trades.append(
            ClosedTrade(
                ticket=int(position.ticket),
                direction="BULLISH" if int(position.type) == self.mt5_module.ORDER_TYPE_BUY else "BEARISH",
                volume=float(position.volume),
                entry_price=float(position.price_open),
                exit_price=float(exit_price),
                profit=profit,
                opened_at=position.opened_at,
                closed_at=closed_at,
                reason=reason,
            )
        )

    def _profit_for_position(self, position, exit_price: float) -> float:
        points = (
            float(exit_price) - float(position.price_open)
            if int(position.type) == self.mt5_module.ORDER_TYPE_BUY
            else float(position.price_open) - float(exit_price)
        )
        return points * self.contract_size * float(position.volume)

def main() -> int:
    args = parse_args()
    settings = load_settings(args.env)
    symbol = str(args.symbol or settings.trading_symbol).strip()

    print(f"Connecting to MT5 at {settings.mt5_terminal_path} to fetch tick data...")
    if not mt5.initialize(settings.mt5_terminal_path):
        print(f"Failed to initialize MT5: {mt5.last_error()}")
        return 1
    if not mt5.login(login=settings.hfm_login, password=settings.hfm_password, server=settings.hfm_server):
        print(f"Failed to log into MT5: {mt5.last_error()}")
        return 1

    if not mt5.symbol_select(symbol, True):
        print(f"Unable to select symbol {symbol}")
        return 1

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbol {symbol} not available")
        return 1
        
    point = float(symbol_info.point)
    contract_size = float(symbol_info.trade_contract_size)

    requested_end = _parse_timestamp(args.end) or datetime.now(UTC)
    latest_candle_time = _latest_available_candle_time(symbol)
    end_time = requested_end
    if args.end is None and latest_candle_time is not None and latest_candle_time < requested_end:
        end_time = latest_candle_time
    start_time = _parse_timestamp(args.start) or (end_time - timedelta(hours=args.hours))

    print(f"Fetching M1 and M15 candles from {start_time.isoformat()} to {end_time.isoformat()}...")
    m1_rates = _fetch_rates(symbol, mt5.TIMEFRAME_M1, start_time - timedelta(days=2), end_time)
    m15_rates = _fetch_rates(symbol, mt5.TIMEFRAME_M15, start_time - timedelta(days=14), end_time)
    
    print(f"Fetching tick data... (this might take a moment)")
    ticks = _fetch_ticks(symbol, start_time, end_time)
    
    mt5.shutdown()

    if not ticks:
        print("No tick data available for the requested period.")
        return 1
        
    print(f"Fetched {len(ticks)} ticks.")

    replay_mt5 = ReplayQuickMt5Module(
        symbol=symbol,
        ticks=ticks,
        m1_rates=m1_rates,
        m15_rates=m15_rates,
        point=point,
        initial_equity=args.initial_equity,
    )
    executor = ReplayQuickExecutor(
        mt5_module=replay_mt5,
        symbol=symbol,
        contract_size=contract_size,
    )

    def sleep_fn(seconds: int):
        # In a tick replayer, "sleep" means fast-forwarding ticks.
        # But wait, the loop sleeps. If it sleeps 1 second, we should fast-forward ticks by 1 second.
        target_time = replay_mt5.ticks[replay_mt5.current_tick_index]["time"] + seconds
        
        while replay_mt5.current_tick_index < len(replay_mt5.ticks) - 1:
            next_idx = replay_mt5.current_tick_index + 1
            replay_mt5.current_tick_index = next_idx
            executor.apply_tick_exits()
            executor.update_profits()
            if replay_mt5.ticks[next_idx]["time"] >= target_time:
                break
                
    def reload_check_fn():
        # Stop loop if we reach the end of ticks
        return replay_mt5.current_tick_index >= len(replay_mt5.ticks) - 1

    # Apply defaults
    from quick import build_quick_runtime_settings
    q_settings = build_quick_runtime_settings(settings)

    print("\nStarting Quick Scalper Backtest...")
    print(f"window={start_time.isoformat()} -> {end_time.isoformat()}")
    if end_time != requested_end:
        print(f"requested_end={requested_end.isoformat()} latest_active_end={end_time.isoformat()}")
    print(f"initial_equity={args.initial_equity:.2f}")

    run_quick_scalp_loop(
        mt5_module=replay_mt5,
        executor=executor,
        symbol=symbol,
        lot=q_settings.quick_trade_lot,
        max_positions=q_settings.quick_max_positions,
        profit_target=q_settings.quick_profit_target,
        poll_seconds=q_settings.quick_poll_seconds,
        max_loss=q_settings.quick_max_loss,
        min_free_margin=q_settings.quick_min_free_margin,
        max_loops=len(ticks), # Fallback, actual exit is in reload_check_fn
        reload_check_fn=reload_check_fn,
        sleep_fn=sleep_fn,
        log_fn=lambda msg: None, # Silence per-tick logs for speed
    )

    print("\n--- RESULTS ---")
    closed = executor.closed_trades
    wins = [t for t in closed if t.profit > 0]
    losses = [t for t in closed if t.profit <= 0]
    
    gross_profit = sum(t.profit for t in wins)
    gross_loss = sum(abs(t.profit) for t in losses)
    net_profit = gross_profit - gross_loss
    
    print(f"closed_trades={len(closed)}")
    print(f"wins={len(wins)} losses={len(losses)} win_rate={(len(wins)/len(closed)*100) if closed else 0:.2f}%")
    print(f"gross_profit={gross_profit:.2f}")
    print(f"gross_loss={gross_loss:.2f}")
    print(f"net_closed={net_profit:.2f}")
    print(f"exit_reasons={dict(executor.close_reasons)}")
    _print_breakdowns(closed)
    return 0

if __name__ == "__main__":
    sys.exit(main())
