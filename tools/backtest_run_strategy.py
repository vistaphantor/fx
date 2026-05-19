from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

import MetaTrader5 as mt5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.live_trade_loop import run_live_signal_loop
from src.market_data import _build_supply_zone, timeframe_label
from src.mt5_client import Mt5Session


UTC = timezone.utc
TF_ATTRS = (
    "TIMEFRAME_D1",
    "TIMEFRAME_H4",
    "TIMEFRAME_H1",
    "TIMEFRAME_M30",
    "TIMEFRAME_M15",
    "TIMEFRAME_M10",
    "TIMEFRAME_M5",
)
LOOKBACK_COUNTS = {
    "TIMEFRAME_D1": 6,
    "TIMEFRAME_H4": 6,
    "TIMEFRAME_H1": 5,
    "TIMEFRAME_M30": 5,
    "TIMEFRAME_M15": 50,
    "TIMEFRAME_M10": 8,
    "TIMEFRAME_M5": 30,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the standard run.py strategy path on MT5 historical candles.")
    parser.add_argument("--env", default=".env", help="Path to the .env file")
    parser.add_argument("--symbol", default=None, help="Override trading symbol")
    parser.add_argument("--hours", type=float, default=8.0, help="Backtest duration in hours when --start is omitted")
    parser.add_argument("--start", default=None, help="UTC ISO timestamp, e.g. 2026-05-07T00:00:00+00:00")
    parser.add_argument("--end", default=None, help="UTC ISO timestamp, defaults to latest completed M5 candle")
    parser.add_argument("--initial-equity", type=float, default=500000.0, help="Starting account equity")
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


def _fetch_rates(symbol: str, timeframe_attr: str, start: datetime, end: datetime) -> list[dict[str, float]]:
    timeframe_value = getattr(mt5, timeframe_attr)
    chunk_days = {
        "TIMEFRAME_M5": 14,
        "TIMEFRAME_M10": 30,
        "TIMEFRAME_M15": 45,
        "TIMEFRAME_M30": 90,
        "TIMEFRAME_H1": 120,
        "TIMEFRAME_H4": 365,
        "TIMEFRAME_D1": 730,
    }.get(timeframe_attr, 30)
    cursor_end = end
    collected: list[dict[str, float]] = []
    seen_times: set[int] = set()

    while cursor_end > start:
        cursor_start = max(start, cursor_end - timedelta(days=chunk_days))
        rates = mt5.copy_rates_range(symbol, timeframe_value, cursor_start, cursor_end)
        if rates is None or len(rates) == 0:
            if collected:
                break
            raise RuntimeError(f"No historical data returned for {symbol} {timeframe_attr}")
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

    if not collected:
        raise RuntimeError(f"No historical data returned for {symbol} {timeframe_attr}")
    collected.sort(key=lambda row: int(row["time"]))
    return collected


def _completed_m5_end(symbol: str) -> datetime:
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 1, 1)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Unable to resolve latest completed M5 candle for {symbol}")
    return datetime.fromtimestamp(int(rates[-1]["time"]), tz=UTC)


def _earliest_available_m5(symbol: str) -> datetime:
    end = _completed_m5_end(symbol)
    probe_start = end - timedelta(days=14)
    earliest = end
    found_any = False

    for _ in range(24):
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, probe_start, end)
        if rates is None or len(rates) == 0:
            break
        found_any = True
        earliest = datetime.fromtimestamp(int(rates[0]["time"]), tz=UTC)
        if earliest > probe_start + timedelta(hours=12):
            break
        probe_start -= timedelta(days=14)

    if not found_any:
        raise RuntimeError(f"Unable to resolve earliest available M5 candle for {symbol}")
    return earliest


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


class ReplayMt5Module:
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    TIMEFRAME_D1 = mt5.TIMEFRAME_D1
    TIMEFRAME_H4 = mt5.TIMEFRAME_H4
    TIMEFRAME_H1 = mt5.TIMEFRAME_H1
    TIMEFRAME_M30 = mt5.TIMEFRAME_M30
    TIMEFRAME_M15 = mt5.TIMEFRAME_M15
    TIMEFRAME_M10 = mt5.TIMEFRAME_M10
    TIMEFRAME_M5 = mt5.TIMEFRAME_M5

    def __init__(
        self,
        *,
        symbol: str,
        rates_by_timeframe: dict[str, list[dict[str, float]]],
        point: float,
        initial_equity: float,
    ) -> None:
        self.symbol = symbol
        self.rates_by_timeframe = rates_by_timeframe
        self.point = point
        self.current_index = 0
        self.equity = float(initial_equity)

    def set_index(self, index: int) -> None:
        self.current_index = index

    def account_info(self):
        return SimpleNamespace(equity=self.equity)

    def symbol_info(self, symbol: str):
        if symbol != self.symbol:
            return None
        info = mt5.symbol_info(symbol)
        return info

    def symbol_info_tick(self, symbol: str):
        if symbol != self.symbol:
            return None
        bar = self.rates_by_timeframe["TIMEFRAME_M5"][self.current_index]
        spread_points = int(bar.get("spread", 0) or 0)
        spread = spread_points * self.point
        close = float(bar["close"])
        bid = close - (spread / 2.0)
        ask = close + (spread / 2.0)
        return SimpleNamespace(bid=bid, ask=ask)

    def order_calc_margin(self, order_type, symbol: str, volume: float, price: float):
        return mt5.order_calc_margin(order_type, symbol, volume, price)

    def copy_rates_from_pos(self, symbol: str, timeframe_value, start_pos: int, count: int):
        if symbol != self.symbol:
            return []
        timeframe_attr = None
        for candidate in TF_ATTRS:
            if getattr(self, candidate) == timeframe_value:
                timeframe_attr = candidate
                break
        if timeframe_attr is None:
            raise ValueError(f"Unsupported timeframe value: {timeframe_value}")
        current_time = self.rates_by_timeframe["TIMEFRAME_M5"][self.current_index]["time"]
        completed = [rate for rate in self.rates_by_timeframe[timeframe_attr] if int(rate["time"]) <= current_time]
        if not completed:
            return []
        return completed[-count:]


class ReplayExecutor:
    def __init__(self, *, mt5_module: ReplayMt5Module, symbol: str, contract_size: float) -> None:
        self.mt5_module = mt5_module
        self.symbol = symbol
        self.contract_size = float(contract_size)
        self.next_ticket = 1
        self.open_positions: list[SimpleNamespace] = []
        self.closed_trades: list[ClosedTrade] = []
        self.open_reasons = Counter()
        self.close_reasons = Counter()
        self.max_concurrent_positions = 0

    def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
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
        )
        self.next_ticket += 1
        self.open_positions.append(position)
        self.open_reasons["entry" if len(self.open_positions) == 1 else "campaign_add"] += 1
        self.max_concurrent_positions = max(self.max_concurrent_positions, len(self.open_positions))
        return position

    def update_position_stop_loss(self, position, stop_loss, take_profit=None):
        position.sl = float(stop_loss)
        position.stop_loss = float(stop_loss)
        if take_profit is not None:
            position.tp = float(take_profit)

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
            int(self.mt5_module.rates_by_timeframe["TIMEFRAME_M5"][self.mt5_module.current_index]["time"]),
            tz=UTC,
        )

    def apply_bar_exits(self, bar: dict[str, float], closed_at: datetime):
        survivors: list[SimpleNamespace] = []
        for position in self.open_positions:
            exit_price, reason = self._bar_exit(position, bar)
            if reason is None:
                survivors.append(position)
                continue
            self._finalize_position(position, exit_price=exit_price, reason=reason, closed_at=closed_at)
        self.open_positions = survivors

    def mark_to_market(self, close_price: float) -> float:
        return sum(self._profit_for_position(position, close_price) for position in self.open_positions)

    def _bar_exit(self, position, bar: dict[str, float]) -> tuple[float | None, str | None]:
        high = float(bar["high"])
        low = float(bar["low"])
        stop_loss = float(getattr(position, "sl", 0.0) or 0.0)
        take_profit = float(getattr(position, "tp", 0.0) or 0.0)
        is_buy = int(position.type) == self.mt5_module.ORDER_TYPE_BUY

        stop_hit = stop_loss > 0.0 and low <= stop_loss <= high
        tp_hit = take_profit > 0.0 and low <= take_profit <= high

        if stop_hit and tp_hit:
            return stop_loss, "sl"
        if stop_hit:
            return stop_loss, "sl"
        if tp_hit:
            return take_profit, "tp"
        return None, None

    def _finalize_position(self, position, *, exit_price: float, reason: str, closed_at: datetime) -> None:
        profit = self._profit_for_position(position, exit_price)
        self.mt5_module.equity += profit
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


def _build_rates(settings, symbol: str, start: datetime, end: datetime) -> dict[str, list[dict[str, float]]]:
    lookback_start = start - timedelta(days=14)
    rates_by_timeframe = {}
    for timeframe_attr in TF_ATTRS:
        rates_by_timeframe[timeframe_attr] = _fetch_rates(symbol, timeframe_attr, lookback_start, end)
    return rates_by_timeframe


def _eligible_m5_indices(rates_by_timeframe: dict[str, list[dict[str, float]]], start: datetime, end: datetime) -> list[int]:
    m5_rates = rates_by_timeframe["TIMEFRAME_M5"]
    eligible: list[int] = []
    for index, rate in enumerate(m5_rates):
        timestamp = datetime.fromtimestamp(int(rate["time"]), tz=UTC)
        if timestamp < start or timestamp > end:
            continue
        enough_history = True
        current_epoch = int(rate["time"])
        for timeframe_attr, count in LOOKBACK_COUNTS.items():
            completed = [row for row in rates_by_timeframe[timeframe_attr] if int(row["time"]) <= current_epoch]
            minimum = 1 if timeframe_attr == "TIMEFRAME_D1" else count
            if len(completed) < minimum:
                enough_history = False
                break
        if enough_history:
            eligible.append(index)
    if not eligible:
        raise RuntimeError("No eligible M5 bars found for the requested backtest window")
    return eligible


def _build_sleep_fn(executor: ReplayExecutor, replay_mt5: ReplayMt5Module, eligible_indices: list[int]):
    cursor = {"position": 0}

    def sleep_fn(_seconds: int):
        current_pos = cursor["position"]
        if current_pos + 1 >= len(eligible_indices):
            return
        next_index = eligible_indices[current_pos + 1]
        next_bar = replay_mt5.rates_by_timeframe["TIMEFRAME_M5"][next_index]
        closed_at = datetime.fromtimestamp(int(next_bar["time"]), tz=UTC)
        executor.apply_bar_exits(next_bar, closed_at=closed_at)
        replay_mt5.set_index(next_index)
        cursor["position"] = current_pos + 1

    return sleep_fn


def _format_dt(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def main() -> int:
    args = parse_args()
    settings = load_settings(args.env)
    symbol = (args.symbol or settings.trading_symbol).strip().upper()

    session = Mt5Session(
        terminal_path=settings.mt5_terminal_path,
        startup_wait_seconds=settings.mt5_startup_wait_seconds,
        mt5_module=mt5,
    )
    session.launch_terminal()
    session.initialize_and_login(
        login=settings.hfm_login,
        password=settings.hfm_password,
        server=settings.hfm_server,
    )

    try:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Unable to select symbol for replay: {symbol}")
        end = _parse_timestamp(args.end) or _completed_m5_end(symbol)
        requested_start = _parse_timestamp(args.start) or (end - timedelta(hours=float(args.hours)))
        earliest_available = _earliest_available_m5(symbol)
        start = max(requested_start, earliest_available)
        rates_by_timeframe = _build_rates(settings, symbol, start, end)
        eligible_indices = _eligible_m5_indices(rates_by_timeframe, start, end)

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise RuntimeError(f"Unable to load symbol info for {symbol}")
        point = float(getattr(symbol_info, "point", 0.01) or 0.01)
        contract_size = float(getattr(symbol_info, "trade_contract_size", 100.0) or 100.0)

        replay_mt5 = ReplayMt5Module(
            symbol=symbol,
            rates_by_timeframe=rates_by_timeframe,
            point=point,
            initial_equity=float(args.initial_equity),
        )
        replay_mt5.set_index(eligible_indices[0])
        executor = ReplayExecutor(
            mt5_module=replay_mt5,
            symbol=symbol,
            contract_size=contract_size,
        )
        sleep_fn = _build_sleep_fn(executor, replay_mt5, eligible_indices)
        run_live_signal_loop(
            mt5_module=replay_mt5,
            executor=executor,
            symbol=symbol,
            lot=settings.default_trade_lot,
            add_on_lot_increment=settings.add_on_lot_increment,
            campaign_max_exposure_pct=settings.campaign_max_exposure_pct,
            risk_buffer=settings.risk_buffer,
            max_candles_since_breakout=settings.max_candles_since_breakout,
            poll_seconds=settings.loop_poll_seconds,
            max_loops=len(eligible_indices),
            tradingview_alert_store=None,
            strategy_profile=settings.strategy_profiles.get(symbol),
            reload_check_fn=None,
            sleep_fn=sleep_fn,
            log_fn=lambda _message: None,
            settings=settings,
        )

        final_bar = replay_mt5.rates_by_timeframe["TIMEFRAME_M5"][eligible_indices[-1]]
        final_close = float(final_bar["close"])
        open_unrealized = executor.mark_to_market(final_close)

        equity_curve = [float(args.initial_equity)]
        running_equity = float(args.initial_equity)
        max_equity = running_equity
        max_drawdown = 0.0
        for trade in executor.closed_trades:
            running_equity += trade.profit
            equity_curve.append(running_equity)
            max_equity = max(max_equity, running_equity)
            if max_equity > 0:
                max_drawdown = max(max_drawdown, max_equity - running_equity)

        wins = sum(1 for trade in executor.closed_trades if trade.profit > 0)
        losses = sum(1 for trade in executor.closed_trades if trade.profit < 0)
        gross_profit = sum(trade.profit for trade in executor.closed_trades if trade.profit > 0)
        gross_loss = -sum(trade.profit for trade in executor.closed_trades if trade.profit < 0)
        net_closed = gross_profit - gross_loss
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        print(
            "\n".join(
                [
                    f"STANDARD RUN.PY REPLAY {symbol}",
                    f"window={_format_dt(start)} -> {_format_dt(end)}",
                    f"requested_start={_format_dt(requested_start)}",
                    f"earliest_m5_available={_format_dt(earliest_available)}",
                    f"loops={len(eligible_indices)}",
                    f"initial_equity={float(args.initial_equity):.2f}",
                    f"closed_trades={len(executor.closed_trades)} open_positions={len(executor.open_positions)}",
                    f"wins={wins} losses={losses} win_rate={(wins / len(executor.closed_trades) * 100.0) if executor.closed_trades else 0.0:.2f}",
                    f"gross_profit={gross_profit:.2f}",
                    f"gross_loss={gross_loss:.2f}",
                    f"net_closed={net_closed:.2f}",
                    f"open_unrealized={open_unrealized:.2f}",
                    f"net_with_open={net_closed + open_unrealized:.2f}",
                    f"profit_factor={profit_factor:.3f}",
                    f"max_closed_drawdown={max_drawdown:.2f}",
                    f"max_concurrent_positions={executor.max_concurrent_positions}",
                    f"entry_breakdown={dict(executor.open_reasons)}",
                    f"exit_breakdown={dict(executor.close_reasons)}",
                ]
            )
        )
        return 0
    finally:
        session.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
