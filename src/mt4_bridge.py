from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import time
import uuid
from types import SimpleNamespace


BRIDGE_EA_NAME = "FxPythonBridge.mq4"


@dataclass(frozen=True)
class Mt4BridgeInstall:
    source: Path
    destination: Path
    common_files_dir: Path


@dataclass(frozen=True)
class Mt4BridgeResult:
    command_id: str
    status: str
    message: str
    ticket: int
    error_code: int


class Mt4FileBridgeClient:
    def __init__(self, common_files_dir: str | Path):
        self.common_files_dir = Path(common_files_dir)
        self.command_file = self.common_files_dir / "fx_bridge_commands.csv"
        self.result_file = self.common_files_dir / "fx_bridge_results.csv"
        self.heartbeat_file = self.common_files_dir / "fx_bridge_heartbeat.csv"

    def is_alive(self, max_age_seconds: float = 5.0) -> bool:
        if not self.heartbeat_file.exists():
            return False
        return time.time() - self.heartbeat_file.stat().st_mtime <= max_age_seconds

    def open_order(self, side: str, symbol: str, lot: float, stop_loss: float = 0.0, take_profit: float = 0.0, comment: str = "fx-python"):
        return self._send_command(side.upper(), symbol, lot, stop_loss, take_profit, comment)

    def close_symbol_orders(self, symbol: str, comment: str = "fx-python-close", ticket: int | None = None):
        # The MT4 EA uses the numeric `lot` command column as an optional
        # ticket selector for CLOSE/MODIFY commands. 0 preserves the legacy
        # "all symbol orders" behavior.
        return self._send_command("CLOSE", symbol, float(ticket or 0), 0.0, 0.0, comment)

    def modify_symbol_orders(
        self,
        symbol: str,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        comment: str = "fx-python-modify",
        ticket: int | None = None,
    ):
        return self._send_command("MODIFY", symbol, float(ticket or 0), stop_loss, take_profit, comment)

    def _send_command(self, action: str, symbol: str, lot: float, stop_loss: float, take_profit: float, comment: str):
        self.common_files_dir.mkdir(parents=True, exist_ok=True)
        command_id = uuid.uuid4().hex
        self._write_text_with_retry(
            self.command_file,
            f"{command_id},{action},{symbol},{float(lot)},{float(stop_loss)},{float(take_profit)},{comment}\n",
        )
        return command_id

    def _write_text_with_retry(self, path: Path, text: str, *, attempts: int = 8, delay_seconds: float = 0.05) -> None:
        last_error: PermissionError | None = None
        for attempt in range(max(1, int(attempts))):
            try:
                path.write_text(text, encoding="utf-8")
                return
            except PermissionError as exc:
                last_error = exc
                if attempt == attempts - 1:
                    break
                time.sleep(delay_seconds)
        if last_error is not None:
            raise last_error

    def read_result(self) -> Mt4BridgeResult | None:
        if not self.result_file.exists():
            return None
        text = ""
        for attempt in range(5):
            try:
                text = self.result_file.read_text(encoding="utf-8", errors="replace").strip()
                break
            except PermissionError:
                if attempt == 4:
                    return None
                time.sleep(0.05)
        parts = text.split(",")
        if len(parts) < 5:
            return None
        return Mt4BridgeResult(
            command_id=parts[0],
            status=parts[1],
            message=parts[2],
            ticket=int(float(parts[3] or 0)),
            error_code=int(float(parts[4] or 0)),
        )

    def wait_for_result(self, command_id: str, timeout_seconds: float = 15.0) -> Mt4BridgeResult | None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            result = self.read_result()
            if result is not None and result.command_id == command_id:
                return result
            time.sleep(0.25)
        return None


class Mt4BridgeModule:
    TIMEFRAME_D1 = "D1"
    TIMEFRAME_H4 = "H4"
    TIMEFRAME_H1 = "H1"
    TIMEFRAME_M30 = "M30"
    TIMEFRAME_M15 = "M15"
    TIMEFRAME_M10 = "M10"
    TIMEFRAME_M5 = "M5"
    TIMEFRAME_M1 = "M1"

    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 10
    TRADE_ACTION_SLTP = 11
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    SYMBOL_TRADE_EXECUTION_MARKET = 2
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_INVALID_FILL = 10030
    COPY_TICKS_ALL = 0
    COPY_TICKS_INFO = 1

    def __init__(self, common_files_dir: str | Path):
        self.common_files_dir = Path(common_files_dir)
        self.client = Mt4FileBridgeClient(self.common_files_dir)
        self._last_error = (0, "")
        self._tick_history = defaultdict(lambda: deque(maxlen=500))

    def initialize(self):
        return True

    def login(self, login, password=None, server=None):
        return True

    def shutdown(self):
        return None

    def last_error(self):
        return self._last_error

    def symbol_select(self, symbol, enable=True):
        return True

    def account_info(self):
        row = self._read_first_row(self.common_files_dir / "fx_bridge_heartbeat.csv")
        if len(row) < 6:
            return None
        return SimpleNamespace(
            login=int(float(row[1] or 0)),
            balance=float(row[2] or 0.0),
            equity=float(row[3] or 0.0),
            margin_free=float(row[4] or 0.0),
            free_margin=float(row[4] or 0.0),
            profit=0.0,
            currency="",
            trade_allowed=str(row[5]).strip().lower() in {"1", "true", "yes", "on"},
        )

    def symbol_info_tick(self, symbol):
        tick = self._read_latest_tick_file(symbol)
        if tick is None:
            return None
        self._remember_tick(symbol, tick)
        return tick

    def symbol_info(self, symbol):
        row = self._read_first_row(self._symbol_file("fx_bridge_tick", symbol, "csv"))
        point = float(row[3]) if len(row) > 3 and row[3] else 0.01
        digits = int(float(row[4])) if len(row) > 4 and row[4] else 2
        volume_min = float(row[5]) if len(row) > 5 and row[5] else 0.01
        volume_max = float(row[6]) if len(row) > 6 and row[6] else 100.0
        volume_step = float(row[7]) if len(row) > 7 and row[7] else 0.01
        stops_level = int(float(row[8])) if len(row) > 8 and row[8] else 0
        freeze_level = int(float(row[9])) if len(row) > 9 and row[9] else 0

        # Contract size and tick value calculation for MT4 bridge
        raw_contract = float(row[10]) if len(row) > 10 and row[10] else None
        symbol_upper = str(symbol).upper()
        if raw_contract is not None and raw_contract > 0:
            contract_size = raw_contract
        elif "XAU" in symbol_upper or "GOLD" in symbol_upper:
            contract_size = 100.0
        else:
            contract_size = 100000.0

        raw_tick_val = float(row[11]) if len(row) > 11 and row[11] else None
        tick_value = raw_tick_val if (raw_tick_val is not None and raw_tick_val > 0) else (contract_size * point)

        return SimpleNamespace(
            name=symbol,
            point=point,
            digits=digits,
            volume_min=volume_min,
            volume_max=volume_max,
            volume_step=volume_step,
            trade_contract_size=contract_size,
            trade_tick_value=tick_value,
            trade_tick_value_loss=tick_value,
            trade_tick_value_profit=tick_value,
            trade_tick_size=point,
            filling_mode=self.ORDER_FILLING_IOC,
            trade_execution=self.SYMBOL_TRADE_EXECUTION_MARKET,
            trade_stops_level=stops_level,
            trade_freeze_level=freeze_level,
        )

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        label = self._timeframe_label(timeframe)
        path = self.common_files_dir / f"fx_bridge_rates_{self._safe_symbol(symbol)}_{label}.csv"
        rows = self._read_rows(path)
        if not rows:
            aggregated = self._aggregate_timeframe(symbol, label, count)
            if aggregated:
                return aggregated
            return self._synthetic_rates_from_ticks(symbol, label, count)
        selected = rows[-int(count):]
        parsed_rates = [
            {
                "time": int(float(row[0])),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "tick_volume": int(float(row[5] or 0)),
            }
            for row in selected
            if len(row) >= 6
        ]
        if len(parsed_rates) < int(count):
            synthetic_rates = self._synthetic_rates_from_ticks(symbol, label, count)
            if synthetic_rates:
                return (synthetic_rates + parsed_rates)[-int(count):]
        return parsed_rates

    def copy_ticks_from_pos(self, symbol, start_pos, count, flags):
        for history_tick in self._read_tick_history_file(symbol, count):
            self._remember_tick(symbol, history_tick)
        tick = self._read_latest_tick_file(symbol)
        if tick is not None:
            self._remember_tick(symbol, tick)
        history = list(self._tick_history[self._safe_symbol(symbol)])
        if not history:
            return []
        return history[-int(count):]

    def copy_ticks_from(self, symbol, start_time, count, flags):
        return self.copy_ticks_from_pos(symbol, 0, count, flags)

    def persisted_tick_history_count(self, symbol) -> int:
        return len(self._read_rows(self._tick_history_file(symbol)))

    def tick_history_file_path(self, symbol) -> Path:
        return self._tick_history_file(symbol)

    def positions_get(self, **kwargs):
        rows = self._read_rows(self.common_files_dir / "fx_bridge_positions.csv")
        positions = []
        for row in rows:
            if len(row) < 9:
                continue
            position = SimpleNamespace(
                ticket=int(float(row[0])),
                symbol=row[1],
                volume=float(row[2]),
                type=int(float(row[3])),
                price_open=float(row[4]),
                sl=float(row[5] or 0.0),
                tp=float(row[6] or 0.0),
                profit=float(row[7] or 0.0),
                comment=row[8],
            )
            positions.append(position)
        if "ticket" in kwargs:
            ticket = int(kwargs["ticket"])
            positions = [position for position in positions if int(position.ticket) == ticket]
        if "symbol" in kwargs:
            symbol = str(kwargs["symbol"])
            positions = [position for position in positions if position.symbol == symbol]
        return positions

    def history_get(self):
        rows = self._read_rows(self.common_files_dir / "fx_bridge_history.csv")
        history = []
        for row in rows:
            if len(row) < 12:
                continue
            deal = SimpleNamespace(
                ticket=int(float(row[0])),
                symbol=row[1],
                volume=float(row[2]),
                type=int(float(row[3])),
                price_open=float(row[4]),
                price_close=float(row[5]),
                sl=float(row[6] or 0.0),
                tp=float(row[7] or 0.0),
                profit=float(row[8] or 0.0),
                open_time=row[9],
                close_time=row[10],
                comment=row[11],
                magic_number=int(float(row[12])) if len(row) > 12 else 0
            )
            history.append(deal)
        return history

    def order_calc_margin(self, order_type, symbol, lot, price):
        return 0.0

    def order_check(self, request):
        return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE)

    def order_send(self, request):
        action = request.get("action")
        symbol = request.get("symbol")
        if action == self.TRADE_ACTION_SLTP:
            command_id = self.client.modify_symbol_orders(
                symbol,
                stop_loss=float(request.get("sl", 0.0) or 0.0),
                take_profit=float(request.get("tp", 0.0) or 0.0),
                ticket=int(request.get("position", 0) or 0),
            )
            return self._result_for(command_id)

        if action != self.TRADE_ACTION_DEAL:
            self._last_error = (1, "unsupported MT4 bridge action")
            return SimpleNamespace(retcode=1, order=0, comment="unsupported action")

        if request.get("position"):
            command_id = self.client.close_symbol_orders(symbol, ticket=int(request.get("position", 0) or 0))
            return self._result_for(command_id)

        order_type = int(request.get("type", self.ORDER_TYPE_BUY))
        side = "BUY" if order_type == self.ORDER_TYPE_BUY else "SELL"
        command_id = self.client.open_order(
            side,
            symbol,
            float(request.get("volume", 0.01)),
            stop_loss=float(request.get("sl", 0.0) or 0.0),
            take_profit=float(request.get("tp", 0.0) or 0.0),
            comment=str(request.get("comment", "strategy-live")),
        )
        return self._result_for(command_id)

    def _result_for(self, command_id):
        result = self.client.wait_for_result(command_id)
        if result is None:
            self._last_error = (2, "MT4 bridge command timed out")
            return SimpleNamespace(retcode=2, order=0, comment="bridge timeout")
        if result.status.upper() == "OK":
            self._last_error = (0, "")
            time.sleep(1.0)
            return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, order=result.ticket, comment=result.message)
        self._last_error = (result.error_code, result.message)
        return SimpleNamespace(retcode=result.error_code or 1, order=result.ticket, comment=result.message)

    def _aggregate_timeframe(self, symbol, label, count):
        if label in {"M5", "M10", "M15", "M30"}:
            source = self._read_rates_file(symbol, "M1")
            factor = {"M5": 5, "M10": 10, "M15": 15, "M30": 30}[label]
        elif label in {"H4", "D1"}:
            source = self._read_rates_file(symbol, "H1")
            factor = {"H4": 4, "D1": 24}[label]
        else:
            return None

        required = int(count) * factor
        if len(source) < required:
            return None
        source = source[-required:]
        aggregated = []
        for index in range(0, len(source), factor):
            chunk = source[index:index + factor]
            if len(chunk) < factor:
                continue
            aggregated.append(
                {
                    "time": chunk[0]["time"],
                    "open": chunk[0]["open"],
                    "high": max(row["high"] for row in chunk),
                    "low": min(row["low"] for row in chunk),
                    "close": chunk[-1]["close"],
                    "tick_volume": sum(row["tick_volume"] for row in chunk),
                }
            )
        return aggregated[-int(count):]

    def _remember_tick(self, symbol, tick) -> None:
        safe_symbol = self._safe_symbol(symbol)
        history = self._tick_history[safe_symbol]
        tick_time = int(getattr(tick, "time", 0) or 0)
        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        if bid <= 0.0 and ask <= 0.0:
            return
        if history:
            latest = history[-1]
            if (
                int(getattr(latest, "time", 0) or 0) == tick_time
                and float(getattr(latest, "bid", 0.0) or 0.0) == bid
                and float(getattr(latest, "ask", 0.0) or 0.0) == ask
            ):
                return
        history.append(tick)

    def _synthetic_rates_from_ticks(self, symbol, label, count):
        tick = self.symbol_info_tick(symbol)
        if tick is None:
            return None
        history = list(self._tick_history[self._safe_symbol(symbol)])
        if not history:
            history = [tick]

        timeframe_seconds = {
            "M1": 60,
            "M5": 300,
            "M10": 600,
            "M15": 900,
            "M30": 1800,
            "H1": 3600,
            "H4": 14400,
            "D1": 86400,
        }.get(label, 60)
        latest_time = int(getattr(history[-1], "time", 0) or time.time())
        mids = [self._tick_mid(row) for row in history if self._tick_mid(row) > 0.0]
        if not mids:
            return None

        rates = []
        requested = max(int(count), 1)
        for index in range(requested):
            stamp = latest_time - ((requested - index) * timeframe_seconds)
            price = mids[-1]
            if label == "M1" and len(mids) >= 2 and index == requested - 1:
                open_price = mids[-2]
                close_price = mids[-1]
                high = max(open_price, close_price)
                low = min(open_price, close_price)
            else:
                open_price = high = low = close_price = price
            rates.append(
                {
                    "time": stamp,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close_price,
                    "tick_volume": max(len(mids), 1),
                }
            )
        return rates

    def _tick_mid(self, tick) -> float:
        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        if bid > 0.0 and ask > 0.0:
            return (bid + ask) / 2.0
        return max(bid, ask)

    def _read_rates_file(self, symbol, label):
        path = self.common_files_dir / f"fx_bridge_rates_{self._safe_symbol(symbol)}_{label}.csv"
        rates = []
        for row in self._read_rows(path):
            if len(row) < 6:
                continue
            rates.append(
                {
                    "time": int(float(row[0])),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "tick_volume": int(float(row[5] or 0)),
                }
            )
        return rates

    def _read_tick_history_file(self, symbol, count):
        path = self._tick_history_file(symbol)
        ticks = []
        for row in self._read_rows(path)[-int(count):]:
            if len(row) < 3:
                continue
            try:
                tick = SimpleNamespace(
                    time=int(float(row[0] or 0)),
                    bid=float(row[1] or 0.0),
                    ask=float(row[2] or 0.0),
                    last=float(row[2] or row[1] or 0.0),
                    volume=0,
                )
            except (TypeError, ValueError):
                continue
            if tick.bid > 0.0 or tick.ask > 0.0:
                ticks.append(tick)
        return ticks

    def _read_latest_tick_file(self, symbol):
        row = self._read_first_row(self._symbol_file("fx_bridge_tick", symbol, "csv"))
        if len(row) < 3:
            return None
        try:
            return SimpleNamespace(
                time=int(float(row[0] or 0)),
                bid=float(row[1] or 0.0),
                ask=float(row[2] or 0.0),
                last=float(row[2] or row[1] or 0.0),
                volume=0,
            )
        except (TypeError, ValueError):
            return None

    def _tick_history_file(self, symbol):
        return self.common_files_dir / f"fx_bridge_ticks_{self._safe_symbol(symbol)}.csv"

    def _timeframe_label(self, timeframe):
        if isinstance(timeframe, str) and timeframe.startswith("TIMEFRAME_"):
            return timeframe.replace("TIMEFRAME_", "")
        return str(timeframe)

    def _symbol_file(self, prefix, symbol, extension):
        return self.common_files_dir / f"{prefix}_{self._safe_symbol(symbol)}.{extension}"

    def _safe_symbol(self, symbol):
        return str(symbol).replace(".", "_").replace("#", "_").replace("/", "_").replace("\\", "_")

    def _read_first_row(self, path):
        rows = self._read_rows(path)
        return rows[0] if rows else []

    def _read_rows(self, path):
        if not Path(path).exists():
            return []
        text = ""
        for attempt in range(5):
            try:
                text = Path(path).read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    break
            except PermissionError:
                if attempt == 4:
                    return []
            time.sleep(0.05)
        if not text:
            return []
        rows = [line.split(",") for line in text.splitlines() if line.strip()]
        return [row for row in rows if any(str(item).strip() for item in row)]


def discover_mt4_data_dir(explicit_path: str = "") -> Path | None:
    if explicit_path:
        path = Path(explicit_path)
        return path if path.exists() else None

    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None

    terminal_root = Path(appdata) / "MetaQuotes" / "Terminal"
    if not terminal_root.exists():
        return None

    candidates = [
        path
        for path in terminal_root.iterdir()
        if path.is_dir() and (path / "MQL4").exists()
    ]
    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


def install_bridge_ea(project_root: str | Path, mt4_data_path: str = "") -> Mt4BridgeInstall:
    root = Path(project_root)
    source = root / "tools" / "mt4" / BRIDGE_EA_NAME
    if not source.exists():
        raise FileNotFoundError(f"MT4 bridge EA source not found: {source}")

    data_dir = discover_mt4_data_dir(mt4_data_path)
    if data_dir is None:
        raise FileNotFoundError("Could not find an MT4 data folder under AppData\\Roaming\\MetaQuotes\\Terminal")

    experts_dir = data_dir / "MQL4" / "Experts"
    experts_dir.mkdir(parents=True, exist_ok=True)
    destination = experts_dir / BRIDGE_EA_NAME
    shutil.copy2(source, destination)
    compiled_source = source.with_suffix(".ex4")
    if compiled_source.exists() and compiled_source.stat().st_mtime >= source.stat().st_mtime:
        shutil.copy2(compiled_source, destination.with_suffix(".ex4"))

    common_files_dir = Path(os.environ.get("APPDATA", "")) / "MetaQuotes" / "Terminal" / "Common" / "Files"
    common_files_dir.mkdir(parents=True, exist_ok=True)

    return Mt4BridgeInstall(source=source, destination=destination, common_files_dir=common_files_dir)
