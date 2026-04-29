from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import ctypes

from src.config import load_settings
from src.live_trade_loop import run_live_signal_loop
from src.mt5_client import Mt5Session
from src.trade_executor import TradeExecutor
from src.tradingview import TradingViewAlertStore, start_tradingview_webhook_server


RELOAD_REQUESTED = "reload_requested"


def create_mt5_module():
    import MetaTrader5 as mt5

    return mt5


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class CodeWatcher:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        self._snapshot = self._build_snapshot()

    def has_changes(self) -> bool:
        current_snapshot = self._build_snapshot()
        if current_snapshot != self._snapshot:
            self._snapshot = current_snapshot
            return True
        return False

    def _build_snapshot(self) -> dict[str, int]:
        snapshot = {}
        for path in self.project_root.rglob("*.py"):
            if any(part == "__pycache__" for part in path.parts):
                continue
            if path.is_file():
                snapshot[str(path.resolve())] = path.stat().st_mtime_ns
        return snapshot


def build_code_watcher(project_root: str | Path | None = None) -> CodeWatcher:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parent
    return CodeWatcher(root)


def restart_process() -> None:
    os.execv(sys.executable, [sys.executable, *sys.argv])


def set_keep_awake():
    """Prevents Windows from entering sleep mode while the bot is running."""
    if os.name == 'nt':
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000040)
        logging.info("SYSTEM STAY-AWAKE ACTIVE (Windows Power Management Override)")


def main() -> int:
    configure_logging()
    set_keep_awake()
    settings = load_settings()
    code_watcher = build_code_watcher()

    mt5 = create_mt5_module()
    session = Mt5Session(
        terminal_path=settings.mt5_terminal_path,
        startup_wait_seconds=settings.mt5_startup_wait_seconds,
        mt5_module=mt5,
    )

    reload_requested = False
    try:
        tradingview_alert_store = None
        tradingview_server = None
        if settings.tradingview_webhook_enabled:
            tradingview_alert_store = TradingViewAlertStore(
                max_age_seconds=settings.tradingview_alert_max_age_seconds,
            )
            tradingview_server = start_tradingview_webhook_server(
                host=settings.tradingview_webhook_host,
                port=settings.tradingview_webhook_port,
                expected_secret=settings.tradingview_webhook_secret,
                store=tradingview_alert_store,
                log_fn=logging.info,
            )
            logging.info(
                "TradingView webhook receiver listening on http://%s:%s/tradingview",
                settings.tradingview_webhook_host,
                settings.tradingview_webhook_port,
            )

        logging.info("Launching MT5 terminal")
        session.launch_terminal()

        logging.info("Initializing MT5 and logging into HFM")
        account_info = session.initialize_and_login(
            login=settings.hfm_login,
            password=settings.hfm_password,
            server=settings.hfm_server,
        )
        if account_info is not None:
            logging.info("Connected account login=%s", getattr(account_info, "login", "unknown"))

        logging.info(
            "Starting live/demo strategy loop for symbol=%s lot=%s",
            settings.trading_symbol,
            settings.default_trade_lot,
        )
        executor = TradeExecutor(mt5)
        loop_result = run_live_signal_loop(
            mt5_module=mt5,
            executor=executor,
            symbol=settings.trading_symbol,
            lot=settings.default_trade_lot,
            add_on_lot_increment=settings.add_on_lot_increment,
            campaign_max_exposure_pct=settings.campaign_max_exposure_pct,
            risk_buffer=settings.risk_buffer,
            max_candles_since_breakout=settings.max_candles_since_breakout,
            poll_seconds=settings.loop_poll_seconds,
            max_loops=settings.max_live_loops,
            tradingview_alert_store=tradingview_alert_store,
            strategy_profile=settings.strategy_profiles.get(settings.trading_symbol.strip().upper()),
            reload_check_fn=code_watcher.has_changes,
            log_fn=logging.info,
            settings=settings,
        )
        reload_requested = loop_result == RELOAD_REQUESTED
        return 0
    finally:
        if "tradingview_server" in locals() and tradingview_server is not None:
            tradingview_server.shutdown()
        session.shutdown()
        if reload_requested:
            logging.info("RELOADING BOT")
            restart_process()


if __name__ == "__main__":
    raise SystemExit(main())
