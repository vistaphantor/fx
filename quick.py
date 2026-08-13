from __future__ import annotations

import logging
import os
import socket
from dataclasses import is_dataclass, replace
from typing import TYPE_CHECKING

from src.requirements_check import ensure_requirements_satisfied

from run import (
    RELOAD_REQUESTED,
    build_code_watcher,
    configure_logging,
    create_mt5_module,
    launch_mt4_fallback,
    restart_process,
    set_keep_awake,
)
import threading
import subprocess
import sys


def start_dashboard():
    """Start the dashboard server in a separate process."""
    if _dashboard_is_running():
        logging.info("DASHBOARD SERVER ALREADY RUNNING at http://localhost:8000")
        return

    def run_server():
        subprocess.Popen([sys.executable, "dashboard_server.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    logging.info("DASHBOARD SERVER STARTED at http://localhost:8000")


def _dashboard_is_running(host: str = "127.0.0.1", port: int = 8000) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=0.25):
            return True
    except OSError:
        return False

if TYPE_CHECKING:
    from src.config import Settings


def load_settings():
    from src.config import load_settings as load_app_settings

    return load_app_settings()


def Mt5Session(*args, **kwargs):
    from src.mt5_client import Mt5Session as AppMt5Session

    return AppMt5Session(*args, **kwargs)


def run_quick_scalp_loop(**kwargs):
    from src.quick_scalp_loop import run_quick_scalp_loop as run_app_quick_scalp_loop

    return run_app_quick_scalp_loop(**kwargs)


def clear_dashboard_state() -> None:
    from src.quick_scalp_loop import BOT_STATE_FILE

    if os.path.exists(BOT_STATE_FILE):
        os.remove(BOT_STATE_FILE)


def log_tick_history_health(mt5_module, symbol: str, minimum_ticks: int = 4) -> None:
    count_fn = getattr(mt5_module, "persisted_tick_history_count", None)
    path_fn = getattr(mt5_module, "tick_history_file_path", None)
    if count_fn is None:
        return
    count = int(count_fn(symbol) or 0)
    if count >= int(minimum_ticks):
        logging.info("MT4 TICK HISTORY READY %s ticks=%s", symbol, count)
        return
    path = path_fn(symbol) if path_fn is not None else f"fx_bridge_ticks_{symbol}.csv"
    logging.warning(
        "MT4 TICK HISTORY NOT READY %s ticks=%s minimum=%s file=%s reload_or_recompile_bridge_ea=true",
        symbol,
        count,
        int(minimum_ticks),
        path,
    )


def TradeExecutor(*args, **kwargs):
    from src.trade_executor import TradeExecutor as AppTradeExecutor

    return AppTradeExecutor(*args, **kwargs)


def force_quick_settings(settings: Settings) -> Settings:
    if is_dataclass(settings):
        return replace(settings, quick_scalp_enabled=True)
    setattr(settings, "quick_scalp_enabled", True)
    return settings


def validate_quick_settings(settings: Settings) -> None:
    if float(settings.quick_trade_lot) <= 0:
        raise ValueError("quick_trade_lot must be greater than 0")
    if int(settings.quick_max_positions) <= 0:
        raise ValueError("quick_max_positions must be greater than 0")
    if float(settings.quick_profit_target) < 0:
        raise ValueError("quick_profit_target must be 0 or greater")
    if float(settings.quick_max_loss) < 0:
        raise ValueError("quick_max_loss must be 0 or greater")
    if int(settings.quick_poll_seconds) <= 0:
        raise ValueError("quick_poll_seconds must be greater than 0")
    if float(settings.quick_min_free_margin) < 0:
        raise ValueError("quick_min_free_margin must be 0 or greater")


def build_quick_runtime_settings(settings: Settings) -> Settings:
    quick_settings = force_quick_settings(settings)
    validate_quick_settings(quick_settings)
    return quick_settings


def format_quick_startup_summary(settings: Settings, account_info=None) -> str:
    login = getattr(account_info, "login", settings.hfm_login)
    currency = getattr(account_info, "currency", "unknown")
    balance = getattr(account_info, "balance", "unknown")
    equity = getattr(account_info, "equity", "unknown")
    return (
        f"login={login} currency={currency} balance={balance} equity={equity} "
        f"symbol={settings.trading_symbol} lot={settings.quick_trade_lot} "
        f"max_positions={settings.quick_max_positions} profit_target={settings.quick_profit_target} "
        f"max_loss={settings.quick_max_loss} "
        f"poll_seconds={settings.quick_poll_seconds} min_free_margin={settings.quick_min_free_margin}"
    )


def main() -> int:
    ensure_requirements_satisfied()
    configure_logging()
    set_keep_awake()
    settings = build_quick_runtime_settings(load_settings())
    # Delete stale state file to prevent dashboard from showing old data
    try:
        clear_dashboard_state()
    except OSError:
        pass
    
    code_watcher = build_code_watcher()
    start_dashboard()

    mt5 = create_mt5_module()
    session = Mt5Session(
        terminal_path=settings.mt5_terminal_path,
        startup_wait_seconds=settings.mt5_startup_wait_seconds,
        mt5_module=mt5,
    )

    reload_requested = False
    try:
        prefer_mt4 = bool(getattr(settings, "quick_prefer_mt4", False))
        if prefer_mt4:
            logging.info("QUICK_PREFER_MT4 enabled; launching MT4 bridge for quick scalper")
            mt4 = launch_mt4_fallback(settings)
            if mt4 is None:
                logging.error("MT4 bridge did not become ready; quick scalper will not trade")
                return 0
            mt5 = mt4
            session = None
            account_info = mt5.account_info() if hasattr(mt5, "account_info") else None
        else:
            logging.info("Launching MT5 terminal for quick scalper")
            try:
                session.launch_terminal()
            except FileNotFoundError:
                mt4 = launch_mt4_fallback(settings)
                if mt4 is None:
                    if getattr(settings, "mt4_fallback_enabled", False):
                        return 0
                    raise
                mt5 = mt4
                session = None
                account_info = mt5.account_info() if hasattr(mt5, "account_info") else None
            else:
                logging.info("Initializing MT5 and logging into HFM quick account")
                account_info = session.initialize_and_login(
                    login=settings.hfm_login,
                    password=settings.hfm_password,
                    server=settings.hfm_server,
                )
        # Auto-Correct: Remove Cent suffix if on Demo server
        if "Demo" in settings.hfm_server and settings.trading_symbol.endswith("c"):
            logging.info("DEMO SERVER DETECTED: Stripping 'c' from symbol for compatibility")
            settings = replace(settings, trading_symbol=settings.trading_symbol[:-1])
            
        logging.info("QUICK SCALPER START %s", format_quick_startup_summary(settings, account_info))
        if getattr(settings, "quick_tick_in_out_mode", False):
            log_tick_history_health(mt5, settings.trading_symbol)
        
        # Clear stale state on startup
        clear_dashboard_state()

        executor = TradeExecutor(mt5)
        try:
            loop_result = run_quick_scalp_loop(
                mt5_module=mt5,
                executor=executor,
                symbol=settings.trading_symbol,
                lot=settings.quick_trade_lot,
                max_positions=settings.quick_max_positions,
                profit_target=settings.quick_profit_target,
                max_loss=settings.quick_max_loss,
                poll_seconds=settings.quick_poll_seconds,
                min_free_margin=settings.quick_min_free_margin,
                max_loops=settings.max_live_loops,
                atr_sl_multiplier=settings.quick_atr_sl_mult,
                atr_tp_multiplier=settings.quick_atr_tp_mult,
                max_spread_pips=settings.quick_max_spread_pips,
                tick_in_out_mode=settings.quick_tick_in_out_mode,
                daily_profit_target=settings.quick_daily_profit_target,
                daily_max_loss=settings.quick_daily_max_loss,
                min_estimated_profit=settings.quick_min_estimated_profit,
                execution_buffer=settings.quick_execution_buffer,
                loss_cooldown_seconds=settings.quick_loss_cooldown_seconds,
                entry_cooldown_seconds=settings.quick_entry_cooldown_seconds,
                shadow_on_daily_halt=settings.quick_shadow_on_daily_halt,
                shadow_on_unproven_edge=settings.quick_shadow_on_unproven_edge,
                shadow_only=settings.quick_shadow_only,
                shadow_policy_enabled=settings.quick_shadow_policy_enabled,
                shadow_policy_path=settings.quick_shadow_policy_path,
                allow_inverted_shadow_policy=settings.quick_allow_inverted_shadow_policy,
                live_pilot_max_trades=settings.quick_live_pilot_max_trades,
                reload_check_fn=code_watcher.has_changes,
                log_fn=logging.info,
            )
            reload_requested = loop_result == RELOAD_REQUESTED
        except KeyboardInterrupt:
            logging.info("Quick scalper interrupted; shutting down")
            return 130
        return 0
    except Exception:
        logging.exception("Quick scalper stopped with error")
        raise
    finally:
        if session is not None:
            session.shutdown()
        if reload_requested:
            logging.info("RELOADING QUICK SCALPER")
            restart_process()


if __name__ == "__main__":
    raise SystemExit(main())
