from __future__ import annotations

import logging
from dataclasses import is_dataclass, replace
from typing import TYPE_CHECKING

from src.requirements_check import ensure_requirements_satisfied

from run import RELOAD_REQUESTED, build_code_watcher, configure_logging, create_mt5_module, restart_process, set_keep_awake

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
        f"poll_seconds={settings.quick_poll_seconds} min_free_margin={settings.quick_min_free_margin}"
    )


def main() -> int:
    ensure_requirements_satisfied()
    configure_logging()
    set_keep_awake()
    settings = build_quick_runtime_settings(load_settings())
    code_watcher = build_code_watcher()

    mt5 = create_mt5_module()
    session = Mt5Session(
        terminal_path=settings.mt5_terminal_path,
        startup_wait_seconds=settings.mt5_startup_wait_seconds,
        mt5_module=mt5,
    )

    reload_requested = False
    try:
        logging.info("Launching MT5 terminal for quick scalper")
        session.launch_terminal()

        logging.info("Initializing MT5 and logging into HFM quick account")
        account_info = session.initialize_and_login(
            login=settings.hfm_login,
            password=settings.hfm_password,
            server=settings.hfm_server,
        )
        logging.info("QUICK SCALPER START %s", format_quick_startup_summary(settings, account_info))

        executor = TradeExecutor(mt5)
        try:
            loop_result = run_quick_scalp_loop(
                mt5_module=mt5,
                executor=executor,
                symbol=settings.trading_symbol,
                lot=settings.quick_trade_lot,
                max_positions=settings.quick_max_positions,
                profit_target=settings.quick_profit_target,
                poll_seconds=settings.quick_poll_seconds,
                min_free_margin=settings.quick_min_free_margin,
                max_loops=settings.max_live_loops,
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
        session.shutdown()
        if reload_requested:
            logging.info("RELOADING QUICK SCALPER")
            restart_process()


if __name__ == "__main__":
    raise SystemExit(main())
