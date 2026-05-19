from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import itertools
from dataclasses import replace

import MetaTrader5 as mt5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.live_trade_loop import run_live_signal_loop
from src.mt5_client import Mt5Session
from tools.backtest_run_strategy import (
    _completed_m5_end,
    _earliest_available_m5,
    _build_rates,
    _eligible_m5_indices,
    ReplayMt5Module,
    ReplayExecutor,
    _build_sleep_fn,
)

UTC = timezone.utc

def main():
    settings = load_settings(".env")
    symbol = "XAUUSD"

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

    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Unable to select symbol for replay: {symbol}")

    end = _completed_m5_end(symbol)
    start = end - timedelta(hours=72)
    rates_by_timeframe = _build_rates(settings, symbol, start, end)
    eligible_indices = _eligible_m5_indices(rates_by_timeframe, start, end)

    symbol_info = mt5.symbol_info(symbol)
    point = float(getattr(symbol_info, "point", 0.01) or 0.01)
    contract_size = float(getattr(symbol_info, "trade_contract_size", 100.0) or 100.0)

    # Specific parameter tuples to test: (min_edge, exp_move, omega)
    test_params = [
        (0.0, 1.0, 0.5), # Current
        (1.0, 1.0, 0.5), # Moderate
        (1.5, 1.2, 0.5), # Conservative
        (2.0, 1.5, 0.6), # Ultra Conservative
    ]

    best_net = -float("inf")
    best_params = None
    results = []

    print(f"Starting Grid Search. 72-hour backtest. Loops: {len(eligible_indices)}")

    for min_edge, exp_move, omega in test_params:
        print(f"Testing min_edge={min_edge}, exp_move={exp_move}, omega={omega}...", end="", flush=True)

        # Update settings
        profile = settings.strategy_profiles[symbol]
        new_profile = replace(
            profile, 
            min_edge_threshold=min_edge,
            minimum_expected_move_multiple=exp_move,
        )
        new_profiles = dict(settings.strategy_profiles)
        new_profiles[symbol] = new_profile
        
        test_settings = replace(
            settings,
            strategy_profiles=new_profiles,
            quant_omega_threshold=omega,
        )

        replay_mt5 = ReplayMt5Module(
            symbol=symbol,
            rates_by_timeframe=rates_by_timeframe,
            point=point,
            initial_equity=500000.0,
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
            lot=test_settings.default_trade_lot,
            add_on_lot_increment=test_settings.add_on_lot_increment,
            campaign_max_exposure_pct=test_settings.campaign_max_exposure_pct,
            risk_buffer=test_settings.risk_buffer,
            max_candles_since_breakout=test_settings.max_candles_since_breakout,
            poll_seconds=test_settings.loop_poll_seconds,
            max_loops=len(eligible_indices),
            tradingview_alert_store=None,
            strategy_profile=new_profile,
            reload_check_fn=None,
            sleep_fn=sleep_fn,
            log_fn=lambda _message: None,
            settings=test_settings,
        )

        final_bar = replay_mt5.rates_by_timeframe["TIMEFRAME_M5"][eligible_indices[-1]]
        final_close = float(final_bar["close"])
        open_unrealized = executor.mark_to_market(final_close)

        gross_profit = sum(trade.profit for trade in executor.closed_trades if trade.profit > 0)
        gross_loss = -sum(trade.profit for trade in executor.closed_trades if trade.profit < 0)
        net_closed = gross_profit - gross_loss
        net_with_open = net_closed + open_unrealized
        wins = sum(1 for trade in executor.closed_trades if trade.profit > 0)
        losses = sum(1 for trade in executor.closed_trades if trade.profit < 0)

        results.append({
            "params": (min_edge, exp_move, omega),
            "net": net_with_open,
            "trades": len(executor.closed_trades),
            "wins": wins
        })

        print(f" Net: {net_with_open:.2f}, Trades: {len(executor.closed_trades)}")

        if net_with_open > best_net:
            best_net = net_with_open
            best_params = (min_edge, exp_move, omega)

    print("\n--- Top Results ---")
    results.sort(key=lambda x: x["net"], reverse=True)
    for r in results[:5]:
        print(f"Params: min_edge={r['params'][0]}, exp_move={r['params'][1]}, omega={r['params'][2]} -> Net: {r['net']:.2f}, Trades: {r['trades']}, Wins: {r['wins']}")

if __name__ == "__main__":
    main()
