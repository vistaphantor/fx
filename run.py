import argparse
import logging
import os
from pathlib import Path
import sys
import ctypes
from dataclasses import is_dataclass, replace

from src.requirements_check import ensure_requirements_satisfied


RELOAD_REQUESTED = "reload_requested"


def load_settings():
    from src.config import load_settings as load_app_settings

    return load_app_settings()


def create_mt5_module():
    import MetaTrader5 as mt5

    return mt5


def run_live_signal_loop(**kwargs):
    from src.live_trade_loop import run_live_signal_loop as run_app_live_signal_loop

    return run_app_live_signal_loop(**kwargs)


def Mt5Session(*args, **kwargs):
    from src.mt5_client import Mt5Session as AppMt5Session

    return AppMt5Session(*args, **kwargs)


def TradeExecutor(*args, **kwargs):
    from src.trade_executor import TradeExecutor as AppTradeExecutor

    return AppTradeExecutor(*args, **kwargs)


def TradingViewAlertStore(*args, **kwargs):
    from src.tradingview import TradingViewAlertStore as AppTradingViewAlertStore

    return AppTradingViewAlertStore(*args, **kwargs)


def start_tradingview_webhook_server(**kwargs):
    from src.tradingview import start_tradingview_webhook_server as start_app_tradingview_webhook_server

    return start_app_tradingview_webhook_server(**kwargs)


import threading
import subprocess
import sys
import time

NGROK_DOMAIN = "heftiness-handler-vocalize.ngrok-free.app"
NGROK_LOCAL_PORT = 8000


def start_dashboard():
    """Start the dashboard server in a separate process."""
    def run_server():
        subprocess.Popen([sys.executable, "dashboard_server.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    logging.info("DASHBOARD SERVER STARTED at http://localhost:%s", NGROK_LOCAL_PORT)


def start_ngrok():
    """Start ngrok tunnel pointing to the dashboard server using the fixed free domain."""
    def run_tunnel():
        try:
            cmd = [
                "ngrok", "http",
                f"--domain={NGROK_DOMAIN}",
                str(NGROK_LOCAL_PORT),
                "--log=stdout",
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            # Stream first few lines so startup errors are visible
            for i, line in enumerate(proc.stdout):
                if i < 10:
                    logging.info("[ngrok] %s", line.rstrip())
        except FileNotFoundError:
            logging.warning(
                "ngrok not found — GoCharting webhooks will NOT reach this bot. "
                "Install ngrok and ensure it is on PATH."
            )
        except Exception as exc:
            logging.warning("ngrok tunnel failed to start: %s", exc)

    # Small delay so the dashboard server is ready before ngrok connects
    time.sleep(1.5)
    thread = threading.Thread(target=run_tunnel, daemon=True)
    thread.start()
    logging.info(
        "NGROK TUNNEL STARTING → https://%s  (orderflow webhook: https://%s/api/orderflow)",
        NGROK_DOMAIN,
        NGROK_DOMAIN,
    )


def configure_logging() -> None:
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


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


def force_standard_settings(settings):
    if is_dataclass(settings):
        return replace(settings, quick_scalp_enabled=False)
    setattr(settings, "quick_scalp_enabled", False)
    return settings


def run_backtest_mode(mt5, settings, days):
    """Run a high-fidelity backtest with Session Gating, DXY Correlation, and Trailing Stop."""
    symbol = settings.trading_symbol
    logging.info("STARTING ADVANCED MACRO BACKTEST: symbol=%s days=%s", symbol, days)
    
    from src.market_data import fetch_candles, Candle
    from src.strategy.m5_engine import M5SignalEngine
    from src.strategy.m15_engine import M15SignalEngine
    from src.strategy.context_engine import ContextEngine
    from src.strategy.session_engine import SessionEngine
    from src.strategy.quant_engine import evaluate_master_equation, QuantParams
    from src.strategy.features import FeatureExtractor
    import math
    from dataclasses import dataclass
    from datetime import datetime, timezone

    def aggregate_candles(m1_candles, factor, label):
        aggregated = []
        for i in range(0, len(m1_candles), factor):
            chunk = m1_candles[i:i + factor]
            if not chunk: continue
            aggregated.append(Candle(
                timestamp=chunk[0].timestamp,
                open=chunk[0].open,
                high=max(c.high for c in chunk),
                low=min(c.low for c in chunk),
                close=chunk[-1].close,
                volume=sum(c.volume for c in chunk),
                timeframe=label
            ))
        return aggregated

    @dataclass
    class Trade:
        direction: str
        entry: float
        sl: float
        tp: float
        omega: float
        peak_price: float

    count = int(days * 1440)
    logging.info("Fetching %s M1 candles for %s and USDIndex...", count, symbol)
    try:
        candles = fetch_candles(mt5, symbol, "TIMEFRAME_M1", count)
        dxy_candles = fetch_candles(mt5, "USDIndex", "TIMEFRAME_M1", count)
    except Exception as e:
        logging.error("Failed to fetch backtest data: %s", e)
        return

    m5_eng = M5SignalEngine()
    m15_eng = M15SignalEngine()
    ctx_eng = ContextEngine()
    session_eng = SessionEngine()
    feat_ext = FeatureExtractor(window=20)
    quant_params = QuantParams()
    
    closed_pnls = []
    active_trade = None
    start_idx = 4320 
    
    if len(candles) <= start_idx or len(dxy_candles) <= start_idx:
        logging.error("Not enough history for backtest")
        return

    # Sync DXY and Gold by timestamp
    dxy_map = {c.timestamp: c.close for c in dxy_candles}

    for i in range(start_idx, len(candles) - 1):
        if i % 1000 == 0:
            logging.info("Backtest Progress: %d / %d", i, len(candles))
            
        curr = candles[i]
        nxt = candles[i+1]
        
        if active_trade:
            trail_dist = 1.0 
            if active_trade.direction == "BUY":
                if curr.high > active_trade.peak_price:
                    active_trade.peak_price = curr.high
                    active_trade.sl = max(active_trade.sl, active_trade.peak_price - trail_dist)
                if nxt.low <= active_trade.sl:
                    closed_pnls.append(active_trade.sl - active_trade.entry)
                    active_trade = None
                elif nxt.high >= active_trade.tp:
                    closed_pnls.append(active_trade.tp - active_trade.entry)
                    active_trade = None
            else:
                if curr.low < active_trade.peak_price:
                    active_trade.peak_price = curr.low
                    active_trade.sl = min(active_trade.sl, active_trade.peak_price + trail_dist)
                if nxt.high >= active_trade.sl:
                    closed_pnls.append(active_trade.entry - active_trade.sl)
                    active_trade = None
                elif nxt.low <= active_trade.tp:
                    closed_pnls.append(active_trade.entry - active_trade.tp)
                    active_trade = None
        
        if active_trade: continue
        
        m5_c = aggregate_candles(candles[i-300:i+1], 5, "M5")
        m15_c = aggregate_candles(candles[i-900:i+1], 15, "M15")
        m30_c = aggregate_candles(candles[i-1800:i+1], 30, "M30")
        h1_c = aggregate_candles(candles[i-3600:i+1], 60, "H1")
        d1_c = aggregate_candles(candles[i-4320:i+1], 1440, "D1")
        
        m5_s = m5_eng.compute(m5_c)
        m15_s = m15_eng.compute(m15_c)
        ctx_s = ctx_eng.compute(h1_c, m30_c, d1_c)
        
        if not (m5_s and m15_s and ctx_s): continue

        # Session Score
        dt = curr.timestamp
        s_score = session_eng.compute_session_score(dt)

        # DXY Trend (M15 equivalent)
        dxy_trend = 0.0
        try:
            dxy_now = dxy_map.get(curr.timestamp, dxy_candles[i].close)
            dxy_prev = dxy_map.get(candles[i-15].timestamp, dxy_candles[i-15].close)
            dxy_trend = (dxy_now - dxy_prev) / dxy_prev * 1000.0
        except Exception: pass

        # Features for Master Equation
        returns = [(candles[j].close - candles[j-1].close)/candles[j-1].close for j in range(i-50, i+1)]
        feat_snap = feat_ext.update(
            momentum_raw=m15_s.composite_score,
            trend_raw=ctx_s.composite_context,
            volume_raw=m5_s.composite_score,
            order_block_raw=0.5,
            volatility_risk_raw=0.5,
            entry_distance_raw=0.0,
            spread_danger_raw=0.0,
            orderflow_raw=0.0,
            m5_score_raw=m5_s.composite_score,
            m15_score_raw=m15_s.composite_score,
            context_score_raw=ctx_s.composite_context,
            expected_return=sum(returns)/len(returns),
            return_std=math.sqrt(sum(r**2 for r in returns)/len(returns)),
            timestamp=curr.timestamp
        )

        decision = evaluate_master_equation(
            features=feat_snap,
            params=quant_params,
            equity=10000,
            drawdown_ratio=0.0,
            recent_returns=returns,
            transaction_cost=0.0001,
            session_score=s_score,
            dxy_trend=0.0 # Temporarily zeroed for diagnostic
        )
        
        if decision.omega_t > 0.4:
            logging.debug("Signal Check: %s Omega=%.3f session=%.2f", dt.strftime("%H:%M"), decision.omega_t, s_score)

        if decision.omega_t >= 0.45:
            atr = (max(c.high for c in candles[i-14:i+1]) - min(c.low for c in candles[i-14:i+1])) / 2.0
            if atr <= 0: continue
            entry = nxt.open
            direction = "BUY" if decision.action == 1 else "SELL"
            active_trade = Trade(direction, entry, entry - atr*1.5 if direction=="BUY" else entry+atr*1.5, 
                                 entry + atr*3.0 if direction=="BUY" else entry-atr*3.0, decision.omega_t, entry)

    logging.info("BACKTEST COMPLETE")
    if not closed_pnls:
        logging.info("No trades executed.")
        return

    net = sum(closed_pnls)
    wins = sum(1 for p in closed_pnls if p > 0)
    wr = wins / len(closed_pnls) * 100
    logging.info("RESULTS: Trades=%s NetPnL=%.4f WinRate=%.2f%%", len(closed_pnls), net, wr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true", help="Run in backtest mode")
    parser.add_argument("--days", type=int, default=15, help="Number of days to backtest")
    args = parser.parse_args()

    ensure_requirements_satisfied()
    configure_logging()
    set_keep_awake()
    settings = force_standard_settings(load_settings())
    
    if not args.backtest:
        code_watcher = build_code_watcher()
        start_dashboard()
        start_ngrok()

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

        if args.backtest:
            run_backtest_mode(mt5, settings, args.days)
            return 0

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
