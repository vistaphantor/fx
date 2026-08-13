from types import SimpleNamespace
from datetime import datetime, timezone
import pytest

from src.strategy.breakout import BreakoutDirection
from src.strategy.decision_tree import TopDownTradePlan, TopDownNoTrade
from src.live_trade_loop import run_live_signal_loop, _check_correlation_limit
from src.config import _load_strategy_profiles


def test_multi_symbol_config_parsing():
    values = {
        "TRADING_SYMBOL": "XAUUSD, EURJPY, GBPUSD",
        "XAUUSD_MIN_EDGE_THRESHOLD": "1.0",
        "EURJPY_MIN_EDGE_THRESHOLD": "2.0",
        # GBPUSD has no config values, should fallback to XAUUSD defaults
    }
    profiles = _load_strategy_profiles(values)
    
    assert "XAUUSD" in profiles
    assert "EURJPY" in profiles
    assert "GBPUSD" in profiles
    
    assert profiles["XAUUSD"].min_edge_threshold == 1.0
    assert profiles["EURJPY"].min_edge_threshold == 2.0
    # GBPUSD falls back to default profile which is XAUUSD (min_edge_threshold = 2.5)
    assert profiles["GBPUSD"].min_edge_threshold == 2.5


def test_correlation_limit_gating_blocks_trade(monkeypatch):
    class FakePosition:
        def __init__(self, symbol, type, volume, price_open=2350.0):
            self.symbol = symbol
            self.type = type # 0 = BUY, 1 = SELL
            self.volume = volume
            self.price_open = price_open

    class FakeExecutor:
        def list_bot_positions(self, symbol):
            if symbol == "XAUUSD":
                return [FakePosition("XAUUSD", 0, 2.0)] # Already holding 2.0 lot Buy
            return []

    # New trade: Buying EURUSD (corr with XAUUSD is 0.82)
    # Aligned exposure = 2.0 * 0.82 * 1 * 1 = 1.64
    # With base lot = 1.0, 1.64 + 1.0 = 2.64 > 2.5 * 1.0 (limit) -> blocked!
    blocked = _check_correlation_limit(
        executor=FakeExecutor(),
        symbols=["XAUUSD", "EURUSD"],
        new_symbol="EURUSD",
        new_direction=BreakoutDirection.BULLISH,
        base_lot=1.0,
        log_fn=lambda x: None
    )
    assert blocked is True


def test_correlation_limit_gating_allows_non_correlated():
    class FakePosition:
        def __init__(self, symbol, type, volume, price_open=2350.0):
            self.symbol = symbol
            self.type = type
            self.volume = volume
            self.price_open = price_open

    class FakeExecutor:
        def list_bot_positions(self, symbol):
            if symbol == "EURJPY":
                return [FakePosition("EURJPY", 0, 2.0)] # EURJPY is uncorrelated with XAUUSD
            return []

    # New trade: Buying XAUUSD (no correlation)
    # Aligned exposure = 0.0
    blocked = _check_correlation_limit(
        executor=FakeExecutor(),
        symbols=["EURJPY", "XAUUSD"],
        new_symbol="XAUUSD",
        new_direction=BreakoutDirection.BULLISH,
        base_lot=1.0,
        log_fn=lambda x: None
    )
    assert blocked is False


def test_live_loop_correlation_blocking_integration(monkeypatch):
    events = []
    live_input = SimpleNamespace(
        symbol="EURUSD",
        d1_candles=["d1"],
        h4_candles=["h4"],
        h1_candles=["h1"],
        m30_candles=["m30"],
        m15_candles=[SimpleNamespace(close=1.0850)],
        m10_candles=[SimpleNamespace(close=1.0850)],
        m5_candles=[SimpleNamespace(close=1.0850)],
        spread=0.0001
    )
    trade_plan = TopDownTradePlan(
        is_trade=True,
        direction=BreakoutDirection.BULLISH,
        entry_price=1.0850,
        stop_loss=1.0800,
        take_profit=1.0950,
        objective_price=1.0900,
        reason="breakout_confirmed",
        metadata={},
    )

    class FakePosition:
        def __init__(self, symbol, type, volume, price_open=1.0850):
            self.symbol = symbol
            self.type = type
            self.volume = volume
            self.price_open = price_open

    class FakeExecutor:
        def list_bot_positions(self, symbol):
            if symbol == "XAUUSD":
                return [FakePosition("XAUUSD", 0, 2.0, price_open=2350.0)] # Correlation risk
            return []

        def open_strategy_trade(self, **kwargs):
            events.append(("open", kwargs))
            return SimpleNamespace(ticket=99)

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr("src.live_trade_loop.evaluate_top_down_decision_tree", lambda **kwargs: trade_plan)

    run_live_signal_loop(
        mt5_module=SimpleNamespace(),
        executor=FakeExecutor(),
        symbol="XAUUSD,EURUSD", # Multiple symbols active
        lot=1.0,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(("log", message)),
        settings=SimpleNamespace(fusion_enabled=False, quant_enabled=False, strategy_profiles={})
    )

    # Verify that the trade was blocked and no order was opened
    assert not any(e[0] == "open" for e in events)
    assert any("CORRELATION GATING BLOCKED" in e[1] for e in events if e[0] == "log")
