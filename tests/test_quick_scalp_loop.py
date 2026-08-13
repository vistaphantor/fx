from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def isolate_quick_session_state(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.quick_scalp_loop.QUICK_SESSION_STATE_FILE",
        str(tmp_path / "quick_session_state.json"),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.BOT_STATE_FILE",
        str(tmp_path / "bot_state.json"),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.QUICK_SHADOW_JOURNAL_FILE",
        str(tmp_path / "quick_shadow_trades.csv"),
    )


def bullish_ticks():
    # Net move ~2.0, well above the 1-pip minimum; 4/4 up-moves = 100% directional
    return [
        SimpleNamespace(bid=100.00, ask=100.04),
        SimpleNamespace(bid=100.50, ask=100.54),
        SimpleNamespace(bid=101.00, ask=101.04),
        SimpleNamespace(bid=102.00, ask=102.04),
    ]


def bearish_ticks():
    # Net move ~2.0 downward; 4/4 down-moves = 100% directional
    return [
        SimpleNamespace(bid=102.00, ask=102.04),
        SimpleNamespace(bid=101.00, ask=101.04),
        SimpleNamespace(bid=100.50, ask=100.54),
        SimpleNamespace(bid=100.00, ask=100.04),
    ]


def test_quick_loop_holds_and_retries_when_m1_candles_unavailable(monkeypatch):
    from src.quick_scalp_loop import run_quick_scalp_loop

    events = []
    sleeps = []

    class FakeMt5:
        def symbol_select(self, symbol, enabled):
            events.append(("select", symbol, enabled))
            return True

    class FakeExecutor:
        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            events.append(("positions", symbol, comment_prefix))
            return []

        def open_strategy_trade(self, **kwargs):
            raise AssertionError("missing candles should not open a trade")

    def raise_missing_candles(mt5_module, symbol, count=30):
        raise RuntimeError(f"Not enough TIMEFRAME_M1 candle data for {symbol}")

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", raise_missing_candles)

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=2,
        sleep_fn=lambda seconds: sleeps.append(seconds),
        log_fn=lambda message: events.append(("log", message)),
    )

    assert sleeps == [1]
    assert ("positions", "XAUUSD", "quick-scalp") in events
    assert any("reason=candle_data_unavailable" in event[1] for event in events if event[0] == "log")


def test_tick_in_out_mode_opens_from_tick_momentum_without_m1_candles(monkeypatch):
    from src.quick_scalp_loop import run_quick_scalp_loop
    from src.strategy.breakout import BreakoutDirection

    events = []
    opened = []

    class FakeMt5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1

        def symbol_select(self, symbol, enabled):
            return True

        def symbol_info(self, symbol):
            return SimpleNamespace(
                point=0.01,
                trade_tick_size=0.01,
                trade_tick_value=1.0,
                trade_stops_level=0,
                trade_freeze_level=0,
            )

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.10, bid=99.90)

        def order_calc_margin(self, order_type, symbol, lot, price):
            return 1.0

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=1)

    def raise_missing_candles(mt5_module, symbol, count=30):
        raise RuntimeError(f"Not enough TIMEFRAME_M1 candle data for {symbol}")

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", raise_missing_candles)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=100: bullish_ticks())
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (96.0, 105.0))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        max_spread_pips=4.0,
        tick_in_out_mode=True,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert opened
    assert opened[0]["direction"] is BreakoutDirection.BULLISH
    assert any("mode=tick_in_out" in message and "QUICK TRADE OPENED" in message for message in events)


def test_quick_loop_shadow_only_collects_data_without_opening_trade(tmp_path, monkeypatch):
    import csv

    from src.quick_scalp_loop import run_quick_scalp_loop

    events = []
    snapshots = []
    opened = []
    monkeypatch.setattr("src.quick_scalp_loop.save_training_snapshot", lambda state: snapshots.append(state))

    class FakeMt5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1

        def account_info(self):
            return SimpleNamespace(login=123, balance=100.0, equity=100.0, trade_allowed=True)

        def history_get(self):
            return []

        def symbol_select(self, symbol, enabled):
            return True

        def symbol_info(self, symbol):
            return SimpleNamespace(point=0.01, trade_tick_size=0.01, trade_tick_value=1.0)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(time=4, ask=102.04, bid=102.00)

        def order_calc_margin(self, order_type, symbol, lot, price):
            return 1.0

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            raise AssertionError("shadow-only mode must not open trades")

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: [])
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=100: bullish_ticks())
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (96.0, 105.0))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.05,
        poll_seconds=1,
        max_loops=1,
        max_spread_pips=4.0,
        tick_in_out_mode=True,
        shadow_only=True,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert opened == []
    assert snapshots
    assert snapshots[0]["trading"]["status"] == "shadow_only_training"
    assert snapshots[0]["trading"]["shadow_learning"] is True
    assert any("QUICK SHADOW ONLY ACTIVE" in message for message in events)
    journal = tmp_path / "quick_shadow_trades.csv"
    rows = list(csv.DictReader(journal.open(newline="", encoding="utf-8")))
    assert rows
    assert rows[-1]["status"] == "pending"
    assert rows[-1]["direction"] == "BULLISH"


def test_tick_in_out_mode_blocks_when_estimated_profit_is_too_small(monkeypatch):
    from src.quick_scalp_loop import run_quick_scalp_loop

    events = []
    opened = []

    class FakeMt5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1

        def account_info(self):
            return SimpleNamespace(balance=100.0, equity=100.0)

        def symbol_select(self, symbol, enabled):
            return True

        def symbol_info(self, symbol):
            return SimpleNamespace(point=0.01, trade_tick_size=0.01, trade_tick_value=1.0)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.10, bid=99.90)

        def order_calc_margin(self, order_type, symbol, lot, price):
            return 1.0

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=1)

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: [])
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=100: bullish_ticks())
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (96.0, 105.0))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        max_spread_pips=4.0,
        tick_in_out_mode=True,
        min_estimated_profit=5.0,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert opened == []
    assert any("reason=estimated_profit_too_small" in message for message in events)


def test_tick_in_out_mode_blocks_when_tick_quality_reverses_before_entry(monkeypatch):
    from src.quick_scalp_loop import run_quick_scalp_loop
    from src.strategy.breakout import BreakoutDirection

    events = []
    opened = []

    class FakeMt5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1

        def account_info(self):
            return SimpleNamespace(balance=100.0, equity=100.0)

        def symbol_select(self, symbol, enabled):
            return True

        def symbol_info(self, symbol):
            return SimpleNamespace(
                point=0.01,
                trade_tick_size=0.01,
                trade_tick_value=1.0,
                trade_stops_level=0,
                trade_freeze_level=0,
            )

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.21, bid=100.17)

        def order_calc_margin(self, order_type, symbol, lot, price):
            return 1.0

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=1)

    noisy_ticks = [
        SimpleNamespace(bid=100.00, ask=100.04),
        SimpleNamespace(bid=100.05, ask=100.09),
        SimpleNamespace(bid=100.12, ask=100.16),
        SimpleNamespace(bid=100.20, ask=100.24),
        SimpleNamespace(bid=100.17, ask=100.21),
    ]

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: [])
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=100: noisy_ticks)
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_tick_direction",
        lambda ticks, point=0.0, fast=False: SimpleNamespace(
            direction=BreakoutDirection.BULLISH,
            reason="tick_micro_momentum",
            tick_count=len(ticks),
            net_move=0.17,
            up_moves=3,
            down_moves=1,
        ),
    )
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (99.80, 100.80))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.05,
        poll_seconds=1,
        max_loops=1,
        max_spread_pips=5.0,
        tick_in_out_mode=True,
        min_estimated_profit=0.05,
        execution_buffer=0.02,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert opened == []
    assert any("reason=quality_no_last_tick_continuation" in message for message in events)


def test_tick_in_out_mode_blocks_when_shadow_policy_is_unapproved(tmp_path, monkeypatch):
    import json

    from src.quick_scalp_loop import run_quick_scalp_loop

    events = []
    opened = []
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "allowed": False,
                "reason": "insufficient_shadow_samples",
                "policy": {
                    "min_estimated_profit": 0.05,
                    "min_consistency": 0.60,
                    "max_pullback_ratio": 0.65,
                    "require_quality_allowed": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeMt5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1

        def account_info(self):
            return SimpleNamespace(balance=100.0, equity=100.0)

        def symbol_select(self, symbol, enabled):
            return True

        def symbol_info(self, symbol):
            return SimpleNamespace(point=0.01, trade_tick_size=0.01, trade_tick_value=1.0)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.10, bid=99.90)

        def order_calc_margin(self, order_type, symbol, lot, price):
            return 1.0

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            raise AssertionError("unapproved shadow policy must not open trades")

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: [])
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=100: bullish_ticks())
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (96.0, 105.0))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.05,
        poll_seconds=1,
        max_loops=1,
        max_spread_pips=4.0,
        tick_in_out_mode=True,
        min_estimated_profit=0.05,
        execution_buffer=0.02,
        shadow_policy_enabled=True,
        shadow_policy_path=policy_path,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert opened == []
    assert any("reason=insufficient_shadow_samples" in message for message in events)


def test_tick_in_out_mode_allows_when_shadow_policy_matches_features(tmp_path, monkeypatch):
    import json

    from src.quick_scalp_loop import run_quick_scalp_loop
    from src.strategy.breakout import BreakoutDirection

    events = []
    opened = []
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "allowed": True,
                "reason": "shadow_policy_ok",
                "policy": {
                    "min_estimated_profit": 0.05,
                    "min_consistency": 0.60,
                    "max_pullback_ratio": 0.65,
                    "require_quality_allowed": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeMt5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1

        def account_info(self):
            return SimpleNamespace(balance=100.0, equity=100.0)

        def symbol_select(self, symbol, enabled):
            return True

        def symbol_info(self, symbol):
            return SimpleNamespace(
                point=0.01,
                trade_tick_size=0.01,
                trade_tick_value=1.0,
                trade_stops_level=0,
                trade_freeze_level=0,
            )

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.10, bid=99.90)

        def order_calc_margin(self, order_type, symbol, lot, price):
            return 1.0

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=1)

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: [])
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=100: bullish_ticks())
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (96.0, 105.0))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.05,
        poll_seconds=1,
        max_loops=1,
        max_spread_pips=4.0,
        tick_in_out_mode=True,
        min_estimated_profit=0.05,
        execution_buffer=0.02,
        shadow_policy_enabled=True,
        shadow_policy_path=policy_path,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert opened
    assert opened[0]["direction"] is BreakoutDirection.BULLISH


def test_tick_in_out_mode_flips_execution_direction_for_approved_inverted_policy(tmp_path, monkeypatch):
    import json

    from src.quick_scalp_loop import run_quick_scalp_loop
    from src.strategy.breakout import BreakoutDirection

    events = []
    opened = []
    level_directions = []
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "allowed": True,
                "reason": "shadow_policy_ok",
                "policy": {
                    "min_estimated_profit": 0.05,
                    "min_consistency": 0.60,
                    "max_pullback_ratio": 0.65,
                    "require_quality_allowed": True,
                    "invert_direction": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeMt5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1

        def account_info(self):
            return SimpleNamespace(balance=100.0, equity=100.0)

        def symbol_select(self, symbol, enabled):
            return True

        def symbol_info(self, symbol):
            return SimpleNamespace(
                point=0.01,
                trade_tick_size=0.01,
                trade_tick_value=1.0,
                trade_stops_level=0,
                trade_freeze_level=0,
            )

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.10, bid=99.90)

        def order_calc_margin(self, order_type, symbol, lot, price):
            return 1.0

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=1)

    def fake_levels(**kwargs):
        level_directions.append(kwargs["direction"])
        if kwargs["direction"] is BreakoutDirection.BULLISH:
            return 96.0, 105.0
        return 104.0, 95.0

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: [])
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=100: bullish_ticks())
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", fake_levels)

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.05,
        poll_seconds=1,
        max_loops=1,
        max_spread_pips=4.0,
        tick_in_out_mode=True,
        min_estimated_profit=0.05,
        execution_buffer=0.02,
        shadow_policy_enabled=True,
        shadow_policy_path=policy_path,
        allow_inverted_shadow_policy=True,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert opened
    assert opened[0]["direction"] is BreakoutDirection.BEARISH
    assert level_directions == [BreakoutDirection.BULLISH, BreakoutDirection.BEARISH]
    assert any("QUICK POLICY INVERSION" in message for message in events)


def test_approved_shadow_policy_can_override_tick_quality_block(tmp_path, monkeypatch):
    import json

    from src.quick_scalp_loop import run_quick_scalp_loop
    from src.strategy.breakout import BreakoutDirection

    events = []
    opened = []
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "allowed": True,
                "reason": "shadow_policy_ok",
                "policy": {
                    "min_estimated_profit": 0.05,
                    "min_consistency": 0.60,
                    "max_pullback_ratio": 0.65,
                    "require_quality_allowed": False,
                    "invert_direction": True,
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeMt5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1

        def account_info(self):
            return SimpleNamespace(balance=100.0, equity=100.0)

        def symbol_select(self, symbol, enabled):
            return True

        def symbol_info(self, symbol):
            return SimpleNamespace(
                point=0.01,
                trade_tick_size=0.01,
                trade_tick_value=1.0,
                trade_stops_level=0,
                trade_freeze_level=0,
            )

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.10, bid=99.90)

        def order_calc_margin(self, order_type, symbol, lot, price):
            return 1.0

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=1)

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: [])
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=100: bullish_ticks())
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (96.0, 105.0))
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_tick_entry_quality",
        lambda *args, **kwargs: SimpleNamespace(
            allowed=False,
            reason="quality_no_last_tick_continuation",
            impulse=2.0,
            pullback=1.0,
            pullback_ratio=0.5,
            continuation=-0.1,
            consistency=1.0,
        ),
    )

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.05,
        poll_seconds=1,
        max_loops=1,
        max_spread_pips=4.0,
        tick_in_out_mode=True,
        min_estimated_profit=0.05,
        execution_buffer=0.02,
        shadow_policy_enabled=True,
        shadow_policy_path=policy_path,
        allow_inverted_shadow_policy=True,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert opened
    assert opened[0]["direction"] is BreakoutDirection.BEARISH
    assert not any("QUICK HOLD" in message and "quality_no_last_tick_continuation" in message for message in events)


def test_tick_in_out_mode_stops_opening_after_live_pilot_cap(monkeypatch):
    from src.quick_scalp_loop import run_quick_scalp_loop

    events = []
    opened = []

    class FakeMt5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1

        def account_info(self):
            return SimpleNamespace(balance=100.0, equity=100.0)

        def symbol_select(self, symbol, enabled):
            return True

        def symbol_info(self, symbol):
            return SimpleNamespace(
                point=0.01,
                trade_tick_size=0.01,
                trade_tick_value=1.0,
                trade_stops_level=0,
                trade_freeze_level=0,
            )

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.10, bid=99.90)

        def order_calc_margin(self, order_type, symbol, lot, price):
            return 1.0

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=len(opened))

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: [])
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=100: bullish_ticks())
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (96.0, 105.0))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.05,
        poll_seconds=1,
        max_loops=2,
        max_spread_pips=4.0,
        tick_in_out_mode=True,
        min_estimated_profit=0.05,
        execution_buffer=0.02,
        live_pilot_max_trades=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert len(opened) == 1
    assert any("reason=live_pilot_trade_cap" in message for message in events)


def test_tick_in_out_mode_entry_cooldown_blocks_immediate_duplicate_open(monkeypatch):
    from src.quick_scalp_loop import run_quick_scalp_loop

    events = []
    opened = []
    clock = {"now": 1000.0}

    class FakeMt5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1

        def account_info(self):
            return SimpleNamespace(balance=100.0, equity=100.0, trade_allowed=True)

        def symbol_select(self, symbol, enabled):
            return True

        def symbol_info(self, symbol):
            return SimpleNamespace(
                point=0.01,
                trade_tick_size=0.01,
                trade_tick_value=1.0,
                trade_stops_level=0,
                trade_freeze_level=0,
            )

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.10, bid=99.90)

        def order_calc_margin(self, order_type, symbol, lot, price):
            return 1.0

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=len(opened))

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: [])
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=100: bullish_ticks())
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (96.0, 105.0))
    monkeypatch.setattr("src.quick_scalp_loop.time", lambda: clock["now"])

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.05,
        poll_seconds=1,
        max_loops=2,
        max_spread_pips=4.0,
        tick_in_out_mode=True,
        min_estimated_profit=0.05,
        execution_buffer=0.02,
        entry_cooldown_seconds=20,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert len(opened) == 1
    assert any("reason=entry_cooldown" in message for message in events)


def test_tick_in_out_mode_live_pilot_cap_counts_broker_history_after_restart(monkeypatch):
    from src.quick_scalp_loop import run_quick_scalp_loop

    events = []
    opened = []

    class FakeMt5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1

        def account_info(self):
            return SimpleNamespace(login=123, balance=100.0, equity=100.0, trade_allowed=True)

        def history_get(self):
            return [
                SimpleNamespace(
                    symbol="XAUUSD",
                    profit=0.10,
                    close_time="2026.08.13 15:21:40",
                    comment="quick-scalp",
                )
            ]

        def symbol_select(self, symbol, enabled):
            return True

        def symbol_info(self, symbol):
            return SimpleNamespace(
                point=0.01,
                trade_tick_size=0.01,
                trade_tick_value=1.0,
                trade_stops_level=0,
                trade_freeze_level=0,
            )

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.10, bid=99.90)

        def order_calc_margin(self, order_type, symbol, lot, price):
            return 1.0

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=len(opened))

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: [])
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=100: bullish_ticks())
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (96.0, 105.0))
    monkeypatch.setattr(
        "src.quick_scalp_loop.datetime",
        SimpleNamespace(
            now=lambda tz=None: __import__("datetime").datetime(2026, 8, 13, 16, 0, tzinfo=tz),
            fromisoformat=__import__("datetime").datetime.fromisoformat,
        ),
    )

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.05,
        poll_seconds=1,
        max_loops=1,
        max_spread_pips=4.0,
        tick_in_out_mode=True,
        min_estimated_profit=0.05,
        execution_buffer=0.02,
        live_pilot_max_trades=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert opened == []
    assert any("reason=live_pilot_trade_cap" in message and "opened=1" in message for message in events)


def test_quick_session_state_reuses_same_day_start_equity(tmp_path):
    from datetime import datetime, timezone

    from src.quick_scalp_loop import load_or_create_quick_session_state

    path = tmp_path / "quick_session_state.json"
    first = load_or_create_quick_session_state(
        symbol="XAUUSD",
        account_info=SimpleNamespace(login=123, balance=500.0, equity=500.0),
        path=path,
        now=datetime(2026, 8, 13, 8, 0, tzinfo=timezone.utc),
    )
    second = load_or_create_quick_session_state(
        symbol="XAUUSD",
        account_info=SimpleNamespace(login=123, balance=540.0, equity=540.0),
        path=path,
        now=datetime(2026, 8, 13, 9, 0, tzinfo=timezone.utc),
    )

    assert first["start_equity"] == pytest.approx(500.0)
    assert second["start_equity"] == pytest.approx(500.0)
    assert second["last_equity"] == pytest.approx(540.0)


def test_quick_session_state_raises_same_day_start_from_broker_history_override(tmp_path):
    from datetime import datetime, timezone

    from src.quick_scalp_loop import load_or_create_quick_session_state, save_quick_session_state

    path = tmp_path / "quick_session_state.json"
    save_quick_session_state(
        {
            "key": "2026-08-13:123:XAUUSD",
            "symbol": "XAUUSD",
            "login": 123,
            "date": "2026-08-13",
            "start_equity": 591.57,
            "last_equity": 591.57,
            "updated_at": "2026-08-13T08:00:00+00:00",
        },
        path=path,
    )

    state = load_or_create_quick_session_state(
        symbol="XAUUSD",
        account_info=SimpleNamespace(login=123, balance=591.95, equity=591.95),
        path=path,
        now=datetime(2026, 8, 13, 9, 0, tzinfo=timezone.utc),
        start_equity_override=596.0,
    )

    assert state["start_equity"] == pytest.approx(596.0)
    assert state["last_equity"] == pytest.approx(591.95)


def test_closed_quick_pnl_today_sums_broker_history_for_symbol_and_date():
    from datetime import datetime, timezone

    from src.quick_scalp_loop import closed_quick_pnl_today

    class FakeMt5:
        def history_get(self):
            return [
                SimpleNamespace(symbol="XAUUSD", profit=-0.62, close_time="2026.08.13 15:12:33", comment="quick-scalp"),
                SimpleNamespace(symbol="XAUUSD", profit=0.30, close_time="2026.08.13 15:13:19", comment="quick-scalp"),
                SimpleNamespace(symbol="XAUUSD", profit=9.99, close_time="2026.08.12 15:13:19", comment="quick-scalp"),
                SimpleNamespace(symbol="EURUSD", profit=9.99, close_time="2026.08.13 15:13:19", comment="quick-scalp"),
                SimpleNamespace(symbol="XAUUSD", profit=9.99, close_time="2026.08.13 15:13:19", comment="manual"),
            ]

    assert closed_quick_pnl_today(
        FakeMt5(),
        "XAUUSD",
        now=datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc),
    ) == pytest.approx(-0.32)


def test_closed_quick_daily_metrics_ignore_blank_close_time_rows():
    from datetime import datetime, timezone

    from src.quick_scalp_loop import closed_quick_pnl_today, closed_quick_trade_count_today

    class FakeMt5:
        def history_get(self):
            return [
                SimpleNamespace(symbol="XAUUSD", profit=-5.00, close_time="", comment="quick-scalp"),
                SimpleNamespace(symbol="XAUUSD", profit=0.21, close_time="2026.08.13 15:21:40", comment="quick-scalp"),
                SimpleNamespace(symbol="XAUUSD", profit=0.08, close_time="2026.08.13 15:24:19", comment="quick-scalp"),
            ]

    now = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)

    assert closed_quick_pnl_today(FakeMt5(), "XAUUSD", now=now) == pytest.approx(0.29)
    assert closed_quick_trade_count_today(FakeMt5(), "XAUUSD", now=now) == 2


def test_quick_history_edge_blocks_negative_real_history():
    from src.quick_scalp_loop import resolve_quick_history_edge

    class FakeMt5:
        def history_get(self):
            return [
                SimpleNamespace(symbol="XAUUSD", profit=0.10, comment="quick-scalp"),
                SimpleNamespace(symbol="XAUUSD", profit=-0.30, comment="quick-scalp"),
                SimpleNamespace(symbol="XAUUSD", profit=9.99, comment="manual"),
            ]

    edge = resolve_quick_history_edge(
        FakeMt5(),
        "XAUUSD",
        min_trades=2,
        min_win_rate=0.50,
        min_profit_factor=1.20,
        min_expectancy=0.02,
        max_loss_streak=3,
    )

    assert edge.allowed is False
    assert edge.reason == "unproven_history_edge"
    assert edge.trade_count == 2
    assert edge.net_profit == pytest.approx(-0.20)


def test_quick_history_edge_allows_positive_real_history():
    from src.quick_scalp_loop import resolve_quick_history_edge

    class FakeMt5:
        def history_get(self):
            return [
                SimpleNamespace(symbol="XAUUSD", profit=0.12, comment="quick-scalp"),
                SimpleNamespace(symbol="XAUUSD", profit=0.08, comment="quick-scalp"),
                SimpleNamespace(symbol="XAUUSD", profit=-0.03, comment="quick-scalp"),
                SimpleNamespace(symbol="XAUUSD", profit=0.09, comment="quick-scalp"),
            ]

    edge = resolve_quick_history_edge(
        FakeMt5(),
        "XAUUSD",
        min_trades=4,
        min_win_rate=0.55,
        min_profit_factor=1.20,
        min_expectancy=0.02,
        max_loss_streak=3,
    )

    assert edge.allowed is True
    assert edge.reason == "history_edge_ok"
    assert edge.profit_factor > 1.20


def test_shadow_journal_records_and_resolves_virtual_buy(tmp_path):
    import csv

    from src.quick_scalp_loop import update_quick_shadow_journal
    from src.strategy.breakout import BreakoutDirection

    class FakeMt5:
        def symbol_info(self, symbol):
            return SimpleNamespace(point=0.01, trade_tick_size=0.01, trade_tick_value=1.0)

    path = tmp_path / "shadow.csv"
    update_quick_shadow_journal(
        mt5_module=FakeMt5(),
        symbol="XAUUSD",
        direction=BreakoutDirection.BULLISH,
        tick=SimpleNamespace(time=10, bid=100.00, ask=100.04),
        profit_target=0.05,
        max_loss=0.50,
        lot=0.01,
        path=path,
    )
    update_quick_shadow_journal(
        mt5_module=FakeMt5(),
        symbol="XAUUSD",
        direction=None,
        tick=SimpleNamespace(time=12, bid=100.10, ask=100.14),
        profit_target=0.05,
        max_loss=0.50,
        lot=0.01,
        path=path,
    )

    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["status"] == "resolved"
    assert rows[0]["label_outcome"] == "win"
    assert rows[0]["label_tp_before_sl"] == "1"
    assert float(rows[0]["label_max_favorable"]) >= 0.05


def test_shadow_journal_records_decision_features(tmp_path):
    import csv

    from src.quick_scalp_loop import update_quick_shadow_journal
    from src.strategy.breakout import BreakoutDirection

    class FakeMt5:
        def symbol_info(self, symbol):
            return SimpleNamespace(point=0.01, trade_tick_size=0.01, trade_tick_value=1.0)

    path = tmp_path / "shadow.csv"
    update_quick_shadow_journal(
        mt5_module=FakeMt5(),
        symbol="XAUUSD",
        direction=BreakoutDirection.BULLISH,
        tick=SimpleNamespace(time=10, bid=100.00, ask=100.04),
        profit_target=0.05,
        max_loss=0.50,
        lot=0.01,
        metadata={
            "tick_count": 8,
            "tick_net_move": 0.12,
            "tick_up_moves": 6,
            "tick_down_moves": 1,
            "tick_directional_consistency": 6 / 7,
            "estimated_tick_profit": 0.08,
            "executable_price_profit": 0.08,
            "quality_allowed": True,
            "quality_reason": "quality_ok",
            "quality_impulse": 0.12,
            "quality_pullback": 0.01,
            "quality_pullback_ratio": 0.083333,
            "quality_continuation": 0.02,
            "quality_consistency": 0.857143,
        },
        path=path,
    )

    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert rows[0]["spread"] == "0.040000"
    assert rows[0]["tick_count"] == "8"
    assert rows[0]["tick_up_moves"] == "6"
    assert rows[0]["quality_allowed"] == "1"
    assert rows[0]["quality_reason"] == "quality_ok"
    assert rows[0]["estimated_tick_profit"] == "0.080000"


def test_shadow_journal_deduplicates_same_tick_decision(tmp_path):
    import csv

    from src.quick_scalp_loop import update_quick_shadow_journal
    from src.strategy.breakout import BreakoutDirection

    class FakeMt5:
        def symbol_info(self, symbol):
            return SimpleNamespace(point=0.01, trade_tick_size=0.01, trade_tick_value=1.0)

    path = tmp_path / "shadow.csv"
    tick = SimpleNamespace(time=10, bid=100.00, ask=100.04)
    for _ in range(3):
        update_quick_shadow_journal(
            mt5_module=FakeMt5(),
            symbol="XAUUSD",
            direction=BreakoutDirection.BULLISH,
            tick=tick,
            profit_target=0.05,
            max_loss=0.50,
            lot=0.01,
            path=path,
        )

    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["decision_time"] == "1970-01-01T00:00:10+00:00"


def test_shadow_journal_records_and_resolves_virtual_sell_loss(tmp_path):
    import csv

    from src.quick_scalp_loop import update_quick_shadow_journal
    from src.strategy.breakout import BreakoutDirection

    class FakeMt5:
        def symbol_info(self, symbol):
            return SimpleNamespace(point=0.01, trade_tick_size=0.01, trade_tick_value=1.0)

    path = tmp_path / "shadow.csv"
    update_quick_shadow_journal(
        mt5_module=FakeMt5(),
        symbol="XAUUSD",
        direction=BreakoutDirection.BEARISH,
        tick=SimpleNamespace(time=10, bid=100.00, ask=100.04),
        profit_target=0.05,
        max_loss=0.05,
        lot=0.01,
        path=path,
    )
    update_quick_shadow_journal(
        mt5_module=FakeMt5(),
        symbol="XAUUSD",
        direction=None,
        tick=SimpleNamespace(time=12, bid=100.04, ask=100.08),
        profit_target=0.05,
        max_loss=0.05,
        lot=0.01,
        path=path,
    )

    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["status"] == "resolved"
    assert rows[0]["label_outcome"] == "loss"
    assert rows[0]["label_tp_before_sl"] == "0"
    assert float(rows[0]["label_max_adverse"]) >= 0.05


def test_quick_shadow_edge_blocks_small_or_losing_shadow_sample(tmp_path):
    import csv

    from src.quick_scalp_loop import QUICK_SHADOW_JOURNAL_FIELDS, resolve_quick_shadow_edge

    path = tmp_path / "shadow.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUICK_SHADOW_JOURNAL_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "symbol": "XAUUSD",
                "status": "resolved",
                "label_outcome": "loss",
                "label_max_favorable": "0.01",
                "label_max_adverse": "0.05",
            }
        )

    edge = resolve_quick_shadow_edge(path=path, symbol="XAUUSD", min_samples=2)

    assert edge.allowed is False
    assert edge.reason == "shadow_edge_unproven"
    assert edge.sample_count == 1


def test_quick_shadow_edge_allows_large_positive_shadow_sample(tmp_path):
    import csv

    from src.quick_scalp_loop import QUICK_SHADOW_JOURNAL_FIELDS, resolve_quick_shadow_edge

    path = tmp_path / "shadow.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUICK_SHADOW_JOURNAL_FIELDS)
        writer.writeheader()
        for index in range(8):
            writer.writerow(
                {
                    "symbol": "XAUUSD",
                    "status": "resolved",
                    "label_outcome": "win" if index != 3 else "loss",
                    "label_max_favorable": "0.08",
                    "label_max_adverse": "0.02",
                }
            )

    edge = resolve_quick_shadow_edge(
        path=path,
        symbol="XAUUSD",
        min_samples=8,
        min_win_rate=0.70,
        min_expectancy_proxy=0.02,
        max_loss_streak=2,
    )

    assert edge.allowed is True
    assert edge.reason == "shadow_edge_ok"
    assert edge.win_rate == pytest.approx(0.875)


def test_shadow_edge_can_lift_only_insufficient_positive_broker_sample():
    from src.quick_scalp_loop import QuickHistoryEdge, QuickShadowEdge, should_block_for_unproven_edge

    history = QuickHistoryEdge(False, "unproven_history_edge", 4, 0.05, 0.50, 1.1, 0.0125, 1)
    shadow = QuickShadowEdge(True, "shadow_edge_ok", 80, 0.65, 0.08, 0.03, 0.05, 2)

    assert should_block_for_unproven_edge(history, shadow) is False


def test_shadow_edge_cannot_lift_negative_broker_sample():
    from src.quick_scalp_loop import QuickHistoryEdge, QuickShadowEdge, should_block_for_unproven_edge

    history = QuickHistoryEdge(False, "unproven_history_edge", 34, -4.05, 0.44, 0.75, -0.119, 8)
    shadow = QuickShadowEdge(True, "shadow_edge_ok", 80, 0.65, 0.08, 0.03, 0.05, 2)

    assert should_block_for_unproven_edge(history, shadow) is True


def test_shadow_policy_gate_blocks_inversion_until_reviewed(tmp_path):
    import json

    from src.quick_scalp_loop import resolve_shadow_policy_entry_gate

    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "allowed": True,
                "reason": "shadow_policy_ok",
                "policy": {
                    "min_estimated_profit": 0.05,
                    "min_consistency": 0.60,
                    "max_pullback_ratio": 0.65,
                    "require_quality_allowed": False,
                    "allowed_quality_reasons": [],
                    "invert_direction": True,
                },
            }
        ),
        encoding="utf-8",
    )

    allowed, reason, _payload = resolve_shadow_policy_entry_gate(
        enabled=True,
        policy_path=policy_path,
        features={
            "estimated_tick_profit": 0.20,
            "tick_directional_consistency": 0.75,
            "quality_pullback_ratio": 0.10,
            "quality_allowed": False,
            "quality_reason": "quality_ok",
        },
    )

    assert allowed is False
    assert reason == "shadow_policy_inversion_requires_review"

    allowed, reason, _payload = resolve_shadow_policy_entry_gate(
        enabled=True,
        policy_path=policy_path,
        allow_inverted_policy=True,
        features={
            "estimated_tick_profit": 0.20,
            "tick_directional_consistency": 0.75,
            "quality_pullback_ratio": 0.10,
            "quality_allowed": False,
            "quality_reason": "quality_ok",
        },
    )

    assert allowed is True
    assert reason == "shadow_policy_ok"


def test_quick_loop_daily_target_uses_persisted_start_equity(tmp_path, monkeypatch):
    from src.quick_scalp_loop import run_quick_scalp_loop, save_quick_session_state

    events = []
    closed = []
    monkeypatch.chdir(tmp_path)
    session_path = tmp_path / "quick_session_state.json"
    save_quick_session_state(
        {
            "key": "2026-08-13:123:XAUUSD",
            "symbol": "XAUUSD",
            "login": 123,
            "date": "2026-08-13",
            "start_equity": 500.0,
            "last_equity": 500.0,
            "updated_at": "2026-08-13T08:00:00+00:00",
        },
        path=session_path,
    )
    monkeypatch.setattr("src.quick_scalp_loop.QUICK_SESSION_STATE_FILE", str(session_path))

    class FakeMt5:
        def account_info(self):
            return SimpleNamespace(login=123, balance=551.0, equity=551.0, margin_free=551.0, trade_allowed=True)

    class FakeExecutor:
        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return [SimpleNamespace(ticket=1)]

        def close_position(self, position, comment):
            closed.append((position.ticket, comment))

    result = run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.05,
        poll_seconds=1,
        max_loops=1,
        daily_profit_target=50.0,
        daily_max_loss=5.0,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert result == "daily_profit_target_hit"
    assert closed == [(1, "DAILY-TARGET-EXIT")]
    assert any("QUICK DAILY TARGET HIT" in message and "pnl=51.00" in message for message in events)


def test_quick_loop_daily_loss_shadow_mode_collects_data_without_trading(tmp_path, monkeypatch):
    from src.quick_scalp_loop import run_quick_scalp_loop, save_quick_session_state

    events = []
    snapshots = []
    opened = []
    monkeypatch.chdir(tmp_path)
    session_path = tmp_path / "quick_session_state.json"
    save_quick_session_state(
        {
            "key": "2026-08-13:123:XAUUSD",
            "symbol": "XAUUSD",
            "login": 123,
            "date": "2026-08-13",
            "start_equity": 100.0,
            "last_equity": 100.0,
            "updated_at": "2026-08-13T08:00:00+00:00",
        },
        path=session_path,
    )
    monkeypatch.setattr("src.quick_scalp_loop.QUICK_SESSION_STATE_FILE", str(session_path))
    monkeypatch.setattr("src.quick_scalp_loop.save_training_snapshot", lambda state: snapshots.append(state))

    class FakeMt5:
        COPY_TICKS_ALL = 0
        COPY_TICKS_INFO = 1

        def account_info(self):
            return SimpleNamespace(login=123, balance=95.0, equity=95.0, margin_free=95.0, trade_allowed=True)

        def symbol_info(self, symbol):
            return SimpleNamespace(point=0.01, trade_tick_size=0.01, trade_tick_value=1.0)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(time=3, ask=100.09, bid=100.05)

        def copy_ticks_from_pos(self, symbol, start_pos, count, flags):
            return [
                SimpleNamespace(time=1, bid=100.00, ask=100.04),
                SimpleNamespace(time=2, bid=100.02, ask=100.06),
                SimpleNamespace(time=3, bid=100.05, ask=100.09),
            ]

    class FakeExecutor:
        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            raise AssertionError("daily loss shadow mode must not open trades")

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: [])

    result = run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.05,
        poll_seconds=1,
        max_loops=1,
        daily_max_loss=4.0,
        shadow_on_daily_halt=True,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert result is None
    assert opened == []
    assert snapshots
    assert snapshots[0]["trading"]["status"] == "daily_loss_limit_hit"
    assert snapshots[0]["trading"]["shadow_learning"] is True
    assert len(snapshots[0]["market_data"]["ticks"]) == 3


def test_quick_loop_unproven_history_edge_shadow_mode_blocks_new_trades(tmp_path, monkeypatch):
    import csv

    from src.quick_scalp_loop import run_quick_scalp_loop, save_quick_session_state

    events = []
    snapshots = []
    opened = []
    monkeypatch.chdir(tmp_path)
    session_path = tmp_path / "quick_session_state.json"
    save_quick_session_state(
        {
            "key": "2026-08-13:123:XAUUSD",
            "symbol": "XAUUSD",
            "login": 123,
            "date": "2026-08-13",
            "start_equity": 100.0,
            "last_equity": 100.0,
            "updated_at": "2026-08-13T08:00:00+00:00",
        },
        path=session_path,
    )
    monkeypatch.setattr("src.quick_scalp_loop.QUICK_SESSION_STATE_FILE", str(session_path))
    monkeypatch.setattr("src.quick_scalp_loop.save_training_snapshot", lambda state: snapshots.append(state))

    class FakeMt5:
        COPY_TICKS_ALL = 0
        COPY_TICKS_INFO = 1

        def account_info(self):
            return SimpleNamespace(login=123, balance=100.0, equity=100.0, margin_free=100.0, trade_allowed=True)

        def history_get(self):
            return [
                SimpleNamespace(symbol="XAUUSD", profit=0.10, comment="quick-scalp"),
                SimpleNamespace(symbol="XAUUSD", profit=-0.30, comment="quick-scalp"),
            ]

        def symbol_info(self, symbol):
            return SimpleNamespace(point=0.01, trade_tick_size=0.01, trade_tick_value=1.0)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(time=3, ask=100.09, bid=100.05)

        def copy_ticks_from_pos(self, symbol, start_pos, count, flags):
            return bullish_ticks()

    class FakeExecutor:
        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            raise AssertionError("unproven history edge must not open trades")

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: [])

    result = run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.05,
        poll_seconds=1,
        max_loops=1,
        daily_max_loss=4.0,
        shadow_on_unproven_edge=True,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert result is None
    assert opened == []
    assert snapshots
    assert snapshots[0]["trading"]["status"] == "unproven_history_edge"
    assert snapshots[0]["trading"]["shadow_learning"] is True
    assert any("QUICK HISTORY EDGE BLOCK" in message for message in events)
    journal = tmp_path / "quick_shadow_trades.csv"
    rows = list(csv.DictReader(journal.open(newline="", encoding="utf-8")))
    assert rows
    assert rows[-1]["status"] == "pending"
    assert rows[-1]["direction"] == "BULLISH"
    assert rows[-1]["tick_count"] == "4"
    assert rows[-1]["tick_up_moves"] == "3"
    assert rows[-1]["tick_down_moves"] == "0"
    assert rows[-1]["quality_allowed"] == "1"
    assert float(rows[-1]["estimated_tick_profit"]) > 0.0


def test_tick_in_out_mode_blocks_when_executable_profit_does_not_clear_spread_cost(monkeypatch):
    from src.quick_scalp_loop import run_quick_scalp_loop

    events = []
    opened = []

    class FakeMt5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1

        def account_info(self):
            return SimpleNamespace(balance=100.0, equity=100.0, margin_free=100.0, trade_allowed=True)

        def symbol_select(self, symbol, enabled):
            return True

        def symbol_info(self, symbol):
            return SimpleNamespace(point=0.01, trade_tick_size=0.01, trade_tick_value=1.0)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.64, bid=100.30)

        def order_calc_margin(self, order_type, symbol, lot, price):
            return 1.0

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=1)

    ticks = [
        SimpleNamespace(bid=100.00, ask=100.34),
        SimpleNamespace(bid=100.15, ask=100.49),
        SimpleNamespace(bid=100.30, ask=100.64),
    ]

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: [])
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=100: ticks)
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (96.0, 105.0))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.05,
        poll_seconds=1,
        max_loops=1,
        max_spread_pips=4.0,
        tick_in_out_mode=True,
        min_estimated_profit=0.05,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert opened == []
    assert any("reason=estimated_profit_too_small" in message and "spread_cost=0.34" in message for message in events)


def test_tick_in_out_mode_applies_execution_buffer_to_required_profit(monkeypatch):
    from src.quick_scalp_loop import run_quick_scalp_loop

    events = []
    opened = []

    class FakeMt5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1

        def account_info(self):
            return SimpleNamespace(balance=100.0, equity=100.0, margin_free=100.0, trade_allowed=True)

        def symbol_select(self, symbol, enabled):
            return True

        def symbol_info(self, symbol):
            return SimpleNamespace(point=0.01, trade_tick_size=0.01, trade_tick_value=1.0)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.40, bid=100.20)

        def order_calc_margin(self, order_type, symbol, lot, price):
            return 1.0

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=1)

    ticks = [
        SimpleNamespace(bid=100.00, ask=100.20),
        SimpleNamespace(bid=100.10, ask=100.30),
        SimpleNamespace(bid=100.26, ask=100.46),
    ]

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: [])
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=100: ticks)
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (96.0, 105.0))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.05,
        poll_seconds=1,
        max_loops=1,
        max_spread_pips=5.0,
        tick_in_out_mode=True,
        min_estimated_profit=0.05,
        execution_buffer=0.02,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert opened == []
    assert any("minimum=0.07" in message and "buffer=0.02" in message for message in events)


def test_save_training_snapshot_writes_rich_weekly_ml_row(tmp_path, monkeypatch):
    import csv

    import src.quick_scalp_loop as quick_scalp_loop

    output = tmp_path / "ml_training_data.csv"
    monkeypatch.setattr(quick_scalp_loop, "ML_TRAINING_FILE", str(output))

    state = {
        "timestamp": "2026-05-11T09:07:14+00:00",
        "account": {
            "balance": 3660.67,
            "equity": 3561.23,
            "profit": -99.44,
            "currency": "KES",
        },
        "signals": {
            "tick_dir": "BULLISH",
            "m1_dir": "BEARISH",
            "fib_dir": "BULLISH",
            "fib_zone": "in_market_mover",
            "rsi": 47.89,
            "sar_dir": "BULLISH",
            "quant": {
                "hurst": 0.61,
                "reversion": 0.14,
                "ofi": -0.22,
                "kelly_lot": 0.01,
                "z_score": -0.5,
                "smoothness": 0.72,
            },
            "mtf": {"m1": "BEARISH", "m5": "BEARISH", "m15": "BULLISH", "h1": "BULLISH"},
            "confluence": {"fib_ok": True, "sar_ok": True, "rsi_ok": True},
        },
        "market_data": {
            "m1_candles": [
                {"time": 1, "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5},
                {"time": 2, "open": 100.5, "high": 101.2, "low": 99.8, "close": 100.0},
            ],
            "ticks": [
                {"time": 1, "bid": 100.0, "ask": 100.2},
                {"time": 2, "bid": 100.4, "ask": 100.6},
                {"time": 3, "bid": 100.1, "ask": 100.3},
            ],
        },
        "trading": {
            "symbol": "XAUUSD",
            "positions_count": 1,
            "status": "mixed_guidance",
            "is_tradeable": True,
            "target_value": 100.0,
            "target_progress": 0.0,
        },
    }

    quick_scalp_loop.save_training_snapshot(state)

    with open(output, newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 1
    row = rows[0]
    assert row["schema_version"] == "2"
    assert row["strategy_mode"] == "quick_scalp"
    assert row["candle_timeframe"] == "M1"
    assert row["training_week_id"] == "2026-W20"
    assert row["training_version"] == "quick-scalp-2026-W20"
    assert row["symbol"] == "XAUUSD"
    assert row["account_currency"] == "KES"
    assert row["tick_dir"] == "BULLISH"
    assert row["m1_dir"] == "BEARISH"
    assert row["mtf_m5_dir"] == "BEARISH"
    assert row["mtf_h1_dir"] == "BULLISH"
    assert row["confluence_score"] == "3"
    assert row["m1_body_ratio"] == "0.357143"
    assert row["tick_net_move"] == "0.100000"
    assert row["tick_up_moves"] == "1"
    assert row["tick_down_moves"] == "1"


def test_resolve_m1_direction_uses_latest_closed_bullish_candle():
    from src.quick_scalp_loop import resolve_m1_direction
    from src.strategy.breakout import BreakoutDirection

    candles = [
        SimpleNamespace(open=100.0, close=99.5),
        SimpleNamespace(open=99.5, close=100.2),
    ]

    assert resolve_m1_direction(candles) is BreakoutDirection.BULLISH


def test_resolve_m1_direction_uses_latest_closed_bearish_candle():
    from src.quick_scalp_loop import resolve_m1_direction
    from src.strategy.breakout import BreakoutDirection

    candles = [
        SimpleNamespace(open=100.0, close=100.5),
        SimpleNamespace(open=100.5, close=99.8),
    ]

    assert resolve_m1_direction(candles) is BreakoutDirection.BEARISH


def test_resolve_tick_direction_follows_upward_ticks():
    from src.quick_scalp_loop import resolve_tick_direction
    from src.strategy.breakout import BreakoutDirection

    ticks = [
        SimpleNamespace(bid=100.00, ask=100.04),
        SimpleNamespace(bid=100.02, ask=100.06),
        SimpleNamespace(bid=100.05, ask=100.09),
        SimpleNamespace(bid=100.09, ask=100.13),
        SimpleNamespace(bid=100.14, ask=100.18),
    ]

    guidance = resolve_tick_direction(ticks, point=0.01)

    assert guidance.direction is BreakoutDirection.BULLISH
    assert guidance.reason == "tick_momentum"


def test_resolve_tick_direction_follows_downward_ticks():
    from src.quick_scalp_loop import resolve_tick_direction
    from src.strategy.breakout import BreakoutDirection

    ticks = [
        SimpleNamespace(bid=100.14, ask=100.18),
        SimpleNamespace(bid=100.09, ask=100.13),
        SimpleNamespace(bid=100.05, ask=100.09),
        SimpleNamespace(bid=100.02, ask=100.06),
        SimpleNamespace(bid=100.00, ask=100.04),
    ]

    guidance = resolve_tick_direction(ticks, point=0.01)

    assert guidance.direction is BreakoutDirection.BEARISH
    assert guidance.reason == "tick_momentum"


def test_fast_tick_direction_uses_micro_upward_net_move():
    from src.quick_scalp_loop import resolve_tick_direction
    from src.strategy.breakout import BreakoutDirection

    ticks = [
        SimpleNamespace(bid=100.00, ask=100.04),
        SimpleNamespace(bid=100.01, ask=100.05),
        SimpleNamespace(bid=100.00, ask=100.04),
        SimpleNamespace(bid=100.02, ask=100.06),
    ]

    guidance = resolve_tick_direction(ticks, point=0.01, fast=True)

    assert guidance.direction is BreakoutDirection.BULLISH
    assert guidance.reason in {"tick_momentum", "tick_micro_momentum"}


def test_fast_tick_direction_uses_micro_downward_net_move():
    from src.quick_scalp_loop import resolve_tick_direction
    from src.strategy.breakout import BreakoutDirection

    ticks = [
        SimpleNamespace(bid=100.02, ask=100.06),
        SimpleNamespace(bid=100.01, ask=100.05),
        SimpleNamespace(bid=100.02, ask=100.06),
        SimpleNamespace(bid=100.00, ask=100.04),
    ]

    guidance = resolve_tick_direction(ticks, point=0.01, fast=True)

    assert guidance.direction is BreakoutDirection.BEARISH
    assert guidance.reason in {"tick_momentum", "tick_micro_momentum"}


def test_tick_entry_quality_allows_smooth_continuation():
    from src.quick_scalp_loop import resolve_tick_entry_quality
    from src.strategy.breakout import BreakoutDirection

    ticks = [
        SimpleNamespace(bid=100.00, ask=100.04),
        SimpleNamespace(bid=100.02, ask=100.06),
        SimpleNamespace(bid=100.05, ask=100.09),
        SimpleNamespace(bid=100.09, ask=100.13),
        SimpleNamespace(bid=100.14, ask=100.18),
    ]

    quality = resolve_tick_entry_quality(ticks, BreakoutDirection.BULLISH, point=0.01)

    assert quality.allowed is True
    assert quality.reason == "quality_ok"
    assert quality.continuation > 0.0
    assert quality.consistency == pytest.approx(1.0)


def test_tick_entry_quality_blocks_last_tick_reversal():
    from src.quick_scalp_loop import resolve_tick_entry_quality
    from src.strategy.breakout import BreakoutDirection

    ticks = [
        SimpleNamespace(bid=100.00, ask=100.04),
        SimpleNamespace(bid=100.05, ask=100.09),
        SimpleNamespace(bid=100.12, ask=100.16),
        SimpleNamespace(bid=100.20, ask=100.24),
        SimpleNamespace(bid=100.17, ask=100.21),
    ]

    quality = resolve_tick_entry_quality(ticks, BreakoutDirection.BULLISH, point=0.01)

    assert quality.allowed is False
    assert quality.reason == "quality_no_last_tick_continuation"


def test_fetch_recent_ticks_uses_mt5_copy_ticks_from_api():
    from src.quick_scalp_loop import fetch_recent_ticks

    calls = []
    expected_ticks = [SimpleNamespace(bid=100.0, ask=100.1)]

    class FakeMt5:
        COPY_TICKS_ALL = 7

        def copy_ticks_from(self, symbol, date_from, count, flags):
            calls.append((symbol, count, flags))
            return expected_ticks

    ticks = fetch_recent_ticks(FakeMt5(), "XAUUSD", count=60)

    assert ticks == expected_ticks
    assert calls == [("XAUUSD", 60, 7)]


def test_quick_loop_holds_when_tick_api_is_unavailable(monkeypatch):
    from src.quick_scalp_loop import QuickFibonacciGuidance, QuickIndicatorGuidance, run_quick_scalp_loop
    from src.strategy.breakout import BreakoutDirection

    opened = []
    events = []

    class FakeMt5:
        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.1, bid=99.9)

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=len(opened))

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: [])
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m15_candles", lambda mt5_module, symbol, count=50: [])
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: BreakoutDirection.BEARISH)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=60: None)
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fibonacci_guidance",
        lambda candles: QuickFibonacciGuidance(BreakoutDirection.BEARISH, "in_market_mover", 100.0, 90.0, 111.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_indicator_guidance",
        lambda candles: QuickIndicatorGuidance(55.0, 99.0, BreakoutDirection.BEARISH),
    )

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert opened == []
    assert any("reason=tick_api_unavailable" in message for message in events)


def test_quick_loop_closes_profitable_quick_positions_before_opening(monkeypatch):
    from src.quick_scalp_loop import QuickFibonacciGuidance, QuickGridPermission, QuickIndicatorGuidance, run_quick_scalp_loop
    from src.strategy.breakout import BreakoutDirection

    events = []
    positions = [
        SimpleNamespace(ticket=1, symbol="XAUUSD", type=1, profit=0.21),
        SimpleNamespace(ticket=2, symbol="XAUUSD", type=0, profit=0.19),
    ]

    class FakeMt5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.1, bid=99.9)

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            assert comment_prefix == "quick-scalp"
            return list(positions)

        def close_position(self, position, comment):
            events.append(("close", position.ticket, comment))
            positions.remove(position)

        def open_strategy_trade(self, **kwargs):
            events.append(("open", kwargs))
            return SimpleNamespace(ticket=3)

    candles = [
        SimpleNamespace(high=101.0, low=99.0, open=99.4, close=100.6)
        for _ in range(30)
    ]
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: candles)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m15_candles", lambda mt5_module, symbol, count=50: [])
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: BreakoutDirection.BULLISH)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=60: bullish_ticks())
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fibonacci_guidance",
        lambda candles: QuickFibonacciGuidance(BreakoutDirection.BULLISH, "golden_zone", 100.0, 90.0, 111.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_indicator_guidance",
        lambda candles: QuickIndicatorGuidance(55.0, 101.0, BreakoutDirection.BULLISH),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_grid_permission",
        lambda **kwargs: QuickGridPermission(True, "ok", 2, 0.2, 100.0),
    )
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (96.0, 105.0))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=2,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(("log", message)),
    )

    assert ("close", 1, "quick-scalp-profit-exit") in events
    assert events[-2][0] == "open"
    assert events[-2][1]["direction"] is BreakoutDirection.BULLISH
    assert events[-2][1]["comment"] == "quick-scalp"


def test_quick_loop_holds_when_tick_disagrees_with_all_structure(monkeypatch):
    """When tick is bullish but all structural signals (fib, SAR) are bearish,
    the bot should hold — tick alone without structural backing is not enough."""
    from src.quick_scalp_loop import (
        QuickFibonacciGuidance,
        QuickFvgGuidance,
        QuickGridPermission,
        QuickIndicatorGuidance,
        run_quick_scalp_loop,
    )
    from src.strategy.breakout import BreakoutDirection

    opened = []
    events = []

    class FakeMt5:
        def symbol_info(self, symbol):
            return SimpleNamespace(point=0.01)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.18, bid=100.14)

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=len(opened))

    candles = [
        SimpleNamespace(high=101.0, low=99.0, open=100.6, close=99.4)
        for _ in range(30)
    ]
    upward_ticks = [
        SimpleNamespace(bid=100.00, ask=100.04),
        SimpleNamespace(bid=100.04, ask=100.08),
        SimpleNamespace(bid=100.08, ask=100.12),
        SimpleNamespace(bid=100.14, ask=100.18),
    ]
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: candles)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m15_candles", lambda mt5_module, symbol, count=50: [])
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: BreakoutDirection.BEARISH)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=60: upward_ticks)
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fibonacci_guidance",
        lambda candles: QuickFibonacciGuidance(BreakoutDirection.BEARISH, "golden_zone", 100.0, 90.0, 111.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_indicator_guidance",
        lambda candles: QuickIndicatorGuidance(55.0, 101.0, BreakoutDirection.BEARISH),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_grid_permission",
        lambda **kwargs: QuickGridPermission(True, "ok", 1, 0.2, 100.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fvg_guidance",
        lambda *args, **kwargs: QuickFvgGuidance(BreakoutDirection.BEARISH, 101.0, 100.0, 2, 100.0, "matching_fvg"),
    )
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (96.0, 105.0))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    # Tick is bullish but fib and SAR are both bearish → signal_override → hold
    assert len(opened) == 0
    assert any("reason=signal_override" in message for message in events)


def test_quick_loop_opens_when_tick_direction_agrees_with_m1(monkeypatch):
    from src.quick_scalp_loop import (
        QuickFibonacciGuidance,
        QuickFvgGuidance,
        QuickGridPermission,
        QuickIndicatorGuidance,
        run_quick_scalp_loop,
    )
    from src.strategy.breakout import BreakoutDirection

    opened = []

    class FakeMt5:
        def symbol_info(self, symbol):
            return SimpleNamespace(point=0.01)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.18, bid=100.14)

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=len(opened))

    candles = [
        SimpleNamespace(high=101.0, low=99.0, open=99.4, close=100.6)
        for _ in range(30)
    ]
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: candles)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m15_candles", lambda mt5_module, symbol, count=50: [])
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: BreakoutDirection.BULLISH)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=60: bullish_ticks())
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fibonacci_guidance",
        lambda candles: QuickFibonacciGuidance(BreakoutDirection.BULLISH, "golden_zone", 100.0, 90.0, 111.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_indicator_guidance",
        lambda candles: QuickIndicatorGuidance(55.0, 99.0, BreakoutDirection.BULLISH),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_grid_permission",
        lambda **kwargs: QuickGridPermission(True, "ok", 1, 0.2, 100.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fvg_guidance",
        lambda *args, **kwargs: QuickFvgGuidance(BreakoutDirection.BULLISH, 99.0, 98.0, 2, 100.0, "matching_fvg"),
    )
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (96.0, 105.0))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: None,
    )

    assert len(opened) == 1
    assert opened[0]["direction"] is BreakoutDirection.BULLISH


def test_resolve_quick_fvg_guidance_finds_nearest_matching_bullish_imbalance():
    from src.quick_scalp_loop import resolve_quick_fvg_guidance
    from src.strategy.breakout import BreakoutDirection

    candles = [
        SimpleNamespace(high=100.0, low=98.0, open=99.0, close=99.5, timestamp=1),
        SimpleNamespace(high=103.0, low=99.5, open=100.0, close=102.5, timestamp=2),
        SimpleNamespace(high=105.0, low=101.0, open=102.5, close=104.0, timestamp=3),
    ]

    guidance = resolve_quick_fvg_guidance(
        candles,
        direction=BreakoutDirection.BULLISH,
        current_price=104.0,
    )

    assert guidance.allows(BreakoutDirection.BULLISH) is True
    assert guidance.bottom == 100.0
    assert guidance.top == 101.0
    assert guidance.bars_since == 1


def test_quick_loop_holds_when_selected_direction_has_no_matching_fvg(monkeypatch):
    from src.quick_scalp_loop import QuickFibonacciGuidance, QuickGridPermission, QuickIndicatorGuidance, run_quick_scalp_loop
    from src.strategy.breakout import BreakoutDirection

    opened = []
    events = []

    class FakeMt5:
        def symbol_info(self, symbol):
            return SimpleNamespace(point=0.01)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.18, bid=100.14)

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=len(opened))

    candles = [
        SimpleNamespace(high=101.0, low=99.0, open=99.4, close=100.6)
        for _ in range(30)
    ]
    m15_candles = [
        SimpleNamespace(high=100.0, low=98.0, open=99.0, close=99.5, timestamp=1),
        SimpleNamespace(high=103.0, low=99.5, open=100.0, close=102.5, timestamp=2),
        SimpleNamespace(high=105.0, low=101.0, open=102.5, close=104.0, timestamp=3),
    ]
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: candles)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m15_candles", lambda mt5_module, symbol, count=50: m15_candles)
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: BreakoutDirection.BEARISH)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=60: bearish_ticks())
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fibonacci_guidance",
        lambda candles: QuickFibonacciGuidance(BreakoutDirection.BEARISH, "in_market_mover", 104.0, 98.0, 105.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_indicator_guidance",
        lambda candles: QuickIndicatorGuidance(55.0, 101.0, BreakoutDirection.BULLISH),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_grid_permission",
        lambda **kwargs: QuickGridPermission(True, "ok", 1, 0.2, 100.0),
    )

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert opened == []
    assert any("reason=no_matching_fvg_mixed_guidance" in message for message in events)
    assert any("swing_low=98.00" in message and "swing_high=105.00" in message for message in events)


def test_quick_loop_allows_full_guidance_without_matching_fvg(monkeypatch):
    from src.quick_scalp_loop import QuickFibonacciGuidance, QuickFvgGuidance, QuickGridPermission, QuickIndicatorGuidance, run_quick_scalp_loop
    from src.strategy.breakout import BreakoutDirection

    opened = []
    events = []

    class FakeMt5:
        def symbol_info(self, symbol):
            return SimpleNamespace(point=0.01)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.18, bid=100.14)

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=len(opened))

    candles = [
        SimpleNamespace(high=101.0, low=99.0, open=99.2, close=100.8)
        for _ in range(30)
    ]
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: candles)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m15_candles", lambda mt5_module, symbol, count=50: candles)
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: BreakoutDirection.BULLISH)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=60: bullish_ticks())
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fibonacci_guidance",
        lambda candles: QuickFibonacciGuidance(BreakoutDirection.BULLISH, "golden_zone", 100.0, 90.0, 111.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_indicator_guidance",
        lambda candles: QuickIndicatorGuidance(55.0, 99.0, BreakoutDirection.BULLISH),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_grid_permission",
        lambda **kwargs: QuickGridPermission(True, "ok", 1, 0.2, 100.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fvg_guidance",
        lambda *args, **kwargs: QuickFvgGuidance(None, 0.0, 0.0, 0, 100.0, "no_matching_fvg"),
    )
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (96.0, 105.0))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert len(opened) == 1
    assert opened[0]["direction"] is BreakoutDirection.BULLISH
    assert any("guidance=full_guidance" in message and "fvg_bars_since=0" in message for message in events)


def test_quick_loop_allows_structure_override_without_matching_fvg(monkeypatch):
    from src.quick_scalp_loop import QuickFibonacciGuidance, QuickFvgGuidance, QuickGridPermission, QuickIndicatorGuidance, run_quick_scalp_loop
    from src.strategy.breakout import BreakoutDirection

    opened = []
    events = []

    class FakeMt5:
        def symbol_info(self, symbol):
            return SimpleNamespace(point=0.01)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.18, bid=100.14)

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=len(opened))

    candles = [
        SimpleNamespace(high=101.0, low=99.0, open=99.2, close=100.8)
        for _ in range(30)
    ]
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: candles)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m15_candles", lambda mt5_module, symbol, count=50: candles)
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: BreakoutDirection.BULLISH)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=60: bullish_ticks())
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fibonacci_guidance",
        lambda candles: QuickFibonacciGuidance(BreakoutDirection.BULLISH, "golden_zone", 100.0, 90.0, 111.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_indicator_guidance",
        lambda candles: QuickIndicatorGuidance(82.0, 101.0, BreakoutDirection.BEARISH),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_grid_permission",
        lambda **kwargs: QuickGridPermission(True, "ok", 1, 0.2, 100.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fvg_guidance",
        lambda *args, **kwargs: QuickFvgGuidance(None, 0.0, 0.0, 0, 100.0, "no_matching_fvg"),
    )
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (96.0, 105.0))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert len(opened) == 1
    assert opened[0]["direction"] is BreakoutDirection.BULLISH
    assert any("guidance=structure_override" in message and "fvg_bars_since=0" in message for message in events)


def test_quick_loop_holds_mixed_guidance_without_matching_fvg(monkeypatch):
    from src.quick_scalp_loop import QuickFibonacciGuidance, QuickFvgGuidance, QuickGridPermission, QuickIndicatorGuidance, run_quick_scalp_loop
    from src.strategy.breakout import BreakoutDirection

    opened = []
    events = []

    class FakeMt5:
        def symbol_info(self, symbol):
            return SimpleNamespace(point=0.01)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.18, bid=100.14)

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=len(opened))

    candles = [
        SimpleNamespace(high=101.0, low=99.0, open=99.2, close=100.8)
        for _ in range(30)
    ]
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: candles)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m15_candles", lambda mt5_module, symbol, count=50: candles)
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: BreakoutDirection.BULLISH)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=60: bullish_ticks())
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fibonacci_guidance",
        lambda candles: QuickFibonacciGuidance(BreakoutDirection.BULLISH, "golden_zone", 100.0, 90.0, 111.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_indicator_guidance",
        lambda candles: QuickIndicatorGuidance(55.0, 101.0, BreakoutDirection.BEARISH),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_grid_permission",
        lambda **kwargs: QuickGridPermission(True, "ok", 1, 0.2, 100.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fvg_guidance",
        lambda *args, **kwargs: QuickFvgGuidance(None, 0.0, 0.0, 0, 100.0, "no_matching_fvg"),
    )
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (96.0, 105.0))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert opened == []
    assert any("reason=no_matching_fvg_mixed_guidance" in message for message in events)


def test_quick_loop_holds_when_spread_is_too_wide_for_tick_scalp(monkeypatch):
    from src.quick_scalp_loop import QuickFibonacciGuidance, QuickIndicatorGuidance, run_quick_scalp_loop
    from src.strategy.breakout import BreakoutDirection

    opened = []
    events = []

    class FakeMt5:
        def symbol_info(self, symbol):
            return SimpleNamespace(point=0.01)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.50, bid=100.00)

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=len(opened))

    ticks = [
        SimpleNamespace(bid=100.00, ask=100.50),
        SimpleNamespace(bid=100.04, ask=100.54),
        SimpleNamespace(bid=100.08, ask=100.58),
        SimpleNamespace(bid=100.12, ask=100.62),
    ]
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: [])
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m15_candles", lambda mt5_module, symbol, count=50: [])
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: BreakoutDirection.BULLISH)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=60: ticks)
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fibonacci_guidance",
        lambda candles: QuickFibonacciGuidance(BreakoutDirection.BULLISH, "golden_zone", 100.0, 90.0, 111.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_indicator_guidance",
        lambda candles: QuickIndicatorGuidance(55.0, 99.0, BreakoutDirection.BULLISH),
    )

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert opened == []
    assert any("reason=spread_too_wide" in message for message in events)


def test_quick_loop_closes_when_profit_reaches_target(monkeypatch):
    from src.quick_scalp_loop import run_quick_scalp_loop

    events = []
    positions = [
        SimpleNamespace(ticket=1, symbol="XAUUSD", profit=0.21),
        SimpleNamespace(ticket=2, symbol="XAUUSD", profit=0.2),
        SimpleNamespace(ticket=3, symbol="XAUUSD", profit=0.19),
    ]

    class FakeExecutor:
        mt5_module = SimpleNamespace()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return list(positions)

        def close_position(self, position, comment):
            events.append(("close", position.ticket, comment))
            positions.remove(position)

        def open_strategy_trade(self, **kwargs):
            events.append(("open", kwargs))
            return SimpleNamespace(ticket=4)

    candles = [
        SimpleNamespace(high=101.0, low=99.0, open=99.4, close=100.6)
        for _ in range(30)
    ]
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: candles)
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: None)

    run_quick_scalp_loop(
        mt5_module=SimpleNamespace(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=3,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(("log", message)),
    )

    assert ("close", 1, "quick-scalp-profit-exit") in events
    assert ("close", 2, "quick-scalp-profit-exit") in events
    assert all(event != ("close", 3, "quick-scalp-profit-exit") for event in events)


def test_close_profitable_quick_positions_closes_exact_target_profit():
    from src.quick_scalp_loop import close_profitable_quick_positions

    events = []
    positions = [
        SimpleNamespace(ticket=1, profit=0.19),
        SimpleNamespace(ticket=2, profit=0.2),
    ]

    class FakeExecutor:
        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return list(positions)

        def close_position(self, position, comment):
            events.append(("close", position.ticket, comment))
            positions.remove(position)

    closed = close_profitable_quick_positions(
        executor=FakeExecutor(),
        symbol="XAUUSD",
        profit_target=0.2,
        log_fn=lambda message: events.append(message),
    )

    assert closed == 1
    assert ("close", 2, "quick-scalp-profit-exit") in events
    assert any("reason=profit_target" in event for event in events if isinstance(event, str))


def test_close_profitable_quick_positions_logs_basket_context_after_exact_target_exit():
    from src.quick_scalp_loop import close_profitable_quick_positions

    events = []
    positions = [
        SimpleNamespace(ticket=1, profit=-1.25),
        SimpleNamespace(ticket=2, profit=0.20),
        SimpleNamespace(ticket=3, profit=0.0),
    ]

    class FakeExecutor:
        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return list(positions)

        def close_position(self, position, comment):
            events.append(("close", position.ticket, comment))
            positions.remove(position)

    closed = close_profitable_quick_positions(
        executor=FakeExecutor(),
        symbol="XAUUSD",
        profit_target=0.2,
        log_fn=lambda message: events.append(message),
    )

    assert closed == 1
    assert ("close", 2, "quick-scalp-profit-exit") in events
    assert any("reason=profit_target" in event for event in events if isinstance(event, str))


def test_close_profitable_quick_positions_closes_target_profit_even_when_ticks_still_favor_trade():
    from src.quick_scalp_loop import close_profitable_quick_positions
    from src.strategy.breakout import BreakoutDirection

    events = []

    class FakeMt5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return [SimpleNamespace(ticket=1, type=0, profit=1.25)]

        def close_position(self, position, comment):
            events.append(("close", position.ticket, comment))

    closed = close_profitable_quick_positions(
        executor=FakeExecutor(),
        symbol="XAUUSD",
        profit_target=0.2,
        tick_direction=BreakoutDirection.BULLISH,
        log_fn=lambda message: events.append(message),
    )

    assert closed == 1
    assert ("close", 1, "quick-scalp-profit-exit") in events
    assert any("reason=profit_target" in event for event in events if isinstance(event, str))


def test_close_profitable_quick_positions_locks_profit_when_ticks_turn_against_trade():
    from src.quick_scalp_loop import close_profitable_quick_positions
    from src.strategy.breakout import BreakoutDirection

    events = []
    positions = [SimpleNamespace(ticket=1, type=0, profit=0.05)]

    class FakeMt5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return list(positions)

        def close_position(self, position, comment):
            events.append(("close", position.ticket, comment))
            positions.remove(position)

    closed = close_profitable_quick_positions(
        executor=FakeExecutor(),
        symbol="XAUUSD",
        profit_target=0.2,
        tick_direction=BreakoutDirection.BEARISH,
        log_fn=lambda message: events.append(("log", message)),
    )

    assert closed == 1
    assert ("close", 1, "quick-scalp-tick-turn-profit-exit") in events
    assert any("reason=tick_turn" in event[1] for event in events if event[0] == "log")


def test_close_profitable_quick_positions_cuts_loser_when_ticks_turn_against_trade():
    from src.quick_scalp_loop import close_profitable_quick_positions
    from src.strategy.breakout import BreakoutDirection

    events = []
    positions = [SimpleNamespace(ticket=1, type=1, profit=-0.06)]

    class FakeMt5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return list(positions)

        def close_position(self, position, comment):
            events.append(("close", position.ticket, comment))
            positions.remove(position)

    closed = close_profitable_quick_positions(
        executor=FakeExecutor(),
        symbol="XAUUSD",
        profit_target=0.05,
        max_loss=0.50,
        tick_direction=BreakoutDirection.BULLISH,
        log_fn=lambda message: events.append(("log", message)),
    )

    assert closed == 1
    assert ("close", 1, "quick-scalp-tick-turn-loss-exit") in events
    assert any("reason=tick_turn_loss" in event[1] for event in events if event[0] == "log")


def test_close_profitable_quick_positions_closes_loser_at_max_loss():
    from src.quick_scalp_loop import close_profitable_quick_positions

    events = []
    positions = [
        SimpleNamespace(ticket=1, profit=-19.99),
        SimpleNamespace(ticket=2, profit=-20.00),
    ]

    class FakeExecutor:
        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return list(positions)

        def close_position(self, position, comment):
            events.append(("close", position.ticket, comment))
            positions.remove(position)

    closed = close_profitable_quick_positions(
        executor=FakeExecutor(),
        symbol="XAUUSD",
        profit_target=0.2,
        max_loss=20.0,
        log_fn=lambda message: events.append(("log", message)),
    )

    assert closed == 1
    assert ("close", 2, "quick-scalp-loss-exit") in events
    assert all(event != ("close", 1, "quick-scalp-loss-exit") for event in events)
    assert any("reason=max_loss" in event[1] for event in events if event[0] == "log")


def test_tick_in_out_mode_enters_loss_cooldown_after_max_loss_close(monkeypatch):
    from src.quick_scalp_loop import run_quick_scalp_loop

    events = []
    opened = []
    positions = [SimpleNamespace(ticket=1, type=0, volume=0.01, profit=-0.55)]

    class FakeMt5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1

        def account_info(self):
            return SimpleNamespace(balance=100.0, equity=100.0, margin_free=100.0, trade_allowed=True)

        def symbol_select(self, symbol, enabled):
            return True

        def symbol_info(self, symbol):
            return SimpleNamespace(point=0.01, trade_tick_size=0.01, trade_tick_value=1.0)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.10, bid=99.90)

        def order_calc_margin(self, order_type, symbol, lot, price):
            return 1.0

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return list(positions)

        def close_position(self, position, comment):
            positions.remove(position)

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=2)

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: [])
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=100: bullish_ticks())
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (96.0, 105.0))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.05,
        max_loss=0.50,
        poll_seconds=1,
        max_loops=1,
        max_spread_pips=5.0,
        tick_in_out_mode=True,
        min_estimated_profit=0.05,
        execution_buffer=0.02,
        loss_cooldown_seconds=60,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert opened == []
    assert any("reason=loss_cooldown" in message for message in events)


def test_close_profitable_quick_positions_does_not_close_loser_when_max_loss_disabled():
    from src.quick_scalp_loop import close_profitable_quick_positions

    events = []

    class FakeExecutor:
        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return [SimpleNamespace(ticket=1, profit=-36.16)]

        def close_position(self, position, comment):
            raise AssertionError("max_loss=0 should leave the broker stop-loss in control")

    closed = close_profitable_quick_positions(
        executor=FakeExecutor(),
        symbol="XAUUSD",
        profit_target=0.2,
        max_loss=0.0,
        log_fn=lambda message: events.append(message),
    )

    assert closed == 0
    assert any("worst_profit=-36.16" in message for message in events)


def test_quick_loop_keeps_monitoring_when_profitable_close_is_rejected(monkeypatch):
    from src.quick_scalp_loop import run_quick_scalp_loop

    events = []
    positions = [
        SimpleNamespace(ticket=1, symbol="XAUUSD", profit=0.31),
        SimpleNamespace(ticket=2, symbol="XAUUSD", profit=0.32),
    ]

    class FakeExecutor:
        mt5_module = SimpleNamespace()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return list(positions)

        def close_position(self, position, comment):
            events.append(("close_attempt", position.ticket, comment))
            if position.ticket == 1:
                raise RuntimeError("Close trade rejected: retcode=10036")
            events.append(("close", position.ticket, comment))
            positions.remove(position)

        def open_strategy_trade(self, **kwargs):
            events.append(("open", kwargs))
            return SimpleNamespace(ticket=3)

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=2: [])
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: None)

    run_quick_scalp_loop(
        mt5_module=SimpleNamespace(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=100,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(("log", message)),
    )

    assert ("close_attempt", 1, "quick-scalp-profit-exit") in events
    assert ("close_attempt", 2, "quick-scalp-profit-exit") in events
    assert ("close", 2, "quick-scalp-profit-exit") in events
    assert any("QUICK PROFIT EXIT REJECTED XAUUSD ticket=1" in event[1] for event in events if event[0] == "log")


def test_quick_loop_opens_until_max_positions(monkeypatch):
    from src.quick_scalp_loop import QuickFibonacciGuidance, QuickGridPermission, QuickIndicatorGuidance, run_quick_scalp_loop
    from src.strategy.breakout import BreakoutDirection

    opened = []

    class FakeMt5:
        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.1, bid=99.9)

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return [
                SimpleNamespace(ticket=index, symbol=symbol, profit=-0.01)
                for index in range(len(opened))
            ]

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=len(opened))

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=2: [])
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m15_candles", lambda mt5_module, symbol, count=50: [])
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: BreakoutDirection.BEARISH)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=60: bearish_ticks())
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fibonacci_guidance",
        lambda candles: QuickFibonacciGuidance(BreakoutDirection.BEARISH, "in_market_mover", 92.0, 90.0, 111.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_indicator_guidance",
        lambda candles: QuickIndicatorGuidance(45.0, 99.0, BreakoutDirection.BEARISH),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_grid_permission",
        lambda **kwargs: QuickGridPermission(True, "ok", 5, 0.2, 100.0),
    )
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (104.0, 95.0))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=5,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: None,
    )

    assert len(opened) == 5
    assert all(order["direction"] is BreakoutDirection.BEARISH for order in opened)


def test_quick_loop_follows_m1_signal_before_opening_trade(monkeypatch):
    from src.quick_scalp_loop import QuickFibonacciGuidance, QuickGridPermission, QuickIndicatorGuidance, run_quick_scalp_loop
    from src.strategy.breakout import BreakoutDirection

    opened = []

    class FakeMt5:
        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.1, bid=99.9)

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=len(opened))

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=2: [])
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m15_candles", lambda mt5_module, symbol, count=50: [])
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: BreakoutDirection.BULLISH)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=60: bullish_ticks())
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fibonacci_guidance",
        lambda candles: QuickFibonacciGuidance(BreakoutDirection.BULLISH, "golden_zone", 96.0, 90.0, 111.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_indicator_guidance",
        lambda candles: QuickIndicatorGuidance(55.0, 99.0, BreakoutDirection.BULLISH),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_grid_permission",
        lambda **kwargs: QuickGridPermission(True, "ok", 1, 0.2, 100.0),
    )
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (104.0, 95.0))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: None,
    )

    assert len(opened) == 1
    assert opened[0]["direction"] is BreakoutDirection.BULLISH


def test_resolve_quick_fibonacci_guidance_allows_bullish_golden_zone():
    from src.quick_scalp_loop import resolve_quick_fibonacci_guidance
    from src.strategy.breakout import BreakoutDirection

    candles = [
        SimpleNamespace(high=100.0, low=90.0, close=92.0),
        SimpleNamespace(high=111.0, low=99.0, close=104.0),
        SimpleNamespace(high=110.0, low=95.0, close=96.0),
    ]

    guidance = resolve_quick_fibonacci_guidance(candles)

    assert guidance.direction is BreakoutDirection.BULLISH
    assert guidance.zone == "golden_zone"
    assert guidance.allows(BreakoutDirection.BULLISH) is True
    assert guidance.allows(BreakoutDirection.BEARISH) is False


def test_resolve_quick_fibonacci_guidance_allows_bearish_market_mover():
    from src.quick_scalp_loop import resolve_quick_fibonacci_guidance
    from src.strategy.breakout import BreakoutDirection

    candles = [
        SimpleNamespace(high=111.0, low=99.0, close=108.0),
        SimpleNamespace(high=101.0, low=90.0, close=94.0),
        SimpleNamespace(high=97.0, low=91.0, close=92.0),
    ]

    guidance = resolve_quick_fibonacci_guidance(candles)

    assert guidance.direction is BreakoutDirection.BEARISH
    assert guidance.zone == "in_market_mover"
    assert guidance.allows(BreakoutDirection.BEARISH) is True
    assert guidance.allows(BreakoutDirection.BULLISH) is False


def test_quick_loop_allows_reduced_grid_when_signal_conflicts_with_guidance(monkeypatch):
    from src.quick_scalp_loop import (
        QuickFibonacciGuidance,
        QuickFvgGuidance,
        QuickIndicatorGuidance,
        QuickTickGuidance,
        run_quick_scalp_loop,
    )
    from src.strategy.breakout import BreakoutDirection

    opened = []
    events = []

    class FakeMt5:
        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.1, bid=99.9)

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=len(opened))

    guidance = QuickFibonacciGuidance(
        direction=BreakoutDirection.BULLISH,
        zone="golden_zone",
        current_price=92.0,
        swing_low=90.0,
        swing_high=111.0,
    )
    candles = [
        SimpleNamespace(high=101.0, low=99.0, open=99.4, close=100.6)
        for _ in range(30)
    ]
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: candles)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m15_candles", lambda mt5_module, symbol, count=50: candles)
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: BreakoutDirection.BULLISH)
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_tick_direction",
        lambda ticks, point=0.0, fast=False: QuickTickGuidance(BreakoutDirection.BEARISH, "tick_momentum", 60, -1.2, 20, 40),
    )
    monkeypatch.setattr("src.quick_scalp_loop.resolve_quick_fibonacci_guidance", lambda candles: guidance)
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fvg_guidance",
        lambda candles, direction, current_price: QuickFvgGuidance(direction, 97.0, 95.0, 3, current_price, "matching_fvg"),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_indicator_guidance",
        lambda candles: QuickIndicatorGuidance(55.0, 101.0, BreakoutDirection.BULLISH),
    )
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (96.0, 105.0))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert opened == []
    assert any("reason=signal_override" in message for message in events)


def test_quick_loop_logs_repeated_same_hold_state_once(monkeypatch):
    from src.quick_scalp_loop import QuickFibonacciGuidance, QuickIndicatorGuidance, run_quick_scalp_loop
    from src.strategy.breakout import BreakoutDirection

    events = []

    class FakeExecutor:
        mt5_module = SimpleNamespace()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            raise AssertionError("conflicted guidance should not open a trade")

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: [])
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m15_candles", lambda mt5_module, symbol, count=50: [])
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: BreakoutDirection.BULLISH)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=60: bullish_ticks())
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fibonacci_guidance",
        lambda candles: QuickFibonacciGuidance(BreakoutDirection.BEARISH, "towards_market_mover", 4690.82, 4680.0, 4700.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_indicator_guidance",
        lambda candles: QuickIndicatorGuidance(88.0, 4696.42, BreakoutDirection.BEARISH),
    )

    run_quick_scalp_loop(
        mt5_module=SimpleNamespace(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=10,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=3,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    indicator_logs = [message for message in events if "reason=indicator_filter" in message]
    assert len(indicator_logs) == 1


def test_quick_loop_allows_reduced_grid_when_only_indicator_agrees(monkeypatch):
    from src.quick_scalp_loop import QuickFibonacciGuidance, QuickIndicatorGuidance, QuickFvgGuidance, run_quick_scalp_loop
    from src.strategy.breakout import BreakoutDirection

    opened = []

    class FakeMt5:
        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.1, bid=99.9)

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=len(opened))

    candles = [
        SimpleNamespace(high=101.0, low=99.0, open=99.4, close=100.6)
        for _ in range(30)
    ]
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: candles)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m15_candles", lambda mt5_module, symbol, count=50: [])
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: BreakoutDirection.BULLISH)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=60: bullish_ticks())
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fvg_guidance",
        lambda candles, direction, current_price: QuickFvgGuidance(direction, 101.0, 99.0, 3, current_price, "matching_fvg"),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fibonacci_guidance",
        lambda candles: QuickFibonacciGuidance(BreakoutDirection.BEARISH, "towards_market_mover", 100.0, 90.0, 111.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_indicator_guidance",
        lambda candles: QuickIndicatorGuidance(55.0, 99.0, BreakoutDirection.BULLISH),
    )
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (96.0, 105.0))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=10,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: None,
    )

    assert len(opened) == 1


def test_quick_loop_allows_reduced_grid_when_only_fibonacci_agrees(monkeypatch):
    from src.quick_scalp_loop import QuickFibonacciGuidance, QuickIndicatorGuidance, QuickFvgGuidance, run_quick_scalp_loop
    from src.strategy.breakout import BreakoutDirection

    opened = []

    class FakeMt5:
        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.1, bid=99.9)

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=len(opened))

    candles = [
        SimpleNamespace(high=101.0, low=99.0, open=100.6, close=99.4)
        for _ in range(30)
    ]
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: candles)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m15_candles", lambda mt5_module, symbol, count=50: [])
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: BreakoutDirection.BEARISH)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=60: bearish_ticks())
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fvg_guidance",
        lambda candles, direction, current_price: QuickFvgGuidance(direction, 101.0, 99.0, 3, current_price, "matching_fvg"),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fibonacci_guidance",
        lambda candles: QuickFibonacciGuidance(BreakoutDirection.BEARISH, "towards_market_mover", 100.0, 90.0, 111.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_indicator_guidance",
        lambda candles: QuickIndicatorGuidance(55.0, 99.0, BreakoutDirection.BULLISH),
    )
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (104.0, 95.0))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=10,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: None,
    )

    assert len(opened) == 1


def test_resolve_quick_indicator_guidance_allows_bullish_when_rsi_and_sar_agree():
    from src.quick_scalp_loop import resolve_quick_indicator_guidance
    from src.strategy.breakout import BreakoutDirection

    closes = [
        100.0,
        100.4,
        100.1,
        100.7,
        100.3,
        101.0,
        100.6,
        101.2,
        100.8,
        101.5,
        101.0,
        101.7,
        101.1,
        101.8,
        101.3,
        102.0,
        101.5,
        102.2,
        101.7,
        102.4,
    ]
    candles = [
        SimpleNamespace(high=close + 0.8, low=close - 0.8, close=close)
        for close in closes
    ]

    guidance = resolve_quick_indicator_guidance(candles)

    assert guidance.rsi < 70.0
    assert guidance.sar_direction is BreakoutDirection.BULLISH
    assert guidance.allows(BreakoutDirection.BULLISH) is True
    assert guidance.allows(BreakoutDirection.BEARISH) is False


def test_resolve_quick_indicator_guidance_blocks_buy_when_rsi_is_overbought():
    from src.quick_scalp_loop import resolve_quick_indicator_guidance
    from src.strategy.breakout import BreakoutDirection

    candles = [
        SimpleNamespace(high=100.8 + (index * 2), low=99.2 + (index * 2), close=100.0 + (index * 2))
        for index in range(20)
    ]

    guidance = resolve_quick_indicator_guidance(candles)

    assert guidance.rsi >= 70.0
    assert guidance.allows(BreakoutDirection.BULLISH) is False


def test_quick_guidance_allows_moderate_overbought_rsi_when_structure_agrees():
    from src.quick_scalp_loop import (
        QuickFibonacciGuidance,
        QuickIndicatorGuidance,
        resolve_quick_guidance_decision,
    )
    from src.strategy.breakout import BreakoutDirection

    decision = resolve_quick_guidance_decision(
        direction=BreakoutDirection.BULLISH,
        fibonacci_guidance=QuickFibonacciGuidance(BreakoutDirection.BULLISH, "golden_zone", 100.0, 90.0, 110.0),
        indicator_guidance=QuickIndicatorGuidance(rsi=82.0, sar=99.0, sar_direction=BreakoutDirection.BEARISH),
    )

    assert decision.allowed is True
    assert decision.reason == "structure_override"
    assert decision.rsi_allows is False


def test_quick_guidance_blocks_extreme_rsi_exhaustion():
    from src.quick_scalp_loop import (
        QuickFibonacciGuidance,
        QuickIndicatorGuidance,
        resolve_quick_guidance_decision,
    )
    from src.strategy.breakout import BreakoutDirection

    decision = resolve_quick_guidance_decision(
        direction=BreakoutDirection.BULLISH,
        fibonacci_guidance=QuickFibonacciGuidance(BreakoutDirection.BULLISH, "golden_zone", 100.0, 90.0, 110.0),
        indicator_guidance=QuickIndicatorGuidance(rsi=88.0, sar=99.0, sar_direction=BreakoutDirection.BULLISH),
    )

    assert decision.allowed is False
    assert decision.reason == "indicator_filter"


def test_resolve_quick_indicator_guidance_flips_sar_bearish_after_reversal():
    from src.quick_scalp_loop import resolve_quick_indicator_guidance
    from src.strategy.breakout import BreakoutDirection

    closes = [
        100.0,
        100.5,
        101.0,
        101.6,
        102.1,
        102.8,
        103.2,
        103.8,
        104.2,
        104.7,
        105.1,
        104.5,
        103.9,
        103.2,
        102.4,
        101.6,
        100.8,
        99.9,
        99.0,
        98.2,
    ]
    candles = [
        SimpleNamespace(high=close + 0.5, low=close - 0.5, close=close)
        for close in closes
    ]

    guidance = resolve_quick_indicator_guidance(candles)

    assert guidance.sar_direction is BreakoutDirection.BEARISH
    assert guidance.sar > closes[-1]


def test_quick_loop_holds_when_m1_indicator_guidance_blocks_signal(monkeypatch):
    from src.quick_scalp_loop import (
        QuickFibonacciGuidance,
        QuickFvgGuidance,
        QuickIndicatorGuidance,
        QuickTickGuidance,
        run_quick_scalp_loop,
    )
    from src.strategy.breakout import BreakoutDirection

    opened = []
    events = []

    class FakeMt5:
        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.1, bid=99.9)

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=len(opened))

    candles = [
        SimpleNamespace(high=101.0, low=99.0, open=99.4, close=100.6)
        for _ in range(30)
    ]
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: candles)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m15_candles", lambda mt5_module, symbol, count=50: candles)
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: BreakoutDirection.BULLISH)
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_tick_direction",
        lambda ticks, point=0.0, fast=False: QuickTickGuidance(BreakoutDirection.BULLISH, "tick_momentum", 60, 1.2, 40, 20),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fibonacci_guidance",
        lambda candles: QuickFibonacciGuidance(BreakoutDirection.BULLISH, "golden_zone", 96.0, 90.0, 111.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fvg_guidance",
        lambda candles, direction, current_price: QuickFvgGuidance(direction, 97.0, 95.0, 3, current_price, "matching_fvg"),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_indicator_guidance",
        lambda candles: QuickIndicatorGuidance(rsi=88.0, sar=99.0, sar_direction=BreakoutDirection.BULLISH),
    )

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    assert opened == []
    assert any("reason=indicator_filter" in message for message in events)


def test_resolve_quick_grid_permission_blocks_weak_doji_candle():
    from src.quick_scalp_loop import QuickFibonacciGuidance, resolve_quick_grid_permission
    from src.strategy.breakout import BreakoutDirection

    candles = [
        SimpleNamespace(high=101.0, low=100.0, open=100.49, close=100.51)
        for _ in range(15)
    ]
    tick = SimpleNamespace(ask=100.52, bid=100.48)
    guidance = QuickFibonacciGuidance(BreakoutDirection.BULLISH, "golden_zone", 100.5, 95.0, 105.0)

    permission = resolve_quick_grid_permission(
        candles=candles,
        positions=[],
        direction=BreakoutDirection.BULLISH,
        fibonacci_guidance=guidance,
        mt5_module=SimpleNamespace(ORDER_TYPE_BUY=0, ORDER_TYPE_SELL=1),
        tick=tick,
    )

    assert permission.allowed is False
    assert permission.reason == "weak_candle"


def test_resolve_quick_grid_permission_requires_spacing_from_same_side_grid():
    from src.quick_scalp_loop import QuickFibonacciGuidance, resolve_quick_grid_permission
    from src.strategy.breakout import BreakoutDirection

    candles = [
        SimpleNamespace(high=102.0, low=100.0, open=100.4, close=101.6)
        for _ in range(15)
    ]
    mt5 = SimpleNamespace(ORDER_TYPE_BUY=0, ORDER_TYPE_SELL=1)
    tick = SimpleNamespace(ask=101.62, bid=101.58)
    guidance = QuickFibonacciGuidance(BreakoutDirection.BULLISH, "golden_zone", 101.6, 95.0, 105.0)
    positions = [
        SimpleNamespace(type=0, price_open=101.55),
        SimpleNamespace(type=1, price_open=101.60),
    ]

    permission = resolve_quick_grid_permission(
        candles=candles,
        positions=positions,
        direction=BreakoutDirection.BULLISH,
        fibonacci_guidance=guidance,
        mt5_module=mt5,
        tick=tick,
    )

    assert permission.allowed is False
    assert permission.reason == "grid_spacing"


def test_resolve_quick_grid_permission_sets_zone_entry_caps():
    from src.quick_scalp_loop import QuickFibonacciGuidance, resolve_quick_grid_permission
    from src.strategy.breakout import BreakoutDirection

    candles = [
        SimpleNamespace(high=102.0, low=100.0, open=100.4, close=101.6)
        for _ in range(15)
    ]
    mt5 = SimpleNamespace(ORDER_TYPE_BUY=0, ORDER_TYPE_SELL=1)
    tick = SimpleNamespace(ask=101.6, bid=101.5)

    golden = resolve_quick_grid_permission(
        candles=candles,
        positions=[],
        direction=BreakoutDirection.BULLISH,
        fibonacci_guidance=QuickFibonacciGuidance(BreakoutDirection.BULLISH, "golden_zone", 101.6, 95.0, 105.0),
        mt5_module=mt5,
        tick=tick,
    )
    market_mover = resolve_quick_grid_permission(
        candles=candles,
        positions=[],
        direction=BreakoutDirection.BULLISH,
        fibonacci_guidance=QuickFibonacciGuidance(BreakoutDirection.BULLISH, "in_market_mover", 101.6, 95.0, 105.0),
        mt5_module=mt5,
        tick=tick,
    )

    assert golden.allowed is True
    assert golden.max_new_entries == 3
    assert market_mover.allowed is True
    assert market_mover.max_new_entries == 1


def test_build_quick_trade_levels_uses_one_to_three_rr_with_spread_padded_take_profit():
    from src.quick_scalp_loop import build_quick_trade_levels
    from src.strategy.breakout import BreakoutDirection

    class FakeMt5:
        def symbol_info(self, symbol):
            return SimpleNamespace(point=0.01)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.05, bid=100.00)

    buy_sl, buy_tp = build_quick_trade_levels(
        mt5_module=FakeMt5(),
        symbol="XAUUSD",
        direction=BreakoutDirection.BULLISH,
    )
    sell_sl, sell_tp = build_quick_trade_levels(
        mt5_module=FakeMt5(),
        symbol="XAUUSD",
        direction=BreakoutDirection.BEARISH,
    )

    assert buy_sl == pytest.approx(92.05)
    assert buy_tp == pytest.approx(124.10)
    assert sell_sl == pytest.approx(108.00)
    assert sell_tp == pytest.approx(75.95)


def test_build_quick_trade_levels_respects_broker_minimum_stop_distance():
    from src.quick_scalp_loop import build_quick_trade_levels
    from src.strategy.breakout import BreakoutDirection

    class FakeMt5:
        def symbol_info(self, symbol):
            return SimpleNamespace(point=0.01, trade_stops_level=60)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.05, bid=100.00)

    buy_sl, buy_tp = build_quick_trade_levels(
        mt5_module=FakeMt5(),
        symbol="XAUUSD",
        direction=BreakoutDirection.BULLISH,
    )
    sell_sl, sell_tp = build_quick_trade_levels(
        mt5_module=FakeMt5(),
        symbol="XAUUSD",
        direction=BreakoutDirection.BEARISH,
    )

    assert buy_sl == pytest.approx(92.05)
    assert buy_tp == pytest.approx(124.10)
    assert sell_sl == pytest.approx(108.00)
    assert sell_tp == pytest.approx(75.95)


def test_estimate_account_profit_from_price_move_uses_tick_value_and_lot():
    from src.quick_scalp_loop import estimate_account_profit_from_price_move

    class FakeMt5:
        def symbol_info(self, symbol):
            return SimpleNamespace(trade_tick_size=0.01, trade_tick_value=1.0, point=0.01)

    assert estimate_account_profit_from_price_move(FakeMt5(), "XAUUSDc", 0.34, 0.01) == pytest.approx(0.34)
    assert estimate_account_profit_from_price_move(FakeMt5(), "XAUUSDc", 0.34, 0.10) == pytest.approx(3.4)


def test_estimate_price_move_for_account_profit_inverts_tick_value_math():
    from src.quick_scalp_loop import estimate_price_move_for_account_profit

    class FakeMt5:
        def symbol_info(self, symbol):
            return SimpleNamespace(trade_tick_size=0.01, trade_tick_value=1.0, point=0.01)

    assert estimate_price_move_for_account_profit(FakeMt5(), "XAUUSDc", 0.50, 0.01) == pytest.approx(0.50)
    assert estimate_price_move_for_account_profit(FakeMt5(), "XAUUSDc", 0.50, 0.10) == pytest.approx(0.05)


def test_cap_stop_loss_to_account_risk_caps_buy_and_sell_distance():
    from src.quick_scalp_loop import cap_stop_loss_to_account_risk
    from src.strategy.breakout import BreakoutDirection

    class FakeMt5:
        def symbol_info(self, symbol):
            return SimpleNamespace(
                point=0.01,
                trade_tick_size=0.01,
                trade_tick_value=1.0,
                trade_stops_level=0,
                trade_freeze_level=0,
            )

    buy_sl, buy_distance = cap_stop_loss_to_account_risk(
        mt5_module=FakeMt5(),
        symbol="XAUUSDc",
        direction=BreakoutDirection.BULLISH,
        entry=100.0,
        stop_loss=96.0,
        lot=0.01,
        max_loss=0.50,
    )
    sell_sl, sell_distance = cap_stop_loss_to_account_risk(
        mt5_module=FakeMt5(),
        symbol="XAUUSDc",
        direction=BreakoutDirection.BEARISH,
        entry=100.0,
        stop_loss=104.0,
        lot=0.01,
        max_loss=0.50,
    )

    assert buy_sl == pytest.approx(99.50)
    assert buy_distance == pytest.approx(0.50)
    assert sell_sl == pytest.approx(100.50)
    assert sell_distance == pytest.approx(0.50)


def test_cap_stop_loss_to_account_risk_respects_broker_minimum_stop_distance():
    from src.quick_scalp_loop import cap_stop_loss_to_account_risk
    from src.strategy.breakout import BreakoutDirection

    class FakeMt5:
        def symbol_info(self, symbol):
            return SimpleNamespace(
                point=0.01,
                trade_tick_size=0.01,
                trade_tick_value=1.0,
                trade_stops_level=80,
                trade_freeze_level=0,
            )

    stop_loss, distance = cap_stop_loss_to_account_risk(
        mt5_module=FakeMt5(),
        symbol="XAUUSDc",
        direction=BreakoutDirection.BULLISH,
        entry=100.0,
        stop_loss=96.0,
        lot=0.01,
        max_loss=0.50,
    )

    assert stop_loss == pytest.approx(99.20)
    assert distance == pytest.approx(0.80)


def test_estimate_executable_tick_profit_subtracts_spread_before_account_profit():
    from src.quick_scalp_loop import estimate_executable_tick_profit
    from src.strategy.breakout import BreakoutDirection

    class FakeMt5:
        def symbol_info(self, symbol):
            return SimpleNamespace(trade_tick_size=0.01, trade_tick_value=1.0, point=0.01)

    ticks = [
        SimpleNamespace(bid=100.00, ask=100.34),
        SimpleNamespace(bid=100.30, ask=100.64),
    ]
    current_tick = SimpleNamespace(bid=100.30, ask=100.64)

    estimated_profit, executable_price_profit = estimate_executable_tick_profit(
        FakeMt5(),
        "XAUUSDc",
        ticks,
        current_tick,
        BreakoutDirection.BULLISH,
        0.01,
    )

    assert executable_price_profit == pytest.approx(-0.04)
    assert estimated_profit == pytest.approx(-0.04)


def test_estimate_executable_tick_profit_allows_only_move_that_clears_spread():
    from src.quick_scalp_loop import estimate_executable_tick_profit
    from src.strategy.breakout import BreakoutDirection

    class FakeMt5:
        def symbol_info(self, symbol):
            return SimpleNamespace(trade_tick_size=0.01, trade_tick_value=1.0, point=0.01)

    ticks = [
        SimpleNamespace(bid=100.00, ask=100.34),
        SimpleNamespace(bid=100.45, ask=100.79),
    ]
    current_tick = SimpleNamespace(bid=100.45, ask=100.79)

    estimated_profit, executable_price_profit = estimate_executable_tick_profit(
        FakeMt5(),
        "XAUUSDc",
        ticks,
        current_tick,
        BreakoutDirection.BULLISH,
        0.01,
    )

    assert executable_price_profit == pytest.approx(0.11)
    assert estimated_profit == pytest.approx(0.11)


def test_quick_loop_limits_new_entries_by_grid_permission(monkeypatch):
    from src.quick_scalp_loop import (
        QuickFibonacciGuidance,
        QuickGridPermission,
        QuickIndicatorGuidance,
        run_quick_scalp_loop,
    )
    from src.strategy.breakout import BreakoutDirection

    opened = []

    class FakeMt5:
        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.1, bid=99.9)

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=len(opened))

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: [])
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m15_candles", lambda mt5_module, symbol, count=50: [])
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: BreakoutDirection.BULLISH)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=60: bullish_ticks())
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fibonacci_guidance",
        lambda candles: QuickFibonacciGuidance(BreakoutDirection.BULLISH, "golden_zone", 96.0, 90.0, 111.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_indicator_guidance",
        lambda candles: QuickIndicatorGuidance(55.0, 99.0, BreakoutDirection.BULLISH),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_grid_permission",
        lambda **kwargs: QuickGridPermission(True, "ok", 3, 0.2, 100.0),
    )
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (96.0, 105.0))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=10,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: None,
    )

    assert len(opened) == 3


def test_quick_loop_rechecks_grid_spacing_after_each_new_entry(monkeypatch):
    from src.quick_scalp_loop import QuickFibonacciGuidance, QuickIndicatorGuidance, run_quick_scalp_loop
    from src.strategy.breakout import BreakoutDirection

    opened = []
    positions = []

    class FakeMt5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.1, bid=99.9)

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return list(positions)

        def open_strategy_trade(self, **kwargs):
            opened_position = SimpleNamespace(
                ticket=len(opened) + 1,
                type=FakeMt5.ORDER_TYPE_BUY,
                price_open=100.1,
            )
            opened.append(kwargs)
            positions.append(opened_position)
            return opened_position

    candles = [
        SimpleNamespace(high=101.0, low=99.0, open=99.4, close=100.6)
        for _ in range(30)
    ]
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=30: candles)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m15_candles", lambda mt5_module, symbol, count=50: [])
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: BreakoutDirection.BULLISH)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=60: bullish_ticks())
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fibonacci_guidance",
        lambda candles: QuickFibonacciGuidance(BreakoutDirection.BULLISH, "golden_zone", 96.0, 90.0, 111.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_indicator_guidance",
        lambda candles: QuickIndicatorGuidance(55.0, 99.0, BreakoutDirection.BULLISH),
    )
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (96.0, 105.0))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=10,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: None,
    )

    assert len(opened) == 1


def test_quick_loop_stops_opening_when_free_margin_cannot_support_next_order(monkeypatch):
    from src.quick_scalp_loop import run_quick_scalp_loop
    from src.strategy.breakout import BreakoutDirection

    opened = []

    class FakeMt5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1

        def account_info(self):
            return SimpleNamespace(margin_free=0.5)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.1, bid=99.9)

        def order_calc_margin(self, order_type, symbol, lot, price):
            return 1.0

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=len(opened))

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=2: [])
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: BreakoutDirection.BULLISH)

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=100,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: None,
    )

    assert opened == []


def test_quick_loop_logs_insufficient_margin_once_until_trade_opens(monkeypatch):
    from src.quick_scalp_loop import QuickFibonacciGuidance, QuickGridPermission, QuickIndicatorGuidance, run_quick_scalp_loop
    from src.strategy.breakout import BreakoutDirection

    events = []
    margin_checks = iter([False, False, True, False])
    opened = []

    class FakeMt5:
        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=100.1, bid=99.9)

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=len(opened))

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=2: [])
    monkeypatch.setattr("src.quick_scalp_loop.fetch_m15_candles", lambda mt5_module, symbol, count=50: [])
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: BreakoutDirection.BULLISH)
    monkeypatch.setattr("src.quick_scalp_loop.fetch_recent_ticks", lambda mt5_module, symbol, count=60: bullish_ticks())
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fibonacci_guidance",
        lambda candles: QuickFibonacciGuidance(BreakoutDirection.BULLISH, "golden_zone", 96.0, 90.0, 111.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_indicator_guidance",
        lambda candles: QuickIndicatorGuidance(55.0, 99.0, BreakoutDirection.BULLISH),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_grid_permission",
        lambda **kwargs: QuickGridPermission(True, "ok", 1, 0.2, 100.0),
    )
    monkeypatch.setattr("src.quick_scalp_loop.build_quick_trade_levels", lambda **kwargs: (104.0, 95.0))
    monkeypatch.setattr("src.quick_scalp_loop.has_margin_for_quick_order", lambda **kwargs: next(margin_checks))

    run_quick_scalp_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=1,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=4,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(message),
    )

    insufficient_logs = [
        message for message in events if "reason=insufficient_free_margin" in message
    ]
    assert len(opened) == 1
    assert len(insufficient_logs) == 2


def test_quick_loop_holds_when_latest_closed_m1_candle_is_flat(monkeypatch):
    from src.quick_scalp_loop import run_quick_scalp_loop

    opened = []

    class FakeExecutor:
        mt5_module = SimpleNamespace()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            opened.append(kwargs)
            return SimpleNamespace(ticket=1)

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=2: [])
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: None)

    run_quick_scalp_loop(
        mt5_module=SimpleNamespace(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=100,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: None,
    )

    assert opened == []


def test_quick_loop_returns_reload_requested_when_callback_detects_code_change(monkeypatch):
    from src.quick_scalp_loop import run_quick_scalp_loop

    class FakeExecutor:
        mt5_module = SimpleNamespace()

        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return []

        def open_strategy_trade(self, **kwargs):
            raise AssertionError("quick loop should not open before returning reload")

    monkeypatch.setattr("src.quick_scalp_loop.fetch_m1_candles", lambda mt5_module, symbol, count=2: [])
    monkeypatch.setattr("src.quick_scalp_loop.resolve_m1_direction", lambda candles: None)

    result = run_quick_scalp_loop(
        mt5_module=SimpleNamespace(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        max_positions=100,
        profit_target=0.2,
        poll_seconds=1,
        max_loops=1,
        reload_check_fn=lambda: True,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: None,
    )

    assert result == "reload_requested"
