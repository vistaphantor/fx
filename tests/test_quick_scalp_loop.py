from types import SimpleNamespace

import pytest


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

    assert ("close", 1, "quick-scalp-tick-turn-profit-exit") in events
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


def test_quick_loop_closes_only_when_profit_is_greater_than_target(monkeypatch):
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
    assert all(event != ("close", 2, "quick-scalp-profit-exit") for event in events)
    assert all(event != ("close", 3, "quick-scalp-profit-exit") for event in events)


def test_close_profitable_quick_positions_logs_best_profit_when_nothing_closes():
    from src.quick_scalp_loop import close_profitable_quick_positions

    events = []

    class FakeExecutor:
        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return [
                SimpleNamespace(ticket=1, profit=0.19),
                SimpleNamespace(ticket=2, profit=0.2),
            ]

        def close_position(self, position, comment):
            raise AssertionError("positions at or below target should not close")

    closed = close_profitable_quick_positions(
        executor=FakeExecutor(),
        symbol="XAUUSD",
        profit_target=0.2,
        log_fn=lambda message: events.append(message),
    )

    assert closed == 0
    assert events == [
        "QUICK PROFIT WAIT XAUUSD positions=2 positive=2 negative=0 flat=0 "
        "net_profit=0.39 best_ticket=2 best_profit=0.20 worst_ticket=1 worst_profit=0.19 "
        "target=0.20 tick_direction=NONE"
    ]


def test_close_profitable_quick_positions_logs_positive_and_negative_basket_context():
    from src.quick_scalp_loop import close_profitable_quick_positions

    events = []

    class FakeExecutor:
        def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
            return [
                SimpleNamespace(ticket=1, profit=-1.25),
                SimpleNamespace(ticket=2, profit=0.20),
                SimpleNamespace(ticket=3, profit=0.0),
            ]

        def close_position(self, position, comment):
            raise AssertionError("positions at or below target should not close")

    closed = close_profitable_quick_positions(
        executor=FakeExecutor(),
        symbol="XAUUSD",
        profit_target=0.2,
        log_fn=lambda message: events.append(message),
    )

    assert closed == 0
    assert events == [
        "QUICK PROFIT WAIT XAUUSD positions=3 positive=1 negative=1 flat=1 "
        "net_profit=-1.05 best_ticket=2 best_profit=0.20 worst_ticket=1 worst_profit=-1.25 "
        "target=0.20 tick_direction=NONE"
    ]


def test_close_profitable_quick_positions_holds_target_profit_when_ticks_still_favor_trade():
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
            raise AssertionError("profitable buy should keep running while ticks stay bullish")

    closed = close_profitable_quick_positions(
        executor=FakeExecutor(),
        symbol="XAUUSD",
        profit_target=0.2,
        tick_direction=BreakoutDirection.BULLISH,
        log_fn=lambda message: events.append(message),
    )

    assert closed == 0
    assert "target=0.20" in events[0]
    assert "tick_direction=BULLISH" in events[0]


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
        direction=BreakoutDirection.BEARISH,
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
        lambda ticks, point=0.0: QuickTickGuidance(BreakoutDirection.BEARISH, "tick_momentum", 60, -1.2, 20, 40),
    )
    monkeypatch.setattr("src.quick_scalp_loop.resolve_quick_fibonacci_guidance", lambda candles: guidance)
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fvg_guidance",
        lambda candles, direction, current_price: QuickFvgGuidance(direction, 97.0, 95.0, 3, current_price, "matching_fvg"),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_indicator_guidance",
        lambda candles: QuickIndicatorGuidance(55.0, 101.0, BreakoutDirection.BEARISH),
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
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_fibonacci_guidance",
        lambda candles: QuickFibonacciGuidance(BreakoutDirection.BEARISH, "towards_market_mover", 4690.82, 4680.0, 4700.0),
    )
    monkeypatch.setattr(
        "src.quick_scalp_loop.resolve_quick_indicator_guidance",
        lambda candles: QuickIndicatorGuidance(82.0, 4696.42, BreakoutDirection.BEARISH),
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
    from src.quick_scalp_loop import QuickFibonacciGuidance, QuickIndicatorGuidance, run_quick_scalp_loop
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
    from src.quick_scalp_loop import QuickFibonacciGuidance, QuickIndicatorGuidance, run_quick_scalp_loop
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
        lambda ticks, point=0.0: QuickTickGuidance(BreakoutDirection.BULLISH, "tick_momentum", 60, 1.2, 40, 20),
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
