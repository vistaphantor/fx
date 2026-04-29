from pathlib import Path
from types import SimpleNamespace

import run


def test_main_always_starts_live_signal_loop(monkeypatch):
    events = []

    class FakeSession:
        def __init__(self, terminal_path, startup_wait_seconds, subprocess_module=None, sleep_fn=None, mt5_module=None):
            events.append(("session_init", terminal_path, startup_wait_seconds))

        def launch_terminal(self):
            events.append("launch")

        def initialize_and_login(self, login, password, server):
            events.append(("login", login, password, server))
            return SimpleNamespace(login=login)

        def shutdown(self):
            events.append("shutdown")

    settings = SimpleNamespace(
        mt5_terminal_path=str(Path("C:/MT5/terminal64.exe")),
        hfm_login=123456,
        hfm_password="secret",
        hfm_server="HFMarketsGlobal-Demo",
        trading_symbol="XAUUSD",
        mt5_startup_wait_seconds=5,
        loop_poll_seconds=60,
        max_live_loops=1,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        default_trade_lot=0.01,
        add_on_lot_increment=0.01,
        campaign_max_exposure_pct=10.0,
        tradingview_webhook_enabled=False,
        strategy_profiles={"XAUUSD": SimpleNamespace(symbol="XAUUSD", min_edge_threshold=2.0)},
    )

    monkeypatch.setattr(run, "load_settings", lambda: settings)
    monkeypatch.setattr(run, "Mt5Session", FakeSession)
    monkeypatch.setattr(run, "create_mt5_module", lambda: "mt5")
    monkeypatch.setattr(
        run,
        "run_live_signal_loop",
        lambda **kwargs: events.append(
            (
                "live_loop",
                kwargs["symbol"],
                kwargs["lot"],
                kwargs["add_on_lot_increment"],
                kwargs["campaign_max_exposure_pct"],
                kwargs["poll_seconds"],
                kwargs["max_loops"],
                kwargs["strategy_profile"].symbol,
            )
        ),
    )

    class FakeExecutor:
        def __init__(self, mt5_module):
            events.append(("executor_init", mt5_module))

    monkeypatch.setattr(run, "TradeExecutor", FakeExecutor)

    exit_code = run.main()

    assert exit_code == 0
    assert events == [
        ("session_init", settings.mt5_terminal_path, 5),
        "launch",
        ("login", 123456, "secret", "HFMarketsGlobal-Demo"),
        ("executor_init", "mt5"),
        ("live_loop", "XAUUSD", 0.01, 0.01, 10.0, 60, 1, "XAUUSD"),
        "shutdown",
    ]


def test_main_starts_tradingview_server_when_enabled(monkeypatch):
    events = []

    class FakeSession:
        def __init__(self, terminal_path, startup_wait_seconds, subprocess_module=None, sleep_fn=None, mt5_module=None):
            events.append(("session_init", terminal_path, startup_wait_seconds))

        def launch_terminal(self):
            events.append("launch")

        def initialize_and_login(self, login, password, server):
            events.append(("login", login, password, server))
            return SimpleNamespace(login=login)

        def shutdown(self):
            events.append("shutdown")

    class FakeServer:
        def shutdown(self):
            events.append("webhook_shutdown")

    settings = SimpleNamespace(
        mt5_terminal_path=str(Path("C:/MT5/terminal64.exe")),
        hfm_login=123456,
        hfm_password="secret",
        hfm_server="HFMarketsGlobal-Demo",
        trading_symbol="XAUUSD",
        mt5_startup_wait_seconds=5,
        loop_poll_seconds=60,
        max_live_loops=1,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        default_trade_lot=0.01,
        add_on_lot_increment=0.01,
        campaign_max_exposure_pct=10.0,
        tradingview_webhook_enabled=True,
        tradingview_webhook_host="127.0.0.1",
        tradingview_webhook_port=8080,
        tradingview_webhook_secret="tv-secret",
        tradingview_alert_max_age_seconds=900,
        strategy_profiles={"XAUUSD": SimpleNamespace(symbol="XAUUSD", min_edge_threshold=2.0)},
    )

    monkeypatch.setattr(run, "load_settings", lambda: settings)
    monkeypatch.setattr(run, "Mt5Session", FakeSession)
    monkeypatch.setattr(run, "create_mt5_module", lambda: "mt5")
    monkeypatch.setattr(run, "TradeExecutor", lambda mt5_module: SimpleNamespace())
    monkeypatch.setattr(run, "TradingViewAlertStore", lambda max_age_seconds: ("store", max_age_seconds))
    monkeypatch.setattr(
        run,
        "start_tradingview_webhook_server",
        lambda **kwargs: events.append(("webhook_start", kwargs["host"], kwargs["port"])) or FakeServer(),
    )
    monkeypatch.setattr(
        run,
        "run_live_signal_loop",
        lambda **kwargs: events.append(("live_loop", kwargs["symbol"])),
    )

    exit_code = run.main()

    assert exit_code == 0
    assert ("webhook_start", "127.0.0.1", 8080) in events
    assert "webhook_shutdown" in events


def test_main_restarts_process_when_live_loop_requests_reload(monkeypatch):
    events = []

    class FakeSession:
        def __init__(self, terminal_path, startup_wait_seconds, subprocess_module=None, sleep_fn=None, mt5_module=None):
            events.append(("session_init", terminal_path, startup_wait_seconds))

        def launch_terminal(self):
            events.append("launch")

        def initialize_and_login(self, login, password, server):
            events.append(("login", login, password, server))
            return SimpleNamespace(login=login)

        def shutdown(self):
            events.append("shutdown")

    settings = SimpleNamespace(
        mt5_terminal_path=str(Path("C:/MT5/terminal64.exe")),
        hfm_login=123456,
        hfm_password="secret",
        hfm_server="HFMarketsGlobal-Demo",
        trading_symbol="XAUUSD",
        mt5_startup_wait_seconds=5,
        loop_poll_seconds=60,
        max_live_loops=1,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        default_trade_lot=0.01,
        add_on_lot_increment=0.01,
        campaign_max_exposure_pct=10.0,
        tradingview_webhook_enabled=False,
        strategy_profiles={"XAUUSD": SimpleNamespace(symbol="XAUUSD", min_edge_threshold=2.0)},
    )

    monkeypatch.setattr(run, "load_settings", lambda: settings)
    monkeypatch.setattr(run, "Mt5Session", FakeSession)
    monkeypatch.setattr(run, "create_mt5_module", lambda: "mt5")
    monkeypatch.setattr(run, "TradeExecutor", lambda mt5_module: SimpleNamespace())
    monkeypatch.setattr(run, "build_code_watcher", lambda project_root=None: SimpleNamespace(has_changes=lambda: False))
    monkeypatch.setattr(run, "run_live_signal_loop", lambda **kwargs: "reload_requested")
    monkeypatch.setattr(run, "restart_process", lambda: events.append("restart"))

    exit_code = run.main()

    assert exit_code == 0
    assert events[-2:] == ["shutdown", "restart"]
