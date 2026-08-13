from dataclasses import replace
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _isolate_dashboard_cleanup(monkeypatch):
    import quick

    monkeypatch.setattr(quick, "clear_dashboard_state", lambda: None)


def _settings():
    from src.config import Settings

    return Settings(
        mt5_terminal_path="C:\\Program Files\\MT5\\terminal64.exe",
        hfm_login=123456,
        hfm_password="secret",
        hfm_server="HFMarketsGlobal-Demo",
        mt4_fallback_enabled=False,
        mt4_terminal_path="",
        mt4_data_path="",
        mt4_chart_symbol="",
        mt4_chart_period="",
        mt4_login=None,
        mt4_password="",
        mt4_server="",
        trading_symbol="XAUUSD",
        mt5_startup_wait_seconds=0,
        loop_poll_seconds=60,
        max_live_loops=1,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        default_trade_lot=0.03,
        add_on_lot_increment=0.01,
        campaign_max_exposure_pct=10.0,
        tradingview_webhook_enabled=False,
        tradingview_webhook_host="127.0.0.1",
        tradingview_webhook_port=8080,
        tradingview_webhook_secret="",
        tradingview_alert_max_age_seconds=900,
        strategy_profiles={},
    )


def test_force_quick_settings_enables_scalper_and_keeps_same_env_values():
    from quick import force_quick_settings

    settings = _settings()

    quick_settings = force_quick_settings(settings)

    assert quick_settings.quick_scalp_enabled is True
    assert quick_settings.hfm_login == settings.hfm_login
    assert quick_settings.quick_trade_lot == settings.quick_trade_lot


def test_force_quick_settings_supports_test_double_settings():
    from quick import force_quick_settings

    settings = SimpleNamespace(quick_scalp_enabled=False, hfm_login=123456)

    quick_settings = force_quick_settings(settings)

    assert quick_settings is settings
    assert quick_settings.quick_scalp_enabled is True
    assert quick_settings.hfm_login == 123456


def test_build_quick_runtime_settings_rejects_invalid_quick_values():
    from quick import build_quick_runtime_settings

    settings = replace(_settings(), quick_trade_lot=0.0)

    try:
        build_quick_runtime_settings(settings)
    except ValueError as exc:
        assert "quick_trade_lot" in str(exc)
    else:
        raise AssertionError("invalid quick trade lot should be rejected before MT5 starts")


def test_format_quick_startup_summary_includes_operating_values():
    from quick import format_quick_startup_summary

    settings = replace(
        _settings(),
        quick_trade_lot=0.02,
        quick_max_positions=100,
        quick_profit_target=0.2,
        quick_max_loss=0.0,
        quick_poll_seconds=1,
        quick_min_free_margin=20.0,
    )
    account_info = SimpleNamespace(login=123456, currency="USD", balance=20.0, equity=20.0)

    summary = format_quick_startup_summary(settings, account_info)

    assert "login=123456" in summary
    assert "currency=USD" in summary
    assert "symbol=XAUUSD" in summary
    assert "lot=0.02" in summary
    assert "max_positions=100" in summary
    assert "profit_target=0.2" in summary
    assert "max_loss=0.0" in summary
    assert "poll_seconds=1" in summary
    assert "min_free_margin=20.0" in summary


def test_log_tick_history_health_warns_when_mt4_history_file_is_missing(monkeypatch):
    import quick

    events = []
    mt4 = SimpleNamespace(
        persisted_tick_history_count=lambda symbol: 0,
        tick_history_file_path=lambda symbol: f"C:\\Common\\Files\\fx_bridge_ticks_{symbol}.csv",
    )
    monkeypatch.setattr(quick.logging, "warning", lambda *args: events.append(args))

    quick.log_tick_history_health(mt4, "XAUUSD", minimum_ticks=4)

    assert events
    assert "MT4 TICK HISTORY NOT READY" in events[0][0]
    assert events[0][2] == 0
    assert events[0][3] == 4


def test_log_tick_history_health_reports_ready_when_enough_ticks(monkeypatch):
    import quick

    events = []
    mt4 = SimpleNamespace(persisted_tick_history_count=lambda symbol: 12)
    monkeypatch.setattr(quick.logging, "info", lambda *args: events.append(args))

    quick.log_tick_history_health(mt4, "XAUUSD", minimum_ticks=4)

    assert events == [("MT4 TICK HISTORY READY %s ticks=%s", "XAUUSD", 12)]


def test_quick_main_runs_quick_loop_with_forced_enabled_settings(monkeypatch):
    import quick

    events = []
    settings = replace(_settings(), quick_scalp_enabled=False, quick_trade_lot=0.02, quick_max_positions=7)

    class FakeSession:
        def __init__(self, **kwargs):
            events.append(("session", kwargs))

        def launch_terminal(self):
            events.append(("launch",))

        def initialize_and_login(self, **kwargs):
            events.append(("login", kwargs))
            return SimpleNamespace(login=123456)

        def shutdown(self):
            events.append(("shutdown",))

    monkeypatch.setattr(quick, "configure_logging", lambda: None)
    monkeypatch.setattr(quick, "set_keep_awake", lambda: None)
    monkeypatch.setattr(quick, "ensure_requirements_satisfied", lambda: events.append(("requirements",)))
    monkeypatch.setattr(quick, "load_settings", lambda: settings)
    monkeypatch.setattr(quick, "create_mt5_module", lambda: SimpleNamespace())
    monkeypatch.setattr(quick, "Mt5Session", lambda **kwargs: FakeSession(**kwargs))
    monkeypatch.setattr(quick, "TradeExecutor", lambda mt5: SimpleNamespace(mt5_module=mt5))
    monkeypatch.setattr(
        quick,
        "run_quick_scalp_loop",
        lambda **kwargs: events.append(("quick_loop", kwargs)),
    )

    assert quick.main() == 0

    assert events[0] == ("requirements",)
    loop_event = [event for event in events if event[0] == "quick_loop"][0]
    assert loop_event[1]["lot"] == 0.02
    assert loop_event[1]["max_positions"] == 7
    assert loop_event[1]["max_loss"] == settings.quick_max_loss
    assert loop_event[1]["execution_buffer"] == settings.quick_execution_buffer
    assert loop_event[1]["loss_cooldown_seconds"] == settings.quick_loss_cooldown_seconds
    assert loop_event[1]["shadow_on_daily_halt"] == settings.quick_shadow_on_daily_halt
    assert loop_event[1]["shadow_on_unproven_edge"] == settings.quick_shadow_on_unproven_edge
    assert loop_event[1]["shadow_only"] == settings.quick_shadow_only
    assert loop_event[1]["shadow_policy_enabled"] == settings.quick_shadow_policy_enabled
    assert loop_event[1]["shadow_policy_path"] == settings.quick_shadow_policy_path
    assert loop_event[1]["allow_inverted_shadow_policy"] == settings.quick_allow_inverted_shadow_policy
    assert loop_event[1]["symbol"] == "XAUUSD"
    assert loop_event[1]["reload_check_fn"]() is False
    assert events[-1] == ("shutdown",)


def test_quick_main_restarts_process_when_quick_loop_requests_reload(monkeypatch):
    import quick

    events = []
    settings = replace(_settings(), quick_scalp_enabled=False)

    class FakeSession:
        def __init__(self, **kwargs):
            events.append(("session", kwargs))

        def launch_terminal(self):
            events.append(("launch",))

        def initialize_and_login(self, **kwargs):
            events.append(("login", kwargs))
            return SimpleNamespace(login=123456)

        def shutdown(self):
            events.append(("shutdown",))

    monkeypatch.setattr(quick, "configure_logging", lambda: None)
    monkeypatch.setattr(quick, "set_keep_awake", lambda: None)
    monkeypatch.setattr(quick, "load_settings", lambda: settings)
    monkeypatch.setattr(quick, "create_mt5_module", lambda: SimpleNamespace())
    monkeypatch.setattr(quick, "Mt5Session", lambda **kwargs: FakeSession(**kwargs))
    monkeypatch.setattr(quick, "TradeExecutor", lambda mt5: SimpleNamespace(mt5_module=mt5))
    monkeypatch.setattr(quick, "build_code_watcher", lambda: SimpleNamespace(has_changes=lambda: True))
    monkeypatch.setattr(quick, "run_quick_scalp_loop", lambda **kwargs: quick.RELOAD_REQUESTED)
    monkeypatch.setattr(quick, "restart_process", lambda: events.append(("restart",)))

    assert quick.main() == 0
    assert events[-2:] == [("shutdown",), ("restart",)]


def test_quick_main_handles_keyboard_interrupt_cleanly(monkeypatch):
    import quick

    events = []
    settings = replace(_settings(), quick_scalp_enabled=False)

    class FakeSession:
        def __init__(self, **kwargs):
            pass

        def launch_terminal(self):
            events.append(("launch",))

        def initialize_and_login(self, **kwargs):
            events.append(("login", kwargs))
            return SimpleNamespace(login=123456)

        def shutdown(self):
            events.append(("shutdown",))

    monkeypatch.setattr(quick, "configure_logging", lambda: None)
    monkeypatch.setattr(quick, "set_keep_awake", lambda: None)
    monkeypatch.setattr(quick, "load_settings", lambda: settings)
    monkeypatch.setattr(quick, "create_mt5_module", lambda: SimpleNamespace())
    monkeypatch.setattr(quick, "Mt5Session", lambda **kwargs: FakeSession(**kwargs))
    monkeypatch.setattr(quick, "TradeExecutor", lambda mt5: SimpleNamespace(mt5_module=mt5))
    monkeypatch.setattr(quick.logging, "info", lambda *args, **kwargs: events.append(("log", args)))
    monkeypatch.setattr(
        quick,
        "run_quick_scalp_loop",
        lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    assert quick.main() == 130
    assert any("interrupted" in event[1][0].lower() for event in events if event[0] == "log")
    assert events[-1] == ("shutdown",)


def test_quick_main_uses_mt4_fallback_when_mt5_terminal_missing(monkeypatch):
    import quick

    events = []
    settings = replace(
        _settings(),
        mt4_fallback_enabled=True,
        mt4_terminal_path="C:\\Program Files\\MT4\\terminal.exe",
    )
    mt4_bridge = SimpleNamespace(account_info=lambda: SimpleNamespace(login=654321))

    class MissingMt5Session:
        def __init__(self, **kwargs):
            events.append(("session", kwargs))

        def launch_terminal(self):
            events.append(("launch_mt5",))
            raise FileNotFoundError("missing mt5")

        def shutdown(self):
            events.append(("shutdown",))

    monkeypatch.setattr(quick, "configure_logging", lambda: None)
    monkeypatch.setattr(quick, "set_keep_awake", lambda: None)
    monkeypatch.setattr(quick, "ensure_requirements_satisfied", lambda: events.append(("requirements",)))
    monkeypatch.setattr(quick, "load_settings", lambda: settings)
    monkeypatch.setattr(quick, "create_mt5_module", lambda: "mt5")
    monkeypatch.setattr(quick, "Mt5Session", lambda **kwargs: MissingMt5Session(**kwargs))
    monkeypatch.setattr(quick, "launch_mt4_fallback", lambda fallback_settings: events.append(("mt4_fallback", fallback_settings)) or mt4_bridge)
    monkeypatch.setattr(quick, "TradeExecutor", lambda mt5: events.append(("executor", mt5)) or SimpleNamespace(mt5_module=mt5))
    monkeypatch.setattr(quick, "run_quick_scalp_loop", lambda **kwargs: events.append(("quick_loop", kwargs)))

    assert quick.main() == 0

    fallback_event = [event for event in events if event[0] == "mt4_fallback"][0]
    assert fallback_event[1].mt4_fallback_enabled is True
    assert fallback_event[1].quick_scalp_enabled is True
    assert ("executor", mt4_bridge) in events
    loop_event = [event for event in events if event[0] == "quick_loop"][0]
    assert loop_event[1]["mt5_module"] is mt4_bridge
    assert ("shutdown",) not in events


def test_quick_main_exits_when_mt4_fallback_enabled_but_bridge_unavailable(monkeypatch):
    import quick

    events = []
    settings = replace(
        _settings(),
        mt4_fallback_enabled=True,
        mt4_terminal_path="C:\\Program Files\\MT4\\terminal.exe",
    )

    class MissingMt5Session:
        def __init__(self, **kwargs):
            events.append(("session", kwargs))

        def launch_terminal(self):
            events.append(("launch_mt5",))
            raise FileNotFoundError("missing mt5")

        def shutdown(self):
            events.append(("shutdown",))

    monkeypatch.setattr(quick, "configure_logging", lambda: None)
    monkeypatch.setattr(quick, "set_keep_awake", lambda: None)
    monkeypatch.setattr(quick, "ensure_requirements_satisfied", lambda: events.append(("requirements",)))
    monkeypatch.setattr(quick, "load_settings", lambda: settings)
    monkeypatch.setattr(quick, "create_mt5_module", lambda: "mt5")
    monkeypatch.setattr(quick, "Mt5Session", lambda **kwargs: MissingMt5Session(**kwargs))
    monkeypatch.setattr(quick, "launch_mt4_fallback", lambda fallback_settings: events.append(("mt4_fallback", fallback_settings)) or None)
    monkeypatch.setattr(quick, "run_quick_scalp_loop", lambda **kwargs: events.append(("quick_loop", kwargs)))

    assert quick.main() == 0

    fallback_event = [event for event in events if event[0] == "mt4_fallback"][0]
    assert fallback_event[1].mt4_fallback_enabled is True
    assert fallback_event[1].quick_scalp_enabled is True
    assert not any(event[0] == "quick_loop" for event in events)
    assert events[-1] == ("shutdown",)
