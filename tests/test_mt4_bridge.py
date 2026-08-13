from types import SimpleNamespace


def test_mt4_bridge_synthesizes_m1_rates_from_tick_when_history_missing(tmp_path):
    from src.mt4_bridge import Mt4BridgeModule
    from src.market_data import fetch_candles

    (tmp_path / "fx_bridge_tick_XAUUSD.csv").write_text(
        "1779292800,4502.67,4503.02,0.01,2,0.01,100,0.01,0,0\n",
        encoding="utf-8",
    )
    bridge = Mt4BridgeModule(tmp_path)

    candles = fetch_candles(bridge, "XAUUSD", "TIMEFRAME_M1", 30, minimum=1)

    assert len(candles) == 30
    assert candles[-1].open == 4502.845
    assert candles[-1].close == 4502.845
    assert candles[-1].timeframe == "M1"


def test_mt4_bridge_uses_tick_history_for_recent_ticks_and_latest_m1_direction(tmp_path):
    from src.mt4_bridge import Mt4BridgeModule

    bridge = Mt4BridgeModule(tmp_path)
    bridge._remember_tick("XAUUSD", SimpleNamespace(time=1, bid=100.0, ask=100.2))
    bridge._remember_tick("XAUUSD", SimpleNamespace(time=2, bid=101.0, ask=101.2))
    (tmp_path / "fx_bridge_tick_XAUUSD.csv").write_text(
        "3,102.0,102.2,0.01,2,0.01,100,0.01,0,0\n",
        encoding="utf-8",
    )

    ticks = bridge.copy_ticks_from_pos("XAUUSD", 0, 100, bridge.COPY_TICKS_ALL)
    rates = bridge.copy_rates_from_pos("XAUUSD", bridge.TIMEFRAME_M1, 1, 30)

    assert [tick.bid for tick in ticks] == [100.0, 101.0, 102.0]
    assert len(rates) == 30
    assert rates[-1]["open"] == 101.1
    assert rates[-1]["close"] == 102.1


def test_mt4_bridge_recovers_recent_ticks_from_persisted_history_file(tmp_path):
    from src.mt4_bridge import Mt4BridgeModule

    (tmp_path / "fx_bridge_ticks_XAUUSD.csv").write_text(
        "\n".join(
            [
                "1,100.0,100.2",
                "2,100.3,100.5",
                "3,100.6,100.8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "fx_bridge_tick_XAUUSD.csv").write_text(
        "4,100.9,101.1,0.01,2,0.01,100,0.01,0,0\n",
        encoding="utf-8",
    )

    ticks = Mt4BridgeModule(tmp_path).copy_ticks_from_pos("XAUUSD", 0, 10, 0)

    assert [tick.time for tick in ticks] == [1, 2, 3, 4]
    assert [tick.bid for tick in ticks] == [100.0, 100.3, 100.6, 100.9]


def test_mt4_bridge_deduplicates_latest_tick_when_history_file_already_has_it(tmp_path):
    from src.mt4_bridge import Mt4BridgeModule

    (tmp_path / "fx_bridge_ticks_XAUUSD.csv").write_text(
        "1,100.0,100.2\n2,100.3,100.5\n",
        encoding="utf-8",
    )
    (tmp_path / "fx_bridge_tick_XAUUSD.csv").write_text(
        "2,100.3,100.5,0.01,2,0.01,100,0.01,0,0\n",
        encoding="utf-8",
    )

    ticks = Mt4BridgeModule(tmp_path).copy_ticks_from_pos("XAUUSD", 0, 10, 0)

    assert [tick.time for tick in ticks] == [1, 2]


def test_mt4_bridge_reports_persisted_tick_history_count_and_path(tmp_path):
    from src.mt4_bridge import Mt4BridgeModule

    (tmp_path / "fx_bridge_ticks_XAUUSD.csv").write_text(
        "1,100.0,100.2\n2,100.3,100.5\n",
        encoding="utf-8",
    )

    bridge = Mt4BridgeModule(tmp_path)

    assert bridge.persisted_tick_history_count("XAUUSD") == 2
    assert bridge.tick_history_file_path("XAUUSD") == tmp_path / "fx_bridge_ticks_XAUUSD.csv"


def test_mt4_bridge_returns_none_for_synthetic_rates_without_tick(tmp_path):
    from src.mt4_bridge import Mt4BridgeModule

    bridge = Mt4BridgeModule(tmp_path)

    assert bridge.copy_rates_from_pos("XAUUSD", bridge.TIMEFRAME_M1, 1, 30) is None


def test_mt4_bridge_backfills_short_rate_file_from_tick(tmp_path):
    from src.mt4_bridge import Mt4BridgeModule

    (tmp_path / "fx_bridge_tick_XAUUSD.csv").write_text(
        "1779292800,4502.67,4503.02,0.01,2,0.01,100,0.01,0,0\n",
        encoding="utf-8",
    )
    (tmp_path / "fx_bridge_rates_XAUUSD_M15.csv").write_text(
        "1779291900,4490,4491,4489,4490.5,12\n",
        encoding="utf-8",
    )
    bridge = Mt4BridgeModule(tmp_path)

    rates = bridge.copy_rates_from_pos("XAUUSD", bridge.TIMEFRAME_M15, 1, 50)

    assert len(rates) == 50
    assert rates[-1]["open"] == 4490.0
    assert rates[-1]["close"] == 4490.5


def test_mt4_bridge_account_info_exposes_trade_allowed(tmp_path):
    from src.mt4_bridge import Mt4BridgeModule

    (tmp_path / "fx_bridge_heartbeat.csv").write_text(
        "2026.08.13 15:07:08,60296768,152.5,152.5,152.5,false\n",
        encoding="utf-8",
    )

    account = Mt4BridgeModule(tmp_path).account_info()

    assert account.login == 60296768
    assert account.balance == 152.5
    assert account.trade_allowed is False


def test_mt4_bridge_retries_empty_file_read(tmp_path, monkeypatch):
    from pathlib import Path

    from src.mt4_bridge import Mt4BridgeModule

    heartbeat = tmp_path / "fx_bridge_heartbeat.csv"
    heartbeat.write_text("2026.08.13 15:07:08,60340356,596,596,596,true\n", encoding="utf-8")
    original_read_text = Path.read_text
    attempts = {"count": 0}

    def flaky_read_text(self, *args, **kwargs):
        if self == heartbeat:
            attempts["count"] += 1
            if attempts["count"] == 1:
                return ""
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)

    account = Mt4BridgeModule(tmp_path).account_info()

    assert account.login == 60340356
    assert account.balance == 596.0
    assert account.trade_allowed is True


def test_mt4_bridge_close_position_sends_ticket_selector(tmp_path, monkeypatch):
    from src.mt4_bridge import Mt4BridgeModule
    from src.trade_executor import TradeExecutor

    bridge = Mt4BridgeModule(tmp_path)
    bridge.client.wait_for_result = lambda command_id: SimpleNamespace(
        command_id=command_id,
        status="OK",
        message="closed",
        ticket=123,
        error_code=0,
    )
    monkeypatch.setattr(bridge, "symbol_info", lambda symbol: SimpleNamespace(filling_mode=2))
    monkeypatch.setattr(bridge, "symbol_info_tick", lambda symbol: SimpleNamespace(bid=100.0, ask=100.2))

    TradeExecutor(bridge).close_position(
        SimpleNamespace(ticket=123, symbol="XAUUSD", volume=0.01, type=bridge.ORDER_TYPE_BUY),
        comment="unit-close",
    )

    command = (tmp_path / "fx_bridge_commands.csv").read_text(encoding="utf-8").strip().split(",")
    assert command[1] == "CLOSE"
    assert command[3] == "123.0"


def test_mt4_bridge_order_send_maps_broker_side_to_command_action(tmp_path):
    from src.mt4_bridge import Mt4BridgeModule

    bridge = Mt4BridgeModule(tmp_path)
    bridge.client.wait_for_result = lambda command_id: SimpleNamespace(
        command_id=command_id,
        status="OK",
        message="opened",
        ticket=321,
        error_code=0,
    )

    result = bridge.order_send(
        {
            "action": bridge.TRADE_ACTION_DEAL,
            "symbol": "XAUUSD",
            "volume": 0.01,
            "type": bridge.ORDER_TYPE_SELL,
            "sl": 2351.1,
            "tp": 2349.1,
            "comment": "quick-scalp",
        }
    )

    command = (tmp_path / "fx_bridge_commands.csv").read_text(encoding="utf-8").strip().split(",")
    assert result.retcode == bridge.TRADE_RETCODE_DONE
    assert command[1] == "SELL"
    assert command[2] == "XAUUSD"
    assert command[3] == "0.01"
    assert command[4] == "2351.1"
    assert command[5] == "2349.1"
    assert command[6] == "quick-scalp"


def test_mt4_bridge_stop_update_sends_ticket_selector(tmp_path, monkeypatch):
    from src.mt4_bridge import Mt4BridgeModule
    from src.trade_executor import TradeExecutor

    bridge = Mt4BridgeModule(tmp_path)
    bridge.client.wait_for_result = lambda command_id: SimpleNamespace(
        command_id=command_id,
        status="OK",
        message="modified",
        ticket=456,
        error_code=0,
    )
    monkeypatch.setattr(bridge, "symbol_info", lambda symbol: SimpleNamespace(filling_mode=2))

    TradeExecutor(bridge).update_position_stop_loss(
        SimpleNamespace(ticket=456, symbol="XAUUSD", tp=105.0),
        stop_loss=99.0,
    )

    command = (tmp_path / "fx_bridge_commands.csv").read_text(encoding="utf-8").strip().split(",")
    assert command[1] == "MODIFY"
    assert command[3] == "456.0"


def test_mt4_bridge_retries_command_write_when_file_is_briefly_locked(tmp_path, monkeypatch):
    from pathlib import Path

    from src.mt4_bridge import Mt4FileBridgeClient

    original_write_text = Path.write_text
    attempts = {"count": 0}

    def flaky_write_text(self, text, *args, **kwargs):
        if self.name == "fx_bridge_commands.csv":
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise PermissionError("locked by MT4")
        return original_write_text(self, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", flaky_write_text)
    client = Mt4FileBridgeClient(tmp_path)

    command_id = client.open_order("BUY", "XAUUSD", 0.01, 99.0, 101.0, "unit")

    assert attempts["count"] == 2
    assert (tmp_path / "fx_bridge_commands.csv").read_text(encoding="utf-8").startswith(command_id)


def test_mt4_bridge_retries_result_read_when_file_is_briefly_locked(tmp_path, monkeypatch):
    from pathlib import Path

    from src.mt4_bridge import Mt4FileBridgeClient

    result_file = tmp_path / "fx_bridge_results.csv"
    result_file.write_text("abc,OK,opened,123,0\n", encoding="utf-8")
    original_read_text = Path.read_text
    attempts = {"count": 0}

    def flaky_read_text(self, *args, **kwargs):
        if self == result_file:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise PermissionError("locked by MT4")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)

    result = Mt4FileBridgeClient(tmp_path).read_result()

    assert attempts["count"] == 2
    assert result is not None
    assert result.command_id == "abc"
    assert result.ticket == 123
