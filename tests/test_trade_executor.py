from types import SimpleNamespace

import pytest

from src.trade_executor import build_close_order_request, build_modify_sl_tp_request, build_open_order_request
from src.trade_executor import normalize_lot_size, resolve_filling_mode, TradeExecutor
from src.strategy.breakout import BreakoutDirection


class FakeMt5:
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 10
    TRADE_ACTION_SLTP = 11
    ORDER_TIME_GTC = 20
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_MARKET_CLOSED = 10018
    TRADE_RETCODE_CLIENT_DISABLES_AT = 10027
    TRADE_RETCODE_INVALID_FILL = 10030
    SYMBOL_TRADE_EXECUTION_MARKET = 2


def test_build_open_order_request_for_buy_uses_ask_price():
    tick = SimpleNamespace(ask=2350.5, bid=2350.1)

    request = build_open_order_request(FakeMt5(), "XAUUSD", "buy", 0.01, tick, 0, "skeleton-test")

    assert request["symbol"] == "XAUUSD"
    assert request["type"] == FakeMt5.ORDER_TYPE_BUY
    assert request["price"] == 2350.5
    assert request["type_filling"] == 0
    assert request["comment"] == "skeleton-test"


def test_resolve_filling_mode_maps_symbol_bitmask_to_order_fill_enum():
    symbol_info = SimpleNamespace(filling_mode=1, trade_exemode=FakeMt5.SYMBOL_TRADE_EXECUTION_MARKET)

    filling_mode = resolve_filling_mode(FakeMt5(), symbol_info)

    assert filling_mode == FakeMt5.ORDER_FILLING_FOK


def test_build_close_order_request_targets_existing_position():
    tick = SimpleNamespace(ask=2350.5, bid=2350.1)
    position = SimpleNamespace(ticket=99, symbol="XAUUSD", volume=0.01, type=FakeMt5.ORDER_TYPE_BUY)

    request = build_close_order_request(FakeMt5(), position, tick, 1, "close-test")

    assert request["position"] == 99
    assert request["symbol"] == "XAUUSD"
    assert request["type"] == FakeMt5.ORDER_TYPE_SELL
    assert request["price"] == 2350.1
    assert request["type_filling"] == 1
    assert request["comment"] == "close-test"


def test_open_test_trade_falls_back_to_supported_filling_mode():
    mt5 = FakeMt5()
    captured_requests = []
    mt5.symbol_select = lambda symbol, enabled: True
    mt5.symbol_info = lambda symbol: SimpleNamespace(filling_mode=1)
    mt5.symbol_info_tick = lambda symbol: SimpleNamespace(ask=2350.5, bid=2350.1)
    mt5.order_send = lambda request: captured_requests.append(request) or SimpleNamespace(retcode=mt5.TRADE_RETCODE_DONE, order=77)
    mt5.positions_get = lambda **kwargs: [SimpleNamespace(ticket=77, symbol="XAUUSD", volume=0.01, type=mt5.ORDER_TYPE_BUY)]
    mt5.last_error = lambda: (0, "ok")
    mt5.order_check = lambda request: SimpleNamespace(
        retcode=0 if request["type_filling"] == mt5.ORDER_FILLING_FOK else mt5.TRADE_RETCODE_INVALID_FILL,
        comment="Done" if request["type_filling"] == mt5.ORDER_FILLING_FOK else "Unsupported filling mode",
    )

    executor = TradeExecutor(mt5)
    executor.open_test_trade("XAUUSD", "buy", 0.01, "skeleton-test")

    assert captured_requests[0]["type_filling"] == 0


def test_open_test_trade_reports_when_client_algo_trading_is_disabled():
    mt5 = FakeMt5()
    mt5.symbol_select = lambda symbol, enabled: True
    mt5.symbol_info = lambda symbol: SimpleNamespace(filling_mode=0)
    mt5.symbol_info_tick = lambda symbol: SimpleNamespace(ask=2350.5, bid=2350.1)
    mt5.order_send = lambda request: SimpleNamespace(retcode=mt5.TRADE_RETCODE_CLIENT_DISABLES_AT)
    mt5.order_check = lambda request: SimpleNamespace(retcode=0, comment="Done")
    mt5.last_error = lambda: (0, "ok")

    executor = TradeExecutor(mt5)

    with pytest.raises(RuntimeError, match="algorithmic trading is disabled in the MT5 client"):
        executor.open_test_trade("XAUUSD", "buy", 0.01, "skeleton-test")


def test_open_test_trade_reports_when_market_is_closed():
    mt5 = FakeMt5()
    mt5.symbol_select = lambda symbol, enabled: True
    mt5.symbol_info = lambda symbol: SimpleNamespace(filling_mode=0)
    mt5.symbol_info_tick = lambda symbol: SimpleNamespace(ask=2350.5, bid=2350.1)
    mt5.order_check = lambda request: SimpleNamespace(retcode=0, comment="Done")
    mt5.order_send = lambda request: SimpleNamespace(retcode=mt5.TRADE_RETCODE_MARKET_CLOSED)
    mt5.last_error = lambda: (0, "ok")

    executor = TradeExecutor(mt5)

    with pytest.raises(RuntimeError, match="market is closed for this symbol"):
        executor.open_test_trade("XAUUSD", "buy", 0.01, "skeleton-test")


def test_open_strategy_trade_sends_stop_loss_and_take_profit():
    mt5 = FakeMt5()
    captured_requests = []
    mt5.symbol_select = lambda symbol, enabled: True
    mt5.symbol_info = lambda symbol: SimpleNamespace(filling_mode=0)
    mt5.symbol_info_tick = lambda symbol: SimpleNamespace(ask=2350.5, bid=2350.1)
    mt5.order_check = lambda request: SimpleNamespace(retcode=0, comment="Done")
    mt5.order_send = lambda request: captured_requests.append(request) or SimpleNamespace(
        retcode=mt5.TRADE_RETCODE_DONE,
        order=88,
    )
    mt5.positions_get = lambda **kwargs: [SimpleNamespace(ticket=88, symbol="XAUUSD", volume=0.01, type=mt5.ORDER_TYPE_BUY)]
    mt5.last_error = lambda: (0, "ok")

    executor = TradeExecutor(mt5)
    position = executor.open_strategy_trade(
        symbol="XAUUSD",
        direction=BreakoutDirection.BULLISH,
        lot=0.01,
        stop_loss=2345.0,
        take_profit=2365.0,
        comment="strategy-live",
    )

    assert position.ticket == 88
    assert captured_requests[0]["sl"] == 2345.0
    assert captured_requests[0]["tp"] == 2365.0
    assert captured_requests[0]["comment"] == "strategy-live"


def test_open_strategy_trade_uses_fok_when_symbol_bitmask_is_fok():
    mt5 = FakeMt5()
    captured_requests = []
    mt5.symbol_select = lambda symbol, enabled: True
    mt5.symbol_info = lambda symbol: SimpleNamespace(
        filling_mode=1,
        trade_exemode=mt5.SYMBOL_TRADE_EXECUTION_MARKET,
        volume_min=0.01,
        volume_max=1.0,
        volume_step=0.01,
    )
    mt5.symbol_info_tick = lambda symbol: SimpleNamespace(ask=2350.5, bid=2350.1)
    mt5.order_check = lambda request: SimpleNamespace(
        retcode=0 if request["type_filling"] == mt5.ORDER_FILLING_FOK else mt5.TRADE_RETCODE_INVALID_FILL,
        comment="Done" if request["type_filling"] == mt5.ORDER_FILLING_FOK else "Unsupported filling mode",
    )
    mt5.order_send = lambda request: captured_requests.append(request) or SimpleNamespace(
        retcode=mt5.TRADE_RETCODE_DONE,
        order=88,
    )
    mt5.positions_get = lambda **kwargs: [SimpleNamespace(ticket=88, symbol="XAUUSD", volume=0.01, type=mt5.ORDER_TYPE_BUY)]
    mt5.last_error = lambda: (0, "ok")

    executor = TradeExecutor(mt5)
    executor.open_strategy_trade(
        symbol="XAUUSD",
        direction=BreakoutDirection.BULLISH,
        lot=0.01,
        stop_loss=2345.0,
        take_profit=2365.0,
        comment="strategy-live",
    )

    assert captured_requests[0]["type_filling"] == mt5.ORDER_FILLING_FOK


def test_update_position_stop_loss_builds_modify_request():
    position = SimpleNamespace(ticket=88, symbol="XAUUSD", tp=2365.0)

    request = build_modify_sl_tp_request(FakeMt5(), position, stop_loss=2352.5)

    assert request["action"] == FakeMt5.TRADE_ACTION_SLTP
    assert request["position"] == 88
    assert request["symbol"] == "XAUUSD"
    assert request["sl"] == 2352.5
    assert request["tp"] == 2365.0


def test_normalize_lot_size_uses_symbol_volume_step_and_limits():
    symbol_info = SimpleNamespace(volume_min=0.01, volume_max=1.0, volume_step=0.01)

    normalized = normalize_lot_size(symbol_info, 0.037)

    assert normalized == pytest.approx(0.03)


def test_list_bot_positions_filters_by_comment_prefix():
    mt5 = FakeMt5()
    mt5.positions_get = lambda **kwargs: [
        SimpleNamespace(ticket=1, symbol="XAUUSD", comment="strategy-live:001"),
        SimpleNamespace(ticket=2, symbol="XAUUSD", comment="manual"),
        SimpleNamespace(ticket=3, symbol="XAUUSD", comment="strategy-live:002"),
    ]

    executor = TradeExecutor(mt5)
    positions = executor.list_bot_positions("XAUUSD")

    assert [position.ticket for position in positions] == [1, 3]
