from __future__ import annotations

import math

from src.strategy.breakout import BreakoutDirection


def describe_retcode(mt5_module, retcode):
    if retcode == getattr(mt5_module, "TRADE_RETCODE_MARKET_CLOSED", None):
        return "market is closed for this symbol"
    if retcode == getattr(mt5_module, "TRADE_RETCODE_CLIENT_DISABLES_AT", None):
        return "algorithmic trading is disabled in the MT5 client"
    if retcode == getattr(mt5_module, "TRADE_RETCODE_INVALID_FILL", None):
        return "unsupported filling mode for this symbol/account"
    return f"retcode={retcode}"


def resolve_filling_mode(mt5_module, symbol_info):
    if symbol_info is not None:
        allowed_filling = getattr(symbol_info, "filling_mode", None)
        if allowed_filling is not None:
            # MetaTrader exposes SYMBOL_FILLING_MODE as a bitmask of allowed
            # symbol policies: 1 = FOK, 2 = IOC. Those values are not the same
            # enum as ORDER_FILLING_* (0 = FOK, 1 = IOC, 2 = RETURN).
            allowed_filling = int(allowed_filling)
            if allowed_filling & 1:
                return mt5_module.ORDER_FILLING_FOK
            if allowed_filling & 2:
                return mt5_module.ORDER_FILLING_IOC

        trade_exemode = getattr(symbol_info, "trade_exemode", None)
        trade_execution_market = getattr(mt5_module, "SYMBOL_TRADE_EXECUTION_MARKET", 2)
        if trade_exemode is not None and int(trade_exemode) != int(trade_execution_market):
            return mt5_module.ORDER_FILLING_RETURN
    return mt5_module.ORDER_FILLING_IOC


def choose_supported_filling_mode(mt5_module, request, preferred_mode):
    modes = _candidate_filling_modes(mt5_module, preferred_mode)

    order_check = getattr(mt5_module, "order_check", None)
    if order_check is None:
        return preferred_mode

    for mode in modes:
        check = order_check({**request, "type_filling": mode})
        retcode = getattr(check, "retcode", None)
        if retcode in (0, getattr(mt5_module, "TRADE_RETCODE_DONE", None)):
            return mode

    return preferred_mode


def _candidate_filling_modes(mt5_module, preferred_mode):
    modes = []
    for mode in [
        preferred_mode,
        getattr(mt5_module, "ORDER_FILLING_FOK", None),
        getattr(mt5_module, "ORDER_FILLING_IOC", None),
        getattr(mt5_module, "ORDER_FILLING_RETURN", None),
    ]:
        if mode is not None and mode not in modes:
            modes.append(mode)
    return modes


def send_order_with_filling_fallback(mt5_module, request, preferred_mode):
    last_result = None
    last_mode = preferred_mode
    invalid_fill_retcode = getattr(mt5_module, "TRADE_RETCODE_INVALID_FILL", None)

    for mode in _candidate_filling_modes(mt5_module, preferred_mode):
        candidate_request = {**request, "type_filling": mode}
        result = mt5_module.order_send(candidate_request)
        if result is None:
            code, message = mt5_module.last_error()
            raise RuntimeError(f"Order send failed: {code} {message}")
        last_result = result
        last_mode = mode
        if getattr(result, "retcode", None) == invalid_fill_retcode:
            continue
        return result, mode

    return last_result, last_mode


def build_open_order_request(mt5_module, symbol, side, lot, tick, filling_mode, comment):
    order_type = mt5_module.ORDER_TYPE_BUY if side == "buy" else mt5_module.ORDER_TYPE_SELL
    price = tick.ask if side == "buy" else tick.bid
    return {
        "action": mt5_module.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "deviation": 20,
        "type_time": mt5_module.ORDER_TIME_GTC,
        "type_filling": filling_mode,
        "comment": comment,
    }


def build_strategy_order_request(mt5_module, symbol, direction, lot, tick, filling_mode, stop_loss, take_profit, comment):
    side = "buy" if direction is BreakoutDirection.BULLISH else "sell"
    request = build_open_order_request(mt5_module, symbol, side, lot, tick, filling_mode, comment)
    request["sl"] = float(stop_loss)
    request["tp"] = float(take_profit)
    return request


def build_close_order_request(mt5_module, position, tick, filling_mode, comment):
    is_buy_position = position.type == mt5_module.ORDER_TYPE_BUY
    order_type = mt5_module.ORDER_TYPE_SELL if is_buy_position else mt5_module.ORDER_TYPE_BUY
    price = tick.bid if is_buy_position else tick.ask
    return {
        "action": mt5_module.TRADE_ACTION_DEAL,
        "position": position.ticket,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": order_type,
        "price": price,
        "deviation": 20,
        "type_time": mt5_module.ORDER_TIME_GTC,
        "type_filling": filling_mode,
        "comment": comment,
    }


def build_modify_sl_tp_request(mt5_module, position, stop_loss, take_profit=None):
    resolved_take_profit = getattr(position, "tp", None) if take_profit is None else take_profit
    return {
        "action": mt5_module.TRADE_ACTION_SLTP,
        "position": position.ticket,
        "symbol": position.symbol,
        "sl": float(stop_loss),
        "tp": float(resolved_take_profit) if resolved_take_profit is not None else 0.0,
    }


def normalize_lot_size(symbol_info, requested_lot):
    volume_min = float(getattr(symbol_info, "volume_min", requested_lot))
    volume_max = float(getattr(symbol_info, "volume_max", requested_lot))
    volume_step = float(getattr(symbol_info, "volume_step", 0.01))
    clamped = min(max(float(requested_lot), volume_min), volume_max)
    steps = math.floor((clamped - volume_min) / volume_step)
    normalized = volume_min + (steps * volume_step)
    return round(normalized, 8)


def _resolve_opened_position(mt5_module, *, result, symbol, order_type, lot, comment):
    positions = mt5_module.positions_get(ticket=getattr(result, "order", 0))
    if positions:
        return positions[0]

    positions = mt5_module.positions_get(symbol=symbol) or []
    matching_positions = [
        position
        for position in positions
        if str(getattr(position, "symbol", "")).upper() == str(symbol).upper()
        and int(getattr(position, "type", order_type)) == int(order_type)
        and abs(float(getattr(position, "volume", lot) or 0.0) - float(lot)) < 1e-8
        and str(getattr(position, "comment", "")).startswith(str(comment))
    ]
    if matching_positions:
        return matching_positions[-1]

    raise RuntimeError("Order reported success but no matching open position was found")


class TradeExecutor:
    def __init__(self, mt5_module):
        self.mt5_module = mt5_module

    def list_bot_positions(self, symbol, comment_prefix="strategy-live"):
        positions = self.mt5_module.positions_get(symbol=symbol) or []
        return [position for position in positions if str(getattr(position, "comment", "")).startswith(comment_prefix)]

    def open_test_trade(self, symbol, side, lot, comment):
        if not self.mt5_module.symbol_select(symbol, True):
            raise RuntimeError(f"Unable to select symbol: {symbol}")
        symbol_info = self.mt5_module.symbol_info(symbol)
        tick = self.mt5_module.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No tick data available for symbol: {symbol}")

        filling_mode = resolve_filling_mode(self.mt5_module, symbol_info)
        request = build_open_order_request(self.mt5_module, symbol, side, lot, tick, filling_mode, comment)
        request["type_filling"] = choose_supported_filling_mode(self.mt5_module, request, filling_mode)
        result, chosen_mode = send_order_with_filling_fallback(self.mt5_module, request, request["type_filling"])
        if getattr(result, "retcode", None) != self.mt5_module.TRADE_RETCODE_DONE:
            raise RuntimeError(f"Open trade rejected: {describe_retcode(self.mt5_module, result.retcode)}")

        order_type = self.mt5_module.ORDER_TYPE_BUY if side == "buy" else self.mt5_module.ORDER_TYPE_SELL
        return _resolve_opened_position(
            self.mt5_module,
            result=result,
            symbol=symbol,
            order_type=order_type,
            lot=lot,
            comment=comment,
        )

    def open_strategy_trade(self, symbol, direction, lot, stop_loss, take_profit, comment):
        if not self.mt5_module.symbol_select(symbol, True):
            raise RuntimeError(f"Unable to select symbol: {symbol}")
        symbol_info = self.mt5_module.symbol_info(symbol)
        tick = self.mt5_module.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No tick data available for symbol: {symbol}")
        normalized_lot = normalize_lot_size(symbol_info, lot)

        filling_mode = resolve_filling_mode(self.mt5_module, symbol_info)
        request = build_strategy_order_request(
            self.mt5_module,
            symbol,
            direction,
            normalized_lot,
            tick,
            filling_mode,
            stop_loss,
            take_profit,
            comment,
        )
        request["type_filling"] = choose_supported_filling_mode(self.mt5_module, request, filling_mode)
        result, chosen_mode = send_order_with_filling_fallback(self.mt5_module, request, request["type_filling"])
        if getattr(result, "retcode", None) != self.mt5_module.TRADE_RETCODE_DONE:
            raise RuntimeError(f"Strategy trade rejected: {describe_retcode(self.mt5_module, result.retcode)}")

        return _resolve_opened_position(
            self.mt5_module,
            result=result,
            symbol=symbol,
            order_type=request["type"],
            lot=normalized_lot,
            comment=comment,
        )

    def close_position(self, position, comment):
        symbol_info = self.mt5_module.symbol_info(position.symbol)
        tick = self.mt5_module.symbol_info_tick(position.symbol)
        if tick is None:
            raise RuntimeError(f"No tick data available for symbol: {position.symbol}")

        filling_mode = resolve_filling_mode(self.mt5_module, symbol_info)
        request = build_close_order_request(self.mt5_module, position, tick, filling_mode, comment)
        request["type_filling"] = choose_supported_filling_mode(self.mt5_module, request, filling_mode)
        result, chosen_mode = send_order_with_filling_fallback(self.mt5_module, request, request["type_filling"])
        if getattr(result, "retcode", None) != self.mt5_module.TRADE_RETCODE_DONE:
            raise RuntimeError(f"Close trade rejected: {describe_retcode(self.mt5_module, result.retcode)}")
        return result

    def update_position_stop_loss(self, position, stop_loss, take_profit=None):
        request = build_modify_sl_tp_request(
            self.mt5_module,
            position,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        result = self.mt5_module.order_send(request)
        if result is None:
            code, message = self.mt5_module.last_error()
            raise RuntimeError(f"Stop update send failed: {code} {message}")
        if getattr(result, "retcode", None) != self.mt5_module.TRADE_RETCODE_DONE:
            raise RuntimeError(f"Stop update rejected: {describe_retcode(self.mt5_module, result.retcode)}")
        return result
