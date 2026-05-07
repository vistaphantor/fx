from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import sleep

from src.market_data import fetch_candles
from src.strategy.breakout import BreakoutDirection
from src.strategy.structural import detect_fair_value_gaps


QUICK_COMMENT_PREFIX = "quick-scalp"


def fetch_m1_candles(mt5_module, symbol: str, count: int = 2):
    return fetch_candles(mt5_module, symbol, "TIMEFRAME_M1", count, minimum=1)


def fetch_m15_candles(mt5_module, symbol: str, count: int = 50):
    return fetch_candles(mt5_module, symbol, "TIMEFRAME_M15", count, minimum=10)


def resolve_m1_direction(candles) -> BreakoutDirection | None:
    if not candles:
        return None
    latest = candles[-1]
    open_price = float(getattr(latest, "open"))
    close_price = float(getattr(latest, "close"))
    if close_price > open_price:
        return BreakoutDirection.BULLISH
    if close_price < open_price:
        return BreakoutDirection.BEARISH
    return None


def fetch_recent_ticks(mt5_module, symbol: str, count: int = 60):
    flags = getattr(mt5_module, "COPY_TICKS_ALL", 0)
    if hasattr(mt5_module, "copy_ticks_from_pos"):
        ticks = mt5_module.copy_ticks_from_pos(symbol, 0, count, flags)
        return list(ticks) if ticks is not None else []
    if hasattr(mt5_module, "copy_ticks_from"):
        ticks = mt5_module.copy_ticks_from(symbol, datetime.now(timezone.utc), count, flags)
        return list(ticks) if ticks is not None else []
    return None


def resolve_tick_direction(ticks, *, point: float = 0.0):
    if ticks is None:
        return QuickTickGuidance(None, "tick_api_unavailable", 0, 0.0, 0, 0)
    if len(ticks) < 3:
        return QuickTickGuidance(None, "insufficient_ticks", len(ticks), 0.0, 0, 0)

    mids = [_tick_mid_price(tick) for tick in ticks]
    mids = [mid for mid in mids if mid > 0.0]
    if len(mids) < 3:
        return QuickTickGuidance(None, "invalid_ticks", len(ticks), 0.0, 0, 0)

    up_moves = 0
    down_moves = 0
    for previous_mid, current_mid in zip(mids, mids[1:]):
        if current_mid > previous_mid:
            up_moves += 1
        elif current_mid < previous_mid:
            down_moves += 1

    net_move = mids[-1] - mids[0]
    minimum_move = max(float(point or 0.0) * 2.0, abs(mids[-1]) * 0.000005)
    if net_move >= minimum_move and up_moves > down_moves:
        return QuickTickGuidance(BreakoutDirection.BULLISH, "tick_momentum", len(mids), net_move, up_moves, down_moves)
    if net_move <= -minimum_move and down_moves > up_moves:
        return QuickTickGuidance(BreakoutDirection.BEARISH, "tick_momentum", len(mids), net_move, up_moves, down_moves)
    return QuickTickGuidance(None, "tick_chop", len(mids), net_move, up_moves, down_moves)


def _tick_mid_price(tick) -> float:
    bid = _tick_value(tick, "bid")
    ask = _tick_value(tick, "ask")
    if bid > 0.0 and ask > 0.0:
        return (bid + ask) / 2.0
    return max(bid, ask)


def _tick_value(tick, name: str) -> float:
    if hasattr(tick, name):
        return float(getattr(tick, name) or 0.0)
    try:
        return float(tick[name] or 0.0)
    except (KeyError, IndexError, TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class QuickFibonacciGuidance:
    direction: BreakoutDirection | None
    zone: str
    current_price: float
    swing_low: float
    swing_high: float

    def allows(self, direction: BreakoutDirection) -> bool:
        return self.direction is direction and self.zone in {
            "in_market_mover",
            "towards_market_mover",
            "golden_zone",
        }


@dataclass(frozen=True)
class QuickIndicatorGuidance:
    rsi: float
    sar: float
    sar_direction: BreakoutDirection | None

    def allows(self, direction: BreakoutDirection) -> bool:
        if self.sar_direction is not direction:
            return False
        if direction is BreakoutDirection.BULLISH:
            return self.rsi < 70.0
        return self.rsi > 30.0


@dataclass(frozen=True)
class QuickFvgGuidance:
    direction: BreakoutDirection | None
    top: float
    bottom: float
    bars_since: int
    current_price: float
    reason: str

    def allows(self, direction: BreakoutDirection) -> bool:
        return self.direction is direction and self.reason == "matching_fvg"


@dataclass(frozen=True)
class QuickTickGuidance:
    direction: BreakoutDirection | None
    reason: str
    tick_count: int
    net_move: float
    up_moves: int
    down_moves: int


@dataclass(frozen=True)
class QuickSpreadState:
    allowed: bool
    reason: str
    spread: float
    spread_pips: float
    max_spread_pips: float


@dataclass(frozen=True)
class QuickGridPermission:
    allowed: bool
    reason: str
    max_new_entries: int
    min_spacing: float
    current_price: float


@dataclass(frozen=True)
class QuickGuidanceDecision:
    allowed: bool
    reason: str
    max_new_entries_override: int | None
    fib_allows: bool
    sar_allows: bool
    rsi_allows: bool


def resolve_quick_fibonacci_guidance(candles) -> QuickFibonacciGuidance:
    if len(candles) < 2:
        return QuickFibonacciGuidance(None, "unavailable", 0.0, 0.0, 0.0)

    swing_low_index, swing_low_candle = min(enumerate(candles), key=lambda item: float(getattr(item[1], "low")))
    swing_high_index, swing_high_candle = max(enumerate(candles), key=lambda item: float(getattr(item[1], "high")))
    swing_low = float(getattr(swing_low_candle, "low"))
    swing_high = float(getattr(swing_high_candle, "high"))
    current_price = float(getattr(candles[-1], "close"))
    impulse_range = swing_high - swing_low
    if impulse_range <= 0:
        return QuickFibonacciGuidance(None, "flat", current_price, swing_low, swing_high)

    if swing_low_index < swing_high_index:
        zone = _classify_bullish_fibonacci_zone(current_price, swing_low, swing_high)
        return QuickFibonacciGuidance(BreakoutDirection.BULLISH, zone, current_price, swing_low, swing_high)

    zone = _classify_bearish_fibonacci_zone(current_price, swing_low, swing_high)
    return QuickFibonacciGuidance(BreakoutDirection.BEARISH, zone, current_price, swing_low, swing_high)


def resolve_quick_fvg_guidance(
    candles,
    *,
    direction: BreakoutDirection,
    current_price: float,
    max_bars_since: int = 20,
) -> QuickFvgGuidance:
    direction_value = 1 if direction is BreakoutDirection.BULLISH else -1
    matching_gaps = [
        gap
        for gap in detect_fair_value_gaps(candles, lookback=max_bars_since)
        if int(getattr(gap, "direction", 0)) == direction_value
        and int(getattr(gap, "bars_since", max_bars_since + 1)) <= int(max_bars_since)
        and _fvg_is_on_chase_side(gap, direction, current_price)
    ]
    if not matching_gaps:
        return QuickFvgGuidance(None, 0.0, 0.0, 0, float(current_price), "no_matching_fvg")

    nearest_gap = min(
        matching_gaps,
        key=lambda gap: abs(float(current_price) - ((float(gap.top) + float(gap.bottom)) / 2.0)),
    )
    return QuickFvgGuidance(
        direction,
        float(nearest_gap.top),
        float(nearest_gap.bottom),
        int(getattr(nearest_gap, "bars_since", 0)),
        float(current_price),
        "matching_fvg",
    )


def _fvg_is_on_chase_side(fvg, direction: BreakoutDirection, current_price: float) -> bool:
    top = float(getattr(fvg, "top"))
    bottom = float(getattr(fvg, "bottom"))
    if direction is BreakoutDirection.BULLISH:
        return float(current_price) >= bottom
    return float(current_price) <= top


def _classify_bullish_fibonacci_zone(current_price: float, swing_low: float, swing_high: float) -> str:
    impulse_range = swing_high - swing_low
    retrace_382 = swing_high - (impulse_range * 0.382)
    retrace_618 = swing_high - (impulse_range * 0.618)
    retrace_786 = swing_high - (impulse_range * 0.786)
    if current_price >= retrace_382:
        return "in_market_mover"
    if current_price >= retrace_618:
        return "towards_market_mover"
    if current_price >= retrace_786:
        return "golden_zone"
    return "outside"


def _classify_bearish_fibonacci_zone(current_price: float, swing_low: float, swing_high: float) -> str:
    impulse_range = swing_high - swing_low
    retrace_382 = swing_low + (impulse_range * 0.382)
    retrace_618 = swing_low + (impulse_range * 0.618)
    retrace_786 = swing_low + (impulse_range * 0.786)
    if current_price <= retrace_382:
        return "in_market_mover"
    if current_price <= retrace_618:
        return "towards_market_mover"
    if current_price <= retrace_786:
        return "golden_zone"
    return "outside"


def resolve_quick_indicator_guidance(candles) -> QuickIndicatorGuidance:
    if len(candles) < 15:
        return QuickIndicatorGuidance(50.0, 0.0, None)

    rsi = _calculate_rsi([float(getattr(candle, "close")) for candle in candles], period=14)
    sar = _calculate_parabolic_sar(candles)
    latest_close = float(getattr(candles[-1], "close"))
    if latest_close > sar:
        sar_direction = BreakoutDirection.BULLISH
    elif latest_close < sar:
        sar_direction = BreakoutDirection.BEARISH
    else:
        sar_direction = None
    return QuickIndicatorGuidance(rsi, sar, sar_direction)


def resolve_quick_guidance_decision(
    *,
    direction: BreakoutDirection,
    fibonacci_guidance: QuickFibonacciGuidance,
    indicator_guidance: QuickIndicatorGuidance,
) -> QuickGuidanceDecision:
    fib_allows = fibonacci_guidance.allows(direction)
    sar_allows = indicator_guidance.sar_direction is direction
    rsi_allows = _rsi_allows_direction(indicator_guidance.rsi, direction)
    if not rsi_allows:
        return QuickGuidanceDecision(False, "indicator_filter", None, fib_allows, sar_allows, rsi_allows)
    if fib_allows and sar_allows:
        return QuickGuidanceDecision(True, "full_guidance", None, fib_allows, sar_allows, rsi_allows)
    if fib_allows or sar_allows:
        return QuickGuidanceDecision(True, "mixed_guidance", 1, fib_allows, sar_allows, rsi_allows)
    return QuickGuidanceDecision(True, "signal_override", 1, fib_allows, sar_allows, rsi_allows)


def _rsi_allows_direction(rsi: float, direction: BreakoutDirection) -> bool:
    if direction is BreakoutDirection.BULLISH:
        return rsi < 70.0
    return rsi > 30.0


def _calculate_rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) <= period:
        return 50.0

    changes = [closes[index] - closes[index - 1] for index in range(1, len(closes))]
    recent_changes = changes[-period:]
    gains = [max(change, 0.0) for change in recent_changes]
    losses = [abs(min(change, 0.0)) for change in recent_changes]
    average_gain = sum(gains) / period
    average_loss = sum(losses) / period
    if average_loss == 0:
        return 100.0 if average_gain > 0 else 50.0
    relative_strength = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def _calculate_parabolic_sar(candles, step: float = 0.02, maximum: float = 0.2) -> float:
    highs = [float(getattr(candle, "high")) for candle in candles]
    lows = [float(getattr(candle, "low")) for candle in candles]
    if len(candles) < 2:
        return lows[-1] if lows else 0.0

    bullish = float(getattr(candles[1], "close")) >= float(getattr(candles[0], "close"))
    sar = lows[0] if bullish else highs[0]
    extreme_point = highs[0] if bullish else lows[0]
    acceleration = step

    for index in range(1, len(candles)):
        previous_sar = sar
        sar = previous_sar + acceleration * (extreme_point - previous_sar)

        if bullish:
            if lows[index] < sar:
                bullish = False
                sar = extreme_point
                extreme_point = lows[index]
                acceleration = step
            else:
                sar = min(sar, lows[index - 1], lows[index])
                if highs[index] > extreme_point:
                    extreme_point = highs[index]
                    acceleration = min(acceleration + step, maximum)
        else:
            if highs[index] > sar:
                bullish = True
                sar = extreme_point
                extreme_point = highs[index]
                acceleration = step
            else:
                sar = max(sar, highs[index - 1], highs[index])
                if lows[index] < extreme_point:
                    extreme_point = lows[index]
                    acceleration = min(acceleration + step, maximum)

    return sar


def resolve_quick_grid_permission(
    *,
    candles,
    positions,
    direction: BreakoutDirection,
    fibonacci_guidance: QuickFibonacciGuidance,
    mt5_module,
    tick,
) -> QuickGridPermission:
    current_price = _grid_current_price(direction, tick)
    atr = _calculate_average_true_range(candles[-14:]) if len(candles) >= 2 else 0.0
    min_spacing = max(atr * 0.35, abs(current_price) * 0.00003)
    max_new_entries = _max_new_entries_for_zone(fibonacci_guidance.zone)

    if not _latest_candle_has_tradeable_body(candles):
        return QuickGridPermission(False, "weak_candle", max_new_entries, min_spacing, current_price)

    same_side_positions = [
        position
        for position in positions
        if _position_direction(position, mt5_module) is direction
    ]
    for position in same_side_positions:
        open_price = float(getattr(position, "price_open", getattr(position, "price", current_price)) or current_price)
        if abs(current_price - open_price) < min_spacing:
            return QuickGridPermission(False, "grid_spacing", max_new_entries, min_spacing, current_price)

    return QuickGridPermission(True, "ok", max_new_entries, min_spacing, current_price)


def _grid_current_price(direction: BreakoutDirection, tick) -> float:
    if direction is BreakoutDirection.BULLISH:
        return float(getattr(tick, "ask", getattr(tick, "bid", 0.0)) or 0.0)
    return float(getattr(tick, "bid", getattr(tick, "ask", 0.0)) or 0.0)


def _max_new_entries_for_zone(zone: str) -> int:
    if zone == "golden_zone":
        return 3
    if zone == "towards_market_mover":
        return 2
    if zone == "in_market_mover":
        return 1
    return 0


def _latest_candle_has_tradeable_body(candles) -> bool:
    if not candles:
        return False
    latest = candles[-1]
    high = float(getattr(latest, "high"))
    low = float(getattr(latest, "low"))
    open_price = float(getattr(latest, "open", getattr(latest, "close")))
    close_price = float(getattr(latest, "close"))
    candle_range = high - low
    if candle_range <= 0:
        return False
    return abs(close_price - open_price) / candle_range >= 0.35


def _position_direction(position, mt5_module) -> BreakoutDirection | None:
    position_type = getattr(position, "type", None)
    if position_type is None:
        return None
    if int(position_type) == int(getattr(mt5_module, "ORDER_TYPE_BUY", 0)):
        return BreakoutDirection.BULLISH
    if int(position_type) == int(getattr(mt5_module, "ORDER_TYPE_SELL", 1)):
        return BreakoutDirection.BEARISH
    return None


def _calculate_average_true_range(candles) -> float:
    if len(candles) < 2:
        return 0.0

    true_ranges = []
    previous_close = float(getattr(candles[0], "close"))
    for candle in candles[1:]:
        high = float(getattr(candle, "high"))
        low = float(getattr(candle, "low"))
        true_ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
        previous_close = float(getattr(candle, "close"))
    if not true_ranges:
        return 0.0
    return sum(true_ranges) / len(true_ranges)


def resolve_spread_state(*, mt5_module, symbol: str, max_spread_pips: float = 3.5) -> QuickSpreadState:
    tick = mt5_module.symbol_info_tick(symbol) if hasattr(mt5_module, "symbol_info_tick") else None
    if tick is None:
        return QuickSpreadState(False, "no_tick_for_spread", 0.0, 0.0, float(max_spread_pips))
    spread = _tick_value(tick, "ask") - _tick_value(tick, "bid")
    symbol_info = mt5_module.symbol_info(symbol) if hasattr(mt5_module, "symbol_info") else None
    point = float(getattr(symbol_info, "point", 0.0) or 0.0)
    if point <= 0.0:
        return QuickSpreadState(True, "spread_point_unavailable", spread, 0.0, float(max_spread_pips))
    pip_size = _symbol_pip_size(mt5_module, symbol, tick=tick)
    spread_pips = spread / pip_size if pip_size > 0.0 else 0.0
    if spread <= 0.0:
        return QuickSpreadState(False, "invalid_spread", spread, spread_pips, float(max_spread_pips))
    if spread_pips > float(max_spread_pips):
        return QuickSpreadState(False, "spread_too_wide", spread, spread_pips, float(max_spread_pips))
    return QuickSpreadState(True, "ok", spread, spread_pips, float(max_spread_pips))


def _symbol_pip_size(mt5_module, symbol: str, *, tick=None) -> float:
    symbol_info = mt5_module.symbol_info(symbol) if hasattr(mt5_module, "symbol_info") else None
    point = float(getattr(symbol_info, "point", 0.0) or 0.0)
    if point <= 0:
        price = _tick_value(tick, "ask") if tick is not None else 1.0
        point = max(abs(price) * 0.00001, 0.00001)
    return point * 10.0


def _broker_minimum_stop_distance(mt5_module, symbol: str) -> float:
    symbol_info = mt5_module.symbol_info(symbol) if hasattr(mt5_module, "symbol_info") else None
    point = float(getattr(symbol_info, "point", 0.0) or 0.0)
    stops_level = float(getattr(symbol_info, "trade_stops_level", 0.0) or 0.0)
    freeze_level = float(getattr(symbol_info, "trade_freeze_level", 0.0) or 0.0)
    return max(stops_level, freeze_level) * point


def build_quick_trade_levels(
    *,
    mt5_module,
    symbol: str,
    direction: BreakoutDirection,
    stop_loss_pips: float = 40.0,
    take_profit_pips: float = 120.0,
) -> tuple[float, float]:
    tick = mt5_module.symbol_info_tick(symbol) if hasattr(mt5_module, "symbol_info_tick") else None
    if tick is None:
        raise RuntimeError(f"No tick data available for symbol: {symbol}")

    ask = _tick_value(tick, "ask")
    bid = _tick_value(tick, "bid")
    spread = max(ask - bid, 0.0)
    pip_size = _symbol_pip_size(mt5_module, symbol, tick=tick)
    broker_minimum_distance = _broker_minimum_stop_distance(mt5_module, symbol)
    stop_distance = max(float(stop_loss_pips) * pip_size, broker_minimum_distance)
    take_profit_distance = max(float(take_profit_pips) * pip_size, broker_minimum_distance) + spread

    if direction is BreakoutDirection.BULLISH:
        entry = ask
        return entry - stop_distance, entry + take_profit_distance

    entry = bid
    return entry + stop_distance, entry - take_profit_distance


def has_margin_for_quick_order(
    *,
    mt5_module,
    symbol: str,
    direction: BreakoutDirection,
    lot: float,
    min_free_margin: float,
) -> bool:
    if not all(hasattr(mt5_module, name) for name in ("account_info", "symbol_info_tick", "order_calc_margin")):
        return True

    account_info = mt5_module.account_info()
    if account_info is None:
        return True

    margin_free = float(
        getattr(account_info, "margin_free", getattr(account_info, "free_margin", 0.0)) or 0.0
    )
    if margin_free <= float(min_free_margin):
        return False

    tick = mt5_module.symbol_info_tick(symbol)
    if tick is None:
        return False

    if direction is BreakoutDirection.BULLISH:
        order_type = getattr(mt5_module, "ORDER_TYPE_BUY", 0)
        price = float(getattr(tick, "ask"))
    else:
        order_type = getattr(mt5_module, "ORDER_TYPE_SELL", 1)
        price = float(getattr(tick, "bid"))

    required_margin = mt5_module.order_calc_margin(order_type, symbol, float(lot), price)
    if required_margin is None:
        return True
    return margin_free - float(required_margin) >= float(min_free_margin)


def close_profitable_quick_positions(
    *,
    executor,
    symbol: str,
    profit_target: float,
    max_loss: float | None = None,
    tick_direction: BreakoutDirection | None = None,
    log_fn=print,
) -> int:
    closed = 0
    positions = executor.list_bot_positions(symbol, comment_prefix=QUICK_COMMENT_PREFIX)
    mt5_module = getattr(executor, "mt5_module", None)
    best_position = None
    best_profit = None
    worst_position = None
    worst_profit = None
    positive_count = 0
    negative_count = 0
    flat_count = 0
    net_profit = 0.0
    for position in positions:
        profit = float(getattr(position, "profit", 0.0) or 0.0)
        net_profit += profit
        if profit > 0:
            positive_count += 1
        elif profit < 0:
            negative_count += 1
        else:
            flat_count += 1
        if best_profit is None or profit > best_profit:
            best_position = position
            best_profit = profit
        if worst_profit is None or profit < worst_profit:
            worst_position = position
            worst_profit = profit
        position_direction = _position_direction(position, mt5_module) if mt5_module is not None else None
        exit_reason = None
        if (
            profit > 0.0
            and tick_direction is not None
            and position_direction is not None
            and tick_direction is not position_direction
        ):
            exit_reason = "tick_turn"
        elif tick_direction is None and profit > float(profit_target):
            exit_reason = "profit_target"
        elif max_loss is not None and float(max_loss) > 0.0 and profit <= -abs(float(max_loss)):
            exit_reason = "max_loss"

        if exit_reason is not None:
            if exit_reason == "tick_turn":
                exit_comment = f"{QUICK_COMMENT_PREFIX}-tick-turn-profit-exit"
            elif exit_reason == "max_loss":
                exit_comment = f"{QUICK_COMMENT_PREFIX}-loss-exit"
            else:
                exit_comment = f"{QUICK_COMMENT_PREFIX}-profit-exit"
            try:
                executor.close_position(position, comment=exit_comment)
            except Exception as exc:
                log_fn(
                    f"QUICK PROFIT EXIT REJECTED {symbol} "
                    f"ticket={getattr(position, 'ticket', 'unknown')} "
                    f"profit={profit:.2f} reason={exc} exit_reason={exit_reason}"
                )
                continue
            closed += 1
            log_fn(
                f"QUICK PROFIT EXIT {symbol} ticket={getattr(position, 'ticket', 'unknown')} "
                f"profit={profit:.2f} reason={exit_reason}"
            )
    if closed == 0 and best_position is not None:
        log_fn(
            f"QUICK PROFIT WAIT {symbol} positions={len(positions)} "
            f"positive={positive_count} negative={negative_count} flat={flat_count} "
            f"net_profit={net_profit:.2f} "
            f"best_ticket={getattr(best_position, 'ticket', 'unknown')} "
            f"best_profit={float(best_profit):.2f} "
            f"worst_ticket={getattr(worst_position, 'ticket', 'unknown')} "
            f"worst_profit={float(worst_profit):.2f} "
            f"target={float(profit_target):.2f} "
            f"tick_direction={getattr(tick_direction, 'value', 'NONE')}"
        )
    return closed


def run_quick_scalp_loop(
    *,
    mt5_module,
    executor,
    symbol: str,
    lot: float,
    max_positions: int,
    profit_target: float,
    poll_seconds: int,
    max_loss: float = 0.0,
    min_free_margin: float = 0.0,
    max_loops: int | None = None,
    reload_check_fn=None,
    sleep_fn=sleep,
    log_fn=print,
):
    loop_count = 0
    insufficient_margin_logged = False
    last_hold_log = None
    def log_hold_once(message: str) -> None:
        nonlocal last_hold_log
        if message != last_hold_log:
            log_fn(message)
            last_hold_log = message

    while max_loops is None or loop_count < max_loops:
        candles = fetch_m1_candles(mt5_module, symbol, count=30)
        m1_direction = resolve_m1_direction(candles)
        current_tick = mt5_module.symbol_info_tick(symbol) if hasattr(mt5_module, "symbol_info_tick") else None
        point = _symbol_pip_size(mt5_module, symbol, tick=current_tick) / 10.0
        recent_ticks = fetch_recent_ticks(mt5_module, symbol, count=60)
        tick_guidance = resolve_tick_direction(recent_ticks, point=point)
        direction = tick_guidance.direction

        close_profitable_quick_positions(
            executor=executor,
            symbol=symbol,
            profit_target=profit_target,
            max_loss=max_loss,
            tick_direction=direction,
            log_fn=lambda message: log_hold_once(message) if "QUICK PROFIT WAIT" in message else log_fn(message),
        )

        positions = executor.list_bot_positions(symbol, comment_prefix=QUICK_COMMENT_PREFIX)
        if direction is None:
            log_hold_once(
                f"QUICK HOLD {symbol} reason={tick_guidance.reason} "
                f"m1_signal={getattr(m1_direction, 'value', 'NONE')} "
                f"ticks={tick_guidance.tick_count} net_move={tick_guidance.net_move:.2f}"
            )
        elif m1_direction is None:
            log_hold_once(
                f"QUICK HOLD {symbol} reason=m1_signal_unavailable "
                f"tick_direction={direction.value} "
                f"tick_reason={tick_guidance.reason} "
                f"ticks={tick_guidance.tick_count} net_move={tick_guidance.net_move:.2f}"
            )
        else:
            trade_direction = m1_direction if m1_direction is not direction else direction
            indicator_guidance = resolve_quick_indicator_guidance(candles)
            try:
                m15_candles = fetch_m15_candles(mt5_module, symbol, count=50)
                fibonacci_guidance = resolve_quick_fibonacci_guidance(m15_candles)
                fvg_guidance = resolve_quick_fvg_guidance(
                    m15_candles,
                    direction=trade_direction,
                    current_price=fibonacci_guidance.current_price,
                )
            except Exception as exc:
                log_hold_once(f"QUICK HOLD {symbol} reason=structure_unavailable detail={exc}")
                fibonacci_guidance = None
                fvg_guidance = None
            if fibonacci_guidance is None:
                loop_count += 1
                if reload_check_fn is not None and reload_check_fn():
                    log_fn(f"CODE CHANGE DETECTED {symbol} reloading bot")
                    return "reload_requested"
                if max_loops is not None and loop_count >= max_loops:
                    break
                sleep_fn(poll_seconds)
                continue
            if fvg_guidance is None or not fvg_guidance.allows(trade_direction):
                log_hold_once(
                    f"QUICK HOLD {symbol} reason=no_matching_fvg signal={direction.value} "
                    f"m1_signal={getattr(m1_direction, 'value', 'NONE')} "
                    f"direction={trade_direction.value} "
                    f"fib_direction={getattr(fibonacci_guidance.direction, 'value', 'NONE')} "
                    f"fib_zone={fibonacci_guidance.zone} "
                    f"swing_low={fibonacci_guidance.swing_low:.2f} "
                    f"swing_high={fibonacci_guidance.swing_high:.2f} "
                    f"current_price={fibonacci_guidance.current_price:.2f}"
                )
                loop_count += 1
                if reload_check_fn is not None and reload_check_fn():
                    log_fn(f"CODE CHANGE DETECTED {symbol} reloading bot")
                    return "reload_requested"
                if max_loops is not None and loop_count >= max_loops:
                    break
                sleep_fn(poll_seconds)
                continue
            guidance_decision = resolve_quick_guidance_decision(
                direction=trade_direction,
                fibonacci_guidance=fibonacci_guidance,
                indicator_guidance=indicator_guidance,
            )
            if not guidance_decision.allowed:
                log_hold_once(
                    f"QUICK HOLD {symbol} reason={guidance_decision.reason} signal={direction.value} "
                    f"m1_signal={getattr(m1_direction, 'value', 'NONE')} "
                    f"tick_reason={tick_guidance.reason} "
                    f"fib_direction={getattr(fibonacci_guidance.direction, 'value', 'NONE')} "
                    f"fib_zone={fibonacci_guidance.zone} "
                    f"rsi={indicator_guidance.rsi:.2f} "
                    f"sar_direction={getattr(indicator_guidance.sar_direction, 'value', 'NONE')} "
                    f"sar={indicator_guidance.sar:.2f} "
                    f"fvg_direction={getattr(fvg_guidance.direction, 'value', 'NONE')} "
                    f"fvg_top={fvg_guidance.top:.2f} fvg_bottom={fvg_guidance.bottom:.2f} "
                    f"fvg_bars_since={fvg_guidance.bars_since}"
                )
            else:
                spread_state = resolve_spread_state(mt5_module=mt5_module, symbol=symbol)
                if not spread_state.allowed:
                    log_hold_once(
                        f"QUICK HOLD {symbol} reason={spread_state.reason} "
                        f"spread_pips={spread_state.spread_pips:.2f} "
                        f"max_spread_pips={spread_state.max_spread_pips:.2f}"
                    )
                    tick = None
                    grid_permission = QuickGridPermission(False, spread_state.reason, 0, 0.0, 0.0)
                else:
                    tick = current_tick
                if tick is None:
                    if spread_state.allowed:
                        log_hold_once(f"QUICK HOLD {symbol} reason=no_tick_for_grid")
                        grid_permission = QuickGridPermission(False, "no_tick_for_grid", 0, 0.0, 0.0)
                elif spread_state.allowed:
                    grid_permission = resolve_quick_grid_permission(
                        candles=candles,
                        positions=positions,
                        direction=trade_direction,
                        fibonacci_guidance=fibonacci_guidance,
                        mt5_module=mt5_module,
                        tick=tick,
                    )
                    if not grid_permission.allowed:
                        log_hold_once(
                            f"QUICK HOLD {symbol} reason={grid_permission.reason} "
                            f"direction={trade_direction.value} price={grid_permission.current_price:.2f} "
                            f"spacing={grid_permission.min_spacing:.2f}"
                        )

                new_entries = 0
                max_new_entries = grid_permission.max_new_entries
                if guidance_decision.max_new_entries_override is not None:
                    max_new_entries = min(max_new_entries, guidance_decision.max_new_entries_override)
                while (
                    grid_permission.allowed
                    and len(positions) < int(max_positions)
                    and new_entries < int(max_new_entries)
                ):
                    if not has_margin_for_quick_order(
                        mt5_module=mt5_module,
                        symbol=symbol,
                        direction=trade_direction,
                        lot=lot,
                        min_free_margin=min_free_margin,
                    ):
                        if not insufficient_margin_logged:
                            log_fn(f"QUICK HOLD {symbol} reason=insufficient_free_margin positions={len(positions)}")
                            insufficient_margin_logged = True
                        break

                    try:
                        stop_loss, take_profit = build_quick_trade_levels(
                            mt5_module=mt5_module,
                            symbol=symbol,
                            direction=trade_direction,
                        )
                        position = executor.open_strategy_trade(
                            symbol=symbol,
                            direction=trade_direction,
                            lot=lot,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            comment=QUICK_COMMENT_PREFIX,
                        )
                    except Exception as exc:
                        log_fn(f"QUICK ORDER REJECTED {symbol} reason={exc} positions={len(positions)}")
                        break

                    positions.append(position)
                    new_entries += 1
                    insufficient_margin_logged = False
                    last_hold_log = None
                    log_fn(
                        f"QUICK TRADE OPENED {symbol} ticket={getattr(position, 'ticket', 'unknown')} "
                        f"signal={direction.value} m1_signal={getattr(m1_direction, 'value', 'NONE')} "
                        f"tick_reason={tick_guidance.reason} direction={trade_direction.value} "
                        f"guidance={guidance_decision.reason} fib_zone={fibonacci_guidance.zone} "
                        f"swing_low={fibonacci_guidance.swing_low:.2f} "
                        f"swing_high={fibonacci_guidance.swing_high:.2f} "
                        f"fvg_top={fvg_guidance.top:.2f} fvg_bottom={fvg_guidance.bottom:.2f} "
                        f"fvg_direction={getattr(fvg_guidance.direction, 'value', 'NONE')} "
                        f"fvg_bars_since={fvg_guidance.bars_since} "
                        f"rsi={indicator_guidance.rsi:.2f} "
                        f"sar={indicator_guidance.sar:.2f} spacing={grid_permission.min_spacing:.2f} "
                        f"lot={lot} positions={len(positions)}"
                    )
                    grid_permission = resolve_quick_grid_permission(
                        candles=candles,
                        positions=positions,
                        direction=trade_direction,
                        fibonacci_guidance=fibonacci_guidance,
                        mt5_module=mt5_module,
                        tick=tick,
                    )

        loop_count += 1
        if reload_check_fn is not None and reload_check_fn():
            log_fn(f"CODE CHANGE DETECTED {symbol} reloading bot")
            return "reload_requested"
        if max_loops is not None and loop_count >= max_loops:
            break
        sleep_fn(poll_seconds)
