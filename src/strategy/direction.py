from __future__ import annotations

from dataclasses import dataclass

from src.market_data import Candle
from src.strategy.breakout import BreakoutDirection
from src.strategy.context import DailyContext, H4Context
from src.strategy.gap import GapDecision
from src.strategy.orderflow import OrderflowSignal, score_orderflow_for_direction
from src.strategy.patterns import detect_candlestick_patterns
from src.strategy.tradingview_confluence import TradingViewConfluence


@dataclass(frozen=True, slots=True)
class DirectionDecision:
    is_valid: bool
    direction: BreakoutDirection | None
    reason: str
    metadata: dict[str, object]
    bullish_contribution: float = 0.0
    bearish_contribution: float = 0.0


def determine_h1_bias(
    *,
    h1_candles: list[Candle],
    daily_context: DailyContext,
    h4_context: H4Context,
    gap_decision: GapDecision | None = None,
    tradingview_confluence: TradingViewConfluence | None = None,
    orderflow_signal: OrderflowSignal | None = None,
    h4_candles: list[Candle] | None = None,
    m15_candles: list[Candle] | None = None,
) -> DirectionDecision:
    _require_timeframe(h1_candles, "H1", minimum=4)

    latest = h1_candles[-1]
    previous = h1_candles[-2]
    current_price = float(daily_context.current_price)
    latest_h1_close = float(h1_candles[-1].close)
    momentum = latest_h1_close - float(h1_candles[-4].close)
    demand_distance = _distance_to_nearest_demand(current_price, h4_context.demand_zones)
    supply_distance = _distance_to_nearest_supply(current_price, h4_context.supply_zones)
    day_range = max(float(daily_context.daily_high) - float(daily_context.daily_low), 0.0)
    momentum_threshold = max(day_range * 0.01, 0.5)
    volume_support_distance = _distance_to_volume_support(current_price, h4_context.volume_profile_levels)
    volume_resistance_distance = _distance_to_volume_resistance(current_price, h4_context.volume_profile_levels)
    volume_threshold = max(day_range * 0.03, 1.0)
    zone_break_threshold = max(day_range * 0.01, 0.5)

    h4_trend_direction = 0
    if h4_candles is not None and len(h4_candles) >= 2:
        h4_start_close = float(h4_candles[0].close)
        h4_end_close = float(h4_candles[-1].close)
        h4_start_low = float(h4_candles[0].low)
        h4_end_low = float(h4_candles[-1].low)
        h4_start_high = float(h4_candles[0].high)
        h4_end_high = float(h4_candles[-1].high)
        if h4_end_close > h4_start_close and h4_end_low >= h4_start_low:
            h4_trend_direction = 1
        elif h4_end_close < h4_start_close and h4_end_high <= h4_start_high:
            h4_trend_direction = -1

    bullish_override_pressure = (h4_trend_direction == -1) and (momentum > momentum_threshold * 2.0)
    bearish_override_pressure = (h4_trend_direction == 1) and (momentum < -(momentum_threshold * 2.0))

    common_metadata = {
        "current_price": current_price,
        "momentum": momentum,
        "demand_distance": demand_distance,
        "supply_distance": supply_distance,
        "daily_range_position": daily_context.range_position,
        "momentum_threshold": momentum_threshold,
        "volume_support_distance": volume_support_distance,
        "volume_resistance_distance": volume_resistance_distance,
        "latest_h1_close": latest_h1_close,
        "bullish_override_pressure": bullish_override_pressure,
        "bearish_override_pressure": bearish_override_pressure,
    }

    in_demand_zone = _is_inside_zone(current_price, h4_context.demand_zones)
    in_supply_zone = _is_inside_zone(current_price, h4_context.supply_zones)
    below_demand_breakdown = _is_below_all_zones(current_price, h4_context.demand_zones, zone_break_threshold)
    above_supply_breakout = _is_above_all_zones(current_price, h4_context.supply_zones, zone_break_threshold)

    common_metadata.update(
        {
            "in_demand_zone": in_demand_zone,
            "in_supply_zone": in_supply_zone,
            "below_demand_breakdown": below_demand_breakdown,
            "above_supply_breakout": above_supply_breakout,
            "zone_break_threshold": zone_break_threshold,
        }
    )

    bullish_score = 0
    bearish_score = 0

    if daily_context.range_position <= 0.35:
        bullish_score += 2
    elif daily_context.range_position >= 0.65:
        bearish_score += 1

    if in_demand_zone:
        bullish_score += 2
    elif _is_favorably_positioned(demand_distance, supply_distance):
        bullish_score += 1

    if in_supply_zone:
        bearish_score += 2
    elif _is_favorably_positioned(supply_distance, demand_distance):
        bearish_score += 1

    if below_demand_breakdown:
        bearish_score += 2
    if above_supply_breakout:
        bullish_score += 2

    if momentum > momentum_threshold:
        bullish_score += 1
    elif momentum < -momentum_threshold:
        bearish_score += 1

    if float(latest.close) > float(latest.open):
        bullish_score += 1
    elif float(latest.close) < float(latest.open):
        bearish_score += 1

    if float(latest.close) > float(previous.close):
        bullish_score += 1
    elif float(latest.close) < float(previous.close):
        bearish_score += 1

    if volume_support_distance is not None and volume_support_distance <= volume_threshold:
        bullish_score += 1
    if volume_resistance_distance is not None and volume_resistance_distance <= volume_threshold:
        bearish_score += 1

    if gap_decision is not None and gap_decision.has_gap and gap_decision.preferred_trade_direction is not None:
        gap_bonus = 2 if gap_decision.fill_preferred else 1
        if gap_decision.preferred_trade_direction is BreakoutDirection.BULLISH:
            bullish_score += gap_bonus
        else:
            bearish_score += gap_bonus

    if tradingview_confluence is not None and tradingview_confluence.is_active:
        if tradingview_confluence.preferred_direction is BreakoutDirection.BULLISH:
            bullish_score += tradingview_confluence.direction_bonus
            bearish_score = max(0, bearish_score - tradingview_confluence.direction_penalty)
        elif tradingview_confluence.preferred_direction is BreakoutDirection.BEARISH:
            bearish_score += tradingview_confluence.direction_bonus
            bullish_score = max(0, bullish_score - tradingview_confluence.direction_penalty)

    # Evaluate candlestick patterns (Pinbars, Engulfing, Momentum Expansion, Morning/Evening Star)
    pattern_candles = m15_candles if m15_candles and len(m15_candles) >= 2 else h1_candles
    candlestick_pattern = detect_candlestick_patterns(pattern_candles)
    if candlestick_pattern.is_present:
        pat_dir = candlestick_pattern.metadata.get("direction")
        if pat_dir == "BULLISH":
            bullish_score += candlestick_pattern.confluence_score
        elif pat_dir == "BEARISH":
            bearish_score += candlestick_pattern.confluence_score

    # Evaluate live orderflow signal support
    if orderflow_signal is not None:
        bull_of = score_orderflow_for_direction(orderflow_signal, BreakoutDirection.BULLISH)
        bear_of = score_orderflow_for_direction(orderflow_signal, BreakoutDirection.BEARISH)
        if bull_of.is_supportive:
            bullish_score += 1
        elif bear_of.is_supportive:
            bearish_score += 1

    common_metadata.update(
        {
            "candlestick_pattern": candlestick_pattern.reason if candlestick_pattern.is_present else None,
            "candlestick_pattern_score": candlestick_pattern.confluence_score if candlestick_pattern.is_present else 0,
            "bullish_score": bullish_score,
            "bearish_score": bearish_score,
            "volume_threshold": volume_threshold,
            "gap_reason": getattr(gap_decision, "reason", None),
            "gap_size_class": getattr(gap_decision, "size_class", None),
            "gap_fill_preferred": getattr(gap_decision, "fill_preferred", None),
            "tradingview_reason": getattr(tradingview_confluence, "reason", None),
            "tradingview_bonus": getattr(tradingview_confluence, "direction_bonus", 0),
            "tradingview_penalty": getattr(tradingview_confluence, "direction_penalty", 0),
        }
    )

    bullish_reversal_context = (
        in_demand_zone
        and not in_supply_zone
        and daily_context.range_position <= 0.10
        and (supply_distance is None or supply_distance >= max(day_range * 0.20, 5.0))
        and bearish_score <= bullish_score + 2
        and momentum >= -(momentum_threshold * 4.0)
    )
    bearish_reversal_context = (
        in_supply_zone
        and not in_demand_zone
        and daily_context.range_position >= 0.90
        and (demand_distance is None or demand_distance >= max(day_range * 0.20, 5.0))
        and bullish_score <= bearish_score + 2
        and momentum <= momentum_threshold * 4.0
    )

    common_metadata.update(
        {
            "bullish_reversal_context": bullish_reversal_context,
            "bearish_reversal_context": bearish_reversal_context,
        }
    )

    if (
        bullish_score >= 2
        and bullish_score > bearish_score
        and not in_supply_zone
        and _supports_direction(
            direction=BreakoutDirection.BULLISH,
            latest=latest,
            previous=previous,
            momentum=momentum,
            momentum_threshold=momentum_threshold,
            in_context_zone=in_demand_zone,
            candlestick_pattern=candlestick_pattern,
            orderflow_signal=orderflow_signal,
        )
    ):
        conflict = _orderflow_conflict_decision(
            direction=BreakoutDirection.BULLISH,
            signal=orderflow_signal,
            metadata=common_metadata,
            bullish_score=bullish_score,
            bearish_score=bearish_score,
        )
        if conflict is not None:
            return conflict
        return DirectionDecision(
            is_valid=True,
            direction=BreakoutDirection.BULLISH,
            reason="h1_bias_bullish",
            metadata=common_metadata,
            bullish_contribution=float(bullish_score),
            bearish_contribution=float(bearish_score),
        )

    if bullish_reversal_context and current_price < daily_context.objective_high:
        conflict = _orderflow_conflict_decision(
            direction=BreakoutDirection.BULLISH,
            signal=orderflow_signal,
            metadata=common_metadata,
            bullish_score=bullish_score,
            bearish_score=bearish_score,
        )
        if conflict is not None:
            return conflict
        return DirectionDecision(
            is_valid=True,
            direction=BreakoutDirection.BULLISH,
            reason="h1_reversal_context_bullish",
            metadata=common_metadata,
            bullish_contribution=float(bullish_score),
            bearish_contribution=float(bearish_score),
        )

    if (
        bearish_score >= 2
        and bearish_score > bullish_score
        and not in_demand_zone
        and _supports_direction(
            direction=BreakoutDirection.BEARISH,
            latest=latest,
            previous=previous,
            momentum=momentum,
            momentum_threshold=momentum_threshold,
            in_context_zone=in_supply_zone,
            candlestick_pattern=candlestick_pattern,
            orderflow_signal=orderflow_signal,
        )
    ):
        conflict = _orderflow_conflict_decision(
            direction=BreakoutDirection.BEARISH,
            signal=orderflow_signal,
            metadata=common_metadata,
            bullish_score=bullish_score,
            bearish_score=bearish_score,
        )
        if conflict is not None:
            return conflict
        return DirectionDecision(
            is_valid=True,
            direction=BreakoutDirection.BEARISH,
            reason="h1_bias_bearish",
            metadata=common_metadata,
            bullish_contribution=float(bullish_score),
            bearish_contribution=float(bearish_score),
        )

    if bearish_reversal_context:
        conflict = _orderflow_conflict_decision(
            direction=BreakoutDirection.BEARISH,
            signal=orderflow_signal,
            metadata=common_metadata,
            bullish_score=bullish_score,
            bearish_score=bearish_score,
        )
        if conflict is not None:
            return conflict
        return DirectionDecision(
            is_valid=True,
            direction=BreakoutDirection.BEARISH,
            reason="h1_reversal_context_bearish",
            metadata=common_metadata,
            bullish_contribution=float(bullish_score),
            bearish_contribution=float(bearish_score),
        )

    return DirectionDecision(
        is_valid=False,
        direction=None,
        reason="h1_context_conflict",
        metadata=common_metadata,
        bullish_contribution=float(bullish_score),
        bearish_contribution=float(bearish_score),
    )


def _orderflow_conflict_decision(
    *,
    direction: BreakoutDirection,
    signal: OrderflowSignal | None,
    metadata: dict[str, object],
    bullish_score: int,
    bearish_score: int,
) -> DirectionDecision | None:
    score = score_orderflow_for_direction(signal, direction)
    metadata.update(score.metadata)
    if not score.is_conflicting:
        return None
    return DirectionDecision(
        is_valid=False,
        direction=None,
        reason="h1_orderflow_conflict",
        metadata=metadata,
        bullish_contribution=float(bullish_score),
        bearish_contribution=float(bearish_score),
    )


def _distance_to_nearest_demand(current_price: float, demand_zones: tuple[tuple[float, float], ...]) -> float | None:
    distances = [current_price - zone[1] for zone in demand_zones if current_price >= zone[1]]
    return min(distances) if distances else None


def _distance_to_nearest_supply(current_price: float, supply_zones: tuple[tuple[float, float], ...]) -> float | None:
    distances = [zone[0] - current_price for zone in supply_zones if current_price <= zone[0]]
    return min(distances) if distances else None


def _is_inside_zone(current_price: float, zones: tuple[tuple[float, float], ...]) -> bool:
    return any(lower <= current_price <= upper for lower, upper in zones)


def _distance_to_volume_support(current_price: float, volume_profile_levels: tuple[float, ...]) -> float | None:
    distances = [current_price - level for level in volume_profile_levels if level <= current_price]
    return min(distances) if distances else None


def _distance_to_volume_resistance(current_price: float, volume_profile_levels: tuple[float, ...]) -> float | None:
    distances = [level - current_price for level in volume_profile_levels if level >= current_price]
    return min(distances) if distances else None


def _is_favorably_positioned(primary_distance: float | None, opposing_distance: float | None) -> bool:
    if primary_distance is None:
        return False
    if opposing_distance is None:
        return True
    return primary_distance <= (opposing_distance + 1.5)


def _is_below_all_zones(current_price: float, zones: tuple[tuple[float, float], ...], threshold: float) -> bool:
    if not zones:
        return False
    lowest_lower = min(lower for lower, _ in zones)
    return current_price < (lowest_lower - threshold)


def _is_above_all_zones(current_price: float, zones: tuple[tuple[float, float], ...], threshold: float) -> bool:
    if not zones:
        return False
    highest_upper = max(upper for _, upper in zones)
    return current_price > (highest_upper + threshold)


def _supports_direction(
    *,
    direction: BreakoutDirection,
    latest: Candle,
    previous: Candle,
    momentum: float,
    momentum_threshold: float,
    in_context_zone: bool,
    candlestick_pattern: PatternDecision | None = None,
    orderflow_signal: OrderflowSignal | None = None,
) -> bool:
    latest_open = float(latest.open)
    latest_close = float(latest.close)
    previous_high = float(previous.high)
    previous_low = float(previous.low)

    pattern_supports = (
        candlestick_pattern is not None
        and candlestick_pattern.is_present
        and candlestick_pattern.metadata.get("direction") == ("BULLISH" if direction is BreakoutDirection.BULLISH else "BEARISH")
    )
    of_supports = False
    if orderflow_signal is not None:
        of_score = score_orderflow_for_direction(orderflow_signal, direction)
        of_supports = of_score.is_supportive or of_score.alignment_score > 0.10

    if direction is BreakoutDirection.BULLISH:
        return (
            momentum > momentum_threshold
            or latest_close > latest_open
            or latest_close > previous_high
            or pattern_supports
            or of_supports
            or (in_context_zone and latest_close >= float(previous.close))
        )

    return (
        momentum < -momentum_threshold
        or latest_close < latest_open
        or latest_close < previous_low
        or pattern_supports
        or of_supports
        or (in_context_zone and latest_close <= float(previous.close))
    )


def _require_timeframe(candles: list[Candle], timeframe: str, *, minimum: int) -> None:
    if len(candles) < minimum:
        raise ValueError(f"{timeframe} direction input requires at least {minimum} candles")
    if any(candle.timeframe != timeframe for candle in candles):
        raise ValueError(f"{timeframe} direction input received a different timeframe")
