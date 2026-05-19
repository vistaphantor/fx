from datetime import datetime, timedelta, timezone

from src.market_data import Candle
from src.strategy.breakout import BreakoutDirection
from src.strategy.context import DailyContext, H4Context
from src.strategy.direction import determine_h1_bias
from src.strategy.gap import GapDecision
from src.strategy.orderflow import parse_orderflow_payload
from src.strategy.tradingview_confluence import TradingViewConfluence


def _candles(closes: list[float]) -> list[Candle]:
    start = datetime(2026, 4, 6, 8, 0, tzinfo=timezone.utc)
    candles = []
    for index, close in enumerate(closes):
        open_price = closes[index - 1] if index else close - 1.0
        candles.append(
            Candle(
                timestamp=start + timedelta(hours=index),
                open=open_price,
                high=max(open_price, close) + 1.0,
                low=min(open_price, close) - 1.0,
                close=close,
                volume=100 + index,
                timeframe="H1",
            )
        )
    return candles


def test_determine_h1_bias_returns_bullish_when_context_and_reaction_align():
    daily_context = DailyContext(
        daily_high=112.0,
        daily_low=100.0,
        current_price=106.0,
        range_position=0.5,
        objective_high=112.0,
        objective_low=100.0,
    )
    h4_context = H4Context(
        previous_session_high=110.0,
        previous_session_low=96.0,
        demand_zones=((101.0, 104.0),),
        supply_zones=((108.0, 131.0),),
        volume_profile_levels=(111.0, 107.0, 103.0),
    )

    result = determine_h1_bias(
        h1_candles=_candles([100.0, 101.5, 103.0, 104.0, 106.0]),
        daily_context=daily_context,
        h4_context=h4_context,
    )

    assert result.is_valid is True
    assert result.direction is BreakoutDirection.BULLISH
    assert result.reason == "h1_bias_bullish"
    assert result.bullish_contribution > result.bearish_contribution


def test_determine_h1_bias_returns_no_trade_when_context_conflicts():
    daily_context = DailyContext(
        daily_high=112.0,
        daily_low=100.0,
        current_price=108.5,
        range_position=0.7,
        objective_high=112.0,
        objective_low=100.0,
    )
    h4_context = H4Context(
        previous_session_high=110.0,
        previous_session_low=96.0,
        demand_zones=((100.0, 102.0),),
        supply_zones=((108.0, 109.5),),
        volume_profile_levels=(109.0, 107.0, 103.0),
    )

    result = determine_h1_bias(
        h1_candles=_candles([103.0, 104.0, 105.0, 107.0, 108.5]),
        daily_context=daily_context,
        h4_context=h4_context,
    )

    assert result.is_valid is False
    assert result.direction is None
    assert result.reason == "h1_context_conflict"


def test_determine_h1_bias_allows_context_led_bullish_bias_when_price_is_near_demand():
    daily_context = DailyContext(
        daily_high=4729.95,
        daily_low=4667.11,
        current_price=4681.73,
        range_position=0.2326543602800741,
        objective_high=4729.95,
        objective_low=4667.11,
    )
    h4_context = H4Context(
        previous_session_high=4729.95,
        previous_session_low=4667.11,
        demand_zones=((4667.11, 4676.89),),
        supply_zones=((4725.05, 4729.95),),
        volume_profile_levels=(4682.44, 4715.233333333334, 4712.486666666667),
    )

    result = determine_h1_bias(
        h1_candles=_candles([4676.89, 4682.1, 4680.18, 4681.01, 4681.73]),
        daily_context=daily_context,
        h4_context=h4_context,
    )

    assert result.is_valid is True
    assert result.direction is BreakoutDirection.BULLISH
    assert result.reason == "h1_bias_bullish"
    assert result.bullish_contribution > 0


def test_determine_h1_bias_uses_gap_fill_preference_as_extra_confluence():
    daily_context = DailyContext(
        daily_high=112.0,
        daily_low=100.0,
        current_price=105.5,
        range_position=0.46,
        objective_high=112.0,
        objective_low=100.0,
    )
    h4_context = H4Context(
        previous_session_high=110.0,
        previous_session_low=99.0,
        demand_zones=((100.0, 102.0),),
        supply_zones=((109.0, 110.5),),
        volume_profile_levels=(103.0, 107.0, 109.5),
    )
    gap_decision = GapDecision(
        has_gap=True,
        gap_size=2.0,
        size_class="moderate",
        gap_direction=BreakoutDirection.BEARISH,
        preferred_trade_direction=BreakoutDirection.BULLISH,
        fill_preferred=True,
        reason="gap_fill_preferred",
        metadata={},
    )

    result = determine_h1_bias(
        h1_candles=_candles([103.0, 104.0, 104.6, 105.0, 105.5]),
        daily_context=daily_context,
        h4_context=h4_context,
        gap_decision=gap_decision,
    )

    assert result.is_valid is True
    assert result.direction is BreakoutDirection.BULLISH


def test_determine_h1_bias_uses_tradingview_bonus_to_pass_borderline_alignment():
    daily_context = DailyContext(
        daily_high=112.0,
        daily_low=100.0,
        current_price=105.6,
        range_position=0.46,
        objective_high=112.0,
        objective_low=100.0,
    )
    h4_context = H4Context(
        previous_session_high=110.0,
        previous_session_low=99.0,
        demand_zones=((101.0, 103.0),),
        supply_zones=((109.5, 110.5),),
        volume_profile_levels=(103.2, 106.0, 109.7),
    )
    confluence = TradingViewConfluence(
        is_active=True,
        reason="tradingview_gap_fill_alignment",
        direction_bonus=2,
        direction_penalty=0,
        setup_bonus=1,
        trigger_bonus=0,
        preferred_direction=BreakoutDirection.BULLISH,
        metadata={},
    )

    result = determine_h1_bias(
        h1_candles=_candles([103.0, 103.8, 104.5, 105.0, 105.6]),
        daily_context=daily_context,
        h4_context=h4_context,
        tradingview_confluence=confluence,
    )

    assert result.is_valid is True
    assert result.direction is BreakoutDirection.BULLISH


def test_determine_h1_bias_blocks_when_orderflow_strongly_conflicts():
    daily_context = DailyContext(
        daily_high=112.0,
        daily_low=100.0,
        current_price=106.0,
        range_position=0.5,
        objective_high=112.0,
        objective_low=100.0,
    )
    h4_context = H4Context(
        previous_session_high=110.0,
        previous_session_low=96.0,
        demand_zones=((101.0, 104.0),),
        supply_zones=((108.0, 131.0),),
        volume_profile_levels=(111.0, 107.0, 103.0),
    )
    orderflow = parse_orderflow_payload(
        {
            "symbol": "XAUUSD",
            "delta": -1200,
            "buyvolume": 300,
            "sellvolume": 1500,
            "cvd_slope": -0.8,
            "imbalance": "sell_stacked",
            "vwap_bias": "below",
        }
    )

    result = determine_h1_bias(
        h1_candles=_candles([100.0, 101.5, 103.0, 104.0, 106.0]),
        daily_context=daily_context,
        h4_context=h4_context,
        orderflow_signal=orderflow,
    )

    assert result.is_valid is False
    assert result.reason == "h1_orderflow_conflict"
    assert result.metadata["orderflow_alignment"] < -0.35


def test_determine_h1_bias_allows_reversal_context_bullish_bias_at_extreme_demand():
    daily_context = DailyContext(
        daily_high=4729.95,
        daily_low=4667.11,
        current_price=4667.58,
        range_position=0.0075,
        objective_high=4729.95,
        objective_low=4667.11,
    )
    h4_context = H4Context(
        previous_session_high=4725.39,
        previous_session_low=4666.35,
        demand_zones=((4667.11, 4676.89),),
        supply_zones=((4690.17, 4701.25),),
        volume_profile_levels=(4682.44, 4712.48, 4702.78),
        session_range=59.04,
    )
    gap_decision = GapDecision(
        has_gap=True,
        gap_size=4.95,
        size_class="small",
        gap_direction=BreakoutDirection.BULLISH,
        preferred_trade_direction=None,
        fill_preferred=False,
        reason="gap_fill_completed",
        metadata={},
    )

    result = determine_h1_bias(
        h1_candles=_candles([4690.17, 4669.81, 4668.91, 4670.81, 4667.58]),
        daily_context=daily_context,
        h4_context=h4_context,
        gap_decision=gap_decision,
    )

    assert result.is_valid is True
    assert result.direction is BreakoutDirection.BULLISH
    assert result.reason == "h1_reversal_context_bullish"


def test_determine_h1_bias_flips_bearish_when_live_price_breaks_below_demand():
    daily_context = DailyContext(
        daily_high=4729.95,
        daily_low=4667.11,
        current_price=4652.22,
        range_position=0.0,
        objective_high=4729.95,
        objective_low=4667.11,
    )
    h4_context = H4Context(
        previous_session_high=4725.39,
        previous_session_low=4666.35,
        demand_zones=((4667.11, 4676.89),),
        supply_zones=((4690.17, 4701.25),),
        volume_profile_levels=(4682.44, 4712.48, 4702.78),
        session_range=59.04,
    )
    gap_decision = GapDecision(
        has_gap=True,
        gap_size=4.95,
        size_class="small",
        gap_direction=BreakoutDirection.BULLISH,
        preferred_trade_direction=None,
        fill_preferred=False,
        reason="gap_fill_completed",
        metadata={},
    )

    result = determine_h1_bias(
        h1_candles=_candles([4690.17, 4669.81, 4668.91, 4670.81, 4667.58]),
        daily_context=daily_context,
        h4_context=h4_context,
        gap_decision=gap_decision,
    )

    assert result.is_valid is True
    assert result.direction is BreakoutDirection.BEARISH
    assert result.reason == "h1_bias_bearish"
