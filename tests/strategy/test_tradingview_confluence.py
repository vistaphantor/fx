from datetime import datetime, timezone

from src.strategy.breakout import BreakoutDirection
from src.strategy.gap import GapDecision
from src.strategy.tradingview_confluence import build_tradingview_confluence
from src.tradingview import TradingViewAlert


def test_build_tradingview_confluence_boosts_gap_fill_alignment():
    alert = TradingViewAlert(
        is_valid=True,
        reason="accepted",
        symbol="XAUUSD",
        direction=BreakoutDirection.BEARISH,
        setup="gap_fill",
        level=4682.4,
        timestamp=datetime(2026, 4, 28, 6, 15, tzinfo=timezone.utc),
        timeframe="15",
        confidence=0.82,
        context={"gap_class": "moderate"},
    )
    gap_decision = GapDecision(
        has_gap=True,
        gap_size=2.1,
        size_class="moderate",
        gap_direction=BreakoutDirection.BULLISH,
        preferred_trade_direction=BreakoutDirection.BEARISH,
        fill_preferred=True,
        reason="gap_fill_preferred",
        metadata={},
    )

    result = build_tradingview_confluence(
        symbol="XAUUSD",
        direction=BreakoutDirection.BEARISH,
        gap_decision=gap_decision,
        alert=alert,
    )

    assert result.is_active is True
    assert result.direction_bonus >= 2
    assert result.setup_bonus >= 1
    assert result.reason == "tradingview_gap_fill_alignment"


def test_build_tradingview_confluence_penalizes_conflicting_bias():
    alert = TradingViewAlert(
        is_valid=True,
        reason="accepted",
        symbol="XAUUSD",
        direction=BreakoutDirection.BULLISH,
        setup="three_drives",
        level=4682.4,
        timestamp=datetime(2026, 4, 28, 6, 15, tzinfo=timezone.utc),
        timeframe="15",
        confidence=0.9,
        context={},
    )
    gap_decision = GapDecision(
        has_gap=False,
        gap_size=0.0,
        size_class="none",
        gap_direction=None,
        preferred_trade_direction=None,
        fill_preferred=False,
        reason="no_session_gap_detected",
        metadata={},
    )

    result = build_tradingview_confluence(
        symbol="XAUUSD",
        direction=BreakoutDirection.BEARISH,
        gap_decision=gap_decision,
        alert=alert,
    )

    assert result.is_active is True
    assert result.direction_penalty >= 1
    assert result.direction_bonus == 0
    assert result.reason == "tradingview_direction_conflict"
