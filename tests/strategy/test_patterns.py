from datetime import datetime, timedelta, timezone

from src.market_data import Candle
from src.strategy.patterns import detect_candlestick_patterns, detect_three_drives


def _candles(rows: list[tuple[float, float, float, float, int]]) -> list[Candle]:
    start = datetime(2026, 4, 6, 0, 0, tzinfo=timezone.utc)
    return [
        Candle(
            timestamp=start + timedelta(minutes=15 * i),
            open=open_p,
            high=high_p,
            low=low_p,
            close=close_p,
            volume=vol,
            timeframe="M15",
        )
        for i, (open_p, high_p, low_p, close_p, vol) in enumerate(rows)
    ]


def test_detect_bullish_engulfing():
    # Candle 1: Bearish (100 -> 95)
    # Candle 2: Bullish Engulfing (94 -> 101)
    candles = _candles([
        (100.0, 101.0, 94.5, 95.0, 100),
        (94.0, 101.5, 93.5, 101.0, 200),
    ])
    res = detect_candlestick_patterns(candles)
    assert res.is_present is True
    assert res.reason == "bullish_engulfing"
    assert res.metadata["direction"] == "BULLISH"
    assert res.confluence_score == 2


def test_detect_bearish_engulfing():
    # Candle 1: Bullish (95 -> 100)
    # Candle 2: Bearish Engulfing (101 -> 94)
    candles = _candles([
        (95.0, 100.5, 94.5, 100.0, 100),
        (101.0, 101.5, 93.5, 94.0, 200),
    ])
    res = detect_candlestick_patterns(candles)
    assert res.is_present is True
    assert res.reason == "bearish_engulfing"
    assert res.metadata["direction"] == "BEARISH"
    assert res.confluence_score == 2


def test_detect_bullish_pinbar():
    # Hammer/Pinbar: Long lower wick (6.0), small body (1.0), total range (8.0) -> lower wick = 75%
    candles = _candles([
        (100.0, 101.0, 99.0, 100.0, 100),
        (99.0, 100.0, 92.0, 99.5, 200),
    ])
    res = detect_candlestick_patterns(candles)
    assert res.is_present is True
    assert res.reason == "bullish_pinbar"
    assert res.metadata["direction"] == "BULLISH"


def test_detect_bearish_pinbar():
    # Shooting star/Pinbar: Long upper wick (6.0), small body (1.0), total range (8.0) -> upper wick = 75%
    candles = _candles([
        (100.0, 101.0, 99.0, 100.0, 100),
        (92.5, 100.0, 92.0, 93.0, 200),
    ])
    res = detect_candlestick_patterns(candles)
    assert res.is_present is True
    assert res.reason == "bearish_pinbar"
    assert res.metadata["direction"] == "BEARISH"


def test_detect_morning_star():
    # Morning star: Bearish -> Small doji -> Bullish closing above c1 midpoint
    candles = _candles([
        (100.0, 101.0, 90.0, 91.0, 100),   # Bearish
        (91.0, 91.5, 89.5, 90.5, 50),      # Small body
        (90.5, 98.0, 90.0, 97.0, 200),     # Bullish expansion > midpoint (95.5)
    ])
    res = detect_candlestick_patterns(candles)
    assert res.is_present is True
    assert res.reason == "morning_star"
    assert res.metadata["direction"] == "BULLISH"
