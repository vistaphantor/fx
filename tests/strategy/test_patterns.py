from datetime import datetime, timedelta, timezone

from src.market_data import Candle
from src.strategy.patterns import detect_three_drives


def _candles(rows: list[tuple[float, float, float, float]]) -> list[Candle]:
    start = datetime(2026, 4, 6, 8, 0, tzinfo=timezone.utc)
    candles = []
    for index, (open_price, high, low, close) in enumerate(rows):
        candles.append(
            Candle(
                timestamp=start + timedelta(minutes=15 * index),
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=100 + index,
                timeframe="M15",
            )
        )
    return candles


def test_detect_three_drives_returns_confluence_near_key_level():
    candles = _candles(
        [
            (103.0, 103.5, 100.4, 101.2),
            (101.2, 104.2, 100.8, 103.8),
            (103.8, 104.1, 100.2, 101.0),
            (101.0, 104.5, 100.9, 104.0),
            (104.0, 104.2, 100.1, 101.1),
            (101.1, 105.0, 100.8, 104.4),
        ]
    )

    result = detect_three_drives(candles=candles, reference_levels=(100.0, 109.0))

    assert result.is_present is True
    assert result.reason == "three_drives_detected"
    assert result.confluence_score > 0


def test_detect_three_drives_returns_absent_when_swings_are_not_clean():
    candles = _candles(
        [
            (103.0, 104.0, 101.8, 103.5),
            (103.5, 104.2, 102.4, 103.9),
            (103.9, 104.3, 102.7, 103.8),
            (103.8, 104.5, 102.8, 104.1),
            (104.1, 104.6, 103.0, 104.2),
            (104.2, 104.7, 103.2, 104.4),
        ]
    )

    result = detect_three_drives(candles=candles, reference_levels=(100.0, 109.0))

    assert result.is_present is False
    assert result.reason == "three_drives_absent"
    assert result.confluence_score == 0
