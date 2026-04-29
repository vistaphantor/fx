from datetime import datetime, timedelta, timezone

from src.market_data import Candle
from src.strategy.breakout import BreakoutDirection
from src.strategy.gap import evaluate_gap_context


def _candles(timeframe: str, rows: list[tuple[float, float, float, float, int]], *, start: datetime, minutes: int) -> list[Candle]:
    candles = []
    for index, (open_price, high, low, close, volume) in enumerate(rows):
        candles.append(
            Candle(
                timestamp=start + timedelta(minutes=minutes * index),
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                timeframe=timeframe,
            )
        )
    return candles


def test_evaluate_gap_context_prefers_fill_for_moderate_gap_up():
    friday = datetime(2026, 4, 24, 20, 0, tzinfo=timezone.utc)
    monday = datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc)
    h4_candles = _candles(
        "H4",
        [
            (100.0, 104.0, 99.0, 103.0, 80),
            (103.0, 106.0, 102.0, 105.0, 90),
            (105.0, 108.0, 104.0, 107.0, 100),
            (107.0, 110.0, 106.0, 109.0, 110),
            (109.0, 112.0, 108.0, 110.0, 120),
        ],
        start=friday,
        minutes=240,
    )
    m15_candles = [
        Candle(timestamp=friday, open=109.0, high=110.0, low=108.5, close=110.0, volume=100, timeframe="M15"),
        Candle(timestamp=monday, open=113.0, high=113.4, low=111.6, close=112.2, volume=120, timeframe="M15"),
        Candle(timestamp=monday + timedelta(minutes=15), open=112.2, high=112.5, low=111.2, close=111.8, volume=130, timeframe="M15"),
    ]

    result = evaluate_gap_context(h4_candles=h4_candles, m15_candles=m15_candles)

    assert result.has_gap is True
    assert result.gap_direction is BreakoutDirection.BULLISH
    assert result.preferred_trade_direction is BreakoutDirection.BEARISH
    assert result.fill_preferred is True
    assert result.size_class in {"small", "moderate"}


def test_evaluate_gap_context_avoids_forcing_fill_for_large_acceptance_gap():
    friday = datetime(2026, 4, 24, 20, 0, tzinfo=timezone.utc)
    monday = datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc)
    h4_candles = _candles(
        "H4",
        [
            (100.0, 104.0, 99.0, 103.0, 80),
            (103.0, 106.0, 102.0, 105.0, 90),
            (105.0, 108.0, 104.0, 107.0, 100),
            (107.0, 110.0, 106.0, 109.0, 110),
            (109.0, 112.0, 108.0, 110.0, 120),
        ],
        start=friday,
        minutes=240,
    )
    m15_candles = [
        Candle(timestamp=friday, open=109.0, high=110.0, low=108.5, close=110.0, volume=100, timeframe="M15"),
        Candle(timestamp=monday, open=116.0, high=117.5, low=115.8, close=117.2, volume=160, timeframe="M15"),
        Candle(timestamp=monday + timedelta(minutes=15), open=117.2, high=118.4, low=117.0, close=118.1, volume=170, timeframe="M15"),
    ]

    result = evaluate_gap_context(h4_candles=h4_candles, m15_candles=m15_candles)

    assert result.has_gap is True
    assert result.gap_direction is BreakoutDirection.BULLISH
    assert result.preferred_trade_direction is BreakoutDirection.BULLISH
    assert result.fill_preferred is False
    assert result.size_class == "large"


def test_evaluate_gap_context_neutralizes_bias_once_fill_is_completed_and_overextended():
    friday = datetime(2026, 4, 24, 20, 0, tzinfo=timezone.utc)
    monday = datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc)
    h4_candles = _candles(
        "H4",
        [
            (4668.0, 4688.0, 4666.0, 4681.73, 80),
            (4681.73, 4696.0, 4678.0, 4690.0, 90),
            (4690.0, 4704.0, 4688.0, 4699.0, 100),
            (4699.0, 4716.0, 4694.0, 4710.0, 110),
            (4710.0, 4725.0, 4700.0, 4718.0, 120),
        ],
        start=friday,
        minutes=240,
    )
    m15_candles = [
        Candle(timestamp=friday, open=4680.5, high=4684.0, low=4679.8, close=4681.73, volume=100, timeframe="M15"),
        Candle(timestamp=monday, open=4686.68, high=4688.2, low=4685.5, close=4687.1, volume=120, timeframe="M15"),
        Candle(timestamp=monday + timedelta(minutes=15), open=4687.1, high=4688.0, low=4678.5, close=4679.2, volume=130, timeframe="M15"),
        Candle(timestamp=monday + timedelta(minutes=30), open=4679.2, high=4680.0, low=4667.4, close=4667.58, volume=140, timeframe="M15"),
    ]

    result = evaluate_gap_context(h4_candles=h4_candles, m15_candles=m15_candles)

    assert result.has_gap is True
    assert result.reason == "gap_fill_completed"
    assert result.preferred_trade_direction is None
    assert result.fill_preferred is False
