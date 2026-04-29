from datetime import datetime, timedelta, timezone

from src.market_data import Candle
from src.strategy.breakout import BreakoutDirection
from src.strategy.gap import GapDecision
from src.strategy.regime import classify_regime
from src.strategy.volatility import VolatilityState


def _candles(timeframe: str, rows: list[tuple[float, float, float, float, int]]) -> list[Candle]:
    start = datetime(2026, 4, 28, 0, 0, tzinfo=timezone.utc)
    minutes = {"H1": 60, "M30": 30, "M15": 15}[timeframe]
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


def test_classify_regime_marks_expansion_when_range_and_impulse_expand():
    regime = classify_regime(
        h1_candles=_candles(
            "H1",
            [
                (100.0, 103.0, 99.0, 102.0, 100),
                (102.0, 106.0, 101.0, 105.0, 105),
                (105.0, 110.0, 104.0, 109.0, 110),
                (109.0, 115.0, 108.0, 114.0, 115),
            ],
        ),
        m30_candles=_candles(
            "M30",
            [
                (110.0, 112.0, 109.0, 111.5, 100),
                (111.5, 115.0, 111.0, 114.0, 120),
                (114.0, 119.0, 113.5, 118.5, 130),
            ],
        ),
        m15_candles=_candles(
            "M15",
            [
                (115.0, 116.0, 114.7, 115.8, 100),
                (115.8, 118.5, 115.4, 118.1, 120),
                (118.1, 122.0, 117.8, 121.7, 150),
            ],
        ),
        volatility_state=VolatilityState(
            short_atr=4.0,
            medium_atr=3.0,
            realized_range=23.0,
            body_efficiency=0.82,
            range_expansion_ratio=1.35,
        ),
        gap_decision=None,
    )

    assert regime.name == "expansion"
    assert regime.tradable is True
    assert regime.continuation_bias > regime.reversion_bias


def test_classify_regime_marks_compression_when_ranges_contract():
    regime = classify_regime(
        h1_candles=_candles(
            "H1",
            [
                (100.0, 102.0, 99.0, 101.0, 100),
                (101.0, 102.5, 100.5, 101.4, 98),
                (101.4, 102.0, 100.9, 101.3, 97),
                (101.3, 101.9, 100.8, 101.2, 96),
            ],
        ),
        m30_candles=_candles(
            "M30",
            [
                (101.1, 101.7, 100.9, 101.3, 100),
                (101.3, 101.6, 101.0, 101.25, 100),
                (101.25, 101.55, 101.05, 101.2, 100),
            ],
        ),
        m15_candles=_candles(
            "M15",
            [
                (101.2, 101.4, 101.0, 101.25, 100),
                (101.25, 101.4, 101.1, 101.22, 100),
                (101.22, 101.35, 101.08, 101.2, 100),
            ],
        ),
        volatility_state=VolatilityState(
            short_atr=0.45,
            medium_atr=0.8,
            realized_range=3.5,
            body_efficiency=0.28,
            range_expansion_ratio=0.62,
        ),
        gap_decision=GapDecision(
            has_gap=False,
            gap_size=0.0,
            size_class="none",
            gap_direction=None,
            preferred_trade_direction=None,
            fill_preferred=False,
            reason="no_gap",
            metadata={},
        ),
    )

    assert regime.name == "compression"
    assert regime.tradable is False
    assert regime.reversion_bias >= regime.continuation_bias
