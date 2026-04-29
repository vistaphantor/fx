import pytest
from datetime import datetime, timezone
from src.market_data import Candle
from src.strategy.structural import detect_order_blocks, detect_fair_value_gaps, score_structural_context

def _make_candle(open_p, high, low, close, vol=100):
    return Candle(
        timestamp=datetime.now(tz=timezone.utc),
        open=float(open_p),
        high=float(high),
        low=float(low),
        close=float(close),
        volume=float(vol),
        timeframe="M15"
    )

def test_detect_order_blocks_bullish():
    candles = [
        _make_candle(100, 101, 99, 98, vol=100), # Bearish
        _make_candle(98, 105, 98, 104, vol=200), # Strong Bullish
        _make_candle(104, 106, 103, 105, vol=150)
    ]
    obs = detect_order_blocks(candles)
    assert len(obs) >= 1
    assert obs[0].direction == 1
    assert obs[0].price_low == 99.0
    assert obs[0].price_high == 101.0

def test_detect_order_blocks_bearish():
    candles = [
        _make_candle(100, 102, 99, 101, vol=100), # Bullish
        _make_candle(101, 101, 95, 96, vol=200), # Strong Bearish
        _make_candle(96, 97, 94, 95, vol=150)
    ]
    obs = detect_order_blocks(candles)
    assert len(obs) >= 1
    assert obs[0].direction == -1
    assert obs[0].price_low == 99.0
    assert obs[0].price_high == 102.0

def test_detect_fair_value_gaps_bullish():
    candles = [
        _make_candle(100, 102, 98, 101), # High 102
        _make_candle(101, 110, 101, 109),
        _make_candle(109, 115, 105, 114)  # Low 105
    ]
    # post.low (105) > prev.high (102)
    fvgs = detect_fair_value_gaps(candles)
    assert len(fvgs) >= 1
    assert fvgs[0].direction == 1
    assert fvgs[0].top == 105.0
    assert fvgs[0].bottom == 102.0

def test_score_structural_context_neutral():
    candles = [_make_candle(100, 101, 99, 100) for _ in range(10)]
    score = score_structural_context(
        price=100.0,
        candles=candles,
        demand_zones=((90.0, 95.0),),
        supply_zones=((105.0, 110.0),),
        atr=2.0
    )
    # No OBs, No FVGs, Zones far away
    assert 0 <= score <= 1.0
