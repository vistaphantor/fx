from datetime import datetime, timedelta, timezone

import pytest

from src.market_data import Candle
from src.strategy.breakout import BreakoutDirection
from src.strategy.top_down import evaluate_top_down_strategy


def _candles(timeframe: str, closes: list[float], *, start_price: float = 100.0) -> list[Candle]:
    start = datetime(2026, 4, 6, 8, 0, tzinfo=timezone.utc)
    minutes = {"M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}[timeframe]
    output = []
    for index, close in enumerate(closes):
        open_price = start_price if index == 0 else closes[index - 1]
        output.append(
            Candle(
                timestamp=start + timedelta(minutes=minutes * index),
                open=open_price,
                high=max(open_price, close) + 1.0,
                low=min(open_price, close) - 1.0,
                close=close,
                volume=100 + index,
                timeframe=timeframe,
            )
        )
    return output


def test_top_down_strategy_builds_bullish_plan_from_htf_bias_and_m15_velocity():
    result = evaluate_top_down_strategy(
        d1_candles=_candles("D1", [95, 110]),
        h4_candles=_candles("H4", [96, 98, 101, 103, 106, 108]),
        h1_candles=_candles("H1", [100, 101, 103, 105, 107]),
        m30_candles=_candles("M30", [102, 103, 105, 106, 108]),
        m15_candles=_candles("M15", [102, 103, 104, 105, 106, 108]),
        risk_buffer=0.05,
    )

    assert result.is_trade is True
    assert result.direction is BreakoutDirection.BULLISH
    assert result.reason == "top_down_trade_plan_ready"
    assert result.entry_price == pytest.approx(108)
    assert result.stop_loss == pytest.approx(100.95)
    assert result.take_profit == pytest.approx(125.625)  # 2.5R base: entry + risk*2.5 = 108 + 7.05*2.5
    assert result.metadata["daily_objective"] == pytest.approx(111.0)


def test_top_down_strategy_returns_no_trade_when_htf_bias_conflicts_with_m15_velocity():
    result = evaluate_top_down_strategy(
        d1_candles=_candles("D1", [95, 110]),
        h4_candles=_candles("H4", [96, 98, 101, 103, 106, 108]),
        h1_candles=_candles("H1", [100, 101, 103, 105, 107]),
        m30_candles=_candles("M30", [102, 103, 105, 106, 108]),
        m15_candles=_candles("M15", [108, 107, 106, 105, 104, 103]),
        risk_buffer=0.05,
    )

    assert result.is_trade is False
    assert result.reason == "m15_velocity_disagrees_with_top_down_bias"
    assert result.metadata["stage"] == "top_down_strategy"
