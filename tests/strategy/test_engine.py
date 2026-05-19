from datetime import datetime, timezone


def _supply_zone() -> dict[str, float | str]:
    return {
        "kind": "SUPPLY",
        "major_low": 2344.5,
        "major_high": 2351.25,
        "refinement_low": 2347.0,
        "refinement_high": 2351.25,
    }


def _breakout_candle():
    from src.market_data import Candle

    return Candle(
        timestamp=datetime(2026, 4, 4, 8, 5, tzinfo=timezone.utc),
        open=2348.5,
        high=2352.5,
        low=2347.25,
        close=2351.5,
        volume=120,
        timeframe="M5",
    )


def _retest_candle():
    from src.market_data import Candle

    return Candle(
        timestamp=datetime(2026, 4, 4, 8, 10, tzinfo=timezone.utc),
        open=2351.1,
        high=2351.9,
        low=2347.8,
        close=2351.45,
        volume=104,
        timeframe="M5",
    )


def test_evaluate_break_and_retest_setup_returns_trade_plan():
    from src.strategy.engine import TradePlan, evaluate_break_and_retest_setup

    result = evaluate_break_and_retest_setup(
        session_timestamp=datetime(2026, 4, 4, 8, 10, tzinfo=timezone.utc),
        breakout_candle=_breakout_candle(),
        retest_candle=_retest_candle(),
        zone=_supply_zone(),
        risk_buffer=0.05,
        max_candles_since_breakout=3,
    )

    assert isinstance(result, TradePlan)
    assert result.is_trade is True
    assert result.direction.value == "BULLISH"
    assert result.entry_price == 2351.45
    assert result.stop_loss == 2346.95
    assert result.take_profit == 2362.70  # 2.5R base: entry + risk*2.5 = 2351.45 + 4.50*2.5
    assert result.reason == "trade_plan_ready"
    assert result.metadata["confirmation_reason"] == "rejection_candle_confirmed"


def test_evaluate_break_and_retest_setup_returns_no_trade_when_session_is_closed():
    from src.strategy.engine import NoTradeResult, evaluate_break_and_retest_setup

    result = evaluate_break_and_retest_setup(
        session_timestamp=datetime(2026, 4, 4, 6, 30, tzinfo=timezone.utc),
        breakout_candle=_breakout_candle(),
        retest_candle=_retest_candle(),
        zone=_supply_zone(),
        risk_buffer=0.05,
        max_candles_since_breakout=3,
    )

    assert isinstance(result, NoTradeResult)
    assert result.is_trade is False
    assert result.reason == "session_not_allowed"
    assert result.metadata["stage"] == "session_filter"


def test_evaluate_break_and_retest_setup_returns_no_trade_when_breakout_is_missing():
    from src.market_data import Candle
    from src.strategy.engine import NoTradeResult, evaluate_break_and_retest_setup

    no_breakout_candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 5, tzinfo=timezone.utc),
        open=2349.0,
        high=2351.0,
        low=2348.25,
        close=2350.75,
        volume=98,
        timeframe="M5",
    )

    result = evaluate_break_and_retest_setup(
        session_timestamp=datetime(2026, 4, 4, 8, 10, tzinfo=timezone.utc),
        breakout_candle=no_breakout_candle,
        retest_candle=_retest_candle(),
        zone=_supply_zone(),
        risk_buffer=0.05,
        max_candles_since_breakout=3,
    )

    assert isinstance(result, NoTradeResult)
    assert result.is_trade is False
    assert result.reason == "no_breakout"
    assert result.metadata["stage"] == "breakout"


def test_evaluate_break_and_retest_setup_returns_no_trade_when_confirmation_fails():
    from src.market_data import Candle
    from src.strategy.engine import NoTradeResult, evaluate_break_and_retest_setup

    weak_retest_candle = Candle(
        timestamp=datetime(2026, 4, 4, 8, 10, tzinfo=timezone.utc),
        open=2351.0,
        high=2351.2,
        low=2347.8,
        close=2351.1,
        volume=104,
        timeframe="M5",
    )

    result = evaluate_break_and_retest_setup(
        session_timestamp=datetime(2026, 4, 4, 8, 10, tzinfo=timezone.utc),
        breakout_candle=_breakout_candle(),
        retest_candle=weak_retest_candle,
        zone=_supply_zone(),
        risk_buffer=0.05,
        max_candles_since_breakout=3,
    )

    assert isinstance(result, NoTradeResult)
    assert result.is_trade is False
    assert result.reason == "no_rejection_confirmation"
    assert result.metadata["stage"] == "confirmation"
