from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.market_data import Candle
from src.strategy.breakout import BreakoutDirection
from src.strategy.campaign_add import (
    _adaptive_add_threshold,
    _adaptive_add_trigger_r,
    evaluate_campaign_add,
)
from src.strategy.decision_tree import _default_strategy_profile


def _candles(timeframe: str, rows: list[tuple[float, float, float, float]], step_minutes: int) -> list[Candle]:
    start = datetime(2026, 4, 28, 8, 0, tzinfo=timezone.utc)
    return [
        Candle(
            timestamp=start + timedelta(minutes=step_minutes * index),
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=100 + index,
            timeframe=timeframe,
        )
        for index, (open_price, high, low, close) in enumerate(rows)
    ]


def _bearish_campaign_live_input():
    return type(
        "LiveInput",
        (),
        {
            "h4_candles": _candles(
                "H4",
                [
                    (4700.0, 4725.39, 4688.0, 4712.0),
                    (4712.0, 4720.0, 4684.0, 4694.0),
                    (4694.0, 4700.0, 4666.35, 4672.0),
                    (4672.0, 4676.89, 4668.2, 4675.0),
                    (4675.0, 4701.25, 4672.0, 4693.0),
                    (4693.0, 4698.0, 4682.0, 4688.0),
                ],
                240,
            ),
            "m15_candles": _candles(
                "M15",
                [
                    (4650.5, 4651.2, 4648.8, 4650.4),
                    (4650.4, 4654.2, 4649.6, 4653.5),
                    (4653.5, 4657.73, 4624.3, 4625.15),
                    (4625.2, 4647.18, 4624.9, 4641.44),
                ],
                15,
            ),
            "m10_candles": _candles(
                "M10",
                [
                    (4657.08, 4657.73, 4637.22, 4637.34),
                    (4637.28, 4642.79, 4631.40, 4641.05),
                    (4646.10, 4646.60, 4638.89, 4641.72),
                ],
                10,
            ),
            "m5_candles": _candles(
                "M5",
                [
                    (4631.46, 4634.04, 4625.73, 4633.99),
                    (4633.98, 4634.20, 4631.40, 4633.30),
                    (4633.50, 4634.00, 4631.10, 4632.00),
                ],
                5,
            ),
        },
    )()


def test_campaign_add_accepts_bearish_continuation_with_valid_m15_m10_m5_stack():
    live_input = _bearish_campaign_live_input()

    result = evaluate_campaign_add(
        symbol="XAUUSD",
        live_input=live_input,
        direction=BreakoutDirection.BEARISH,
        risk_buffer=0.05,
        latest_trade_r_multiple=2.2,
        continuation_edge=1.8,
        continuation_threshold=1.5,
        strategy_profile=_default_strategy_profile("XAUUSD"),
    )

    assert result.is_ready is True
    assert result.reason == "campaign_add_ready"
    assert result.direction is BreakoutDirection.BEARISH
    assert result.entry_price is not None
    assert result.stop_loss is not None
    assert result.take_profit is not None
    assert result.quality_score >= 0.56


def test_campaign_add_rejects_when_m10_continuation_quality_is_too_weak():
    live_input = type(
        "LiveInput",
        (),
        {
            "h4_candles": _candles(
                "H4",
                [
                    (4700.0, 4725.39, 4688.0, 4712.0),
                    (4712.0, 4720.0, 4684.0, 4694.0),
                    (4694.0, 4700.0, 4666.35, 4672.0),
                    (4672.0, 4676.89, 4668.2, 4675.0),
                    (4675.0, 4701.25, 4672.0, 4693.0),
                    (4693.0, 4698.0, 4682.0, 4688.0),
                ],
                240,
            ),
            "m15_candles": _candles(
                "M15",
                [
                    (4650.5, 4651.2, 4648.8, 4650.4),
                    (4650.4, 4654.2, 4649.6, 4653.5),
                    (4653.5, 4657.73, 4624.3, 4625.15),
                    (4625.2, 4647.18, 4624.9, 4641.44),
                ],
                15,
            ),
            "m10_candles": _candles(
                "M10",
                [
                    (4657.08, 4657.73, 4637.22, 4637.34),
                    (4637.28, 4654.60, 4631.40, 4653.40),
                    (4656.80, 4657.10, 4652.89, 4654.95),
                ],
                10,
            ),
            "m5_candles": _candles(
                "M5",
                [
                    (4631.46, 4634.04, 4625.73, 4633.99),
                    (4633.98, 4634.20, 4631.40, 4633.30),
                    (4633.50, 4634.00, 4631.10, 4632.00),
                ],
                5,
            ),
        },
    )()

    result = evaluate_campaign_add(
        symbol="XAUUSD",
        live_input=live_input,
        direction=BreakoutDirection.BEARISH,
        risk_buffer=0.05,
        latest_trade_r_multiple=2.2,
        continuation_edge=1.8,
        continuation_threshold=1.5,
        strategy_profile=_default_strategy_profile("XAUUSD"),
    )

    assert result.is_ready is False
    assert result.reason == "campaign_add_m10_not_ready"
    assert "continuation_score" in result.metadata


def test_campaign_add_is_hard_blocked_when_quant_flips_against_direction():
    result = evaluate_campaign_add(
        symbol="XAUUSD",
        live_input=_bearish_campaign_live_input(),
        direction=BreakoutDirection.BEARISH,
        risk_buffer=0.05,
        latest_trade_r_multiple=2.2,
        continuation_edge=1.8,
        continuation_threshold=1.5,
        strategy_profile=_default_strategy_profile("XAUUSD"),
        quant_decision=SimpleNamespace(
            action=1,
            cvar=0.001,
            drawdown_dampener=1.0,
            reason="master_equation_long_approved",
        ),
    )

    assert result.is_ready is False
    assert result.reason == "campaign_add_quant_blocked"
    assert result.quant_state == "hard_block"


def test_campaign_add_is_reduced_not_blocked_when_quant_is_flat_but_campaign_is_protected():
    result = evaluate_campaign_add(
        symbol="XAUUSD",
        live_input=_bearish_campaign_live_input(),
        direction=BreakoutDirection.BEARISH,
        risk_buffer=0.05,
        latest_trade_r_multiple=2.2,
        continuation_edge=1.8,
        continuation_threshold=1.5,
        strategy_profile=_default_strategy_profile("XAUUSD"),
        quant_decision=SimpleNamespace(
            action=0,
            cvar=0.001,
            drawdown_dampener=1.0,
            reason="master_equation_flat",
        ),
    )

    assert result.is_ready is True
    assert result.reason == "campaign_add_quant_reduced"
    assert result.lot_multiplier == 0.5
    assert result.quant_state == "soft_reduce"


def test_campaign_add_trigger_relaxes_when_momentum_accelerates():
    relaxed = _adaptive_add_trigger_r(
        base_trigger_r=1.5,
        accel_bonus=0.20,
        execution_penalty=0.0,
        volatility_penalty=0.0,
        floor_r=1.25,
        ceiling_r=1.75,
    )

    assert relaxed == 1.3


def test_campaign_add_trigger_tightens_when_execution_penalty_rises():
    tightened = _adaptive_add_trigger_r(
        base_trigger_r=1.5,
        accel_bonus=0.0,
        execution_penalty=0.18,
        volatility_penalty=0.04,
        floor_r=1.25,
        ceiling_r=1.75,
    )

    assert tightened == 1.72


def test_campaign_add_threshold_relaxes_in_protected_campaign():
    protected = _adaptive_add_threshold(
        base_threshold=0.56,
        protected_bonus=0.08,
        accel_bonus=0.03,
        execution_penalty=0.01,
        quant_penalty=0.0,
        lower_bound=0.45,
        upper_bound=0.75,
    )
    unprotected = _adaptive_add_threshold(
        base_threshold=0.56,
        protected_bonus=0.0,
        accel_bonus=0.0,
        execution_penalty=0.01,
        quant_penalty=0.0,
        lower_bound=0.45,
        upper_bound=0.75,
    )

    assert protected < unprotected
