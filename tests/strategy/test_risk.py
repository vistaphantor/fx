from datetime import datetime, timezone

import pytest


@pytest.mark.parametrize(
    (
        "direction_name",
        "entry_price",
        "retest_structure_low",
        "retest_structure_high",
        "buffer",
        "expected_stop_loss",
        "expected_take_profit",
    ),
    [
        (
            "BULLISH",
            2351.45,
            2347.0,
            2351.25,
            0.05,
            2346.95,
            2362.70,   # base 2.5R: entry + risk*2.5 = 2351.45 + 4.50*2.5
        ),
        (
            "BEARISH",
            2322.55,
            2323.75,
            2328.0,
            0.05,
            2328.05,
            2308.80,   # base 2.5R: entry - risk*2.5 = 2322.55 - 5.50*2.5
        ),
    ],
)
def test_build_trade_levels_sets_stop_from_structure_and_target_at_three_r(
    direction_name,
    entry_price,
    retest_structure_low,
    retest_structure_high,
    buffer,
    expected_stop_loss,
    expected_take_profit,
):
    from src.strategy.breakout import BreakoutDirection
    from src.strategy.risk import build_trade_levels

    direction = getattr(BreakoutDirection, direction_name)

    levels = build_trade_levels(
        entry_price=entry_price,
        direction=direction,
        retest_structure_low=retest_structure_low,
        retest_structure_high=retest_structure_high,
        buffer=buffer,
        candle_timestamp=datetime(2026, 4, 4, 8, 10, tzinfo=timezone.utc),
    )

    assert levels.direction is direction
    assert levels.entry_price == pytest.approx(entry_price)
    assert levels.stop_loss == pytest.approx(expected_stop_loss)
    assert levels.take_profit == pytest.approx(expected_take_profit)
    assert levels.risk == pytest.approx(abs(entry_price - expected_stop_loss))
    assert levels.reward == pytest.approx(levels.risk * 2.5)
    assert levels.risk_reward_ratio == pytest.approx(2.5)


def test_build_trade_levels_rejects_non_positive_buffer():
    from src.strategy.breakout import BreakoutDirection
    from src.strategy.risk import build_trade_levels

    with pytest.raises(ValueError, match="buffer must be positive"):
        build_trade_levels(
            entry_price=2351.45,
            direction=BreakoutDirection.BULLISH,
            retest_structure_low=2347.0,
            retest_structure_high=2351.25,
            buffer=0.0,
            candle_timestamp=datetime(2026, 4, 4, 8, 10, tzinfo=timezone.utc),
        )
