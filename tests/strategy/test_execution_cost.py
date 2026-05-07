from types import SimpleNamespace

from src.strategy.breakout import BreakoutDirection
from src.strategy.execution_cost import assess_market_order_execution
from src.strategy.volatility import VolatilityState


def test_execution_cost_rejects_trade_when_spread_crushes_stop_distance():
    assessment = assess_market_order_execution(
        direction=BreakoutDirection.BULLISH,
        planned_entry=100.0,
        stop_loss=99.7,
        take_profit=101.5,
        current_bid=99.95,
        current_ask=100.15,
        spread=0.20,
        volatility_state=VolatilityState(
            short_atr=0.40,
            medium_atr=0.50,
            realized_range=1.0,
            body_efficiency=0.7,
            range_expansion_ratio=1.0,
        ),
        requested_lot=1.0,
        campaign_exposure_pct=2.0,
    )

    assert assessment.is_tradeable is False
    assert assessment.reason == "spread_pressure_too_high"


def test_execution_cost_reduces_size_when_penalty_is_elevated_but_trade_survives():
    assessment = assess_market_order_execution(
        direction=BreakoutDirection.BULLISH,
        planned_entry=100.0,
        stop_loss=98.0,
        take_profit=106.0,
        current_bid=99.98,
        current_ask=100.02,
        spread=0.04,
        volatility_state=VolatilityState(
            short_atr=0.55,
            medium_atr=0.65,
            realized_range=1.6,
            body_efficiency=0.7,
            range_expansion_ratio=1.05,
        ),
        requested_lot=5.0,
        campaign_exposure_pct=8.0,
    )

    assert assessment.is_tradeable is True
    assert assessment.recommended_lot_multiplier < 1.0
    assert assessment.effective_rr > 1.5


def test_execution_cost_allows_strong_continuation_below_legacy_rr_floor():
    assessment = assess_market_order_execution(
        direction=BreakoutDirection.BULLISH,
        planned_entry=100.0,
        stop_loss=98.7,
        take_profit=101.5,
        current_bid=99.98,
        current_ask=100.02,
        spread=0.04,
        volatility_state=VolatilityState(
            short_atr=0.35,
            medium_atr=0.45,
            realized_range=1.0,
            body_efficiency=0.82,
            range_expansion_ratio=1.25,
        ),
        requested_lot=0.01,
        campaign_exposure_pct=0.0,
        continuation_context={
            "is_continuation_setup": True,
            "m15_quality": 0.95,
            "m10_quality": 0.92,
            "m5_quality": 0.90,
            "range_expansion_ratio": 1.25,
            "body_efficiency": 0.82,
            "regime_confidence": 0.90,
        },
    )

    assert assessment.is_tradeable is True
    assert assessment.effective_rr < 1.5


def test_execution_cost_rejects_trade_below_continuation_disaster_floor():
    assessment = assess_market_order_execution(
        direction=BreakoutDirection.BULLISH,
        planned_entry=100.0,
        stop_loss=98.0,
        take_profit=100.8,
        current_bid=99.98,
        current_ask=100.02,
        spread=0.04,
        volatility_state=VolatilityState(
            short_atr=0.35,
            medium_atr=0.45,
            realized_range=1.0,
            body_efficiency=0.90,
            range_expansion_ratio=1.30,
        ),
        requested_lot=0.01,
        campaign_exposure_pct=0.0,
        continuation_context={
            "is_continuation_setup": True,
            "m15_quality": 0.98,
            "m10_quality": 0.95,
            "m5_quality": 0.94,
            "range_expansion_ratio": 1.30,
            "body_efficiency": 0.90,
            "regime_confidence": 0.95,
        },
    )

    assert assessment.is_tradeable is False
    assert assessment.reason == "continuation_rr_below_disaster_floor"


def test_execution_cost_keeps_legacy_rr_rejection_for_fresh_entries():
    assessment = assess_market_order_execution(
        direction=BreakoutDirection.BULLISH,
        planned_entry=100.0,
        stop_loss=98.7,
        take_profit=101.5,
        current_bid=99.98,
        current_ask=100.02,
        spread=0.04,
        volatility_state=VolatilityState(
            short_atr=0.35,
            medium_atr=0.45,
            realized_range=1.0,
            body_efficiency=0.82,
            range_expansion_ratio=1.25,
        ),
        requested_lot=0.01,
        campaign_exposure_pct=0.0,
    )

    assert assessment.is_tradeable is False
    assert assessment.reason == "execution_rr_degraded"


def test_execution_cost_exposes_continuation_mu_for_continuation_setup():
    assessment = assess_market_order_execution(
        direction=BreakoutDirection.BULLISH,
        planned_entry=100.0,
        stop_loss=98.7,
        take_profit=101.5,
        current_bid=99.98,
        current_ask=100.02,
        spread=0.04,
        volatility_state=VolatilityState(
            short_atr=0.35,
            medium_atr=0.45,
            realized_range=1.0,
            body_efficiency=0.82,
            range_expansion_ratio=1.25,
        ),
        requested_lot=0.01,
        campaign_exposure_pct=0.0,
        continuation_context={
            "is_continuation_setup": True,
            "m15_quality": 0.95,
            "m10_quality": 0.92,
            "m5_quality": 0.90,
            "range_expansion_ratio": 1.25,
            "body_efficiency": 0.82,
            "regime_confidence": 0.90,
        },
    )

    assert assessment.continuation_probability > 0.0
    assert assessment.continuation_mu > -1.0
    assert assessment.effective_gain_remaining > 0.0
    assert assessment.effective_loss_if_failed > 0.0


def test_execution_cost_exposes_directional_tail_proxy_by_trade_direction():
    common_kwargs = dict(
        planned_entry=100.0,
        stop_loss=98.7,
        take_profit=101.5,
        current_bid=99.98,
        current_ask=100.02,
        spread=0.04,
        volatility_state=VolatilityState(
            short_atr=0.35,
            medium_atr=0.45,
            realized_range=1.0,
            body_efficiency=0.62,
            range_expansion_ratio=1.35,
        ),
        requested_lot=0.01,
        campaign_exposure_pct=0.0,
        continuation_context={
            "is_continuation_setup": True,
            "m15_quality": 0.90,
            "m10_quality": 0.88,
            "m5_quality": 0.86,
            "range_expansion_ratio": 1.35,
            "body_efficiency": 0.62,
            "regime_confidence": 0.84,
        },
    )

    long_assessment = assess_market_order_execution(
        direction=BreakoutDirection.BULLISH,
        **common_kwargs,
    )
    short_assessment = assess_market_order_execution(
        direction=BreakoutDirection.BEARISH,
        **common_kwargs,
    )

    assert long_assessment.directional_tail_proxy != short_assessment.directional_tail_proxy


def test_execution_cost_uses_improved_live_market_price_for_buy_entries():
    assessment = assess_market_order_execution(
        direction=BreakoutDirection.BULLISH,
        planned_entry=100.0,
        stop_loss=98.8,
        take_profit=101.6,
        current_bid=99.88,
        current_ask=99.90,
        spread=0.02,
        volatility_state=VolatilityState(
            short_atr=0.02,
            medium_atr=0.03,
            realized_range=0.2,
            body_efficiency=0.8,
            range_expansion_ratio=1.0,
        ),
        requested_lot=0.01,
        campaign_exposure_pct=0.0,
    )

    assert assessment.effective_entry < 100.0
