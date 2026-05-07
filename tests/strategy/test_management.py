from types import SimpleNamespace

import pytest

from src.strategy.breakout import BreakoutDirection
from src.strategy.management import evaluate_campaign_action


@pytest.fixture(autouse=True)
def isolated_initial_stop_cache(monkeypatch, tmp_path):
    import src.strategy.management as management

    monkeypatch.setattr(management, "_INITIAL_STOP_CACHE_PATH", tmp_path / "initial_stop_cache.json")
    management._INITIAL_STOP_CACHE.clear()
    yield
    management._INITIAL_STOP_CACHE.clear()


def test_management_allows_add_after_latest_trade_reaches_two_r():
    positions = [
        SimpleNamespace(entry_price=100.0, initial_stop_loss=95.0, stop_loss=110.0, volume=0.01),
        SimpleNamespace(entry_price=105.0, initial_stop_loss=100.0, stop_loss=110.0, volume=0.01),
    ]

    result = evaluate_campaign_action(
        positions=positions,
        current_price=115.0,
        direction=BreakoutDirection.BULLISH,
        latest_trade_r_multiple=2.0,
        default_lot=0.01,
        add_on_lot_increment=0.01,
        max_exposure_pct=10.0,
        margin_snapshot={
            "campaign_exposure_pct": 6.0,
            "preferred_add_exposure_pct": 2.0,
            "fallback_add_exposure_pct": 1.0,
        },
        reversal_confirmed=False,
    )

    assert result.action == "add_position"
    assert result.reason == "campaign_add_ready"
    assert result.add_lot == 0.02


def test_campaign_moves_stop_to_entry_at_min_breakeven_trigger():
    positions = [
        SimpleNamespace(entry_price=100.0, initial_stop_loss=95.0, stop_loss=98.0, volume=0.01),
        SimpleNamespace(entry_price=105.0, initial_stop_loss=101.0, stop_loss=103.0, volume=0.01),
    ]

    result = evaluate_campaign_action(
        positions=positions,
        current_price=106.5,
        direction=BreakoutDirection.BULLISH,
        latest_trade_r_multiple=0.375,
        default_lot=0.01,
        add_on_lot_increment=0.01,
        max_exposure_pct=10.0,
        margin_snapshot={
            "campaign_exposure_pct": 4.0,
            "preferred_add_exposure_pct": 1.5,
            "fallback_add_exposure_pct": 1.0,
        },
        reversal_confirmed=False,
        breakeven_distance=1.5,
    )

    assert result.action == "trail_all"
    assert result.reason == "campaign_breakeven_earned"
    assert result.stop_updates == ((0, 100.0), (1, 105.0))


def test_campaign_locks_plus_1r_at_plus_2r():
    positions = [
        SimpleNamespace(entry_price=100.0, initial_stop_loss=95.0, stop_loss=100.0, volume=0.01),
        SimpleNamespace(entry_price=105.0, initial_stop_loss=101.0, stop_loss=105.0, volume=0.01),
    ]

    result = evaluate_campaign_action(
        positions=positions,
        current_price=113.0,
        direction=BreakoutDirection.BULLISH,
        latest_trade_r_multiple=2.0,
        default_lot=0.01,
        add_on_lot_increment=0.01,
        max_exposure_pct=10.0,
        margin_snapshot={
            "campaign_exposure_pct": 4.0,
            "preferred_add_exposure_pct": 1.5,
            "fallback_add_exposure_pct": 1.0,
        },
        reversal_confirmed=False,
    )

    assert result.action == "trail_all"
    assert result.reason == "campaign_profit_lock_progression"
    assert result.stop_updates == ((0, 105.0), (1, 109.0))


def test_campaign_locks_more_profit_on_older_trade_than_newer_trade():
    positions = [
        SimpleNamespace(entry_price=100.0, initial_stop_loss=95.0, stop_loss=100.0, volume=0.01),
        SimpleNamespace(entry_price=108.0, initial_stop_loss=104.0, stop_loss=108.0, volume=0.01),
    ]

    result = evaluate_campaign_action(
        positions=positions,
        current_price=116.0,
        direction=BreakoutDirection.BULLISH,
        latest_trade_r_multiple=2.0,
        default_lot=0.01,
        add_on_lot_increment=0.01,
        max_exposure_pct=10.0,
        margin_snapshot={
            "campaign_exposure_pct": 4.0,
            "preferred_add_exposure_pct": 1.5,
            "fallback_add_exposure_pct": 1.0,
        },
        reversal_confirmed=False,
    )

    assert result.action == "trail_all"
    stop_map = dict(result.stop_updates)
    older_locked_profit = stop_map[0] - positions[0].entry_price
    newer_locked_profit = stop_map[1] - positions[1].entry_price
    assert older_locked_profit > newer_locked_profit


def test_management_supports_mt5_trade_position_field_names_when_trailing():
    positions = [
        SimpleNamespace(price_open=100.0, sl=95.0, volume=0.01, type=0),
        SimpleNamespace(price_open=105.0, sl=101.0, volume=0.01, type=0),
    ]

    result = evaluate_campaign_action(
        positions=positions,
        current_price=106.5,
        direction=BreakoutDirection.BULLISH,
        latest_trade_r_multiple=0.375,
        default_lot=0.01,
        add_on_lot_increment=0.01,
        max_exposure_pct=10.0,
        margin_snapshot={
            "campaign_exposure_pct": 4.0,
            "preferred_add_exposure_pct": 1.5,
            "fallback_add_exposure_pct": 1.0,
        },
        reversal_confirmed=False,
        breakeven_distance=1.5,
    )

    assert result.action == "trail_all"
    assert result.reason == "campaign_breakeven_earned"
    assert result.stop_updates == ((0, 100.0), (1, 105.0))


def test_management_preserves_original_mt5_risk_after_stop_moves_for_same_ticket():
    first_snapshot = [
        SimpleNamespace(ticket=11, price_open=100.0, sl=95.0, volume=0.01, type=0),
    ]
    evaluate_campaign_action(
        positions=first_snapshot,
        current_price=100.5,
        direction=BreakoutDirection.BULLISH,
        latest_trade_r_multiple=0.1,
        default_lot=0.01,
        add_on_lot_increment=0.01,
        max_exposure_pct=10.0,
        margin_snapshot={
            "campaign_exposure_pct": 4.0,
            "preferred_add_exposure_pct": 1.5,
            "fallback_add_exposure_pct": 1.0,
        },
        reversal_confirmed=False,
        breakeven_distance=1.5,
    )

    tightened_snapshot = [
        SimpleNamespace(ticket=11, price_open=100.0, sl=100.0, volume=0.01, type=0),
    ]
    result = evaluate_campaign_action(
        positions=tightened_snapshot,
        current_price=110.0,
        direction=BreakoutDirection.BULLISH,
        latest_trade_r_multiple=2.0,
        default_lot=0.01,
        add_on_lot_increment=0.01,
        max_exposure_pct=10.0,
        margin_snapshot={
            "campaign_exposure_pct": 4.0,
            "preferred_add_exposure_pct": 1.5,
            "fallback_add_exposure_pct": 1.0,
        },
        reversal_confirmed=False,
    )

    assert result.action == "trail_all"
    assert result.reason == "campaign_profit_lock_progression"
    assert result.stop_updates == ((0, 105.0),)


def test_management_preserves_original_mt5_risk_after_process_reload(monkeypatch, tmp_path):
    import src.strategy.management as management

    monkeypatch.setattr(
        management,
        "_INITIAL_STOP_CACHE_PATH",
        tmp_path / "initial_stop_cache.json",
        raising=False,
    )
    management._INITIAL_STOP_CACHE.clear()

    initial_position = SimpleNamespace(
        ticket=22,
        price_open=100.0,
        sl=95.0,
        volume=0.01,
        type=0,
        symbol="XAUUSD",
    )
    management.remember_position_initial_stop_loss(initial_position)

    management._INITIAL_STOP_CACHE.clear()

    tightened_snapshot = [
        SimpleNamespace(
            ticket=22,
            price_open=100.0,
            sl=100.0,
            volume=0.01,
            type=0,
            symbol="XAUUSD",
        ),
    ]
    result = management.evaluate_campaign_action(
        positions=tightened_snapshot,
        current_price=110.0,
        direction=BreakoutDirection.BULLISH,
        latest_trade_r_multiple=2.0,
        default_lot=0.01,
        add_on_lot_increment=0.01,
        max_exposure_pct=10.0,
        margin_snapshot={
            "campaign_exposure_pct": 4.0,
            "preferred_add_exposure_pct": 1.5,
            "fallback_add_exposure_pct": 1.0,
        },
        reversal_confirmed=False,
    )

    assert result.action == "trail_all"
    assert result.reason == "campaign_profit_lock_progression"
    assert result.stop_updates == ((0, 105.0),)


def test_management_blocks_add_when_campaign_exposure_exceeds_limit():
    positions = [
        SimpleNamespace(entry_price=100.0, initial_stop_loss=95.0, stop_loss=110.0, volume=0.01),
        SimpleNamespace(entry_price=105.0, initial_stop_loss=100.0, stop_loss=110.0, volume=0.01),
    ]

    result = evaluate_campaign_action(
        positions=positions,
        current_price=115.0,
        direction=BreakoutDirection.BULLISH,
        latest_trade_r_multiple=2.0,
        default_lot=0.01,
        add_on_lot_increment=0.01,
        max_exposure_pct=10.0,
        margin_snapshot={
            "campaign_exposure_pct": 9.8,
            "preferred_add_exposure_pct": 0.5,
            "fallback_add_exposure_pct": 0.3,
        },
        reversal_confirmed=False,
    )

    assert result.action == "hold"
    assert result.reason == "campaign_exposure_limit_reached"


def test_management_uses_fallback_lot_when_preferred_add_exposure_exceeds_limit():
    positions = [
        SimpleNamespace(entry_price=100.0, initial_stop_loss=95.0, stop_loss=110.0, volume=0.01),
        SimpleNamespace(entry_price=105.0, initial_stop_loss=100.0, stop_loss=110.0, volume=0.01),
    ]

    result = evaluate_campaign_action(
        positions=positions,
        current_price=115.0,
        direction=BreakoutDirection.BULLISH,
        latest_trade_r_multiple=2.0,
        default_lot=0.01,
        add_on_lot_increment=0.01,
        max_exposure_pct=10.0,
        margin_snapshot={
            "campaign_exposure_pct": 8.8,
            "preferred_add_exposure_pct": 1.5,
            "fallback_add_exposure_pct": 0.7,
        },
        reversal_confirmed=False,
    )

    assert result.action == "add_position"
    assert result.reason == "campaign_add_ready"
    assert result.add_lot == 0.01


def test_management_closes_all_when_reversal_is_confirmed():
    positions = [
        SimpleNamespace(entry_price=100.0, initial_stop_loss=95.0, stop_loss=101.0, volume=0.01),
    ]

    result = evaluate_campaign_action(
        positions=positions,
        current_price=98.0,
        direction=BreakoutDirection.BULLISH,
        latest_trade_r_multiple=-0.4,
        default_lot=0.01,
        add_on_lot_increment=0.01,
        max_exposure_pct=10.0,
        margin_snapshot={
            "campaign_exposure_pct": 2.0,
            "preferred_add_exposure_pct": 1.0,
            "fallback_add_exposure_pct": 0.5,
        },
        reversal_confirmed=True,
    )

    assert result.action == "close_all"
    assert result.reason == "reversal_confirmed_exit"


def test_management_defers_continuation_edge_filtering_to_campaign_add_engine():
    positions = [
        SimpleNamespace(entry_price=100.0, initial_stop_loss=95.0, stop_loss=110.0, volume=0.01),
        SimpleNamespace(entry_price=105.0, initial_stop_loss=100.0, stop_loss=110.0, volume=0.01),
    ]

    result = evaluate_campaign_action(
        positions=positions,
        current_price=115.0,
        direction=BreakoutDirection.BULLISH,
        latest_trade_r_multiple=2.0,
        default_lot=0.01,
        add_on_lot_increment=0.01,
        max_exposure_pct=10.0,
        margin_snapshot={
            "campaign_exposure_pct": 6.0,
            "preferred_add_exposure_pct": 2.0,
            "fallback_add_exposure_pct": 1.0,
        },
        reversal_confirmed=False,
        continuation_edge=1.8,
        continuation_threshold=2.4,
    )

    assert result.action == "add_position"
    assert result.reason == "campaign_add_ready"
    assert result.metadata["continuation_edge"] == 1.8
    assert result.metadata["continuation_threshold"] == 2.4
