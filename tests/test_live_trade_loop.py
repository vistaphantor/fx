from types import SimpleNamespace
from datetime import datetime, timezone
import json

import pytest


@pytest.fixture(autouse=True)
def isolated_initial_stop_cache(monkeypatch, tmp_path):
    import src.strategy.management as management

    monkeypatch.setattr(management, "_INITIAL_STOP_CACHE_PATH", tmp_path / "initial_stop_cache.json")
    management._INITIAL_STOP_CACHE.clear()
    yield
    management._INITIAL_STOP_CACHE.clear()


def test_live_loop_opens_entry_when_decision_tree_returns_trade(monkeypatch):
    from src.live_trade_loop import run_live_signal_loop
    from src.strategy.breakout import BreakoutDirection
    from src.strategy.decision_tree import TopDownTradePlan

    events = []
    live_input = SimpleNamespace(
        d1_candles=["d1"],
        h4_candles=["h4"],
        h1_candles=["h1"],
        m30_candles=["m30"],
        m15_candles=[SimpleNamespace(close=2350.0)],
    )
    trade_plan = TopDownTradePlan(
        is_trade=True,
        direction=BreakoutDirection.BULLISH,
        entry_price=2350.0,
        stop_loss=2345.0,
        take_profit=2365.0,
        objective_price=2360.0,
        reason="top_down_trade_plan_ready",
        metadata={},
    )

    class FakeExecutor:
        def list_bot_positions(self, symbol):
            return []

        def open_strategy_trade(self, **kwargs):
            events.append(("open", kwargs))
            return SimpleNamespace(ticket=99)

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr("src.live_trade_loop.evaluate_top_down_decision_tree", lambda **kwargs: trade_plan)

    run_live_signal_loop(
        mt5_module=SimpleNamespace(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(("log", message)),
    )

    assert events[0][0] == "open"
    assert events[0][1]["symbol"] == "XAUUSD"
    assert events[0][1]["direction"] is BreakoutDirection.BULLISH
    assert events[0][1]["lot"] == 0.01
    assert ("log", "LIVE TRADE OPENED XAUUSD ticket=99 direction=BULLISH entry=2350.0 sl=2345.0 tp=2365.0") in events


def test_live_loop_writes_standard_mode_training_snapshot(monkeypatch):
    from src.live_trade_loop import run_live_signal_loop
    from src.strategy.breakout import BreakoutDirection
    from src.strategy.decision_tree import TopDownNoTrade

    snapshots = []
    candle = SimpleNamespace(
        timestamp=datetime(2026, 5, 11, 9, 15, tzinfo=timezone.utc),
        open=2348.0,
        high=2352.0,
        low=2347.5,
        close=2350.0,
    )
    live_input = SimpleNamespace(
        d1_candles=[candle],
        h4_candles=[candle],
        h1_candles=[candle],
        m30_candles=[candle],
        m15_candles=[candle],
        spread=0.25,
        tick_data={"bid": 2349.9, "ask": 2350.15},
    )
    no_trade = TopDownNoTrade(
        is_trade=False,
        reason="m15_setup_not_ready",
        failed_node="m15_setup",
        metadata={},
    )

    class FakeMt5:
        def account_info(self):
            return SimpleNamespace(balance=3660.67, equity=3601.25, profit=-59.42, currency="KES")

    class FakeExecutor:
        def list_bot_positions(self, symbol):
            return []

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr("src.live_trade_loop.evaluate_top_down_decision_tree", lambda **kwargs: no_trade)
    monkeypatch.setattr("src.live_trade_loop.save_training_snapshot", lambda state: snapshots.append(state))

    run_live_signal_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: None,
    )

    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot["trading"]["strategy_mode"] == "standard_live"
    assert snapshot["trading"]["candle_timeframe"] == "M15"
    assert snapshot["trading"]["decision_reason"] == "m15_setup_not_ready"
    assert snapshot["trading"]["failed_node"] == "m15_setup"
    assert snapshot["trading"]["planned_direction"] == "None"
    assert snapshot["market_data"]["m1_candles"][-1]["close"] == 2350.0

    from src.quick_scalp_loop import build_training_snapshot_row

    row = build_training_snapshot_row(snapshot)
    assert row["strategy_mode"] == "standard_live"
    assert row["training_version"] == "standard-live-2026-W20"


def test_live_loop_passes_orderflow_signal_to_strategy_and_quant(monkeypatch):
    from src.live_trade_loop import run_live_signal_loop
    from src.strategy.decision_tree import TopDownNoTrade
    from src.strategy.features import FeatureSnapshot
    from src.strategy.orderflow import parse_orderflow_payload
    from src.strategy.quant_engine import QuantDecision

    captured = {}
    signal = parse_orderflow_payload(
        {
            "symbol": "GC",
            "target_symbol": "XAUUSD",
            "delta": -900,
            "cvd_slope": -0.6,
            "imbalance": "sell_stacked",
            "vwap_bias": "below",
        }
    )
    candle = SimpleNamespace(
        timestamp=datetime(2026, 5, 13, 9, 15, tzinfo=timezone.utc),
        open=2350.0,
        high=2351.0,
        low=2348.0,
        close=2349.0,
    )
    live_input = SimpleNamespace(
        d1_candles=[candle],
        h4_candles=[candle],
        h1_candles=[candle],
        m30_candles=[candle],
        m15_candles=[candle],
        spread=0.25,
        tick_data={},
    )
    settings = SimpleNamespace(
        quant_enabled=True,
        quant_gamma=2.0,
        quant_cvar_alpha=0.05,
        quant_cvar_eta=1.2,
        quant_dd_max=0.20,
        quant_dd_rho=0.5,
        quant_omega_threshold=0.45,
        quant_position_r_max=0.05,
        quant_transaction_lambda=1.0,
        quant_zscore_window=100,
        ml_enabled=False,
        feature_logging_enabled=False,
    )

    class FakeOrderflowStore:
        def latest_for(self, symbol, now=None):
            captured["store_symbol"] = symbol
            captured["store_now"] = now
            return signal

    class FakeEquityTracker:
        drawdown_ratio = 0.0

        def update(self, equity):
            return None

    class FakeExecutor:
        def list_bot_positions(self, symbol):
            return []

    def fake_tree(**kwargs):
        captured["tree_orderflow"] = kwargs["orderflow_signal"]
        return TopDownNoTrade(False, "m30_orderflow_conflict", "m30_setup", {})

    def fake_extract(**kwargs):
        captured["extract_orderflow"] = kwargs["orderflow_signal"]
        return (
            FeatureSnapshot(
                timestamp=candle.timestamp,
                momentum_raw=0.0,
                trend_raw=0.0,
                volume_raw=1.0,
                order_block_raw=0.0,
                volatility_risk_raw=1.0,
                entry_distance_raw=0.0,
                spread_danger_raw=0.0,
                momentum_z=0.0,
                trend_z=0.0,
                volume_z=0.0,
                order_block_z=0.0,
                volatility_risk_z=0.0,
                entry_distance_z=0.0,
                spread_danger_z=0.0,
                expected_return=0.0,
                return_std=1.0,
            ),
            0.0,
            1.0,
        )

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr("src.live_trade_loop.evaluate_top_down_decision_tree", fake_tree)
    monkeypatch.setattr("src.live_trade_loop._build_feature_extractor", lambda settings: SimpleNamespace(snapshot_count=5))
    monkeypatch.setattr("src.live_trade_loop._build_equity_tracker", lambda settings: FakeEquityTracker())
    monkeypatch.setattr("src.live_trade_loop._estimate_strategy_trade_statistics", lambda **kwargs: {
        "win_rate": 0.5,
        "avg_win": 1.0,
        "avg_loss": 1.0,
        "expected_return": 0.0,
        "return_std": 1.0,
        "recent_returns": [],
        "transaction_cost": 0.0,
        "continuation_context": None,
    })
    monkeypatch.setattr("src.live_trade_loop._extract_features_from_strategy", fake_extract)
    monkeypatch.setattr("src.live_trade_loop.save_training_snapshot", lambda state: None)
    monkeypatch.setattr(
        "src.strategy.quant_engine.evaluate_master_equation",
        lambda **kwargs: QuantDecision(
            action=0,
            omega_t=0.0,
            position_size_fraction=0.0,
            expected_return=0.0,
            cvar=0.0,
            drawdown_ratio=0.0,
            drawdown_dampener=1.0,
            utility_scores={-1: 0.0, 0: 0.0, 1: 0.0},
            sharpe_signal=0.0,
            reason="master_equation_flat",
            is_trade=False,
            metadata={},
        ),
    )

    run_live_signal_loop(
        mt5_module=SimpleNamespace(account_info=lambda: SimpleNamespace(equity=10000.0)),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        orderflow_signal_store=FakeOrderflowStore(),
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: None,
        settings=settings,
    )

    assert captured["store_symbol"] == "XAUUSD"
    assert captured["store_now"] == candle.timestamp
    assert captured["tree_orderflow"] is signal
    assert captured["extract_orderflow"] is signal


def test_live_loop_logs_order_rejection_without_crashing(monkeypatch):
    from src.live_trade_loop import run_live_signal_loop
    from src.strategy.breakout import BreakoutDirection
    from src.strategy.decision_tree import TopDownTradePlan

    events = []
    live_input = SimpleNamespace(
        d1_candles=["d1"],
        h4_candles=["h4"],
        h1_candles=["h1"],
        m30_candles=["m30"],
        m15_candles=[SimpleNamespace(close=2350.0)],
    )
    trade_plan = TopDownTradePlan(
        is_trade=True,
        direction=BreakoutDirection.BULLISH,
        entry_price=2350.0,
        stop_loss=2345.0,
        take_profit=2365.0,
        objective_price=2360.0,
        reason="top_down_trade_plan_ready",
        metadata={},
    )

    class FakeExecutor:
        def list_bot_positions(self, symbol):
            return []

        def open_strategy_trade(self, **kwargs):
            raise RuntimeError("Strategy trade rejected: unsupported filling mode for this symbol/account")

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr("src.live_trade_loop.evaluate_top_down_decision_tree", lambda **kwargs: trade_plan)

    run_live_signal_loop(
        mt5_module=SimpleNamespace(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(("log", message)),
    )

    assert (
        "log",
        "LIVE ORDER REJECTED XAUUSD reason=Strategy trade rejected: unsupported filling mode for this symbol/account",
    ) in events


def test_live_loop_opens_strong_continuation_trade_below_legacy_rr_floor(monkeypatch):
    from src.live_trade_loop import run_live_signal_loop
    from src.strategy.breakout import BreakoutDirection
    from src.strategy.decision_tree import TopDownTradePlan
    from src.strategy.volatility import VolatilityState

    events = []
    live_input = SimpleNamespace(
        d1_candles=["d1"],
        h4_candles=["h4"],
        h1_candles=["h1"],
        m30_candles=["m30"],
        m15_candles=[SimpleNamespace(close=100.0)],
        spread=0.04,
        tick_data={"bid": 99.98, "ask": 100.02},
    )
    trade_plan = TopDownTradePlan(
        is_trade=True,
        direction=BreakoutDirection.BULLISH,
        entry_price=100.0,
        stop_loss=98.7,
        take_profit=101.5,
        objective_price=102.0,
        reason="top_down_trade_plan_ready",
        metadata={
            "is_continuation_setup": True,
            "m15_quality": 0.95,
            "m10_quality": 0.92,
            "m5_quality": 0.90,
            "range_expansion_ratio": 1.25,
            "body_efficiency": 0.82,
            "regime_confidence": 0.90,
            "volatility_state": VolatilityState(
                short_atr=0.35,
                medium_atr=0.45,
                realized_range=1.0,
                body_efficiency=0.82,
                range_expansion_ratio=1.25,
            ),
        },
    )

    class FakeExecutor:
        def list_bot_positions(self, symbol):
            return []

        def open_strategy_trade(self, **kwargs):
            events.append(("open", kwargs))
            return SimpleNamespace(ticket=101)

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr("src.live_trade_loop.evaluate_top_down_decision_tree", lambda **kwargs: trade_plan)

    run_live_signal_loop(
        mt5_module=SimpleNamespace(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(("log", message)),
    )

    assert events[0][0] == "open"
    assert events[0][1]["symbol"] == "XAUUSD"


def test_live_loop_trails_existing_campaign_before_opening_new_trade(monkeypatch):
    from src.live_trade_loop import run_live_signal_loop
    from src.strategy.breakout import BreakoutDirection
    from src.strategy.decision_tree import TopDownTradePlan
    from src.strategy.management import CampaignAction

    events = []
    positions = [
        SimpleNamespace(ticket=1, entry_price=100.0, initial_stop_loss=95.0, stop_loss=98.0, symbol="XAUUSD", tp=120.0),
        SimpleNamespace(ticket=2, entry_price=105.0, initial_stop_loss=101.0, stop_loss=103.0, symbol="XAUUSD", tp=125.0),
    ]
    live_input = SimpleNamespace(
        d1_candles=["d1"],
        h4_candles=["h4"],
        h1_candles=["h1"],
        m30_candles=["m30"],
        m15_candles=[SimpleNamespace(close=109.0)],
    )
    trade_plan = TopDownTradePlan(
        is_trade=True,
        direction=BreakoutDirection.BULLISH,
        entry_price=109.0,
        stop_loss=104.0,
        take_profit=124.0,
        objective_price=120.0,
        reason="top_down_trade_plan_ready",
        metadata={},
    )
    campaign_action = CampaignAction(
        action="trail_all",
        reason="campaign_trail_progression",
        new_stop_loss=106.0,
        metadata={},
    )

    class FakeExecutor:
        def list_bot_positions(self, symbol):
            return positions

        def update_position_stop_loss(self, position, stop_loss, take_profit=None):
            events.append(("trail", position.ticket, stop_loss, take_profit))

        def open_strategy_trade(self, **kwargs):
            raise AssertionError("should not open a new trade while trailing the active campaign")

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr("src.live_trade_loop.evaluate_top_down_decision_tree", lambda **kwargs: trade_plan)
    monkeypatch.setattr("src.live_trade_loop.evaluate_campaign_action", lambda **kwargs: campaign_action)

    run_live_signal_loop(
        mt5_module=SimpleNamespace(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(("log", message)),
    )

    assert ("trail", 1, 106.0, None) in events
    assert ("trail", 2, 106.0, None) in events


def test_live_loop_logs_campaign_breakeven_earned_with_indexed_stop_updates(monkeypatch):
    from src.live_trade_loop import run_live_signal_loop
    from src.strategy.breakout import BreakoutDirection
    from src.strategy.decision_tree import TopDownTradePlan
    from src.strategy.management import CampaignAction

    events = []
    positions = [
        SimpleNamespace(ticket=1, entry_price=100.0, initial_stop_loss=95.0, stop_loss=98.0, symbol="XAUUSD", tp=120.0),
        SimpleNamespace(ticket=2, entry_price=105.0, initial_stop_loss=101.0, stop_loss=103.0, symbol="XAUUSD", tp=125.0),
    ]
    live_input = SimpleNamespace(
        d1_candles=["d1"],
        h4_candles=["h4"],
        h1_candles=["h1"],
        m30_candles=["m30"],
        m15_candles=[SimpleNamespace(close=106.5)],
    )
    trade_plan = TopDownTradePlan(
        is_trade=True,
        direction=BreakoutDirection.BULLISH,
        entry_price=106.5,
        stop_loss=102.0,
        take_profit=119.0,
        objective_price=114.0,
        reason="top_down_trade_plan_ready",
        metadata={},
    )
    campaign_action = CampaignAction(
        action="trail_all",
        reason="campaign_breakeven_earned",
        stop_updates=((0, 100.0), (1, 105.0)),
        metadata={},
    )

    class FakeExecutor:
        def list_bot_positions(self, symbol):
            return positions

        def update_position_stop_loss(self, position, stop_loss, take_profit=None):
            events.append(("trail", position.ticket, stop_loss, take_profit))

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr("src.live_trade_loop.evaluate_top_down_decision_tree", lambda **kwargs: trade_plan)
    monkeypatch.setattr("src.live_trade_loop.evaluate_campaign_action", lambda **kwargs: campaign_action)

    run_live_signal_loop(
        mt5_module=SimpleNamespace(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(("log", message)),
    )

    assert ("log", "LIVE CAMPAIGN TRAIL XAUUSD positions=2 updated=2 reason=campaign_breakeven_earned new_sl=105.0") in events


def test_live_loop_adds_position_when_campaign_action_allows(monkeypatch):
    from src.live_trade_loop import run_live_signal_loop
    from src.strategy.breakout import BreakoutDirection
    from src.strategy.campaign_add import CampaignAddDecision
    from src.strategy.decision_tree import TopDownNoTrade, TopDownTradePlan
    from src.strategy.management import CampaignAction

    events = []
    positions = [
        SimpleNamespace(ticket=1, entry_price=100.0, initial_stop_loss=95.0, stop_loss=101.0, symbol="XAUUSD", tp=120.0),
        SimpleNamespace(ticket=2, entry_price=105.0, initial_stop_loss=100.0, stop_loss=106.0, symbol="XAUUSD", tp=125.0),
    ]
    live_input = SimpleNamespace(
        d1_candles=["d1"],
        h4_candles=["h4"],
        h1_candles=["h1"],
        m30_candles=["m30"],
        m15_candles=[SimpleNamespace(close=115.0)],
    )
    trade_plan = TopDownNoTrade(
        is_trade=False,
        reason="m10_setup_not_ready",
        failed_node="m10_setup",
        metadata={},
    )
    campaign_action = CampaignAction(
        action="add_position",
        reason="campaign_add_ready",
        add_lot=0.02,
        metadata={},
    )

    class FakeExecutor:
        def list_bot_positions(self, symbol):
            return positions

        def update_position_stop_loss(self, position, stop_loss, take_profit=None):
            raise AssertionError("should not trail when the campaign action is add_position")

        def open_strategy_trade(self, **kwargs):
            events.append(("open", kwargs))
            return SimpleNamespace(ticket=3)

    captured_management_kwargs = {}

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr("src.live_trade_loop.evaluate_top_down_decision_tree", lambda **kwargs: trade_plan)
    monkeypatch.setattr(
        "src.live_trade_loop.evaluate_campaign_add",
        lambda **kwargs: CampaignAddDecision(
            is_ready=True,
            reason="campaign_add_ready",
            direction=BreakoutDirection.BULLISH,
            entry_price=115.0,
            stop_loss=110.0,
            take_profit=130.0,
            metadata={},
            quality_score=0.8,
        ),
    )
    monkeypatch.setattr(
        "src.live_trade_loop._assess_live_execution",
        lambda **kwargs: SimpleNamespace(
            is_tradeable=True,
            recommended_lot_multiplier=1.0,
            reason="execution_ok",
            effective_rr=2.0,
        ),
    )
    monkeypatch.setattr(
        "src.live_trade_loop.evaluate_campaign_action",
        lambda **kwargs: captured_management_kwargs.update(kwargs) or campaign_action,
    )

    run_live_signal_loop(
        mt5_module=SimpleNamespace(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        add_on_lot_increment=0.03,
        campaign_max_exposure_pct=8.5,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(("log", message)),
    )

    assert events[0][0] == "open"
    assert events[0][1]["lot"] == 0.02
    assert events[0][1]["stop_loss"] == 110.0
    assert events[0][1]["take_profit"] == 130.0
    assert captured_management_kwargs["add_on_lot_increment"] == 0.03
    assert captured_management_kwargs["max_exposure_pct"] == 8.5


def test_live_loop_reduces_campaign_add_when_quant_engine_is_flat_but_campaign_add_allows_reduced_size(monkeypatch):
    from src.live_trade_loop import run_live_signal_loop
    from src.strategy.breakout import BreakoutDirection
    from src.strategy.campaign_add import CampaignAddDecision
    from src.strategy.decision_tree import TopDownNoTrade
    from src.strategy.management import CampaignAction

    events = []
    positions = [
        SimpleNamespace(ticket=1, entry_price=100.0, initial_stop_loss=95.0, stop_loss=101.0, symbol="XAUUSD", tp=120.0),
    ]
    live_input = SimpleNamespace(
        d1_candles=["d1"],
        h4_candles=["h4"],
        h1_candles=[SimpleNamespace(open=100.0, close=101.0)],
        m30_candles=[SimpleNamespace(open=100.0, close=101.0)],
        m15_candles=[SimpleNamespace(close=115.0)],
    )
    quant_blocked = TopDownNoTrade(
        is_trade=False,
        reason="master_equation_flat",
        failed_node="quant_engine",
        metadata={"quant_decision": SimpleNamespace(action=0, reason="master_equation_flat")},
    )
    campaign_action = CampaignAction(
        action="add_position",
        reason="campaign_add_ready",
        add_lot=0.02,
        metadata={},
    )

    class FakeExecutor:
        def list_bot_positions(self, symbol):
            return positions

        def open_strategy_trade(self, **kwargs):
            events.append(("open", kwargs))
            return SimpleNamespace(ticket=2)

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr("src.live_trade_loop.evaluate_top_down_decision_tree", lambda **kwargs: quant_blocked)
    monkeypatch.setattr("src.live_trade_loop.evaluate_campaign_action", lambda **kwargs: campaign_action)
    monkeypatch.setattr(
        "src.live_trade_loop.evaluate_campaign_add",
        lambda **kwargs: CampaignAddDecision(
            is_ready=True,
            reason="campaign_add_quant_reduced",
            direction=BreakoutDirection.BULLISH,
            entry_price=115.0,
            stop_loss=110.0,
            take_profit=130.0,
            metadata={},
            quality_score=0.82,
            lot_multiplier=0.5,
            quant_state="soft_reduce",
        ),
    )
    monkeypatch.setattr(
        "src.live_trade_loop._assess_live_execution",
        lambda **kwargs: SimpleNamespace(
            is_tradeable=True,
            recommended_lot_multiplier=1.0,
            reason="execution_ok",
            effective_rr=2.0,
        ),
    )

    run_live_signal_loop(
        mt5_module=SimpleNamespace(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        add_on_lot_increment=0.03,
        campaign_max_exposure_pct=8.5,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(("log", message)),
    )

    assert events[0][0] == "open"
    assert events[0][1]["lot"] == 0.01
    assert (
        "log",
        "LIVE CAMPAIGN ADD XAUUSD ticket=2 direction=BULLISH entry=115.0 sl=110.0 tp=130.0 lot=0.01 reason=campaign_add_quant_reduced",
    ) in events


def test_live_loop_holds_campaign_add_when_campaign_add_engine_returns_hard_quant_block(monkeypatch):
    from src.live_trade_loop import run_live_signal_loop
    from src.strategy.campaign_add import CampaignAddDecision
    from src.strategy.decision_tree import TopDownNoTrade
    from src.strategy.management import CampaignAction

    events = []
    positions = [
        SimpleNamespace(ticket=1, entry_price=100.0, initial_stop_loss=95.0, stop_loss=101.0, symbol="XAUUSD", tp=120.0),
    ]
    live_input = SimpleNamespace(
        d1_candles=["d1"],
        h4_candles=["h4"],
        h1_candles=[SimpleNamespace(open=100.0, close=101.0)],
        m30_candles=[SimpleNamespace(open=100.0, close=101.0)],
        m15_candles=[SimpleNamespace(close=115.0)],
    )
    quant_blocked = TopDownNoTrade(
        is_trade=False,
        reason="master_equation_flat",
        failed_node="quant_engine",
        metadata={"quant_decision": SimpleNamespace(action=1, reason="master_equation_long_approved")},
    )
    campaign_action = CampaignAction(
        action="add_position",
        reason="campaign_add_ready",
        add_lot=0.02,
        metadata={},
    )

    class FakeExecutor:
        def list_bot_positions(self, symbol):
            return positions

        def open_strategy_trade(self, **kwargs):
            raise AssertionError("should not add when campaign add engine hard-blocks quant")

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr("src.live_trade_loop.evaluate_top_down_decision_tree", lambda **kwargs: quant_blocked)
    monkeypatch.setattr("src.live_trade_loop.evaluate_campaign_action", lambda **kwargs: campaign_action)
    monkeypatch.setattr(
        "src.live_trade_loop.evaluate_campaign_add",
        lambda **kwargs: CampaignAddDecision(
            is_ready=False,
            reason="campaign_add_quant_blocked",
            direction=None,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            metadata={},
            quality_score=0.0,
            quant_state="hard_block",
        ),
    )

    run_live_signal_loop(
        mt5_module=SimpleNamespace(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        add_on_lot_increment=0.03,
        campaign_max_exposure_pct=8.5,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(("log", message)),
    )

    assert ("log", "LIVE CAMPAIGN HOLD XAUUSD positions=1 reason=campaign_add_quant_blocked") in events


def test_live_loop_passes_quant_edge_into_campaign_management(monkeypatch):
    from src.live_trade_loop import run_live_signal_loop
    from src.strategy.breakout import BreakoutDirection
    from src.strategy.decision_tree import TopDownTradePlan
    from src.strategy.scoring import ScoreDecision, SideScore

    events = []
    positions = [
        SimpleNamespace(ticket=1, entry_price=100.0, initial_stop_loss=95.0, stop_loss=101.0, symbol="XAUUSD", tp=120.0),
        SimpleNamespace(ticket=2, entry_price=105.0, initial_stop_loss=100.0, stop_loss=106.0, symbol="XAUUSD", tp=125.0),
    ]
    live_input = SimpleNamespace(
        d1_candles=["d1"],
        h4_candles=["h4"],
        h1_candles=["h1"],
        m30_candles=["m30"],
        m15_candles=[SimpleNamespace(close=115.0)],
    )
    trade_plan = TopDownTradePlan(
        is_trade=True,
        direction=BreakoutDirection.BULLISH,
        entry_price=115.0,
        stop_loss=110.0,
        take_profit=130.0,
        objective_price=126.0,
        reason="top_down_trade_plan_ready",
        metadata={
            "score_decision": ScoreDecision(
                bullish=SideScore(1, 1, 1, 1, 0.2, 0.0, 0.8, 4.0),
                bearish=SideScore(0.8, 0.7, 0.5, 0.4, 0.0, 0.0, 0.8, 2.2),
                uncertainty_penalty=0.2,
                edge=1.8,
                threshold=1.5,
                expected_move_multiple=2.6,
                is_tradeable=True,
            )
        },
    )

    class FakeExecutor:
        def list_bot_positions(self, symbol):
            return positions

    captured_management_kwargs = {}

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr("src.live_trade_loop.evaluate_top_down_decision_tree", lambda **kwargs: trade_plan)
    monkeypatch.setattr(
        "src.live_trade_loop.evaluate_campaign_action",
        lambda **kwargs: captured_management_kwargs.update(kwargs)
        or SimpleNamespace(action="hold", reason="campaign_edge_below_add_threshold", metadata={}),
    )

    run_live_signal_loop(
        mt5_module=SimpleNamespace(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        add_on_lot_increment=0.03,
        campaign_max_exposure_pct=8.5,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(("log", message)),
    )

    assert captured_management_kwargs["continuation_edge"] == 1.8
    assert captured_management_kwargs["continuation_threshold"] == 1.5
    assert ("log", "LIVE CAMPAIGN HOLD XAUUSD positions=2 reason=campaign_edge_below_add_threshold") in events


def test_live_loop_does_not_loosen_existing_stop_losses_when_trailing(monkeypatch):
    from src.live_trade_loop import run_live_signal_loop
    from src.strategy.breakout import BreakoutDirection
    from src.strategy.decision_tree import TopDownTradePlan
    from src.strategy.management import CampaignAction

    events = []
    positions = [
        SimpleNamespace(
            ticket=1,
            entry_price=100.0,
            initial_stop_loss=95.0,
            stop_loss=108.0,
            symbol="XAUUSD",
            tp=120.0,
        ),
        SimpleNamespace(
            ticket=2,
            entry_price=105.0,
            initial_stop_loss=101.0,
            stop_loss=103.0,
            symbol="XAUUSD",
            tp=125.0,
        ),
    ]
    live_input = SimpleNamespace(
        d1_candles=["d1"],
        h4_candles=["h4"],
        h1_candles=["h1"],
        m30_candles=["m30"],
        m15_candles=[SimpleNamespace(close=109.0)],
    )
    trade_plan = TopDownTradePlan(
        is_trade=True,
        direction=BreakoutDirection.BULLISH,
        entry_price=109.0,
        stop_loss=104.0,
        take_profit=124.0,
        objective_price=120.0,
        reason="top_down_trade_plan_ready",
        metadata={},
    )
    campaign_action = CampaignAction(
        action="trail_all",
        reason="campaign_trail_progression",
        new_stop_loss=106.0,
        metadata={},
    )

    class FakeExecutor:
        def list_bot_positions(self, symbol):
            return positions

        def update_position_stop_loss(self, position, stop_loss, take_profit=None):
            events.append(("trail", position.ticket, stop_loss, take_profit))

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr("src.live_trade_loop.evaluate_top_down_decision_tree", lambda **kwargs: trade_plan)
    monkeypatch.setattr("src.live_trade_loop.evaluate_campaign_action", lambda **kwargs: campaign_action)

    run_live_signal_loop(
        mt5_module=SimpleNamespace(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(("log", message)),
    )

    assert ("trail", 1, 106.0, None) not in events
    assert ("trail", 2, 106.0, None) in events


def test_live_loop_passes_real_margin_snapshot_to_campaign_management(monkeypatch):
    from src.live_trade_loop import run_live_signal_loop
    from src.strategy.breakout import BreakoutDirection
    from src.strategy.decision_tree import TopDownTradePlan
    from src.strategy.management import CampaignAction

    captured_management_kwargs = {}
    positions = [
        SimpleNamespace(
            ticket=1,
            entry_price=100.0,
            initial_stop_loss=95.0,
            stop_loss=101.0,
            symbol="XAUUSD",
            tp=120.0,
            volume=0.02,
            type=0,
        ),
        SimpleNamespace(
            ticket=2,
            entry_price=105.0,
            initial_stop_loss=100.0,
            stop_loss=106.0,
            symbol="XAUUSD",
            tp=125.0,
            volume=0.01,
            type=0,
        ),
    ]
    live_input = SimpleNamespace(
        d1_candles=["d1"],
        h4_candles=["h4"],
        h1_candles=["h1"],
        m30_candles=["m30"],
        m15_candles=[SimpleNamespace(close=115.0)],
    )
    trade_plan = TopDownTradePlan(
        is_trade=True,
        direction=BreakoutDirection.BULLISH,
        entry_price=115.0,
        stop_loss=110.0,
        take_profit=130.0,
        objective_price=126.0,
        reason="top_down_trade_plan_ready",
        metadata={},
    )

    class FakeMt5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1

        def account_info(self):
            return SimpleNamespace(equity=1000.0)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(ask=115.2, bid=114.8)

        def order_calc_margin(self, order_type, symbol, volume, price):
            return 100.0 * float(volume)

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol):
            return positions

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr("src.live_trade_loop.evaluate_top_down_decision_tree", lambda **kwargs: trade_plan)
    monkeypatch.setattr(
        "src.live_trade_loop.evaluate_campaign_action",
        lambda **kwargs: captured_management_kwargs.update(kwargs)
        or CampaignAction(action="hold", reason="campaign_waiting_for_progress", metadata={}),
    )

    run_live_signal_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        add_on_lot_increment=0.01,
        campaign_max_exposure_pct=10.0,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: None,
    )

    margin_snapshot = captured_management_kwargs["margin_snapshot"]
    assert margin_snapshot["campaign_exposure_pct"] == 0.3
    assert margin_snapshot["preferred_add_exposure_pct"] == 0.2
    assert margin_snapshot["fallback_add_exposure_pct"] == 0.1


def test_live_loop_closes_campaign_when_multi_timeframe_reversal_is_confirmed(monkeypatch):
    from src.live_trade_loop import run_live_signal_loop
    from src.strategy.decision_tree import TopDownNoTrade

    events = []
    positions = [
        SimpleNamespace(
            ticket=1,
            entry_price=100.0,
            initial_stop_loss=95.0,
            stop_loss=103.0,
            symbol="XAUUSD",
            tp=120.0,
            volume=0.01,
        ),
    ]
    live_input = SimpleNamespace(
        d1_candles=["d1"],
        h4_candles=["h4"],
        h1_candles=[
            SimpleNamespace(open=112.0, close=114.0, high=115.0, low=111.0),
            SimpleNamespace(open=114.0, close=109.0, high=114.5, low=108.5),
        ],
        m30_candles=[
            SimpleNamespace(open=111.0, close=113.0, high=113.5, low=110.5),
            SimpleNamespace(open=113.0, close=108.0, high=113.2, low=107.8),
        ],
        m15_candles=[
            SimpleNamespace(open=110.0, close=112.0, high=112.5, low=109.5),
            SimpleNamespace(open=112.0, close=107.0, high=112.2, low=106.8),
        ],
    )
    no_trade = TopDownNoTrade(
        is_trade=False,
        reason="m15_trigger_missing",
        failed_node="m15_trigger",
        metadata={},
    )

    class FakeMt5:
        def account_info(self):
            return SimpleNamespace(equity=1000.0)

    class FakeExecutor:
        mt5_module = FakeMt5()

        def list_bot_positions(self, symbol):
            return positions

        def close_position(self, position, comment):
            events.append(("close", position.ticket, comment))

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr("src.live_trade_loop.evaluate_top_down_decision_tree", lambda **kwargs: no_trade)

    run_live_signal_loop(
        mt5_module=FakeMt5(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(("log", message)),
    )

    assert ("close", 1, "strategy-live-reversal-exit") in events


def test_live_loop_uses_open_position_direction_for_campaign_management(monkeypatch):
    from src.live_trade_loop import run_live_signal_loop
    from src.strategy.breakout import BreakoutDirection
    from src.strategy.decision_tree import TopDownTradePlan
    from src.strategy.management import CampaignAction

    captured_management_kwargs = {}
    positions = [
        SimpleNamespace(
            ticket=1,
            entry_price=100.0,
            initial_stop_loss=95.0,
            stop_loss=103.0,
            symbol="XAUUSD",
            tp=120.0,
            volume=0.01,
        ),
    ]
    live_input = SimpleNamespace(
        d1_candles=["d1"],
        h4_candles=["h4"],
        h1_candles=["h1"],
        m30_candles=["m30"],
        m15_candles=[SimpleNamespace(close=110.0)],
    )
    opposite_trade_plan = TopDownTradePlan(
        is_trade=True,
        direction=BreakoutDirection.BEARISH,
        entry_price=110.0,
        stop_loss=112.0,
        take_profit=104.0,
        objective_price=100.0,
        reason="top_down_trade_plan_ready",
        metadata={},
    )

    class FakeExecutor:
        mt5_module = SimpleNamespace(account_info=lambda: SimpleNamespace(equity=1000.0))

        def list_bot_positions(self, symbol):
            return positions

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr("src.live_trade_loop.evaluate_top_down_decision_tree", lambda **kwargs: opposite_trade_plan)
    monkeypatch.setattr(
        "src.live_trade_loop.evaluate_campaign_action",
        lambda **kwargs: captured_management_kwargs.update(kwargs)
        or CampaignAction(action="hold", reason="campaign_waiting_for_progress", metadata={}),
    )

    run_live_signal_loop(
        mt5_module=SimpleNamespace(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: None,
    )

    assert captured_management_kwargs["direction"] is BreakoutDirection.BULLISH


def test_live_loop_supports_mt5_trade_position_field_names(monkeypatch):
    from src.live_trade_loop import run_live_signal_loop
    from src.strategy.breakout import BreakoutDirection
    from src.strategy.decision_tree import TopDownTradePlan
    from src.strategy.management import CampaignAction

    captured_management_kwargs = {}
    positions = [
        SimpleNamespace(
            ticket=95856608,
            price_open=4689.24,
            sl=4678.28,
            tp=4692.08,
            type=0,
            volume=0.01,
            symbol="XAUUSD",
            comment="strategy-live",
        ),
    ]
    live_input = SimpleNamespace(
        d1_candles=["d1"],
        h4_candles=["h4"],
        h1_candles=["h1"],
        m30_candles=["m30"],
        m15_candles=[SimpleNamespace(close=4688.90)],
    )
    trade_plan = TopDownTradePlan(
        is_trade=True,
        direction=BreakoutDirection.BULLISH,
        entry_price=4688.90,
        stop_loss=4680.00,
        take_profit=4695.00,
        objective_price=4700.00,
        reason="top_down_trade_plan_ready",
        metadata={},
    )

    class FakeExecutor:
        mt5_module = SimpleNamespace(
            ORDER_TYPE_BUY=0,
            ORDER_TYPE_SELL=1,
            account_info=lambda: SimpleNamespace(equity=1000.0),
        )

        def list_bot_positions(self, symbol):
            return positions

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr("src.live_trade_loop.evaluate_top_down_decision_tree", lambda **kwargs: trade_plan)
    monkeypatch.setattr(
        "src.live_trade_loop.evaluate_campaign_action",
        lambda **kwargs: captured_management_kwargs.update(kwargs)
        or CampaignAction(action="hold", reason="campaign_waiting_for_progress", metadata={}),
    )

    run_live_signal_loop(
        mt5_module=SimpleNamespace(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: None,
    )

    assert captured_management_kwargs["direction"] is BreakoutDirection.BULLISH
    assert captured_management_kwargs["latest_trade_r_multiple"] < 0.0


def test_live_loop_returns_reload_requested_when_callback_detects_code_change(monkeypatch):
    from src.live_trade_loop import run_live_signal_loop
    from src.strategy.decision_tree import TopDownNoTrade

    live_input = SimpleNamespace(
        d1_candles=["d1"],
        h4_candles=["h4"],
        h1_candles=["h1"],
        m30_candles=["m30"],
        m15_candles=[SimpleNamespace(close=2350.0)],
    )
    no_trade = TopDownNoTrade(
        is_trade=False,
        reason="m15_trigger_missing",
        failed_node="m15_trigger",
        metadata={},
    )

    class FakeExecutor:
        def list_bot_positions(self, symbol):
            return []

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr("src.live_trade_loop.evaluate_top_down_decision_tree", lambda **kwargs: no_trade)

    result = run_live_signal_loop(
        mt5_module=SimpleNamespace(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        reload_check_fn=lambda: True,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: None,
    )

    assert result == "reload_requested"


def test_estimate_strategy_trade_statistics_uses_trade_plan_expectancy():
    from src.live_trade_loop import _estimate_strategy_trade_statistics
    from src.strategy.breakout import BreakoutDirection
    from src.strategy.decision_tree import TopDownTradePlan
    from src.strategy.regime import RegimeState
    from src.strategy.scoring import ScoreDecision, SideScore

    trade_plan = TopDownTradePlan(
        is_trade=True,
        direction=BreakoutDirection.BEARISH,
        entry_price=4700.0,
        stop_loss=4704.0,
        take_profit=4688.0,
        objective_price=4680.0,
        reason="top_down_trade_plan_ready",
        metadata={
            "score_decision": ScoreDecision(
                bullish=SideScore(0.4, 0.3, 0.2, 0.2, 0.0, 0.0, 0.8, 1.9),
                bearish=SideScore(1.2, 1.1, 1.0, 0.9, 0.3, 0.2, 0.8, 5.5),
                uncertainty_penalty=0.25,
                edge=-3.6,
                threshold=1.5,
                expected_move_multiple=3.0,
                is_tradeable=True,
                z_score_normalized_edge=-2.4,
            ),
            "regime_state": RegimeState(
                name="trend",
                tradable=True,
                continuation_bias=1.1,
                reversion_bias=0.6,
                confidence=0.84,
            ),
            "pattern_confluence_score": 1.0,
            "tradingview_confluence": SimpleNamespace(direction_bonus=0.4, setup_bonus=0.25),
        },
    )
    live_input = SimpleNamespace(
        m15_candles=[
            SimpleNamespace(open=4696.0, high=4698.0, low=4695.0, close=4697.0),
            SimpleNamespace(open=4697.0, high=4699.0, low=4696.0, close=4698.5),
            SimpleNamespace(open=4698.5, high=4699.0, low=4696.0, close=4696.5),
            SimpleNamespace(open=4696.5, high=4701.0, low=4696.0, close=4700.0),
        ],
    )

    stats = _estimate_strategy_trade_statistics(
        strategy_result=trade_plan,
        live_input=live_input,
        spread=0.35,
        requested_lot=0.01,
        campaign_exposure_pct=0.0,
    )

    assert stats["win_rate"] > 0.5
    assert stats["avg_win"] > stats["avg_loss"]
    assert stats["expected_return"] < 0.0
    assert 0.0 < stats["transaction_cost"] < 0.01
    assert stats["recent_returns"] == [
        -((4698.5 - 4697.0) / 4697.0),
        -((4696.5 - 4698.5) / 4698.5),
        -((4700.0 - 4696.5) / 4696.5),
    ]


def test_live_loop_passes_strategy_trade_stats_into_quant_engine(monkeypatch):
    from src.live_trade_loop import run_live_signal_loop
    from src.strategy.breakout import BreakoutDirection
    from src.strategy.decision_tree import TopDownTradePlan
    from src.strategy.features import FeatureSnapshot
    from src.strategy.quant_engine import QuantDecision

    captured_quant_kwargs = {}
    live_input = SimpleNamespace(
        d1_candles=["d1"],
        h4_candles=["h4"],
        h1_candles=["h1"],
        m30_candles=["m30"],
        m15_candles=[SimpleNamespace(close=4700.0, timestamp=datetime(2026, 4, 28, 6, 15, tzinfo=timezone.utc))],
        spread=0.35,
    )
    trade_plan = TopDownTradePlan(
        is_trade=True,
        direction=BreakoutDirection.BULLISH,
        entry_price=4700.0,
        stop_loss=4696.0,
        take_profit=4712.0,
        objective_price=4720.0,
        reason="top_down_trade_plan_ready",
        metadata={},
    )
    quant_trade_stats = {
        "win_rate": 0.64,
        "avg_win": 0.0024,
        "avg_loss": 0.0009,
        "expected_return": 0.0012,
        "return_std": 0.0018,
        "recent_returns": [0.0024, 0.0024, -0.0009, 0.0024, -0.0009],
        "transaction_cost": 0.00012,
        "continuation_context": {
            "is_continuation_setup": True,
            "continuation_probability": 0.78,
            "mu_cont": 0.0092,
            "cvar_dir": 0.0024,
            "effective_rr": 1.08,
            "execution_penalty": 0.11,
            "directional_tail_proxy": 0.0024,
        },
    }

    class FakeEquityTracker:
        drawdown_ratio = 0.0

        def update(self, equity):
            return None

    class FakeExecutor:
        def list_bot_positions(self, symbol):
            return []

        def open_strategy_trade(self, **kwargs):
            return SimpleNamespace(ticket=123)

    settings = SimpleNamespace(
        quant_enabled=True,
        quant_gamma=2.0,
        quant_cvar_alpha=0.05,
        quant_cvar_eta=1.5,
        quant_dd_max=0.20,
        quant_dd_rho=0.5,
        quant_omega_threshold=0.5,
        quant_position_r_max=0.02,
        quant_transaction_lambda=1.0,
        quant_zscore_window=100,
        ml_enabled=False,
        feature_logging_enabled=False,
    )

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr("src.live_trade_loop.evaluate_top_down_decision_tree", lambda **kwargs: trade_plan)
    monkeypatch.setattr("src.live_trade_loop._build_feature_extractor", lambda settings: SimpleNamespace(snapshot_count=5))
    monkeypatch.setattr("src.live_trade_loop._build_equity_tracker", lambda settings: FakeEquityTracker())
    monkeypatch.setattr("src.live_trade_loop._estimate_strategy_trade_statistics", lambda **kwargs: quant_trade_stats)
    monkeypatch.setattr(
        "src.live_trade_loop._extract_features_from_strategy",
        lambda **kwargs: (
            FeatureSnapshot(
                timestamp=datetime(2026, 4, 28, 6, 15, tzinfo=timezone.utc),
                momentum_raw=1.0,
                trend_raw=1.0,
                volume_raw=1.0,
                order_block_raw=1.0,
                volatility_risk_raw=1.0,
                entry_distance_raw=0.2,
                spread_danger_raw=0.1,
                momentum_z=1.0,
                trend_z=1.0,
                volume_z=0.5,
                order_block_z=0.8,
                volatility_risk_z=0.1,
                entry_distance_z=-0.4,
                spread_danger_z=0.2,
                expected_return=quant_trade_stats["expected_return"],
                return_std=quant_trade_stats["return_std"],
            ),
            quant_trade_stats["expected_return"],
            quant_trade_stats["return_std"],
        ),
    )

    def fake_evaluate_master_equation(**kwargs):
        captured_quant_kwargs.update(kwargs)
        return QuantDecision(
            action=1,
            omega_t=0.82,
            position_size_fraction=0.01,
            expected_return=kwargs["features"].expected_return,
            cvar=0.0009,
            drawdown_ratio=kwargs["drawdown_ratio"],
            drawdown_dampener=1.0,
            utility_scores={-1: -0.4, 0: -0.2, 1: -0.1},
            sharpe_signal=0.66,
            reason="master_equation_long_approved",
            is_trade=True,
            metadata={},
        )

    monkeypatch.setattr("src.strategy.quant_engine.evaluate_master_equation", fake_evaluate_master_equation)

    run_live_signal_loop(
        mt5_module=SimpleNamespace(account_info=lambda: SimpleNamespace(equity=10000.0)),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: None,
        settings=settings,
    )

    assert captured_quant_kwargs["win_rate"] == 0.64
    assert captured_quant_kwargs["avg_win"] == 0.0024
    assert captured_quant_kwargs["avg_loss"] == 0.0009
    assert captured_quant_kwargs["recent_returns"] == quant_trade_stats["recent_returns"]
    assert captured_quant_kwargs["transaction_cost"] == 0.00012
    assert captured_quant_kwargs["continuation_context"] == quant_trade_stats["continuation_context"]


def test_live_loop_appends_equity_history_when_quant_enabled(monkeypatch, tmp_path):
    from src.live_trade_loop import run_live_signal_loop
    from src.strategy.decision_tree import TopDownNoTrade
    from src.strategy.features import FeatureSnapshot
    from src.strategy.quant_engine import QuantDecision

    equity_log_path = tmp_path / "nested" / "equity.jsonl"
    live_input = SimpleNamespace(
        d1_candles=["d1"],
        h4_candles=["h4"],
        h1_candles=["h1"],
        m30_candles=["m30"],
        m15_candles=[SimpleNamespace(close=4700.0, timestamp=datetime(2026, 4, 28, 6, 15, tzinfo=timezone.utc))],
        spread=0.35,
    )
    no_trade = TopDownNoTrade(is_trade=False, reason="waiting", failed_node="setup", metadata={})

    class FakeExecutor:
        def list_bot_positions(self, symbol):
            return []

    settings = SimpleNamespace(
        quant_enabled=True,
        quant_gamma=2.0,
        quant_cvar_alpha=0.05,
        quant_cvar_eta=1.5,
        quant_dd_max=0.20,
        quant_dd_rho=0.5,
        quant_omega_threshold=0.5,
        quant_position_r_max=0.02,
        quant_transaction_lambda=1.0,
        quant_zscore_window=100,
        ml_enabled=False,
        feature_logging_enabled=False,
        equity_log_path=str(equity_log_path),
    )

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr("src.live_trade_loop.evaluate_top_down_decision_tree", lambda **kwargs: no_trade)
    monkeypatch.setattr("src.live_trade_loop._build_feature_extractor", lambda settings: SimpleNamespace(snapshot_count=5))
    monkeypatch.setattr(
        "src.live_trade_loop._estimate_strategy_trade_statistics",
        lambda **kwargs: {
            "win_rate": 0.5,
            "avg_win": 0.001,
            "avg_loss": 0.001,
            "expected_return": 0.0,
            "return_std": 0.001,
            "recent_returns": [0.001, -0.001, 0.001, -0.001, 0.001],
            "transaction_cost": 0.0001,
            "continuation_context": None,
        },
    )
    monkeypatch.setattr(
        "src.live_trade_loop._extract_features_from_strategy",
        lambda **kwargs: (
            FeatureSnapshot(
                timestamp=datetime(2026, 4, 28, 6, 15, tzinfo=timezone.utc),
                momentum_raw=0.0,
                trend_raw=0.0,
                volume_raw=0.0,
                order_block_raw=0.0,
                volatility_risk_raw=0.0,
                entry_distance_raw=0.0,
                spread_danger_raw=0.0,
                momentum_z=0.0,
                trend_z=0.0,
                volume_z=0.0,
                order_block_z=0.0,
                volatility_risk_z=0.0,
                entry_distance_z=0.0,
                spread_danger_z=0.0,
                expected_return=0.0,
                return_std=0.001,
            ),
            0.0,
            0.001,
        ),
    )
    monkeypatch.setattr(
        "src.strategy.quant_engine.evaluate_master_equation",
        lambda **kwargs: QuantDecision(
            action=0,
            omega_t=0.4,
            position_size_fraction=0.0,
            expected_return=kwargs["features"].expected_return,
            cvar=0.001,
            drawdown_ratio=kwargs["drawdown_ratio"],
            drawdown_dampener=1.0,
            utility_scores={-1: -0.1, 0: 0.0, 1: -0.1},
            sharpe_signal=0.0,
            reason="master_equation_flat",
            is_trade=False,
            metadata={},
        ),
    )

    run_live_signal_loop(
        mt5_module=SimpleNamespace(account_info=lambda: SimpleNamespace(equity=9800.0)),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: None,
        settings=settings,
    )

    lines = equity_log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    snapshot = json.loads(lines[0])
    assert snapshot["equity"] == 9800.0
    assert snapshot["peak_equity"] == 9800.0
    assert snapshot["drawdown"] == 0.0
    assert snapshot["drawdown_ratio"] == 0.0
    assert snapshot["timestamp"].endswith("+00:00")


def test_estimate_strategy_trade_statistics_forwards_requested_lot_and_campaign_exposure(monkeypatch):
    from src.live_trade_loop import _estimate_strategy_trade_statistics
    from src.strategy.breakout import BreakoutDirection
    from src.strategy.decision_tree import TopDownTradePlan

    captured_execution_kwargs = {}
    trade_plan = TopDownTradePlan(
        is_trade=True,
        direction=BreakoutDirection.BULLISH,
        entry_price=100.0,
        stop_loss=98.5,
        take_profit=103.0,
        objective_price=104.0,
        reason="top_down_trade_plan_ready",
        metadata={},
    )
    live_input = SimpleNamespace(
        m15_candles=[SimpleNamespace(close=100.0)],
        spread=0.04,
        tick_data={"bid": 99.98, "ask": 100.02},
    )

    monkeypatch.setattr(
        "src.live_trade_loop.assess_market_order_execution",
        lambda **kwargs: captured_execution_kwargs.update(kwargs)
        or SimpleNamespace(
            normalized_transaction_cost=0.0004,
            effective_reward_distance=3.0,
            effective_stop_distance=1.5,
            continuation_probability=0.0,
            continuation_mu=0.0,
            directional_tail_proxy=0.0,
            effective_rr=2.0,
            execution_penalty=0.1,
            dynamic_rr_floor=1.5,
            continuation_ev=0.0,
        ),
    )

    _estimate_strategy_trade_statistics(
        strategy_result=trade_plan,
        live_input=live_input,
        spread=0.04,
        requested_lot=0.25,
        campaign_exposure_pct=6.0,
    )

    assert captured_execution_kwargs["requested_lot"] == 0.25
    assert captured_execution_kwargs["campaign_exposure_pct"] == 6.0


def test_live_loop_passes_actual_lot_and_exposure_into_quant_trade_stats(monkeypatch):
    from src.live_trade_loop import run_live_signal_loop
    from src.strategy.breakout import BreakoutDirection
    from src.strategy.decision_tree import TopDownTradePlan
    from src.strategy.features import FeatureSnapshot
    from src.strategy.quant_engine import QuantDecision

    captured_trade_stats_kwargs = {}
    live_input = SimpleNamespace(
        d1_candles=["d1"],
        h4_candles=["h4"],
        h1_candles=["h1"],
        m30_candles=["m30"],
        m15_candles=[SimpleNamespace(close=4700.0, timestamp=datetime(2026, 4, 28, 6, 15, tzinfo=timezone.utc))],
        spread=0.35,
    )
    trade_plan = TopDownTradePlan(
        is_trade=True,
        direction=BreakoutDirection.BULLISH,
        entry_price=4700.0,
        stop_loss=4696.0,
        take_profit=4712.0,
        objective_price=4720.0,
        reason="top_down_trade_plan_ready",
        metadata={},
    )
    positions = [
        SimpleNamespace(
            ticket=1,
            entry_price=4688.0,
            initial_stop_loss=4680.0,
            stop_loss=4690.0,
            symbol="XAUUSD",
            tp=4702.0,
            volume=0.25,
            type=0,
        ),
    ]

    class FakeEquityTracker:
        drawdown_ratio = 0.0

        def update(self, equity):
            return None

    class FakeExecutor:
        mt5_module = SimpleNamespace(account_info=lambda: SimpleNamespace(equity=10000.0))

        def list_bot_positions(self, symbol):
            return positions

    settings = SimpleNamespace(
        quant_enabled=True,
        quant_gamma=2.0,
        quant_cvar_alpha=0.05,
        quant_cvar_eta=1.5,
        quant_dd_max=0.20,
        quant_dd_rho=0.5,
        quant_omega_threshold=0.5,
        quant_position_r_max=0.02,
        quant_transaction_lambda=1.0,
        quant_zscore_window=100,
        ml_enabled=False,
        feature_logging_enabled=False,
    )

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr("src.live_trade_loop.evaluate_top_down_decision_tree", lambda **kwargs: trade_plan)
    monkeypatch.setattr("src.live_trade_loop._build_feature_extractor", lambda settings: SimpleNamespace(snapshot_count=5))
    monkeypatch.setattr("src.live_trade_loop._build_equity_tracker", lambda settings: FakeEquityTracker())
    monkeypatch.setattr(
        "src.live_trade_loop._build_margin_snapshot",
        lambda **kwargs: {
            "campaign_exposure_pct": 6.0,
            "preferred_add_exposure_pct": 2.0,
            "fallback_add_exposure_pct": 1.0,
        },
    )
    monkeypatch.setattr(
        "src.live_trade_loop._estimate_strategy_trade_statistics",
        lambda **kwargs: captured_trade_stats_kwargs.update(kwargs)
        or {
            "win_rate": 0.64,
            "avg_win": 0.0024,
            "avg_loss": 0.0009,
            "expected_return": 0.0012,
            "return_std": 0.0018,
            "recent_returns": [0.0024, 0.0024, -0.0009, 0.0024, -0.0009],
            "transaction_cost": 0.00012,
            "continuation_context": None,
        },
    )
    monkeypatch.setattr(
        "src.live_trade_loop._extract_features_from_strategy",
        lambda **kwargs: (
            FeatureSnapshot(
                timestamp=datetime(2026, 4, 28, 6, 15, tzinfo=timezone.utc),
                momentum_raw=1.0,
                trend_raw=1.0,
                volume_raw=1.0,
                order_block_raw=1.0,
                volatility_risk_raw=1.0,
                entry_distance_raw=0.2,
                spread_danger_raw=0.1,
                momentum_z=1.0,
                trend_z=1.0,
                volume_z=0.5,
                order_block_z=0.8,
                volatility_risk_z=0.1,
                entry_distance_z=-0.4,
                spread_danger_z=0.2,
                expected_return=0.0012,
                return_std=0.0018,
            ),
            0.0012,
            0.0018,
        ),
    )
    monkeypatch.setattr(
        "src.strategy.quant_engine.evaluate_master_equation",
        lambda **kwargs: QuantDecision(
            action=0,
            omega_t=0.3,
            position_size_fraction=0.0,
            expected_return=kwargs["features"].expected_return,
            cvar=0.0009,
            drawdown_ratio=kwargs["drawdown_ratio"],
            drawdown_dampener=1.0,
            utility_scores={-1: -0.4, 0: -0.2, 1: -0.3},
            sharpe_signal=0.2,
            reason="master_equation_flat",
            is_trade=False,
            metadata={},
        ),
    )

    run_live_signal_loop(
        mt5_module=SimpleNamespace(account_info=lambda: SimpleNamespace(equity=10000.0)),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.25,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: None,
        settings=settings,
    )

    assert captured_trade_stats_kwargs["requested_lot"] == 0.25
    assert captured_trade_stats_kwargs["campaign_exposure_pct"] == 6.0


def test_latest_trade_r_multiple_preserves_original_risk_for_same_mt5_ticket():
    from src.live_trade_loop import _latest_trade_r_multiple
    from src.strategy.breakout import BreakoutDirection

    seeded_position = SimpleNamespace(ticket=22, price_open=100.0, sl=95.0, type=0)
    tightened_position = SimpleNamespace(ticket=22, price_open=100.0, sl=100.0, type=0)

    _latest_trade_r_multiple(seeded_position, 100.5, BreakoutDirection.BULLISH)
    result = _latest_trade_r_multiple(tightened_position, 110.0, BreakoutDirection.BULLISH)

    assert result == 2.0


def test_live_loop_preserves_strategy_reason_when_quant_is_enabled_but_tree_is_not_trade(monkeypatch):
    from src.live_trade_loop import run_live_signal_loop
    from src.strategy.decision_tree import TopDownNoTrade
    from src.strategy.features import FeatureSnapshot
    from src.strategy.quant_engine import QuantDecision

    events = []
    live_input = SimpleNamespace(
        d1_candles=["d1"],
        h4_candles=["h4"],
        h1_candles=["h1"],
        m30_candles=["m30"],
        m15_candles=[SimpleNamespace(close=4700.0, timestamp=datetime(2026, 4, 28, 6, 15, tzinfo=timezone.utc))],
        spread=0.35,
    )
    no_trade = TopDownNoTrade(
        is_trade=False,
        reason="m10_setup_not_ready",
        failed_node="m10_setup",
        metadata={},
    )

    class FakeEquityTracker:
        drawdown_ratio = 0.0

        def update(self, equity):
            return None

    class FakeExecutor:
        def list_bot_positions(self, symbol):
            return []

    settings = SimpleNamespace(
        quant_enabled=True,
        quant_gamma=2.0,
        quant_cvar_alpha=0.05,
        quant_cvar_eta=1.5,
        quant_dd_max=0.20,
        quant_dd_rho=0.5,
        quant_omega_threshold=0.5,
        quant_position_r_max=0.02,
        quant_transaction_lambda=1.0,
        quant_zscore_window=100,
        ml_enabled=False,
        feature_logging_enabled=False,
    )

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr("src.live_trade_loop.evaluate_top_down_decision_tree", lambda **kwargs: no_trade)
    monkeypatch.setattr("src.live_trade_loop._build_feature_extractor", lambda settings: SimpleNamespace(snapshot_count=5))
    monkeypatch.setattr("src.live_trade_loop._build_equity_tracker", lambda settings: FakeEquityTracker())
    monkeypatch.setattr(
        "src.live_trade_loop._estimate_strategy_trade_statistics",
        lambda **kwargs: {
            "win_rate": 0.5,
            "avg_win": 0.002,
            "avg_loss": 0.001,
            "expected_return": 0.0004,
            "return_std": 0.001,
            "recent_returns": [0.001, -0.001, 0.001],
            "transaction_cost": 0.00012,
            "continuation_context": None,
        },
    )
    monkeypatch.setattr(
        "src.live_trade_loop._extract_features_from_strategy",
        lambda **kwargs: (
            FeatureSnapshot(
                timestamp=datetime(2026, 4, 28, 6, 15, tzinfo=timezone.utc),
                momentum_raw=1.0,
                trend_raw=1.0,
                volume_raw=1.0,
                order_block_raw=1.0,
                volatility_risk_raw=1.0,
                entry_distance_raw=0.2,
                spread_danger_raw=0.1,
                momentum_z=1.0,
                trend_z=1.0,
                volume_z=0.5,
                order_block_z=0.8,
                volatility_risk_z=0.1,
                entry_distance_z=-0.4,
                spread_danger_z=0.2,
                expected_return=0.0004,
                return_std=0.001,
            ),
            0.0004,
            0.001,
        ),
    )
    monkeypatch.setattr(
        "src.strategy.quant_engine.evaluate_master_equation",
        lambda **kwargs: QuantDecision(
            action=0,
            omega_t=0.82,
            position_size_fraction=0.0,
            expected_return=kwargs["features"].expected_return,
            cvar=0.0009,
            drawdown_ratio=kwargs["drawdown_ratio"],
            drawdown_dampener=1.0,
            utility_scores={-1: -0.4, 0: -0.2, 1: -0.1},
            sharpe_signal=0.66,
            reason="continuation_quant_blocked",
            is_trade=False,
            metadata={"ce_scores": {-1: -0.002, 0: 0.0, 1: -0.001}},
        ),
    )

    run_live_signal_loop(
        mt5_module=SimpleNamespace(account_info=lambda: SimpleNamespace(equity=10000.0)),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(("log", message)),
        settings=settings,
    )

    assert ("log", "LIVE NO TRADE XAUUSD reason=m10_setup_not_ready node=m10_setup") in events


def test_live_loop_blocks_trade_when_quant_direction_disagrees(monkeypatch):
    from src.live_trade_loop import run_live_signal_loop
    from src.strategy.breakout import BreakoutDirection
    from src.strategy.decision_tree import TopDownTradePlan
    from src.strategy.features import FeatureSnapshot
    from src.strategy.quant_engine import QuantDecision

    events = []
    live_input = SimpleNamespace(
        d1_candles=["d1"],
        h4_candles=["h4"],
        h1_candles=["h1"],
        m30_candles=["m30"],
        m15_candles=[SimpleNamespace(close=4700.0, timestamp=datetime(2026, 4, 28, 6, 15, tzinfo=timezone.utc))],
        spread=0.35,
    )
    trade_plan = TopDownTradePlan(
        is_trade=True,
        direction=BreakoutDirection.BULLISH,
        entry_price=4700.0,
        stop_loss=4696.0,
        take_profit=4712.0,
        objective_price=4720.0,
        reason="top_down_trade_plan_ready",
        metadata={},
    )

    class FakeEquityTracker:
        drawdown_ratio = 0.0

        def update(self, equity):
            return None

    class FakeExecutor:
        def list_bot_positions(self, symbol):
            return []

        def open_strategy_trade(self, **kwargs):
            raise AssertionError("opposite quant direction should block strategy trade")

    settings = SimpleNamespace(
        quant_enabled=True,
        quant_gamma=2.0,
        quant_cvar_alpha=0.05,
        quant_cvar_eta=1.5,
        quant_dd_max=0.20,
        quant_dd_rho=0.5,
        quant_omega_threshold=0.5,
        quant_position_r_max=0.02,
        quant_transaction_lambda=1.0,
        quant_zscore_window=100,
        ml_enabled=False,
        feature_logging_enabled=False,
    )

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr("src.live_trade_loop.evaluate_top_down_decision_tree", lambda **kwargs: trade_plan)
    monkeypatch.setattr("src.live_trade_loop._build_feature_extractor", lambda settings: SimpleNamespace(snapshot_count=5))
    monkeypatch.setattr("src.live_trade_loop._build_equity_tracker", lambda settings: FakeEquityTracker())
    monkeypatch.setattr(
        "src.live_trade_loop._estimate_strategy_trade_statistics",
        lambda **kwargs: {
            "win_rate": 0.64,
            "avg_win": 0.0024,
            "avg_loss": 0.0009,
            "expected_return": 0.0012,
            "return_std": 0.0018,
            "recent_returns": [0.001, -0.001, 0.001],
            "transaction_cost": 0.00012,
            "continuation_context": None,
        },
    )
    monkeypatch.setattr(
        "src.live_trade_loop._extract_features_from_strategy",
        lambda **kwargs: (
            FeatureSnapshot(
                timestamp=datetime(2026, 4, 28, 6, 15, tzinfo=timezone.utc),
                momentum_raw=1.0,
                trend_raw=1.0,
                volume_raw=1.0,
                order_block_raw=1.0,
                volatility_risk_raw=1.0,
                entry_distance_raw=0.2,
                spread_danger_raw=0.1,
                momentum_z=1.0,
                trend_z=1.0,
                volume_z=0.5,
                order_block_z=0.8,
                volatility_risk_z=0.1,
                entry_distance_z=-0.4,
                spread_danger_z=0.2,
                expected_return=0.0012,
                return_std=0.0018,
            ),
            0.0012,
            0.0018,
        ),
    )
    monkeypatch.setattr(
        "src.strategy.quant_engine.evaluate_master_equation",
        lambda **kwargs: QuantDecision(
            action=-1,
            omega_t=0.82,
            position_size_fraction=0.01,
            expected_return=kwargs["features"].expected_return,
            cvar=0.0009,
            drawdown_ratio=kwargs["drawdown_ratio"],
            drawdown_dampener=1.0,
            utility_scores={-1: -0.1, 0: -0.4, 1: -0.3},
            sharpe_signal=0.66,
            reason="master_equation_short_approved",
            is_trade=True,
            metadata={"ce_scores": {-1: 0.002, 0: 0.0, 1: 0.001}},
        ),
    )

    run_live_signal_loop(
        mt5_module=SimpleNamespace(account_info=lambda: SimpleNamespace(equity=10000.0)),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: events.append(("log", message)),
        settings=settings,
    )

    assert ("log", "LIVE NO TRADE XAUUSD reason=quant_direction_mismatch node=quant_engine") in events


def test_live_loop_passes_latest_tradingview_alert_into_decision_tree(monkeypatch):
    from src.live_trade_loop import run_live_signal_loop
    from src.strategy.decision_tree import TopDownNoTrade
    from src.strategy.breakout import BreakoutDirection
    from src.tradingview import TradingViewAlert

    captured_kwargs = {}
    live_input = SimpleNamespace(
        d1_candles=["d1"],
        h4_candles=["h4"],
        h1_candles=["h1"],
        m30_candles=["m30"],
        m10_candles=["m10"],
        m15_candles=[SimpleNamespace(close=2350.0, timestamp=datetime(2026, 4, 28, 6, 15, tzinfo=timezone.utc))],
        m5_candles=["m5"],
    )
    no_trade = TopDownNoTrade(
        is_trade=False,
        reason="m15_trigger_missing",
        failed_node="m15_trigger",
        metadata={},
    )
    alert = TradingViewAlert(
        is_valid=True,
        reason="accepted",
        symbol="XAUUSD",
        direction=BreakoutDirection.BULLISH,
        setup="three_drives",
        level=2350.0,
        timestamp=datetime(2026, 4, 28, 6, 14, tzinfo=timezone.utc),
        timeframe="15",
        confidence=0.76,
        context={},
    )

    class FakeStore:
        def latest_for(self, symbol, *, now=None):
            return alert

    class FakeExecutor:
        def list_bot_positions(self, symbol):
            return []

    monkeypatch.setattr("src.live_trade_loop.build_live_strategy_input", lambda mt5_module, symbol: live_input)
    monkeypatch.setattr(
        "src.live_trade_loop.evaluate_top_down_decision_tree",
        lambda **kwargs: captured_kwargs.update(kwargs) or no_trade,
    )

    run_live_signal_loop(
        mt5_module=SimpleNamespace(),
        executor=FakeExecutor(),
        symbol="XAUUSD",
        lot=0.01,
        risk_buffer=0.05,
        max_candles_since_breakout=3,
        poll_seconds=10,
        max_loops=1,
        tradingview_alert_store=FakeStore(),
        reload_check_fn=lambda: False,
        sleep_fn=lambda seconds: None,
        log_fn=lambda message: None,
    )

    assert captured_kwargs["tradingview_alert"] is alert
    assert captured_kwargs["m10_candles"] == ["m10"]
    assert captured_kwargs["m5_candles"] == ["m5"]
