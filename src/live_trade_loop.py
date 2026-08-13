from __future__ import annotations

from datetime import datetime, timezone
import json
from math import exp, sqrt
from pathlib import Path
from time import sleep
from uuid import uuid4

from src.market_data import build_live_strategy_input
from src.quick_scalp_loop import save_bot_state, save_training_snapshot
from src.strategy.breakout import BreakoutDirection
from src.strategy.campaign_add import evaluate_campaign_add
from src.strategy.decision_tree import TopDownNoTrade, TopDownTradePlan, evaluate_top_down_decision_tree
from src.strategy.execution_cost import assess_market_order_execution
from src.strategy.management import (
    evaluate_campaign_action,
    evaluate_fixed_trailing_stop,
    remember_position_initial_stop_loss,
)
from src.strategy.math_engine import resolve_quant_metrics
from src.strategy.volatility import build_volatility_state
from src.strategy.session_engine import SessionEngine


# ---------------------------------------------------------------------------
# Quant engine integration helpers
# ---------------------------------------------------------------------------

def _build_quant_params(settings):
    """Build QuantParams from settings, importing lazily."""
    from src.strategy.quant_engine import OmegaWeights, QuantParams

    return QuantParams(
        gamma=settings.quant_gamma,
        cvar_alpha=settings.quant_cvar_alpha,
        cvar_eta=settings.quant_cvar_eta,
        dd_rho=settings.quant_dd_rho,
        dd_max=settings.quant_dd_max,
        omega_threshold=settings.quant_omega_threshold,
        position_r_max=settings.quant_position_r_max,
        transaction_lambda=settings.quant_transaction_lambda,
        omega_weights=OmegaWeights(),
    )


def _build_feature_extractor(settings):
    """Build FeatureExtractor from settings."""
    from src.strategy.features import FeatureExtractor

    return FeatureExtractor(window=settings.quant_zscore_window)


def _build_equity_tracker(settings):
    """Build EquityTracker from settings."""
    from src.strategy.equity_tracker import EquityTracker

    return EquityTracker(dd_max=settings.quant_dd_max)


def _build_ml_classifier(settings):
    """Build ML regime classifier if enabled."""
    if not settings.ml_enabled:
        return None
    from src.strategy.ml_regime import MLRegimeClassifier
    from src.strategy.regime import classify_regime

    return MLRegimeClassifier(
        model_path=settings.ml_model_path if settings.ml_model_path else None,
        fallback_fn=classify_regime,
    )


def _build_local_edge_model(settings, log_fn):
    if not settings or not getattr(settings, "local_edge_enabled", False):
        return None
    try:
        from src.strategy.local_edge_model import LocalEdgeModel

        model = LocalEdgeModel.load(getattr(settings, "local_edge_model_path", "data/models/local_edge_model.npz"))
        model.threshold = float(getattr(settings, "local_edge_threshold", model.threshold) or model.threshold)
        log_fn(f"LOCAL EDGE MODEL ACTIVE threshold={model.threshold:.3f}")
        return model
    except Exception as exc:
        log_fn(f"LOCAL EDGE MODEL DISABLED reason={exc}")
        return None


def _run_online_edge_training(settings, log_fn):
    if not settings or not getattr(settings, "local_edge_online_train_enabled", False):
        return None
    try:
        from argparse import Namespace
        from src.strategy.intelligence_pipeline import run_pipeline

        candidate_path = str(Path(getattr(settings, "local_edge_model_path", "data/models/local_edge_model.npz")).with_name(
            "local_edge_model.online_candidate.npz"
        ))
        report = run_pipeline(
            Namespace(
                features=getattr(settings, "feature_log_path", "data/features.jsonl"),
                diagnostics=getattr(settings, "diagnostics_log_path", "data/live_diagnostics.jsonl"),
                paper_trades=getattr(settings, "paper_trade_log_path", "data/paper_trades.jsonl"),
                candles=getattr(settings, "local_edge_online_train_candles_path", "") or None,
                candidate_model=candidate_path,
                live_model=getattr(settings, "local_edge_model_path", "data/models/local_edge_model.npz"),
                report="data/models/intelligence_online.report.json",
                horizon_bars=120,
                folds=4,
                epochs=500,
                learning_rate=0.03,
                hidden_size=16,
                threshold=float(getattr(settings, "local_edge_threshold", 0.55) or 0.55),
                min_abs_return=0.0,
                promote=bool(getattr(settings, "local_edge_online_train_promote", False)),
                min_promotion_samples=int(getattr(settings, "local_edge_online_train_min_rows", 120) or 120),
                min_promotion_precision=0.52,
                min_promotion_profit_factor=1.05,
            )
        )
        promotion = report.get("promotion") or {}
        status = "promoted" if promotion.get("promoted") else "candidate_only"
        log_fn(
            f"ONLINE EDGE TRAIN {status} rows={report.get('rows', 0)} "
            f"precision={promotion.get('precision', 0.0)} profit_factor={promotion.get('profit_factor', 0.0)}"
        )
        if promotion.get("promoted"):
            return _build_local_edge_model(settings, log_fn)
    except Exception as exc:
        log_fn(f"ONLINE EDGE TRAIN WARN reason={exc}")
    return None


def _extract_features_from_strategy(
    *,
    live_input,
    strategy_result,
    feature_extractor,
    spread,
    expected_return_override=None,
    return_std_override=None,
    orderflow_signal=None,
):
    """Extract raw features from strategy outputs and compute z-scores."""
    from src.strategy.features import (
        extract_entry_distance,
        extract_expected_return,
        extract_momentum,
        extract_order_block_quality,
        extract_spread_danger,
        extract_trend,
        extract_volatility_risk,
        extract_volume,
    )

    metadata = getattr(strategy_result, "metadata", {}) or {}
    volatility_state = metadata.get("volatility_state")
    regime_state = metadata.get("regime_state")
    score_decision = metadata.get("score_decision")
    h4_context = metadata.get("h4_context")

    # Extract raw features
    momentum_raw = extract_momentum(
        d1_candles=live_input.d1_candles,
        h4_candles=live_input.h4_candles,
        h1_candles=live_input.h1_candles,
        m30_candles=live_input.m30_candles,
        m15_candles=live_input.m15_candles,
    )
    trend_raw = extract_trend(
        d1_candles=live_input.d1_candles,
        h4_candles=live_input.h4_candles,
        h1_candles=live_input.h1_candles,
        m30_candles=live_input.m30_candles,
        m15_candles=live_input.m15_candles,
    )
    volume_raw = extract_volume(live_input.m15_candles, lookback=20)
    
    current_price = float(live_input.m15_candles[-1].close)
    atr = float(getattr(volatility_state, "short_atr", 1.0)) if volatility_state else 1.0
    
    order_block_raw = extract_order_block_quality(
        price=current_price,
        candles=live_input.m15_candles,
        demand_zones=getattr(h4_context, "demand_zones", ()),
        supply_zones=getattr(h4_context, "supply_zones", ()),
        atr=atr,
    )

    volatility_risk_raw = extract_volatility_risk(volatility_state) if volatility_state else 1.0

    # Entry distance from demand/supply zones
    entry_distance_raw = 0.0
    if h4_context is not None:
        entry_distance_raw = extract_entry_distance(
            current_price,
            getattr(h4_context, "demand_zones", ()),
            getattr(h4_context, "supply_zones", ()),
        )

    # Spread danger
    spread_danger_raw = extract_spread_danger(spread, atr)

    # Expected return
    if expected_return_override is None or return_std_override is None:
        expected_return, return_std = extract_expected_return(live_input.m15_candles, lookback=20)
    else:
        expected_return = float(expected_return_override)
        return_std = max(float(return_std_override), 1e-9)

    # M5 context signal
    m5_score_raw = 0.0
    try:
        from src.strategy.m5_engine import M5SignalEngine
        m5_candles = getattr(live_input, "m5_candles", None)
        if m5_candles and len(m5_candles) >= 50:
            m5_engine = M5SignalEngine()
            m5_snap = m5_engine.compute(m5_candles)
            if m5_snap:
                m5_score_raw = m5_snap.composite_score
    except Exception:
        pass

    # M15 structural signal
    m15_score_raw = 0.0
    try:
        from src.strategy.m15_engine import M15SignalEngine
        m15_engine = M15SignalEngine()
        m15_snap = m15_engine.compute(live_input.m15_candles)
        if m15_snap:
            m15_score_raw = m15_snap.composite_score
    except Exception:
        pass

    # M30/H1 Context signal
    context_score_raw = 0.0
    try:
        from src.strategy.context_engine import ContextEngine
        context_engine = ContextEngine()
        context_snap = context_engine.compute(
            h1_candles=live_input.h1_candles,
            m30_candles=live_input.m30_candles,
            d1_candles=live_input.d1_candles,
        )
        if context_snap:
            context_score_raw = context_snap.composite_context
    except Exception:
        pass

    structure_score_raw = 0.0
    volatility_score_raw = 0.0
    momentum_indicator_raw = 0.0
    trend_indicator_raw = 0.0
    orderflow_volume_raw = 0.0
    risk_math_raw = 0.0
    statistical_score_raw = 0.0
    try:
        from src.strategy.indicator_math import build_indicator_math_pack

        indicator_pack = build_indicator_math_pack(live_input=live_input, orderflow_signal=orderflow_signal)
        structure_score_raw = indicator_pack.structure_score
        volatility_score_raw = indicator_pack.volatility_score
        momentum_indicator_raw = indicator_pack.momentum_score
        trend_indicator_raw = indicator_pack.trend_score
        orderflow_volume_raw = indicator_pack.orderflow_volume_score
        risk_math_raw = indicator_pack.risk_score
        statistical_score_raw = indicator_pack.statistical_score
    except Exception:
        pass

    snapshot = feature_extractor.update(
        momentum_raw=momentum_raw,
        trend_raw=trend_raw,
        volume_raw=volume_raw,
        order_block_raw=order_block_raw,
        volatility_risk_raw=volatility_risk_raw,
        entry_distance_raw=entry_distance_raw,
        spread_danger_raw=spread_danger_raw,
        orderflow_raw=orderflow_signal,
        m5_score_raw=m5_score_raw,
        m15_score_raw=m15_score_raw,
        context_score_raw=context_score_raw,
        structure_score_raw=structure_score_raw,
        volatility_score_raw=volatility_score_raw,
        momentum_indicator_raw=momentum_indicator_raw,
        trend_indicator_raw=trend_indicator_raw,
        orderflow_volume_raw=orderflow_volume_raw,
        risk_math_raw=risk_math_raw,
        statistical_score_raw=statistical_score_raw,
        expected_return=expected_return,
        return_std=return_std,
        # Use wall-clock time so every poll cycle advances snapshot_count.
        # Candle timestamp only changes every 15 min, which would freeze warmup.
        timestamp=datetime.now(tz=timezone.utc),
    )

    return snapshot, expected_return, return_std


def _collect_recent_returns(m15_candles, lookback: int = 50) -> list[float]:
    """Collect recent bar returns for CVaR computation."""
    candles = m15_candles[-min(lookback + 1, len(m15_candles)):]
    returns: list[float] = []
    for i in range(1, len(candles)):
        prev = float(candles[i - 1].close)
        cur = float(candles[i].close)
        if prev > 0:
            returns.append((cur - prev) / prev)
    return returns


def _collect_directional_recent_returns(m15_candles, direction: BreakoutDirection, lookback: int = 50) -> list[float]:
    returns = _collect_recent_returns(m15_candles, lookback=lookback)
    if direction is BreakoutDirection.BEARISH:
        return [-value for value in returns]
    return returns


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    exp_value = exp(value)
    return exp_value / (1.0 + exp_value)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _local_edge_lot_multiplier(probability: float | None, settings) -> float:
    if probability is None or settings is None:
        return 1.0
    threshold = float(getattr(settings, "local_edge_threshold", 0.55) or 0.55)
    full_size_threshold = float(getattr(settings, "local_edge_full_size_threshold", 0.72) or 0.72)
    max_size_threshold = float(getattr(settings, "local_edge_max_size_threshold", 0.88) or 0.88)
    min_multiplier = float(getattr(settings, "local_edge_min_lot_multiplier", 0.35) or 0.35)
    max_multiplier = float(getattr(settings, "local_edge_max_lot_multiplier", 1.25) or 1.25)
    probability = float(probability)
    if probability < threshold:
        return 0.0
    if probability < full_size_threshold:
        span = max(full_size_threshold - threshold, 1e-9)
        return min_multiplier + ((probability - threshold) / span) * (1.0 - min_multiplier)
    if probability < max_size_threshold:
        span = max(max_size_threshold - full_size_threshold, 1e-9)
        return 1.0 + ((probability - full_size_threshold) / span) * (max_multiplier - 1.0)
    return max_multiplier


def _normalize_transaction_cost(
    *,
    price: float,
    spread: float,
    commission_per_lot: float = 6.0,
    lot: float = 0.01,
    volatility_state=None,
) -> float:
    """Normalize total round-trip transaction cost into return space.

    Includes:
    - Spread (bid/ask cost)
    - Realistic ATR-scaled slippage buffer: max(spread * 0.50, short_atr * 0.12)
    - Round-trip commission in price units ($6.00/lot default for HFM Zero/ECN)
    """
    safe_price = max(float(price), 1e-9)
    safe_spread = max(float(spread), 0.0)

    # ATR-scaled slippage buffer matching execution_cost.py
    short_atr = float(getattr(volatility_state, "short_atr", safe_spread) or safe_spread)
    slippage_est = max(safe_spread * 0.50, short_atr * 0.12)

    # Commission round-trip per lot in account currency (default $6.00 per lot round-trip for HFM)
    comm_per_lot = float(commission_per_lot if commission_per_lot is not None else 6.0)
    contract_multiplier = 100.0 if (safe_price > 1000 and safe_price < 10000) else 100000.0
    commission_price_units = comm_per_lot / contract_multiplier

    return (safe_spread + slippage_est + commission_price_units) / safe_price


def _fallback_volatility_state(live_input):
    candles = getattr(live_input, "m15_candles", []) or []
    if len(candles) >= 3:
        return build_volatility_state(candles=candles[-min(12, len(candles)) :])

    from src.strategy.volatility import VolatilityState

    close_price = float(candles[-1].close) if candles else 1.0
    synthetic_atr = max(abs(close_price) * 0.001, 0.1)
    return VolatilityState(
        short_atr=synthetic_atr,
        medium_atr=synthetic_atr,
        realized_range=synthetic_atr,
        body_efficiency=0.6,
        range_expansion_ratio=1.0,
    )


def _estimate_strategy_trade_statistics(
    *,
    strategy_result,
    live_input,
    spread,
    requested_lot: float,
    campaign_exposure_pct: float,
    commission_per_lot: float = 6.0,
) -> dict[str, object]:
    from src.strategy.breakout import BreakoutDirection
    from src.strategy.features import extract_expected_return

    metadata = getattr(strategy_result, "metadata", {}) or {}
    volatility_state = metadata.get("volatility_state") or _fallback_volatility_state(live_input)
    current_price = float(live_input.m15_candles[-1].close)
    normalized_cost = _normalize_transaction_cost(
        price=current_price,
        spread=spread,
        commission_per_lot=commission_per_lot,
        lot=requested_lot,
        volatility_state=volatility_state,
    )

    if not getattr(strategy_result, "is_trade", False):
        expected_return, return_std = extract_expected_return(live_input.m15_candles, lookback=20)
        return {
            "win_rate": 0.5,
            "avg_win": 1.0,
            "avg_loss": 1.0,
            "expected_return": expected_return,
            "return_std": return_std,
            "recent_returns": _collect_recent_returns(live_input.m15_candles, lookback=50),
            "transaction_cost": normalized_cost,
            "continuation_context": None,
        }

    entry_price = float(strategy_result.entry_price)
    stop_loss = float(strategy_result.stop_loss)
    take_profit = float(strategy_result.take_profit)
    if entry_price <= 0:
        expected_return, return_std = extract_expected_return(live_input.m15_candles, lookback=20)
        return {
            "win_rate": 0.5,
            "avg_win": 1.0,
            "avg_loss": 1.0,
            "expected_return": expected_return,
            "return_std": return_std,
            "recent_returns": _collect_recent_returns(live_input.m15_candles, lookback=50),
            "transaction_cost": normalized_cost,
            "continuation_context": None,
        }

    risk_distance = abs(entry_price - stop_loss)
    reward_distance = abs(take_profit - entry_price)
    if risk_distance <= 0 or reward_distance <= 0:
        expected_return, return_std = extract_expected_return(live_input.m15_candles, lookback=20)
        return {
            "win_rate": 0.5,
            "avg_win": 1.0,
            "avg_loss": 1.0,
            "expected_return": expected_return,
            "return_std": return_std,
            "recent_returns": _collect_recent_returns(live_input.m15_candles, lookback=50),
            "transaction_cost": normalized_cost,
            "continuation_context": None,
        }

    reward_to_risk = reward_distance / risk_distance
    metadata = getattr(strategy_result, "metadata", {}) or {}
    score_decision = metadata.get("score_decision")
    regime_state = metadata.get("regime_state")
    pattern_confluence_score = float(metadata.get("pattern_confluence_score", 0.0) or 0.0)
    tradingview_confluence = metadata.get("tradingview_confluence")

    threshold = abs(float(getattr(score_decision, "threshold", 1.0) or 1.0))
    threshold = max(threshold, 1e-6)
    edge_ratio = abs(float(getattr(score_decision, "edge", 0.0) or 0.0)) / threshold
    expected_move_multiple = float(getattr(score_decision, "expected_move_multiple", reward_to_risk) or reward_to_risk)
    uncertainty_penalty = float(getattr(score_decision, "uncertainty_penalty", 0.0) or 0.0)
    regime_confidence = float(getattr(regime_state, "confidence", 0.55) or 0.55)
    direction_bonus = abs(float(getattr(tradingview_confluence, "direction_bonus", 0.0) or 0.0))
    setup_bonus = abs(float(getattr(tradingview_confluence, "setup_bonus", 0.0) or 0.0))
    pattern_bonus = _clamp(pattern_confluence_score / 2.0, 0.0, 1.0)

    confidence_signal = (
        (0.75 * edge_ratio)
        + (0.55 * ((regime_confidence - 0.5) * 2.0))
        + (0.20 * max(expected_move_multiple - 1.0, 0.0))
        + (0.15 * pattern_bonus)
        + (0.10 * direction_bonus)
        + (0.08 * setup_bonus)
        - (0.18 * uncertainty_penalty)
    )
    confidence = _sigmoid(confidence_signal)

    tick_data = getattr(live_input, "tick_data", {}) or {}
    current_bid = float(tick_data.get("bid", current_price - (spread / 2.0)))
    current_ask = float(tick_data.get("ask", current_price + (spread / 2.0)))
    volatility_state = _fallback_volatility_state(live_input)
    continuation_context = _build_continuation_context(metadata)
    execution_assessment = assess_market_order_execution(
        direction=strategy_result.direction,
        planned_entry=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        current_bid=current_bid,
        current_ask=current_ask,
        spread=float(spread),
        volatility_state=volatility_state,
        requested_lot=float(requested_lot),
        campaign_exposure_pct=float(campaign_exposure_pct),
        continuation_context=continuation_context,
    )

    normalized_cost = max(normalized_cost, execution_assessment.normalized_transaction_cost)
    avg_win = max((execution_assessment.effective_reward_distance / entry_price) - normalized_cost, 1e-6)
    avg_loss = max((execution_assessment.effective_stop_distance / entry_price) + normalized_cost, 1e-6)
    breakeven_probability = avg_loss / (avg_win + avg_loss)
    # Dynamic win rate floor based on alignment, regime trend, and volatility expansion
    wanted_dir = "BULLISH" if strategy_result.direction is BreakoutDirection.BULLISH else "BEARISH"
    align_groups = [
        getattr(live_input, "h1_candles", []),
        getattr(live_input, "m30_candles", []),
        getattr(live_input, "m15_candles", []),
        getattr(live_input, "m5_candles", []),
    ]
    votes = 0
    align_total = 0
    for candles in align_groups:
        if len(candles) < 1:
            continue
        candle = candles[-1]
        if isinstance(candle, str) or not hasattr(candle, "close"):
            continue
        align_total += 1
        open_val = float(getattr(candle, "open", candle.close))
        actual = "BULLISH" if float(candle.close) >= open_val else "BEARISH"
        votes += 1 if actual == wanted_dir else -1
    alignment_score = votes / max(align_total, 1) if align_total > 0 else 0.0

    regime_name = getattr(regime_state, "name", "ranging")
    is_trend = (regime_name == "trend")

    # Rigorous win rate formulation: scale win rate relative to breakeven_probability
    # based on setup confidence, regime quality, and directional alignment.
    # High-quality setups get win_rate > breakeven_probability (+EV).
    # Weak setups (poor alignment, execution penalty) get win_rate <= breakeven_probability (-EV after costs).
    confidence_delta = (confidence - 0.50) * 0.20
    regime_bonus = 0.03 if is_trend else -0.01
    align_bonus = 0.03 if alignment_score >= 0.0 else -0.03
    exec_penalty = float(getattr(execution_assessment, "execution_penalty", 0.0) or 0.0)

    # Base win rate starts at breakeven + 0.04 (giving valid setups ~51-55% baseline win rate)
    raw_win_rate = (breakeven_probability + 0.04) + confidence_delta + regime_bonus + align_bonus - (exec_penalty * 0.05)
    win_rate = _clamp(raw_win_rate, 0.05, 0.95)
    loss_rate = 1.0 - win_rate

    # True trade expectancy (can be negative for sub-par setups)
    trade_expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    variance = (
        (win_rate * ((avg_win - trade_expectancy) ** 2))
        + (loss_rate * (((-avg_loss) - trade_expectancy) ** 2))
    )
    return_std = max(sqrt(max(variance, 0.0)), 1e-9)

    # Always provide the quant engine with the actual trade edge (mu_cont).
    # Without this, evaluate_master_equation falls back to raw market EWMA
    # (features.expected_return) which is often negative in ranging markets,
    # causing CE to prefer action=-1 on a BUY setup and triggering hard_block.
    direction_sign = 1.0 if strategy_result.direction is BreakoutDirection.BULLISH else -1.0
    recent_returns = _collect_directional_recent_returns(
        live_input.m15_candles,
        strategy_result.direction,
        lookback=50,
    )
    empirical_wins = [value for value in recent_returns if value > normalized_cost]
    empirical_losses = [abs(value) for value in recent_returns if value < -normalized_cost]
    if len(recent_returns) >= 8 and empirical_wins and empirical_losses:
        empirical_win_rate = len(empirical_wins) / len(recent_returns)
        empirical_avg_win = max((sum(empirical_wins) / len(empirical_wins)) - normalized_cost, 1e-6)
        empirical_avg_loss = max((sum(empirical_losses) / len(empirical_losses)) + normalized_cost, 1e-6)
        win_rate = _clamp((0.65 * win_rate) + (0.35 * empirical_win_rate), 0.05, 0.95)
        avg_win = max((0.70 * avg_win) + (0.30 * empirical_avg_win), 1e-6)
        avg_loss = max((0.70 * avg_loss) + (0.30 * empirical_avg_loss), 1e-6)
        loss_rate = 1.0 - win_rate
        trade_expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        variance = (
            (win_rate * ((avg_win - trade_expectancy) ** 2))
            + (loss_rate * (((-avg_loss) - trade_expectancy) ** 2))
        )
        return_std = max(sqrt(max(variance, 0.0)), 1e-9)
    trade_mu = direction_sign * trade_expectancy
    base_context = continuation_context or {}
    return {
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expected_return": trade_mu,
        "return_std": return_std,
        "recent_returns": recent_returns,
        "transaction_cost": normalized_cost,
        "continuation_context": {
            **base_context,
            "is_continuation_setup": True,
            "mu_cont": trade_mu,
            "continuation_probability": execution_assessment.continuation_probability,
            "cvar_dir": execution_assessment.directional_tail_proxy,
            "effective_rr": execution_assessment.effective_rr,
            "execution_penalty": execution_assessment.execution_penalty,
            "directional_tail_proxy": execution_assessment.directional_tail_proxy,
            "dynamic_rr_floor": execution_assessment.dynamic_rr_floor,
            "continuation_ev": execution_assessment.continuation_ev,
        },
    }


def _assess_live_execution(*, live_input, strategy_result, requested_lot: float, campaign_exposure_pct: float = 0.0):
    metadata = getattr(strategy_result, "metadata", {}) or {}
    volatility_state = metadata.get("volatility_state")
    if volatility_state is None:
        volatility_state = _fallback_volatility_state(live_input)

    current_price = float(live_input.m15_candles[-1].close)
    spread = float(getattr(live_input, "spread", 0.0) or 0.0)
    tick_data = getattr(live_input, "tick_data", {}) or {}
    current_bid = float(tick_data.get("bid", current_price - (spread / 2.0)))
    current_ask = float(tick_data.get("ask", current_price + (spread / 2.0)))

    return assess_market_order_execution(
        direction=strategy_result.direction,
        planned_entry=float(strategy_result.entry_price),
        stop_loss=float(strategy_result.stop_loss),
        take_profit=float(strategy_result.take_profit),
        current_bid=current_bid,
        current_ask=current_ask,
        spread=spread,
        volatility_state=volatility_state,
        requested_lot=float(requested_lot),
        campaign_exposure_pct=float(campaign_exposure_pct),
        continuation_context=_build_continuation_context(metadata),
    )


def _build_continuation_context(metadata):
    if not metadata or not metadata.get("is_continuation_setup"):
        return None

    return {
        "is_continuation_setup": True,
        "m15_quality": float(metadata.get("m15_quality", 0.0) or 0.0),
        "m10_quality": float(metadata.get("m10_quality", 0.0) or 0.0),
        "m5_quality": float(metadata.get("m5_quality", 0.0) or 0.0),
        "range_expansion_ratio": float(metadata.get("range_expansion_ratio", 1.0) or 1.0),
        "body_efficiency": float(metadata.get("body_efficiency", 0.5) or 0.5),
        "regime_confidence": float(metadata.get("regime_confidence", 0.5) or 0.5),
    }


def _format_quant_metrics(quant_decision, log_fn):
    """Format quant engine metrics for logging."""
    return (
        f"Ω_t={quant_decision.omega_t:.4f} "
        f"E[R]={quant_decision.expected_return:.6f} "
        f"CVaR={quant_decision.cvar:.6f} "
        f"DD={quant_decision.drawdown_ratio:.4f} "
        f"Sharpe={quant_decision.sharpe_signal:.4f} "
        f"U={{-1:{quant_decision.utility_scores.get(-1, 0):.6f}, "
        f"0:{quant_decision.utility_scores.get(0, 0):.6f}, "
        f"1:{quant_decision.utility_scores.get(1, 0):.6f}}}"
    )


def _build_standard_training_state(
    *,
    mt5_module,
    symbol: str,
    live_input,
    strategy_result,
    positions,
    quant_decision=None,
) -> dict:
    account_info = mt5_module.account_info() if hasattr(mt5_module, "account_info") else None
    latest_candle = (getattr(live_input, "m15_candles", None) or [None])[-1]
    timestamp = getattr(latest_candle, "timestamp", None)
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    elif getattr(timestamp, "tzinfo", None) is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)

    is_trade = bool(getattr(strategy_result, "is_trade", False))
    direction = getattr(strategy_result, "direction", None)
    planned_direction = direction.value if direction is not None else "None"
    reason = str(getattr(strategy_result, "reason", "") or "")
    metadata = getattr(strategy_result, "metadata", {}) or {}
    score_decision = metadata.get("score_decision")
    regime_state = metadata.get("regime_state")
    volatility_state = metadata.get("volatility_state")
    tick_data = getattr(live_input, "tick_data", {}) or {}
    spread = float(getattr(live_input, "spread", 0.0) or 0.0)

    confluence_score = float(metadata.get("pattern_confluence_score", 0.0) or 0.0)
    dashboard_candles = (getattr(live_input, "m1_candles", None) or getattr(live_input, "m5_candles", None) or getattr(live_input, "m15_candles", []) or [])[-80:]
    dashboard_ticks = _dashboard_ticks(getattr(live_input, "ticks", None), tick_data, timestamp)
    quant_payload = _dashboard_quant_payload(
        candles=dashboard_candles,
        ticks=dashboard_ticks,
        account_info=account_info,
        quant_decision=quant_decision,
    )
    execution_history = _dashboard_execution_history(
        positions=positions,
        strategy_result=strategy_result,
        timestamp=timestamp,
    )

    return {
        "timestamp": timestamp.isoformat(),
        "account": {
            "balance": float(getattr(account_info, "balance", 0.0) or 0.0),
            "equity": float(getattr(account_info, "equity", 0.0) or 0.0),
            "profit": float(getattr(account_info, "profit", 0.0) or 0.0),
            "currency": str(getattr(account_info, "currency", "")) if account_info is not None else "",
        },
        "signals": {
            "tick_dir": "None",
            "m1_dir": "None",
            "fib_dir": planned_direction if is_trade else "None",
            "fib_zone": reason,
            "rsi": 0.0,
            "sar_dir": "None",
            "quant": quant_payload,
            "mtf": {
                "m1": "None",
                "m5": _latest_candle_direction(getattr(live_input, "m5_candles", [])),
                "m15": _latest_candle_direction(getattr(live_input, "m15_candles", [])),
                "h1": _latest_candle_direction(getattr(live_input, "h1_candles", [])),
            },
            "confluence": {
                "fib_ok": is_trade,
                "sar_ok": bool(regime_state and float(getattr(regime_state, "confidence", 0.0) or 0.0) >= 0.5),
                "rsi_ok": bool(score_decision and float(getattr(score_decision, "edge", 0.0) or 0.0) > 0.0),
            },
        },
        "market_data": {
            "m1_candles": [_candle_to_training_dict(candle) for candle in dashboard_candles],
            "ticks": dashboard_ticks,
        },
        "trading": {
            "symbol": symbol,
            "strategy_mode": "standard_live",
            "candle_timeframe": "M15",
            "positions_count": len(positions),
            "status": reason,
            "is_tradeable": is_trade,
            "decision_reason": reason,
            "failed_node": str(getattr(strategy_result, "failed_node", "") or ""),
            "planned_direction": planned_direction,
            "entry_price": float(getattr(strategy_result, "entry_price", 0.0) or 0.0),
            "stop_loss": float(getattr(strategy_result, "stop_loss", 0.0) or 0.0),
            "take_profit": float(getattr(strategy_result, "take_profit", 0.0) or 0.0),
            "spread": spread,
            "target_value": float(getattr(strategy_result, "take_profit", 0.0) or 0.0),
            "target_progress": confluence_score,
            "volatility_short_atr": float(getattr(volatility_state, "short_atr", 0.0) or 0.0) if volatility_state else 0.0,
            "history": execution_history,
        },
    }


def _dashboard_quant_payload(*, candles, ticks, account_info, quant_decision=None) -> dict:
    prices = [float(getattr(candle, "close", 0.0) or 0.0) for candle in candles or []]
    prices = [price for price in prices if price > 0.0]
    balance = float(getattr(account_info, "balance", 0.0) or 0.0) if account_info is not None else 0.0
    metrics = resolve_quant_metrics(prices, ticks or [], balance)
    payload = {
        "hurst": float(metrics.hurst_exponent),
        "reversion": float(metrics.ou_reversion_speed),
        "ofi": float(metrics.order_flow_imbalance),
        "kelly_lot": float(metrics.kelly_suggested_lot),
        "z_score": float(metrics.z_score),
        "smoothness": float(metrics.autocorrelation),
        "valid": bool(metrics.is_mathematically_valid),
        "sample_count": len(prices),
    }
    if quant_decision is not None:
        payload.update(
            {
                "omega": float(getattr(quant_decision, "omega_t", 0.0) or 0.0),
                "sharpe": float(getattr(quant_decision, "sharpe_signal", 0.0) or 0.0),
                "drawdown_dampener": float(getattr(quant_decision, "drawdown_dampener", 0.0) or 0.0),
            }
        )
    return payload


def _dashboard_ticks(raw_ticks, tick_data: dict, timestamp: datetime) -> list[dict]:
    ticks = []
    for tick in raw_ticks or []:
        bid = float(getattr(tick, "bid", 0.0) if not isinstance(tick, dict) else tick.get("bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) if not isinstance(tick, dict) else tick.get("ask", 0.0) or 0.0)
        tick_time = getattr(tick, "time", None) if not isinstance(tick, dict) else tick.get("time")
        if hasattr(tick_time, "timestamp"):
            tick_time = int(tick_time.timestamp())
        ticks.append({"time": int(tick_time or timestamp.timestamp()), "bid": bid, "ask": ask})
    if not ticks and tick_data:
        ticks.append(
            {
                "time": int(timestamp.timestamp()),
                "bid": float(tick_data.get("bid", 0.0) or 0.0),
                "ask": float(tick_data.get("ask", 0.0) or 0.0),
            }
        )
    return ticks[-100:]


def _dashboard_execution_history(*, positions, strategy_result, timestamp: datetime) -> list[dict]:
    rows = []
    for position in positions or []:
        ticket = getattr(position, "ticket", "open")
        profit = float(getattr(position, "profit", 0.0) or 0.0)
        volume = float(getattr(position, "volume", 0.0) or 0.0)
        entry = float(getattr(position, "entry_price", getattr(position, "price_open", 0.0)) or 0.0)
        rows.append(
            {
                "time": timestamp.isoformat(),
                "ticket": ticket,
                "profit": round(profit, 2),
                "reason": f"OPEN lot={volume:.2f} entry={entry:.2f}",
            }
        )
    if not rows:
        reason = str(getattr(strategy_result, "reason", "waiting") or "waiting")
        rows.append(
            {
                "time": timestamp.isoformat(),
                "ticket": "-",
                "profit": 0.0,
                "reason": f"WAIT {reason}",
            }
        )
    return rows[-20:]


def _append_jsonl(path: str | Path, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _score_payload(score_decision) -> dict:
    if score_decision is None:
        return {}
    return {
        "edge": float(getattr(score_decision, "edge", 0.0) or 0.0),
        "threshold": float(getattr(score_decision, "threshold", 0.0) or 0.0),
        "expected_move_multiple": float(getattr(score_decision, "expected_move_multiple", 0.0) or 0.0),
        "uncertainty_penalty": float(getattr(score_decision, "uncertainty_penalty", 0.0) or 0.0),
    }


def _quant_payload(quant_decision) -> dict:
    if quant_decision is None:
        return {}
    return {
        "is_trade": bool(getattr(quant_decision, "is_trade", False)),
        "action": int(getattr(quant_decision, "action", 0) or 0),
        "reason": str(getattr(quant_decision, "reason", "") or ""),
        "omega_t": float(getattr(quant_decision, "omega_t", 0.0) or 0.0),
        "lot_multiplier": float(getattr(quant_decision, "lot_multiplier", 0.0) or 0.0),
        "drawdown_dampener": float(getattr(quant_decision, "drawdown_dampener", 0.0) or 0.0),
        "sharpe_signal": float(getattr(quant_decision, "sharpe_signal", 0.0) or 0.0),
    }


def _fusion_payload(fusion_decision) -> dict:
    if fusion_decision is None:
        return {}
    return {
        "action": str(getattr(fusion_decision, "action", "WAIT")),
        "score": float(getattr(fusion_decision, "score", 0.0) or 0.0),
        "confidence": float(getattr(fusion_decision, "confidence", 0.0) or 0.0),
        "lot_multiplier": float(getattr(fusion_decision, "lot_multiplier", 0.0) or 0.0),
        "reason": str(getattr(fusion_decision, "reason", "") or ""),
        "hard_block": bool(getattr(fusion_decision, "hard_block", False)),
        "components": dict(getattr(fusion_decision, "components", {}) or {}),
    }


def _diagnostic_payload(
    *,
    symbol: str,
    live_input,
    strategy_result,
    positions,
    quant_decision=None,
    fusion_decision=None,
    execution_assessment=None,
    effective_lot: float = 0.0,
    event: str,
) -> dict:
    latest_m15 = (getattr(live_input, "m15_candles", None) or [None])[-1]
    timestamp = getattr(latest_m15, "timestamp", None) or datetime.now(timezone.utc)
    if getattr(timestamp, "tzinfo", None) is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)

    metadata = getattr(strategy_result, "metadata", {}) or {}
    direction = getattr(strategy_result, "direction", None)
    payload = {
        "timestamp": timestamp.isoformat(),
        "event": event,
        "symbol": symbol,
        "positions_count": len(positions),
        "lot": float(effective_lot or 0.0),
        "strategy": {
            "is_trade": bool(getattr(strategy_result, "is_trade", False)),
            "reason": str(getattr(strategy_result, "reason", "") or ""),
            "failed_node": str(getattr(strategy_result, "failed_node", "") or ""),
            "direction": direction.value if direction is not None else "None",
            "entry_price": float(getattr(strategy_result, "entry_price", 0.0) or 0.0),
            "stop_loss": float(getattr(strategy_result, "stop_loss", 0.0) or 0.0),
            "take_profit": float(getattr(strategy_result, "take_profit", 0.0) or 0.0),
            "score": _score_payload(metadata.get("score_decision")),
        },
        "market": {
            "spread": float(getattr(live_input, "spread", 0.0) or 0.0),
            "m5_direction": _latest_candle_direction(getattr(live_input, "m5_candles", [])),
            "m15_direction": _latest_candle_direction(getattr(live_input, "m15_candles", [])),
            "h1_direction": _latest_candle_direction(getattr(live_input, "h1_candles", [])),
        },
        "quant": _quant_payload(quant_decision),
        "fusion": _fusion_payload(fusion_decision),
    }
    if execution_assessment is not None:
        payload["execution"] = {
            "is_tradeable": bool(getattr(execution_assessment, "is_tradeable", False)),
            "reason": str(getattr(execution_assessment, "reason", "") or ""),
            "effective_rr": float(getattr(execution_assessment, "effective_rr", 0.0) or 0.0),
            "recommended_lot_multiplier": float(
                getattr(execution_assessment, "recommended_lot_multiplier", 0.0) or 0.0
            ),
        }
    return payload


def _write_live_diagnostic(settings, payload: dict, log_fn) -> None:
    if not settings or not getattr(settings, "diagnostics_enabled", True):
        return
    try:
        _append_jsonl(getattr(settings, "diagnostics_log_path", "data/live_diagnostics.jsonl"), payload)
    except Exception as exc:
        log_fn(f"DIAGNOSTIC LOG ERROR {payload.get('symbol', 'unknown')} reason={exc}")


def _write_paper_trade(settings, payload: dict, log_fn) -> None:
    if not settings or not getattr(settings, "paper_trade_enabled", True):
        return
    try:
        _append_jsonl(getattr(settings, "paper_trade_log_path", "data/paper_trades.jsonl"), payload)
    except Exception as exc:
        log_fn(f"PAPER TRADE LOG ERROR {payload.get('symbol', 'unknown')} reason={exc}")


def _latest_candle_timestamp(candle) -> str:
    timestamp = getattr(candle, "timestamp", None) or datetime.now(timezone.utc)
    if getattr(timestamp, "tzinfo", None) is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)
    return timestamp.isoformat()


def _paper_trade_state_from_payload(payload: dict) -> dict:
    trade_id = uuid4().hex
    strategy = payload.get("strategy", {})
    direction = str(strategy.get("direction", "None"))
    entry_price = float(strategy.get("entry_price", 0.0) or 0.0)
    stop_loss = float(strategy.get("stop_loss", 0.0) or 0.0)
    take_profit = float(strategy.get("take_profit", 0.0) or 0.0)
    state = {
        "paper_trade_id": trade_id,
        "symbol": payload.get("symbol", ""),
        "direction": direction,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "lot": float(payload.get("lot", 0.0) or 0.0),
        "opened_at": payload.get("timestamp"),
        "max_favorable": 0.0,
        "max_adverse": 0.0,
    }
    return state


def _update_open_paper_trades(settings, open_trades: list[dict], live_input, log_fn) -> None:
    if not open_trades:
        return
    latest_m5 = (getattr(live_input, "m5_candles", None) or [None])[-1]
    if latest_m5 is None:
        return

    high = float(getattr(latest_m5, "high", 0.0) or 0.0)
    low = float(getattr(latest_m5, "low", 0.0) or 0.0)
    close = float(getattr(latest_m5, "close", 0.0) or 0.0)
    checked_at = _latest_candle_timestamp(latest_m5)
    remaining = []

    for trade in open_trades:
        if trade.get("symbol") and trade.get("symbol") != getattr(live_input, "symbol", ""):
            remaining.append(trade)
            continue
        direction = str(trade.get("direction", "None"))
        entry = float(trade.get("entry_price", 0.0) or 0.0)
        stop_loss = float(trade.get("stop_loss", 0.0) or 0.0)
        take_profit = float(trade.get("take_profit", 0.0) or 0.0)
        outcome = None
        exit_price = close

        if direction == BreakoutDirection.BULLISH.value:
            trade["max_favorable"] = max(float(trade.get("max_favorable", 0.0) or 0.0), high - entry)
            trade["max_adverse"] = max(float(trade.get("max_adverse", 0.0) or 0.0), entry - low)
            if stop_loss and low <= stop_loss:
                outcome = "SL"
                exit_price = stop_loss
            elif take_profit and high >= take_profit:
                outcome = "TP"
                exit_price = take_profit
        elif direction == BreakoutDirection.BEARISH.value:
            trade["max_favorable"] = max(float(trade.get("max_favorable", 0.0) or 0.0), entry - low)
            trade["max_adverse"] = max(float(trade.get("max_adverse", 0.0) or 0.0), high - entry)
            if stop_loss and high >= stop_loss:
                outcome = "SL"
                exit_price = stop_loss
            elif take_profit and low <= take_profit:
                outcome = "TP"
                exit_price = take_profit

        if outcome is None:
            remaining.append(trade)
            continue

        risk = abs(entry - stop_loss) if stop_loss else 0.0
        pnl_points = exit_price - entry if direction == BreakoutDirection.BULLISH.value else entry - exit_price
        close_payload = {
            "event": "paper_trade_close",
            "paper_trade_id": trade.get("paper_trade_id"),
            "symbol": trade.get("symbol"),
            "direction": direction,
            "opened_at": trade.get("opened_at"),
            "closed_at": checked_at,
            "outcome": outcome,
            "entry_price": entry,
            "exit_price": float(exit_price),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "lot": float(trade.get("lot", 0.0) or 0.0),
            "pnl_points": float(pnl_points),
            "r_multiple": float(pnl_points / risk) if risk > 0 else 0.0,
            "max_favorable": float(trade.get("max_favorable", 0.0) or 0.0),
            "max_adverse": float(trade.get("max_adverse", 0.0) or 0.0),
        }
        _write_paper_trade(settings, close_payload, log_fn)
        log_fn(
            f"PAPER TRADE CLOSED {trade.get('symbol')} id={trade.get('paper_trade_id')} "
            f"outcome={outcome} r={close_payload['r_multiple']:.2f} "
            f"mfe={close_payload['max_favorable']:.2f} mae={close_payload['max_adverse']:.2f}"
        )

    open_trades[:] = remaining


def _latest_candle_direction(candles) -> str:
    if not candles:
        return "None"
    candle = candles[-1]
    open_price = float(getattr(candle, "open", getattr(candle, "close", 0.0)) or 0.0)
    close_price = float(getattr(candle, "close", 0.0) or 0.0)
    if close_price > open_price:
        return BreakoutDirection.BULLISH.value
    if close_price < open_price:
        return BreakoutDirection.BEARISH.value
    return "None"


def _candle_to_training_dict(candle) -> dict:
    timestamp = getattr(candle, "timestamp", None)
    if hasattr(timestamp, "timestamp"):
        candle_time = int(timestamp.timestamp())
    else:
        candle_time = 0
    return {
        "time": candle_time,
        "open": float(getattr(candle, "open", getattr(candle, "close", 0.0)) or 0.0),
        "high": float(getattr(candle, "high", getattr(candle, "close", 0.0)) or 0.0),
        "low": float(getattr(candle, "low", getattr(candle, "close", 0.0)) or 0.0),
        "close": float(getattr(candle, "close", 0.0) or 0.0),
    }


# ---------------------------------------------------------------------------
# Main live loop
# ---------------------------------------------------------------------------

def _check_correlation_limit(
    executor,
    symbols: list[str],
    new_symbol: str,
    new_direction,
    base_lot: float,
    log_fn,
) -> bool:
    """
    Returns True if the correlation-aligned exposure limit is exceeded, blocking the trade.
    """
    CORRELATION_MATRIX = {
        ("XAUUSD", "EURUSD"): 0.82,
        ("XAUUSD", "GBPUSD"): 0.78,
        ("EURUSD", "GBPUSD"): 0.88,
        ("XAUUSD", "USDIndex"): -0.85,
        ("EURUSD", "USDIndex"): -0.92,
        ("GBPUSD", "USDIndex"): -0.89,
    }
    
    all_positions = []
    for sym in symbols:
        try:
            positions = executor.list_bot_positions(sym)
            if positions:
                all_positions.extend(positions)
        except Exception:
            pass
            
    if not all_positions:
        return False
        
    def get_corr(s1, s2):
        s1, s2 = s1.upper(), s2.upper()
        if s1 == s2:
            return 1.0
        if (s1, s2) in CORRELATION_MATRIX:
            return CORRELATION_MATRIX[(s1, s2)]
        if (s2, s1) in CORRELATION_MATRIX:
            return CORRELATION_MATRIX[(s2, s1)]
        return 0.0

    new_sign = 1 if str(getattr(new_direction, "value", new_direction)) == "BULLISH" else -1
    aligned_exposure = 0.0
    
    for pos in all_positions:
        pos_sym = getattr(pos, "symbol", "")
        if not pos_sym:
            continue
        pos_type = getattr(pos, "type", 0)
        pos_sign = 1 if pos_type == 0 else -1
        pos_volume = float(getattr(pos, "volume", 0.0) or 0.0)
        
        corr = get_corr(new_symbol, pos_sym)
        aligned_exposure += pos_volume * corr * pos_sign * new_sign
        
    limit = 2.5 * base_lot
    if aligned_exposure + base_lot > limit:
        log_fn(
            f"CORRELATION GATING BLOCKED {new_symbol} directional exposure={aligned_exposure:.4f} "
            f"limit={limit:.4f} (over-leveraging risk)"
        )
        return True
        
    return False


def run_live_signal_loop(
    *,
    mt5_module,
    executor,
    symbol: str,
    lot: float,
    add_on_lot_increment: float = 0.01,
    campaign_max_exposure_pct: float = 10.0,
    risk_buffer: float,
    max_candles_since_breakout: int,
    poll_seconds: int,
    max_loops: int | None = None,
    tradingview_alert_store=None,
    orderflow_signal_store=None,
    strategy_profile=None,
    reload_check_fn=None,
    sleep_fn=sleep,
    log_fn=print,
    settings=None,
) -> None:
    # Initialize quant engine components if settings provided and enabled
    symbols = [s.strip() for s in symbol.split(",") if s.strip()]
    if not symbols:
        log_fn("No active symbols specified for trading loop.")
        return

    quant_enabled = False
    quant_params = None
    equity_tracker = None
    ml_classifier = None
    local_edge_model = _build_local_edge_model(settings, log_fn)

    if settings is not None and getattr(settings, "quant_enabled", False):
        quant_enabled = True
        quant_params = _build_quant_params(settings)
        equity_tracker = _build_equity_tracker(settings)
        ml_classifier = _build_ml_classifier(settings)
        log_fn(
            f"QUANT ENGINE ACTIVE γ={quant_params.gamma} "
            f"Ω_threshold={quant_params.omega_threshold} "
            f"CVaR_α={quant_params.cvar_alpha} "
            f"DD_max={quant_params.dd_max} "
            f"r_max={quant_params.position_r_max}"
        )
        if ml_classifier and ml_classifier.is_loaded:
            log_fn("ML REGIME CLASSIFIER ACTIVE")
        else:
            log_fn("ML REGIME CLASSIFIER DISABLED (rule-based fallback)")

    if orderflow_signal_store is None:
        try:
            from src.strategy.orderflow import OrderflowSignalStore

            orderflow_signal_store = OrderflowSignalStore()
        except Exception:
            orderflow_signal_store = None

    _live_orderflow_engines = {}
    feature_extractors = {}
    strategy_profiles = {}

    for sym in symbols:
        profile_key = sym.upper()
        profile = None
        if settings and hasattr(settings, "strategy_profiles"):
            profile = settings.strategy_profiles.get(sym) or settings.strategy_profiles.get(profile_key)
        if profile is None:
            profile = strategy_profile
        strategy_profiles[sym] = profile
        try:
            from src.strategy.orderflow_engine import LiveOrderflowEngine

            _live_orderflow_engines[sym] = LiveOrderflowEngine(sym, window=500)
            log_fn(f"ORDERFLOW ENGINE ACTIVE {sym} (MT5 tick-based, no external feed needed)")
        except Exception as _oe_err:
            log_fn(f"ORDERFLOW ENGINE DISABLED {sym} reason={_oe_err}")

        if settings is not None and getattr(settings, "quant_enabled", False):
            feature_extractors[sym] = _build_feature_extractor(settings)

    loop_count = 0
    open_paper_trades: list[dict] = []
    while max_loops is None or loop_count < max_loops:
        import os
        if os.path.exists("panic.signal"):
            log_fn("!!! PANIC SIGNAL DETECTED - CLOSING ALL ACTIVE POSITIONS !!!")
            for sym in symbols:
                all_pos = executor.list_bot_positions(sym, comment_prefix="strategy-live")
                for p in all_pos:
                    try:
                        executor.close_position(p, comment="PANIC-EXIT")
                        log_fn(f"Panic closed position ticket={getattr(p, 'ticket', 'unknown')} for {sym}")
                    except Exception as e:
                        log_fn(f"Panic close failed for ticket={getattr(p, 'ticket', 'unknown')} on {sym}: {e}")
            try:
                os.remove("panic.signal")
            except Exception:
                pass
            log_fn("Panic exit complete. Bot paused for 30s.")
            sleep_fn(30)
            continue

        for sym in symbols:
            try:
                live_input = build_live_strategy_input(mt5_module, sym)
            except RuntimeError as exc:
                log_fn(f"LIVE DATA WAIT {sym} reason={exc}")
                continue

            _update_open_paper_trades(settings, open_paper_trades, live_input, log_fn)
            tradingview_alert = None
            if tradingview_alert_store is not None:
                tradingview_alert = tradingview_alert_store.latest_for(
                    sym,
                    now=getattr(live_input.m15_candles[-1], "timestamp", None),
                )

            _live_orderflow_engine = _live_orderflow_engines.get(sym)
            if _live_orderflow_engine is not None and orderflow_signal_store is not None:
                try:
                    from src.quick_scalp_loop import fetch_recent_ticks
                    from src.strategy.orderflow import parse_orderflow_payload

                    raw_ticks = fetch_recent_ticks(mt5_module, sym, count=500)
                    added = _live_orderflow_engine.ingest(raw_ticks)
                    if added > 0:
                        payload = _live_orderflow_engine.to_signal_payload()
                        if payload is not None:
                            sig = parse_orderflow_payload(payload)
                            orderflow_signal_store.record(sig)
                            log_fn(
                                f"ORDERFLOW {sym} "
                                f"ticks={len(_live_orderflow_engine._ticks)} "
                                f"delta={payload['delta']:.2f} "
                                f"cvd_slope={payload['cvd_slope']:.3f} "
                                f"vwap={payload.get('vwap', 0):.2f} "
                                f"vwap_bias={payload['vwap_bias']:.3f} "
                                f"imbalance={payload['imbalance']:.3f} "
                                f"profile_osc={payload.get('profile_location', 0):.3f} "
                                f"POC={payload.get('poc', 0):.2f} "
                                f"VAH={payload.get('vah', 0):.2f} "
                                f"VAL={payload.get('val', 0):.2f} "
                                f"vol_mom={payload.get('volume_momentum', 0):.3f}"
                            )
                except Exception as _of_poll_err:
                    import logging as _logging
                    _logging.warning("ORDERFLOW POLL WARN %s reason=%s", sym, _of_poll_err)

            orderflow_signal = None
            if orderflow_signal_store is not None:
                orderflow_signal = orderflow_signal_store.latest_for(
                    sym,
                    now=getattr(live_input.m15_candles[-1], "timestamp", None),
                )

            strategy_profile = strategy_profiles.get(sym)
            strategy_result = evaluate_top_down_decision_tree(
                symbol=sym,
                d1_candles=live_input.d1_candles,
                h4_candles=live_input.h4_candles,
                h1_candles=live_input.h1_candles,
                m30_candles=live_input.m30_candles,
                m15_candles=live_input.m15_candles,
                m10_candles=getattr(live_input, "m10_candles", None),
                m5_candles=getattr(live_input, "m5_candles", None),
                risk_buffer=risk_buffer,
                tradingview_alert=tradingview_alert,
                strategy_profile=strategy_profile,
                orderflow_signal=orderflow_signal,
            )

            campaign_positions = executor.list_bot_positions(sym)

            trail_distance = getattr(strategy_profile, "trail_distance", 1.0) if strategy_profile else 1.0
            current_price = float(live_input.m15_candles[-1].close)

            for pos in campaign_positions:
                new_sl = evaluate_fixed_trailing_stop(
                    position=pos,
                    current_price=current_price,
                    direction=_campaign_direction(pos),
                    trail_distance=trail_distance
                )
                if new_sl is not None:
                    try:
                        executor.update_position_stop_loss(pos, new_sl)
                        log_fn(f"TRAILING {sym} ticket={getattr(pos, 'ticket', 'unknown')} new_sl={new_sl:.2f}")
                    except Exception as e:
                        log_fn(f"TRAILING ERR {sym} ticket={getattr(pos, 'ticket', 'unknown')} reason={e}")

            campaign_margin_snapshot = _build_margin_snapshot(
                mt5_module=getattr(executor, "mt5_module", None),
                symbol=sym,
                positions=campaign_positions,
                direction=_campaign_direction(campaign_positions[-1]) if campaign_positions else BreakoutDirection.BULLISH,
                default_lot=lot,
                add_on_lot_increment=add_on_lot_increment,
            )

            quant_decision = None
            effective_lot = lot
            local_edge_probability = None
            local_edge_lot_multiplier = 1.0
            fusion_decision = None
            strategy_result_for_execution = strategy_result  # default; may be overridden below

            feature_extractor = feature_extractors.get(sym)
            if quant_enabled and feature_extractor is not None and equity_tracker is not None:
                account_info = mt5_module.account_info() if hasattr(mt5_module, "account_info") else None
                current_equity = float(getattr(account_info, "equity", 10000.0) or 10000.0) if account_info else 10000.0
                equity_snapshot = equity_tracker.update(current_equity)
                equity_log_path = getattr(settings, "equity_log_path", None) if settings else None
                if equity_log_path:
                    try:
                        from src.strategy.equity_tracker import append_equity_snapshot_to_file

                        append_equity_snapshot_to_file(equity_snapshot, equity_log_path)
                    except Exception:
                        pass

                spread = getattr(live_input, "spread", 0.0)
                quant_trade_stats = _estimate_strategy_trade_statistics(
                    strategy_result=strategy_result,
                    live_input=live_input,
                    spread=spread,
                    requested_lot=lot,
                    campaign_exposure_pct=float(campaign_margin_snapshot["campaign_exposure_pct"]),
                    commission_per_lot=getattr(settings, "hfm_commission_per_lot", 6.0),
                )
                features, expected_return, return_std = _extract_features_from_strategy(
                    live_input=live_input,
                    strategy_result=strategy_result,
                    feature_extractor=feature_extractor,
                    spread=spread,
                    expected_return_override=quant_trade_stats["expected_return"],
                    return_std_override=quant_trade_stats["return_std"],
                    orderflow_signal=orderflow_signal,
                )
                local_edge_allows = True
                if local_edge_model is not None:
                    if feature_extractor.snapshot_count < 5:
                        log_fn(
                            f"LOCAL EDGE {sym} WARMUP samples={feature_extractor.snapshot_count}/5"
                            f" — skipping inference until buffers are populated"
                        )
                    else:
                        try:
                            local_edge_allows, local_edge_probability = local_edge_model.allows_trade(
                                features, drawdown_ratio=equity_tracker.drawdown_ratio
                            )
                            local_edge_lot_multiplier = _local_edge_lot_multiplier(local_edge_probability, settings)
                            log_fn(
                                f"LOCAL EDGE {sym} p_win={local_edge_probability:.3f} "
                                f"threshold={local_edge_model.threshold:.3f} allow={local_edge_allows} "
                                f"lot_mult={local_edge_lot_multiplier:.2f}"
                            )
                        except Exception as exc:
                            log_fn(f"LOCAL EDGE WARN {sym} reason={exc}")


                if settings and getattr(settings, "feature_logging_enabled", False):
                    try:
                        from src.strategy.features import append_enriched_snapshot_to_file

                        _meta = quant_decision.metadata if quant_decision else {}
                        _vst = (getattr(strategy_result, "metadata", {}) or {}).get("volatility_state")
                        _atr_val = float(getattr(_vst, "short_atr", spread or 1.0)) if _vst else float(spread or 1.0)
                        append_enriched_snapshot_to_file(
                            features,
                            settings.feature_log_path,
                            # Quant decision
                            quant_is_trade=bool(quant_decision.is_trade) if quant_decision else False,
                            quant_action=int(quant_decision.action) if quant_decision else 0,
                            omega_t=float(quant_decision.omega_t) if quant_decision else 0.0,
                            kelly_fraction=float(_meta.get("kelly_fraction", 0.0)),
                            ce_score_trade=float((_meta.get("ce_scores") or {}).get(1, 0.0)),
                            ce_score_flat=float((_meta.get("ce_scores") or {}).get(0, 0.0)),
                            sharpe_signal=float(quant_decision.sharpe_signal) if quant_decision else 0.0,
                            drawdown_dampener=float(quant_decision.drawdown_dampener) if quant_decision else 1.0,
                            lot_multiplier=float(quant_decision.lot_multiplier) if quant_decision else 1.0,
                            # Trade economics
                            transaction_cost=float(quant_trade_stats["transaction_cost"]),
                            win_rate=float(quant_trade_stats["win_rate"]),
                            avg_win=float(quant_trade_stats["avg_win"]),
                            avg_loss=float(quant_trade_stats["avg_loss"]),
                            # Session & macro
                            session_score=session_score,
                            dxy_trend=dxy_trend,
                            drawdown_ratio=float(equity_tracker.drawdown_ratio),
                            # Execution context
                            spread=float(spread),
                            atr=_atr_val,
                            lot_requested=float(effective_lot),
                            commission_per_lot=float(getattr(settings, "hfm_commission_per_lot", 6.0)),
                            current_equity=float(current_equity),
                            # outcome_label left at -1 sentinel; patched on trade close
                        )
                    except Exception:
                        pass

                recent_returns = list(quant_trade_stats["recent_returns"])

                session_eng = SessionEngine()
                session_score = session_eng.compute_session_score(datetime.now(timezone.utc))

                dxy_trend = 0.0
                try:
                    from src.market_data import fetch_candles
                    dxy_candles = fetch_candles(mt5_module, "USDIndex", "TIMEFRAME_M15", 5)
                    if dxy_candles and len(dxy_candles) >= 5:
                        dxy_trend = (dxy_candles[-1].close - dxy_candles[0].close) / dxy_candles[0].close * 1000.0
                except Exception:
                    pass

                from src.strategy.quant_engine import evaluate_master_equation

                quant_decision = evaluate_master_equation(
                    features=features,
                    params=quant_params,
                    equity=current_equity,
                    drawdown_ratio=equity_tracker.drawdown_ratio,
                    recent_returns=recent_returns,
                    transaction_cost=float(quant_trade_stats["transaction_cost"]),
                    win_rate=float(quant_trade_stats["win_rate"]),
                    avg_win=float(quant_trade_stats["avg_win"]),
                    avg_loss=float(quant_trade_stats["avg_loss"]),
                    continuation_context=quant_trade_stats.get("continuation_context"),
                    session_score=session_score,
                    dxy_trend=dxy_trend,
                )

                log_fn(f"QUANT {sym} {_format_quant_metrics(quant_decision, log_fn)}")

                WARMUP_MIN_SAMPLES = 5
                if feature_extractor.snapshot_count < WARMUP_MIN_SAMPLES:
                    # ── WARMUP HARD BLOCK ─────────────────────────────────
                    # Previously this just logged and let effective_lot stay at
                    # DEFAULT_TRADE_LOT (0.1 on gold = ~$10/point), which allowed
                    # a full-risk trade to fire on every restart until 5 samples
                    # accumulated.  That is the primary cause of the spike-and-
                    # collapse pattern observed in the live equity log.
                    # We now block the trade entirely for this loop iteration.
                    effective_lot = 0.01  # absolute safe minimum; never inherits default
                    log_fn(
                        f"QUANT WARMUP BLOCKED {sym} samples={feature_extractor.snapshot_count}/"
                        f"{WARMUP_MIN_SAMPLES} — no trade until buffers populated"
                    )
                    # Override strategy decision so execution layer cannot open a position
                    strategy_result_for_execution = TopDownNoTrade(
                        is_trade=False,
                        reason="quant_warmup_block",
                        failed_node="quant_engine",
                        metadata={"warmup_samples": feature_extractor.snapshot_count,
                                  "warmup_required": WARMUP_MIN_SAMPLES},
                    )
                else:
                    if quant_decision.is_trade:
                        from src.strategy.equity_tracker import compute_position_size, compute_price_risk_per_lot

                        symbol_info = mt5_module.symbol_info(sym) if hasattr(mt5_module, "symbol_info") else None
                        price_risk_per_lot = (
                            compute_price_risk_per_lot(
                                entry_price=float(strategy_result.entry_price),
                                stop_loss=float(strategy_result.stop_loss),
                                symbol_info=symbol_info,
                            )
                            if symbol_info is not None and getattr(strategy_result, "is_trade", False)
                            else 0.0
                        )

                        effective_lot = compute_position_size(
                            equity=current_equity,
                            win_rate=float(quant_trade_stats["win_rate"]),
                            avg_win=float(quant_trade_stats["avg_win"]),
                            avg_loss=float(quant_trade_stats["avg_loss"]),
                            omega_t=quant_decision.omega_t,
                            r_max=quant_params.position_r_max,
                            volume_min=0.01,
                            volume_step=0.01,
                            price_per_lot=price_risk_per_lot,
                        )
                        effective_lot *= max(float(getattr(quant_decision, "lot_multiplier", 1.0) or 1.0), 0.0)
                        log_fn(
                            f"QUANT LOT {sym} kelly_lot={effective_lot:.4f} "
                            f"Ω_t={quant_decision.omega_t:.4f} "
                            f"DD_damp={quant_decision.drawdown_dampener:.4f}"
                        )
            else:
                local_edge_allows = True

            # strategy_result_for_execution was initialised above; the warmup
            # block may have already set it to a NoTrade — don't overwrite it.
            # If quant is disabled or warmup didn't fire, it stays as strategy_result.
            if settings is not None and getattr(settings, "fusion_enabled", True):
                try:
                    from src.strategy.decision_fusion import fuse_decision
                    strategy_trade = getattr(strategy_result, "is_trade", False)

                    fusion_decision = fuse_decision(
                        strategy_result=strategy_result,
                        live_input=live_input,
                        quant_decision=quant_decision,
                        local_edge_probability=local_edge_probability,
                        orderflow_signal=orderflow_signal,
                        features=features if quant_enabled and feature_extractor is not None else None,
                        base_lot_multiplier=local_edge_lot_multiplier,
                        settings=settings,
                    )
                    log_fn(
                        f"DECISION {sym} action={fusion_decision.action} "
                        f"score={fusion_decision.score:.3f} confidence={fusion_decision.confidence:.3f} "
                        f"confluence={fusion_decision.components.get('confluence', 0.0):.3f} "
                        f"support={int(fusion_decision.components.get('support_count', 0.0))} "
                        f"conflict={int(fusion_decision.components.get('conflict_count', 0.0))} "
                        f"threshold={fusion_decision.components.get('adjusted_threshold', 0.0):.3f} "
                        f"lot_mult={fusion_decision.lot_multiplier:.2f} reason={fusion_decision.reason}"
                    )
                    if not fusion_decision.is_trade:
                        if strategy_trade:
                            reason = fusion_decision.reason
                            failed_node = "decision_fusion"
                            if reason == "quant_direction_mismatch":
                                failed_node = "quant_engine"
                            strategy_result_for_execution = TopDownNoTrade(
                                is_trade=False,
                                reason=reason,
                                failed_node=failed_node,
                                metadata={"fusion_decision": _fusion_payload(fusion_decision)},
                            )
                        else:
                            strategy_result_for_execution = TopDownNoTrade(
                                is_trade=False,
                                reason=strategy_result.reason,
                                failed_node=getattr(strategy_result, "failed_node", "decision_tree"),
                                metadata={
                                    **(getattr(strategy_result, "metadata", {}) or {}),
                                    "fusion_decision": _fusion_payload(fusion_decision),
                                },
                            )
                    elif fusion_decision.action == "BUY" and strategy_result.direction is not BreakoutDirection.BULLISH:
                        strategy_result_for_execution = TopDownNoTrade(
                            is_trade=False,
                            reason="fusion_direction_mismatch",
                            failed_node="decision_fusion",
                            metadata={"fusion_decision": _fusion_payload(fusion_decision)},
                        )
                    elif fusion_decision.action == "SELL" and strategy_result.direction is not BreakoutDirection.BEARISH:
                        strategy_result_for_execution = TopDownNoTrade(
                            is_trade=False,
                            reason="fusion_direction_mismatch",
                            failed_node="decision_fusion",
                            metadata={"fusion_decision": _fusion_payload(fusion_decision)},
                        )
                    else:
                        effective_lot = max(effective_lot * fusion_decision.lot_multiplier, 0.01)
                except Exception as exc:
                    log_fn(f"DECISION FUSION WARN {sym} reason={exc}")
            elif getattr(strategy_result, "is_trade", False) and not local_edge_allows:
                strategy_result_for_execution = TopDownNoTrade(
                    is_trade=False,
                    reason=f"local_edge_low_probability_{local_edge_probability:.3f}",
                    failed_node="local_edge_model",
                    metadata={"local_edge_probability": local_edge_probability},
                )

            if strategy_result_for_execution.is_trade:
                if _check_correlation_limit(
                    executor=executor,
                    symbols=symbols,
                    new_symbol=sym,
                    new_direction=strategy_result_for_execution.direction,
                    base_lot=lot,
                    log_fn=log_fn,
                ):
                    strategy_result_for_execution = TopDownNoTrade(
                        is_trade=False,
                        reason="correlation_limit_exceeded",
                        failed_node="correlation_gate",
                        metadata={"aligned_exposure_limit": 2.5 * lot},
                    )

            dashboard_state = _build_standard_training_state(
                mt5_module=mt5_module,
                symbol=sym,
                live_input=live_input,
                strategy_result=strategy_result_for_execution,
                positions=campaign_positions,
                quant_decision=quant_decision,
            )
            save_training_snapshot(dashboard_state)
            save_bot_state(dashboard_state)

            if campaign_positions:
                try:
                    _handle_campaign_action(
                        executor=executor,
                        symbol=sym,
                        lot=effective_lot,
                        add_on_lot_increment=add_on_lot_increment,
                        campaign_max_exposure_pct=campaign_max_exposure_pct,
                        risk_buffer=risk_buffer,
                        live_input=live_input,
                        positions=campaign_positions,
                        strategy_result=strategy_result_for_execution,
                        strategy_profile=strategy_profile,
                        log_fn=log_fn,
                    )
                except Exception as exc:
                    log_fn(f"LIVE CAMPAIGN ERROR {sym} reason={exc}")
            elif strategy_result_for_execution.is_trade:
                execution_assessment = _assess_live_execution(
                    live_input=live_input,
                    strategy_result=strategy_result_for_execution,
                    requested_lot=effective_lot,
                    campaign_exposure_pct=0.0,
                )
                diagnostic_payload = _diagnostic_payload(
                    symbol=sym,
                    live_input=live_input,
                    strategy_result=strategy_result_for_execution,
                    positions=campaign_positions,
                    quant_decision=quant_decision,
                    fusion_decision=fusion_decision,
                    execution_assessment=execution_assessment,
                    effective_lot=effective_lot,
                    event="trade_candidate",
                )
                _write_live_diagnostic(settings, diagnostic_payload, log_fn)
                if not execution_assessment.is_tradeable:
                    log_fn(
                        f"LIVE NO TRADE {sym} reason={execution_assessment.reason} "
                        f"node=execution_engine rr={execution_assessment.effective_rr:.3f}"
                    )
                    diagnostic_payload["event"] = "no_trade"
                    _write_live_diagnostic(settings, diagnostic_payload, log_fn)
                else:
                    adjusted_lot = max(effective_lot * execution_assessment.recommended_lot_multiplier, 0.01)
                    paper_payload = dict(diagnostic_payload)
                    paper_payload["event"] = "paper_trade_open"
                    paper_payload["lot"] = float(adjusted_lot)
                    paper_state = _paper_trade_state_from_payload(paper_payload)
                    paper_payload["paper_trade_id"] = paper_state["paper_trade_id"]
                    _write_paper_trade(settings, paper_payload, log_fn)
                    open_paper_trades.append(paper_state)
                    log_fn(
                        f"PAPER TRADE OPEN {sym} id={paper_state['paper_trade_id']} "
                        f"direction={strategy_result_for_execution.direction.value} "
                        f"entry={strategy_result_for_execution.entry_price} sl={strategy_result_for_execution.stop_loss} "
                        f"tp={strategy_result_for_execution.take_profit} lot={adjusted_lot}"
                    )
                    try:
                        position = executor.open_strategy_trade(
                            symbol=sym,
                            direction=strategy_result_for_execution.direction,
                            lot=adjusted_lot,
                            stop_loss=strategy_result_for_execution.stop_loss,
                            take_profit=strategy_result_for_execution.take_profit,
                            comment="strategy-live",
                        )
                    except Exception as exc:
                        log_fn(f"LIVE ORDER REJECTED {sym} reason={exc}")
                    else:
                        remember_position_initial_stop_loss(
                            position,
                            strategy_result_for_execution.stop_loss,
                            entry_price=strategy_result_for_execution.entry_price,
                        )
                        quant_info = ""
                        if quant_decision:
                            quant_info = f" Ω={quant_decision.omega_t:.3f} lot={adjusted_lot}"
                        log_fn(
                            f"LIVE TRADE OPENED {sym} ticket={getattr(position, 'ticket', 'unknown')} "
                            f"direction={strategy_result_for_execution.direction.value} entry={strategy_result_for_execution.entry_price} "
                            f"sl={strategy_result_for_execution.stop_loss} tp={strategy_result_for_execution.take_profit}{quant_info}"
                        )
            else:
                _write_live_diagnostic(
                    settings,
                    _diagnostic_payload(
                        symbol=sym,
                        live_input=live_input,
                        strategy_result=strategy_result_for_execution,
                        positions=campaign_positions,
                        quant_decision=quant_decision,
                        fusion_decision=fusion_decision,
                        effective_lot=effective_lot,
                        event="no_trade",
                    ),
                    log_fn,
                )
                log_fn(_format_no_trade(sym, strategy_result_for_execution))

        loop_count += 1
        if (
            settings is not None
            and getattr(settings, "local_edge_online_train_enabled", False)
            and loop_count % int(getattr(settings, "local_edge_online_train_interval_loops", 30) or 30) == 0
        ):
            refreshed_model = _run_online_edge_training(settings, log_fn)
            if refreshed_model is not None:
                local_edge_model = refreshed_model
        if reload_check_fn is not None and reload_check_fn():
            log_fn(f"CODE CHANGE DETECTED reloading bot")
            return "reload_requested"
        if max_loops is not None and loop_count >= max_loops:
            break
        sleep_fn(poll_seconds)

def _handle_campaign_action(
    *,
    executor,
    symbol,
    lot,
    add_on_lot_increment,
    campaign_max_exposure_pct,
    risk_buffer,
    live_input,
    positions,
    strategy_result,
    strategy_profile,
    log_fn,
) -> None:
    current_price = float(live_input.m15_candles[-1].close)
    direction = _campaign_direction(positions[-1])
    latest_trade_r_multiple = _latest_trade_r_multiple(positions[-1], current_price, direction)
    action = evaluate_campaign_action(
        positions=positions,
        current_price=current_price,
        direction=direction,
        latest_trade_r_multiple=latest_trade_r_multiple,
        default_lot=lot,
        add_on_lot_increment=add_on_lot_increment,
        max_exposure_pct=campaign_max_exposure_pct,
        margin_snapshot=_build_margin_snapshot(
            mt5_module=getattr(executor, "mt5_module", None),
            symbol=symbol,
            positions=positions,
            direction=direction,
            default_lot=lot,
            add_on_lot_increment=add_on_lot_increment,
        ),
        reversal_confirmed=_is_reversal_confirmed(
            live_input=live_input,
            direction=direction,
            strategy_result=strategy_result,
        ),
        continuation_edge=_continuation_edge(strategy_result),
        continuation_threshold=_continuation_threshold(strategy_result),
        breakeven_distance=float(getattr(strategy_profile, "breakeven_distance", 1.5) or 1.5),
        campaign_add_floor_r=float(getattr(strategy_profile, "campaign_add_trigger_floor_r", 1.25) or 1.25),
    )

    if action.action == "trail_all":
        updated_positions = 0
        stop_updates = list(action.stop_updates)
        if not stop_updates and action.new_stop_loss is not None:
            stop_updates = [(index, action.new_stop_loss) for index, _ in enumerate(positions)]
        for index, stop_loss in stop_updates:
            position = positions[index]
            if _should_update_stop_loss(position, stop_loss, direction):
                executor.update_position_stop_loss(position, stop_loss)
                updated_positions += 1
        if updated_positions:
            log_fn(
                f"LIVE CAMPAIGN TRAIL {symbol} positions={len(positions)} "
                f"updated={updated_positions} reason={action.reason} "
                f"new_sl={action.new_stop_loss if action.new_stop_loss is not None else action.stop_updates[-1][1]}"
            )
        else:
            log_fn(f"LIVE CAMPAIGN HOLD {symbol} positions={len(positions)} reason=campaign_stop_already_tighter")
        return

    if action.action == "add_position" and action.add_lot is not None:
        strategy_metadata = getattr(strategy_result, "metadata", {}) or {}
        add_decision = evaluate_campaign_add(
            symbol=symbol,
            live_input=live_input,
            direction=direction,
            risk_buffer=risk_buffer,
            latest_trade_r_multiple=latest_trade_r_multiple,
            continuation_edge=_continuation_edge(strategy_result),
            continuation_threshold=_continuation_threshold(strategy_result),
            strategy_profile=strategy_profile,
            quant_decision=strategy_metadata.get("quant_decision"),
        )
        if (
            not add_decision.is_ready
            or add_decision.entry_price is None
            or add_decision.stop_loss is None
            or add_decision.take_profit is None
            or add_decision.direction is None
        ):
            log_fn(f"LIVE CAMPAIGN HOLD {symbol} positions={len(positions)} reason={add_decision.reason}")
            return
        execution_assessment = _assess_live_execution(
            live_input=live_input,
            strategy_result=add_decision,
            requested_lot=action.add_lot,
            campaign_exposure_pct=float(action.metadata.get("campaign_exposure_pct", 0.0))
            if getattr(action, "metadata", None)
            else 0.0,
        )
        if not execution_assessment.is_tradeable:
            log_fn(
                f"LIVE CAMPAIGN HOLD {symbol} positions={len(positions)} "
                f"reason={execution_assessment.reason}"
            )
            return
        adjusted_add_lot = max(
            action.add_lot * add_decision.lot_multiplier * execution_assessment.recommended_lot_multiplier,
            0.01,
        )
        position = executor.open_strategy_trade(
            symbol=symbol,
            direction=add_decision.direction,
            lot=adjusted_add_lot,
            stop_loss=add_decision.stop_loss,
            take_profit=add_decision.take_profit,
            comment="strategy-live",
        )
        log_fn(
            f"LIVE CAMPAIGN ADD {symbol} ticket={getattr(position, 'ticket', 'unknown')} "
            f"direction={add_decision.direction.value} entry={add_decision.entry_price} "
            f"sl={add_decision.stop_loss} tp={add_decision.take_profit} lot={adjusted_add_lot} "
            f"reason={add_decision.reason}"
        )
        return

    if action.action == "close_all":
        for position in positions:
            executor.close_position(position, comment="strategy-live-reversal-exit")
        log_fn(f"LIVE CAMPAIGN EXIT {symbol} positions={len(positions)} reason={action.reason}")
        return

    log_fn(f"LIVE CAMPAIGN HOLD {symbol} positions={len(positions)} reason={action.reason}")


def _latest_trade_r_multiple(position, current_price: float, direction: BreakoutDirection) -> float:
    entry_price = _position_entry_price(position)
    initial_stop_loss = _position_initial_stop_loss(position, entry_price)
    risk = abs(entry_price - initial_stop_loss)
    if risk == 0:
        return 0.0
    if direction is BreakoutDirection.BULLISH:
        return (current_price - entry_price) / risk
    return (entry_price - current_price) / risk


def _campaign_direction(position) -> BreakoutDirection:
    order_type = getattr(position, "type", None)
    if order_type is not None:
        return BreakoutDirection.BULLISH if int(order_type) == 0 else BreakoutDirection.BEARISH
    entry_price = _position_entry_price(position)
    initial_stop_loss = _position_initial_stop_loss(position, entry_price)
    if entry_price >= initial_stop_loss:
        return BreakoutDirection.BULLISH
    return BreakoutDirection.BEARISH


def _build_margin_snapshot(*, mt5_module, symbol, positions, direction, default_lot, add_on_lot_increment):
    zero_snapshot = {
        "campaign_exposure_pct": 0.0,
        "preferred_add_exposure_pct": 0.0,
        "fallback_add_exposure_pct": 0.0,
    }
    if mt5_module is None or not hasattr(mt5_module, "account_info"):
        return zero_snapshot
    account_info = mt5_module.account_info()
    if account_info is None:
        return zero_snapshot
    equity = float(getattr(account_info, "equity", 0.0) or 0.0)
    if equity <= 0:
        return zero_snapshot
    tick_getter = getattr(mt5_module, "symbol_info_tick", None)
    margin_calculator = getattr(mt5_module, "order_calc_margin", None)
    if tick_getter is None or margin_calculator is None:
        return zero_snapshot
    tick = tick_getter(symbol)
    if tick is None:
        return zero_snapshot

    campaign_margin = sum(
        _estimate_position_margin(
            mt5_module=mt5_module,
            symbol=symbol,
            tick=tick,
            position=position,
            direction=direction,
        )
        for position in positions
    )
    preferred_add_margin = _estimate_order_margin(
        mt5_module=mt5_module,
        symbol=symbol,
        tick=tick,
        direction=direction,
        lot=round(default_lot + add_on_lot_increment, 8),
    )
    fallback_add_margin = _estimate_order_margin(
        mt5_module=mt5_module,
        symbol=symbol,
        tick=tick,
        direction=direction,
        lot=default_lot,
    )
    return {
        "campaign_exposure_pct": (campaign_margin / equity) * 100.0,
        "preferred_add_exposure_pct": (preferred_add_margin / equity) * 100.0,
        "fallback_add_exposure_pct": (fallback_add_margin / equity) * 100.0,
    }


def _continuation_edge(strategy_result) -> float | None:
    metadata = getattr(strategy_result, "metadata", {}) or {}
    score_decision = metadata.get("score_decision")
    if score_decision is None:
        return None
    edge = getattr(score_decision, "edge", None)
    return float(edge) if edge is not None else None


def _continuation_threshold(strategy_result) -> float | None:
    metadata = getattr(strategy_result, "metadata", {}) or {}
    score_decision = metadata.get("score_decision")
    if score_decision is None:
        return None
    threshold = getattr(score_decision, "threshold", None)
    if threshold is None:
        return None
    symbol_profile = metadata.get("symbol_profile")
    multiplier = float(getattr(symbol_profile, "add_on_edge_multiplier", 1.0) or 1.0)
    return float(threshold) * multiplier


def _format_no_trade(symbol: str, result: TopDownNoTrade | TopDownTradePlan) -> str:
    failed_node = getattr(result, "failed_node", None)
    if failed_node is not None:
        return f"LIVE NO TRADE {symbol} reason={result.reason} node={failed_node}"
    return f"LIVE NO TRADE {symbol} reason={result.reason}"


def _estimate_position_margin(*, mt5_module, symbol, tick, position, direction: BreakoutDirection) -> float:
    volume = float(getattr(position, "volume", 0.0) or 0.0)
    if volume <= 0:
        return 0.0
    order_type = getattr(position, "type", None)
    if order_type is None:
        order_type = _order_type_for_direction(mt5_module, direction)
    price = _price_for_order_type(mt5_module, tick, order_type)
    margin = mt5_module.order_calc_margin(order_type, symbol, volume, price)
    return float(margin or 0.0)


def _estimate_order_margin(*, mt5_module, symbol, tick, direction: BreakoutDirection, lot: float) -> float:
    if lot <= 0:
        return 0.0
    order_type = _order_type_for_direction(mt5_module, direction)
    price = _price_for_order_type(mt5_module, tick, order_type)
    margin = mt5_module.order_calc_margin(order_type, symbol, float(lot), price)
    return float(margin or 0.0)


def _order_type_for_direction(mt5_module, direction: BreakoutDirection):
    if direction is BreakoutDirection.BULLISH:
        return getattr(mt5_module, "ORDER_TYPE_BUY", 0)
    return getattr(mt5_module, "ORDER_TYPE_SELL", 1)


def _price_for_order_type(mt5_module, tick, order_type) -> float:
    if order_type == getattr(mt5_module, "ORDER_TYPE_SELL", 1):
        return float(getattr(tick, "bid"))
    return float(getattr(tick, "ask"))


def _should_update_stop_loss(position, new_stop_loss: float, direction: BreakoutDirection) -> bool:
    current_stop_loss = _position_current_stop_loss(position)
    if current_stop_loss is None:
        return True
    if current_stop_loss == 0.0:
        return True
    if direction is BreakoutDirection.BULLISH:
        return new_stop_loss > current_stop_loss
    return new_stop_loss < current_stop_loss


def _is_reversal_confirmed(*, live_input, direction: BreakoutDirection, strategy_result) -> bool:
    if getattr(strategy_result, "is_trade", False):
        return strategy_result.direction is not direction
    return all(
        _timeframe_reversal(candles, direction)
        for candles in (live_input.m15_candles, live_input.m30_candles, live_input.h1_candles)
    )


def _timeframe_reversal(candles, direction: BreakoutDirection) -> bool:
    if len(candles) < 2:
        return False
    latest = candles[-1]
    previous = candles[-2]
    required_fields = ("open", "close", "high", "low")
    if any(not hasattr(latest, field) for field in required_fields):
        return False
    if any(not hasattr(previous, field) for field in required_fields):
        return False
    latest_open = float(latest.open)
    latest_close = float(latest.close)
    if direction is BreakoutDirection.BULLISH:
        return latest_close < latest_open and latest_close < float(previous.low)
    return latest_close > latest_open and latest_close > float(previous.high)


def _position_entry_price(position) -> float:
    if hasattr(position, "entry_price"):
        return float(position.entry_price)
    if hasattr(position, "price_open"):
        return float(position.price_open)
    raise AttributeError("Position object is missing entry price fields")


def _position_current_stop_loss(position) -> float | None:
    if hasattr(position, "stop_loss"):
        return float(position.stop_loss)
    if hasattr(position, "sl"):
        return float(position.sl)
    return None


def _position_initial_stop_loss(position, entry_price: float) -> float:
    from src.strategy.management import _position_initial_stop_loss as resolve_initial_stop_loss

    return resolve_initial_stop_loss(position, entry_price)
