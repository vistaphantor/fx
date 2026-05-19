from __future__ import annotations

from datetime import datetime, timezone
from math import exp, sqrt
from time import sleep

from src.market_data import build_live_strategy_input
from src.quick_scalp_loop import save_training_snapshot
from src.strategy.breakout import BreakoutDirection
from src.strategy.campaign_add import evaluate_campaign_add
from src.strategy.decision_tree import TopDownNoTrade, TopDownTradePlan, evaluate_top_down_decision_tree
from src.strategy.execution_cost import assess_market_order_execution
from src.strategy.management import (
    evaluate_campaign_action,
    evaluate_fixed_trailing_stop,
    remember_position_initial_stop_loss,
)
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
        expected_return=expected_return,
        return_std=return_std,
        timestamp=getattr(live_input.m15_candles[-1], "timestamp", None),
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


def _normalize_transaction_cost(*, price: float, spread: float, commission_per_lot: float = 0.0, lot: float = 0.01) -> float:
    """Normalize total round-trip transaction cost into return space.

    Includes:
    - Spread (bid/ask cost, paid at entry and effectively again at exit)
    - Estimated slippage (10% of spread as a conservative buffer)
    - Commission round-trip converted to price units (commission_per_lot / lot_value)

    All divided by price so the result is comparable with expected_return / CVaR.
    """
    safe_price = max(float(price), 1e-9)
    safe_spread = max(float(spread), 0.0)
    # Slippage estimated at 10% of spread
    slippage_est = safe_spread * 0.10
    # Commission: round-trip per lot, converted to price-return units
    # commission_per_lot is in account currency per lot (e.g. $6 round-trip for HFM)
    # Approximate lot value = price (works for XAUUSD/forex as crude proxy)
    commission_cost = max(float(commission_per_lot), 0.0) / max(safe_price * max(float(lot), 0.01), 1e-9)
    return (safe_spread + slippage_est + commission_cost) / safe_price


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
) -> dict[str, object]:
    from src.strategy.breakout import BreakoutDirection
    from src.strategy.features import extract_expected_return

    current_price = float(live_input.m15_candles[-1].close)
    normalized_cost = _normalize_transaction_cost(price=current_price, spread=spread, lot=requested_lot)

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
    confidence_floor = 0.40 + (0.35 * confidence)
    win_rate = _clamp(max(breakeven_probability + 0.03, confidence_floor), 0.05, 0.95)
    loss_rate = 1.0 - win_rate

    trade_expectancy = max((win_rate * avg_win) - (loss_rate * avg_loss), 1e-6)
    variance = (
        (win_rate * ((avg_win - trade_expectancy) ** 2))
        + (loss_rate * (((-avg_loss) - trade_expectancy) ** 2))
    )
    return_std = max(sqrt(max(variance, 0.0)), 1e-9)

    direction_sign = 1.0 if strategy_result.direction is BreakoutDirection.BULLISH else -1.0
    recent_returns = _collect_directional_recent_returns(
        live_input.m15_candles,
        strategy_result.direction,
        lookback=50,
    )

    return {
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expected_return": direction_sign * trade_expectancy,
        "return_std": return_std,
        "recent_returns": recent_returns,
        "transaction_cost": normalized_cost,
        "continuation_context": {
            **continuation_context,
            "continuation_probability": execution_assessment.continuation_probability,
            "mu_cont": execution_assessment.continuation_mu,
            "cvar_dir": execution_assessment.directional_tail_proxy,
            "effective_rr": execution_assessment.effective_rr,
            "execution_penalty": execution_assessment.execution_penalty,
            "directional_tail_proxy": execution_assessment.directional_tail_proxy,
            "dynamic_rr_floor": execution_assessment.dynamic_rr_floor,
            "continuation_ev": execution_assessment.continuation_ev,
        }
        if continuation_context
        else None,
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
    quant_payload = {}
    if quant_decision is not None:
        quant_payload = {
            "hurst": 0.0,
            "reversion": 0.0,
            "ofi": float(getattr(quant_decision, "omega_t", 0.0) or 0.0),
            "kelly_lot": float(getattr(quant_decision, "lot_multiplier", 0.0) or 0.0),
            "z_score": float(getattr(quant_decision, "sharpe_signal", 0.0) or 0.0),
            "smoothness": float(getattr(quant_decision, "drawdown_dampener", 0.0) or 0.0),
        }

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
            "m1_candles": [_candle_to_training_dict(candle) for candle in (getattr(live_input, "m15_candles", []) or [])[-50:]],
            "ticks": [
                {
                    "time": int(timestamp.timestamp()),
                    "bid": float(tick_data.get("bid", 0.0) or 0.0),
                    "ask": float(tick_data.get("ask", 0.0) or 0.0),
                }
            ]
            if tick_data
            else [],
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
        },
    }


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
    quant_enabled = False
    quant_params = None
    feature_extractor = None
    equity_tracker = None
    ml_classifier = None

    if settings is not None and getattr(settings, "quant_enabled", False):
        quant_enabled = True
        quant_params = _build_quant_params(settings)
        feature_extractor = _build_feature_extractor(settings)
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

    # MT5-native orderflow engine: computes delta/CVD/VWAP/imbalance
    # directly from ticks — no external service required.
    _live_orderflow_engine = None
    try:
        from src.strategy.orderflow_engine import LiveOrderflowEngine

        _live_orderflow_engine = LiveOrderflowEngine(symbol, window=500)
        log_fn(f"ORDERFLOW ENGINE ACTIVE {symbol} (MT5 tick-based, no external feed needed)")
    except Exception as _oe_err:
        log_fn(f"ORDERFLOW ENGINE DISABLED {symbol} reason={_oe_err}")

    loop_count = 0
    while max_loops is None or loop_count < max_loops:
        live_input = build_live_strategy_input(mt5_module, symbol)
        tradingview_alert = None
        if tradingview_alert_store is not None:
            tradingview_alert = tradingview_alert_store.latest_for(
                symbol,
                now=getattr(live_input.m15_candles[-1], "timestamp", None),
            )
        # ── MT5 orderflow polling ─────────────────────────────────────────
        # Fetch latest ticks from MT5 and push into the engine every cycle.
        # The computed signal is stored in orderflow_signal_store so the
        # decision tree consumes it exactly as if it arrived via a webhook.
        if _live_orderflow_engine is not None and orderflow_signal_store is not None:
            try:
                from src.quick_scalp_loop import fetch_recent_ticks
                from src.strategy.orderflow import parse_orderflow_payload

                raw_ticks = fetch_recent_ticks(mt5_module, symbol, count=500)
                added = _live_orderflow_engine.ingest(raw_ticks)
                if added > 0:
                    payload = _live_orderflow_engine.to_signal_payload()
                    if payload is not None:
                        sig = parse_orderflow_payload(payload)
                        orderflow_signal_store.record(sig)
                        log_fn(
                            f"ORDERFLOW {symbol} "
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
                log_fn(f"ORDERFLOW POLL WARN {symbol} reason={_of_poll_err}")
        # ──────────────────────────────────────────────────────────────────

        orderflow_signal = None
        if orderflow_signal_store is not None:
            orderflow_signal = orderflow_signal_store.latest_for(
                symbol,
                now=getattr(live_input.m15_candles[-1], "timestamp", None),
            )
        strategy_result = evaluate_top_down_decision_tree(
            symbol=symbol,
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

        campaign_positions = executor.list_bot_positions(symbol)
        
        # --- 10-pip Fixed Trailing Stop Logic ---
        trail_distance = 1.0 # 10 pips in Gold
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
                    log_fn(f"TRAILING {symbol} ticket={getattr(pos, 'ticket', 'unknown')} new_sl={new_sl:.2f}")
                except Exception as e:
                    log_fn(f"TRAILING ERR {symbol} ticket={getattr(pos, 'ticket', 'unknown')} reason={e}")

        campaign_margin_snapshot = _build_margin_snapshot(
            mt5_module=getattr(executor, "mt5_module", None),
            symbol=symbol,
            positions=campaign_positions,
            direction=_campaign_direction(campaign_positions[-1]) if campaign_positions else BreakoutDirection.BULLISH,
            default_lot=lot,
            add_on_lot_increment=add_on_lot_increment,
        )

        # ---------------------------------------------------------------
        # Quant engine overlay
        # ---------------------------------------------------------------
        quant_decision = None
        effective_lot = lot

        if quant_enabled and feature_extractor is not None and equity_tracker is not None:
            # Update equity tracker
            account_info = mt5_module.account_info() if hasattr(mt5_module, "account_info") else None
            current_equity = float(getattr(account_info, "equity", 10000.0) or 10000.0) if account_info else 10000.0
            equity_snapshot = equity_tracker.update(current_equity)
            equity_log_path = getattr(settings, "equity_log_path", None) if settings else None
            if equity_log_path:
                try:
                    from src.strategy.equity_tracker import append_equity_snapshot_to_file

                    append_equity_snapshot_to_file(equity_snapshot, equity_log_path)
                except Exception:
                    pass  # Don't crash on logging failure

            # Extract features
            spread = getattr(live_input, "spread", 0.0)
            quant_trade_stats = _estimate_strategy_trade_statistics(
                strategy_result=strategy_result,
                live_input=live_input,
                spread=spread,
                requested_lot=lot,
                campaign_exposure_pct=float(campaign_margin_snapshot["campaign_exposure_pct"]),
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

            # Log features if enabled
            if settings and getattr(settings, "feature_logging_enabled", False):
                try:
                    from src.strategy.features import append_snapshot_to_file

                    append_snapshot_to_file(features, settings.feature_log_path)
                except Exception:
                    pass  # Don't crash on logging failure

            # Collect recent returns for CVaR
            recent_returns = list(quant_trade_stats["recent_returns"])

            # Session and DXY Correlation Logic
            session_eng = SessionEngine()
            session_score = session_eng.compute_session_score(datetime.now(timezone.utc))
            
            dxy_trend = 0.0
            try:
                # Fetch USDIndex M15 candles to compute trend
                from src.market_data import fetch_candles
                dxy_candles = fetch_candles(mt5_module, "USDIndex", "TIMEFRAME_M15", 5)
                if dxy_candles and len(dxy_candles) >= 5:
                    dxy_trend = (dxy_candles[-1].close - dxy_candles[0].close) / dxy_candles[0].close * 1000.0 # Normalized
            except Exception:
                pass

            # Run master equation
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

            # Log quant metrics
            log_fn(f"QUANT {symbol} {_format_quant_metrics(quant_decision, log_fn)}")

            # Warm-up bypass: z-scores are meaningless with < 10 samples.
            # During warm-up, log metrics but pass through strategy decision.
            WARMUP_MIN_SAMPLES = 5
            if feature_extractor.snapshot_count < WARMUP_MIN_SAMPLES:
                log_fn(
                    f"QUANT WARMUP {symbol} samples={feature_extractor.snapshot_count}/"
                    f"{WARMUP_MIN_SAMPLES} — passing through strategy decision"
                )
                strategy_result_for_execution = strategy_result
            else:
                # Compute effective lot from quant engine position sizing
                if quant_decision.is_trade:
                    from src.strategy.equity_tracker import compute_position_size, compute_price_risk_per_lot

                    symbol_info = mt5_module.symbol_info(symbol) if hasattr(mt5_module, "symbol_info") else None
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
                        volume_min=lot,  # Use configured lot as floor
                        volume_step=0.01,
                        price_per_lot=price_risk_per_lot or float("inf"),
                    )
                    effective_lot *= max(float(getattr(quant_decision, "lot_multiplier", 1.0) or 1.0), 0.0)
                    log_fn(
                        f"QUANT LOT {symbol} kelly_lot={effective_lot:.4f} "
                        f"Ω_t={quant_decision.omega_t:.4f} "
                        f"DD_damp={quant_decision.drawdown_dampener:.4f}"
                    )

                if not strategy_result.is_trade:
                    strategy_result_for_execution = strategy_result
                elif not quant_decision.is_trade:
                    log_fn(
                        f"QUANT OVERRIDE {symbol} strategy_wanted_trade "
                        f"but quant_says_flat reason={quant_decision.reason}"
                    )
                    strategy_result_for_execution = TopDownNoTrade(
                        is_trade=False,
                        reason=quant_decision.reason,
                        failed_node="quant_engine",
                        metadata={"quant_decision": quant_decision},
                    )
                else:
                    if quant_decision.action == 1 and strategy_result.direction is BreakoutDirection.BULLISH:
                        strategy_result_for_execution = strategy_result
                    elif quant_decision.action == -1 and strategy_result.direction is BreakoutDirection.BEARISH:
                        strategy_result_for_execution = strategy_result
                    else:
                        log_fn(
                            f"QUANT DIRECTION {symbol} quant_action={quant_decision.action} "
                            f"strategy_direction={strategy_result.direction.value}"
                        )
                        strategy_result_for_execution = TopDownNoTrade(
                            is_trade=False,
                            reason="quant_direction_mismatch",
                            failed_node="quant_engine",
                            metadata={"quant_decision": quant_decision},
                        )
        else:
            strategy_result_for_execution = strategy_result

        save_training_snapshot(
            _build_standard_training_state(
                mt5_module=mt5_module,
                symbol=symbol,
                live_input=live_input,
                strategy_result=strategy_result_for_execution,
                positions=campaign_positions,
                quant_decision=quant_decision,
            )
        )

        if campaign_positions:
            try:
                _handle_campaign_action(
                    executor=executor,
                    symbol=symbol,
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
                log_fn(f"LIVE CAMPAIGN ERROR {symbol} reason={exc}")
        elif strategy_result_for_execution.is_trade:
            execution_assessment = _assess_live_execution(
                live_input=live_input,
                strategy_result=strategy_result_for_execution,
                requested_lot=effective_lot,
                campaign_exposure_pct=0.0,
            )
            if not execution_assessment.is_tradeable:
                log_fn(
                    f"LIVE NO TRADE {symbol} reason={execution_assessment.reason} "
                    f"node=execution_engine rr={execution_assessment.effective_rr:.3f}"
                )
            else:
                adjusted_lot = max(effective_lot * execution_assessment.recommended_lot_multiplier, 0.01)
                try:
                    position = executor.open_strategy_trade(
                        symbol=symbol,
                        direction=strategy_result_for_execution.direction,
                        lot=adjusted_lot,
                        stop_loss=strategy_result_for_execution.stop_loss,
                        take_profit=strategy_result_for_execution.take_profit,
                        comment="strategy-live",
                    )
                except Exception as exc:
                    log_fn(f"LIVE ORDER REJECTED {symbol} reason={exc}")
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
                        f"LIVE TRADE OPENED {symbol} ticket={getattr(position, 'ticket', 'unknown')} "
                        f"direction={strategy_result_for_execution.direction.value} entry={strategy_result_for_execution.entry_price} "
                        f"sl={strategy_result_for_execution.stop_loss} tp={strategy_result_for_execution.take_profit}{quant_info}"
                    )
        else:
            log_fn(_format_no_trade(symbol, strategy_result_for_execution))

        loop_count += 1
        if reload_check_fn is not None and reload_check_fn():
            log_fn(f"CODE CHANGE DETECTED {symbol} reloading bot")
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
