import warnings

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

# RankWarning moved to np.exceptions in NumPy 2.x
_RankWarning = getattr(getattr(np, "exceptions", np), "RankWarning", UserWarning)

@dataclass
class QuantMetrics:
    hurst_exponent: float
    ou_reversion_speed: float
    order_flow_imbalance: float
    kelly_suggested_lot: float
    z_score: float
    autocorrelation: float
    is_mathematically_valid: bool

def calculate_hurst_exponent(prices: List[float]) -> float:
    """Calculate the Hurst Exponent to determine trend persistence."""
    if len(prices) < 20:
        return 0.5

    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag]))) for lag in lags]

    # Filter out zero/negative tau values before log to avoid divide-by-zero
    valid = [(lag, t) for lag, t in zip(lags, tau) if t > 0]
    if len(valid) < 2:
        return 0.5  # Not enough data — assume random walk

    log_lags = np.log([lag for lag, _ in valid])
    log_tau  = np.log([t   for _, t  in valid])

    # Guard against poorly-conditioned fit (all lags identical after filtering)
    if np.ptp(log_lags) < 1e-10:
        return 0.5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", _RankWarning)
        poly = np.polyfit(log_lags, log_tau, 1)
    return float(np.clip(poly[0] * 2.0, 0.0, 2.0))

def calculate_ou_params(prices: List[float]):
    """Estimate Ornstein-Uhlenbeck parameters (Reversion Speed)."""
    if len(prices) < 20:
        return 0.0

    # Simple linear regression on the differences
    # dP = lambda * (mu - P) * dt + sigma * dW
    p = np.array(prices, dtype=np.float64)
    x = p[:-1]
    y = p[1:]

    # Guard: polyfit is poorly conditioned when x has near-zero variance
    if np.std(x) < 1e-10:
        return 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", _RankWarning)
        poly = np.polyfit(x, y, 1)
    # The 'reversion speed' (lambda) is related to the slope
    # slope = 1 - lambda * dt
    reversion_speed = 1 - poly[0]
    return max(0, reversion_speed)

def calculate_order_flow_imbalance(ticks: List[dict]) -> float:
    """Calculate Order Flow Imbalance (OFI) from tick stream."""
    if not ticks:
        return 0.0
    
    buy_vol = 0
    sell_vol = 0
    
    for i in range(1, len(ticks)):
        curr = ticks[i]
        prev = ticks[i-1]
        
        # Price improvement logic (standard OFI calculation)
        if curr['bid'] > prev['bid']: buy_vol += 1
        elif curr['bid'] < prev['bid']: sell_vol += 1
        
        if curr['ask'] > prev['ask']: buy_vol += 1
        elif curr['ask'] < prev['ask']: sell_vol += 1
            
    total = buy_vol + sell_vol
    if total == 0: return 0.0
    return (buy_vol - sell_vol) / total

def calculate_kelly_lot(balance: float, win_rate: float, avg_win: float, avg_loss: float, min_lot: float = 0.01) -> float:
    """Calculate optimal position size using Kelly Criterion."""
    if avg_loss == 0: return min_lot
    
    # Kelly % = w - (1-w)/R where R is Reward/Risk ratio
    reward_risk = avg_win / avg_loss
    kelly_f = win_rate - (1 - win_rate) / reward_risk
    
    # We use "Fractional Kelly" (10% of Kelly) for safety on $50 account
    safe_f = max(0, kelly_f * 0.1)
    
    # On $50 account, 1% risk at 100 pips SL is ~0.01 lot
    # This is a simplified mapping for micro accounts
    suggested = (balance * safe_f) / 50.0 * 0.01
    return max(min_lot, round(suggested, 2))

def calculate_z_score(prices: List[float]) -> float:
    """Calculate Z-Score of the current price relative to the window."""
    if len(prices) < 20: return 0.0
    mean = np.mean(prices)
    std = np.std(prices)
    if std == 0: return 0.0
    return (prices[-1] - mean) / std

def calculate_autocorrelation(prices: List[float]) -> float:
    """Calculate 1-lag autocorrelation (Smoothness)."""
    if len(prices) < 20: return 0.0
    p = np.array(prices)
    return np.corrcoef(p[1:], p[:-1])[0, 1]

def resolve_quant_metrics(prices: List[float], ticks: List[dict], balance: float) -> QuantMetrics:
    """Run all pillars of the Advanced Quant Math Engine."""
    try:
        hurst = calculate_hurst_exponent(prices)
        reversion = calculate_ou_params(prices)
        ofi = calculate_order_flow_imbalance(ticks)
        z_score = calculate_z_score(prices)
        smoothness = calculate_autocorrelation(prices)
        
        kelly = calculate_kelly_lot(balance, 0.55, 0.75, 0.25)
        
        return QuantMetrics(
            hurst_exponent=round(hurst, 3),
            ou_reversion_speed=round(reversion, 3),
            order_flow_imbalance=round(ofi, 3),
            kelly_suggested_lot=kelly,
            z_score=round(z_score, 2),
            autocorrelation=round(smoothness, 3),
            is_mathematically_valid=True
        )
    except Exception as e:
        return QuantMetrics(0.5, 0.0, 0.0, 0.01, 0.0, 0.0, False)
