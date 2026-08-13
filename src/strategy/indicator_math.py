from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True, slots=True)
class IndicatorMathPack:
    structure_score: float = 0.0
    volatility_score: float = 0.0
    momentum_score: float = 0.0
    trend_score: float = 0.0
    orderflow_volume_score: float = 0.0
    risk_score: float = 0.0
    statistical_score: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def build_indicator_math_pack(*, live_input, orderflow_signal=None) -> IndicatorMathPack:
    m1 = getattr(live_input, "m1_candles", []) or []
    m5 = getattr(live_input, "m5_candles", []) or []
    m15 = getattr(live_input, "m15_candles", []) or []
    m30 = getattr(live_input, "m30_candles", []) or []
    h1 = getattr(live_input, "h1_candles", []) or []
    candles = m15 or m5 or m1
    return IndicatorMathPack(
        structure_score=_structure_score(m5=m5, m15=m15, h1=h1),
        volatility_score=_volatility_score(candles),
        momentum_score=_momentum_score(candles),
        trend_score=_trend_score(m5=m5, m15=m15, m30=m30, h1=h1),
        orderflow_volume_score=_orderflow_volume_score(candles, orderflow_signal),
        risk_score=_risk_score(live_input, candles),
        statistical_score=_statistical_score(candles),
    )


def _structure_score(*, m5: list[Any], m15: list[Any], h1: list[Any]) -> float:
    candles = m15 if len(m15) >= 20 else m5
    if len(candles) < 12:
        return 0.0
    highs = [float(c.high) for c in candles[-20:]]
    lows = [float(c.low) for c in candles[-20:]]
    close = float(candles[-1].close)
    prev_close = float(candles[-2].close)
    recent_high = max(highs[:-1])
    recent_low = min(lows[:-1])
    breakout = 0.0
    if close > recent_high:
        breakout = 1.0
    elif close < recent_low:
        breakout = -1.0
    sweep = 0.0
    if float(candles[-1].high) > recent_high and close < recent_high:
        sweep = -0.7
    elif float(candles[-1].low) < recent_low and close > recent_low:
        sweep = 0.7
    magnet = _clamp((close - ((recent_high + recent_low) / 2.0)) / max(recent_high - recent_low, 1e-9), -1.0, 1.0)
    h1_bias = _slope_score(h1[-8:]) if len(h1) >= 8 else 0.0
    return _clamp((0.45 * breakout) + (0.30 * sweep) + (0.15 * magnet) + (0.10 * h1_bias), -1.0, 1.0)


def _volatility_score(candles: list[Any]) -> float:
    if len(candles) < 25:
        return 0.0
    atr_values = _atr_series(candles, period=14)
    if len(atr_values) < 10:
        return 0.0
    atr = atr_values[-1]
    atr_percentile = _percentile_rank(atr_values[-40:], atr)
    closes = [float(c.close) for c in candles[-20:]]
    stdev = _std(closes)
    mid = sum(closes) / len(closes)
    boll_width = (4.0 * stdev) / max(abs(mid), 1e-9)
    range_now = abs(float(candles[-1].high) - float(candles[-1].low))
    avg_range = sum(abs(float(c.high) - float(c.low)) for c in candles[-20:]) / 20.0
    expansion = range_now / max(avg_range, 1e-9)
    compression_break = _clamp((expansion - 1.0) / 2.0, -1.0, 1.0)
    return _clamp((0.45 * ((atr_percentile - 0.5) * 2.0)) + (0.25 * compression_break) - (0.30 * min(boll_width * 200.0, 1.0)), -1.0, 1.0)


def _momentum_score(candles: list[Any]) -> float:
    if len(candles) < 35:
        return 0.0
    closes = [float(c.close) for c in candles]
    rsi = _rsi(closes[-15:])
    prev_rsi = _rsi(closes[-20:-5])
    rsi_slope = _clamp((rsi - prev_rsi) / 25.0, -1.0, 1.0)
    macd_hist = _macd_histogram(closes)
    macd_slope = _clamp((macd_hist[-1] - macd_hist[-4]) / max(abs(closes[-1]) * 0.0004, 1e-9), -1.0, 1.0) if len(macd_hist) >= 4 else 0.0
    roc_fast = (closes[-1] - closes[-4]) / max(abs(closes[-4]), 1e-9)
    roc_slow = (closes[-1] - closes[-12]) / max(abs(closes[-12]), 1e-9)
    acceleration = _clamp((roc_fast - roc_slow) * 800.0, -1.0, 1.0)
    body_eff = _body_efficiency(candles[-1])
    body_sign = 1.0 if candles[-1].close >= candles[-1].open else -1.0
    return _clamp((0.30 * rsi_slope) + (0.35 * macd_slope) + (0.25 * acceleration) + (0.10 * body_eff * body_sign), -1.0, 1.0)


def _trend_score(*, m5: list[Any], m15: list[Any], m30: list[Any], h1: list[Any]) -> float:
    scores = []
    for candles, weight in ((m5, 0.15), (m15, 0.35), (m30, 0.25), (h1, 0.25)):
        if len(candles) < 30:
            continue
        closes = [float(c.close) for c in candles]
        ema20 = _ema(closes[-30:], 20)
        ema50 = _ema(closes[-60:], 50) if len(closes) >= 60 else _ema(closes, min(30, len(closes)))
        slope = _slope_score(candles[-12:])
        adx = _adx_proxy(candles[-20:])
        alignment = 1.0 if closes[-1] > ema20 > ema50 else -1.0 if closes[-1] < ema20 < ema50 else 0.0
        scores.append(weight * _clamp((0.55 * alignment) + (0.30 * slope) + (0.15 * adx * (1 if slope >= 0 else -1)), -1.0, 1.0))
    return _clamp(sum(scores), -1.0, 1.0)


def _orderflow_volume_score(candles: list[Any], orderflow_signal=None) -> float:
    volume_score = 0.0
    if len(candles) >= 20:
        volumes = [float(getattr(c, "volume", 0.0) or 0.0) for c in candles[-20:]]
        volume_z = (volumes[-1] - (sum(volumes) / len(volumes))) / max(_std(volumes), 1.0)
        body_sign = 1.0 if candles[-1].close >= candles[-1].open else -1.0
        volume_score = _clamp((volume_z / 3.0) * body_sign, -1.0, 1.0)
    orderflow_score = 0.0
    if orderflow_signal is not None:
        delta = float(getattr(orderflow_signal, "delta_bias", 0.0) or 0.0)
        cvd = float(getattr(orderflow_signal, "cvd_slope", 0.0) or 0.0)
        imbalance = float(getattr(orderflow_signal, "imbalance_score", 0.0) or 0.0)
        vwap = float(getattr(orderflow_signal, "vwap_alignment", 0.0) or 0.0)
        orderflow_score = _clamp((0.30 * delta) + (0.30 * cvd) + (0.25 * imbalance) + (0.15 * vwap), -1.0, 1.0)
    return _clamp((0.45 * volume_score) + (0.55 * orderflow_score), -1.0, 1.0)


def _risk_score(live_input, candles: list[Any]) -> float:
    if len(candles) < 20:
        return 0.0
    spread = abs(float(getattr(live_input, "spread", 0.0) or 0.0))
    atr = _atr_series(candles, period=14)[-1]
    spread_risk = _clamp(spread / max(atr, 1e-9), 0.0, 1.0)
    ranges = [abs(float(c.high) - float(c.low)) for c in candles[-20:]]
    range_risk = _clamp(ranges[-1] / max(sum(ranges) / len(ranges), 1e-9) - 1.0, 0.0, 1.0)
    wick_risk = 1.0 - _body_efficiency(candles[-1])
    return _clamp((0.45 * spread_risk) + (0.30 * range_risk) + (0.25 * wick_risk), 0.0, 1.0)


def _statistical_score(candles: list[Any]) -> float:
    if len(candles) < 40:
        return 0.0
    closes = [float(c.close) for c in candles[-50:]]
    returns = [(closes[i] - closes[i - 1]) / max(abs(closes[i - 1]), 1e-9) for i in range(1, len(closes))]
    z_price = _clamp((closes[-1] - (sum(closes[-20:]) / 20.0)) / max(_std(closes[-20:]), 1e-9) / 3.0, -1.0, 1.0)
    autocorr = _autocorr(returns[-30:])
    entropy = _entropy_score(returns[-30:])
    sharpe = _clamp((_mean(returns[-20:]) / max(_std(returns[-20:]), 1e-9)) / 2.0, -1.0, 1.0)
    return _clamp((0.30 * z_price) + (0.25 * autocorr) + (0.25 * sharpe) - (0.20 * entropy), -1.0, 1.0)


def _atr_series(candles: list[Any], period: int = 14) -> list[float]:
    if len(candles) < 2:
        return [0.0]
    trs = []
    for index in range(1, len(candles)):
        high = float(candles[index].high)
        low = float(candles[index].low)
        prev_close = float(candles[index - 1].close)
        trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    values = []
    for index in range(len(trs)):
        window = trs[max(0, index - period + 1) : index + 1]
        values.append(sum(window) / len(window))
    return values


def _rsi(closes: list[float]) -> float:
    if len(closes) < 3:
        return 50.0
    gains = []
    losses = []
    for idx in range(1, len(closes)):
        diff = closes[idx] - closes[idx - 1]
        gains.append(max(diff, 0.0))
        losses.append(abs(min(diff, 0.0)))
    avg_gain = sum(gains) / len(gains)
    avg_loss = sum(losses) / max(len(losses), 1)
    if avg_loss <= 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _macd_histogram(closes: list[float]) -> list[float]:
    if len(closes) < 35:
        return []
    ema12 = _ema_series(closes, 12)
    ema26 = _ema_series(closes, 26)
    macd = [a - b for a, b in zip(ema12, ema26)]
    signal = _ema_series(macd, 9)
    return [a - b for a, b in zip(macd, signal)]


def _ema(values: list[float], period: int) -> float:
    series = _ema_series(values, period)
    return series[-1] if series else 0.0


def _ema_series(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (period + 1.0)
    output = [float(values[0])]
    for value in values[1:]:
        output.append((float(value) * alpha) + (output[-1] * (1.0 - alpha)))
    return output


def _adx_proxy(candles: list[Any]) -> float:
    if len(candles) < 5:
        return 0.0
    movement = abs(float(candles[-1].close) - float(candles[0].close))
    ranges = sum(abs(float(c.high) - float(c.low)) for c in candles)
    return _clamp(movement / max(ranges, 1e-9), 0.0, 1.0)


def _slope_score(candles: list[Any]) -> float:
    if len(candles) < 2:
        return 0.0
    closes = [float(c.close) for c in candles]
    avg_range = sum(abs(float(c.high) - float(c.low)) for c in candles) / len(candles)
    return _clamp((closes[-1] - closes[0]) / max(avg_range * len(candles), 1e-9), -1.0, 1.0)


def _body_efficiency(candle: Any) -> float:
    body = abs(float(candle.close) - float(candle.open))
    candle_range = max(abs(float(candle.high) - float(candle.low)), 1e-9)
    return _clamp(body / candle_range, 0.0, 1.0)


def _percentile_rank(values: list[float], value: float) -> float:
    if not values:
        return 0.5
    return sum(1 for item in values if item <= value) / len(values)


def _autocorr(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    x = values[1:]
    y = values[:-1]
    mx = _mean(x)
    my = _mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return _clamp(numerator / denominator, -1.0, 1.0) if denominator else 0.0


def _entropy_score(values: list[float]) -> float:
    if len(values) < 5:
        return 0.0
    positives = sum(1 for value in values if value > 0)
    negatives = sum(1 for value in values if value < 0)
    total = positives + negatives
    if total == 0:
        return 1.0
    entropy = 0.0
    for count in (positives, negatives):
        if count:
            p = count / total
            entropy -= p * math.log(p, 2)
    return _clamp(entropy, 0.0, 1.0)


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
