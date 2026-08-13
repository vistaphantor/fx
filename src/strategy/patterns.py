from __future__ import annotations

from dataclasses import dataclass

from src.market_data import Candle


@dataclass(frozen=True, slots=True)
class PatternDecision:
    is_present: bool
    reason: str
    confluence_score: int
    metadata: dict[str, object]


def detect_three_drives(*, candles: list[Candle], reference_levels: tuple[float, ...], tolerance: float = 0.5) -> PatternDecision:
    if len(candles) < 5:
        return PatternDecision(
            is_present=False,
            reason="three_drives_absent",
            confluence_score=0,
            metadata={"touches": 0},
        )

    touches = []
    for index, candle in enumerate(candles):
        if candle.close >= candle.open:
            continue
        if any(abs(candle.low - level) <= tolerance for level in reference_levels):
            touches.append(index)

    if len(touches) >= 3 and _spaced_touches(touches):
        return PatternDecision(
            is_present=True,
            reason="three_drives_detected",
            confluence_score=3,
            metadata={"touches": tuple(touches[-3:])},
        )

    return PatternDecision(
        is_present=False,
        reason="three_drives_absent",
        confluence_score=0,
        metadata={"touches": tuple(touches)},
    )


def _spaced_touches(indices: list[int]) -> bool:
    recent = indices[-3:]
    return recent[0] + 1 < recent[1] and recent[1] + 1 < recent[2]


def detect_candlestick_patterns(candles: list[Candle]) -> PatternDecision:
    """Detect key price action candlestick patterns:
    - Bullish/Bearish Engulfing
    - Bullish Pinbar (Hammer / Lower Wick Rejection)
    - Bearish Pinbar (Shooting Star / Upper Wick Rejection)
    - Bullish/Bearish Momentum Expansion (Marubozu/Breakout)
    - Morning Star / Evening Star
    """
    if len(candles) < 2:
        return PatternDecision(
            is_present=False,
            reason="insufficient_candles_for_patterns",
            confluence_score=0,
            metadata={},
        )

    latest = candles[-1]
    previous = candles[-2]

    latest_range = max(float(latest.high) - float(latest.low), 1e-9)
    latest_body = abs(float(latest.close) - float(latest.open))
    latest_upper_wick = float(latest.high) - max(float(latest.open), float(latest.close))
    latest_lower_wick = min(float(latest.open), float(latest.close)) - float(latest.low)

    # 1. Morning Star / Evening Star (3-candle patterns take precedence)
    if len(candles) >= 3:
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        c2_body = abs(float(c2.close) - float(c2.open))
        c2_range = max(float(c2.high) - float(c2.low), 1e-9)
        c1_midpoint = (float(c1.open) + float(c1.close)) / 2.0

        if float(c1.close) < float(c1.open) and c2_body <= (c2_range * 0.40) and float(c3.close) > float(c3.open):
            if float(c3.close) >= c1_midpoint:
                return PatternDecision(
                    is_present=True,
                    reason="morning_star",
                    confluence_score=2,
                    metadata={"pattern": "morning_star", "direction": "BULLISH"},
                )

        if float(c1.close) > float(c1.open) and c2_body <= (c2_range * 0.40) and float(c3.close) < float(c3.open):
            if float(c3.close) <= c1_midpoint:
                return PatternDecision(
                    is_present=True,
                    reason="evening_star",
                    confluence_score=2,
                    metadata={"pattern": "evening_star", "direction": "BEARISH"},
                )

    # 2. Engulfing Patterns
    if float(previous.close) < float(previous.open) and float(latest.close) > float(latest.open):
        if float(latest.close) >= float(previous.open) and float(latest.open) <= float(previous.close):
            return PatternDecision(
                is_present=True,
                reason="bullish_engulfing",
                confluence_score=2,
                metadata={"pattern": "bullish_engulfing", "direction": "BULLISH"},
            )
    if float(previous.close) > float(previous.open) and float(latest.close) < float(latest.open):
        if float(latest.close) <= float(previous.open) and float(latest.open) >= float(previous.close):
            return PatternDecision(
                is_present=True,
                reason="bearish_engulfing",
                confluence_score=2,
                metadata={"pattern": "bearish_engulfing", "direction": "BEARISH"},
            )

    # 3. Pinbars / Rejection Wicks
    if latest_lower_wick >= (latest_range * 0.55) and latest_body <= (latest_range * 0.35):
        return PatternDecision(
            is_present=True,
            reason="bullish_pinbar",
            confluence_score=2,
            metadata={"pattern": "bullish_pinbar", "direction": "BULLISH", "lower_wick_ratio": latest_lower_wick / latest_range},
        )
    if latest_upper_wick >= (latest_range * 0.55) and latest_body <= (latest_range * 0.35):
        return PatternDecision(
            is_present=True,
            reason="bearish_pinbar",
            confluence_score=2,
            metadata={"pattern": "bearish_pinbar", "direction": "BEARISH", "upper_wick_ratio": latest_upper_wick / latest_range},
        )

    # 4. Momentum Expansion Bar
    sample_candles = candles[-min(10, len(candles)):]
    avg_range = sum(max(float(c.high) - float(c.low), 1e-9) for c in sample_candles) / len(sample_candles)
    if latest_body >= (latest_range * 0.70) and latest_range >= (avg_range * 1.25):
        if float(latest.close) > float(latest.open):
            return PatternDecision(
                is_present=True,
                reason="bullish_momentum_expansion",
                confluence_score=1,
                metadata={"pattern": "bullish_momentum_expansion", "direction": "BULLISH"},
            )
        elif float(latest.close) < float(latest.open):
            return PatternDecision(
                is_present=True,
                reason="bearish_momentum_expansion",
                confluence_score=1,
                metadata={"pattern": "bearish_momentum_expansion", "direction": "BEARISH"},
            )

    return PatternDecision(
        is_present=False,
        reason="no_candlestick_pattern",
        confluence_score=0,
        metadata={},
    )

