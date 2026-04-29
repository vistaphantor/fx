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
