from __future__ import annotations

import random
import re
from dataclasses import dataclass

CURRICULUM_STAGES = ("foundation", "reasoning", "trading_reasoning")

# One strong term is sufficient because these are highly specific to trading.
_STRONG_TRADING_TERMS = re.compile(
    r"\b(?:"
    r"xauusd|eurusd|gbpusd|usdjpy|usdchf|audusd|nzdusd|"
    r"candlestick|candlesticks|slippage|stop loss|stop-loss|take profit|"
    r"risk reward|risk-reward|position size|lot size|leverage|margin call|"
    r"pip|pips|order flow|orderflow|rsi|atr|macd|"
    r"bid[- ]ask|bid ask|long position|short position|broker|spread"
    r")\b",
    flags=re.IGNORECASE,
)

# These terms are meaningful in trading context but individually ambiguous.
# Require at least two distinct hits before calling an example trading-domain.
_CONTEXT_TRADING_TERMS = re.compile(
    r"\b(?:"
    r"forex|fx|trading|trader|trade|bullish|bearish|"
    r"liquidity|support|resistance|breakout|retest|volatility|drawdown|"
    r"timeframe|higher timeframe|lower timeframe|entry|exit|execution|"
    r"moving average|ema|sma|price action|risk management"
    r")\b",
    flags=re.IGNORECASE,
)

_MATH_TERMS = re.compile(
    r"\b(?:solve|calculate|equation|probability|percentage|percent|ratio|"
    r"mean|median|variance|standard deviation|integral|derivative|formula|"
    r"arithmetic|algebra|geometry|statistics)\b|[=+*/^]",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class CurriculumSelection:
    stage: str
    texts: list[str]
    total_available: int
    trading_available: int
    reasoning_available: int
    math_available: int
    replay_examples: int


def is_reasoning_example(text: str) -> bool:
    return "<think>" in text and "</think>" in text


def trading_evidence_score(text: str) -> int:
    if "<market>" in text or "<decision>" in text:
        return 3
    if _STRONG_TRADING_TERMS.search(text):
        return 3
    context_hits = {match.group(0).casefold() for match in _CONTEXT_TRADING_TERMS.finditer(text)}
    return len(context_hits)


def is_trading_example(text: str) -> bool:
    return trading_evidence_score(text) >= 2


def is_math_example(text: str) -> bool:
    return bool(_MATH_TERMS.search(text))


def _stable_sample(values: list[str], count: int, *, seed: int) -> list[str]:
    if count <= 0 or not values:
        return []
    if count >= len(values):
        return list(values)
    rng = random.Random(seed)
    indexes = list(range(len(values)))
    rng.shuffle(indexes)
    chosen = sorted(indexes[:count])
    return [values[index] for index in chosen]


def select_curriculum(
    texts: list[str],
    *,
    stage: str,
    seed: int = 42,
    min_trading_examples: int = 100,
) -> CurriculumSelection:
    """Select a deterministic, non-duplicating curriculum for one stage."""
    normalized_stage = stage.strip().casefold()
    if normalized_stage == "general_language":
        normalized_stage = "foundation"
    if normalized_stage not in CURRICULUM_STAGES:
        raise ValueError(f"unsupported_training_stage:{stage}")

    trading = [text for text in texts if is_trading_example(text)]
    reasoning = [text for text in texts if is_reasoning_example(text)]
    math_examples = [text for text in texts if is_math_example(text)]

    if normalized_stage == "foundation":
        selected = list(texts)
        replay = 0

    elif normalized_stage == "reasoning":
        primary_set = set(reasoning) | set(math_examples)
        primary = [text for text in texts if text in primary_set]
        general_pool = [text for text in texts if text not in primary_set]
        replay_count = min(len(general_pool), max(50, len(primary) // 4))
        replay_values = _stable_sample(general_pool, replay_count, seed=seed + 101)
        selected_set = set(primary) | set(replay_values)
        selected = [text for text in texts if text in selected_set]
        replay = len(replay_values)
        if not primary:
            raise RuntimeError("reasoning_curriculum_has_no_reasoning_or_math_examples")

    else:
        if len(trading) < min_trading_examples:
            raise RuntimeError(
                f"trading_curriculum_insufficient_examples:{len(trading)}<{min_trading_examples}"
            )
        trading_set = set(trading)
        reasoning_pool = [text for text in reasoning if text not in trading_set]
        reasoning_pool_set = set(reasoning_pool)
        general_pool = [
            text for text in texts
            if text not in trading_set and text not in reasoning_pool_set
        ]
        reasoning_replay_count = min(len(reasoning_pool), max(25, len(trading) // 3))
        general_replay_count = min(len(general_pool), max(10, len(trading) // 10))
        replay_values = (
            _stable_sample(reasoning_pool, reasoning_replay_count, seed=seed + 211)
            + _stable_sample(general_pool, general_replay_count, seed=seed + 307)
        )
        selected_set = trading_set | set(replay_values)
        selected = [text for text in texts if text in selected_set]
        replay = len(replay_values)

    if not selected:
        raise RuntimeError("curriculum_selection_empty")

    return CurriculumSelection(
        stage=normalized_stage,
        texts=selected,
        total_available=len(texts),
        trading_available=len(trading),
        reasoning_available=len(reasoning),
        math_available=len(math_examples),
        replay_examples=replay,
    )
