from __future__ import annotations

import pytest

from src.language.curriculum import (
    is_math_example,
    is_reasoning_example,
    is_trading_example,
    select_curriculum,
    trading_evidence_score,
)


def test_domain_classification_requires_substantive_trading_evidence():
    assert is_trading_example("XAUUSD is bullish on H1 with high spread.")
    assert is_trading_example("A bullish breakout entry needs risk management.")
    assert is_reasoning_example("<assistant><think>Work it out.</think>Answer.</assistant>")
    assert is_math_example("Solve 2x + 5 = 11")
    assert not is_trading_example("The capital of France is Paris.")
    assert not is_trading_example("The market price increased this year.")
    assert not is_trading_example("International trade affects the economy.")
    assert trading_evidence_score("The market price increased this year.") < 2


def test_foundation_preserves_unique_canonical_corpus():
    texts = ["one", "two", "three"]
    selection = select_curriculum(texts, stage="foundation")
    assert selection.texts == texts
    assert selection.replay_examples == 0


def test_reasoning_stage_keeps_reasoning_and_small_replay():
    reasoning = [f"<think>step {i}</think> answer" for i in range(20)]
    general = [f"general example {i}" for i in range(200)]
    selection = select_curriculum(reasoning + general, stage="reasoning", seed=42)
    assert all(text in selection.texts for text in reasoning)
    assert selection.replay_examples > 0
    assert len(selection.texts) < len(reasoning) + len(general)


def test_trading_stage_fails_closed_when_domain_data_is_shallow():
    texts = ["general text"] * 200 + ["XAUUSD bullish"] * 5
    with pytest.raises(RuntimeError, match="trading_curriculum_insufficient_examples"):
        select_curriculum(texts, stage="trading_reasoning", min_trading_examples=20)


def test_trading_stage_is_domain_dominant_without_duplicate_oversampling():
    trading = [f"XAUUSD trading setup number {i}" for i in range(120)]
    reasoning = [f"<think>reasoning {i}</think> answer" for i in range(120)]
    general = [f"general knowledge {i}" for i in range(300)]
    selection = select_curriculum(
        trading + reasoning + general,
        stage="trading_reasoning",
        min_trading_examples=100,
        seed=42,
    )
    assert all(text in selection.texts for text in trading)
    assert len(selection.texts) == len(set(selection.texts))
    assert len(trading) > selection.replay_examples
