from __future__ import annotations

from types import SimpleNamespace

from src.language.exam_feedback import derive_exam_feedback


def _answer(category: str, *, correct: bool, quality: float = 0.8, flags=(), text="attempt"):
    return SimpleNamespace(
        category=category,
        correct=correct,
        quality_score=quality,
        gibberish_flags=tuple(flags),
        normalized_output=text,
    )


def _result(answers, *, quality=80.0, diversity=100.0, gibberish=0, collapse=False):
    correct = sum(answer.correct for answer in answers)
    return SimpleNamespace(
        answers=tuple(answers),
        total_questions=len(answers),
        correct_questions=correct,
        correctness_percent=100.0 * correct / max(len(answers), 1),
        mean_quality_percent=quality,
        answer_diversity_percent=diversity,
        gibberish_answers=gibberish,
        mode_collapse=collapse,
    )


def test_wrong_arithmetic_and_economics_raise_their_next_epoch_weights() -> None:
    result = _result([
        _answer("primitive_arithmetic", correct=False),
        _answer("number_sense", correct=False),
        _answer("foundation_economics", correct=False),
        _answer("grammar", correct=True),
    ])
    policy = derive_exam_feedback(result)
    assert policy.arithmetic_weight > 1.5
    assert policy.economics_weight > 1.5


def test_grammar_gibberish_and_low_quality_raise_language_pressure() -> None:
    result = _result([
        _answer("grammar", correct=False, quality=0.2, flags=("gibberish",)),
        _answer("semantic_plausibility", correct=False, quality=0.2, flags=("gibberish",)),
        _answer("language_control", correct=False, quality=0.2, flags=("high_word_repetition",)),
    ], quality=20.0, diversity=50.0, gibberish=3)
    policy = derive_exam_feedback(result)
    assert policy.language_quality_weight >= 2.0
    assert policy.repetition_penalty_scale > 1.5


def test_repeated_answer_sheet_and_mode_collapse_raise_creativity_and_repetition() -> None:
    result = _result([
        _answer("creativity", correct=False),
        _answer("language_control", correct=False),
        _answer("foundation_economics", correct=False),
        _answer("primitive_arithmetic", correct=False),
    ], diversity=25.0, collapse=True)
    policy = derive_exam_feedback(result)
    assert policy.creativity_weight > 1.8
    assert policy.repetition_penalty_scale > 2.0


def test_coherent_attempt_is_punished_less_than_empty_or_gibberish_failure() -> None:
    attempted = _result([
        _answer("primitive_arithmetic", correct=False, quality=0.85, text="I think the result is five."),
        _answer("foundation_economics", correct=False, quality=0.85, text="Price is related to exchange."),
    ], quality=85.0)
    abandoned = _result([
        _answer("primitive_arithmetic", correct=False, quality=0.0, flags=("empty",), text=""),
        _answer("foundation_economics", correct=False, quality=0.0, flags=("empty",), text=""),
    ], quality=0.0, gibberish=2)
    assert derive_exam_feedback(attempted).hard_negative_scale < derive_exam_feedback(abandoned).hard_negative_scale
