from __future__ import annotations

from types import SimpleNamespace

from src.language.semantic_checkpointing import (
    checkpoint_is_better,
    checkpoint_rank,
    mastery_report,
)


def _answer(category: str, correct: bool):
    return SimpleNamespace(category=category, correct=correct)


def _exam(*, correct: int, total: int = 12, quality: float = 80.0, diversity: float = 80.0,
          gibberish: int = 0, collapse: bool = False, answers=None):
    return SimpleNamespace(
        correct_questions=correct,
        total_questions=total,
        mean_quality_percent=quality,
        answer_diversity_percent=diversity,
        gibberish_answers=gibberish,
        mode_collapse=collapse,
        answers=list(answers or []),
    )


def test_more_semantic_correctness_beats_lower_validation_loss() -> None:
    fluent_wrong = _exam(correct=2, quality=99.0)
    better_semantics = _exam(correct=3, quality=70.0)
    incumbent = checkpoint_rank(fluent_wrong, 1.1)
    promoted, rank = checkpoint_is_better(
        candidate_result=better_semantics,
        candidate_validation_loss=3.0,
        incumbent_rank=incumbent,
    )
    assert promoted
    assert rank > incumbent


def test_validation_loss_only_breaks_semantic_ties() -> None:
    result = _exam(correct=5)
    incumbent = checkpoint_rank(result, 2.0)
    promoted, rank = checkpoint_is_better(
        candidate_result=result,
        candidate_validation_loss=1.5,
        incumbent_rank=incumbent,
    )
    assert promoted
    assert rank > incumbent


def test_foundation_cannot_master_with_weak_arithmetic() -> None:
    answers = [
        _answer("primitive_arithmetic", False),
        _answer("primitive_arithmetic", False),
        _answer("primitive_arithmetic", False),
        _answer("primitive_arithmetic", True),
        _answer("number_sense", True),
        _answer("foundation_economics", True),
        _answer("foundation_economics", True),
        _answer("foundation_economics", True),
        _answer("grammar", True),
        _answer("semantic_plausibility", True),
        _answer("language_control", True),
        _answer("creativity", True),
    ]
    report = mastery_report(_exam(correct=9, answers=answers), "foundation")
    assert report.arithmetic_accuracy == 0.4
    assert not report.mastered


def test_foundation_mastery_requires_clean_language_and_no_collapse() -> None:
    answers = [
        *[_answer("primitive_arithmetic", True) for _ in range(4)],
        _answer("number_sense", True),
        *[_answer("foundation_economics", True) for _ in range(3)],
        _answer("grammar", True),
        _answer("semantic_plausibility", True),
        _answer("language_control", True),
        _answer("creativity", True),
    ]
    good = mastery_report(_exam(correct=12, answers=answers), "foundation")
    assert good.mastered
    collapsed = mastery_report(_exam(correct=12, answers=answers, collapse=True), "foundation")
    assert not collapsed.mastered
    gibberish = mastery_report(_exam(correct=12, answers=answers, gibberish=1), "foundation")
    assert not gibberish.mastered
