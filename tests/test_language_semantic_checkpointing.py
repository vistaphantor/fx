from __future__ import annotations

from types import SimpleNamespace

from src.language.foundation_contract import (
    FOUNDATION_EXAM_QUESTIONS_PER_SKILL,
    FOUNDATION_SKILLS,
)
from src.language.semantic_checkpointing import checkpoint_is_better, checkpoint_rank, mastery_report


def _answer(skill: str, index: int, correct: bool = True):
    return SimpleNamespace(
        skill=skill,
        conceptual_gate=(index == 0),
        correct=correct,
    )


def _exam(*, answers, quality: float = 80.0, diversity: float = 80.0,
          gibberish: int = 0, collapse: bool = False):
    answers = list(answers)
    return SimpleNamespace(
        training_stage="foundation",
        correct_questions=sum(1 for answer in answers if answer.correct),
        total_questions=len(answers),
        mean_quality_percent=quality,
        answer_diversity_percent=diversity,
        gibberish_answers=gibberish,
        mode_collapse=collapse,
        answers=answers,
    )


def _perfect_answers():
    return [
        _answer(skill, index)
        for skill in FOUNDATION_SKILLS
        for index in range(FOUNDATION_EXAM_QUESTIONS_PER_SKILL)
    ]


def test_foundation_has_one_isolated_fifty_question_mastery_contract_per_skill() -> None:
    report = mastery_report(_exam(answers=_perfect_answers()), "foundation")
    assert len(report.skill_results) == len(FOUNDATION_SKILLS) == 14
    assert all(item.total == 50 for item in report.skill_results)
    assert all(item.conceptual_gate_passed for item in report.skill_results)
    assert report.mastered_skills == 14
    assert report.mastered


def test_conceptual_question_one_is_a_mandatory_gate_even_above_eighty_percent() -> None:
    answers = _perfect_answers()
    for answer in answers:
        if answer.skill == "addition" and answer.conceptual_gate:
            answer.correct = False
            break
    report = mastery_report(_exam(answers=answers), "foundation")
    addition = next(item for item in report.skill_results if item.skill == "addition")
    assert addition.accuracy == 49 / 50
    assert not addition.conceptual_gate_passed
    assert not addition.mastered
    assert not report.mastered


def test_skill_requires_at_least_eighty_percent_after_conceptual_gate() -> None:
    answers = _perfect_answers()
    failed = 0
    for answer in answers:
        if answer.skill == "poetry" and not answer.conceptual_gate and failed < 11:
            answer.correct = False
            failed += 1
    report = mastery_report(_exam(answers=answers), "foundation")
    poetry = next(item for item in report.skill_results if item.skill == "poetry")
    assert poetry.conceptual_gate_passed
    assert poetry.accuracy == 39 / 50
    assert not poetry.mastered
    assert not report.mastered


def test_checkpoint_rank_prefers_more_mastered_skills_over_lower_validation_loss() -> None:
    weaker_answers = _perfect_answers()
    for answer in weaker_answers:
        if answer.skill == "shairi" and answer.conceptual_gate:
            answer.correct = False
            break
    weaker = _exam(answers=weaker_answers, quality=99.0)
    stronger = _exam(answers=_perfect_answers(), quality=70.0)
    incumbent = checkpoint_rank(weaker, 1.0)
    promoted, rank = checkpoint_is_better(
        candidate_result=stronger,
        candidate_validation_loss=3.0,
        incumbent_rank=incumbent,
    )
    assert promoted
    assert rank > incumbent


def test_mastery_requires_clean_outputs_and_no_mode_collapse() -> None:
    good = mastery_report(_exam(answers=_perfect_answers()), "foundation")
    assert good.mastered
    assert not mastery_report(_exam(answers=_perfect_answers(), gibberish=1), "foundation").mastered
    assert not mastery_report(_exam(answers=_perfect_answers(), collapse=True), "foundation").mastered
