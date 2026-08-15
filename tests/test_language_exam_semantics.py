from __future__ import annotations

from src.language.exam import (
    ExamAnswer,
    _collapse_metrics,
    _is_correct,
    _normalize_output,
    exam_questions,
)


def _answer(output: str) -> ExamAnswer:
    return ExamAnswer(
        question_id="probe",
        category="probe",
        prompt="probe",
        raw_output=output,
        normalized_output=_normalize_output(output),
        generated_tokens=4,
        correct=False,
        quality_score=1.0,
        repetition_ratio=0.0,
        gibberish_flags=(),
    )


def test_foundation_exam_only_tests_taught_foundation_capabilities() -> None:
    questions = exam_questions("foundation")
    categories = {question.category for question in questions}
    assert "algebra" not in categories
    assert "trading_language" not in categories
    assert "primitive_arithmetic" in categories
    assert "foundation_economics" in categories


def test_incidental_expected_digit_does_not_pass_numeric_question() -> None:
    question = next(q for q in exam_questions("foundation") if q.question_id == "arithmetic_2_plus_2")
    assert not _is_correct(question, "4 - 9 = -3", ())
    assert not _is_correct(question, "there are 4 cats and 9 dogs", ())
    assert _is_correct(question, "4", ())
    assert _is_correct(question, "the answer is 4", ())
    assert _is_correct(question, "2 + 2 = 4", ())


def test_unrelated_prompt_answer_collision_is_detected() -> None:
    answers = [_answer("4 - 9 = -3.") for _ in range(5)]
    answers += [_answer("The price can rise."), _answer("Purchasing power falls."), _answer("13.")]
    diversity, exact_collision, prefix_collision, collapsed = _collapse_metrics(answers)
    assert diversity < 60.0
    assert exact_collision == 5
    assert prefix_collision >= 5
    assert collapsed is True


def test_distinct_short_answers_do_not_trigger_mode_collapse() -> None:
    outputs = ["4.", "7.", "-3.", "12.", "5.", "A price is money paid.", "13.", "Purchasing power falls."]
    diversity, exact_collision, prefix_collision, collapsed = _collapse_metrics([_answer(x) for x in outputs])
    assert diversity == 100.0
    assert exact_collision == 1
    assert prefix_collision == 1
    assert collapsed is False
