from __future__ import annotations

from dataclasses import dataclass

from src.language.exam import EpochExamResult
from src.language.foundation_contract import (
    FOUNDATION_EXAM_QUESTIONS_PER_SKILL,
    FOUNDATION_SKILL_MASTERY_THRESHOLD,
    FOUNDATION_SKILLS,
)

SEMANTIC_CHECKPOINT_POLICY_VERSION = 2


@dataclass(frozen=True, slots=True)
class SkillMastery:
    skill: str
    correct: int
    total: int
    accuracy: float
    conceptual_gate_passed: bool
    mastered: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "skill": self.skill,
            "correct": self.correct,
            "total": self.total,
            "accuracy": self.accuracy,
            "conceptual_gate_passed": self.conceptual_gate_passed,
            "mastered": self.mastered,
        }


@dataclass(frozen=True, slots=True)
class MasteryReport:
    stage: str
    skill_results: tuple[SkillMastery, ...]
    overall_accuracy: float
    gibberish_answers: int
    mode_collapse: bool
    mastered: bool

    @property
    def mastered_skills(self) -> int:
        return sum(1 for item in self.skill_results if item.mastered)

    @property
    def conceptual_gates_passed(self) -> int:
        return sum(1 for item in self.skill_results if item.conceptual_gate_passed)

    @property
    def minimum_skill_accuracy(self) -> float:
        return min((item.accuracy for item in self.skill_results), default=0.0)

    @property
    def arithmetic_accuracy(self) -> float:
        selected = [item.accuracy for item in self.skill_results if item.skill in {"addition", "subtraction", "multiplication"}]
        return sum(selected) / len(selected) if selected else 0.0

    @property
    def economics_accuracy(self) -> float:
        return next((item.accuracy for item in self.skill_results if item.skill == "economics"), 0.0)

    @property
    def language_control_accuracy(self) -> float:
        selected = [item.accuracy for item in self.skill_results if item.skill in {"english", "swahili"}]
        return sum(selected) / len(selected) if selected else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "questions_per_skill": FOUNDATION_EXAM_QUESTIONS_PER_SKILL,
            "skill_mastery_threshold": FOUNDATION_SKILL_MASTERY_THRESHOLD,
            "skills": {item.skill: item.to_dict() for item in self.skill_results},
            "mastered_skills": self.mastered_skills,
            "conceptual_gates_passed": self.conceptual_gates_passed,
            "minimum_skill_accuracy": self.minimum_skill_accuracy,
            "arithmetic_accuracy": self.arithmetic_accuracy,
            "economics_accuracy": self.economics_accuracy,
            "language_control_accuracy": self.language_control_accuracy,
            "overall_accuracy": self.overall_accuracy,
            "gibberish_answers": self.gibberish_answers,
            "mode_collapse": self.mode_collapse,
            "mastered": self.mastered,
        }


def mastery_report(result: EpochExamResult, stage: str) -> MasteryReport:
    normalized_stage = stage.strip().casefold()
    answers = list(result.answers)
    skill_results: list[SkillMastery] = []
    for skill in FOUNDATION_SKILLS:
        selected = [answer for answer in answers if getattr(answer, "skill", None) == skill]
        gates = [answer for answer in selected if getattr(answer, "conceptual_gate", False)]
        correct = sum(1 for answer in selected if answer.correct)
        total = len(selected)
        accuracy = correct / total if total else 0.0
        conceptual_gate_passed = len(gates) == 1 and bool(gates[0].correct)
        mastered = (
            total == FOUNDATION_EXAM_QUESTIONS_PER_SKILL
            and conceptual_gate_passed
            and accuracy >= FOUNDATION_SKILL_MASTERY_THRESHOLD
        )
        skill_results.append(SkillMastery(
            skill=skill,
            correct=correct,
            total=total,
            accuracy=accuracy,
            conceptual_gate_passed=conceptual_gate_passed,
            mastered=mastered,
        ))

    foundation_answers = [answer for answer in answers if getattr(answer, "skill", None) in FOUNDATION_SKILLS]
    overall = sum(1 for answer in foundation_answers if answer.correct) / max(len(foundation_answers), 1)
    foundation_mastered = (
        all(item.mastered for item in skill_results)
        and result.gibberish_answers == 0
        and not result.mode_collapse
    )
    if normalized_stage == "foundation":
        mastered = foundation_mastered
    elif normalized_stage in {"reasoning", "trading_reasoning"}:
        extension_answers = [answer for answer in answers if getattr(answer, "skill", None) not in FOUNDATION_SKILLS]
        extension_accuracy = sum(1 for answer in extension_answers if answer.correct) / max(len(extension_answers), 1)
        mastered = foundation_mastered and extension_accuracy >= FOUNDATION_SKILL_MASTERY_THRESHOLD
    else:
        raise ValueError(f"unsupported_training_stage:{stage}")

    return MasteryReport(
        stage=normalized_stage,
        skill_results=tuple(skill_results),
        overall_accuracy=overall,
        gibberish_answers=int(result.gibberish_answers),
        mode_collapse=bool(result.mode_collapse),
        mastered=mastered,
    )


def checkpoint_rank(result: EpochExamResult, validation_loss: float) -> tuple[float, ...]:
    report = mastery_report(result, result.training_stage)
    return (
        float(report.mastered_skills),
        float(report.conceptual_gates_passed),
        float(report.minimum_skill_accuracy),
        float(result.correct_questions),
        float(-result.gibberish_answers),
        float(not result.mode_collapse),
        float(result.mean_quality_percent),
        float(result.answer_diversity_percent),
        -float(validation_loss),
    )


def checkpoint_is_better(
    *,
    candidate_result: EpochExamResult,
    candidate_validation_loss: float,
    incumbent_rank: tuple[float, ...] | None,
) -> tuple[bool, tuple[float, ...]]:
    candidate_rank = checkpoint_rank(candidate_result, candidate_validation_loss)
    return incumbent_rank is None or candidate_rank > incumbent_rank, candidate_rank


def rank_from_checkpoint_payload(payload: dict | None) -> tuple[float, ...] | None:
    if not payload:
        return None
    raw = payload.get("semantic_checkpoint_rank")
    if not isinstance(raw, (list, tuple)) or len(raw) != 9:
        return None
    try:
        return tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None
