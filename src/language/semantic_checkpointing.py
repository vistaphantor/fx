from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.language.exam import EpochExamResult

SEMANTIC_CHECKPOINT_POLICY_VERSION = 1
FOUNDATION_ARITHMETIC_MASTERY = 0.80
FOUNDATION_ECONOMICS_MASTERY = 2.0 / 3.0
FOUNDATION_LANGUAGE_CONTROL_MASTERY = 2.0 / 3.0
FOUNDATION_OVERALL_MASTERY = 0.70


@dataclass(frozen=True, slots=True)
class MasteryReport:
    stage: str
    arithmetic_accuracy: float
    economics_accuracy: float
    language_control_accuracy: float
    overall_accuracy: float
    gibberish_answers: int
    mode_collapse: bool
    mastered: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "arithmetic_accuracy": self.arithmetic_accuracy,
            "economics_accuracy": self.economics_accuracy,
            "language_control_accuracy": self.language_control_accuracy,
            "overall_accuracy": self.overall_accuracy,
            "gibberish_answers": self.gibberish_answers,
            "mode_collapse": self.mode_collapse,
            "mastered": self.mastered,
        }


def _category_accuracy(result: EpochExamResult, categories: Iterable[str]) -> float:
    category_set = set(categories)
    selected = [answer for answer in result.answers if answer.category in category_set]
    if not selected:
        return 1.0
    return sum(1 for answer in selected if answer.correct) / len(selected)


def checkpoint_rank(result: EpochExamResult, validation_loss: float) -> tuple[float, ...]:
    """Return the authoritative lexicographic checkpoint rank.

    Semantic correctness is primary. Fluent but wrong checkpoints therefore can
    no longer displace a checkpoint that answers more holdout questions correctly.
    Lower gibberish and no mode collapse are next, then surface quality/diversity.
    Validation cross-entropy is only the final tie-breaker.
    """
    return (
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


def mastery_report(result: EpochExamResult, stage: str) -> MasteryReport:
    normalized_stage = stage.strip().casefold()
    arithmetic = _category_accuracy(result, {"primitive_arithmetic", "number_sense"})
    economics = _category_accuracy(result, {"foundation_economics"})
    language_control = _category_accuracy(
        result,
        {"grammar", "semantic_plausibility", "language_control"},
    )
    overall = result.correct_questions / max(result.total_questions, 1)

    foundation_mastered = (
        arithmetic >= FOUNDATION_ARITHMETIC_MASTERY
        and economics >= FOUNDATION_ECONOMICS_MASTERY
        and language_control >= FOUNDATION_LANGUAGE_CONTROL_MASTERY
        and overall >= FOUNDATION_OVERALL_MASTERY
        and result.gibberish_answers == 0
        and not result.mode_collapse
    )
    if normalized_stage == "foundation":
        mastered = foundation_mastered
    elif normalized_stage in {"reasoning", "trading_reasoning"}:
        # Later stages may contain additional probes, but they are not allowed to
        # erase the foundation prerequisites. Overall mastery ensures the added
        # stage-specific questions are also contributing to the checkpoint.
        mastered = foundation_mastered and overall >= FOUNDATION_OVERALL_MASTERY
    else:
        raise ValueError(f"unsupported_training_stage:{stage}")

    return MasteryReport(
        stage=normalized_stage,
        arithmetic_accuracy=arithmetic,
        economics_accuracy=economics,
        language_control_accuracy=language_control,
        overall_accuracy=overall,
        gibberish_answers=int(result.gibberish_answers),
        mode_collapse=bool(result.mode_collapse),
        mastered=mastered,
    )


def rank_from_checkpoint_payload(payload: dict | None) -> tuple[float, ...] | None:
    if not payload:
        return None
    raw = payload.get("semantic_checkpoint_rank")
    if not isinstance(raw, (list, tuple)) or len(raw) != 6:
        return None
    try:
        return tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None
