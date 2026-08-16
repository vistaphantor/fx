from __future__ import annotations

from dataclasses import asdict, dataclass

EXAM_FEEDBACK_VERSION = 3


@dataclass(frozen=True, slots=True)
class ExamFeedbackPolicy:
    version: int = EXAM_FEEDBACK_VERSION
    arithmetic_weight: float = 1.0
    economics_weight: float = 1.0
    language_quality_weight: float = 1.0
    conversation_weight: float = 1.0
    creativity_weight: float = 1.0
    hard_negative_scale: float = 1.0
    repetition_penalty_scale: float = 1.0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "ExamFeedbackPolicy":
        if not payload:
            return cls()
        version = int(payload.get("version", EXAM_FEEDBACK_VERSION))
        if version not in {1, 2, EXAM_FEEDBACK_VERSION}:
            raise RuntimeError(f"unsupported_exam_feedback_version:{version}")
        values = {field: payload[field] for field in cls.__dataclass_fields__ if field in payload}
        values["version"] = EXAM_FEEDBACK_VERSION
        return cls(**values)

    def multiplier_for_tags(self, tags: tuple[str, ...]) -> float:
        mapping = {
            "arithmetic": self.arithmetic_weight,
            "economics": self.economics_weight,
            "language_quality": self.language_quality_weight,
            "conversation": self.conversation_weight,
            "creativity": self.creativity_weight,
        }
        values = [mapping[tag] for tag in tags if tag in mapping]
        return sum(values) / len(values) if values else 1.0


def _bounded(value: float, low: float = 0.65, high: float = 3.75) -> float:
    return max(low, min(high, float(value)))


def _category_accuracy(result, categories: set[str]) -> float:
    answers = [answer for answer in result.answers if answer.category in categories]
    if not answers:
        return 1.0
    return sum(1.0 for answer in answers if answer.correct) / len(answers)


def _attempt_rate(result) -> float:
    """Credit relevant, non-collapsed effort even when the final answer is wrong."""
    answers = list(result.answers)
    if not answers:
        return 0.0
    attempted = 0.0
    for answer in answers:
        if not answer.normalized_output:
            continue
        if answer.quality_score >= 0.55 and not answer.gibberish_flags:
            attempted += 1.0
        elif answer.quality_score >= 0.30:
            attempted += 0.5
    return attempted / len(answers)


def derive_exam_feedback(result) -> ExamFeedbackPolicy:
    """Convert holdout failures into next-interval remediation pressure.

    Exams remain holdouts: their prompts/targets are never replayed as training
    examples. Failure instead reallocates *supervised-token quota* toward the
    weak skill and strengthens answer-anchor/repetition penalties. Coherent
    attempts retain a small reward so the model is not trained to retreat into
    silence or generic boilerplate.
    """
    arithmetic_accuracy = _category_accuracy(result, {"primitive_arithmetic", "number_sense"})
    economics_accuracy = _category_accuracy(result, {"foundation_economics"})
    grammar_accuracy = _category_accuracy(result, {"grammar", "semantic_plausibility"})
    language_control_accuracy = _category_accuracy(result, {"language_control"})
    creativity_accuracy = _category_accuracy(result, {"creativity"})

    quality = max(0.0, min(1.0, result.mean_quality_percent / 100.0))
    diversity = max(0.0, min(1.0, result.answer_diversity_percent / 100.0))
    attempt = _attempt_rate(result)
    gibberish_rate = result.gibberish_answers / max(result.total_questions, 1)

    # Arithmetic is a mastery prerequisite for later reasoning. A zero-score exam
    # now receives 3.5x arithmetic token quota instead of the former 2.3x ceiling.
    arithmetic = _bounded(1.0 + 2.50 * (1.0 - arithmetic_accuracy), high=3.50)
    economics = _bounded(1.0 + 2.00 * (1.0 - economics_accuracy), high=3.00)
    language_quality = _bounded(
        1.0
        + 1.25 * (1.0 - quality)
        + 1.10 * gibberish_rate
        + 1.10 * (1.0 - grammar_accuracy)
        + 0.60 * (1.0 - language_control_accuracy),
        high=3.25,
    )
    conversation = _bounded(
        1.0 + 0.65 * (1.0 - quality) + 0.45 * (1.0 - attempt),
        high=2.25,
    )
    creativity_weight = _bounded(
        1.0
        + 0.90 * (1.0 - diversity)
        + 0.65 * (1.0 - creativity_accuracy)
        + (0.50 if result.mode_collapse else 0.0),
        high=2.75,
    )

    raw_hard_negative = 1.0 + 1.55 * (1.0 - result.correctness_percent / 100.0)
    # Attempts reduce only 10% of the extra pressure. The answer-anchor objective
    # is now narrow enough that strong correction does not indiscriminately punish
    # every ordinary language token in an otherwise useful answer.
    hard_negative = _bounded(
        raw_hard_negative * (1.0 - 0.10 * attempt),
        low=0.90,
        high=2.75,
    )
    repetition = _bounded(
        1.0
        + 1.75 * (1.0 - diversity)
        + 0.80 * (1.0 - language_control_accuracy)
        + (0.90 if result.mode_collapse else 0.0)
        + 0.90 * gibberish_rate,
        low=0.90,
        high=3.50,
    )
    return ExamFeedbackPolicy(
        arithmetic_weight=arithmetic,
        economics_weight=economics,
        language_quality_weight=language_quality,
        conversation_weight=conversation,
        creativity_weight=creativity_weight,
        hard_negative_scale=hard_negative,
        repetition_penalty_scale=repetition,
    )
