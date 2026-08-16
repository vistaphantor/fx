from __future__ import annotations

from dataclasses import asdict, dataclass

EXAM_FEEDBACK_VERSION = 1


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
        if version != EXAM_FEEDBACK_VERSION:
            raise RuntimeError(f"unsupported_exam_feedback_version:{version}")
        return cls(**{field: payload[field] for field in cls.__dataclass_fields__ if field in payload})

    def multiplier_for_tags(self, tags: tuple[str, ...]) -> float:
        mapping = {
            "arithmetic": self.arithmetic_weight,
            "economics": self.economics_weight,
            "language_quality": self.language_quality_weight,
            "conversation": self.conversation_weight,
            "creativity": self.creativity_weight,
        }
        values = [mapping[tag] for tag in tags if tag in mapping]
        if not values:
            return 1.0
        # Geometric-like moderation without importing math: multiple tags should
        # reinforce a source, but not multiply into runaway sampling weights.
        return sum(values) / len(values)


def _bounded(value: float, low: float = 0.65, high: float = 2.50) -> float:
    return max(low, min(high, float(value)))


def _category_accuracy(result, categories: set[str]) -> float:
    answers = [answer for answer in result.answers if answer.category in categories]
    if not answers:
        return 1.0
    return sum(1.0 for answer in answers if answer.correct) / len(answers)


def derive_exam_feedback(result) -> ExamFeedbackPolicy:
    """Convert exam failures into bounded next-epoch curriculum pressure.

    The exam remains a holdout: its prompts and target answers never enter the
    training stream. Only aggregate failure signals alter source sampling and
    objective strength for the following completed exam interval.
    """
    arithmetic_accuracy = _category_accuracy(result, {"primitive_arithmetic", "number_sense"})
    economics_accuracy = _category_accuracy(result, {"foundation_economics"})
    grammar_accuracy = _category_accuracy(result, {"grammar", "semantic_plausibility"})

    quality = max(0.0, min(1.0, result.mean_quality_percent / 100.0))
    diversity = max(0.0, min(1.0, result.answer_diversity_percent / 100.0))
    attempt = max(0.0, min(1.0, getattr(result, "mean_attempt_percent", 100.0) / 100.0))
    creativity = max(0.0, min(1.0, getattr(result, "mean_creativity_percent", 100.0) / 100.0))
    gibberish_rate = result.gibberish_answers / max(result.total_questions, 1)

    arithmetic = _bounded(1.0 + 1.30 * (1.0 - arithmetic_accuracy))
    economics = _bounded(1.0 + 1.20 * (1.0 - economics_accuracy))
    language_quality = _bounded(
        1.0
        + 1.10 * (1.0 - quality)
        + 0.90 * gibberish_rate
        + 0.80 * (1.0 - grammar_accuracy)
    )
    conversation = _bounded(1.0 + 0.65 * (1.0 - quality) + 0.45 * (1.0 - attempt))
    creativity_weight = _bounded(
        1.0
        + 0.90 * (1.0 - diversity)
        + 0.70 * (1.0 - creativity)
        + (0.50 if result.mode_collapse else 0.0)
    )
    hard_negative = _bounded(
        1.0 + 1.15 * (1.0 - result.correctness_percent / 100.0),
        low=0.80,
        high=2.75,
    )
    repetition = _bounded(
        1.0
        + 1.40 * (1.0 - diversity)
        + (0.75 if result.mode_collapse else 0.0)
        + 0.75 * gibberish_rate,
        low=0.80,
        high=3.00,
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
