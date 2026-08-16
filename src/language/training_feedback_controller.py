from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from src.language.exam_feedback import ExamFeedbackPolicy, derive_exam_feedback
from src.language.hard_negative_objective import (
    hard_negative_answer_penalty,
    repetition_unlikelihood_penalty,
)


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def feedback_from_exam_json(path: str | Path) -> ExamFeedbackPolicy:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return derive_exam_feedback(_namespace(payload))


def load_latest_exam_feedback(exams_dir: str | Path) -> ExamFeedbackPolicy:
    """Recover the most recent completed exam signal without retraining on it."""
    directory = Path(exams_dir)
    if not directory.exists():
        return ExamFeedbackPolicy()
    candidates = sorted(
        directory.glob("*.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or "answers" not in payload:
                continue
            return derive_exam_feedback(_namespace(payload))
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return ExamFeedbackPolicy()


def adaptive_training_loss(
    *,
    base_loss: torch.Tensor,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
    feedback: ExamFeedbackPolicy,
) -> torch.Tensor:
    """Apply next-epoch exam pressure while preserving the base objective.

    ``VistaReasoningGPT`` already includes one unit of hard-negative correction
    during training. This controller adds only the extra amount requested by the
    exam policy, plus the repetition-specific unlikelihood term.
    """
    if not torch.isfinite(base_loss):
        return base_loss
    extra_hard_scale = max(0.0, float(feedback.hard_negative_scale) - 1.0)
    loss = base_loss
    if extra_hard_scale > 0:
        loss = loss + extra_hard_scale * hard_negative_answer_penalty(
            logits,
            targets,
            pad_id=pad_id,
        )
    repetition_scale = max(0.0, float(feedback.repetition_penalty_scale))
    if repetition_scale > 0:
        loss = loss + repetition_scale * repetition_unlikelihood_penalty(
            logits,
            input_ids,
            targets,
            pad_id=pad_id,
        )
    return loss


def feedback_summary(feedback: ExamFeedbackPolicy) -> str:
    return (
        f"math={feedback.arithmetic_weight:.2f} "
        f"economics={feedback.economics_weight:.2f} "
        f"language={feedback.language_quality_weight:.2f} "
        f"conversation={feedback.conversation_weight:.2f} "
        f"creativity={feedback.creativity_weight:.2f} "
        f"hard_negative={feedback.hard_negative_scale:.2f} "
        f"repetition={feedback.repetition_penalty_scale:.2f}"
    )
