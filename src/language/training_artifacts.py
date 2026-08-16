from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch

from src.language.exam import EpochExamResult
from src.language.exam_feedback import ExamFeedbackPolicy
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.semantic_checkpointing import (
    SEMANTIC_CHECKPOINT_POLICY_VERSION,
    checkpoint_is_better,
    mastery_report,
    rank_from_checkpoint_payload,
)
from src.language.training_session_contract import UPGRADABLE_DATA_CONTRACT_KEYS


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_json_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def save_training_state(
    path: Path,
    *,
    contract: dict,
    epoch: int,
    interval_steps: int,
    interval_elapsed_seconds: float,
    interval_loss_sum: float,
    interval_prediction_tokens: int,
    cumulative_tokens: int,
    optimizer_steps: int,
    best_val: float,
    model: VistaReasoningGPT,
    optimizer: torch.optim.Optimizer,
    feedback: ExamFeedbackPolicy,
) -> None:
    atomic_torch_save({
        **contract,
        "epoch": epoch,
        "interval_steps": interval_steps,
        "interval_elapsed_seconds": float(interval_elapsed_seconds),
        "interval_loss_sum": float(interval_loss_sum),
        "interval_prediction_tokens": int(interval_prediction_tokens),
        "cumulative_prediction_tokens": cumulative_tokens,
        "optimizer_steps": optimizer_steps,
        "best_validation_loss": best_val,
        "exam_feedback": feedback.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_random_state": random.getstate(),
    }, path)


def contract_mismatch_allowed(*, key: str, curriculum_upgrade: bool) -> bool:
    return bool(curriculum_upgrade and key in UPGRADABLE_DATA_CONTRACT_KEYS)


def load_best_rank(best_path: Path) -> tuple[float, ...] | None:
    if not best_path.exists():
        return None
    try:
        payload = torch.load(best_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError):
        return None
    return rank_from_checkpoint_payload(payload if isinstance(payload, dict) else None)


def promote_checkpoint(
    *,
    result: EpochExamResult,
    val_loss: float,
    model: VistaReasoningGPT,
    model_config: dict,
    cumulative_tokens: int,
    best_path: Path,
    incumbent_rank: tuple[float, ...] | None,
) -> tuple[tuple[float, ...] | None, float, bool]:
    promoted, candidate_rank = checkpoint_is_better(
        candidate_result=result,
        candidate_validation_loss=val_loss,
        incumbent_rank=incumbent_rank,
    )
    if not promoted:
        return incumbent_rank, val_loss, False
    mastery = mastery_report(result, result.training_stage)
    atomic_torch_save({
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "validation_loss": val_loss,
        "epoch": result.epoch,
        "cumulative_prediction_tokens": cumulative_tokens,
        "semantic_checkpoint_policy_version": SEMANTIC_CHECKPOINT_POLICY_VERSION,
        "semantic_checkpoint_rank": list(candidate_rank),
        "exam_correct_questions": result.correct_questions,
        "exam_total_questions": result.total_questions,
        "exam_correctness_percent": result.correctness_percent,
        "exam_quality_percent": result.mean_quality_percent,
        "exam_gibberish_answers": result.gibberish_answers,
        "exam_answer_diversity_percent": result.answer_diversity_percent,
        "exam_mode_collapse": result.mode_collapse,
        "mastery": mastery.to_dict(),
    }, best_path)
    print(
        f"[BestCheckpoint] promoted skills={mastery.mastered_skills}/{len(mastery.skill_results)} "
        f"concept_gates={mastery.conceptual_gates_passed}/{len(mastery.skill_results)} "
        f"semantic={result.correct_questions}/{result.total_questions} val={val_loss:.4f}"
    )
    return candidate_rank, val_loss, True
