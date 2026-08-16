from __future__ import annotations

import math

import torch

from src.language.compute_budget import reference_token_target
from src.language.foundation_contract import FOUNDATION_TARGET_PREDICTION_TOKENS
from src.language.training_profiles import DEFAULT_STAGE_TOKENS_PER_PARAMETER


def set_token_scheduled_lr(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    min_lr: float,
    cumulative_tokens: int,
    target_tokens: int,
    warmup_fraction: float,
) -> float:
    progress = min(1.0, cumulative_tokens / max(target_tokens, 1))
    if warmup_fraction > 0 and progress < warmup_fraction:
        lr = min_lr + (base_lr - min_lr) * (progress / warmup_fraction)
    else:
        span = max(1e-9, 1.0 - warmup_fraction)
        decay = min(1.0, max(0.0, (progress - warmup_fraction) / span))
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * decay))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def cap_prediction_targets(
    targets: torch.Tensor, *, pad_id: int, remaining_tokens: int,
) -> tuple[torch.Tensor, int]:
    """Mask only the final batch tail so the prediction counter lands exactly on target."""
    if remaining_tokens <= 0:
        raise ValueError("remaining_tokens must be positive")
    valid = int((targets != pad_id).sum().item())
    if valid <= remaining_tokens:
        return targets, valid
    capped = targets.clone()
    flat = capped.reshape(-1)
    valid_positions = torch.nonzero(flat != pad_id, as_tuple=False).flatten()
    flat[valid_positions[remaining_tokens:]] = pad_id
    return capped, remaining_tokens


def target_prediction_tokens(
    stage: str, total_params: int, target_tpp: float | None,
) -> tuple[int, float | None]:
    if stage == "foundation":
        if target_tpp is not None:
            raise ValueError("foundation_target_is_fixed_at_8b_prediction_tokens")
        return FOUNDATION_TARGET_PREDICTION_TOKENS, None
    tpp = float(target_tpp if target_tpp is not None else DEFAULT_STAGE_TOKENS_PER_PARAMETER[stage])
    if tpp <= 0:
        raise ValueError("target_tokens_per_parameter_must_be_positive")
    return reference_token_target(total_params, tokens_per_parameter=tpp), tpp
