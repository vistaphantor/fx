from __future__ import annotations

import torch
import torch.nn.functional as F

HARD_NEGATIVE_OBJECTIVE_VERSION = 3
HARD_NEGATIVE_MARGIN = 1.5
HARD_NEGATIVE_MARGIN_WEIGHT = 0.30
HARD_NEGATIVE_UNLIKELIHOOD_WEIGHT = 0.10
HARD_NEGATIVE_CONFIDENCE_MULTIPLIER = 4.0
REPETITION_WINDOW = 16
REPETITION_MIN_PRIOR_OCCURRENCES = 2
REPETITION_UNLIKELIHOOD_WEIGHT = 0.08


def hard_negative_answer_penalty(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pad_id: int,
) -> torch.Tensor:
    """Push the strongest wrong token below the supervised target."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, time, vocab]")
    if targets.shape != logits.shape[:2]:
        raise ValueError("targets must match logits batch/time dimensions")
    if logits.shape[-1] < 2:
        return logits.new_zeros(())

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    valid = flat_targets != int(pad_id)
    if not torch.any(valid):
        return logits.new_zeros(())

    supervised_logits = flat_logits[valid]
    supervised_targets = flat_targets[valid]
    correct_logits = supervised_logits.gather(1, supervised_targets[:, None]).squeeze(1)
    top_values, top_indices = torch.topk(supervised_logits, k=2, dim=1)
    target_is_top = top_indices[:, 0] == supervised_targets
    hard_wrong_logits = torch.where(target_is_top, top_values[:, 1], top_values[:, 0])

    log_normalizer = torch.logsumexp(supervised_logits, dim=1)
    hard_wrong_probability = torch.exp(hard_wrong_logits - log_normalizer).clamp(
        min=0.0,
        max=1.0 - 1e-6,
    )
    confidence_scale = 1.0 + (
        HARD_NEGATIVE_CONFIDENCE_MULTIPLIER * hard_wrong_probability.detach()
    )
    margin_violation = F.relu(
        HARD_NEGATIVE_MARGIN - (correct_logits - hard_wrong_logits)
    )
    wrong_unlikelihood = -torch.log1p(-hard_wrong_probability)
    return (
        HARD_NEGATIVE_MARGIN_WEIGHT * (confidence_scale * margin_violation).mean()
        + HARD_NEGATIVE_UNLIKELIHOOD_WEIGHT * (confidence_scale * wrong_unlikelihood).mean()
    )


def repetition_unlikelihood_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    pad_id: int,
    window: int = REPETITION_WINDOW,
) -> torch.Tensor:
    """Suppress recent loop-attractor tokens when the teacher exits the loop.

    For each supervised position, collect a bounded causal history. A token is a
    repetition negative only when it appears at least twice in that history and
    differs from the current target. The strongest such candidate receives an
    unlikelihood penalty. The implementation is vectorized across batch/time so
    this signal can run every optimizer step without a Python loop per token.
    """
    if logits.shape[:2] != input_ids.shape or targets.shape != input_ids.shape:
        raise ValueError("logits, input_ids and targets must share batch/time dimensions")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, time]")

    batch, steps = input_ids.shape
    width = min(max(2, int(window)), steps)
    if steps < 2:
        return logits.new_zeros(())

    # history[b, t, k] is the token k+1 causal positions before/current at t.
    # Out-of-range positions are filled with pad and therefore ignored.
    histories: list[torch.Tensor] = []
    for lag in range(width):
        shifted = torch.full_like(input_ids, int(pad_id))
        if lag == 0:
            shifted.copy_(input_ids)
        else:
            shifted[:, lag:] = input_ids[:, :-lag]
        histories.append(shifted)
    history = torch.stack(histories, dim=-1)  # [B, T, W]

    equality = history.unsqueeze(-1) == history.unsqueeze(-2)
    occurrence_count = equality.sum(dim=-1)
    duplicated = occurrence_count >= REPETITION_MIN_PRIOR_OCCURRENCES
    current_target = targets.unsqueeze(-1)
    valid_history = (
        duplicated
        & (history != int(pad_id))
        & (history != current_target)
        & (targets.unsqueeze(-1) != int(pad_id))
    )
    if not torch.any(valid_history):
        return logits.new_zeros(())

    safe_ids = history.clamp(min=0, max=logits.shape[-1] - 1)
    candidate_logits = logits.gather(dim=-1, index=safe_ids)
    candidate_logits = candidate_logits.masked_fill(~valid_history, -torch.inf)
    hardest = candidate_logits.max(dim=-1).values
    valid_positions = torch.isfinite(hardest)
    if not torch.any(valid_positions):
        return logits.new_zeros(())

    rows = logits[valid_positions]
    hard = hardest[valid_positions]
    probability = torch.exp(hard - torch.logsumexp(rows, dim=-1)).clamp(
        min=0.0,
        max=1.0 - 1e-6,
    )
    return REPETITION_UNLIKELIHOOD_WEIGHT * (-torch.log1p(-probability)).mean()
