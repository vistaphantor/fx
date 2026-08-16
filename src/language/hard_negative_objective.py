from __future__ import annotations

import torch
import torch.nn.functional as F

HARD_NEGATIVE_OBJECTIVE_VERSION = 5
HARD_NEGATIVE_MARGIN = 1.5
HARD_NEGATIVE_MARGIN_WEIGHT = 0.30
HARD_NEGATIVE_UNLIKELIHOOD_WEIGHT = 0.10
HARD_NEGATIVE_CONFIDENCE_MULTIPLIER = 4.0
HARD_NEGATIVE_ANCHOR_TOKENS = 2
REPETITION_WINDOW = 16
REPETITION_MIN_PRIOR_OCCURRENCES = 2
REPETITION_UNLIKELIHOOD_WEIGHT = 0.08


def _answer_anchor_mask(targets: torch.Tensor, *, pad_id: int) -> torch.Tensor:
    """Select the first supervised tokens of genuine assistant-answer runs.

    The authoritative chat loss masks user/system prompt targets before the first
    assistant-content target. Ordinary document LM rows begin supervised at time
    zero and may only contain padding at the tail. Requiring a pad->valid
    transition therefore prevents the answer-specific hard-negative objective
    from accidentally treating the first words of normal documents as answers.

    Packed chat sequences may contain more than one assistant run; each transition
    is independently anchored. A run beginning at position zero is deliberately
    excluded because it has no evidence of masked prompt context.
    """
    if targets.ndim != 2:
        raise ValueError("targets must have shape [batch, time]")
    valid = targets != int(pad_id)
    if not torch.any(valid):
        return valid

    previous_is_pad = torch.zeros_like(valid)
    if valid.shape[1] > 1:
        previous_is_pad[:, 1:] = ~valid[:, :-1]
    run_start = valid & previous_is_pad
    if not torch.any(run_start):
        return torch.zeros_like(valid)

    anchors = torch.zeros_like(valid)
    frontier = run_start
    for _ in range(HARD_NEGATIVE_ANCHOR_TOKENS):
        anchors |= frontier & valid
        shifted = torch.zeros_like(frontier)
        if frontier.shape[1] > 1:
            shifted[:, 1:] = frontier[:, :-1]
        frontier = shifted
    return anchors & valid


def hard_negative_answer_penalty(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pad_id: int,
) -> torch.Tensor:
    """Push the strongest wrong token below answer-critical target anchors."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, time, vocab]")
    if targets.shape != logits.shape[:2]:
        raise ValueError("targets must match logits batch/time dimensions")
    if logits.shape[-1] < 2:
        return logits.new_zeros(())

    anchor = _answer_anchor_mask(targets, pad_id=pad_id)
    if not torch.any(anchor):
        return logits.new_zeros(())

    supervised_logits = logits[anchor]
    supervised_targets = targets[anchor]
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
    unlikelihood penalty. This applies to both documents and conversations because
    pathological repetition is a language failure in either objective.
    """
    if logits.shape[:2] != input_ids.shape or targets.shape != input_ids.shape:
        raise ValueError("logits, input_ids and targets must share batch/time dimensions")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, time]")

    _, steps = input_ids.shape
    width = min(max(2, int(window)), steps)
    if steps < 2:
        return logits.new_zeros(())

    histories: list[torch.Tensor] = []
    for lag in range(width):
        shifted = torch.full_like(input_ids, int(pad_id))
        if lag == 0:
            shifted.copy_(input_ids)
        else:
            shifted[:, lag:] = input_ids[:, :-lag]
        histories.append(shifted)
    history = torch.stack(histories, dim=-1)

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
