from __future__ import annotations

import torch
import torch.nn.functional as F

HARD_NEGATIVE_OBJECTIVE_VERSION = 2
HARD_NEGATIVE_MARGIN = 1.5
HARD_NEGATIVE_MARGIN_WEIGHT = 0.30
HARD_NEGATIVE_UNLIKELIHOOD_WEIGHT = 0.10
HARD_NEGATIVE_CONFIDENCE_MULTIPLIER = 4.0
REPETITION_WINDOW = 24
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
    """Penalize loop-attractor tokens only when the teacher target differs.

    A candidate becomes a repetition negative when the same token has already
    appeared at least twice in the recent causal context and it is *not* the
    current supervised target. This preserves legitimate repetitions taught by
    the data while directly suppressing the common ``new time to get a new
    time...`` failure mode.
    """
    if logits.shape[:2] != input_ids.shape or targets.shape != input_ids.shape:
        raise ValueError("logits, input_ids and targets must share batch/time dimensions")
    batch, steps = input_ids.shape
    penalties: list[torch.Tensor] = []
    for batch_index in range(batch):
        for position in range(steps):
            target = int(targets[batch_index, position].item())
            if target == int(pad_id):
                continue
            start = max(0, position - max(1, int(window)) + 1)
            history = input_ids[batch_index, start : position + 1]
            if history.numel() < REPETITION_MIN_PRIOR_OCCURRENCES:
                continue
            values, counts = torch.unique(history, return_counts=True)
            repeated = values[counts >= REPETITION_MIN_PRIOR_OCCURRENCES]
            repeated = repeated[(repeated != target) & (repeated != int(pad_id))]
            if repeated.numel() == 0:
                continue
            row = logits[batch_index, position]
            repeated_logits = row.index_select(0, repeated)
            hardest = torch.max(repeated_logits)
            probability = torch.exp(hardest - torch.logsumexp(row, dim=0)).clamp(
                min=0.0,
                max=1.0 - 1e-6,
            )
            penalties.append(-torch.log1p(-probability))
    if not penalties:
        return logits.new_zeros(())
    return REPETITION_UNLIKELIHOOD_WEIGHT * torch.stack(penalties).mean()
