from __future__ import annotations

import torch
import torch.nn.functional as F

HARD_NEGATIVE_OBJECTIVE_VERSION = 1
HARD_NEGATIVE_MARGIN = 1.5
HARD_NEGATIVE_MARGIN_WEIGHT = 0.30
HARD_NEGATIVE_UNLIKELIHOOD_WEIGHT = 0.10
HARD_NEGATIVE_CONFIDENCE_MULTIPLIER = 4.0


def hard_negative_answer_penalty(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pad_id: int,
) -> torch.Tensor:
    """Actively push down the strongest wrong token on supervised positions.

    Cross-entropy rewards the correct token but does not directly express that a
    confidently preferred wrong token is worse than a diffuse uncertain error.
    This objective adds a target-vs-hard-negative margin and unlikelihood on the
    strongest wrong token. The penalty scales with detached wrong-token
    confidence, so confidently wrong predictions receive materially stronger
    correction without amplifying gradient through the scale itself.

    The top-two selection avoids cloning a full supervised [tokens, vocab]
    tensor merely to mask the correct target. If top-1 is the target, top-2 is
    necessarily the strongest wrong candidate; otherwise top-1 is the hard
    negative. Prompt and padding positions remain excluded by the target mask.
    """
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

    margin_term = (confidence_scale * margin_violation).mean()
    unlikelihood_term = (confidence_scale * wrong_unlikelihood).mean()
    return (
        HARD_NEGATIVE_MARGIN_WEIGHT * margin_term
        + HARD_NEGATIVE_UNLIKELIHOOD_WEIGHT * unlikelihood_term
    )
