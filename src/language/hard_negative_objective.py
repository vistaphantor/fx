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
    This objective adds two terms on every supervised assistant target:

    * a logit margin requiring the correct target to beat the strongest wrong
      candidate by ``HARD_NEGATIVE_MARGIN``;
    * unlikelihood on that strongest wrong candidate.

    Both terms are scaled by the model's detached probability for the hard wrong
    candidate, so confidently wrong predictions receive materially larger
    gradients while uncertain early-training distributions are not destabilized.
    Prompt and padding positions remain excluded by the authoritative target mask.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, time, vocab]")
    if targets.shape != logits.shape[:2]:
        raise ValueError("targets must match logits batch/time dimensions")

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    valid = flat_targets != int(pad_id)
    if not torch.any(valid):
        return logits.new_zeros(())

    supervised_logits = flat_logits[valid]
    supervised_targets = flat_targets[valid]
    correct_logits = supervised_logits.gather(1, supervised_targets[:, None]).squeeze(1)

    wrong_logits = supervised_logits.clone()
    wrong_logits.scatter_(1, supervised_targets[:, None], -torch.inf)
    hard_wrong_logits, _ = wrong_logits.max(dim=1)

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
