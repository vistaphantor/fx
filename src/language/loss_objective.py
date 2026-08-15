"""Authoritative target construction and weighted loss for Vista language training.

Documents use full next-token language modelling. Conversational examples use
assistant-content supervision: user/system prompt tokens and the opening
<assistant> control token provide causal context but do not consume gradient
budget. Packed <eos> -> <bos> transitions are always masked because they are
batching artifacts, not language.

A tiny model can otherwise reduce corpus LM loss while barely learning the
conditional user -> assistant mapping. The authoritative weighted objective
therefore gives assistant-response targets more optimization mass than ordinary
document continuation targets. This is loss weighting, not example replay: all
examples still pass through the same causal transformer and tokenizer contract.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.language.tokenizer import ASSISTANT, BOS, ENDASSISTANT, EOS, SPECIAL_TOKENS

LOSS_OBJECTIVE_VERSION = 4
DEFAULT_ASSISTANT_TARGET_WEIGHT = 4.0

_BOS_ID = SPECIAL_TOKENS.index(BOS)
_EOS_ID = SPECIAL_TOKENS.index(EOS)
_ASSISTANT_ID = SPECIAL_TOKENS.index(ASSISTANT)
_END_ASSISTANT_ID = SPECIAL_TOKENS.index(ENDASSISTANT)


@dataclass(frozen=True, slots=True)
class LossTargetStats:
    prediction_tokens: int
    masked_prompt_tokens: int
    masked_boundary_tokens: int


@dataclass(frozen=True, slots=True)
class WeightedLossStats:
    prediction_tokens: int
    assistant_prediction_tokens: int
    document_prediction_tokens: int
    effective_weight_sum: float


def _example_ranges(sequence: list[int]) -> list[tuple[int, int]]:
    if not sequence:
        return []
    ranges: list[tuple[int, int]] = []
    start = 0
    for index, token_id in enumerate(sequence):
        if token_id == _BOS_ID and index != start:
            if index > start:
                ranges.append((start, index - 1))
            start = index
        if token_id == _EOS_ID:
            ranges.append((start, index))
            start = index + 1
    if start < len(sequence):
        ranges.append((start, len(sequence) - 1))
    return [(a, b) for a, b in ranges if b > a]


def build_loss_targets(
    sequence: list[int],
    *,
    seq_len: int,
    pad_id: int,
) -> tuple[list[int], list[int], LossTargetStats]:
    """Build x/y lists with role-aware supervision under tokenizer v4 IDs."""
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    clipped = list(sequence[: seq_len + 1])
    if len(clipped) < 2:
        raise ValueError("sequence must contain at least two tokens")

    x = clipped[:-1]
    raw_y = clipped[1:]
    y = [pad_id] * len(raw_y)
    prompt_masked = 0
    boundary_masked = 0

    for start, end in _example_ranges(clipped):
        example = clipped[start : end + 1]
        is_chat = _ASSISTANT_ID in example

        if not is_chat:
            for target_index in range(start + 1, end + 1):
                y_index = target_index - 1
                target = clipped[target_index]
                if target == _BOS_ID:
                    boundary_masked += 1
                    continue
                y[y_index] = target
            continue

        inside_assistant = False
        saw_assistant = False
        for target_index in range(start + 1, end + 1):
            y_index = target_index - 1
            target = clipped[target_index]
            if target == _BOS_ID:
                boundary_masked += 1
                continue
            if target == _ASSISTANT_ID:
                inside_assistant = True
                saw_assistant = True
                prompt_masked += 1
                continue
            if inside_assistant:
                y[y_index] = target
                if target == _END_ASSISTANT_ID:
                    inside_assistant = False
                continue
            if target == _EOS_ID and saw_assistant:
                y[y_index] = target
                continue
            prompt_masked += 1

    for index, target in enumerate(raw_y):
        if target == _BOS_ID and y[index] != pad_id:
            y[index] = pad_id
            boundary_masked += 1

    prediction_tokens = sum(1 for token_id in y if token_id != pad_id)

    if len(x) < seq_len:
        padding = seq_len - len(x)
        x.extend([pad_id] * padding)
        y.extend([pad_id] * padding)

    return x, y, LossTargetStats(
        prediction_tokens=prediction_tokens,
        masked_prompt_tokens=prompt_masked,
        masked_boundary_tokens=boundary_masked,
    )


def _assistant_weight_mask(
    x: torch.Tensor,
    targets: torch.Tensor,
    *,
    pad_id: int,
    assistant_weight: float,
) -> tuple[torch.Tensor, WeightedLossStats]:
    if x.shape != targets.shape or x.ndim != 2:
        raise ValueError("weighted_loss_requires_matching_rank2_x_targets")
    if assistant_weight < 1.0:
        raise ValueError("assistant_weight_must_be_at_least_one")

    weights = torch.ones_like(targets, dtype=torch.float32)
    valid = targets != pad_id
    weights.masked_fill_(~valid, 0.0)
    assistant_positions = torch.zeros_like(valid)

    # Packing can place documents and conversations in one row, so classify
    # positions from the actual control-token state rather than at row level.
    for row in range(x.shape[0]):
        inside_assistant = False
        for column in range(x.shape[1]):
            current = int(x[row, column].item())
            target = int(targets[row, column].item())
            if current == _ASSISTANT_ID:
                inside_assistant = True
            if target != pad_id and (
                inside_assistant or current == _END_ASSISTANT_ID
            ):
                assistant_positions[row, column] = True
            if target == _END_ASSISTANT_ID:
                inside_assistant = False
            if target == _EOS_ID and current == _END_ASSISTANT_ID:
                inside_assistant = False

    weights[assistant_positions] = float(assistant_weight)
    prediction_tokens = int(valid.sum().item())
    assistant_tokens = int((assistant_positions & valid).sum().item())
    document_tokens = prediction_tokens - assistant_tokens
    return weights, WeightedLossStats(
        prediction_tokens=prediction_tokens,
        assistant_prediction_tokens=assistant_tokens,
        document_prediction_tokens=document_tokens,
        effective_weight_sum=float(weights.sum().item()),
    )


def weighted_next_token_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    targets: torch.Tensor,
    *,
    pad_id: int,
    assistant_weight: float = DEFAULT_ASSISTANT_TARGET_WEIGHT,
) -> tuple[torch.Tensor, WeightedLossStats]:
    """Cross entropy where assistant targets carry deliberate extra gradient mass."""
    if logits.ndim != 3 or logits.shape[:2] != targets.shape:
        raise ValueError("weighted_loss_logits_shape_mismatch")
    weights, stats = _assistant_weight_mask(
        x,
        targets,
        pad_id=pad_id,
        assistant_weight=assistant_weight,
    )
    if stats.prediction_tokens <= 0 or stats.effective_weight_sum <= 0:
        raise RuntimeError("weighted_loss_has_no_prediction_tokens")
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=pad_id,
        reduction="none",
    ).view_as(targets)
    loss = (token_losses * weights.to(token_losses.device)).sum() / weights.sum().to(token_losses.device)
    return loss, stats
