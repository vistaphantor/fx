"""Authoritative target construction for Vista language training.

Documents use full next-token language modelling. Conversational examples use
assistant-content supervision: user/system prompt tokens and the opening
<assistant> control token provide causal context but do not consume gradient
budget. Inference already seeds <assistant>, so teaching the model to emit that
same opener creates a duplicated-control-token attractor. Packed <eos> -> <bos>
transitions are always masked because they are batching artifacts, not language.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.language.tokenizer import ASSISTANT, BOS, ENDASSISTANT, EOS, SPECIAL_TOKENS

LOSS_OBJECTIVE_VERSION = 2

_BOS_ID = SPECIAL_TOKENS.index(BOS)
_EOS_ID = SPECIAL_TOKENS.index(EOS)
_ASSISTANT_ID = SPECIAL_TOKENS.index(ASSISTANT)
_END_ASSISTANT_ID = SPECIAL_TOKENS.index(ENDASSISTANT)


@dataclass(frozen=True, slots=True)
class LossTargetStats:
    prediction_tokens: int
    masked_prompt_tokens: int
    masked_boundary_tokens: int


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
                # Generation is seeded with <assistant>; this token is context,
                # never a prediction target. Masking it prevents repeated role
                # markers from becoming a cheap high-confidence attractor.
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
    if prediction_tokens <= 0:
        raise RuntimeError("loss_objective_has_no_prediction_tokens")

    if len(x) < seq_len:
        padding = seq_len - len(x)
        x.extend([pad_id] * padding)
        y.extend([pad_id] * padding)

    return x, y, LossTargetStats(
        prediction_tokens=prediction_tokens,
        masked_prompt_tokens=prompt_masked,
        masked_boundary_tokens=boundary_masked,
    )
