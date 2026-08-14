"""Authoritative target construction for Vista language training.

Documents use full next-token language modelling. Conversational examples use
assistant-supervised loss: user/system prompt tokens provide causal context but
do not consume gradient budget. Packed <eos> -> <bos> transitions are always
masked because they are batching artifacts, not language.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.language.tokenizer import BPETokenizer


@dataclass(frozen=True, slots=True)
class LossTargetStats:
    prediction_tokens: int
    masked_prompt_tokens: int
    masked_boundary_tokens: int


def _example_ranges(sequence: list[int], *, bos_id: int, eos_id: int) -> list[tuple[int, int]]:
    """Return inclusive [start, end] ranges for packed canonical examples."""
    if not sequence:
        return []
    ranges: list[tuple[int, int]] = []
    start = 0
    for index, token_id in enumerate(sequence):
        if token_id == bos_id and index != start:
            # Recover safely from a missing EOS rather than merging examples.
            if index > start:
                ranges.append((start, index - 1))
            start = index
        if token_id == eos_id:
            ranges.append((start, index))
            start = index + 1
    if start < len(sequence):
        ranges.append((start, len(sequence) - 1))
    return [(a, b) for a, b in ranges if b > a]


def build_loss_targets(
    sequence: list[int],
    tokenizer: BPETokenizer,
    *,
    seq_len: int,
) -> tuple[list[int], list[int], LossTargetStats]:
    """Build x/y tensors as lists with role-aware supervision.

    For a canonical conversational example, only assistant spans are optimized.
    The opening <assistant> token, assistant content/reasoning, </assistant>, and
    final <eos> are supervised. User/system/context spans are masked with pad_id.

    For a document example (no <assistant> token), every ordinary next-token
    target is supervised except a packed <bos> boundary.
    """
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    clipped = list(sequence[: seq_len + 1])
    if len(clipped) < 2:
        raise ValueError("sequence must contain at least two tokens")

    pad_id = tokenizer.pad_id()
    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()
    assistant_id = tokenizer.vocab["<assistant>"]
    end_assistant_id = tokenizer.vocab["</assistant>"]

    x = clipped[:-1]
    raw_y = clipped[1:]
    y = [pad_id] * len(raw_y)
    prompt_masked = 0
    boundary_masked = 0

    for start, end in _example_ranges(clipped, bos_id=bos_id, eos_id=eos_id):
        example = clipped[start : end + 1]
        is_chat = assistant_id in example

        if not is_chat:
            # Full LM objective for documents. y position p predicts clipped[p+1].
            for target_index in range(start + 1, end + 1):
                y_index = target_index - 1
                target = clipped[target_index]
                if target == bos_id:
                    boundary_masked += 1
                    continue
                y[y_index] = target
            continue

        inside_assistant = False
        saw_assistant = False
        for target_index in range(start + 1, end + 1):
            y_index = target_index - 1
            target = clipped[target_index]
            if target == bos_id:
                boundary_masked += 1
                continue
            if target == assistant_id:
                inside_assistant = True
                saw_assistant = True
                y[y_index] = target
                continue
            if inside_assistant:
                y[y_index] = target
                if target == end_assistant_id:
                    inside_assistant = False
                continue
            if target == eos_id and saw_assistant:
                # Teach clean response termination after the last assistant turn.
                y[y_index] = target
                continue
            prompt_masked += 1

    # Any packed BOS target not encountered through malformed ranges is masked.
    for index, target in enumerate(raw_y):
        if target == bos_id and y[index] != pad_id:
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
