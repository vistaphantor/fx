from __future__ import annotations

import torch

from src.language.hard_negative_objective import (
    hard_negative_answer_penalty,
    repetition_unlikelihood_penalty,
)


def _penalty(values: list[float], target: int = 1) -> float:
    # Answer-specific punishment requires evidence of masked prompt context.
    logits = torch.tensor([[[0.0] * len(values), values]], dtype=torch.float32)
    targets = torch.tensor([[0, target]], dtype=torch.long)
    return float(hard_negative_answer_penalty(logits, targets, pad_id=0).item())


def test_confident_wrong_answer_is_punished_more_than_uncertain_wrong_answer() -> None:
    uncertain = _penalty([0.0, 0.0, 0.2, 0.1])
    confident_wrong = _penalty([0.0, -1.0, 5.0, -2.0])
    assert confident_wrong > uncertain * 2.0


def test_correct_answer_with_clear_margin_has_small_penalty() -> None:
    correct = _penalty([-3.0, 5.0, -2.0, -4.0])
    wrong = _penalty([-3.0, -2.0, 5.0, -4.0])
    assert correct < 0.01
    assert wrong > correct


def test_gradient_pushes_correct_up_and_hard_wrong_down() -> None:
    logits = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 4.0, 0.5]]],
        requires_grad=True,
    )
    targets = torch.tensor([[0, 1]], dtype=torch.long)
    loss = hard_negative_answer_penalty(logits, targets, pad_id=0)
    loss.backward()

    gradient = logits.grad[0, 1]
    assert gradient[1].item() < 0
    assert gradient[2].item() > 0


def test_document_start_receives_no_answer_specific_penalty() -> None:
    logits = torch.tensor([[[0.0, -1.0, 5.0, -2.0]]], dtype=torch.float32)
    targets = torch.tensor([[1]], dtype=torch.long)
    penalty = hard_negative_answer_penalty(logits, targets, pad_id=0)
    assert penalty.item() == 0.0


def test_padding_positions_receive_no_penalty() -> None:
    logits = torch.randn(1, 2, 5)
    targets = torch.zeros(1, 2, dtype=torch.long)
    penalty = hard_negative_answer_penalty(logits, targets, pad_id=0)
    assert penalty.item() == 0.0


def test_repetition_loop_candidate_is_directly_penalized() -> None:
    # Token 3 has already appeared repeatedly in context. At the final position
    # the teacher wants token 4, while the model is trying to emit token 3 again.
    input_ids = torch.tensor([[1, 3, 2, 3, 3]], dtype=torch.long)
    targets = torch.tensor([[0, 0, 0, 0, 4]], dtype=torch.long)
    logits = torch.zeros(1, 5, 6, requires_grad=True)
    with torch.no_grad():
        logits[0, 4, 3] = 5.0
        logits[0, 4, 4] = 0.5
    penalty = repetition_unlikelihood_penalty(
        logits,
        input_ids,
        targets,
        pad_id=0,
    )
    assert penalty.item() > 0
    penalty.backward()
    assert logits.grad[0, 4, 3].item() > 0


def test_teacher_requested_repetition_is_not_punished() -> None:
    input_ids = torch.tensor([[1, 3, 2, 3, 3]], dtype=torch.long)
    targets = torch.tensor([[0, 0, 0, 0, 3]], dtype=torch.long)
    logits = torch.zeros(1, 5, 6)
    penalty = repetition_unlikelihood_penalty(
        logits,
        input_ids,
        targets,
        pad_id=0,
    )
    assert penalty.item() == 0.0
