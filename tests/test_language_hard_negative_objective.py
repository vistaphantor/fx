from __future__ import annotations

import torch

from src.language.hard_negative_objective import hard_negative_answer_penalty


def _penalty(values: list[float], target: int = 1) -> float:
    logits = torch.tensor([[values]], dtype=torch.float32)
    targets = torch.tensor([[target]], dtype=torch.long)
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
    logits = torch.tensor([[[0.0, -1.0, 4.0, 0.5]]], requires_grad=True)
    targets = torch.tensor([[1]], dtype=torch.long)
    loss = hard_negative_answer_penalty(logits, targets, pad_id=0)
    loss.backward()

    gradient = logits.grad[0, 0]
    # Gradient descent subtracts the gradient: negative raises the correct logit,
    # positive lowers the confidently preferred wrong logit.
    assert gradient[1].item() < 0
    assert gradient[2].item() > 0


def test_padding_positions_receive_no_penalty() -> None:
    logits = torch.randn(1, 2, 5)
    targets = torch.zeros(1, 2, dtype=torch.long)
    penalty = hard_negative_answer_penalty(logits, targets, pad_id=0)
    assert penalty.item() == 0.0
