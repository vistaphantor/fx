import torch
import torch.nn.functional as F

from src.language.pytorch_transformer import VistaReasoningGPT


def _tiny_model():
    torch.manual_seed(42)
    return VistaReasoningGPT(
        vocab_size=64,
        d_model=32,
        n_layers=1,
        n_heads=4,
        ffn_dim=64,
        max_seq_len=16,
        dropout=0.0,
    )


def test_next_token_targets_are_not_shifted_twice():
    model = _tiny_model()
    model.eval()

    x = torch.tensor(
        [[2, 10, 11, 12, 13]],
        dtype=torch.long,
    )

    # Correct next-token alignment:
    # 2 -> 10
    # 10 -> 11
    # 11 -> 12
    # 12 -> 13
    # 13 -> 3
    y = torch.tensor(
        [[10, 11, 12, 13, 3]],
        dtype=torch.long,
    )

    with torch.no_grad():
        logits, direct_loss = model(
            x,
            targets=y,
            pad_id=0,
        )

    expected_loss = F.cross_entropy(
        logits.reshape(-1, model.vocab_size),
        y.reshape(-1),
        ignore_index=0,
    )

    assert torch.allclose(
        direct_loss,
        expected_loss,
        atol=1e-7,
    )


def test_targets_must_match_input_shape():
    model = _tiny_model()

    x = torch.tensor(
        [[2, 10, 11, 12]],
        dtype=torch.long,
    )
    invalid_targets = torch.tensor(
        [[10, 11, 12]],
        dtype=torch.long,
    )

    try:
        model(
            x,
            targets=invalid_targets,
            pad_id=0,
        )
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError(
            "Model accepted misaligned training targets"
        )
