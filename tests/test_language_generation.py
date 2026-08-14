from __future__ import annotations

import torch

from src.language.pytorch_transformer import VistaReasoningGPT


def test_generation_stops_on_configured_token(monkeypatch):
    model = VistaReasoningGPT(
        vocab_size=16,
        d_model=16,
        n_layers=1,
        n_heads=4,
        ffn_dim=32,
        max_seq_len=8,
        dropout=0.0,
    )
    stop_id = 7

    def fake_multinomial(_probs, num_samples):
        assert num_samples == 1
        return torch.tensor([[stop_id]], dtype=torch.long)

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)
    prompt = torch.tensor([[2, 3]], dtype=torch.long)
    output = model.generate(
        prompt,
        max_new_tokens=5,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        stop_ids={stop_id},
    )
    assert output.shape[1] == prompt.shape[1] + 1
    assert output[0, -1].item() == stop_id


def test_generation_rejects_untrained_batch_mode():
    model = VistaReasoningGPT(
        vocab_size=16,
        d_model=16,
        n_layers=1,
        n_heads=4,
        ffn_dim=32,
        max_seq_len=8,
        dropout=0.0,
    )
    prompt = torch.tensor([[2, 3], [2, 4]], dtype=torch.long)
    try:
        model.generate(prompt, max_new_tokens=1)
    except ValueError as exc:
        assert "single batch item" in str(exc)
    else:
        raise AssertionError("generation accepted unsupported multi-item batch")
