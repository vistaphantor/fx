from __future__ import annotations

import torch

from src.language.pytorch_transformer import VistaReasoningGPT


def _model(*, max_seq_len: int = 8) -> VistaReasoningGPT:
    torch.manual_seed(42)
    model = VistaReasoningGPT(
        vocab_size=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        ffn_dim=64,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )
    model.eval()
    return model


def test_generation_stops_on_configured_token(monkeypatch):
    model = _model()
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
    model = _model()
    prompt = torch.tensor([[2, 3], [2, 4]], dtype=torch.long)
    try:
        model.generate(prompt, max_new_tokens=1)
    except ValueError as exc:
        assert "single batch item" in str(exc)
    else:
        raise AssertionError("generation accepted unsupported multi-item batch")


def test_kv_cached_greedy_generation_matches_reference():
    model = _model(max_seq_len=16)
    prompt = torch.tensor([[2, 5, 8, 11]], dtype=torch.long)
    cached = model.generate(
        prompt.clone(),
        max_new_tokens=6,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        use_kv_cache=True,
    )
    reference = model.generate(
        prompt.clone(),
        max_new_tokens=6,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        use_kv_cache=False,
    )
    assert torch.equal(cached, reference)


def test_kv_cached_generation_matches_reference_across_context_rollover():
    model = _model(max_seq_len=6)
    prompt = torch.tensor([[2, 5, 8, 11, 13]], dtype=torch.long)
    cached = model.generate(
        prompt.clone(),
        max_new_tokens=8,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        use_kv_cache=True,
    )
    reference = model.generate(
        prompt.clone(),
        max_new_tokens=8,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        use_kv_cache=False,
    )
    assert torch.equal(cached, reference)
