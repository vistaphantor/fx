from __future__ import annotations

import hashlib

import torch

from src.language.loss_objective import build_loss_targets
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.streaming_sources import (
    FOUNDATION_INTERACTION_FRACTION,
    _continuation_interaction,
    _use_foundation_interaction,
)
from src.language.tokenizer import ASSISTANT, BPETokenizer, USER


def _document() -> str:
    words = [
        "The", "small", "bird", "flew", "over", "the", "green", "field",
        "while", "the", "children", "walked", "home", "after", "school", "and",
        "talked", "about", "their", "day", "because", "the", "weather", "was",
        "warm", "and", "the", "road", "was", "quiet", "near", "the",
        "river", "where", "their", "parents", "were", "waiting", "for", "them",
        "with", "food", "and", "water", "before", "the", "evening", "rain",
        "started", "falling", "across", "the", "village", "and", "everyone", "went",
        "inside", "to", "rest", "and", "prepare", "for", "the", "next", "morning",
        "when", "the", "sun", "would", "rise", "again", "over", "the", "hills",
        "and", "the", "birds", "would", "begin", "singing", "outside", "again",
    ]
    return "<bos>\n" + " ".join(words) + "\n<eos>"


def _interaction_and_tokenizer() -> tuple[str, BPETokenizer]:
    transformed = _continuation_interaction(_document())
    assert transformed is not None
    tok = BPETokenizer()
    tok.train(transformed, vocab_size=512, min_frequency=1)
    return transformed, tok


def test_foundation_interaction_is_real_prefix_to_real_continuation() -> None:
    transformed = _continuation_interaction(_document())
    assert transformed is not None
    assert "<user>" in transformed
    assert "<assistant>" in transformed
    assert "Continue this passage naturally:" in transformed
    assert "river where their parents" in transformed


def test_foundation_interaction_targets_only_assistant_lane() -> None:
    transformed, tok = _interaction_and_tokenizer()
    ids = tok.encode(transformed, add_bos=False, add_eos=False)
    _, targets, stats = build_loss_targets(ids, seq_len=191, pad_id=tok.pad_id())

    assert stats.prediction_tokens > 0
    user_id = tok.vocab[USER]
    assistant_id = tok.vocab[ASSISTANT]
    for index, token_id in enumerate(ids[1:], start=0):
        if token_id in {user_id, assistant_id}:
            assert targets[index] == tok.pad_id()


def test_tiny_model_can_learn_assistant_continuation_objective() -> None:
    torch.manual_seed(7)
    transformed, tok = _interaction_and_tokenizer()
    ids = tok.encode(transformed, add_bos=False, add_eos=False)
    seq_len = min(191, len(ids) - 1)
    x, y, stats = build_loss_targets(ids, seq_len=seq_len, pad_id=tok.pad_id())
    assert stats.prediction_tokens > 0

    model = VistaReasoningGPT(
        vocab_size=tok.vocab_size,
        d_model=48,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=128,
        max_seq_len=seq_len,
        dropout=0.0,
        ffn_type="dense",
        num_experts=1,
        experts_per_token=1,
        moe_ffn_dim=128,
        shared_expert_ffn_dim=0,
        router_aux_loss_coef=0.0,
        router_jitter=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    xb = torch.tensor([x], dtype=torch.long)
    yb = torch.tensor([y], dtype=torch.long)

    first = None
    last = None
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, targets=yb, pad_id=tok.pad_id())
        assert loss is not None and torch.isfinite(loss)
        if first is None:
            first = float(loss.item())
        loss.backward()
        optimizer.step()
        last = float(loss.item())

    assert first is not None and last is not None
    assert last < first * 0.30


def test_interaction_selection_is_deterministic_and_near_configured_fraction() -> None:
    selected = 0
    total = 2000
    for index in range(total):
        digest = hashlib.sha256(f"example-{index}".encode()).hexdigest()
        first = _use_foundation_interaction(digest)
        second = _use_foundation_interaction(digest)
        assert first == second
        selected += int(first)

    observed = selected / total
    assert abs(observed - FOUNDATION_INTERACTION_FRACTION) < 0.04
