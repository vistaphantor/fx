from __future__ import annotations

import hashlib

from src.language.loss_objective import build_loss_targets
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


def test_foundation_interaction_is_real_prefix_to_real_continuation() -> None:
    transformed = _continuation_interaction(_document())
    assert transformed is not None
    assert "<user>" in transformed
    assert "<assistant>" in transformed
    assert "Continue this passage naturally:" in transformed
    assert "river where their parents" in transformed


def test_foundation_interaction_targets_only_assistant_lane() -> None:
    transformed = _continuation_interaction(_document())
    assert transformed is not None

    tok = BPETokenizer()
    tok.train(transformed, vocab_size=512, min_frequency=1)
    ids = tok.encode(transformed, add_bos=False, add_eos=False)
    _, targets, stats = build_loss_targets(ids, seq_len=191, pad_id=tok.pad_id())

    assert stats.prediction_tokens > 0
    user_id = tok.vocab[USER]
    assistant_id = tok.vocab[ASSISTANT]
    for index, token_id in enumerate(ids[1:], start=0):
        if token_id in {user_id, assistant_id}:
            assert targets[index] == tok.pad_id()


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
