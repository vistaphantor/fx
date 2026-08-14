from __future__ import annotations

from tools.train_pytorch_50m import (
    PackedSequenceDataset,
    _normalize_prompt_family,
    build_example_sequences,
    split_by_prompt_family,
)
from src.language.tokenizer import BPETokenizer


def _tokenizer() -> BPETokenizer:
    tok = BPETokenizer()
    tok.train(
        (
            "<bos><user>What is RSI?</user><assistant>RSI is momentum.</assistant><eos>"
            "<bos><user>What is ATR?</user><assistant>ATR is volatility.</assistant><eos>"
        ),
        vocab_size=512,
        min_frequency=1,
    )
    return tok


def test_prompt_family_normalizes_terminal_punctuation():
    a = "<bos><user>Solve for x: 2x + 5 = 11.</user><assistant>x=3</assistant><eos>"
    b = "<bos><user>solve for x: 2x + 5 = 11</user><assistant>x is 3</assistant><eos>"
    assert _normalize_prompt_family(a) == _normalize_prompt_family(b)


def test_family_split_never_leaks_same_prompt_family():
    texts = [
        "<bos><user>Solve x.</user><assistant>A</assistant><eos>",
        "<bos><user>solve x</user><assistant>B</assistant><eos>",
        "<bos><user>What is RSI?</user><assistant>C</assistant><eos>",
        "<bos><user>What is ATR?</user><assistant>D</assistant><eos>",
    ]
    train, val = split_by_prompt_family(texts, val_fraction=0.5, seed=42)
    train_families = {_normalize_prompt_family(t) for t in train}
    val_families = {_normalize_prompt_family(t) for t in val}
    assert train_families.isdisjoint(val_families)


def test_short_examples_are_packed_without_destroying_boundaries():
    tok = _tokenizer()
    texts = [
        "<bos><user>A?</user><assistant>A.</assistant><eos>",
        "<bos><user>B?</user><assistant>B.</assistant><eos>",
    ]
    sequences = build_example_sequences(texts, tok, seq_len=128)
    assert len(sequences) == 1
    decoded = tok.decode(sequences[0], skip_special=False)
    assert "<eos><bos>" in decoded
    assert "<user>A?</user>" in decoded
    assert "<user>B?</user>" in decoded


def test_long_example_is_chunked_without_crossing_other_examples():
    tok = _tokenizer()
    long_answer = "reasoning " * 200
    first = f"<bos><user>Long?</user><assistant>{long_answer}</assistant><eos>"
    second = "<bos><user>Short?</user><assistant>Yes.</assistant><eos>"
    sequences = build_example_sequences([first, second], tok, seq_len=64)
    decoded = [tok.decode(seq, skip_special=False) for seq in sequences]
    # No long-example chunk may contain the second prompt; the short example is
    # emitted only after the long example's independent windows.
    long_chunks = [text for text in decoded if "Long?" in text or "reasoning" in text]
    assert long_chunks
    assert all("Short?" not in text for text in long_chunks)


def test_packed_dataset_returns_aligned_fixed_shapes():
    tok = _tokenizer()
    sequences = build_example_sequences(
        ["<bos><user>A?</user><assistant>A.</assistant><eos>"],
        tok,
        seq_len=32,
    )
    ds = PackedSequenceDataset(sequences, seq_len=32, pad_id=tok.pad_id())
    x, y = ds[0]
    assert tuple(x.shape) == (32,)
    assert tuple(y.shape) == (32,)
    # The first target must be the second token of the original sequence.
    assert y[0].item() == sequences[0][1]
