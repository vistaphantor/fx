from __future__ import annotations

from src.language.tokenizer import BPETokenizer
from src.language.training_pipeline import (
    PackedSequenceDataset,
    build_example_sequences,
    normalize_prompt_family,
    run_training_preflight,
    split_by_prompt_family,
)


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
    assert normalize_prompt_family(a) == normalize_prompt_family(b)


def test_family_split_never_leaks_same_prompt_family():
    texts = [
        "<bos><user>Solve x.</user><assistant>A</assistant><eos>",
        "<bos><user>solve x</user><assistant>B</assistant><eos>",
        "<bos><user>What is RSI?</user><assistant>C</assistant><eos>",
        "<bos><user>What is ATR?</user><assistant>D</assistant><eos>",
    ]
    train, val = split_by_prompt_family(texts, val_fraction=0.5, seed=42)
    train_families = {normalize_prompt_family(text) for text in train}
    val_families = {normalize_prompt_family(text) for text in val}
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
    decoded = [tok.decode(sequence, skip_special=False) for sequence in sequences]
    long_chunks = [text for text in decoded if "Long?" in text or "reasoning" in text]
    assert long_chunks
    assert all("Short?" not in text for text in long_chunks)


def test_packed_dataset_masks_prompt_and_supervises_assistant():
    tok = _tokenizer()
    sequences = build_example_sequences(
        ["<bos><user>A?</user><assistant>A.</assistant><eos>"],
        tok,
        seq_len=32,
    )
    dataset = PackedSequenceDataset(sequences, seq_len=32, pad_id=tok.pad_id())
    x, y = dataset[0]
    assert tuple(x.shape) == (32,)
    assert tuple(y.shape) == (32,)
    # <user> and its prompt are context, not gradient targets.
    assert y[0].item() == tok.pad_id()
    assistant_id = tok.vocab["<assistant>"]
    assistant_target_positions = [
        index - 1 for index, token in enumerate(sequences[0]) if token == assistant_id
    ]
    assert assistant_target_positions
    assert all(y[position].item() == assistant_id for position in assistant_target_positions)


def test_preflight_proves_roundtrip_alignment_and_learning():
    tok = _tokenizer()
    texts = [
        "<bos><user>What is RSI?</user><assistant>RSI is momentum.</assistant><eos>",
        "<bos><user>What is ATR?</user><assistant>ATR is volatility.</assistant><eos>",
        "<bos><user>What is risk?</user><assistant>Risk is potential loss.</assistant><eos>",
        "<bos><user>What is spread?</user><assistant>Spread is bid minus ask.</assistant><eos>",
    ]
    train, val = split_by_prompt_family(texts, val_fraction=0.25, seed=42)
    train_sequences = build_example_sequences(train, tok, seq_len=64)
    val_sequences = build_example_sequences(val, tok, seq_len=64)
    report = run_training_preflight(
        tokenizer=tok,
        train_texts=train,
        val_texts=val,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        seq_len=64,
    )
    assert report.tokenizer_algorithm_version == 4
    assert report.roundtrip_cases >= 8
    assert report.overfit_final_loss < report.overfit_initial_loss * 0.70
