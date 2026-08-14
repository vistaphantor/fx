from __future__ import annotations

from src.language.tokenizer import BPETokenizer
from src.language.training_pipeline import build_example_sequences


def test_long_reasoning_chunks_repeat_question_context_and_only_final_chunk_closes_reasoning():
    tokenizer = BPETokenizer()
    tokenizer.train(
        "<bos><user>Evaluate setup?</user><assistant><think>reasoning evidence market risk</think>WAIT</assistant><eos>",
        vocab_size=512,
        min_frequency=1,
    )
    reasoning = "reasoning evidence market risk " * 120
    text = (
        "<bos><user>Evaluate setup?</user><assistant><think>"
        + reasoning
        + "</think>WAIT</assistant><eos>"
    )
    sequences = build_example_sequences([text], tokenizer, seq_len=96)
    assert len(sequences) > 1
    decoded = [tokenizer.decode(sequence, skip_special=False) for sequence in sequences]
    assert all("Evaluate setup?" in chunk for chunk in decoded)
    assert all("<assistant><think>" in chunk for chunk in decoded)
    assert all("</think>" not in chunk for chunk in decoded[:-1])
    assert "</think>WAIT</assistant><eos>" in decoded[-1]
