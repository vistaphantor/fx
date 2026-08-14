from __future__ import annotations

from src.language.compute_budget import (
    benchmark_training_throughput,
    reference_token_target,
    required_tokens_per_second,
)
from src.language.tokenizer import BPETokenizer
from src.language.training_pipeline import build_example_sequences


def test_reference_token_math_is_explicit():
    assert reference_token_target(8_000_000) == 160_000_000
    assert required_tokens_per_second(160_000_000, 4.0) > 11_000


def test_real_training_probe_reports_positive_throughput():
    text = (
        "<bos>\n<user>\nWhat is 2 + 2?\n</user>\n"
        "<assistant>\n4\n</assistant>\n<eos>"
    )
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=512, min_frequency=1)
    sequences = build_example_sequences([text], tokenizer, seq_len=16)
    report = benchmark_training_throughput(
        model_config={
            "vocab_size": tokenizer.vocab_size,
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 4,
            "ffn_dim": 64,
            "max_seq_len": 16,
            "dropout": 0.0,
        },
        tokenizer=tokenizer,
        sequences=sequences,
        batch_size=1,
        steps=1,
        wall_clock_hours=4.0,
    )
    assert report.parameter_count > 0
    assert report.useful_tokens_per_second > 0
    assert report.projected_useful_tokens > 0
    assert report.reference_target_tokens > report.parameter_count
    assert report.projected_hours_to_reference_target > 0
