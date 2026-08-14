from __future__ import annotations

import torch

from src.language.training_profiles import DEFAULT_STAGE_TOKENS_PER_PARAMETER, PROFILES
from tools.train_language_reasoner import _set_token_scheduled_lr


def test_profiles_have_valid_attention_geometry_and_compute_ordering():
    expected_order = ["2m", "4m", "8m", "15m"]
    previous_work = 0
    for name in expected_order:
        cfg = PROFILES[name]
        d_model = int(cfg["d_model"])
        heads = int(cfg["n_heads"])
        assert d_model % heads == 0
        assert int(cfg["seq_len"]) > 0
        assert int(cfg["batch_size"]) > 0
        # Rough architecture work proxy must increase with profile size.
        work = int(cfg["n_layers"]) * d_model * d_model
        assert work > previous_work
        previous_work = work


def test_stage_token_targets_are_positive_and_foundation_is_largest():
    assert DEFAULT_STAGE_TOKENS_PER_PARAMETER["foundation"] > 0
    assert DEFAULT_STAGE_TOKENS_PER_PARAMETER["reasoning"] > 0
    assert DEFAULT_STAGE_TOKENS_PER_PARAMETER["trading_reasoning"] > 0
    assert DEFAULT_STAGE_TOKENS_PER_PARAMETER["foundation"] > DEFAULT_STAGE_TOKENS_PER_PARAMETER["reasoning"]
    assert DEFAULT_STAGE_TOKENS_PER_PARAMETER["foundation"] > DEFAULT_STAGE_TOKENS_PER_PARAMETER["trading_reasoning"]


def test_learning_rate_is_a_function_of_cumulative_tokens():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    target = 1_000_000

    warm = _set_token_scheduled_lr(
        optimizer,
        base_lr=1e-3,
        min_lr=1e-5,
        cumulative_tokens=5_000,
        target_tokens=target,
        warmup_fraction=0.02,
    )
    peak = _set_token_scheduled_lr(
        optimizer,
        base_lr=1e-3,
        min_lr=1e-5,
        cumulative_tokens=20_000,
        target_tokens=target,
        warmup_fraction=0.02,
    )
    middle = _set_token_scheduled_lr(
        optimizer,
        base_lr=1e-3,
        min_lr=1e-5,
        cumulative_tokens=500_000,
        target_tokens=target,
        warmup_fraction=0.02,
    )
    final = _set_token_scheduled_lr(
        optimizer,
        base_lr=1e-3,
        min_lr=1e-5,
        cumulative_tokens=target,
        target_tokens=target,
        warmup_fraction=0.02,
    )

    assert 1e-5 < warm < peak
    assert peak == 1e-3
    assert 1e-5 < middle < peak
    assert final == 1e-5


def test_four_hour_target_is_not_hardcoded_to_billions():
    # Training target scales from model parameters; the four-hour value is a
    # resumable session limit measured by preflight, not a fabricated token goal.
    assert DEFAULT_STAGE_TOKENS_PER_PARAMETER["foundation"] == 20.0
