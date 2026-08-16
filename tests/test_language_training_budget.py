from __future__ import annotations

import torch

from src.language.foundation_contract import (
    FOUNDATION_EXAM_INTERVAL_SECONDS,
    FOUNDATION_MIN_AVAILABLE_CURRICULUM_TOKENS,
    FOUNDATION_TARGET_PREDICTION_TOKENS,
)
from src.language.streaming_sources import load_hf_source_config, require_curriculum_capacity
from src.language.training_profiles import DEFAULT_STAGE_TOKENS_PER_PARAMETER, PROFILES
from src.language.training_runtime import cap_prediction_targets, set_token_scheduled_lr


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
        work = int(cfg["n_layers"]) * d_model * d_model
        assert work > previous_work
        previous_work = work


def test_foundation_budget_is_exactly_eight_billion_prediction_tokens():
    assert FOUNDATION_MIN_AVAILABLE_CURRICULUM_TOKENS == 8_000_000_000
    assert FOUNDATION_TARGET_PREDICTION_TOKENS == 8_000_000_000
    assert FOUNDATION_EXAM_INTERVAL_SECONDS == 4 * 60 * 60


def test_pinned_curriculum_exposes_more_than_required_eight_billion_tokens():
    inventory = require_curriculum_capacity(load_hf_source_config("config/hf_sources.json"), "foundation")
    assert inventory.available_tokens >= FOUNDATION_MIN_AVAILABLE_CURRICULUM_TOKENS
    assert inventory.available_tokens == 24_700_000_000
    assert len(inventory.skills) == 14


def test_later_stage_parameter_targets_remain_positive():
    assert "foundation" not in DEFAULT_STAGE_TOKENS_PER_PARAMETER
    assert DEFAULT_STAGE_TOKENS_PER_PARAMETER["reasoning"] > 0
    assert DEFAULT_STAGE_TOKENS_PER_PARAMETER["trading_reasoning"] > 0


def test_learning_rate_is_a_function_of_cumulative_tokens():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    target = 1_000_000
    warm = set_token_scheduled_lr(
        optimizer, base_lr=1e-3, min_lr=1e-5,
        cumulative_tokens=5_000, target_tokens=target, warmup_fraction=0.02,
    )
    peak = set_token_scheduled_lr(
        optimizer, base_lr=1e-3, min_lr=1e-5,
        cumulative_tokens=20_000, target_tokens=target, warmup_fraction=0.02,
    )
    middle = set_token_scheduled_lr(
        optimizer, base_lr=1e-3, min_lr=1e-5,
        cumulative_tokens=500_000, target_tokens=target, warmup_fraction=0.02,
    )
    final = set_token_scheduled_lr(
        optimizer, base_lr=1e-3, min_lr=1e-5,
        cumulative_tokens=target, target_tokens=target, warmup_fraction=0.02,
    )
    assert 1e-5 < warm < peak
    assert peak == 1e-3
    assert 1e-5 < middle < peak
    assert final == 1e-5


def test_final_batch_is_masked_to_land_on_exact_prediction_token_target():
    pad = 0
    targets = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
    capped, counted = cap_prediction_targets(targets, pad_id=pad, remaining_tokens=4)
    assert counted == 4
    assert int((capped != pad).sum().item()) == 4
    assert capped.tolist() == [[1, 2, 3, 4], [0, 0, 0, 0]]
