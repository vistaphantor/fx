from __future__ import annotations

PROFILES: dict[str, dict[str, int | float | str | None]] = {
    "smoke": {
        "vocab_size": 1024, "d_model": 128, "n_heads": 4, "n_kv_heads": 2,
        "n_layers": 4, "ffn_dim": 384, "ffn_type": "dense", "num_experts": 1,
        "experts_per_token": 1, "moe_ffn_dim": 384, "shared_expert_ffn_dim": 0,
        "router_aux_loss_coef": 0.0, "router_jitter": 0.0, "rope_theta": 10000.0,
        "seq_len": 128, "batch_size": 4, "lr": 8e-4, "dropout": 0.05, "tokenizer_chars": 1_500_000,
        "exam_max_new_tokens": 48, "hf_preflight_sample_examples": 1000,
        "checkpoint_every_steps": 50,
    },
    "2m_dense": {
        "vocab_size": 2048, "d_model": 160, "n_heads": 8, "n_kv_heads": 2,
        "n_layers": 5, "ffn_dim": 512, "ffn_type": "dense", "num_experts": 1,
        "experts_per_token": 1, "moe_ffn_dim": 512, "shared_expert_ffn_dim": 0,
        "router_aux_loss_coef": 0.0, "router_jitter": 0.0, "rope_theta": 10000.0,
        "seq_len": 192, "batch_size": 4, "lr": 4e-4, "dropout": 0.04, "tokenizer_chars": 3_000_000,
        "exam_max_new_tokens": 64, "hf_preflight_sample_examples": 2500,
        "checkpoint_every_steps": 100,
    },
    "2m": {
        "vocab_size": 2048, "d_model": 128, "n_heads": 8, "n_kv_heads": 2,
        "n_layers": 4, "ffn_dim": 384, "ffn_type": "moe", "num_experts": 8,
        "experts_per_token": 1, "moe_ffn_dim": 128, "shared_expert_ffn_dim": 64,
        "router_aux_loss_coef": 0.01, "router_jitter": 0.01, "rope_theta": 10000.0,
        "seq_len": 192, "batch_size": 4, "lr": 4e-4, "dropout": 0.04, "tokenizer_chars": 3_000_000,
        "exam_max_new_tokens": 64, "hf_preflight_sample_examples": 2500,
        "checkpoint_every_steps": 100,
    },
    "4m": {
        "vocab_size": 3072, "d_model": 160, "n_heads": 8, "n_kv_heads": 2,
        "n_layers": 6, "ffn_dim": 480, "ffn_type": "moe", "num_experts": 8,
        "experts_per_token": 1, "moe_ffn_dim": 160, "shared_expert_ffn_dim": 80,
        "router_aux_loss_coef": 0.01, "router_jitter": 0.01, "rope_theta": 10000.0,
        "seq_len": 256, "batch_size": 2, "lr": 4e-4, "dropout": 0.06, "tokenizer_chars": 4_000_000,
        "exam_max_new_tokens": 64, "hf_preflight_sample_examples": 3000,
        "checkpoint_every_steps": 100,
    },
    "8m": {
        "vocab_size": 4096, "d_model": 192, "n_heads": 8, "n_kv_heads": 2,
        "n_layers": 8, "ffn_dim": 576, "ffn_type": "moe", "num_experts": 8,
        "experts_per_token": 1, "moe_ffn_dim": 192, "shared_expert_ffn_dim": 96,
        "router_aux_loss_coef": 0.01, "router_jitter": 0.01, "rope_theta": 10000.0,
        "seq_len": 256, "batch_size": 2, "lr": 3.5e-4, "dropout": 0.07, "tokenizer_chars": 4_000_000,
        "exam_max_new_tokens": 64, "hf_preflight_sample_examples": 3000,
        "checkpoint_every_steps": 100,
    },
    "15m": {
        "vocab_size": 8192, "d_model": 256, "n_heads": 8, "n_kv_heads": 2,
        "n_layers": 10, "ffn_dim": 768, "ffn_type": "moe", "num_experts": 8,
        "experts_per_token": 2, "moe_ffn_dim": 256, "shared_expert_ffn_dim": 128,
        "router_aux_loss_coef": 0.01, "router_jitter": 0.01, "rope_theta": 10000.0,
        "seq_len": 320, "batch_size": 1, "lr": 3e-4, "dropout": 0.08, "tokenizer_chars": 8_000_000,
        "exam_max_new_tokens": 64, "hf_preflight_sample_examples": 5000,
        "checkpoint_every_steps": 100,
    },
}

# Foundation no longer has a tokens-per-parameter stop condition. Its canonical
# target is FOUNDATION_TARGET_PREDICTION_TOKENS in foundation_contract.py.
DEFAULT_STAGE_TOKENS_PER_PARAMETER = {
    "reasoning": 6.0,
    "trading_reasoning": 8.0,
}

DEFAULT_LOCAL_REPLAY_WEIGHTS = {
    "foundation": 0.0,
    "reasoning": 0.0,
    "trading_reasoning": 0.0,
}


def profile(name: str) -> dict:
    if name not in PROFILES:
        raise ValueError(f"unknown_training_profile:{name}")
    return dict(PROFILES[name])
