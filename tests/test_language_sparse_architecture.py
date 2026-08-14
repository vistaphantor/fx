from __future__ import annotations

import torch

from src.language.pytorch_transformer import SparseMoE, VistaReasoningGPT
from src.language.training_profiles import PROFILES


def _sparse_model() -> VistaReasoningGPT:
    torch.manual_seed(42)
    return VistaReasoningGPT(
        vocab_size=512,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=128,
        max_seq_len=32,
        dropout=0.0,
        ffn_type="moe",
        num_experts=4,
        experts_per_token=1,
        moe_ffn_dim=64,
        shared_expert_ffn_dim=32,
        router_aux_loss_coef=0.01,
        router_jitter=0.0,
    )


def test_sparse_model_has_less_active_than_total_parameters():
    model = _sparse_model()
    assert model.get_num_params() > model.get_active_params_per_token() > 0


def test_router_and_experts_receive_gradients():
    model = _sparse_model()
    model.train()
    x = torch.randint(1, 500, (2, 16), dtype=torch.long)
    y = torch.randint(1, 500, (2, 16), dtype=torch.long)
    _, loss = model(x, targets=y, pad_id=0)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    router_grads = []
    expert_grads = []
    for block in model.blocks:
        assert isinstance(block.ffn, SparseMoE)
        router_grads.append(block.ffn.router.weight.grad)
        expert_grads.extend(expert.down_proj.weight.grad for expert in block.ffn.experts)
    assert all(grad is not None and torch.isfinite(grad).all() for grad in router_grads)
    assert any(grad is not None and torch.isfinite(grad).all() for grad in expert_grads)


def test_gqa_cache_is_compact_and_preallocated():
    model = _sparse_model().eval()
    prompt = torch.randint(1, 500, (1, 8), dtype=torch.long)
    _, cache = model._forward_cached(prompt, None)
    assert len(cache) == len(model.blocks)
    layer = cache[0]
    assert layer.k.shape[1] == model.n_kv_heads
    assert layer.k.shape[2] == model.max_seq_len
    assert layer.length == prompt.shape[1]
    storage_ptr = layer.k.data_ptr()
    next_token = torch.randint(1, 500, (1, 1), dtype=torch.long)
    _, next_cache = model._forward_cached(next_token, cache)
    assert next_cache[0].k.data_ptr() == storage_ptr
    assert next_cache[0].length == prompt.shape[1] + 1


def test_cached_and_uncached_greedy_generation_match():
    model = _sparse_model().eval()
    prompt = torch.tensor([[2, 10, 11, 12]], dtype=torch.long)
    cached = model.generate(
        prompt.clone(), max_new_tokens=8, temperature=1.0,
        top_k=1, top_p=1.0, use_kv_cache=True,
    )
    uncached = model.generate(
        prompt.clone(), max_new_tokens=8, temperature=1.0,
        top_k=1, top_p=1.0, use_kv_cache=False,
    )
    assert torch.equal(cached, uncached)


def test_2m_profile_is_sparse_gqa_and_has_valid_geometry():
    cfg = PROFILES["2m"]
    assert cfg["ffn_type"] == "moe"
    assert int(cfg["num_experts"]) > int(cfg["experts_per_token"])
    assert int(cfg["n_heads"]) % int(cfg["n_kv_heads"]) == 0
    model = VistaReasoningGPT(
        vocab_size=int(cfg["vocab_size"]),
        d_model=int(cfg["d_model"]),
        n_layers=int(cfg["n_layers"]),
        n_heads=int(cfg["n_heads"]),
        n_kv_heads=int(cfg["n_kv_heads"]),
        ffn_dim=int(cfg["ffn_dim"]),
        max_seq_len=int(cfg["seq_len"]),
        dropout=float(cfg["dropout"]),
        rope_theta=float(cfg["rope_theta"]),
        ffn_type=str(cfg["ffn_type"]),
        num_experts=int(cfg["num_experts"]),
        experts_per_token=int(cfg["experts_per_token"]),
        moe_ffn_dim=int(cfg["moe_ffn_dim"]),
        shared_expert_ffn_dim=int(cfg["shared_expert_ffn_dim"]),
        router_aux_loss_coef=float(cfg["router_aux_loss_coef"]),
        router_jitter=float(cfg["router_jitter"]),
    )
    assert 1_500_000 <= model.get_num_params() <= 2_500_000
    assert model.get_active_params_per_token() < model.get_num_params() * 0.50
