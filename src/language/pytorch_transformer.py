"""Authoritative PyTorch decoder-only transformer for Vista language reasoning.

The architecture supports a dense FFN mode for smoke/regression models and a
sparse MoE mode for capacity-efficient reasoning profiles. The sparse path keeps
a dense attention/residual spine while routing each token through only a small
subset of FFN experts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KVCacheState:
    """Preallocated compact GQA key/value cache for one transformer layer."""

    k: torch.Tensor
    v: torch.Tensor
    length: int


ModelKVCache = tuple[KVCacheState, ...]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10_000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even")
        inv_freq = 1.0 / (
            float(theta)
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin(
        self,
        positions: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.outer(positions.float(), self.inv_freq)
        return freqs.cos().to(dtype=dtype), freqs.sin().to(dtype=dtype)

    @staticmethod
    def apply(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # x: [B, H, T, D], cos/sin: [T, D/2]
        even = x[..., 0::2]
        odd = x[..., 1::2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        out = torch.empty_like(x)
        out[..., 0::2] = even * cos - odd * sin
        out[..., 1::2] = even * sin + odd * cos
        return out


class SwiGLUExpert(nn.Module):
    """Bias-free SwiGLU FFN expert."""

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoE(nn.Module):
    """Token-routed SwiGLU experts with optional always-on shared expert.

    Top-1 is the CPU profile default. Top-k > 1 is supported for later GPU
    profiles without changing the model family or bundle format.
    """

    def __init__(
        self,
        d_model: int,
        *,
        num_experts: int,
        experts_per_token: int,
        expert_hidden_dim: int,
        shared_expert_hidden_dim: int = 0,
        router_jitter: float = 0.01,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        if not 1 <= experts_per_token <= num_experts:
            raise ValueError("experts_per_token must be in [1, num_experts]")
        if expert_hidden_dim <= 0:
            raise ValueError("expert_hidden_dim must be positive")
        if shared_expert_hidden_dim < 0:
            raise ValueError("shared_expert_hidden_dim must be >= 0")

        self.num_experts = int(num_experts)
        self.experts_per_token = int(experts_per_token)
        self.router_jitter = float(router_jitter)
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUExpert(d_model, expert_hidden_dim) for _ in range(num_experts)]
        )
        self.shared_expert = (
            SwiGLUExpert(d_model, shared_expert_hidden_dim)
            if shared_expert_hidden_dim > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        router_logits = self.router(flat)
        if self.training and self.router_jitter > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_jitter

        probs = F.softmax(router_logits, dim=-1)
        top_probs, top_indices = torch.topk(
            probs,
            k=self.experts_per_token,
            dim=-1,
        )
        if self.experts_per_token > 1:
            top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        routed = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            token_positions, slots = torch.where(top_indices == expert_index)
            if token_positions.numel() == 0:
                continue
            expert_out = expert(flat[token_positions])
            weights = top_probs[token_positions, slots].unsqueeze(-1)
            routed.index_add_(0, token_positions, expert_out * weights)

        if self.shared_expert is not None:
            routed = routed + self.shared_expert(flat)

        # Switch-style differentiable load-balancing signal. This is only added
        # to the training loss by the parent model; eval loss remains pure CE.
        with torch.no_grad():
            assignment = F.one_hot(
                top_indices[:, 0], num_classes=self.num_experts
            ).float().mean(dim=0)
        mean_probability = probs.mean(dim=0)
        load_balance = self.num_experts * torch.sum(mean_probability * assignment)
        router_z = torch.mean(torch.logsumexp(router_logits, dim=-1).pow(2))
        aux = load_balance + 0.01 * router_z
        return routed.reshape(shape), aux

    def expert_parameter_count(self) -> int:
        if not self.experts:
            return 0
        return sum(parameter.numel() for parameter in self.experts[0].parameters())


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        *,
        max_seq_len: int,
        dropout: float,
        rope_theta: float,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = d_model // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("attention head_dim must be even for RoPE")
        self.max_seq_len = int(max_seq_len)
        self.kv_repeat = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, theta=rope_theta)

    def _project(
        self,
        x: torch.Tensor,
        *,
        position_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, time_steps, _ = x.shape
        q = self.q_proj(x).view(
            batch, time_steps, self.n_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch, time_steps, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch, time_steps, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)
        positions = torch.arange(
            position_start,
            position_start + time_steps,
            device=x.device,
            dtype=torch.long,
        )
        cos, sin = self.rope.cos_sin(positions, dtype=q.dtype)
        q = self.rope.apply(q, cos, sin)
        k = self.rope.apply(k, cos, sin)
        return q, k, v

    def _expand_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.kv_repeat == 1:
            return tensor
        return tensor.repeat_interleave(self.kv_repeat, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, _ = x.shape
        q, k, v = self._project(x, position_start=0)
        attention = F.scaled_dot_product_attention(
            q,
            self._expand_kv(k),
            self._expand_kv(v),
            is_causal=True,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        attention = attention.transpose(1, 2).contiguous().view(
            batch, time_steps, self.d_model
        )
        return self.dropout(self.out_proj(attention))

    def forward_cached(
        self,
        x: torch.Tensor,
        cache: KVCacheState | None,
    ) -> tuple[torch.Tensor, KVCacheState]:
        batch, time_steps, _ = x.shape
        if batch != 1:
            raise ValueError("cached_attention_requires_single_batch_item")
        past_length = 0 if cache is None else cache.length
        if past_length + time_steps > self.max_seq_len:
            raise ValueError("kv_cache_exceeds_model_context")

        q, k_new, v_new = self._project(x, position_start=past_length)
        if cache is None:
            k_store = torch.empty(
                (batch, self.n_kv_heads, self.max_seq_len, self.head_dim),
                dtype=k_new.dtype,
                device=k_new.device,
            )
            v_store = torch.empty_like(k_store)
            cache = KVCacheState(k=k_store, v=v_store, length=0)
        elif time_steps != 1:
            raise ValueError("cached_continuation_requires_single_new_token")

        start = cache.length
        end = start + time_steps
        cache.k[:, :, start:end, :].copy_(k_new)
        cache.v[:, :, start:end, :].copy_(v_new)
        cache.length = end

        k_all = cache.k[:, :, :end, :]
        v_all = cache.v[:, :, :end, :]
        is_causal = past_length == 0 and time_steps > 1
        attention = F.scaled_dot_product_attention(
            q,
            self._expand_kv(k_all),
            self._expand_kv(v_all),
            is_causal=is_causal,
            dropout_p=0.0,
        )
        attention = attention.transpose(1, 2).contiguous().view(
            batch, time_steps, self.d_model
        )
        return self.out_proj(attention), cache


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_dim: int,
        *,
        max_seq_len: int,
        dropout: float,
        rope_theta: float,
        ffn_type: str,
        num_experts: int,
        experts_per_token: int,
        moe_ffn_dim: int,
        shared_expert_ffn_dim: int,
        router_jitter: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(
            d_model,
            n_heads,
            n_kv_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            rope_theta=rope_theta,
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn_type = str(ffn_type)
        if self.ffn_type == "dense":
            self.ffn: nn.Module = SwiGLUExpert(d_model, ffn_dim)
        elif self.ffn_type == "moe":
            self.ffn = SparseMoE(
                d_model,
                num_experts=num_experts,
                experts_per_token=experts_per_token,
                expert_hidden_dim=moe_ffn_dim,
                shared_expert_hidden_dim=shared_expert_ffn_dim,
                router_jitter=router_jitter,
            )
        else:
            raise ValueError(f"unsupported_ffn_type:{ffn_type}")
        self.dropout = nn.Dropout(dropout)

    def _ffn_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.ffn_norm(x)
        if self.ffn_type == "moe":
            out, aux = self.ffn(normalized)  # type: ignore[misc]
            return out, aux
        out = self.ffn(normalized)
        return out, x.new_zeros(())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.attn_norm(x))
        ffn_out, aux = self._ffn_forward(x)
        x = x + self.dropout(ffn_out)
        return x, aux

    def forward_cached(
        self,
        x: torch.Tensor,
        cache: KVCacheState | None,
    ) -> tuple[torch.Tensor, KVCacheState]:
        attention, next_cache = self.attn.forward_cached(self.attn_norm(x), cache)
        x = x + attention
        ffn_out, _ = self._ffn_forward(x)
        x = x + ffn_out
        return x, next_cache


class VistaReasoningGPT(nn.Module):
    """Vista decoder-only reasoning model with optional sparse MoE FFNs."""

    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 512,
        n_layers: int = 16,
        n_heads: int = 16,
        n_kv_heads: int | None = None,
        ffn_dim: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        rope_theta: float = 10_000.0,
        ffn_type: str = "dense",
        num_experts: int = 1,
        experts_per_token: int = 1,
        moe_ffn_dim: int | None = None,
        shared_expert_ffn_dim: int = 0,
        router_aux_loss_coef: float = 0.01,
        router_jitter: float = 0.01,
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if max_seq_len < 2:
            raise ValueError("max_seq_len must be >= 2")
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if router_aux_loss_coef < 0:
            raise ValueError("router_aux_loss_coef must be >= 0")

        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.max_seq_len = int(max_seq_len)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads or n_heads)
        self.ffn_type = str(ffn_type)
        self.num_experts = int(num_experts)
        self.experts_per_token = int(experts_per_token)
        self.router_aux_loss_coef = float(router_aux_loss_coef)

        resolved_moe_dim = int(moe_ffn_dim or ffn_dim)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_kv_heads=self.n_kv_heads,
                    ffn_dim=ffn_dim,
                    max_seq_len=max_seq_len,
                    dropout=dropout,
                    rope_theta=rope_theta,
                    ffn_type=self.ffn_type,
                    num_experts=self.num_experts,
                    experts_per_token=self.experts_per_token,
                    moe_ffn_dim=resolved_moe_dim,
                    shared_expert_ffn_dim=shared_expert_ffn_dim,
                    router_jitter=router_jitter,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        self._initialize_routers()
        self._scale_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def _initialize_routers(self) -> None:
        # Near-uniform but non-identical routing at step zero. Exact zeros would
        # deterministically collapse top-1 routing into expert 0 on ties.
        std = 0.01 / math.sqrt(max(1, self.d_model))
        for block in self.blocks:
            if isinstance(block.ffn, SparseMoE):
                block.ffn.router.weight.data.normal_(mean=0.0, std=std)

    def _scale_residual_projections(self) -> None:
        residual_std = 0.02 / math.sqrt(2.0 * self.n_layers)
        for block in self.blocks:
            block.attn.out_proj.weight.data.normal_(mean=0.0, std=residual_std)
            if isinstance(block.ffn, SwiGLUExpert):
                block.ffn.down_proj.weight.data.normal_(mean=0.0, std=residual_std)
            elif isinstance(block.ffn, SparseMoE):
                for expert in block.ffn.experts:
                    expert.down_proj.weight.data.normal_(mean=0.0, std=residual_std)
                if block.ffn.shared_expert is not None:
                    block.ffn.shared_expert.down_proj.weight.data.normal_(
                        mean=0.0, std=residual_std
                    )

    def get_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def get_active_params_per_token(self) -> int:
        """Parameter-equivalent path touched by one token.

        Router logits and shared experts are always active. For MoE blocks only
        selected routed experts are counted; inactive experts remain capacity
        without per-token FFN compute.
        """
        active = self.get_num_params()
        for block in self.blocks:
            if isinstance(block.ffn, SparseMoE):
                per_expert = block.ffn.expert_parameter_count()
                inactive = block.ffn.num_experts - block.ffn.experts_per_token
                active -= inactive * per_expert
        return active

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        pad_id: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, time_steps = idx.shape
        if time_steps > self.max_seq_len:
            raise ValueError(
                f"sequence_length_exceeds_model_context:{time_steps}>{self.max_seq_len}"
            )

        x = self.drop(self.tok_emb(idx))
        aux_losses: list[torch.Tensor] = []
        for block in self.blocks:
            x, aux = block(x)
            aux_losses.append(aux)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError(
                    "targets must have the same shape as idx; "
                    f"got idx={tuple(idx.shape)} targets={tuple(targets.shape)}"
                )
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=pad_id,
            )
            if self.training and self.ffn_type == "moe" and aux_losses:
                router_aux = torch.stack(aux_losses).mean()
                loss = loss + self.router_aux_loss_coef * router_aux
        return logits, loss

    def _forward_cached(
        self,
        idx: torch.Tensor,
        cache: ModelKVCache | None,
    ) -> tuple[torch.Tensor, ModelKVCache]:
        if idx.ndim != 2 or idx.shape[0] != 1:
            raise ValueError("cached_generation_requires_single_batch_item")
        if cache is not None and len(cache) != len(self.blocks):
            raise ValueError("kv_cache_layer_count_mismatch")
        past_length = 0 if cache is None else cache[0].length
        if past_length + idx.shape[1] > self.max_seq_len:
            raise ValueError("kv_cache_exceeds_model_context")

        x = self.tok_emb(idx)
        next_cache: list[KVCacheState] = []
        for layer_index, block in enumerate(self.blocks):
            layer_cache = None if cache is None else cache[layer_index]
            x, layer_cache = block.forward_cached(x, layer_cache)
            next_cache.append(layer_cache)
        logits = self.lm_head(self.ln_f(x))
        return logits, tuple(next_cache)

    @staticmethod
    def _sample_next_token(
        logits_last: torch.Tensor,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        logits_last = logits_last / temperature
        if top_k > 0:
            threshold, _ = torch.topk(
                logits_last,
                min(top_k, logits_last.size(-1)),
            )
            logits_last = logits_last.masked_fill(
                logits_last < threshold[:, [-1]],
                -float("inf"),
            )

        probs = F.softmax(logits_last, dim=-1)
        if top_p < 1.0:
            if top_p <= 0.0:
                raise ValueError("top_p must be > 0")
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove_sorted = cumulative > top_p
            remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
            remove_sorted[..., 0] = False
            remove_mask = torch.zeros_like(remove_sorted, dtype=torch.bool)
            remove_mask.scatter_(1, sorted_indices, remove_sorted)
            probs = probs.masked_fill(remove_mask, 0.0)
            normalizer = probs.sum(dim=-1, keepdim=True)
            if torch.any(normalizer <= 0):
                raise RuntimeError("generation_probability_mass_collapsed")
            probs = probs / normalizer
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        *,
        max_new_tokens: int = 300,
        temperature: float = 0.75,
        top_k: int = 50,
        top_p: float = 0.92,
        stop_ids: set[int] | frozenset[int] | None = None,
        use_kv_cache: bool = True,
    ) -> torch.Tensor:
        if idx.ndim != 2 or idx.shape[0] != 1:
            raise ValueError("generation currently requires a single batch item")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.eval()
        stops = set(stop_ids or ())
        if max_new_tokens == 0:
            return idx

        if not use_kv_cache:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.max_seq_len :]
                logits, _ = self(idx_cond)
                idx_next = self._sample_next_token(
                    logits[:, -1, :],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                idx = torch.cat((idx, idx_next), dim=1)
                if stops and int(idx_next.item()) in stops:
                    break
            return idx

        context = idx[:, -self.max_seq_len :]
        logits, cache = self._forward_cached(context, None)
        for _ in range(max_new_tokens):
            idx_next = self._sample_next_token(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            idx = torch.cat((idx, idx_next), dim=1)
            if stops and int(idx_next.item()) in stops:
                break

            cache_length = cache[0].length
            if cache_length >= self.max_seq_len:
                context = idx[:, -self.max_seq_len :]
                logits, cache = self._forward_cached(context, None)
            else:
                logits, cache = self._forward_cached(idx_next, cache)
        return idx
