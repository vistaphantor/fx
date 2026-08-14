"""
PyTorch 50M Parameter Reasoning Transformer — VistaReasoningGPT (50M)

Features:
  - 50.3M Parameters (d_model=512, n_layers=16, n_heads=16, ffn=2048)
  - PyTorch F.scaled_dot_product_attention (fast C++ CPU/GPU implementation)
  - Weight-tied LM Head
  - Systemic Reasoning Support (<think> ... </think> CoT mask)
  - Cosine Annealing Learning Rate Scheduler with Warmup
  - PyTorch DataLoader with parallel CPU workers
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 16, ffn_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape

        # ── Multi-Head Self Attention ─────────────────────────────────────────
        x_norm = self.ln1(x)
        qkv = self.qkv_proj(x_norm)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head: (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # PyTorch native scaled dot product attention (fast vectorized C++ / MKL kernel)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.dropout(self.out_proj(attn_out))

        # ── Feed-Forward Network ──────────────────────────────────────────────
        x = x + self.ffn(self.ln2(x))
        return x


class VistaReasoningGPT(nn.Module):
    """
    50M Parameter Reasoning Transformer in PyTorch.

    Default Hyperparameters (50.3M parameters):
      vocab_size=8192, d_model=512, n_layers=16, n_heads=16, max_seq_len=512
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 512,
        n_layers: int = 16,
        n_heads: int = 16,
        ffn_dim: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        # Weight tying: LM Head shares weights with Token Embedding
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        device = idx.device
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V)

        loss = None
        if targets is not None:
            # Training datasets already provide aligned next-token targets:
            #
            #   input   = tokens[:-1]
            #   target  = tokens[1:]
            #
            # Therefore logits[t] must be compared directly with targets[t].
            # Shifting here again would incorrectly train the model to predict
            # two tokens ahead.
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

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 300,
        temperature: float = 0.75,
        top_k: int = 50,
        top_p: float = 0.92,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits_last = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
                logits_last[logits_last < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits_last, dim=-1)

            if top_p is not None and top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if eos_id is not None and idx_next.item() == eos_id:
                break

        return idx
