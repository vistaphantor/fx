"""Authoritative PyTorch decoder-only transformer for Vista language reasoning."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 16,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels = x.shape
        x_norm = self.ln1(x)
        qkv = self.qkv_proj(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, time_steps, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, time_steps, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, time_steps, self.n_heads, self.head_dim).transpose(1, 2)

        attention = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        attention = attention.transpose(1, 2).contiguous().view(batch, time_steps, channels)
        x = x + self.dropout(self.out_proj(attention))
        x = x + self.ffn(self.ln2(x))
        return x


class VistaReasoningGPT(nn.Module):
    """Decoder-only GPT used by every authoritative Vista language bundle."""

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
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if max_seq_len < 2:
            raise ValueError("max_seq_len must be >= 2")
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")

        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.max_seq_len = int(max_seq_len)

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

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

        positions = torch.arange(0, time_steps, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x)
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
        return logits, loss

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
    ) -> torch.Tensor:
        if idx.ndim != 2 or idx.shape[0] != 1:
            raise ValueError("generation currently requires a single batch item")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.eval()
        stops = set(stop_ids or ())
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len :]
            logits, _ = self(idx_cond)
            logits_last = logits[:, -1, :] / temperature

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

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if stops and int(idx_next.item()) in stops:
                break
        return idx
