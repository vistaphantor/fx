"""
Pure-NumPy Transformer — GPT-style decoder-only language model.

Architecture:
  Token Embedding  (vocab_size × d_model)
  Positional Embedding (max_seq_len × d_model)
  N × Transformer Block:
    Layer Norm 1
    Causal Self-Attention (n_heads, d_head = d_model // n_heads)
    Residual
    Layer Norm 2
    Feed-Forward  (d_model → 4*d_model → d_model)  [GELU activation]
    Residual
  Final Layer Norm
  LM Head  (d_model → vocab_size)   [tied with token embedding]

Default "nano" config (~8M params):
  d_model=256, n_heads=8, n_layers=6, max_seq=512, vocab=4096
"""
from __future__ import annotations

import math
import numpy as np
from pathlib import Path


# ── Activation ────────────────────────────────────────────────────────────────

def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit — smooth alternative to ReLU."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of GELU."""
    tanh_arg = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
    tanh_val = np.tanh(tanh_arg)
    sech2 = 1.0 - tanh_val ** 2
    dtanh = math.sqrt(2.0 / math.pi) * (1.0 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * dtanh


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5):
    """Layer normalisation. x: (... × d), g/b: (d,)"""
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(var + eps) + b


# ── Weight container ──────────────────────────────────────────────────────────

class Weights:
    """Flat dictionary of named NumPy arrays with Adam moment tracking."""

    def __init__(self):
        self.params: dict[str, np.ndarray] = {}
        self.m:      dict[str, np.ndarray] = {}  # 1st moment
        self.v:      dict[str, np.ndarray] = {}  # 2nd moment
        self.t: int = 0

    def add(self, name: str, arr: np.ndarray) -> None:
        self.params[name] = arr
        self.m[name] = np.zeros_like(arr)
        self.v[name] = np.zeros_like(arr)

    def adam_update(self, grads: dict[str, np.ndarray],
                    lr: float = 3e-4, beta1: float = 0.9,
                    beta2: float = 0.999, eps: float = 1e-8,
                    weight_decay: float = 0.01,
                    grad_clip: float = 1.0) -> None:
        # Global gradient norm clipping
        total_norm = math.sqrt(sum(float(np.sum(g ** 2)) for g in grads.values()))
        if total_norm > grad_clip:
            scale = grad_clip / (total_norm + 1e-8)
            grads = {k: v * scale for k, v in grads.items()}

        self.t += 1
        bc1 = 1.0 - beta1 ** self.t
        bc2 = 1.0 - beta2 ** self.t

        for name, grad in grads.items():
            if name not in self.params:
                continue
            # AdamW: weight decay applied before Adam update
            self.params[name] *= (1.0 - lr * weight_decay)
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * grad
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * grad ** 2
            self.params[name] -= lr * (self.m[name] / bc1) / (np.sqrt(self.v[name] / bc2) + eps)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), **self.params,
                            _t=np.array(self.t, dtype=np.int64))
        print(f"[Weights] Saved {len(self.params)} tensors to {path}")

    def load(self, path: str | Path) -> None:
        data = np.load(str(path) + ".npz" if not str(path).endswith(".npz") else str(path))
        for k in data.files:
            if k == "_t":
                self.t = int(data[k])
            else:
                self.params[k] = data[k]
                self.m[k] = np.zeros_like(data[k])
                self.v[k] = np.zeros_like(data[k])
        print(f"[Weights] Loaded {len(self.params)} tensors from {path} (step={self.t})")

    def numel(self) -> int:
        return sum(p.size for p in self.params.values())


# ── Transformer Model ─────────────────────────────────────────────────────────

class NanoGPT:
    """
    Decoder-only GPT language model in pure NumPy.

    Supports training (forward + backward pass) and generation.
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        max_seq_len: int = 512,
        dropout_p: float = 0.10,
        seed: int = 42,
    ):
        self.vocab_size  = vocab_size
        self.d_model     = d_model
        self.n_heads     = n_heads
        self.d_head      = d_model // n_heads
        self.n_layers    = n_layers
        self.max_seq_len = max_seq_len
        self.dropout_p   = dropout_p
        self.rng = np.random.default_rng(seed)
        self.W = Weights()
        self._init_weights()

    def _init_weights(self) -> None:
        W = self.W
        rng = self.rng
        d = self.d_model
        V = self.vocab_size
        H = self.max_seq_len

        # Token + position embeddings
        W.add("tok_emb", rng.normal(0, 0.02, (V, d)).astype(np.float32))
        W.add("pos_emb", rng.normal(0, 0.01, (H, d)).astype(np.float32))

        for i in range(self.n_layers):
            p = f"l{i}_"
            # Attention layer norm
            W.add(p + "ln1_g", np.ones(d, dtype=np.float32))
            W.add(p + "ln1_b", np.zeros(d, dtype=np.float32))
            # QKV projection (d → 3d)
            W.add(p + "qkv_w", rng.normal(0, 0.02, (d, 3 * d)).astype(np.float32))
            W.add(p + "qkv_b", np.zeros(3 * d, dtype=np.float32))
            # Output projection (d → d)
            W.add(p + "attn_proj_w", rng.normal(0, 0.02 / math.sqrt(2 * self.n_layers), (d, d)).astype(np.float32))
            W.add(p + "attn_proj_b", np.zeros(d, dtype=np.float32))
            # FFN layer norm
            W.add(p + "ln2_g", np.ones(d, dtype=np.float32))
            W.add(p + "ln2_b", np.zeros(d, dtype=np.float32))
            # FFN weights (d → 4d → d)
            W.add(p + "ffn_w1", rng.normal(0, 0.02, (d, 4 * d)).astype(np.float32))
            W.add(p + "ffn_b1", np.zeros(4 * d, dtype=np.float32))
            W.add(p + "ffn_w2", rng.normal(0, 0.02 / math.sqrt(2 * self.n_layers), (4 * d, d)).astype(np.float32))
            W.add(p + "ffn_b2", np.zeros(d, dtype=np.float32))

        # Final layer norm
        W.add("final_ln_g", np.ones(d, dtype=np.float32))
        W.add("final_ln_b", np.zeros(d, dtype=np.float32))

        # LM head — weight tied to tok_emb (no separate storage)
        total = W.numel()
        print(f"[NanoGPT] Initialized  vocab={self.vocab_size}  d_model={self.d_model}  "
              f"n_heads={self.n_heads}  n_layers={self.n_layers}  "
              f"params={total/1e6:.2f}M")

    # ── Forward pass (inference — no dropout, no gradient tracking) ───────────

    def forward(self, ids: np.ndarray) -> np.ndarray:
        """
        ids: (T,) or (B, T)   integer token IDs
        returns logits: (B, T, V)
        """
        if ids.ndim == 1:
            ids = ids[np.newaxis, :]  # (1, T)
        B, T = ids.shape
        W = self.W.params

        # Embeddings
        x = W["tok_emb"][ids] + W["pos_emb"][:T]  # (B, T, d)
        mask = self._causal_mask(T)  # (T, T)

        for i in range(self.n_layers):
            x = self._block_forward(x, i, mask)

        x = layer_norm(x, W["final_ln_g"], W["final_ln_b"])
        logits = x @ W["tok_emb"].T  # weight tying  (B, T, V)
        return logits

    def _block_forward(self, x: np.ndarray, i: int, mask: np.ndarray) -> np.ndarray:
        W = self.W.params
        p = f"l{i}_"
        # Self-attention sublayer
        x_norm = layer_norm(x, W[p + "ln1_g"], W[p + "ln1_b"])
        attn_out = self._attention(x_norm, p, mask)
        x = x + attn_out  # residual
        # FFN sublayer
        x_norm2 = layer_norm(x, W[p + "ln2_g"], W[p + "ln2_b"])
        ffn_out = self._ffn(x_norm2, p)
        x = x + ffn_out  # residual
        return x

    def _attention(self, x: np.ndarray, p: str, mask: np.ndarray) -> np.ndarray:
        """Causal multi-head self-attention."""
        W = self.W.params
        B, T, d = x.shape
        H = self.n_heads
        Hd = self.d_head

        qkv = x @ W[p + "qkv_w"] + W[p + "qkv_b"]  # (B, T, 3d)
        q, k, v = np.split(qkv, 3, axis=-1)          # each (B, T, d)

        # Reshape to multi-head: (B, H, T, Hd)
        def to_heads(t):
            return t.reshape(B, T, H, Hd).transpose(0, 2, 1, 3)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(Hd)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale  # (B, H, T, T)
        scores = scores + mask  # causal mask (additive -inf)
        attn = softmax(scores)  # (B, H, T, T)

        out = attn @ v  # (B, H, T, Hd)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, d)
        return out @ W[p + "attn_proj_w"] + W[p + "attn_proj_b"]

    def _ffn(self, x: np.ndarray, p: str) -> np.ndarray:
        W = self.W.params
        h = gelu(x @ W[p + "ffn_w1"] + W[p + "ffn_b1"])  # (B, T, 4d)
        return h @ W[p + "ffn_w2"] + W[p + "ffn_b2"]      # (B, T, d)

    @staticmethod
    def _causal_mask(T: int) -> np.ndarray:
        """Upper-triangular mask filled with -1e9 (additive)."""
        mask = np.full((T, T), -1e9, dtype=np.float32)
        mask = np.tril(np.zeros((T, T), dtype=np.float32)) + np.triu(mask, k=1)
        return mask  # (T, T) — broadcastable to (B, H, T, T)

    # ── Loss ─────────────────────────────────────────────────────────────────

    def loss(self, ids: np.ndarray, pad_id: int = 0) -> tuple[float, np.ndarray]:
        """
        Cross-entropy language model loss.
        ids: (B, T)
        Returns: (scalar_loss, grads_dict)
        """
        B, T = ids.shape
        logits = self.forward(ids)   # (B, T, V)

        # Shift: predict token t+1 from position t
        logits_shifted = logits[:, :-1, :]       # (B, T-1, V)
        targets = ids[:, 1:]                     # (B, T-1)

        # Cross-entropy with ignore_index for PAD
        V = self.vocab_size
        log_probs = logits_shifted - logits_shifted.max(axis=-1, keepdims=True)
        log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True) + 1e-9)

        mask = (targets != pad_id).astype(np.float32)   # (B, T-1)
        n_tokens = mask.sum()

        # Gather log-prob at correct token
        idx_b = np.arange(B)[:, None]
        idx_t = np.arange(T - 1)[None, :]
        loss_tokens = -log_probs[idx_b, idx_t, targets] * mask
        loss_val = float(loss_tokens.sum() / (n_tokens + 1e-9))

        # ── Backward pass ────────────────────────────────────────────────────
        grads = self._backward(ids, logits, log_probs, targets, mask, n_tokens)

        return loss_val, grads

    def _backward(self, ids, logits, log_probs, targets, mask, n_tokens):
        """
        Reverse-mode auto-diff through the transformer.
        Returns gradient dictionary matching self.W.params keys.
        """
        B, T = ids.shape
        W = self.W.params
        grads: dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in W.items()}

        # ── dL / d logits_shifted ────────────────────────────────────────────
        probs = np.exp(log_probs)                          # (B, T-1, V)
        dlogits = probs.copy()
        idx_b = np.arange(B)[:, None]
        idx_t = np.arange(T - 1)[None, :]
        dlogits[idx_b, idx_t, targets] -= 1.0
        dlogits *= mask[:, :, None]
        dlogits /= (n_tokens + 1e-9)

        # Pad back to (B, T, V) — last position has 0 gradient
        dlogits_full = np.concatenate([dlogits, np.zeros((B, 1, self.vocab_size), dtype=np.float32)], axis=1)

        # ── LM head gradient (weight-tied with tok_emb) ─────────────────────
        # logits = x_final @ tok_emb.T => d_x_final = dlogits @ tok_emb
        # Need forward activations — re-run forward pass storing intermediates
        x, cache = self._forward_with_cache(ids)  # cache has per-layer activations

        # d_tok_emb from LM head
        x_ln = layer_norm(x, W["final_ln_g"], W["final_ln_b"])  # (B, T, d)
        grads["tok_emb"] += (dlogits_full.reshape(-1, self.vocab_size).T @
                              x_ln.reshape(-1, self.d_model)).T  # (V, d)

        # Gradient w.r.t. x_ln
        dx_ln = dlogits_full @ W["tok_emb"]  # (B, T, d)  (tok_emb acts as weight matrix)

        # Backward through final layer norm
        dx, dg, db = self._ln_backward(dx_ln, x, W["final_ln_g"])
        grads["final_ln_g"] += dg
        grads["final_ln_b"] += db

        # ── Backward through each transformer block (reverse order) ──────────
        for i in reversed(range(self.n_layers)):
            dx, block_grads = self._block_backward(dx, i, cache[i])
            for k, v in block_grads.items():
                grads[k] += v

        # ── Token embedding gradient ─────────────────────────────────────────
        np.add.at(grads["tok_emb"], ids, dx)
        grads["pos_emb"] += dx.sum(axis=0)[:T]

        return grads

    def _forward_with_cache(self, ids: np.ndarray):
        """Forward pass that saves intermediate values needed for backprop."""
        B, T = ids.shape
        W = self.W.params
        mask = self._causal_mask(T)

        x = W["tok_emb"][ids] + W["pos_emb"][:T]
        cache = []

        for i in range(self.n_layers):
            c = {"x_in": x.copy()}
            p = f"l{i}_"

            # Attention sub-layer
            x_norm1 = layer_norm(x, W[p + "ln1_g"], W[p + "ln1_b"])
            c["x_norm1"] = x_norm1

            # QKV
            qkv = x_norm1 @ W[p + "qkv_w"] + W[p + "qkv_b"]
            q, k, v = np.split(qkv, 3, axis=-1)
            H, Hd = self.n_heads, self.d_head

            def to_heads(t):
                return t.reshape(B, T, H, Hd).transpose(0, 2, 1, 3)

            q_h, k_h, v_h = to_heads(q), to_heads(k), to_heads(v)
            scale = 1.0 / math.sqrt(Hd)
            scores = (q_h @ k_h.transpose(0, 1, 3, 2)) * scale + mask
            attn_w = softmax(scores)

            attn_out_h = attn_w @ v_h
            attn_out = attn_out_h.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
            proj_out = attn_out @ W[p + "attn_proj_w"] + W[p + "attn_proj_b"]

            c.update({"q_h": q_h, "k_h": k_h, "v_h": v_h,
                       "attn_w": attn_w, "attn_out": attn_out, "proj_out": proj_out})
            x_attn = x + proj_out

            # FFN sub-layer
            x_norm2 = layer_norm(x_attn, W[p + "ln2_g"], W[p + "ln2_b"])
            h_pre = x_norm2 @ W[p + "ffn_w1"] + W[p + "ffn_b1"]
            h_act = gelu(h_pre)
            ffn_out = h_act @ W[p + "ffn_w2"] + W[p + "ffn_b2"]

            c.update({"x_attn": x_attn, "x_norm2": x_norm2,
                       "h_pre": h_pre, "h_act": h_act})
            x = x_attn + ffn_out
            cache.append(c)

        return x, cache

    def _block_backward(self, dx: np.ndarray, i: int, cache: dict):
        W = self.W.params
        p = f"l{i}_"
        grads = {}
        B, T, d = dx.shape
        H, Hd = self.n_heads, self.d_head

        # ── FFN backward ─────────────────────────────────────────────────────
        # x = x_attn + ffn_out  →  d_ffn = dx, d_x_attn accumulated
        d_ffn = dx  # gradient flows through residual
        d_x_attn = dx.copy()

        # ffn_out = h_act @ ffn_w2 + ffn_b2
        grads[p + "ffn_w2"] = cache["h_act"].reshape(-1, 4 * d).T @ d_ffn.reshape(-1, d)
        grads[p + "ffn_b2"] = d_ffn.sum(axis=(0, 1))
        d_h_act = d_ffn @ W[p + "ffn_w2"].T

        # h_act = gelu(h_pre)
        d_h_pre = d_h_act * gelu_grad(cache["h_pre"])
        grads[p + "ffn_w1"] = cache["x_norm2"].reshape(-1, d).T @ d_h_pre.reshape(-1, 4 * d)
        grads[p + "ffn_b1"] = d_h_pre.sum(axis=(0, 1))
        d_x_norm2 = d_h_pre @ W[p + "ffn_w1"].T

        # Layer norm 2 backward
        d_x_attn2, dg2, db2 = self._ln_backward(d_x_norm2, cache["x_attn"], W[p + "ln2_g"])
        grads[p + "ln2_g"] = dg2
        grads[p + "ln2_b"] = db2
        d_x_attn = d_x_attn + d_x_attn2  # accumulate

        # ── Attention backward ────────────────────────────────────────────────
        d_proj = d_x_attn
        d_x_in = d_x_attn.copy()

        # proj_out = attn_out @ attn_proj_w + attn_proj_b
        grads[p + "attn_proj_w"] = cache["attn_out"].reshape(-1, d).T @ d_proj.reshape(-1, d)
        grads[p + "attn_proj_b"] = d_proj.sum(axis=(0, 1))
        d_attn_out = d_proj @ W[p + "attn_proj_w"].T  # (B, T, d)

        # Reshape to heads
        d_attn_out_h = d_attn_out.reshape(B, T, H, Hd).transpose(0, 2, 1, 3)  # (B, H, T, Hd)

        # attn_out = attn_w @ v_h
        d_attn_w = d_attn_out_h @ cache["v_h"].transpose(0, 1, 3, 2)   # (B, H, T, T)
        d_v_h    = cache["attn_w"].transpose(0, 1, 3, 2) @ d_attn_out_h # (B, H, T, Hd)

        # Softmax backward (d_attn_w through softmax)
        aw = cache["attn_w"]
        d_scores = aw * (d_attn_w - (d_attn_w * aw).sum(axis=-1, keepdims=True))
        scale = 1.0 / math.sqrt(Hd)
        d_scores *= scale

        # scores = q_h @ k_h.T
        d_q_h = d_scores @ cache["k_h"]              # (B, H, T, Hd)
        d_k_h = d_scores.transpose(0, 1, 3, 2) @ cache["q_h"]  # (B, H, T, Hd)

        # Back to (B, T, d)
        def from_heads(t):
            return t.transpose(0, 2, 1, 3).reshape(B, T, d)

        d_q = from_heads(d_q_h)
        d_k = from_heads(d_k_h)
        d_v = from_heads(d_v_h)
        d_qkv = np.concatenate([d_q, d_k, d_v], axis=-1)  # (B, T, 3d)

        # qkv = x_norm1 @ qkv_w + qkv_b
        grads[p + "qkv_w"] = cache["x_norm1"].reshape(-1, d).T @ d_qkv.reshape(-1, 3 * d)
        grads[p + "qkv_b"] = d_qkv.sum(axis=(0, 1))
        d_x_norm1 = d_qkv @ W[p + "qkv_w"].T

        # Layer norm 1 backward
        d_x_in2, dg1, db1 = self._ln_backward(d_x_norm1, cache["x_in"], W[p + "ln1_g"])
        grads[p + "ln1_g"] = dg1
        grads[p + "ln1_b"] = db1
        d_x_in = d_x_in + d_x_in2  # accumulate from residual

        return d_x_in, grads

    @staticmethod
    def _ln_backward(dy: np.ndarray, x: np.ndarray, g: np.ndarray, eps: float = 1e-5):
        """Backward pass for layer normalization."""
        mean = x.mean(axis=-1, keepdims=True)
        var  = x.var(axis=-1, keepdims=True)
        std  = np.sqrt(var + eps)
        x_hat = (x - mean) / std

        N = x.shape[-1]
        dg = (dy * x_hat).sum(axis=tuple(range(dy.ndim - 1)))
        db = dy.sum(axis=tuple(range(dy.ndim - 1)))

        dx_hat = dy * g
        dx = (1.0 / (N * std)) * (
            N * dx_hat
            - dx_hat.sum(axis=-1, keepdims=True)
            - x_hat * (dx_hat * x_hat).sum(axis=-1, keepdims=True)
        )
        return dx, dg, db

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        eos_id: int | None = None,
    ) -> list[int]:
        """
        Autoregressive token generation with top-k / top-p / temperature sampling.
        """
        ids = list(prompt_ids)

        for _ in range(max_new_tokens):
            ctx = ids[-self.max_seq_len:]
            logits = self.forward(np.array(ctx, dtype=np.int32))  # (1, T, V)
            logits_last = logits[0, -1, :].astype(np.float64)     # (V,)

            # Temperature
            logits_last /= max(temperature, 1e-6)

            # Top-k filtering
            if top_k > 0:
                kth = np.partition(logits_last, -top_k)[-top_k]
                logits_last[logits_last < kth] = -1e9

            # Top-p (nucleus) filtering
            probs = softmax(logits_last)
            sorted_idx = np.argsort(-probs)
            cumprobs = np.cumsum(probs[sorted_idx])
            cutoff = cumprobs > top_p
            if cutoff.any():
                cutoff_idx = np.argmax(cutoff)
                logits_last[sorted_idx[cutoff_idx + 1:]] = -1e9
                probs = softmax(logits_last)

            # Sample
            next_id = int(self.rng.choice(len(probs), p=probs / probs.sum()))
            ids.append(next_id)

            if eos_id is not None and next_id == eos_id:
                break

        return ids[len(prompt_ids):]  # return only generated tokens

    def save(self, path: str | Path) -> None:
        self.W.save(path)

    def load(self, path: str | Path) -> None:
        self.W.load(path)
