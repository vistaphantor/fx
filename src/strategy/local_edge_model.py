from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


FEATURE_NAMES = [
    # ── Core Z-scored Market Signals (Normalized) ────────────────────────────
    "momentum_z",
    "trend_z",
    "volume_z",
    "order_block_z",
    "volatility_risk_z",
    "entry_distance_z",
    "spread_danger_z",
    "orderflow_z",
    # ── Multi-Timeframe Structural Signals ────────────────────────────────────
    "m5_score_z",
    "m15_score_z",
    "context_score_z",
    "structure_score_z",
    "statistical_score_z",
    # ── Portfolio & Economic Edge Context ─────────────────────────────────────
    "omega_t",
    "kelly_fraction",
    "session_score",
    "spread",
    "atr",
]


# ─────────────────────────────────────────────────────────────────────────────
# Deep forward pass (shared by inference methods)
# ─────────────────────────────────────────────────────────────────────────────

def _deep_forward(weights: dict, x: np.ndarray) -> np.ndarray:
    """
    Forward pass through a deep residual network.

    Supports three weight formats:
      * Deep residual: weights["num_layers"] + weights["layer_i_w/b"]
      * 2-layer MLP:   w1/b1/w2/b2/w3/b3
      * Legacy 1-layer: w1/b1/w2/b2
    """
    # ── Deep residual format ──────────────────────────────────────────────────
    if "num_layers" in weights:
        num_layers = int(np.asarray(weights["num_layers"]).item())

        # Layer 0: input projection
        h = _leaky_relu(x @ weights["layer_0_w"] + weights["layer_0_b"])

        # Residual blocks: pairs of hidden layers (1,2), (3,4), …, (n-3, n-2)
        i = 1
        while i < num_layers - 1:
            h_skip = h
            # Layer normalization helper
            h_mean = h.mean(axis=-1, keepdims=True)
            h_std = np.sqrt(h.var(axis=-1, keepdims=True) + 1e-5)
            h_norm = (h - h_mean) / h_std

            # First layer of block
            h = _leaky_relu(h_norm @ weights[f"layer_{i}_w"] + weights[f"layer_{i}_b"])
            i += 1
            # Second layer of block (if available)
            if i < num_layers - 1:
                h_mean2 = h.mean(axis=-1, keepdims=True)
                h_std2 = np.sqrt(h.var(axis=-1, keepdims=True) + 1e-5)
                h_norm2 = (h - h_mean2) / h_std2
                h = _leaky_relu(h_norm2 @ weights[f"layer_{i}_w"] + weights[f"layer_{i}_b"])
                h = 0.5 * h + h_skip  # scaled residual skip connection
                i += 1

        # Output layer
        h_mean_out = h.mean(axis=-1, keepdims=True)
        h_std_out = np.sqrt(h.var(axis=-1, keepdims=True) + 1e-5)
        h_norm_out = (h - h_mean_out) / h_std_out
        logit = h_norm_out @ weights[f"layer_{num_layers - 1}_w"] + weights[f"layer_{num_layers - 1}_b"]
        return logit

    # ── 2-layer MLP legacy format ─────────────────────────────────────────────
    if "w3" in weights:
        h1 = _leaky_relu(x @ weights["w1"] + weights["b1"])
        h2 = _leaky_relu(h1 @ weights["w2"] + weights["b2"])
        return h2 @ weights["w3"] + weights["b3"]

    # ── 1-layer legacy format ─────────────────────────────────────────────────
    h = _leaky_relu(x @ weights["w1"] + weights["b1"])
    return h @ weights["w2"] + weights["b2"]


# ─────────────────────────────────────────────────────────────────────────────
# Model class
# ─────────────────────────────────────────────────────────────────────────────

class LocalEdgeModel:
    def __init__(
        self,
        weights: dict[str, np.ndarray],
        feature_mean,
        feature_std,
        threshold: float = 0.55,
    ):
        self.weights = weights
        self.feature_mean = np.asarray(feature_mean, dtype=np.float64)
        self.feature_std = np.asarray(feature_std, dtype=np.float64)
        self.threshold = float(threshold)

    @classmethod
    def load(cls, path: str | Path) -> "LocalEdgeModel":
        data = np.load(Path(path), allow_pickle=False)
        weights: dict[str, np.ndarray] = {}

        # Deep residual format
        if "num_layers" in data:
            num_layers = int(np.asarray(data["num_layers"]).item())
            weights["num_layers"] = data["num_layers"]
            weights["hidden_size"] = data["hidden_size"]
            for i in range(num_layers):
                weights[f"layer_{i}_w"] = data[f"layer_{i}_w"]
                weights[f"layer_{i}_b"] = data[f"layer_{i}_b"]
        else:
            # Legacy 1-layer or 2-layer format
            weights = {"w1": data["w1"], "b1": data["b1"], "w2": data["w2"], "b2": data["b2"]}
            if "w3" in data:
                weights["w3"] = data["w3"]
                weights["b3"] = data["b3"]

        threshold = float(data["threshold"]) if "threshold" in data else 0.55
        return cls(weights, data["feature_mean"], data["feature_std"], threshold=threshold)

    def predict_probability(self, features: dict) -> float:
        x = vectorize_features(features)
        expected_size = int(self.feature_mean.shape[0])
        if x.shape[0] > expected_size:
            x = x[:expected_size]
        elif x.shape[0] < expected_size:
            x = np.pad(x, (0, expected_size - x.shape[0]))
        x = (x - self.feature_mean) / self.feature_std
        logit = _deep_forward(self.weights, x)
        prob = _sigmoid(logit)
        if isinstance(prob, np.ndarray):
            return float(prob.item() if prob.size == 1 else prob.ravel()[0])
        return float(prob)

    def explain(self, features: dict, drawdown_ratio: float = 0.0) -> dict:
        """Generate a full chain-of-thought reasoning explanation for a prediction.

        Returns a structured dict with probability, decision, confidence band,
        feature-level insights, regime detection, and a natural language reasoning
        paragraph — suitable as a training target for a language model.
        """
        x_raw = vectorize_features(features)
        expected_size = int(self.feature_mean.shape[0])
        if x_raw.shape[0] > expected_size:
            x_raw = x_raw[:expected_size]
        elif x_raw.shape[0] < expected_size:
            x_raw = np.pad(x_raw, (0, expected_size - x_raw.shape[0]))
        x_norm = (x_raw - self.feature_mean) / self.feature_std

        # Probability
        logit = _deep_forward(self.weights, x_norm)
        prob = float(_sigmoid(logit).ravel()[0] if isinstance(_sigmoid(logit), np.ndarray) else _sigmoid(logit))
        dynamic_threshold = min(0.75, max(0.45, self.threshold + 0.10 * float(drawdown_ratio)))
        decision = "TRADE" if prob >= dynamic_threshold else "SKIP"

        # Confidence band
        margin = abs(prob - dynamic_threshold)
        if margin > 0.15:
            confidence = "high"
        elif margin > 0.07:
            confidence = "medium"
        else:
            confidence = "low"

        # Top feature signals (z-scored contribution)
        feat_names = FEATURE_NAMES[:len(x_norm)]
        feat_z = x_norm  # already normalised
        top_idx = np.argsort(np.abs(feat_z))[::-1][:5]
        top_features = [
            {"name": feat_names[i], "z_score": round(float(feat_z[i]), 3)}
            for i in top_idx if i < len(feat_names)
        ]

        # Regime detection from raw feature values
        # Regime detection from z-scored feature values (raw versions pruned)
        vol_z   = float(x_raw[feat_names.index("volatility_risk_z")] if "volatility_risk_z" in feat_names else 0.0)
        mom_z   = float(x_raw[feat_names.index("momentum_z")] if "momentum_z" in feat_names else 0.0)
        trend_z = float(x_raw[feat_names.index("trend_z")] if "trend_z" in feat_names else 0.0)

        if vol_z > 1.5:
            regime = "high-volatility"
        elif abs(trend_z) > 1.0:
            regime = "trending" if trend_z > 0 else "downtrending"
        elif abs(mom_z) > 0.8:
            regime = "momentum"
        else:
            regime = "ranging"

        # Construct natural language reasoning paragraph
        top_feat_desc = ", ".join(
            f"{f['name'].replace('_', ' ')} (z={f['z_score']:+.2f})"
            for f in top_features
        )
        drawdown_note = (
            f" The model is under drawdown ({drawdown_ratio:.0%}), raising threshold to {dynamic_threshold:.2f}."
            if drawdown_ratio > 0.05 else ""
        )
        action_verb = "initiating" if decision == "TRADE" else "skipping"
        confidence_phrase = {
            "high":   "with high conviction",
            "medium": "with moderate conviction",
            "low":    "but conviction is marginal",
        }[confidence]

        reasoning = (
            f"In a {regime} market regime, the deep residual model assigned a probability of "
            f"{prob:.3f} against a dynamic threshold of {dynamic_threshold:.2f}, "
            f"{action_verb} this setup {confidence_phrase}. "
            f"The strongest input signals driving this decision were: {top_feat_desc}. "
            f"The network processed these through {self.weights.get('num_layers', '?')} residual layers, "
            f"compressing the {len(FEATURE_NAMES)}-feature space into a binary edge signal."
            f"{drawdown_note}"
        )

        return {
            "probability": round(prob, 5),
            "threshold": round(dynamic_threshold, 3),
            "decision": decision,
            "confidence": confidence,
            "regime": regime,
            "top_features": top_features,
            "reasoning": reasoning,
            "num_layers": int(np.asarray(self.weights.get("num_layers", 2)).item()),
        }

    def allows_trade(self, features: dict, drawdown_ratio: float = 0.0) -> tuple[bool, float]:
        probability = self.predict_probability(features)
        # Dynamically scale the threshold based on drawdown to be more selective when losing
        dynamic_threshold = self.threshold + (0.10 * float(drawdown_ratio))
        dynamic_threshold = min(0.75, max(0.45, dynamic_threshold))
        return probability >= dynamic_threshold, probability


# ─────────────────────────────────────────────────────────────────────────────
# Feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def vectorize_features(row: dict) -> np.ndarray:
    values = []
    for name in FEATURE_NAMES:
        if isinstance(row, dict):
            value = row.get(name, 0.0)
        else:
            value = getattr(row, name, 0.0)
        values.append(float(value or 0.0))
    return np.asarray(values, dtype=np.float64)


def load_feature_rows(path: str | Path) -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    rows: list[dict] = []
    with file_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def build_dataset(rows: Iterable[dict], min_abs_return: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    x_rows = []
    labels = []
    for row in rows:
        expected_return = float(row.get("expected_return", 0.0) or 0.0)
        if abs(expected_return) < min_abs_return:
            continue
        x_rows.append(vectorize_features(row))
        labels.append(1.0 if expected_return > 0 else 0.0)
    if not x_rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64), np.empty((0,), dtype=np.float64)
    return np.vstack(x_rows).astype(np.float64), np.asarray(labels, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Deep Residual Network Training
# ─────────────────────────────────────────────────────────────────────────────

def train_local_edge_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hidden_size: int = 48,
    num_layers: int = 32,
    epochs: int = 400,
    learning_rate: float = 0.003,
    threshold: float = 0.55,
    seed: int = 42,
    l1_lambda: float = 0.0001,
    l2_lambda: float = 0.001,
    batch_size: int = 128,
    grad_clip: float = 1.0,
) -> tuple[LocalEdgeModel, dict]:
    """
    Train a deep residual MLP.

    Architecture (num_layers=32, hidden_size=48):
      Layer 0  : Input projection   74  → 48
      Layers 1-30 : 15 residual blocks, each: 48→48→48 with skip
      Layer 31 : Output             48  → 1
    """
    if len(x) < 10:
        raise ValueError(f"Need at least 10 samples to train local edge model, got {len(x)}")
    if len(set(y.tolist())) < 2:
        raise ValueError("Need both positive and negative labels to train local edge model")

    # Ensure num_layers is at least 2 (projection + output)
    num_layers = max(2, int(num_layers))
    # Ensure hidden layers form complete residual blocks (even number of hidden layers)
    num_hidden = num_layers - 2          # hidden layers (excl. projection & output)
    if num_hidden % 2 != 0:
        num_hidden += 1                  # round up to even
    num_layers = num_hidden + 2          # recompute total

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    x = x[order]
    y = y[order]
    split = min(max(1, int(len(x) * 0.8)), len(x) - 1)
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    x_train_norm = (x_train - mean) / std
    x_val_norm = (x_val - mean) / std

    fan_in = x_train.shape[1]

    # ── Weight initialisation (He / Kaiming Normal) ───────────────────────────
    ws: list[np.ndarray] = []   # weight matrices
    bs: list[np.ndarray] = []   # bias vectors

    # Layer 0: input projection (fan_in → hidden_size)
    ws.append(rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, hidden_size)))
    bs.append(np.zeros(hidden_size))

    # Layers 1 … num_layers-2: hidden residual layers (hidden_size → hidden_size)
    for _ in range(num_hidden):
        # Small init to keep residual branches near zero at start
        ws.append(rng.normal(0.0, np.sqrt(2.0 / hidden_size) * 0.1, size=(hidden_size, hidden_size)))
        bs.append(np.zeros(hidden_size))

    # Layer num_layers-1: output (hidden_size → 1)
    ws.append(rng.normal(0.0, np.sqrt(2.0 / hidden_size), size=(hidden_size,)))
    bs.append(np.zeros(1))

    # ── Adam moments ─────────────────────────────────────────────────────────
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    mws = [np.zeros_like(w) for w in ws]
    vws = [np.zeros_like(w) for w in ws]
    mbs = [np.zeros_like(b) for b in bs]
    vbs = [np.zeros_like(b) for b in bs]

    actual_batch_size = min(batch_size, len(x_train))
    step_logs: list[dict] = []

    # Pre-compute class weight (outside batch loop for efficiency)
    n_pos = float(np.sum(y_train == 1))
    n_neg = float(np.sum(y_train == 0))
    pos_weight = min(1.6, max(1.1, (n_neg / max(n_pos, 1.0)) ** 0.4))

    dropout_p = 0.20
    dropout_scale = 1.0 / (1.0 - dropout_p)

    for epoch in range(1, int(epochs) + 1):
        perm = rng.permutation(len(x_train))
        x_shuffled = x_train_norm[perm]
        y_shuffled = y_train[perm]

        epoch_loss = 0.0
        epoch_logits: list[float] = []
        epoch_probs: list[float] = []
        total_grad_norm = 0.0

        for start_idx in range(0, len(x_train), actual_batch_size):
            end_idx = start_idx + actual_batch_size
            x_batch = x_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            bs_size = len(y_batch)

            # ── Forward pass ──────────────────────────────────────────────────
            hs: list[np.ndarray] = []           # pre-dropout activations
            hds: list[np.ndarray] = []          # post-dropout activations
            masks: list[np.ndarray] = []        # dropout masks

            # Layer 0: input projection (no residual)
            h = _leaky_relu(x_batch @ ws[0] + bs[0])
            mask = (rng.uniform(0.0, 1.0, size=h.shape) >= dropout_p).astype(np.float64) * dropout_scale
            hd = h * mask
            hs.append(h); hds.append(hd); masks.append(mask)

            # Hidden residual blocks: (1,2), (3,4), …, (num_hidden-1, num_hidden)
            layer_idx = 1
            while layer_idx <= num_hidden:
                h_skip = hds[-1]  # skip connection from previous layer output

                # First layer of block
                h_a = _leaky_relu(hds[-1] @ ws[layer_idx] + bs[layer_idx])
                mask_a = (rng.uniform(0.0, 1.0, size=h_a.shape) >= dropout_p).astype(np.float64) * dropout_scale
                hd_a = h_a * mask_a
                hs.append(h_a); hds.append(hd_a); masks.append(mask_a)
                layer_idx += 1

                if layer_idx <= num_hidden:
                    # Second layer of block + residual
                    h_b = _leaky_relu(hd_a @ ws[layer_idx] + bs[layer_idx])
                    mask_b = (rng.uniform(0.0, 1.0, size=h_b.shape) >= dropout_p).astype(np.float64) * dropout_scale
                    hd_b = h_b * mask_b
                    # Residual: add skip to post-dropout output
                    hd_b = hd_b + h_skip
                    hs.append(h_b); hds.append(hd_b); masks.append(mask_b)
                    layer_idx += 1

            # Output layer
            out_idx = num_layers - 1
            logits = hds[-1] @ ws[out_idx] + bs[out_idx]
            probabilities = _sigmoid(logits)

            # ── Loss: Weighted Focal (gamma=1.0) ──────────────────────────────
            probs_c = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
            sample_w = np.where(y_batch == 1, pos_weight, 1.0)
            p_t = np.where(y_batch == 1, probs_c, 1.0 - probs_c)
            focal = (1.0 - p_t) ** 1.0
            batch_loss = -np.mean(sample_w * focal * (y_batch * np.log(probs_c) + (1.0 - y_batch) * np.log(1.0 - probs_c)))
            epoch_loss += batch_loss * bs_size
            epoch_logits.extend(logits.tolist())
            epoch_probs.extend(probabilities.tolist())

            # ── Backward pass ─────────────────────────────────────────────────
            # Gradient at logit layer
            grad_out = sample_w * focal * (probabilities - y_batch) / bs_size  # (bs,)

            # Initialise per-layer gradient lists
            grad_ws = [None] * num_layers
            grad_bs = [None] * num_layers

            # Output layer gradients
            hidden_before_out = hds[-1]  # post-dropout activations feeding output
            grad_ws[out_idx] = hidden_before_out.T @ grad_out + l2_lambda * ws[out_idx] + l1_lambda * np.sign(ws[out_idx])
            grad_bs[out_idx] = np.asarray([grad_out.sum()])

            # Backprop delta into last hidden layer
            delta = grad_out[:, None] * ws[out_idx][None, :]   # (bs, hidden_size)

            # Backprop through residual blocks in reverse
            # `hs` and `hds` and `masks` are indexed 0..num_hidden (num_hidden+1 entries)
            # hs[0] = projection layer activations; hs[1..num_hidden] = residual block activations
            hidden_layer_idx = num_hidden  # current position in hs/hds/masks arrays

            layer_idx = num_hidden  # current weight layer index (1-indexed)
            while layer_idx >= 1:
                if layer_idx % 2 == 0 and layer_idx <= num_hidden:
                    # Second layer of a residual block
                    h = hs[hidden_layer_idx]
                    mask = masks[hidden_layer_idx]
                    # delta passes through dropout mask and LeakyReLU
                    delta_drop = delta * mask
                    delta_relu = delta_drop * np.where(h > 0, 1.0, 0.01)

                    prev_hd = hds[hidden_layer_idx - 1]
                    grad_ws[layer_idx] = prev_hd.T @ delta_relu + l2_lambda * ws[layer_idx] + l1_lambda * np.sign(ws[layer_idx])
                    grad_bs[layer_idx] = delta_relu.sum(axis=0)

                    # Carry delta back to first layer of block
                    delta_to_prev = delta_relu @ ws[layer_idx].T
                    # Residual: also pass original delta forward (skip connection gradient)
                    delta_skip = delta  # gradient flows through skip connection unchanged
                    hidden_layer_idx -= 1
                    layer_idx -= 1

                    # First layer of the same block
                    h_a = hs[hidden_layer_idx]
                    mask_a = masks[hidden_layer_idx]
                    delta_a = delta_to_prev * mask_a
                    delta_a_relu = delta_a * np.where(h_a > 0, 1.0, 0.01)

                    prev_hd_a = hds[hidden_layer_idx - 1] if hidden_layer_idx > 0 else x_batch
                    grad_ws[layer_idx] = prev_hd_a.T @ delta_a_relu + l2_lambda * ws[layer_idx] + l1_lambda * np.sign(ws[layer_idx])
                    grad_bs[layer_idx] = delta_a_relu.sum(axis=0)

                    # Delta for next block = from first layer + skip
                    delta = delta_a_relu @ ws[layer_idx].T + delta_skip

                    hidden_layer_idx -= 1
                    layer_idx -= 1
                else:
                    # Single layer (happens only if num_hidden is odd after clamping, or layer 0)
                    h = hs[hidden_layer_idx]
                    mask = masks[hidden_layer_idx]
                    delta_drop = delta * mask
                    delta_relu = delta_drop * np.where(h > 0, 1.0, 0.01)

                    prev_hd = hds[hidden_layer_idx - 1] if hidden_layer_idx > 0 else x_batch
                    grad_ws[layer_idx] = prev_hd.T @ delta_relu + l2_lambda * ws[layer_idx] + l1_lambda * np.sign(ws[layer_idx])
                    grad_bs[layer_idx] = delta_relu.sum(axis=0)
                    delta = delta_relu @ ws[layer_idx].T

                    hidden_layer_idx -= 1
                    layer_idx -= 1

            # Input projection layer (layer 0)
            h0 = hs[0]
            mask0 = masks[0]
            delta0 = delta * mask0
            delta0_relu = delta0 * np.where(h0 > 0, 1.0, 0.01)
            grad_ws[0] = x_batch.T @ delta0_relu + l2_lambda * ws[0] + l1_lambda * np.sign(ws[0])
            grad_bs[0] = delta0_relu.sum(axis=0)

            # ── Gradient clipping (global norm) ───────────────────────────────
            all_grads = [g for g in grad_ws if g is not None] + [g for g in grad_bs if g is not None]
            global_norm = float(np.sqrt(sum(float(np.sum(g ** 2)) for g in all_grads)))
            total_grad_norm += global_norm
            if global_norm > grad_clip:
                scale = grad_clip / (global_norm + 1e-8)
                grad_ws = [g * scale if g is not None else None for g in grad_ws]
                grad_bs = [g * scale if g is not None else None for g in grad_bs]

            # ── Cosine Learning Rate Schedule ──────────────────────────────────
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * epoch / float(epochs)))
            effective_lr = learning_rate * max(0.1, lr_scale)

            # ── Adam parameter updates ────────────────────────────────────────
            t += 1
            bc1 = 1.0 - beta1 ** t
            bc2 = 1.0 - beta2 ** t

            for i in range(num_layers):
                gw = grad_ws[i]
                gb = grad_bs[i]
                if gw is None or gb is None:
                    continue
                mws[i] = beta1 * mws[i] + (1.0 - beta1) * gw
                vws[i] = beta2 * vws[i] + (1.0 - beta2) * (gw ** 2)
                ws[i] -= effective_lr * (mws[i] / bc1) / (np.sqrt(vws[i] / bc2) + eps)

                mbs[i] = beta1 * mbs[i] + (1.0 - beta1) * gb
                vbs[i] = beta2 * vbs[i] + (1.0 - beta2) * (gb ** 2)
                bs[i] -= effective_lr * (mbs[i] / bc1) / (np.sqrt(vbs[i] / bc2) + eps)

        num_batches = max(1, math.ceil(len(x_train) / actual_batch_size))
        avg_loss = epoch_loss / len(x_train)
        mean_prob = float(np.mean(epoch_probs))
        avg_grad = total_grad_norm / num_batches
        w0_norm = float(np.linalg.norm(ws[0]))
        wout_norm = float(np.linalg.norm(ws[-1]))

        if epoch == 1 or epoch == epochs or epoch % 20 == 0:
            pos_ratio = float(np.mean(np.array(epoch_probs) >= threshold))
            mean_logit = float(np.mean(epoch_logits))
            std_logit = float(np.std(epoch_logits))
            logit_spread = "collapsed" if std_logit < 0.05 else ("narrow" if std_logit < 0.3 else ("healthy" if std_logit < 1.2 else "wide"))
            prob_stance = (
                "aggressively bullish" if mean_prob > 0.65 else
                "mildly bullish" if mean_prob > 0.52 else
                "neutral" if mean_prob > 0.45 else
                "mildly bearish" if mean_prob > 0.35 else
                "strongly bearish"
            )
            grad_health = (
                "gradient has vanished" if avg_grad < 1e-5 else
                "gradient is very small" if avg_grad < 1e-3 else
                "gradient is healthy" if avg_grad < 0.5 else
                "gradient is large — may need clipping" if avg_grad < 2.0 else
                "gradient is exploding"
            )
            trade_rate_desc = (
                "refusing nearly all setups" if pos_ratio < 0.03 else
                "highly selective" if pos_ratio < 0.12 else
                "selective" if pos_ratio < 0.25 else
                "moderately active" if pos_ratio < 0.45 else
                "trading frequently"
            )
            progress_pct = round(100.0 * epoch / epochs, 1)
            loss_desc = (
                "still in early descent" if epoch <= epochs * 0.1 else
                "converging" if avg_loss > 0.5 else
                "well-converged" if avg_loss < 0.38 else
                "plateauing"
            )

            reasoning = (
                f"[Epoch {epoch}/{epochs} | {progress_pct}% complete] "
                f"The {num_layers}-layer deep residual model is {loss_desc} with loss={avg_loss:.5f}. "
                f"The mean output probability is {mean_prob:.3f}, reflecting a {prob_stance} posture. "
                f"At the decision threshold of {threshold:.2f}, the model is {trade_rate_desc} "
                f"(allowing {pos_ratio:.1%} of setups). "
                f"The logit distribution is {logit_spread} (std={std_logit:.3f}, mean={mean_logit:.3f}), "
                f"indicating {'good separation between trade and no-trade signals' if logit_spread == 'healthy' else 'the network needs more training to separate signals'}. "
                f"Global gradient norm is {avg_grad:.4f}: {grad_health}. "
                f"Input projection layer L0 weight norm={w0_norm:.3f}; output layer weight norm={wout_norm:.3f}."
            )

            step_logs.append({
                "epoch": epoch,
                "loss": round(avg_loss, 6),
                "progress_pct": progress_pct,
                "weights_norm": {"layer_0": round(w0_norm, 4), "output": round(wout_norm, 4)},
                "gradients_norm": {"avg_global": round(avg_grad, 4)},
                "logits_stat": {
                    "mean": round(mean_logit, 4),
                    "std": round(std_logit, 4),
                    "spread": logit_spread,
                },
                "probabilities_stat": {
                    "mean": round(mean_prob, 4),
                    "positive_ratio": round(pos_ratio, 4),
                    "stance": prob_stance,
                    "trade_rate": trade_rate_desc,
                },
                "gradient_health": grad_health,
                "loss_status": loss_desc,
                "reasoning": reasoning,
            })

    # ── Build weight dict ─────────────────────────────────────────────────────
    weight_dict: dict[str, np.ndarray] = {
        "num_layers": np.asarray(num_layers),
        "hidden_size": np.asarray(hidden_size),
    }
    for i, (w, b) in enumerate(zip(ws, bs)):
        weight_dict[f"layer_{i}_w"] = w
        weight_dict[f"layer_{i}_b"] = b

    model = LocalEdgeModel(weight_dict, mean, std, threshold=threshold)
    train_probs = _predict_matrix(model, x_train)
    val_probs = _predict_matrix(model, x_val)
    report = {
        "samples": int(len(x)),
        "train_samples": int(len(x_train)),
        "validation_samples": int(len(x_val)),
        "positive_rate": float(y.mean()),
        "threshold": float(threshold),
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "train_accuracy": _accuracy(train_probs, y_train, threshold),
        "validation_accuracy": _accuracy(val_probs, y_val, threshold),
        "validation_positive_rate": float(y_val.mean()) if len(y_val) else 0.0,
        "feature_names": FEATURE_NAMES,
        "step_logs": step_logs,
    }
    return model, report


# ─────────────────────────────────────────────────────────────────────────────
# Model I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model: LocalEdgeModel, path: str | Path, report: dict | None = None) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, np.ndarray] = {
        "feature_mean": model.feature_mean,
        "feature_std": model.feature_std,
        "threshold": np.asarray(model.threshold),
    }
    for key, val in model.weights.items():
        save_kwargs[key] = np.asarray(val)
    np.savez(output_path, **save_kwargs)
    if report is not None:
        output_path.with_suffix(".report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )


def _predict_matrix(model: LocalEdgeModel, x: np.ndarray) -> np.ndarray:
    """Batch inference — returns 1-D probability array of shape (n,)."""
    if len(x) == 0:
        return np.empty((0,), dtype=np.float64)
    x_norm = (x - model.feature_mean) / model.feature_std
    logits = _deep_forward(model.weights, x_norm)
    return _sigmoid(logits).ravel()


# ─────────────────────────────────────────────────────────────────────────────
# Activations & utilities
# ─────────────────────────────────────────────────────────────────────────────

def _leaky_relu(z: np.ndarray, leak: float = 0.01) -> np.ndarray:
    return np.maximum(leak * z, z)


def _accuracy(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    if len(labels) == 0:
        return 0.0
    return float(((probabilities >= threshold).astype(np.float64) == labels).mean())


def _sigmoid(values):
    values = np.clip(values, -60, 60)
    return 1.0 / (1.0 + np.exp(-values))


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features.jsonl")
    parser.add_argument("--output", default="data/models/local_edge_model.npz")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--min-abs-return", type=float, default=0.0)
    args = parser.parse_args(argv)

    rows = load_feature_rows(args.features)
    x, y = build_dataset(rows, min_abs_return=args.min_abs_return)
    model, report = train_local_edge_model(
        x,
        y,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        threshold=args.threshold,
    )
    save_model(model, args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
