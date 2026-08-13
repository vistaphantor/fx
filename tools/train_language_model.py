"""
Language Model Trainer — orchestrates tokenizer training + model training.

Usage:
    python tools/train_language_model.py

Config is set in the CONFIG dict at the top of this file.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# ── Project imports ───────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.language.tokenizer import BPETokenizer
from src.language.transformer import NanoGPT
from src.language.data_pipeline import (
    load_all_training_text,
    build_corpus_string,
    make_batches,
)

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit these to scale up/down
# ═════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Data ──────────────────────────────────────────────────────────────────
    "data_root":      "data/data/trainingdata",
    "max_examples":   None,          # None = load all available
    "val_split":      0.05,          # fraction of data for validation

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    "vocab_size":     4096,          # BPE vocabulary size
    "tok_save_path":  "data/models/language/tokenizer.json",
    "tok_retrain":    False,         # True = re-train even if tokenizer exists

    # ── Model architecture ────────────────────────────────────────────────────
    "d_model":        256,           # embedding / hidden dim
    "n_heads":        8,             # attention heads
    "n_layers":       6,             # transformer blocks  (~8M params)
    "max_seq_len":    256,           # context window
    "dropout_p":      0.10,

    # ── Training ──────────────────────────────────────────────────────────────
    "epochs":         20,
    "batch_size":     4,             # keep small on CPU
    "seq_len":        128,           # tokens per sequence
    "lr":             3e-4,
    "lr_min":         1e-5,          # cosine decay floor
    "weight_decay":   0.01,
    "grad_clip":      1.0,

    # ── Checkpointing & logging ───────────────────────────────────────────────
    "model_save_path": "data/models/language/nanogpt.npz",
    "log_path":        "data/models/language/training_log.jsonl",
    "save_every_steps": 500,
    "log_every_steps":  50,
    "generate_every":   500,        # generate sample text every N steps
    "generate_prompt":  "Human: What is the best way to learn programming?\n\nAssistant:",
    "generate_tokens":  150,
}

SEP = "═" * 90


def cosine_lr(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def train():
    cfg = CONFIG
    print(SEP)
    print(" VISTA LANGUAGE MODEL — NanoGPT Training")
    print(SEP)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading training data...")
    texts = load_all_training_text(
        data_root=Path(cfg["data_root"]),
        max_examples=cfg["max_examples"],
        shuffle=True,
    )
    if not texts:
        print("ERROR: No training texts found. Check data/data/trainingdata/")
        return

    corpus = build_corpus_string(texts)
    n_val = max(1, int(len(texts) * cfg["val_split"]))
    val_texts   = texts[:n_val]
    train_texts = texts[n_val:]
    train_corpus = build_corpus_string(train_texts)
    val_corpus   = build_corpus_string(val_texts)

    print(f"  Train: {len(train_texts):,} examples  |  Val: {len(val_texts):,} examples")
    print(f"  Corpus size: {len(corpus):,} chars")

    # ── 2. Train / load tokenizer ─────────────────────────────────────────────
    print("\n[2/5] Tokenizer...")
    tok_path = Path(cfg["tok_save_path"])
    if tok_path.exists() and not cfg["tok_retrain"]:
        print(f"  Loading existing tokenizer from {tok_path}")
        tok = BPETokenizer.load(tok_path)
    else:
        print("  Training new BPE tokenizer...")
        tok = BPETokenizer()
        tok.train(corpus[:500_000], vocab_size=cfg["vocab_size"])  # use first 500k chars for speed
        tok.save(tok_path)
    print(f"  Vocabulary size: {tok.vocab_size:,}")

    # ── 3. Tokenize corpus ────────────────────────────────────────────────────
    print("\n[3/5] Tokenizing corpus...")
    t0 = time.time()
    train_ids = tok.encode(train_corpus, add_bos=False, add_eos=False)
    val_ids   = tok.encode(val_corpus,   add_bos=False, add_eos=False)
    print(f"  Train tokens: {len(train_ids):,}  |  Val tokens: {len(val_ids):,}  ({time.time()-t0:.1f}s)")

    # ── 4. Build model ────────────────────────────────────────────────────────
    print("\n[4/5] Building model...")
    model = NanoGPT(
        vocab_size  = tok.vocab_size,
        d_model     = cfg["d_model"],
        n_heads     = cfg["n_heads"],
        n_layers    = cfg["n_layers"],
        max_seq_len = cfg["max_seq_len"],
        dropout_p   = cfg["dropout_p"],
    )

    model_path = Path(cfg["model_save_path"])
    if model_path.exists():
        print(f"  Found checkpoint at {model_path} — loading...")
        model.load(model_path)
    else:
        print("  Starting from random init.")

    # ── 5. Training loop ──────────────────────────────────────────────────────
    print(f"\n[5/5] Training  ({cfg['epochs']} epochs, seq_len={cfg['seq_len']}, batch={cfg['batch_size']})...")
    print(SEP)

    log_path = Path(cfg["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    seq_len    = cfg["seq_len"]
    batch_size = cfg["batch_size"]
    lr_max     = cfg["lr"]
    lr_min     = cfg["lr_min"]

    # Estimate total steps for LR schedule
    steps_per_epoch = max(1, len(train_ids) // (seq_len * batch_size))
    total_steps     = cfg["epochs"] * steps_per_epoch
    warmup_steps    = min(200, total_steps // 10)

    global_step = model.W.t  # resume from checkpoint step
    best_val_loss = float("inf")

    for epoch in range(cfg["epochs"]):
        epoch_losses = []
        t_epoch = time.time()

        for batch in make_batches(train_ids, seq_len=seq_len, batch_size=batch_size, shuffle=True):
            # Pad batch to same length
            max_len = max(len(s) for s in batch)
            arr = np.full((len(batch), max_len), tok.pad_id(), dtype=np.int32)
            for i, seq in enumerate(batch):
                arr[i, :len(seq)] = seq

            lr = cosine_lr(global_step, warmup_steps, total_steps, lr_max, lr_min)
            loss_val, grads = model.loss(arr, pad_id=tok.pad_id())
            model.W.adam_update(grads, lr=lr, weight_decay=cfg["weight_decay"],
                                 grad_clip=cfg["grad_clip"])

            epoch_losses.append(loss_val)
            global_step += 1

            # ── Logging ───────────────────────────────────────────────────────
            if global_step % cfg["log_every_steps"] == 0:
                recent = float(np.mean(epoch_losses[-cfg["log_every_steps"]:]))
                perplexity = math.exp(min(recent, 20))
                print(f"  Step {global_step:6d} | epoch {epoch+1}/{cfg['epochs']} | "
                      f"loss={recent:.4f} | ppl={perplexity:.1f} | lr={lr:.2e}")
                log_path.open("a").write(json.dumps({
                    "step": global_step, "epoch": epoch + 1,
                    "loss": round(recent, 4), "perplexity": round(perplexity, 2),
                    "lr": round(lr, 8),
                }) + "\n")

            # ── Checkpoint ────────────────────────────────────────────────────
            if global_step % cfg["save_every_steps"] == 0:
                model.save(model_path)
                print(f"  [Checkpoint] Saved to {model_path}")

            # ── Sample generation ─────────────────────────────────────────────
            if global_step % cfg["generate_every"] == 0:
                _generate_sample(model, tok, cfg)

        # ── End-of-epoch validation ───────────────────────────────────────────
        val_loss = _compute_val_loss(model, val_ids, seq_len, batch_size, tok)
        avg_train = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        elapsed = time.time() - t_epoch

        print(f"\n{SEP}")
        print(f"  Epoch {epoch+1}/{cfg['epochs']}  "
              f"train_loss={avg_train:.4f}  val_loss={val_loss:.4f}  "
              f"ppl={math.exp(min(val_loss,20)):.1f}  time={elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = model_path.with_stem(model_path.stem + "_best")
            model.save(best_path)
            print(f"  *** New best val_loss={val_loss:.4f} → saved to {best_path}")
        print(SEP + "\n")

    # ── Final checkpoint + generation sample ─────────────────────────────────
    model.save(model_path)
    _generate_sample(model, tok, cfg, n_samples=3)
    print(f"\n[Done] Training complete. Model saved to {model_path}")


def _compute_val_loss(model, val_ids, seq_len, batch_size, tok):
    losses = []
    for batch in make_batches(val_ids, seq_len=seq_len, batch_size=batch_size, shuffle=False):
        max_len = max(len(s) for s in batch)
        arr = np.full((len(batch), max_len), tok.pad_id(), dtype=np.int32)
        for i, seq in enumerate(batch):
            arr[i, :len(seq)] = seq
        loss_val, _ = model.loss(arr, pad_id=tok.pad_id())
        losses.append(loss_val)
        if len(losses) >= 20:  # limit validation time
            break
    return float(np.mean(losses)) if losses else 0.0


def _generate_sample(model, tok, cfg, n_samples: int = 1):
    prompt = cfg["generate_prompt"]
    print(f"\n  [Generate] Prompt: {prompt!r}")
    prompt_ids = tok.encode(prompt, add_bos=True, add_eos=False)

    for i in range(n_samples):
        gen_ids = model.generate(
            prompt_ids,
            max_new_tokens=cfg["generate_tokens"],
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            eos_id=tok.eos_id(),
        )
        gen_text = tok.decode(gen_ids)
        print(f"\n  [Sample {i+1}] {gen_text}")
    print()


if __name__ == "__main__":
    train()
