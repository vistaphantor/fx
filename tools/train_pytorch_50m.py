"""
PyTorch 50M Parameter Reasoning Transformer — Live Streaming Console Trainer

Streams training telemetry (Loss, Perplexity, LR, Step/sec) and live generated
Chain-of-Thought (<think>...</think>) responses directly to your terminal in real time!

Usage:
    python tools/train_pytorch_50m.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

# Ensure output streams unbuffered so progress prints instantly
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.language.tokenizer import BPETokenizer
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.data_pipeline import load_all_training_text, build_corpus_string

# Multi-threaded CPU execution
num_cores = os.cpu_count() or 4
torch.set_num_threads(num_cores)
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)

CONFIG = {
    "data_root":       "data/data/trainingdata",
    "val_split":       0.05,
    "vocab_size":      8192,
    "tok_save_path":   "data/models/language_50m/tokenizer.json",
    "d_model":         512,
    "n_heads":         16,
    "n_layers":        16,
    "ffn_dim":         2048,
    "max_seq_len":     512,
    "dropout":         0.10,
    "epochs":          15,
    "batch_size":      8,
    "seq_len":         256,
    "lr":              4e-4,
    "lr_min":          1e-5,
    "weight_decay":    0.01,
    "grad_clip":       1.0,
    "model_save_path": "data/models/language_50m/vista_50m.pt",
    "log_every":       5,            # Print telemetry every 5 steps
    "generate_every":  25,           # Stream a live sample response every 25 steps!
    "sample_prompts": [
        "Human: Solve 15 * 14.\n\nAssistant: <think>",
        "Human: If a car travels 120 km in 2 hours, what is its average speed?\n\nAssistant: <think>",
        "Human: What is the derivative of x^2?\n\nAssistant: <think>",
    ]
}

SEP  = "═" * 80
LINE = "─" * 80


class TokenSequenceDataset(Dataset):
    def __init__(self, token_ids: list[int], seq_len: int = 256):
        self.seq_len = seq_len
        n_chunks = (len(token_ids) - 1) // seq_len
        self.data = [token_ids[i * seq_len : i * seq_len + seq_len + 1] for i in range(n_chunks)]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)


def stream_train():
    cfg = CONFIG
    print(SEP, flush=True)
    print(f" 🚀 VISTA-50M-REASONING — REAL-TIME LIVE STREAMING TRAINER", flush=True)
    print(f" CPU Cores: {num_cores}  |  PyTorch: {torch.__version__}  |  Params: 50.3 Million", flush=True)
    print(SEP, flush=True)

    print("\n[1/4] Loading training texts...", flush=True)
    texts = load_all_training_text(data_root=Path(cfg["data_root"]), shuffle=True)
    if not texts:
        print("ERROR: No training texts found.", flush=True)
        return

    n_val = max(1, int(len(texts) * cfg["val_split"]))
    val_texts, train_texts = texts[:n_val], texts[n_val:]
    full_corpus  = build_corpus_string(texts)
    train_corpus = build_corpus_string(train_texts)
    val_corpus   = build_corpus_string(val_texts)

    print(f"  Corpus: {len(texts):,} conversations ({len(full_corpus):,} chars)", flush=True)

    # Tokenizer
    print("\n[2/4] BPE Tokenizer...", flush=True)
    tok_path = Path(cfg["tok_save_path"])
    if tok_path.exists():
        tok = BPETokenizer.load(tok_path)
    else:
        print(f"  Training 8K subword BPE tokenizer on corpus...", flush=True)
        tok = BPETokenizer()
        tok.train(full_corpus[:600_000], vocab_size=cfg["vocab_size"])
        tok.save(tok_path)
    print(f"  Vocabulary: {tok.vocab_size:,} subword tokens", flush=True)

    # Tokenize
    print("\n[3/4] Tokenizing training sequences...", flush=True)
    train_ids = tok.encode(train_corpus, add_bos=False, add_eos=False)
    val_ids   = tok.encode(val_corpus,   add_bos=False, add_eos=False)
    print(f"  Total Tokens: {len(train_ids):,} train | {len(val_ids):,} val", flush=True)

    train_loader = DataLoader(TokenSequenceDataset(train_ids, cfg["seq_len"]), batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    val_loader   = DataLoader(TokenSequenceDataset(val_ids,   cfg["seq_len"]), batch_size=cfg["batch_size"], shuffle=False)

    # Model
    print("\n[4/4] Initializing VistaReasoningGPT (50.3M Parameters)...", flush=True)
    device = torch.device("cpu")
    model = VistaReasoningGPT(
        vocab_size=tok.vocab_size, d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], ffn_dim=cfg["ffn_dim"], max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"]
    ).to(device)

    model_path = Path(cfg["model_save_path"])
    if model_path.exists():
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded existing weights from {model_path}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = len(train_loader) * cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=cfg["lr_min"])

    print(f"\n{SEP}", flush=True)
    print(f" 🎬 LIVE TRAINING STARTED — Watch loss drop and reasoning improve in real time!", flush=True)
    print(f"{SEP}\n", flush=True)

    global_step = 0
    t_start = time.time()

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0.0

        for step, (x, y) in enumerate(train_loader):
            step_t0 = time.time()
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits, loss = model(x, targets=y, pad_id=tok.pad_id())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            global_step += 1
            step_time = time.time() - step_t0
            speed = 1.0 / max(step_time, 1e-4)

            # ── Real-Time Progress Stream ──────────────────────────────────────
            if global_step % cfg["log_every"] == 0:
                ppl = math.exp(min(loss_val, 20))
                lr = scheduler.get_last_lr()[0]
                pct = (global_step / total_steps) * 100
                bar_len = 20
                filled = int(bar_len * global_step / total_steps)
                bar = "█" * filled + "░" * (bar_len - filled)

                print(f"\r[{bar}] {pct:5.1f}% | Step {global_step:5d}/{total_steps} | "
                      f"Epoch {epoch+1:2d} | Loss: {loss_val:.4f} | PPL: {ppl:6.1f} | "
                      f"Speed: {speed:4.1f} stp/s | LR: {lr:.1e}", flush=True)

            # ── Stream Live Reasoning Generation Sample ────────────────────────
            if global_step % cfg["generate_every"] == 0:
                print(f"\n\n{LINE}", flush=True)
                print(f" 🧠 LIVE REASONING SAMPLE (Step {global_step} | Loss: {loss_val:.4f})", flush=True)
                print(f"{LINE}", flush=True)

                prompt = cfg["sample_prompts"][(global_step // cfg["generate_every"]) % len(cfg["sample_prompts"])]
                print(f" Prompt: {prompt!r}\n", flush=True)
                print(" Generation: ", end="", flush=True)

                model.eval()
                with torch.no_grad():
                    prompt_ids = tok.encode(prompt, add_bos=True, add_eos=False)
                    inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                    out = model.generate(inp, max_new_tokens=100, temperature=0.75, top_k=40, eos_id=tok.eos_id())
                    gen_text = tok.decode(out[0].tolist())
                    print(gen_text, flush=True)

                model.train()
                print(f"{LINE}\n", flush=True)
                torch.save({"model_state_dict": model.state_dict(), "config": cfg}, model_path)

        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"\n>>> Epoch {epoch+1} Complete! Avg Loss: {avg_loss:.4f}  (Total Time: {time.time()-t_start:.1f}s)\n", flush=True)

    print(f"\n{SEP}", flush=True)
    print(" 🎉 TRAINING COMPLETE! Model saved to data/models/language_50m/vista_50m.pt", flush=True)
    print(SEP, flush=True)


if __name__ == "__main__":
    stream_train()
