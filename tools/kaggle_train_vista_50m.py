"""
Vista-50M-Reasoning — Kaggle GPU Trainer v5.2 (Data Subsystem Edition)

Powered by the corpus package (SQLite registry, DatasetSource abstraction, binary VST1 sharding,
quality filtering, exact deduplication, and HuggingFace streaming support).

Paste into a Kaggle Notebook with GPU T4 x1 enabled.
Output: /kaggle/working/vista_50m_best.pt, training_log.csv, vista_50m_final.pt
"""

from __future__ import annotations

import math
import os
import sys
import json
import time
import warnings
import re
import random
import csv
from pathlib import Path
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

# Ensure local repo root is on sys.path
for p in ["/kaggle/input/models/victormutwirib/vistae/pytorch/default/1/fx", "/kaggle/working/fx", "."]:
    if Path(p).exists() and str(Path(p).resolve()) not in sys.path:
        sys.path.insert(0, str(Path(p).resolve()))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.language.tokenizer import BPETokenizer
from corpus import CorpusManager

# ── 1. Complete Deterministic Reproducibility & Matmul Precision Setup ───────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

# ── 2. Environment & CUDA Setup ───────────────────────────────────────────────
def _check_cuda():
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1).cuda()
        return True
    except Exception:
        return False

use_cuda = _check_cuda()
device   = torch.device("cuda" if use_cuda else "cpu")
use_amp  = use_cuda

print("=" * 75)
print(f" 🚀 VISTA-50M KAGGLE TRAINER v5.2 (CORPUS DATASUBSYSTEM EDITION)")
print(f" Running on Device: {device}  |  AMP (FP16): {use_amp}  |  Seed: {SEED}")
if use_cuda:
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f" GPU: {gpu_name} (VRAM: {vram_gb:.2f} GB)")
print("=" * 75)

# ── 3. Directory & Path Resolution ────────────────────────────────────────────
REPO = Path("/kaggle/input/models/victormutwirib/vistae/pytorch/default/1/fx")
if not REPO.exists():
    REPO = Path("/kaggle/working/fx")
if not REPO.exists():
    REPO = Path(".")

print(f"Repo Root Directory: {REPO.resolve()}")

TOK_PATHS = [
    REPO / "data/models/language_50m/tokenizer.json",
    REPO / "data/models/language/tokenizer.json",
    Path("/kaggle/working/fx/data/models/language_50m/tokenizer.json"),
    Path("data/models/language_50m/tokenizer.json"),
]

tok = None
for tp in TOK_PATHS:
    if Path(tp).exists():
        print(f"📦 Loading BPE Tokenizer: {tp}")
        tok = BPETokenizer.load(tp)
        break

if tok is None:
    raise FileNotFoundError(f"❌ Tokenizer not found in: {[str(p) for p in TOK_PATHS]}")

print(f"✅ BPE Tokenizer Loaded: {tok.vocab_size:,} subword tokens (Fingerprint: {tok.fingerprint()})")

DATA_ROOT = REPO / "data/data/trainingdata"
if not DATA_ROOT.exists():
    DATA_ROOT = Path("/kaggle/working/fx/data/data/trainingdata")
if not DATA_ROOT.exists():
    DATA_ROOT = Path("data/data/trainingdata")

out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

# ── 4. CorpusManager Pipeline Integration ──────────────────────────────────────
SEQ_LEN = 512
STRIDE = 256
MICRO_BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
EFFECTIVE_BATCH_SIZE = MICRO_BATCH_SIZE * ACCUMULATION_STEPS

cm = CorpusManager(
    local_roots=[DATA_ROOT],
    hf_sources=[
        {"path": "open-thoughts/OpenThoughts-114k", "split": "train", "max_examples": 20_000},
    ],
    tokenizer=tok,
    target_tokens=1_800_000_000_000,
    seq_len=SEQ_LEN,
    stride=STRIDE,
    db_path=out_dir / "corpus_registry.db",
    shard_dir=out_dir / "corpus_shards",
    json_export_path=out_dir / "registry.json",
    checkpoint_path=out_dir / "index_checkpoint.json",
)

cm.scan()
cm.status()
train_loader, val_loader = cm.build_loaders(batch_size=MICRO_BATCH_SIZE, pin_memory=use_cuda)

# ── 5. Transformer Model Architecture (VistaReasoningGPT v5.2) ────────────────
class TransformerBlock(nn.Module):
    def __init__(self, d=512, h=16, ffn=2048, drop=0.1, n_layers=16):
        super().__init__()
        self.nh = h
        self.hd = d // h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3*d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, ffn),
            nn.GELU(),
            nn.Linear(ffn, d),
            nn.Dropout(drop)
        )
        self.drop = nn.Dropout(drop)
        self.n_layers = n_layers

        # Mark residual projection layers for deep GPT initialization scaling
        self.proj._is_residual = True
        self.ff[2]._is_residual = True

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.drop.p if self.training else 0.0
        )
        x = x + self.drop(self.proj(attn_out.transpose(1, 2).reshape(B, T, C)))
        return x + self.ff(self.ln2(x))

class VistaReasoningGPT(nn.Module):
    def __init__(self, V=8192, d=512, L=16, H=16, ffn=2048, S=512, drop=0.1):
        super().__init__()
        self.vocab_size = V
        self.max_seq_len = S
        self.d_model = d
        self.n_layers = L
        self.tok_emb = nn.Embedding(V, d)
        self.pos_emb = nn.Embedding(S, d)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([TransformerBlock(d, H, ffn, drop, n_layers=L) for _ in range(L)])
        self.ln_f = nn.LayerNorm(d)
        
        self.lm_head = nn.Linear(d, V, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = 0.02 / math.sqrt(2 * self.n_layers) if getattr(module, "_is_residual", False) else 0.02
            module.weight.data.normal_(mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, targets=None, pad_ignore_id=0):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop((self.tok_emb(idx) * math.sqrt(self.d_model)) + self.pos_emb(pos))
        for b in self.blocks: x = b(x)
        logits = self.lm_head(self.ln_f(x))
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=pad_ignore_id
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=300, temperature=0.75, top_k=50, top_p=0.92, eos_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.max_seq_len:]
            logits, _ = self(ctx)
            logits_last = logits[:, -1, :]

            if temperature <= 0.0:
                nxt = logits_last.argmax(dim=-1, keepdim=True)
            else:
                logits_last = logits_last / max(temperature, 1e-6)
                if top_k > 0:
                    v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
                    logits_last[logits_last < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits_last, dim=-1)
                if top_p < 1.0:
                    sp, si = torch.sort(probs, descending=True)
                    cp = torch.cumsum(sp, dim=-1)
                    sp[(cp - sp) > top_p] = 0.0
                    sp /= sp.sum(dim=-1, keepdim=True)
                    nxt = si.gather(-1, torch.multinomial(sp, 1))
                else:
                    nxt = torch.multinomial(probs, 1)

            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and nxt.item() == eos_id:
                break
        return idx

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

BENCHMARK_PROMPTS = [
    ("MATH", "Human: What is 15% of 200?\n\nAssistant: <think>\n"),
    ("LOGIC", "Human: If Alice is older than Bob, and Bob is older than Charlie, who is youngest?\n\nAssistant: <think>\n"),
    ("CODE", "Human: Write a Python function `is_even(n)`.\n\nAssistant: <think>\n"),
    ("QA", "Human: What is the capital of France?\n\nAssistant: <think>\n"),
    ("COT", "Human: A store gives a 20% discount on a $50 jacket. What is the final price?\n\nAssistant: <think>\n"),
]

def run_epoch_benchmark(model: VistaReasoningGPT, epoch: int):
    model.eval()
    benchmark_lines = [f"=== BENCHMARK REPORT (DETERMINISTIC GREEDY) — EPOCH {epoch} ==="]
    for tag, prompt in BENCHMARK_PROMPTS:
        inp = torch.tensor([tok.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
        out = model.generate(inp, max_new_tokens=150, temperature=0.0, top_k=1, eos_id=tok.eos_id())
        gen = tok.decode(out[0].tolist())
        benchmark_lines.append(f"\n[{tag}] Prompt: {prompt.splitlines()[0]}")
        benchmark_lines.append(f"Result: {gen[:200]}...")
    
    report_text = "\n".join(benchmark_lines)
    log_file = out_dir / f"benchmark_epoch_{epoch:02d}.txt"
    log_file.write_text(report_text, encoding="utf-8")
    return report_text

csv_log_path = out_dir / "training_log.csv"
if not csv_log_path.exists():
    with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_ppl", "val_ppl", "val_accuracy", "lr", "elapsed_seconds"])

best_path = out_dir / "vista_50m_best.pt"
final_path = out_dir / "vista_50m_final.pt"

CFG = dict(vocab_size=tok.vocab_size, d_model=512, n_layers=16, n_heads=16, ffn_dim=2048, max_seq_len=512, dropout=0.1)

raw_model = VistaReasoningGPT(V=tok.vocab_size, drop=CFG["dropout"]).to(device)

decay_params = []
no_decay_params = []

for name, param in raw_model.named_parameters():
    if not param.requires_grad: continue
    if param.ndim == 1 or "bias" in name or "ln" in name.lower() or "emb" in name.lower():
        no_decay_params.append(param)
    else:
        decay_params.append(param)

FORCE_SCRATCH = "--scratch" in sys.argv
start_val_loss = float("inf")
EPOCHS = 15
BASE_LR = 3e-4

fused_available = use_cuda and hasattr(torch.optim.AdamW, "fused")
optimizer = torch.optim.AdamW([
    {"params": decay_params, "weight_decay": 0.01},
    {"params": no_decay_params, "weight_decay": 0.0}
], lr=BASE_LR, fused=fused_available)

scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
global_step = 0

if best_path.exists() and not FORCE_SCRATCH:
    print(f"\n[Resume] Found existing checkpoint: {best_path}")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    prev_cfg = ckpt.get("config", {})
    if "version" in ckpt and prev_cfg.get("vocab_size") == tok.vocab_size:
        try:
            raw_model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in ckpt and use_amp:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            global_step = ckpt.get("global_step", 0)
            start_val_loss = ckpt.get("val_loss", float("inf"))
            EPOCHS = 10
            BASE_LR = 1e-4
            print(f"[Fine-tune] Resuming from step={global_step} | prev_val_loss={start_val_loss:.4f} | LR={BASE_LR}")
        except Exception as e:
            print(f"[Scratch] Loading error: {e} -> Starting fresh from scratch")
    else:
        print("[Scratch] Legacy or un-versioned checkpoint -> Starting fresh from scratch")
else:
    print(f"\n[Scratch] Training fresh from scratch (Epochs={EPOCHS}, LR={BASE_LR})")

model = raw_model
if hasattr(torch, "compile") and use_cuda:
    try:
        model = torch.compile(raw_model)
        print("⚡ PyTorch 2.x torch.compile() enabled!")
    except Exception as e:
        print(f"⚠️ torch.compile() skipped: {e}")

print(f"Model Parameters: {raw_model.get_num_params()/1e6:.1f}M")
print(f"Optimizer Parameter Groups: Decay={sum(p.numel() for p in decay_params)/1e6:.1f}M | No-Decay={sum(p.numel() for p in no_decay_params)/1e6:.1f}M | Fused={fused_available}")
print(f"Batch Config: Micro={MICRO_BATCH_SIZE} | Accum={ACCUMULATION_STEPS} | Effective Batch={EFFECTIVE_BATCH_SIZE}")
print(f"DataLoader: {len(train_loader)} micro-batches ({len(train_loader)//ACCUMULATION_STEPS} optimizer steps/epoch)")
print("=" * 75)

total_optimizer_steps = (EPOCHS * len(train_loader)) // ACCUMULATION_STEPS
warmup_steps = min(200, total_optimizer_steps // 10)

def get_lr(opt_step):
    if opt_step < warmup_steps:
        return BASE_LR * opt_step / max(1, warmup_steps)
    progress = (opt_step - warmup_steps) / max(1, total_optimizer_steps - warmup_steps)
    return 1e-6 + 0.5 * (BASE_LR - 1e-6) * (1 + math.cos(math.pi * progress))

best_val_loss = start_val_loss
patience_count = 0
PATIENCE = 5
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss_accum = 0.0
    optimizer.zero_grad()

    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        opt_step = global_step // ACCUMULATION_STEPS
        lr = get_lr(opt_step)
        for pg in optimizer.param_groups: pg["lr"] = lr
        
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model(x, y, pad_ignore_id=tok.pad_id())
            loss = loss / ACCUMULATION_STEPS

        scaler.scale(loss).backward()
        train_loss_accum += loss.item() * ACCUMULATION_STEPS
        global_step += 1

        if global_step % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    avg_train_loss = train_loss_accum / len(train_loader)
    train_ppl = math.exp(min(avg_train_loss, 20))

    model.eval()
    val_loss_accum = 0.0
    val_correct_tokens = 0
    val_total_tokens = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, loss = model(x, y, pad_ignore_id=tok.pad_id())
            
            val_loss_accum += loss.item()
            
            preds = logits.argmax(dim=-1)
            targets = y
            mask = (targets != tok.pad_id())
            if mask.sum() > 0:
                correct = (preds == targets) & mask
                val_correct_tokens += correct.sum().item()
                val_total_tokens += mask.sum().item()

    avg_val_loss = val_loss_accum / len(val_loader)
    val_ppl = math.exp(min(avg_val_loss, 20))
    val_acc = (val_correct_tokens / max(1, val_total_tokens)) * 100
    elapsed = time.time() - t0

    with open(csv_log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{avg_train_loss:.4f}", f"{avg_val_loss:.4f}", f"{train_ppl:.2f}", f"{val_ppl:.2f}", f"{val_acc:.2f}", f"{lr:.6f}", f"{elapsed:.0f}"])

    if avg_val_loss < best_val_loss - 0.001:
        best_val_loss = avg_val_loss
        patience_count = 0
        torch.save({
            "version": "5.2",
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "global_step": global_step,
            "config": CFG,
            "epoch": epoch,
            "loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_acc,
            "tokenizer_path": "data/models/language_50m/tokenizer.json",
        }, best_path)
        marker = "  <-- BEST VAL SAVED ✅"
    else:
        patience_count += 1
        marker = f"  [patience {patience_count}/{PATIENCE}]"

    print(f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} (PPL: {val_ppl:4.1f} | Acc: {val_acc:5.1f}%) | {elapsed:.0f}s{marker}")

    if epoch % 2 == 0 or epoch == EPOCHS:
        bench_text = run_epoch_benchmark(raw_model, epoch)

    if patience_count >= PATIENCE:
        print(f"\n[Early Stop] Validation loss stopped improving for {PATIENCE} epochs.")
        break

torch.save({
    "version": "5.2",
    "model_state_dict": raw_model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scaler_state_dict": scaler.state_dict() if use_amp else None,
    "global_step": global_step,
    "config": CFG,
    "epoch": epoch,
    "loss": avg_train_loss,
    "val_loss": avg_val_loss,
    "val_accuracy": val_acc,
    "tokenizer_path": "data/models/language_50m/tokenizer.json",
}, final_path)

print("\n" + "=" * 75)
print(f" 🎉 Training Complete! Best Validation Loss: {best_val_loss:.4f}")
print(f" 💾 Best Checkpoint:  {best_path}")
print(f" 💾 Final Checkpoint: {final_path}")
print(f" 📊 Telemetry Log:    {csv_log_path}")
print("=" * 75)
