from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.language.data_pipeline import (
    build_corpus_string,
    build_tokenizer_training_sample,
    load_all_training_text,
)
from src.language.model_bundle import save_model_bundle
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.tokenizer import BPETokenizer


PROFILES = {
    "smoke": {
        "vocab_size": 1024,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 4,
        "ffn_dim": 512,
        "max_seq_len": 256,
        "seq_len": 128,
        "batch_size": 4,
        "epochs": 20,
        "lr": 8e-4,
        "max_examples": 512,
        "tokenizer_chars": 1_500_000,
    },
    "8m": {
        "vocab_size": 4096,
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 6,
        "ffn_dim": 1024,
        "max_seq_len": 384,
        "seq_len": 192,
        "batch_size": 2,
        "epochs": 12,
        "lr": 4e-4,
        "max_examples": None,
        "tokenizer_chars": 4_000_000,
    },
    "15m": {
        "vocab_size": 8192,
        "d_model": 320,
        "n_heads": 8,
        "n_layers": 10,
        "ffn_dim": 1280,
        "max_seq_len": 512,
        "seq_len": 256,
        "batch_size": 1,
        "epochs": 10,
        "lr": 3e-4,
        "max_examples": None,
        "tokenizer_chars": 8_000_000,
    },
}


class TokenSequenceDataset(Dataset):
    def __init__(self, token_ids: list[int], seq_len: int):
        self.seq_len = seq_len
        self.starts = list(range(0, max(0, len(token_ids) - seq_len - 1), seq_len))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        chunk = self.token_ids[start:start + self.seq_len + 1]
        return (
            torch.tensor(chunk[:-1], dtype=torch.long),
            torch.tensor(chunk[1:], dtype=torch.long),
        )

    @property
    def token_ids(self):
        return self._token_ids

    @token_ids.setter
    def token_ids(self, value):
        self._token_ids = value


def corpus_fingerprint(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8", errors="replace"))
        digest.update(b"\\0")
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="15m")
    parser.add_argument("--data-root", default="data/data/trainingdata")
    parser.add_argument("--bundle-dir", default=None)
    parser.add_argument("--training-stage", default="general_language")
    parser.add_argument("--threads", type=int, default=max(1, min(4, os.cpu_count() or 2)))
    args = parser.parse_args()

    cfg = dict(PROFILES[args.profile])
    cfg.update({
        "dropout": 0.10,
        "lr_min": 1e-5,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "val_split": 0.05,
    })

    torch.set_num_threads(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    texts = load_all_training_text(
        data_root=Path(args.data_root),
        max_examples=cfg["max_examples"],
        shuffle=True,
        seed=42,
    )
    if not texts:
        raise SystemExit("No training data found")

    n_val = max(1, int(len(texts) * cfg["val_split"]))
    val_texts = texts[:n_val]
    train_texts = texts[n_val:]

    tokenizer_sample = build_tokenizer_training_sample(
        texts,
        max_chars=cfg["tokenizer_chars"],
        seed=42,
    )

    tokenizer = BPETokenizer()
    tokenizer.train(tokenizer_sample, vocab_size=cfg["vocab_size"])

    train_ids = tokenizer.encode(build_corpus_string(train_texts), add_bos=False, add_eos=False)
    val_ids = tokenizer.encode(build_corpus_string(val_texts), add_bos=False, add_eos=False)

    train_ds = TokenSequenceDataset(train_ids, cfg["seq_len"])
    val_ds = TokenSequenceDataset(val_ids, cfg["seq_len"])
    train_ds.token_ids = train_ids
    val_ds.token_ids = val_ids

    if len(train_ds) == 0:
        raise SystemExit("Training corpus too small for selected profile")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    model_cfg = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": cfg["d_model"],
        "n_layers": cfg["n_layers"],
        "n_heads": cfg["n_heads"],
        "ffn_dim": cfg["ffn_dim"],
        "max_seq_len": cfg["max_seq_len"],
        "dropout": cfg["dropout"],
    }

    model = VistaReasoningGPT(**model_cfg).to("cpu")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    total_steps = max(1, len(train_loader) * cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=cfg["lr_min"],
    )

    best_val = float("inf")
    best_state = None
    started = time.time()

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, targets=y, pad_id=tokenizer.pad_id())
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("non_finite_training_loss")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                _, loss = model(x, targets=y, pad_id=tokenizer.pad_id())
                if loss is not None:
                    val_loss += float(loss.item())
                    val_batches += 1

        train_loss = total_loss / max(1, len(train_loader))
        current_val = val_loss / max(1, val_batches) if val_batches else train_loss

        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={current_val:.4f} "
            f"params={model.get_num_params()/1e6:.2f}M"
        )

        if current_val < best_val:
            best_val = current_val
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    bundle_dir = Path(
        args.bundle_dir
        or f"data/models/trading_language/{args.profile}"
    )

    metrics = {
        "best_validation_loss": best_val,
        "perplexity": math.exp(min(best_val, 20)),
        "training_seconds": time.time() - started,
        "profile": args.profile,
    }

    save_model_bundle(
        bundle_dir=bundle_dir,
        model=model,
        tokenizer=tokenizer,
        model_config=model_cfg,
        training_stage=args.training_stage,
        corpus_fingerprint=corpus_fingerprint(texts),
        metrics=metrics,
    )

    print(f"bundle={bundle_dir}")


if __name__ == "__main__":
    main()
