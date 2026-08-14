from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.language.data_pipeline import (
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


def _normalize_prompt_family(text: str) -> str:
    """Stable split key that keeps near-identical prompt variants together."""
    match = re.search(r"<user>\s*(.*?)\s*</user>", text, flags=re.DOTALL)
    if match:
        value = match.group(1)
    else:
        # General-text examples are grouped by their normalized prefix.
        value = text[:512]
    value = value.casefold()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[\s\.,;:!?]+$", "", value)
    return value.strip()


def split_by_prompt_family(
    texts: list[str],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Split entire prompt families, never individual variants, across train/val."""
    groups: dict[str, list[str]] = defaultdict(list)
    for text in texts:
        groups[_normalize_prompt_family(text)].append(text)

    keys = list(groups)
    random.Random(seed).shuffle(keys)
    target_val = max(1, int(round(len(texts) * val_fraction)))

    train: list[str] = []
    val: list[str] = []
    for key in keys:
        group = groups[key]
        if len(val) < target_val:
            val.extend(group)
        else:
            train.extend(group)

    if not train:
        raise RuntimeError("family_split_produced_empty_training_set")
    if not val:
        raise RuntimeError("family_split_produced_empty_validation_set")
    return train, val


def _content_windows(token_ids: list[int], seq_len: int) -> list[list[int]]:
    """Chunk one canonical example without crossing into another example."""
    if len(token_ids) < 2:
        return []

    # Each training item needs seq_len + 1 tokens so x/y stay aligned.
    window = seq_len + 1
    if len(token_ids) <= window:
        return [token_ids]

    # Overlap long examples so the model sees continuity instead of disconnected
    # fixed slices. No window ever crosses an example boundary.
    stride = max(1, seq_len // 2)
    chunks: list[list[int]] = []
    start = 0
    while start < len(token_ids) - 1:
        chunk = token_ids[start:start + window]
        if len(chunk) >= 2:
            chunks.append(chunk)
        if start + window >= len(token_ids):
            break
        start += stride
    return chunks


def build_example_sequences(
    texts: list[str],
    tokenizer: BPETokenizer,
    *,
    seq_len: int,
) -> list[list[int]]:
    """
    Tokenize examples independently, preserve complete short examples, and pack
    only complete examples together. Long examples are intentionally overlapped.
    """
    encoded_examples = [
        tokenizer.encode(text, add_bos=False, add_eos=False)
        for text in texts
        if text.strip()
    ]

    sequences: list[list[int]] = []
    pack: list[int] = []
    max_tokens = seq_len + 1

    for ids in encoded_examples:
        if len(ids) < 2:
            continue

        if len(ids) > max_tokens:
            if len(pack) >= 2:
                sequences.append(pack)
                pack = []
            sequences.extend(_content_windows(ids, seq_len))
            continue

        if not pack:
            pack = list(ids)
            continue

        if len(pack) + len(ids) <= max_tokens:
            pack.extend(ids)
        else:
            if len(pack) >= 2:
                sequences.append(pack)
            pack = list(ids)

    if len(pack) >= 2:
        sequences.append(pack)

    return sequences


class PackedSequenceDataset(Dataset):
    def __init__(self, sequences: list[list[int]], seq_len: int, pad_id: int):
        if not sequences:
            raise ValueError("sequences must not be empty")
        self.seq_len = int(seq_len)
        self.pad_id = int(pad_id)
        self.sequences = [list(seq) for seq in sequences]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx][: self.seq_len + 1]
        x = seq[:-1]
        y = seq[1:]

        if len(x) < self.seq_len:
            padding = self.seq_len - len(x)
            x = x + [self.pad_id] * padding
            y = y + [self.pad_id] * padding

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


def corpus_fingerprint(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
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
        shuffle=False,
        seed=42,
    )
    if not texts:
        raise SystemExit("No training data found")

    train_texts, val_texts = split_by_prompt_family(
        texts,
        val_fraction=cfg["val_split"],
        seed=42,
    )
    print(
        f"[Split] train={len(train_texts):,} val={len(val_texts):,} "
        f"family-isolated=true"
    )

    tokenizer_sample = build_tokenizer_training_sample(
        train_texts,
        max_chars=cfg["tokenizer_chars"],
        seed=42,
    )
    tokenizer = BPETokenizer()
    tokenizer.train(tokenizer_sample, vocab_size=cfg["vocab_size"])

    train_sequences = build_example_sequences(
        train_texts,
        tokenizer,
        seq_len=cfg["seq_len"],
    )
    val_sequences = build_example_sequences(
        val_texts,
        tokenizer,
        seq_len=cfg["seq_len"],
    )
    print(
        f"[Packing] train_sequences={len(train_sequences):,} "
        f"val_sequences={len(val_sequences):,} seq_len={cfg['seq_len']}"
    )

    train_ds = PackedSequenceDataset(train_sequences, cfg["seq_len"], tokenizer.pad_id())
    val_ds = PackedSequenceDataset(val_sequences, cfg["seq_len"], tokenizer.pad_id())

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

    bundle_dir = Path(args.bundle_dir or f"data/models/trading_language/{args.profile}")
    metrics = {
        "best_validation_loss": best_val,
        "perplexity": math.exp(min(best_val, 20)),
        "training_seconds": time.time() - started,
        "profile": args.profile,
        "training_examples": len(train_texts),
        "validation_examples": len(val_texts),
        "train_sequences": len(train_sequences),
        "validation_sequences": len(val_sequences),
        "family_isolated_validation": True,
        "example_aware_packing": True,
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
