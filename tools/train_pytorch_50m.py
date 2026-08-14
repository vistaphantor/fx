from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.language.data_pipeline import build_tokenizer_training_sample, load_all_training_text
from src.language.model_bundle import save_model_bundle
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.tokenizer import BPETokenizer, TOKENIZER_ALGORITHM_VERSION
from src.language.training_pipeline import (
    PackedSequenceDataset,
    build_example_sequences,
    corpus_fingerprint,
    run_training_preflight,
    split_by_prompt_family,
    split_fingerprint,
)

TRAINING_STATE_VERSION = 1
PREFLIGHT_MANIFEST_VERSION = 1
SEED = 42

PROFILES = {
    "smoke": {
        "vocab_size": 1024,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 4,
        "ffn_dim": 512,
        "seq_len": 128,
        "batch_size": 4,
        "epochs": 20,
        "lr": 8e-4,
        "max_examples": 512,
        "tokenizer_chars": 1_500_000,
        "early_stop_patience": 5,
        "early_stop_min_delta": 0.003,
    },
    "8m": {
        "vocab_size": 4096,
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 8,
        "ffn_dim": 1024,
        "seq_len": 192,
        "batch_size": 2,
        "epochs": 12,
        "lr": 4e-4,
        "max_examples": None,
        "tokenizer_chars": 4_000_000,
        "early_stop_patience": 3,
        "early_stop_min_delta": 0.005,
    },
    "15m": {
        "vocab_size": 8192,
        "d_model": 320,
        "n_heads": 8,
        "n_layers": 10,
        "ffn_dim": 1280,
        "seq_len": 256,
        "batch_size": 1,
        "epochs": 10,
        "lr": 3e-4,
        "max_examples": None,
        "tokenizer_chars": 8_000_000,
        "early_stop_patience": 3,
        "early_stop_min_delta": 0.005,
    },
}


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _model_config(cfg: dict, tokenizer: BPETokenizer) -> dict:
    return {
        "vocab_size": tokenizer.vocab_size,
        "d_model": cfg["d_model"],
        "n_layers": cfg["n_layers"],
        "n_heads": cfg["n_heads"],
        "ffn_dim": cfg["ffn_dim"],
        # Never expose positional embeddings that were not trained.
        "max_seq_len": cfg["seq_len"],
        "dropout": cfg["dropout"],
    }


def _new_optimizer(model: VistaReasoningGPT, cfg: dict) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )


def _epoch_loader(dataset: PackedSequenceDataset, *, batch_size: int, epoch: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(SEED + int(epoch))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        num_workers=0,
    )


def _validation_loader(dataset: PackedSequenceDataset, *, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )


def _token_weighted_loss(
    model: VistaReasoningGPT,
    loader: DataLoader,
    *,
    pad_id: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_clip: float = 1.0,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, y in loader:
            valid_tokens = int((y != pad_id).sum().item())
            if valid_tokens <= 0:
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)

            _, loss = model(x, targets=y, pad_id=pad_id)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(
                    "non_finite_training_loss" if training else "non_finite_validation_loss"
                )

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += float(loss.item()) * valid_tokens
            total_tokens += valid_tokens

    if total_tokens <= 0:
        raise RuntimeError("no_non_padding_tokens_seen")
    return total_loss / total_tokens


def _static_preflight_contract(
    *,
    profile: str,
    training_stage: str,
    cfg: dict,
    corpus_hash: str,
    split_hash: str,
) -> dict:
    return {
        "preflight_manifest_version": PREFLIGHT_MANIFEST_VERSION,
        "profile": profile,
        "training_stage": training_stage,
        "tokenizer_algorithm_version": TOKENIZER_ALGORITHM_VERSION,
        "target_vocab_size": cfg["vocab_size"],
        "tokenizer_chars": cfg["tokenizer_chars"],
        "seq_len": cfg["seq_len"],
        "corpus_fingerprint": corpus_hash,
        "split_fingerprint": split_hash,
    }


def _validate_contract(saved: dict, expected: dict, *, prefix: str) -> None:
    for key, expected_value in expected.items():
        if saved.get(key) != expected_value:
            raise RuntimeError(f"{prefix}_contract_mismatch:{key}")


def _load_or_train_tokenizer(
    *,
    tokenizer_path: Path,
    preflight_path: Path,
    static_contract: dict,
    train_texts: list[str],
    cfg: dict,
) -> tuple[BPETokenizer, bool]:
    if tokenizer_path.exists():
        if not preflight_path.exists():
            raise RuntimeError(
                "unverified_preflight_tokenizer_exists:delete_bundle_.training_and_run_preflight"
            )
        saved = json.loads(preflight_path.read_text(encoding="utf-8"))
        _validate_contract(saved, static_contract, prefix="preflight")
        tokenizer = BPETokenizer.load(tokenizer_path)
        if saved.get("tokenizer_fingerprint") != tokenizer.fingerprint():
            raise RuntimeError("preflight_tokenizer_fingerprint_mismatch")
        if saved.get("actual_vocab_size") != tokenizer.vocab_size:
            raise RuntimeError("preflight_tokenizer_vocab_size_mismatch")
        return tokenizer, True

    if preflight_path.exists():
        raise RuntimeError("preflight_manifest_exists_without_tokenizer")

    tokenizer_sample = build_tokenizer_training_sample(
        train_texts,
        max_chars=cfg["tokenizer_chars"],
        seed=SEED,
    )
    tokenizer = BPETokenizer()
    tokenizer.train(tokenizer_sample, vocab_size=cfg["vocab_size"])
    tokenizer.save(tokenizer_path)
    return tokenizer, False


def _checkpoint_contract(
    *,
    profile: str,
    training_stage: str,
    tokenizer: BPETokenizer,
    model_config: dict,
    corpus_hash: str,
    split_hash: str,
) -> dict:
    return {
        "training_state_version": TRAINING_STATE_VERSION,
        "profile": profile,
        "training_stage": training_stage,
        "tokenizer_algorithm_version": tokenizer.algorithm_version,
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "model_config": dict(model_config),
        "corpus_fingerprint": corpus_hash,
        "split_fingerprint": split_hash,
    }


def _print_budget(*, report, model: VistaReasoningGPT, epochs: int) -> dict:
    parameter_count = model.get_num_params()
    per_epoch = report.train_prediction_tokens
    planned = per_epoch * epochs
    unique_ratio = per_epoch / max(parameter_count, 1)
    planned_ratio = planned / max(parameter_count, 1)
    print(
        f"[Budget] params={parameter_count:,} train_prediction_tokens/epoch={per_epoch:,} "
        f"planned_token_updates={planned:,} tokens_per_param/epoch={unique_ratio:.3f} "
        f"planned_tokens_per_param={planned_ratio:.3f}"
    )
    if unique_ratio < 1.0:
        print(
            "[Budget] WARNING: fewer than one packed training prediction token per model parameter "
            "per epoch. Architecture is valid, but scratch-model language quality may be data-limited."
        )
    return {
        "parameter_count": parameter_count,
        "train_prediction_tokens_per_epoch": per_epoch,
        "planned_token_updates": planned,
        "prediction_tokens_per_parameter_per_epoch": unique_ratio,
        "planned_prediction_tokens_per_parameter": planned_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="15m")
    parser.add_argument("--data-root", default="data/data/trainingdata")
    parser.add_argument("--bundle-dir", default=None)
    parser.add_argument("--training-stage", default="general_language")
    parser.add_argument("--threads", type=int, default=max(1, min(4, os.cpu_count() or 2)))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = dict(PROFILES[args.profile])
    cfg.update(
        {
            "dropout": 0.10,
            "lr_min": 1e-5,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "val_split": 0.05,
        }
    )

    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    bundle_dir = Path(args.bundle_dir or f"data/models/trading_language/{args.profile}")
    work_dir = bundle_dir / ".training"
    tokenizer_path = work_dir / "tokenizer.json"
    preflight_path = work_dir / "preflight.json"
    state_path = work_dir / "training_state.pt"
    best_model_path = work_dir / "best_model.pt"

    texts = load_all_training_text(
        data_root=Path(args.data_root),
        max_examples=cfg["max_examples"],
        shuffle=False,
        seed=SEED,
    )
    if not texts:
        raise SystemExit("No training data found")

    train_texts, val_texts = split_by_prompt_family(
        texts,
        val_fraction=cfg["val_split"],
        seed=SEED,
    )
    corpus_hash = corpus_fingerprint(texts)
    split_hash = split_fingerprint(train_texts, val_texts)
    print(
        f"[Split] train={len(train_texts):,} val={len(val_texts):,} "
        f"family-isolated=true split={split_hash[:12]}"
    )

    static_contract = _static_preflight_contract(
        profile=args.profile,
        training_stage=args.training_stage,
        cfg=cfg,
        corpus_hash=corpus_hash,
        split_hash=split_hash,
    )
    tokenizer, reused_verified_tokenizer = _load_or_train_tokenizer(
        tokenizer_path=tokenizer_path,
        preflight_path=preflight_path,
        static_contract=static_contract,
        train_texts=train_texts,
        cfg=cfg,
    )
    print(f"[Tokenizer] verified_reuse={str(reused_verified_tokenizer).lower()}")

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

    report = run_training_preflight(
        tokenizer=tokenizer,
        train_texts=train_texts,
        val_texts=val_texts,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        seq_len=cfg["seq_len"],
    )
    verified_manifest = {
        **static_contract,
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "actual_vocab_size": tokenizer.vocab_size,
        "roundtrip_cases": report.roundtrip_cases,
        "overfit_initial_loss": report.overfit_initial_loss,
        "overfit_final_loss": report.overfit_final_loss,
        "train_prediction_tokens": report.train_prediction_tokens,
        "validation_prediction_tokens": report.validation_prediction_tokens,
    }
    _atomic_json_save(verified_manifest, preflight_path)
    print(
        "[Preflight] PASS "
        f"tokenizer=v{report.tokenizer_algorithm_version} "
        f"roundtrips={report.roundtrip_cases} "
        f"overfit_loss={report.overfit_initial_loss:.4f}->{report.overfit_final_loss:.4f}"
    )

    model_config = _model_config(cfg, tokenizer)
    model = VistaReasoningGPT(**model_config).to("cpu")
    budget = _print_budget(report=report, model=model, epochs=cfg["epochs"])

    if args.preflight_only:
        print("[Preflight] Full training intentionally not started. Verified tokenizer is reusable.")
        return

    train_ds = PackedSequenceDataset(train_sequences, cfg["seq_len"], tokenizer.pad_id())
    val_ds = PackedSequenceDataset(val_sequences, cfg["seq_len"], tokenizer.pad_id())
    val_loader = _validation_loader(val_ds, batch_size=cfg["batch_size"])

    optimizer = _new_optimizer(model, cfg)
    steps_per_epoch = math.ceil(len(train_ds) / cfg["batch_size"])
    total_steps = max(1, steps_per_epoch * cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=cfg["lr_min"],
    )

    contract = _checkpoint_contract(
        profile=args.profile,
        training_stage=args.training_stage,
        tokenizer=tokenizer,
        model_config=model_config,
        corpus_hash=corpus_hash,
        split_hash=split_hash,
    )

    start_epoch = 0
    best_val = float("inf")
    epochs_without_improvement = 0
    if args.resume:
        if not state_path.exists():
            raise RuntimeError("resume_training_state_missing")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        _validate_contract(state, contract, prefix="resume")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state["epoch_completed"])
        best_val = float(state["best_validation_loss"])
        epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
        print(f"[Resume] continuing at epoch={start_epoch + 1} best_val={best_val:.4f}")
    elif state_path.exists():
        raise RuntimeError("training_state_already_exists:use_--resume")

    started = time.time()
    last_train_loss = float("nan")
    last_val_loss = float("nan")

    for epoch in range(start_epoch, cfg["epochs"]):
        epoch_started = time.time()
        train_loader = _epoch_loader(
            train_ds,
            batch_size=cfg["batch_size"],
            epoch=epoch,
        )
        train_loss = _token_weighted_loss(
            model,
            train_loader,
            pad_id=tokenizer.pad_id(),
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=cfg["grad_clip"],
        )
        val_loss = _token_weighted_loss(model, val_loader, pad_id=tokenizer.pad_id())
        last_train_loss = train_loss
        last_val_loss = val_loss

        improved = val_loss < (best_val - cfg["early_stop_min_delta"])
        if improved:
            best_val = val_loss
            epochs_without_improvement = 0
            _atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "validation_loss": best_val,
                },
                best_model_path,
            )
        else:
            epochs_without_improvement += 1

        state = {
            **contract,
            "epoch_completed": epoch + 1,
            "best_validation_loss": best_val,
            "epochs_without_improvement": epochs_without_improvement,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        _atomic_torch_save(state, state_path)

        elapsed = time.time() - epoch_started
        print(
            f"epoch={epoch + 1}/{cfg['epochs']} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} best_val={best_val:.4f} "
            f"params={model.get_num_params()/1e6:.2f}M epoch_seconds={elapsed:.1f} checkpoint=saved"
        )

        if epochs_without_improvement >= cfg["early_stop_patience"]:
            print(
                f"[EarlyStop] no validation improvement >= {cfg['early_stop_min_delta']:.4f} "
                f"for {epochs_without_improvement} epochs"
            )
            break

    if not best_model_path.exists():
        raise RuntimeError("best_model_checkpoint_missing")
    best_checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
    if best_checkpoint.get("model_config") != model_config:
        raise RuntimeError("best_model_config_mismatch")
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    metrics = {
        "best_validation_loss": best_val,
        "perplexity": math.exp(min(best_val, 20)),
        "training_seconds_this_run": time.time() - started,
        "profile": args.profile,
        "training_examples": len(train_texts),
        "validation_examples": len(val_texts),
        "train_sequences": len(train_sequences),
        "validation_sequences": len(val_sequences),
        "family_isolated_validation": True,
        "example_aware_packing": True,
        "preflight_passed": True,
        "preflight_overfit_initial_loss": report.overfit_initial_loss,
        "preflight_overfit_final_loss": report.overfit_final_loss,
        "last_training_loss": last_train_loss,
        "last_validation_loss": last_val_loss,
        "trained_context_length": cfg["seq_len"],
        "resumable_training": True,
        **budget,
    }

    save_model_bundle(
        bundle_dir=bundle_dir,
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        training_stage=args.training_stage,
        corpus_fingerprint=corpus_hash,
        metrics=metrics,
    )
    print(f"bundle={bundle_dir}")


if __name__ == "__main__":
    main()
