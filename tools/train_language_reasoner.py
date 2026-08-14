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

from src.language.curriculum import CURRICULUM_STAGES, select_curriculum
from src.language.data_pipeline import build_tokenizer_training_sample, load_all_training_text
from src.language.model_bundle import load_model_bundle, save_model_bundle
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
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)


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
                raise RuntimeError("non_finite_training_loss" if training else "non_finite_validation_loss")
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


def _normalize_stage(value: str) -> str:
    stage = value.strip().casefold()
    if stage == "general_language":
        stage = "foundation"
    if stage not in CURRICULUM_STAGES:
        raise ValueError(f"unsupported_training_stage:{value}")
    return stage


def _static_preflight_contract(
    *,
    profile: str,
    stage: str,
    cfg: dict,
    source_hash: str,
    curriculum_hash: str,
    split_hash: str,
    lineage_fingerprint: str | None,
) -> dict:
    return {
        "preflight_manifest_version": PREFLIGHT_MANIFEST_VERSION,
        "profile": profile,
        "training_stage": stage,
        "tokenizer_algorithm_version": TOKENIZER_ALGORITHM_VERSION,
        "target_vocab_size": cfg["vocab_size"],
        "tokenizer_chars": cfg["tokenizer_chars"],
        "seq_len": cfg["seq_len"],
        "source_corpus_fingerprint": source_hash,
        "curriculum_fingerprint": curriculum_hash,
        "split_fingerprint": split_hash,
        "lineage_tokenizer_fingerprint": lineage_fingerprint,
    }


def _validate_contract(saved: dict, expected: dict, *, prefix: str) -> None:
    for key, expected_value in expected.items():
        if saved.get(key) != expected_value:
            raise RuntimeError(f"{prefix}_contract_mismatch:{key}")


def _prepare_lineage(
    *,
    stage: str,
    init_bundle: Path | None,
    cfg: dict,
) -> tuple[VistaReasoningGPT | None, BPETokenizer | None, str | None]:
    if stage == "foundation":
        if init_bundle is not None:
            raise RuntimeError("foundation_stage_must_start_without_init_bundle")
        return None, None, None
    if init_bundle is None:
        raise RuntimeError(f"{stage}_requires_--init-bundle")

    model, tokenizer, manifest = load_model_bundle(init_bundle, device="cpu")
    expected = _model_config(cfg, tokenizer)
    if manifest.model_config != expected:
        raise RuntimeError("init_bundle_model_config_mismatch")
    return model, tokenizer, tokenizer.fingerprint()


def _load_or_prepare_tokenizer(
    *,
    tokenizer_path: Path,
    preflight_path: Path,
    static_contract: dict,
    train_texts: list[str],
    cfg: dict,
    lineage_tokenizer: BPETokenizer | None,
) -> tuple[BPETokenizer, bool]:
    if tokenizer_path.exists():
        if not preflight_path.exists():
            raise RuntimeError("unverified_preflight_tokenizer_exists:delete_bundle_.training")
        saved = json.loads(preflight_path.read_text(encoding="utf-8"))
        _validate_contract(saved, static_contract, prefix="preflight")
        tokenizer = BPETokenizer.load(tokenizer_path)
        if saved.get("tokenizer_fingerprint") != tokenizer.fingerprint():
            raise RuntimeError("preflight_tokenizer_fingerprint_mismatch")
        return tokenizer, True

    if preflight_path.exists():
        raise RuntimeError("preflight_manifest_exists_without_tokenizer")

    if lineage_tokenizer is not None:
        tokenizer = lineage_tokenizer
        tokenizer.save(tokenizer_path)
        return tokenizer, False

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
    stage: str,
    tokenizer: BPETokenizer,
    model_config: dict,
    curriculum_hash: str,
    split_hash: str,
    lineage_fingerprint: str | None,
) -> dict:
    return {
        "training_state_version": TRAINING_STATE_VERSION,
        "profile": profile,
        "training_stage": stage,
        "tokenizer_algorithm_version": tokenizer.algorithm_version,
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "model_config": dict(model_config),
        "curriculum_fingerprint": curriculum_hash,
        "split_fingerprint": split_hash,
        "lineage_tokenizer_fingerprint": lineage_fingerprint,
    }


def _raw_training_tokens(train_texts: list[str], tokenizer: BPETokenizer) -> int:
    return sum(
        max(0, len(tokenizer.encode(text, add_bos=False, add_eos=False)) - 1)
        for text in train_texts
    )


def _print_budget(
    *,
    report,
    model: VistaReasoningGPT,
    tokenizer: BPETokenizer,
    train_texts: list[str],
    epochs: int,
) -> dict:
    parameter_count = model.get_num_params()
    raw_tokens = _raw_training_tokens(train_texts, tokenizer)
    packed_per_epoch = report.train_prediction_tokens
    planned = packed_per_epoch * epochs
    raw_ratio = raw_tokens / max(parameter_count, 1)
    planned_ratio = planned / max(parameter_count, 1)
    print(
        f"[Budget] params={parameter_count:,} raw_canonical_tokens={raw_tokens:,} "
        f"packed_tokens/epoch={packed_per_epoch:,} planned_token_updates={planned:,} "
        f"raw_tokens_per_param={raw_ratio:.3f} planned_tokens_per_param={planned_ratio:.3f}"
    )
    return {
        "parameter_count": parameter_count,
        "raw_canonical_training_tokens": raw_tokens,
        "train_prediction_tokens_per_epoch": packed_per_epoch,
        "planned_token_updates": planned,
        "raw_training_tokens_per_parameter": raw_ratio,
        "planned_prediction_tokens_per_parameter": planned_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="8m")
    parser.add_argument("--data-root", default="data/data/trainingdata")
    parser.add_argument("--bundle-dir", default=None)
    parser.add_argument("--training-stage", default="foundation")
    parser.add_argument("--init-bundle", default=None)
    parser.add_argument("--threads", type=int, default=max(1, min(4, os.cpu_count() or 2)))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-data-starved", action="store_true")
    args = parser.parse_args()

    stage = _normalize_stage(args.training_stage)
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

    bundle_dir = Path(args.bundle_dir or f"data/models/trading_language/{args.profile}/{stage}")
    work_dir = bundle_dir / ".training"
    tokenizer_path = work_dir / "tokenizer.json"
    preflight_path = work_dir / "preflight.json"
    state_path = work_dir / "training_state.pt"
    best_model_path = work_dir / "best_model.pt"

    all_texts = load_all_training_text(
        data_root=Path(args.data_root),
        max_examples=cfg["max_examples"],
        shuffle=False,
        seed=SEED,
    )
    if not all_texts:
        raise SystemExit("No training data found")

    selection = select_curriculum(all_texts, stage=stage, seed=SEED)
    texts = selection.texts
    print(
        f"[Curriculum] stage={selection.stage} selected={len(texts):,}/{selection.total_available:,} "
        f"trading_available={selection.trading_available:,} reasoning_available={selection.reasoning_available:,} "
        f"math_available={selection.math_available:,} replay={selection.replay_examples:,}"
    )

    lineage_model, lineage_tokenizer, lineage_fingerprint = _prepare_lineage(
        stage=stage,
        init_bundle=Path(args.init_bundle) if args.init_bundle else None,
        cfg=cfg,
    )

    train_texts, val_texts = split_by_prompt_family(
        texts,
        val_fraction=cfg["val_split"],
        seed=SEED,
    )
    source_hash = corpus_fingerprint(all_texts)
    curriculum_hash = corpus_fingerprint(texts)
    split_hash = split_fingerprint(train_texts, val_texts)
    print(
        f"[Split] train={len(train_texts):,} val={len(val_texts):,} "
        f"family-isolated=true split={split_hash[:12]}"
    )

    static_contract = _static_preflight_contract(
        profile=args.profile,
        stage=stage,
        cfg=cfg,
        source_hash=source_hash,
        curriculum_hash=curriculum_hash,
        split_hash=split_hash,
        lineage_fingerprint=lineage_fingerprint,
    )
    tokenizer, reused = _load_or_prepare_tokenizer(
        tokenizer_path=tokenizer_path,
        preflight_path=preflight_path,
        static_contract=static_contract,
        train_texts=train_texts,
        cfg=cfg,
        lineage_tokenizer=lineage_tokenizer,
    )
    print(
        f"[Tokenizer] verified_reuse={str(reused).lower()} "
        f"fingerprint={tokenizer.fingerprint()}"
    )

    train_sequences = build_example_sequences(train_texts, tokenizer, seq_len=cfg["seq_len"])
    val_sequences = build_example_sequences(val_texts, tokenizer, seq_len=cfg["seq_len"])
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
    _atomic_json_save(
        {
            **static_contract,
            "tokenizer_fingerprint": tokenizer.fingerprint(),
            "actual_vocab_size": tokenizer.vocab_size,
            "roundtrip_cases": report.roundtrip_cases,
            "overfit_initial_loss": report.overfit_initial_loss,
            "overfit_final_loss": report.overfit_final_loss,
            "train_prediction_tokens": report.train_prediction_tokens,
            "validation_prediction_tokens": report.validation_prediction_tokens,
        },
        preflight_path,
    )
    print(
        f"[Preflight] PASS tokenizer=v{report.tokenizer_algorithm_version} "
        f"roundtrips={report.roundtrip_cases} "
        f"overfit_loss={report.overfit_initial_loss:.4f}->{report.overfit_final_loss:.4f}"
    )

    model_config = _model_config(cfg, tokenizer)
    if lineage_model is not None:
        model = lineage_model
        if model.max_seq_len != model_config["max_seq_len"] or model.vocab_size != tokenizer.vocab_size:
            raise RuntimeError("lineage_model_runtime_contract_mismatch")
    else:
        model = VistaReasoningGPT(**model_config).to("cpu")

    budget = _print_budget(
        report=report,
        model=model,
        tokenizer=tokenizer,
        train_texts=train_texts,
        epochs=cfg["epochs"],
    )
    data_starved = budget["raw_training_tokens_per_parameter"] < 1.0
    if stage == "foundation" and args.profile != "smoke" and data_starved:
        print(
            "[Budget] BLOCK: foundation corpus has fewer than 1.0 raw canonical training token "
            "per model parameter. This is a conservative no-waste guardrail, not an optimality claim."
        )
        if not args.preflight_only and not args.allow_data_starved:
            raise RuntimeError(
                "foundation_training_blocked_by_data_budget:add_corpus_or_explicitly_use_--allow-data-starved"
            )

    if args.preflight_only:
        print("[Preflight] Full training intentionally not started. Verified artifacts are reusable.")
        return

    train_ds = PackedSequenceDataset(train_sequences, cfg["seq_len"], tokenizer.pad_id())
    val_ds = PackedSequenceDataset(val_sequences, cfg["seq_len"], tokenizer.pad_id())
    val_loader = _validation_loader(val_ds, batch_size=cfg["batch_size"])
    optimizer = _new_optimizer(model, cfg)
    steps_per_epoch = math.ceil(len(train_ds) / cfg["batch_size"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, steps_per_epoch * cfg["epochs"]),
        eta_min=cfg["lr_min"],
    )

    contract = _checkpoint_contract(
        profile=args.profile,
        stage=stage,
        tokenizer=tokenizer,
        model_config=model_config,
        curriculum_hash=curriculum_hash,
        split_hash=split_hash,
        lineage_fingerprint=lineage_fingerprint,
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
        print(f"[Resume] epoch={start_epoch + 1} best_val={best_val:.4f}")
    elif state_path.exists():
        raise RuntimeError("training_state_already_exists:use_--resume")

    started = time.time()
    last_train_loss = float("nan")
    last_val_loss = float("nan")

    for epoch in range(start_epoch, cfg["epochs"]):
        epoch_started = time.time()
        train_loss = _token_weighted_loss(
            model,
            _epoch_loader(train_ds, batch_size=cfg["batch_size"], epoch=epoch),
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

        _atomic_torch_save(
            {
                **contract,
                "epoch_completed": epoch + 1,
                "best_validation_loss": best_val,
                "epochs_without_improvement": epochs_without_improvement,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            state_path,
        )
        print(
            f"epoch={epoch + 1}/{cfg['epochs']} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} best_val={best_val:.4f} "
            f"params={model.get_num_params()/1e6:.2f}M "
            f"epoch_seconds={time.time() - epoch_started:.1f} checkpoint=saved"
        )
        if epochs_without_improvement >= cfg["early_stop_patience"]:
            print(
                f"[EarlyStop] no improvement >= {cfg['early_stop_min_delta']:.4f} "
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
        "training_stage": stage,
        "source_examples": len(all_texts),
        "curriculum_examples": len(texts),
        "trading_examples_available": selection.trading_available,
        "reasoning_examples_available": selection.reasoning_available,
        "math_examples_available": selection.math_available,
        "replay_examples": selection.replay_examples,
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
        "lineage_tokenizer_fingerprint": lineage_fingerprint,
        "data_starved_override": bool(args.allow_data_starved),
        **budget,
    }
    save_model_bundle(
        bundle_dir=bundle_dir,
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        training_stage=stage,
        corpus_fingerprint=curriculum_hash,
        metrics=metrics,
    )
    print(f"bundle={bundle_dir}")


if __name__ == "__main__":
    main()
