from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from corpus.streamer import CorpusStreamer
from src.language.canonical_contract import canonicalize_serialized, prompt_family
from src.language.compute_budget import benchmark_training_throughput, reference_token_target
from src.language.curriculum import CURRICULUM_STAGES, select_curriculum
from src.language.data_pipeline import build_tokenizer_training_sample, load_all_training_text
from src.language.exam import EpochExamResult, build_exam_prompt, exam_questions, run_epoch_exam, save_epoch_exam
from src.language.loss_objective import LOSS_OBJECTIVE_VERSION
from src.language.model_bundle import load_model_bundle, save_model_bundle
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.streaming_sources import (
    HFSourceSpec,
    build_training_stream,
    hf_source_from_spec,
    load_hf_source_config,
    sample_training_stream,
    specs_fingerprint,
    stage_specs,
)
from src.language.tokenizer import BPETokenizer, TOKENIZER_ALGORITHM_VERSION
from src.language.training_pipeline import (
    PackedSequenceDataset,
    build_example_sequences,
    corpus_fingerprint,
    run_training_preflight,
    split_by_prompt_family,
    split_fingerprint,
)
from src.language.training_profiles import (
    DEFAULT_LOCAL_REPLAY_WEIGHTS,
    DEFAULT_STAGE_TOKENS_PER_PARAMETER,
    PROFILES,
    profile as load_profile,
)

TRAINER_VERSION = 6
TRAINING_STATE_VERSION = 6
PREFLIGHT_MANIFEST_VERSION = 6
SEED = 42


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _stable_unique(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in texts:
        text = canonicalize_serialized(raw)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _normalize_stage(value: str) -> str:
    stage = value.strip().casefold()
    if stage == "general_language":
        stage = "foundation"
    if stage not in CURRICULUM_STAGES:
        raise ValueError(f"unsupported_training_stage:{value}")
    return stage


def _exam_holdout_texts(stage: str) -> list[str]:
    return [build_exam_prompt(question.prompt) for question in exam_questions(stage)]


def _remove_exam_families(texts: list[str], stage: str) -> tuple[list[str], int]:
    held = {prompt_family(text) for text in _exam_holdout_texts(stage)}
    result = [text for text in texts if prompt_family(text) not in held]
    return result, len(texts) - len(result)


def _model_config(cfg: dict, tokenizer: BPETokenizer) -> dict:
    return {
        "vocab_size": tokenizer.vocab_size,
        "d_model": int(cfg["d_model"]),
        "n_layers": int(cfg["n_layers"]),
        "n_heads": int(cfg["n_heads"]),
        "ffn_dim": int(cfg["ffn_dim"]),
        "max_seq_len": int(cfg["seq_len"]),
        "dropout": float(cfg["dropout"]),
    }


def _new_optimizer(model: VistaReasoningGPT, cfg: dict) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(cfg["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(cfg["lr"]),
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def _set_token_scheduled_lr(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    min_lr: float,
    cumulative_tokens: int,
    target_tokens: int,
    warmup_fraction: float,
) -> float:
    progress = min(1.0, cumulative_tokens / max(target_tokens, 1))
    if warmup_fraction > 0 and progress < warmup_fraction:
        warmup_progress = progress / warmup_fraction
        lr = min_lr + (base_lr - min_lr) * warmup_progress
    else:
        denominator = max(1e-9, 1.0 - warmup_fraction)
        decay_progress = min(
            1.0,
            max(0.0, (progress - warmup_fraction) / denominator),
        )
        lr = min_lr + 0.5 * (base_lr - min_lr) * (
            1.0 + math.cos(math.pi * decay_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def _validation_loader(dataset: PackedSequenceDataset, *, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )


def _finite_loader(dataset: PackedSequenceDataset, *, batch_size: int, epoch: int) -> DataLoader:
    generator = torch.Generator().manual_seed(SEED + epoch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def _stream_loader(
    *,
    specs: tuple[HFSourceSpec, ...],
    stage: str,
    epoch: int,
    local_replay: list[str],
    excluded_texts: list[str],
    tokenizer: BPETokenizer,
    cfg: dict,
    local_replay_weight: float,
) -> DataLoader:
    stream = build_training_stream(
        specs=specs,
        stage=stage,
        seed=SEED + epoch * 100_003,
        local_replay=local_replay,
        local_weight=local_replay_weight,
        excluded_texts=excluded_texts,
        repeat=True,
    )
    dataset = CorpusStreamer(
        stream,
        tokenizer,
        seq_len=int(cfg["seq_len"]),
        seed=SEED + epoch,
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        num_workers=0,
        drop_last=False,
    )


def _validation_loss(
    model: VistaReasoningGPT,
    loader: DataLoader,
    *,
    pad_id: int,
) -> tuple[float, int]:
    model.eval()
    total_loss = 0.0
    tokens = 0
    with torch.no_grad():
        for x, y in loader:
            valid = int((y != pad_id).sum().item())
            if valid <= 0:
                continue
            _, loss = model(x, targets=y, pad_id=pad_id)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("non_finite_validation_loss")
            total_loss += float(loss.item()) * valid
            tokens += valid
    if tokens <= 0:
        raise RuntimeError("validation_has_no_prediction_tokens")
    return total_loss / tokens, tokens


def _load_or_sample_hf(
    *,
    specs: tuple[HFSourceSpec, ...],
    stage: str,
    limit: int,
    sample_path: Path,
    metadata_path: Path,
) -> tuple[list[str], str]:
    fingerprint = specs_fingerprint(specs)
    expected = {
        "config_fingerprint": fingerprint,
        "stage": stage,
        "limit": int(limit),
    }
    if sample_path.exists() or metadata_path.exists():
        if not sample_path.exists() or not metadata_path.exists():
            raise RuntimeError("incomplete_hf_preflight_sample_cache")
        if json.loads(metadata_path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError(
                "hf_preflight_sample_contract_mismatch:use_a_clean_bundle_directory"
            )
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        if not isinstance(sample, list) or not sample:
            raise RuntimeError("cached_hf_preflight_sample_invalid")
        return [canonicalize_serialized(value) for value in sample], fingerprint

    sample = sample_training_stream(
        specs=specs,
        stage=stage,
        limit=limit,
        seed=SEED,
    )
    _atomic_json_save(sample, sample_path)
    _atomic_json_save(expected, metadata_path)
    return sample, fingerprint


def _audit_hf_sources(
    specs: tuple[HFSourceSpec, ...],
    *,
    stage: str,
    rows_per_source: int,
    min_serialization_rate: float,
) -> list[dict]:
    reports: list[dict] = []
    failures: list[str] = []
    for index, spec in enumerate(stage_specs(specs, stage)):
        source = hf_source_from_spec(spec, seed=SEED + index * 997)
        report = source.audit(max_rows=rows_per_source)
        payload = asdict(report)
        payload["serialization_rate"] = report.serialization_rate
        reports.append(payload)
        print(
            f"[SourceAudit] {spec.path}@{spec.revision} rows={report.rows_scanned:,} "
            f"serialized={report.rows_serialized:,} rate={report.serialization_rate:.1%} "
            f"chars(mean/min/max)={report.mean_serialized_chars:.0f}/"
            f"{report.min_serialized_chars}/{report.max_serialized_chars}"
        )
        if report.rows_scanned <= 0:
            failures.append(f"{spec.path}:empty")
        elif report.serialization_rate < min_serialization_rate:
            failures.append(
                f"{spec.path}:serialization_rate={report.serialization_rate:.3f}"
            )
    if failures:
        raise RuntimeError("hf_source_audit_failed:" + ";".join(failures))
    return reports


def _tokenizer_efficiency(texts: list[str], tokenizer: BPETokenizer) -> dict:
    sample = texts[: min(512, len(texts))]
    lengths: list[int] = []
    chars = 0
    for text in sample:
        ids = tokenizer.encode(text, add_bos=False, add_eos=False)
        lengths.append(len(ids))
        chars += len(text)
    if not lengths:
        raise RuntimeError("tokenizer_efficiency_sample_empty")
    ordered = sorted(lengths)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    total_tokens = sum(lengths)
    return {
        "sample_examples": len(sample),
        "sample_chars": chars,
        "sample_tokens": total_tokens,
        "chars_per_token": chars / max(total_tokens, 1),
        "mean_tokens_per_example": total_tokens / len(sample),
        "p95_tokens_per_example": ordered[p95_index],
    }


def _data_mix_report(local_texts: list[str], hf_sample: list[str]) -> dict:
    def describe(values: list[str]) -> dict:
        chats = sum("<assistant>" in text for text in values)
        documents = len(values) - chats
        return {
            "examples": len(values),
            "chat_examples": chats,
            "document_examples": documents,
            "characters": sum(len(text) for text in values),
        }
    return {
        "local": describe(local_texts),
        "hf_preflight": describe(hf_sample),
    }


def _prepare_lineage(
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


def _profile_contract(cfg: dict) -> dict:
    keys = (
        "vocab_size",
        "d_model",
        "n_heads",
        "n_layers",
        "ffn_dim",
        "seq_len",
        "batch_size",
        "lr",
        "dropout",
        "tokenizer_chars",
        "early_stop_patience",
        "early_stop_min_delta",
        "early_stop_min_tokens_per_parameter",
        "exam_max_new_tokens",
        "checkpoint_every_steps",
    )
    return {key: cfg[key] for key in keys}


def _static_contract(
    *,
    profile_name: str,
    stage: str,
    cfg: dict,
    source_hash: str,
    curriculum_hash: str,
    split_hash: str,
    local_replay_hash: str,
    lineage: str | None,
    hf_fingerprint: str | None,
) -> dict:
    return {
        "trainer_version": TRAINER_VERSION,
        "preflight_manifest_version": PREFLIGHT_MANIFEST_VERSION,
        "profile": profile_name,
        "training_stage": stage,
        "profile_contract": _profile_contract(cfg),
        "tokenizer_algorithm_version": TOKENIZER_ALGORITHM_VERSION,
        "loss_objective_version": LOSS_OBJECTIVE_VERSION,
        "source_corpus_fingerprint": source_hash,
        "curriculum_fingerprint": curriculum_hash,
        "split_fingerprint": split_hash,
        "local_replay_fingerprint": local_replay_hash,
        "lineage_tokenizer_fingerprint": lineage,
        "hf_sources_fingerprint": hf_fingerprint,
    }


def _validate_contract(saved: dict, expected: dict, *, prefix: str) -> None:
    for key, expected_value in expected.items():
        if saved.get(key) != expected_value:
            raise RuntimeError(f"{prefix}_contract_mismatch:{key}")


def _prepare_tokenizer(
    *,
    path: Path,
    manifest_path: Path,
    contract: dict,
    train_texts: list[str],
    cfg: dict,
    lineage_tokenizer: BPETokenizer | None,
) -> tuple[BPETokenizer, bool]:
    if path.exists():
        if not manifest_path.exists():
            raise RuntimeError("unverified_preflight_tokenizer_exists")
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_contract(saved, contract, prefix="preflight")
        tokenizer = BPETokenizer.load(path)
        if saved.get("tokenizer_fingerprint") != tokenizer.fingerprint():
            raise RuntimeError("preflight_tokenizer_fingerprint_mismatch")
        return tokenizer, True
    if manifest_path.exists():
        raise RuntimeError("preflight_manifest_exists_without_tokenizer")

    if lineage_tokenizer is not None:
        lineage_tokenizer.save(path)
        return lineage_tokenizer, False

    sample = build_tokenizer_training_sample(
        train_texts,
        max_chars=int(cfg["tokenizer_chars"]),
        seed=SEED,
    )
    tokenizer = BPETokenizer()
    tokenizer.train(sample, vocab_size=int(cfg["vocab_size"]))
    tokenizer.save(path)
    return tokenizer, False


def _checkpoint_contract(
    *,
    static: dict,
    tokenizer: BPETokenizer,
    model_config: dict,
    target_tokens: int,
    target_tokens_per_parameter: float,
    steps_per_exam_epoch: int,
    warmup_fraction: float,
    min_lr: float,
    local_replay_weight: float,
) -> dict:
    return {
        "training_state_version": TRAINING_STATE_VERSION,
        **static,
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "model_config": model_config,
        "target_prediction_tokens": int(target_tokens),
        "target_tokens_per_parameter": float(target_tokens_per_parameter),
        "steps_per_exam_epoch": int(steps_per_exam_epoch),
        "warmup_fraction": float(warmup_fraction),
        "min_lr": float(min_lr),
        "local_replay_weight": float(local_replay_weight),
    }


def _run_exam(
    model: VistaReasoningGPT,
    tokenizer: BPETokenizer,
    *,
    epoch: int,
    stage: str,
    train_loss: float | None,
    val_loss: float | None,
    exams_dir: Path,
    previous: EpochExamResult | None,
    max_new_tokens: int,
    prefix: str | None = None,
) -> EpochExamResult:
    started = time.time()
    result = run_epoch_exam(
        model=model,
        tokenizer=tokenizer,
        epoch=epoch,
        training_stage=stage,
        train_loss=train_loss,
        validation_loss=val_loss,
        max_new_tokens=max_new_tokens,
    )
    text_path, _ = save_epoch_exam(
        result=result,
        exams_dir=exams_dir,
        previous=previous,
        prefix=prefix,
    )
    print(
        f"[Exam] epoch={epoch} correct={result.correct_questions}/{result.total_questions} "
        f"score={result.correctness_percent:.1f}% quality={result.mean_quality_percent:.1f}% "
        f"gibberish={result.gibberish_answers}/{result.total_questions} "
        f"signal={result.training_signal} seconds={time.time() - started:.1f} file={text_path}"
    )
    return result


def _save_state(
    path: Path,
    *,
    contract: dict,
    epoch_index: int,
    step_in_epoch: int,
    epoch_loss_sum: float,
    epoch_tokens: int,
    cumulative_tokens: int,
    optimizer_steps: int,
    best_val: float,
    best_epoch: int,
    stale_epochs: int,
    model: VistaReasoningGPT,
    optimizer: torch.optim.Optimizer,
) -> None:
    _atomic_torch_save(
        {
            **contract,
            "epoch_index": int(epoch_index),
            "step_in_epoch": int(step_in_epoch),
            "epoch_loss_sum": float(epoch_loss_sum),
            "epoch_tokens": int(epoch_tokens),
            "cumulative_prediction_tokens": int(cumulative_tokens),
            "optimizer_steps": int(optimizer_steps),
            "best_validation_loss": float(best_val),
            "best_epoch": int(best_epoch),
            "epochs_without_improvement": int(stale_epochs),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "python_random_state": random.getstate(),
        },
        path,
    )


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        try:
            return next(iterator), iterator
        except StopIteration as exc:
            raise RuntimeError("training_loader_produced_no_batches") from exc


def _train_exam_epoch(
    *,
    model: VistaReasoningGPT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pad_id: int,
    grad_clip: float,
    steps_per_epoch: int,
    resume_step: int,
    loss_sum: float,
    epoch_tokens: int,
    cumulative_tokens: int,
    optimizer_steps: int,
    target_tokens: int,
    base_lr: float,
    min_lr: float,
    warmup_fraction: float,
    checkpoint_every: int,
    checkpoint_callback,
    session_deadline: float | None,
) -> tuple[float, int, int, int, int, bool, bool]:
    model.train()
    iterator = iter(loader)
    for _ in range(resume_step):
        _, iterator = _next_batch(iterator, loader)

    completed = resume_step
    session_expired = False
    target_reached = cumulative_tokens >= target_tokens

    while completed < steps_per_epoch and not target_reached:
        if session_deadline is not None and time.monotonic() >= session_deadline:
            session_expired = True
            break

        (x, y), iterator = _next_batch(iterator, loader)
        valid = int((y != pad_id).sum().item())
        completed += 1
        if valid <= 0:
            continue

        _set_token_scheduled_lr(
            optimizer,
            base_lr=base_lr,
            min_lr=min_lr,
            cumulative_tokens=cumulative_tokens,
            target_tokens=target_tokens,
            warmup_fraction=warmup_fraction,
        )
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y, pad_id=pad_id)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("non_finite_training_loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_sum += float(loss.item()) * valid
        epoch_tokens += valid
        cumulative_tokens += valid
        optimizer_steps += 1
        target_reached = cumulative_tokens >= target_tokens

        if completed % checkpoint_every == 0 and completed < steps_per_epoch:
            checkpoint_callback(
                completed,
                loss_sum,
                epoch_tokens,
                cumulative_tokens,
                optimizer_steps,
            )

    if epoch_tokens <= 0:
        raise RuntimeError("exam_epoch_has_no_prediction_tokens")

    return (
        loss_sum / epoch_tokens,
        epoch_tokens,
        cumulative_tokens,
        optimizer_steps,
        completed,
        session_expired,
        target_reached,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="2m")
    parser.add_argument("--data-root", default="data/data/trainingdata")
    parser.add_argument("--bundle-dir", default=None)
    parser.add_argument("--training-stage", default="foundation")
    parser.add_argument("--init-bundle", default=None)
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--hf-sample-examples", type=int, default=None)
    parser.add_argument("--local-replay-weight", type=float, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=None)
    parser.add_argument("--threads", type=int, default=max(1, min(4, os.cpu_count() or 2)))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--session-hours", type=float, default=4.0)
    parser.add_argument("--exam-interval-minutes", type=float, default=20.0)
    parser.add_argument("--target-tokens-per-parameter", type=float, default=None)
    parser.add_argument("--compute-probe-steps", type=int, default=3)
    parser.add_argument("--min-session-tokens-per-parameter", type=float, default=1.0)
    parser.add_argument("--source-audit-rows", type=int, default=500)
    parser.add_argument("--min-source-serialization-rate", type=float, default=0.80)
    parser.add_argument("--warmup-fraction", type=float, default=0.02)
    args = parser.parse_args()

    stage = _normalize_stage(args.training_stage)
    cfg = load_profile(args.profile)
    cfg.update(
        lr_min=1e-5,
        weight_decay=0.01,
        grad_clip=1.0,
        val_split=0.05,
    )

    checkpoint_every = args.checkpoint_every_steps or int(cfg["checkpoint_every_steps"])
    target_tpp = (
        float(args.target_tokens_per_parameter)
        if args.target_tokens_per_parameter is not None
        else float(DEFAULT_STAGE_TOKENS_PER_PARAMETER[stage])
    )
    replay_weight = (
        float(args.local_replay_weight)
        if args.local_replay_weight is not None
        else float(DEFAULT_LOCAL_REPLAY_WEIGHTS[stage])
    )

    if checkpoint_every <= 0:
        raise ValueError("checkpoint interval must be positive")
    if replay_weight < 0:
        raise ValueError("local replay weight must be >= 0")
    if args.session_hours <= 0 or args.exam_interval_minutes <= 0:
        raise ValueError("session-hours and exam-interval-minutes must be positive")
    if target_tpp <= 0 or args.compute_probe_steps <= 0:
        raise ValueError("target token ratio and compute probe steps must be positive")
    if not 0.0 <= args.warmup_fraction < 0.25:
        raise ValueError("warmup-fraction must be in [0, 0.25)")

    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    bundle_dir = Path(
        args.bundle_dir or f"data/models/trading_language/{args.profile}/{stage}"
    )
    work_dir = bundle_dir / ".training"
    tokenizer_path = work_dir / "tokenizer.json"
    preflight_path = work_dir / "preflight.json"
    compute_path = work_dir / "compute_probe.json"
    state_path = work_dir / "training_state.pt"
    best_model_path = work_dir / "best_model.pt"
    exams_dir = bundle_dir / "exams"
    hf_sample_path = work_dir / "hf_preflight_sample.json"
    hf_sample_meta = work_dir / "hf_preflight_sample_meta.json"

    local_raw = load_all_training_text(
        Path(args.data_root),
        cfg["max_examples"],
        False,
        SEED,
    )
    local_texts = _stable_unique(local_raw)

    hf_specs: tuple[HFSourceSpec, ...] = ()
    hf_sample: list[str] = []
    hf_fingerprint: str | None = None
    source_audit: list[dict] = []
    if args.hf_config:
        hf_specs = load_hf_source_config(args.hf_config)
        source_audit = _audit_hf_sources(
            hf_specs,
            stage=stage,
            rows_per_source=args.source_audit_rows,
            min_serialization_rate=args.min_source_serialization_rate,
        )
        sample_limit = args.hf_sample_examples or int(cfg["hf_preflight_sample_examples"])
        hf_sample, hf_fingerprint = _load_or_sample_hf(
            specs=hf_specs,
            stage=stage,
            limit=sample_limit,
            sample_path=hf_sample_path,
            metadata_path=hf_sample_meta,
        )
        print(
            f"[HF] sources={len(stage_specs(hf_specs, stage))} "
            f"accepted_preflight_sample={len(hf_sample):,} "
            f"fingerprint={hf_fingerprint[:12]}"
        )

    all_texts = _stable_unique(local_texts + hf_sample)
    all_texts, exam_removed = _remove_exam_families(all_texts, stage)
    print(
        f"[ExamHoldout] families={len(exam_questions(stage))} "
        f"removed_from_preflight={exam_removed}"
    )
    if not all_texts:
        raise RuntimeError("no_training_data_after_exam_holdout")

    selection = select_curriculum(all_texts, stage=stage, seed=SEED)
    texts = selection.texts
    print(
        f"[Curriculum] stage={stage} selected={len(texts):,}/{selection.total_available:,} "
        f"trading={selection.trading_available:,} reasoning={selection.reasoning_available:,} "
        f"math={selection.math_available:,} replay={selection.replay_examples:,}"
    )

    train_texts, val_texts = split_by_prompt_family(
        texts,
        val_fraction=float(cfg["val_split"]),
        seed=SEED,
    )
    train_set = set(train_texts)
    local_replay = [text for text in local_texts if text in train_set]
    if replay_weight == 0.0:
        local_replay = []
    print(
        f"[Replay] local_only={len(local_replay):,} "
        f"weight={replay_weight:.3f} hf_preflight_replayed=0"
    )

    lineage_model, lineage_tokenizer, lineage_fp = _prepare_lineage(
        stage,
        Path(args.init_bundle) if args.init_bundle else None,
        cfg,
    )

    source_hash = corpus_fingerprint(all_texts)
    curriculum_hash = corpus_fingerprint(texts)
    split_hash = split_fingerprint(train_texts, val_texts)
    local_replay_hash = corpus_fingerprint(local_replay)
    print(
        f"[Split] train={len(train_texts):,} val={len(val_texts):,} "
        f"family_isolated=true split={split_hash[:12]}"
    )

    static = _static_contract(
        profile_name=args.profile,
        stage=stage,
        cfg=cfg,
        source_hash=source_hash,
        curriculum_hash=curriculum_hash,
        split_hash=split_hash,
        local_replay_hash=local_replay_hash,
        lineage=lineage_fp,
        hf_fingerprint=hf_fingerprint,
    )

    tokenizer, reused = _prepare_tokenizer(
        path=tokenizer_path,
        manifest_path=preflight_path,
        contract=static,
        train_texts=train_texts,
        cfg=cfg,
        lineage_tokenizer=lineage_tokenizer,
    )
    print(
        f"[Tokenizer] reuse={str(reused).lower()} "
        f"fingerprint={tokenizer.fingerprint()}"
    )

    train_sequences = build_example_sequences(
        train_texts,
        tokenizer,
        seq_len=int(cfg["seq_len"]),
    )
    val_sequences = build_example_sequences(
        val_texts,
        tokenizer,
        seq_len=int(cfg["seq_len"]),
    )
    report = run_training_preflight(
        tokenizer=tokenizer,
        train_texts=train_texts,
        val_texts=val_texts,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        seq_len=int(cfg["seq_len"]),
    )

    tokenizer_stats = _tokenizer_efficiency(train_texts, tokenizer)
    data_mix = _data_mix_report(local_texts, hf_sample)
    print(
        f"[TokenizerEfficiency] chars/token={tokenizer_stats['chars_per_token']:.2f} "
        f"mean_tokens/example={tokenizer_stats['mean_tokens_per_example']:.1f} "
        f"p95={tokenizer_stats['p95_tokens_per_example']}"
    )
    print(
        f"[DataMix] local={data_mix['local']['examples']:,} "
        f"(chat={data_mix['local']['chat_examples']:,},doc={data_mix['local']['document_examples']:,}) "
        f"hf_sample={data_mix['hf_preflight']['examples']:,} "
        f"(chat={data_mix['hf_preflight']['chat_examples']:,},doc={data_mix['hf_preflight']['document_examples']:,})"
    )

    model_config = _model_config(cfg, tokenizer)
    probe = benchmark_training_throughput(
        model_config=model_config,
        tokenizer=tokenizer,
        sequences=train_sequences,
        batch_size=int(cfg["batch_size"]),
        steps=args.compute_probe_steps,
        wall_clock_hours=args.session_hours,
        reference_tokens_per_parameter=target_tpp,
    )
    _atomic_json_save(probe.to_dict(), compute_path)

    step_seconds = probe.elapsed_seconds / max(probe.benchmark_steps, 1)
    steps_per_exam_epoch = max(
        1,
        int(round(args.exam_interval_minutes * 60.0 / max(step_seconds, 1e-9))),
    )
    steps_per_exam_epoch = min(20_000, steps_per_exam_epoch)
    target_tokens = reference_token_target(
        probe.parameter_count,
        tokens_per_parameter=target_tpp,
    )
    expected_exam_minutes = steps_per_exam_epoch * step_seconds / 60.0

    preflight_payload = {
        **static,
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "actual_vocab_size": tokenizer.vocab_size,
        "roundtrip_cases": report.roundtrip_cases,
        "overfit_initial_loss": report.overfit_initial_loss,
        "overfit_final_loss": report.overfit_final_loss,
        "train_prediction_tokens": report.train_prediction_tokens,
        "validation_prediction_tokens": report.validation_prediction_tokens,
        "source_audit": source_audit,
        "tokenizer_efficiency": tokenizer_stats,
        "data_mix": data_mix,
        "local_replay_examples": len(local_replay),
        "local_replay_weight": replay_weight,
        "compute_probe": probe.to_dict(),
        "target_tokens_per_parameter": target_tpp,
        "target_prediction_tokens": target_tokens,
        "steps_per_exam_epoch": steps_per_exam_epoch,
        "expected_exam_interval_minutes": expected_exam_minutes,
    }
    _atomic_json_save(preflight_payload, preflight_path)

    print(
        f"[Preflight] PASS roundtrips={report.roundtrip_cases} "
        f"overfit={report.overfit_initial_loss:.4f}->{report.overfit_final_loss:.4f} "
        f"objective=v{LOSS_OBJECTIVE_VERSION}"
    )
    print(
        f"[Compute] params={probe.parameter_count:,} measured_supervised_tokens/s="
        f"{probe.useful_tokens_per_second:,.1f} projected_{args.session_hours:g}h="
        f"{probe.projected_useful_tokens:,} "
        f"({probe.projected_tokens_per_parameter:.2f} supervised_tokens/param)"
    )
    print(
        f"[Compute] target={target_tokens:,} ({target_tpp:.1f} supervised_tokens/param) "
        f"projected_hours_to_target={probe.projected_hours_to_reference_target:.2f} "
        f"exam_steps={steps_per_exam_epoch:,} "
        f"expected_exam_interval={expected_exam_minutes:.1f}m"
    )

    if (
        stage == "foundation"
        and args.profile != "smoke"
        and probe.projected_tokens_per_parameter < args.min_session_tokens_per_parameter
    ):
        raise RuntimeError(
            "profile_too_large_for_session_compute:choose_smaller_profile_or_longer_session:"
            f"projected_tpp={probe.projected_tokens_per_parameter:.3f}"
        )

    if args.preflight_only:
        print(
            "[Preflight] Full training intentionally not started. "
            "All verified artifacts are reusable."
        )
        return

    # Preflight uses independent tiny models and therefore consumes torch RNG.
    # Reseed here so changing probe steps can never change scratch initialization.
    torch.manual_seed(SEED)
    random.seed(SEED)

    model = (
        lineage_model
        if lineage_model is not None
        else VistaReasoningGPT(**model_config).to("cpu")
    )
    if model.max_seq_len != int(cfg["seq_len"]) or model.vocab_size != tokenizer.vocab_size:
        raise RuntimeError("model_runtime_contract_mismatch")

    finite_ds = None
    if not hf_specs:
        finite_ds = PackedSequenceDataset(
            train_sequences,
            int(cfg["seq_len"]),
            tokenizer.pad_id(),
        )
    val_ds = PackedSequenceDataset(
        val_sequences,
        int(cfg["seq_len"]),
        tokenizer.pad_id(),
    )
    val_loader = _validation_loader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
    )
    optimizer = _new_optimizer(model, cfg)

    contract = _checkpoint_contract(
        static=static,
        tokenizer=tokenizer,
        model_config=model_config,
        target_tokens=target_tokens,
        target_tokens_per_parameter=target_tpp,
        steps_per_exam_epoch=steps_per_exam_epoch,
        warmup_fraction=args.warmup_fraction,
        min_lr=float(cfg["lr_min"]),
        local_replay_weight=replay_weight,
    )

    epoch_index = 0
    resume_step = 0
    epoch_loss_sum = 0.0
    epoch_tokens = 0
    cumulative_tokens = 0
    optimizer_steps = 0
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0

    if args.resume:
        if not state_path.exists():
            raise RuntimeError("resume_training_state_missing")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        _validate_contract(state, contract, prefix="resume")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        epoch_index = int(state["epoch_index"])
        resume_step = int(state["step_in_epoch"])
        epoch_loss_sum = float(state.get("epoch_loss_sum", 0.0))
        epoch_tokens = int(state.get("epoch_tokens", 0))
        cumulative_tokens = int(state.get("cumulative_prediction_tokens", 0))
        optimizer_steps = int(state.get("optimizer_steps", 0))
        best_val = float(state["best_validation_loss"])
        best_epoch = int(state.get("best_epoch", 0))
        stale_epochs = int(state.get("epochs_without_improvement", 0))
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"])
        if "python_random_state" in state:
            random.setstate(state["python_random_state"])
        print(
            f"[Resume] exam_epoch={epoch_index + 1} "
            f"step={resume_step}/{steps_per_exam_epoch} "
            f"tokens={cumulative_tokens:,}/{target_tokens:,} best_val={best_val:.4f}"
        )
    elif state_path.exists():
        raise RuntimeError("training_state_already_exists:use_--resume_or_clean_bundle")

    previous_exam: EpochExamResult | None = None
    if epoch_index == 0 and resume_step == 0:
        previous_exam = _run_exam(
            model,
            tokenizer,
            epoch=0,
            stage=stage,
            train_loss=None,
            val_loss=None,
            exams_dir=exams_dir,
            previous=None,
            max_new_tokens=int(cfg["exam_max_new_tokens"]),
        )

    session_started_wall = time.time()
    session_deadline = time.monotonic() + args.session_hours * 3600.0
    completed_training = cumulative_tokens >= target_tokens
    stopped_for_session = False

    while not completed_training:
        excluded = list(val_texts) + _exam_holdout_texts(stage)
        if hf_specs:
            loader = _stream_loader(
                specs=hf_specs,
                stage=stage,
                epoch=epoch_index,
                local_replay=local_replay,
                excluded_texts=excluded,
                tokenizer=tokenizer,
                cfg=cfg,
                local_replay_weight=replay_weight,
            )
        else:
            if finite_ds is None:
                raise RuntimeError("finite_training_dataset_missing")
            loader = _finite_loader(
                finite_ds,
                batch_size=int(cfg["batch_size"]),
                epoch=epoch_index,
            )

        def checkpoint(
            completed_step: int,
            loss_sum: float,
            current_epoch_tokens: int,
            current_cumulative_tokens: int,
            current_optimizer_steps: int,
        ) -> None:
            _save_state(
                state_path,
                contract=contract,
                epoch_index=epoch_index,
                step_in_epoch=completed_step,
                epoch_loss_sum=loss_sum,
                epoch_tokens=current_epoch_tokens,
                cumulative_tokens=current_cumulative_tokens,
                optimizer_steps=current_optimizer_steps,
                best_val=best_val,
                best_epoch=best_epoch,
                stale_epochs=stale_epochs,
                model=model,
                optimizer=optimizer,
            )
            print(
                f"[Checkpoint] exam_epoch={epoch_index + 1} "
                f"step={completed_step}/{steps_per_exam_epoch} "
                f"cumulative_supervised_tokens={current_cumulative_tokens:,}"
            )

        epoch_started = time.time()
        (
            train_loss,
            epoch_tokens,
            cumulative_tokens,
            optimizer_steps,
            completed_steps,
            session_expired,
            target_reached,
        ) = _train_exam_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            pad_id=tokenizer.pad_id(),
            grad_clip=float(cfg["grad_clip"]),
            steps_per_epoch=steps_per_exam_epoch,
            resume_step=resume_step,
            loss_sum=epoch_loss_sum,
            epoch_tokens=epoch_tokens,
            cumulative_tokens=cumulative_tokens,
            optimizer_steps=optimizer_steps,
            target_tokens=target_tokens,
            base_lr=float(cfg["lr"]),
            min_lr=float(cfg["lr_min"]),
            warmup_fraction=args.warmup_fraction,
            checkpoint_every=checkpoint_every,
            checkpoint_callback=checkpoint,
            session_deadline=session_deadline,
        )

        if session_expired and not target_reached and completed_steps < steps_per_exam_epoch:
            _save_state(
                state_path,
                contract=contract,
                epoch_index=epoch_index,
                step_in_epoch=completed_steps,
                epoch_loss_sum=train_loss * epoch_tokens,
                epoch_tokens=epoch_tokens,
                cumulative_tokens=cumulative_tokens,
                optimizer_steps=optimizer_steps,
                best_val=best_val,
                best_epoch=best_epoch,
                stale_epochs=stale_epochs,
                model=model,
                optimizer=optimizer,
            )
            _run_exam(
                model,
                tokenizer,
                epoch=epoch_index,
                stage=stage,
                train_loss=train_loss,
                val_loss=None,
                exams_dir=exams_dir,
                previous=previous_exam,
                max_new_tokens=int(cfg["exam_max_new_tokens"]),
                prefix="session_end_exam",
            )
            print(
                f"[SessionStop] mid-epoch step={completed_steps}/{steps_per_exam_epoch} "
                f"tokens={cumulative_tokens:,}/{target_tokens:,}. Resume with --resume."
            )
            stopped_for_session = True
            break

        val_loss, val_tokens = _validation_loss(
            model,
            val_loader,
            pad_id=tokenizer.pad_id(),
        )
        improved = val_loss < best_val - float(cfg["early_stop_min_delta"])
        if improved:
            best_val = val_loss
            best_epoch = epoch_index + 1
            stale_epochs = 0
            _atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "validation_loss": best_val,
                    "epoch": best_epoch,
                    "cumulative_prediction_tokens": cumulative_tokens,
                },
                best_model_path,
            )
        else:
            stale_epochs += 1

        previous_exam = _run_exam(
            model,
            tokenizer,
            epoch=epoch_index + 1,
            stage=stage,
            train_loss=train_loss,
            val_loss=val_loss,
            exams_dir=exams_dir,
            previous=previous_exam,
            max_new_tokens=int(cfg["exam_max_new_tokens"]),
        )

        max_slots = max(
            1,
            completed_steps * int(cfg["batch_size"]) * int(cfg["seq_len"]),
        )
        utilization = epoch_tokens / max_slots
        current_lr = optimizer.param_groups[0]["lr"]
        current_tpp = cumulative_tokens / probe.parameter_count
        print(
            f"epoch={epoch_index + 1} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} best_val={best_val:.4f} "
            f"supervised_tokens={epoch_tokens:,} "
            f"cumulative_tokens={cumulative_tokens:,}/{target_tokens:,} "
            f"supervised_tokens_per_param={current_tpp:.3f} "
            f"supervision_density={utilization:.3f} lr={current_lr:.8f} "
            f"val_tokens={val_tokens:,} epoch_seconds={time.time() - epoch_started:.1f}"
        )

        epoch_index += 1
        resume_step = 0
        epoch_loss_sum = 0.0
        epoch_tokens = 0
        completed_training = target_reached or cumulative_tokens >= target_tokens

        _save_state(
            state_path,
            contract=contract,
            epoch_index=epoch_index,
            step_in_epoch=0,
            epoch_loss_sum=0.0,
            epoch_tokens=0,
            cumulative_tokens=cumulative_tokens,
            optimizer_steps=optimizer_steps,
            best_val=best_val,
            best_epoch=best_epoch,
            stale_epochs=stale_epochs,
            model=model,
            optimizer=optimizer,
        )

        min_early_stop_tpp = float(cfg["early_stop_min_tokens_per_parameter"])
        if (
            current_tpp >= min_early_stop_tpp
            and stale_epochs >= int(cfg["early_stop_patience"])
        ):
            print(
                f"[EarlyStop] no material validation improvement for {stale_epochs} "
                f"exam epochs after {current_tpp:.2f} supervised tokens/param."
            )
            completed_training = True
        elif time.monotonic() >= session_deadline and not completed_training:
            stopped_for_session = True
            print(
                f"[SessionStop] completed exam epoch {epoch_index}; "
                f"tokens={cumulative_tokens:,}/{target_tokens:,}. Resume with --resume."
            )
            break

    if stopped_for_session and cumulative_tokens < target_tokens:
        summary = {
            "trainer_version": TRAINER_VERSION,
            "training_stage": stage,
            "profile": args.profile,
            "session_hours": (time.time() - session_started_wall) / 3600.0,
            "cumulative_prediction_tokens": cumulative_tokens,
            "target_prediction_tokens": target_tokens,
            "progress_percent": 100.0 * cumulative_tokens / target_tokens,
            "supervised_tokens_per_parameter": cumulative_tokens / probe.parameter_count,
            "optimizer_steps": optimizer_steps,
        }
        _atomic_json_save(summary, work_dir / "session_summary.json")
        print(
            f"[SessionSummary] hours={summary['session_hours']:.2f} "
            f"tokens={cumulative_tokens:,}/{target_tokens:,} "
            f"progress={summary['progress_percent']:.2f}% "
            f"supervised_tpp={summary['supervised_tokens_per_parameter']:.2f}"
        )
        return

    if not best_model_path.exists():
        raise RuntimeError("best_model_checkpoint_missing")
    best = torch.load(best_model_path, map_location="cpu", weights_only=False)
    if best.get("model_config") != model_config:
        raise RuntimeError("best_model_config_mismatch")
    model.load_state_dict(best["model_state_dict"])
    model.eval()

    best_exam = _run_exam(
        model,
        tokenizer,
        epoch=int(best.get("epoch", best_epoch)),
        stage=stage,
        train_loss=None,
        val_loss=best_val,
        exams_dir=exams_dir,
        previous=None,
        max_new_tokens=int(cfg["exam_max_new_tokens"]),
        prefix="best_model_exam",
    )

    metrics = {
        "trainer_version": TRAINER_VERSION,
        "loss_objective_version": LOSS_OBJECTIVE_VERSION,
        "best_validation_loss": best_val,
        "best_epoch": best_epoch,
        "perplexity": math.exp(min(best_val, 20)),
        "training_seconds_this_session": time.time() - session_started_wall,
        "profile": args.profile,
        "training_stage": stage,
        "parameter_count": probe.parameter_count,
        "hf_sources_fingerprint": hf_fingerprint,
        "family_isolated_validation": True,
        "validation_excluded_from_stream": True,
        "exam_prompt_families_held_out": True,
        "example_aware_stream_packing": True,
        "assistant_supervised_chat_loss": True,
        "packed_bos_transition_masking": True,
        "near_duplicate_filtering": True,
        "source_schema_audit": bool(hf_specs),
        "step_resumable_training": True,
        "rng_resumable_training": True,
        "token_budgeted_training": True,
        "token_scheduled_learning_rate": True,
        "local_replay_examples": len(local_replay),
        "local_replay_weight": replay_weight,
        "checkpoint_every_steps": checkpoint_every,
        "target_tokens_per_parameter": target_tpp,
        "target_prediction_tokens": target_tokens,
        "cumulative_prediction_tokens": cumulative_tokens,
        "optimizer_steps": optimizer_steps,
        "tokenizer_efficiency": tokenizer_stats,
        "data_mix": data_mix,
        "compute_probe": probe.to_dict(),
        "best_model_exam": {
            "correctness_percent": best_exam.correctness_percent,
            "mean_quality_percent": best_exam.mean_quality_percent,
            "gibberish_answers": best_exam.gibberish_answers,
            "training_signal": best_exam.training_signal,
        },
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
