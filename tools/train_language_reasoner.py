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
from src.language.data_pipeline import build_tokenizer_training_sample
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
    DEFAULT_STAGE_TOKENS_PER_PARAMETER,
    PROFILES,
    profile as load_profile,
)

TRAINER_VERSION = 8
TRAINING_STATE_VERSION = 8
PREFLIGHT_MANIFEST_VERSION = 8
SEED = 42


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _atomic_json_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


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
    kept = [text for text in texts if prompt_family(text) not in held]
    return kept, len(texts) - len(kept)


def _model_config(cfg: dict, tokenizer: BPETokenizer) -> dict:
    return {
        "vocab_size": tokenizer.vocab_size,
        "d_model": int(cfg["d_model"]),
        "n_layers": int(cfg["n_layers"]),
        "n_heads": int(cfg["n_heads"]),
        "n_kv_heads": int(cfg["n_kv_heads"]),
        "ffn_dim": int(cfg["ffn_dim"]),
        "max_seq_len": int(cfg["seq_len"]),
        "dropout": float(cfg["dropout"]),
        "rope_theta": float(cfg["rope_theta"]),
        "ffn_type": str(cfg["ffn_type"]),
        "num_experts": int(cfg["num_experts"]),
        "experts_per_token": int(cfg["experts_per_token"]),
        "moe_ffn_dim": int(cfg["moe_ffn_dim"]),
        "shared_expert_ffn_dim": int(cfg["shared_expert_ffn_dim"]),
        "router_aux_loss_coef": float(cfg["router_aux_loss_coef"]),
        "router_jitter": float(cfg["router_jitter"]),
    }


def _profile_contract(cfg: dict) -> dict:
    keys = (
        "vocab_size", "d_model", "n_heads", "n_kv_heads", "n_layers",
        "ffn_dim", "ffn_type", "num_experts", "experts_per_token",
        "moe_ffn_dim", "shared_expert_ffn_dim", "router_aux_loss_coef",
        "router_jitter", "rope_theta", "seq_len", "batch_size", "lr",
        "dropout", "tokenizer_chars", "early_stop_patience",
        "early_stop_min_delta", "early_stop_min_tokens_per_parameter",
        "exam_max_new_tokens", "checkpoint_every_steps",
    )
    return {key: cfg[key] for key in keys}


def _new_optimizer(model: VistaReasoningGPT, cfg: dict) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(cfg["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(cfg["lr"]), betas=(0.9, 0.95), eps=1e-8,
    )


def _set_token_scheduled_lr(
    optimizer: torch.optim.Optimizer,
    *, base_lr: float, min_lr: float, cumulative_tokens: int,
    target_tokens: int, warmup_fraction: float,
) -> float:
    progress = min(1.0, cumulative_tokens / max(target_tokens, 1))
    if warmup_fraction > 0 and progress < warmup_fraction:
        lr = min_lr + (base_lr - min_lr) * (progress / warmup_fraction)
    else:
        span = max(1e-9, 1.0 - warmup_fraction)
        decay = min(1.0, max(0.0, (progress - warmup_fraction) / span))
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * decay))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def _validation_loss(model: VistaReasoningGPT, loader: DataLoader, *, pad_id: int) -> tuple[float, int]:
    model.eval()
    weighted = 0.0
    tokens = 0
    with torch.no_grad():
        for x, y in loader:
            valid = int((y != pad_id).sum().item())
            if valid <= 0:
                continue
            _, loss = model(x, targets=y, pad_id=pad_id)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("non_finite_validation_loss")
            weighted += float(loss.item()) * valid
            tokens += valid
    if tokens <= 0:
        raise RuntimeError("validation_has_no_prediction_tokens")
    return weighted / tokens, tokens


def _audit_hf_sources(specs: tuple[HFSourceSpec, ...], *, stage: str, rows: int, min_rate: float) -> list[dict]:
    reports: list[dict] = []
    for index, spec in enumerate(stage_specs(specs, stage)):
        report = hf_source_from_spec(spec, seed=SEED + index * 997).audit(max_rows=rows)
        payload = asdict(report)
        payload["serialization_rate"] = report.serialization_rate
        reports.append(payload)
        print(
            f"[SourceAudit] {spec.path}@{spec.revision} rows={report.rows_scanned:,} "
            f"serialized={report.rows_serialized:,} rate={report.serialization_rate:.1%} "
            f"chars(mean/min/max)={report.mean_serialized_chars:.0f}/"
            f"{report.min_serialized_chars}/{report.max_serialized_chars}"
        )
        if report.rows_scanned <= 0 or report.serialization_rate < min_rate:
            raise RuntimeError(f"hf_source_audit_failed:{spec.path}")
    return reports


def _load_hf_sample(
    *, specs: tuple[HFSourceSpec, ...], stage: str, limit: int,
    sample_path: Path, metadata_path: Path,
) -> tuple[list[str], str]:
    fingerprint = specs_fingerprint(specs)
    expected = {"config_fingerprint": fingerprint, "stage": stage, "limit": int(limit)}
    if sample_path.exists() or metadata_path.exists():
        if not sample_path.exists() or not metadata_path.exists():
            raise RuntimeError("incomplete_hf_preflight_sample_cache")
        if json.loads(metadata_path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("hf_preflight_sample_contract_mismatch:use_clean_bundle")
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        return [canonicalize_serialized(value) for value in sample], fingerprint
    sample = sample_training_stream(specs=specs, stage=stage, limit=limit, seed=SEED)
    _atomic_json_save(sample, sample_path)
    _atomic_json_save(expected, metadata_path)
    return sample, fingerprint


def _tokenizer_efficiency(texts: list[str], tokenizer: BPETokenizer) -> dict:
    sample = texts[: min(512, len(texts))]
    lengths = [len(tokenizer.encode(text, add_bos=False, add_eos=False)) for text in sample]
    chars = sum(len(text) for text in sample)
    if not lengths:
        raise RuntimeError("tokenizer_efficiency_sample_empty")
    ordered = sorted(lengths)
    return {
        "sample_examples": len(sample),
        "sample_chars": chars,
        "sample_tokens": sum(lengths),
        "chars_per_token": chars / max(sum(lengths), 1),
        "mean_tokens_per_example": sum(lengths) / len(lengths),
        "p95_tokens_per_example": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
    }


def _run_exam(
    model: VistaReasoningGPT, tokenizer: BPETokenizer, *, epoch: int, stage: str,
    train_loss: float | None, val_loss: float | None, exams_dir: Path,
    previous: EpochExamResult | None, max_new_tokens: int, prefix: str | None = None,
) -> EpochExamResult:
    started = time.time()
    result = run_epoch_exam(
        model=model, tokenizer=tokenizer, epoch=epoch, training_stage=stage,
        train_loss=train_loss, validation_loss=val_loss, max_new_tokens=max_new_tokens,
    )
    text_path, _ = save_epoch_exam(result=result, exams_dir=exams_dir, previous=previous, prefix=prefix)
    print(
        f"[Exam] epoch={epoch} correct={result.correct_questions}/{result.total_questions} "
        f"score={result.correctness_percent:.1f}% quality={result.mean_quality_percent:.1f}% "
        f"gibberish={result.gibberish_answers}/{result.total_questions} "
        f"signal={result.training_signal} seconds={time.time()-started:.1f} file={text_path}"
    )
    return result


def _stream_loader(
    *, specs: tuple[HFSourceSpec, ...], stage: str, stream_generation: int,
    excluded: list[str], tokenizer: BPETokenizer, cfg: dict,
) -> DataLoader:
    stream = build_training_stream(
        specs=specs,
        stage=stage,
        seed=SEED + stream_generation * 100_003,
        local_replay=(),
        local_weight=0.0,
        excluded_texts=excluded,
        repeat=True,
    )
    return DataLoader(
        CorpusStreamer(
            stream,
            tokenizer,
            seq_len=int(cfg["seq_len"]),
            seed=SEED + stream_generation,
        ),
        batch_size=int(cfg["batch_size"]),
        num_workers=0,
        drop_last=False,
    )


def _save_state(
    path: Path, *, contract: dict, epoch: int, step: int, cumulative_tokens: int,
    optimizer_steps: int, best_val: float, stale: int, model: VistaReasoningGPT,
    optimizer: torch.optim.Optimizer,
) -> None:
    _atomic_torch_save(
        {
            **contract,
            "epoch": epoch,
            "step": step,
            "cumulative_prediction_tokens": cumulative_tokens,
            "optimizer_steps": optimizer_steps,
            "best_validation_loss": best_val,
            "epochs_without_improvement": stale,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "python_random_state": random.getstate(),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="2m")
    parser.add_argument("--bundle-dir", default=None)
    parser.add_argument("--training-stage", default="foundation")
    parser.add_argument("--init-bundle", default=None)
    parser.add_argument("--hf-config", required=True)
    parser.add_argument("--hf-sample-examples", type=int, default=None)
    parser.add_argument("--threads", type=int, default=max(1, min(4, os.cpu_count() or 2)))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--session-hours", type=float, default=4.0)
    parser.add_argument("--exam-interval-minutes", type=float, default=5.0)
    parser.add_argument("--target-tokens-per-parameter", type=float, default=None)
    parser.add_argument("--compute-probe-steps", type=int, default=3)
    parser.add_argument("--source-audit-rows", type=int, default=500)
    parser.add_argument("--min-source-serialization-rate", type=float, default=0.80)
    parser.add_argument("--warmup-fraction", type=float, default=0.02)
    args = parser.parse_args()

    stage = _normalize_stage(args.training_stage)
    cfg = load_profile(args.profile)
    cfg.update(lr_min=1e-5, weight_decay=0.01, grad_clip=1.0, val_split=0.05)
    target_tpp = float(
        args.target_tokens_per_parameter
        if args.target_tokens_per_parameter is not None
        else DEFAULT_STAGE_TOKENS_PER_PARAMETER[stage]
    )
    if args.session_hours <= 0 or args.exam_interval_minutes <= 0 or target_tpp <= 0:
        raise ValueError("training budgets must be positive")

    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    bundle = Path(args.bundle_dir or f"data/models/trading_language/{args.profile}/{stage}")
    work = bundle / ".training"
    exams_dir = bundle / "exams"
    tokenizer_path = work / "tokenizer.json"
    preflight_path = work / "preflight.json"
    state_path = work / "training_state.pt"
    best_path = work / "best_model.pt"
    hf_sample_path = work / "hf_preflight_sample.json"
    hf_sample_meta = work / "hf_preflight_sample_meta.json"

    hf_specs = load_hf_source_config(args.hf_config)
    source_audit = _audit_hf_sources(
        hf_specs,
        stage=stage,
        rows=args.source_audit_rows,
        min_rate=args.min_source_serialization_rate,
    )
    hf_sample, hf_fp = _load_hf_sample(
        specs=hf_specs,
        stage=stage,
        limit=args.hf_sample_examples or int(cfg["hf_preflight_sample_examples"]),
        sample_path=hf_sample_path,
        metadata_path=hf_sample_meta,
    )
    print(
        f"[HF] sources={len(stage_specs(hf_specs, stage))} "
        f"accepted_preflight_sample={len(hf_sample):,} fingerprint={hf_fp[:12]}"
    )

    all_texts, removed = _remove_exam_families(_stable_unique(hf_sample), stage)
    print(f"[ExamHoldout] families={len(exam_questions(stage))} removed_from_preflight={removed}")
    selected = select_curriculum(all_texts, stage=stage, seed=SEED)
    texts = selected.texts
    print(
        f"[Curriculum] stage={stage} selected={len(texts):,}/{selected.total_available:,} "
        f"trading={selected.trading_available:,} reasoning={selected.reasoning_available:,} "
        f"math={selected.math_available:,} replay=0"
    )
    train_texts, val_texts = split_by_prompt_family(
        texts,
        val_fraction=float(cfg["val_split"]),
        seed=SEED,
    )
    split_fp = split_fingerprint(train_texts, val_texts)
    print(
        f"[Split] train={len(train_texts):,} val={len(val_texts):,} "
        f"family_isolated=true split={split_fp[:12]}"
    )
    print("[Replay] local=disabled hf_preflight_replayed=0")

    lineage_model = None
    lineage_tokenizer = None
    if stage != "foundation":
        if not args.init_bundle:
            raise RuntimeError(f"{stage}_requires_--init-bundle")
        lineage_model, lineage_tokenizer, manifest = load_model_bundle(args.init_bundle)
        if manifest.model_config != _model_config(cfg, lineage_tokenizer):
            raise RuntimeError("init_bundle_model_config_mismatch")
    elif args.init_bundle:
        raise RuntimeError("foundation_stage_must_start_without_init_bundle")

    contract = {
        "trainer_version": TRAINER_VERSION,
        "training_state_version": TRAINING_STATE_VERSION,
        "preflight_manifest_version": PREFLIGHT_MANIFEST_VERSION,
        "profile": args.profile,
        "training_stage": stage,
        "profile_contract": _profile_contract(cfg),
        "tokenizer_algorithm_version": TOKENIZER_ALGORITHM_VERSION,
        "loss_objective_version": LOSS_OBJECTIVE_VERSION,
        "source_corpus_fingerprint": corpus_fingerprint(all_texts),
        "curriculum_fingerprint": corpus_fingerprint(texts),
        "split_fingerprint": split_fp,
        "hf_sources_fingerprint": hf_fp,
        "stream_only": True,
    }

    if tokenizer_path.exists():
        if not preflight_path.exists():
            raise RuntimeError("unverified_preflight_tokenizer_exists")
        saved = json.loads(preflight_path.read_text(encoding="utf-8"))
        for key, value in contract.items():
            if saved.get(key) != value:
                raise RuntimeError(f"preflight_contract_mismatch:{key}:use_clean_bundle")
        tokenizer = BPETokenizer.load(tokenizer_path)
        reused = True
    elif lineage_tokenizer is not None:
        tokenizer = lineage_tokenizer
        tokenizer.save(tokenizer_path)
        reused = False
    else:
        sample = build_tokenizer_training_sample(
            train_texts,
            max_chars=int(cfg["tokenizer_chars"]),
            seed=SEED,
        )
        tokenizer = BPETokenizer()
        tokenizer.train(sample, vocab_size=int(cfg["vocab_size"]))
        tokenizer.save(tokenizer_path)
        reused = False
    print(f"[Tokenizer] reuse={str(reused).lower()} fingerprint={tokenizer.fingerprint()}")

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

    torch.manual_seed(SEED)
    random.seed(SEED)
    probe_model = VistaReasoningGPT(**model_config)
    total_params = probe_model.get_num_params()
    active_params = probe_model.get_active_params_per_token()
    activation_ratio = active_params / max(total_params, 1)
    target_tokens = reference_token_target(total_params, tokens_per_parameter=target_tpp)
    step_seconds = probe.elapsed_seconds / max(probe.benchmark_steps, 1)
    exam_steps = max(
        1,
        min(
            20_000,
            int(round(args.exam_interval_minutes * 60 / max(step_seconds, 1e-9))),
        ),
    )

    preflight = {
        **contract,
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "model_config": model_config,
        "total_parameters": total_params,
        "active_parameters_per_token": active_params,
        "activation_ratio": activation_ratio,
        "target_tokens_per_total_parameter": target_tpp,
        "target_prediction_tokens": target_tokens,
        "roundtrip_cases": report.roundtrip_cases,
        "overfit_initial_loss": report.overfit_initial_loss,
        "overfit_final_loss": report.overfit_final_loss,
        "tokenizer_efficiency": tokenizer_stats,
        "source_audit": source_audit,
        "compute_probe": probe.to_dict(),
        "exam_steps": exam_steps,
    }
    _atomic_json_save(preflight, preflight_path)
    print(
        f"[Preflight] PASS roundtrips={report.roundtrip_cases} "
        f"overfit={report.overfit_initial_loss:.4f}->{report.overfit_final_loss:.4f} "
        f"objective=v{LOSS_OBJECTIVE_VERSION}"
    )
    print(
        f"[Architecture] ffn={cfg['ffn_type']} total_params={total_params:,} "
        f"active_params/token={active_params:,} activation={activation_ratio:.1%} "
        f"GQA={cfg['n_heads']}Q/{cfg['n_kv_heads']}KV "
        f"experts={cfg['num_experts']} top_k={cfg['experts_per_token']}"
    )
    print(
        f"[Compute] measured_supervised_tokens/s={probe.useful_tokens_per_second:,.1f} "
        f"projected_{args.session_hours:g}h={probe.projected_useful_tokens:,} "
        f"tokens/total_param={probe.projected_useful_tokens/max(total_params,1):.2f} "
        f"tokens/active_param={probe.projected_useful_tokens/max(active_params,1):.2f}"
    )
    print(
        f"[Compute] target={target_tokens:,} ({target_tpp:.1f} tokens/total_param) "
        f"exam_steps={exam_steps:,}"
    )
    if args.preflight_only:
        print("[Preflight] Full training intentionally not started. Verified stream-only artifacts are reusable.")
        return

    torch.manual_seed(SEED)
    random.seed(SEED)
    model = lineage_model if lineage_model is not None else VistaReasoningGPT(**model_config)
    optimizer = _new_optimizer(model, cfg)
    val_ds = PackedSequenceDataset(
        val_sequences,
        int(cfg["seq_len"]),
        tokenizer.pad_id(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    state_contract = {
        **contract,
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "model_config": model_config,
        "target_prediction_tokens": target_tokens,
        "exam_steps": exam_steps,
    }
    epoch = 0
    resume_step = 0
    cumulative_tokens = 0
    optimizer_steps = 0
    best_val = float("inf")
    stale = 0
    if args.resume:
        if not state_path.exists():
            raise RuntimeError("resume_training_state_missing")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        for key, value in state_contract.items():
            if state.get(key) != value:
                raise RuntimeError(f"resume_contract_mismatch:{key}")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        epoch = int(state["epoch"])
        resume_step = int(state["step"])
        cumulative_tokens = int(state["cumulative_prediction_tokens"])
        optimizer_steps = int(state["optimizer_steps"])
        best_val = float(state["best_validation_loss"])
        stale = int(state["epochs_without_improvement"])
        torch.set_rng_state(state["torch_rng_state"])
        random.setstate(state["python_random_state"])
        print(
            f"[Resume] epoch={epoch} step={resume_step}/{exam_steps} "
            f"tokens={cumulative_tokens:,}/{target_tokens:,}"
        )
    elif state_path.exists():
        raise RuntimeError("training_state_already_exists:use_--resume_or_clean_bundle")

    previous_exam = None
    if epoch == 0 and resume_step == 0:
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

    excluded_stream_texts = list(val_texts) + _exam_holdout_texts(stage)
    stream_generation = epoch if args.resume else 0
    train_loader = _stream_loader(
        specs=hf_specs,
        stage=stage,
        stream_generation=stream_generation,
        excluded=excluded_stream_texts,
        tokenizer=tokenizer,
        cfg=cfg,
    )
    iterator = iter(train_loader)

    # Resume cannot persist a remote HTTP cursor across processes. Replaying only
    # the in-interval batch count keeps recovery bounded; GuardedSource removes
    # exact/near duplicates during the reconstructed stream.
    if resume_step > 0:
        for _ in range(resume_step):
            try:
                next(iterator)
            except StopIteration:
                stream_generation += 1
                train_loader = _stream_loader(
                    specs=hf_specs,
                    stage=stage,
                    stream_generation=stream_generation,
                    excluded=excluded_stream_texts,
                    tokenizer=tokenizer,
                    cfg=cfg,
                )
                iterator = iter(train_loader)
                next(iterator)

    deadline = time.monotonic() + args.session_hours * 3600
    while cumulative_tokens < target_tokens and time.monotonic() < deadline:
        model.train()
        loss_sum = 0.0
        epoch_tokens = 0
        completed = resume_step

        while (
            completed < exam_steps
            and cumulative_tokens < target_tokens
            and time.monotonic() < deadline
        ):
            try:
                x, y = next(iterator)
            except StopIteration:
                stream_generation += 1
                train_loader = _stream_loader(
                    specs=hf_specs,
                    stage=stage,
                    stream_generation=stream_generation,
                    excluded=excluded_stream_texts,
                    tokenizer=tokenizer,
                    cfg=cfg,
                )
                iterator = iter(train_loader)
                x, y = next(iterator)

            completed += 1
            valid = int((y != tokenizer.pad_id()).sum().item())
            if valid <= 0:
                continue

            _set_token_scheduled_lr(
                optimizer,
                base_lr=float(cfg["lr"]),
                min_lr=float(cfg["lr_min"]),
                cumulative_tokens=cumulative_tokens,
                target_tokens=target_tokens,
                warmup_fraction=args.warmup_fraction,
            )
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, targets=y, pad_id=tokenizer.pad_id())
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("non_finite_training_loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            optimizer.step()

            loss_sum += float(loss.item()) * valid
            epoch_tokens += valid
            cumulative_tokens += valid
            optimizer_steps += 1

            if completed % int(cfg["checkpoint_every_steps"]) == 0:
                _save_state(
                    state_path,
                    contract=state_contract,
                    epoch=epoch,
                    step=completed,
                    cumulative_tokens=cumulative_tokens,
                    optimizer_steps=optimizer_steps,
                    best_val=best_val,
                    stale=stale,
                    model=model,
                    optimizer=optimizer,
                )
                print(
                    f"[Checkpoint] exam_epoch={epoch+1} step={completed}/{exam_steps} "
                    f"cumulative_supervised_tokens={cumulative_tokens:,}"
                )

        if epoch_tokens <= 0:
            raise RuntimeError("exam_epoch_has_no_prediction_tokens")
        train_loss = loss_sum / epoch_tokens

        if completed < exam_steps and time.monotonic() >= deadline:
            _save_state(
                state_path,
                contract=state_contract,
                epoch=epoch,
                step=completed,
                cumulative_tokens=cumulative_tokens,
                optimizer_steps=optimizer_steps,
                best_val=best_val,
                stale=stale,
                model=model,
                optimizer=optimizer,
            )
            _run_exam(
                model,
                tokenizer,
                epoch=epoch,
                stage=stage,
                train_loss=train_loss,
                val_loss=None,
                exams_dir=exams_dir,
                previous=previous_exam,
                max_new_tokens=int(cfg["exam_max_new_tokens"]),
                prefix="session_end_exam",
            )
            print(
                f"[SessionStop] step={completed}/{exam_steps} "
                f"tokens={cumulative_tokens:,}/{target_tokens:,}. Resume with --resume."
            )
            return

        val_loss, val_tokens = _validation_loss(
            model,
            val_loader,
            pad_id=tokenizer.pad_id(),
        )
        if val_loss < best_val - float(cfg["early_stop_min_delta"]):
            best_val = val_loss
            stale = 0
            _atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "validation_loss": best_val,
                    "epoch": epoch + 1,
                    "cumulative_prediction_tokens": cumulative_tokens,
                },
                best_path,
            )
        else:
            stale += 1

        previous_exam = _run_exam(
            model,
            tokenizer,
            epoch=epoch + 1,
            stage=stage,
            train_loss=train_loss,
            val_loss=val_loss,
            exams_dir=exams_dir,
            previous=previous_exam,
            max_new_tokens=int(cfg["exam_max_new_tokens"]),
        )
        epoch += 1
        resume_step = 0
        _save_state(
            state_path,
            contract=state_contract,
            epoch=epoch,
            step=0,
            cumulative_tokens=cumulative_tokens,
            optimizer_steps=optimizer_steps,
            best_val=best_val,
            stale=stale,
            model=model,
            optimizer=optimizer,
        )
        current_tpp = cumulative_tokens / max(total_params, 1)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"best_val={best_val:.4f} supervised_tokens={epoch_tokens:,} "
            f"cumulative={cumulative_tokens:,}/{target_tokens:,} "
            f"tokens/total_param={current_tpp:.3f} val_tokens={val_tokens:,}"
        )
        if (
            current_tpp >= float(cfg["early_stop_min_tokens_per_parameter"])
            and stale >= int(cfg["early_stop_patience"])
        ):
            print(
                f"[EarlyStop] validation stalled for {stale} exam epochs "
                f"after {current_tpp:.2f} tokens/total-param"
            )
            break

    if not best_path.exists():
        _atomic_torch_save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "validation_loss": best_val,
                "epoch": epoch,
                "cumulative_prediction_tokens": cumulative_tokens,
            },
            best_path,
        )

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    model.eval()
    final_exam = _run_exam(
        model,
        tokenizer,
        epoch=int(best.get("epoch", epoch)),
        stage=stage,
        train_loss=None,
        val_loss=float(best.get("validation_loss", best_val)),
        exams_dir=exams_dir,
        previous=None,
        max_new_tokens=int(cfg["exam_max_new_tokens"]),
        prefix="best_model_exam",
    )
    metrics = {
        "trainer_version": TRAINER_VERSION,
        "loss_objective_version": LOSS_OBJECTIVE_VERSION,
        "profile": args.profile,
        "training_stage": stage,
        "parameter_count": total_params,
        "active_parameter_count": active_params,
        "activation_ratio": activation_ratio,
        "cumulative_prediction_tokens": cumulative_tokens,
        "target_prediction_tokens": target_tokens,
        "optimizer_steps": optimizer_steps,
        "best_validation_loss": float(best.get("validation_loss", best_val)),
        "tokenizer_efficiency": tokenizer_stats,
        "stream_only": True,
        "persistent_stream_per_session": True,
        "sparse_moe": str(cfg["ffn_type"]) == "moe",
        "gqa": {
            "query_heads": int(cfg["n_heads"]),
            "kv_heads": int(cfg["n_kv_heads"]),
        },
        "routing": {
            "experts": int(cfg["num_experts"]),
            "top_k": int(cfg["experts_per_token"]),
        },
        "best_model_exam": {
            "correctness_percent": final_exam.correctness_percent,
            "mean_quality_percent": final_exam.mean_quality_percent,
            "gibberish_answers": final_exam.gibberish_answers,
            "training_signal": final_exam.training_signal,
        },
    }
    save_model_bundle(
        bundle_dir=bundle,
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        training_stage=stage,
        corpus_fingerprint=contract["curriculum_fingerprint"],
        metrics=metrics,
    )
    print(f"bundle={bundle}")


if __name__ == "__main__":
    main()
