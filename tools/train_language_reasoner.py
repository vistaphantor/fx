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
from src.language.curriculum import CURRICULUM_STAGES, select_curriculum
from src.language.data_pipeline import build_tokenizer_training_sample, load_all_training_text
from src.language.exam import (
    EpochExamResult,
    build_exam_prompt,
    exam_questions,
    render_exam_text,
    run_epoch_exam,
    save_epoch_exam,
)
from src.language.model_bundle import load_model_bundle, save_model_bundle
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.streaming_sources import (
    HFSourceSpec,
    build_training_stream,
    load_hf_source_config,
    sample_training_stream,
    specs_fingerprint,
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

TRAINING_STATE_VERSION = 4
PREFLIGHT_MANIFEST_VERSION = 4
SEED = 42

PROFILES = {
    "smoke": dict(vocab_size=1024, d_model=128, n_heads=4, n_layers=4, ffn_dim=512,
                  seq_len=128, batch_size=4, epochs=20, lr=8e-4, max_examples=512,
                  tokenizer_chars=1_500_000, early_stop_patience=5, early_stop_min_delta=0.003,
                  exam_max_new_tokens=48, stream_steps_per_epoch=256, hf_preflight_sample_examples=1000,
                  checkpoint_every_steps=50),
    "8m": dict(vocab_size=4096, d_model=256, n_heads=8, n_layers=8, ffn_dim=1024,
               seq_len=192, batch_size=2, epochs=12, lr=4e-4, max_examples=None,
               tokenizer_chars=4_000_000, early_stop_patience=3, early_stop_min_delta=0.005,
               exam_max_new_tokens=48, stream_steps_per_epoch=1000, hf_preflight_sample_examples=3000,
               checkpoint_every_steps=100),
    "15m": dict(vocab_size=8192, d_model=320, n_heads=8, n_layers=10, ffn_dim=1280,
                seq_len=256, batch_size=1, epochs=10, lr=3e-4, max_examples=None,
                tokenizer_chars=8_000_000, early_stop_patience=3, early_stop_min_delta=0.005,
                exam_max_new_tokens=48, stream_steps_per_epoch=800, hf_preflight_sample_examples=5000,
                checkpoint_every_steps=100),
}


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
    result = [text for text in texts if prompt_family(text) not in held]
    return result, len(texts) - len(result)


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
    return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])


def _finite_loader(dataset: PackedSequenceDataset, *, batch_size: int, epoch: int) -> DataLoader:
    generator = torch.Generator().manual_seed(SEED + epoch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)


def _validation_loader(dataset: PackedSequenceDataset, *, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def _stream_loader(
    *,
    specs: tuple[HFSourceSpec, ...],
    stage: str,
    epoch: int,
    train_texts: list[str],
    excluded_texts: list[str],
    tokenizer: BPETokenizer,
    cfg: dict,
    local_replay_weight: float,
) -> DataLoader:
    stream = build_training_stream(
        specs=specs,
        stage=stage,
        seed=SEED + epoch * 100_003,
        local_replay=train_texts,
        local_weight=local_replay_weight,
        excluded_texts=excluded_texts,
        repeat=True,
    )
    dataset = CorpusStreamer(stream, tokenizer, seq_len=cfg["seq_len"], seed=SEED + epoch)
    return DataLoader(dataset, batch_size=cfg["batch_size"], num_workers=0)


def _validation_loss(model: VistaReasoningGPT, loader: DataLoader, *, pad_id: int) -> tuple[float, int]:
    model.eval()
    total = 0.0
    tokens = 0
    with torch.no_grad():
        for x, y in loader:
            valid = int((y != pad_id).sum().item())
            if valid <= 0:
                continue
            _, loss = model(x, targets=y, pad_id=pad_id)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("non_finite_validation_loss")
            total += float(loss.item()) * valid
            tokens += valid
    if tokens <= 0:
        raise RuntimeError("validation_has_no_prediction_tokens")
    return total / tokens, tokens


def _load_or_sample_hf(
    *, specs: tuple[HFSourceSpec, ...], stage: str, limit: int,
    sample_path: Path, metadata_path: Path,
) -> tuple[list[str], str]:
    fingerprint = specs_fingerprint(specs)
    expected = {"config_fingerprint": fingerprint, "stage": stage, "limit": int(limit)}
    if sample_path.exists() or metadata_path.exists():
        if not sample_path.exists() or not metadata_path.exists():
            raise RuntimeError("incomplete_hf_preflight_sample_cache")
        if json.loads(metadata_path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("hf_preflight_sample_contract_mismatch:delete_bundle_.training_hf_sample")
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        if not isinstance(sample, list) or not sample:
            raise RuntimeError("cached_hf_preflight_sample_invalid")
        return [canonicalize_serialized(value) for value in sample], fingerprint
    sample = sample_training_stream(specs=specs, stage=stage, limit=limit, seed=SEED)
    _atomic_json_save(sample, sample_path)
    _atomic_json_save(expected, metadata_path)
    return sample, fingerprint


def _prepare_lineage(stage: str, init_bundle: Path | None, cfg: dict):
    if stage == "foundation":
        if init_bundle is not None:
            raise RuntimeError("foundation_stage_must_start_without_init_bundle")
        return None, None, None
    if init_bundle is None:
        raise RuntimeError(f"{stage}_requires_--init-bundle")
    model, tokenizer, manifest = load_model_bundle(init_bundle, device="cpu")
    if manifest.model_config != _model_config(cfg, tokenizer):
        raise RuntimeError("init_bundle_model_config_mismatch")
    return model, tokenizer, tokenizer.fingerprint()


def _static_contract(
    *, profile: str, stage: str, cfg: dict, source_hash: str, curriculum_hash: str,
    split_hash: str, lineage: str | None, hf_fingerprint: str | None,
    stream_steps: int | None,
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
        "lineage_tokenizer_fingerprint": lineage,
        "hf_sources_fingerprint": hf_fingerprint,
        "stream_steps_per_epoch": stream_steps,
    }


def _validate_contract(saved: dict, expected: dict, *, prefix: str) -> None:
    for key, value in expected.items():
        if saved.get(key) != value:
            raise RuntimeError(f"{prefix}_contract_mismatch:{key}")


def _prepare_tokenizer(
    *, path: Path, manifest_path: Path, contract: dict, train_texts: list[str], cfg: dict,
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
    sample = build_tokenizer_training_sample(train_texts, max_chars=cfg["tokenizer_chars"], seed=SEED)
    tokenizer = BPETokenizer()
    tokenizer.train(sample, vocab_size=cfg["vocab_size"])
    tokenizer.save(path)
    return tokenizer, False


def _checkpoint_contract(
    *, profile: str, stage: str, tokenizer: BPETokenizer, model_config: dict,
    curriculum_hash: str, split_hash: str, lineage: str | None,
    hf_fingerprint: str | None, stream_steps: int | None,
) -> dict:
    return {
        "training_state_version": TRAINING_STATE_VERSION,
        "profile": profile,
        "training_stage": stage,
        "tokenizer_algorithm_version": tokenizer.algorithm_version,
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "model_config": model_config,
        "curriculum_fingerprint": curriculum_hash,
        "split_fingerprint": split_hash,
        "lineage_tokenizer_fingerprint": lineage,
        "hf_sources_fingerprint": hf_fingerprint,
        "stream_steps_per_epoch": stream_steps,
    }


def _budget(report, model, *, epochs: int, batch_size: int, seq_len: int, stream_steps: int | None) -> dict:
    params = model.get_num_params()
    if stream_steps is None:
        useful_per_epoch = report.train_prediction_tokens
        utilization = 1.0
        mode = "finite"
    else:
        capacity = max(1, report.train_sequences * seq_len)
        utilization = min(1.0, report.train_prediction_tokens / capacity)
        useful_per_epoch = int(stream_steps * batch_size * seq_len * utilization)
        mode = "streaming"
    planned = useful_per_epoch * epochs
    print(
        f"[Budget] mode={mode} params={params:,} estimated_context_utilization={utilization:.3f} "
        f"estimated_useful_tokens/epoch={useful_per_epoch:,} planned_useful_tokens={planned:,} "
        f"planned_useful_tokens_per_param={planned / max(params, 1):.3f}"
    )
    return {
        "training_mode": mode,
        "parameter_count": params,
        "estimated_context_utilization": utilization,
        "estimated_useful_training_tokens_per_epoch": useful_per_epoch,
        "estimated_planned_useful_tokens": planned,
        "estimated_planned_useful_tokens_per_parameter": planned / max(params, 1),
    }


def _run_exam(model, tokenizer, *, epoch: int, stage: str, train_loss, val_loss, exams_dir: Path,
              previous: EpochExamResult | None, max_new_tokens: int) -> EpochExamResult:
    started = time.time()
    result = run_epoch_exam(
        model=model, tokenizer=tokenizer, epoch=epoch, training_stage=stage,
        train_loss=train_loss, validation_loss=val_loss, max_new_tokens=max_new_tokens,
    )
    text_path, _ = save_epoch_exam(result=result, exams_dir=exams_dir, previous=previous)
    print(
        f"[Exam] epoch={epoch} correct={result.correct_questions}/{result.total_questions} "
        f"score={result.correctness_percent:.1f}% quality={result.mean_quality_percent:.1f}% "
        f"gibberish={result.gibberish_answers}/{result.total_questions} "
        f"seconds={time.time() - started:.1f} file={text_path}"
    )
    if epoch >= 2 and result.correct_questions == 0 and result.gibberish_answers == result.total_questions:
        print("[Exam] ALERT: no correct answers and every answer is still gibberish; inspect before continuing.")
    return result


def _save_best_exam(result: EpochExamResult, bundle_dir: Path) -> None:
    exams = bundle_dir / "exams"
    exams.mkdir(parents=True, exist_ok=True)
    (exams / "best_model_exam.txt").write_text(render_exam_text(result), encoding="utf-8")
    (exams / "best_model_exam.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _save_state(
    path: Path, *, contract: dict, epoch_index: int, step_in_epoch: int,
    epoch_loss_sum: float, epoch_tokens: int, best_val: float, best_epoch: int,
    stale_epochs: int, model, optimizer, scheduler,
) -> None:
    _atomic_torch_save(
        {
            **contract,
            "epoch_index": epoch_index,
            "step_in_epoch": step_in_epoch,
            "epoch_loss_sum": epoch_loss_sum,
            "epoch_tokens": epoch_tokens,
            "best_validation_loss": best_val,
            "best_epoch": best_epoch,
            "epochs_without_improvement": stale_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )


def _train_epoch(
    *, model, loader: DataLoader, optimizer, scheduler, pad_id: int, grad_clip: float,
    steps_per_epoch: int, resume_step: int, loss_sum: float, token_count: int,
    checkpoint_every: int, checkpoint_callback,
) -> tuple[float, int]:
    model.train()
    iterator = iter(loader)
    for _ in range(resume_step):
        try:
            next(iterator)
        except StopIteration as exc:
            raise RuntimeError("resume_stream_ended_before_saved_step") from exc

    for step in range(resume_step, steps_per_epoch):
        try:
            x, y = next(iterator)
        except StopIteration as exc:
            raise RuntimeError(f"stream_ended_before_epoch_budget:{step}<{steps_per_epoch}") from exc
        valid = int((y != pad_id).sum().item())
        if valid <= 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y, pad_id=pad_id)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("non_finite_training_loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        loss_sum += float(loss.item()) * valid
        token_count += valid
        completed = step + 1
        if completed % checkpoint_every == 0 and completed < steps_per_epoch:
            checkpoint_callback(completed, loss_sum, token_count)
    if token_count <= 0:
        raise RuntimeError("epoch_has_no_prediction_tokens")
    return loss_sum / token_count, token_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="8m")
    parser.add_argument("--data-root", default="data/data/trainingdata")
    parser.add_argument("--bundle-dir", default=None)
    parser.add_argument("--training-stage", default="foundation")
    parser.add_argument("--init-bundle", default=None)
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--hf-sample-examples", type=int, default=None)
    parser.add_argument("--stream-steps-per-epoch", type=int, default=None)
    parser.add_argument("--local-replay-weight", type=float, default=0.25)
    parser.add_argument("--checkpoint-every-steps", type=int, default=None)
    parser.add_argument("--threads", type=int, default=max(1, min(4, os.cpu_count() or 2)))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-data-starved", action="store_true")
    args = parser.parse_args()

    stage = _normalize_stage(args.training_stage)
    cfg = dict(PROFILES[args.profile])
    cfg.update(dropout=0.10, lr_min=1e-5, weight_decay=0.01, grad_clip=1.0, val_split=0.05)
    checkpoint_every = args.checkpoint_every_steps or cfg["checkpoint_every_steps"]
    if checkpoint_every <= 0 or args.local_replay_weight <= 0:
        raise ValueError("checkpoint interval and local replay weight must be positive")

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
    exams_dir = bundle_dir / "exams"
    hf_sample_path = work_dir / "hf_preflight_sample.json"
    hf_sample_meta = work_dir / "hf_preflight_sample_meta.json"

    local_texts = load_all_training_text(Path(args.data_root), cfg["max_examples"], False, SEED)
    hf_specs: tuple[HFSourceSpec, ...] = ()
    hf_sample: list[str] = []
    hf_fingerprint: str | None = None
    stream_steps: int | None = None
    if args.hf_config:
        hf_specs = load_hf_source_config(args.hf_config)
        sample_limit = args.hf_sample_examples or cfg["hf_preflight_sample_examples"]
        hf_sample, hf_fingerprint = _load_or_sample_hf(
            specs=hf_specs, stage=stage, limit=sample_limit,
            sample_path=hf_sample_path, metadata_path=hf_sample_meta,
        )
        stream_steps = args.stream_steps_per_epoch or cfg["stream_steps_per_epoch"]
        if stream_steps <= 0:
            raise ValueError("--stream-steps-per-epoch must be positive")
        print(f"[HF] sources={len(hf_specs)} sample={len(hf_sample):,} steps/epoch={stream_steps:,} fingerprint={hf_fingerprint[:12]}")
    elif args.stream_steps_per_epoch is not None:
        raise RuntimeError("--stream-steps-per-epoch_requires_--hf-config")

    all_texts = _stable_unique(local_texts + hf_sample)
    all_texts, exam_removed = _remove_exam_families(all_texts, stage)
    print(f"[ExamHoldout] excluded_prompt_families={len(exam_questions(stage))} removed_from_preflight={exam_removed}")
    if not all_texts:
        raise RuntimeError("no_training_data_after_exam_holdout")

    selection = select_curriculum(all_texts, stage=stage, seed=SEED)
    texts = selection.texts
    print(
        f"[Curriculum] stage={stage} selected={len(texts):,}/{selection.total_available:,} "
        f"trading={selection.trading_available:,} reasoning={selection.reasoning_available:,} math={selection.math_available:,}"
    )

    lineage_model, lineage_tokenizer, lineage_fp = _prepare_lineage(
        stage, Path(args.init_bundle) if args.init_bundle else None, cfg
    )
    train_texts, val_texts = split_by_prompt_family(texts, val_fraction=cfg["val_split"], seed=SEED)
    source_hash = corpus_fingerprint(all_texts)
    curriculum_hash = corpus_fingerprint(texts)
    split_hash = split_fingerprint(train_texts, val_texts)
    print(f"[Split] train={len(train_texts):,} val={len(val_texts):,} family_isolated=true split={split_hash[:12]}")

    static = _static_contract(
        profile=args.profile, stage=stage, cfg=cfg, source_hash=source_hash,
        curriculum_hash=curriculum_hash, split_hash=split_hash, lineage=lineage_fp,
        hf_fingerprint=hf_fingerprint, stream_steps=stream_steps,
    )
    tokenizer, reused = _prepare_tokenizer(
        path=tokenizer_path, manifest_path=preflight_path, contract=static,
        train_texts=train_texts, cfg=cfg, lineage_tokenizer=lineage_tokenizer,
    )
    print(f"[Tokenizer] reuse={str(reused).lower()} fingerprint={tokenizer.fingerprint()}")

    train_sequences = build_example_sequences(train_texts, tokenizer, seq_len=cfg["seq_len"])
    val_sequences = build_example_sequences(val_texts, tokenizer, seq_len=cfg["seq_len"])
    report = run_training_preflight(
        tokenizer=tokenizer, train_texts=train_texts, val_texts=val_texts,
        train_sequences=train_sequences, val_sequences=val_sequences, seq_len=cfg["seq_len"],
    )
    _atomic_json_save(
        {**static, "tokenizer_fingerprint": tokenizer.fingerprint(), "actual_vocab_size": tokenizer.vocab_size,
         "roundtrip_cases": report.roundtrip_cases, "overfit_initial_loss": report.overfit_initial_loss,
         "overfit_final_loss": report.overfit_final_loss, "train_prediction_tokens": report.train_prediction_tokens,
         "validation_prediction_tokens": report.validation_prediction_tokens},
        preflight_path,
    )
    print(f"[Preflight] PASS roundtrips={report.roundtrip_cases} overfit={report.overfit_initial_loss:.4f}->{report.overfit_final_loss:.4f}")

    model_config = _model_config(cfg, tokenizer)
    model = lineage_model if lineage_model is not None else VistaReasoningGPT(**model_config).to("cpu")
    if model.max_seq_len != cfg["seq_len"] or model.vocab_size != tokenizer.vocab_size:
        raise RuntimeError("model_runtime_contract_mismatch")
    budget = _budget(report, model, epochs=cfg["epochs"], batch_size=cfg["batch_size"], seq_len=cfg["seq_len"], stream_steps=stream_steps)
    if stage == "foundation" and args.profile != "smoke" and budget["estimated_planned_useful_tokens_per_parameter"] < 1.0:
        print("[Budget] BLOCK: useful token exposure is below the conservative no-waste floor.")
        if not args.preflight_only and not args.allow_data_starved:
            raise RuntimeError("foundation_training_blocked_by_data_budget")
    if args.preflight_only:
        print("[Preflight] Full training intentionally not started.")
        return

    finite_ds = None if hf_specs else PackedSequenceDataset(train_sequences, cfg["seq_len"], tokenizer.pad_id())
    val_ds = PackedSequenceDataset(val_sequences, cfg["seq_len"], tokenizer.pad_id())
    val_loader = _validation_loader(val_ds, batch_size=cfg["batch_size"])
    optimizer = _new_optimizer(model, cfg)
    steps_per_epoch = int(stream_steps) if hf_specs else math.ceil(len(finite_ds) / cfg["batch_size"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, steps_per_epoch * cfg["epochs"]), eta_min=cfg["lr_min"]
    )
    contract = _checkpoint_contract(
        profile=args.profile, stage=stage, tokenizer=tokenizer, model_config=model_config,
        curriculum_hash=curriculum_hash, split_hash=split_hash, lineage=lineage_fp,
        hf_fingerprint=hf_fingerprint, stream_steps=stream_steps,
    )

    epoch_index = 0
    resume_step = 0
    epoch_loss_sum = 0.0
    epoch_tokens = 0
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
        scheduler.load_state_dict(state["scheduler_state_dict"])
        epoch_index = int(state["epoch_index"])
        resume_step = int(state["step_in_epoch"])
        epoch_loss_sum = float(state.get("epoch_loss_sum", 0.0))
        epoch_tokens = int(state.get("epoch_tokens", 0))
        best_val = float(state["best_validation_loss"])
        best_epoch = int(state.get("best_epoch", 0))
        stale_epochs = int(state.get("epochs_without_improvement", 0))
        print(f"[Resume] epoch={epoch_index + 1} step={resume_step}/{steps_per_epoch} best_val={best_val:.4f}")
    elif state_path.exists():
        raise RuntimeError("training_state_already_exists:use_--resume")

    previous_exam: EpochExamResult | None = None
    if epoch_index == 0 and resume_step == 0:
        previous_exam = _run_exam(
            model, tokenizer, epoch=0, stage=stage, train_loss=None, val_loss=None,
            exams_dir=exams_dir, previous=None, max_new_tokens=cfg["exam_max_new_tokens"],
        )

    actual_epoch_tokens: list[int] = []
    started = time.time()
    for epoch in range(epoch_index, cfg["epochs"]):
        epoch_started = time.time()
        excluded = list(val_texts) + _exam_holdout_texts(stage)
        if hf_specs:
            loader = _stream_loader(
                specs=hf_specs, stage=stage, epoch=epoch, train_texts=train_texts,
                excluded_texts=excluded, tokenizer=tokenizer, cfg=cfg,
                local_replay_weight=args.local_replay_weight,
            )
        else:
            assert finite_ds is not None
            loader = _finite_loader(finite_ds, batch_size=cfg["batch_size"], epoch=epoch)

        current_resume = resume_step if epoch == epoch_index else 0
        current_loss_sum = epoch_loss_sum if epoch == epoch_index else 0.0
        current_tokens = epoch_tokens if epoch == epoch_index else 0

        def checkpoint(completed: int, loss_sum: float, tokens: int) -> None:
            _save_state(
                state_path, contract=contract, epoch_index=epoch, step_in_epoch=completed,
                epoch_loss_sum=loss_sum, epoch_tokens=tokens, best_val=best_val, best_epoch=best_epoch,
                stale_epochs=stale_epochs, model=model, optimizer=optimizer, scheduler=scheduler,
            )
            print(f"[Checkpoint] epoch={epoch + 1} step={completed}/{steps_per_epoch} tokens={tokens:,}")

        train_loss, trained_tokens = _train_epoch(
            model=model, loader=loader, optimizer=optimizer, scheduler=scheduler,
            pad_id=tokenizer.pad_id(), grad_clip=cfg["grad_clip"], steps_per_epoch=steps_per_epoch,
            resume_step=current_resume, loss_sum=current_loss_sum, token_count=current_tokens,
            checkpoint_every=checkpoint_every, checkpoint_callback=checkpoint,
        )
        actual_epoch_tokens.append(trained_tokens)
        val_loss, val_tokens = _validation_loss(model, val_loader, pad_id=tokenizer.pad_id())
        improved = val_loss < best_val - cfg["early_stop_min_delta"]
        if improved:
            best_val = val_loss
            best_epoch = epoch + 1
            stale_epochs = 0
            _atomic_torch_save(
                {"model_state_dict": model.state_dict(), "model_config": model_config,
                 "validation_loss": best_val, "epoch": best_epoch},
                best_model_path,
            )
        else:
            stale_epochs += 1

        _save_state(
            state_path, contract=contract, epoch_index=epoch + 1, step_in_epoch=0,
            epoch_loss_sum=0.0, epoch_tokens=0, best_val=best_val, best_epoch=best_epoch,
            stale_epochs=stale_epochs, model=model, optimizer=optimizer, scheduler=scheduler,
        )
        previous_exam = _run_exam(
            model, tokenizer, epoch=epoch + 1, stage=stage, train_loss=train_loss, val_loss=val_loss,
            exams_dir=exams_dir, previous=previous_exam, max_new_tokens=cfg["exam_max_new_tokens"],
        )
        utilization = trained_tokens / max(1, steps_per_epoch * cfg["batch_size"] * cfg["seq_len"])
        print(
            f"epoch={epoch + 1}/{cfg['epochs']} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"best_val={best_val:.4f} train_tokens={trained_tokens:,} val_tokens={val_tokens:,} "
            f"actual_context_utilization={utilization:.3f} epoch_seconds={time.time() - epoch_started:.1f}"
        )
        resume_step = 0
        epoch_loss_sum = 0.0
        epoch_tokens = 0
        if stale_epochs >= cfg["early_stop_patience"]:
            print(f"[EarlyStop] no material validation improvement for {stale_epochs} epochs")
            break

    if not best_model_path.exists():
        raise RuntimeError("best_model_checkpoint_missing")
    best = torch.load(best_model_path, map_location="cpu", weights_only=False)
    if best.get("model_config") != model_config:
        raise RuntimeError("best_model_config_mismatch")
    model.load_state_dict(best["model_state_dict"])
    model.eval()

    best_exam = run_epoch_exam(
        model=model, tokenizer=tokenizer, epoch=int(best.get("epoch", best_epoch)), training_stage=stage,
        train_loss=None, validation_loss=best_val, max_new_tokens=cfg["exam_max_new_tokens"],
    )
    _save_best_exam(best_exam, bundle_dir)
    print(f"[BestExam] epoch={best_epoch} score={best_exam.correctness_percent:.1f}% quality={best_exam.mean_quality_percent:.1f}%")

    metrics = {
        "best_validation_loss": best_val,
        "best_epoch": best_epoch,
        "perplexity": math.exp(min(best_val, 20)),
        "training_seconds_this_run": time.time() - started,
        "profile": args.profile,
        "training_stage": stage,
        "hf_sources_fingerprint": hf_fingerprint,
        "stream_steps_per_epoch": stream_steps,
        "family_isolated_validation": True,
        "validation_excluded_from_stream": True,
        "exam_prompt_families_held_out": True,
        "example_aware_stream_packing": True,
        "step_resumable_training": True,
        "checkpoint_every_steps": checkpoint_every,
        "actual_epoch_prediction_tokens": actual_epoch_tokens,
        "preflight_passed": True,
        "best_model_exam": {
            "correctness_percent": best_exam.correctness_percent,
            "mean_quality_percent": best_exam.mean_quality_percent,
            "gibberish_answers": best_exam.gibberish_answers,
        },
        **budget,
    }
    save_model_bundle(
        bundle_dir=bundle_dir, model=model, tokenizer=tokenizer, model_config=model_config,
        training_stage=stage, corpus_fingerprint=curriculum_hash, metrics=metrics,
    )
    print(f"bundle={bundle_dir}")


if __name__ == "__main__":
    main()
