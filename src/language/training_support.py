from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from corpus.streamer import CorpusStreamer
from src.language.canonical_contract import canonicalize_serialized, prompt_family
from src.language.curriculum import CURRICULUM_STAGES
from src.language.exam import EpochExamResult, build_exam_prompt, exam_questions, run_epoch_exam, save_epoch_exam
from src.language.exam_feedback import ExamFeedbackPolicy
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.semantic_checkpointing import mastery_report
from src.language.streaming_sources import (
    HFSourceSpec,
    build_training_stream,
    hf_source_from_spec,
    sample_training_stream,
    specs_fingerprint,
    stage_specs,
)
from src.language.tokenizer import BPETokenizer
from src.language.training_artifacts import atomic_json_save
from src.language.training_session_contract import SEED


def stable_unique(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in texts:
        text = canonicalize_serialized(raw)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def normalize_stage(value: str) -> str:
    stage = value.strip().casefold()
    if stage == "general_language":
        stage = "foundation"
    if stage not in CURRICULUM_STAGES:
        raise ValueError(f"unsupported_training_stage:{value}")
    return stage


def exam_holdout_texts(stage: str) -> list[str]:
    return [build_exam_prompt(question.prompt) for question in exam_questions(stage)]


def remove_exam_families(texts: list[str], stage: str) -> tuple[list[str], int]:
    held = {prompt_family(text) for text in exam_holdout_texts(stage)}
    kept = [text for text in texts if prompt_family(text) not in held]
    return kept, len(texts) - len(kept)


def model_config(cfg: dict, tokenizer: BPETokenizer) -> dict:
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


def profile_contract(cfg: dict) -> dict:
    keys = (
        "vocab_size", "d_model", "n_heads", "n_kv_heads", "n_layers",
        "ffn_dim", "ffn_type", "num_experts", "experts_per_token",
        "moe_ffn_dim", "shared_expert_ffn_dim", "router_aux_loss_coef",
        "router_jitter", "rope_theta", "seq_len", "batch_size", "lr",
        "dropout", "tokenizer_chars", "exam_max_new_tokens", "checkpoint_every_steps",
    )
    return {key: cfg[key] for key in keys}


def new_optimizer(model: VistaReasoningGPT, cfg: dict) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(cfg["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(cfg["lr"]), betas=(0.9, 0.95), eps=1e-8,
    )


def validation_loss(model: VistaReasoningGPT, loader: DataLoader, *, pad_id: int) -> tuple[float, int]:
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


def audit_hf_sources(
    specs: tuple[HFSourceSpec, ...], *, stage: str, rows: int, min_rate: float,
) -> list[dict]:
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


def load_hf_sample(
    *,
    specs: tuple[HFSourceSpec, ...],
    stage: str,
    limit: int,
    sample_path: Path,
    metadata_path: Path,
    refresh: bool = False,
) -> tuple[list[str], str]:
    fingerprint = specs_fingerprint(specs)
    expected = {"config_fingerprint": fingerprint, "stage": stage, "limit": int(limit)}
    if sample_path.exists() or metadata_path.exists():
        if not sample_path.exists() or not metadata_path.exists():
            raise RuntimeError("incomplete_hf_preflight_sample_cache")
        saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if saved_metadata == expected:
            sample = json.loads(sample_path.read_text(encoding="utf-8"))
            return [canonicalize_serialized(value) for value in sample], fingerprint
        if not refresh:
            raise RuntimeError("hf_preflight_sample_contract_mismatch:use_--curriculum-upgrade_or_clean_bundle")
        print("[CurriculumUpgrade] refreshing HF preflight sample for new source contract")
    sample = sample_training_stream(specs=specs, stage=stage, limit=limit, seed=SEED)
    atomic_json_save(sample, sample_path)
    atomic_json_save(expected, metadata_path)
    return sample, fingerprint


def tokenizer_efficiency(texts: list[str], tokenizer: BPETokenizer) -> dict:
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


def run_exam(
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
        model=model, tokenizer=tokenizer, epoch=epoch, training_stage=stage,
        train_loss=train_loss, validation_loss=val_loss, max_new_tokens=max_new_tokens,
    )
    text_path, _ = save_epoch_exam(result=result, exams_dir=exams_dir, previous=previous, prefix=prefix)
    mastery = mastery_report(result, stage)
    print(
        f"[Exam] epoch={epoch} correct={result.correct_questions}/{result.total_questions} "
        f"score={result.correctness_percent:.1f}% skills={mastery.mastered_skills}/{len(mastery.skill_results)} "
        f"concept_gates={mastery.conceptual_gates_passed}/{len(mastery.skill_results)} "
        f"gibberish={result.gibberish_answers}/{result.total_questions} "
        f"seconds={time.time()-started:.1f} file={text_path}"
    )
    return result


def stream_loader(
    *,
    specs: tuple[HFSourceSpec, ...],
    stage: str,
    stream_generation: int,
    excluded: list[str],
    tokenizer: BPETokenizer,
    cfg: dict,
    feedback: ExamFeedbackPolicy,
) -> DataLoader:
    stream = build_training_stream(
        specs=specs,
        stage=stage,
        seed=SEED + stream_generation * 100_003,
        excluded_texts=excluded,
        repeat=True,
        feedback=feedback,
    )
    return DataLoader(
        CorpusStreamer(stream, tokenizer, seq_len=int(cfg["seq_len"]), seed=SEED + stream_generation),
        batch_size=int(cfg["batch_size"]), num_workers=0, drop_last=False,
    )
