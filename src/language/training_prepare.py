from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.language.compute_budget import benchmark_training_throughput
from src.language.curriculum import select_curriculum
from src.language.data_pipeline import build_tokenizer_training_sample
from src.language.exam import exam_questions
from src.language.exam_feedback import EXAM_FEEDBACK_VERSION
from src.language.foundation_contract import (
    FOUNDATION_CONTRACT_VERSION,
    FOUNDATION_EXAM_INTERVAL_SECONDS,
    FOUNDATION_MIN_AVAILABLE_CURRICULUM_TOKENS,
)
from src.language.hard_negative_objective import HARD_NEGATIVE_OBJECTIVE_VERSION
from src.language.loss_objective import LOSS_OBJECTIVE_VERSION
from src.language.model_bundle import load_model_bundle
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.semantic_checkpointing import SEMANTIC_CHECKPOINT_POLICY_VERSION
from src.language.streaming_sources import HFSourceSpec, load_hf_source_config, require_curriculum_capacity, stage_specs
from src.language.tokenizer import BPETokenizer, TOKENIZER_ALGORITHM_VERSION
from src.language.training_artifacts import atomic_json_save, contract_mismatch_allowed
from src.language.training_budget import target_prediction_tokens
from src.language.training_pipeline import (
    build_example_sequences,
    corpus_fingerprint,
    run_training_preflight,
    split_by_prompt_family,
    split_fingerprint,
)
from src.language.training_profiles import profile as load_profile
from src.language.training_session_contract import (
    PREFLIGHT_MANIFEST_VERSION,
    SEED,
    TRAINER_VERSION,
    TRAINING_STATE_VERSION,
)
from src.language.training_support import (
    audit_hf_sources,
    load_hf_sample,
    model_config,
    profile_contract,
    remove_exam_families,
    stable_unique,
    tokenizer_efficiency,
)


@dataclass(slots=True)
class PreparedTraining:
    stage: str
    cfg: dict
    bundle: Path
    exams_dir: Path
    state_path: Path
    best_path: Path
    hf_specs: tuple[HFSourceSpec, ...]
    val_texts: list[str]
    contract: dict
    tokenizer: BPETokenizer
    model_config: dict
    tokenizer_stats: dict
    total_params: int
    active_params: int
    activation_ratio: float
    target_tokens: int
    lineage_model: VistaReasoningGPT | None
    val_sequences: list[list[int]]


def prepare_training(args: Any) -> PreparedTraining | None:
    stage = args.training_stage.strip().casefold()
    if stage == "general_language":
        stage = "foundation"
    cfg = load_profile(args.profile)
    cfg.update(lr_min=1e-5, weight_decay=0.01, grad_clip=1.0, val_split=0.05)

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
    inventory = require_curriculum_capacity(hf_specs, stage)
    if stage == "foundation":
        print(
            f"[CurriculumInventory] available_tokens={inventory.available_tokens:,} "
            f"required={FOUNDATION_MIN_AVAILABLE_CURRICULUM_TOKENS:,} "
            f"sources={inventory.source_count} declared_token_sources={inventory.declared_token_sources} "
            f"skills={len(inventory.skills)}"
        )
    source_audit = audit_hf_sources(
        hf_specs, stage=stage, rows=args.source_audit_rows,
        min_rate=args.min_source_serialization_rate,
    )
    hf_sample, hf_fp = load_hf_sample(
        specs=hf_specs, stage=stage,
        limit=args.hf_sample_examples or int(cfg["hf_preflight_sample_examples"]),
        sample_path=hf_sample_path, metadata_path=hf_sample_meta,
        refresh=bool(args.resume and args.curriculum_upgrade),
    )
    print(
        f"[HF] sources={len(stage_specs(hf_specs, stage))} "
        f"accepted_preflight_sample={len(hf_sample):,} fingerprint={hf_fp[:12]}"
    )

    all_texts, removed = remove_exam_families(stable_unique(hf_sample), stage)
    print(f"[ExamHoldout] families={len(exam_questions(stage))} removed_from_preflight={removed}")
    selected = select_curriculum(all_texts, stage=stage, seed=SEED)
    texts = selected.texts
    print(
        f"[Curriculum] stage={stage} selected={len(texts):,}/{selected.total_available:,} "
        f"trading={selected.trading_available:,} reasoning={selected.reasoning_available:,} "
        f"math={selected.math_available:,} replay=0"
    )
    train_texts, val_texts = split_by_prompt_family(texts, val_fraction=float(cfg["val_split"]), seed=SEED)
    split_fp = split_fingerprint(train_texts, val_texts)
    print(f"[Split] train={len(train_texts):,} val={len(val_texts):,} family_isolated=true split={split_fp[:12]}")

    lineage_model = None
    lineage_tokenizer = None
    if stage != "foundation":
        if not args.init_bundle:
            raise RuntimeError(f"{stage}_requires_--init-bundle")
        lineage_model, lineage_tokenizer, manifest = load_model_bundle(args.init_bundle)
        if manifest.model_config != model_config(cfg, lineage_tokenizer):
            raise RuntimeError("init_bundle_model_config_mismatch")
    elif args.init_bundle:
        raise RuntimeError("foundation_stage_must_start_without_init_bundle")

    contract = {
        "trainer_version": TRAINER_VERSION,
        "training_state_version": TRAINING_STATE_VERSION,
        "preflight_manifest_version": PREFLIGHT_MANIFEST_VERSION,
        "foundation_contract_version": FOUNDATION_CONTRACT_VERSION,
        "profile": args.profile,
        "training_stage": stage,
        "profile_contract": profile_contract(cfg),
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
        changed_data_keys: list[str] = []
        for key, value in contract.items():
            if saved.get(key) == value:
                continue
            if contract_mismatch_allowed(key=key, curriculum_upgrade=args.curriculum_upgrade):
                changed_data_keys.append(key)
                continue
            raise RuntimeError(f"preflight_contract_mismatch:{key}:use_clean_bundle")
        if changed_data_keys:
            print(f"[CurriculumUpgrade] preflight data contract changed: {','.join(changed_data_keys)}")
        tokenizer = BPETokenizer.load(tokenizer_path)
        reused = True
    elif lineage_tokenizer is not None:
        tokenizer = lineage_tokenizer
        tokenizer.save(tokenizer_path)
        reused = False
    else:
        sample = build_tokenizer_training_sample(train_texts, max_chars=int(cfg["tokenizer_chars"]), seed=SEED)
        tokenizer = BPETokenizer()
        tokenizer.train(sample, vocab_size=int(cfg["vocab_size"]))
        tokenizer.save(tokenizer_path)
        reused = False
    print(f"[Tokenizer] reuse={str(reused).lower()} fingerprint={tokenizer.fingerprint()}")

    train_sequences = build_example_sequences(train_texts, tokenizer, seq_len=int(cfg["seq_len"]))
    val_sequences = build_example_sequences(val_texts, tokenizer, seq_len=int(cfg["seq_len"]))
    report = run_training_preflight(
        tokenizer=tokenizer, train_texts=train_texts, val_texts=val_texts,
        train_sequences=train_sequences, val_sequences=val_sequences,
        seq_len=int(cfg["seq_len"]),
    )
    tokenizer_stats = tokenizer_efficiency(train_texts, tokenizer)
    resolved_model_config = model_config(cfg, tokenizer)

    torch.manual_seed(SEED)
    random.seed(SEED)
    probe_model = VistaReasoningGPT(**resolved_model_config)
    total_params = probe_model.get_num_params()
    active_params = probe_model.get_active_params_per_token()
    activation_ratio = active_params / max(total_params, 1)
    target_tokens, target_tpp = target_prediction_tokens(stage, total_params, args.target_tokens_per_parameter)
    reference_tpp = target_tokens / max(total_params, 1) if stage == "foundation" else float(target_tpp)
    probe = benchmark_training_throughput(
        model_config=resolved_model_config, tokenizer=tokenizer, sequences=train_sequences,
        batch_size=int(cfg["batch_size"]), steps=args.compute_probe_steps,
        wall_clock_hours=FOUNDATION_EXAM_INTERVAL_SECONDS / 3600.0,
        reference_tokens_per_parameter=reference_tpp,
    )

    preflight = {
        **contract,
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "model_config": resolved_model_config,
        "total_parameters": total_params,
        "active_parameters_per_token": active_params,
        "activation_ratio": activation_ratio,
        "target_tokens_per_total_parameter": target_tpp,
        "target_prediction_tokens": target_tokens,
        "foundation_available_curriculum_tokens": inventory.available_tokens if stage == "foundation" else None,
        "exam_interval_seconds": FOUNDATION_EXAM_INTERVAL_SECONDS,
        "roundtrip_cases": report.roundtrip_cases,
        "overfit_initial_loss": report.overfit_initial_loss,
        "overfit_final_loss": report.overfit_final_loss,
        "tokenizer_efficiency": tokenizer_stats,
        "source_audit": source_audit,
        "compute_probe": probe.to_dict(),
        "exam_feedback_version": EXAM_FEEDBACK_VERSION,
        "hard_negative_objective_version": HARD_NEGATIVE_OBJECTIVE_VERSION,
        "semantic_checkpoint_policy_version": SEMANTIC_CHECKPOINT_POLICY_VERSION,
    }
    atomic_json_save(preflight, preflight_path)
    print(
        f"[Preflight] PASS roundtrips={report.roundtrip_cases} "
        f"overfit={report.overfit_initial_loss:.4f}->{report.overfit_final_loss:.4f} "
        f"objective=v{LOSS_OBJECTIVE_VERSION} semantic_checkpoint=v{SEMANTIC_CHECKPOINT_POLICY_VERSION}"
    )
    print(
        f"[Compute] measured_supervised_tokens/s={probe.useful_tokens_per_second:,.1f} "
        f"projected_4h={probe.projected_useful_tokens:,} target={target_tokens:,} "
        f"exam_interval=4h session_limit={'none' if args.session_hours is None else str(args.session_hours) + 'h'}"
    )
    if args.preflight_only:
        print("[Preflight] Full training intentionally not started. Verified artifacts are reusable.")
        return None

    return PreparedTraining(
        stage=stage, cfg=cfg, bundle=bundle, exams_dir=exams_dir,
        state_path=state_path, best_path=best_path, hf_specs=hf_specs,
        val_texts=val_texts, contract=contract, tokenizer=tokenizer,
        model_config=resolved_model_config, tokenizer_stats=tokenizer_stats,
        total_params=total_params, active_params=active_params,
        activation_ratio=activation_ratio, target_tokens=target_tokens,
        lineage_model=lineage_model, val_sequences=val_sequences,
    )
