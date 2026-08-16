from __future__ import annotations

import random
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.language.exam import EpochExamResult
from src.language.exam_feedback import EXAM_FEEDBACK_VERSION, ExamFeedbackPolicy, derive_exam_feedback
from src.language.foundation_contract import FOUNDATION_CONTRACT_VERSION, FOUNDATION_EXAM_INTERVAL_SECONDS
from src.language.hard_negative_objective import HARD_NEGATIVE_OBJECTIVE_VERSION
from src.language.loss_objective import LOSS_OBJECTIVE_VERSION
from src.language.model_bundle import save_model_bundle
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.semantic_checkpointing import SEMANTIC_CHECKPOINT_POLICY_VERSION, mastery_report
from src.language.training_artifacts import (
    contract_mismatch_allowed,
    load_best_rank,
    promote_checkpoint,
    save_training_state,
)
from src.language.training_budget import cap_prediction_targets, set_token_scheduled_lr
from src.language.training_feedback_controller import adaptive_training_loss, feedback_summary
from src.language.training_pipeline import PackedSequenceDataset
from src.language.training_prepare import PreparedTraining
from src.language.training_session_contract import SEED, TRAINER_VERSION
from src.language.training_support import (
    exam_holdout_texts,
    new_optimizer,
    run_exam,
    stream_loader,
    validation_loss,
)


def run_training(prep: PreparedTraining, args: Any) -> None:
    stage, cfg, tokenizer = prep.stage, prep.cfg, prep.tokenizer
    target_tokens = prep.target_tokens
    torch.manual_seed(SEED)
    random.seed(SEED)
    model = prep.lineage_model if prep.lineage_model is not None else VistaReasoningGPT(**prep.model_config)
    optimizer = new_optimizer(model, cfg)
    val_loader = DataLoader(
        PackedSequenceDataset(prep.val_sequences, int(cfg["seq_len"]), tokenizer.pad_id()),
        batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0,
    )

    state_contract = {
        **prep.contract,
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "model_config": prep.model_config,
        "target_prediction_tokens": target_tokens,
        "exam_interval_seconds": FOUNDATION_EXAM_INTERVAL_SECONDS,
    }
    epoch = 0
    interval_steps = 0
    interval_elapsed_seconds = 0.0
    interval_loss_sum = 0.0
    interval_prediction_tokens = 0
    cumulative_tokens = 0
    optimizer_steps = 0
    best_val = float("inf")
    feedback = ExamFeedbackPolicy()
    best_semantic_rank = load_best_rank(prep.best_path)

    if args.resume:
        if not prep.state_path.exists():
            raise RuntimeError("resume_training_state_missing")
        state = torch.load(prep.state_path, map_location="cpu", weights_only=False)
        changed_state_data_keys: list[str] = []
        for key, value in state_contract.items():
            if state.get(key) == value:
                continue
            if contract_mismatch_allowed(key=key, curriculum_upgrade=args.curriculum_upgrade):
                changed_state_data_keys.append(key)
                continue
            raise RuntimeError(f"resume_contract_mismatch:{key}")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        epoch = int(state["epoch"])
        interval_steps = int(state.get("interval_steps", 0))
        interval_elapsed_seconds = float(state.get("interval_elapsed_seconds", 0.0))
        interval_loss_sum = float(state.get("interval_loss_sum", 0.0))
        interval_prediction_tokens = int(state.get("interval_prediction_tokens", 0))
        cumulative_tokens = int(state["cumulative_prediction_tokens"])
        optimizer_steps = int(state["optimizer_steps"])
        best_val = float(state["best_validation_loss"])
        feedback = ExamFeedbackPolicy.from_dict(state.get("exam_feedback"))
        if changed_state_data_keys:
            interval_steps = 0
            interval_elapsed_seconds = 0.0
            interval_loss_sum = 0.0
            interval_prediction_tokens = 0
            best_semantic_rank = None
            best_val = float("inf")
            prep.best_path.unlink(missing_ok=True)
            print(f"[CurriculumUpgrade] reset interval/checkpoint baseline: {','.join(changed_state_data_keys)}")
        torch.set_rng_state(state["torch_rng_state"])
        random.setstate(state["python_random_state"])
        print(
            f"[Resume] exams_completed={epoch} interval_steps={interval_steps} "
            f"interval_elapsed={interval_elapsed_seconds/3600:.2f}h "
            f"tokens={cumulative_tokens:,}/{target_tokens:,}"
        )
        print(f"[ExamFeedback] restored {feedback_summary(feedback)}")
    elif prep.state_path.exists():
        raise RuntimeError("training_state_already_exists:use_--resume_or_clean_bundle")

    excluded_stream_texts = list(prep.val_texts) + exam_holdout_texts(stage)
    train_loader = stream_loader(
        specs=prep.hf_specs, stage=stage, stream_generation=epoch,
        excluded=excluded_stream_texts, tokenizer=tokenizer, cfg=cfg, feedback=feedback,
    )
    iterator = iter(train_loader)
    if interval_steps > 0:
        for _ in range(interval_steps):
            try:
                next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                next(iterator)

    interval_started = time.monotonic() - interval_elapsed_seconds
    session_deadline = None if args.session_hours is None else time.monotonic() + args.session_hours * 3600.0
    previous_exam: EpochExamResult | None = None

    while cumulative_tokens < target_tokens:
        if session_deadline is not None and time.monotonic() >= session_deadline:
            elapsed = min(float(FOUNDATION_EXAM_INTERVAL_SECONDS), time.monotonic() - interval_started)
            save_training_state(
                prep.state_path, contract=state_contract, epoch=epoch, interval_steps=interval_steps,
                interval_elapsed_seconds=elapsed, interval_loss_sum=interval_loss_sum,
                interval_prediction_tokens=interval_prediction_tokens,
                cumulative_tokens=cumulative_tokens, optimizer_steps=optimizer_steps,
                best_val=best_val, model=model, optimizer=optimizer, feedback=feedback,
            )
            print(
                f"[SessionStop] optional safety override reached; tokens={cumulative_tokens:,}/{target_tokens:,} "
                f"exam_interval_progress={elapsed/3600:.2f}/4.00h. Resume with --resume."
            )
            return

        model.train()
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)

        y, valid = cap_prediction_targets(
            y, pad_id=tokenizer.pad_id(), remaining_tokens=target_tokens - cumulative_tokens,
        )
        if valid <= 0:
            continue
        set_token_scheduled_lr(
            optimizer, base_lr=float(cfg["lr"]), min_lr=float(cfg["lr_min"]),
            cumulative_tokens=cumulative_tokens, target_tokens=target_tokens,
            warmup_fraction=args.warmup_fraction,
        )
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(x, targets=y, pad_id=tokenizer.pad_id())
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("non_finite_training_loss")
        loss = adaptive_training_loss(
            base_loss=loss, logits=logits, input_ids=x, targets=y,
            pad_id=tokenizer.pad_id(), feedback=feedback,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non_finite_adaptive_training_loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
        optimizer.step()

        interval_steps += 1
        optimizer_steps += 1
        interval_loss_sum += float(loss.item()) * valid
        interval_prediction_tokens += valid
        cumulative_tokens += valid

        if interval_steps % int(cfg["checkpoint_every_steps"]) == 0:
            elapsed = min(float(FOUNDATION_EXAM_INTERVAL_SECONDS), time.monotonic() - interval_started)
            save_training_state(
                prep.state_path, contract=state_contract, epoch=epoch, interval_steps=interval_steps,
                interval_elapsed_seconds=elapsed, interval_loss_sum=interval_loss_sum,
                interval_prediction_tokens=interval_prediction_tokens,
                cumulative_tokens=cumulative_tokens, optimizer_steps=optimizer_steps,
                best_val=best_val, model=model, optimizer=optimizer, feedback=feedback,
            )
            print(f"[Checkpoint] tokens={cumulative_tokens:,}/{target_tokens:,} interval={elapsed/3600:.2f}/4.00h")

        if (time.monotonic() - interval_started) >= FOUNDATION_EXAM_INTERVAL_SECONDS and cumulative_tokens < target_tokens:
            if interval_prediction_tokens <= 0:
                raise RuntimeError("exam_interval_has_no_prediction_tokens")
            train_loss = interval_loss_sum / interval_prediction_tokens
            val_loss, val_tokens = validation_loss(model, val_loader, pad_id=tokenizer.pad_id())
            previous_exam = run_exam(
                model, tokenizer, epoch=epoch + 1, stage=stage,
                train_loss=train_loss, val_loss=val_loss, exams_dir=prep.exams_dir,
                previous=previous_exam, max_new_tokens=int(cfg["exam_max_new_tokens"]),
            )
            best_semantic_rank, promoted_val, promoted = promote_checkpoint(
                result=previous_exam, val_loss=val_loss, model=model, model_config=prep.model_config,
                cumulative_tokens=cumulative_tokens, best_path=prep.best_path,
                incumbent_rank=best_semantic_rank,
            )
            if promoted:
                best_val = promoted_val
            feedback = derive_exam_feedback(previous_exam)
            print(f"[ExamFeedback] next_interval {feedback_summary(feedback)}")
            mastery = mastery_report(previous_exam, stage)
            epoch += 1
            print(
                f"exam_epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_tokens={val_tokens:,} cumulative={cumulative_tokens:,}/{target_tokens:,} "
                f"skills_mastered={mastery.mastered_skills}/{len(mastery.skill_results)}"
            )
            interval_steps = 0
            interval_loss_sum = 0.0
            interval_prediction_tokens = 0
            interval_started = time.monotonic()
            train_loader = stream_loader(
                specs=prep.hf_specs, stage=stage, stream_generation=epoch,
                excluded=excluded_stream_texts, tokenizer=tokenizer, cfg=cfg, feedback=feedback,
            )
            iterator = iter(train_loader)
            save_training_state(
                prep.state_path, contract=state_contract, epoch=epoch, interval_steps=0,
                interval_elapsed_seconds=0.0, interval_loss_sum=0.0,
                interval_prediction_tokens=0, cumulative_tokens=cumulative_tokens,
                optimizer_steps=optimizer_steps, best_val=best_val,
                model=model, optimizer=optimizer, feedback=feedback,
            )

    if cumulative_tokens != target_tokens:
        raise RuntimeError(f"prediction_token_target_not_exact:{cumulative_tokens}!={target_tokens}")

    completion_train_loss = interval_loss_sum / interval_prediction_tokens if interval_prediction_tokens > 0 else None
    val_loss, _ = validation_loss(model, val_loader, pad_id=tokenizer.pad_id())
    completion_exam = run_exam(
        model, tokenizer, epoch=epoch + 1, stage=stage,
        train_loss=completion_train_loss, val_loss=val_loss, exams_dir=prep.exams_dir,
        previous=previous_exam, max_new_tokens=int(cfg["exam_max_new_tokens"]),
        prefix="target_completion_exam",
    )
    best_semantic_rank, promoted_val, promoted = promote_checkpoint(
        result=completion_exam, val_loss=val_loss, model=model, model_config=prep.model_config,
        cumulative_tokens=cumulative_tokens, best_path=prep.best_path,
        incumbent_rank=best_semantic_rank,
    )
    if promoted:
        best_val = promoted_val

    if not prep.best_path.exists():
        raise RuntimeError("semantic_checkpoint_selection_produced_no_best_model")
    best = torch.load(prep.best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    model.eval()
    final_exam = run_exam(
        model, tokenizer, epoch=int(best.get("epoch", epoch + 1)), stage=stage,
        train_loss=None, val_loss=float(best.get("validation_loss", best_val)),
        exams_dir=prep.exams_dir, previous=None,
        max_new_tokens=int(cfg["exam_max_new_tokens"]), prefix="best_model_exam",
    )
    final_mastery = mastery_report(final_exam, stage)
    metrics = {
        "trainer_version": TRAINER_VERSION,
        "foundation_contract_version": FOUNDATION_CONTRACT_VERSION,
        "loss_objective_version": LOSS_OBJECTIVE_VERSION,
        "hard_negative_objective_version": HARD_NEGATIVE_OBJECTIVE_VERSION,
        "exam_feedback_version": EXAM_FEEDBACK_VERSION,
        "semantic_checkpoint_policy_version": SEMANTIC_CHECKPOINT_POLICY_VERSION,
        "profile": args.profile,
        "training_stage": stage,
        "parameter_count": prep.total_params,
        "active_parameter_count": prep.active_params,
        "activation_ratio": prep.activation_ratio,
        "cumulative_prediction_tokens": cumulative_tokens,
        "target_prediction_tokens": target_tokens,
        "optimizer_steps": optimizer_steps,
        "best_validation_loss": float(best.get("validation_loss", best_val)),
        "tokenizer_efficiency": prep.tokenizer_stats,
        "stream_only": True,
        "exam_interval_hours": FOUNDATION_EXAM_INTERVAL_SECONDS / 3600.0,
        "session_deadline_default": None,
        "exact_prediction_token_target": True,
        "semantic_checkpoint_selection": True,
        "best_model_exam": {
            "correctness_percent": final_exam.correctness_percent,
            "mean_quality_percent": final_exam.mean_quality_percent,
            "gibberish_answers": final_exam.gibberish_answers,
            "training_signal": final_exam.training_signal,
            "mastery": final_mastery.to_dict(),
        },
    }
    save_model_bundle(
        bundle_dir=prep.bundle, model=model, tokenizer=tokenizer, model_config=prep.model_config,
        training_stage=stage, corpus_fingerprint=prep.contract["curriculum_fingerprint"], metrics=metrics,
    )
    prep.state_path.unlink(missing_ok=True)
    print(f"[TrainingComplete] exact_supervised_prediction_tokens={cumulative_tokens:,}")
    print(f"bundle={prep.bundle}")
