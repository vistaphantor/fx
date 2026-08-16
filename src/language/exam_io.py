from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.language.exam_types import EpochExamResult, TokenCandidate

BINOCULARS_RENDER_STEPS = 24


def _candidate_line(candidates: tuple[TokenCandidate, ...]) -> str:
    return " | ".join(f"#{c.rank} {c.token!r} p={c.probability:.4%} logit={c.logit:.4f}" for c in candidates)


def render_exam_text(result: EpochExamResult, *, previous: EpochExamResult | None = None) -> str:
    if previous is not None and previous.exam_contract_fingerprint != result.exam_contract_fingerprint:
        raise RuntimeError("exam_contract_changed_between_epochs")
    lines = [
        f"Vista Reasoner Exam v{result.exam_version}", f"Contract: {result.exam_contract_fingerprint[:16]}",
        f"Decoding: {result.decoding_mode}", f"Epoch: {result.epoch}", f"Stage: {result.training_stage}",
        f"Train loss: {result.train_loss}", f"Validation loss: {result.validation_loss}",
        f"Semantic correct: {result.correct_questions}/{result.total_questions} ({result.correctness_percent:.1f}%)",
        f"Surface quality: {result.mean_quality_percent:.1f}%", f"Gibberish: {result.gibberish_answers}/{result.total_questions}",
        f"Answer diversity: {result.answer_diversity_percent:.1f}%", f"Max exact collision: {result.max_answer_collision}",
        f"Max prefix collision: {result.max_prefix_collision}", f"Mode collapse: {'YES' if result.mode_collapse else 'NO'}",
        f"Training signal: {result.training_signal}", "", "BINOCULARS — CONCEPTUAL GATES / EXTENSION PROBES",
        f"Mean first-token Jensen-Shannon divergence: {result.mean_prompt_js_bits:.6f} bits",
        f"Min first-token Jensen-Shannon divergence: {result.min_prompt_js_bits:.6f} bits",
        f"Max first-token Jensen-Shannon divergence: {result.max_prompt_js_bits:.6f} bits",
        "", "BINOCULARS — PARAMETER HEALTH",
    ]
    for stat in result.parameter_health:
        lines.append(f"{stat.group}: params={stat.parameters:,} rms={stat.rms:.6f} l2={stat.l2_norm:.3f} max_abs={stat.max_abs:.6f}")
    if previous is not None:
        lines += [
            f"Correctness delta: {result.correctness_percent - previous.correctness_percent:+.1f}pp",
            f"Surface-quality delta: {result.mean_quality_percent - previous.mean_quality_percent:+.1f}pp",
            f"Gibberish delta: {result.gibberish_answers - previous.gibberish_answers:+d}",
            f"Diversity delta: {result.answer_diversity_percent - previous.answer_diversity_percent:+.1f}pp",
        ]
    for a in result.answers:
        gate = " conceptual-gate" if a.conceptual_gate else ""
        lines += [
            "", "=" * 80, f"QUESTION [{a.question_id}] skill={a.skill or 'extension'}{gate} category={a.category}", a.prompt,
            "", "RAW OUTPUT", a.raw_output, "", "NORMALIZED OUTPUT", a.normalized_output, "",
            f"Semantically correct: {'YES' if a.correct else 'NO'}", f"Surface quality: {a.quality_score * 100:.1f}%",
            f"Repetition: {a.repetition_ratio:.3f}", f"Flags: {', '.join(a.gibberish_flags) if a.gibberish_flags else 'none'}",
            f"Generated tokens: {a.generated_tokens}",
        ]
        if a.decision_trace:
            total_evaluations = sum(step.candidates_evaluated for step in a.decision_trace)
            lines += ["", "MODEL DECISION TRACE", f"Token alternatives scored: {len(a.decision_trace)} steps; total vocabulary evaluations={total_evaluations:,}"]
            for step in a.decision_trace[:BINOCULARS_RENDER_STEPS]:
                lines.append(f"step={step.step:02d} chose={step.chosen_token!r} p={step.chosen_probability:.4%} logit={step.chosen_logit:.4f} entropy={step.entropy_bits:.4f}b margin={step.winner_margin_probability:.4%}")
                lines.append("  " + _candidate_line(step.top_candidates))
            if len(a.decision_trace) > BINOCULARS_RENDER_STEPS:
                lines.append(f"... {len(a.decision_trace) - BINOCULARS_RENDER_STEPS} later decision steps omitted from TXT; full trace is in JSON.")
        if a.target_trace:
            mean_rank = sum(step.target_rank for step in a.target_trace) / len(a.target_trace)
            lines += ["", "CORRECT TARGET PATH (teacher-forced diagnostic only; does not affect training)", f"Mean correct-token rank: {mean_rank:.2f}/{a.target_trace[0].candidates_evaluated}"]
            for step in a.target_trace:
                lines.append(f"step={step.step:02d} target={step.target_token!r} rank={step.target_rank}/{step.candidates_evaluated} p={step.target_probability:.4%} logit={step.target_logit:.4f} | winner={step.winning_token!r} p={step.winning_probability:.4%} entropy={step.entropy_bits:.4f}b")
                lines.append("  " + _candidate_line(step.top_candidates))
    return "\n".join(lines) + "\n"


def save_epoch_exam(*, result: EpochExamResult, exams_dir: str | Path, previous: EpochExamResult | None = None, prefix: str | None = None) -> tuple[Path, Path]:
    directory = Path(exams_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = prefix or f"epoch_{result.epoch:03d}_exam"
    text_path, json_path = directory / f"{stem}.txt", directory / f"{stem}.json"
    text_path.write_text(render_exam_text(result, previous=previous), encoding="utf-8")
    json_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return text_path, json_path
