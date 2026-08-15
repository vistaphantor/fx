from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os
import random

import torch

from src.language.canonical_contract import CanonicalMessage, serialize_messages
from src.language.loss_objective import build_loss_targets
from src.language.protocol import build_exam_prompt, extract_assistant_response, generation_stop_ids
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.tokenizer import BPETokenizer


@dataclass(frozen=True, slots=True)
class ConditioningCheck:
    optimizer_updates: int
    recall: int
    full_loss: float
    learning_rate: float


@dataclass(frozen=True, slots=True)
class ConditioningResponse:
    prompt: str
    expected: str
    actual: str
    expected_token_id: int
    expected_rank: int
    expected_probability: float
    winner: str
    winner_token_id: int
    winner_probability: float
    margin: float


@dataclass(frozen=True, slots=True)
class ConditioningStageResult:
    size: int
    recall: int
    optimizer_updates: int
    final_loss: float
    responses: tuple[ConditioningResponse, ...]
    checks: tuple[ConditioningCheck, ...]


def _cases() -> list[tuple[str, str]]:
    """Arbitrary one-token labels that cannot be passed by a global answer prior."""
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ012345")
    return [
        (f"Conditioning key {index:02d}. Reply with its assigned label only.", label)
        for index, label in enumerate(labels)
    ]


def _new_model(tokenizer: BPETokenizer, *, seq_len: int, seed: int) -> VistaReasoningGPT:
    torch.manual_seed(seed)
    return VistaReasoningGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=192,
        max_seq_len=seq_len,
        dropout=0.0,
        ffn_type="dense",
        num_experts=1,
        experts_per_token=1,
        moe_ffn_dim=192,
        shared_expert_ffn_dim=0,
        router_aux_loss_coef=0.0,
        router_jitter=0.0,
    )


def _build_examples(
    tokenizer: BPETokenizer,
    cases: list[tuple[str, str]],
    *,
    seq_len: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    examples: list[tuple[torch.Tensor, torch.Tensor]] = []
    for prompt, answer in cases:
        answer_ids = tokenizer.encode(answer, add_bos=False, add_eos=False)
        if len(answer_ids) != 1:
            raise RuntimeError(
                f"conditioning_label_must_be_one_token:{answer!r}:{answer_ids}"
            )
        text = serialize_messages((
            CanonicalMessage("user", prompt),
            CanonicalMessage("assistant", answer),
        ))
        ids = tokenizer.encode(text, add_bos=False, add_eos=False)
        if len(ids) < 2 or len(ids) > seq_len + 1:
            raise RuntimeError("conditioning_example_length_invalid")
        x, y, stats = build_loss_targets(ids, seq_len=seq_len, pad_id=tokenizer.pad_id())
        if stats.prediction_tokens <= 0:
            raise RuntimeError("conditioning_example_has_no_supervision")
        examples.append((torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)))
    return examples


@torch.no_grad()
def _full_dataset_loss(
    model: VistaReasoningGPT,
    tokenizer: BPETokenizer,
    examples: list[tuple[torch.Tensor, torch.Tensor]],
) -> float:
    was_training = model.training
    try:
        model.eval()
        x = torch.stack([example[0] for example in examples])
        y = torch.stack([example[1] for example in examples])
        _, loss = model(x, targets=y, pad_id=tokenizer.pad_id())
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("conditioning_gate_full_loss_invalid")
        return float(loss.item())
    finally:
        model.train(was_training)


@torch.no_grad()
def _recall(
    model: VistaReasoningGPT,
    tokenizer: BPETokenizer,
    cases: list[tuple[str, str]],
) -> tuple[int, tuple[ConditioningResponse, ...]]:
    was_training = model.training
    rows: list[ConditioningResponse] = []
    correct = 0
    stops = generation_stop_ids(tokenizer)
    try:
        model.eval()
        for prompt, expected in cases:
            prompt_ids = tokenizer.encode(
                build_exam_prompt(prompt), add_bos=False, add_eos=False,
            )
            if len(prompt_ids) >= model.max_seq_len:
                raise RuntimeError("conditioning_prompt_exceeds_tiny_context")
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)
            logits, _ = model(prompt_tensor)
            next_logits = logits[0, len(prompt_ids) - 1].float()
            probabilities = torch.softmax(next_logits, dim=-1)

            expected_ids = tokenizer.encode(expected, add_bos=False, add_eos=False)
            if len(expected_ids) != 1:
                raise RuntimeError("conditioning_expected_not_single_token")
            expected_id = int(expected_ids[0])
            target_logit = float(next_logits[expected_id].item())
            expected_rank = 1 + int((next_logits > next_logits[expected_id]).sum().item())
            expected_probability = float(probabilities[expected_id].item())
            winner_id = int(torch.argmax(next_logits).item())
            winner_probability = float(probabilities[winner_id].item())
            winner = tokenizer.decode([winner_id], skip_special=False, errors="replace")

            generated = model.generate(
                prompt_tensor,
                max_new_tokens=6,
                stop_ids=stops,
                use_kv_cache=True,
                do_sample=False,
            )
            decoded = tokenizer.decode(generated[0].tolist(), skip_special=False)
            actual = extract_assistant_response(decoded).strip()
            if actual == expected:
                correct += 1

            rows.append(ConditioningResponse(
                prompt=prompt,
                expected=expected,
                actual=actual,
                expected_token_id=expected_id,
                expected_rank=expected_rank,
                expected_probability=expected_probability,
                winner=winner,
                winner_token_id=winner_id,
                winner_probability=winner_probability,
                margin=float(next_logits[winner_id].item()) - target_logit,
            ))
    finally:
        model.train(was_training)
    return correct, tuple(rows)


def _set_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _run_stage(
    tokenizer: BPETokenizer,
    *,
    size: int,
    seed: int,
    max_updates: int,
    check_every: int,
) -> ConditioningStageResult:
    """Train an independent exact-mapping gate with deterministic full batches.

    Mini-batch loss was previously misleading: a sampled batch could have near-zero
    loss while omitted mappings were wrong. Full-batch optimization and full-dataset
    evaluation make the gate measure conditioning rather than stochastic interference.
    """
    seq_len = 64
    cases = _cases()[:size]
    examples = _build_examples(tokenizer, cases, seq_len=seq_len)
    model = _new_model(tokenizer, seq_len=seq_len, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    x_all = torch.stack([example[0] for example in examples])
    y_all = torch.stack([example[1] for example in examples])

    updates = 0
    perfect_checks = 0
    checks: list[ConditioningCheck] = []
    best_recall = -1
    checks_without_recall_gain = 0
    learning_rate = 3e-3

    while updates < max_updates:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x_all, targets=y_all, pad_id=tokenizer.pad_id())
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("conditioning_gate_non_finite_loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        updates += 1

        if updates % check_every != 0 and updates < max_updates:
            continue

        recalled, rows = _recall(model, tokenizer, cases)
        full_loss = _full_dataset_loss(model, tokenizer, examples)
        checks.append(ConditioningCheck(updates, recalled, full_loss, learning_rate))
        print(
            f"[ConditioningGate] size={size} recall={recalled}/{size} "
            f"updates={updates} full_loss={full_loss:.6f} lr={learning_rate:.6g}"
        )

        if recalled > best_recall:
            best_recall = recalled
            checks_without_recall_gain = 0
        else:
            checks_without_recall_gain += 1

        if recalled == size:
            perfect_checks += 1
            if perfect_checks >= 2:
                return ConditioningStageResult(
                    size, recalled, updates, full_loss, rows, tuple(checks)
                )
        else:
            perfect_checks = 0

        # Once exact recall stalls, consolidate instead of continuing high-LR
        # updates that can overwrite already learned arbitrary mappings.
        if checks_without_recall_gain >= 4 and learning_rate > 5e-4:
            learning_rate = max(5e-4, learning_rate * 0.5)
            _set_lr(optimizer, learning_rate)
            checks_without_recall_gain = 0
            print(
                f"[ConditioningGate] size={size} consolidation_lr={learning_rate:.6g}"
            )

    recalled, rows = _recall(model, tokenizer, cases)
    full_loss = _full_dataset_loss(model, tokenizer, examples)
    if not checks or checks[-1].optimizer_updates != updates:
        checks.append(ConditioningCheck(updates, recalled, full_loss, learning_rate))
    return ConditioningStageResult(
        size, recalled, updates, full_loss, rows, tuple(checks)
    )


def _render_stage_result(
    result: ConditioningStageResult,
    *,
    tokenizer: BPETokenizer,
) -> str:
    status = "PASS" if result.recall == result.size else "FAIL"
    lines = [
        "Vista Prompt Conditioning Gate",
        f"Stage size: {result.size}",
        f"Status: {status}",
        f"Recall: {result.recall}/{result.size}",
        f"Optimizer updates: {result.optimizer_updates}",
        f"Final full-dataset loss: {result.final_loss:.8f}",
        f"Tokenizer vocab: {tokenizer.vocab_size}",
        f"Tokenizer fingerprint: {tokenizer.fingerprint()}",
        "",
        "CONVERGENCE TRACE",
        "update\trecall\tfull_loss\tlr",
    ]
    for check in result.checks:
        lines.append(
            f"{check.optimizer_updates}\t{check.recall}/{result.size}\t"
            f"{check.full_loss:.8f}\t{check.learning_rate:.8g}"
        )

    lines.extend(("", "CASE RESULTS"))
    for index, row in enumerate(result.responses, start=1):
        marker = "OK" if row.expected == row.actual else "MISS"
        lines.extend((
            "",
            "-" * 80,
            f"CASE {index:02d} [{marker}]",
            f"PROMPT: {row.prompt}",
            f"EXPECTED: {row.expected!r}",
            f"ACTUAL: {row.actual!r}",
            f"EXPECTED TOKEN: id={row.expected_token_id} rank={row.expected_rank}/{tokenizer.vocab_size} "
            f"p={row.expected_probability:.8%}",
            f"WINNER: {row.winner!r} id={row.winner_token_id} p={row.winner_probability:.8%}",
            f"WINNER-TARGET LOGIT MARGIN: {row.margin:.8f}",
        ))

    lines.extend((
        "",
        "INTERPRETATION",
        "This gate tests exact prompt-to-answer conditioning, not arithmetic or world knowledge.",
        "Loss values are full-dataset teacher-forced losses, not last-mini-batch losses.",
        "Expected-token rank/probability shows whether a wrong answer was narrowly beaten or never learned.",
        "A PASS requires two consecutive perfect-recall checks for this stage size.",
        "A FAIL means the long training run remains blocked at this cardinality.",
    ))
    return "\n".join(lines) + "\n"


def _save_stage_result(
    result: ConditioningStageResult,
    *,
    tokenizer: BPETokenizer,
    diagnostics_dir: str | Path,
) -> Path:
    directory = Path(diagnostics_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"conditioning_stage_{result.size:02d}.txt"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_render_stage_result(result, tokenizer=tokenizer), encoding="utf-8")
    os.replace(tmp, path)
    return path


def run_prompt_conditioning_gate(
    tokenizer: BPETokenizer,
    *,
    diagnostics_dir: str | Path,
) -> tuple[int, int, int]:
    """Prove 1/1, 8/8 and 32/32 prompt conditioning before long training."""
    random.seed(1701)
    recalls: list[int] = []
    configs = (
        (1, 1702, 400, 20),
        (8, 1709, 1200, 40),
        (32, 1733, 3000, 80),
    )
    for size, seed, max_updates, check_every in configs:
        result = _run_stage(
            tokenizer,
            size=size,
            seed=seed,
            max_updates=max_updates,
            check_every=check_every,
        )
        stage_path = _save_stage_result(
            result,
            tokenizer=tokenizer,
            diagnostics_dir=diagnostics_dir,
        )
        print(f"[ConditioningGate] stage_file={stage_path}")
        recalls.append(result.recall)
        if result.recall != size:
            for row in result.responses:
                marker = "OK" if row.expected == row.actual else "MISS"
                print(
                    f"[ConditioningGate:{marker}] prompt={row.prompt!r} "
                    f"expected={row.expected!r} actual={row.actual!r} "
                    f"rank={row.expected_rank}/{tokenizer.vocab_size} "
                    f"p={row.expected_probability:.6%}"
                )
            raise RuntimeError(
                f"prompt_conditioning_gate_failed:{result.recall}/{size}:"
                f"updates={result.optimizer_updates}:full_loss={result.final_loss:.6f}:"
                f"diagnostics={stage_path}"
            )
    return recalls[0], recalls[1], recalls[2]
