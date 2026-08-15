from __future__ import annotations

from dataclasses import dataclass
import random

import torch

from src.language.canonical_contract import CanonicalMessage, serialize_messages
from src.language.loss_objective import build_loss_targets
from src.language.protocol import build_exam_prompt, extract_assistant_response, generation_stop_ids
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.tokenizer import BPETokenizer


@dataclass(frozen=True, slots=True)
class ConditioningStageResult:
    size: int
    recall: int
    optimizer_updates: int
    final_loss: float
    responses: tuple[tuple[str, str, str], ...]


def _cases() -> list[tuple[str, str]]:
    """Arbitrary one-token labels that cannot be passed by a global answer prior.

    Single-character targets make this a direct test of prompt -> next-token
    conditioning rather than a test of multi-token decoding endurance. The
    ordinary tiny-overfit gate separately checks full sequence learning.
    """
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
def _recall(
    model: VistaReasoningGPT,
    tokenizer: BPETokenizer,
    cases: list[tuple[str, str]],
) -> tuple[int, tuple[tuple[str, str, str], ...]]:
    was_training = model.training
    rows: list[tuple[str, str, str]] = []
    correct = 0
    stops = generation_stop_ids(tokenizer)
    try:
        model.eval()
        for prompt, expected in cases:
            prompt_ids = tokenizer.encode(build_exam_prompt(prompt), add_bos=False, add_eos=False)
            if len(prompt_ids) >= model.max_seq_len:
                raise RuntimeError("conditioning_prompt_exceeds_tiny_context")
            generated = model.generate(
                torch.tensor([prompt_ids], dtype=torch.long),
                max_new_tokens=6,
                stop_ids=stops,
                use_kv_cache=True,
                do_sample=False,
            )
            decoded = tokenizer.decode(generated[0].tolist(), skip_special=False)
            actual = extract_assistant_response(decoded).strip()
            rows.append((prompt, expected, actual))
            if actual == expected:
                correct += 1
    finally:
        model.train(was_training)
    return correct, tuple(rows)


def _run_stage(
    tokenizer: BPETokenizer,
    *,
    size: int,
    seed: int,
    max_updates: int,
    check_every: int,
) -> ConditioningStageResult:
    seq_len = 64
    cases = _cases()[:size]
    examples = _build_examples(tokenizer, cases, seq_len=seq_len)
    model = _new_model(tokenizer, seq_len=seq_len, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed + 1)

    updates = 0
    final_loss = float("inf")
    perfect_checks = 0
    while updates < max_updates:
        order = torch.randperm(len(examples), generator=generator).tolist()
        for start in range(0, len(order), min(8, len(order))):
            selected = order[start : start + min(8, len(order))]
            x = torch.stack([examples[index][0] for index in selected])
            y = torch.stack([examples[index][1] for index in selected])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, targets=y, pad_id=tokenizer.pad_id())
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("conditioning_gate_non_finite_loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            final_loss = float(loss.item())
            updates += 1

            if updates % check_every == 0 or updates >= max_updates:
                recalled, rows = _recall(model, tokenizer, cases)
                print(
                    f"[ConditioningGate] size={size} recall={recalled}/{size} "
                    f"updates={updates} loss={final_loss:.4f}"
                )
                if recalled == size:
                    perfect_checks += 1
                    if perfect_checks >= 2:
                        return ConditioningStageResult(size, recalled, updates, final_loss, rows)
                else:
                    perfect_checks = 0
            if updates >= max_updates:
                break

    recalled, rows = _recall(model, tokenizer, cases)
    return ConditioningStageResult(size, recalled, updates, final_loss, rows)


def run_prompt_conditioning_gate(tokenizer: BPETokenizer) -> tuple[int, int, int]:
    """Prove 1/1, 8/8 and 32/32 prompt conditioning before long training.

    Every cardinality starts from a fresh deterministic initialization. This
    avoids measuring curriculum carry-over/catastrophic forgetting inside the
    diagnostic itself. Training is convergence-driven rather than tied to one
    arbitrary epoch count, and exact generation recall remains the acceptance
    criterion.
    """
    random.seed(1701)
    recalls: list[int] = []
    configs = (
        (1, 1702, 400, 20),
        (8, 1709, 1600, 40),
        (32, 1733, 4000, 80),
    )
    for size, seed, max_updates, check_every in configs:
        result = _run_stage(
            tokenizer,
            size=size,
            seed=seed,
            max_updates=max_updates,
            check_every=check_every,
        )
        recalls.append(result.recall)
        if result.recall != size:
            for prompt, expected, actual in result.responses:
                marker = "OK" if expected == actual else "MISS"
                print(
                    f"[ConditioningGate:{marker}] prompt={prompt!r} "
                    f"expected={expected!r} actual={actual!r}"
                )
            raise RuntimeError(
                f"prompt_conditioning_gate_failed:{result.recall}/{size}:"
                f"updates={result.optimizer_updates}:loss={result.final_loss:.4f}"
            )
    return recalls[0], recalls[1], recalls[2]
