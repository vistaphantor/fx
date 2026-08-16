from __future__ import annotations

import math
from collections import defaultdict

import torch

from src.language.exam_types import DecisionTrace, ExamQuestion, ParameterHealth, TargetTokenTrace, TokenCandidate
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.tokenizer import BPETokenizer

BINOCULARS_TOP_K = 8


def _token_text(tokenizer: BPETokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special=False, errors="replace")


def _distribution_snapshot(logits: torch.Tensor, tokenizer: BPETokenizer, *, top_k: int = BINOCULARS_TOP_K) -> tuple[torch.Tensor, float, tuple[TokenCandidate, ...]]:
    values = logits.float()
    probs = torch.softmax(values, dim=-1)
    entropy = float((-(probs * torch.log2(probs.clamp_min(1e-30))).sum()).item())
    top_prob, top_ids = torch.topk(probs, min(top_k, probs.numel()))
    candidates = tuple(
        TokenCandidate(rank, int(token_id), _token_text(tokenizer, int(token_id)), float(values[int(token_id)].item()), float(probability))
        for rank, (probability, token_id) in enumerate(zip(top_prob.tolist(), top_ids.tolist()), start=1)
    )
    return probs, entropy, candidates


@torch.no_grad()
def _trace_generated_decisions(model: VistaReasoningGPT, tokenizer: BPETokenizer, prompt_ids: list[int], continuation: list[int]) -> tuple[DecisionTrace, ...]:
    if not continuation:
        return ()
    full = prompt_ids + continuation
    context = full[:-1]
    if len(context) > model.max_seq_len:
        context = context[-model.max_seq_len:]
        prompt_offset = max(1, len(prompt_ids) - (len(full) - 1 - len(context)))
    else:
        prompt_offset = len(prompt_ids)
    logits, _ = model(torch.tensor([context], dtype=torch.long))
    traces: list[DecisionTrace] = []
    start = prompt_offset - 1
    for step, token_id in enumerate(continuation):
        position = start + step
        if position < 0 or position >= logits.shape[1]:
            continue
        row = logits[0, position]
        probs, entropy, candidates = _distribution_snapshot(row, tokenizer)
        chosen_probability = float(probs[token_id].item())
        runner = candidates[1].probability if len(candidates) > 1 else 0.0
        traces.append(DecisionTrace(
            step + 1, int(token_id), _token_text(tokenizer, token_id), float(row[token_id].item()),
            chosen_probability, entropy, chosen_probability - runner, tokenizer.vocab_size, candidates,
        ))
    return tuple(traces)


@torch.no_grad()
def _trace_target_path(model: VistaReasoningGPT, tokenizer: BPETokenizer, prompt_ids: list[int], target_text: str | None) -> tuple[TargetTokenTrace, ...]:
    if not target_text:
        return ()
    target_ids = tokenizer.encode(target_text, add_bos=False, add_eos=False)
    available = model.max_seq_len - len(prompt_ids)
    if not target_ids or available <= 0:
        return ()
    target_ids = target_ids[:available]
    context = prompt_ids + target_ids[:-1]
    logits, _ = model(torch.tensor([context], dtype=torch.long))
    traces: list[TargetTokenTrace] = []
    start = len(prompt_ids) - 1
    for step, token_id in enumerate(target_ids):
        row = logits[0, start + step]
        probs, entropy, candidates = _distribution_snapshot(row, tokenizer)
        winner = candidates[0]
        traces.append(TargetTokenTrace(
            step + 1, int(token_id), _token_text(tokenizer, token_id),
            int((row > row[token_id]).sum().item()) + 1, float(row[token_id].item()),
            float(probs[token_id].item()), winner.token_id, winner.token, winner.probability,
            entropy, tokenizer.vocab_size, candidates,
        ))
    return tuple(traces)


@torch.no_grad()
def _first_token_distribution(model: VistaReasoningGPT, prompt_ids: list[int]) -> torch.Tensor:
    logits, _ = model(torch.tensor([prompt_ids], dtype=torch.long))
    return torch.softmax(logits[0, -1].float(), dim=-1)


def _js_bits(a: torch.Tensor, b: torch.Tensor) -> float:
    m = 0.5 * (a + b)
    kl_a = (a * torch.log2((a / m.clamp_min(1e-30)).clamp_min(1e-30))).sum()
    kl_b = (b * torch.log2((b / m.clamp_min(1e-30)).clamp_min(1e-30))).sum()
    return float((0.5 * (kl_a + kl_b)).item())


def _prompt_sensitivity(distributions: list[torch.Tensor]) -> tuple[float, float, float]:
    divergences = [_js_bits(distributions[i], distributions[j]) for i in range(len(distributions)) for j in range(i + 1, len(distributions))]
    if not divergences:
        return 0.0, 0.0, 0.0
    return sum(divergences) / len(divergences), min(divergences), max(divergences)


def _parameter_group(name: str) -> str:
    if name.startswith("tok_emb") or name.startswith("lm_head"):
        return "token_embedding_tied_head"
    for label in ("q_proj", "k_proj", "v_proj", "out_proj"):
        if f".attn.{label}." in name:
            return f"attention_{label}"
    for label in ("gate_proj", "up_proj", "down_proj"):
        if f".{label}." in name:
            return f"ffn_{label}"
    if ".router." in name:
        return "moe_router"
    if "norm" in name or name.startswith("ln_f"):
        return "normalization"
    return "other"


@torch.no_grad()
def _parameter_health(model: VistaReasoningGPT) -> tuple[ParameterHealth, ...]:
    stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for name, parameter in model.named_parameters():
        data = parameter.detach().float()
        group = _parameter_group(name)
        stats[group][0] += float(data.numel())
        stats[group][1] += float((data * data).sum().item())
        stats[group][2] = max(stats[group][2], float(data.abs().max().item()))
    return tuple(
        ParameterHealth(group, int(count), math.sqrt(sum_sq / max(int(count), 1)), math.sqrt(sum_sq), max_abs)
        for group, (count, sum_sq, max_abs) in sorted(stats.items())
    )


def _deep_diagnostic(question: ExamQuestion) -> bool:
    return bool(question.conceptual_gate or question.skill is None)
