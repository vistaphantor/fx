from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from src.language.protocol import build_exam_prompt
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.tokenizer import BPETokenizer, ENDASSISTANT, EOS

EXAM_VERSION = 10
EXAM_DECODING_MODE = "greedy_argmax_v1"
BINOCULARS_TOP_K = 8
BINOCULARS_RENDER_STEPS = 24


@dataclass(frozen=True)
class ExamQuestion:
    question_id: str
    category: str
    prompt: str
    expected_all: tuple[str, ...] = ()
    expected_any: tuple[str, ...] = ()
    numeric_answer: str | None = None
    expected_expression: str | None = None
    numeric_units: tuple[str, ...] = ()
    diagnostic_target: str | None = None


@dataclass(frozen=True)
class TokenCandidate:
    rank: int
    token_id: int
    token: str
    logit: float
    probability: float


@dataclass(frozen=True)
class DecisionTrace:
    step: int
    chosen_token_id: int
    chosen_token: str
    chosen_logit: float
    chosen_probability: float
    entropy_bits: float
    winner_margin_probability: float
    candidates_evaluated: int
    top_candidates: tuple[TokenCandidate, ...]


@dataclass(frozen=True)
class TargetTokenTrace:
    step: int
    target_token_id: int
    target_token: str
    target_rank: int
    target_logit: float
    target_probability: float
    winning_token_id: int
    winning_token: str
    winning_probability: float
    entropy_bits: float
    candidates_evaluated: int
    top_candidates: tuple[TokenCandidate, ...]


@dataclass(frozen=True)
class ParameterHealth:
    group: str
    parameters: int
    rms: float
    l2_norm: float
    max_abs: float


@dataclass(frozen=True)
class ExamAnswer:
    question_id: str
    category: str
    prompt: str
    raw_output: str
    normalized_output: str
    generated_tokens: int
    correct: bool
    quality_score: float
    repetition_ratio: float
    gibberish_flags: tuple[str, ...]
    decision_trace: tuple[DecisionTrace, ...] = ()
    target_trace: tuple[TargetTokenTrace, ...] = ()


@dataclass(frozen=True)
class EpochExamResult:
    exam_version: int
    exam_contract_fingerprint: str
    decoding_mode: str
    epoch: int
    training_stage: str
    train_loss: float | None
    validation_loss: float | None
    total_questions: int
    correct_questions: int
    correctness_percent: float
    mean_quality_percent: float
    gibberish_answers: int
    mean_generated_tokens: float
    training_signal: str
    answer_diversity_percent: float
    max_answer_collision: int
    max_prefix_collision: int
    mode_collapse: bool
    answers: tuple[ExamAnswer, ...]
    mean_prompt_js_bits: float = 0.0
    min_prompt_js_bits: float = 0.0
    max_prompt_js_bits: float = 0.0
    parameter_health: tuple[ParameterHealth, ...] = ()


FOUNDATION_EXAM: tuple[ExamQuestion, ...] = (
    ExamQuestion("arithmetic_2_plus_2", "primitive_arithmetic", "What is 2 + 2?", numeric_answer="4", expected_expression="2 + 2", diagnostic_target="4."),
    ExamQuestion("arithmetic_chain", "primitive_arithmetic", "What is 2 + 2 + 3?", numeric_answer="7", expected_expression="2 + 2 + 3", diagnostic_target="7."),
    ExamQuestion("arithmetic_negative", "primitive_arithmetic", "What is 9 - 12?", numeric_answer="-3", expected_expression="9 - 12", diagnostic_target="-3."),
    ExamQuestion("arithmetic_multiply", "primitive_arithmetic", "What is 3 times 4?", numeric_answer="12", expected_expression="3 times 4", diagnostic_target="12."),
    ExamQuestion("number_successor", "number_sense", "What number comes after 4?", numeric_answer="5", diagnostic_target="5."),
    ExamQuestion(
        "economics_price", "foundation_economics", "In simple economics, what is a price?",
        expected_any=("money", "amount", "paid", "asked"),
        diagnostic_target="A price is the amount of money asked or paid for a good or service.",
    ),
    ExamQuestion(
        "economics_profit", "foundation_economics",
        "A business receives 20 shillings and has costs of 7 shillings. What is its profit?",
        numeric_answer="13", numeric_units=("shilling", "shillings"), diagnostic_target="13 shillings.",
    ),
    ExamQuestion(
        "economics_inflation", "foundation_economics",
        "If prices rise while income stays the same, what happens to purchasing power?",
        expected_all=("purchasing", "power"),
        expected_any=("falls", "fall", "decreases", "declines", "lower"),
        diagnostic_target="Purchasing power falls.",
    ),
    ExamQuestion(
        "grammar_subject_verb", "grammar",
        "Which sentence is grammatical? A) The children are playing outside. B) The children is playing outside.",
        expected_any=("a", "children are"),
        diagnostic_target="A. The children are playing outside.",
    ),
    ExamQuestion(
        "semantic_plausibility_eat", "semantic_plausibility",
        "Which sentence makes sense? A) They ate lunch in the park. B) They ate the park.",
        expected_any=("a", "lunch"),
        diagnostic_target="A. They ate lunch in the park.",
    ),
    ExamQuestion(
        "repetition_control", "language_control",
        "Answer once, in one short sentence: What does a key normally open?",
        expected_any=("lock", "door"),
        diagnostic_target="A key normally opens a lock or a door.",
    ),
    ExamQuestion(
        "creative_kite", "creativity",
        "Write one sensible sentence about a red kite in the sky.",
        expected_all=("kite", "sky"),
        diagnostic_target="A red kite floats high in the sky.",
    ),
)

REASONING_EXTENSION: tuple[ExamQuestion, ...] = (
    ExamQuestion("algebra_simple", "algebra", "If 2x + 5 = 11, what is x?", expected_all=("x", "3"), diagnostic_target="x = 3."),
    ExamQuestion("logic_youngest", "logic", "Alice is older than Bob, and Bob is older than Charlie. Who is youngest?", expected_all=("charlie",), diagnostic_target="Charlie is youngest."),
)

TRADING_EXTENSION: tuple[ExamQuestion, ...] = (
    ExamQuestion("trading_bullish", "trading_language", "In trading, what does bullish mean?", expected_any=("rise", "rising", "higher", "upward", "increase", "buyers"), diagnostic_target="Bullish means prices are expected to rise."),
    ExamQuestion("trading_risk", "trading_language", "What does risk mean in trading?", expected_any=("loss", "lose", "uncertainty", "exposure", "capital"), diagnostic_target="Risk is the possibility of loss or uncertainty in a trade."),
    ExamQuestion("trading_spread", "trading_language", "What is the bid-ask spread?", expected_all=("bid", "ask"), expected_any=("difference", "distance", "gap"), diagnostic_target="The bid-ask spread is the difference between the bid and ask prices."),
    ExamQuestion("trading_atr", "trading_language", "What does ATR measure in market analysis?", expected_any=("volatility", "range", "true range"), diagnostic_target="ATR measures market volatility using true range."),
)


def exam_questions(training_stage: str) -> tuple[ExamQuestion, ...]:
    stage = training_stage.strip().casefold()
    if stage == "foundation":
        return FOUNDATION_EXAM
    if stage == "reasoning":
        return FOUNDATION_EXAM + REASONING_EXTENSION
    if stage == "trading_reasoning":
        return FOUNDATION_EXAM + REASONING_EXTENSION + TRADING_EXTENSION
    raise ValueError(f"unsupported_exam_stage:{training_stage}")


def exam_contract_payload(training_stage: str) -> dict:
    stage = training_stage.strip().casefold()
    return {
        "exam_version": EXAM_VERSION,
        "decoding_mode": EXAM_DECODING_MODE,
        "training_stage": stage,
        "questions": [{**asdict(q), "serialized_prompt": build_exam_prompt(q.prompt)} for q in exam_questions(stage)],
        "stop_tokens": [ENDASSISTANT, EOS],
        "semantic_grading": "expression_bound_numeric_v3+language_probes_v1",
        "collapse_detection": "exact_and_prefix_v1",
        "binoculars": "logit_trace_target_rank_prompt_js_weight_health_v1",
    }


def exam_contract_fingerprint(training_stage: str) -> str:
    payload = json.dumps(exam_contract_payload(training_stage), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def exam_prompt_families(training_stage: str) -> set[str]:
    from src.language.canonical_contract import prompt_family_key
    return {prompt_family_key(q.prompt) for q in exam_questions(training_stage)}


def _normalize_output(value: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", str(value or ""), flags=re.I | re.S)
    text = re.sub(r"</?[^>]+>", " ", text).replace("\ufffd", " ")
    return re.sub(r"\s+", " ", text).strip().casefold()


def _norm_math(value: str) -> str:
    text = value.casefold().replace("multiplied by", "*").replace("times", "*").replace("plus", "+").replace("minus", "-")
    return re.sub(r"\s+", "", text)


def _numeric_answer_matches(question: ExamQuestion, normalized: str) -> bool:
    expected = str(question.numeric_answer)
    cleaned = normalized.strip(" \t\r\n.!,:;")
    if cleaned == expected:
        return True
    if question.numeric_units:
        units = "|".join(re.escape(unit) for unit in question.numeric_units)
        if re.fullmatch(rf"{re.escape(expected)}\s+(?:{units})", cleaned):
            return True
    if re.fullmatch(rf"(?:the\s+)?(?:answer|result|value)\s+(?:is|=)\s*{re.escape(expected)}", cleaned):
        return True
    if question.expected_expression and "=" in cleaned:
        lhs, rhs = cleaned.rsplit("=", 1)
        rhs = rhs.strip(" .!,:;")
        if rhs == expected and _norm_math(lhs) == _norm_math(question.expected_expression):
            return True
    return False


def _is_correct(question: ExamQuestion, normalized: str, flags: tuple[str, ...]) -> bool:
    if not normalized or flags:
        return False
    if question.numeric_answer is not None:
        return _numeric_answer_matches(question, normalized)

    def present(term: str) -> bool:
        term = term.casefold().strip()
        return bool(term and re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized))

    if question.expected_all and not all(present(term) for term in question.expected_all):
        return False
    if question.expected_any and not any(present(term) for term in question.expected_any):
        return False
    return bool(question.expected_all or question.expected_any)


def _character_repetition_ratio(text: str, ngram: int = 4) -> float:
    compact = re.sub(r"\s+", "", text.casefold())
    if len(compact) < 12:
        return 0.0
    grams = [compact[i:i + ngram] for i in range(len(compact) - ngram + 1)]
    return 1.0 - len(set(grams)) / max(len(grams), 1)


def _periodic_repetition_ratio(text: str) -> float:
    compact = re.sub(r"\s+", "", text.casefold())
    if len(compact) < 16:
        return 0.0
    best = 0.0
    for width in range(1, min(13, len(compact) // 3 + 1)):
        comparisons = len(compact) - width
        if comparisons > 0:
            matches = sum(compact[i] == compact[i - width] for i in range(width, len(compact)))
            best = max(best, matches / comparisons)
    return best


def _quality(value: str) -> tuple[float, float, tuple[str, ...]]:
    raw = str(value or "")
    normalized = _normalize_output(raw)
    flags: list[str] = []
    if "\ufffd" in raw:
        flags.append("invalid_utf8_bytes")
    if not normalized:
        return 0.0, 1.0, tuple(flags + ["empty"])
    words = re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", normalized)
    if not words:
        return 0.0, 1.0, tuple(flags + ["no_lexical_content"])
    word_rep = 1.0 - len(set(words)) / len(words)
    char_rep = _character_repetition_ratio(normalized)
    periodic = _periodic_repetition_ratio(normalized)
    repetition = max(word_rep, char_rep, periodic)
    if len(words) >= 8 and word_rep > 0.55:
        flags.append("high_word_repetition")
    if len(normalized) >= 24 and char_rep > 0.58:
        flags.append("high_character_repetition")
    if len(normalized) >= 24 and periodic > 0.72:
        flags.append("periodic_repetition")
    if re.search(r"(\d)\1{7,}", normalized):
        flags.append("digit_run_repetition")
    unusual = sum(1 for ch in normalized if not (ch.isalnum() or ch.isspace() or ch in ".,!?;:'-+*/=%()[]")) / max(len(normalized), 1)
    if unusual > 0.20:
        flags.append("high_symbol_noise")
    quality = 1.0 - min(0.65, repetition * 0.72) - min(0.35, unusual)
    if flags:
        quality -= min(0.45, 0.10 * len(flags))
    return max(0.0, min(1.0, quality)), repetition, tuple(dict.fromkeys(flags))


def _collapse_metrics(answers: list[ExamAnswer]) -> tuple[float, int, int, bool]:
    outputs = [a.normalized_output for a in answers if a.normalized_output]
    if not outputs:
        return 0.0, len(answers), len(answers), True
    exact = Counter(outputs)
    prefixes = Counter(" ".join(output.split()[:5]) for output in outputs)
    diversity = 100.0 * len(exact) / len(outputs)
    max_exact = max(exact.values())
    max_prefix = max(prefixes.values())
    total = len(answers)
    collapsed = max_exact >= max(3, (total + 2) // 3) or max_prefix >= max(4, (total + 1) // 2) or (len(outputs) >= 6 and diversity <= 37.5)
    return diversity, max_exact, max_prefix, collapsed


def _training_signal(correctness: float, quality: float, gibberish: int, total: int, collapse: bool) -> str:
    if collapse and quality >= 45.0:
        return "MODE_COLLAPSE"
    if total <= 0 or gibberish >= total or quality < 20.0:
        return "GIBBERISH"
    if correctness <= 0 and gibberish >= max(1, total // 2):
        return "GIBBERISH"
    if correctness <= 0 and quality < 55.0:
        return "EARLY_SIGNAL"
    if correctness < 60.0:
        return "LEARNING"
    return "FUNCTIONAL"


def _token_text(tokenizer: BPETokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special=False, errors="replace")


def _distribution_snapshot(logits: torch.Tensor, tokenizer: BPETokenizer, *, top_k: int = BINOCULARS_TOP_K) -> tuple[torch.Tensor, float, tuple[TokenCandidate, ...]]:
    values = logits.float()
    probs = torch.softmax(values, dim=-1)
    entropy = float((-(probs * torch.log2(probs.clamp_min(1e-30))).sum()).item())
    top_prob, top_ids = torch.topk(probs, min(top_k, probs.numel()))
    candidates = tuple(
        TokenCandidate(
            rank=rank,
            token_id=int(token_id),
            token=_token_text(tokenizer, int(token_id)),
            logit=float(values[int(token_id)].item()),
            probability=float(probability),
        )
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
        chosen_logit = float(row[token_id].item())
        runner = candidates[1].probability if len(candidates) > 1 else 0.0
        traces.append(DecisionTrace(
            step=step + 1,
            chosen_token_id=int(token_id),
            chosen_token=_token_text(tokenizer, token_id),
            chosen_logit=chosen_logit,
            chosen_probability=chosen_probability,
            entropy_bits=entropy,
            winner_margin_probability=chosen_probability - runner,
            candidates_evaluated=tokenizer.vocab_size,
            top_candidates=candidates,
        ))
    return tuple(traces)


@torch.no_grad()
def _trace_target_path(model: VistaReasoningGPT, tokenizer: BPETokenizer, prompt_ids: list[int], target_text: str | None) -> tuple[TargetTokenTrace, ...]:
    if not target_text:
        return ()
    target_ids = tokenizer.encode(target_text, add_bos=False, add_eos=False)
    if not target_ids:
        return ()
    available = model.max_seq_len - len(prompt_ids)
    if available <= 0:
        return ()
    target_ids = target_ids[:available]
    context = prompt_ids + target_ids[:-1]
    logits, _ = model(torch.tensor([context], dtype=torch.long))
    traces: list[TargetTokenTrace] = []
    start = len(prompt_ids) - 1
    for step, token_id in enumerate(target_ids):
        row = logits[0, start + step]
        probs, entropy, candidates = _distribution_snapshot(row, tokenizer)
        target_logit = float(row[token_id].item())
        target_rank = int((row > row[token_id]).sum().item()) + 1
        winner = candidates[0]
        traces.append(TargetTokenTrace(
            step=step + 1,
            target_token_id=int(token_id),
            target_token=_token_text(tokenizer, token_id),
            target_rank=target_rank,
            target_logit=target_logit,
            target_probability=float(probs[token_id].item()),
            winning_token_id=winner.token_id,
            winning_token=winner.token,
            winning_probability=winner.probability,
            entropy_bits=entropy,
            candidates_evaluated=tokenizer.vocab_size,
            top_candidates=candidates,
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
    divergences = [
        _js_bits(distributions[i], distributions[j])
        for i in range(len(distributions))
        for j in range(i + 1, len(distributions))
    ]
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
    result: list[ParameterHealth] = []
    for group in sorted(stats):
        count, sum_sq, max_abs = stats[group]
        n = int(count)
        result.append(ParameterHealth(group, n, math.sqrt(sum_sq / max(n, 1)), math.sqrt(sum_sq), max_abs))
    return tuple(result)


@torch.no_grad()
def run_epoch_exam(*, model: VistaReasoningGPT, tokenizer: BPETokenizer, epoch: int, training_stage: str, train_loss: float | None, validation_loss: float | None, max_new_tokens: int = 64) -> EpochExamResult:
    if epoch < 0 or max_new_tokens <= 0:
        raise ValueError("invalid exam epoch/token budget")
    stage = training_stage.strip().casefold()
    questions = exam_questions(stage)
    stop_ids = {tokenizer.vocab[ENDASSISTANT], tokenizer.vocab[EOS]}
    answers: list[ExamAnswer] = []
    first_distributions: list[torch.Tensor] = []
    was_training = model.training
    rng = torch.get_rng_state()
    try:
        model.eval()
        for question in questions:
            prompt_ids = tokenizer.encode(build_exam_prompt(question.prompt), add_bos=False, add_eos=False)
            if len(prompt_ids) >= model.max_seq_len:
                raise RuntimeError(f"exam_prompt_exceeds_model_context:{question.question_id}")
            first_distributions.append(_first_token_distribution(model, prompt_ids))
            ids = torch.tensor([prompt_ids], dtype=torch.long)
            generated = model.generate(ids, max_new_tokens=min(max_new_tokens, model.max_seq_len - len(prompt_ids)), stop_ids=stop_ids, do_sample=False)
            continuation = generated[0, len(prompt_ids):].tolist()
            raw = tokenizer.decode(continuation, skip_special=False, errors="replace")
            normalized = _normalize_output(raw)
            quality, repetition, flags = _quality(raw)
            answers.append(ExamAnswer(
                question.question_id, question.category, question.prompt, raw, normalized,
                len(continuation), _is_correct(question, normalized, flags), quality, repetition, flags,
                _trace_generated_decisions(model, tokenizer, prompt_ids, continuation),
                _trace_target_path(model, tokenizer, prompt_ids, question.diagnostic_target),
            ))
    finally:
        torch.set_rng_state(rng)
        model.train(was_training)

    correct = sum(a.correct for a in answers)
    gibberish = sum(bool(a.gibberish_flags) for a in answers)
    quality = 100.0 * sum(a.quality_score for a in answers) / max(len(answers), 1)
    correctness = 100.0 * correct / max(len(answers), 1)
    diversity, max_exact, max_prefix, collapse = _collapse_metrics(answers)
    mean_js, min_js, max_js = _prompt_sensitivity(first_distributions)
    return EpochExamResult(
        EXAM_VERSION, exam_contract_fingerprint(stage), EXAM_DECODING_MODE, epoch, stage,
        train_loss, validation_loss, len(answers), correct, correctness, quality, gibberish,
        sum(a.generated_tokens for a in answers) / max(len(answers), 1),
        _training_signal(correctness, quality, gibberish, len(answers), collapse),
        diversity, max_exact, max_prefix, collapse, tuple(answers),
        mean_js, min_js, max_js, _parameter_health(model),
    )


def _candidate_line(candidates: tuple[TokenCandidate, ...]) -> str:
    return " | ".join(
        f"#{c.rank} {c.token!r} p={c.probability:.4%} logit={c.logit:.4f}"
        for c in candidates
    )


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
        f"Training signal: {result.training_signal}",
        "", "BINOCULARS — PROMPT SENSITIVITY",
        f"Mean first-token Jensen-Shannon divergence: {result.mean_prompt_js_bits:.6f} bits",
        f"Min first-token Jensen-Shannon divergence: {result.min_prompt_js_bits:.6f} bits",
        f"Max first-token Jensen-Shannon divergence: {result.max_prompt_js_bits:.6f} bits",
        "Interpretation: near 0 means different questions produce nearly identical next-token distributions.",
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
        lines += [
            "", "=" * 80, f"QUESTION [{a.question_id}] ({a.category})", a.prompt,
            "", "RAW OUTPUT", a.raw_output, "", "NORMALIZED OUTPUT", a.normalized_output, "",
            f"Semantically correct: {'YES' if a.correct else 'NO'}", f"Surface quality: {a.quality_score * 100:.1f}%",
            f"Repetition: {a.repetition_ratio:.3f}", f"Flags: {', '.join(a.gibberish_flags) if a.gibberish_flags else 'none'}",
            f"Generated tokens: {a.generated_tokens}",
        ]
        if a.decision_trace:
            total_evaluations = sum(step.candidates_evaluated for step in a.decision_trace)
            lines += ["", "MODEL DECISION TRACE", f"Token alternatives scored: {len(a.decision_trace)} steps × {a.decision_trace[0].candidates_evaluated:,} vocabulary = {total_evaluations:,}"]
            for step in a.decision_trace[:BINOCULARS_RENDER_STEPS]:
                lines.append(
                    f"step={step.step:02d} chose={step.chosen_token!r} p={step.chosen_probability:.4%} "
                    f"logit={step.chosen_logit:.4f} entropy={step.entropy_bits:.4f}b margin={step.winner_margin_probability:.4%}"
                )
                lines.append("  " + _candidate_line(step.top_candidates))
            if len(a.decision_trace) > BINOCULARS_RENDER_STEPS:
                lines.append(f"... {len(a.decision_trace) - BINOCULARS_RENDER_STEPS} later decision steps omitted from TXT; full trace is in JSON.")
        if a.target_trace:
            mean_rank = sum(step.target_rank for step in a.target_trace) / len(a.target_trace)
            lines += ["", "CORRECT TARGET PATH (teacher-forced diagnostic only; does not affect training)", f"Mean correct-token rank: {mean_rank:.2f}/{a.target_trace[0].candidates_evaluated}"]
            for step in a.target_trace:
                lines.append(
                    f"step={step.step:02d} target={step.target_token!r} rank={step.target_rank}/{step.candidates_evaluated} "
                    f"p={step.target_probability:.4%} logit={step.target_logit:.4f} | "
                    f"winner={step.winning_token!r} p={step.winning_probability:.4%} entropy={step.entropy_bits:.4f}b"
                )
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
