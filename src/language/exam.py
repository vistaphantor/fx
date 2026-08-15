from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from src.language.protocol import build_exam_prompt
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.tokenizer import BPETokenizer, ENDASSISTANT, EOS

EXAM_VERSION = 8
EXAM_DECODING_MODE = "greedy_argmax_v1"


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


FOUNDATION_EXAM: tuple[ExamQuestion, ...] = (
    ExamQuestion("arithmetic_2_plus_2", "primitive_arithmetic", "What is 2 + 2?", numeric_answer="4", expected_expression="2 + 2"),
    ExamQuestion("arithmetic_chain", "primitive_arithmetic", "What is 2 + 2 + 3?", numeric_answer="7", expected_expression="2 + 2 + 3"),
    ExamQuestion("arithmetic_negative", "primitive_arithmetic", "What is 9 - 12?", numeric_answer="-3", expected_expression="9 - 12"),
    ExamQuestion("arithmetic_multiply", "primitive_arithmetic", "What is 3 times 4?", numeric_answer="12", expected_expression="3 times 4"),
    ExamQuestion("number_successor", "number_sense", "What number comes after 4?", numeric_answer="5"),
    ExamQuestion("economics_price", "foundation_economics", "In simple economics, what is a price?", expected_any=("money", "amount", "paid", "asked")),
    ExamQuestion(
        "economics_profit", "foundation_economics",
        "A business receives 20 shillings and has costs of 7 shillings. What is its profit?",
        numeric_answer="13", numeric_units=("shilling", "shillings"),
    ),
    ExamQuestion(
        "economics_inflation", "foundation_economics",
        "If prices rise while income stays the same, what happens to purchasing power?",
        expected_all=("purchasing", "power"),
        expected_any=("falls", "fall", "decreases", "declines", "lower"),
    ),
)

REASONING_EXTENSION: tuple[ExamQuestion, ...] = (
    ExamQuestion("algebra_simple", "algebra", "If 2x + 5 = 11, what is x?", expected_all=("x", "3")),
    ExamQuestion("logic_youngest", "logic", "Alice is older than Bob, and Bob is older than Charlie. Who is youngest?", expected_all=("charlie",)),
)

TRADING_EXTENSION: tuple[ExamQuestion, ...] = (
    ExamQuestion("trading_bullish", "trading_language", "In trading, what does bullish mean?", expected_any=("rise", "rising", "higher", "upward", "increase", "buyers")),
    ExamQuestion("trading_risk", "trading_language", "What does risk mean in trading?", expected_any=("loss", "lose", "uncertainty", "exposure", "capital")),
    ExamQuestion("trading_spread", "trading_language", "What is the bid-ask spread?", expected_all=("bid", "ask"), expected_any=("difference", "distance", "gap")),
    ExamQuestion("trading_atr", "trading_language", "What does ATR measure in market analysis?", expected_any=("volatility", "range", "true range")),
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
        "semantic_grading": "expression_bound_numeric_v3",
        "collapse_detection": "exact_and_prefix_v1",
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


def run_epoch_exam(*, model: VistaReasoningGPT, tokenizer: BPETokenizer, epoch: int, training_stage: str, train_loss: float | None, validation_loss: float | None, max_new_tokens: int = 64) -> EpochExamResult:
    if epoch < 0 or max_new_tokens <= 0:
        raise ValueError("invalid exam epoch/token budget")
    stage = training_stage.strip().casefold()
    questions = exam_questions(stage)
    stop_ids = {tokenizer.vocab[ENDASSISTANT], tokenizer.vocab[EOS]}
    answers: list[ExamAnswer] = []
    was_training = model.training
    rng = torch.get_rng_state()
    try:
        model.eval()
        for question in questions:
            prompt_ids = tokenizer.encode(build_exam_prompt(question.prompt), add_bos=False, add_eos=False)
            if len(prompt_ids) >= model.max_seq_len:
                raise RuntimeError(f"exam_prompt_exceeds_model_context:{question.question_id}")
            ids = torch.tensor([prompt_ids], dtype=torch.long)
            generated = model.generate(ids, max_new_tokens=min(max_new_tokens, model.max_seq_len - len(prompt_ids)), stop_ids=stop_ids, do_sample=False)
            continuation = generated[0, len(prompt_ids):].tolist()
            raw = tokenizer.decode(continuation, skip_special=False, errors="replace")
            normalized = _normalize_output(raw)
            quality, repetition, flags = _quality(raw)
            answers.append(ExamAnswer(question.question_id, question.category, question.prompt, raw, normalized, len(continuation), _is_correct(question, normalized, flags), quality, repetition, flags))
    finally:
        torch.set_rng_state(rng)
        model.train(was_training)

    correct = sum(a.correct for a in answers)
    gibberish = sum(bool(a.gibberish_flags) for a in answers)
    quality = 100.0 * sum(a.quality_score for a in answers) / max(len(answers), 1)
    correctness = 100.0 * correct / max(len(answers), 1)
    diversity, max_exact, max_prefix, collapse = _collapse_metrics(answers)
    return EpochExamResult(
        EXAM_VERSION, exam_contract_fingerprint(stage), EXAM_DECODING_MODE, epoch, stage,
        train_loss, validation_loss, len(answers), correct, correctness, quality, gibberish,
        sum(a.generated_tokens for a in answers) / max(len(answers), 1),
        _training_signal(correctness, quality, gibberish, len(answers), collapse),
        diversity, max_exact, max_prefix, collapse, tuple(answers),
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
    ]
    if previous is not None:
        lines += [
            f"Correctness delta: {result.correctness_percent - previous.correctness_percent:+.1f}pp",
            f"Surface-quality delta: {result.mean_quality_percent - previous.mean_quality_percent:+.1f}pp",
            f"Gibberish delta: {result.gibberish_answers - previous.gibberish_answers:+d}",
            f"Diversity delta: {result.answer_diversity_percent - previous.answer_diversity_percent:+.1f}pp",
        ]
    for a in result.answers:
        lines += ["", "=" * 80, f"QUESTION [{a.question_id}] ({a.category})", a.prompt, "", "RAW OUTPUT", a.raw_output, "", "NORMALIZED OUTPUT", a.normalized_output, "", f"Semantically correct: {'YES' if a.correct else 'NO'}", f"Surface quality: {a.quality_score * 100:.1f}%", f"Repetition: {a.repetition_ratio:.3f}", f"Flags: {', '.join(a.gibberish_flags) if a.gibberish_flags else 'none'}", f"Generated tokens: {a.generated_tokens}"]
    return "\n".join(lines) + "\n"


def save_epoch_exam(*, result: EpochExamResult, exams_dir: str | Path, previous: EpochExamResult | None = None, prefix: str | None = None) -> tuple[Path, Path]:
    directory = Path(exams_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = prefix or f"epoch_{result.epoch:03d}_exam"
    text_path, json_path = directory / f"{stem}.txt", directory / f"{stem}.json"
    text_path.write_text(render_exam_text(result, previous=previous), encoding="utf-8")
    json_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return text_path, json_path
