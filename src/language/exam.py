"""Deterministic held-out diagnostics for Vista language-model training."""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.tokenizer import BPETokenizer, ENDASSISTANT, EOS

EXAM_VERSION = 2


@dataclass(frozen=True, slots=True)
class ExamQuestion:
    question_id: str
    category: str
    prompt: str
    expected_any: tuple[str, ...] = ()
    expected_all: tuple[str, ...] = ()
    expected_regex: str | None = None


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class EpochExamResult:
    exam_version: int
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
    answers: tuple[ExamAnswer, ...]


FOUNDATION_EXAM: tuple[ExamQuestion, ...] = (
    ExamQuestion("math_2_plus_2", "arithmetic", "What is 2 + 2?", expected_regex=r"(?<!\d)4(?!\d)"),
    ExamQuestion("math_15_times_14", "arithmetic", "What is 15 multiplied by 14?", expected_regex=r"(?<!\d)210(?!\d)"),
    ExamQuestion("algebra_linear", "arithmetic", "If 2x + 5 = 11, what is x?", expected_regex=r"(?<!\d)3(?:\.0+)?(?!\d)"),
    ExamQuestion(
        "language_bullish", "market_language", "In trading, what does bullish mean?",
        expected_any=("rise", "rising", "higher", "increase", "buyers", "buying", "upward"),
    ),
    ExamQuestion(
        "language_risk", "risk_language", "What does risk mean in trading?",
        expected_any=("loss", "uncertainty", "exposure", "lose", "downside"),
    ),
    ExamQuestion(
        "logic_youngest", "logic",
        "Alice is older than Bob, and Bob is older than Charlie. Who is youngest?",
        expected_any=("charlie",),
    ),
    ExamQuestion(
        "trading_spread", "trading", "What is the bid-ask spread?",
        expected_all=("bid", "ask"), expected_any=("difference", "gap"),
    ),
    ExamQuestion(
        "trading_atr", "trading", "What does ATR measure in market analysis?",
        expected_any=("volatility", "true range", "range"),
    ),
)

TRADING_EXTENSION: tuple[ExamQuestion, ...] = (
    ExamQuestion(
        "trading_stop_loss", "risk", "Why does a trader use a stop-loss?",
        expected_any=("limit loss", "limit losses", "risk", "protect", "loss"),
    ),
    ExamQuestion(
        "trading_long_profit", "arithmetic",
        "A long position enters at 100 and exits at 105 before fees. What is the price gain per unit?",
        expected_regex=r"(?<!\d)5(?:\.0+)?(?!\d)",
    ),
    ExamQuestion(
        "trading_direction", "reasoning",
        "If buyers repeatedly lift the ask and price makes higher highs, which side has directional control?",
        expected_any=("buyer", "buyers", "bull", "bullish", "buying"),
    ),
    ExamQuestion(
        "trading_invalidation", "reasoning", "What is an invalidation level in a trading thesis?",
        expected_all=("thesis",), expected_any=("wrong", "invalid", "fails", "failure", "exit"),
    ),
)

CONTROL_TOKEN_RE = re.compile(
    r"</?(?:bos|eos|sep|user|assistant|think|market|account|position|evidence|hypothesis|countercase|tool|tool_result|decision|confidence|invalidation)>",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[._'-][A-Za-z0-9]+)*")


def exam_questions(training_stage: str) -> tuple[ExamQuestion, ...]:
    return FOUNDATION_EXAM + TRADING_EXTENSION if training_stage.strip().casefold() == "trading_reasoning" else FOUNDATION_EXAM


def build_exam_prompt(question: str) -> str:
    return "<bos>\n<user>\n" + question.strip() + "\n</user>\n<assistant>\n"


def _normalize_output(text: str) -> str:
    return re.sub(r"\s+", " ", CONTROL_TOKEN_RE.sub(" ", text)).strip()


def _repetition_ratio(text: str) -> float:
    words = [match.group(0).casefold() for match in WORD_RE.finditer(text)]
    if not words:
        return 1.0
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return max(counts.values()) / len(words)


def _quality(text: str) -> tuple[float, float, tuple[str, ...]]:
    normalized = _normalize_output(text)
    words = WORD_RE.findall(normalized)
    flags: list[str] = []
    repetition = _repetition_ratio(normalized)
    if not normalized:
        flags.append("empty")
    if "<unk>" in text:
        flags.append("unk_token")
    if repetition >= 0.50 and len(words) >= 6:
        flags.append("high_repetition")
    if normalized and not words:
        flags.append("no_lexical_content")
    if normalized and sum(ch.isalnum() for ch in normalized) / max(len(normalized), 1) < 0.35:
        flags.append("low_alphanumeric_density")
    if text.count("<think>") > 1 or text.count("</think>") > 1:
        flags.append("control_token_loop")
    if "<user>" in text or "</user>" in text or "<bos>" in text:
        flags.append("protocol_leak")
    score = 1.0
    penalties = {
        "empty": 1.0, "unk_token": 0.30, "high_repetition": 0.50,
        "no_lexical_content": 0.70, "low_alphanumeric_density": 0.35,
        "control_token_loop": 0.50, "protocol_leak": 0.50,
    }
    for flag in flags:
        score -= penalties[flag]
    return max(0.0, min(1.0, score)), repetition, tuple(flags)


def _contains(value: str, keyword: str) -> bool:
    keyword = keyword.casefold()
    if " " in keyword:
        return keyword in value
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", value))


def _is_correct(question: ExamQuestion, normalized: str) -> bool:
    value = normalized.casefold()
    if question.expected_regex and not re.search(question.expected_regex, normalized, flags=re.IGNORECASE):
        return False
    if question.expected_all and not all(_contains(value, word) for word in question.expected_all):
        return False
    if question.expected_any and not any(_contains(value, word) for word in question.expected_any):
        return False
    return bool(question.expected_regex or question.expected_all or question.expected_any)


def _training_signal(correctness: float, quality: float, gibberish: int, total: int) -> str:
    ratio = gibberish / max(total, 1)
    if correctness == 0.0 and ratio >= 0.75:
        return "GIBBERISH"
    if correctness < 25.0 or quality < 50.0:
        return "EARLY_SIGNAL"
    if correctness < 60.0:
        return "LEARNING"
    return "FUNCTIONAL"


def run_epoch_exam(
    *, model: VistaReasoningGPT, tokenizer: BPETokenizer, epoch: int,
    training_stage: str, train_loss: float | None, validation_loss: float | None,
    max_new_tokens: int = 64,
) -> EpochExamResult:
    if epoch < 0 or max_new_tokens <= 0:
        raise ValueError("invalid exam epoch/token budget")
    questions = exam_questions(training_stage)
    stop_ids = {tokenizer.vocab[ENDASSISTANT], tokenizer.vocab[EOS]}
    answers: list[ExamAnswer] = []
    model.eval()
    for question in questions:
        prompt_ids = tokenizer.encode(build_exam_prompt(question.prompt), add_bos=False, add_eos=False)
        if len(prompt_ids) >= model.max_seq_len:
            raise RuntimeError(f"exam_prompt_exceeds_model_context:{question.question_id}")
        ids = torch.tensor([prompt_ids], dtype=torch.long)
        generated = model.generate(
            ids, max_new_tokens=min(max_new_tokens, model.max_seq_len - len(prompt_ids)),
            temperature=1.0, top_k=1, top_p=1.0, stop_ids=stop_ids,
        )
        continuation = generated[0, len(prompt_ids):].tolist()
        raw = tokenizer.decode(continuation, skip_special=False)
        normalized = _normalize_output(raw)
        quality, repetition, flags = _quality(raw)
        answers.append(
            ExamAnswer(
                question.question_id, question.category, question.prompt, raw, normalized,
                len(continuation), _is_correct(question, normalized), quality, repetition, flags,
            )
        )
    correct = sum(answer.correct for answer in answers)
    gibberish = sum(bool(answer.gibberish_flags) for answer in answers)
    mean_quality = sum(answer.quality_score for answer in answers) / len(answers)
    correctness = 100.0 * correct / len(answers)
    quality_percent = 100.0 * mean_quality
    return EpochExamResult(
        EXAM_VERSION, epoch, training_stage, train_loss, validation_loss, len(answers), correct,
        correctness, quality_percent, gibberish,
        sum(answer.generated_tokens for answer in answers) / len(answers),
        _training_signal(correctness, quality_percent, gibberish, len(answers)), tuple(answers),
    )


def _fmt_loss(value: float | None) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.4f}"


def render_exam_text(result: EpochExamResult, previous: EpochExamResult | None = None) -> str:
    delta = None if previous is None else result.correctness_percent - previous.correctness_percent
    lines = [
        "VISTA LANGUAGE REASONER - EPOCH EXAM", "=" * 80,
        f"Exam version     : {result.exam_version}",
        f"Epoch            : {result.epoch}",
        f"Training stage   : {result.training_stage}",
        f"Training signal  : {result.training_signal}",
        f"Train loss       : {_fmt_loss(result.train_loss)}",
        f"Validation loss  : {_fmt_loss(result.validation_loss)}",
        f"Correct          : {result.correct_questions}/{result.total_questions} ({result.correctness_percent:.1f}%)",
        f"Quality          : {result.mean_quality_percent:.1f}%",
        f"Gibberish flags  : {result.gibberish_answers}/{result.total_questions} answers",
        f"Mean new tokens  : {result.mean_generated_tokens:.1f}",
        f"Score delta      : {'n/a' if delta is None else f'{delta:+.1f} points'}", "",
    ]
    for index, answer in enumerate(result.answers, start=1):
        lines.extend([
            f"QUESTION {index:02d} | {answer.question_id} | {answer.category}", "-" * 80,
            f"Prompt       : {answer.prompt}", f"Correct      : {'YES' if answer.correct else 'NO'}",
            f"Quality      : {answer.quality_score * 100:.1f}%", f"Repetition   : {answer.repetition_ratio:.3f}",
            f"Flags        : {', '.join(answer.gibberish_flags) if answer.gibberish_flags else 'none'}",
            f"Tokens       : {answer.generated_tokens}", "Raw output:", answer.raw_output or "<EMPTY>", "",
            "Normalized output:", answer.normalized_output or "<EMPTY>", "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def save_epoch_exam(
    *, result: EpochExamResult, exams_dir: Path, previous: EpochExamResult | None = None,
) -> tuple[Path, Path]:
    exams_dir.mkdir(parents=True, exist_ok=True)
    stem = f"epoch_{result.epoch:03d}_exam"
    text_path = exams_dir / f"{stem}.txt"
    json_path = exams_dir / f"{stem}.json"
    text_path.write_text(render_exam_text(result, previous), encoding="utf-8")
    json_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return text_path, json_path
