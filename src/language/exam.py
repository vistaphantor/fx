from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

from src.language.protocol import build_exam_prompt
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.tokenizer import BPETokenizer, ENDASSISTANT, EOS

EXAM_VERSION = 2


@dataclass(frozen=True)
class ExamQuestion:
    question_id: str
    category: str
    prompt: str
    expected_all: tuple[str, ...] = ()
    expected_any: tuple[str, ...] = ()


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
    epoch: int
    training_stage: str
    train_loss: float | None
    validation_loss: float | None
    question_count: int
    correct: int
    correctness_percent: float
    quality_percent: float
    gibberish_count: int
    mean_generated_tokens: float
    training_signal: str
    answers: tuple[ExamAnswer, ...]


def exam_questions(training_stage: str) -> tuple[ExamQuestion, ...]:
    base = (
        ExamQuestion("arithmetic_2_plus_2", "arithmetic", "What is 2 + 2?", expected_all=("4",)),
        ExamQuestion("arithmetic_15_times_14", "arithmetic", "What is 15 multiplied by 14?", expected_all=("210",)),
        ExamQuestion("algebra_simple", "algebra", "If 2x + 5 = 11, what is x?", expected_all=("3",)),
        ExamQuestion(
            "trading_bullish",
            "trading_language",
            "In trading, what does bullish mean?",
            expected_any=("rise", "rising", "higher", "upward", "increase", "buyers"),
        ),
        ExamQuestion(
            "trading_risk",
            "trading_language",
            "What does risk mean in trading?",
            expected_any=("loss", "lose", "uncertainty", "exposure", "capital"),
        ),
        ExamQuestion(
            "logic_youngest",
            "logic",
            "Alice is older than Bob. Bob is older than Charlie. Who is youngest?",
            expected_all=("charlie",),
        ),
        ExamQuestion(
            "trading_spread",
            "trading_language",
            "What is the bid-ask spread?",
            expected_all=("bid", "ask"),
            expected_any=("difference", "distance", "gap"),
        ),
        ExamQuestion(
            "trading_atr",
            "trading_language",
            "What does ATR measure in trading?",
            expected_any=("volatility", "range", "true range"),
        ),
    )
    if training_stage.strip().casefold() != "trading_reasoning":
        return base
    return base + (
        ExamQuestion(
            "trading_stop_loss",
            "trading_reasoning",
            "Why would a trader use a stop-loss?",
            expected_all=("loss",),
            expected_any=("limit", "protect", "control", "risk"),
        ),
        ExamQuestion(
            "trading_gain_per_unit",
            "trading_math",
            "A trader buys at 100 and exits at 105. What is the gain per unit?",
            expected_all=("5",),
        ),
        ExamQuestion(
            "trading_orderflow",
            "trading_reasoning",
            "Buyers repeatedly lift the ask and price makes higher highs. Who is controlling direction?",
            expected_any=("buyers", "buyer", "bulls", "bullish"),
        ),
        ExamQuestion(
            "trading_invalidation",
            "trading_reasoning",
            "What is an invalidation level in a trading thesis?",
            expected_all=("level",),
            expected_any=("wrong", "invalid", "fails", "failure", "thesis"),
        ),
    )


def exam_prompt_families(training_stage: str) -> set[str]:
    from src.language.canonical_contract import prompt_family_key

    return {prompt_family_key(question.prompt) for question in exam_questions(training_stage)}


def _normalize_output(value: str) -> str:
    text = str(value or "")
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?[^>]+>", " ", text)
    text = text.replace("\ufffd", " ")
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text


def _is_correct(question: ExamQuestion, normalized: str) -> bool:
    if not normalized:
        return False

    def present(term: str) -> bool:
        term_norm = term.casefold().strip()
        if not term_norm:
            return False
        return bool(re.search(rf"(?<!\w){re.escape(term_norm)}(?!\w)", normalized))

    if question.expected_all and not all(present(term) for term in question.expected_all):
        return False
    if question.expected_any and not any(present(term) for term in question.expected_any):
        return False
    return bool(question.expected_all or question.expected_any)


def _quality(value: str) -> tuple[float, float, tuple[str, ...]]:
    raw = str(value or "")
    normalized = _normalize_output(raw)
    flags: list[str] = []
    if not normalized:
        return 0.0, 1.0, ("empty",)

    words = re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", normalized)
    if not words:
        return 0.0, 1.0, ("no_lexical_content",)

    repetition = 1.0 - (len(set(words)) / max(1, len(words)))
    if len(words) >= 8 and repetition > 0.55:
        flags.append("high_repetition")

    replacement_count = raw.count("\ufffd")
    if replacement_count:
        flags.append("invalid_utf8_bytes")

    unusual = sum(1 for ch in normalized if not (ch.isalnum() or ch.isspace() or ch in ".,!?;:'-+*/=%()[]"))
    unusual_ratio = unusual / max(1, len(normalized))
    if unusual_ratio > 0.20:
        flags.append("high_symbol_noise")

    single_char_words = sum(1 for word in words if len(word) == 1 and not word.isdigit())
    if len(words) >= 8 and single_char_words / len(words) > 0.45:
        flags.append("fragmented_words")

    quality = 1.0
    quality -= min(0.55, repetition * 0.65)
    quality -= min(0.35, unusual_ratio)
    quality -= min(0.25, replacement_count / max(1, len(raw)))
    if flags:
        quality -= min(0.40, 0.12 * len(flags))
    return max(0.0, min(1.0, quality)), repetition, tuple(flags)


def _training_signal(correctness: float, quality_percent: float, gibberish: int, total: int) -> str:
    if total <= 0:
        return "GIBBERISH"
    if gibberish >= total or quality_percent < 20.0:
        return "GIBBERISH"
    if correctness <= 0.0 and quality_percent < 45.0:
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
        # A freshly initialized byte-level LM can generate arbitrary byte IDs,
        # including incomplete UTF-8 sequences. The exam must observe and score
        # that gibberish rather than crash before epoch 0. Strict decoding
        # remains the tokenizer's default for corpus/checkpoint integrity tests.
        raw = tokenizer.decode(continuation, skip_special=False, errors="replace")
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


def write_exam_artifacts(result: EpochExamResult, output_dir: str | Path, *, prefix: str | None = None) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = prefix or f"epoch_{result.epoch:03d}_exam"
    json_path = output / f"{stem}.json"
    txt_path = output / f"{stem}.txt"
    payload = asdict(result)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"Vista Reasoner Exam v{result.exam_version}",
        f"Epoch: {result.epoch}",
        f"Stage: {result.training_stage}",
        f"Train loss: {result.train_loss}",
        f"Validation loss: {result.validation_loss}",
        f"Correct: {result.correct}/{result.question_count} ({result.correctness_percent:.1f}%)",
        f"Quality: {result.quality_percent:.1f}%",
        f"Gibberish: {result.gibberish_count}/{result.question_count}",
        f"Training signal: {result.training_signal}",
        "",
    ]
    for answer in result.answers:
        lines.extend(
            [
                "=" * 80,
                f"QUESTION [{answer.question_id}] ({answer.category})",
                answer.prompt,
                "",
                "RAW OUTPUT",
                answer.raw_output,
                "",
                "NORMALIZED OUTPUT",
                answer.normalized_output,
                "",
                f"Correct: {'YES' if answer.correct else 'NO'}",
                f"Quality: {answer.quality_score * 100:.1f}%",
                f"Repetition: {answer.repetition_ratio:.3f}",
                f"Flags: {', '.join(answer.gibberish_flags) if answer.gibberish_flags else 'none'}",
                f"Generated tokens: {answer.generated_tokens}",
                "",
            ]
        )
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return txt_path, json_path
