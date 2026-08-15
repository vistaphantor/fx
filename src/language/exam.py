from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from src.language.protocol import build_exam_prompt
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.tokenizer import BPETokenizer, ENDASSISTANT, EOS

EXAM_VERSION = 6
EXAM_DECODING_MODE = "greedy_argmax_v1"


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
    answers: tuple[ExamAnswer, ...]


FOUNDATION_EXAM: tuple[ExamQuestion, ...] = (
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
        "Alice is older than Bob, and Bob is older than Charlie. Who is youngest?",
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
        "What does ATR measure in market analysis?",
        expected_any=("volatility", "range", "true range"),
    ),
)


TRADING_EXTENSION: tuple[ExamQuestion, ...] = (
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


_NUMERIC_ANSWERS = {
    "arithmetic_2_plus_2": "4",
    "arithmetic_15_times_14": "210",
    "algebra_simple": "3",
    "trading_gain_per_unit": "5",
}


def exam_questions(training_stage: str) -> tuple[ExamQuestion, ...]:
    if training_stage.strip().casefold() == "trading_reasoning":
        return FOUNDATION_EXAM + TRADING_EXTENSION
    return FOUNDATION_EXAM


def exam_contract_payload(training_stage: str) -> dict:
    """Return the immutable benchmark contract for one training stage."""
    stage = training_stage.strip().casefold()
    questions = exam_questions(stage)
    return {
        "exam_version": EXAM_VERSION,
        "decoding_mode": EXAM_DECODING_MODE,
        "training_stage": stage,
        "questions": [
            {
                **asdict(question),
                "serialized_prompt": build_exam_prompt(question.prompt),
            }
            for question in questions
        ],
        "numeric_answers": {
            key: value
            for key, value in sorted(_NUMERIC_ANSWERS.items())
            if any(question.question_id == key for question in questions)
        },
        "stop_tokens": [ENDASSISTANT, EOS],
    }


def exam_contract_fingerprint(training_stage: str) -> str:
    encoded = json.dumps(
        exam_contract_payload(training_stage),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _numeric_answer_matches(question_id: str, normalized: str) -> bool:
    expected = _NUMERIC_ANSWERS[question_id]
    numbers = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", normalized)
    if question_id == "algebra_simple":
        if re.search(rf"\bx\s*=\s*{re.escape(expected)}\b", normalized):
            return True
    if normalized.strip(" .!,:;") == expected:
        return True
    if re.search(rf"\b(?:answer|result|value)\s+(?:is|=)\s*{re.escape(expected)}\b", normalized):
        return True
    return len(numbers) <= 2 and bool(numbers) and numbers[-1] == expected


def _is_correct(
    question: ExamQuestion,
    normalized: str,
    gibberish_flags: tuple[str, ...] = (),
) -> bool:
    if not normalized or gibberish_flags:
        return False
    if question.question_id in _NUMERIC_ANSWERS:
        return _numeric_answer_matches(question.question_id, normalized)

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


def _character_repetition_ratio(text: str, *, ngram: int = 4) -> float:
    compact = re.sub(r"\s+", "", text.casefold())
    if len(compact) < max(ngram * 3, 12):
        return 0.0
    grams = [compact[index:index + ngram] for index in range(len(compact) - ngram + 1)]
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def _periodic_repetition_ratio(text: str) -> float:
    compact = re.sub(r"\s+", "", text.casefold())
    if len(compact) < 16:
        return 0.0
    best = 0.0
    for width in range(1, min(13, len(compact) // 3 + 1)):
        matches = 0
        comparisons = 0
        for index in range(width, len(compact)):
            comparisons += 1
            if compact[index] == compact[index - width]:
                matches += 1
        if comparisons:
            best = max(best, matches / comparisons)
    return best


def _quality(value: str) -> tuple[float, float, tuple[str, ...]]:
    raw = str(value or "")
    replacement_count = raw.count("\ufffd")
    normalized = _normalize_output(raw)
    flags: list[str] = []
    if replacement_count:
        flags.append("invalid_utf8_bytes")
    if not normalized:
        if "empty" not in flags:
            flags.append("empty")
        return 0.0, 1.0, tuple(flags)

    words = re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", normalized)
    if not words:
        flags.append("no_lexical_content")
        return 0.0, 1.0, tuple(dict.fromkeys(flags))

    word_repetition = 1.0 - (len(set(words)) / max(1, len(words)))
    char_repetition = _character_repetition_ratio(normalized)
    periodic_repetition = _periodic_repetition_ratio(normalized)
    repetition = max(word_repetition, char_repetition, periodic_repetition)

    if len(words) >= 8 and word_repetition > 0.55:
        flags.append("high_word_repetition")
    if len(normalized) >= 24 and char_repetition > 0.58:
        flags.append("high_character_repetition")
    if len(normalized) >= 24 and periodic_repetition > 0.72:
        flags.append("periodic_repetition")
    if re.search(r"(\d)\1{7,}", normalized):
        flags.append("digit_run_repetition")

    unusual = sum(
        1
        for ch in normalized
        if not (ch.isalnum() or ch.isspace() or ch in ".,!?;:'-+*/=%()[]")
    )
    unusual_ratio = unusual / max(1, len(normalized))
    if unusual_ratio > 0.20:
        flags.append("high_symbol_noise")

    single_char_words = sum(1 for word in words if len(word) == 1 and not word.isdigit())
    if len(words) >= 8 and single_char_words / len(words) > 0.45:
        flags.append("fragmented_words")

    longest_run = max((len(part) for part in re.findall(r"[A-Za-z]+", normalized)), default=0)
    if longest_run >= 40:
        flags.append("run_on_fragment")

    quality = 1.0
    quality -= min(0.65, repetition * 0.72)
    quality -= min(0.35, unusual_ratio)
    quality -= min(0.25, replacement_count / max(1, len(raw)))
    if "run_on_fragment" in flags or "digit_run_repetition" in flags:
        quality -= 0.25
    if flags:
        quality -= min(0.45, 0.10 * len(flags))
    return max(0.0, min(1.0, quality)), repetition, tuple(dict.fromkeys(flags))


def _training_signal(
    correctness: float,
    quality_percent: float,
    gibberish: int,
    total: int,
) -> str:
    if total <= 0:
        return "GIBBERISH"
    if gibberish >= total or quality_percent < 20.0:
        return "GIBBERISH"
    if correctness <= 0.0 and gibberish >= max(1, total // 2):
        return "GIBBERISH"
    if correctness <= 0.0 and quality_percent < 55.0:
        return "EARLY_SIGNAL"
    if correctness < 60.0:
        return "LEARNING"
    return "FUNCTIONAL"


def run_epoch_exam(
    *,
    model: VistaReasoningGPT,
    tokenizer: BPETokenizer,
    epoch: int,
    training_stage: str,
    train_loss: float | None,
    validation_loss: float | None,
    max_new_tokens: int = 64,
) -> EpochExamResult:
    if epoch < 0 or max_new_tokens <= 0:
        raise ValueError("invalid exam epoch/token budget")

    stage = training_stage.strip().casefold()
    questions = exam_questions(stage)
    contract_fingerprint = exam_contract_fingerprint(stage)
    stop_ids = {tokenizer.vocab[ENDASSISTANT], tokenizer.vocab[EOS]}
    answers: list[ExamAnswer] = []
    was_training = model.training

    try:
        model.eval()
        for question in questions:
            serialized_prompt = build_exam_prompt(question.prompt)
            prompt_ids = tokenizer.encode(
                serialized_prompt,
                add_bos=False,
                add_eos=False,
            )
            if len(prompt_ids) >= model.max_seq_len:
                raise RuntimeError(f"exam_prompt_exceeds_model_context:{question.question_id}")

            ids = torch.tensor([prompt_ids], dtype=torch.long)
            generated = model.generate(
                ids,
                max_new_tokens=min(max_new_tokens, model.max_seq_len - len(prompt_ids)),
                stop_ids=stop_ids,
                do_sample=False,
            )
            continuation = generated[0, len(prompt_ids):].tolist()
            raw = tokenizer.decode(continuation, skip_special=False, errors="replace")
            normalized = _normalize_output(raw)
            quality, repetition, flags = _quality(raw)
            answers.append(
                ExamAnswer(
                    question_id=question.question_id,
                    category=question.category,
                    prompt=question.prompt,
                    raw_output=raw,
                    normalized_output=normalized,
                    generated_tokens=len(continuation),
                    correct=_is_correct(question, normalized, flags),
                    quality_score=quality,
                    repetition_ratio=repetition,
                    gibberish_flags=flags,
                )
            )
    finally:
        model.train(was_training)

    correct = sum(answer.correct for answer in answers)
    gibberish = sum(bool(answer.gibberish_flags) for answer in answers)
    mean_quality = sum(answer.quality_score for answer in answers) / len(answers)
    correctness = 100.0 * correct / len(answers)
    quality_percent = 100.0 * mean_quality

    return EpochExamResult(
        exam_version=EXAM_VERSION,
        exam_contract_fingerprint=contract_fingerprint,
        decoding_mode=EXAM_DECODING_MODE,
        epoch=epoch,
        training_stage=stage,
        train_loss=train_loss,
        validation_loss=validation_loss,
        total_questions=len(answers),
        correct_questions=correct,
        correctness_percent=correctness,
        mean_quality_percent=quality_percent,
        gibberish_answers=gibberish,
        mean_generated_tokens=sum(answer.generated_tokens for answer in answers) / len(answers),
        training_signal=_training_signal(correctness, quality_percent, gibberish, len(answers)),
        answers=tuple(answers),
    )


def render_exam_text(
    result: EpochExamResult,
    *,
    previous: EpochExamResult | None = None,
) -> str:
    if previous is not None and previous.exam_contract_fingerprint != result.exam_contract_fingerprint:
        raise RuntimeError("exam_contract_changed_between_results")

    lines = [
        f"Vista Reasoner Exam v{result.exam_version}",
        f"Contract: {result.exam_contract_fingerprint[:16]}",
        f"Decoding: {result.decoding_mode}",
        f"Epoch: {result.epoch}",
        f"Stage: {result.training_stage}",
        f"Train loss: {result.train_loss}",
        f"Validation loss: {result.validation_loss}",
        f"Correct: {result.correct_questions}/{result.total_questions} "
        f"({result.correctness_percent:.1f}%)",
        f"Quality: {result.mean_quality_percent:.1f}%",
        f"Gibberish: {result.gibberish_answers}/{result.total_questions}",
        f"Training signal: {result.training_signal}",
    ]

    if previous is not None:
        lines.extend(
            [
                f"Correctness delta: {result.correctness_percent - previous.correctness_percent:+.1f}pp",
                f"Quality delta: {result.mean_quality_percent - previous.mean_quality_percent:+.1f}pp",
                f"Gibberish delta: {result.gibberish_answers - previous.gibberish_answers:+d}",
            ]
        )

    lines.append("")
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
    return "\n".join(lines)


def save_epoch_exam(
    *,
    result: EpochExamResult,
    exams_dir: str | Path,
    previous: EpochExamResult | None = None,
    prefix: str | None = None,
) -> tuple[Path, Path]:
    output = Path(exams_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = prefix or f"epoch_{result.epoch:03d}_exam"
    text_path = output / f"{stem}.txt"
    json_path = output / f"{stem}.json"

    text_path.write_text(render_exam_text(result, previous=previous), encoding="utf-8")
    json_path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return text_path, json_path
