from __future__ import annotations

import json
from pathlib import Path

import torch

from src.language.exam import (
    FOUNDATION_EXAM,
    TRADING_EXTENSION,
    build_exam_prompt,
    exam_questions,
    run_epoch_exam,
    save_epoch_exam,
)
from src.language.tokenizer import BPETokenizer, ENDASSISTANT


class _ScriptedModel:
    def __init__(self, tokenizer: BPETokenizer, outputs: dict[str, str]):
        self.tokenizer = tokenizer
        self.outputs = outputs
        self.max_seq_len = 256

    def eval(self):
        return self

    def generate(self, idx: torch.Tensor, **_: object) -> torch.Tensor:
        prompt = self.tokenizer.decode(idx[0].tolist(), skip_special=False)
        question = next(
            question.prompt
            for question in FOUNDATION_EXAM + TRADING_EXTENSION
            if question.prompt in prompt
        )
        text = self.outputs.get(question, "I do not know.") + ENDASSISTANT
        continuation = self.tokenizer.encode(text, add_bos=False, add_eos=False)
        return torch.cat((idx, torch.tensor([continuation], dtype=torch.long)), dim=1)


def _tokenizer() -> BPETokenizer:
    text = "\n".join(
        build_exam_prompt(question.prompt)
        for question in FOUNDATION_EXAM + TRADING_EXTENSION
    ) + "\n4 210 3 rising loss Charlie difference bid ask volatility range"
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=1024, min_frequency=1)
    return tokenizer


def test_foundation_exam_is_fixed_and_trading_stage_extends_it():
    foundation = exam_questions("foundation")
    trading = exam_questions("trading_reasoning")
    assert foundation == FOUNDATION_EXAM
    assert trading[: len(FOUNDATION_EXAM)] == FOUNDATION_EXAM
    assert len(trading) > len(foundation)


def test_epoch_exam_scores_known_answers_and_writes_artifacts(tmp_path: Path):
    tokenizer = _tokenizer()
    outputs = {
        "What is 2 + 2?": "4",
        "What is 15 multiplied by 14?": "210",
        "If 2x + 5 = 11, what is x?": "x = 3",
        "In trading, what does bullish mean?": "Bullish means price is expected to rise.",
        "What does risk mean in trading?": "Risk is the possibility of loss.",
        "Alice is older than Bob, and Bob is older than Charlie. Who is youngest?": "Charlie.",
        "What is the bid-ask spread?": "The difference between bid and ask.",
        "What does ATR measure in market analysis?": "ATR measures volatility and true range.",
    }
    model = _ScriptedModel(tokenizer, outputs)
    result = run_epoch_exam(
        model=model,  # type: ignore[arg-type]
        tokenizer=tokenizer,
        epoch=1,
        training_stage="foundation",
        train_loss=3.0,
        validation_loss=3.2,
        max_new_tokens=32,
    )
    assert result.correct_questions == result.total_questions
    assert result.correctness_percent == 100.0
    text_path, json_path = save_epoch_exam(result=result, exams_dir=tmp_path)
    assert text_path.name == "epoch_001_exam.txt"
    assert json_path.name == "epoch_001_exam.json"
    assert "What is 2 + 2?" in text_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["epoch"] == 1
    assert payload["correctness_percent"] == 100.0


def test_repetitive_gibberish_is_visible_in_exam_result():
    tokenizer = _tokenizer()
    model = _ScriptedModel(
        tokenizer,
        {question.prompt: "the the the the the the the the" for question in FOUNDATION_EXAM},
    )
    result = run_epoch_exam(
        model=model,  # type: ignore[arg-type]
        tokenizer=tokenizer,
        epoch=2,
        training_stage="foundation",
        train_loss=2.9,
        validation_loss=3.1,
        max_new_tokens=32,
    )
    assert result.correct_questions == 0
    assert result.gibberish_answers == result.total_questions
    assert all("high_word_repetition" in answer.gibberish_flags for answer in result.answers)


def test_concatenated_subword_loops_are_gibberish():
    tokenizer = _tokenizer()
    loops = {
        question.prompt: "parkparkparkparkparkparklesslesslesslesslessless"
        for question in FOUNDATION_EXAM
    }
    model = _ScriptedModel(tokenizer, loops)
    result = run_epoch_exam(
        model=model,  # type: ignore[arg-type]
        tokenizer=tokenizer,
        epoch=0,
        training_stage="foundation",
        train_loss=None,
        validation_loss=None,
        max_new_tokens=32,
    )
    assert result.correct_questions == 0
    assert result.gibberish_answers == result.total_questions
    assert result.training_signal == "GIBBERISH"
    assert all(
        any(
            flag in answer.gibberish_flags
            for flag in ("high_character_repetition", "periodic_repetition", "run_on_fragment")
        )
        for answer in result.answers
    )
    assert result.mean_quality_percent < 50.0
