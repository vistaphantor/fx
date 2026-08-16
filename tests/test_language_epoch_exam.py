from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
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
from src.language.foundation_contract import FOUNDATION_EXAM_QUESTIONS_PER_SKILL, FOUNDATION_SKILLS
from src.language.tokenizer import BPETokenizer, ENDASSISTANT


_ALL_QUESTIONS = FOUNDATION_EXAM + TRADING_EXTENSION


class _ScriptedModel:
    def __init__(self, tokenizer: BPETokenizer, outputs: dict[str, str]):
        self.tokenizer = tokenizer
        self.outputs = outputs
        self.max_seq_len = 512
        self.training = False

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = bool(mode)
        return self

    def named_parameters(self):
        return iter(())

    def __call__(self, idx: torch.Tensor, **_: object):
        batch, steps = idx.shape
        logits = torch.zeros(batch, steps, self.tokenizer.vocab_size, dtype=torch.float32)
        return logits, None

    def generate(self, idx: torch.Tensor, **_: object) -> torch.Tensor:
        prompt = self.tokenizer.decode(idx[0].tolist(), skip_special=False)
        question = next(question.prompt for question in _ALL_QUESTIONS if question.prompt in prompt)
        text = self.outputs.get(question, "I do not know.") + ENDASSISTANT
        continuation = self.tokenizer.encode(text, add_bos=False, add_eos=False)
        return torch.cat((idx, torch.tensor([continuation], dtype=torch.long)), dim=1)


@lru_cache(maxsize=1)
def _tokenizer() -> BPETokenizer:
    text = "\n".join(build_exam_prompt(question.prompt) for question in _ALL_QUESTIONS)
    text += "\n" + "\n".join(question.diagnostic_target or "" for question in _ALL_QUESTIONS)
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=2048, min_frequency=1)
    return tokenizer


def test_foundation_exam_has_fifty_questions_per_skill_and_question_one_is_gate():
    foundation = exam_questions("foundation")
    assert foundation == FOUNDATION_EXAM
    assert len(foundation) == len(FOUNDATION_SKILLS) * FOUNDATION_EXAM_QUESTIONS_PER_SKILL == 700
    counts = Counter(question.skill for question in foundation)
    assert counts == Counter({skill: 50 for skill in FOUNDATION_SKILLS})
    for skill in FOUNDATION_SKILLS:
        questions = [question for question in foundation if question.skill == skill]
        assert questions[0].conceptual_gate
        assert sum(question.conceptual_gate for question in questions) == 1


def test_trading_stage_extends_the_complete_foundation_exam():
    trading = exam_questions("trading_reasoning")
    assert trading[: len(FOUNDATION_EXAM)] == FOUNDATION_EXAM
    assert len(trading) > len(FOUNDATION_EXAM)


def test_epoch_exam_scores_known_answers_and_writes_artifacts(tmp_path: Path):
    tokenizer = _tokenizer()
    outputs = {question.prompt: question.diagnostic_target or "I do not know." for question in FOUNDATION_EXAM}
    model = _ScriptedModel(tokenizer, outputs)
    result = run_epoch_exam(
        model=model,  # type: ignore[arg-type]
        tokenizer=tokenizer,
        epoch=1,
        training_stage="foundation",
        train_loss=3.0,
        validation_loss=3.2,
        max_new_tokens=64,
    )
    assert result.correct_questions == result.total_questions == 700
    assert result.correctness_percent == 100.0
    assert sum(answer.conceptual_gate for answer in result.answers) == 14
    assert sum(bool(answer.decision_trace) for answer in result.answers) == 14
    text_path, json_path = save_epoch_exam(result=result, exams_dir=tmp_path)
    assert text_path.name == "epoch_001_exam.txt"
    assert json_path.name == "epoch_001_exam.json"
    assert "What is addition? Explain what it means." in text_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["epoch"] == 1
    assert payload["correctness_percent"] == 100.0


def test_repetitive_gibberish_fails_the_large_exam():
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
    assert result.gibberish_answers == result.total_questions == 700
    assert result.training_signal == "GIBBERISH"
