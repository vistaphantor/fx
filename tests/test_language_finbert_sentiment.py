from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.language.finbert_sentiment import FinBertSentimentAnalyzer, resolve_torch_device


class _FakeTokenizer:
    def __call__(self, texts, *, padding, truncation, max_length, return_tensors):
        batch = len(texts)
        return {
            "input_ids": torch.ones((batch, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch, 4), dtype=torch.long),
        }


class _FakeFinBertModel:
    def __init__(self, logits: torch.Tensor) -> None:
        self._logits = logits
        self.config = SimpleNamespace(id2label={0: "positive", 1: "negative", 2: "neutral"})

    def __call__(self, **kwargs):
        batch = kwargs["input_ids"].shape[0]
        return SimpleNamespace(logits=self._logits[:batch])


def test_finbert_analyzer_scores_direction_confidence_and_net_sentiment() -> None:
    logits = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=torch.float32,
    )
    analyzer = FinBertSentimentAnalyzer(
        model=_FakeFinBertModel(logits),
        tokenizer=_FakeTokenizer(),
        device=torch.device("cpu"),
    )
    results = analyzer.score_batch(
        [
            "Strong US payrolls lift dollar yields.",
            "Dovish central bank cut weakens currency.",
            "Markets await tomorrow's inflation print.",
        ],
        batch_size=3,
    )
    assert [item.label for item in results] == ["positive", "negative", "neutral"]
    assert results[0].sentiment_score > 0.8
    assert results[1].sentiment_score < -0.8
    assert abs(results[2].sentiment_score) < 0.05
    assert all(0.0 < item.confidence <= 1.0 for item in results)


def test_finbert_analyzer_rejects_empty_text() -> None:
    analyzer = FinBertSentimentAnalyzer(
        model=_FakeFinBertModel(torch.zeros((1, 3))),
        tokenizer=_FakeTokenizer(),
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="empty"):
        analyzer.score_text("   ")


def test_resolve_torch_device_requires_cuda_when_requested(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_torch_device("auto", require_cuda=False) == torch.device("cpu")
    with pytest.raises(RuntimeError, match="finbert_cuda_required"):
        resolve_torch_device("auto", require_cuda=True)
