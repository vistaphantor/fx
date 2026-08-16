from __future__ import annotations

from itertools import islice

import pytest
import torch

from corpus.source import DatasetSource, SourceMetadata
from corpus.streamer import CorpusStreamer, WeightedSourceStream
from src.language.foundation_skill_sources import PrimitiveArithmeticSource
from src.language.hard_negative_objective import _answer_anchor_mask
from src.language.streaming_sources import _parse_spec


class _FixedSource(DatasetSource):
    def __init__(self, name: str, token_count: int):
        self.name = name
        self.token_count = token_count

    @property
    def source_id(self) -> str:
        return f"test:{self.name}"

    def scan(self) -> SourceMetadata:
        return SourceMetadata(source_type="test", path=self.source_id, estimated_docs=1)

    def metadata(self) -> dict:
        return {"source_id": self.source_id}

    def stream(self):
        while True:
            yield f"{self.name}:{self.token_count}"


class _PadOnlyTokenizer:
    def pad_id(self) -> int:
        return 0


class _SyntheticPairStreamer(CorpusStreamer):
    def _pairs_from_texts(self, texts):
        for text in texts:
            name, raw_count = text.split(":", 1)
            count = int(raw_count)
            marker = 11 if name == "short" else 22
            x = torch.full((self.seq_len,), marker, dtype=torch.long)
            y = torch.zeros((self.seq_len,), dtype=torch.long)
            y[:count] = marker
            yield x, y


def test_weighted_stream_balances_supervised_tokens_not_document_count():
    stream = WeightedSourceStream(
        [(_FixedSource("short", 2), 1.0), (_FixedSource("long", 8), 1.0)],
        seed=7,
        repeat=True,
    )
    dataset = _SyntheticPairStreamer(stream, _PadOnlyTokenizer(), seq_len=16, seed=7)
    consumed = {11: 0, 22: 0}
    pairs = list(islice(iter(dataset), 120))
    for x, y in pairs:
        marker = int(x[0].item())
        consumed[marker] += int((y != 0).sum().item())

    total = consumed[11] + consumed[22]
    assert total > 0
    short_share = consumed[11] / total
    assert 0.45 <= short_share <= 0.55
    short_examples = sum(1 for x, _ in pairs if int(x[0].item()) == 11)
    long_examples = len(pairs) - short_examples
    assert short_examples > long_examples * 2


def test_hard_negative_objective_targets_answer_anchors_only():
    pad = 0
    targets = torch.tensor([
        [0, 0, 4, 5, 6, 0, 0, 7, 8, 9],
        [0, 3, 4, 5, 6, 7, 0, 0, 0, 0],
    ])
    mask = _answer_anchor_mask(targets, pad_id=pad)
    expected = torch.tensor([
        [False, False, True, True, False, False, False, True, True, False],
        [False, True, True, False, False, False, False, False, False, False],
    ])
    assert torch.equal(mask, expected)


def test_hf_revision_must_be_immutable_commit_sha():
    payload = {
        "path": "owner/dataset",
        "revision": "main",
        "weight": 1.0,
        "stages": ["foundation"],
        "text_fields": ["text"],
    }
    with pytest.raises(ValueError, match="hf_source_revision_must_be_immutable_commit"):
        _parse_spec(payload)

    payload["revision"] = "a" * 40
    spec = _parse_spec(payload)
    assert spec.revision == "a" * 40


def test_primitive_arithmetic_teaches_symbol_surface_with_result_first():
    target_prompt = "<user>\nWhat is 2 + 2?\n</user>"
    found = None
    for text in PrimitiveArithmeticSource().stream():
        if target_prompt in text:
            found = text
            break
    assert found is not None
    assistant = found.split("<assistant>\n", 1)[1].split("\n</assistant>", 1)[0]
    assert assistant.startswith("4")
    assert "2 + 2 + 2" not in assistant
