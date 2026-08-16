from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from corpus.source import DatasetSource, HFSource, SourceMetadata
from corpus.streamer import CorpusStreamer
from src.language.canonical_contract import prompt_family
from src.language.foundation_contract import FOUNDATION_MIN_AVAILABLE_CURRICULUM_TOKENS, FOUNDATION_SKILLS
from src.language.streaming_sources import (
    GuardedSource,
    HFSourceSpec,
    build_training_stream,
    load_hf_source_config,
    require_curriculum_capacity,
    stream_quality_accepts,
)
from src.language.tokenizer import BPETokenizer


class _MemorySource(DatasetSource):
    def __init__(self, rows: list[str], name: str = "memory"):
        self.rows = rows
        self.name = name

    @property
    def source_id(self) -> str:
        return f"test:{self.name}"

    def scan(self) -> SourceMetadata:
        return SourceMetadata(source_type="test", path=self.source_id, estimated_docs=len(self.rows))

    def stream(self):
        yield from self.rows

    def metadata(self) -> dict:
        return {"source_type": "test", "source_id": self.source_id}


class _FakeIterableDataset:
    def __init__(self, rows):
        self.rows = list(rows)
        self.shuffle_called = False

    def shuffle(self, *args, **kwargs):
        self.shuffle_called = True
        raise AssertionError("remote HF shuffle must not be used")

    def __iter__(self):
        yield from self.rows


def test_hf_source_requires_pinned_revision(monkeypatch):
    dataset = _FakeIterableDataset([{"question": "What is 2 + 2?", "answer": "4"}])
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **k: dataset))
    with pytest.raises(RuntimeError, match="hf_revision_must_be_pinned"):
        list(HFSource("example/test", shuffle_buffer_size=0).stream())


def test_hf_source_streams_canonical_chat_without_remote_shuffle(monkeypatch):
    calls = []
    dataset = _FakeIterableDataset(
        [
            {"question": "What is 2 + 2?", "answer": "<think>Calculate.</think>4"},
            {"messages": [
                {"role": "user", "content": "What is ATR?"},
                {"role": "assistant", "content": "ATR measures range."},
            ]},
        ]
    )

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return dataset

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    source = HFSource(
        "example/test", revision="abc123", max_examples=2,
        shuffle_buffer_size=128, seed=7,
    )
    texts = list(source.stream())
    assert len(texts) == 2
    assert all("<assistant>" in text for text in texts)
    assert calls[0][1]["revision"] == "abc123"
    assert dataset.shuffle_called is False


def test_guarded_source_excludes_validation_family_and_duplicates():
    heldout = "<bos>\n<user>\nWhat is RSI?\n</user>\n<assistant>\nHeldout.\n</assistant>\n<eos>"
    same_family = "<bos>\n<user>\nWhat is RSI?\n</user>\n<assistant>\nDifferent answer.\n</assistant>\n<eos>"
    safe = "<bos>\n<user>\nExplain ATR.\n</user>\n<assistant>\nATR measures range.\n</assistant>\n<eos>"
    guarded = GuardedSource(
        _MemorySource([same_family, safe, safe]),
        stage="foundation",
        excluded_families=frozenset({prompt_family(heldout)}),
    )
    assert list(guarded.stream()) == [safe]


def test_stream_quality_rejects_malformed_and_pathological_records():
    good = "<bos>\n<user>\nExplain volatility.\n</user>\n<assistant>\nVolatility describes how much price changes over time.\n</assistant>\n<eos>"
    malformed = "<bos>\n<user>\nQuestion only.\n</user>\n<eos>"
    repetitive = "<bos>\n" + "word " * 100 + "\n<eos>"
    assert stream_quality_accepts(good)
    assert not stream_quality_accepts(malformed)
    assert not stream_quality_accepts(repetitive)


def test_guarded_source_drops_bad_rows_before_training():
    good = "<bos>\n<user>\nExplain ATR.\n</user>\n<assistant>\nATR measures market range and volatility.\n</assistant>\n<eos>"
    bad = "<bos>\n<user>\nMissing assistant.\n</user>\n<eos>"
    guarded = GuardedSource(_MemorySource([bad, good]), stage="foundation")
    assert list(guarded.stream()) == [good]


def test_local_replay_is_rejected_by_authoritative_stream_only_builder():
    spec = HFSourceSpec(
        path="example/empty",
        revision="abc123",
        weight=1.0,
        stages=("foundation",),
        shuffle_buffer_size=0,
    )
    with pytest.raises(RuntimeError, match="local_language_training_disabled"):
        build_training_stream(
            specs=(spec,),
            stage="foundation",
            seed=7,
            local_replay=("<bos>\ntext\n<eos>",),
            local_weight=1.0,
            repeat=False,
        )


def test_corpus_streamer_packs_short_examples_into_context():
    texts = [
        "<bos>\n<user>\nA?\n</user>\n<assistant>\nA.\n</assistant>\n<eos>",
        "<bos>\n<user>\nB?\n</user>\n<assistant>\nB.\n</assistant>\n<eos>",
    ]
    tokenizer = BPETokenizer()
    tokenizer.train("\n".join(texts), vocab_size=512, min_frequency=1)
    dataset = CorpusStreamer(texts, tokenizer, seq_len=64)
    samples = list(dataset)
    assert samples
    decoded = [tokenizer.decode(x.tolist(), skip_special=False) for x, _ in samples]
    joined = "\n".join(decoded)
    assert "A?" in joined and "B?" in joined


def test_hf_config_rejects_unpinned_revision(tmp_path):
    path = tmp_path / "hf.json"
    path.write_text(
        json.dumps({"sources": [{"path": "example/test", "weight": 1.0, "stages": ["foundation"]}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="revision_required"):
        load_hf_source_config(path)


def test_foundation_capacity_is_declared_and_enforced(tmp_path):
    path = tmp_path / "hf.json"
    path.write_text(json.dumps({"sources": [{
        "path": "example/test",
        "revision": "a" * 40,
        "weight": 1.0,
        "stages": ["foundation"],
        "text_fields": ["text"],
        "available_tokens": FOUNDATION_MIN_AVAILABLE_CURRICULUM_TOKENS - 1,
        "skills": list(FOUNDATION_SKILLS),
    }]}), encoding="utf-8")
    specs = load_hf_source_config(path)
    with pytest.raises(RuntimeError, match="foundation_curriculum_capacity_insufficient"):
        require_curriculum_capacity(specs, "foundation")

    path.write_text(json.dumps({"sources": [{
        "path": "example/test",
        "revision": "a" * 40,
        "weight": 1.0,
        "stages": ["foundation"],
        "text_fields": ["text"],
        "available_tokens": FOUNDATION_MIN_AVAILABLE_CURRICULUM_TOKENS,
        "skills": list(FOUNDATION_SKILLS),
    }]}), encoding="utf-8")
    inventory = require_curriculum_capacity(load_hf_source_config(path), "foundation")
    assert inventory.available_tokens == FOUNDATION_MIN_AVAILABLE_CURRICULUM_TOKENS
    assert inventory.skills == tuple(sorted(FOUNDATION_SKILLS))
