from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from corpus.source import HFSource
from corpus.streamer import CorpusStreamer, WeightedSourceStream
from src.language.canonical_contract import prompt_family
from src.language.streaming_sources import (
    CanonicalMemorySource,
    GuardedSource,
    build_training_stream,
    load_hf_source_config,
    stream_quality_accepts,
)
from src.language.tokenizer import BPETokenizer


class _FakeIterableDataset:
    def __init__(self, rows):
        self.rows = list(rows)
        self.shuffle_args = None

    def shuffle(self, *, seed: int, buffer_size: int):
        self.shuffle_args = (seed, buffer_size)
        return self

    def __iter__(self):
        yield from self.rows


def test_hf_source_requires_pinned_revision(monkeypatch):
    dataset = _FakeIterableDataset([{"question": "What is 2 + 2?", "answer": "4"}])
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **k: dataset))
    with pytest.raises(RuntimeError, match="hf_revision_must_be_pinned"):
        list(HFSource("example/test", shuffle_buffer_size=0).stream())


def test_hf_source_streams_same_canonical_chat_contract(monkeypatch):
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
    assert texts[0] == (
        "<bos>\n<user>\nWhat is 2 + 2?\n</user>\n"
        "<assistant>\n<think>\nCalculate.\n</think>\n4\n</assistant>\n<eos>"
    )
    assert calls[0][1]["revision"] == "abc123"
    assert dataset.shuffle_args == (7, 128)


def test_guarded_source_excludes_validation_family_and_duplicates():
    heldout = "<bos>\n<user>\nWhat is RSI?\n</user>\n<assistant>\nHeldout.\n</assistant>\n<eos>"
    same_family = "<bos>\n<user>\nWhat is RSI?\n</user>\n<assistant>\nDifferent answer.\n</assistant>\n<eos>"
    safe = "<bos>\n<user>\nExplain ATR.\n</user>\n<assistant>\nATR measures range.\n</assistant>\n<eos>"
    source = CanonicalMemorySource([same_family, safe, safe])
    guarded = GuardedSource(
        source,
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
    guarded = GuardedSource(CanonicalMemorySource([bad, good]), stage="foundation")
    assert list(guarded.stream()) == [good]


def test_specialist_local_replay_preserves_general_language_examples(monkeypatch):
    general = (
        "<bos>\n<user>\nExplain photosynthesis.\n</user>\n"
        "<assistant>\nPlants convert light into chemical energy.\n</assistant>\n<eos>"
    )
    trading = (
        "<bos>\n<user>\nWhat does ATR measure?\n</user>\n"
        "<assistant>\nATR measures market volatility and range.\n</assistant>\n<eos>"
    )
    # No HF row is required to prove replay behavior; use a tiny fake source
    # configuration whose iterator is empty, while local replay must remain.
    dataset = _FakeIterableDataset([])
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *a, **k: dataset),
    )
    from src.language.streaming_sources import HFSourceSpec

    spec = HFSourceSpec(
        path="example/empty",
        revision="abc123",
        weight=1.0,
        stages=("trading_reasoning",),
        shuffle_buffer_size=0,
    )
    stream = build_training_stream(
        specs=(spec,),
        stage="trading_reasoning",
        seed=7,
        local_replay=(general, trading),
        local_weight=1.0,
        repeat=False,
    )
    values = list(stream)
    assert general in values
    assert trading in values


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
    individual = sum(len(tokenizer.encode(text, False, False)) for text in texts)
    if individual <= 65:
        assert len(samples) == 1


def test_weighted_stream_repeats_finite_sources_deterministically():
    left = CanonicalMemorySource(["<bos>\nleft\n<eos>"], source_name="left")
    right = CanonicalMemorySource(["<bos>\nright\n<eos>"], source_name="right")
    stream_a = WeightedSourceStream([(left, 1.0), (right, 1.0)], seed=9, repeat=True)
    stream_b = WeightedSourceStream([(left, 1.0), (right, 1.0)], seed=9, repeat=True)
    iterator_a = iter(stream_a)
    iterator_b = iter(stream_b)
    assert [next(iterator_a) for _ in range(12)] == [next(iterator_b) for _ in range(12)]


def test_hf_config_rejects_unpinned_revision(tmp_path):
    path = tmp_path / "hf.json"
    path.write_text(
        json.dumps({"sources": [{"path": "example/test", "weight": 1.0, "stages": ["foundation"]}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="revision_required"):
        load_hf_source_config(path)
