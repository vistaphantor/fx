from __future__ import annotations

import sys
from types import SimpleNamespace

from corpus.source import HFSource
from corpus.streamer import CorpusStreamer, WeightedSourceStream
from src.language.streaming_sources import CanonicalMemorySource
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


def test_hf_source_streams_canonical_chat(monkeypatch):
    dataset = _FakeIterableDataset(
        [
            {"question": "What is 2 + 2?", "answer": "4"},
            {
                "messages": [
                    {"role": "user", "content": "What is ATR?"},
                    {"role": "assistant", "content": "ATR measures range."},
                ]
            },
        ]
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: dataset),
    )
    source = HFSource(
        "example/test",
        max_examples=2,
        shuffle_buffer_size=128,
        seed=7,
    )
    texts = list(source.stream())
    assert len(texts) == 2
    assert texts[0].startswith("<bos>\n<user>\n")
    assert "Human:" not in texts[0]
    assert "Assistant:" not in texts[0]
    assert "<assistant>\n4\n</assistant>" in texts[0]
    assert texts[0].endswith("<eos>")
    assert dataset.shuffle_args == (7, 128)


def test_corpus_streamer_never_crosses_examples_without_explicit_boundaries():
    texts = [
        "<bos>\n<user>\nA?\n</user>\n<assistant>\nA.\n</assistant>\n<eos>",
        "<bos>\n<user>\nB?\n</user>\n<assistant>\nB.\n</assistant>\n<eos>",
    ]
    tokenizer = BPETokenizer()
    tokenizer.train("\n".join(texts), vocab_size=512, min_frequency=1)
    dataset = CorpusStreamer(texts, tokenizer, seq_len=64)
    samples = list(dataset)
    assert samples
    decoded = [
        tokenizer.decode(x.tolist(), skip_special=False)
        for x, _ in samples
    ]
    assert any("A?" in text for text in decoded)
    assert any("B?" in text for text in decoded)


def test_weighted_stream_repeats_finite_sources_deterministically():
    left = CanonicalMemorySource(["<bos>\nleft\n<eos>"], source_name="left")
    right = CanonicalMemorySource(["<bos>\nright\n<eos>"], source_name="right")
    stream_a = WeightedSourceStream([(left, 1.0), (right, 1.0)], seed=9, repeat=True)
    stream_b = WeightedSourceStream([(left, 1.0), (right, 1.0)], seed=9, repeat=True)
    iterator_a = iter(stream_a)
    iterator_b = iter(stream_b)
    first_a = [next(iterator_a) for _ in range(12)]
    first_b = [next(iterator_b) for _ in range(12)]
    assert first_a == first_b
    assert set(first_a) == {"<bos>\nleft\n<eos>", "<bos>\nright\n<eos>"}
