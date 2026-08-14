from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence

import torch
from torch.utils.data import Dataset, IterableDataset

if TYPE_CHECKING:
    from corpus.shard import ShardWriter
    from corpus.source import DatasetSource
    from src.language.tokenizer import BPETokenizer


class ShardDataset(Dataset):
    def __init__(self, shard_paths: list[Path]):
        from corpus.shard import ShardWriter

        self.samples: list[tuple[list[int], list[int]]] = []
        for path in shard_paths:
            self.samples.extend(ShardWriter.read_shard(path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def buffered_shuffle(
    values: Iterable[str],
    *,
    buffer_size: int,
    seed: int,
) -> Iterator[str]:
    """Bounded-memory deterministic shuffle for arbitrary text streams."""
    if buffer_size <= 1:
        yield from values
        return

    rng = random.Random(seed)
    iterator = iter(values)
    buffer: list[str] = []
    for _ in range(buffer_size):
        try:
            buffer.append(next(iterator))
        except StopIteration:
            break

    while buffer:
        index = rng.randrange(len(buffer))
        yield buffer[index]
        try:
            buffer[index] = next(iterator)
        except StopIteration:
            buffer.pop(index)


class WeightedSourceStream:
    """Deterministically mix independent dataset sources without materializing them.

    A source is reopened only after exhaustion when ``repeat=True``. Weights are
    sampling probabilities, not duplication factors, so large external datasets
    do not have to be copied locally to influence the curriculum.
    """

    def __init__(
        self,
        sources: Sequence[tuple["DatasetSource", float]],
        *,
        seed: int = 42,
        repeat: bool = False,
    ):
        if not sources:
            raise ValueError("WeightedSourceStream requires at least one source")
        normalized: list[tuple["DatasetSource", float]] = []
        for source, weight in sources:
            value = float(weight)
            if value <= 0:
                raise ValueError("source weights must be positive")
            normalized.append((source, value))
        self.sources = tuple(normalized)
        self.seed = int(seed)
        self.repeat = bool(repeat)

    def __iter__(self) -> Iterator[str]:
        rng = random.Random(self.seed)
        active = [
            {
                "source": source,
                "weight": weight,
                "iterator": iter(source.stream()),
            }
            for source, weight in self.sources
        ]

        while active:
            weights = [entry["weight"] for entry in active]
            chosen = rng.choices(range(len(active)), weights=weights, k=1)[0]
            entry = active[chosen]
            try:
                yield next(entry["iterator"])
            except StopIteration:
                if self.repeat:
                    entry["iterator"] = iter(entry["source"].stream())
                    try:
                        yield next(entry["iterator"])
                    except StopIteration:
                        active.pop(chosen)
                else:
                    active.pop(chosen)


class CorpusStreamer(IterableDataset):
    """Boundary-preserving streaming next-token dataset.

    Each canonical example is tokenized independently. Short examples may be
    packed together only at explicit ``<eos><bos>`` boundaries; long examples
    are chunked by the authoritative language training pipeline. There is no
    global flat token stream.
    """

    def __init__(
        self,
        texts: Iterable[str],
        tokenizer: "BPETokenizer",
        seq_len: int = 512,
        *,
        seed: int = 42,
        shuffle_buffer_size: int = 0,
    ):
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        if shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be >= 0")
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.seed = int(seed)
        self.shuffle_buffer_size = int(shuffle_buffer_size)

    def _stream_texts(self) -> Iterator[str]:
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = self.seed if worker_info is None else self.seed + worker_info.id
        source = iter(self.texts)
        if self.shuffle_buffer_size > 1:
            yield from buffered_shuffle(
                source,
                buffer_size=self.shuffle_buffer_size,
                seed=worker_seed,
            )
        else:
            yield from source

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        from src.language.training_pipeline import build_example_sequences

        pending: deque[list[int]] = deque()
        for text in self._stream_texts():
            if not text or not text.strip():
                continue
            sequences = build_example_sequences(
                [text],
                self.tokenizer,
                seq_len=self.seq_len,
            )
            pending.extend(sequences)
            while pending:
                sequence = pending.popleft()[: self.seq_len + 1]
                x = sequence[:-1]
                y = sequence[1:]
                if len(x) < self.seq_len:
                    padding = self.seq_len - len(x)
                    x = x + [self.tokenizer.pad_id()] * padding
                    y = y + [self.tokenizer.pad_id()] * padding
                yield (
                    torch.tensor(x, dtype=torch.long),
                    torch.tensor(y, dtype=torch.long),
                )
