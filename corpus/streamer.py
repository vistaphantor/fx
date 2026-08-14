from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence

import torch
from torch.utils.data import Dataset, IterableDataset

if TYPE_CHECKING:
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


def buffered_shuffle(values: Iterable[str], *, buffer_size: int, seed: int) -> Iterator[str]:
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
    """Deterministically mix independent dataset sources without materializing them."""

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
            {"source": source, "weight": weight, "iterator": iter(source.stream())}
            for source, weight in self.sources
        ]
        while active:
            weights = [entry["weight"] for entry in active]
            chosen = rng.choices(range(len(active)), weights=weights, k=1)[0]
            entry = active[chosen]
            try:
                yield next(entry["iterator"])
            except StopIteration:
                if not self.repeat:
                    active.pop(chosen)
                    continue
                entry["iterator"] = iter(entry["source"].stream())
                try:
                    yield next(entry["iterator"])
                except StopIteration:
                    active.pop(chosen)


class CorpusStreamer(IterableDataset):
    """Boundary-preserving, bounded-memory role-aware training stream."""

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
            yield from buffered_shuffle(source, buffer_size=self.shuffle_buffer_size, seed=worker_seed)
        else:
            yield from source

    def _tensor_pair(self, sequence: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        from src.language.loss_objective import build_loss_targets

        x, y, _ = build_loss_targets(
            sequence,
            self.tokenizer,
            seq_len=self.seq_len,
        )
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        from src.language.training_pipeline import build_example_sequences

        max_tokens = self.seq_len + 1
        pack: list[int] = []

        for text in self._stream_texts():
            if not text or not text.strip():
                continue
            raw_ids = self.tokenizer.encode(text, add_bos=False, add_eos=False)
            if len(raw_ids) < 2:
                continue

            if len(raw_ids) > max_tokens:
                if len(pack) >= 2:
                    yield self._tensor_pair(pack)
                    pack = []
                for chunk in build_example_sequences([text], self.tokenizer, seq_len=self.seq_len):
                    yield self._tensor_pair(chunk)
                continue

            if not pack:
                pack = list(raw_ids)
            elif len(pack) + len(raw_ids) <= max_tokens:
                pack.extend(raw_ids)
            else:
                yield self._tensor_pair(pack)
                pack = list(raw_ids)

        if len(pack) >= 2:
            yield self._tensor_pair(pack)
