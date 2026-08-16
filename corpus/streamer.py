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
    """Deterministically mix independent dataset sources without materializing them.

    ``CorpusStreamer`` treats this object as an authoritative weighted source
    contract and schedules training sequences by *supervised prediction tokens*,
    not by document count. ``iter_with_source`` remains useful for lightweight
    sampling/auditing where tokenization is intentionally unavailable.
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

    def iter_with_source(self) -> Iterator[tuple[str, str]]:
        """Yield accepted text together with the authoritative source id."""
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
                yield entry["source"].source_id, next(entry["iterator"])
            except StopIteration:
                if not self.repeat:
                    active.pop(chosen)
                    continue
                entry["iterator"] = iter(entry["source"].stream())
                try:
                    yield entry["source"].source_id, next(entry["iterator"])
                except StopIteration:
                    active.pop(chosen)

    def __iter__(self) -> Iterator[str]:
        for _, text in self.iter_with_source():
            yield text


class CorpusStreamer(IterableDataset):
    """Boundary-preserving, bounded-memory role-aware training stream.

    When fed a ``WeightedSourceStream`` the scheduler measures the actual number
    of non-pad loss targets emitted by every source. The next source is selected
    by weighted fair queuing on that measured token debt. This prevents a single
    long web document from receiving many optimizer windows merely because one
    document draw happened to be selected, which previously overwhelmed short
    arithmetic and conversation examples despite apparently large source weights.
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
            yield from buffered_shuffle(source, buffer_size=self.shuffle_buffer_size, seed=worker_seed)
        else:
            yield from source

    def _tensor_pair(self, sequence: list[int]) -> tuple[torch.Tensor, torch.Tensor] | None:
        from src.language.loss_objective import build_loss_targets

        x, y, stats = build_loss_targets(
            sequence,
            seq_len=self.seq_len,
            pad_id=self.tokenizer.pad_id(),
        )
        if stats.prediction_tokens <= 0:
            return None
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def _yield_pair(self, sequence: list[int]) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        pair = self._tensor_pair(sequence)
        if pair is not None:
            yield pair

    def _pairs_from_texts(self, texts: Iterable[str]) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Pack one source independently so every emitted token has one owner."""
        from src.language.training_pipeline import build_example_sequences

        max_tokens = self.seq_len + 1
        pack: list[int] = []
        for text in texts:
            if not text or not text.strip():
                continue
            raw_ids = self.tokenizer.encode(text, add_bos=False, add_eos=False)
            if len(raw_ids) < 2:
                continue

            if len(raw_ids) > max_tokens:
                if len(pack) >= 2:
                    yield from self._yield_pair(pack)
                    pack = []
                for chunk in build_example_sequences([text], self.tokenizer, seq_len=self.seq_len):
                    yield from self._yield_pair(chunk)
                continue

            if not pack:
                pack = list(raw_ids)
            elif len(pack) + len(raw_ids) <= max_tokens:
                pack.extend(raw_ids)
            else:
                yield from self._yield_pair(pack)
                pack = list(raw_ids)

        if len(pack) >= 2:
            yield from self._yield_pair(pack)

    def _weighted_pairs(self, stream: WeightedSourceStream) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Weighted-fair schedule using actual supervised-token consumption."""
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = stream.seed if worker_info is None else stream.seed + worker_info.id
        rng = random.Random(worker_seed)
        total_weight = sum(weight for _, weight in stream.sources)
        entries = [
            {
                "source": source,
                "weight": float(weight) / total_weight,
                "tokens": 0,
                "iterator": self._pairs_from_texts(source.stream()),
            }
            for source, weight in stream.sources
        ]
        pad_id = self.tokenizer.pad_id()

        while entries:
            # Weighted fair queuing: the source with the smallest consumed-token
            # share relative to its requested share has the largest training debt.
            scores = [entry["tokens"] / entry["weight"] for entry in entries]
            minimum = min(scores)
            tied = [index for index, score in enumerate(scores) if abs(score - minimum) <= 1e-9]
            chosen = tied[rng.randrange(len(tied))]
            entry = entries[chosen]
            try:
                pair = next(entry["iterator"])
            except StopIteration:
                if not stream.repeat:
                    entries.pop(chosen)
                    continue
                entry["iterator"] = self._pairs_from_texts(entry["source"].stream())
                try:
                    pair = next(entry["iterator"])
                except StopIteration:
                    entries.pop(chosen)
                    continue

            _, targets = pair
            supervised_tokens = int((targets != pad_id).sum().item())
            if supervised_tokens <= 0:
                continue
            entry["tokens"] += supervised_tokens
            yield pair

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(self.texts, WeightedSourceStream):
            yield from self._weighted_pairs(self.texts)
            return
        yield from self._pairs_from_texts(self._stream_texts())
