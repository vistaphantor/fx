from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence

import torch
from torch.utils.data import Dataset, IterableDataset

if TYPE_CHECKING:
    from corpus.source import DatasetSource
    from src.language.tokenizer import BPETokenizer

SOURCE_ACCOUNTING_INTERVAL_TOKENS = 50_000


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
    contract and schedules training sequences by supervised prediction tokens,
    not by document count. Requested weights are therefore interpretable as token
    shares and are audited against the realized shares during iteration.
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

    Weighted sources are scheduled by realized non-pad target tokens. The
    scheduler prefetches one candidate sequence per source and chooses using the
    projected post-emission debt. This bounds short-interval overshoot better than
    choosing only from historical debt, which matters when five-minute exam
    remediation changes weights frequently.
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
        self.source_token_accounting: dict[str, int] = {}
        self.source_requested_shares: dict[str, float] = {}

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

    @staticmethod
    def _pair_supervised_tokens(pair: tuple[torch.Tensor, torch.Tensor], pad_id: int) -> int:
        return int((pair[1] != pad_id).sum().item())

    def _print_source_accounting(self, *, total_tokens: int) -> None:
        if total_tokens <= 0 or not self.source_token_accounting:
            return
        fragments: list[str] = []
        for source_id in sorted(self.source_token_accounting):
            actual = self.source_token_accounting[source_id] / total_tokens
            requested = self.source_requested_shares.get(source_id, 0.0)
            fragments.append(
                f"{source_id} requested={requested:.1%} actual={actual:.1%} "
                f"tokens={self.source_token_accounting[source_id]:,}"
            )
        print("[SourceTokens] " + " | ".join(fragments))

    def _weighted_pairs(self, stream: WeightedSourceStream) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Projected weighted-fair scheduling with realized-token accounting."""
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = stream.seed if worker_info is None else stream.seed + worker_info.id
        rng = random.Random(worker_seed)
        total_weight = sum(weight for _, weight in stream.sources)
        pad_id = self.tokenizer.pad_id()
        entries = [
            {
                "source": source,
                "source_id": source.source_id,
                "weight": float(weight) / total_weight,
                "tokens": 0,
                "iterator": self._pairs_from_texts(source.stream()),
                "pending": None,
                "pending_tokens": 0,
            }
            for source, weight in stream.sources
        ]
        self.source_requested_shares = {
            entry["source_id"]: entry["weight"] for entry in entries
        }
        self.source_token_accounting = {entry["source_id"]: 0 for entry in entries}
        total_tokens = 0
        next_report = SOURCE_ACCOUNTING_INTERVAL_TOKENS

        def refill(entry: dict) -> bool:
            while entry["pending"] is None:
                try:
                    pair = next(entry["iterator"])
                except StopIteration:
                    if not stream.repeat:
                        return False
                    entry["iterator"] = self._pairs_from_texts(entry["source"].stream())
                    try:
                        pair = next(entry["iterator"])
                    except StopIteration:
                        return False
                count = self._pair_supervised_tokens(pair, pad_id)
                if count <= 0:
                    continue
                entry["pending"] = pair
                entry["pending_tokens"] = count
            return True

        while entries:
            exhausted: list[int] = []
            for index, entry in enumerate(entries):
                if not refill(entry):
                    exhausted.append(index)
            for index in reversed(exhausted):
                entries.pop(index)
            if not entries:
                break

            # Compare where each source would land after its next indivisible
            # sequence. This sharply reduces oscillation from 512-token web chunks
            # competing with tiny arithmetic answers.
            scores = [
                (entry["tokens"] + entry["pending_tokens"]) / entry["weight"]
                for entry in entries
            ]
            minimum = min(scores)
            tied = [index for index, score in enumerate(scores) if abs(score - minimum) <= 1e-9]
            chosen = tied[rng.randrange(len(tied))]
            entry = entries[chosen]
            pair = entry["pending"]
            supervised_tokens = int(entry["pending_tokens"])
            entry["pending"] = None
            entry["pending_tokens"] = 0
            entry["tokens"] += supervised_tokens
            source_id = entry["source_id"]
            self.source_token_accounting[source_id] = self.source_token_accounting.get(source_id, 0) + supervised_tokens
            total_tokens += supervised_tokens
            if total_tokens >= next_report:
                self._print_source_accounting(total_tokens=total_tokens)
                while next_report <= total_tokens:
                    next_report += SOURCE_ACCOUNTING_INTERVAL_TOKENS
            yield pair

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(self.texts, WeightedSourceStream):
            yield from self._weighted_pairs(self.texts)
            return
        yield from self._pairs_from_texts(self._stream_texts())
