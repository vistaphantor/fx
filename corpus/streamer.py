from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import torch
from torch.utils.data import IterableDataset, Dataset

if TYPE_CHECKING:
    from src.language.tokenizer import BPETokenizer
    from corpus.shard import ShardWriter


class ShardDataset(Dataset):
    def __init__(self, shard_paths: list[Path]):
        from corpus.shard import ShardWriter
        self.samples: list[tuple[list[int], list[int]]] = []
        for p in shard_paths:
            self.samples.extend(ShardWriter.read_shard(p))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class CorpusStreamer(IterableDataset):
    def __init__(
        self,
        texts: list[str] | Iterator[str],
        tokenizer: "BPETokenizer",
        seq_len: int = 512,
        stride: int = 256,
        seed: int = 42,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        self.seed = seed

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = self.seed if worker_info is None else self.seed + worker_info.id
        rng = random.Random(worker_seed)

        if isinstance(self.texts, list):
            doc_list = list(self.texts)
            rng.shuffle(doc_list)
            text_iter = iter(doc_list)
        else:
            text_iter = self.texts

        token_stream = []
        for text in text_iter:
            ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            token_stream.extend(ids)

            while len(token_stream) >= self.seq_len + 1:
                chunk = token_stream[: self.seq_len + 1]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y
                token_stream = token_stream[self.stride :]

        if len(token_stream) > 1:
            pad_needed = (self.seq_len + 1) - len(token_stream)
            chunk = token_stream + [self.tokenizer.pad_id()] * pad_needed
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            yield x, y
