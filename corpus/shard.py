from __future__ import annotations

import hashlib
import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from src.language.tokenizer import BPETokenizer

MAGIC = b"VST1"  # 0x56535431


def get_tokenizer_fingerprint(tokenizer: "BPETokenizer") -> str:
    vocab_items = sorted(tokenizer.vocab.items())
    merges_items = tokenizer.merges
    payload = json.dumps({"v": vocab_items, "m": merges_items}, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class ShardHeader:
    num_chunks: int
    seq_len: int
    vocab_size: int


class ShardWriter:
    def __init__(
        self,
        output_dir: str | Path,
        tokenizer: "BPETokenizer",
        seq_len: int = 512,
        stride: int = 256,
        chunks_per_shard: int = 100_000,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        self.chunks_per_shard = chunks_per_shard
        self.fingerprint = get_tokenizer_fingerprint(tokenizer)

    def write_shards(
        self,
        texts: Iterator[str] | list[str],
        split: str = "train",
        on_shard_created: None | callable = None,
    ) -> list[Path]:
        shard_paths = []
        shard_idx = 1
        buffer_chunks = []
        token_stream = []

        for text in texts:
            ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            token_stream.extend(ids)

            while len(token_stream) >= self.seq_len + 1:
                chunk = token_stream[: self.seq_len + 1]
                buffer_chunks.append(chunk)
                token_stream = token_stream[self.stride :]

                if len(buffer_chunks) >= self.chunks_per_shard:
                    sp = self._flush_shard(buffer_chunks, split, shard_idx)
                    shard_paths.append(sp)
                    if on_shard_created:
                        on_shard_created(sp, len(buffer_chunks))
                    shard_idx += 1
                    buffer_chunks = []

        if len(token_stream) > 1:
            pad_needed = (self.seq_len + 1) - len(token_stream)
            chunk = token_stream + [self.tokenizer.pad_id()] * pad_needed
            buffer_chunks.append(chunk)

        if buffer_chunks:
            sp = self._flush_shard(buffer_chunks, split, shard_idx)
            shard_paths.append(sp)
            if on_shard_created:
                on_shard_created(sp, len(buffer_chunks))

        self.write_manifest(split_shards={split: [p.name for p in shard_paths]})
        return shard_paths

    def _flush_shard(self, chunks: list[list[int]], split: str, index: int) -> Path:
        shard_name = f"{split}_{index:05d}.bin"
        path = self.output_dir / shard_name
        num_chunks = len(chunks)

        with path.open("wb") as fh:
            # Write Header: magic (4B), num_chunks (4B uint32), seq_len (4B uint32), vocab_size (4B uint32)
            header = MAGIC + struct.pack("<III", num_chunks, self.seq_len, self.tokenizer.vocab_size)
            fh.write(header)

            # Flatten uint16 token IDs
            flat_ids = []
            for c in chunks:
                flat_ids.extend(c)

            fmt = f"<{len(flat_ids)}H"
            fh.write(struct.pack(fmt, *flat_ids))

        return path

    def write_manifest(self, split_shards: dict[str, list[str]]) -> Path:
        manifest_path = self.output_dir / "manifest.json"
        manifest = {
            "version": 1,
            "tokenizer_fingerprint": self.fingerprint,
            "seq_len": self.seq_len,
            "stride": self.stride,
            "vocab_size": self.tokenizer.vocab_size,
            "created_at": int(time.time()),
            "splits": split_shards,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path

    @staticmethod
    def read_shard_header(path: Path) -> ShardHeader:
        with path.open("rb") as fh:
            magic = fh.read(4)
            if magic != MAGIC:
                raise ValueError(f"Invalid shard magic header in {path}: {magic}")
            num_chunks, seq_len, vocab_size = struct.unpack("<III", fh.read(12))
            return ShardHeader(num_chunks=num_chunks, seq_len=seq_len, vocab_size=vocab_size)

    @staticmethod
    def read_shard(path: Path) -> list[tuple[list[int], list[int]]]:
        header = ShardWriter.read_shard_header(path)
        with path.open("rb") as fh:
            fh.seek(16)  # skip 16-byte header
            total_elements = header.num_chunks * (header.seq_len + 1)
            raw = fh.read(total_elements * 2)
            fmt = f"<{total_elements}H"
            flat_ids = list(struct.unpack(fmt, raw))

        chunk_size = header.seq_len + 1
        samples = []
        for i in range(header.num_chunks):
            start = i * chunk_size
            c = flat_ids[start : start + chunk_size]
            x = c[:-1]
            y = c[1:]
            samples.append((x, y))
        return samples
