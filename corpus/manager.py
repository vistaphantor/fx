from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from corpus.auditor import CorpusAuditor
from corpus.checkpoint import IndexCheckpoint
from corpus.dashboard import CorpusDashboard
from corpus.dedup import Deduplicator
from corpus.discovery import DatasetDiscovery
from corpus.quality import QualityFilter
from corpus.registry import CorpusRegistry, DatasetRecord, file_sha256
from corpus.shard import ShardWriter, get_tokenizer_fingerprint
from corpus.source import DatasetSource, HFSource, LocalSource
from corpus.streamer import ShardDataset

if TYPE_CHECKING:
    from src.language.tokenizer import BPETokenizer


class SequenceDataset(Dataset):
    def __init__(self, data_chunks: list[list[int]]):
        self.chunks = data_chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.chunks[idx]
        return torch.tensor(c[:-1], dtype=torch.long), torch.tensor(c[1:], dtype=torch.long)


class CorpusManager:
    def __init__(
        self,
        local_roots: Optional[list[str | Path]] = None,
        hf_sources: Optional[list[dict | HFSource]] = None,
        tokenizer: Optional["BPETokenizer"] = None,
        target_tokens: int = 1_800_000_000_000,
        seq_len: int = 512,
        stride: int = 256,
        db_path: str | Path = "data/corpus_registry.db",
        shard_dir: str | Path = "data/corpus_shards",
        json_export_path: str | Path = "data/datasets/registry.json",
        checkpoint_path: str | Path = "data/index_checkpoint.json",
    ):
        self.local_roots = [Path(r) for r in (local_roots or [])]
        self.tokenizer = tokenizer
        self.target_tokens = target_tokens
        self.seq_len = seq_len
        self.stride = stride
        self.db_path = Path(db_path)
        self.shard_dir = Path(shard_dir)
        self.json_export_path = Path(json_export_path)
        self.checkpoint_path = Path(checkpoint_path)

        self.sources: list[DatasetSource] = []

        if local_roots:
            disco = DatasetDiscovery(self.local_roots)
            self.sources.extend(disco.discover())

        if hf_sources:
            for item in hf_sources:
                if isinstance(item, HFSource):
                    self.sources.append(item)
                elif isinstance(item, dict):
                    self.sources.append(HFSource(**item))

        self.registry = CorpusRegistry(self.db_path)
        self.auditor = CorpusAuditor(target_tokens=target_tokens)
        self.quality = QualityFilter()
        self.dedup = Deduplicator()
        self.dashboard = CorpusDashboard()
        self.checkpoint = IndexCheckpoint(self.checkpoint_path)

        self.cached_cleaned_texts: list[str] = []
        self.train_chunks: list[list[int]] = []
        self.val_chunks: list[list[int]] = []

    def index(self, force_reindex: bool = False) -> None:
        print(f"[CorpusManager] Indexing {len(self.sources)} dataset sources into SQLite registry...", flush=True)
        t0 = time.time()
        self.auditor.reset()
        self.dedup.reset()
        self.cached_cleaned_texts.clear()

        ckpt_state = self.checkpoint.load()
        processed_files = ckpt_state.get("files_processed", 0)

        file_counter = 0
        total_raw = 0
        accepted_count = 0
        duplicate_count = 0

        for src in self.sources:
            meta = src.scan()
            src_id = src.source_id
            dataset_name = Path(meta.path).name if meta.source_type == "local" else src_id

            sha = ""
            if meta.source_type == "local" and Path(meta.path).is_file():
                sha = file_sha256(Path(meta.path))

            record = DatasetRecord(
                source_type=meta.source_type,
                path=meta.path,
                name=dataset_name,
                sha256=sha,
                size_bytes=meta.size_bytes,
            )
            dataset_id = self.registry.upsert_dataset(record)

            doc_offset = 0
            for raw_text in src.stream():
                file_counter += 1
                if file_counter < processed_files and not force_reindex:
                    continue

                total_raw += 1
                qscore = self.quality.score(raw_text)
                if not qscore.accepted:
                    continue
                accepted_count += 1

                dedup_res = self.dedup.exact([raw_text])
                if not dedup_res.kept:
                    duplicate_count += 1
                    continue

                text = dedup_res.kept[0]
                self.cached_cleaned_texts.append(text)

                if self.tokenizer:
                    tok_count = len(self.tokenizer.encode(text, add_bos=True, add_eos=True))
                else:
                    tok_count = len(text.split())

                cat = self.auditor.record_document(text, tok_count)
                doc_id = f"{dataset_id}_{doc_offset}"
                self.registry.insert_document(
                    doc_id=doc_id,
                    dataset_id=dataset_id,
                    offset=doc_offset,
                    char_count=len(text),
                    token_count=tok_count,
                    category=cat,
                    quality_score=qscore.score,
                    is_duplicate=False,
                )
                doc_offset += 1

                self.checkpoint.update(
                    files_processed=file_counter,
                    tokens_processed=self.auditor.stats.total_tokens,
                )

        elapsed = time.time() - t0
        self.auditor.stats.duplicate_rate = duplicate_count / max(total_raw, 1)
        self.auditor.stats.quality_pass_rate = accepted_count / max(total_raw, 1)
        self.auditor.update_throughput(
            docs_per_sec=self.auditor.stats.total_documents / max(elapsed, 1e-6),
            tokens_per_sec=self.auditor.stats.total_tokens / max(elapsed, 1e-6),
        )
        self.auditor.update_memory()

        self.registry.record_statistics(self.auditor.to_registry_dict())
        self.registry.export_json(self.json_export_path)
        print(
            f"[CorpusManager] Indexing complete in {elapsed:.2f}s! "
            f"({self.auditor.stats.total_documents:,} docs, {self.auditor.stats.total_tokens:,} tokens)"
        )

    def scan(self, force_reindex: bool = False) -> None:
        self.index(force_reindex=force_reindex)

    def build(self) -> list[Path]:
        if not self.tokenizer:
            raise ValueError("BPETokenizer required for build() sharding")

        print(f"[CorpusManager] Building VST1 binary shards in {self.shard_dir}...", flush=True)
        if not self.cached_cleaned_texts:
            print("[CorpusManager] No cached text found. Running index() first...")
            self.index()

        texts = list(self.cached_cleaned_texts)
        random.seed(42)
        random.shuffle(texts)

        split_idx = int(len(texts) * 0.90)
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]

        writer = ShardWriter(
            output_dir=self.shard_dir,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            stride=self.stride,
        )

        train_shards = writer.write_shards(train_texts, split="train")
        val_shards = writer.write_shards(val_texts, split="val")

        self.auditor.stats.shards_train = len(train_shards)
        self.auditor.stats.shards_val = len(val_shards)

        for sp in train_shards + val_shards:
            hdr = ShardWriter.read_shard_header(sp)
            self.registry.insert_shard(
                filename=sp.name,
                split="train" if "train" in sp.name else "val",
                chunk_count=hdr.num_chunks,
                token_count=hdr.num_chunks * (hdr.seq_len + 1),
                avg_seq_len=float(hdr.seq_len),
                reasoning_pct=self.auditor.stats.reasoning_tokens / max(1, self.auditor.stats.total_tokens),
                code_pct=self.auditor.stats.code_tokens / max(1, self.auditor.stats.total_tokens),
                math_pct=self.auditor.stats.math_tokens / max(1, self.auditor.stats.total_tokens),
                duplicate_rate=self.auditor.stats.duplicate_rate,
                quality_rate=self.auditor.stats.quality_pass_rate,
                tokenizer_fp=writer.fingerprint,
            )

        self.registry.export_json(self.json_export_path)
        print(
            f"[CorpusManager] Build complete! Written {len(train_shards)} train shards "
            f"and {len(val_shards)} val shards."
        )
        return train_shards + val_shards

    def verify(self) -> bool:
        print("[CorpusManager] Verifying SQLite registry & shard integrity...", flush=True)
        manifest_file = self.shard_dir / "manifest.json"
        if not manifest_file.exists():
            print("⚠️ Shard manifest.json not found.")
            return False

        if self.tokenizer:
            fp = get_tokenizer_fingerprint(self.tokenizer)
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if manifest.get("tokenizer_fingerprint") != fp:
                print("❌ Tokenizer fingerprint mismatch in manifest!")
                return False

        train_shards = list(self.shard_dir.glob("train_*.bin"))
        val_shards = list(self.shard_dir.glob("val_*.bin"))
        for shard in train_shards + val_shards:
            try:
                hdr = ShardWriter.read_shard_header(shard)
                if hdr.seq_len != self.seq_len:
                    print(f"❌ Sequence length mismatch in {shard.name}")
                    return False
            except Exception as exc:
                print(f"❌ Shard corruption detected in {shard.name}: {exc}")
                return False

        print("✅ Integrity verification passed!")
        return True

    def status(self) -> None:
        local_cnt = sum(1 for source in self.sources if isinstance(source, LocalSource))
        hf_cnt = sum(1 for source in self.sources if isinstance(source, HFSource))
        self.dashboard.display(self.auditor.stats, local_sources=local_cnt, hf_sources=hf_cnt)

    def build_loaders(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> tuple[DataLoader, DataLoader]:
        train_shards = list(self.shard_dir.glob("train_*.bin"))
        val_shards = list(self.shard_dir.glob("val_*.bin"))

        if train_shards and val_shards:
            print(
                f"[CorpusManager] Loading DataLoaders directly from {len(train_shards)} train "
                f"and {len(val_shards)} val binary shards..."
            )
            train_ds = ShardDataset(train_shards)
            val_ds = ShardDataset(val_shards)
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                pin_memory=pin_memory,
                num_workers=num_workers,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=pin_memory,
                num_workers=num_workers,
            )
            return train_loader, val_loader

        print("[CorpusManager] Binary shards not found. Loading from memory chunks...")
        if not self.cached_cleaned_texts:
            self.index()
        if self.tokenizer is None:
            raise ValueError("BPETokenizer required for in-memory loader construction")

        texts = list(self.cached_cleaned_texts)
        random.seed(42)
        random.shuffle(texts)

        split_idx = int(len(texts) * 0.90)
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]

        def _chunk_texts(doc_list: list[str]) -> list[list[int]]:
            chunks: list[list[int]] = []
            for text in doc_list:
                ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
                for start in range(0, len(ids) - 1, self.stride):
                    chunk = ids[start : start + self.seq_len + 1]
                    if len(chunk) < 2:
                        continue
                    if len(chunk) < self.seq_len + 1:
                        chunk = chunk + [self.tokenizer.pad_id()] * (self.seq_len + 1 - len(chunk))
                    chunks.append(chunk)
            return chunks

        train_chunks = _chunk_texts(train_texts)
        val_chunks = _chunk_texts(val_texts)

        train_ds = SequenceDataset(train_chunks)
        val_ds = SequenceDataset(val_chunks)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        return train_loader, val_loader

    def loaders(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> tuple[DataLoader, DataLoader]:
        return self.build_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
