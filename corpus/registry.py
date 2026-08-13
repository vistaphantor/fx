from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


REGISTRY_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS sources (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,
    path        TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT '',
    added_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id       INTEGER NOT NULL REFERENCES sources(id),
    name            TEXT NOT NULL,
    sha256          TEXT DEFAULT '',
    size_bytes      INTEGER DEFAULT 0,
    last_indexed    INTEGER DEFAULT 0,
    sharded         INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    dataset_id      INTEGER NOT NULL REFERENCES datasets(id),
    doc_offset      INTEGER NOT NULL,
    char_count      INTEGER DEFAULT 0,
    token_count     INTEGER DEFAULT 0,
    category        TEXT DEFAULT 'general',
    quality_score   REAL DEFAULT 1.0,
    is_duplicate    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS shards (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT NOT NULL UNIQUE,
    split           TEXT NOT NULL,
    chunk_count     INTEGER DEFAULT 0,
    token_count     INTEGER DEFAULT 0,
    avg_seq_len     REAL DEFAULT 0.0,
    reasoning_pct   REAL DEFAULT 0.0,
    code_pct        REAL DEFAULT 0.0,
    math_pct        REAL DEFAULT 0.0,
    duplicate_rate  REAL DEFAULT 0.0,
    quality_rate    REAL DEFAULT 1.0,
    tokenizer_fp    TEXT DEFAULT '',
    created_at      INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS statistics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at     INTEGER NOT NULL,
    total_docs      INTEGER DEFAULT 0,
    total_tokens    INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    code_tokens     INTEGER DEFAULT 0,
    math_tokens     INTEGER DEFAULT 0,
    instruction_tokens INTEGER DEFAULT 0,
    general_tokens  INTEGER DEFAULT 0,
    duplicate_rate  REAL DEFAULT 0.0,
    quality_rate    REAL DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_documents_dataset ON documents(dataset_id);
CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category);
CREATE INDEX IF NOT EXISTS idx_shards_split ON shards(split);
"""


@dataclass
class DatasetRecord:
    source_type: str
    path: str
    name: str
    sha256: str = ""
    size_bytes: int = 0
    last_indexed: int = 0
    sharded: bool = False
    dataset_id: Optional[int] = None
    source_id: Optional[int] = None


class CorpusRegistry:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self) -> None:
        self._con.executescript(DDL)
        row = self._con.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            self._con.execute("INSERT INTO schema_version VALUES (?)", (REGISTRY_VERSION,))
        self._con.commit()

    def upsert_source(self, source_type: str, path: str, description: str = "") -> int:
        cur = self._con.execute("SELECT id FROM sources WHERE path = ?", (path,))
        row = cur.fetchone()
        if row:
            return row["id"]
        cur = self._con.execute(
            "INSERT INTO sources (source_type, path, description, added_at) VALUES (?, ?, ?, ?)",
            (source_type, path, description, int(time.time()))
        )
        self._con.commit()
        return cur.lastrowid

    def upsert_dataset(self, record: DatasetRecord) -> int:
        source_id = self.upsert_source(record.source_type, record.path)
        cur = self._con.execute("SELECT id, sha256 FROM datasets WHERE name = ?", (record.name,))
        row = cur.fetchone()
        if row:
            if row["sha256"] != record.sha256:
                self._con.execute(
                    "UPDATE datasets SET sha256=?, size_bytes=?, last_indexed=?, sharded=? WHERE id=?",
                    (record.sha256, record.size_bytes, int(time.time()), int(record.sharded), row["id"])
                )
                self._con.commit()
            return row["id"]
        cur = self._con.execute(
            "INSERT INTO datasets (source_id, name, sha256, size_bytes, last_indexed, sharded) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (source_id, record.name, record.sha256, record.size_bytes,
             int(time.time()), int(record.sharded))
        )
        self._con.commit()
        return cur.lastrowid

    def is_sharded(self, dataset_name: str) -> bool:
        row = self._con.execute(
            "SELECT sharded FROM datasets WHERE name = ?", (dataset_name,)
        ).fetchone()
        return bool(row and row["sharded"])

    def mark_sharded(self, dataset_name: str) -> None:
        self._con.execute(
            "UPDATE datasets SET sharded = 1 WHERE name = ?", (dataset_name,)
        )
        self._con.commit()

    def insert_document(
        self,
        doc_id: str,
        dataset_id: int,
        offset: int,
        char_count: int,
        token_count: int,
        category: str,
        quality_score: float,
        is_duplicate: bool,
    ) -> None:
        self._con.execute(
            "INSERT OR IGNORE INTO documents "
            "(id, dataset_id, doc_offset, char_count, token_count, category, quality_score, is_duplicate) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (doc_id, dataset_id, offset, char_count, token_count, category,
             quality_score, int(is_duplicate))
        )

    def flush_documents(self) -> None:
        self._con.commit()

    def insert_shard(
        self,
        filename: str,
        split: str,
        chunk_count: int,
        token_count: int,
        avg_seq_len: float,
        reasoning_pct: float,
        code_pct: float,
        math_pct: float,
        duplicate_rate: float,
        quality_rate: float,
        tokenizer_fp: str,
    ) -> None:
        self._con.execute(
            "INSERT OR REPLACE INTO shards "
            "(filename, split, chunk_count, token_count, avg_seq_len, "
            "reasoning_pct, code_pct, math_pct, duplicate_rate, quality_rate, "
            "tokenizer_fp, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (filename, split, chunk_count, token_count, avg_seq_len,
             reasoning_pct, code_pct, math_pct, duplicate_rate, quality_rate,
             tokenizer_fp, int(time.time()))
        )
        self._con.commit()

    def record_statistics(self, stats: dict) -> None:
        self._con.execute(
            "INSERT INTO statistics "
            "(recorded_at, total_docs, total_tokens, reasoning_tokens, code_tokens, "
            "math_tokens, instruction_tokens, general_tokens, duplicate_rate, quality_rate) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(time.time()),
                stats.get("total_docs", 0),
                stats.get("total_tokens", 0),
                stats.get("reasoning_tokens", 0),
                stats.get("code_tokens", 0),
                stats.get("math_tokens", 0),
                stats.get("instruction_tokens", 0),
                stats.get("general_tokens", 0),
                stats.get("duplicate_rate", 0.0),
                stats.get("quality_rate", 1.0),
            )
        )
        self._con.commit()

    def latest_statistics(self) -> dict:
        row = self._con.execute(
            "SELECT * FROM statistics ORDER BY recorded_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else {}

    def shard_list(self, split: str) -> list[str]:
        rows = self._con.execute(
            "SELECT filename FROM shards WHERE split = ? ORDER BY filename", (split,)
        ).fetchall()
        return [r["filename"] for r in rows]

    def dataset_needs_reindex(self, name: str, current_sha256: str) -> bool:
        row = self._con.execute(
            "SELECT sha256 FROM datasets WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return True
        return row["sha256"] != current_sha256

    def export_json(self, out_path: str | Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        datasets = [dict(r) for r in self._con.execute(
            "SELECT d.name, s.source_type, s.path, d.size_bytes, d.sha256, "
            "d.last_indexed, d.sharded "
            "FROM datasets d JOIN sources s ON d.source_id = s.id"
        ).fetchall()]
        latest_stats = self.latest_statistics()
        shards = [dict(r) for r in self._con.execute("SELECT * FROM shards").fetchall()]
        summary = {
            "registry_version": REGISTRY_VERSION,
            "exported_at": int(time.time()),
            "datasets": datasets,
            "shards": shards,
            "latest_statistics": latest_stats,
        }
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    def close(self) -> None:
        self._con.close()


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()
