from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional


class IndexCheckpoint:
    def __init__(self, path: str | Path, file_interval: int = 100, token_interval: int = 5_000_000):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file_interval = file_interval
        self._token_interval = token_interval
        self._state: dict = {}
        self._files_since_save = 0
        self._tokens_since_save = 0

    def load(self) -> dict:
        if self._path.exists():
            try:
                self._state = json.loads(self._path.read_text(encoding="utf-8"))
                print(f"[Checkpoint] Resumed from: {self._path}")
            except Exception:
                self._state = {}
        return self._state

    def update(self, files_processed: int, tokens_processed: int, extra: dict | None = None) -> None:
        self._state["files_processed"] = files_processed
        self._state["tokens_processed"] = tokens_processed
        self._state["updated_at"] = int(time.time())
        if extra:
            self._state.update(extra)

        self._files_since_save += 1
        self._tokens_since_save += tokens_processed - self._state.get("_last_tok_snap", 0)
        self._state["_last_tok_snap"] = tokens_processed

        if (self._files_since_save >= self._file_interval
                or self._tokens_since_save >= self._token_interval):
            self.save()
            self._files_since_save = 0
            self._tokens_since_save = 0

    def save(self) -> None:
        self._path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")

    def get(self, key: str, default=None):
        return self._state.get(key, default)

    def reset(self) -> None:
        self._state = {}
        self._files_since_save = 0
        self._tokens_since_save = 0
        if self._path.exists():
            self._path.unlink()
