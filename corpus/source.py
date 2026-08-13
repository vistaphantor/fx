from __future__ import annotations

import os
import re
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


@dataclass
class SourceMetadata:
    source_type: str
    path: str
    size_bytes: int = 0
    estimated_docs: int = 0
    description: str = ""
    extra: dict = field(default_factory=dict)


class DatasetSource(ABC):
    @abstractmethod
    def scan(self) -> SourceMetadata: ...

    @abstractmethod
    def stream(self) -> Iterator[str]: ...

    @abstractmethod
    def metadata(self) -> dict: ...

    @property
    @abstractmethod
    def source_id(self) -> str: ...


class LocalSource(DatasetSource):
    SKIP_NAMES = {"master_index.json", "__pycache__", ".DS_Store"}

    def __init__(self, path: str | Path):
        self._path = Path(path)

    @property
    def source_id(self) -> str:
        return f"local:{self._path}"

    def scan(self) -> SourceMetadata:
        size = self._path.stat().st_size if self._path.is_file() else sum(
            f.stat().st_size for f in self._path.rglob("*") if f.is_file()
        )
        return SourceMetadata(
            source_type="local",
            path=str(self._path),
            size_bytes=size,
        )

    def _files(self) -> list[Path]:
        if self._path.is_file():
            return [self._path]
        exts = {".json", ".jsonl", ".txt", ".gz"}
        return [
            f for f in sorted(self._path.rglob("*"))
            if f.is_file()
            and f.suffix.lower() in exts
            and f.name not in self.SKIP_NAMES
        ]

    def stream(self) -> Iterator[str]:
        from corpus.normalizer import TextNormalizer
        norm = TextNormalizer()
        for fp in self._files():
            yield from norm.parse_file(fp)

    def metadata(self) -> dict:
        return {"source_type": "local", "path": str(self._path)}


class HFSource(DatasetSource):
    COLUMN_SCHEMAS = [
        ("problem",      "solution"),
        ("question",     "solution"),
        ("prompt",       "chosen"),
        ("instruction",  "output"),
        ("instruction",  "response"),
        ("input",        "output"),
    ]

    def __init__(
        self,
        path: str,
        split: str = "train",
        text_fields: Optional[list[str]] = None,
        think_wrap: bool = True,
        max_examples: Optional[int] = None,
        config_name: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self._path = path
        self._split = split
        self._text_fields = text_fields
        self._think_wrap = think_wrap
        self._max_examples = max_examples
        self._config_name = config_name
        self._token = token or os.environ.get("HF_TOKEN")

    @property
    def source_id(self) -> str:
        return f"hf:{self._path}/{self._split}"

    def scan(self) -> SourceMetadata:
        return SourceMetadata(
            source_type="huggingface",
            path=self.source_id,
            description=f"{self._path} [{self._split}]",
            extra={"max_examples": self._max_examples},
        )

    def _detect_schema(self, row: dict) -> Optional[tuple[str, str]]:
        for a, b in self.COLUMN_SCHEMAS:
            if a in row and b in row:
                return a, b
        return None

    def _row_to_text(self, row: dict) -> Optional[str]:
        if self._text_fields:
            parts = [str(row[f]).strip() for f in self._text_fields if f in row]
            return "\n\n".join(parts) if parts else None

        if "conversations" in row:
            turns = row["conversations"]
            if isinstance(turns, list):
                parts = []
                for t in turns:
                    role = t.get("from", t.get("role", ""))
                    val  = t.get("value", t.get("content", "")).strip()
                    if role in ("human", "user"):
                        parts.append(f"Human: {val}")
                    elif role in ("gpt", "assistant"):
                        parts.append(f"Assistant: {val}")
                return "\n\n".join(parts) if parts else None

        if "text" in row:
            return str(row["text"]).strip()

        schema = self._detect_schema(row)
        if schema:
            prompt_col, resp_col = schema
            p = str(row[prompt_col]).strip()
            r = str(row[resp_col]).strip()
            if self._think_wrap:
                return f"Human: {p}\n\nAssistant: <think>\n{r}\n</think>"
            return f"Human: {p}\n\nAssistant: {r}"

        return None

    def stream(self) -> Iterator[str]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Install the 'datasets' library: pip install datasets"
            )

        kwargs: dict = {"split": self._split, "streaming": True}
        if self._config_name:
            kwargs["name"] = self._config_name
        if self._token:
            kwargs["token"] = self._token

        ds = load_dataset(self._path, **kwargs)
        count = 0
        for row in ds:
            if self._max_examples and count >= self._max_examples:
                break
            text = self._row_to_text(row)
            if text and len(text.strip()) > 25:
                yield text.strip()
                count += 1

    def metadata(self) -> dict:
        return {
            "source_type": "huggingface",
            "path": self._path,
            "split": self._split,
            "max_examples": self._max_examples,
        }
