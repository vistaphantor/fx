from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from src.language.canonical_contract import (
    CanonicalMessage,
    serialize_document,
    serialize_messages,
)


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    source_type: str
    path: str
    size_bytes: int = 0
    estimated_docs: int = 0
    description: str = ""
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HFSourceAudit:
    source_id: str
    rows_scanned: int
    rows_serialized: int
    rows_unrecognized: int
    mean_serialized_chars: float
    min_serialized_chars: int
    max_serialized_chars: int

    @property
    def serialization_rate(self) -> float:
        return self.rows_serialized / max(self.rows_scanned, 1)


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
    """Local file adapter retained for non-language corpus tooling.

    The language trainer itself is stream-only and must not construct this
    adapter. Keeping the adapter here avoids coupling unrelated corpus tooling
    to the language training policy.
    """

    SKIP_NAMES = {"master_index.json", "__pycache__", ".DS_Store"}
    SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".txt"}

    def __init__(self, path: str | Path):
        self._path = Path(path)

    @property
    def source_id(self) -> str:
        return f"local:{self._path}"

    def scan(self) -> SourceMetadata:
        size = self._path.stat().st_size if self._path.is_file() else sum(
            f.stat().st_size for f in self._path.rglob("*") if f.is_file()
        )
        return SourceMetadata(source_type="local", path=str(self._path), size_bytes=size)

    def _files(self) -> list[Path]:
        if self._path.is_file():
            if self._path.suffix.casefold() not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(f"unsupported_local_training_file:{self._path}")
            return [self._path]
        return [
            file_path
            for file_path in sorted(self._path.rglob("*"))
            if file_path.is_file()
            and file_path.suffix.casefold() in self.SUPPORTED_EXTENSIONS
            and file_path.name not in self.SKIP_NAMES
        ]

    def stream(self) -> Iterator[str]:
        from src.language.data_pipeline import (
            _load_json_file,
            _load_jsonl_file,
            _load_txt_file,
        )

        loaders = {
            ".json": _load_json_file,
            ".jsonl": _load_jsonl_file,
            ".txt": _load_txt_file,
        }
        for file_path in self._files():
            yield from loaders[file_path.suffix.casefold()](file_path)

    def metadata(self) -> dict:
        return {"source_type": "local", "path": str(self._path)}


class HFSource(DatasetSource):
    """Canonical, bounded-memory, transport-resilient HF streaming adapter."""

    COLUMN_SCHEMAS = (
        ("problem", "solution"),
        ("question", "solution"),
        ("question", "answer"),
        ("instruction", "output"),
        ("instruction", "response"),
        ("input", "output"),
        ("prompt", "response"),
        ("prompt", "answer"),
    )

    _TRANSIENT_MARKERS = (
        "timed out",
        "timeout",
        "peer closed connection",
        "incomplete message body",
        "remoteprotocolerror",
        "connection reset",
        "connection aborted",
        "connection refused",
        "server disconnected",
        "got disconnected from remote data host",
        "winerror 10038",
        "winerror 10053",
        "winerror 10054",
        "winerror 10060",
        "temporarily unavailable",
        "http 429",
        "status code: 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
    )

    def __init__(
        self,
        path: str,
        split: str = "train",
        text_fields: Optional[list[str]] = None,
        prompt_field: Optional[str] = None,
        response_field: Optional[str] = None,
        max_examples: Optional[int] = None,
        config_name: Optional[str] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        shuffle_buffer_size: int = 10_000,
        seed: int = 42,
        stream_retry_attempts: int = 20,
        stream_retry_base_seconds: float = 1.0,
    ):
        if not path.strip():
            raise ValueError("HFSource path must not be empty")
        if shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be >= 0")
        if bool(prompt_field) != bool(response_field):
            raise ValueError("prompt_field and response_field must be configured together")
        if text_fields and prompt_field:
            raise ValueError("text_fields cannot be combined with prompt/response fields")
        if stream_retry_attempts < 0:
            raise ValueError("stream_retry_attempts must be >= 0")
        if stream_retry_base_seconds < 0:
            raise ValueError("stream_retry_base_seconds must be >= 0")
        self._path = path.strip()
        self._split = split
        self._text_fields = list(text_fields) if text_fields else None
        self._prompt_field = str(prompt_field) if prompt_field else None
        self._response_field = str(response_field) if response_field else None
        self._max_examples = max_examples
        self._config_name = config_name
        self._revision = revision
        self._token = token or os.environ.get("HF_TOKEN")
        self._shuffle_buffer_size = int(shuffle_buffer_size)
        self._seed = int(seed)
        self._stream_retry_attempts = int(stream_retry_attempts)
        self._stream_retry_base_seconds = float(stream_retry_base_seconds)

    @property
    def source_id(self) -> str:
        config = f"/{self._config_name}" if self._config_name else ""
        revision = f"@{self._revision}" if self._revision else "@UNPINNED"
        return f"hf:{self._path}{config}{revision}/{self._split}"

    def scan(self) -> SourceMetadata:
        return SourceMetadata(
            source_type="huggingface",
            path=self.source_id,
            description=f"{self._path} [{self._split}]",
            extra={
                "revision": self._revision,
                "max_examples": self._max_examples,
                "shuffle_buffer_size": self._shuffle_buffer_size,
                "seed": self._seed,
                "text_fields": self._text_fields,
                "prompt_field": self._prompt_field,
                "response_field": self._response_field,
                "stream_retry_attempts": self._stream_retry_attempts,
            },
        )

    def _detect_schema(self, row: dict) -> Optional[tuple[str, str]]:
        if self._prompt_field and self._response_field:
            if self._prompt_field in row and self._response_field in row:
                return self._prompt_field, self._response_field
            return None
        for prompt_field, response_field in self.COLUMN_SCHEMAS:
            if prompt_field in row and response_field in row:
                return prompt_field, response_field
        return None

    def _messages_from_turns(self, turns: object) -> list[CanonicalMessage]:
        if not isinstance(turns, list):
            return []
        messages: list[CanonicalMessage] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))
            if role and content is not None:
                messages.append(CanonicalMessage(str(role), str(content)))
        return messages

    def _row_to_text(self, row: dict) -> Optional[str]:
        if self._prompt_field and self._response_field:
            schema = self._detect_schema(row)
            if schema is None:
                return None
            prompt = row.get(self._prompt_field)
            response = row.get(self._response_field)
            if prompt is None or response is None:
                return None
            return serialize_messages(
                [CanonicalMessage("user", str(prompt)), CanonicalMessage("assistant", str(response))]
            ) or None

        if self._text_fields:
            parts = [str(row.get(field, "")) for field in self._text_fields]
            document = "\n\n".join(part for part in parts if part and part.strip())
            return serialize_document(document) or None

        for key in ("messages", "conversations"):
            messages = self._messages_from_turns(row.get(key))
            if messages:
                serialized = serialize_messages(messages)
                return serialized or None

        if "text" in row:
            serialized = serialize_document(row.get("text"))
            return serialized or None

        schema = self._detect_schema(row)
        if schema:
            prompt_field, response_field = schema
            prompt = row.get(prompt_field)
            response = row.get(response_field)
            if prompt is not None and response is not None:
                return serialize_messages(
                    [CanonicalMessage("user", str(prompt)), CanonicalMessage("assistant", str(response))]
                ) or None
        return None

    def _load_dataset(self, *, shuffle: bool, retry_generation: int = 0):
        if not self._revision:
            raise RuntimeError(
                f"hf_revision_must_be_pinned:{self._path}:set an immutable commit/tag revision"
            )
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face streaming requires the 'datasets' package declared in requirements.txt"
            ) from exc

        kwargs: dict = {
            "split": self._split,
            "streaming": True,
            "revision": self._revision,
        }
        if self._config_name:
            kwargs["name"] = self._config_name
        if self._token:
            kwargs["token"] = self._token
        dataset = load_dataset(self._path, **kwargs)
        if shuffle and self._shuffle_buffer_size > 0:
            retry_seed = self._seed + int(retry_generation) * 1_000_003
            dataset = dataset.shuffle(seed=retry_seed, buffer_size=self._shuffle_buffer_size)
        return dataset

    @classmethod
    def _is_transient_stream_error(cls, exc: BaseException) -> bool:
        if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
            return True
        message = f"{type(exc).__name__}: {exc}".casefold()
        return any(marker in message for marker in cls._TRANSIENT_MARKERS)

    def _retry_delay(self, retry_number: int) -> float:
        return min(30.0, self._stream_retry_base_seconds * (2 ** max(0, retry_number - 1)))

    def audit(self, *, max_rows: int = 1_000) -> HFSourceAudit:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        scanned = 0
        serialized = 0
        lengths: list[int] = []
        for row in self._load_dataset(shuffle=False):
            if scanned >= max_rows:
                break
            scanned += 1
            if not isinstance(row, dict):
                continue
            text = self._row_to_text(row)
            if not text:
                continue
            serialized += 1
            lengths.append(len(text))
        return HFSourceAudit(
            source_id=self.source_id,
            rows_scanned=scanned,
            rows_serialized=serialized,
            rows_unrecognized=scanned - serialized,
            mean_serialized_chars=(sum(lengths) / len(lengths) if lengths else 0.0),
            min_serialized_chars=(min(lengths) if lengths else 0),
            max_serialized_chars=(max(lengths) if lengths else 0),
        )

    def stream(self) -> Iterator[str]:
        emitted = 0
        retry_generation = 0
        while self._max_examples is None or emitted < self._max_examples:
            try:
                dataset = self._load_dataset(
                    shuffle=True,
                    retry_generation=retry_generation,
                )
                for row in dataset:
                    if self._max_examples is not None and emitted >= self._max_examples:
                        return
                    if not isinstance(row, dict):
                        continue
                    text = self._row_to_text(row)
                    if text and len(text) > 25:
                        yield text
                        emitted += 1
                return
            except Exception as exc:
                if (
                    not self._is_transient_stream_error(exc)
                    or retry_generation >= self._stream_retry_attempts
                ):
                    raise
                retry_generation += 1
                delay = self._retry_delay(retry_generation)
                print(
                    f"[HFRetry] source={self.source_id} retry={retry_generation}/"
                    f"{self._stream_retry_attempts} delay={delay:.1f}s "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                if delay > 0:
                    time.sleep(delay)

    def metadata(self) -> dict:
        return {
            "source_type": "huggingface",
            "path": self._path,
            "split": self._split,
            "config_name": self._config_name,
            "revision": self._revision,
            "max_examples": self._max_examples,
            "shuffle_buffer_size": self._shuffle_buffer_size,
            "seed": self._seed,
            "text_fields": self._text_fields,
            "prompt_field": self._prompt_field,
            "response_field": self._response_field,
            "stream_retry_attempts": self._stream_retry_attempts,
        }