from __future__ import annotations

import gc
import os
import random
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
    @property
    @abstractmethod
    def source_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def scan(self) -> SourceMetadata:
        raise NotImplementedError

    @abstractmethod
    def stream(self) -> Iterator[str]:
        raise NotImplementedError


class LocalSource(DatasetSource):
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @property
    def source_id(self) -> str:
        return f"local:{self.path.resolve()}"

    def scan(self) -> SourceMetadata:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if self.path.is_file():
            return SourceMetadata(
                source_type="local",
                path=str(self.path),
                size_bytes=self.path.stat().st_size,
                estimated_docs=1,
                description=self.path.name,
            )
        files = [path for path in self.path.rglob("*") if path.is_file()]
        return SourceMetadata(
            source_type="local",
            path=str(self.path),
            size_bytes=sum(path.stat().st_size for path in files),
            estimated_docs=len(files),
            description=self.path.name,
        )

    def stream(self) -> Iterator[str]:
        """Stream through the same canonical parsers used by the trainer.

        The historical ``corpus.loader`` path duplicated parsing semantics and no
        longer exists. LocalSource now delegates directly to the authoritative
        language data pipeline instead of maintaining a compatibility facade.
        """
        from src.language.data_pipeline import (
            _load_json_file,
            _load_jsonl_file,
            _load_txt_file,
            load_all_training_text,
        )

        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if self.path.is_dir():
            yield from load_all_training_text(self.path, shuffle=False)
            return
        suffix = self.path.suffix.casefold()
        if suffix == ".json":
            yield from _load_json_file(self.path)
            return
        if suffix == ".jsonl":
            yield from _load_jsonl_file(self.path)
            return
        if suffix == ".txt":
            yield from _load_txt_file(self.path)
            return
        raise ValueError(f"unsupported_local_source_format:{self.path.suffix}")


class HFSource(DatasetSource):
    """Pinned Hugging Face streaming source with local filtering and shuffling."""

    COLUMN_SCHEMAS = (
        ("prompt", "response"),
        ("instruction", "output"),
        ("question", "answer"),
        ("input", "output"),
    )
    _TRANSIENT_MARKERS = (
        "timed out",
        "timeout",
        "peer closed connection",
        "connection reset",
        "connection aborted",
        "got disconnected",
        "winerror 10038",
        "ssl",
        "temporary failure",
    )

    def __init__(
        self,
        path: str,
        *,
        split: str = "train",
        text_fields: list[str] | tuple[str, ...] | None = None,
        prompt_field: str | None = None,
        response_field: str | None = None,
        dialogue_field: str | None = None,
        row_filters: dict[str, tuple[str, ...] | list[str] | str] | None = None,
        max_examples: int | None = None,
        config_name: str | None = None,
        revision: str | None = None,
        token: str | None = None,
        shuffle_buffer_size: int = 0,
        seed: int = 42,
        stream_retry_attempts: int = 20,
        stream_retry_base_seconds: float = 1.0,
    ):
        if not path or not path.strip():
            raise ValueError("path must be non-empty")
        if max_examples is not None and max_examples <= 0:
            raise ValueError("max_examples must be positive when supplied")
        if shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be >= 0")
        if stream_retry_attempts < 0:
            raise ValueError("stream_retry_attempts must be >= 0")
        if stream_retry_base_seconds < 0:
            raise ValueError("stream_retry_base_seconds must be >= 0")
        if bool(prompt_field) != bool(response_field):
            raise ValueError("prompt_field and response_field must be supplied together")
        if text_fields and (prompt_field or response_field):
            raise ValueError("text_fields cannot be combined with prompt/response mappings")
        if dialogue_field and (text_fields or prompt_field or response_field):
            raise ValueError("dialogue_field cannot be combined with text or prompt/response mappings")

        normalized_filters: dict[str, tuple[str, ...]] = {}
        for key, raw_values in (row_filters or {}).items():
            field_name = str(key).strip()
            if not field_name:
                raise ValueError("row_filter_field_must_be_nonempty")
            values = (raw_values,) if isinstance(raw_values, str) else tuple(raw_values)
            cleaned = tuple(str(value).strip() for value in values if str(value).strip())
            if not cleaned:
                raise ValueError(f"row_filter_values_must_be_nonempty:{field_name}")
            normalized_filters[field_name] = cleaned

        self._path = path.strip()
        self._split = split
        self._text_fields = list(text_fields) if text_fields else None
        self._prompt_field = str(prompt_field) if prompt_field else None
        self._response_field = str(response_field) if response_field else None
        self._dialogue_field = str(dialogue_field) if dialogue_field else None
        self._row_filters = normalized_filters
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
                "shuffle_location": "local_reservoir",
                "remote_shard_order": "sequential",
                "seed": self._seed,
                "text_fields": self._text_fields,
                "prompt_field": self._prompt_field,
                "response_field": self._response_field,
                "dialogue_field": self._dialogue_field,
                "row_filters": self._row_filters,
                "stream_retry_attempts": self._stream_retry_attempts,
            },
        )

    def _row_allowed(self, row: dict) -> bool:
        for field_name, allowed in self._row_filters.items():
            value = row.get(field_name)
            if value is None or str(value).strip() not in allowed:
                return False
        return True

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

    @staticmethod
    def _messages_from_dialogue(dialogue: object) -> list[CanonicalMessage]:
        if not isinstance(dialogue, (list, tuple)):
            return []
        messages: list[CanonicalMessage] = []
        for index, content in enumerate(dialogue):
            if content is None or not str(content).strip():
                continue
            role = "user" if index % 2 == 0 else "assistant"
            messages.append(CanonicalMessage(role, str(content)))
        if len(messages) < 2:
            return []
        if messages[-1].role == "user":
            messages.pop()
        return messages

    def _row_to_text(self, row: dict) -> Optional[str]:
        if not self._row_allowed(row):
            return None

        if self._dialogue_field:
            messages = self._messages_from_dialogue(row.get(self._dialogue_field))
            if not messages:
                return None
            return serialize_messages(messages) or None

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

    def _load_dataset(self, *, shuffle: bool = False, retry_generation: int = 0):
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
        return load_dataset(self._path, **kwargs)

    @classmethod
    def _is_transient_stream_error(cls, exc: BaseException) -> bool:
        if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
            return True
        message = f"{type(exc).__name__}: {exc}".casefold()
        return any(marker in message for marker in cls._TRANSIENT_MARKERS)

    def _retry_delay(self, retry_number: int) -> float:
        return min(30.0, self._stream_retry_base_seconds * (2 ** max(0, retry_number - 1)))

    @staticmethod
    def _close_stream(iterator: object | None, dataset: object | None) -> None:
        if iterator is not None:
            close = getattr(iterator, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        if dataset is not None:
            close = getattr(dataset, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def audit(self, *, max_rows: int = 1_000) -> HFSourceAudit:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        scanned = 0
        serialized = 0
        lengths: list[int] = []
        dataset = self._load_dataset(shuffle=False)
        iterator = iter(dataset)
        try:
            for row in iterator:
                if scanned >= max_rows:
                    break
                if not isinstance(row, dict) or not self._row_allowed(row):
                    continue
                scanned += 1
                text = self._row_to_text(row)
                if not text:
                    continue
                serialized += 1
                lengths.append(len(text))
        finally:
            self._close_stream(iterator, dataset)
            del iterator
            del dataset
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
        raw_rows_consumed = 0
        retry_generation = 0
        rng = random.Random(self._seed)
        reservoir: list[str] = []

        while self._max_examples is None or emitted < self._max_examples:
            dataset = None
            iterator = None
            try:
                dataset = self._load_dataset(shuffle=False)
                iterator = iter(dataset)
                skipped = 0
                while skipped < raw_rows_consumed:
                    next(iterator)
                    skipped += 1

                for row in iterator:
                    raw_rows_consumed += 1
                    if self._max_examples is not None and emitted >= self._max_examples:
                        return
                    if not isinstance(row, dict):
                        continue
                    text = self._row_to_text(row)
                    if not text or len(text) <= 25:
                        continue

                    if self._shuffle_buffer_size <= 1:
                        yield text
                        emitted += 1
                        continue
                    if len(reservoir) < self._shuffle_buffer_size:
                        reservoir.append(text)
                        continue
                    index = rng.randrange(len(reservoir))
                    selected = reservoir[index]
                    reservoir[index] = text
                    yield selected
                    emitted += 1

                rng.shuffle(reservoir)
                while reservoir and (
                    self._max_examples is None or emitted < self._max_examples
                ):
                    yield reservoir.pop()
                    emitted += 1
                return
            except StopIteration:
                rng.shuffle(reservoir)
                while reservoir and (
                    self._max_examples is None or emitted < self._max_examples
                ):
                    yield reservoir.pop()
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
                    f"raw_rows={raw_rows_consumed:,} buffered={len(reservoir):,} "
                    f"transport=sequential_remote+local_reservoir "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                if delay > 0:
                    time.sleep(delay)
            finally:
                self._close_stream(iterator, dataset)
                iterator = None
                dataset = None
                gc.collect()

    def metadata(self) -> dict:
        return {
            "source_type": "huggingface",
            "path": self._path,
            "split": self._split,
            "config_name": self._config_name,
            "revision": self._revision,
            "max_examples": self._max_examples,
            "shuffle_buffer_size": self._shuffle_buffer_size,
            "shuffle_location": "local_reservoir",
            "remote_shard_order": "sequential",
            "seed": self._seed,
            "text_fields": self._text_fields,
            "prompt_field": self._prompt_field,
            "response_field": self._response_field,
            "dialogue_field": self._dialogue_field,
            "row_filters": self._row_filters,
            "stream_retry_attempts": self._stream_retry_attempts,
        }
