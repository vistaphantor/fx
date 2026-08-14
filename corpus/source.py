from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


@dataclass(frozen=True, slots=True)
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
        return SourceMetadata(source_type="local", path=str(self._path), size_bytes=size)

    def _files(self) -> list[Path]:
        if self._path.is_file():
            return [self._path]
        exts = {".json", ".jsonl", ".txt", ".gz"}
        return [
            f
            for f in sorted(self._path.rglob("*"))
            if f.is_file() and f.suffix.lower() in exts and f.name not in self.SKIP_NAMES
        ]

    def stream(self) -> Iterator[str]:
        from corpus.normalizer import TextNormalizer

        normalizer = TextNormalizer()
        for file_path in self._files():
            yield from normalizer.parse_file(file_path)

    def metadata(self) -> dict:
        return {"source_type": "local", "path": str(self._path)}


def _clean_content(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _canonical_document(text: str) -> str:
    value = _clean_content(text)
    return f"<bos>\n{value}\n<eos>" if value else ""


def _canonical_messages(messages: list[tuple[str, str]]) -> str:
    parts = ["<bos>"]
    appended = 0
    for role, raw_content in messages:
        content = _clean_content(raw_content)
        if not content:
            continue
        normalized_role = role.strip().casefold()
        if normalized_role in {"human", "user"}:
            parts.extend(["<user>", content, "</user>"])
            appended += 1
        elif normalized_role in {"gpt", "assistant", "ai"}:
            parts.extend(["<assistant>", content, "</assistant>"])
            appended += 1
        elif normalized_role == "system":
            parts.extend(["<evidence>", content, "</evidence>"])
            appended += 1
    if appended == 0:
        return ""
    parts.append("<eos>")
    return "\n".join(parts)


class HFSource(DatasetSource):
    """Authoritative Hugging Face streaming adapter.

    Rows are converted directly to the canonical Vista language grammar. The
    adapter never emits legacy ``Human:``/``Assistant:`` records and never
    materializes the remote dataset in RAM.
    """

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

    def __init__(
        self,
        path: str,
        split: str = "train",
        text_fields: Optional[list[str]] = None,
        max_examples: Optional[int] = None,
        config_name: Optional[str] = None,
        token: Optional[str] = None,
        shuffle_buffer_size: int = 10_000,
        seed: int = 42,
    ):
        if not path.strip():
            raise ValueError("HFSource path must not be empty")
        if shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be >= 0")
        self._path = path.strip()
        self._split = split
        self._text_fields = list(text_fields) if text_fields else None
        self._max_examples = max_examples
        self._config_name = config_name
        self._token = token or os.environ.get("HF_TOKEN")
        self._shuffle_buffer_size = int(shuffle_buffer_size)
        self._seed = int(seed)

    @property
    def source_id(self) -> str:
        config = f"/{self._config_name}" if self._config_name else ""
        return f"hf:{self._path}{config}/{self._split}"

    def scan(self) -> SourceMetadata:
        return SourceMetadata(
            source_type="huggingface",
            path=self.source_id,
            description=f"{self._path} [{self._split}]",
            extra={
                "max_examples": self._max_examples,
                "shuffle_buffer_size": self._shuffle_buffer_size,
                "seed": self._seed,
            },
        )

    def _detect_schema(self, row: dict) -> Optional[tuple[str, str]]:
        for prompt_field, response_field in self.COLUMN_SCHEMAS:
            if prompt_field in row and response_field in row:
                return prompt_field, response_field
        return None

    def _messages_from_turns(self, turns: object) -> list[tuple[str, str]]:
        if not isinstance(turns, list):
            return []
        messages: list[tuple[str, str]] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))
            if role and content is not None:
                messages.append((str(role), str(content)))
        return messages

    def _row_to_text(self, row: dict) -> Optional[str]:
        if self._text_fields:
            parts = [_clean_content(row.get(field)) for field in self._text_fields]
            document = "\n\n".join(part for part in parts if part)
            return _canonical_document(document) or None

        for key in ("messages", "conversations"):
            messages = self._messages_from_turns(row.get(key))
            if messages:
                serialized = _canonical_messages(messages)
                return serialized or None

        if "text" in row:
            serialized = _canonical_document(_clean_content(row.get("text")))
            return serialized or None

        schema = self._detect_schema(row)
        if schema:
            prompt_field, response_field = schema
            prompt = _clean_content(row.get(prompt_field))
            response = _clean_content(row.get(response_field))
            if prompt and response:
                return _canonical_messages([("user", prompt), ("assistant", response)])

        return None

    def stream(self) -> Iterator[str]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face streaming requires the 'datasets' package declared in requirements.txt"
            ) from exc

        kwargs: dict = {
            "split": self._split,
            "streaming": True,
        }
        if self._config_name:
            kwargs["name"] = self._config_name
        if self._token:
            kwargs["token"] = self._token

        dataset = load_dataset(self._path, **kwargs)
        if self._shuffle_buffer_size > 0:
            dataset = dataset.shuffle(
                seed=self._seed,
                buffer_size=self._shuffle_buffer_size,
            )

        emitted = 0
        for row in dataset:
            if self._max_examples is not None and emitted >= self._max_examples:
                break
            if not isinstance(row, dict):
                continue
            text = self._row_to_text(row)
            if text and len(text) > 25:
                yield text
                emitted += 1

    def metadata(self) -> dict:
        return {
            "source_type": "huggingface",
            "path": self._path,
            "split": self._split,
            "config_name": self._config_name,
            "max_examples": self._max_examples,
            "shuffle_buffer_size": self._shuffle_buffer_size,
            "seed": self._seed,
        }
