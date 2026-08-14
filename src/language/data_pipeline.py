"""Authoritative heterogeneous-data parser for Vista language training."""
from __future__ import annotations

import ast
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from src.language.canonical_contract import (
    CanonicalMessage,
    canonicalize_serialized,
    normalize_text,
    serialize_document,
    serialize_messages,
)

DATA_ROOT = Path("data/data/trainingdata")


@dataclass(frozen=True, slots=True)
class TrainingMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class TrainingExample:
    messages: tuple[TrainingMessage, ...]
    source: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


def _clean_text(value: object) -> str:
    return normalize_text(value, strip_role_prefix=True)


def _clean_instruction(value: object) -> str:
    text = _clean_text(value)
    text = re.sub(
        r"^Below is an instruction that describes a task\.\s*"
        r"Write a response that appropriately completes the request\.\s*"
        r"### Instruction:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*### Response:\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def _append_message(messages: list[TrainingMessage], role: str, content: object) -> None:
    role_map = {
        "human": "user",
        "user": "user",
        "assistant": "assistant",
        "ai": "assistant",
        "gpt": "assistant",
        "model": "assistant",
        "system": "system",
    }
    normalized_role = role_map.get(str(role or "").strip().casefold(), "")
    if normalized_role not in {"user", "assistant", "system"}:
        return
    cleaned = _clean_instruction(content) if normalized_role == "user" else _clean_text(content)
    if not cleaned:
        return
    message = TrainingMessage(normalized_role, cleaned)
    if messages and messages[-1] == message:
        return
    messages.append(message)


def _deduplicate_messages(messages: list[TrainingMessage]) -> tuple[TrainingMessage, ...]:
    result: list[TrainingMessage] = []
    for message in messages:
        if message.content and (not result or result[-1] != message):
            result.append(message)
    return tuple(result)


def _split_legacy_prompt(value: object) -> tuple[str, str]:
    text = str(value or "").replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"^\s*(?:Human|User)\s*:\s*", "", text, count=1, flags=re.IGNORECASE)
    match = re.search(r"(?:^|\n)\s*Assistant\s*:\s*", text, flags=re.IGNORECASE)
    if not match:
        return _clean_instruction(text), ""
    return _clean_instruction(text[: match.start()]), _clean_text(text[match.end() :])


def _parse_hh_conversation(value: str) -> list[TrainingMessage]:
    value = str(value or "").replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n").strip()
    markers = list(re.finditer(r"(?:^|\n\n)(Human|Assistant):\s*", value, flags=re.IGNORECASE))
    messages: list[TrainingMessage] = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(value)
        _append_message(messages, marker.group(1), value[marker.end() : end])
    return messages


def _try_parse_embedded_object(value: str):
    raw = str(value or "").strip()
    if not raw:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(raw)
        except Exception:
            continue
    return None


def _messages_from_list(values) -> list[TrainingMessage]:
    messages: list[TrainingMessage] = []
    if not isinstance(values, list):
        return messages
    for item in values:
        if not isinstance(item, dict):
            continue
        role = item.get("from", item.get("role", ""))
        content = item.get("value", item.get("content", ""))
        _append_message(messages, role, content)
    return messages


def _parse_serialized_chat_turns(value: str) -> list[TrainingMessage]:
    messages: list[TrainingMessage] = []
    for raw_line in str(value or "").strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        obj = _try_parse_embedded_object(line)
        if not isinstance(obj, dict) or "role" not in obj or "content" not in obj:
            return []
        _append_message(messages, obj.get("role", ""), obj.get("content", ""))
    return messages


def _example_from_embedded_object(obj, *, source: str) -> TrainingExample | None:
    if not isinstance(obj, dict):
        return None
    messages: list[TrainingMessage] = []

    if "chosen" in obj or "rejected" in obj:
        messages.extend(_parse_hh_conversation(str(obj.get("chosen") or obj.get("rejected") or "")))

    for key in ("content", "messages", "conversations"):
        if isinstance(obj.get(key), list):
            messages.extend(_messages_from_list(obj[key]))

    instruction = obj.get("instruction")
    output = obj.get("output") or obj.get("response") or obj.get("answer")
    if instruction:
        user = _clean_instruction(instruction)
        extra = _clean_text(obj.get("input"))
        if extra:
            user = f"{user}\n\n{extra}"
        _append_message(messages, "user", user)
        _append_message(messages, "assistant", output)
    elif obj.get("prompt") and output:
        user, embedded_answer = _split_legacy_prompt(obj.get("prompt"))
        _append_message(messages, "user", user)
        _append_message(messages, "assistant", output or embedded_answer)

    teacher = _clean_text(obj.get("teacher_response"))
    if teacher:
        _append_message(messages, "assistant", teacher)

    deduped = _deduplicate_messages(messages)
    return TrainingExample(deduped, source=source) if deduped else None


def _parse_source_example(ex: dict, *, source: str) -> TrainingExample | None:
    if not isinstance(ex, dict):
        return None
    prompt = ex.get("prompt")

    if isinstance(prompt, str):
        embedded = _try_parse_embedded_object(prompt)
        if embedded is not None:
            parsed = _example_from_embedded_object(embedded, source=source)
            if parsed is not None:
                outer = ex.get("response") or ex.get("output") or ex.get("answer")
                if outer:
                    messages = list(parsed.messages)
                    _append_message(messages, "assistant", outer)
                    return TrainingExample(_deduplicate_messages(messages), source=source)
                return parsed

        serialized = _parse_serialized_chat_turns(prompt)
        if serialized:
            outer = ex.get("response") or ex.get("output") or ex.get("answer")
            if outer:
                _append_message(serialized, "assistant", outer)
            return TrainingExample(_deduplicate_messages(serialized), source=source)

    parsed = _example_from_embedded_object(ex, source=source)
    if parsed is not None:
        return parsed

    text = _clean_text(ex.get("text"))
    if text:
        return TrainingExample((TrainingMessage("document", text),), source=source)
    return None


def serialize_training_example(example: TrainingExample) -> str:
    messages = [CanonicalMessage(message.role, message.content) for message in example.messages]
    return serialize_messages(messages)


def _load_json_file(path: Path) -> Iterator[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        print(f"  [DataLoader] Skip {path.name}: {exc}")
        return
    examples = data.get("examples") or data.get("data") or [] if isinstance(data, dict) else data if isinstance(data, list) else []
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        parsed = _parse_source_example(ex, source=str(path))
        if parsed is None:
            continue
        serialized = serialize_training_example(parsed)
        if len(serialized) > 20:
            yield serialized


def _load_txt_file(path: Path) -> Iterator[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    for paragraph in re.split(r"\n{2,}", text):
        serialized = serialize_document(paragraph)
        if len(serialized) > 30:
            yield serialized


def _stable_unique(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in texts:
        text = canonicalize_serialized(raw)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def load_all_training_text(
    data_root: Path = DATA_ROOT,
    max_examples: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> list[str]:
    all_texts: list[str] = []
    json_files = sorted(data_root.glob("*.json")) if data_root.exists() else []
    txt_files = sorted(data_root.glob("*.txt")) if data_root.exists() else []
    print(f"[DataLoader] Found {len(json_files)} root JSON files and {len(txt_files)} TXT files in {data_root}")

    for path in json_files:
        if "master_index" not in path.name.casefold():
            before = len(all_texts)
            all_texts.extend(_load_json_file(path))
            print(f"  {path.name}: +{len(all_texts) - before} examples")

    if data_root.exists():
        for subdir in sorted(data_root.iterdir()):
            if not subdir.is_dir():
                continue
            for path in sorted(subdir.glob("*.json")):
                before = len(all_texts)
                all_texts.extend(_load_json_file(path))
                count = len(all_texts) - before
                if count:
                    print(f"  {subdir.name}/{path.name}: +{count} examples")

    for path in txt_files:
        all_texts.extend(_load_txt_file(path))

    raw_count = len(all_texts)
    all_texts = _stable_unique(all_texts)
    print(f"[DataLoader] Parsed: {raw_count:,} | Unique canonical examples: {len(all_texts):,}")
    if shuffle:
        random.Random(seed).shuffle(all_texts)
    if max_examples is not None:
        all_texts = all_texts[: max(0, int(max_examples))]
    return all_texts


def build_corpus_string(texts: list[str], sep: str = "\n<sep>\n") -> str:
    return sep.join(canonicalize_serialized(text) for text in texts if text and text.strip())


def make_batches(
    token_ids: list[int],
    seq_len: int = 256,
    batch_size: int = 8,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[list[list[int]]]:
    windows = [
        token_ids[start : start + seq_len + 1]
        for start in range(0, max(0, len(token_ids) - seq_len - 1), seq_len)
    ]
    if shuffle:
        random.Random(seed).shuffle(windows)
    batch: list[list[int]] = []
    for window in windows:
        batch.append(window)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_tokenizer_training_sample(
    texts: list[str],
    *,
    max_chars: int = 8_000_000,
    seed: int = 42,
) -> str:
    if not texts or max_chars <= 0:
        return ""
    canonical = [canonicalize_serialized(text) for text in texts if text and text.strip()]
    rng = random.Random(seed)
    indices = list(range(len(canonical)))
    rng.shuffle(indices)
    target_per_text = max(256, max_chars // max(len(canonical), 1))
    pieces: list[str] = []
    remaining = max_chars
    for index in indices:
        if remaining <= 0:
            break
        text = canonical[index]
        take = min(len(text), target_per_text, remaining)
        if len(text) > take:
            start = rng.randint(0, len(text) - take)
            piece = text[start : start + take]
        else:
            piece = text
        pieces.append(piece)
        remaining -= len(piece)
    return "\n<sep>\n".join(pieces)
