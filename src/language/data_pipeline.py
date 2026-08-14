"""
Authoritative language-model data pipeline.

All heterogeneous local datasets are normalized into the same canonical
protocol used by streamed sources before tokenization.
"""
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


ROLE_PREFIX_RE = re.compile(
    r"^\s*(?:Human|User|Assistant|AI|System)\s*:\s*",
    flags=re.IGNORECASE,
)


def _clean_text(value: object) -> str:
    return normalize_text(value, strip_role_prefix=True)


def _clean_instruction(text: str) -> str:
    text = _clean_text(text)
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


def _split_legacy_prompt(value: object) -> tuple[str, str]:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"^\s*(?:Human|User)\s*:\s*", "", text, count=1, flags=re.IGNORECASE)
    match = re.search(r"(?:^|\n)\s*Assistant\s*:\s*", text, flags=re.IGNORECASE)
    if not match:
        return _clean_instruction(text), ""
    return _clean_instruction(text[: match.start()]), _clean_text(text[match.end() :])


def _append_message(messages: list[TrainingMessage], role: str, content: object) -> None:
    role = str(role or "").strip().casefold()
    role_map = {
        "human": "user",
        "user": "user",
        "assistant": "assistant",
        "ai": "assistant",
        "gpt": "assistant",
        "model": "assistant",
        "system": "system",
    }
    normalized_role = role_map.get(role, role)
    if normalized_role not in {"user", "assistant", "system"}:
        return
    cleaned = _clean_text(content)
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


def _parse_hh_conversation(value: str) -> list[TrainingMessage]:
    value = str(value or "").strip()
    if not value:
        return []
    # Some downloaded JSON escaped newlines into literal backslash-n pairs.
    if "\\n\\nHuman:" in value and "\n\nHuman:" not in value:
        value = value.replace("\\n", "\n")
    markers = list(re.finditer(r"(?:^|\n\n)(Human|Assistant):\s*", value, flags=re.IGNORECASE))
    if not markers:
        return []
    messages: list[TrainingMessage] = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(value)
        _append_message(messages, marker.group(1), value[marker.end() : end])
    return messages


def _try_parse_embedded_object(prompt_str: str):
    prompt_str = str(prompt_str or "").strip()
    if not prompt_str:
        return None
    try:
        return json.loads(prompt_str)
    except Exception:
        pass
    try:
        return ast.literal_eval(prompt_str)
    except Exception:
        return None


def _messages_from_conversation_list(values) -> list[TrainingMessage]:
    messages: list[TrainingMessage] = []
    if not isinstance(values, list):
        return messages
    for item in values:
        if not isinstance(item, dict):
            continue
        role = item.get("from", item.get("role", ""))
        content = item.get("value", item.get("content", ""))
        if str(role).strip().casefold() in {"human", "user"}:
            content = _clean_instruction(str(content))
        _append_message(messages, role, content)
    return messages


def _messages_from_message_list(values) -> list[TrainingMessage]:
    messages: list[TrainingMessage] = []
    if not isinstance(values, list):
        return messages
    for item in values:
        if not isinstance(item, dict):
            continue
        role = item.get("role", "")
        content = item.get("content", "")
        if str(role).strip().casefold() in {"human", "user"}:
            content = _clean_instruction(str(content))
        _append_message(messages, role, content)
    return messages


def _parse_serialized_chat_turns(value: str) -> list[TrainingMessage]:
    raw = str(value or "").strip()
    if not raw:
        return []
    messages: list[TrainingMessage] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = _try_parse_embedded_object(line)
        if not isinstance(obj, dict) or "role" not in obj or "content" not in obj:
            return []
        role = obj.get("role", "")
        content = obj.get("content", "")
        if str(role).strip().casefold() in {"human", "user"}:
            content = _clean_instruction(str(content))
        _append_message(messages, role, content)
    return messages


def _example_from_embedded_object(obj, *, source: str) -> TrainingExample | None:
    if not isinstance(obj, dict):
        return None
    messages: list[TrainingMessage] = []

    if "chosen" in obj or "rejected" in obj:
        messages.extend(_parse_hh_conversation(str(obj.get("chosen") or obj.get("rejected") or "")))

    contents = obj.get("content")
    if isinstance(contents, list):
        messages.extend(_messages_from_message_list(contents))
    messages_value = obj.get("messages")
    if isinstance(messages_value, list):
        messages.extend(_messages_from_message_list(messages_value))
    conversations = obj.get("conversations")
    if isinstance(conversations, list):
        messages.extend(_messages_from_conversation_list(conversations))

    instruction = obj.get("instruction")
    input_text = obj.get("input")
    output = obj.get("output") or obj.get("response") or obj.get("answer")
    if instruction:
        user = _clean_instruction(str(instruction))
        extra = _clean_text(input_text)
        if extra:
            user = f"{user}\n\n{extra}"
        _append_message(messages, "user", user)
        _append_message(messages, "assistant", output)
    elif obj.get("prompt") and output:
        user_text, embedded_assistant = _split_legacy_prompt(obj.get("prompt", ""))
        _append_message(messages, "user", user_text)
        authoritative_answer = _clean_text(output)
        _append_message(messages, "assistant", authoritative_answer or embedded_assistant)

    teacher_response = _clean_text(obj.get("teacher_response"))
    if teacher_response:
        if not (messages and messages[-1].role == "assistant" and messages[-1].content == teacher_response):
            _append_message(messages, "assistant", teacher_response)

    messages_tuple = _deduplicate_messages(messages)
    if not messages_tuple:
        return None
    return TrainingExample(messages_tuple, source=source)


def _parse_source_example(ex: dict, *, source: str) -> TrainingExample | None:
    if not isinstance(ex, dict):
        return None
    prompt = ex.get("prompt")
    if isinstance(prompt, str):
        embedded = _try_parse_embedded_object(prompt)
        if embedded is not None:
            parsed = _example_from_embedded_object(embedded, source=source)
            if parsed is not None:
                outer_response = ex.get("response") or ex.get("output") or ex.get("answer")
                if outer_response:
                    messages = list(parsed.messages)
                    _append_message(messages, "assistant", outer_response)
                    return TrainingExample(_deduplicate_messages(messages), source=source)
                return parsed

        serialized_chat = _parse_serialized_chat_turns(prompt)
        if serialized_chat:
            outer_response = ex.get("response") or ex.get("output") or ex.get("answer")
            if outer_response:
                _append_message(serialized_chat, "assistant", outer_response)
            return TrainingExample(_deduplicate_messages(serialized_chat), source=source)

    parsed = _example_from_embedded_object(ex, source=source)
    if parsed is not None:
        return parsed

    text = _clean_text(ex.get("text"))
    if text:
        return TrainingExample((TrainingMessage("document", text),), source=source)
    return None


def serialize_training_example(example: TrainingExample) -> str:
    canonical_messages = [CanonicalMessage(message.role, message.content) for message in example.messages]
    return serialize_messages(canonical_messages)


def _serialize_record(ex: object, *, source: str) -> str | None:
    if not isinstance(ex, dict):
        return None
    parsed = _parse_source_example(ex, source=source)
    if parsed is None:
        return None
    serialized = serialize_training_example(parsed)
    return serialized if len(serialized) > 20 else None


def _load_json_file(path: Path) -> Iterator[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        print(f"  [DataLoader] Skip {path.name}: {exc}")
        return
    examples = data.get("examples") or data.get("data") or [] if isinstance(data, dict) else data if isinstance(data, list) else []
    for ex in examples:
        serialized = _serialize_record(ex, source=str(path))
        if serialized:
            yield serialized


def _load_jsonl_file(path: Path) -> Iterator[str]:
    try:
        handle = path.open("r", encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"  [DataLoader] Skip {path.name}: {exc}")
        return
    with handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Bad individual rows should not discard the rest of a large file.
                continue
            serialized = _serialize_record(
                record,
                source=f"{path}:{line_number}",
            )
            if serialized:
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
    if not data_root.exists():
        print(f"[DataLoader] Training root does not exist: {data_root}")
        return []

    json_files = sorted(data_root.rglob("*.json"))
    jsonl_files = sorted(data_root.rglob("*.jsonl"))
    txt_files = sorted(data_root.rglob("*.txt"))
    print(
        f"[DataLoader] Found {len(json_files)} JSON, {len(jsonl_files)} JSONL "
        f"and {len(txt_files)} TXT files in {data_root}"
    )

    for path in json_files:
        if "master_index" in path.name.casefold():
            continue
        before = len(all_texts)
        all_texts.extend(_load_json_file(path))
        count = len(all_texts) - before
        if count:
            print(f"  {path.relative_to(data_root)}: +{count} examples")

    for path in jsonl_files:
        before = len(all_texts)
        all_texts.extend(_load_jsonl_file(path))
        count = len(all_texts) - before
        if count:
            print(f"  {path.relative_to(data_root)}: +{count} examples")

    for path in txt_files:
        before = len(all_texts)
        all_texts.extend(_load_txt_file(path))
        count = len(all_texts) - before
        if count:
            print(f"  {path.relative_to(data_root)}: +{count} examples")

    raw_count = len(all_texts)
    all_texts = _stable_unique(all_texts)
    print(
        f"[DataLoader] Parsed: {raw_count:,} | "
        f"Unique canonical examples: {len(all_texts):,}"
    )
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
    """Build a tokenizer sample from complete canonical examples only.

    Structural tokens are atomic model grammar. Random character slicing can
    cut `<assistant>` or `<think>` in half and teach BPE useless fragments.
    The character budget is therefore a soft ceiling: examples are shuffled
    deterministically and accepted whole until the budget is reached.
    """
    if not texts or max_chars <= 0:
        return ""
    canonical = [canonicalize_serialized(text) for text in texts if text and text.strip()]
    random.Random(seed).shuffle(canonical)
    pieces: list[str] = []
    used = 0
    separator_chars = len("\n<sep>\n")
    for text in canonical:
        additional = len(text) + (separator_chars if pieces else 0)
        if pieces and used + additional > max_chars:
            continue
        pieces.append(text)
        used += additional
        if used >= max_chars:
            break
    if not pieces:
        # A single unusually large record is still safer whole than sliced
        # through control-token boundaries.
        pieces.append(canonical[0])
    return "\n<sep>\n".join(pieces)
