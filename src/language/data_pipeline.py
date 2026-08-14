"""
Authoritative language-model data pipeline.

All heterogeneous source datasets are normalized into one canonical training
protocol before tokenization:

    <user>...</user>
    <assistant>...</assistant>
    <eos>

Reasoning examples may additionally contain:

    <think>...</think>

The source files remain untouched. This module owns parsing, normalization,
deduplication and canonical serialization.
"""
from __future__ import annotations

import ast
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


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


def _normalize_structural_tokens(text: str) -> str:
    """
    Canonicalize model-control tokens onto explicit structural boundaries.

    The language model should never depend on adjacency such as:
        foo</think>
        <assistant><think>

    Canonical representation is:
        foo
        </think>
        <assistant>
        <think>

    This makes tokenizer behaviour deterministic and keeps training,
    evaluation, and inference on one grammar.
    """
    value = str(text or "")

    structural_tokens = [
        "<bos>", "<eos>", "<sep>",
        "<think>", "</think>",
        "<user>", "</user>",
        "<assistant>", "</assistant>",
        "<market>", "</market>",
        "<account>", "</account>",
        "<position>", "</position>",
        "<evidence>", "</evidence>",
        "<hypothesis>", "</hypothesis>",
        "<countercase>", "</countercase>",
        "<tool>", "</tool>",
        "<tool_result>", "</tool_result>",
        "<decision>", "</decision>",
        "<confidence>", "</confidence>",
        "<invalidation>", "</invalidation>",
    ]

    for token in structural_tokens:
        value = value.replace(token, f"\n{token}\n")

    # Normalize whitespace introduced around structural boundaries while
    # preserving paragraph breaks inside natural-language content.
    lines = [line.rstrip() for line in value.splitlines()]

    result: list[str] = []
    previous_blank = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if result and not previous_blank:
                result.append("")
            previous_blank = True
            continue

        result.append(stripped)
        previous_blank = False

    return "\n".join(result).strip()


def _clean_text(value: object) -> str:
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    text = _normalize_structural_tokens(text)

    # Remove duplicated source-role prefixes such as:
    #   Human: Human: question
    previous = None
    while text and text != previous:
        previous = text
        text = ROLE_PREFIX_RE.sub("", text, count=1).strip()

    return text


def _split_legacy_prompt(value: object) -> tuple[str, str]:
    """
    Split legacy records that embed the expected answer inside the prompt:

        Human: question
        Assistant: answer

    Returns (user_text, embedded_assistant_text).
    """
    text = str(value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Remove only the leading user marker here. Do not globally strip role
    # markers before locating the assistant boundary.
    text = re.sub(
        r"^\s*(?:Human|User)\s*:\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )

    match = re.search(
        r"(?:^|\n)\s*Assistant\s*:\s*",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return _clean_instruction(text), ""

    user_text = text[:match.start()].strip()
    assistant_text = text[match.end():].strip()

    return (
        _clean_instruction(user_text),
        _clean_text(assistant_text),
    )


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
    text = re.sub(
        r"\s*### Response:\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def _append_message(
    messages: list[TrainingMessage],
    role: str,
    content: object,
) -> None:
    role = str(role or "").strip().lower()

    role_map = {
        "human": "user",
        "user": "user",
        "assistant": "assistant",
        "ai": "assistant",
        "gpt": "assistant",
        "model": "assistant",
        "system": "system",
    }
    role = role_map.get(role, role)

    if role not in {"user", "assistant", "system"}:
        return

    cleaned = _clean_text(content)
    if not cleaned:
        return

    # Exact consecutive duplicate suppression.
    if messages:
        previous = messages[-1]
        if previous.role == role and previous.content == cleaned:
            return

    messages.append(TrainingMessage(role=role, content=cleaned))


def _deduplicate_messages(
    messages: list[TrainingMessage],
) -> tuple[TrainingMessage, ...]:
    result: list[TrainingMessage] = []

    for message in messages:
        if not message.content:
            continue

        # Prevent duplicated answers caused by teacher_response mirroring content.
        if result and result[-1] == message:
            continue

        result.append(message)

    return tuple(result)


def _parse_hh_conversation(value: str) -> list[TrainingMessage]:
    value = str(value or "").strip()
    if not value:
        return []

    markers = list(
        re.finditer(
            r"(?:^|\n\n)(Human|Assistant):\s*",
            value,
            flags=re.IGNORECASE,
        )
    )
    if not markers:
        return []

    messages: list[TrainingMessage] = []

    for index, marker in enumerate(markers):
        content_start = marker.end()
        content_end = (
            markers[index + 1].start()
            if index + 1 < len(markers)
            else len(value)
        )
        role = marker.group(1)
        content = value[content_start:content_end]
        _append_message(messages, role, content)

    return messages


def _try_parse_embedded_object(prompt_str: str):
    prompt_str = str(prompt_str or "").strip()
    if not prompt_str:
        return None

    # Proper JSON first.
    try:
        return json.loads(prompt_str)
    except Exception:
        pass

    # Python-literal datasets (TeichAI and similar).
    try:
        return ast.literal_eval(prompt_str)
    except Exception:
        return None


def _messages_from_conversation_list(values) -> list[TrainingMessage]:
    """Normalize ShareGPT/Kimi-style {from, value} conversations."""
    messages: list[TrainingMessage] = []

    if not isinstance(values, list):
        return messages

    for item in values:
        if not isinstance(item, dict):
            continue

        role = item.get("from", item.get("role", ""))
        content = item.get("value", item.get("content", ""))

        if str(role).strip().lower() in {"human", "user"}:
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

        if str(role).lower() in {"user", "human"}:
            content = _clean_instruction(str(content))

        _append_message(messages, role, content)

    return messages


def _parse_serialized_chat_turns(value: str) -> list[TrainingMessage]:
    """
    Parse datasets containing one Python/JSON-style message object per line.

    Example:
        {'role': 'system', 'content': ''}
        {'role': 'user', 'content': '...'}
        {'role': 'assistant', 'content': '...'}
    """
    raw = str(value or "").strip()
    if not raw:
        return []

    messages: list[TrainingMessage] = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        obj = _try_parse_embedded_object(line)
        if not isinstance(obj, dict):
            return []

        if "role" not in obj or "content" not in obj:
            return []

        role = obj.get("role", "")
        content = obj.get("content", "")

        if str(role).strip().lower() in {"human", "user"}:
            content = _clean_instruction(str(content))

        _append_message(messages, role, content)

    return messages


def _example_from_embedded_object(
    obj,
    *,
    source: str,
) -> TrainingExample | None:
    if not isinstance(obj, dict):
        return None

    messages: list[TrainingMessage] = []

    # Anthropic HH-RLHF.
    if "chosen" in obj or "rejected" in obj:
        chosen = obj.get("chosen") or obj.get("rejected") or ""
        messages.extend(_parse_hh_conversation(str(chosen)))

    # Standard messages/content list.
    contents = obj.get("content")
    if isinstance(contents, list):
        messages.extend(_messages_from_message_list(contents))

    messages_value = obj.get("messages")
    if isinstance(messages_value, list):
        messages.extend(_messages_from_message_list(messages_value))

    conversations = obj.get("conversations")
    if isinstance(conversations, list):
        messages.extend(
            _messages_from_conversation_list(conversations)
        )

    # Generic instruction schemas.
    instruction = obj.get("instruction")
    input_text = obj.get("input")
    output = (
        obj.get("output")
        or obj.get("response")
        or obj.get("answer")
    )

    if instruction:
        user = _clean_instruction(str(instruction))
        extra = _clean_text(input_text)
        if extra:
            user = f"{user}\n\n{extra}"
        _append_message(messages, "user", user)
        _append_message(messages, "assistant", output)

    # Generic prompt-response schema. Some downloaded records contain
    # "Human: ... Assistant: ..." inside prompt while response/output repeats
    # the assistant answer. Split that legacy representation before
    # canonical serialization.
    elif obj.get("prompt") and output:
        user_text, embedded_assistant = _split_legacy_prompt(
            obj.get("prompt", "")
        )

        _append_message(
            messages,
            "user",
            user_text,
        )

        authoritative_answer = _clean_text(output)

        # Prefer the explicit response/output field. The embedded assistant
        # section is retained only when the record has no usable outer answer.
        _append_message(
            messages,
            "assistant",
            authoritative_answer or embedded_assistant,
        )

    # LMSYS teacher response.
    teacher_response = _clean_text(obj.get("teacher_response"))
    if teacher_response:
        # Do not append the teacher response when the immediately preceding
        # assistant turn already contains exactly the same answer.
        if not (
            messages
            and messages[-1].role == "assistant"
            and messages[-1].content == teacher_response
        ):
            _append_message(messages, "assistant", teacher_response)

    messages_tuple = _deduplicate_messages(messages)

    if not messages_tuple:
        return None

    return TrainingExample(
        messages=messages_tuple,
        source=source,
    )


def _parse_source_example(
    ex: dict,
    *,
    source: str,
) -> TrainingExample | None:
    if not isinstance(ex, dict):
        return None

    prompt = ex.get("prompt")

    # Many downloaded datasets store a complete nested record inside prompt.
    if isinstance(prompt, str):
        embedded = _try_parse_embedded_object(prompt)
        if embedded is not None:
            parsed = _example_from_embedded_object(
                embedded,
                source=source,
            )
            if parsed is not None:
                # Some wrapper records store the true answer outside the
                # embedded prompt.
                outer_response = (
                    ex.get("response")
                    or ex.get("output")
                    or ex.get("answer")
                )

                if outer_response:
                    msgs = list(parsed.messages)
                    _append_message(
                        msgs,
                        "assistant",
                        outer_response,
                    )
                    return TrainingExample(
                        messages=_deduplicate_messages(msgs),
                        source=source,
                    )
                return parsed

    # TeichAI-style prompt containing one serialized message object per line.
    if isinstance(prompt, str):
        serialized_chat = _parse_serialized_chat_turns(prompt)

        if serialized_chat:
            outer_response = (
                ex.get("response")
                or ex.get("output")
                or ex.get("answer")
            )

            if outer_response:
                _append_message(
                    serialized_chat,
                    "assistant",
                    outer_response,
                )

            return TrainingExample(
                messages=_deduplicate_messages(serialized_chat),
                source=source,
            )

    # Direct record schema.
    parsed = _example_from_embedded_object(
        ex,
        source=source,
    )
    if parsed is not None:
        return parsed

    # Plain text record: retain as general pretraining material.
    text = _clean_text(ex.get("text"))
    if text:
        return TrainingExample(
            messages=(
                TrainingMessage(
                    role="document",
                    content=text,
                ),
            ),
            source=source,
        )

    return None


def serialize_training_example(example: TrainingExample) -> str:
    parts: list[str] = ["<bos>"]

    for message in example.messages:
        content = message.content.strip()
        if not content:
            continue

        if message.role == "user":
            parts.extend([
                "<user>",
                content,
                "</user>",
            ])
        elif message.role == "assistant":
            parts.extend([
                "<assistant>",
                content,
                "</assistant>",
            ])
        elif message.role == "system":
            # Until a dedicated system token is added, preserve system
            # context as evidence rather than inventing another grammar.
            parts.extend([
                "<evidence>",
                content,
                "</evidence>",
            ])
        elif message.role == "document":
            # General pretraining material is intentionally left unwrapped
            # inside BOS/EOS. It teaches language without pretending to be
            # conversation.
            parts.append(content)

    parts.append("<eos>")
    return "\n".join(parts)


def _load_json_file(path: Path) -> Iterator[str]:
    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )
    except Exception as exc:
        print(f"  [DataLoader] Skip {path.name}: {exc}")
        return

    if isinstance(data, dict):
        examples = (
            data.get("examples")
            or data.get("data")
            or []
        )
    elif isinstance(data, list):
        examples = data
    else:
        examples = []

    for ex in examples:
        if not isinstance(ex, dict):
            continue

        parsed = _parse_source_example(
            ex,
            source=str(path),
        )
        if parsed is None:
            continue

        serialized = serialize_training_example(parsed)

        if len(serialized.strip()) > 20:
            yield serialized


def _load_txt_file(path: Path) -> Iterator[str]:
    try:
        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return

    for paragraph in re.split(r"\n{2,}", text):
        paragraph = _clean_text(paragraph)
        if len(paragraph) <= 30:
            continue

        example = TrainingExample(
            messages=(
                TrainingMessage(
                    role="document",
                    content=paragraph,
                ),
            ),
            source=str(path),
        )
        yield serialize_training_example(example)


def _stable_unique(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for text in texts:
        normalized = text.strip()
        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        result.append(normalized)

    return result


def load_all_training_text(
    data_root: Path = DATA_ROOT,
    max_examples: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> list[str]:
    """
    Load all supported datasets and return canonical serialized examples.
    """
    all_texts: list[str] = []
    rng = random.Random(seed)

    json_files = sorted(data_root.glob("*.json"))
    txt_files = sorted(data_root.glob("*.txt"))

    print(
        f"[DataLoader] Found {len(json_files)} root JSON files "
        f"and {len(txt_files)} TXT files in {data_root}"
    )

    for path in json_files:
        if "master_index" in path.name.lower():
            continue

        before = len(all_texts)
        all_texts.extend(_load_json_file(path))
        print(
            f"  {path.name}: "
            f"+{len(all_texts) - before} examples"
        )

    for subdir in sorted(data_root.iterdir()):
        if not subdir.is_dir():
            continue

        for path in sorted(subdir.glob("*.json")):
            before = len(all_texts)
            all_texts.extend(_load_json_file(path))
            count = len(all_texts) - before

            if count:
                print(
                    f"  {subdir.name}/{path.name}: "
                    f"+{count} examples"
                )

    for path in txt_files:
        all_texts.extend(_load_txt_file(path))

    raw_count = len(all_texts)
    all_texts = _stable_unique(all_texts)

    print(
        f"[DataLoader] Parsed: {raw_count:,} | "
        f"Unique canonical examples: {len(all_texts):,}"
    )

    if shuffle:
        rng.shuffle(all_texts)

    if max_examples is not None:
        all_texts = all_texts[:max(0, int(max_examples))]

    return all_texts


def build_corpus_string(
    texts: list[str],
    sep: str = "\n<sep>\n",
) -> str:
    return sep.join(
        text.strip()
        for text in texts
        if text.strip()
    )


def make_batches(
    token_ids: list[int],
    seq_len: int = 256,
    batch_size: int = 8,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[list[list[int]]]:
    rng = random.Random(seed)

    windows = [
        token_ids[start:start + seq_len + 1]
        for start in range(
            0,
            len(token_ids) - seq_len - 1,
            seq_len,
        )
    ]

    if shuffle:
        rng.shuffle(windows)

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
    """
    Sample across the complete canonical corpus instead of taking an initial
    contiguous prefix.
    """
    if not texts or max_chars <= 0:
        return ""

    rng = random.Random(seed)
    indices = list(range(len(texts)))
    rng.shuffle(indices)

    target_per_text = max(
        256,
        max_chars // max(len(texts), 1),
    )

    pieces: list[str] = []
    remaining = max_chars

    for index in indices:
        if remaining <= 0:
            break

        text = texts[index].strip()
        if not text:
            continue

        take = min(
            len(text),
            target_per_text,
            remaining,
        )

        if len(text) > take:
            start = rng.randint(
                0,
                len(text) - take,
            )
            piece = text[start:start + take]
        else:
            piece = text

        pieces.append(piece)
        remaining -= len(piece)

    # Real newlines. Do not emit literal backslash-n sequences.
    return "\n<sep>\n".join(pieces)
