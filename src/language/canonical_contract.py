from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

CANONICAL_CONTRACT_VERSION = 2

STRUCTURAL_TOKENS: tuple[str, ...] = (
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
)

_STRUCTURAL_RE = re.compile(
    "(" + "|".join(re.escape(token) for token in sorted(STRUCTURAL_TOKENS, key=len, reverse=True)) + ")"
)
_ROLE_PREFIX_RE = re.compile(r"^\s*(?:Human|User|Assistant|AI|System)\s*:\s*", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class CanonicalMessage:
    role: str
    content: str


def _escape_structural_literals(value: object) -> str:
    """Make reserved grammar strings inert when they originate in corpus payloads.

    Structural tokens are protocol, not ordinary dataset text. Raw web/chat data
    must never be able to create role/control state by merely containing a literal
    string such as ``<assistant>``. HTML-style escaping preserves the visible
    meaning while ensuring only the serializer itself can emit grammar tokens.
    """
    text = "" if value is None else str(value)
    for token in STRUCTURAL_TOKENS:
        if token in text:
            escaped = token.replace("<", "&lt;").replace(">", "&gt;")
            text = text.replace(token, escaped)
    return text


def normalize_text(value: object, *, strip_role_prefix: bool = False) -> str:
    """Normalize already-serialized text and structural tokens idempotently.

    This function is for canonical protocol strings. Dataset payloads must pass
    through :func:`normalize_payload_text` before wrappers are introduced.
    Structural tokens are always emitted on their own lines. Applying this
    function repeatedly produces byte-identical output.
    """
    if value is None:
        return ""
    text = str(value).replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    pieces: list[str] = []
    for part in _STRUCTURAL_RE.split(text):
        if not part:
            continue
        if part in STRUCTURAL_TOKENS:
            pieces.append(part)
            continue
        ordinary = part.strip()
        if ordinary:
            pieces.append(ordinary)

    normalized = "\n".join(pieces).strip()
    if strip_role_prefix:
        previous = None
        while normalized and normalized != previous:
            previous = normalized
            normalized = _ROLE_PREFIX_RE.sub("", normalized, count=1).strip()
    return normalized


def normalize_payload_text(value: object, *, strip_role_prefix: bool = False) -> str:
    """Normalize untrusted corpus content without allowing grammar injection."""
    return normalize_text(
        _escape_structural_literals(value),
        strip_role_prefix=strip_role_prefix,
    )


def serialize_messages(messages: Iterable[CanonicalMessage]) -> str:
    parts: list[str] = ["<bos>"]
    appended = 0
    for message in messages:
        role = str(message.role or "").strip().casefold()
        content = normalize_payload_text(message.content, strip_role_prefix=True)
        if not content:
            continue
        if role in {"human", "user"}:
            parts.extend(("<user>", content, "</user>"))
        elif role in {"assistant", "ai", "gpt", "model"}:
            parts.extend(("<assistant>", content, "</assistant>"))
        elif role == "system":
            parts.extend(("<evidence>", content, "</evidence>"))
        elif role == "document":
            parts.append(content)
        else:
            continue
        appended += 1
    if appended == 0:
        return ""
    parts.append("<eos>")
    return normalize_text("\n".join(parts))


def serialize_document(text: object) -> str:
    content = normalize_payload_text(text)
    if not content:
        return ""
    return normalize_text(f"<bos>\n{content}\n<eos>")


def canonicalize_serialized(text: object) -> str:
    return normalize_text(text)


def canonical_hash(text: str) -> str:
    value = canonicalize_serialized(text)
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def prompt_family(text: str) -> str:
    value = canonicalize_serialized(text)
    match = re.search(r"<user>\s*(.*?)\s*</user>", value, flags=re.DOTALL)
    prompt = match.group(1) if match else value[:512]
    prompt = prompt.casefold()
    prompt = re.sub(r"\s+", " ", prompt)
    prompt = re.sub(r"[\s\.,;:!?]+$", "", prompt)
    return prompt.strip()
