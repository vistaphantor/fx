from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

CANONICAL_CONTRACT_VERSION = 3

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

# Strong signals that UTF-8 bytes were decoded as Latin-1/Windows-1252. These
# characters are not blanket-banned: they only trigger a conservative repair
# attempt whose result must strictly reduce the corruption score.
_MOJIBAKE_MARKERS: tuple[str, ...] = (
    "Ã", "Â", "â€", "â€™", "â€œ", "â€�", "â€“", "â€”", "â€¦",
    "ðŸ", "ï»¿", "ï¿½", "�",
)


@dataclass(frozen=True, slots=True)
class CanonicalMessage:
    role: str
    content: str


def mojibake_score(value: object) -> int:
    """Return a bounded corruption score for common UTF-8 mojibake patterns."""
    text = "" if value is None else str(value)
    score = text.count("�") * 8
    for marker in _MOJIBAKE_MARKERS:
        if marker == "�":
            continue
        score += text.count(marker) * 2
    # C1 controls often appear after a bad single-byte decode and should never
    # survive in natural-language training text.
    score += sum(1 for char in text if 0x80 <= ord(char) <= 0x9F) * 3
    return score


def repair_mojibake(value: object) -> str:
    """Conservatively repair common UTF-8-as-single-byte decoding corruption.

    We never modify clean text. A candidate repair is accepted only when the
    corruption score strictly decreases. At most two passes are attempted so
    doubly-encoded text can be repaired without creating an unbounded heuristic.
    """
    text = "" if value is None else str(value)
    if mojibake_score(text) <= 0:
        return text

    current = text
    current_score = mojibake_score(current)
    for _ in range(2):
        best = current
        best_score = current_score
        for encoding in ("cp1252", "latin-1"):
            try:
                candidate = current.encode(encoding, errors="strict").decode("utf-8", errors="strict")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            candidate_score = mojibake_score(candidate)
            if candidate_score < best_score:
                best = candidate
                best_score = candidate_score
        if best == current:
            break
        current, current_score = best, best_score
        if current_score == 0:
            break
    return current


def _escape_structural_literals(value: object) -> str:
    """Make reserved grammar strings inert when they originate in corpus payloads."""
    text = "" if value is None else str(value)
    for token in STRUCTURAL_TOKENS:
        if token in text:
            escaped = token.replace("<", "&lt;").replace(">", "&gt;")
            text = text.replace(token, escaped)
    return text


def normalize_text(value: object, *, strip_role_prefix: bool = False) -> str:
    """Normalize already-serialized text and structural tokens idempotently."""
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
    """Normalize untrusted corpus content without grammar injection or mojibake."""
    repaired = repair_mojibake(value)
    return normalize_text(
        _escape_structural_literals(repaired),
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
