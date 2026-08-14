from __future__ import annotations

import re
from dataclasses import dataclass, field

_STRUCTURAL_TOKEN_RE = re.compile(
    r"</?(?:bos|eos|sep|user|assistant|think|market|account|position|evidence|"
    r"hypothesis|countercase|tool|tool_result|decision|confidence|invalidation)>",
    flags=re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class QualityScore:
    accepted: bool
    score: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


class QualityFilter:
    """Authoritative bounded-cost quality gate for local and streamed text."""

    def __init__(
        self,
        min_chars: int = 25,
        max_chars: int = 100_000,
        min_score: float = 0.55,
        max_word_repetition_ratio: float = 0.50,
        min_unique_word_ratio: float = 0.06,
        max_html_ratio: float = 0.30,
        min_printable_ratio: float = 0.80,
        min_alnum_ratio: float = 0.12,
    ):
        if min_chars < 0 or max_chars <= min_chars:
            raise ValueError("invalid quality length bounds")
        self.min_chars = int(min_chars)
        self.max_chars = int(max_chars)
        self.min_score = float(min_score)
        self.max_word_repetition_ratio = float(max_word_repetition_ratio)
        self.min_unique_word_ratio = float(min_unique_word_ratio)
        self.max_html_ratio = float(max_html_ratio)
        self.min_printable_ratio = float(min_printable_ratio)
        self.min_alnum_ratio = float(min_alnum_ratio)

    @staticmethod
    def _structural_reasons(text: str) -> list[str]:
        reasons: list[str] = []
        if "<bos>" in text or "<eos>" in text:
            if text.count("<bos>") != 1 or text.count("<eos>") != 1:
                reasons.append("invalid_bos_eos")

        has_user = "<user>" in text or "</user>" in text
        has_assistant = "<assistant>" in text or "</assistant>" in text
        if has_user or has_assistant:
            # Anything encoded as conversation must contain both sides. Plain
            # documents have neither and remain valid pretraining material.
            if not has_user:
                reasons.append("missing_user")
            if not has_assistant:
                reasons.append("missing_assistant")
            if text.count("<user>") != text.count("</user>"):
                reasons.append("unbalanced_user")
            if text.count("<assistant>") != text.count("</assistant>"):
                reasons.append("unbalanced_assistant")
            if text.count("<user>") <= 0 or text.count("<assistant>") <= 0:
                reasons.append("empty_chat_side")

        if text.count("<think>") != text.count("</think>"):
            reasons.append("unbalanced_think")
        return reasons

    def score(self, text: str) -> QualityScore:
        value = str(text or "")
        reasons: list[str] = []
        penalties = 0.0

        if len(value) < self.min_chars:
            return QualityScore(False, 0.0, ("too_short",))
        if len(value) > self.max_chars:
            return QualityScore(False, 0.0, ("too_long",))

        structural = self._structural_reasons(value)
        if structural:
            return QualityScore(False, 0.0, tuple(structural))

        content = _STRUCTURAL_TOKEN_RE.sub(" ", value)
        printable = sum(1 for char in content if char.isprintable() or char in "\n\t")
        printable_ratio = printable / max(len(content), 1)
        if printable_ratio < self.min_printable_ratio:
            reasons.append("low_printable")
            penalties += 0.55

        alnum = sum(char.isalnum() for char in content)
        alnum_ratio = alnum / max(len(content), 1)
        if alnum_ratio < self.min_alnum_ratio:
            reasons.append("low_alphanumeric_density")
            penalties += 0.55

        words = [match.group(0).casefold() for match in _WORD_RE.finditer(content)]
        if len(words) >= 40:
            counts: dict[str, int] = {}
            for word in words:
                counts[word] = counts.get(word, 0) + 1
            repetition = max(counts.values()) / len(words)
            unique_ratio = len(counts) / len(words)
            if repetition >= self.max_word_repetition_ratio:
                reasons.append("high_word_repetition")
                penalties += 0.55
            if unique_ratio < self.min_unique_word_ratio:
                reasons.append("low_lexical_diversity")
                penalties += 0.45

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if len(lines) >= 12:
            counts: dict[str, int] = {}
            for line in lines:
                counts[line] = counts.get(line, 0) + 1
            if max(counts.values(), default=0) >= 8:
                reasons.append("repeated_lines")
                penalties += 0.55

        html_tags = len(re.findall(r"<[a-zA-Z][^>]*>", content))
        html_ratio = html_tags / max(len(words), 1)
        if html_ratio > self.max_html_ratio:
            reasons.append("mostly_html")
            penalties += 0.40

        lower = content.casefold()
        boilerplate_patterns = (
            r"(?:^|\n)\s*(?:cookie policy|privacy policy|terms of service|all rights reserved)\b",
            r"\b(?:subscribe to our newsletter|accept all cookies|click here to unsubscribe)\b",
        )
        if any(re.search(pattern, lower) for pattern in boilerplate_patterns):
            reasons.append("boilerplate")
            penalties += 0.35

        score = max(0.0, 1.0 - penalties)
        accepted = score >= self.min_score
        return QualityScore(accepted=accepted, score=round(score, 4), reasons=tuple(reasons))

    def accepts(self, text: str) -> bool:
        return self.score(text).accepted

    def filter(self, texts: list[str]) -> tuple[list[str], float]:
        passed = [text for text in texts if self.accepts(text)]
        return passed, len(passed) / max(len(texts), 1)


LANGUAGE_QUALITY_FILTER = QualityFilter()
