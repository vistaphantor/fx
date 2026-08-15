from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.language.canonical_contract import mojibake_score

_STRUCTURAL_TOKEN_RE = re.compile(
    r"</?(?:bos|eos|sep|user|assistant|think|market|account|position|evidence|"
    r"hypothesis|countercase|tool|tool_result|decision|confidence|invalidation)>",
    flags=re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_URL_RE = re.compile(r"(?:https?://|www\.|\b[a-z0-9.-]+\.(?:com|org|net|io|co|ai)\b)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_CODE_RE = re.compile(
    r"(?:\b(?:def|class|function|import|return|SELECT|INSERT|const|let|var)\b|"
    r"```|=>|::|</?[a-z][^>]*>)",
    re.IGNORECASE,
)
_MATH_RE = re.compile(
    r"(?:\\(?:frac|begin|end|mathrm|mathbf|sqrt|det|sum|int)\b|"
    r"\$[^$]{2,}\$|\b[a-zA-Z]\s*[=<>]\s*[-+]?\d|[{}]{2,})"
)


@dataclass(frozen=True)
class QualityScore:
    accepted: bool
    score: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


class QualityFilter:
    """Authoritative bounded-cost quality gate for canonical streamed text."""

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

        corruption = mojibake_score(value)
        if "�" in value:
            return QualityScore(False, 0.0, ("unicode_replacement_character",))
        if corruption >= 4:
            return QualityScore(False, 0.0, ("residual_mojibake",))

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
        return QualityScore(score >= self.min_score, round(score, 4), tuple(reasons))

    def accepts(self, text: str) -> bool:
        return self.score(text).accepted

    def filter(self, texts: list[str]) -> tuple[list[str], float]:
        passed = [text for text in texts if self.accepts(text)]
        return passed, len(passed) / max(len(texts), 1)


class FoundationEnglishFilter(QualityFilter):
    """Stricter gate for the first language stage of a very small model."""

    def __init__(self) -> None:
        super().__init__(
            min_chars=40,
            max_chars=20_000,
            min_score=0.70,
            max_word_repetition_ratio=0.32,
            min_unique_word_ratio=0.10,
            max_html_ratio=0.08,
            min_printable_ratio=0.95,
            min_alnum_ratio=0.30,
        )

    def score(self, text: str) -> QualityScore:
        base = super().score(text)
        if not base.accepted:
            return base

        value = _STRUCTURAL_TOKEN_RE.sub(" ", str(text or ""))
        words = _WORD_RE.findall(value)
        reasons = list(base.reasons)
        penalties = 1.0 - base.score

        if len(words) < 8:
            return QualityScore(False, 0.0, tuple(reasons + ["too_few_words_for_foundation"]))

        ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in value)
        all_letters = sum(ch.isalpha() for ch in value)
        if all_letters and ascii_letters / all_letters < 0.90:
            reasons.append("non_english_script_density")
            penalties += 0.45

        url_count = len(_URL_RE.findall(value)) + len(_EMAIL_RE.findall(value))
        if url_count >= 3 or url_count / max(len(words), 1) > 0.025:
            reasons.append("link_or_identifier_heavy")
            penalties += 0.45

        code_hits = len(_CODE_RE.findall(value))
        if code_hits >= 3 or code_hits / max(len(words), 1) > 0.03:
            reasons.append("code_or_markup_heavy")
            penalties += 0.50

        math_hits = len(_MATH_RE.findall(value))
        if math_hits >= 3 or math_hits / max(len(words), 1) > 0.025:
            reasons.append("math_notation_heavy")
            penalties += 0.50

        punctuation = sum(ch in ".!?" for ch in value)
        if len(words) >= 30 and punctuation == 0:
            reasons.append("no_sentence_boundaries")
            penalties += 0.35

        digit_words = sum(word.isdigit() for word in words)
        if digit_words / max(len(words), 1) > 0.16:
            reasons.append("numeric_heavy")
            penalties += 0.40

        score = max(0.0, 1.0 - penalties)
        return QualityScore(score >= self.min_score, round(score, 4), tuple(reasons))


LANGUAGE_QUALITY_FILTER = QualityFilter()
FOUNDATION_ENGLISH_FILTER = FoundationEnglishFilter()
