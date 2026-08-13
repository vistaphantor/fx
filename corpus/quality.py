from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QualityScore:
    accepted: bool
    score: float
    reasons: list[str] = field(default_factory=list)


class QualityFilter:
    def __init__(
        self,
        min_chars: int = 25,
        min_score: float = 0.4,
        max_repetition_ratio: float = 0.5,
        max_html_ratio: float = 0.3,
        min_printable_ratio: float = 0.8,
    ):
        self.min_chars = min_chars
        self.min_score = min_score
        self.max_repetition_ratio = max_repetition_ratio
        self.max_html_ratio = max_html_ratio
        self.min_printable_ratio = min_printable_ratio

    def score(self, text: str) -> QualityScore:
        reasons: list[str] = []
        penalties = 0.0

        if len(text) < self.min_chars:
            return QualityScore(accepted=False, score=0.0, reasons=["too_short"])

        lines = text.splitlines()
        total_lines = max(len(lines), 1)
        line_counts: dict[str, int] = {}
        for ln in lines:
            stripped = ln.strip()
            if stripped:
                line_counts[stripped] = line_counts.get(stripped, 0) + 1
        if line_counts:
            top_freq = max(line_counts.values()) / total_lines
            if top_freq > self.max_repetition_ratio:
                reasons.append("high_repetition")
                penalties += 0.4

        html_tags = len(re.findall(r"<[a-zA-Z][^>]*>", text))
        words = len(text.split())
        html_ratio = html_tags / max(words, 1)
        if html_ratio > self.max_html_ratio:
            reasons.append("mostly_html")
            penalties += 0.3

        printable = sum(1 for c in text if c.isprintable())
        printable_ratio = printable / max(len(text), 1)
        if printable_ratio < self.min_printable_ratio:
            reasons.append("low_printable")
            penalties += 0.5

        boilerplate_patterns = [
            r"^(cookie|privacy|terms of service|all rights reserved)",
            r"(subscribe to our newsletter|click here to)",
            r"^\s*\d+\s*$",
        ]
        lower = text.lower()
        for pat in boilerplate_patterns:
            if re.search(pat, lower):
                reasons.append("boilerplate")
                penalties += 0.2
                break

        if len(text) < 80:
            penalties += 0.1
            reasons.append("very_short")

        score = max(0.0, 1.0 - penalties)
        accepted = score >= self.min_score and "too_short" not in reasons
        return QualityScore(accepted=accepted, score=round(score, 4), reasons=reasons)

    def filter(self, texts: list[str]) -> tuple[list[str], float]:
        passed = []
        for t in texts:
            qs = self.score(t)
            if qs.accepted:
                passed.append(t)
        pass_rate = len(passed) / max(len(texts), 1)
        return passed, pass_rate
