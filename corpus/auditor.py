from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CorpusStats:
    total_documents: int = 0
    total_tokens: int = 0
    target_tokens: int = 1_800_000_000_000

    reasoning_tokens: int = 0
    instruction_tokens: int = 0
    code_tokens: int = 0
    math_tokens: int = 0
    general_tokens: int = 0

    reasoning_docs: int = 0
    instruction_docs: int = 0
    code_docs: int = 0
    math_docs: int = 0
    general_docs: int = 0

    duplicate_rate: float = 0.0
    quality_pass_rate: float = 1.0

    docs_per_sec: float = 0.0
    tokens_per_sec: float = 0.0
    eta_seconds: float = 0.0
    memory_mb: float = 0.0

    shards_train: int = 0
    shards_val: int = 0

    @property
    def progress_pct(self) -> float:
        return (self.total_tokens / max(self.target_tokens, 1)) * 100

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.target_tokens - self.total_tokens)


def classify_text(text: str) -> str:
    if "<think>" in text:
        return "reasoning"
    lower = text[:500].lower()
    if any(k in lower for k in ["def ", "import ", "class ", "return ", "function "]):
        return "code"
    if any(k in lower for k in ["solve", "calculate", "equation", "integral", "formula", "= "]):
        return "math"
    if any(k in lower for k in ["human:", "assistant:", "user:", "instruction:"]):
        return "instruction"
    return "general"


class CorpusAuditor:
    def __init__(self, target_tokens: int = 1_800_000_000_000):
        self._target = target_tokens
        self.stats = CorpusStats(target_tokens=target_tokens)

    def reset(self) -> None:
        self.stats = CorpusStats(target_tokens=self._target)

    def record_document(self, text: str, token_count: int) -> str:
        category = classify_text(text)
        self.stats.total_documents += 1
        self.stats.total_tokens += token_count

        if category == "reasoning":
            self.stats.reasoning_tokens += token_count
            self.stats.reasoning_docs += 1
        elif category == "code":
            self.stats.code_tokens += token_count
            self.stats.code_docs += 1
        elif category == "math":
            self.stats.math_tokens += token_count
            self.stats.math_docs += 1
        elif category == "instruction":
            self.stats.instruction_tokens += token_count
            self.stats.instruction_docs += 1
        else:
            self.stats.general_tokens += token_count
            self.stats.general_docs += 1

        return category

    def update_throughput(self, docs_per_sec: float, tokens_per_sec: float) -> None:
        self.stats.docs_per_sec = docs_per_sec
        self.stats.tokens_per_sec = tokens_per_sec
        remaining = self.stats.remaining_tokens
        if tokens_per_sec > 0:
            self.stats.eta_seconds = remaining / tokens_per_sec

    def update_memory(self) -> None:
        try:
            import psutil
            proc = psutil.Process()
            self.stats.memory_mb = proc.memory_info().rss / 1e6
        except ImportError:
            pass

    def to_registry_dict(self) -> dict:
        s = self.stats
        return {
            "total_docs": s.total_documents,
            "total_tokens": s.total_tokens,
            "reasoning_tokens": s.reasoning_tokens,
            "code_tokens": s.code_tokens,
            "math_tokens": s.math_tokens,
            "instruction_tokens": s.instruction_tokens,
            "general_tokens": s.general_tokens,
            "duplicate_rate": s.duplicate_rate,
            "quality_rate": s.quality_pass_rate,
        }
