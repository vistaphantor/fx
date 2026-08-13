from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.language.tokenizer import BPETokenizer


@dataclass
class TokenHistogramBin:
    min_len: int
    max_len: int
    count: int


@dataclass
class TokenizerStatsResult:
    total_tokens: int
    total_documents: int
    average_length: float
    min_length: int
    max_length: int
    histogram: list[TokenHistogramBin]
    tokens_per_sec: float


class TokenizerStats:
    def __init__(self, tokenizer: "BPETokenizer"):
        self._tok = tokenizer
        self._space_id = tokenizer.vocab.get(" ", tokenizer.vocab.get("<unk>", 1))

    def count_tokens(self, text: str) -> int:
        words = re.findall(r"\S+|\n", text)
        total = 1  # BOS
        for w in words:
            ids = self._tok._tokenize_word(w)
            total += len(ids) + 1  # word subwords + space
        return total

    def count_documents(self, texts: list[str]) -> int:
        return len(texts)

    def average_document_length(self, texts: list[str]) -> float:
        if not texts:
            return 0.0
        counts = [self.count_tokens(t) for t in texts]
        return sum(counts) / len(counts)

    def token_histogram(self, texts: list[str], bins: int = 20) -> list[TokenHistogramBin]:
        if not texts:
            return []
        counts = sorted(self.count_tokens(t) for t in texts)
        min_c, max_c = counts[0], counts[-1]
        if min_c == max_c:
            return [TokenHistogramBin(min_c, max_c, len(counts))]
        width = max(1, (max_c - min_c) // bins)
        result: list[TokenHistogramBin] = []
        for i in range(bins):
            lo = min_c + i * width
            hi = lo + width if i < bins - 1 else max_c + 1
            cnt = sum(1 for c in counts if lo <= c < hi)
            result.append(TokenHistogramBin(lo, hi - 1, cnt))
        return result

    def estimate_epochs(self, dataset_tokens: int, target_tokens: int) -> float:
        if dataset_tokens <= 0:
            return float("inf")
        return target_tokens / dataset_tokens

    def tokens_per_category(self, categorized: dict[str, list[str]]) -> dict[str, int]:
        return {cat: sum(self.count_tokens(t) for t in texts)
                for cat, texts in categorized.items()}

    def benchmark(self, sample_text: str, repeats: int = 100) -> float:
        t0 = time.perf_counter()
        total = 0
        for _ in range(repeats):
            total += self.count_tokens(sample_text)
        elapsed = time.perf_counter() - t0
        tps = total / elapsed
        print(f"[TokenizerStats] {tps:,.0f} tokens/sec  ({repeats} repeats, {total:,} total tokens)")
        return tps

    def full_stats(self, texts: list[str]) -> TokenizerStatsResult:
        t0 = time.perf_counter()
        counts = [self.count_tokens(t) for t in texts]
        elapsed = time.perf_counter() - t0
        total = sum(counts)
        return TokenizerStatsResult(
            total_tokens=total,
            total_documents=len(texts),
            average_length=total / max(len(counts), 1),
            min_length=min(counts) if counts else 0,
            max_length=max(counts) if counts else 0,
            histogram=self.token_histogram(texts),
            tokens_per_sec=total / max(elapsed, 1e-9),
        )
