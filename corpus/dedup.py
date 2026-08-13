from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class DeduplicationResult:
    kept: list[str]
    total: int
    exact_removed: int
    near_removed: int

    @property
    def duplicate_rate(self) -> float:
        removed = self.exact_removed + self.near_removed
        return removed / max(self.total, 1)


class Deduplicator:
    def __init__(self, seen_hashes: set[str] | None = None):
        self._seen: set[str] = seen_hashes if seen_hashes is not None else set()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    def exact(self, texts: list[str]) -> DeduplicationResult:
        kept = []
        exact_removed = 0
        for t in texts:
            h = self._hash(t)
            if h in self._seen:
                exact_removed += 1
            else:
                self._seen.add(h)
                kept.append(t)
        return DeduplicationResult(
            kept=kept,
            total=len(texts),
            exact_removed=exact_removed,
            near_removed=0,
        )

    def near(self, texts: list[str], threshold: float = 0.85) -> DeduplicationResult:
        raise NotImplementedError(
            "Near-deduplication (MinHash + LSH) is scheduled for Phase 2. "
            "Use exact() for Phase 1."
        )

    def seen_count(self) -> int:
        return len(self._seen)

    def reset(self) -> None:
        self._seen.clear()
