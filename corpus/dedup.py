from __future__ import annotations

import hashlib
import re
from collections import defaultdict, deque
from dataclasses import dataclass

_WORD_RE = re.compile(r"[\w']+", flags=re.UNICODE)


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


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD_RE.finditer(text)]


def simhash64(text: str) -> int:
    """Return a deterministic 64-bit SimHash over lexical shingles.

    Byte-exact hashing is insufficient for streamed corpora because copied
    examples commonly differ only in punctuation, whitespace or a few words.
    SimHash gives us a bounded-memory near-duplicate signal without adding a
    large dependency or materializing the corpus.
    """
    words = _tokens(text)
    if not words:
        payloads = [text.casefold().strip()]
    else:
        width = 3 if len(words) >= 6 else 1
        payloads = [
            " ".join(words[index : index + width])
            for index in range(max(1, len(words) - width + 1))
        ]

    weights = [0] * 64
    for payload in payloads:
        digest = hashlib.blake2b(payload.encode("utf-8", errors="strict"), digest_size=8).digest()
        value = int.from_bytes(digest, "little", signed=False)
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1

    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming_distance64(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class NearDuplicateIndex:
    """Bounded-memory SimHash index using four 16-bit candidate bands."""

    def __init__(self, *, max_entries: int = 50_000, max_hamming_distance: int = 4):
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if not 0 <= max_hamming_distance <= 16:
            raise ValueError("max_hamming_distance must be between 0 and 16")
        self.max_entries = int(max_entries)
        self.max_hamming_distance = int(max_hamming_distance)
        self._next_id = 0
        self._order: deque[int] = deque()
        self._fingerprints: dict[int, int] = {}
        self._bands: tuple[dict[int, set[int]], ...] = tuple(defaultdict(set) for _ in range(4))

    @staticmethod
    def _band_values(fingerprint: int) -> tuple[int, int, int, int]:
        mask = (1 << 16) - 1
        return tuple((fingerprint >> (index * 16)) & mask for index in range(4))  # type: ignore[return-value]

    def _candidates(self, fingerprint: int) -> set[int]:
        candidates: set[int] = set()
        for band, value in zip(self._bands, self._band_values(fingerprint)):
            candidates.update(band.get(value, ()))
        return candidates

    def contains_near(self, text: str) -> bool:
        fingerprint = simhash64(text)
        for entry_id in self._candidates(fingerprint):
            existing = self._fingerprints.get(entry_id)
            if existing is not None and hamming_distance64(fingerprint, existing) <= self.max_hamming_distance:
                return True
        return False

    def add(self, text: str) -> None:
        fingerprint = simhash64(text)
        entry_id = self._next_id
        self._next_id += 1
        self._fingerprints[entry_id] = fingerprint
        self._order.append(entry_id)
        for band, value in zip(self._bands, self._band_values(fingerprint)):
            band[value].add(entry_id)

        while len(self._order) > self.max_entries:
            old_id = self._order.popleft()
            old = self._fingerprints.pop(old_id, None)
            if old is None:
                continue
            for band, value in zip(self._bands, self._band_values(old)):
                ids = band.get(value)
                if ids is None:
                    continue
                ids.discard(old_id)
                if not ids:
                    band.pop(value, None)

    def accept(self, text: str) -> bool:
        if self.contains_near(text):
            return False
        self.add(text)
        return True

    def __len__(self) -> int:
        return len(self._fingerprints)


class Deduplicator:
    def __init__(self, seen_hashes: set[str] | None = None):
        self._seen: set[str] = seen_hashes if seen_hashes is not None else set()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="strict")).hexdigest()

    def exact(self, texts: list[str]) -> DeduplicationResult:
        kept: list[str] = []
        exact_removed = 0
        for text in texts:
            digest = self._hash(text)
            if digest in self._seen:
                exact_removed += 1
            else:
                self._seen.add(digest)
                kept.append(text)
        return DeduplicationResult(
            kept=kept,
            total=len(texts),
            exact_removed=exact_removed,
            near_removed=0,
        )

    def near(
        self,
        texts: list[str],
        threshold: float = 0.85,
        *,
        max_entries: int = 50_000,
    ) -> DeduplicationResult:
        """Remove exact and near duplicates using a bounded SimHash index.

        `threshold` is retained as the public similarity control. Values closer
        to 1.0 are stricter; it is converted to a 64-bit Hamming radius.
        """
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        max_distance = max(0, min(16, int(round((1.0 - threshold) * 64))))
        index = NearDuplicateIndex(
            max_entries=max_entries,
            max_hamming_distance=max_distance,
        )
        kept: list[str] = []
        exact_removed = 0
        near_removed = 0
        for text in texts:
            digest = self._hash(text)
            if digest in self._seen:
                exact_removed += 1
                continue
            if not index.accept(text):
                near_removed += 1
                self._seen.add(digest)
                continue
            self._seen.add(digest)
            kept.append(text)
        return DeduplicationResult(
            kept=kept,
            total=len(texts),
            exact_removed=exact_removed,
            near_removed=near_removed,
        )

    def seen_count(self) -> int:
        return len(self._seen)

    def reset(self) -> None:
        self._seen.clear()
