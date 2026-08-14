from __future__ import annotations

from corpus.dedup import Deduplicator, NearDuplicateIndex, hamming_distance64, simhash64


def test_simhash_is_stable_and_similar_for_small_edits():
    left = "The trader uses ATR to measure market volatility and manage risk."
    right = "The trader uses ATR to measure market volatility, and manage risk."
    assert simhash64(left) == simhash64(left)
    assert hamming_distance64(simhash64(left), simhash64(right)) <= 4


def test_near_duplicate_index_rejects_reformatted_copy():
    index = NearDuplicateIndex(max_entries=100, max_hamming_distance=4)
    original = "A bullish market is one where buyers are pushing price higher over time."
    copy = "A bullish market is one where buyers are pushing price higher over time!"
    assert index.accept(original)
    assert not index.accept(copy)


def test_deduplicator_near_is_authoritative_not_stub():
    values = [
        "Risk management limits potential loss before a trade is entered.",
        "Risk management limits potential loss before a trade is entered!",
        "ATR measures the average true range and is commonly used as a volatility measure.",
    ]
    result = Deduplicator().near(values, threshold=0.93)
    assert result.total == 3
    assert result.near_removed >= 1
    assert len(result.kept) <= 2


def test_near_duplicate_index_is_memory_bounded():
    index = NearDuplicateIndex(max_entries=8, max_hamming_distance=2)
    for number in range(50):
        index.add(f"Unique training record number {number} with distinct lexical content.")
    assert len(index) == 8
