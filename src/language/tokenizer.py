"""Lossless byte-level BPE tokenizer for the Vista trading reasoner."""
from __future__ import annotations

import hashlib
import json
import re
import struct
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOKENIZER_ALGORITHM_VERSION = 4

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
SEP = "<sep>"
THINK = "<think>"
ENDTHINK = "</think>"
USER = "<user>"
ENDUSER = "</user>"
ASSISTANT = "<assistant>"
ENDASSISTANT = "</assistant>"
MARKET = "<market>"
ENDMARKET = "</market>"
ACCOUNT = "<account>"
ENDACCOUNT = "</account>"
POSITION = "<position>"
ENDPOSITION = "</position>"
EVIDENCE = "<evidence>"
ENDEVIDENCE = "</evidence>"
HYPOTHESIS = "<hypothesis>"
ENDHYPOTHESIS = "</hypothesis>"
COUNTERCASE = "<countercase>"
ENDCOUNTERCASE = "</countercase>"
TOOL = "<tool>"
ENDTOOL = "</tool>"
TOOL_RESULT = "<tool_result>"
ENDTOOL_RESULT = "</tool_result>"
DECISION = "<decision>"
ENDDECISION = "</decision>"
CONFIDENCE = "<confidence>"
ENDCONFIDENCE = "</confidence>"
INVALIDATION = "<invalidation>"
ENDINVALIDATION = "</invalidation>"

SPECIAL_TOKENS = [
    PAD, UNK, BOS, EOS, SEP, THINK, ENDTHINK,
    USER, ENDUSER, ASSISTANT, ENDASSISTANT,
    MARKET, ENDMARKET, ACCOUNT, ENDACCOUNT,
    POSITION, ENDPOSITION, EVIDENCE, ENDEVIDENCE,
    HYPOTHESIS, ENDHYPOTHESIS, COUNTERCASE, ENDCOUNTERCASE,
    TOOL, ENDTOOL, TOOL_RESULT, ENDTOOL_RESULT,
    DECISION, ENDDECISION, CONFIDENCE, ENDCONFIDENCE,
    INVALIDATION, ENDINVALIDATION,
]

_BYTE_PREFIX = "<byte:"
_SEGMENT_RE = re.compile(r"\s+|\S+")


def _byte_token(value: int) -> str:
    return f"{_BYTE_PREFIX}{value:02x}>"


def _adjacent_pair_counts(symbols: list[str]) -> Counter[tuple[str, str]]:
    return Counter(zip(symbols, symbols[1:]))


class BPETokenizer:
    """Byte-level BPE with atomic control tokens and exact UTF-8 round trips."""

    def __init__(self, cache_max_size: int = 65536):
        self.algorithm_version = TOKENIZER_ALGORITHM_VERSION
        self.vocab: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []
        self._merge_map: dict[tuple[str, str], str] = {}
        self._merge_ranks: dict[tuple[str, str], int] = {}
        self._word_cache: dict[str, list[int]] = {}
        self._cache_max_size = cache_max_size

    @staticmethod
    def _base_vocab() -> dict[str, int]:
        vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for value in range(256):
            vocab[_byte_token(value)] = len(vocab)
        return vocab

    @staticmethod
    def _bytes_to_symbols(value: str) -> list[str]:
        return [_byte_token(b) for b in value.encode("utf-8")]

    @staticmethod
    def _special_pattern() -> str:
        return "(" + "|".join(
            re.escape(tok) for tok in sorted(SPECIAL_TOKENS, key=len, reverse=True)
        ) + ")"

    @classmethod
    def _ordinary_segments(cls, text: str) -> list[str]:
        segments: list[str] = []
        for part in re.split(cls._special_pattern(), text):
            if not part or part in SPECIAL_TOKENS:
                continue
            segments.extend(_SEGMENT_RE.findall(part))
        return segments

    @staticmethod
    def _merge_pair(symbols: list[str], pair: tuple[str, str], merged: str) -> list[str]:
        out: list[str] = []
        i = 0
        while i < len(symbols):
            if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return out

    def train(self, text: str, vocab_size: int = 8192, min_frequency: int = 2) -> None:
        print(
            f"[Tokenizer] Training byte BPE v4 on {len(text):,} chars, "
            f"target vocab={vocab_size}...",
            flush=True,
        )
        vocab = self._base_vocab()
        if vocab_size < len(vocab):
            raise ValueError(f"vocab_size must be >= {len(vocab)} for byte-level BPE")

        segment_freqs = Counter(self._ordinary_segments(text))
        splits = {segment: self._bytes_to_symbols(segment) for segment in segment_freqs}

        pair_counts: Counter[tuple[str, str]] = Counter()
        pair_segments: dict[tuple[str, str], set[str]] = defaultdict(set)
        for segment, symbols in splits.items():
            frequency = segment_freqs[segment]
            for pair, occurrences in _adjacent_pair_counts(symbols).items():
                pair_counts[pair] += occurrences * frequency
                pair_segments[pair].add(segment)

        merges: list[tuple[str, str]] = []
        target_merges = vocab_size - len(vocab)
        started = time.perf_counter()

        while len(vocab) < vocab_size and pair_counts:
            best_pair, best_frequency = max(pair_counts.items(), key=lambda item: item[1])
            if best_frequency < min_frequency:
                break

            merged = best_pair[0] + best_pair[1]
            if merged in vocab:
                # A byte sequence can be reachable through a different merge tree.
                # Do not create duplicate token IDs; discard this redundant rule.
                pair_counts.pop(best_pair, None)
                pair_segments.pop(best_pair, None)
                continue

            affected = tuple(pair_segments.get(best_pair, ()))
            if not affected:
                pair_counts.pop(best_pair, None)
                continue

            merges.append(best_pair)
            vocab[merged] = len(vocab)

            touched_pairs: set[tuple[str, str]] = set()
            for segment in affected:
                old_symbols = splits[segment]
                frequency = segment_freqs[segment]
                old_pairs = _adjacent_pair_counts(old_symbols)
                for pair, occurrences in old_pairs.items():
                    pair_counts[pair] -= occurrences * frequency
                    pair_segments[pair].discard(segment)
                    touched_pairs.add(pair)

                new_symbols = self._merge_pair(old_symbols, best_pair, merged)
                splits[segment] = new_symbols
                new_pairs = _adjacent_pair_counts(new_symbols)
                for pair, occurrences in new_pairs.items():
                    pair_counts[pair] += occurrences * frequency
                    pair_segments[pair].add(segment)
                    touched_pairs.add(pair)

            for pair in touched_pairs:
                if pair_counts.get(pair, 0) <= 0:
                    pair_counts.pop(pair, None)
                    pair_segments.pop(pair, None)

            if len(merges) % 500 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  [BPE] merges={len(merges):,}/{target_merges:,} "
                    f"vocab={len(vocab):,} pairs={len(pair_counts):,} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.merges = merges
        self._merge_map = {pair: pair[0] + pair[1] for pair in merges}
        self._merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self._word_cache.clear()
        print(
            f"[Tokenizer] Training Done. Vocabulary size: {len(self.vocab):,}; "
            f"unique segments={len(segment_freqs):,}",
            flush=True,
        )

    def _tokenize_word(self, text: str) -> list[int]:
        if text in self._word_cache:
            return list(self._word_cache[text])

        symbols = self._bytes_to_symbols(text)
        while len(symbols) > 1:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            pair = min(pairs, key=lambda value: self._merge_ranks.get(value, float("inf")))
            if pair not in self._merge_ranks:
                break
            symbols = self._merge_pair(symbols, pair, self._merge_map[pair])

        ids = [self.vocab[symbol] for symbol in symbols]
        if len(self._word_cache) >= self._cache_max_size:
            for key in list(self._word_cache)[: max(1, self._cache_max_size // 4)]:
                del self._word_cache[key]
        self._word_cache[text] = ids
        return list(ids)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        tokens: list[int] = []
        if add_bos:
            tokens.append(self.bos_id())

        for part in re.split(self._special_pattern(), text):
            if not part:
                continue
            if part in SPECIAL_TOKENS:
                tokens.append(self.vocab[part])
                continue
            for segment in _SEGMENT_RE.findall(part):
                tokens.extend(self._tokenize_word(segment))

        if add_eos:
            tokens.append(self.eos_id())
        return tokens

    def encode_batch(
        self,
        texts: list[str],
        workers: int = 4,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[list[int]]:
        if workers <= 1 or len(texts) < 10:
            return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(
                executor.map(
                    lambda value: self.encode(value, add_bos=add_bos, add_eos=add_eos),
                    texts,
                )
            )

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text, add_bos=True, add_eos=True))

    @staticmethod
    def _symbol_bytes(symbol: str) -> bytes:
        matches = re.findall(r"<byte:([0-9a-f]{2})>", symbol)
        if not matches or "".join(f"<byte:{value}>" for value in matches) != symbol:
            raise ValueError(f"invalid_byte_symbol:{symbol!r}")
        return bytes(int(value, 16) for value in matches)

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        chunks: list[str] = []
        byte_buffer = bytearray()

        def flush_bytes() -> None:
            if byte_buffer:
                chunks.append(bytes(byte_buffer).decode("utf-8", errors="strict"))
                byte_buffer.clear()

        for token_id in ids:
            symbol = self.id_to_token.get(int(token_id), UNK)
            if symbol in SPECIAL_TOKENS:
                flush_bytes()
                if not skip_special:
                    chunks.append(symbol)
                continue
            byte_buffer.extend(self._symbol_bytes(symbol))
        flush_bytes()
        return "".join(chunks)

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "algorithm_version": self.algorithm_version,
                "v": sorted(self.vocab.items()),
                "m": self.merges,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _payload(self) -> dict:
        return {
            "algorithm_version": self.algorithm_version,
            "vocab": self.vocab,
            "merges": [[a, b] for a, b in self.merges],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._payload(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Tokenizer] Saved to {path} ({len(self.vocab):,} tokens)", flush=True)

    def save_binary(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._payload(), ensure_ascii=False).encode("utf-8")
        with path.open("wb") as fh:
            fh.write(b"BPE4")
            fh.write(struct.pack("<I", len(payload)))
            fh.write(payload)

    @classmethod
    def _from_payload(cls, data: dict) -> "BPETokenizer":
        version = int(data.get("algorithm_version", 1))
        if version != TOKENIZER_ALGORITHM_VERSION:
            raise RuntimeError(f"unsupported_tokenizer_algorithm_version:{version}")
        tok = cls()
        tok.algorithm_version = version
        tok.vocab = {k: int(v) for k, v in data["vocab"].items()}
        tok.id_to_token = {int(v): k for k, v in data["vocab"].items()}
        tok.merges = [tuple(merge) for merge in data["merges"]]
        tok._merge_map = {tuple(merge): merge[0] + merge[1] for merge in data["merges"]}
        tok._merge_ranks = {tuple(merge): i for i, merge in enumerate(data["merges"])}
        required = set(cls._base_vocab())
        missing = required.difference(tok.vocab)
        if missing:
            raise RuntimeError(f"invalid_byte_tokenizer_missing_base_tokens:{len(missing)}")
        return tok

    @classmethod
    def load_binary(cls, path: str | Path) -> "BPETokenizer":
        with Path(path).open("rb") as fh:
            if fh.read(4) != b"BPE4":
                raise ValueError("Invalid binary tokenizer magic")
            (length,) = struct.unpack("<I", fh.read(4))
            return cls._from_payload(json.loads(fh.read(length).decode("utf-8")))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        if path.suffix == ".bin":
            return cls.load_binary(path)
        return cls._from_payload(json.loads(path.read_text(encoding="utf-8")))

    def benchmark(self, sample_text: str, repeats: int = 50) -> float:
        t0 = time.perf_counter()
        total_tokens = sum(len(self.encode(sample_text)) for _ in range(repeats))
        elapsed = time.perf_counter() - t0
        tps = total_tokens / max(elapsed, 1e-9)
        print(
            f"[BPETokenizer] Speed: {tps:,.0f} tokens/sec "
            f"({repeats} runs, {total_tokens:,} tokens total)"
        )
        return tps

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def pad_id(self) -> int:
        return self.vocab.get(PAD, 0)

    def unk_id(self) -> int:
        return self.vocab.get(UNK, 1)

    def bos_id(self) -> int:
        return self.vocab.get(BOS, 2)

    def eos_id(self) -> int:
        return self.vocab.get(EOS, 3)
