"""
Byte-Pair Encoding (BPE) Tokenizer — pure Python, optimized with Priority-Queue BPE merges,
LRU Word Cache, multi-threaded batch encoding, and binary serialization.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import struct
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

TOKENIZER_ALGORITHM_VERSION = 3

PAD      = "<pad>"      # 0
UNK      = "<unk>"      # 1
BOS      = "<bos>"      # 2
EOS      = "<eos>"      # 3
SEP      = "<sep>"      # 4
THINK       = "<think>"        # 5
ENDTHINK    = "</think>"       # 6
USER        = "<user>"
ENDUSER     = "</user>"
ASSISTANT   = "<assistant>"
ENDASSISTANT= "</assistant>"
MARKET      = "<market>"
ENDMARKET   = "</market>"
ACCOUNT     = "<account>"
ENDACCOUNT  = "</account>"
POSITION    = "<position>"
ENDPOSITION = "</position>"
EVIDENCE    = "<evidence>"
ENDEVIDENCE = "</evidence>"
HYPOTHESIS  = "<hypothesis>"
ENDHYPOTHESIS = "</hypothesis>"
COUNTERCASE = "<countercase>"
ENDCOUNTERCASE = "</countercase>"
TOOL        = "<tool>"
ENDTOOL     = "</tool>"
TOOL_RESULT = "<tool_result>"
ENDTOOL_RESULT = "</tool_result>"
DECISION    = "<decision>"
ENDDECISION = "</decision>"
CONFIDENCE  = "<confidence>"
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


class BPETokenizer:

    def __init__(self, cache_max_size: int = 65536):
        self.algorithm_version = TOKENIZER_ALGORITHM_VERSION
        self.vocab: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []
        self._merge_map: dict[tuple[str, str], str] = {}
        self._merge_ranks: dict[tuple[str, str], int] = {}
        self._word_cache: dict[str, list[int]] = {}
        self._cache_max_size = cache_max_size

    def train(self, text: str, vocab_size: int = 8192, min_frequency: int = 2) -> None:
        print(f"[Tokenizer] Training BPE on {len(text):,} chars, target vocab={vocab_size}...", flush=True)

        vocab: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        chars = set(text)
        for ch in sorted(chars):
            if ch not in vocab:
                vocab[ch] = len(vocab)

        special_pattern = "(" + "|".join(
            re.escape(tok)
            for tok in sorted(
                SPECIAL_TOKENS,
                key=len,
                reverse=True,
            )
        ) + ")"

        ordinary_parts = [
            part
            for part in re.split(special_pattern, text)
            if part and part not in SPECIAL_TOKENS
        ]

        word_freqs = Counter()

        for part in ordinary_parts:
            word_freqs.update(
                re.findall(r'\S+', part)
            )

        def word_to_chars(word):
            return tuple(word)

        splits = {
            word: list(word_to_chars(word))
            for word in word_freqs
        }

        merges = []
        merge_map = {}
        merge_ranks = {}

        n_merges = vocab_size - len(vocab)
        for step in range(n_merges):
            pair_freqs: dict[tuple[str, str], int] = defaultdict(int)
            for word, freq in word_freqs.items():
                syms = splits[word]
                for i in range(len(syms) - 1):
                    pair_freqs[(syms[i], syms[i + 1])] += freq

            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.__getitem__)
            best_freq = pair_freqs[best_pair]

            if best_freq < min_frequency:
                break

            merged = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            merge_map[best_pair] = merged
            merge_ranks[best_pair] = step
            vocab[merged] = len(vocab)

            new_splits = {}
            for word, syms in splits.items():
                new_syms = []
                i = 0
                while i < len(syms):
                    if i < len(syms) - 1 and (syms[i], syms[i + 1]) == best_pair:
                        new_syms.append(merged)
                        i += 2
                    else:
                        new_syms.append(syms[i])
                        i += 1
                new_splits[word] = new_syms
            splits = new_splits

            if (step + 1) % 500 == 0:
                print(f"  [BPE] Step {step+1}/{n_merges}  vocab={len(vocab)}  best='{best_pair[0]}'+'{best_pair[1]}'->'{merged}'", flush=True)

        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.merges = merges
        self._merge_map = merge_map
        self._merge_ranks = merge_ranks
        self._word_cache.clear()
        print(f"[Tokenizer] Training Done. Vocabulary size: {len(self.vocab):,}", flush=True)

    def _tokenize_word(self, word: str) -> list[int]:
        if word in self._word_cache:
            return self._word_cache[word]

        syms = list(word)

        if self._merge_ranks:
            while len(syms) > 1:
                pairs = [(syms[i], syms[i + 1]) for i in range(len(syms) - 1)]
                min_pair = min(pairs, key=lambda p: self._merge_ranks.get(p, float("inf")))
                if min_pair not in self._merge_ranks:
                    break
                merged = self._merge_map[min_pair]
                new_syms = []
                i = 0
                while i < len(syms):
                    if i < len(syms) - 1 and (syms[i], syms[i + 1]) == min_pair:
                        new_syms.append(merged)
                        i += 2
                    else:
                        new_syms.append(syms[i])
                        i += 1
                syms = new_syms
        else:
            for pair, merged in self._merge_map.items():
                new_syms = []
                i = 0
                while i < len(syms):
                    if i < len(syms) - 1 and (syms[i], syms[i + 1]) == pair:
                        new_syms.append(merged)
                        i += 2
                    else:
                        new_syms.append(syms[i])
                        i += 1
                syms = new_syms

        unk_id = self.vocab.get(UNK, 1)
        token_ids = [self.vocab.get(s, unk_id) for s in syms]

        if len(self._word_cache) >= self._cache_max_size:
            # Evict 25% of cache when full
            evict_keys = list(self._word_cache.keys())[: self._cache_max_size // 4]
            for k in evict_keys:
                del self._word_cache[k]

        self._word_cache[word] = token_ids
        return token_ids

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        tokens: list[int] = []

        if add_bos:
            tokens.append(self.vocab.get(BOS, 2))

        special_pattern = "(" + "|".join(
            re.escape(tok)
            for tok in sorted(
                SPECIAL_TOKENS,
                key=len,
                reverse=True,
            )
        ) + ")"

        parts = re.split(special_pattern, text)
        unk_id = self.vocab.get(UNK, 1)

        for part in parts:
            if not part:
                continue

            if part in SPECIAL_TOKENS:
                tokens.append(
                    self.vocab.get(part, unk_id)
                )
                continue

            for segment in re.findall(r'\s+|\S+', part):
                if segment.isspace():
                    for char in segment:
                        tokens.append(
                            self.vocab.get(char, unk_id)
                        )
                else:
                    tokens.extend(
                        self._tokenize_word(segment)
                    )

        if add_eos:
            tokens.append(self.vocab.get(EOS, 3))

        return tokens


    def encode_batch(self, texts: list[str], workers: int = 4, add_bos: bool = True, add_eos: bool = True) -> list[list[int]]:
        if workers <= 1 or len(texts) < 10:
            return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(lambda t: self.encode(t, add_bos=add_bos, add_eos=add_eos), texts))

    def count_tokens(self, text: str) -> int:
        """Count tokens using the exact same grammar as encode()."""
        return len(
            self.encode(
                text,
                add_bos=True,
                add_eos=True,
            )
        )

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        tokens: list[str] = []

        for i in ids:
            tok = self.id_to_token.get(i, UNK)

            if skip_special and tok in SPECIAL_TOKENS:
                continue

            tokens.append(tok)

        return "".join(tokens)

    def fingerprint(self) -> str:
        vocab_items = sorted(self.vocab.items())
        merges_items = self.merges

        payload = json.dumps(
            {
                "algorithm_version": self.algorithm_version,
                "v": vocab_items,
                "m": merges_items,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

        return hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "algorithm_version": self.algorithm_version,
            "vocab": self.vocab,
            "merges": [[a, b] for a, b in self.merges],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Tokenizer] Saved to {path}  ({len(self.vocab):,} tokens)", flush=True)

    def save_binary(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({
            "v": self.vocab,
            "m": [[a, b] for a, b in self.merges]
        }, ensure_ascii=False).encode("utf-8")

        with path.open("wb") as fh:
            fh.write(b"BPE1")
            fh.write(struct.pack("<I", len(payload)))
            fh.write(payload)

    @classmethod
    def load_binary(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        with path.open("rb") as fh:
            magic = fh.read(4)
            if magic != b"BPE1":
                raise ValueError(f"Invalid binary tokenizer magic: {magic}")
            (length,) = struct.unpack("<I", fh.read(4))
            payload = fh.read(length).decode("utf-8")
            data = json.loads(payload)

        tok = cls()
        tok.vocab = {k: int(v) for k, v in data["v"].items()}
        tok.id_to_token = {int(v): k for k, v in data["v"].items()}
        tok.merges = [tuple(m) for m in data["m"]]
        tok._merge_map = {tuple(m): m[0] + m[1] for m in data["m"]}
        tok._merge_ranks = {tuple(m): i for i, m in enumerate(data["m"])}
        return tok

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        if path.suffix == ".bin":
            return cls.load_binary(path)

        data = json.loads(path.read_text(encoding="utf-8"))

        version = int(
            data.get(
                "algorithm_version",
                1,
            )
        )

        if version != TOKENIZER_ALGORITHM_VERSION:
            raise RuntimeError(
                "unsupported_tokenizer_algorithm_version:"
                f"{version}"
            )

        tok = cls()
        tok.algorithm_version = version
        tok.vocab = {k: int(v) for k, v in data["vocab"].items()}
        tok.id_to_token = {int(v): k for k, v in data["vocab"].items()}
        tok.merges = [tuple(m) for m in data["merges"]]
        tok._merge_map = {tuple(m): m[0] + m[1] for m in data["merges"]}
        tok._merge_ranks = {tuple(m): i for i, m in enumerate(data["merges"])}
        return tok

    def benchmark(self, sample_text: str, repeats: int = 50) -> float:
        t0 = time.perf_counter()
        total_tokens = 0
        for _ in range(repeats):
            total_tokens += len(self.encode(sample_text))
        elapsed = time.perf_counter() - t0
        tps = total_tokens / max(elapsed, 1e-9)
        print(f"[BPETokenizer] Speed: {tps:,.0f} tokens/sec ({repeats} runs, {total_tokens:,} tokens total)")
        return tps

    @property
    def vocab_size(self) -> int: return len(self.vocab)
    def pad_id(self) -> int:     return self.vocab.get(PAD, 0)
    def unk_id(self) -> int:     return self.vocab.get(UNK, 1)
    def bos_id(self) -> int:     return self.vocab.get(BOS, 2)
    def eos_id(self) -> int:     return self.vocab.get(EOS, 3)
