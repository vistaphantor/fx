from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.tokenizer import BPETokenizer, TOKENIZER_ALGORITHM_VERSION


@dataclass(frozen=True)
class TrainingPreflightReport:
    train_examples: int
    validation_examples: int
    train_sequences: int
    validation_sequences: int
    train_prediction_tokens: int
    validation_prediction_tokens: int
    tokenizer_vocab_size: int
    tokenizer_algorithm_version: int
    roundtrip_cases: int
    overfit_initial_loss: float
    overfit_final_loss: float


def normalize_prompt_family(text: str) -> str:
    """Stable family key used to prevent prompt variants leaking into validation."""
    match = re.search(r"<user>\s*(.*?)\s*</user>", text, flags=re.DOTALL)
    value = match.group(1) if match else text[:512]
    value = value.casefold()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[\s\.,;:!?]+$", "", value)
    return value.strip()


def split_by_prompt_family(
    texts: list[str],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    groups: dict[str, list[str]] = defaultdict(list)
    for text in texts:
        groups[normalize_prompt_family(text)].append(text)

    keys = list(groups)
    random.Random(seed).shuffle(keys)
    target_val = max(1, int(round(len(texts) * val_fraction)))
    train: list[str] = []
    val: list[str] = []

    for key in keys:
        group = groups[key]
        if len(val) < target_val:
            val.extend(group)
        else:
            train.extend(group)

    if not train:
        raise RuntimeError("family_split_produced_empty_training_set")
    if not val:
        raise RuntimeError("family_split_produced_empty_validation_set")

    train_families = {normalize_prompt_family(text) for text in train}
    val_families = {normalize_prompt_family(text) for text in val}
    overlap = train_families.intersection(val_families)
    if overlap:
        raise RuntimeError(f"prompt_family_leakage:{len(overlap)}")
    return train, val


def _content_windows(token_ids: list[int], seq_len: int) -> list[list[int]]:
    """Fallback for documents/prompts that cannot be structurally chunked."""
    if len(token_ids) < 2:
        return []
    window = seq_len + 1
    if len(token_ids) <= window:
        return [token_ids]

    stride = max(1, seq_len // 2)
    chunks: list[list[int]] = []
    start = 0
    while start < len(token_ids) - 1:
        chunk = token_ids[start : start + window]
        if len(chunk) >= 2:
            chunks.append(chunk)
        if start + window >= len(token_ids):
            break
        start += stride
    return chunks


def _contextual_long_chunks(
    text: str,
    tokenizer: BPETokenizer,
    *,
    seq_len: int,
) -> list[list[int]]:
    """Chunk a long answer while repeating its question/assistant context.

    Intermediate reasoning chunks intentionally do not synthesize closing tags or a
    fake final answer. Only the final chunk learns the real reasoning close and final
    response, avoiding the premature-termination signal produced by artificial tails.
    """
    max_tokens = seq_len + 1
    full_ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    if len(full_ids) <= max_tokens:
        return [full_ids]

    reasoning = re.match(
        r"(?s)^(.*?<assistant>\s*<think>\s*)(.*?)(\s*</think>.*)$",
        text,
    )
    if reasoning:
        prefix_text, body_text, final_tail_text = reasoning.groups()
    else:
        assistant = re.match(
            r"(?s)^(.*?<assistant>\s*)(.*?)(\s*</assistant>\s*<eos>\s*)$",
            text,
        )
        if not assistant:
            return _content_windows(full_ids, seq_len)
        prefix_text, body_text, final_tail_text = assistant.groups()

    prefix_ids = tokenizer.encode(prefix_text, add_bos=False, add_eos=False)
    body_ids = tokenizer.encode(body_text, add_bos=False, add_eos=False)
    final_tail_ids = tokenizer.encode(final_tail_text, add_bos=False, add_eos=False)

    # If the question/prefix itself consumes the context, arbitrary windowing is
    # more faithful than silently truncating or manufacturing a shortened prompt.
    if len(prefix_ids) >= max_tokens - 2:
        return _content_windows(full_ids, seq_len)

    intermediate_capacity = max_tokens - len(prefix_ids)
    final_capacity = max_tokens - len(prefix_ids) - len(final_tail_ids)
    if final_capacity < 1:
        return _content_windows(full_ids, seq_len)

    chunks: list[list[int]] = []
    cursor = 0
    while cursor < len(body_ids):
        remaining = len(body_ids) - cursor
        if remaining <= final_capacity:
            chunk = prefix_ids + body_ids[cursor:] + final_tail_ids
            chunks.append(chunk)
            cursor = len(body_ids)
            break

        take = min(intermediate_capacity, remaining - final_capacity)
        if take <= 0:
            return _content_windows(full_ids, seq_len)
        chunks.append(prefix_ids + body_ids[cursor : cursor + take])
        cursor += take

    if not chunks:
        return _content_windows(full_ids, seq_len)
    if any(len(chunk) < 2 or len(chunk) > max_tokens for chunk in chunks):
        raise RuntimeError("contextual_long_chunk_length_invalid")
    return chunks


def build_example_sequences(
    texts: list[str],
    tokenizer: BPETokenizer,
    *,
    seq_len: int,
) -> list[list[int]]:
    """Pack complete short examples and context-preserving long-example chunks."""
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")

    sequences: list[list[int]] = []
    pack: list[int] = []
    max_tokens = seq_len + 1

    for text in texts:
        if not text.strip():
            continue
        ids = tokenizer.encode(text, add_bos=False, add_eos=False)
        if len(ids) < 2:
            continue

        if len(ids) > max_tokens:
            if len(pack) >= 2:
                sequences.append(pack)
                pack = []
            sequences.extend(_contextual_long_chunks(text, tokenizer, seq_len=seq_len))
            continue

        if not pack:
            pack = list(ids)
        elif len(pack) + len(ids) <= max_tokens:
            pack.extend(ids)
        else:
            if len(pack) >= 2:
                sequences.append(pack)
            pack = list(ids)

    if len(pack) >= 2:
        sequences.append(pack)

    if not sequences:
        raise RuntimeError("packing_produced_no_sequences")
    if any(len(sequence) < 2 or len(sequence) > max_tokens for sequence in sequences):
        raise RuntimeError("invalid_packed_sequence_length")
    return sequences


class PackedSequenceDataset(Dataset):
    def __init__(self, sequences: list[list[int]], seq_len: int, pad_id: int):
        if not sequences:
            raise ValueError("sequences must not be empty")
        self.seq_len = int(seq_len)
        self.pad_id = int(pad_id)
        self.sequences = [list(sequence) for sequence in sequences]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx][: self.seq_len + 1]
        x = sequence[:-1]
        y = sequence[1:]
        if len(x) < self.seq_len:
            padding = self.seq_len - len(x)
            x = x + [self.pad_id] * padding
            y = y + [self.pad_id] * padding
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def prediction_token_count(sequences: list[list[int]]) -> int:
    return sum(max(0, len(sequence) - 1) for sequence in sequences)


def corpus_fingerprint(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8", errors="strict"))
        digest.update(b"\0")
    return digest.hexdigest()


def split_fingerprint(train_texts: list[str], val_texts: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(corpus_fingerprint(train_texts).encode("ascii"))
    digest.update(b":")
    digest.update(corpus_fingerprint(val_texts).encode("ascii"))
    return digest.hexdigest()


def _roundtrip_samples(train_texts: list[str]) -> list[str]:
    fixed = [
        "The market is bullish.",
        "one  two\n\nthree",
        "Walae Mkuu Mtaji",
        "KES 50,000 | XAUUSD @ 2,431.75",
        "€ £ ¥ KSh α β Σ ∂ 你好 مرحبا",
        r"\mathrm{ATR} \frac{1}{2}",
        "<user>What is RSI?</user>",
        "<assistant><think>Calculate.</think>Answer.</assistant>",
    ]
    fixed.extend(text[:2000] for text in train_texts[:8])
    return fixed


def validate_tokenizer_contract(tokenizer: BPETokenizer, train_texts: list[str]) -> int:
    if tokenizer.algorithm_version != TOKENIZER_ALGORITHM_VERSION:
        raise RuntimeError("non_authoritative_tokenizer_version")
    if tokenizer.vocab_size < 256:
        raise RuntimeError("tokenizer_vocab_too_small_for_byte_base")

    cases = _roundtrip_samples(train_texts)
    for sample in cases:
        ids = tokenizer.encode(sample, add_bos=False, add_eos=False)
        if tokenizer.unk_id() in ids:
            raise RuntimeError("tokenizer_emitted_unk_for_valid_utf8")
        decoded = tokenizer.decode(ids, skip_special=False)
        if decoded != sample:
            raise RuntimeError(
                "tokenizer_roundtrip_mismatch:"
                f"expected={sample[:80]!r}:actual={decoded[:80]!r}"
            )
    return len(cases)


def validate_sequence_contract(
    train_sequences: list[list[int]],
    val_sequences: list[list[int]],
    *,
    seq_len: int,
    vocab_size: int,
) -> None:
    for label, sequences in (("train", train_sequences), ("validation", val_sequences)):
        if not sequences:
            raise RuntimeError(f"{label}_sequences_empty")
        for sequence in sequences:
            if len(sequence) < 2 or len(sequence) > seq_len + 1:
                raise RuntimeError(f"{label}_sequence_length_invalid")
            if min(sequence) < 0 or max(sequence) >= vocab_size:
                raise RuntimeError(f"{label}_sequence_token_out_of_range")


def run_tiny_overfit_gate(tokenizer: BPETokenizer, train_sequences: list[list[int]]) -> tuple[float, float]:
    seq_len = min(48, max(8, len(train_sequences[0]) - 1))
    dataset = PackedSequenceDataset(
        train_sequences[: min(4, len(train_sequences))],
        seq_len,
        tokenizer.pad_id(),
    )
    x, y = dataset[0]
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)

    torch.manual_seed(20260814)
    model = VistaReasoningGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        ffn_dim=128,
        max_seq_len=seq_len,
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)

    model.train()
    _, initial_loss_tensor = model(x, targets=y, pad_id=tokenizer.pad_id())
    if initial_loss_tensor is None or not torch.isfinite(initial_loss_tensor):
        raise RuntimeError("preflight_initial_loss_invalid")
    initial = float(initial_loss_tensor.item())

    final = initial
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y, pad_id=tokenizer.pad_id())
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("preflight_overfit_loss_invalid")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final = float(loss.item())

    if final >= initial * 0.70:
        raise RuntimeError(
            f"preflight_overfit_failed:initial={initial:.4f}:final={final:.4f}"
        )
    return initial, final


def run_training_preflight(
    *,
    tokenizer: BPETokenizer,
    train_texts: list[str],
    val_texts: list[str],
    train_sequences: list[list[int]],
    val_sequences: list[list[int]],
    seq_len: int,
) -> TrainingPreflightReport:
    train_families = {normalize_prompt_family(text) for text in train_texts}
    val_families = {normalize_prompt_family(text) for text in val_texts}
    if train_families.intersection(val_families):
        raise RuntimeError("preflight_prompt_family_leakage")

    roundtrip_cases = validate_tokenizer_contract(tokenizer, train_texts)
    validate_sequence_contract(
        train_sequences,
        val_sequences,
        seq_len=seq_len,
        vocab_size=tokenizer.vocab_size,
    )
    initial_loss, final_loss = run_tiny_overfit_gate(tokenizer, train_sequences)

    return TrainingPreflightReport(
        train_examples=len(train_texts),
        validation_examples=len(val_texts),
        train_sequences=len(train_sequences),
        validation_sequences=len(val_sequences),
        train_prediction_tokens=prediction_token_count(train_sequences),
        validation_prediction_tokens=prediction_token_count(val_sequences),
        tokenizer_vocab_size=tokenizer.vocab_size,
        tokenizer_algorithm_version=tokenizer.algorithm_version,
        roundtrip_cases=roundtrip_cases,
        overfit_initial_loss=initial_loss,
        overfit_final_loss=final_loss,
    )
