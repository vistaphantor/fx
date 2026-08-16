from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.language.canonical_contract import prompt_family
from src.language.conditioning_gate import run_prompt_conditioning_gate
from src.language.loss_objective import build_loss_targets
from src.language.pytorch_transformer import VistaReasoningGPT
from src.language.tokenizer import (
    ASSISTANT,
    BOS,
    ENDASSISTANT,
    ENDTHINK,
    ENDUSER,
    EOS,
    THINK,
    USER,
    BPETokenizer,
    SPECIAL_TOKENS,
    TOKENIZER_ALGORITHM_VERSION,
)

_BOS_ID = SPECIAL_TOKENS.index(BOS)
_EOS_ID = SPECIAL_TOKENS.index(EOS)
_USER_ID = SPECIAL_TOKENS.index(USER)
_END_USER_ID = SPECIAL_TOKENS.index(ENDUSER)
_ASSISTANT_ID = SPECIAL_TOKENS.index(ASSISTANT)
_END_ASSISTANT_ID = SPECIAL_TOKENS.index(ENDASSISTANT)
_THINK_ID = SPECIAL_TOKENS.index(THINK)
_END_THINK_ID = SPECIAL_TOKENS.index(ENDTHINK)


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
    conditioning_recall_1: int
    conditioning_recall_8: int
    conditioning_recall_32: int


def normalize_prompt_family(text: str) -> str:
    return prompt_family(text)


def split_by_prompt_family(
    texts: list[str], *, val_fraction: float, seed: int,
) -> tuple[list[str], list[str]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    groups: dict[str, list[str]] = defaultdict(list)
    for text in texts:
        groups[prompt_family(text)].append(text)
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
    if not train or not val:
        raise RuntimeError("family_split_produced_empty_partition")
    overlap = {prompt_family(text) for text in train}.intersection(
        prompt_family(text) for text in val
    )
    if overlap:
        raise RuntimeError(f"prompt_family_leakage:{len(overlap)}")
    return train, val


def _content_windows(token_ids: list[int], seq_len: int) -> list[list[int]]:
    """Split long non-chat content without duplicating prediction targets."""
    if len(token_ids) < 2:
        return []
    window = seq_len + 1
    if len(token_ids) <= window:
        return [token_ids]
    stride = seq_len
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


def _current_user_context(full_ids: list[int], assistant_index: int, *, budget: int) -> list[int]:
    """Return bounded prompt context for one assistant turn without prior answer targets."""
    user_index = -1
    for index in range(assistant_index - 1, -1, -1):
        if full_ids[index] == _USER_ID:
            user_index = index
            break
        if full_ids[index] == _END_ASSISTANT_ID:
            break
    if user_index < 0:
        context = full_ids[1:assistant_index] if full_ids and full_ids[0] == _BOS_ID else full_ids[:assistant_index]
    else:
        context = full_ids[user_index:assistant_index]
    if len(context) <= budget:
        return list(context)
    if budget <= 0:
        return []

    tail_budget = max(0, budget - 1)
    tail = list(context[-tail_budget:]) if tail_budget else []
    if context and context[0] == _USER_ID:
        if tail and tail[0] == _USER_ID:
            return tail[-budget:]
        return [_USER_ID] + tail
    return list(context[-budget:])


def _reasoning_body_chunks(
    *,
    body: list[int],
    prefix_without_think: list[int],
    max_tokens: int,
) -> list[list[int]] | None:
    """Chunk a <think> assistant body without dropping reasoning boundaries.

    Every reasoning chunk reopens <think>. Intermediate chunks intentionally do
    not close it; the final reasoning chunk carries the single </think>, final
    answer tokens, </assistant> and <eos>. Prediction targets are partitioned, not
    overlapped, so long reasoning remains causally labelled without duplication.
    """
    if not body or body[0] != _THINK_ID:
        return None
    try:
        end_think = body.index(_END_THINK_ID, 1)
    except ValueError:
        return None

    reasoning = body[1:end_think]
    answer_tail = body[end_think + 1 :]
    prefix = prefix_without_think + [_THINK_ID]
    final_suffix = [_END_THINK_ID] + answer_tail + [_END_ASSISTANT_ID, _EOS_ID]
    intermediate_capacity = max_tokens - len(prefix)
    final_reasoning_capacity = max_tokens - len(prefix) - len(final_suffix)
    if intermediate_capacity < 1 or final_reasoning_capacity < 0:
        return None

    chunks: list[list[int]] = []
    cursor = 0
    while len(reasoning) - cursor > final_reasoning_capacity:
        remaining = len(reasoning) - cursor
        take = min(intermediate_capacity, remaining - final_reasoning_capacity)
        if take <= 0:
            raise RuntimeError("long_reasoning_chunk_progress_failed")
        chunks.append(prefix + reasoning[cursor : cursor + take])
        cursor += take
    chunks.append(prefix + reasoning[cursor:] + final_suffix)
    return chunks


def _chat_long_chunks(full_ids: list[int], *, seq_len: int) -> list[list[int]]:
    """Chunk long chats by assistant turn while preserving role-aware supervision."""
    max_tokens = seq_len + 1
    chunks: list[list[int]] = []
    cursor = 0
    while cursor < len(full_ids):
        try:
            assistant_index = full_ids.index(_ASSISTANT_ID, cursor)
        except ValueError:
            break
        try:
            end_index = full_ids.index(_END_ASSISTANT_ID, assistant_index + 1)
        except ValueError:
            raise RuntimeError("long_chat_missing_endassistant")

        body = full_ids[assistant_index + 1 : end_index]
        context_budget = max(0, max_tokens - 7)
        context = _current_user_context(full_ids, assistant_index, budget=context_budget)
        prefix = [_BOS_ID] + context + [_ASSISTANT_ID]
        if len(prefix) >= max_tokens - 1:
            raise RuntimeError("long_chat_prompt_context_exhausts_sequence")

        reasoning_chunks = _reasoning_body_chunks(
            body=body,
            prefix_without_think=prefix,
            max_tokens=max_tokens,
        )
        if reasoning_chunks is not None:
            chunks.extend(reasoning_chunks)
            cursor = end_index + 1
            continue

        intermediate_capacity = max_tokens - len(prefix)
        final_capacity = max_tokens - len(prefix) - 2
        if final_capacity < 1:
            raise RuntimeError("long_chat_has_no_assistant_capacity")

        body_cursor = 0
        if not body:
            chunks.append(prefix + [_END_ASSISTANT_ID, _EOS_ID])
        while body_cursor < len(body):
            remaining = len(body) - body_cursor
            if remaining <= final_capacity:
                chunk = prefix + body[body_cursor:] + [_END_ASSISTANT_ID, _EOS_ID]
                body_cursor = len(body)
            else:
                take = min(intermediate_capacity, remaining - final_capacity)
                if take <= 0:
                    raise RuntimeError("long_chat_chunk_progress_failed")
                chunk = prefix + body[body_cursor : body_cursor + take]
                body_cursor += take
            chunks.append(chunk)

        cursor = end_index + 1

    if not chunks:
        raise RuntimeError("long_chat_contains_no_assistant_turn")
    if any(len(chunk) < 2 or len(chunk) > max_tokens for chunk in chunks):
        raise RuntimeError("contextual_long_chunk_length_invalid")
    return chunks


def _contextual_long_chunks(
    text: str, tokenizer: BPETokenizer, *, seq_len: int,
) -> list[list[int]]:
    max_tokens = seq_len + 1
    full_ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    if len(full_ids) <= max_tokens:
        return [full_ids]
    if _ASSISTANT_ID in full_ids:
        return _chat_long_chunks(full_ids, seq_len=seq_len)
    return _content_windows(full_ids, seq_len)


def _drop_zero_supervision_sequences(
    sequences: list[list[int]], *, tokenizer: BPETokenizer, seq_len: int,
) -> list[list[int]]:
    supervised: list[list[int]] = []
    for sequence in sequences:
        _, _, stats = build_loss_targets(sequence, seq_len=seq_len, pad_id=tokenizer.pad_id())
        if stats.prediction_tokens > 0:
            supervised.append(sequence)
    return supervised


def build_example_sequences(texts: list[str], tokenizer: BPETokenizer, *, seq_len: int) -> list[list[int]]:
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
    sequences = _drop_zero_supervision_sequences(sequences, tokenizer=tokenizer, seq_len=seq_len)
    if not sequences:
        raise RuntimeError("packing_produced_no_supervised_sequences")
    if any(len(sequence) < 2 or len(sequence) > max_tokens for sequence in sequences):
        raise RuntimeError("invalid_packed_sequence_length")
    return sequences


class PackedSequenceDataset(Dataset):
    """Packed examples with the authoritative role-aware training objective."""
    def __init__(self, sequences: list[list[int]], seq_len: int, pad_id: int):
        if not sequences:
            raise ValueError("sequences must not be empty")
        self.seq_len = int(seq_len)
        self.pad_id = int(pad_id)
        self.sequences = [list(sequence) for sequence in sequences]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y, stats = build_loss_targets(self.sequences[idx], seq_len=self.seq_len, pad_id=self.pad_id)
        if stats.prediction_tokens <= 0:
            raise RuntimeError("packed_dataset_zero_supervision_sequence")
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def prediction_token_count(sequences: list[list[int]], *, seq_len: int, pad_id: int) -> int:
    total = 0
    for sequence in sequences:
        _, _, stats = build_loss_targets(sequence, seq_len=seq_len, pad_id=pad_id)
        total += stats.prediction_tokens
    return total


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
        "The market is bullish.", "one  two\n\nthree", "Walae Mkuu Mtaji",
        "KES 50,000 | XAUUSD @ 2,431.75", "€ £ ¥ KSh α β Σ ∂ 你好 مرحبا",
        r"\mathrm{ATR} \frac{1}{2}", "<user>What is RSI?</user>",
        "<assistant><think>Calculate.</think>Answer.</assistant>",
    ]
    fixed.extend(text[:2000] for text in train_texts[:8])
    return fixed


def validate_tokenizer_contract(tokenizer: BPETokenizer, train_texts: list[str]) -> int:
    if tokenizer.algorithm_version != TOKENIZER_ALGORITHM_VERSION:
        raise RuntimeError("non_authoritative_tokenizer_version")
    cases = _roundtrip_samples(train_texts)
    for sample in cases:
        ids = tokenizer.encode(sample, add_bos=False, add_eos=False)
        if tokenizer.unk_id() in ids:
            raise RuntimeError("tokenizer_emitted_unk_for_valid_utf8")
        decoded = tokenizer.decode(ids, skip_special=False)
        if decoded != sample:
            raise RuntimeError(f"tokenizer_roundtrip_mismatch:{sample[:80]!r}:{decoded[:80]!r}")
    return len(cases)


def validate_sequence_contract(train_sequences: list[list[int]], val_sequences: list[list[int]], *, seq_len: int, vocab_size: int) -> None:
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
    tiny_sequences = [sequence[: seq_len + 1] for sequence in train_sequences[: min(4, len(train_sequences))]]
    tiny_sequences = _drop_zero_supervision_sequences(tiny_sequences, tokenizer=tokenizer, seq_len=seq_len)
    if not tiny_sequences:
        raise RuntimeError("tiny_overfit_has_no_supervised_sequences")
    dataset = PackedSequenceDataset(tiny_sequences, seq_len, tokenizer.pad_id())
    loader = torch.utils.data.DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=False)
    model = VistaReasoningGPT(
        vocab_size=tokenizer.vocab_size, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
        ffn_dim=192, max_seq_len=seq_len, dropout=0.0, ffn_type="dense", num_experts=1,
        experts_per_token=1, moe_ffn_dim=192, shared_expert_ffn_dim=0,
        router_aux_loss_coef=0.0, router_jitter=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3)
    first_loss: float | None = None
    last_loss: float | None = None
    for _ in range(80):
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, targets=y, pad_id=tokenizer.pad_id())
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("tiny_overfit_loss_invalid")
            if first_loss is None:
                first_loss = float(loss.item())
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
    if first_loss is None or last_loss is None:
        raise RuntimeError("tiny_overfit_produced_no_loss")
    if not last_loss < first_loss * 0.35:
        raise RuntimeError(f"tiny_overfit_gate_failed:{first_loss:.4f}->{last_loss:.4f}")
    return first_loss, last_loss


def run_training_preflight(
    *,
    tokenizer: BPETokenizer,
    train_texts: list[str],
    val_texts: list[str],
    train_sequences: list[list[int]],
    val_sequences: list[list[int]],
    seq_len: int,
    conditioning_diagnostics_dir: str | Path = "conditioning_diagnostics",
) -> TrainingPreflightReport:
    from src.language.semantic_audit import audit_training_semantics
    roundtrip_cases = validate_tokenizer_contract(tokenizer, train_texts)
    validate_sequence_contract(train_sequences, val_sequences, seq_len=seq_len, vocab_size=tokenizer.vocab_size)
    semantic_report, _ = audit_training_semantics(texts=train_texts, sequences=train_sequences, tokenizer=tokenizer, seq_len=seq_len)
    print(
        f"[SemanticAudit] PASS examples={semantic_report.examples} documents={semantic_report.documents} "
        f"conversations={semantic_report.conversations} sequences={semantic_report.sequences} "
        f"supervision={semantic_report.mean_sequence_supervision_ratio:.1%} "
        f"zero_target={semantic_report.zero_supervision_sequences}"
    )
    train_prediction_tokens = prediction_token_count(train_sequences, seq_len=seq_len, pad_id=tokenizer.pad_id())
    validation_prediction_tokens = prediction_token_count(val_sequences, seq_len=seq_len, pad_id=tokenizer.pad_id())
    if train_prediction_tokens <= 0 or validation_prediction_tokens <= 0:
        raise RuntimeError("preflight_has_no_prediction_tokens")
    initial_loss, final_loss = run_tiny_overfit_gate(tokenizer, train_sequences)
    recall_1, recall_8, recall_32 = run_prompt_conditioning_gate(tokenizer, diagnostics_dir=conditioning_diagnostics_dir)
    return TrainingPreflightReport(
        train_examples=len(train_texts), validation_examples=len(val_texts),
        train_sequences=len(train_sequences), validation_sequences=len(val_sequences),
        train_prediction_tokens=train_prediction_tokens, validation_prediction_tokens=validation_prediction_tokens,
        tokenizer_vocab_size=tokenizer.vocab_size, tokenizer_algorithm_version=tokenizer.algorithm_version,
        roundtrip_cases=roundtrip_cases, overfit_initial_loss=initial_loss, overfit_final_loss=final_loss,
        conditioning_recall_1=recall_1, conditioning_recall_8=recall_8, conditioning_recall_32=recall_32,
    )
