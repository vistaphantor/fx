"""Fail-closed semantic audit for Vista language-model training data.

The audit operates on the exact canonical examples and packed token sequences
that feed optimization. It verifies protocol shape, tokenizer round-tripping,
packing boundaries and role-aware target masks. It can also persist a compact
human-readable artifact showing what the model sees and what it is asked to
predict.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Sequence

from src.language.canonical_contract import STRUCTURAL_TOKENS, canonicalize_serialized
from src.language.loss_objective import build_loss_targets
from src.language.tokenizer import ASSISTANT, BOS, EOS, USER, BPETokenizer, SPECIAL_TOKENS

SEMANTIC_AUDIT_VERSION = 2

_BOS_ID = SPECIAL_TOKENS.index(BOS)
_EOS_ID = SPECIAL_TOKENS.index(EOS)
_USER_ID = SPECIAL_TOKENS.index(USER)
_ASSISTANT_ID = SPECIAL_TOKENS.index(ASSISTANT)


@dataclass(frozen=True, slots=True)
class SemanticAuditReport:
    version: int
    examples: int
    documents: int
    conversations: int
    sequences: int
    prediction_tokens: int
    masked_prompt_tokens: int
    masked_boundary_tokens: int
    zero_supervision_sequences: int
    mean_sequence_supervision_ratio: float
    min_sequence_supervision_ratio: float
    max_sequence_supervision_ratio: float
    structural_token_counts: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


def _kind_and_validate_text(text: str, *, index: int) -> str:
    if canonicalize_serialized(text) != text:
        raise RuntimeError(f"semantic_audit_noncanonical_example:{index}")
    lines = text.splitlines()
    if not lines or lines[0] != BOS or lines[-1] != EOS:
        raise RuntimeError(f"semantic_audit_missing_bos_eos:{index}")
    if text.count(BOS) != 1 or text.count(EOS) != 1:
        raise RuntimeError(f"semantic_audit_nested_example_boundary:{index}")

    role_pairs = (
        ("<user>", "</user>"),
        ("<assistant>", "</assistant>"),
        ("<evidence>", "</evidence>"),
        ("<think>", "</think>"),
        ("<market>", "</market>"),
        ("<account>", "</account>"),
        ("<position>", "</position>"),
        ("<hypothesis>", "</hypothesis>"),
        ("<countercase>", "</countercase>"),
        ("<tool>", "</tool>"),
        ("<tool_result>", "</tool_result>"),
        ("<decision>", "</decision>"),
        ("<confidence>", "</confidence>"),
        ("<invalidation>", "</invalidation>"),
    )
    for opening, closing in role_pairs:
        if text.count(opening) != text.count(closing):
            raise RuntimeError(
                f"semantic_audit_unbalanced_control_token:{index}:{opening}"
            )

    has_user = "<user>" in text
    has_assistant = "<assistant>" in text
    if has_user != has_assistant:
        raise RuntimeError(f"semantic_audit_incomplete_conversation:{index}")
    return "conversation" if has_assistant else "document"


def _token_label(tokenizer: BPETokenizer, token_id: int) -> str:
    token = tokenizer.id_to_token.get(int(token_id), "<missing>")
    return token.replace("\n", "\\n")


def audit_training_semantics(
    *,
    texts: Sequence[str],
    sequences: Sequence[Sequence[int]],
    tokenizer: BPETokenizer,
    seq_len: int,
    max_examples: int = 64,
    max_sequences: int = 64,
) -> tuple[SemanticAuditReport, str]:
    """Validate exact training semantics and return a human-readable preview."""
    selected_texts = [text for text in texts if text and text.strip()][: max(1, int(max_examples))]
    selected_sequences = [list(sequence) for sequence in sequences][: max(1, int(max_sequences))]
    if not selected_texts or not selected_sequences:
        raise RuntimeError("semantic_audit_has_no_training_data")

    documents = 0
    conversations = 0
    structural_counts: Counter[str] = Counter()
    rendered: list[str] = [
        "Vista Language Training Semantic Audit",
        f"Version: {SEMANTIC_AUDIT_VERSION}",
        f"Sequence length: {seq_len}",
        f"Examples inspected: {len(selected_texts)}",
        f"Packed sequences inspected: {len(selected_sequences)}",
        "",
        "EXAMPLE PREVIEW",
    ]

    for index, text in enumerate(selected_texts, start=1):
        kind = _kind_and_validate_text(text, index=index)
        documents += int(kind == "document")
        conversations += int(kind == "conversation")
        ids = tokenizer.encode(text, add_bos=False, add_eos=False)
        if tokenizer.unk_id() in ids:
            raise RuntimeError(f"semantic_audit_unk_token:{index}")
        decoded = tokenizer.decode(ids, skip_special=False)
        if decoded != text:
            raise RuntimeError(f"semantic_audit_roundtrip_mismatch:{index}")
        for token in STRUCTURAL_TOKENS:
            count = text.count(token)
            if count:
                structural_counts[token] += count
        rendered.extend([
            "-" * 96,
            f"EXAMPLE {index} kind={kind} chars={len(text)} tokens={len(ids)}",
            text[:2400],
        ])

    prediction_tokens = 0
    masked_prompt_tokens = 0
    masked_boundary_tokens = 0
    zero_supervision = 0
    ratios: list[float] = []
    rendered.extend(["", "PACKED TARGET-MASK PREVIEW"])

    for sequence_index, sequence in enumerate(selected_sequences, start=1):
        if len(sequence) < 2 or len(sequence) > seq_len + 1:
            raise RuntimeError(f"semantic_audit_sequence_length:{sequence_index}")
        _, y, stats = build_loss_targets(
            sequence,
            seq_len=seq_len,
            pad_id=tokenizer.pad_id(),
        )
        prediction_tokens += stats.prediction_tokens
        masked_prompt_tokens += stats.masked_prompt_tokens
        masked_boundary_tokens += stats.masked_boundary_tokens
        zero_supervision += int(stats.prediction_tokens <= 0)
        active_targets = len(sequence) - 1
        ratios.append(stats.prediction_tokens / max(active_targets, 1))

        if stats.prediction_tokens <= 0:
            raise RuntimeError(f"semantic_audit_zero_supervision_sequence:{sequence_index}")

        # Opening grammar tokens are context, never desired model output. This
        # catches packed-boundary and role-mask regressions immediately.
        for target_position in range(1, len(sequence)):
            token_id = sequence[target_position]
            y_index = target_position - 1
            if token_id in {_BOS_ID, _USER_ID, _ASSISTANT_ID} and y[y_index] != tokenizer.pad_id():
                raise RuntimeError(
                    f"semantic_audit_control_token_supervised:{sequence_index}:"
                    f"{_token_label(tokenizer, token_id)}"
                )
        if any(target == _BOS_ID for target in y if target != tokenizer.pad_id()):
            raise RuntimeError(f"semantic_audit_bos_supervised:{sequence_index}")

        preview: list[str] = []
        for pos in range(min(len(sequence) - 1, 48)):
            target_id = sequence[pos + 1]
            mode = "MASK" if y[pos] == tokenizer.pad_id() else "LEARN"
            preview.append(f"{pos:03d}:{mode}:{_token_label(tokenizer, target_id)}")
        rendered.extend([
            "-" * 96,
            f"SEQUENCE {sequence_index} raw_tokens={len(sequence)} "
            f"prediction_tokens={stats.prediction_tokens} "
            f"masked_prompt={stats.masked_prompt_tokens} "
            f"masked_boundary={stats.masked_boundary_tokens}",
            " | ".join(preview),
        ])

    report = SemanticAuditReport(
        version=SEMANTIC_AUDIT_VERSION,
        examples=len(selected_texts),
        documents=documents,
        conversations=conversations,
        sequences=len(selected_sequences),
        prediction_tokens=prediction_tokens,
        masked_prompt_tokens=masked_prompt_tokens,
        masked_boundary_tokens=masked_boundary_tokens,
        zero_supervision_sequences=zero_supervision,
        mean_sequence_supervision_ratio=mean(ratios),
        min_sequence_supervision_ratio=min(ratios),
        max_sequence_supervision_ratio=max(ratios),
        structural_token_counts=dict(sorted(structural_counts.items())),
    )
    return report, "\n".join(rendered) + "\n"


def save_semantic_audit(
    *, report: SemanticAuditReport, rendered: str,
    text_path: str | Path, json_path: str | Path,
) -> None:
    text_output = Path(text_path)
    json_output = Path(json_path)
    text_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    text_output.write_text(rendered, encoding="utf-8")
    json_output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
