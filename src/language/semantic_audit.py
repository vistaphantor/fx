"""Fail-closed semantic audit for the exact language-model training representation.

This module inspects canonical examples after source filtering and before model
optimization. It verifies that corpus text cannot mutate protocol grammar, that
tokenization is lossless, that sequence packing preserves boundaries, and that
the role-aware loss mask supervises exactly the intended targets.
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
from src.language.training_pipeline import build_example_sequences

SEMANTIC_AUDIT_VERSION = 1

_BOS_ID = SPECIAL_TOKENS.index(BOS)
_EOS_ID = SPECIAL_TOKENS.index(EOS)
_USER_ID = SPECIAL_TOKENS.index(USER)
_ASSISTANT_ID = SPECIAL_TOKENS.index(ASSISTANT)


@dataclass(frozen=True, slots=True)
class SemanticExampleSummary:
    index: int
    kind: str
    canonical_chars: int
    token_count: int
    sequence_count: int
    prediction_tokens: int
    masked_prompt_tokens: int
    masked_boundary_tokens: int
    supervision_ratio: float


@dataclass(frozen=True, slots=True)
class SemanticAuditReport:
    version: int
    examples: int
    documents: int
    conversations: int
    sequences: int
    prediction_tokens: int
    zero_supervision_sequences: int
    mean_supervision_ratio: float
    min_supervision_ratio: float
    max_supervision_ratio: float
    structural_token_counts: dict[str, int]
    example_summaries: tuple[SemanticExampleSummary, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["example_summaries"] = [asdict(item) for item in self.example_summaries]
        return payload


def _validate_structural_shape(text: str, *, index: int) -> str:
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


def _render_example(
    *,
    index: int,
    text: str,
    tokenizer: BPETokenizer,
    seq_len: int,
) -> tuple[SemanticExampleSummary, list[str], Counter[str]]:
    kind = _validate_structural_shape(text, index=index)
    token_ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    if tokenizer.unk_id() in token_ids:
        raise RuntimeError(f"semantic_audit_unk_token:{index}")
    decoded = tokenizer.decode(token_ids, skip_special=False)
    if decoded != text:
        raise RuntimeError(f"semantic_audit_roundtrip_mismatch:{index}")

    structural_counts: Counter[str] = Counter()
    for token in STRUCTURAL_TOKENS:
        count = text.count(token)
        if count:
            structural_counts[token] += count

    sequences = build_example_sequences([text], tokenizer, seq_len=seq_len)
    prediction_tokens = 0
    prompt_masked = 0
    boundary_masked = 0
    zero_supervision = 0
    sequence_lines: list[str] = []
    ratios: list[float] = []

    for sequence_index, sequence in enumerate(sequences, start=1):
        x, y, stats = build_loss_targets(
            sequence,
            seq_len=seq_len,
            pad_id=tokenizer.pad_id(),
        )
        prediction_tokens += stats.prediction_tokens
        prompt_masked += stats.masked_prompt_tokens
        boundary_masked += stats.masked_boundary_tokens
        if stats.prediction_tokens <= 0:
            zero_supervision += 1

        active_targets = max(0, len(sequence) - 1)
        ratio = stats.prediction_tokens / max(active_targets, 1)
        ratios.append(ratio)

        # Protocol controls that open a prompt/assistant turn must never be
        # learned as output targets. The caller provides them as context.
        for target_position in range(1, len(sequence)):
            token_id = sequence[target_position]
            y_index = target_position - 1
            if token_id in {_BOS_ID, _USER_ID, _ASSISTANT_ID} and y[y_index] != tokenizer.pad_id():
                raise RuntimeError(
                    f"semantic_audit_control_token_supervised:{index}:{sequence_index}:"
                    f"{_token_label(tokenizer, token_id)}"
                )

        if any(target == _BOS_ID for target in y if target != tokenizer.pad_id()):
            raise RuntimeError(f"semantic_audit_bos_supervised:{index}:{sequence_index}")

        preview: list[str] = []
        for pos in range(min(len(sequence) - 1, 48)):
            target_id = sequence[pos + 1]
            mode = "MASK" if y[pos] == tokenizer.pad_id() else "LEARN"
            preview.append(f"{pos:03d}:{mode}:{_token_label(tokenizer, target_id)}")
        sequence_lines.extend([
            f"SEQUENCE {sequence_index} raw_tokens={len(sequence)} "
            f"prediction_tokens={stats.prediction_tokens} "
            f"masked_prompt={stats.masked_prompt_tokens} "
            f"masked_boundary={stats.masked_boundary_tokens}",
            "TARGET MAP (first 48 targets):",
            " | ".join(preview),
        ])

    if prediction_tokens <= 0:
        raise RuntimeError(f"semantic_audit_example_has_no_supervision:{index}")

    summary = SemanticExampleSummary(
        index=index,
        kind=kind,
        canonical_chars=len(text),
        token_count=len(token_ids),
        sequence_count=len(sequences),
        prediction_tokens=prediction_tokens,
        masked_prompt_tokens=prompt_masked,
        masked_boundary_tokens=boundary_masked,
        supervision_ratio=mean(ratios) if ratios else 0.0,
    )
    rendered = [
        "=" * 96,
        f"EXAMPLE {index} kind={kind} chars={len(text)} tokens={len(token_ids)}",
        "CANONICAL TEXT",
        text,
        *sequence_lines,
    ]
    return summary, rendered, structural_counts


def run_semantic_audit(
    *,
    texts: Sequence[str],
    tokenizer: BPETokenizer,
    seq_len: int,
    text_path: str | Path,
    json_path: str | Path,
    max_examples: int = 48,
) -> SemanticAuditReport:
    """Audit and persist the exact canonical/token/loss semantics used in training."""
    selected = [text for text in texts if text and text.strip()][: max(1, int(max_examples))]
    if not selected:
        raise RuntimeError("semantic_audit_has_no_examples")

    summaries: list[SemanticExampleSummary] = []
    rendered: list[str] = [
        "Vista Language Training Semantic Audit",
        f"Version: {SEMANTIC_AUDIT_VERSION}",
        f"Sequence length: {seq_len}",
        f"Examples inspected: {len(selected)}",
        "",
    ]
    structural_counts: Counter[str] = Counter()
    zero_supervision_sequences = 0

    for index, text in enumerate(selected, start=1):
        summary, lines, counts = _render_example(
            index=index,
            text=text,
            tokenizer=tokenizer,
            seq_len=seq_len,
        )
        summaries.append(summary)
        rendered.extend(lines)
        structural_counts.update(counts)
        # A zero-target subwindow can exist for an exceptionally long prompt;
        # CorpusStreamer drops it before optimization. Record it visibly.
        sequences = build_example_sequences([text], tokenizer, seq_len=seq_len)
        for sequence in sequences:
            _, _, stats = build_loss_targets(
                sequence, seq_len=seq_len, pad_id=tokenizer.pad_id()
            )
            if stats.prediction_tokens <= 0:
                zero_supervision_sequences += 1

    ratios = [summary.supervision_ratio for summary in summaries]
    report = SemanticAuditReport(
        version=SEMANTIC_AUDIT_VERSION,
        examples=len(summaries),
        documents=sum(summary.kind == "document" for summary in summaries),
        conversations=sum(summary.kind == "conversation" for summary in summaries),
        sequences=sum(summary.sequence_count for summary in summaries),
        prediction_tokens=sum(summary.prediction_tokens for summary in summaries),
        zero_supervision_sequences=zero_supervision_sequences,
        mean_supervision_ratio=mean(ratios),
        min_supervision_ratio=min(ratios),
        max_supervision_ratio=max(ratios),
        structural_token_counts=dict(sorted(structural_counts.items())),
        example_summaries=tuple(summaries),
    )

    text_output = Path(text_path)
    json_output = Path(json_path)
    text_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    text_output.write_text("\n".join(rendered) + "\n", encoding="utf-8")
    json_output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report
