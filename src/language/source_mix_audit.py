from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.language.loss_objective import build_loss_targets
from src.language.streaming_sources import HFSourceSpec, build_training_stream
from src.language.tokenizer import BPETokenizer


@dataclass(frozen=True, slots=True)
class SourceTokenStat:
    source_id: str
    examples: int
    supervised_tokens: int
    percent: float


def _short_source_name(source_id: str) -> str:
    value = source_id
    if "primitive_arithmetic" in value:
        return "PrimitiveArithmetic"
    if "foundation_economics" in value:
        return "FoundationEconomics"
    for marker, label in (
        ("roneneldan/TinyStories", "TinyStories"),
        ("tiiuae/falcon-refinedweb", "FalconRefinedWeb"),
        ("HuggingFaceTB/everyday-conversations", "EverydayConversations"),
    ):
        if marker in value:
            return label
    return value


def audit_supervised_source_mix(
    *,
    specs: tuple[HFSourceSpec, ...],
    stage: str,
    tokenizer: BPETokenizer,
    seq_len: int,
    seed: int,
    max_examples: int = 512,
    excluded_texts: tuple[str, ...] = (),
) -> tuple[SourceTokenStat, ...]:
    """Measure actual role-aware prediction tokens contributed by each source.

    This deliberately counts supervised target tokens, not configured source
    weights and not keyword classifications. It uses the same authoritative
    stream and loss-target constructor as training. Each example is measured
    independently so attribution remains unambiguous even though the runtime
    packer may combine short examples into one sequence.
    """
    if max_examples <= 0:
        raise ValueError("source_mix_audit_max_examples_must_be_positive")

    stream = build_training_stream(
        specs=specs,
        stage=stage,
        seed=seed,
        excluded_texts=excluded_texts,
        repeat=True,
    )
    examples: dict[str, int] = defaultdict(int)
    tokens: dict[str, int] = defaultdict(int)

    for index, (source_id, text) in enumerate(stream.iter_with_source()):
        if index >= max_examples:
            break
        ids = tokenizer.encode(text, add_bos=False, add_eos=False)
        if len(ids) < 2:
            continue
        # Long examples are chunked without cross-source packing. Counting each
        # contiguous window gives the same role-aware target semantics while
        # retaining exact source ownership.
        start = 0
        source_tokens = 0
        while start < len(ids) - 1:
            chunk = ids[start : start + seq_len + 1]
            if len(chunk) < 2:
                break
            _, _, stats = build_loss_targets(
                chunk,
                seq_len=seq_len,
                pad_id=tokenizer.pad_id(),
            )
            source_tokens += int(stats.prediction_tokens)
            if start + seq_len + 1 >= len(ids):
                break
            start += seq_len
        name = _short_source_name(source_id)
        examples[name] += 1
        tokens[name] += source_tokens

    total = sum(tokens.values())
    if total <= 0:
        raise RuntimeError("source_mix_audit_has_no_supervised_tokens")

    return tuple(
        SourceTokenStat(
            source_id=name,
            examples=examples[name],
            supervised_tokens=tokens[name],
            percent=100.0 * tokens[name] / total,
        )
        for name in sorted(tokens, key=lambda key: (-tokens[key], key))
    )


def format_supervised_source_mix(stats: tuple[SourceTokenStat, ...]) -> str:
    if not stats:
        raise ValueError("source_mix_stats_empty")
    return "[GradientMix] " + " ".join(
        f"{row.source_id}={row.percent:.1f}%({row.supervised_tokens:,}t/{row.examples}e)"
        for row in stats
    )
