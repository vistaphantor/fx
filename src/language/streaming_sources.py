from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Sequence

from corpus.dedup import NearDuplicateIndex
from corpus.quality import FOUNDATION_ENGLISH_FILTER, LANGUAGE_QUALITY_FILTER, QualityFilter
from corpus.source import DatasetSource, HFSource
from corpus.streamer import WeightedSourceStream
from src.language.canonical_contract import (
    CanonicalMessage,
    canonical_hash,
    canonicalize_serialized,
    prompt_family,
    serialize_messages,
)
from src.language.curriculum import is_math_example, is_reasoning_example, is_trading_example


# Foundation must learn two different skills at the same time:
#   1. ordinary causal English from documents;
#   2. that a <user> turn followed by <assistant> requires a useful continuation.
# A finite dialogue dataset alone cannot sustain the second lane for a long stream,
# so a deterministic fraction of real documents becomes a self-supervised
# continuation task. The target text is always genuine corpus text, never an
# invented answer.
FOUNDATION_INTERACTION_FRACTION = 0.25
FOUNDATION_CONTINUATION_PREFIX_WORDS = 32
FOUNDATION_CONTINUATION_TARGET_WORDS = 48


@dataclass(frozen=True, slots=True)
class HFSourceSpec:
    path: str
    weight: float
    stages: tuple[str, ...]
    split: str = "train"
    config_name: str | None = None
    revision: str | None = None
    text_fields: tuple[str, ...] | None = None
    prompt_field: str | None = None
    response_field: str | None = None
    max_examples: int | None = None
    shuffle_buffer_size: int = 10_000


def stream_quality_accepts(text: str) -> bool:
    return LANGUAGE_QUALITY_FILTER.accepts(text)


def _quality_filter_for_stage(stage: str) -> QualityFilter:
    normalized = stage.strip().casefold()
    if normalized == "foundation":
        return FOUNDATION_ENGLISH_FILTER
    return LANGUAGE_QUALITY_FILTER


def _is_chat(text: str) -> bool:
    return "<user>" in text and "<assistant>" in text


def _document_payload(text: str) -> str:
    value = canonicalize_serialized(text)
    if not value:
        return ""
    if value.startswith("<bos>"):
        value = value[len("<bos>"):].lstrip()
    if value.endswith("<eos>"):
        value = value[:-len("<eos>")].rstrip()
    return value.strip()


def _continuation_interaction(text: str) -> str | None:
    """Turn a real document into a bounded self-supervised assistant task.

    This is not synthetic knowledge. The user receives a genuine prefix from the
    document and the assistant target is the genuine text that follows it. The
    bounded word counts keep the full interaction comfortably inside the 192-token
    foundation context for the current byte-BPE profiles.
    """
    if _is_chat(text):
        return None
    payload = _document_payload(text)
    words = payload.split()
    minimum = FOUNDATION_CONTINUATION_PREFIX_WORDS + 12
    if len(words) < minimum:
        return None

    prefix_end = min(FOUNDATION_CONTINUATION_PREFIX_WORDS, len(words) - 12)
    target_end = min(
        len(words),
        prefix_end + FOUNDATION_CONTINUATION_TARGET_WORDS,
    )
    prefix = " ".join(words[:prefix_end]).strip()
    continuation = " ".join(words[prefix_end:target_end]).strip()
    if not prefix or not continuation:
        return None

    return serialize_messages((
        CanonicalMessage(
            "user",
            "Continue this passage naturally:\n" + prefix,
        ),
        CanonicalMessage("assistant", continuation),
    )) or None


def _use_foundation_interaction(digest: str) -> bool:
    # Hash selection is deterministic across machines/runs and does not touch the
    # trainer RNG. That keeps validation/preflight reproducible.
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < FOUNDATION_INTERACTION_FRACTION


class GuardedSource(DatasetSource):
    """Canonicalize, stage-filter, near-deduplicate and enforce holdouts."""

    def __init__(
        self,
        source: DatasetSource,
        *,
        stage: str,
        excluded_hashes: frozenset[str] = frozenset(),
        excluded_families: frozenset[str] = frozenset(),
        near_dedup_entries: int = 50_000,
        near_dedup_hamming: int = 4,
        near_index: NearDuplicateIndex | None = None,
    ):
        self._source = source
        self._stage = stage.strip().casefold()
        self._excluded_hashes = excluded_hashes
        self._excluded_families = excluded_families
        self._near_dedup_entries = int(near_dedup_entries)
        self._near_dedup_hamming = int(near_dedup_hamming)
        self._near_index = near_index
        self._quality_filter = _quality_filter_for_stage(self._stage)

    @property
    def source_id(self) -> str:
        return f"guarded:{self._source.source_id}:{self._stage}"

    def scan(self):
        return self._source.scan()

    def metadata(self) -> dict:
        return {
            **self._source.metadata(),
            "guarded_stage": self._stage,
            "quality_filter": type(self._quality_filter).__name__,
            "near_dedup_entries": self._near_dedup_entries,
            "near_dedup_hamming": self._near_dedup_hamming,
            "shared_near_dedup": self._near_index is not None,
            "foundation_interaction_fraction": (
                FOUNDATION_INTERACTION_FRACTION if self._stage == "foundation" else 0.0
            ),
        }

    def _stage_accepts(self, text: str) -> bool:
        if self._stage == "foundation":
            return True
        if self._stage == "reasoning":
            return is_reasoning_example(text) or is_math_example(text)
        if self._stage == "trading_reasoning":
            return is_trading_example(text)
        raise ValueError(f"unsupported_training_stage:{self._stage}")

    def stream(self) -> Iterator[str]:
        seen: set[str] = set()
        near = self._near_index or NearDuplicateIndex(
            max_entries=self._near_dedup_entries,
            max_hamming_distance=self._near_dedup_hamming,
        )
        for raw in self._source.stream():
            text = canonicalize_serialized(raw)
            if not text or not self._quality_filter.accepts(text):
                continue
            digest = canonical_hash(text)
            if digest in seen or digest in self._excluded_hashes:
                continue
            family = prompt_family(text)
            if family and family in self._excluded_families:
                continue
            if not self._stage_accepts(text):
                continue
            if not near.accept(text):
                continue
            seen.add(digest)

            if (
                self._stage == "foundation"
                and not _is_chat(text)
                and _use_foundation_interaction(digest)
            ):
                interaction = _continuation_interaction(text)
                if interaction:
                    yield interaction
                    continue
            yield text


def _parse_spec(payload: dict) -> HFSourceSpec:
    path = str(payload.get("path", "")).strip()
    if not path:
        raise ValueError("hf_source_missing_path")
    weight = float(payload.get("weight", 1.0))
    if weight <= 0:
        raise ValueError(f"hf_source_weight_must_be_positive:{path}")
    raw_stages = payload.get("stages", ["foundation", "reasoning", "trading_reasoning"])
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError(f"hf_source_stages_must_be_nonempty_list:{path}")
    stages = tuple(str(value).strip().casefold() for value in raw_stages if str(value).strip())
    allowed = {"foundation", "reasoning", "trading_reasoning"}
    invalid = set(stages).difference(allowed)
    if invalid:
        raise ValueError(f"hf_source_invalid_stages:{path}:{','.join(sorted(invalid))}")

    text_fields_payload = payload.get("text_fields")
    text_fields = None
    if text_fields_payload is not None:
        if not isinstance(text_fields_payload, list) or not text_fields_payload:
            raise ValueError(f"hf_source_text_fields_invalid:{path}")
        text_fields = tuple(str(value) for value in text_fields_payload)

    prompt_field = str(payload["prompt_field"]).strip() if payload.get("prompt_field") else None
    response_field = str(payload["response_field"]).strip() if payload.get("response_field") else None
    if bool(prompt_field) != bool(response_field):
        raise ValueError(f"hf_source_prompt_response_fields_must_be_paired:{path}")
    if text_fields and prompt_field:
        raise ValueError(f"hf_source_document_and_chat_mapping_conflict:{path}")

    max_examples = payload.get("max_examples")
    if max_examples is not None:
        max_examples = int(max_examples)
        if max_examples <= 0:
            raise ValueError(f"hf_source_max_examples_invalid:{path}")
    buffer_size = int(payload.get("shuffle_buffer_size", 10_000))
    if buffer_size < 0:
        raise ValueError(f"hf_source_shuffle_buffer_invalid:{path}")
    revision = str(payload["revision"]).strip() if payload.get("revision") else None
    if not revision:
        raise ValueError(f"hf_source_revision_required:{path}")
    return HFSourceSpec(
        path=path,
        weight=weight,
        stages=stages,
        split=str(payload.get("split", "train")),
        config_name=(str(payload["config_name"]) if payload.get("config_name") else None),
        revision=revision,
        text_fields=text_fields,
        prompt_field=prompt_field,
        response_field=response_field,
        max_examples=max_examples,
        shuffle_buffer_size=buffer_size,
    )


def load_hf_source_config(path: str | Path) -> tuple[HFSourceSpec, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_sources = payload.get("sources", []) if isinstance(payload, dict) else payload if isinstance(payload, list) else None
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("hf_config_contains_no_sources")
    specs = tuple(_parse_spec(item) for item in raw_sources if isinstance(item, dict))
    if not specs:
        raise ValueError("hf_config_contains_no_valid_sources")
    return specs


def specs_fingerprint(specs: Sequence[HFSourceSpec]) -> str:
    payload = {
        "foundation_interaction_fraction": FOUNDATION_INTERACTION_FRACTION,
        "foundation_continuation_prefix_words": FOUNDATION_CONTINUATION_PREFIX_WORDS,
        "foundation_continuation_target_words": FOUNDATION_CONTINUATION_TARGET_WORDS,
        "sources": [
            {
                "path": spec.path,
                "weight": spec.weight,
                "stages": list(spec.stages),
                "split": spec.split,
                "config_name": spec.config_name,
                "revision": spec.revision,
                "text_fields": list(spec.text_fields) if spec.text_fields else None,
                "prompt_field": spec.prompt_field,
                "response_field": spec.response_field,
                "max_examples": spec.max_examples,
                "shuffle_buffer_size": spec.shuffle_buffer_size,
            }
            for spec in specs
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hf_source_from_spec(spec: HFSourceSpec, *, seed: int) -> HFSource:
    return HFSource(
        path=spec.path,
        split=spec.split,
        text_fields=list(spec.text_fields) if spec.text_fields else None,
        prompt_field=spec.prompt_field,
        response_field=spec.response_field,
        max_examples=spec.max_examples,
        config_name=spec.config_name,
        revision=spec.revision,
        shuffle_buffer_size=spec.shuffle_buffer_size,
        seed=seed,
    )


def stage_specs(specs: Sequence[HFSourceSpec], stage: str) -> tuple[HFSourceSpec, ...]:
    normalized = stage.strip().casefold()
    selected = tuple(spec for spec in specs if normalized in spec.stages)
    if not selected:
        raise RuntimeError(f"no_hf_sources_for_training_stage:{normalized}")
    return selected


def build_training_stream(
    *,
    specs: Sequence[HFSourceSpec],
    stage: str,
    seed: int,
    local_replay: Sequence[str] = (),
    local_weight: float = 0.0,
    excluded_texts: Sequence[str] = (),
    repeat: bool = True,
) -> WeightedSourceStream:
    if local_replay or local_weight > 0:
        raise RuntimeError(
            "local_language_training_disabled:configure a pinned streaming source instead"
        )

    selected = stage_specs(specs, stage)
    excluded_hashes = frozenset(canonical_hash(text) for text in excluded_texts if text and text.strip())
    excluded_families = frozenset(prompt_family(text) for text in excluded_texts if text and text.strip())
    sources: list[tuple[DatasetSource, float]] = []

    hf_near_index = NearDuplicateIndex(max_entries=50_000, max_hamming_distance=4)
    for index, spec in enumerate(selected):
        sources.append((
            GuardedSource(
                hf_source_from_spec(spec, seed=seed + index * 997),
                stage=stage,
                excluded_hashes=excluded_hashes,
                excluded_families=excluded_families,
                near_index=hf_near_index,
            ),
            spec.weight,
        ))
    return WeightedSourceStream(sources, seed=seed, repeat=repeat)


def sample_training_stream(
    *, specs: Sequence[HFSourceSpec], stage: str, limit: int, seed: int,
) -> list[str]:
    """Build a deterministic, network-light preflight sample.

    Preflight is for tokenizer/data-contract inspection, not stochastic training.
    Disabling per-source HF shuffle here prevents the datasets library from
    scattering a small sample across thousands of remote parquet shards.
    WeightedSourceStream still mixes the configured sources according to their
    weights, while each individual source is consumed sequentially.
    """
    if limit <= 0:
        raise ValueError("stream sample limit must be positive")

    selected = stage_specs(specs, stage)
    preflight_specs = tuple(
        replace(spec, shuffle_buffer_size=0)
        for spec in selected
    )
    stream = build_training_stream(
        specs=preflight_specs,
        stage=stage,
        seed=seed,
        local_replay=(),
        local_weight=0.0,
        repeat=False,
    )
    result: list[str] = []
    for text in stream:
        result.append(text)
        if len(result) >= limit:
            break
    if not result:
        raise RuntimeError("hf_stream_sample_produced_no_examples")
    return result
