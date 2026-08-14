from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from corpus.dedup import NearDuplicateIndex
from corpus.quality import LANGUAGE_QUALITY_FILTER
from corpus.source import DatasetSource, HFSource, SourceMetadata
from corpus.streamer import WeightedSourceStream
from src.language.canonical_contract import canonical_hash, canonicalize_serialized, prompt_family
from src.language.curriculum import is_math_example, is_reasoning_example, is_trading_example


@dataclass(frozen=True, slots=True)
class HFSourceSpec:
    path: str
    weight: float
    stages: tuple[str, ...]
    split: str = "train"
    config_name: str | None = None
    revision: str | None = None
    text_fields: tuple[str, ...] | None = None
    max_examples: int | None = None
    shuffle_buffer_size: int = 10_000


class CanonicalMemorySource(DatasetSource):
    def __init__(self, texts: Sequence[str], *, source_name: str = "local_replay"):
        self._texts = tuple(canonicalize_serialized(text) for text in texts if text and text.strip())
        self._source_name = source_name

    @property
    def source_id(self) -> str:
        return f"memory:{self._source_name}"

    def scan(self) -> SourceMetadata:
        return SourceMetadata(source_type="memory", path=self.source_id, estimated_docs=len(self._texts))

    def stream(self) -> Iterator[str]:
        yield from self._texts

    def metadata(self) -> dict:
        return {"source_type": "memory", "source_name": self._source_name, "examples": len(self._texts)}


def stream_quality_accepts(text: str) -> bool:
    """Compatibility name for the single authoritative corpus quality gate."""
    return LANGUAGE_QUALITY_FILTER.accepts(text)


class GuardedSource(DatasetSource):
    """Canonicalize, quality-filter, near-deduplicate and enforce holdouts.

    Raw HF sources are classified per example for the active specialist stage.
    Local replay has already been selected by `select_curriculum`, so it must
    not be stage-filtered again or the anti-forgetting replay is erased.
    """

    def __init__(
        self,
        source: DatasetSource,
        *,
        stage: str,
        excluded_hashes: frozenset[str] = frozenset(),
        excluded_families: frozenset[str] = frozenset(),
        enforce_stage: bool = True,
        near_dedup_entries: int = 50_000,
        near_dedup_hamming: int = 4,
    ):
        self._source = source
        self._stage = stage.strip().casefold()
        self._excluded_hashes = excluded_hashes
        self._excluded_families = excluded_families
        self._enforce_stage = bool(enforce_stage)
        self._near_dedup_entries = int(near_dedup_entries)
        self._near_dedup_hamming = int(near_dedup_hamming)

    @property
    def source_id(self) -> str:
        mode = "stage" if self._enforce_stage else "replay"
        return f"guarded:{self._source.source_id}:{self._stage}:{mode}"

    def scan(self) -> SourceMetadata:
        return self._source.scan()

    def metadata(self) -> dict:
        return {
            **self._source.metadata(),
            "guarded_stage": self._stage,
            "enforce_stage": self._enforce_stage,
            "near_dedup_entries": self._near_dedup_entries,
            "near_dedup_hamming": self._near_dedup_hamming,
        }

    def _stage_accepts(self, text: str) -> bool:
        if not self._enforce_stage:
            return True
        if self._stage == "foundation":
            return True
        if self._stage == "reasoning":
            return is_reasoning_example(text) or is_math_example(text)
        if self._stage == "trading_reasoning":
            return is_trading_example(text)
        raise ValueError(f"unsupported_training_stage:{self._stage}")

    def stream(self) -> Iterator[str]:
        seen: set[str] = set()
        near = NearDuplicateIndex(
            max_entries=self._near_dedup_entries,
            max_hamming_distance=self._near_dedup_hamming,
        )
        for raw in self._source.stream():
            text = canonicalize_serialized(raw)
            if not text or not LANGUAGE_QUALITY_FILTER.accepts(text):
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
    payload = [
        {
            "path": spec.path,
            "weight": spec.weight,
            "stages": list(spec.stages),
            "split": spec.split,
            "config_name": spec.config_name,
            "revision": spec.revision,
            "text_fields": list(spec.text_fields) if spec.text_fields else None,
            "max_examples": spec.max_examples,
            "shuffle_buffer_size": spec.shuffle_buffer_size,
        }
        for spec in specs
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hf_source(spec: HFSourceSpec, *, seed: int) -> HFSource:
    return HFSource(
        path=spec.path,
        split=spec.split,
        text_fields=list(spec.text_fields) if spec.text_fields else None,
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
    local_weight: float = 0.25,
    excluded_texts: Sequence[str] = (),
    repeat: bool = True,
) -> WeightedSourceStream:
    selected = stage_specs(specs, stage)
    excluded_hashes = frozenset(canonical_hash(text) for text in excluded_texts if text and text.strip())
    excluded_families = frozenset(prompt_family(text) for text in excluded_texts if text and text.strip())
    sources: list[tuple[DatasetSource, float]] = []
    for index, spec in enumerate(selected):
        sources.append((
            GuardedSource(
                _hf_source(spec, seed=seed + index * 997),
                stage=stage,
                excluded_hashes=excluded_hashes,
                excluded_families=excluded_families,
                enforce_stage=True,
            ),
            spec.weight,
        ))
    if local_replay:
        if local_weight <= 0:
            raise ValueError("local_weight must be positive when local replay is enabled")
        sources.append((
            GuardedSource(
                CanonicalMemorySource(local_replay),
                stage=stage,
                excluded_hashes=excluded_hashes,
                excluded_families=excluded_families,
                enforce_stage=False,
            ),
            float(local_weight),
        ))
    return WeightedSourceStream(sources, seed=seed, repeat=repeat)


def sample_training_stream(
    *, specs: Sequence[HFSourceSpec], stage: str, limit: int, seed: int,
) -> list[str]:
    if limit <= 0:
        raise ValueError("stream sample limit must be positive")
    stream = build_training_stream(specs=specs, stage=stage, seed=seed, local_replay=(), repeat=False)
    result: list[str] = []
    for text in stream:
        result.append(text)
        if len(result) >= limit:
            break
    if not result:
        raise RuntimeError("hf_stream_sample_produced_no_examples")
    return result
