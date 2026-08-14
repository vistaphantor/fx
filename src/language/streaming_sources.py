from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from corpus.source import DatasetSource, HFSource, SourceMetadata
from corpus.streamer import WeightedSourceStream


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
    """Replay already-canonical local examples inside the weighted stream."""

    def __init__(self, texts: Sequence[str], *, source_name: str = "local_replay"):
        self._texts = tuple(text for text in texts if text and text.strip())
        self._source_name = source_name

    @property
    def source_id(self) -> str:
        return f"memory:{self._source_name}"

    def scan(self) -> SourceMetadata:
        return SourceMetadata(
            source_type="memory",
            path=self.source_id,
            estimated_docs=len(self._texts),
        )

    def stream(self) -> Iterator[str]:
        yield from self._texts

    def metadata(self) -> dict:
        return {
            "source_type": "memory",
            "source_name": self._source_name,
            "examples": len(self._texts),
        }


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
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw_sources = payload.get("sources", [])
    elif isinstance(payload, list):
        raw_sources = payload
    else:
        raise ValueError("hf_config_must_be_list_or_sources_object")
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
            "revision": spec.revision or "main",
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
    repeat: bool = True,
) -> WeightedSourceStream:
    selected = stage_specs(specs, stage)
    sources: list[tuple[DatasetSource, float]] = [
        (_hf_source(spec, seed=seed + index * 997), spec.weight)
        for index, spec in enumerate(selected)
    ]
    if local_replay:
        if local_weight <= 0:
            raise ValueError("local_weight must be positive when local replay is enabled")
        sources.append((CanonicalMemorySource(local_replay), float(local_weight)))
    return WeightedSourceStream(sources, seed=seed, repeat=repeat)


def sample_training_stream(
    *,
    specs: Sequence[HFSourceSpec],
    stage: str,
    limit: int,
    seed: int,
) -> list[str]:
    if limit <= 0:
        raise ValueError("stream sample limit must be positive")
    stream = build_training_stream(
        specs=specs,
        stage=stage,
        seed=seed,
        local_replay=(),
        repeat=True,
    )
    result: list[str] = []
    seen: set[str] = set()
    for text in stream:
        normalized = text.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
        if len(result) >= limit:
            break
    if not result:
        raise RuntimeError("hf_stream_sample_produced_no_examples")
    return result
