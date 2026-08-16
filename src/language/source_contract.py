from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.language.canonical_contract import CANONICAL_CONTRACT_VERSION
from src.language.foundation_contract import (
    FOUNDATION_CONTRACT_VERSION,
    FOUNDATION_MIN_AVAILABLE_CURRICULUM_TOKENS,
    FOUNDATION_SKILL_SET,
)

SOURCE_CONTRACT_VERSION = 2
_IMMUTABLE_HF_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")


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
    dialogue_field: str | None = None
    row_filters: tuple[tuple[str, tuple[str, ...]], ...] = ()
    tags: tuple[str, ...] = ()
    max_examples: int | None = None
    shuffle_buffer_size: int = 10_000
    available_tokens: int | None = None
    skills: tuple[str, ...] = ()

    def row_filter_dict(self) -> dict[str, tuple[str, ...]]:
        return {key: values for key, values in self.row_filters}


def _parse_row_filters(payload: object, *, path: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if payload is None:
        return ()
    if not isinstance(payload, dict):
        raise ValueError(f"hf_source_row_filters_must_be_object:{path}")
    parsed: list[tuple[str, tuple[str, ...]]] = []
    for raw_key, raw_values in payload.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError(f"hf_source_row_filter_field_invalid:{path}")
        values = [raw_values] if isinstance(raw_values, str) else raw_values
        if not isinstance(values, list) or not values:
            raise ValueError(f"hf_source_row_filter_values_invalid:{path}:{key}")
        cleaned = tuple(str(value).strip() for value in values if str(value).strip())
        if not cleaned:
            raise ValueError(f"hf_source_row_filter_values_invalid:{path}:{key}")
        parsed.append((key, cleaned))
    return tuple(sorted(parsed))


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
    allowed_stages = {"foundation", "reasoning", "trading_reasoning"}
    invalid_stages = set(stages).difference(allowed_stages)
    if invalid_stages:
        raise ValueError(f"hf_source_invalid_stages:{path}:{','.join(sorted(invalid_stages))}")

    text_fields_payload = payload.get("text_fields")
    text_fields = None
    if text_fields_payload is not None:
        if not isinstance(text_fields_payload, list) or not text_fields_payload:
            raise ValueError(f"hf_source_text_fields_invalid:{path}")
        text_fields = tuple(str(value) for value in text_fields_payload)

    prompt_field = str(payload["prompt_field"]).strip() if payload.get("prompt_field") else None
    response_field = str(payload["response_field"]).strip() if payload.get("response_field") else None
    dialogue_field = str(payload["dialogue_field"]).strip() if payload.get("dialogue_field") else None
    if bool(prompt_field) != bool(response_field):
        raise ValueError(f"hf_source_prompt_response_fields_must_be_paired:{path}")
    mapping_count = int(bool(text_fields)) + int(bool(prompt_field)) + int(bool(dialogue_field))
    if mapping_count > 1:
        raise ValueError(f"hf_source_mapping_conflict:{path}")

    raw_tags = payload.get("tags", [])
    if not isinstance(raw_tags, list):
        raise ValueError(f"hf_source_tags_must_be_list:{path}")
    tags = tuple(sorted({str(tag).strip().casefold() for tag in raw_tags if str(tag).strip()}))
    valid_tags = {"arithmetic", "economics", "language_quality", "conversation", "creativity"}
    invalid_tags = set(tags).difference(valid_tags)
    if invalid_tags:
        raise ValueError(f"hf_source_invalid_tags:{path}:{','.join(sorted(invalid_tags))}")

    revision = str(payload["revision"]).strip() if payload.get("revision") else None
    if not revision:
        raise ValueError(f"hf_source_revision_required:{path}")
    if not _IMMUTABLE_HF_REVISION_RE.fullmatch(revision):
        raise ValueError(f"hf_source_revision_must_be_immutable_commit:{path}:{revision}")

    max_examples = payload.get("max_examples")
    if max_examples is not None:
        max_examples = int(max_examples)
        if max_examples <= 0:
            raise ValueError(f"hf_source_max_examples_invalid:{path}")
    shuffle_buffer_size = int(payload.get("shuffle_buffer_size", 10_000))
    if shuffle_buffer_size < 0:
        raise ValueError(f"hf_source_shuffle_buffer_invalid:{path}")

    available_tokens = payload.get("available_tokens")
    if available_tokens is not None:
        available_tokens = int(available_tokens)
        if available_tokens <= 0:
            raise ValueError(f"hf_source_available_tokens_invalid:{path}")

    raw_skills = payload.get("skills", [])
    if not isinstance(raw_skills, list):
        raise ValueError(f"hf_source_skills_must_be_list:{path}")
    skills = tuple(sorted({str(value).strip().casefold() for value in raw_skills if str(value).strip()}))
    invalid_skills = set(skills).difference(FOUNDATION_SKILL_SET)
    if invalid_skills:
        raise ValueError(f"hf_source_invalid_skills:{path}:{','.join(sorted(invalid_skills))}")

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
        dialogue_field=dialogue_field,
        row_filters=_parse_row_filters(payload.get("row_filters"), path=path),
        tags=tags,
        max_examples=max_examples,
        shuffle_buffer_size=shuffle_buffer_size,
        available_tokens=available_tokens,
        skills=skills,
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


def stage_specs(specs: Sequence[HFSourceSpec], stage: str) -> tuple[HFSourceSpec, ...]:
    normalized = stage.strip().casefold()
    selected = tuple(spec for spec in specs if normalized in spec.stages)
    if not selected:
        raise RuntimeError(f"no_hf_sources_for_training_stage:{normalized}")
    return selected


@dataclass(frozen=True, slots=True)
class CurriculumCapacity:
    available_tokens: int
    skills: tuple[str, ...]
    source_count: int
    declared_token_sources: int


def curriculum_capacity(specs: Sequence[HFSourceSpec], stage: str) -> CurriculumCapacity:
    selected = stage_specs(specs, stage)
    return CurriculumCapacity(
        available_tokens=sum(spec.available_tokens or 0 for spec in selected),
        skills=tuple(sorted({skill for spec in selected for skill in spec.skills})),
        source_count=len(selected),
        declared_token_sources=sum(1 for spec in selected if spec.available_tokens is not None),
    )


def require_curriculum_capacity(specs: Sequence[HFSourceSpec], stage: str) -> CurriculumCapacity:
    inventory = curriculum_capacity(specs, stage)
    if stage.strip().casefold() != "foundation":
        return inventory
    if inventory.available_tokens < FOUNDATION_MIN_AVAILABLE_CURRICULUM_TOKENS:
        raise RuntimeError(
            "foundation_curriculum_capacity_insufficient:"
            f"declared={inventory.available_tokens}:required={FOUNDATION_MIN_AVAILABLE_CURRICULUM_TOKENS}"
        )
    missing = sorted(FOUNDATION_SKILL_SET.difference(inventory.skills))
    if missing:
        raise RuntimeError("foundation_curriculum_missing_skills:" + ",".join(missing))
    return inventory


def specs_fingerprint(specs: Sequence[HFSourceSpec]) -> str:
    payload = {
        "source_contract_version": SOURCE_CONTRACT_VERSION,
        "canonical_contract_version": CANONICAL_CONTRACT_VERSION,
        "foundation_contract_version": FOUNDATION_CONTRACT_VERSION,
        "foundation_min_available_curriculum_tokens": FOUNDATION_MIN_AVAILABLE_CURRICULUM_TOKENS,
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
                "dialogue_field": spec.dialogue_field,
                "row_filters": {key: list(values) for key, values in spec.row_filters},
                "tags": list(spec.tags),
                "max_examples": spec.max_examples,
                "shuffle_buffer_size": spec.shuffle_buffer_size,
                "available_tokens": spec.available_tokens,
                "skills": list(spec.skills),
            }
            for spec in specs
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
