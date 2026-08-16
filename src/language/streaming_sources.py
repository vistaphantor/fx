from __future__ import annotations

from dataclasses import replace
from typing import Iterator, Sequence

from corpus.dedup import NearDuplicateIndex
from corpus.quality import FOUNDATION_ENGLISH_FILTER, LANGUAGE_QUALITY_FILTER, QualityFilter
from corpus.source import DatasetSource, HFSource
from corpus.streamer import WeightedSourceStream
from src.language.canonical_contract import (
    CANONICAL_CONTRACT_VERSION,
    CanonicalMessage,
    canonical_hash,
    canonicalize_serialized,
    prompt_family,
    serialize_messages,
)
from src.language.conceptual_foundations import ConceptualArithmeticSource, EconomicsCausalSource
from src.language.curriculum import is_math_example, is_reasoning_example, is_trading_example
from src.language.exam_feedback import ExamFeedbackPolicy
from src.language.foundation_exam_source import FoundationExamCurriculumSource
from src.language.foundation_skill_sources import FoundationEconomicsSource, PrimitiveArithmeticSource
from src.language.language_quality_source import LanguageQualityContrastSource
from src.language.source_contract import (
    CurriculumCapacity,
    HFSourceSpec,
    _parse_spec,
    curriculum_capacity,
    load_hf_source_config,
    require_curriculum_capacity,
    specs_fingerprint,
    stage_specs,
)

FOUNDATION_INTERACTION_FRACTION = 0.25
FOUNDATION_CONTINUATION_PREFIX_WORDS = 32
FOUNDATION_CONTINUATION_TARGET_WORDS = 48
# The 8B run is broad-source first. Generated curricula reinforce elementary
# invariants but are deliberately a minority of supervised prediction tokens.
FOUNDATION_ARITHMETIC_WEIGHT = 0.05
FOUNDATION_CONCEPTUAL_ARITHMETIC_WEIGHT = 0.08
FOUNDATION_ECONOMICS_WEIGHT = 0.03
FOUNDATION_ECONOMICS_CAUSAL_WEIGHT = 0.06
FOUNDATION_LANGUAGE_QUALITY_WEIGHT = 0.02
FOUNDATION_EXAM_CURRICULUM_WEIGHT = 0.05


def stream_quality_accepts(text: str) -> bool:
    return LANGUAGE_QUALITY_FILTER.accepts(text)


def _is_chat(text: str) -> bool:
    return "<user>" in text and "<assistant>" in text


def _document_payload(text: str) -> str:
    value = canonicalize_serialized(text)
    if value.startswith("<bos>"):
        value = value[len("<bos>"):].lstrip()
    if value.endswith("<eos>"):
        value = value[:-len("<eos>"):].rstrip()
    return value.strip()


def _continuation_interaction(text: str) -> str | None:
    if _is_chat(text):
        return None
    words = _document_payload(text).split()
    if len(words) < FOUNDATION_CONTINUATION_PREFIX_WORDS + 12:
        return None
    prefix_end = min(FOUNDATION_CONTINUATION_PREFIX_WORDS, len(words) - 12)
    target_end = min(len(words), prefix_end + FOUNDATION_CONTINUATION_TARGET_WORDS)
    prefix = " ".join(words[:prefix_end]).strip()
    continuation = " ".join(words[prefix_end:target_end]).strip()
    if not prefix or not continuation:
        return None
    return serialize_messages((
        CanonicalMessage("user", "Continue this passage naturally:\n" + prefix),
        CanonicalMessage("assistant", continuation),
    )) or None


def _use_foundation_interaction(digest: str) -> bool:
    return int(digest[:8], 16) / 0xFFFFFFFF < FOUNDATION_INTERACTION_FRACTION


class GuardedSource(DatasetSource):
    """Canonical quality gate, deduplicator and exam/validation holdout boundary."""

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
        quality_filter: QualityFilter | None = None,
        transform_foundation_documents: bool = True,
    ):
        self._source = source
        self._stage = stage.strip().casefold()
        self._excluded_hashes = excluded_hashes
        self._excluded_families = excluded_families
        self._near_dedup_entries = int(near_dedup_entries)
        self._near_dedup_hamming = int(near_dedup_hamming)
        self._near_index = near_index
        self._quality_filter_override = quality_filter
        self._transform_foundation_documents = bool(transform_foundation_documents)

    @property
    def source_id(self) -> str:
        return f"guarded:{self._source.source_id}:{self._stage}"

    def scan(self):
        return self._source.scan()

    def metadata(self) -> dict:
        return {
            **self._source.metadata(),
            "guarded_stage": self._stage,
            "near_dedup_entries": self._near_dedup_entries,
            "near_dedup_hamming": self._near_dedup_hamming,
            "shared_near_dedup": self._near_index is not None,
            "canonical_contract_version": CANONICAL_CONTRACT_VERSION,
            "foundation_interaction_fraction": (
                FOUNDATION_INTERACTION_FRACTION
                if self._stage == "foundation" and self._transform_foundation_documents else 0.0
            ),
        }

    def _quality_accepts(self, text: str) -> bool:
        if self._quality_filter_override is not None:
            return self._quality_filter_override.accepts(text)
        if self._stage == "foundation" and _is_chat(text):
            return LANGUAGE_QUALITY_FILTER.accepts(text)
        if self._stage == "foundation":
            return FOUNDATION_ENGLISH_FILTER.accepts(text)
        return LANGUAGE_QUALITY_FILTER.accepts(text)

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
            if not text or not self._quality_accepts(text):
                continue
            digest = canonical_hash(text)
            if digest in seen or digest in self._excluded_hashes:
                continue
            family = prompt_family(text)
            if family and family in self._excluded_families:
                continue
            if not self._stage_accepts(text) or not near.accept(text):
                continue
            seen.add(digest)
            if (
                self._stage == "foundation"
                and self._transform_foundation_documents
                and not _is_chat(text)
                and _use_foundation_interaction(digest)
            ):
                interaction = _continuation_interaction(text)
                if interaction:
                    yield interaction
                    continue
            yield text


def hf_source_from_spec(spec: HFSourceSpec, *, seed: int) -> HFSource:
    return HFSource(
        path=spec.path,
        split=spec.split,
        text_fields=list(spec.text_fields) if spec.text_fields else None,
        prompt_field=spec.prompt_field,
        response_field=spec.response_field,
        dialogue_field=spec.dialogue_field,
        row_filters=spec.row_filter_dict(),
        max_examples=spec.max_examples,
        config_name=spec.config_name,
        revision=spec.revision,
        shuffle_buffer_size=spec.shuffle_buffer_size,
        seed=seed,
    )


def _foundation_skill_sources(
    *, excluded_hashes: frozenset[str], excluded_families: frozenset[str], feedback: ExamFeedbackPolicy,
) -> list[tuple[DatasetSource, float]]:
    specs: tuple[tuple[DatasetSource, float, int, tuple[str, ...]], ...] = (
        (PrimitiveArithmeticSource(), FOUNDATION_ARITHMETIC_WEIGHT, 250_000, ("arithmetic",)),
        (ConceptualArithmeticSource(), FOUNDATION_CONCEPTUAL_ARITHMETIC_WEIGHT, 250_000, ("arithmetic",)),
        (FoundationEconomicsSource(), FOUNDATION_ECONOMICS_WEIGHT, 100_000, ("economics",)),
        (EconomicsCausalSource(), FOUNDATION_ECONOMICS_CAUSAL_WEIGHT, 100_000, ("economics",)),
        (LanguageQualityContrastSource(), FOUNDATION_LANGUAGE_QUALITY_WEIGHT, 100_000,
         ("language_quality", "conversation", "creativity")),
        (FoundationExamCurriculumSource(), FOUNDATION_EXAM_CURRICULUM_WEIGHT, 100_000,
         ("language_quality", "conversation")),
    )
    return [
        (
            GuardedSource(
                source,
                stage="foundation",
                excluded_hashes=excluded_hashes,
                excluded_families=excluded_families,
                near_dedup_entries=near_entries,
                near_dedup_hamming=0,
                quality_filter=LANGUAGE_QUALITY_FILTER,
                transform_foundation_documents=False,
            ),
            weight * feedback.multiplier_for_tags(tags),
        )
        for source, weight, near_entries, tags in specs
    ]


def build_training_stream(
    *,
    specs: Sequence[HFSourceSpec],
    stage: str,
    seed: int,
    local_replay: Sequence[str] = (),
    local_weight: float = 0.0,
    excluded_texts: Sequence[str] = (),
    repeat: bool = True,
    feedback: ExamFeedbackPolicy | None = None,
) -> WeightedSourceStream:
    if local_replay or local_weight > 0:
        raise RuntimeError("local_language_training_disabled:configure_a_pinned_streaming_source_instead")
    policy = feedback or ExamFeedbackPolicy()
    normalized_stage = stage.strip().casefold()
    selected = stage_specs(specs, stage)
    excluded_hashes = frozenset(canonical_hash(text) for text in excluded_texts if text and text.strip())
    excluded_families = frozenset(prompt_family(text) for text in excluded_texts if text and text.strip())
    sources: list[tuple[DatasetSource, float]] = []
    hf_near_index = NearDuplicateIndex(max_entries=50_000, max_hamming_distance=4)
    for index, spec in enumerate(selected):
        unicode_foundation = normalized_stage == "foundation" and {"swahili", "shairi", "poetry", "financial_news_comprehension"}.intersection(spec.skills)
        sources.append((
            GuardedSource(
                hf_source_from_spec(spec, seed=seed + index * 997),
                stage=stage,
                excluded_hashes=excluded_hashes,
                excluded_families=excluded_families,
                near_index=hf_near_index,
                quality_filter=LANGUAGE_QUALITY_FILTER if unicode_foundation else None,
            ),
            spec.weight * policy.multiplier_for_tags(spec.tags),
        ))
    if normalized_stage == "foundation":
        sources.extend(_foundation_skill_sources(
            excluded_hashes=excluded_hashes,
            excluded_families=excluded_families,
            feedback=policy,
        ))
    return WeightedSourceStream(sources, seed=seed, repeat=repeat)


def sample_training_stream(
    *, specs: Sequence[HFSourceSpec], stage: str, limit: int, seed: int,
) -> list[str]:
    if limit <= 0:
        raise ValueError("stream sample limit must be positive")
    selected = stage_specs(specs, stage)
    preflight_specs = tuple(replace(spec, shuffle_buffer_size=0) for spec in selected)
    stream = build_training_stream(
        specs=preflight_specs,
        stage=stage,
        seed=seed,
        repeat=False,
        feedback=ExamFeedbackPolicy(),
    )
    result: list[str] = []
    for text in stream:
        result.append(text)
        if len(result) >= limit:
            break
    if not result:
        raise RuntimeError("hf_stream_sample_produced_no_examples")
    return result
