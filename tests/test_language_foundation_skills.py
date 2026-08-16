from __future__ import annotations

from itertools import islice

from src.language.canonical_contract import prompt_family
from src.language.exam import FOUNDATION_EXAM, build_exam_prompt
from src.language.foundation_contract import FOUNDATION_SKILLS
from src.language.foundation_exam_source import FoundationExamCurriculumSource
from src.language.foundation_skill_sources import FoundationEconomicsSource, PrimitiveArithmeticSource
from src.language.streaming_sources import (
    FOUNDATION_ARITHMETIC_WEIGHT,
    FOUNDATION_CONCEPTUAL_ARITHMETIC_WEIGHT,
    FOUNDATION_ECONOMICS_WEIGHT,
    GuardedSource,
    build_training_stream,
    load_hf_source_config,
    require_curriculum_capacity,
)


def test_primitive_arithmetic_source_contains_exact_small_math() -> None:
    rows = list(islice(PrimitiveArithmeticSource().stream(), 5000))
    joined = "\n".join(rows)
    assert "What is 2 + 2?" in joined
    assert "<assistant>\n4." in joined
    assert "<user>" in joined and "<assistant>" in joined


def test_primitive_arithmetic_source_reaches_three_term_and_tables() -> None:
    checks = {
        "three_term_add": lambda row: "What is 2 + 2 + 3?" in row and "<assistant>\n7." in row,
        "three_term_negative": lambda row: "What is 2 + 2 - 9?" in row and "<assistant>\n-5." in row,
        "multiplication": lambda row: "What is 3 times 4?" in row and "<assistant>\n12." in row,
        "division": lambda row: "What is 12 divided by 3?" in row and "<assistant>\n4." in row,
    }
    found = {name: False for name in checks}
    for row in PrimitiveArithmeticSource().stream():
        for name, predicate in checks.items():
            if not found[name] and predicate(row):
                found[name] = True
        if all(found.values()):
            break
    assert found == {name: True for name in checks}


def test_foundation_economics_source_starts_with_primitive_concepts() -> None:
    rows = list(islice(FoundationEconomicsSource().stream(), 120))
    joined = "\n".join(rows).casefold()
    for term in (
        "scarcity", "opportunity cost", "price", "market", "demand", "supply",
        "revenue", "profit", "saving", "interest", "inflation", "budget",
    ):
        assert term in joined
    assert "<user>" in joined and "<assistant>" in joined


def test_generated_foundation_skills_keep_strong_elementary_pressure() -> None:
    assert FOUNDATION_ARITHMETIC_WEIGHT > 0
    assert FOUNDATION_CONCEPTUAL_ARITHMETIC_WEIGHT > 0
    assert FOUNDATION_ECONOMICS_WEIGHT > 0
    assert FOUNDATION_ARITHMETIC_WEIGHT >= FOUNDATION_ECONOMICS_WEIGHT


def test_authoritative_foundation_stream_includes_generated_skills() -> None:
    specs = load_hf_source_config("config/hf_sources.json")
    stream = build_training_stream(specs=specs, stage="foundation", seed=42, repeat=False)
    source_ids = {source.source_id for source, _ in stream.sources}
    assert any("primitive_arithmetic" in source_id for source_id in source_ids)
    assert any("conceptual_arithmetic" in source_id for source_id in source_ids)
    assert any("foundation_economics" in source_id for source_id in source_ids)
    assert any("foundation_exam_curriculum" in source_id for source_id in source_ids)


def test_reserved_exam_rows_are_owned_by_source_but_removed_before_training() -> None:
    held_families = frozenset(prompt_family(build_exam_prompt(question.prompt)) for question in FOUNDATION_EXAM)
    source = FoundationExamCurriculumSource()
    raw = list(islice(source.stream(), 4))
    assert prompt_family(raw[0]) in held_families

    guarded = GuardedSource(
        source,
        stage="foundation",
        excluded_families=held_families,
        near_dedup_hamming=0,
    )
    kept = list(islice(guarded.stream(), 3))
    assert kept
    assert all(prompt_family(row) not in held_families for row in kept)


def test_foundation_inventory_covers_every_examined_skill() -> None:
    inventory = require_curriculum_capacity(load_hf_source_config("config/hf_sources.json"), "foundation")
    assert set(FOUNDATION_SKILLS).issubset(inventory.skills)
