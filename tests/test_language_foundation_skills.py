from __future__ import annotations

from itertools import islice

from src.language.foundation_skill_sources import (
    FoundationEconomicsSource,
    PrimitiveArithmeticSource,
)
from src.language.streaming_sources import (
    FOUNDATION_ARITHMETIC_WEIGHT,
    FOUNDATION_ECONOMICS_WEIGHT,
    build_training_stream,
    load_hf_source_config,
)


def test_primitive_arithmetic_source_contains_exact_small_math() -> None:
    rows = list(islice(PrimitiveArithmeticSource().stream(), 5000))
    joined = "\n".join(rows)
    assert "1 + 1 = 2" in joined
    assert "2 + 2 = 4" in joined
    assert "0 - 1 = -1" in joined
    assert "<user>" in joined and "<assistant>" in joined


def test_primitive_arithmetic_source_reaches_three_term_and_tables() -> None:
    # Assert exact prompt->answer capabilities, not generator offsets. Earlier
    # prompt/answer template counts are allowed to evolve without silently moving
    # later arithmetic families outside an arbitrary islice window.
    checks = {
        "three_term_add": lambda row: "What is 2 + 2 + 3?" in row and "2 + 2 + 3 = 7." in row,
        "three_term_negative": lambda row: "What is 2 + 2 - 9?" in row and "The answer is -5." in row,
        "multiplication": lambda row: "Calculate 3 multiplied by 4." in row and "3 times 4 is 12." in row,
        "division": lambda row: "Calculate 12 divided by 3." in row and "12 divided by 3 is 4." in row,
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


def test_foundation_skill_weights_are_nontrivial_but_not_dominant() -> None:
    assert 0.10 <= FOUNDATION_ARITHMETIC_WEIGHT <= 0.25
    assert 0.05 <= FOUNDATION_ECONOMICS_WEIGHT <= 0.20
    assert FOUNDATION_ARITHMETIC_WEIGHT + FOUNDATION_ECONOMICS_WEIGHT < 0.40


def test_authoritative_foundation_stream_includes_generated_skills() -> None:
    specs = load_hf_source_config("config/hf_sources.json")
    stream = build_training_stream(
        specs=specs,
        stage="foundation",
        seed=42,
        repeat=False,
    )
    source_ids = {source.source_id for source, _ in stream.sources}
    assert any("primitive_arithmetic" in source_id for source_id in source_ids)
    assert any("foundation_economics" in source_id for source_id in source_ids)
