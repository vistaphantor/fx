from __future__ import annotations

from itertools import islice

from src.language.conceptual_foundations import ConceptualArithmeticSource, EconomicsCausalSource
from src.language.streaming_sources import (
    FOUNDATION_CONCEPTUAL_ARITHMETIC_WEIGHT,
    FOUNDATION_ECONOMICS_CAUSAL_WEIGHT,
    build_training_stream,
    load_hf_source_config,
)


def test_conceptual_arithmetic_teaches_invariants_and_rejects_wrong_answers() -> None:
    rows = list(islice(ConceptualArithmeticSource().stream(), 2500))
    joined = "\n".join(rows).casefold()
    assert "addition combines quantities" in joined
    assert "additive identity" in joined
    assert "commutative" in joined
    assert "inverse relationship" in joined
    assert "is that correct" in joined
    assert "not" in joined


def test_operand_binding_examples_preserve_original_expression() -> None:
    found = False
    for row in ConceptualArithmeticSource().stream():
        if "Solve 2 + 2 + 3. Do not change any operand." in row:
            assert "operands are 2, 2, and 3" in row
            assert "2 + 2 + 3 = 7" in row
            found = True
            break
    assert found


def test_operand_corruption_is_explicitly_corrected() -> None:
    found = False
    for row in ConceptualArithmeticSource().stream():
        if "answered 2 + 2 + 3 by calculating 2 + 2 + 4" in row:
            assert "last operand was changed from 3 to 4" in row
            assert "requested expression equals 7" in row
            found = True
            break
    assert found


def test_economics_causal_source_teaches_mechanisms_and_counterclaims() -> None:
    rows = list(islice(EconomicsCausalSource().stream(), 900))
    joined = "\n".join(rows).casefold()
    assert "purchasing power falls" in joined
    assert "upward pressure on price" in joined
    assert "profit equals revenue minus cost" in joined
    assert "claim is incorrect" in joined


def test_conceptual_sources_have_material_foundation_weight() -> None:
    assert FOUNDATION_CONCEPTUAL_ARITHMETIC_WEIGHT >= 0.08
    assert FOUNDATION_ECONOMICS_CAUSAL_WEIGHT >= 0.06


def test_authoritative_foundation_stream_includes_conceptual_sources() -> None:
    specs = load_hf_source_config("config/hf_sources.json")
    stream = build_training_stream(specs=specs, stage="foundation", seed=42, repeat=False)
    source_ids = {source.source_id for source, _ in stream.sources}
    assert any("conceptual_arithmetic" in source_id for source_id in source_ids)
    assert any("economics_causal" in source_id for source_id in source_ids)
