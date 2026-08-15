from __future__ import annotations

from corpus.quality import FOUNDATION_ENGLISH_FILTER, LANGUAGE_QUALITY_FILTER
from src.language.canonical_contract import (
    CANONICAL_CONTRACT_VERSION,
    mojibake_score,
    normalize_payload_text,
    repair_mojibake,
    serialize_document,
)
from src.language.streaming_sources import load_hf_source_config, specs_fingerprint


def test_common_utf8_mojibake_is_repaired_before_serialization() -> None:
    assert repair_mojibake("Itâ€™s a good day.") == "It’s a good day."
    assert repair_mojibake("cafÃ© prices") == "café prices"
    assert normalize_payload_text("The marketâ€™s price rose.") == "The market’s price rose."
    serialized = serialize_document("The marketâ€™s price rose because demand increased.")
    assert "market’s" in serialized
    assert "â€™" not in serialized


def test_clean_unicode_is_not_rewritten() -> None:
    text = "Málaga café — prices rose 2%."
    assert mojibake_score(text) == 0
    assert repair_mojibake(text) == text
    assert normalize_payload_text(text) == text


def test_quality_gate_rejects_unrecoverable_corruption() -> None:
    corrupted = "<bos>\nThis sentence contains a damaged replacement character � in otherwise normal English text.\n<eos>"
    general = LANGUAGE_QUALITY_FILTER.score(corrupted)
    foundation = FOUNDATION_ENGLISH_FILTER.score(corrupted)
    assert not general.accepted
    assert "unicode_replacement_character" in general.reasons
    assert not foundation.accepted


def test_canonical_contract_version_is_bumped_for_sanitation_change() -> None:
    assert CANONICAL_CONTRACT_VERSION >= 3


def test_stream_fingerprint_binds_canonical_contract_version() -> None:
    specs = load_hf_source_config("config/hf_sources.json")
    fingerprint = specs_fingerprint(specs)
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 64
