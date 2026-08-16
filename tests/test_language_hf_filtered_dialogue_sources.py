from __future__ import annotations

import sys
import types

from corpus.source import HFSource
from src.language.streaming_sources import load_hf_source_config


class _FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def __iter__(self):
        yield from self.rows


def test_row_filters_keep_only_eligible_economics_rows(monkeypatch) -> None:
    rows = [
        {"category": "Physics", "difficulty": "Senior High School", "question": "Q1", "answer": "A1"},
        {"category": "Economics", "difficulty": "University", "question": "Q2", "answer": "A2"},
        {"category": "Economics", "difficulty": "Senior High School", "question": "What is scarcity?", "answer": "Limited resources."},
    ]
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=lambda *args, **kwargs: _FakeDataset(rows)),
    )
    source = HFSource(
        "example/economics",
        revision="deadbeef",
        prompt_field="question",
        response_field="answer",
        row_filters={
            "category": ("Economics",),
            "difficulty": ("Senior High School",),
        },
    )
    output = list(source.stream())
    assert len(output) == 1
    assert "What is scarcity?" in output[0]
    assert "Limited resources." in output[0]


def test_filtered_audit_rates_only_eligible_rows(monkeypatch) -> None:
    rows = [
        {"category": "Physics", "question": "ignored", "answer": "ignored"},
        {"category": "Economics", "question": "Demand?", "answer": "Quantity buyers will buy."},
        {"category": "Economics", "question": "Supply?", "answer": "Quantity sellers will offer."},
    ]
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=lambda *args, **kwargs: _FakeDataset(rows)),
    )
    source = HFSource(
        "example/economics",
        revision="deadbeef",
        prompt_field="question",
        response_field="answer",
        row_filters={"category": ("Economics",)},
    )
    report = source.audit(max_rows=2)
    assert report.rows_scanned == 2
    assert report.rows_serialized == 2
    assert report.serialization_rate == 1.0


def test_dialogue_field_becomes_alternating_user_assistant_chat() -> None:
    source = HFSource(
        "example/dialogue",
        revision="deadbeef",
        dialogue_field="dialogue",
    )
    text = source._row_to_text({"dialogue": ["Hello", "Hi there", "How are you?", "I am well."]})
    assert text is not None
    assert text.count("<user>") == 2
    assert text.count("<assistant>") == 2
    assert "I am well." in text


def test_training_config_contains_external_math_economics_and_dialogue_sources() -> None:
    specs = load_hf_source_config("config/hf_sources.json")
    by_path = {spec.path: spec for spec in specs}
    assert "mkurman/basic-math-operations" in by_path
    assert "TIGER-Lab/WebInstruct-verified" in by_path
    assert "allenai/soda" in by_path
    assert len(by_path["mkurman/basic-math-operations"].revision or "") == 40
    assert by_path["TIGER-Lab/WebInstruct-verified"].row_filter_dict()["category"] == ("Economics",)
    assert by_path["allenai/soda"].dialogue_field == "dialogue"
