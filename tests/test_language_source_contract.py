from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from corpus.source import HFSource, LocalSource


class _FakeIterableDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def shuffle(self, *, seed: int, buffer_size: int):
        return self

    def __iter__(self):
        yield from self.rows


def _install_dataset(monkeypatch, rows):
    dataset = _FakeIterableDataset(rows)
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: dataset),
    )
    return dataset


def test_local_source_uses_same_canonical_parser_as_trainer(tmp_path):
    path = tmp_path / "local.jsonl"
    path.write_text(
        json.dumps({"question": "What is ATR?", "answer": "ATR measures true range."}) + "\n",
        encoding="utf-8",
    )
    text = next(LocalSource(path).stream())
    assert text == (
        "<bos>\n<user>\nWhat is ATR?\n</user>\n"
        "<assistant>\nATR measures true range.\n</assistant>\n<eos>"
    )
    assert "Human:" not in text
    assert "Assistant:" not in text


def test_explicit_prompt_response_mapping_preserves_roles(monkeypatch):
    _install_dataset(
        monkeypatch,
        [{"query_text": "What is ATR?", "target_text": "ATR measures true range."}],
    )
    source = HFSource(
        "example/custom",
        revision="abc123",
        prompt_field="query_text",
        response_field="target_text",
        shuffle_buffer_size=0,
    )
    text = next(source.stream())
    assert "<user>\nWhat is ATR?\n</user>" in text
    assert "<assistant>\nATR measures true range.\n</assistant>" in text


def test_document_fields_cannot_be_combined_with_chat_mapping():
    with pytest.raises(ValueError, match="cannot be combined"):
        HFSource(
            "example/custom",
            revision="abc123",
            text_fields=["query_text", "target_text"],
            prompt_field="query_text",
            response_field="target_text",
        )


def test_source_audit_exposes_bad_schema_rate(monkeypatch):
    _install_dataset(
        monkeypatch,
        [
            {"unknown": "row one"},
            {"unknown": "row two"},
            {"question": "What is spread?", "answer": "Bid minus ask distance."},
        ],
    )
    source = HFSource(
        "example/mixed",
        revision="abc123",
        shuffle_buffer_size=0,
    )
    audit = source.audit(max_rows=3)
    assert audit.rows_scanned == 3
    assert audit.rows_serialized == 1
    assert audit.rows_unrecognized == 2
    assert audit.serialization_rate == pytest.approx(1 / 3)
