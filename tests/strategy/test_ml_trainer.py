"""Tests for offline ML trainer data loading."""

from __future__ import annotations

import json

from src.strategy.ml_trainer import load_feature_rows


def test_load_feature_rows_canonicalizes_duplicate_and_malformed_rows(tmp_path):
    feature_path = tmp_path / "features.jsonl"
    rows = [
        {"timestamp": "2026-04-28T02:00:00+00:00", "momentum_raw": 1},
        {"timestamp": "2026-04-28T02:00:00+00:00", "momentum_raw": 99},
        {"timestamp": "2026-04-28T02:15:00+00:00", "momentum_raw": 2},
    ]
    feature_path.write_text(
        json.dumps(rows[0])
        + "\n"
        + "\x00\x00\n"
        + json.dumps(rows[1])
        + "\n"
        + json.dumps(rows[2])
        + "\n",
        encoding="utf-8",
    )

    loaded = load_feature_rows(feature_path)

    assert loaded == [rows[0], rows[2]]
