"""Tests for the offline feature dataset repair utility."""

from __future__ import annotations

import json
import subprocess
import sys

from src.strategy.feature_dataset_repair import canonicalize_feature_dataset


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


def test_canonicalize_feature_dataset_drops_duplicate_timestamps_and_preserves_order(tmp_path):
    source = tmp_path / "features.jsonl"
    output = tmp_path / "features.cleaned.jsonl"
    rows = [
        {"timestamp": "2026-04-28T02:00:00+00:00", "momentum_raw": 1},
        {"timestamp": "2026-04-28T02:15:00+00:00", "momentum_raw": 2},
        {"timestamp": "2026-04-28T02:00:00+00:00", "momentum_raw": 99},
        {"timestamp": "2026-04-28T02:30:00+00:00", "momentum_raw": 3},
        {"timestamp": "2026-04-28T02:15:00+00:00", "momentum_raw": 88},
    ]
    source.write_text(_jsonl(rows), encoding="utf-8")

    result = canonicalize_feature_dataset(source, output)

    cleaned = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert cleaned == [rows[0], rows[1], rows[3]]
    assert result.rows_read == 5
    assert result.rows_written == 3
    assert result.rows_removed == 2


def test_canonicalize_feature_dataset_ignores_blank_lines_without_counting_them(tmp_path):
    source = tmp_path / "features.jsonl"
    output = tmp_path / "features.cleaned.jsonl"
    source.write_text(
        '{"timestamp": "2026-04-28T02:00:00+00:00", "momentum_raw": 1}\n\n'
        '{"timestamp": "2026-04-28T02:00:00+00:00", "momentum_raw": 2}\n',
        encoding="utf-8",
    )

    result = canonicalize_feature_dataset(source, output)

    assert result.rows_read == 2
    assert result.rows_written == 1
    assert result.rows_removed == 1


def test_canonicalize_feature_dataset_skips_malformed_rows_and_reports_them(tmp_path):
    source = tmp_path / "features.jsonl"
    output = tmp_path / "features.cleaned.jsonl"
    source.write_text(
        '{"timestamp": "2026-04-28T02:00:00+00:00", "momentum_raw": 1}\n'
        "\x00\x00\x00\n"
        '{"timestamp": "2026-04-28T02:15:00+00:00", "momentum_raw": 2}\n',
        encoding="utf-8",
    )

    result = canonicalize_feature_dataset(source, output)

    cleaned = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert cleaned == [
        {"timestamp": "2026-04-28T02:00:00+00:00", "momentum_raw": 1},
        {"timestamp": "2026-04-28T02:15:00+00:00", "momentum_raw": 2},
    ]
    assert result.rows_read == 3
    assert result.invalid_rows_removed == 1
    assert result.rows_removed == 1


def test_cli_writes_cleaned_output_and_reports_removed_rows(tmp_path):
    source = tmp_path / "features.jsonl"
    output = tmp_path / "features.cleaned.jsonl"
    source.write_text(
        _jsonl(
            [
                {"timestamp": "2026-04-28T02:00:00+00:00", "momentum_raw": 1},
                {"timestamp": "2026-04-28T02:00:00+00:00", "momentum_raw": 2},
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.strategy.feature_dataset_repair",
            "--features",
            str(source),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "removed 1 duplicate row" in completed.stdout
    assert output.exists()
