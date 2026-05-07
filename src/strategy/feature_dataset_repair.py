"""Offline repair utility for feature snapshot JSONL datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class FeatureDatasetRepairResult:
    """Summary of a feature dataset canonicalization run."""

    source: Path
    output: Path
    rows_read: int
    rows_written: int
    rows_removed: int
    duplicate_rows_removed: int
    invalid_rows_removed: int


def canonicalize_feature_dataset(
    source: str | Path = "data/features.jsonl",
    output: str | Path = "data/features.cleaned.jsonl",
) -> FeatureDatasetRepairResult:
    """Drop duplicate feature snapshots by timestamp and write a cleaned JSONL file.

    The first row for each timestamp is retained, preserving the original order of
    retained rows. Blank lines are ignored and are not counted as input rows.
    """
    source_path = Path(source)
    output_path = Path(output)
    seen_timestamps: set[Any] = set()
    rows_read = 0
    rows_written = 0
    duplicate_rows_removed = 0
    invalid_rows_removed = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with source_path.open(encoding="utf-8") as source_fh, output_path.open(
        "w",
        encoding="utf-8",
    ) as output_fh:
        for line in source_fh:
            line = line.strip()
            if not line:
                continue

            rows_read += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_rows_removed += 1
                continue

            if not isinstance(row, dict) or "timestamp" not in row:
                invalid_rows_removed += 1
                continue

            timestamp = row["timestamp"]
            if timestamp in seen_timestamps:
                duplicate_rows_removed += 1
                continue

            seen_timestamps.add(timestamp)
            output_fh.write(json.dumps(row) + "\n")
            rows_written += 1

    return FeatureDatasetRepairResult(
        source=source_path,
        output=output_path,
        rows_read=rows_read,
        rows_written=rows_written,
        rows_removed=rows_read - rows_written,
        duplicate_rows_removed=duplicate_rows_removed,
        invalid_rows_removed=invalid_rows_removed,
    )


def _format_row_count(count: int, label: str = "row") -> str:
    suffix = "" if count == 1 else "s"
    return f"{count} {label}{suffix}"


def _format_removed_count(result: FeatureDatasetRepairResult) -> str:
    if result.invalid_rows_removed == 0:
        return _format_row_count(result.rows_removed, "duplicate row")
    return _format_row_count(result.rows_removed)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Canonicalize feature JSONL snapshots by dropping duplicate timestamps.",
    )
    parser.add_argument(
        "--features",
        default="data/features.jsonl",
        help="Path to the source feature JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="data/features.cleaned.jsonl",
        help="Path for the cleaned JSONL output file.",
    )
    args = parser.parse_args()

    result = canonicalize_feature_dataset(args.features, args.output)
    print(
        f"Read {result.rows_read} rows, wrote {result.rows_written} rows, "
        f"removed {_format_removed_count(result)} "
        f"({result.duplicate_rows_removed} duplicate, {result.invalid_rows_removed} invalid). "
        f"Output: {result.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
