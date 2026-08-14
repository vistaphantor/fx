from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.language.streaming_sources import (
    hf_source_from_spec,
    load_hf_source_config,
    stage_specs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit pinned Hugging Face source schemas before Vista language training."
    )
    parser.add_argument("--hf-config", required=True)
    parser.add_argument(
        "--training-stage",
        choices=("foundation", "reasoning", "trading_reasoning"),
        default="foundation",
    )
    parser.add_argument("--rows-per-source", type=int, default=1000)
    parser.add_argument("--min-serialization-rate", type=float, default=0.80)
    args = parser.parse_args()

    if args.rows_per_source <= 0:
        raise ValueError("rows-per-source must be positive")
    if not 0.0 < args.min_serialization_rate <= 1.0:
        raise ValueError("min-serialization-rate must be in (0, 1]")

    specs = load_hf_source_config(args.hf_config)
    selected = stage_specs(specs, args.training_stage)
    reports: list[dict] = []
    failures: list[str] = []

    for index, spec in enumerate(selected):
        source = hf_source_from_spec(spec, seed=42 + index * 997)
        report = source.audit(max_rows=args.rows_per_source)
        payload = asdict(report)
        payload["serialization_rate"] = report.serialization_rate
        reports.append(payload)
        print(
            f"[SourceAudit] {spec.path}@{spec.revision} "
            f"rows={report.rows_scanned:,} serialized={report.rows_serialized:,} "
            f"rate={report.serialization_rate:.1%} "
            f"chars(mean/min/max)={report.mean_serialized_chars:.0f}/"
            f"{report.min_serialized_chars}/{report.max_serialized_chars}"
        )
        if report.rows_scanned <= 0:
            failures.append(f"{spec.path}:empty_source")
        elif report.serialization_rate < args.min_serialization_rate:
            failures.append(
                f"{spec.path}:serialization_rate={report.serialization_rate:.3f}"
            )

    output = Path(args.hf_config).with_suffix(".audit.json")
    output.write_text(
        json.dumps(
            {
                "training_stage": args.training_stage,
                "rows_per_source": args.rows_per_source,
                "min_serialization_rate": args.min_serialization_rate,
                "sources": reports,
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError("hf_source_audit_failed:" + ";".join(failures))
    print(f"[SourceAudit] PASS report={output}")


if __name__ == "__main__":
    main()
