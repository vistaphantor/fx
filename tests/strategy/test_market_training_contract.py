from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tools.fetch_365d_all_timeframes_train import (
    _absolute_spread,
    _candidate_direction,
    _sort_feature_log_by_time,
    _strict_validation_gate,
)


def test_candidate_direction_uses_pre_gate_ce_preference():
    decision = SimpleNamespace(
        action=0,
        metadata={"ce_scores": {-1: 0.04, 0: 0.05, 1: 0.02}},
    )
    assert _candidate_direction(decision, expected_return=0.01) == -1


def test_candidate_direction_falls_back_to_expected_return_on_tie():
    decision = SimpleNamespace(action=0, metadata={"ce_scores": {-1: 0.02, 0: 0.03, 1: 0.02}})
    assert _candidate_direction(decision, expected_return=0.001) == 1
    assert _candidate_direction(decision, expected_return=-0.001) == -1
    assert _candidate_direction(decision, expected_return=0.0) == 0


def test_absolute_spread_is_asset_scale_aware():
    assert _absolute_spread("GC=F") == 0.25
    assert _absolute_spread("USDJPY=X") == 0.015
    assert _absolute_spread("EURUSD=X") == 0.00015


def test_feature_log_is_sorted_globally_by_timestamp(tmp_path: Path):
    path = tmp_path / "features.jsonl"
    rows = [
        {"timestamp": "2026-01-02T10:00:00+00:00", "value": 2},
        {"timestamp": "2026-01-01T10:00:00+00:00", "value": 1},
        {"timestamp": "2026-01-03T10:00:00+00:00", "value": 3},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    _sort_feature_log_by_time(path)

    ordered = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["value"] for row in ordered] == [1, 2, 3]


def _walk_forward(*, recent_precision: float = 0.55, recent_pf: float = 1.2, trades: int = 8):
    folds = []
    for fold in range(1, 6):
        precision = recent_precision if fold >= 4 else 0.60
        pf = recent_pf if fold >= 4 else 1.4
        folds.append(
            {
                "fold": fold,
                "precision": precision,
                "profit_factor": pf,
                "tp": trades,
                "fp": 0,
            }
        )
    return {
        "samples": 2000,
        "average": {"precision": 0.58, "profit_factor": 1.3},
        "folds": folds,
    }


def test_strict_validation_requires_recent_regime_survival():
    passed, report = _strict_validation_gate(_walk_forward())
    assert passed is True
    assert report["passed"] is True

    passed, report = _strict_validation_gate(_walk_forward(recent_pf=0.8))
    assert passed is False
    assert report["recent_folds"][-1]["passed"] is False


def test_strict_validation_requires_large_sample_population():
    walk_forward = _walk_forward()
    walk_forward["samples"] = 1499
    passed, _ = _strict_validation_gate(walk_forward)
    assert passed is False
