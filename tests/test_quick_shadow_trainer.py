from __future__ import annotations

import csv

from src.quick_shadow_trainer import (
    ShadowPolicy,
    evaluate_shadow_policy,
    load_resolved_shadow_rows,
    train_shadow_policy,
)


def _row(*, outcome: str, profit: float, adverse: float, estimated: float, consistency: float, quality: str = "1") -> dict[str, str]:
    return {
        "symbol": "XAUUSD",
        "status": "resolved",
        "label_outcome": outcome,
        "label_max_favorable": str(profit if outcome == "win" else 0.0),
        "label_max_adverse": str(adverse if outcome == "loss" else 0.0),
        "estimated_tick_profit": str(estimated),
        "tick_directional_consistency": str(consistency),
        "quality_pullback_ratio": "0.20",
        "quality_allowed": quality,
    }


def test_train_shadow_policy_selects_profitable_feature_gate():
    rows = []
    rows.extend(_row(outcome="win", profit=0.10, adverse=0.0, estimated=0.08, consistency=0.80) for _ in range(24))
    rows.extend(_row(outcome="loss", profit=0.0, adverse=0.04, estimated=0.08, consistency=0.80) for _ in range(4))
    rows.extend(_row(outcome="loss", profit=0.0, adverse=0.12, estimated=0.02, consistency=0.45, quality="0") for _ in range(20))

    report = train_shadow_policy(
        rows,
        min_samples=40,
        min_selected=20,
        min_win_rate=0.58,
        min_expectancy=0.02,
        min_profit_factor=1.20,
        max_loss_streak=4,
        validation_fraction=0.0,
    )

    assert report.allowed is True
    assert report.reason == "shadow_policy_ok"
    assert report.policy.require_quality_allowed is True
    assert report.selected_count >= 20
    assert report.win_rate > 0.80
    assert report.expectancy > 0.07


def test_evaluate_shadow_policy_rejects_loss_streak_even_when_expectancy_positive():
    rows = []
    rows.extend(_row(outcome="win", profit=0.20, adverse=0.0, estimated=0.08, consistency=0.80) for _ in range(10))
    rows.extend(_row(outcome="loss", profit=0.0, adverse=0.03, estimated=0.08, consistency=0.80) for _ in range(5))

    report = evaluate_shadow_policy(
        rows,
        policy=ShadowPolicy(0.05, 0.60, 0.65, True),
        min_samples=10,
        min_selected=10,
        min_win_rate=0.58,
        min_expectancy=0.02,
        min_profit_factor=1.20,
        max_loss_streak=3,
    )

    assert report.allowed is False
    assert report.reason == "shadow_policy_loss_streak_too_high"
    assert report.max_loss_streak == 5


def test_shadow_policy_can_filter_by_quality_reason():
    rows = [
        _row(outcome="win", profit=0.20, adverse=0.0, estimated=0.08, consistency=0.80),
        _row(outcome="loss", profit=0.0, adverse=0.20, estimated=0.08, consistency=0.80),
    ]
    rows[0]["quality_reason"] = "quality_too_much_pullback"
    rows[1]["quality_reason"] = "quality_ok"

    report = evaluate_shadow_policy(
        rows,
        policy=ShadowPolicy(0.05, 0.60, 1.0, False, ("quality_too_much_pullback",)),
        min_samples=1,
        min_selected=1,
        min_win_rate=0.50,
        min_expectancy=0.01,
        min_profit_factor=1.0,
        max_loss_streak=1,
    )

    assert report.allowed is True
    assert report.selected_count == 1
    assert report.win_rate == 1.0


def test_train_shadow_policy_can_select_inverse_direction_policy():
    rows = []
    for _ in range(24):
        row = _row(outcome="loss", profit=0.0, adverse=0.20, estimated=0.08, consistency=0.70)
        row["target_profit"] = "0.05"
        row["max_loss"] = "0.50"
        rows.append(row)
    for _ in range(4):
        row = _row(outcome="win", profit=0.08, adverse=0.0, estimated=0.08, consistency=0.70)
        row["target_profit"] = "0.05"
        row["max_loss"] = "0.50"
        rows.append(row)

    report = train_shadow_policy(
        rows,
        min_samples=20,
        min_selected=20,
        min_win_rate=0.58,
        min_expectancy=0.02,
        min_profit_factor=1.20,
        max_loss_streak=4,
        validation_fraction=0.0,
    )

    assert report.allowed is True
    assert report.policy.invert_direction is True
    assert report.win_rate > 0.80
    assert report.expectancy > 0.10


def test_train_shadow_policy_prefers_enough_selected_samples_over_tiny_slice():
    rows = []
    rows.extend(_row(outcome="win", profit=5.00, adverse=0.0, estimated=2.00, consistency=0.90) for _ in range(2))
    rows.extend(_row(outcome="win", profit=0.12, adverse=0.0, estimated=0.06, consistency=0.70) for _ in range(18))
    rows.extend(_row(outcome="loss", profit=0.0, adverse=0.04, estimated=0.06, consistency=0.70) for _ in range(4))
    rows.extend(_row(outcome="loss", profit=0.0, adverse=0.20, estimated=0.01, consistency=0.40, quality="0") for _ in range(20))

    report = train_shadow_policy(
        rows,
        min_samples=60,
        min_selected=20,
        min_win_rate=0.58,
        min_expectancy=0.02,
        min_profit_factor=1.20,
        max_loss_streak=4,
        validation_fraction=0.0,
    )

    assert report.allowed is False
    assert report.reason == "insufficient_shadow_samples"
    assert report.selected_count >= 20
    assert report.policy.min_estimated_profit <= 0.06


def test_train_shadow_policy_requires_walk_forward_validation():
    rows = []
    for _ in range(30):
        row = _row(outcome="loss", profit=0.0, adverse=0.20, estimated=0.08, consistency=0.70)
        row["target_profit"] = "0.05"
        row["max_loss"] = "0.50"
        rows.append(row)
    for _ in range(15):
        row = _row(outcome="win", profit=0.20, adverse=0.0, estimated=0.08, consistency=0.70)
        row["target_profit"] = "0.05"
        row["max_loss"] = "0.50"
        rows.append(row)

    report = train_shadow_policy(
        rows,
        min_samples=30,
        min_selected=20,
        min_win_rate=0.58,
        min_expectancy=0.02,
        min_profit_factor=1.20,
        max_loss_streak=4,
        validation_fraction=0.30,
        min_validation_selected=5,
    )

    assert report.policy.invert_direction is True
    assert report.validation_allowed is False
    assert report.allowed is False
    assert report.reason != "shadow_policy_ok"


def test_load_resolved_shadow_rows_filters_symbol_and_pending_rows(tmp_path):
    path = tmp_path / "shadow.csv"
    fieldnames = ["symbol", "status", "label_outcome"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"symbol": "XAUUSD", "status": "pending", "label_outcome": ""})
        writer.writerow({"symbol": "EURUSD", "status": "resolved", "label_outcome": "win"})
        writer.writerow({"symbol": "XAUUSD", "status": "resolved", "label_outcome": "loss"})

    rows = load_resolved_shadow_rows(path, symbol="XAUUSD")

    assert len(rows) == 1
    assert rows[0]["label_outcome"] == "loss"
