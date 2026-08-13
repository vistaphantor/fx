from __future__ import annotations

import csv
import json

import pytest

from src.quick_shadow_monitor import build_quick_shadow_monitor_report


def test_shadow_monitor_reports_training_progress_and_safety(tmp_path):
    journal = tmp_path / "shadow.csv"
    fields = [
        "decision_time",
        "symbol",
        "direction",
        "entry_bid",
        "entry_ask",
        "status",
        "label_outcome",
        "label_max_favorable",
        "label_max_adverse",
        "quality_allowed",
        "quality_reason",
    ]
    with journal.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "decision_time": "2026-08-13T10:00:00+00:00",
                "symbol": "XAUUSD",
                "direction": "BULLISH",
                "entry_bid": "100.0",
                "entry_ask": "100.3",
                "status": "resolved",
                "label_outcome": "win",
                "label_max_favorable": "0.80",
                "label_max_adverse": "0.10",
                "quality_allowed": "1",
                "quality_reason": "quality_ok",
            }
        )
        writer.writerow(
            {
                "decision_time": "2026-08-13T10:01:00+00:00",
                "symbol": "XAUUSD",
                "direction": "BEARISH",
                "entry_bid": "99.8",
                "entry_ask": "100.1",
                "status": "resolved",
                "label_outcome": "loss",
                "label_max_favorable": "0.20",
                "label_max_adverse": "0.40",
                "quality_allowed": "0",
                "quality_reason": "quality_low_consistency",
            }
        )
        writer.writerow(
            {
                "decision_time": "2026-08-13T10:01:00+00:00",
                "symbol": "XAUUSD",
                "direction": "BEARISH",
                "entry_bid": "99.8",
                "entry_ask": "100.1",
                "status": "pending",
                "label_outcome": "",
                "label_max_favorable": "0.0",
                "label_max_adverse": "0.0",
                "quality_allowed": "0",
                "quality_reason": "quality_low_consistency",
            }
        )

    bot_state = tmp_path / "bot_state.json"
    bot_state.write_text(
        json.dumps(
            {
                "account": {"balance": 591.95, "equity": 591.95},
                "trading": {
                    "status": "shadow_only_training",
                    "is_tradeable": False,
                    "positions_count": 0,
                    "session_pnl": -4.05,
                },
            }
        ),
        encoding="utf-8",
    )
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"allowed": False, "reason": "insufficient_shadow_samples"}), encoding="utf-8")
    positions = tmp_path / "positions.csv"
    positions.write_text("ticket,symbol\n", encoding="utf-8")

    report = build_quick_shadow_monitor_report(
        journal_path=journal,
        bot_state_path=bot_state,
        policy_path=policy,
        positions_path=positions,
        symbol="XAUUSD",
    )

    assert report["rows"] == 3
    assert report["resolved"] == 2
    assert report["pending"] == 1
    assert report["wins"] == 1
    assert report["losses"] == 1
    assert report["virtual_net_pnl"] == 0.40
    assert report["inverse_virtual_net_pnl"] == pytest.approx(-0.40)
    assert report["inverse_wins"] == 1
    assert report["inverse_losses"] == 1
    assert report["target_progress"] == pytest.approx(0.008)
    assert report["inverse_target_progress"] == pytest.approx(-0.008)
    assert report["equity_curve"]["ending_pnl"] == pytest.approx(0.40)
    assert report["equity_curve"]["max_drawdown"] == pytest.approx(0.40)
    assert report["equity_curve"]["max_loss_streak"] == 1
    assert report["equity_curve"]["reached_50"] is False
    assert report["duplicates"] == 1
    assert report["diagnostics"]["by_quality_reason"]["quality_ok"]["count"] == 1
    assert report["diagnostics"]["by_quality_reason"]["quality_low_consistency"]["count"] == 1
    assert report["diagnostics"]["by_quality_reason"]["quality_ok"]["inverse_net"] < 0
    assert report["diagnostics"]["by_quality_reason"]["quality_low_consistency"]["inverse_net"] > 0
    assert report["safety_ok"] is True
    assert report["recommendation"] == "deduplicate_shadow_journal"


def test_shadow_monitor_equity_curve_marks_trades_to_50(tmp_path):
    journal = tmp_path / "shadow.csv"
    fields = ["decision_time", "symbol", "direction", "entry_bid", "entry_ask", "status", "label_outcome", "label_max_favorable", "label_max_adverse"]
    with journal.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, profit in enumerate([20.0, -5.0, 40.0], start=1):
            writer.writerow(
                {
                    "decision_time": f"2026-08-13T10:0{idx}:00+00:00",
                    "symbol": "XAUUSD",
                    "direction": "BULLISH",
                    "entry_bid": str(100 + idx),
                    "entry_ask": str(100.3 + idx),
                    "status": "resolved",
                    "label_outcome": "win" if profit > 0 else "loss",
                    "label_max_favorable": str(max(profit, 0.0)),
                    "label_max_adverse": str(abs(min(profit, 0.0))),
                }
            )
    bot_state = tmp_path / "bot_state.json"
    bot_state.write_text(
        json.dumps(
            {
                "trading": {
                    "status": "shadow_only_training",
                    "is_tradeable": False,
                    "daily_profit_target": 50.0,
                }
            }
        ),
        encoding="utf-8",
    )

    report = build_quick_shadow_monitor_report(
        journal_path=journal,
        bot_state_path=bot_state,
        policy_path=tmp_path / "policy.json",
        positions_path=tmp_path / "positions.csv",
        symbol="XAUUSD",
    )

    assert report["virtual_net_pnl"] == pytest.approx(55.0)
    assert report["target_progress"] == pytest.approx(1.1)
    assert report["equity_curve"]["reached_50"] is True
    assert report["equity_curve"]["trades_to_50"] == 3
    assert report["equity_curve"]["max_drawdown"] == pytest.approx(5.0)


def test_shadow_monitor_marks_safety_bad_when_position_file_has_rows(tmp_path):
    journal = tmp_path / "shadow.csv"
    journal.write_text("decision_time,symbol,status,label_outcome\n", encoding="utf-8")
    bot_state = tmp_path / "bot_state.json"
    bot_state.write_text(
        json.dumps({"trading": {"status": "shadow_only_training", "is_tradeable": False}}),
        encoding="utf-8",
    )
    positions = tmp_path / "positions.csv"
    positions.write_text("ticket,symbol\n1,XAUUSD\n", encoding="utf-8")

    report = build_quick_shadow_monitor_report(
        journal_path=journal,
        bot_state_path=bot_state,
        policy_path=tmp_path / "missing.json",
        positions_path=positions,
        symbol="XAUUSD",
    )

    assert report["live_state"]["open_positions_file_count"] == 1
    assert report["safety_ok"] is False
    assert report["recommendation"] == "fix_safety_before_training"
