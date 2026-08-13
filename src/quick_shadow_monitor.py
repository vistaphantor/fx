from __future__ import annotations

import csv
import json
from pathlib import Path

from src.quick_shadow_trainer import (
    load_resolved_shadow_rows,
    save_shadow_policy_report,
    train_shadow_policy,
)


def build_quick_shadow_monitor_report(
    *,
    journal_path: str | Path,
    bot_state_path: str | Path = "bot_state.json",
    policy_path: str | Path = "data/quick_shadow_policy.json",
    positions_path: str | Path | None = None,
    symbol: str = "",
    retrain_policy: bool = False,
    validation_fraction: float = 0.30,
    min_validation_selected: int = 8,
) -> dict:
    rows = _read_csv_rows(journal_path)
    if symbol:
        rows = [row for row in rows if row.get("symbol") == symbol]
    resolved = [row for row in rows if row.get("status") == "resolved" and row.get("label_outcome") in {"win", "loss"}]
    pending = [row for row in rows if row.get("status") == "pending"]
    wins = [row for row in resolved if row.get("label_outcome") == "win"]
    losses = [row for row in resolved if row.get("label_outcome") == "loss"]
    virtual_profits = [_virtual_profit(row) for row in resolved]
    inverse_virtual_profits = [_inverse_virtual_profit(row) for row in resolved]
    original_curve = _equity_curve_metrics(virtual_profits)
    inverse_curve = _equity_curve_metrics(inverse_virtual_profits)
    quality_rows = [row for row in resolved if str(row.get("quality_allowed", "")).strip() in {"1", "true", "True"}]
    duplicate_count = _duplicate_decision_count(rows)

    if retrain_policy:
        training_rows = load_resolved_shadow_rows(journal_path, symbol=symbol)
        policy_report = train_shadow_policy(
            training_rows,
            validation_fraction=validation_fraction,
            min_validation_selected=min_validation_selected,
        )
        save_shadow_policy_report(policy_report, policy_path)
        policy_payload = policy_report.to_dict()
    else:
        policy_payload = _read_json(policy_path)

    bot_state = _read_json(bot_state_path)
    trading = bot_state.get("trading", {}) if isinstance(bot_state.get("trading"), dict) else {}
    account = bot_state.get("account", {}) if isinstance(bot_state.get("account"), dict) else {}
    open_positions = _count_positions(positions_path) if positions_path else 0
    is_shadow_only = trading.get("status") == "shadow_only_training" and trading.get("is_tradeable") is False
    safety_ok = is_shadow_only and open_positions == 0

    report = {
        "journal": str(Path(journal_path)),
        "symbol": symbol,
        "rows": len(rows),
        "resolved": len(resolved),
        "pending": len(pending),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(resolved) if resolved else 0.0,
        "virtual_net_pnl": sum(virtual_profits),
        "avg_virtual_pnl": sum(virtual_profits) / len(virtual_profits) if virtual_profits else 0.0,
        "inverse_virtual_net_pnl": sum(inverse_virtual_profits),
        "inverse_avg_virtual_pnl": sum(inverse_virtual_profits) / len(inverse_virtual_profits) if inverse_virtual_profits else 0.0,
        "inverse_wins": sum(1 for profit in inverse_virtual_profits if profit > 0.0),
        "inverse_losses": sum(1 for profit in inverse_virtual_profits if profit < 0.0),
        "inverse_win_rate": (
            sum(1 for profit in inverse_virtual_profits if profit > 0.0) / len(inverse_virtual_profits)
            if inverse_virtual_profits
            else 0.0
        ),
        "target_progress": _target_progress(sum(virtual_profits), trading),
        "inverse_target_progress": _target_progress(sum(inverse_virtual_profits), trading),
        "equity_curve": original_curve,
        "inverse_equity_curve": inverse_curve,
        "quality_allowed_resolved": len(quality_rows),
        "quality_allowed_win_rate": _win_rate(quality_rows),
        "diagnostics": {
            "by_direction": _bucket_summary(resolved, "direction"),
            "by_quality_reason": _bucket_summary(resolved, "quality_reason"),
            "by_quality_allowed": _bucket_summary(resolved, "quality_allowed"),
            "by_estimated_profit": _numeric_bucket_summary(
                resolved,
                "estimated_tick_profit",
                [(0.5, "<0.5"), (1.0, "0.5-1"), (2.0, "1-2"), (5.0, "2-5")],
                ">=5",
            ),
            "by_consistency": _numeric_bucket_summary(
                resolved,
                "tick_directional_consistency",
                [(0.55, "<0.55"), (0.67, "0.55-0.67"), (0.80, "0.67-0.80")],
                ">=0.80",
            ),
        },
        "duplicates": duplicate_count,
        "latest_decision_time": rows[-1].get("decision_time", "") if rows else "",
        "policy": {
            "allowed": bool(policy_payload.get("allowed", False)) if isinstance(policy_payload, dict) else False,
            "reason": str(policy_payload.get("reason", "missing_policy")) if isinstance(policy_payload, dict) else "missing_policy",
            "sample_count": int(float(policy_payload.get("sample_count", 0) or 0)) if isinstance(policy_payload, dict) else 0,
            "selected_count": int(float(policy_payload.get("selected_count", 0) or 0)) if isinstance(policy_payload, dict) else 0,
            "win_rate": float(policy_payload.get("win_rate", 0.0) or 0.0) if isinstance(policy_payload, dict) else 0.0,
            "expectancy": float(policy_payload.get("expectancy", 0.0) or 0.0) if isinstance(policy_payload, dict) else 0.0,
            "profit_factor": float(policy_payload.get("profit_factor", 0.0) or 0.0) if isinstance(policy_payload, dict) else 0.0,
            "validation": policy_payload.get("validation", {}) if isinstance(policy_payload.get("validation", {}), dict) else {},
        },
        "live_state": {
            "status": trading.get("status", ""),
            "is_tradeable": bool(trading.get("is_tradeable", False)),
            "positions_count": int(float(trading.get("positions_count", 0) or 0)),
            "open_positions_file_count": open_positions,
            "balance": float(account.get("balance", 0.0) or 0.0),
            "equity": float(account.get("equity", 0.0) or 0.0),
            "session_pnl": float(trading.get("session_pnl", 0.0) or 0.0),
        },
        "safety_ok": safety_ok,
    }
    report["recommendation"] = _recommendation(report)
    return report


def save_monitor_report(report: dict, output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: str | Path) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _virtual_profit(row: dict[str, str]) -> float:
    if row.get("label_outcome") == "win":
        return max(_to_float(row.get("label_max_favorable")), 0.0)
    if row.get("label_outcome") == "loss":
        return -max(_to_float(row.get("label_max_adverse")), 0.0)
    return 0.0


def _inverse_virtual_profit(row: dict[str, str]) -> float:
    target = max(_to_float(row.get("target_profit")), 0.0)
    max_loss = max(_to_float(row.get("max_loss")), 0.0)
    original_favorable = max(_to_float(row.get("label_max_favorable")), 0.0)
    original_adverse = max(_to_float(row.get("label_max_adverse")), 0.0)
    seconds = int(_to_float(row.get("label_seconds_to_outcome")))
    if target > 0.0 and original_adverse >= target:
        return original_adverse
    if max_loss > 0.0 and original_favorable >= max_loss:
        return -original_favorable
    if row.get("label_outcome") == "loss" and original_adverse > 0.0:
        return original_adverse
    if row.get("label_outcome") == "win" and original_favorable > 0.0:
        return -original_favorable
    return 0.0


def _win_rate(rows: list[dict[str, str]]) -> float:
    if not rows:
        return 0.0
    wins = sum(1 for row in rows if row.get("label_outcome") == "win")
    return wins / len(rows)


def _duplicate_decision_count(rows: list[dict[str, str]]) -> int:
    seen = set()
    duplicates = 0
    for row in rows:
        key = (
            row.get("decision_time", ""),
            row.get("symbol", ""),
            row.get("direction", ""),
            row.get("entry_bid", ""),
            row.get("entry_ask", ""),
        )
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


def _bucket_summary(rows: list[dict[str, str]], field: str) -> dict[str, dict]:
    buckets: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        key = str(row.get(field, "")).strip() or "missing"
        buckets.setdefault(key, []).append(row)
    return {key: _summarize_rows(value) for key, value in sorted(buckets.items())}


def _numeric_bucket_summary(
    rows: list[dict[str, str]],
    field: str,
    thresholds: list[tuple[float, str]],
    final_label: str,
) -> dict[str, dict]:
    buckets: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        value = _to_float(row.get(field))
        label = final_label
        previous = float("-inf")
        for threshold, threshold_label in thresholds:
            if value < threshold:
                label = threshold_label
                break
            previous = threshold
        buckets.setdefault(label, []).append(row)
    return {key: _summarize_rows(value) for key, value in sorted(buckets.items())}


def _summarize_rows(rows: list[dict[str, str]]) -> dict:
    profits = [_virtual_profit(row) for row in rows]
    inverse_profits = [_inverse_virtual_profit(row) for row in rows]
    wins = sum(1 for profit in profits if profit > 0.0)
    losses = sum(1 for profit in profits if profit < 0.0)
    total = sum(profits)
    inverse_total = sum(inverse_profits)
    inverse_wins = sum(1 for profit in inverse_profits if profit > 0.0)
    return {
        "count": len(rows),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(rows) if rows else 0.0,
        "net": total,
        "avg": total / len(rows) if rows else 0.0,
        "inverse_net": inverse_total,
        "inverse_avg": inverse_total / len(rows) if rows else 0.0,
        "inverse_win_rate": inverse_wins / len(rows) if rows else 0.0,
    }


def _equity_curve_metrics(profits: list[float]) -> dict:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    max_loss_streak = 0
    current_loss_streak = 0
    first_profit_time_index = None
    for idx, profit in enumerate(profits, start=1):
        equity += profit
        if first_profit_time_index is None and equity >= 50.0:
            first_profit_time_index = idx
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
        if profit < 0.0:
            current_loss_streak += 1
            max_loss_streak = max(max_loss_streak, current_loss_streak)
        elif profit > 0.0:
            current_loss_streak = 0
    return {
        "ending_pnl": equity,
        "peak_pnl": peak,
        "max_drawdown": max_drawdown,
        "max_loss_streak": max_loss_streak,
        "reached_50": first_profit_time_index is not None,
        "trades_to_50": first_profit_time_index or 0,
    }


def _target_progress(net_pnl: float, trading: dict) -> float:
    target = _to_float(trading.get("daily_profit_target")) or 50.0
    if target <= 0.0:
        return 0.0
    return net_pnl / target


def _count_positions(path: str | Path | None) -> int:
    if path is None:
        return 0
    rows = _read_csv_rows(path)
    return len(rows)


def _recommendation(report: dict) -> str:
    if not report.get("safety_ok", False):
        return "fix_safety_before_training"
    if report.get("duplicates", 0) > 0:
        return "deduplicate_shadow_journal"
    policy = report.get("policy", {})
    if not policy.get("allowed", False):
        return "continue_shadow_training"
    return "policy_ready_for_review"


def _to_float(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
