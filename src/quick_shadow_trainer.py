from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ShadowPolicy:
    min_estimated_profit: float
    min_consistency: float
    max_pullback_ratio: float
    require_quality_allowed: bool
    allowed_quality_reasons: tuple[str, ...] = ()
    invert_direction: bool = False


@dataclass(frozen=True)
class ShadowPolicyReport:
    policy: ShadowPolicy
    sample_count: int
    selected_count: int
    win_rate: float
    expectancy: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_loss_streak: int
    allowed: bool
    reason: str
    validation_samples: int = 0
    validation_selected_count: int = 0
    validation_win_rate: float = 0.0
    validation_expectancy: float = 0.0
    validation_profit_factor: float = 0.0
    validation_max_loss_streak: int = 0
    validation_allowed: bool = False
    validation_reason: str = "not_validated"

    def to_dict(self) -> dict:
        return {
            "policy": {
                "min_estimated_profit": self.policy.min_estimated_profit,
                "min_consistency": self.policy.min_consistency,
                "max_pullback_ratio": self.policy.max_pullback_ratio,
                "require_quality_allowed": self.policy.require_quality_allowed,
                "allowed_quality_reasons": list(self.policy.allowed_quality_reasons),
                "invert_direction": self.policy.invert_direction,
            },
            "sample_count": self.sample_count,
            "selected_count": self.selected_count,
            "win_rate": self.win_rate,
            "expectancy": self.expectancy,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "max_loss_streak": self.max_loss_streak,
            "allowed": self.allowed,
            "reason": self.reason,
            "validation": {
                "samples": self.validation_samples,
                "selected_count": self.validation_selected_count,
                "win_rate": self.validation_win_rate,
                "expectancy": self.validation_expectancy,
                "profit_factor": self.validation_profit_factor,
                "max_loss_streak": self.validation_max_loss_streak,
                "allowed": self.validation_allowed,
                "reason": self.validation_reason,
            },
        }


def load_resolved_shadow_rows(path: str | Path, *, symbol: str = "") -> list[dict[str, str]]:
    journal_path = Path(path)
    if not journal_path.exists():
        return []
    rows: list[dict[str, str]] = []
    with journal_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "resolved":
                continue
            if row.get("label_outcome") not in {"win", "loss"}:
                continue
            if symbol and row.get("symbol") != symbol:
                continue
            rows.append(dict(row))
    return rows


def train_shadow_policy(
    rows: list[dict[str, str]],
    *,
    min_samples: int = 60,
    min_selected: int = 20,
    min_win_rate: float = 0.58,
    min_expectancy: float = 0.02,
    min_profit_factor: float = 1.20,
    max_loss_streak: int = 4,
    validation_fraction: float = 0.30,
    min_validation_selected: int = 8,
) -> ShadowPolicyReport:
    train_rows, validation_rows = _walk_forward_split(rows, validation_fraction=validation_fraction)
    selection_rows = train_rows if validation_rows else rows
    policies = _candidate_policies(selection_rows)
    best_report: ShadowPolicyReport | None = None
    for policy in policies:
        report = evaluate_shadow_policy(
            selection_rows,
            policy=policy,
            min_samples=min(min_samples, len(selection_rows)),
            min_selected=min_selected,
            min_win_rate=min_win_rate,
            min_expectancy=min_expectancy,
            min_profit_factor=min_profit_factor,
            max_loss_streak=max_loss_streak,
        )
        if best_report is None or _report_score(report, min_selected=min_selected) > _report_score(best_report, min_selected=min_selected):
            best_report = report

    if best_report is None:
        empty_policy = ShadowPolicy(0.0, 0.0, 1.0, True, (), False)
        return ShadowPolicyReport(empty_policy, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, False, "no_resolved_shadow_samples")

    full_report = evaluate_shadow_policy(
        rows,
        policy=best_report.policy,
        min_samples=min_samples,
        min_selected=min_selected,
        min_win_rate=min_win_rate,
        min_expectancy=min_expectancy,
        min_profit_factor=min_profit_factor,
        max_loss_streak=max_loss_streak,
    )
    validation_report = _validation_report(
        validation_rows,
        policy=best_report.policy,
        min_selected=min_validation_selected,
        min_win_rate=min_win_rate,
        min_expectancy=min_expectancy,
        min_profit_factor=min_profit_factor,
        max_loss_streak=max_loss_streak,
    )
    allowed = full_report.allowed and validation_report.allowed
    reason = full_report.reason if full_report.reason != "shadow_policy_ok" else (
        "shadow_policy_ok" if validation_report.allowed else validation_report.reason
    )
    return ShadowPolicyReport(
        full_report.policy,
        full_report.sample_count,
        full_report.selected_count,
        full_report.win_rate,
        full_report.expectancy,
        full_report.avg_win,
        full_report.avg_loss,
        full_report.profit_factor,
        full_report.max_loss_streak,
        allowed,
        reason,
        validation_samples=validation_report.sample_count,
        validation_selected_count=validation_report.selected_count,
        validation_win_rate=validation_report.win_rate,
        validation_expectancy=validation_report.expectancy,
        validation_profit_factor=validation_report.profit_factor,
        validation_max_loss_streak=validation_report.max_loss_streak,
        validation_allowed=validation_report.allowed,
        validation_reason=validation_report.reason,
    )


def evaluate_shadow_policy(
    rows: list[dict[str, str]],
    *,
    policy: ShadowPolicy,
    min_samples: int = 60,
    min_selected: int = 20,
    min_win_rate: float = 0.58,
    min_expectancy: float = 0.02,
    min_profit_factor: float = 1.20,
    max_loss_streak: int = 4,
) -> ShadowPolicyReport:
    selected = [row for row in rows if _row_matches_policy(row, policy)]
    profits = [_row_profit(row, invert_direction=policy.invert_direction) for row in selected]
    wins = [profit for profit in profits if profit > 0.0]
    losses = [profit for profit in profits if profit < 0.0]
    selected_count = len(selected)
    win_rate = len(wins) / selected_count if selected_count else 0.0
    expectancy = sum(profits) / selected_count if selected_count else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_win / gross_loss if gross_loss > 0.0 else (999.0 if gross_win > 0.0 else 0.0)
    loss_streak = _max_loss_streak(profits)
    allowed = (
        len(rows) >= int(min_samples)
        and selected_count >= int(min_selected)
        and win_rate >= float(min_win_rate)
        and expectancy >= float(min_expectancy)
        and profit_factor >= float(min_profit_factor)
        and loss_streak <= int(max_loss_streak)
    )
    reason = "shadow_policy_ok" if allowed else _policy_rejection_reason(
        len(rows),
        selected_count,
        win_rate,
        expectancy,
        profit_factor,
        loss_streak,
        min_samples=min_samples,
        min_selected=min_selected,
        min_win_rate=min_win_rate,
        min_expectancy=min_expectancy,
        min_profit_factor=min_profit_factor,
        max_loss_streak=max_loss_streak,
    )
    return ShadowPolicyReport(
        policy,
        len(rows),
        selected_count,
        win_rate,
        expectancy,
        avg_win,
        avg_loss,
        profit_factor,
        loss_streak,
        allowed,
        reason,
    )


def _walk_forward_split(rows: list[dict[str, str]], *, validation_fraction: float) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if float(validation_fraction) <= 0.0:
        return rows, []
    if len(rows) < 20:
        return rows, []
    fraction = min(max(float(validation_fraction), 0.10), 0.50)
    validation_size = max(1, int(len(rows) * fraction))
    if len(rows) - validation_size < 10:
        return rows, []
    return rows[:-validation_size], rows[-validation_size:]


def _validation_report(
    rows: list[dict[str, str]],
    *,
    policy: ShadowPolicy,
    min_selected: int,
    min_win_rate: float,
    min_expectancy: float,
    min_profit_factor: float,
    max_loss_streak: int,
) -> ShadowPolicyReport:
    if not rows:
        return ShadowPolicyReport(
            policy,
            0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            True,
            "not_validated",
            validation_allowed=True,
            validation_reason="not_validated",
        )
    return evaluate_shadow_policy(
        rows,
        policy=policy,
        min_samples=1,
        min_selected=min_selected,
        min_win_rate=min_win_rate,
        min_expectancy=min_expectancy,
        min_profit_factor=min_profit_factor,
        max_loss_streak=max_loss_streak,
    )


def save_shadow_policy_report(report: ShadowPolicyReport, output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_shadow_policy_payload(path: str | Path) -> dict:
    policy_path = Path(path)
    if not policy_path.exists():
        return {}
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def policy_from_payload(payload: dict) -> ShadowPolicy | None:
    policy = payload.get("policy") if isinstance(payload, dict) else None
    if not isinstance(policy, dict):
        return None
    return ShadowPolicy(
        min_estimated_profit=_to_float(policy.get("min_estimated_profit")),
        min_consistency=_to_float(policy.get("min_consistency")),
        max_pullback_ratio=_to_float(policy.get("max_pullback_ratio")) or 1.0,
        require_quality_allowed=bool(policy.get("require_quality_allowed", True)),
        allowed_quality_reasons=tuple(
            str(reason)
            for reason in (policy.get("allowed_quality_reasons") or [])
            if str(reason).strip()
        ),
        invert_direction=bool(policy.get("invert_direction", False)),
    )


def shadow_policy_allows_features(features: dict[str, object], policy: ShadowPolicy) -> bool:
    row = {key: str(value) for key, value in features.items()}
    return _row_matches_policy(row, policy)


def _candidate_policies(rows: list[dict[str, str]]) -> list[ShadowPolicy]:
    estimated_values = sorted({_to_float(row.get("estimated_tick_profit")) for row in rows})
    estimated_values = [value for value in estimated_values if value >= 0.0]
    if not estimated_values:
        estimated_values = [0.0]
    thresholds = sorted(set([0.0, 0.03, 0.05, 0.07, 0.10] + estimated_values))
    policies: list[ShadowPolicy] = []
    quality_reason_options: list[tuple[str, ...]] = [()]
    reasons = sorted({str(row.get("quality_reason", "")).strip() for row in rows if str(row.get("quality_reason", "")).strip()})
    quality_reason_options.extend((reason,) for reason in reasons)
    for min_profit in thresholds:
        for min_consistency in (0.0, 0.55, 0.60, 0.67, 0.75):
            for max_pullback in (1.00, 0.75, 0.65, 0.50, 0.35):
                for require_quality in (True, False):
                    for allowed_reasons in quality_reason_options:
                        policies.append(ShadowPolicy(min_profit, min_consistency, max_pullback, require_quality, allowed_reasons, False))
                        policies.append(ShadowPolicy(min_profit, min_consistency, max_pullback, require_quality, allowed_reasons, True))
    return policies


def _row_matches_policy(row: dict[str, str], policy: ShadowPolicy) -> bool:
    if _to_float(row.get("estimated_tick_profit")) < policy.min_estimated_profit:
        return False
    if _to_float(row.get("tick_directional_consistency")) < policy.min_consistency:
        return False
    pullback = _to_float(row.get("quality_pullback_ratio"))
    if pullback > policy.max_pullback_ratio:
        return False
    if policy.require_quality_allowed and str(row.get("quality_allowed", "")).strip() not in {"1", "true", "True"}:
        return False
    if policy.allowed_quality_reasons and str(row.get("quality_reason", "")).strip() not in policy.allowed_quality_reasons:
        return False
    return True


def _row_profit(row: dict[str, str], *, invert_direction: bool = False) -> float:
    if invert_direction:
        return _inverse_row_profit(row)
    if row.get("label_outcome") == "win":
        return max(_to_float(row.get("label_max_favorable")), 0.0)
    if row.get("label_outcome") == "loss":
        return -max(_to_float(row.get("label_max_adverse")), 0.0)
    return 0.0


def _inverse_row_profit(row: dict[str, str]) -> float:
    target = max(_to_float(row.get("target_profit")), 0.0)
    max_loss = max(_to_float(row.get("max_loss")), 0.0)
    original_favorable = max(_to_float(row.get("label_max_favorable")), 0.0)
    original_adverse = max(_to_float(row.get("label_max_adverse")), 0.0)
    if target > 0.0 and original_adverse >= target:
        return original_adverse
    if max_loss > 0.0 and original_favorable >= max_loss:
        return -original_favorable
    if row.get("label_outcome") == "loss" and original_adverse > 0.0:
        return original_adverse
    if row.get("label_outcome") == "win" and original_favorable > 0.0:
        return -original_favorable
    return 0.0


def _report_score(report: ShadowPolicyReport, *, min_selected: int = 20) -> tuple:
    return (
        int(report.allowed),
        int(report.selected_count >= int(min_selected)),
        min(report.selected_count, int(min_selected)),
        round(report.expectancy, 8),
        round(report.profit_factor, 8),
        round(report.win_rate, 8),
        -report.max_loss_streak,
    )


def _policy_rejection_reason(
    sample_count: int,
    selected_count: int,
    win_rate: float,
    expectancy: float,
    profit_factor: float,
    loss_streak: int,
    *,
    min_samples: int,
    min_selected: int,
    min_win_rate: float,
    min_expectancy: float,
    min_profit_factor: float,
    max_loss_streak: int,
) -> str:
    if sample_count < min_samples:
        return "insufficient_shadow_samples"
    if selected_count < min_selected:
        return "insufficient_selected_shadow_trades"
    if win_rate < min_win_rate:
        return "shadow_policy_low_win_rate"
    if expectancy < min_expectancy:
        return "shadow_policy_low_expectancy"
    if profit_factor < min_profit_factor:
        return "shadow_policy_low_profit_factor"
    if loss_streak > max_loss_streak:
        return "shadow_policy_loss_streak_too_high"
    return "shadow_policy_unproven"


def _max_loss_streak(profits: list[float]) -> int:
    current = 0
    longest = 0
    for profit in profits:
        if profit < 0.0:
            current += 1
            longest = max(longest, current)
        elif profit > 0.0:
            current = 0
    return longest


def _to_float(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
