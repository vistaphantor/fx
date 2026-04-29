from __future__ import annotations

from dataclasses import dataclass, field

from src.strategy.breakout import BreakoutDirection
from src.strategy.gap import GapDecision
from src.tradingview import TradingViewAlert


@dataclass(frozen=True, slots=True)
class TradingViewConfluence:
    is_active: bool
    reason: str
    direction_bonus: int = 0
    direction_penalty: int = 0
    setup_bonus: int = 0
    trigger_bonus: int = 0
    preferred_direction: BreakoutDirection | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def build_tradingview_confluence(
    *,
    symbol: str,
    direction: BreakoutDirection,
    gap_decision: GapDecision,
    alert: TradingViewAlert | None,
) -> TradingViewConfluence:
    if alert is None or not alert.is_valid or alert.symbol is None:
        return TradingViewConfluence(is_active=False, reason="tradingview_alert_missing")
    if alert.symbol != symbol.strip().upper():
        return TradingViewConfluence(is_active=False, reason="tradingview_symbol_mismatch")

    base_bonus = 2 if float(alert.confidence) >= 0.75 else 1
    direction_penalty = 1 if float(alert.confidence) >= 0.6 else 0
    metadata = {
        "setup": alert.setup,
        "confidence": alert.confidence,
        "timeframe": alert.timeframe,
        "context": alert.context or {},
    }

    if alert.direction is direction:
        reason = "tradingview_direction_alignment"
        setup_bonus = 0
        trigger_bonus = 0
        direction_bonus = base_bonus
        if (
            alert.setup == "gap_fill"
            and gap_decision.has_gap
            and gap_decision.fill_preferred
            and gap_decision.preferred_trade_direction is direction
        ):
            reason = "tradingview_gap_fill_alignment"
            direction_bonus += 1
            setup_bonus = 1
        elif (
            alert.setup == "gap_continuation"
            and gap_decision.has_gap
            and not gap_decision.fill_preferred
            and gap_decision.gap_direction is direction
        ):
            reason = "tradingview_gap_continuation_alignment"
            direction_bonus += 1
            trigger_bonus = 1
        elif alert.setup in {"three_drives", "session_sweep", "vp_reaction", "bos", "choch"}:
            setup_bonus = 1

        return TradingViewConfluence(
            is_active=True,
            reason=reason,
            direction_bonus=direction_bonus,
            direction_penalty=0,
            setup_bonus=setup_bonus,
            trigger_bonus=trigger_bonus,
            preferred_direction=direction,
            metadata=metadata,
        )

    return TradingViewConfluence(
        is_active=True,
        reason="tradingview_direction_conflict",
        direction_bonus=0,
        direction_penalty=direction_penalty,
        setup_bonus=0,
        trigger_bonus=0,
        preferred_direction=alert.direction,
        metadata=metadata,
    )
