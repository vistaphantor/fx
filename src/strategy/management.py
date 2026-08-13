from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.strategy.breakout import BreakoutDirection

_INITIAL_STOP_CACHE: dict[tuple[object, ...], float] = {}
_INITIAL_STOP_CACHE_PATH = Path("data") / "initial_stop_cache.json"


@dataclass(frozen=True, slots=True)
class CampaignAction:
    action: str
    reason: str
    new_stop_loss: float | None = None
    stop_updates: tuple[tuple[int, float], ...] = ()
    add_lot: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def evaluate_fixed_trailing_stop(*, position, current_price: float, direction: BreakoutDirection, trail_distance: float = 1.0) -> float | None:
    """Computes a new stop loss based on a fixed distance from the peak price."""
    entry_price = _position_entry_price(position)
    current_sl = _position_current_stop_loss(position)
    
    if direction is BreakoutDirection.BULLISH:
        # For BUYS: SL = CurrentPrice - Distance
        # We only trail if the new SL is tighter than the current SL
        candidate_sl = current_price - trail_distance
        if current_sl is None or candidate_sl > current_sl:
            return candidate_sl
    else:
        # For SELLS: SL = CurrentPrice + Distance
        candidate_sl = current_price + trail_distance
        if current_sl is None or candidate_sl < current_sl:
            return candidate_sl
            
    return None


def remember_position_initial_stop_loss(position, stop_loss: float | None = None, entry_price: float | None = None) -> None:
    resolved_entry_price = float(entry_price) if entry_price is not None else _position_entry_price(position)
    resolved_stop = stop_loss
    if resolved_stop is None:
        if hasattr(position, "initial_stop_loss"):
            resolved_stop = float(position.initial_stop_loss)
        else:
            resolved_stop = _position_current_stop_loss(position)
    if resolved_stop is None:
        return
    _remember_initial_stop_by_key(_position_identity(position, resolved_entry_price), float(resolved_stop))


def evaluate_campaign_action(
    *,
    positions,
    current_price,
    direction,
    latest_trade_r_multiple,
    default_lot,
    add_on_lot_increment,
    max_exposure_pct,
    margin_snapshot,
    reversal_confirmed,
    continuation_edge=None,
    continuation_threshold=None,
    breakeven_distance: float = 1.5,
    campaign_add_floor_r: float = 1.25,
):
    if reversal_confirmed:
        return CampaignAction(
            action="close_all",
            reason="reversal_confirmed_exit",
            metadata={"current_price": current_price},
        )

    stop_updates, stop_reason, stop_metadata = _build_stop_updates(
        positions=positions,
        current_price=current_price,
        direction=direction,
        breakeven_distance=breakeven_distance,
    )
    if stop_updates:
        return CampaignAction(
            action="trail_all",
            reason=stop_reason,
            new_stop_loss=stop_updates[-1][1] if len({stop_loss for _, stop_loss in stop_updates}) == 1 else None,
            stop_updates=tuple(stop_updates),
            metadata=stop_metadata,
        )

    if latest_trade_r_multiple >= campaign_add_floor_r:
        campaign_exposure_pct = float(margin_snapshot["campaign_exposure_pct"])
        preferred_add_exposure_pct = float(margin_snapshot["preferred_add_exposure_pct"])
        fallback_add_exposure_pct = float(margin_snapshot["fallback_add_exposure_pct"])
        preferred_lot = round(default_lot + add_on_lot_increment, 2)
        if campaign_exposure_pct + preferred_add_exposure_pct <= max_exposure_pct:
            return CampaignAction(
                action="add_position",
                reason="campaign_add_ready",
                add_lot=preferred_lot,
                metadata={
                    "campaign_exposure_pct": campaign_exposure_pct,
                    "continuation_edge": continuation_edge,
                    "continuation_threshold": continuation_threshold,
                },
            )
        if campaign_exposure_pct + fallback_add_exposure_pct <= max_exposure_pct:
            return CampaignAction(
                action="add_position",
                reason="campaign_add_ready",
                add_lot=default_lot,
                metadata={
                    "campaign_exposure_pct": campaign_exposure_pct,
                    "continuation_edge": continuation_edge,
                    "continuation_threshold": continuation_threshold,
                },
            )
        return CampaignAction(
            action="hold",
            reason="campaign_exposure_limit_reached",
            metadata={"campaign_exposure_pct": campaign_exposure_pct},
        )

    if latest_trade_r_multiple >= 1.5:
        latest_position = positions[-1]
        entry_price = _position_entry_price(latest_position)
        initial_stop_loss = _position_initial_stop_loss(latest_position, entry_price)
        risk = abs(entry_price - initial_stop_loss)
        if direction is BreakoutDirection.BULLISH:
            new_stop_loss = entry_price + (risk * 0.25)
        else:
            new_stop_loss = entry_price - (risk * 0.25)
        return CampaignAction(
            action="trail_all",
            reason="campaign_trail_progression",
            new_stop_loss=new_stop_loss,
            metadata={"risk": risk},
        )

    return CampaignAction(
        action="hold",
        reason="campaign_waiting_for_progress",
        metadata={"current_price": current_price},
    )


def _build_stop_updates(*, positions, current_price: float, direction: BreakoutDirection, breakeven_distance: float):
    stop_updates: list[tuple[int, float]] = []
    progress_by_index: dict[int, float] = {}
    locked_r_by_index: dict[int, int] = {}
    breakeven_any = False
    profit_lock_any = False

    for index, position in enumerate(positions):
        entry_price = _position_entry_price(position)
        initial_stop_loss = _position_initial_stop_loss(position, entry_price)
        current_stop_loss = _position_current_stop_loss(position)
        risk = abs(entry_price - initial_stop_loss)
        if risk <= 0:
            continue

        progress_r = _position_progress_r(
            position=position,
            current_price=current_price,
            direction=direction,
        )
        progress_by_index[index] = progress_r
        target_stop_loss = None

        # --- Dynamic Profit-Trailing Ladder ---
        # 1. Early risk reduction: at +0.5 R profit, cut risk by 50%
        if progress_r >= 0.5:
            half_risk_sl = entry_price + (0.5 * (initial_stop_loss - entry_price))
            target_stop_loss = _more_protective_stop_loss(
                existing_stop=target_stop_loss,
                candidate_stop=half_risk_sl,
                direction=direction,
            )

        # 2. Breakeven lock: at +1.0 R profit (or breakeven_trigger_r), move SL to entry
        breakeven_trigger_r = _breakeven_trigger_r(breakeven_distance=breakeven_distance, risk=risk)
        if progress_r >= min(1.0, breakeven_trigger_r):
            target_stop_loss = _more_protective_stop_loss(
                existing_stop=target_stop_loss,
                candidate_stop=entry_price,
                direction=direction,
            )
            breakeven_any = True

        # 3. Profit locking: at +1.5 R profit, lock +0.5 R profit
        if progress_r >= 1.5:
            lock_05_sl = entry_price + (0.5 * risk) if direction is BreakoutDirection.BULLISH else entry_price - (0.5 * risk)
            target_stop_loss = _more_protective_stop_loss(
                existing_stop=target_stop_loss,
                candidate_stop=lock_05_sl,
                direction=direction,
            )
            profit_lock_any = True

        # 4. Dynamic trailing: at >= +2.0 R profit, trail SL in integer R step increments behind peak
        if progress_r >= 2.0:
            locked_r = _locked_r_multiple(progress_r)
            locked_r_by_index[index] = locked_r
            if locked_r > 0:
                dynamic_sl = _locked_stop_loss(
                    entry_price=entry_price,
                    direction=direction,
                    risk=risk,
                    locked_r=locked_r,
                )
                target_stop_loss = _more_protective_stop_loss(
                    existing_stop=target_stop_loss,
                    candidate_stop=dynamic_sl,
                    direction=direction,
                )
                profit_lock_any = True

        if target_stop_loss is None:
            continue
        if _should_improve_stop_loss(current_stop_loss=current_stop_loss, new_stop_loss=target_stop_loss, direction=direction):
            stop_updates.append((index, target_stop_loss))

    reason = "campaign_profit_lock_progression" if profit_lock_any else "campaign_breakeven_earned"
    metadata = {
        "progress_by_index": progress_by_index,
        "locked_r_by_index": locked_r_by_index,
        "breakeven_distance": breakeven_distance,
    }
    return stop_updates, reason, metadata


def _breakeven_trigger_r(*, breakeven_distance: float, risk: float) -> float:
    return float(breakeven_distance) / risk


def _locked_r_multiple(progress_r: float) -> int:
    return max(int(progress_r) - 1, 0)


def _locked_stop_loss(*, entry_price: float, direction: BreakoutDirection, risk: float, locked_r: int) -> float:
    if direction is BreakoutDirection.BULLISH:
        return entry_price + (locked_r * risk)
    return entry_price - (locked_r * risk)


def _more_protective_stop_loss(*, existing_stop: float | None, candidate_stop: float, direction: BreakoutDirection) -> float:
    if existing_stop is None:
        return candidate_stop
    if direction is BreakoutDirection.BULLISH:
        return max(existing_stop, candidate_stop)
    return min(existing_stop, candidate_stop)


def _should_improve_stop_loss(*, current_stop_loss: float | None, new_stop_loss: float, direction: BreakoutDirection) -> bool:
    if current_stop_loss is None or current_stop_loss == 0.0:
        return True
    if direction is BreakoutDirection.BULLISH:
        return new_stop_loss > current_stop_loss
    return new_stop_loss < current_stop_loss


def _position_progress_r(*, position, current_price: float, direction: BreakoutDirection) -> float:
    entry_price = _position_entry_price(position)
    initial_stop_loss = _position_initial_stop_loss(position, entry_price)
    risk = abs(entry_price - initial_stop_loss)
    if risk == 0:
        return 0.0
    if direction is BreakoutDirection.BULLISH:
        return (current_price - entry_price) / risk
    return (entry_price - current_price) / risk


def _position_entry_price(position) -> float:
    if hasattr(position, "entry_price"):
        return float(position.entry_price)
    if hasattr(position, "price_open"):
        return float(position.price_open)
    raise AttributeError("Position object is missing entry price fields")


def _position_initial_stop_loss(position, entry_price: float) -> float:
    if hasattr(position, "initial_stop_loss"):
        initial_stop_loss = float(position.initial_stop_loss)
        _remember_initial_stop_by_key(_position_identity(position, entry_price), initial_stop_loss)
        return initial_stop_loss
    position_key = _position_identity(position, entry_price)
    _load_initial_stop_cache()
    cached_stop_loss = _INITIAL_STOP_CACHE.get(position_key)
    if cached_stop_loss is not None:
        return cached_stop_loss
    current_stop_loss = _position_current_stop_loss(position)
    if current_stop_loss is not None:
        _remember_initial_stop_by_key(position_key, current_stop_loss)
        return current_stop_loss
    return entry_price


def _position_current_stop_loss(position) -> float | None:
    if hasattr(position, "stop_loss"):
        return float(position.stop_loss)
    if hasattr(position, "sl"):
        return float(position.sl)
    return None


def _position_identity(position, entry_price: float) -> tuple[object, ...]:
    ticket = getattr(position, "ticket", None)
    symbol = getattr(position, "symbol", None)
    volume = getattr(position, "volume", None)
    position_type = getattr(position, "type", None)
    return (ticket, symbol, round(float(entry_price), 10), volume, position_type)


def _remember_initial_stop_by_key(position_key: tuple[object, ...], stop_loss: float) -> None:
    _load_initial_stop_cache()
    _INITIAL_STOP_CACHE[position_key] = float(stop_loss)
    _save_initial_stop_cache()


def _load_initial_stop_cache() -> None:
    path = Path(_INITIAL_STOP_CACHE_PATH)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return
    if not isinstance(data, dict):
        return
    for encoded_key, stop_loss in data.items():
        try:
            key_parts = json.loads(encoded_key)
            if isinstance(key_parts, list):
                _INITIAL_STOP_CACHE[tuple(key_parts)] = float(stop_loss)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue


def _save_initial_stop_cache() -> None:
    path = Path(_INITIAL_STOP_CACHE_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            json.dumps(list(position_key), separators=(",", ":")): stop_loss
            for position_key, stop_loss in _INITIAL_STOP_CACHE.items()
        }
        path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    except OSError:
        return
