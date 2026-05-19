from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.strategy.breakout import BreakoutDirection


DEFAULT_ORDERFLOW_STATE_PATH = Path("orderflow_state.json")


@dataclass(frozen=True, slots=True)
class OrderflowSignal:
    symbol: str
    source_symbol: str
    timeframe: str
    timestamp: datetime
    delta_bias: float = 0.0
    cvd_slope: float = 0.0
    imbalance_score: float = 0.0
    absorption_score: float = 0.0
    profile_location_score: float = 0.0
    vwap_alignment: float = 0.0
    liquidity_obstacle_score: float = 0.0
    raw: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class OrderflowScore:
    alignment_score: float
    is_supportive: bool
    is_conflicting: bool
    metadata: dict[str, object]


def parse_orderflow_payload(payload: dict[str, Any]) -> OrderflowSignal:
    source_symbol = str(payload.get("symbol") or payload.get("source_symbol") or "").upper()
    symbol = str(payload.get("target_symbol") or payload.get("broker_symbol") or source_symbol).upper()
    if not symbol:
        raise ValueError("orderflow payload requires symbol or target_symbol")

    buy_volume = _coerce_float(payload.get("buyvolume", payload.get("buy_volume", payload.get("ask_volume"))), 0.0)
    sell_volume = _coerce_float(payload.get("sellvolume", payload.get("sell_volume", payload.get("bid_volume"))), 0.0)
    explicit_delta = payload.get("delta", payload.get("bar_delta"))
    delta = _coerce_float(explicit_delta, buy_volume - sell_volume)
    total_volume = max(buy_volume + sell_volume, abs(delta), 1.0)

    return OrderflowSignal(
        symbol=symbol,
        source_symbol=source_symbol or symbol,
        timeframe=str(payload.get("timeframe") or payload.get("tf") or "").upper(),
        timestamp=_parse_timestamp(payload.get("timestamp") or payload.get("time")),
        delta_bias=_clamp(delta / total_volume, -1.0, 1.0),
        cvd_slope=_clamp(_coerce_float(payload.get("cvd_slope", payload.get("cvdSlope")), 0.0), -1.0, 1.0),
        imbalance_score=_parse_imbalance(payload.get("imbalance", payload.get("stacked_imbalance"))),
        absorption_score=_parse_absorption(payload.get("absorption")),
        profile_location_score=_parse_profile_location(payload.get("profile_location", payload.get("market_profile"))),
        vwap_alignment=_parse_vwap_bias(payload.get("vwap_bias", payload.get("vwap"))),
        liquidity_obstacle_score=_clamp(_coerce_float(payload.get("liquidity_obstacle"), 0.0), 0.0, 1.0),
        raw=dict(payload),
    )


def score_orderflow_for_direction(signal: OrderflowSignal | None, direction: BreakoutDirection | None) -> OrderflowScore:
    if signal is None or direction is None:
        return OrderflowScore(0.0, False, False, {"orderflow_status": "missing"})

    side = 1.0 if direction is BreakoutDirection.BULLISH else -1.0
    directional_components = {
        "delta_bias": signal.delta_bias * side,
        "cvd_slope": signal.cvd_slope * side,
        "imbalance": signal.imbalance_score * side,
        "profile_location": signal.profile_location_score * side,
        "vwap_alignment": signal.vwap_alignment * side,
    }
    alignment = (
        directional_components["delta_bias"] * 0.25
        + directional_components["cvd_slope"] * 0.25
        + directional_components["imbalance"] * 0.20
        + directional_components["profile_location"] * 0.10
        + directional_components["vwap_alignment"] * 0.15
        + signal.absorption_score * 0.10
        - signal.liquidity_obstacle_score * 0.20
    )
    alignment = _clamp(alignment, -1.0, 1.0)
    return OrderflowScore(
        alignment_score=alignment,
        is_supportive=alignment >= 0.35,
        is_conflicting=alignment <= -0.35,
        metadata={
            "orderflow_symbol": signal.symbol,
            "orderflow_source_symbol": signal.source_symbol,
            "orderflow_timeframe": signal.timeframe,
            "orderflow_timestamp": signal.timestamp.isoformat(),
            "orderflow_alignment": alignment,
            **directional_components,
            "absorption_score": signal.absorption_score,
            "liquidity_obstacle_score": signal.liquidity_obstacle_score,
        },
    )


class OrderflowSignalStore:
    def __init__(self, path: str | Path = DEFAULT_ORDERFLOW_STATE_PATH) -> None:
        self.path = Path(path)

    def record(self, signal: OrderflowSignal) -> None:
        state = self._read_state()
        state[signal.symbol.upper()] = _serialize_signal(signal)
        self.path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    def latest_for(self, symbol: str, now: datetime | None = None, max_age_seconds: int = 900) -> OrderflowSignal | None:
        item = self._read_state().get(str(symbol).upper())
        if not item:
            return None
        signal = _deserialize_signal(item)
        reference_time = now or datetime.now(timezone.utc)
        if reference_time.tzinfo is None or reference_time.utcoffset() is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)
        age = reference_time.astimezone(timezone.utc) - signal.timestamp
        if age.total_seconds() > max_age_seconds:
            return None
        return signal

    def _read_state(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}


def _serialize_signal(signal: OrderflowSignal) -> dict[str, Any]:
    data = asdict(signal)
    data["timestamp"] = signal.timestamp.isoformat()
    return data


def _deserialize_signal(data: dict[str, Any]) -> OrderflowSignal:
    return OrderflowSignal(
        symbol=str(data.get("symbol", "")).upper(),
        source_symbol=str(data.get("source_symbol", "")).upper(),
        timeframe=str(data.get("timeframe", "")).upper(),
        timestamp=_parse_timestamp(data.get("timestamp")),
        delta_bias=_coerce_float(data.get("delta_bias"), 0.0),
        cvd_slope=_coerce_float(data.get("cvd_slope"), 0.0),
        imbalance_score=_coerce_float(data.get("imbalance_score"), 0.0),
        absorption_score=_coerce_float(data.get("absorption_score"), 0.0),
        profile_location_score=_coerce_float(data.get("profile_location_score"), 0.0),
        vwap_alignment=_coerce_float(data.get("vwap_alignment"), 0.0),
        liquidity_obstacle_score=_coerce_float(data.get("liquidity_obstacle_score"), 0.0),
        raw=data.get("raw") if isinstance(data.get("raw"), dict) else None,
    )


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if value is None or value == "":
        return datetime.now(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _parse_imbalance(value: Any) -> float:
    if isinstance(value, (int, float)):
        return _clamp(float(value), -1.0, 1.0)
    text = str(value or "").strip().lower()
    if "sell" in text:
        return -0.85 if "stack" in text else -0.55
    if "buy" in text:
        return 0.85 if "stack" in text else 0.55
    return 0.0


def _parse_absorption(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return _clamp(float(value), 0.0, 1.0)
    text = str(value or "").strip().lower()
    return 1.0 if text in {"true", "yes", "absorption", "1"} else 0.0


def _parse_profile_location(value: Any) -> float:
    if isinstance(value, (int, float)):
        return _clamp(float(value), -1.0, 1.0)
    text = str(value or "").strip().lower()
    if "below" in text:
        return -0.6
    if "above" in text:
        return 0.6
    if "inside" in text or "value" in text:
        return 0.0
    return 0.0


def _parse_vwap_bias(value: Any) -> float:
    if isinstance(value, (int, float)):
        return _clamp(float(value), -1.0, 1.0)
    text = str(value or "").strip().lower()
    if "below" in text:
        return -0.8
    if "above" in text:
        return 0.8
    return 0.0


def _coerce_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
