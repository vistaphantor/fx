from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from src.strategy.breakout import BreakoutDirection


DEFAULT_MAX_WEBHOOK_BODY_BYTES = 16 * 1024


@dataclass(frozen=True, slots=True)
class TradingViewAlert:
    is_valid: bool
    reason: str
    symbol: str | None = None
    direction: BreakoutDirection | None = None
    setup: str | None = None
    level: float | None = None
    timestamp: datetime | None = None
    timeframe: str | None = None
    confidence: float = 0.5
    context: dict[str, Any] | None = None


class TradingViewAlertStore:
    def __init__(self, *, max_age_seconds: int) -> None:
        self.max_age_seconds = max_age_seconds
        self._alerts: dict[str, TradingViewAlert] = {}
        self._lock = threading.Lock()

    def put(self, alert: TradingViewAlert) -> None:
        if not alert.is_valid or alert.symbol is None:
            return
        with self._lock:
            self._alerts[alert.symbol] = alert

    def latest_for(self, symbol: str, *, now: datetime | None = None) -> TradingViewAlert | None:
        lookup_symbol = symbol.strip().upper()
        current_time = _utc_now() if now is None else _as_utc(now)
        with self._lock:
            alert = self._alerts.get(lookup_symbol)
        if alert is None or alert.timestamp is None:
            return None
        age_seconds = (current_time - alert.timestamp).total_seconds()
        if age_seconds < 0 or age_seconds > self.max_age_seconds:
            return None
        return alert


def parse_tradingview_alert(payload: dict[str, Any], *, expected_secret: str) -> TradingViewAlert:
    if str(payload.get("secret", "")) != expected_secret:
        return TradingViewAlert(is_valid=False, reason="invalid_secret")

    symbol = str(payload.get("symbol", "")).strip().upper()
    if not symbol:
        return TradingViewAlert(is_valid=False, reason="missing_symbol")

    direction = _parse_direction(payload.get("bias") or payload.get("direction"))
    if direction is None:
        return TradingViewAlert(is_valid=False, reason="invalid_direction")

    timestamp = _parse_timestamp(payload.get("timestamp"))
    if timestamp is None:
        return TradingViewAlert(is_valid=False, reason="invalid_timestamp")

    level = _parse_optional_float(payload.get("level"))
    if payload.get("level") not in {None, ""} and level is None:
        return TradingViewAlert(is_valid=False, reason="invalid_level")
    confidence = _parse_confidence(payload.get("confidence"))
    if confidence is None:
        return TradingViewAlert(is_valid=False, reason="invalid_confidence")
    timeframe = _parse_optional_string(payload.get("timeframe"))
    context = _parse_context(payload.get("context"))
    if context is None:
        return TradingViewAlert(is_valid=False, reason="invalid_context")

    return TradingViewAlert(
        is_valid=True,
        reason="accepted",
        symbol=symbol,
        direction=direction,
        setup=str(payload.get("setup", "tradingview_alert")).strip() or "tradingview_alert",
        level=level,
        timestamp=timestamp,
        timeframe=timeframe,
        confidence=confidence,
        context=context,
    )


def start_tradingview_webhook_server(
    *,
    host: str,
    port: int,
    expected_secret: str,
    store: TradingViewAlertStore,
    max_body_bytes: int = DEFAULT_MAX_WEBHOOK_BODY_BYTES,
    log_fn=print,
) -> ThreadingHTTPServer:
    class TradingViewWebhookHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/tradingview":
                self.send_error(404, "Not Found")
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(400, "Invalid Content-Length")
                return
            if content_length < 0:
                self.send_error(400, "Invalid Content-Length")
                return
            if content_length > max_body_bytes:
                self.send_error(413, "Payload Too Large")
                return
            try:
                body = self.rfile.read(content_length)
                payload = json.loads(body.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.send_error(400, "Invalid JSON")
                return

            alert = parse_tradingview_alert(payload, expected_secret=expected_secret)
            if not alert.is_valid:
                self.send_error(400, alert.reason)
                return

            store.put(alert)
            log_fn(
                "TradingView alert accepted "
                f"symbol={alert.symbol} direction={alert.direction.value if alert.direction else None} "
                f"setup={alert.setup} level={alert.level}"
            )
            self.send_response(202)
            self.end_headers()
            self.wfile.write(b"accepted")

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), TradingViewWebhookHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _parse_direction(value: Any) -> BreakoutDirection | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"buy", "bull", "bullish", "long"}:
        return BreakoutDirection.BULLISH
    if normalized in {"sell", "bear", "bearish", "short"}:
        return BreakoutDirection.BEARISH
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    if value in {None, ""}:
        return _utc_now()
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return _as_utc(parsed)


def _parse_optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_optional_string(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value).strip() or None


def _parse_confidence(value: Any) -> float | None:
    if value is None or value == "":
        return 0.5
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= parsed <= 1.0:
        return parsed
    return None


def _parse_context(value: Any) -> dict[str, Any] | None:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    return None


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
