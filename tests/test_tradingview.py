from datetime import datetime, timedelta, timezone

from src.strategy.breakout import BreakoutDirection
from src.tradingview import TradingViewAlertStore, parse_tradingview_alert


def test_parse_tradingview_alert_requires_shared_secret():
    payload = {
        "secret": "wrong",
        "symbol": "XAUUSD",
        "bias": "bullish",
        "setup": "three_drives",
        "level": 4700.5,
        "timestamp": "2026-04-06T09:30:00+00:00",
    }

    alert = parse_tradingview_alert(payload, expected_secret="correct")

    assert alert.is_valid is False
    assert alert.reason == "invalid_secret"


def test_parse_tradingview_alert_accepts_valid_bias_alert():
    payload = {
        "secret": "correct",
        "symbol": "xauusd",
        "bias": "bullish",
        "timeframe": "15",
        "setup": "three_drives",
        "level": "4700.5",
        "timestamp": "2026-04-06T09:30:00+00:00",
    }

    alert = parse_tradingview_alert(payload, expected_secret="correct")

    assert alert.is_valid is True
    assert alert.symbol == "XAUUSD"
    assert alert.direction is BreakoutDirection.BULLISH
    assert alert.setup == "three_drives"
    assert alert.level == 4700.5
    assert alert.timestamp == datetime(2026, 4, 6, 9, 30, tzinfo=timezone.utc)
    assert alert.timeframe == "15"
    assert alert.confidence == 0.5
    assert alert.context == {}


def test_parse_tradingview_alert_accepts_confidence_timeframe_and_context():
    payload = {
        "secret": "correct",
        "symbol": "xauusd",
        "bias": "bearish",
        "timeframe": "15",
        "setup": "gap_fill",
        "confidence": "0.78",
        "context": {"gap_class": "moderate", "session": "london"},
        "timestamp": "2026-04-06T09:30:00+00:00",
    }

    alert = parse_tradingview_alert(payload, expected_secret="correct")

    assert alert.is_valid is True
    assert alert.direction is BreakoutDirection.BEARISH
    assert alert.timeframe == "15"
    assert alert.confidence == 0.78
    assert alert.context == {"gap_class": "moderate", "session": "london"}


def test_alert_store_returns_fresh_matching_symbol_context():
    store = TradingViewAlertStore(max_age_seconds=300)
    now = datetime(2026, 4, 6, 9, 35, tzinfo=timezone.utc)
    alert = parse_tradingview_alert(
        {
            "secret": "correct",
            "symbol": "XAUUSD",
            "bias": "bearish",
            "setup": "h4_supply_rejection",
            "level": 4705,
            "timestamp": (now - timedelta(seconds=60)).isoformat(),
        },
        expected_secret="correct",
    )
    store.put(alert)

    context = store.latest_for("XAUUSD", now=now)

    assert context is alert


def test_alert_store_ignores_stale_alerts():
    store = TradingViewAlertStore(max_age_seconds=300)
    now = datetime(2026, 4, 6, 9, 35, tzinfo=timezone.utc)
    alert = parse_tradingview_alert(
        {
            "secret": "correct",
            "symbol": "XAUUSD",
            "bias": "bearish",
            "setup": "h4_supply_rejection",
            "level": 4705,
            "timestamp": (now - timedelta(seconds=301)).isoformat(),
        },
        expected_secret="correct",
    )
    store.put(alert)

    assert store.latest_for("XAUUSD", now=now) is None
