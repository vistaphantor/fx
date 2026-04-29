"""Session-based trade helpers.

The strategy only trades during the London and New York sessions.
All checks operate on timezone-aware UTC datetimes so the boundaries
stay deterministic regardless of the machine's local timezone.
"""

from __future__ import annotations

from datetime import datetime, time, timezone

LONDON_SESSION_START = time(7, 0, tzinfo=timezone.utc)
LONDON_SESSION_END = time(11, 0, tzinfo=timezone.utc)
NEW_YORK_SESSION_START = time(12, 0, tzinfo=timezone.utc)
NEW_YORK_SESSION_END = time(16, 0, tzinfo=timezone.utc)


def _is_within_session(timestamp: datetime, start: time, end: time) -> bool:
    """Return True when the UTC time-of-day falls inside ``[start, end)``."""

    current_time = timestamp.timetz()
    return start <= current_time < end


def is_allowed_session(timestamp: datetime) -> bool:
    """Return True when ``timestamp`` is inside the London or New York session."""

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")

    utc_timestamp = timestamp.astimezone(timezone.utc)
    return _is_within_session(utc_timestamp, LONDON_SESSION_START, LONDON_SESSION_END) or _is_within_session(
        utc_timestamp, NEW_YORK_SESSION_START, NEW_YORK_SESSION_END
    )
