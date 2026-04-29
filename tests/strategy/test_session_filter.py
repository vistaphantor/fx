from datetime import datetime, timedelta, timezone

import pytest

from src.strategy.session_filter import is_allowed_session


def test_is_allowed_session_accepts_london_session():
    timestamp = datetime(2026, 4, 4, 8, 30, tzinfo=timezone.utc)

    assert is_allowed_session(timestamp) is True


def test_is_allowed_session_accepts_new_york_session():
    timestamp = datetime(2026, 4, 4, 13, 30, tzinfo=timezone.utc)

    assert is_allowed_session(timestamp) is True


@pytest.mark.parametrize(
    "timestamp",
    [
        datetime(2026, 4, 4, 7, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 4, 10, 59, 59, tzinfo=timezone.utc),
        datetime(2026, 4, 4, 12, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 4, 15, 59, 59, tzinfo=timezone.utc),
    ],
)
def test_is_allowed_session_accepts_session_boundaries(timestamp):
    assert is_allowed_session(timestamp) is True


@pytest.mark.parametrize(
    "timestamp",
    [
        datetime(2026, 4, 4, 11, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 4, 16, 0, tzinfo=timezone.utc),
    ],
)
def test_is_allowed_session_rejects_session_end_boundaries(timestamp):
    assert is_allowed_session(timestamp) is False


@pytest.mark.parametrize(
    "timestamp",
    [
        datetime(2026, 4, 4, 6, 59, tzinfo=timezone.utc),
        datetime(2026, 4, 4, 21, 0, tzinfo=timezone.utc),
    ],
)
def test_is_allowed_session_rejects_outside_london_and_new_york(timestamp):
    assert is_allowed_session(timestamp) is False


def test_is_allowed_session_normalizes_non_utc_aware_datetime():
    timestamp = datetime(2026, 4, 4, 10, 30, tzinfo=timezone(timedelta(hours=2)))

    assert is_allowed_session(timestamp) is True


def test_is_allowed_session_rejects_naive_datetime():
    timestamp = datetime(2026, 4, 4, 8, 30)

    with pytest.raises(ValueError, match="timezone-aware"):
        is_allowed_session(timestamp)
