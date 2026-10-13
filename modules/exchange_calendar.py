"""
NYSE market calendar via exchange_calendars.

CRITICAL: This module is execution-critical.
If the calendar is unavailable, all functions raise CalendarUnavailableError.
There is NO fallback for order execution or execution-price determination.

Callers must catch CalendarUnavailableError and:
  - abort new paper fills
  - log the error
  - send Telegram alert
  - mark orders as pending or failed

A CalendarUnavailableError may be caught silently for non-critical
reporting (e.g. "is today a holiday?" banner) but never for fill decisions.
"""

from __future__ import annotations

import functools

import pandas as pd

_EXCHANGE = "XNYS"
_STANDARD_SESSION_HOURS = 6.5        # 09:30–16:00 ET
_EARLY_CLOSE_THRESHOLD_HOURS = 6.0   # anything shorter counts as early close


class CalendarUnavailableError(Exception):
    """Raised when the NYSE calendar cannot be loaded or queried."""


def _load_calendar():
    """Isolated import so tests can patch this function."""
    import exchange_calendars as ec  # noqa: PLC0415
    return ec.get_calendar(_EXCHANGE)


@functools.lru_cache(maxsize=1)
def get_nyse_calendar():
    """
    Return the cached NYSE exchange calendar.
    Raises CalendarUnavailableError if exchange_calendars is not installed
    or the calendar cannot be loaded.
    """
    try:
        return _load_calendar()
    except ImportError as exc:
        raise CalendarUnavailableError(
            "exchange_calendars is not installed — "
            "run: pip install 'exchange_calendars>=4.0'"
        ) from exc
    except Exception as exc:
        raise CalendarUnavailableError(
            f"Cannot load NYSE calendar: {exc}"
        ) from exc


def clear_calendar_cache() -> None:
    """Clear the cached calendar instance. Intended for tests only."""
    get_nyse_calendar.cache_clear()


# ---------------------------------------------------------------------------
# Public API — all raise CalendarUnavailableError on failure
# ---------------------------------------------------------------------------

def is_trading_session(date_str: str) -> bool:
    """Return True if date_str is a valid NYSE trading session."""
    cal = get_nyse_calendar()
    try:
        return bool(cal.is_session(pd.Timestamp(date_str)))
    except CalendarUnavailableError:
        raise
    except Exception as exc:
        raise CalendarUnavailableError(
            f"Calendar query failed for {date_str}: {exc}"
        ) from exc


def last_completed_session(as_of_utc: "pd.Timestamp | None" = None) -> str:
    """
    Return the most recently CLOSED NYSE trading session as 'YYYY-MM-DD',
    anchored to an absolute UTC instant rather than any wall-clock timezone.

    This must be used instead of a wall-clock "today" (Oslo or NY) to decide
    which session a signal's market data represents. GitHub Actions scheduled
    triggers are best-effort and routinely fire hours late — a cron set for
    21:00 UTC has been observed starting anywhere from on-time to 02:23 UTC
    the next calendar day. Deriving the signal date from wall-clock time means
    the apparent "today" can silently roll over to the next day (or skip a
    session) purely because of scheduler delay, with no error — which is
    exactly how intended_execution_session() ended up one session ahead in
    production. Anchoring to "the session that already closed" is immune to
    both scheduler delay and the multi-hour AI analysis that follows it.

    Examples (all correctly resolve to the same trading day regardless of
    when the calling code actually happens to run):
        Monday 21:00 UTC (on-time cron fire, right after close) → Monday
        Tuesday 02:23 UTC (same cron, 5h23m late)                → Monday
        Tuesday 15:00 UTC (intraday, Tuesday still open)         → Monday
        Saturday 03:00 UTC (Friday cron delayed past midnight)   → Friday
    """
    cal = get_nyse_calendar()
    try:
        now = as_of_utc if as_of_utc is not None else pd.Timestamp.now(tz="UTC")
        session = cal.minute_to_past_session(now, count=1)
        return session.strftime("%Y-%m-%d")
    except CalendarUnavailableError:
        raise
    except Exception as exc:
        raise CalendarUnavailableError(
            f"Cannot determine last completed session as of {as_of_utc}: {exc}"
        ) from exc


def intended_execution_session(signal_date_str: str) -> str:
    """
    Return the next valid NYSE trading session after signal_date_str as 'YYYY-MM-DD'.

    A signal generated after market close on date D is intended for execution
    in the next trading session, which may be several calendar days later if D
    is a Friday or falls before a holiday.

    Examples:
        Wednesday 2026-08-05  →  Thursday 2026-08-06
        Friday    2026-08-07  →  Monday   2026-08-10
        Thursday  2026-07-02  →  Monday   2026-07-06  (Friday 07-03 is closed)
    """
    cal = get_nyse_calendar()
    try:
        signal_date = pd.Timestamp(signal_date_str)
        session = cal.date_to_session(
            signal_date + pd.Timedelta(days=1), direction="next"
        )
        return session.strftime("%Y-%m-%d")
    except CalendarUnavailableError:
        raise
    except Exception as exc:
        raise CalendarUnavailableError(
            f"Cannot determine next session after {signal_date_str}: {exc}"
        ) from exc


def session_open_utc(session_date_str: str) -> pd.Timestamp:
    """
    Return the NYSE open time for session_date_str as a UTC-aware pd.Timestamp.
    Accounts for US DST (EDT summer = UTC-4, EST winter = UTC-5).

    Raises CalendarUnavailableError if the date is not a trading session.
    """
    cal = get_nyse_calendar()
    try:
        ts = pd.Timestamp(session_date_str)
        if not cal.is_session(ts):
            raise CalendarUnavailableError(
                f"{session_date_str} is not a NYSE trading session"
            )
        return cal.session_open(ts)
    except CalendarUnavailableError:
        raise
    except Exception as exc:
        raise CalendarUnavailableError(
            f"Cannot get open time for {session_date_str}: {exc}"
        ) from exc


def session_close_utc(session_date_str: str) -> pd.Timestamp:
    """
    Return the NYSE close time for session_date_str as a UTC-aware pd.Timestamp.
    Returns early-close time on shortened sessions.

    Raises CalendarUnavailableError if the date is not a trading session.
    """
    cal = get_nyse_calendar()
    try:
        ts = pd.Timestamp(session_date_str)
        if not cal.is_session(ts):
            raise CalendarUnavailableError(
                f"{session_date_str} is not a NYSE trading session"
            )
        return cal.session_close(ts)
    except CalendarUnavailableError:
        raise
    except Exception as exc:
        raise CalendarUnavailableError(
            f"Cannot get close time for {session_date_str}: {exc}"
        ) from exc


def is_early_close(session_date_str: str) -> bool:
    """
    Return True if the session has an early close (session duration < 6 hours).
    Standard NYSE sessions are 6.5 hours (09:30–16:00 ET).
    Early close sessions (e.g. day after Thanksgiving, Christmas Eve) are 3.5 hours.
    """
    open_utc = session_open_utc(session_date_str)
    close_utc = session_close_utc(session_date_str)
    duration_hours = (close_utc - open_utc).total_seconds() / 3600
    return duration_hours < _EARLY_CLOSE_THRESHOLD_HOURS


def nth_session_after(date_str: str, n: int) -> str:
    """
    Return the date string of the Nth NYSE trading session after date_str.
    n=1 returns the next session, n=5 returns the 5th session after, etc.

    Used for forward-return tracking (T+1, T+5, T+20, T+60, T+120, T+252).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    cal = get_nyse_calendar()
    try:
        base = pd.Timestamp(date_str)
        # Conservative window: n sessions fit within 3× calendar days, minimum 30 days
        window_end = base + pd.Timedelta(days=max(n * 3, 30))
        sessions = cal.sessions_in_range(base + pd.Timedelta(days=1), window_end)
        if len(sessions) < n:
            # Extend window for very long horizons (T+120, T+252)
            window_end = base + pd.Timedelta(days=n * 2 + 60)
            sessions = cal.sessions_in_range(base + pd.Timedelta(days=1), window_end)
        if len(sessions) < n:
            raise CalendarUnavailableError(
                f"Cannot find {n} sessions after {date_str} within search window"
            )
        return sessions[n - 1].strftime("%Y-%m-%d")
    except CalendarUnavailableError:
        raise
    except Exception as exc:
        raise CalendarUnavailableError(
            f"Cannot compute session T+{n} after {date_str}: {exc}"
        ) from exc


def sessions_between_count(start_date_str: str, end_date_str: str) -> int:
    """
    Count NYSE trading sessions strictly after start_date up to and including end_date.
    Returns 0 if end_date <= start_date.

    Used to classify a forward-return measurement into a T+N bucket.
    """
    cal = get_nyse_calendar()
    try:
        start = pd.Timestamp(start_date_str)
        end = pd.Timestamp(end_date_str)
        if end <= start:
            return 0
        sessions = cal.sessions_in_range(start + pd.Timedelta(days=1), end)
        return len(sessions)
    except CalendarUnavailableError:
        raise
    except Exception as exc:
        raise CalendarUnavailableError(
            f"Cannot count sessions between {start_date_str} and {end_date_str}: {exc}"
        ) from exc
