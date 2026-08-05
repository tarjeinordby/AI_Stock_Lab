"""
Tests for modules/exchange_calendar.py

All tests use the real exchange_calendars library (offline, no network calls).
Calendar data for 2026 is bundled inside the library.

Naming convention: test_<what>_<expected_outcome>
"""

import pytest
import pandas as pd

from modules.exchange_calendar import (
    CalendarUnavailableError,
    clear_calendar_cache,
    get_nyse_calendar,
    intended_execution_session,
    is_early_close,
    is_trading_session,
    nth_session_after,
    session_close_utc,
    session_open_utc,
    sessions_between_count,
)


@pytest.fixture(autouse=True)
def restore_calendar_cache():
    """Ensure a clean calendar cache before each test."""
    clear_calendar_cache()
    yield
    # Re-prime after tests that deliberately clear the cache
    clear_calendar_cache()


# ---------------------------------------------------------------------------
# is_trading_session
# ---------------------------------------------------------------------------

class TestIsTradingSession:
    def test_regular_wednesday_is_session(self):
        assert is_trading_session("2026-08-05") is True

    def test_saturday_is_not_session(self):
        assert is_trading_session("2026-08-08") is False

    def test_sunday_is_not_session(self):
        assert is_trading_session("2026-08-09") is False

    def test_independence_day_adjacent_friday_is_not_session(self):
        # July 4 2026 is Saturday; NYSE observes it on Friday July 3 → closed
        assert is_trading_session("2026-07-03") is False

    def test_christmas_day_is_not_session(self):
        assert is_trading_session("2026-12-25") is False

    def test_thanksgiving_thursday_is_not_session(self):
        # Thanksgiving 2026 = November 26
        assert is_trading_session("2026-11-26") is False


# ---------------------------------------------------------------------------
# intended_execution_session
# ---------------------------------------------------------------------------

class TestIntendedExecutionSession:
    def test_wednesday_signal_gives_thursday_session(self):
        # Signal generated 2026-08-05 (Wednesday after close)
        assert intended_execution_session("2026-08-05") == "2026-08-06"

    def test_friday_signal_gives_monday_session(self):
        # Signal generated 2026-08-07 (Friday after close) → Monday 2026-08-10
        assert intended_execution_session("2026-08-07") == "2026-08-10"

    def test_signal_before_holiday_friday_gives_monday_session(self):
        # 2026-07-02 (Thursday) → next session skips closed Friday 07-03 → Monday 07-06
        assert intended_execution_session("2026-07-02") == "2026-07-06"

    def test_christmas_eve_signal_gives_next_session(self):
        # 2026-12-24 (Thursday, early close) → signal after close → next session 2026-12-28
        assert intended_execution_session("2026-12-24") == "2026-12-28"

    def test_thanksgiving_wednesday_signal_gives_friday_session(self):
        # 2026-11-25 (Wednesday) → next session is Friday 11-27 (Thursday is closed)
        assert intended_execution_session("2026-11-25") == "2026-11-27"


# ---------------------------------------------------------------------------
# session_open_utc — DST
# ---------------------------------------------------------------------------

class TestSessionOpenUtc:
    def test_summer_session_opens_at_1330_utc(self):
        # August = EDT (UTC-4); NYSE opens 09:30 ET = 13:30 UTC
        open_utc = session_open_utc("2026-08-05")
        assert open_utc.hour == 13
        assert open_utc.minute == 30

    def test_winter_session_opens_at_1430_utc(self):
        # January = EST (UTC-5); NYSE opens 09:30 ET = 14:30 UTC
        open_utc = session_open_utc("2026-01-05")
        assert open_utc.hour == 14
        assert open_utc.minute == 30

    def test_dst_transition_spring_forward(self):
        # US clocks spring forward second Sunday March 2026 = March 8
        # Before (March 6, Friday): EST → 14:30 UTC
        # After  (March 9, Monday): EDT → 13:30 UTC
        before = session_open_utc("2026-03-06")
        after = session_open_utc("2026-03-09")
        assert before.hour == 14 and before.minute == 30
        assert after.hour == 13 and after.minute == 30

    def test_dst_transition_fall_back(self):
        # US clocks fall back first Sunday November 2026 = November 1
        # Before (Oct 30, Friday): EDT → 13:30 UTC
        # After  (Nov 2, Monday):  EST → 14:30 UTC
        before = session_open_utc("2026-10-30")
        after = session_open_utc("2026-11-02")
        assert before.hour == 13 and before.minute == 30
        assert after.hour == 14 and after.minute == 30

    def test_non_session_raises_calendar_error(self):
        with pytest.raises(CalendarUnavailableError):
            session_open_utc("2026-08-08")  # Saturday


# ---------------------------------------------------------------------------
# is_early_close
# ---------------------------------------------------------------------------

class TestIsEarlyClose:
    def test_regular_session_is_not_early_close(self):
        assert is_early_close("2026-08-05") is False

    def test_day_after_thanksgiving_is_early_close(self):
        # Friday 2026-11-27: closes at 13:00 ET (3.5 h session)
        assert is_early_close("2026-11-27") is True

    def test_christmas_eve_is_early_close(self):
        # 2026-12-24: closes at 13:00 ET (3.5 h session)
        assert is_early_close("2026-12-24") is True

    def test_non_session_raises_calendar_error(self):
        with pytest.raises(CalendarUnavailableError):
            is_early_close("2026-11-26")  # Thanksgiving — not a session


# ---------------------------------------------------------------------------
# nth_session_after
# ---------------------------------------------------------------------------

class TestNthSessionAfter:
    def test_t1_after_wednesday(self):
        assert nth_session_after("2026-08-05", 1) == "2026-08-06"

    def test_t1_after_friday_is_monday(self):
        assert nth_session_after("2026-08-07", 1) == "2026-08-10"

    def test_t5_after_wednesday(self):
        # Aug 5 (Wed) → Aug 6, 7, 10, 11, 12
        assert nth_session_after("2026-08-05", 5) == "2026-08-12"

    def test_t20_spans_multiple_weeks(self):
        result = nth_session_after("2026-08-05", 20)
        # Must be a valid trading session
        assert is_trading_session(result)
        # Must be approximately 20 trading days (4 calendar weeks)
        result_ts = pd.Timestamp(result)
        base_ts = pd.Timestamp("2026-08-05")
        delta_days = (result_ts - base_ts).days
        assert 25 <= delta_days <= 35

    def test_t1_skips_holiday_friday(self):
        # Signal date 2026-07-02, next session skips closed 07-03
        assert nth_session_after("2026-07-02", 1) == "2026-07-06"

    def test_invalid_n_raises_value_error(self):
        with pytest.raises(ValueError):
            nth_session_after("2026-08-05", 0)


# ---------------------------------------------------------------------------
# sessions_between_count
# ---------------------------------------------------------------------------

class TestSessionsBetweenCount:
    def test_five_sessions_in_one_week(self):
        # Aug 5 (Wed) to Aug 12 (Wed): Thu, Fri, Mon, Tue, Wed = 5
        assert sessions_between_count("2026-08-05", "2026-08-12") == 5

    def test_same_date_is_zero(self):
        assert sessions_between_count("2026-08-05", "2026-08-05") == 0

    def test_end_before_start_is_zero(self):
        assert sessions_between_count("2026-08-12", "2026-08-05") == 0

    def test_week_with_holiday_has_fewer_sessions(self):
        # Week of July 4 2026: July 6 (Mon) – July 10 (Fri) = 5, but July 3 is closed
        # Signal date June 30, end July 10: July 1(W), 2(Th), 6(M), 7(T), 8(W), 9(Th), 10(F) = 7
        count_with_holiday = sessions_between_count("2026-06-30", "2026-07-10")
        # Compare to normal week without holiday
        count_normal = sessions_between_count("2026-08-03", "2026-08-13")
        # Holiday week should have fewer sessions
        assert count_with_holiday < count_normal


# ---------------------------------------------------------------------------
# CalendarUnavailableError — fail closed, no fallback
# ---------------------------------------------------------------------------

class TestCalendarUnavailableFailClosed:
    def test_load_failure_raises_calendar_unavailable(self, monkeypatch):
        """
        Simulates exchange_calendars being unavailable.
        Must raise CalendarUnavailableError — no fallback, no silent degradation.
        """
        import modules.exchange_calendar as ec_mod

        clear_calendar_cache()

        def raise_import_error():
            raise ImportError("exchange_calendars not installed")

        monkeypatch.setattr(ec_mod, "_load_calendar", raise_import_error)

        with pytest.raises(CalendarUnavailableError):
            get_nyse_calendar()

    def test_downstream_functions_propagate_calendar_error(self, monkeypatch):
        """
        All public functions must propagate CalendarUnavailableError upward.
        None may silently return a result or use a default.
        """
        import modules.exchange_calendar as ec_mod

        clear_calendar_cache()

        def raise_import_error():
            raise ImportError("exchange_calendars not installed")

        monkeypatch.setattr(ec_mod, "_load_calendar", raise_import_error)

        with pytest.raises(CalendarUnavailableError):
            is_trading_session("2026-08-05")

        # Restore cache so other functions can be tested
        clear_calendar_cache()
        monkeypatch.setattr(ec_mod, "_load_calendar", raise_import_error)

        with pytest.raises(CalendarUnavailableError):
            intended_execution_session("2026-08-05")

        clear_calendar_cache()
        monkeypatch.setattr(ec_mod, "_load_calendar", raise_import_error)

        with pytest.raises(CalendarUnavailableError):
            session_open_utc("2026-08-05")

        clear_calendar_cache()
        monkeypatch.setattr(ec_mod, "_load_calendar", raise_import_error)

        with pytest.raises(CalendarUnavailableError):
            nth_session_after("2026-08-05", 1)

        clear_calendar_cache()
        monkeypatch.setattr(ec_mod, "_load_calendar", raise_import_error)

        with pytest.raises(CalendarUnavailableError):
            sessions_between_count("2026-08-05", "2026-08-12")
