"""
Tests for modules/signal_validator.py — signal stale-validation, data quality,
and traceability checks.

Key invariants:
  - A signal is only valid for its intended_execution_session.
  - publication_status must be "published".
  - Ledger record must have status "completed".
  - Content hash must match (tamper detection).
  - Stale/invalid signal blocks new buys and pyramids.
  - Protective sells (stop-loss, trailing stop) are ALWAYS allowed.
  - execute-reruns do not create duplicate orders (via order idempotency ledger).
  - Calendar-unavailable → fail-closed (no new buys, Telegram alert).
"""

import hashlib
import json
from datetime import timezone, datetime

import pytest

from modules.signal_validator import (
    COVERAGE_NORMAL,
    COVERAGE_REDUCED,
    STALE_BLOCK_PCT,
    SignalValidationResult,
    compute_signal_content_hash,
    validate_signal,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SESSION = "2026-08-06"   # Thursday — valid NYSE session

def _make_signal(
    date="2026-08-05",                        # Wednesday signal → Thursday execute
    signal_run_id="run-2026-08-05-abc123456",
    intended_execution_session=SESSION,
    publication_status="published",
    published_at="2026-08-05T21:00:00+00:00",
    model_version="quant_baseline_v1",
    universe_count=200,
    valid_ticker_count=190,
    stale_ticker_count=5,
    excluded_ticker_count=5,
    add_content_hash=False,
    **overrides,
) -> dict:
    s = {
        "date": date,
        "created_at_utc": "2026-08-05T20:00:00+00:00",
        "created_at_oslo": "2026-08-05T22:00:00+02:00",
        "created_at_ny": "2026-08-05T16:00:00-04:00",
        "signal_run_id": signal_run_id,
        "intended_execution_session": intended_execution_session,
        "publication_status": publication_status,
        "published_at": published_at,
        "model_version": model_version,
        "universe_count": universe_count,
        "valid_ticker_count": valid_ticker_count,
        "stale_ticker_count": stale_ticker_count,
        "excluded_ticker_count": excluded_ticker_count,
        "missing_ticker_count": 0,
        "signal_content_hash": "",
    }
    s.update(overrides)
    if add_content_hash:
        s["signal_content_hash"] = compute_signal_content_hash(s)
    return s


def _ledger_completed(signal_run_id, year_month):
    return {"status": "completed", "n_signal_records": 100}

def _ledger_failed(signal_run_id, year_month):
    return {"status": "failed", "error": "test error"}

def _ledger_none(signal_run_id, year_month):
    return None

def _ies_thursday(date_str):
    """Always returns Thursday 2026-08-06 — for tests that don't vary session."""
    return SESSION

def _ies_friday_to_monday(date_str):
    """Friday 2026-08-07 → Monday 2026-08-10."""
    mapping = {
        "2026-08-07": "2026-08-10",
        "2026-08-05": "2026-08-06",
    }
    return mapping.get(date_str, SESSION)

def _ies_holiday(date_str):
    """July 3 → July 6 (July 4 holiday)."""
    if date_str == "2026-07-03":
        return "2026-07-06"
    return SESSION


# ---------------------------------------------------------------------------
# Layer 1: Required fields
# ---------------------------------------------------------------------------

class TestRequiredFields:
    def test_missing_signal_run_id_rejected(self):
        s = _make_signal(signal_run_id=None)
        result = validate_signal(s, "/fake/path", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode == "missing_fields"
        assert not result.allow_new_buys

    def test_missing_created_at_utc_rejected(self):
        s = _make_signal()
        del s["created_at_utc"]
        result = validate_signal(s, "/fake/path", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode == "missing_fields"
        assert not result.allow_new_buys

    def test_missing_date_rejected(self):
        s = _make_signal()
        del s["date"]
        result = validate_signal(s, "/fake/path", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode == "missing_fields"
        assert not result.allow_new_buys

    def test_missing_publication_status_rejected(self):
        s = _make_signal(publication_status=None)
        result = validate_signal(s, "/fake/path", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode == "unpublished"
        assert not result.allow_new_buys

    def test_all_required_fields_passes(self):
        s = _make_signal(add_content_hash=False)
        result = validate_signal(s, "/fake/path", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode in ("ok", "data_quality_reduced")
        assert result.allow_protective_sells


# ---------------------------------------------------------------------------
# Layer 2: Session match
# ---------------------------------------------------------------------------

class TestSessionValidation:
    def test_same_session_passes(self):
        s = _make_signal(intended_execution_session=SESSION)
        result = validate_signal(s, "/fake/path", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode not in ("stale_signal", "calendar_unavailable")
        assert result.intended_session == SESSION
        assert result.actual_session == SESSION

    def test_stale_signal_rejected_different_session(self):
        """Yesterday's signal (intended Monday) is invalid when run on Tuesday."""
        s = _make_signal(intended_execution_session="2026-08-03")  # Monday
        result = validate_signal(s, "/fake/path", SESSION,   # session is Thursday
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode == "stale_signal"
        assert not result.allow_new_buys
        assert result.allow_protective_sells  # Always true

    def test_friday_signal_valid_monday(self):
        """Friday evening signal → Monday is the valid execution session."""
        s = _make_signal(
            date="2026-08-07",                        # Friday
            intended_execution_session="2026-08-10",  # Monday
        )
        result = validate_signal(
            s, "/fake/path", "2026-08-10",            # Running on Monday
            _get_ledger_status=_ledger_completed,
            _intended_session_fn=_ies_friday_to_monday,
        )
        assert result.failure_mode not in ("stale_signal",)
        assert result.allow_new_buys

    def test_friday_signal_stale_on_tuesday(self):
        """Friday signal that wasn't consumed on Monday is stale on Tuesday."""
        s = _make_signal(
            date="2026-08-07",
            intended_execution_session="2026-08-10",  # Monday
        )
        result = validate_signal(
            s, "/fake/path", "2026-08-11",            # Running on Tuesday
            _get_ledger_status=_ledger_completed,
            _intended_session_fn=_ies_friday_to_monday,
        )
        assert result.failure_mode == "stale_signal"
        assert not result.allow_new_buys

    def test_pre_holiday_signal_valid_next_session(self):
        """July 3 signal → July 6 is next valid session (July 4 holiday)."""
        s = _make_signal(
            date="2026-07-03",
            intended_execution_session="2026-07-06",
        )
        result = validate_signal(
            s, "/fake/path", "2026-07-06",
            _get_ledger_status=_ledger_completed,
            _intended_session_fn=_ies_holiday,
        )
        assert result.failure_mode not in ("stale_signal",)
        assert result.allow_new_buys

    def test_calendar_unavailable_blocks_buys(self):
        """When calendar is unavailable and field is missing, fail-closed."""
        s = _make_signal()
        del s["intended_execution_session"]  # No stored field

        def _calendar_fails(date_str):
            raise RuntimeError("exchange_calendars not installed")

        result = validate_signal(
            s, "/fake/path", SESSION,
            _get_ledger_status=_ledger_completed,
            _intended_session_fn=_calendar_fails,
        )
        assert result.failure_mode == "calendar_unavailable"
        assert not result.allow_new_buys
        assert result.allow_protective_sells  # Safety always allowed

    def test_stored_intended_session_used_without_calendar(self):
        """When intended_execution_session is in payload, no calendar call needed."""
        s = _make_signal(intended_execution_session=SESSION)

        called = []
        def _calendar_should_not_be_called(date_str):
            called.append(date_str)
            raise RuntimeError("should not be called")

        result = validate_signal(
            s, "/fake/path", SESSION,
            _get_ledger_status=_ledger_completed,
            _intended_session_fn=_calendar_should_not_be_called,
        )
        # Calendar fn should not be called when field exists in payload
        assert len(called) == 0

    def test_filename_not_used_as_date_truth(self):
        """signal_path filename date is irrelevant — content drives validation."""
        s = _make_signal(date="2026-08-05", intended_execution_session=SESSION)
        # Filename says "2026-07-01" but content says 2026-08-05 → 2026-08-06
        result = validate_signal(
            s, "/data/signals/2026-07-01_signal.json", SESSION,
            _get_ledger_status=_ledger_completed,
            _intended_session_fn=_ies_thursday,
        )
        assert result.failure_mode not in ("stale_signal",)

    def test_timezone_utc_generated_at_parsed(self):
        """generated_at in UTC is stored and returned correctly."""
        s = _make_signal()
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert result.generated_at == "2026-08-05T20:00:00+00:00"

    def test_timezone_oslo_and_ny_in_signal(self):
        """Oslo and NY timestamps are in signal for audit; UTC drives logic."""
        s = _make_signal()
        # 20:00 UTC = 22:00 Oslo (CEST summer) = 16:00 NY (EDT summer)
        assert "22:00:00+02:00" in s["created_at_oslo"]
        assert "16:00:00-04:00" in s["created_at_ny"]
        # Validation uses created_at_utc — no mixing
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert result.generated_at == s["created_at_utc"]


# ---------------------------------------------------------------------------
# Layer 3: Publication status
# ---------------------------------------------------------------------------

class TestPublicationStatus:
    def test_published_passes(self):
        s = _make_signal(publication_status="published")
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode not in ("unpublished",)

    def test_draft_rejected(self):
        s = _make_signal(publication_status="draft")
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode == "unpublished"
        assert not result.allow_new_buys

    def test_pending_rejected(self):
        s = _make_signal(publication_status="pending")
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode == "unpublished"
        assert not result.allow_new_buys

    def test_none_publication_status_rejected(self):
        s = _make_signal(publication_status=None)
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode == "unpublished"

    def test_failed_workflow_proxy_rejected(self):
        """A 'failed' publication_status represents a failed workflow run."""
        s = _make_signal(publication_status="failed")
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode == "unpublished"
        assert not result.allow_new_buys


# ---------------------------------------------------------------------------
# Layer 4: Ledger check
# ---------------------------------------------------------------------------

class TestLedgerCheck:
    def test_completed_ledger_passes(self):
        s = _make_signal()
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode not in ("ledger_incomplete", "ledger_error")

    def test_no_ledger_record_rejected(self):
        s = _make_signal()
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_none,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode == "ledger_incomplete"
        assert not result.allow_new_buys

    def test_failed_ledger_status_rejected(self):
        s = _make_signal()
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_failed,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode == "ledger_incomplete"
        assert not result.allow_new_buys

    def test_pending_ledger_status_rejected(self):
        def _pending(run_id, ym):
            return {"status": "pending"}
        s = _make_signal()
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_pending,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode == "ledger_incomplete"

    def test_ledger_error_propagated(self):
        def _raises(run_id, ym):
            raise OSError("disk error")
        s = _make_signal()
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_raises,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode == "ledger_error"
        assert not result.allow_new_buys


# ---------------------------------------------------------------------------
# Layer 5: Content hash
# ---------------------------------------------------------------------------

class TestContentHash:
    def test_valid_hash_passes(self):
        s = _make_signal(add_content_hash=True)
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode not in ("hash_mismatch", "hash_error")
        assert result.allow_new_buys

    def test_tampered_signal_rejected(self):
        """Modifying any field after signing invalidates the hash."""
        s = _make_signal(add_content_hash=True)
        s["universe_count"] = 999999  # tamper
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode == "hash_mismatch"
        assert not result.allow_new_buys

    def test_missing_hash_field_skipped(self):
        """Old signals without signal_content_hash are accepted (backward compat)."""
        s = _make_signal()
        s.pop("signal_content_hash", None)
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode not in ("hash_mismatch",)
        assert result.allow_new_buys

    def test_empty_hash_field_skipped(self):
        """signal_content_hash='' is treated as absent (backward compat)."""
        s = _make_signal()
        s["signal_content_hash"] = ""
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode not in ("hash_mismatch",)

    def test_hash_is_deterministic(self):
        s1 = _make_signal()
        s2 = _make_signal()
        assert compute_signal_content_hash(s1) == compute_signal_content_hash(s2)

    def test_hash_changes_on_field_change(self):
        s = _make_signal()
        h1 = compute_signal_content_hash(s)
        s["universe_count"] = s["universe_count"] + 1
        h2 = compute_signal_content_hash(s)
        assert h1 != h2

    def test_hash_excludes_itself(self):
        """Hash must be computable without circular dependency."""
        s = _make_signal()
        s["signal_content_hash"] = "some_placeholder"
        h1 = compute_signal_content_hash(s)
        s["signal_content_hash"] = "different_placeholder"
        h2 = compute_signal_content_hash(s)
        assert h1 == h2  # Hash value is excluded from computation


# ---------------------------------------------------------------------------
# Layer 6: Data quality thresholds
# ---------------------------------------------------------------------------

class TestDataQuality:
    def _validate(self, universe=200, valid=190, stale=5, excluded=5):
        s = _make_signal(
            universe_count=universe,
            valid_ticker_count=valid,
            stale_ticker_count=stale,
            excluded_ticker_count=excluded,
        )
        return validate_signal(s, "/fake", SESSION,
                               _get_ledger_status=_ledger_completed,
                               _intended_session_fn=_ies_thursday)

    def test_high_coverage_is_normal_tier(self):
        result = self._validate(universe=200, valid=190, stale=5)
        assert result.data_quality_tier == "normal"
        assert result.allow_new_buys

    def test_medium_coverage_is_reduced_tier(self):
        # 160/200 = 80% — between COVERAGE_REDUCED (0.75) and COVERAGE_NORMAL (0.90)
        result = self._validate(universe=200, valid=160, stale=10, excluded=30)
        assert result.data_quality_tier == "reduced"
        assert result.allow_new_buys  # Buys allowed in reduced tier (with warning)

    def test_low_coverage_blocks_buys(self):
        # 140/200 = 70% — below COVERAGE_REDUCED (0.75)
        result = self._validate(universe=200, valid=140, stale=5, excluded=55)
        assert result.data_quality_tier == "blocked"
        assert not result.allow_new_buys
        assert not result.allow_pyramid
        assert result.allow_protective_sells

    def test_high_stale_pct_blocks_buys(self):
        # stale=55/200 = 27.5% — above STALE_BLOCK_PCT (0.25)
        result = self._validate(universe=200, valid=185, stale=55, excluded=0)
        assert result.data_quality_tier == "blocked"
        assert not result.allow_new_buys

    def test_stale_at_threshold_blocks(self):
        # stale=51/200 = 25.5% — just above threshold
        result = self._validate(universe=200, valid=180, stale=51, excluded=0)
        assert result.data_quality_tier == "blocked"

    def test_stale_below_threshold_normal(self):
        # stale=49/200 = 24.5% — just below threshold, coverage 95%
        result = self._validate(universe=200, valid=190, stale=49, excluded=0)
        assert result.data_quality_tier == "normal"
        assert result.allow_new_buys

    def test_data_quality_report_in_result(self):
        result = self._validate(universe=200, valid=190, stale=5, excluded=5)
        assert result.data_quality["universe_count"] == 200
        assert result.data_quality["valid_ticker_count"] == 190
        assert result.data_quality["signal_coverage_rate"] == pytest.approx(0.95, abs=0.01)

    def test_zero_universe_count_is_unknown_tier(self):
        result = self._validate(universe=0, valid=0, stale=0, excluded=0)
        assert result.data_quality_tier == "unknown"
        # Unknown tier: allow buys (we don't have info to block)
        assert result.allow_new_buys

    def test_blocked_quality_is_valid_signal(self):
        """Data quality blocked is is_valid=True — signal itself is fine."""
        result = self._validate(universe=200, valid=140, stale=5, excluded=55)
        assert result.is_valid  # Signal is valid, quality is not
        assert result.failure_mode == "data_quality_blocked"


# ---------------------------------------------------------------------------
# Protective sells always allowed
# ---------------------------------------------------------------------------

class TestProtectiveSells:
    def test_protective_sells_allowed_on_stale_signal(self):
        """Stale signal blocks buys but stop-loss/trailing-stop must proceed."""
        s = _make_signal(intended_execution_session="2026-08-03")  # Wrong session
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode == "stale_signal"
        assert not result.allow_new_buys
        assert result.allow_protective_sells  # Must be True

    def test_protective_sells_allowed_on_unpublished_signal(self):
        s = _make_signal(publication_status="draft")
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert not result.allow_new_buys
        assert result.allow_protective_sells

    def test_protective_sells_allowed_on_calendar_error(self):
        s = _make_signal()
        del s["intended_execution_session"]

        def _fails(d): raise RuntimeError("calendar down")

        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_fails)
        assert not result.allow_new_buys
        assert result.allow_protective_sells

    def test_cash_not_affected_by_stale_signal_blocking_buys(self):
        """
        Verify that the validator returns allow_new_buys=False on stale signal.
        Cash invariant is enforced by run_strategy_execution using this flag.
        The validator itself never modifies cash.
        """
        s = _make_signal(intended_execution_session="2026-08-03")
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert not result.allow_new_buys
        # No cash/position objects exist here — the invariant is contract-based


# ---------------------------------------------------------------------------
# Duplicate protection (order idempotency via order ledger)
# ---------------------------------------------------------------------------

class TestDuplicateProtection:
    def test_valid_signal_allows_full_execution(self):
        """A valid signal for the correct session enables all actions."""
        s = _make_signal(add_content_hash=True)
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert result.is_valid
        assert result.allow_new_buys
        assert result.allow_pyramid

    def test_rerun_with_same_signal_does_not_change_validation(self):
        """
        Running validate_signal twice for the same session/signal gives
        the same result (deterministic — no side effects in validator).
        """
        s = _make_signal(add_content_hash=True)
        r1 = validate_signal(s, "/fake", SESSION,
                              _get_ledger_status=_ledger_completed,
                              _intended_session_fn=_ies_thursday)
        r2 = validate_signal(s, "/fake", SESSION,
                              _get_ledger_status=_ledger_completed,
                              _intended_session_fn=_ies_thursday)
        assert r1.is_valid == r2.is_valid
        assert r1.failure_mode == r2.failure_mode
        assert r1.allow_new_buys == r2.allow_new_buys

    def test_old_signal_cannot_be_used_as_new_analysis(self):
        """
        A signal from a past intended_execution_session is rejected for today.
        This prevents reusing old analysis when today's signal job failed.
        """
        s = _make_signal(intended_execution_session="2026-07-01")  # Past session
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode == "stale_signal"
        assert not result.allow_new_buys

    def test_pending_order_correct_session_allowed(self):
        """
        Signal validation only governs NEW orders.
        Pending orders from the correct session are handled by order ledger
        (get_pending_for_session) regardless of new signal validity.
        This test documents the contract — signal_validator does not touch orders.
        """
        s = _make_signal(add_content_hash=True)
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        # If signal is valid, execution proceeds; pending orders are retried
        assert result.is_valid
        assert result.allow_new_buys

    def test_expired_session_order_blocked_by_session_mismatch(self):
        """
        An order intended for an old session cannot be filled by a new session.
        The signal validator alone doesn't enforce this — order ledger does.
        But when the signal's intended_execution_session is wrong, we block.
        """
        s = _make_signal(intended_execution_session="2026-08-05")  # yesterday
        result = validate_signal(s, "/fake", SESSION,   # today is 2026-08-06
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        assert result.failure_mode == "stale_signal"
        assert not result.allow_new_buys


# ---------------------------------------------------------------------------
# Telegram message content
# ---------------------------------------------------------------------------

class TestTelegramMessage:
    def test_stale_signal_message_contains_signal_run_id(self):
        from modules.reporting import build_stale_signal_message
        s = _make_signal(intended_execution_session="2026-08-03")
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        msg = build_stale_signal_message(result)
        assert result.signal_run_id in msg

    def test_stale_signal_message_contains_intended_and_actual_session(self):
        from modules.reporting import build_stale_signal_message
        s = _make_signal(intended_execution_session="2026-08-03")
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        msg = build_stale_signal_message(result)
        assert "2026-08-03" in msg  # intended session
        assert SESSION in msg        # actual session

    def test_stale_signal_message_contains_failure_mode(self):
        from modules.reporting import build_stale_signal_message
        s = _make_signal(intended_execution_session="2026-08-03")
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        msg = build_stale_signal_message(result)
        assert "stale_signal" in msg

    def test_stale_signal_message_mentions_allowed_actions(self):
        from modules.reporting import build_stale_signal_message
        s = _make_signal(intended_execution_session="2026-08-03")
        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_ies_thursday)
        msg = build_stale_signal_message(result)
        assert "stop-loss" in msg.lower() or "stop" in msg.lower()
        assert "kjøp" in msg.lower()  # blocked buys mentioned

    def test_calendar_unavailable_message_distinct(self):
        from modules.reporting import build_stale_signal_message
        s = _make_signal()
        del s["intended_execution_session"]

        def _fails(d): raise RuntimeError("calendar unavailable")

        result = validate_signal(s, "/fake", SESSION,
                                 _get_ledger_status=_ledger_completed,
                                 _intended_session_fn=_fails)
        msg = build_stale_signal_message(result)
        assert "calendar_unavailable" in msg
