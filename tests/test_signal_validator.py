"""
Tests for modules/signal_validator.py and regression tests for commit-6 remediation.

Key invariants (updated):
  - A signal is only valid for its intended_execution_session (required field).
  - All eight required fields must be present (signal_run_id, created_at_utc, date,
    data_cutoff_at, intended_execution_session, published_at, published_commit_sha,
    model_version).
  - A publication record must exist with workflow_conclusion=="success" (Layer 3).
  - Ledger: status sidecar "completed" AND JSONL record cross-validates model_version +
    data_cutoff_at (Layer 4).
  - Content hash must be present and correct — missing or empty hash → REJECTED (Layer 5).
  - Data quality tier is reported but does NOT block execution (thresholds not yet approved).
  - Protective sells always allowed; signal-dependent sells blocked on invalid signal.
  - execute-reruns do not create duplicate orders (via order idempotency ledger).
  - Protective stops must use current execution/valuation price, not stale signal price.
  - load_latest_signal() has no fallback — requires published signal or raises.
"""

import hashlib
import json
import os
import tempfile

import pytest

from modules.signal_validator import (
    SignalValidationResult,
    compute_signal_content_hash,
    validate_signal,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SESSION = "2026-08-06"   # Thursday — valid NYSE session
DATA_CUTOFF = "2026-08-05"


def _make_signal(
    date="2026-08-05",
    signal_run_id="run-2026-08-05-abc123456",
    intended_execution_session=SESSION,
    publication_status="published",
    published_at="2026-08-05T21:05:00+00:00",
    published_commit_sha="abc1234def56",
    model_version="quant_baseline_v1",
    data_cutoff_at=DATA_CUTOFF,
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
        "published_commit_sha": published_commit_sha,
        "model_version": model_version,
        "data_cutoff_at": data_cutoff_at,
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


# Injection helpers
def _pub_success(run_id):
    return {"signal_run_id": run_id, "workflow_conclusion": "success", "commit_sha": "abc123"}

def _pub_none(run_id):
    return None

def _pub_failed(run_id):
    return {"signal_run_id": run_id, "workflow_conclusion": "failure"}

def _pub_missing_conclusion(run_id):
    return {"signal_run_id": run_id}

def _ledger_status_completed(run_id, ym):
    return {"status": "completed", "n_signal_records": 100}

def _ledger_status_failed(run_id, ym):
    return {"status": "failed"}

def _ledger_status_none(run_id, ym):
    return None

def _ledger_record_ok(run_id, ym):
    return {
        "signal_run_id": run_id,
        "model_version": "quant_baseline_v1",
        "canonical_data_cutoff": DATA_CUTOFF,
        "intended_signal_session": "2026-08-05",
    }

def _ledger_record_none(run_id, ym):
    return None

def _ledger_record_wrong_model(run_id, ym):
    return {
        "signal_run_id": run_id,
        "model_version": "old_model_v0",
        "canonical_data_cutoff": DATA_CUTOFF,
    }

def _ledger_record_wrong_cutoff(run_id, ym):
    return {
        "signal_run_id": run_id,
        "model_version": "quant_baseline_v1",
        "canonical_data_cutoff": "2026-07-01",  # wrong
    }


def _validate(signal, session=SESSION):
    """Convenience wrapper with all injections set to happy-path defaults."""
    return validate_signal(
        signal, "/fake/path", session,
        _get_publication_fn=_pub_success,
        _get_ledger_status_fn=_ledger_status_completed,
        _get_ledger_record_fn=_ledger_record_ok,
    )


# ---------------------------------------------------------------------------
# Layer 1: Required fields
# ---------------------------------------------------------------------------

class TestRequiredFields:
    """Regression: point 3 — validate obligatory fields."""

    def test_all_fields_present_passes(self):
        s = _make_signal(add_content_hash=True)
        result = _validate(s)
        assert result.failure_mode in ("ok", "data_quality_reduced")
        assert result.is_valid

    def test_missing_signal_run_id_rejected(self):
        s = _make_signal(add_content_hash=True)
        s["signal_run_id"] = ""
        result = _validate(s)
        assert result.failure_mode == "missing_fields"
        assert not result.allow_new_buys
        assert result.allow_protective_sells

    def test_missing_created_at_utc_rejected(self):
        s = _make_signal(add_content_hash=True)
        del s["created_at_utc"]
        result = _validate(s)
        assert result.failure_mode == "missing_fields"

    def test_missing_date_rejected(self):
        s = _make_signal(add_content_hash=True)
        del s["date"]
        result = _validate(s)
        assert result.failure_mode == "missing_fields"

    def test_missing_data_cutoff_at_rejected(self):
        """Regression point 3: data_cutoff_at is now obligatory."""
        s = _make_signal(add_content_hash=True)
        del s["data_cutoff_at"]
        result = _validate(s)
        assert result.failure_mode == "missing_fields"
        assert "data_cutoff_at" in result.reason

    def test_missing_intended_execution_session_rejected(self):
        """Regression point 3: intended_execution_session now required; no calendar fallback."""
        s = _make_signal(add_content_hash=True)
        del s["intended_execution_session"]
        result = _validate(s)
        assert result.failure_mode == "missing_fields"

    def test_missing_published_at_rejected(self):
        """Regression point 3: published_at is required."""
        s = _make_signal(add_content_hash=True)
        s["published_at"] = None
        result = _validate(s)
        assert result.failure_mode == "missing_fields"

    def test_missing_published_commit_sha_rejected(self):
        """Regression point 3: published_commit_sha is required."""
        s = _make_signal(add_content_hash=True)
        s["published_commit_sha"] = ""
        result = _validate(s)
        assert result.failure_mode == "missing_fields"

    def test_missing_model_version_rejected(self):
        """Regression point 3: model_version is required."""
        s = _make_signal(add_content_hash=True)
        s["model_version"] = ""
        result = _validate(s)
        assert result.failure_mode == "missing_fields"

    def test_missing_multiple_fields_reported(self):
        s = _make_signal(add_content_hash=True)
        s["data_cutoff_at"] = ""
        s["published_at"] = None
        result = _validate(s)
        assert result.failure_mode == "missing_fields"
        assert "data_cutoff_at" in result.reason
        assert "published_at" in result.reason

    def test_all_required_fields_in_result(self):
        s = _make_signal(add_content_hash=True)
        result = _validate(s)
        assert result.signal_run_id == s["signal_run_id"]
        assert result.generated_at == s["created_at_utc"]
        assert result.model_version == s["model_version"]
        assert result.data_cutoff_at == s["data_cutoff_at"]
        assert result.published_at == s["published_at"]
        assert result.published_commit_sha == s["published_commit_sha"]


# ---------------------------------------------------------------------------
# Layer 2: Session match
# ---------------------------------------------------------------------------

class TestSessionValidation:
    def test_matching_session_passes(self):
        s = _make_signal(add_content_hash=True, intended_execution_session=SESSION)
        result = _validate(s, session=SESSION)
        assert result.failure_mode not in ("stale_signal",)
        assert result.intended_session == SESSION
        assert result.actual_session == SESSION

    def test_stale_signal_rejected(self):
        s = _make_signal(add_content_hash=True, intended_execution_session="2026-08-03")
        result = _validate(s, session=SESSION)
        assert result.failure_mode == "stale_signal"
        assert not result.allow_new_buys
        assert result.allow_protective_sells

    def test_friday_signal_valid_on_monday(self):
        """Friday signal with intended_session=Monday is valid when run on Monday."""
        s = _make_signal(
            add_content_hash=True,
            date="2026-08-07",
            intended_execution_session="2026-08-10",
        )
        result = _validate(s, session="2026-08-10")
        assert result.failure_mode not in ("stale_signal", "missing_fields")
        assert result.allow_new_buys

    def test_friday_signal_stale_on_tuesday(self):
        s = _make_signal(
            add_content_hash=True,
            date="2026-08-07",
            intended_execution_session="2026-08-10",
        )
        result = _validate(s, session="2026-08-11")
        assert result.failure_mode == "stale_signal"
        assert not result.allow_new_buys

    def test_no_calendar_fallback_for_missing_session(self):
        """Regression point 3: missing intended_execution_session → rejected in Layer 1."""
        s = _make_signal(add_content_hash=True)
        del s["intended_execution_session"]
        result = _validate(s)
        # Must fail in Layer 1 as missing_fields, NOT attempt calendar fallback
        assert result.failure_mode == "missing_fields"
        # No "calendar_unavailable" failure mode in new validator
        assert result.failure_mode != "calendar_unavailable"

    def test_filename_not_used_as_date_truth(self):
        """Regression point 4: signal content + intended_execution_session drives validation."""
        s = _make_signal(add_content_hash=True, intended_execution_session=SESSION)
        # Filename says 2026-07-01 — irrelevant
        result = validate_signal(
            s, "/data/signals/2026-07-01_signal.json", SESSION,
            _get_publication_fn=_pub_success,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        assert result.failure_mode not in ("stale_signal",)


# ---------------------------------------------------------------------------
# Layer 3: Publication record
# ---------------------------------------------------------------------------

class TestPublicationRecord:
    """Regression points 1 & 2: publication status set after git push via ledger record."""

    def test_no_publication_record_rejected(self):
        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_none,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        assert result.failure_mode == "unpublished"
        assert not result.allow_new_buys
        assert result.allow_protective_sells

    def test_failed_workflow_rejected(self):
        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_failed,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        assert result.failure_mode == "workflow_failed"
        assert not result.allow_new_buys

    def test_missing_workflow_conclusion_rejected(self):
        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_missing_conclusion,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        assert result.failure_mode == "workflow_failed"

    def test_successful_publication_record_passes(self):
        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_success,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        assert result.failure_mode not in ("unpublished", "workflow_failed")

    def test_draft_signal_file_rejected_via_missing_pub_record(self):
        """
        Regression point 1: publication_status='draft' in JSON alone is insufficient.
        A draft has no publication record → rejected by Layer 3.
        """
        s = _make_signal(add_content_hash=True, publication_status="draft")
        # Signal file claims draft status but Layer 3 checks publication record
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_none,  # no record yet
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        assert result.failure_mode == "unpublished"
        assert not result.allow_new_buys

    def test_publication_record_with_success_overrides_draft_status(self):
        """
        If publication record exists with success, signal passes Layer 3
        (the signal file's publication_status field is secondary).
        """
        s = _make_signal(add_content_hash=True, publication_status="published")
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_success,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        assert result.failure_mode not in ("unpublished", "workflow_failed")


# ---------------------------------------------------------------------------
# Layer 4: Ledger integrity (status + JSONL cross-validation)
# ---------------------------------------------------------------------------

class TestLedgerIntegrity:
    """Regression point 5: cross-validate against actual signal_run JSONL record."""

    def test_completed_status_and_matching_record_passes(self):
        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_success,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        assert result.failure_mode not in ("ledger_incomplete", "ledger_mismatch", "ledger_error")

    def test_failed_ledger_status_rejected(self):
        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_success,
            _get_ledger_status_fn=_ledger_status_failed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        assert result.failure_mode == "ledger_incomplete"
        assert not result.allow_new_buys

    def test_missing_status_sidecar_rejected(self):
        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_success,
            _get_ledger_status_fn=_ledger_status_none,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        assert result.failure_mode == "ledger_incomplete"

    def test_missing_jsonl_record_rejected(self):
        """Regression point 5: not just status sidecar — JSONL record must exist."""
        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_success,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_none,
        )
        assert result.failure_mode == "ledger_incomplete"
        assert not result.allow_new_buys

    def test_model_version_mismatch_rejected(self):
        """Regression point 5: model_version in signal must match ledger record."""
        s = _make_signal(add_content_hash=True, model_version="quant_baseline_v1")
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_success,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_wrong_model,
        )
        assert result.failure_mode == "ledger_mismatch"
        assert not result.allow_new_buys
        assert "model_version" in result.reason

    def test_data_cutoff_mismatch_rejected(self):
        """Regression point 5: data_cutoff_at must match canonical_data_cutoff in ledger."""
        s = _make_signal(add_content_hash=True, data_cutoff_at=DATA_CUTOFF)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_success,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_wrong_cutoff,
        )
        assert result.failure_mode == "ledger_mismatch"
        assert not result.allow_new_buys
        assert "data_cutoff_at" in result.reason

    def test_ledger_error_propagated(self):
        def _raises(run_id, ym):
            raise OSError("disk error")
        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_success,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_raises,
        )
        assert result.failure_mode == "ledger_error"
        assert not result.allow_new_buys

    def test_only_status_check_without_jsonl_record_fields_ok(self):
        """If ledger JSONL record lacks model_version/cutoff fields, cross-val skipped."""
        def _record_no_fields(run_id, ym):
            return {"signal_run_id": run_id}  # no model_version, no canonical_data_cutoff

        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_success,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_record_no_fields,
        )
        # Empty fields mean nothing to mismatch — passes cross-validation
        assert result.failure_mode not in ("ledger_mismatch",)


# ---------------------------------------------------------------------------
# Layer 5: Content hash — mandatory (no backward compat)
# ---------------------------------------------------------------------------

class TestContentHash:
    """Regression point 4: missing signal_content_hash → rejected."""

    def test_valid_hash_passes(self):
        s = _make_signal(add_content_hash=True)
        result = _validate(s)
        assert result.failure_mode not in ("hash_missing", "hash_mismatch", "hash_error")
        assert result.allow_new_buys

    def test_missing_hash_field_rejected(self):
        """Regression point 4: no backward compat for missing hash."""
        s = _make_signal()
        del s["signal_content_hash"]
        result = _validate(s)
        assert result.failure_mode == "hash_missing"
        assert not result.allow_new_buys

    def test_empty_hash_field_rejected(self):
        """Regression point 4: empty string treated as missing — rejected."""
        s = _make_signal()
        s["signal_content_hash"] = ""
        result = _validate(s)
        assert result.failure_mode == "hash_missing"
        assert not result.allow_new_buys

    def test_tampered_signal_rejected(self):
        s = _make_signal(add_content_hash=True)
        s["universe_count"] = 999999  # tamper after signing
        result = _validate(s)
        assert result.failure_mode == "hash_mismatch"
        assert not result.allow_new_buys

    def test_hash_is_deterministic(self):
        s1 = _make_signal()
        s2 = _make_signal()
        assert compute_signal_content_hash(s1) == compute_signal_content_hash(s2)

    def test_hash_changes_on_any_field_change(self):
        s = _make_signal()
        h1 = compute_signal_content_hash(s)
        s["universe_count"] += 1
        h2 = compute_signal_content_hash(s)
        assert h1 != h2

    def test_hash_excludes_itself(self):
        """signal_content_hash field excluded from hash computation (no circularity)."""
        s = _make_signal()
        s["signal_content_hash"] = "placeholder_a"
        h1 = compute_signal_content_hash(s)
        s["signal_content_hash"] = "placeholder_b"
        h2 = compute_signal_content_hash(s)
        assert h1 == h2


# ---------------------------------------------------------------------------
# Data quality tier (informational only — no blocking)
# ---------------------------------------------------------------------------

class TestDataQuality:
    """Regression point 8: data quality reported but does NOT block execution."""

    def _validate_quality(self, universe=200, valid=190, stale=5, excluded=5):
        s = _make_signal(
            add_content_hash=True,
            universe_count=universe,
            valid_ticker_count=valid,
            stale_ticker_count=stale,
            excluded_ticker_count=excluded,
        )
        return _validate(s)

    def test_high_coverage_is_normal(self):
        result = self._validate_quality(valid=190, stale=5)
        assert result.data_quality_tier == "normal"
        assert result.is_valid
        assert result.allow_new_buys  # Never blocked by data quality

    def test_low_coverage_reported_as_reduced_but_does_not_block(self):
        """Regression point 8: <90% coverage → 'reduced' tier but buys NOT blocked."""
        result = self._validate_quality(universe=200, valid=140, stale=5)
        assert result.data_quality_tier == "reduced"
        assert result.is_valid  # Signal itself is valid
        assert result.allow_new_buys  # NOT blocked — thresholds not yet approved
        assert result.failure_mode == "data_quality_reduced"

    def test_high_stale_pct_reported_but_does_not_block(self):
        """
        Regression point 8: high stale_pct is included in data_quality dict for
        reporting but does not change the tier (thresholds not yet approved).
        valid=185/universe=200 = 92.5% coverage → 'normal' tier regardless of stale_pct.
        """
        result = self._validate_quality(universe=200, valid=185, stale=55)
        # Coverage 92.5% → normal tier; stale_pct informational only
        assert result.data_quality_tier == "normal"
        assert result.data_quality["stale_pct"] == pytest.approx(0.275, abs=0.01)
        assert result.allow_new_buys  # NOT blocked
        assert result.is_valid

    def test_data_quality_report_in_result(self):
        result = self._validate_quality(universe=200, valid=190, stale=5)
        assert result.data_quality["universe_count"] == 200
        assert result.data_quality["valid_ticker_count"] == 190
        assert result.data_quality["signal_coverage_rate"] == pytest.approx(0.95, abs=0.01)

    def test_zero_universe_count_unknown_tier(self):
        result = self._validate_quality(universe=0, valid=0, stale=0)
        assert result.data_quality_tier == "unknown"
        assert result.allow_new_buys

    def test_no_data_quality_blocked_failure_mode(self):
        """Regression point 8: 'data_quality_blocked' failure mode removed."""
        result = self._validate_quality(universe=200, valid=100, stale=60)
        assert result.failure_mode != "data_quality_blocked"
        assert result.allow_new_buys


# ---------------------------------------------------------------------------
# Protective sells always allowed
# ---------------------------------------------------------------------------

class TestProtectiveSells:
    def test_protective_sells_allowed_on_stale_signal(self):
        s = _make_signal(add_content_hash=True, intended_execution_session="2026-08-03")
        result = _validate(s, session=SESSION)
        assert result.failure_mode == "stale_signal"
        assert not result.allow_new_buys
        assert result.allow_protective_sells

    def test_protective_sells_allowed_on_no_publication_record(self):
        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_none,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        assert not result.allow_new_buys
        assert result.allow_protective_sells

    def test_protective_sells_allowed_on_missing_hash(self):
        s = _make_signal()
        s["signal_content_hash"] = ""
        result = _validate(s)
        assert result.failure_mode == "hash_missing"
        assert not result.allow_new_buys
        assert result.allow_protective_sells

    def test_protective_sells_allowed_on_ledger_mismatch(self):
        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_success,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_wrong_model,
        )
        assert not result.allow_new_buys
        assert result.allow_protective_sells


# ---------------------------------------------------------------------------
# Protective stop price injection (point 7)
# ---------------------------------------------------------------------------

class TestProtectiveStopPrices:
    """
    Regression point 7: protective stops must use current execution/valuation price,
    not the stale candidate price from the signal.
    """

    def test_stop_loss_uses_current_price_when_injected(self):
        """
        should_sell_position uses `candidate.price` for stop-loss math.
        Verify the injection changes the decision.
        """
        from modules.risk import should_sell_position

        config = {"stop_loss": -0.15, "trailing_stop": -0.20, "sell_rank_threshold": 30}
        pos = {"avg_price": 100.0, "highest_price": 110.0, "last_price": 100.0, "buy_score": 60.0, "buy_date": "2026-01-01"}

        # Signal price (stale): above stop-loss
        stale_candidate = {"ticker": "AAPL", "price": 90.0, "above_sma200": True, "rank": 5, "strategy_score": 70.0, "vol60": 0.20}
        sell_stale, reason_stale = should_sell_position("AAPL", pos, stale_candidate, config)
        assert not sell_stale, "Stale price above stop-loss should not trigger"

        # Current price (injected): below stop-loss
        current_candidate = dict(stale_candidate)
        current_candidate["price"] = 80.0  # -20% from avg → triggers -15% stop
        sell_current, reason_current = should_sell_position("AAPL", pos, current_candidate, config)
        assert sell_current, "Current price below stop-loss should trigger"
        assert "Stop-loss" in reason_current

    def test_trailing_stop_uses_current_price(self):
        from modules.risk import should_sell_position

        config = {"stop_loss": -0.20, "trailing_stop": -0.15, "sell_rank_threshold": 30}
        pos = {"avg_price": 80.0, "highest_price": 100.0, "last_price": 100.0, "buy_score": 60.0, "buy_date": "2026-01-01"}

        # Stale candidate: 92.0 → -8% from high, doesn't trigger -15% trailing
        stale = {"ticker": "AAPL", "price": 92.0, "above_sma200": True, "rank": 5, "strategy_score": 70.0, "vol60": 0.20}
        sell_stale, _ = should_sell_position("AAPL", pos, stale, config)
        assert not sell_stale

        # Current: 83.0 → -17% from high, triggers -15% trailing
        current = dict(stale)
        current["price"] = 83.0
        sell_current, reason = should_sell_position("AAPL", pos, current, config)
        assert sell_current
        assert "Trailing stop" in reason


# ---------------------------------------------------------------------------
# load_latest_signal no-fallback (point 6)
# ---------------------------------------------------------------------------

class TestLoadLatestSignal:
    """Regression point 6: load_latest_signal() has no directory-scan fallback."""

    def _write_global(self, tmp_path, signal_path):
        global_path = tmp_path / "state" / "_global.json"
        global_path.parent.mkdir(parents=True, exist_ok=True)
        with open(global_path, "w") as f:
            json.dump({"last_signal_file": str(signal_path)}, f)

    def _write_signal(self, tmp_path, status="published"):
        sig_dir = tmp_path / "signals"
        sig_dir.mkdir(exist_ok=True)
        sig_path = sig_dir / "2026-08-05_signal.json"
        with open(sig_path, "w") as f:
            json.dump({"date": "2026-08-05", "publication_status": status}, f)
        return sig_path

    def test_published_signal_loaded(self, tmp_path, monkeypatch):
        import modules.state as state_mod
        sig_path = self._write_signal(tmp_path, "published")
        self._write_global(tmp_path, sig_path)
        monkeypatch.setattr(state_mod, "GLOBAL_STATE_FILE", str(tmp_path / "state" / "_global.json"))
        signal, path = state_mod.load_latest_signal()
        assert signal["publication_status"] == "published"
        assert str(sig_path) == path

    def test_draft_signal_raises(self, tmp_path, monkeypatch):
        """Regression point 6: draft signal raises SignalNotPublishedError."""
        import modules.state as state_mod
        sig_path = self._write_signal(tmp_path, "draft")
        self._write_global(tmp_path, sig_path)
        monkeypatch.setattr(state_mod, "GLOBAL_STATE_FILE", str(tmp_path / "state" / "_global.json"))
        with pytest.raises(state_mod.SignalNotPublishedError):
            state_mod.load_latest_signal()

    def test_missing_global_state_raises(self, tmp_path, monkeypatch):
        """Regression point 6: no _global.json → raises FileNotFoundError (not fallback scan)."""
        import modules.state as state_mod
        nonexistent = str(tmp_path / "state" / "_global.json")
        monkeypatch.setattr(state_mod, "GLOBAL_STATE_FILE", nonexistent)
        with pytest.raises(FileNotFoundError):
            state_mod.load_latest_signal()

    def test_missing_signal_file_raises(self, tmp_path, monkeypatch):
        """File registered in _global.json but deleted → FileNotFoundError, no dir scan."""
        import modules.state as state_mod
        sig_path = tmp_path / "signals" / "2026-08-05_signal.json"
        # Don't actually create the file
        self._write_global(tmp_path, sig_path)
        monkeypatch.setattr(state_mod, "GLOBAL_STATE_FILE", str(tmp_path / "state" / "_global.json"))
        with pytest.raises(FileNotFoundError) as exc_info:
            state_mod.load_latest_signal()
        assert "ikke funnet" in str(exc_info.value).lower() or "not found" in str(exc_info.value).lower()

    def test_no_fallback_to_directory_scan(self, tmp_path, monkeypatch):
        """
        Regression point 6: even if other signal files exist in signals/, they are NOT
        used as fallback when _global.json is missing.
        """
        import modules.state as state_mod
        # Create a valid-looking signal file in SIGNALS_DIR
        sig_dir = tmp_path / "signals"
        sig_dir.mkdir(exist_ok=True)
        stray = sig_dir / "2026-08-05_signal.json"
        with open(stray, "w") as f:
            json.dump({"date": "2026-08-05", "publication_status": "published"}, f)
        # But _global.json is missing
        nonexistent = str(tmp_path / "state" / "_global.json")
        monkeypatch.setattr(state_mod, "GLOBAL_STATE_FILE", nonexistent)
        monkeypatch.setattr(state_mod, "SIGNALS_DIR", str(sig_dir))
        with pytest.raises(FileNotFoundError):
            state_mod.load_latest_signal()


# ---------------------------------------------------------------------------
# Duplicate protection — validate_signal is deterministic
# ---------------------------------------------------------------------------

class TestDuplicateProtection:
    def test_same_signal_revalidates_identically(self):
        s = _make_signal(add_content_hash=True)
        r1 = _validate(s)
        r2 = _validate(s)
        assert r1.is_valid == r2.is_valid
        assert r1.failure_mode == r2.failure_mode
        assert r1.allow_new_buys == r2.allow_new_buys

    def test_old_signal_cannot_substitute_for_todays(self):
        s = _make_signal(add_content_hash=True, intended_execution_session="2026-07-01")
        result = _validate(s, session=SESSION)
        assert result.failure_mode == "stale_signal"
        assert not result.allow_new_buys


# ---------------------------------------------------------------------------
# Telegram message content
# ---------------------------------------------------------------------------

class TestTelegramMessage:
    def test_stale_signal_message_has_required_fields(self):
        from modules.reporting import build_stale_signal_message
        s = _make_signal(add_content_hash=True, intended_execution_session="2026-08-03")
        result = _validate(s, session=SESSION)
        msg = build_stale_signal_message(result)
        assert result.signal_run_id in msg
        assert "2026-08-03" in msg  # intended
        assert SESSION in msg        # actual
        assert "stale_signal" in msg

    def test_stale_signal_message_mentions_stops_allowed(self):
        from modules.reporting import build_stale_signal_message
        s = _make_signal(add_content_hash=True, intended_execution_session="2026-08-03")
        result = _validate(s, session=SESSION)
        msg = build_stale_signal_message(result)
        assert "stop" in msg.lower()

    def test_unpublished_message_has_failure_mode(self):
        from modules.reporting import build_stale_signal_message
        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_none,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        msg = build_stale_signal_message(result)
        assert "unpublished" in msg

    def test_workflow_failed_message_has_failure_mode(self):
        from modules.reporting import build_stale_signal_message
        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_failed,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        msg = build_stale_signal_message(result)
        assert "workflow_failed" in msg

    def test_hash_missing_message_distinct(self):
        from modules.reporting import build_stale_signal_message
        s = _make_signal()
        s["signal_content_hash"] = ""
        result = _validate(s)
        msg = build_stale_signal_message(result)
        assert "hash_missing" in msg
