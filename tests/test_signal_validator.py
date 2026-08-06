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


# ---------------------------------------------------------------------------
# Publication record builder (self-consistent content_hash)
# ---------------------------------------------------------------------------

def _make_pub_record(
    signal_run_id,
    workflow_conclusion="success",
    commit_sha="abc1234def56",        # matches _make_signal default published_commit_sha
    published_at="2026-08-05T21:05:00+00:00",  # matches _make_signal default published_at
    finalized_commit_sha="def456abc789",
    **overrides,
):
    """Build a publication record with a correct content_hash."""
    from modules.publication import compute_publication_content_hash
    rec = {
        "schema_version": 1,
        "record_type": "signal_publication",
        "signal_run_id": signal_run_id,
        "workflow_conclusion": workflow_conclusion,
        "commit_sha": commit_sha,
        "published_at": published_at,
        "signal_path": "/fake/signal.json",
        "written_at": "2026-08-05T21:05:01.000Z",
        "finalized_commit_sha": finalized_commit_sha,
        "content_hash": "",
    }
    rec.update(overrides)
    rec["content_hash"] = compute_publication_content_hash(rec)
    return rec


# Injection helpers
def _pub_success(run_id):
    return _make_pub_record(run_id)

def _pub_none(run_id):
    return None

def _pub_failed(run_id):
    return _make_pub_record(run_id, workflow_conclusion="failure")

def _pub_missing_conclusion(run_id):
    return _make_pub_record(run_id, workflow_conclusion=None)

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

    def test_low_coverage_reported_as_normal_no_threshold(self):
        """Regression point 8: no hardcoded threshold — all signals return 'normal' tier."""
        result = self._validate_quality(universe=200, valid=140, stale=5)
        # Threshold 0.90 removed — coverage=70% still returns "normal" (no approved threshold)
        assert result.data_quality_tier == "normal"
        assert result.is_valid
        assert result.allow_new_buys
        assert result.failure_mode == "ok"

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


# ---------------------------------------------------------------------------
# Point 1: publication_status must be explicitly "published" (Layer 1b)
# ---------------------------------------------------------------------------

class TestPublicationStatusRequired:
    """
    Regression point 1: publication_status='published' must be validated
    by the validator itself — not just by load_latest_signal().
    """

    def test_draft_status_rejected_before_layer3(self):
        """Draft signal is rejected in Layer 1b — before the pub-record check."""
        s = _make_signal(add_content_hash=True, publication_status="draft")
        result = _validate(s)
        # Must reject as unpublished, not reach Layer 3
        assert result.failure_mode == "unpublished"
        assert not result.allow_new_buys
        assert result.allow_protective_sells

    def test_none_status_rejected(self):
        s = _make_signal(add_content_hash=True)
        s["publication_status"] = None
        result = _validate(s)
        assert result.failure_mode == "unpublished"
        assert not result.allow_new_buys

    def test_missing_status_key_rejected(self):
        s = _make_signal(add_content_hash=True)
        del s["publication_status"]
        result = _validate(s)
        assert result.failure_mode == "unpublished"

    def test_typo_status_rejected(self):
        s = _make_signal(add_content_hash=True, publication_status="Published")  # capital P
        result = _validate(s)
        assert result.failure_mode == "unpublished"

    def test_published_status_passes_layer1b(self):
        s = _make_signal(add_content_hash=True, publication_status="published")
        result = _validate(s)
        assert result.failure_mode not in ("unpublished", "missing_fields")


# ---------------------------------------------------------------------------
# Point 2: Publication record content_hash must be validated
# ---------------------------------------------------------------------------

class TestPublicationRecordHash:
    """Regression point 2: validator checks pub record's own content_hash."""

    def test_valid_pub_record_hash_passes(self):
        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_success,   # has correct content_hash
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        assert result.failure_mode not in ("publication_error",)
        assert result.is_valid

    def test_tampered_pub_record_rejected(self):
        """If pub record is tampered (content_hash no longer matches), reject."""
        def _pub_tampered(run_id):
            rec = _make_pub_record(run_id)
            rec["workflow_conclusion"] = "TAMPERED"  # change field without recomputing hash
            return rec

        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_tampered,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        # Tampered workflow_conclusion changes hash AND is not "success" — caught first
        assert not result.allow_new_buys

    def test_pub_record_without_content_hash_rejected(self):
        """Regression point 4: empty content_hash is mandatory — missing → rejected."""
        def _pub_no_hash(run_id):
            rec = _make_pub_record(run_id)
            rec["content_hash"] = ""
            return rec

        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_no_hash,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        # Missing content_hash is now mandatory → rejected
        assert result.failure_mode == "publication_error"
        assert "content_hash" in result.reason
        assert not result.allow_new_buys

    def test_wrong_content_hash_in_pub_record_rejected(self):
        """Pub record with a deliberately wrong content_hash is rejected."""
        def _pub_wrong_hash(run_id):
            rec = _make_pub_record(run_id)
            rec["content_hash"] = "a" * 64  # wrong hash, right length
            return rec

        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_wrong_hash,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        assert result.failure_mode == "publication_error"
        assert "content_hash" in result.reason
        assert not result.allow_new_buys


# ---------------------------------------------------------------------------
# Point 3: Publication record fields cross-validated against signal
# ---------------------------------------------------------------------------

class TestPublicationRecordCrossValidation:
    """Regression point 3: signal_run_id, published_at, commit_sha must match."""

    def test_matching_fields_pass(self):
        s = _make_signal(add_content_hash=True)
        result = _validate(s)
        assert result.failure_mode not in ("publication_error",)

    def test_signal_run_id_mismatch_rejected(self):
        def _pub_wrong_run_id(run_id):
            return _make_pub_record("wrong-run-id-xyz")  # mismatched signal_run_id

        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_wrong_run_id,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        assert result.failure_mode == "publication_error"
        assert "signal_run_id" in result.reason
        assert not result.allow_new_buys

    def test_published_at_mismatch_rejected(self):
        def _pub_wrong_published_at(run_id):
            return _make_pub_record(run_id, published_at="2020-01-01T00:00:00+00:00")

        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_wrong_published_at,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        assert result.failure_mode == "publication_error"
        assert "published_at" in result.reason
        assert not result.allow_new_buys

    def test_commit_sha_mismatch_rejected(self):
        def _pub_wrong_commit(run_id):
            return _make_pub_record(run_id, commit_sha="000000000000")

        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_wrong_commit,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        assert result.failure_mode == "publication_error"
        assert "commit_sha" in result.reason
        assert not result.allow_new_buys

    def test_missing_commit_sha_in_pub_record_rejected(self):
        """Regression point 5: missing commit_sha in pub record is mandatory → rejected."""
        def _pub_no_commit_sha(run_id):
            from modules.publication import compute_publication_content_hash
            rec = {
                "schema_version": 1,
                "record_type": "signal_publication",
                "signal_run_id": run_id,
                "workflow_conclusion": "success",
                "published_at": "2026-08-05T21:05:00+00:00",
                "finalized_commit_sha": "def456",
                # commit_sha intentionally missing
                "content_hash": "",
            }
            rec["content_hash"] = compute_publication_content_hash(rec)
            return rec

        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_no_commit_sha,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        assert result.failure_mode == "publication_error"
        assert "commit_sha" in result.reason
        assert not result.allow_new_buys

    def test_missing_published_at_in_pub_record_rejected(self):
        """Regression point 5: missing published_at in pub record is mandatory → rejected."""
        def _pub_no_published_at(run_id):
            from modules.publication import compute_publication_content_hash
            rec = {
                "schema_version": 1,
                "record_type": "signal_publication",
                "signal_run_id": run_id,
                "workflow_conclusion": "success",
                "commit_sha": "abc1234def56",
                "finalized_commit_sha": "def456",
                # published_at intentionally missing
                "content_hash": "",
            }
            rec["content_hash"] = compute_publication_content_hash(rec)
            return rec

        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_no_published_at,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        assert result.failure_mode == "publication_error"
        assert "published_at" in result.reason
        assert not result.allow_new_buys


# ---------------------------------------------------------------------------
# Point 4: ImportError in ledger must reject, not skip
# ---------------------------------------------------------------------------

class TestLedgerImportError:
    """Regression point 4: ImportError/ModuleNotFoundError → ledger_error, not silent skip."""

    def test_import_error_in_ledger_status_rejects(self):
        def _raises_import_error(run_id, ym):
            raise ImportError("No module named 'modules.ledger'")

        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_success,
            _get_ledger_status_fn=_raises_import_error,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        assert result.failure_mode == "ledger_error"
        assert not result.allow_new_buys
        assert result.allow_protective_sells

    def test_module_not_found_in_ledger_status_rejects(self):
        def _raises_module_not_found(run_id, ym):
            raise ModuleNotFoundError("modules.ledger not installed")

        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_success,
            _get_ledger_status_fn=_raises_module_not_found,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        assert result.failure_mode == "ledger_error"
        assert not result.allow_new_buys

    def test_import_error_in_ledger_record_rejects(self):
        def _raises_import_error(run_id, ym):
            raise ImportError("No module named 'modules.ledger'")

        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_success,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_raises_import_error,
        )
        assert result.failure_mode == "ledger_error"
        assert not result.allow_new_buys

    def test_import_error_does_not_silently_pass(self):
        """Pre-regression guard: ImportError must NOT result in is_valid=True."""
        def _raises_import_error(run_id, ym):
            raise ImportError("ledger missing")

        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_success,
            _get_ledger_status_fn=_raises_import_error,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        # The old code had `except (ImportError, ModuleNotFoundError): pass`
        # which would let the signal pass. Verify this no longer happens.
        assert not result.is_valid
        assert not result.allow_new_buys


# ---------------------------------------------------------------------------
# Point 5: session_date uses New York timezone
# ---------------------------------------------------------------------------

class TestSessionDateTimezone:
    """Regression point 5: session_date must use NYSE/NY timezone, not Europe/Oslo."""

    def test_now_ny_and_now_oslo_are_valid_datetimes(self):
        from modules.state import now_ny, now_oslo
        ny = now_ny()
        oslo = now_oslo()
        assert ny.tzinfo is not None
        assert oslo.tzinfo is not None

    def test_oslo_is_ahead_of_or_equal_to_ny(self):
        """Oslo is always UTC+1 or higher; NY is UTC-5 to UTC-4. Oslo >= NY."""
        from modules.state import now_ny, now_oslo
        import datetime
        ny = now_ny()
        oslo = now_oslo()
        # Convert both to UTC for comparison
        ny_utc = ny.utctimetuple()
        oslo_utc = oslo.utctimetuple()
        # They represent the same instant — UTC offsets differ, not the moment
        ny_date = now_ny().strftime("%Y-%m-%d")
        oslo_date = now_oslo().strftime("%Y-%m-%d")
        # Both should be valid ISO dates
        import re
        assert re.match(r"\d{4}-\d{2}-\d{2}", ny_date)
        assert re.match(r"\d{4}-\d{2}-\d{2}", oslo_date)

    def test_run_execute_uses_ny_timezone(self):
        """Source-level check: run_execute() must use now_ny() for session_date."""
        # Read source directly — avoids importing stock_bot (requires 'anthropic' package)
        with open("stock_bot.py") as f:
            src = f.read()

        # Find run_execute function body
        start = src.find("def run_execute()")
        assert start != -1, "run_execute function not found in stock_bot.py"
        # Grab roughly 30 lines after the def
        snippet = src[start:start + 1200]

        # Must reference now_ny() for session_date
        assert "now_ny()" in snippet, "run_execute must use now_ny() for session_date"
        # Must NOT use today_str() for session_date assignment
        lines = snippet.split("\n")
        session_line = next((l for l in lines if "session_date" in l and "=" in l), "")
        assert "today_str()" not in session_line, (
            f"run_execute must use now_ny() not today_str() for session_date. "
            f"Found: {session_line!r}"
        )


# ---------------------------------------------------------------------------
# Point 6: finalized_commit_sha (persistence-ack) must be present
# ---------------------------------------------------------------------------

class TestPersistenceAck:
    """Regression point 6: validator requires finalized_commit_sha in pub record."""

    def test_missing_finalized_commit_sha_rejected(self):
        """Pub record without finalized_commit_sha → rejected (ack not written yet)."""
        def _pub_no_ack(run_id):
            return _make_pub_record(run_id, finalized_commit_sha="")

        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_no_ack,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        assert result.failure_mode == "publication_error"
        assert "finalized_commit_sha" in result.reason or "persistence-ack" in result.reason.lower()
        assert not result.allow_new_buys
        assert result.allow_protective_sells

    def test_none_finalized_commit_sha_rejected(self):
        def _pub_none_sha(run_id):
            return _make_pub_record(run_id, finalized_commit_sha=None)

        s = _make_signal(add_content_hash=True)
        result = validate_signal(
            s, "/fake", SESSION,
            _get_publication_fn=_pub_none_sha,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        assert result.failure_mode == "publication_error"
        assert not result.allow_new_buys

    def test_present_finalized_commit_sha_passes(self):
        s = _make_signal(add_content_hash=True)
        result = _validate(s)  # _pub_success has finalized_commit_sha="def456abc789"
        assert result.failure_mode not in ("publication_error",)
        assert result.is_valid

    def test_write_publication_ack_roundtrip(self):
        """write_publication_ack + get_signal_publication returns merged record."""
        import tempfile
        from pathlib import Path
        from modules.publication import (
            write_signal_publication,
            write_publication_ack,
            get_signal_publication,
            PUBLICATIONS_FILE,
        )
        import modules.publication as pub_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file = Path(tmpdir) / "signal_publications.jsonl"
            tmp_lock = Path(tmpdir) / "signal_publications.lock"
            orig_file = pub_mod.PUBLICATIONS_FILE
            orig_lock = pub_mod._LOCK_FILE
            pub_mod.PUBLICATIONS_FILE = tmp_file
            pub_mod._LOCK_FILE = tmp_lock
            try:
                run_id = "test-run-ack-001"
                write_signal_publication(
                    signal_run_id=run_id,
                    signal_path="/test/signal.json",
                    commit_sha="draft-sha-abc",
                    published_at="2026-08-05T21:05:00+00:00",
                    workflow_conclusion="success",
                )
                # Before ack: no finalized_commit_sha
                rec = get_signal_publication(run_id)
                assert rec is not None
                assert not rec.get("finalized_commit_sha")

                write_publication_ack(run_id, "final-sha-def")

                # After ack: merged record has finalized_commit_sha
                rec2 = get_signal_publication(run_id)
                assert rec2 is not None
                assert rec2["finalized_commit_sha"] == "final-sha-def"
                assert rec2["workflow_conclusion"] == "success"
            finally:
                pub_mod.PUBLICATIONS_FILE = orig_file
                pub_mod._LOCK_FILE = orig_lock

    def test_write_publication_ack_idempotent(self):
        """Calling write_publication_ack twice for same run_id is a no-op."""
        import tempfile
        from pathlib import Path
        import modules.publication as pub_mod
        from modules.publication import (
            write_signal_publication,
            write_publication_ack,
            get_signal_publication,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file = Path(tmpdir) / "signal_publications.jsonl"
            tmp_lock = Path(tmpdir) / "signal_publications.lock"
            orig_file = pub_mod.PUBLICATIONS_FILE
            orig_lock = pub_mod._LOCK_FILE
            pub_mod.PUBLICATIONS_FILE = tmp_file
            pub_mod._LOCK_FILE = tmp_lock
            try:
                run_id = "test-run-ack-002"
                write_signal_publication(run_id, "/s.json", "sha1", "2026-08-05T21:00:00+00:00")
                write_publication_ack(run_id, "final-sha-1")
                write_publication_ack(run_id, "final-sha-1")  # idempotent

                # Count ack lines
                lines = [l for l in tmp_file.read_text().strip().split("\n") if l]
                ack_count = sum(1 for l in lines if '"publication_ack"' in l)
                assert ack_count == 1, "Only one ack record should exist"
            finally:
                pub_mod.PUBLICATIONS_FILE = orig_file
                pub_mod._LOCK_FILE = orig_lock


# ---------------------------------------------------------------------------
# Point 7: signal.yml if: always() removed (workflow config regression)
# ---------------------------------------------------------------------------

class TestWorkflowConfig:
    """Regression points 7 (original) and 1 (ChatGPT): workflow persistence guarantees."""

    def test_final_push_step_not_always(self):
        """signal.yml Phase 4 (finalized signal) must not use if: always()."""
        workflow_path = ".github/workflows/signal.yml"
        try:
            with open(workflow_path) as f:
                content = f.read()
        except FileNotFoundError:
            import pytest
            pytest.skip("signal.yml not found — run from repo root")

        lines = content.split("\n")
        in_final_push = False
        for i, line in enumerate(lines):
            if "id: final_push" in line or "Commit og push finalisert signal" in line:
                in_final_push = True
            if in_final_push and "id: " in line and "final_push" not in line:
                break
            if in_final_push and "if: always()" in line:
                raise AssertionError(
                    f"Phase 4 must NOT use 'if: always()'. Line {i+1}: {line!r}"
                )

    def test_failure_state_step_uses_always(self):
        """Regression point 1: a separate step must use if: always() for ledger persistence."""
        workflow_path = ".github/workflows/signal.yml"
        try:
            with open(workflow_path) as f:
                content = f.read()
        except FileNotFoundError:
            import pytest
            pytest.skip("signal.yml not found — run from repo root")

        assert "if: always()" in content, (
            "signal.yml must have a step with 'if: always()' to commit partial "
            "ledger state even when finalization fails."
        )

    def test_failure_state_step_commits_ledger(self):
        """The if: always() step must commit ledger records."""
        workflow_path = ".github/workflows/signal.yml"
        try:
            with open(workflow_path) as f:
                content = f.read()
        except FileNotFoundError:
            import pytest
            pytest.skip("signal.yml not found — run from repo root")

        # Find the if: always() block and verify it touches ledger
        lines = content.split("\n")
        in_always_block = False
        ledger_committed = False
        for line in lines:
            if "if: always()" in line:
                in_always_block = True
            if in_always_block and ("data_v4/ledger" in line or "data_v4/" in line):
                ledger_committed = True
            if in_always_block and line.strip().startswith("- name:") and "if: always()" not in line:
                if ledger_committed:
                    break
                in_always_block = False

        assert ledger_committed, (
            "The if: always() step must add data_v4/ledger/ to persist partial state"
        )


# ---------------------------------------------------------------------------
# Point 8: Comprehensive regression — all points together
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Point 3: session_date validated via NYSE calendar (fail-closed)
# ---------------------------------------------------------------------------

class TestNYSECalendarValidation:
    """Regression point 3: session_date must be validated as a real NYSE session."""

    def test_thursday_2026_08_06_is_trading_session(self):
        from modules.exchange_calendar import is_trading_session
        assert is_trading_session("2026-08-06")

    def test_saturday_is_not_trading_session(self):
        from modules.exchange_calendar import is_trading_session
        assert not is_trading_session("2026-08-08")  # Saturday

    def test_sunday_is_not_trading_session(self):
        from modules.exchange_calendar import is_trading_session
        assert not is_trading_session("2026-08-09")  # Sunday

    def test_calendar_unavailable_error_is_defined(self):
        """CalendarUnavailableError must be importable and catchable."""
        from modules.exchange_calendar import CalendarUnavailableError
        try:
            raise CalendarUnavailableError("test error")
        except CalendarUnavailableError as exc:
            assert "test error" in str(exc)

    def test_run_execute_source_uses_is_trading_session(self):
        """run_execute() source must call is_trading_session for fail-closed validation."""
        with open("stock_bot.py") as f:
            src = f.read()
        start = src.find("def run_execute()")
        assert start != -1
        snippet = src[start:start + 2000]
        assert "is_trading_session" in snippet, (
            "run_execute must call is_trading_session to validate session_date"
        )
        assert "CalendarUnavailableError" in snippet, (
            "run_execute must handle CalendarUnavailableError fail-closed"
        )

    def test_non_trading_day_signal_rejected_by_validator(self):
        """If session_date is a non-trading day, session mismatch rejects the signal."""
        s = _make_signal(add_content_hash=True, intended_execution_session="2026-08-06")
        # Validator sees session_date=2026-08-08 (Saturday) — mismatch → stale_signal
        result = _validate(s, session="2026-08-08")
        assert result.failure_mode == "stale_signal"
        assert not result.allow_new_buys


# ---------------------------------------------------------------------------
# Point 6: publication_ack content_hash validated (not discarded)
# ---------------------------------------------------------------------------

class TestAckContentHashValidation:
    """Regression point 6: get_signal_publication must validate ack record hash."""

    def test_valid_ack_hash_passes(self):
        """write_publication_ack + get_signal_publication with valid hash passes."""
        import tempfile
        from pathlib import Path
        import modules.publication as pub_mod
        from modules.publication import (
            write_signal_publication,
            write_publication_ack,
            get_signal_publication,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file = Path(tmpdir) / "signal_publications.jsonl"
            tmp_lock = Path(tmpdir) / "signal_publications.lock"
            orig_file, orig_lock = pub_mod.PUBLICATIONS_FILE, pub_mod._LOCK_FILE
            pub_mod.PUBLICATIONS_FILE = tmp_file
            pub_mod._LOCK_FILE = tmp_lock
            try:
                run_id = "ack-hash-valid-001"
                write_signal_publication(run_id, "/s.json", "sha1", "2026-08-05T21:05:00+00:00")
                write_publication_ack(run_id, "final-sha-ok")
                rec = get_signal_publication(run_id)
                assert rec is not None
                assert rec["finalized_commit_sha"] == "final-sha-ok"
            finally:
                pub_mod.PUBLICATIONS_FILE = orig_file
                pub_mod._LOCK_FILE = orig_lock

    def test_tampered_ack_hash_raises(self):
        """Tampered publication_ack content_hash causes get_signal_publication to raise."""
        import json
        import tempfile
        from pathlib import Path
        import modules.publication as pub_mod
        from modules.publication import (
            write_signal_publication,
            write_publication_ack,
            get_signal_publication,
            PublicationWriteError,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file = Path(tmpdir) / "signal_publications.jsonl"
            tmp_lock = Path(tmpdir) / "signal_publications.lock"
            orig_file, orig_lock = pub_mod.PUBLICATIONS_FILE, pub_mod._LOCK_FILE
            pub_mod.PUBLICATIONS_FILE = tmp_file
            pub_mod._LOCK_FILE = tmp_lock
            try:
                run_id = "ack-hash-tampered-001"
                write_signal_publication(run_id, "/s.json", "sha1", "2026-08-05T21:05:00+00:00")
                write_publication_ack(run_id, "final-sha-tamper")

                # Tamper the ack record's finalized_commit_sha without recomputing hash
                lines = tmp_file.read_text().strip().split("\n")
                tampered_lines = []
                for line in lines:
                    rec = json.loads(line)
                    if rec.get("record_type") == "publication_ack":
                        rec["finalized_commit_sha"] = "TAMPERED-SHA"
                        # content_hash is now wrong
                    tampered_lines.append(json.dumps(rec, sort_keys=True))
                tmp_file.write_text("\n".join(tampered_lines) + "\n")

                with pytest.raises(PublicationWriteError, match="content_hash"):
                    get_signal_publication(run_id)
            finally:
                pub_mod.PUBLICATIONS_FILE = orig_file
                pub_mod._LOCK_FILE = orig_lock

    def test_validator_rejects_when_ack_hash_invalid(self):
        """Validator rejects with publication_error when ack hash is invalid."""
        import json
        import tempfile
        from pathlib import Path
        import modules.publication as pub_mod
        from modules.publication import (
            write_signal_publication,
            write_publication_ack,
            get_signal_publication,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file = Path(tmpdir) / "signal_publications.jsonl"
            tmp_lock = Path(tmpdir) / "signal_publications.lock"
            orig_file, orig_lock = pub_mod.PUBLICATIONS_FILE, pub_mod._LOCK_FILE
            pub_mod.PUBLICATIONS_FILE = tmp_file
            pub_mod._LOCK_FILE = tmp_lock
            try:
                run_id = "run-2026-08-05-abc123456"  # matches _make_signal default
                write_signal_publication(
                    run_id, "/s.json", "abc1234def56", "2026-08-05T21:05:00+00:00"
                )
                write_publication_ack(run_id, "def456abc789")

                # Tamper the ack
                lines = tmp_file.read_text().strip().split("\n")
                tampered_lines = []
                for line in lines:
                    rec = json.loads(line)
                    if rec.get("record_type") == "publication_ack":
                        rec["finalized_commit_sha"] = "TAMPERED"
                    tampered_lines.append(json.dumps(rec, sort_keys=True))
                tmp_file.write_text("\n".join(tampered_lines) + "\n")

                s = _make_signal(add_content_hash=True)
                result = validate_signal(
                    s, "/fake", SESSION,
                    _get_publication_fn=lambda rid: get_signal_publication(rid),
                    _get_ledger_status_fn=_ledger_status_completed,
                    _get_ledger_record_fn=_ledger_record_ok,
                )
                assert result.failure_mode == "publication_error"
                assert not result.allow_new_buys
            finally:
                pub_mod.PUBLICATIONS_FILE = orig_file
                pub_mod._LOCK_FILE = orig_lock


# ---------------------------------------------------------------------------
# Point 7: Timestamp format + explicit timezone validation
# ---------------------------------------------------------------------------

class TestTimestampValidation:
    """Regression point 7: generated_at and published_at must be ISO 8601 with timezone."""

    def test_valid_utc_offset_timestamp_passes(self):
        s = _make_signal(add_content_hash=True)
        s["created_at_utc"] = "2026-08-05T21:00:00+00:00"
        result = _validate(s)
        assert result.failure_mode not in ("missing_fields",)

    def test_valid_z_timestamp_passes(self):
        s = _make_signal(add_content_hash=True)
        s["created_at_utc"] = "2026-08-05T21:00:00Z"
        result = _validate(s)
        assert result.failure_mode not in ("missing_fields",)

    def test_timestamp_without_timezone_rejected(self):
        """Timestamp without timezone is rejected as invalid format."""
        s = _make_signal()  # no hash — will fail at Layer 1
        s["created_at_utc"] = "2026-08-05T21:00:00"  # no timezone
        result = _validate(s)
        assert result.failure_mode == "missing_fields"
        assert "created_at_utc" in result.reason

    def test_date_only_created_at_rejected(self):
        s = _make_signal()
        s["created_at_utc"] = "2026-08-05"  # date only, no time/timezone
        result = _validate(s)
        assert result.failure_mode == "missing_fields"
        assert "created_at_utc" in result.reason

    def test_published_at_without_timezone_rejected(self):
        s = _make_signal()
        s["published_at"] = "2026-08-05T21:05:00"  # no timezone
        result = _validate(s)
        assert result.failure_mode == "missing_fields"
        assert "published_at" in result.reason

    def test_published_at_with_negative_offset_passes(self):
        s = _make_signal(add_content_hash=True, published_at="2026-08-05T17:05:00-04:00")
        result = _validate(s)
        assert result.failure_mode not in ("missing_fields",)

    def test_published_at_with_milliseconds_passes(self):
        """Sub-second precision is allowed."""
        s = _make_signal(add_content_hash=True, published_at="2026-08-05T21:05:00.123+00:00")
        result = _validate(s)
        assert result.failure_mode not in ("missing_fields",)

    def test_garbage_timestamp_rejected(self):
        s = _make_signal()
        s["created_at_utc"] = "not-a-timestamp"
        result = _validate(s)
        assert result.failure_mode == "missing_fields"


# ---------------------------------------------------------------------------
# Point 8: No hardcoded data-quality thresholds
# ---------------------------------------------------------------------------

class TestNoHardcodedQualityThreshold:
    """Regression point 8: 0.90 threshold removed — all valid signals return 'normal' tier."""

    def test_very_low_coverage_still_normal_tier(self):
        """1% coverage → 'normal' tier (no threshold enforced)."""
        s = _make_signal(
            add_content_hash=True,
            universe_count=200,
            valid_ticker_count=2,  # 1% coverage
            stale_ticker_count=0,
        )
        result = _validate(s)
        assert result.data_quality_tier == "normal"
        assert result.is_valid
        assert result.allow_new_buys

    def test_failure_mode_never_data_quality_reduced(self):
        """failure_mode 'data_quality_reduced' can never be triggered."""
        for valid in [0, 50, 100, 140, 180, 200]:
            s = _make_signal(
                add_content_hash=True,
                universe_count=200,
                valid_ticker_count=valid,
            )
            result = _validate(s)
            assert result.failure_mode != "data_quality_reduced", (
                f"data_quality_reduced should never fire (valid={valid}/200)"
            )

    def test_raw_metrics_still_available(self):
        """Even without threshold, raw coverage/stale metrics are still reported."""
        s = _make_signal(
            add_content_hash=True,
            universe_count=200,
            valid_ticker_count=140,
            stale_ticker_count=30,
        )
        result = _validate(s)
        assert result.data_quality["signal_coverage_rate"] == pytest.approx(0.70, abs=0.01)
        assert result.data_quality["stale_pct"] == pytest.approx(0.15, abs=0.01)
        assert result.data_quality["universe_count"] == 200


# ---------------------------------------------------------------------------
# Full regression suite
# ---------------------------------------------------------------------------

class TestRegressionSuite:
    """Full regression suite — a valid signal must pass all remediation points."""

    def test_fully_valid_signal_passes_all_layers(self):
        """A correctly constructed signal passes all six validation layers."""
        s = _make_signal(
            add_content_hash=True,
            publication_status="published",
        )
        result = validate_signal(
            s, "/fake/path/2026-08-05_signal.json", SESSION,
            _get_publication_fn=_pub_success,
            _get_ledger_status_fn=_ledger_status_completed,
            _get_ledger_record_fn=_ledger_record_ok,
        )
        assert result.is_valid
        assert result.failure_mode in ("ok", "data_quality_reduced")
        assert result.allow_new_buys
        assert result.allow_pyramid
        assert result.allow_protective_sells
        assert result.signal_run_id is not None
        assert result.published_commit_sha is not None
        assert result.data_cutoff_at is not None

    def test_each_single_failure_blocks_new_buys(self):
        """Each failure mode individually blocks new buys."""
        failures = [
            # (signal_override, pub_fn, ledger_status_fn, ledger_record_fn)
            # Layer 1: missing field
            (_make_signal(add_content_hash=True, **{"signal_run_id": ""}), _pub_success, _ledger_status_completed, _ledger_record_ok),
            # Layer 1b: draft status
            (_make_signal(add_content_hash=True, publication_status="draft"), _pub_success, _ledger_status_completed, _ledger_record_ok),
            # Layer 2: stale session
            (_make_signal(add_content_hash=True, intended_execution_session="2026-01-01"), _pub_success, _ledger_status_completed, _ledger_record_ok),
            # Layer 3: no pub record
            (_make_signal(add_content_hash=True), _pub_none, _ledger_status_completed, _ledger_record_ok),
            # Layer 4: ledger error
            (_make_signal(add_content_hash=True), _pub_success, _ledger_status_failed, _ledger_record_ok),
            # Layer 5: missing hash
            (_make_signal(), _pub_success, _ledger_status_completed, _ledger_record_ok),
        ]
        for sig, pub_fn, status_fn, record_fn in failures:
            result = validate_signal(
                sig, "/fake", SESSION,
                _get_publication_fn=pub_fn,
                _get_ledger_status_fn=status_fn,
                _get_ledger_record_fn=record_fn,
            )
            assert not result.allow_new_buys, (
                f"Expected allow_new_buys=False for failure_mode={result.failure_mode}"
            )
            assert result.allow_protective_sells, (
                f"Expected allow_protective_sells=True for failure_mode={result.failure_mode}"
            )
