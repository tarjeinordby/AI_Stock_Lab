"""
V2B Shadow Observation Ledger — behavioral tests.

All tests operate in a temporary directory to avoid touching real ledger data.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from unittest import mock

import pytest

import modules.v2b_ledger as ledger
from modules.v2b_ledger import (
    ConflictError,
    CorruptionError,
    InvalidTransitionError,
    assert_order_creation_blocked,
    create_observation,
    get_observation_events,
    get_observation_status,
    list_observations,
    make_content_hash,
    make_observation_key,
    make_ticker_record,
    provenance_entry,
    transition_observation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def tmp_ledger(tmp_path, monkeypatch):
    """Redirect all ledger file I/O to a temporary directory."""
    monkeypatch.setattr(ledger, "LEDGER_DIR", tmp_path)
    yield tmp_path


def _ticker(ticker="AAPL", rank=1) -> dict:
    return make_ticker_record(
        ticker=ticker,
        momentum_score=75.0,
        quality_score=60.0,
        value_score=50.0,
        safety_score=80.0,
        composite_score=70.0,
        factor_coverage=0.75,
        rank=rank,
        excluded=False,
        exclusion_reason=None,
        sector="Technology",
        value_sector_adjusted=True,
        provenance=[
            provenance_entry("current_snapshot", "yfinance", note="quality factor"),
            provenance_entry("point_in_time", "price_history", as_of_date="2026-08-11"),
        ],
    )


def _create(session="2026-08-11", model_version="quant_baseline_v2", cfg_hash="abc123") -> dict:
    return create_observation(
        model_version=model_version,
        model_config_hash=cfg_hash,
        intended_execution_session=session,
        universe_count=500,
        signal_coverage=0.85,
        ticker_records=[_ticker()],
        metadata={"source": "test"},
    )


# ---------------------------------------------------------------------------
# T01 — observation_key is deterministic SHA-256
# ---------------------------------------------------------------------------


class TestObservationKey:
    def test_deterministic(self):
        k1 = make_observation_key("v2", "abc", "2026-08-11")
        k2 = make_observation_key("v2", "abc", "2026-08-11")
        assert k1 == k2

    def test_changes_with_model_version(self):
        k1 = make_observation_key("v2", "abc", "2026-08-11")
        k2 = make_observation_key("v3", "abc", "2026-08-11")
        assert k1 != k2

    def test_changes_with_session(self):
        k1 = make_observation_key("v2", "abc", "2026-08-11")
        k2 = make_observation_key("v2", "abc", "2026-08-12")
        assert k1 != k2

    def test_hex_format_64_chars(self):
        k = make_observation_key("v2", "abc", "2026-08-11")
        assert len(k) == 64
        assert all(c in "0123456789abcdef" for c in k)


# ---------------------------------------------------------------------------
# T02 — create_observation returns OBSERVATION_CREATED event
# ---------------------------------------------------------------------------


class TestCreateObservation:
    def test_returns_created_event(self):
        ev = _create()
        assert ev["event_type"] == "OBSERVATION_CREATED"

    def test_order_creation_blocked_always_true(self):
        ev = _create()
        assert ev["order_creation_blocked"] is True

    def test_point_in_time_fundamentals_global_false(self):
        ev = _create()
        assert ev["point_in_time_fundamentals_global"] is False

    def test_run_id_is_uuid4(self):
        ev = _create()
        val = uuid.UUID(ev["observation_run_id"])
        assert val.version == 4

    def test_content_hash_present(self):
        ev = _create()
        assert len(ev["content_hash"]) == 64

    def test_writes_jsonl_file(self, tmp_ledger):
        _create()
        files = list(tmp_ledger.glob("*_v2b_observations.jsonl"))
        assert len(files) == 1
        lines = [l for l in files[0].read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["event_type"] == "OBSERVATION_CREATED"

    def test_index_updated(self, tmp_ledger):
        ev = _create()
        idx = json.loads((tmp_ledger / "v2b_idx.json").read_text())
        assert ev["observation_key"] in idx


# ---------------------------------------------------------------------------
# T03 — idempotency (same key + same content)
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_same_content_returns_idempotent_match(self):
        _create()
        ev2 = _create()
        assert ev2["event_type"] == "IDEMPOTENT_MATCH"

    def test_idempotent_match_does_not_write_new_line(self, tmp_ledger):
        _create()
        _create()
        files = list(tmp_ledger.glob("*_v2b_observations.jsonl"))
        lines = [l for l in files[0].read_text().splitlines() if l.strip()]
        assert len(lines) == 1

    def test_idempotent_returns_prior_run_id(self):
        ev1 = _create()
        ev2 = _create()
        assert ev2["observation_run_id"] == ev1["observation_run_id"]


# ---------------------------------------------------------------------------
# T04 — conflict (same key, different content)
# ---------------------------------------------------------------------------


def _create_conflict_pair(session="2026-08-11"):
    """Create two observations with the same key but different content."""
    common = dict(
        model_version="quant_baseline_v2",
        model_config_hash="fixed_hash",
        intended_execution_session=session,
        ticker_records=[_ticker()],
        metadata={},
    )
    ev1 = create_observation(universe_count=500, signal_coverage=0.85, **common)
    # Same key, different universe_count → different content_hash → CONFLICT
    with pytest.raises(ConflictError):
        create_observation(universe_count=501, signal_coverage=0.85, **common)
    return ev1


class TestConflict:
    def test_raises_conflict_error(self):
        _create_conflict_pair()

    def test_conflict_event_written_to_ledger(self, tmp_ledger):
        ev = _create_conflict_pair()
        files = list(tmp_ledger.glob("*_v2b_observations.jsonl"))
        lines = [l for l in files[0].read_text().splitlines() if l.strip()]
        event_types = [json.loads(l)["event_type"] for l in lines]
        assert "CONFLICT" in event_types

    def test_conflict_status_is_conflict(self):
        ev = _create_conflict_pair()
        key = ev["observation_key"]
        assert get_observation_status(key) == "CONFLICT"


# ---------------------------------------------------------------------------
# T05 — status transitions
# ---------------------------------------------------------------------------


class TestStatusTransitions:
    def test_initial_status_is_created(self):
        ev = _create()
        assert get_observation_status(ev["observation_key"]) == "CREATED"

    def test_created_to_collecting(self):
        ev = _create()
        key = ev["observation_key"]
        transition_observation(key, "COLLECTING")
        assert get_observation_status(key) == "COLLECTING"

    def test_collecting_to_completed(self):
        ev = _create()
        key = ev["observation_key"]
        transition_observation(key, "COLLECTING")
        transition_observation(key, "COMPLETED")
        assert get_observation_status(key) == "COMPLETED"

    def test_collecting_to_failed_data(self):
        ev = _create()
        key = ev["observation_key"]
        transition_observation(key, "COLLECTING")
        transition_observation(key, "FAILED_DATA")
        assert get_observation_status(key) == "FAILED_DATA"

    def test_collecting_to_failed_validation(self):
        ev = _create()
        key = ev["observation_key"]
        transition_observation(key, "COLLECTING")
        transition_observation(key, "FAILED_VALIDATION")
        assert get_observation_status(key) == "FAILED_VALIDATION"

    def test_collecting_to_cancelled(self):
        ev = _create()
        key = ev["observation_key"]
        transition_observation(key, "COLLECTING")
        transition_observation(key, "CANCELLED")
        assert get_observation_status(key) == "CANCELLED"

    def test_illegal_transition_raises(self):
        ev = _create()
        key = ev["observation_key"]
        transition_observation(key, "COLLECTING")
        transition_observation(key, "COMPLETED")
        with pytest.raises(InvalidTransitionError):
            transition_observation(key, "COLLECTING")

    def test_terminal_state_no_transition(self):
        ev = _create()
        key = ev["observation_key"]
        transition_observation(key, "COLLECTING")
        transition_observation(key, "CANCELLED")
        with pytest.raises(InvalidTransitionError):
            transition_observation(key, "COLLECTING")

    def test_direct_created_to_completed_illegal(self):
        ev = _create()
        key = ev["observation_key"]
        with pytest.raises(InvalidTransitionError):
            transition_observation(key, "COMPLETED")


# ---------------------------------------------------------------------------
# T06 — get_observation_events
# ---------------------------------------------------------------------------


class TestGetEvents:
    def test_returns_all_events_in_order(self):
        ev = _create()
        key = ev["observation_key"]
        transition_observation(key, "COLLECTING")
        transition_observation(key, "COMPLETED")
        events = get_observation_events(key)
        assert [e["event_type"] for e in events] == [
            "OBSERVATION_CREATED",
            "COLLECTING",
            "COMPLETED",
        ]

    def test_unknown_key_returns_empty(self):
        assert get_observation_events("0" * 64) == []

    def test_all_events_have_order_creation_blocked_true(self):
        ev = _create()
        key = ev["observation_key"]
        transition_observation(key, "COLLECTING")
        transition_observation(key, "COMPLETED")
        for event in get_observation_events(key):
            assert event["order_creation_blocked"] is True


# ---------------------------------------------------------------------------
# T07 — list_observations
# ---------------------------------------------------------------------------


class TestListObservations:
    def test_lists_single_observation(self):
        _create()
        obs = list_observations()
        assert len(obs) == 1

    def test_multiple_observations(self):
        _create(session="2026-08-11")
        _create(session="2026-08-12", model_version="quant_baseline_v2", cfg_hash="other")
        obs = list_observations()
        assert len(obs) == 2

    def test_summary_has_required_fields(self):
        _create()
        obs = list_observations()
        required = {
            "observation_key", "status", "model_version",
            "intended_execution_session", "content_hash", "created_at",
        }
        assert required.issubset(obs[0].keys())


# ---------------------------------------------------------------------------
# T08 — corrupt ledger handling
# ---------------------------------------------------------------------------


class TestCorruptLedger:
    def test_incomplete_last_line_warns_and_skips(self, tmp_ledger):
        ev = _create()
        key = ev["observation_key"]
        yyyymm = key  # unused
        # Find the file and append a broken last line
        files = list(tmp_ledger.glob("*_v2b_observations.jsonl"))
        assert files
        with open(files[0], "a") as f:
            f.write("{broken json")
        with pytest.warns(UserWarning):
            events = get_observation_events(key)
        # The valid first line should still be returned
        assert len(events) >= 1

    def test_mid_file_corruption_raises_corruption_error(self, tmp_ledger):
        ev1 = _create(session="2026-08-11", cfg_hash="h1")
        key = ev1["observation_key"]
        transition_observation(key, "COLLECTING")
        files = list(tmp_ledger.glob("*_v2b_observations.jsonl"))
        content = files[0].read_text()
        lines = content.splitlines()
        # corrupt the first line (mid-file), append a valid second line
        corrupted = "{bad json}\n" + "\n".join(lines[1:]) + "\n"
        files[0].write_text(corrupted)
        with pytest.raises(CorruptionError):
            get_observation_events(key)


# ---------------------------------------------------------------------------
# T09 — structural invariant
# ---------------------------------------------------------------------------


class TestStructuralInvariant:
    def test_assert_order_creation_blocked_passes(self):
        assert_order_creation_blocked()

    def test_module_constant_is_true(self):
        assert ledger._ORDER_CREATION_BLOCKED is True


# ---------------------------------------------------------------------------
# T10 — content hash changes with payload
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_hash_changes_with_different_universe_count(self):
        payload_a = {"universe_count": 100}
        payload_b = {"universe_count": 200}
        assert make_content_hash(payload_a) != make_content_hash(payload_b)

    def test_hash_is_deterministic(self):
        payload = {"a": 1, "b": [1, 2, 3]}
        assert make_content_hash(payload) == make_content_hash(payload)


# ---------------------------------------------------------------------------
# T11 — ticker record structure
# ---------------------------------------------------------------------------


class TestTickerRecord:
    def test_ticker_record_has_required_keys(self):
        r = _ticker()
        assert "ticker" in r
        assert "scores" in r
        assert "factor_coverage" in r
        assert "excluded" in r
        assert "provenance" in r
        assert "value_sector_adjusted" in r

    def test_provenance_types_valid(self):
        r = _ticker()
        valid = {"point_in_time", "current_snapshot", "unavailable", "unknown"}
        for p in r["provenance"]:
            assert p["type"] in valid

    def test_excluded_ticker_has_reason(self):
        r = make_ticker_record(
            ticker="BAD",
            momentum_score=None,
            quality_score=None,
            value_score=None,
            safety_score=None,
            composite_score=None,
            factor_coverage=None,
            rank=None,
            excluded=True,
            exclusion_reason="below_sma200",
            sector="Energy",
            value_sector_adjusted=False,
            provenance=[provenance_entry("unavailable", "none")],
        )
        assert r["excluded"] is True
        assert r["exclusion_reason"] == "below_sma200"


# ---------------------------------------------------------------------------
# T12 — session date maps to correct YYYY-MM partition
# ---------------------------------------------------------------------------


class TestSessionPartitioning:
    def test_august_session_creates_august_file(self, tmp_ledger):
        _create(session="2026-08-11")
        files = list(tmp_ledger.glob("2026-08_v2b_observations.jsonl"))
        assert len(files) == 1

    def test_different_months_create_different_files(self, tmp_ledger):
        _create(session="2026-08-11", cfg_hash="a")
        _create(session="2026-09-01", cfg_hash="b")
        aug = list(tmp_ledger.glob("2026-08_v2b_observations.jsonl"))
        sep = list(tmp_ledger.glob("2026-09_v2b_observations.jsonl"))
        assert len(aug) == 1
        assert len(sep) == 1


# ---------------------------------------------------------------------------
# T13 — get_observation_status returns None for unknown key
# ---------------------------------------------------------------------------


class TestGetStatus:
    def test_unknown_key_returns_none(self):
        assert get_observation_status("0" * 64) is None
