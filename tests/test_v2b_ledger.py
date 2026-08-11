"""
V2B Shadow Observation Ledger — behavioral tests.

All tests operate in a temporary directory. The module-level LEDGER_DIR is
monkeypatched per test via the `tmp_ledger` fixture.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

import modules.v2b_ledger as ledger
from modules.v2b_ledger import (
    ConflictError,
    CorruptionError,
    InvalidTransitionError,
    ValidationError,
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
# Test fixtures and helpers
# ---------------------------------------------------------------------------

VALID_CFG_HASH = "a" * 64
VALID_SESSION = "2026-08-11"
VALID_SESSION_ALT = "2026-08-12"
VALID_MODEL = "quant_baseline_v2"

WEEKEND = "2026-08-09"       # Sunday
HOLIDAY = "2026-12-25"       # Christmas
INVALID_DATE = "2026-13-01"  # month 13
DST_SUNDAY = "2026-03-08"    # DST transition Sunday — not a session
DST_MONDAY = "2026-03-09"    # Monday after DST — valid session


@pytest.fixture(autouse=True)
def tmp_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_DIR", tmp_path)
    yield tmp_path


def _prov() -> list[dict]:
    return [
        provenance_entry(
            "point_in_time", "price_history",
            as_of_date=VALID_SESSION,
            source_timestamp="2026-08-11T20:00:00+00:00",
            data_cutoff_at="2026-08-10",
        ),
        provenance_entry(
            "current_snapshot", "yfinance",
            note="quality factor via .info",
        ),
    ]


def _ticker(ticker="AAPL", rank=1, excluded=False, exclusion_reason=None) -> dict:
    return make_ticker_record(
        ticker,
        raw_factor_inputs={
            "momentum_12_1": 0.25,
            "momentum_6_1": 0.18,
            "roe": None,
        },
        raw_factor_unavailable_reasons={"roe": "no_data"},
        factor_availability={"momentum": True, "quality": False, "value": True, "safety": True},
        scores={"momentum": 75.0, "quality": None, "value": 50.0, "safety": 80.0, "composite": 70.0},
        factor_coverage=0.75,
        rank=rank if not excluded else None,
        excluded=excluded,
        exclusion_reason=exclusion_reason if excluded else None,
        selected_by_strategy=[] if excluded else ["Core_Quality_Momentum_V2"],
        sector="Technology",
        value_sector_adjusted=True,
        provenance=_prov(),
    )


def _create(
    session=VALID_SESSION,
    model_version=VALID_MODEL,
    cfg_hash=VALID_CFG_HASH,
    universe_count=500,
    valid_ticker_count=480,
    stale_ticker_count=5,
    excluded_ticker_count=10,
    missing_ticker_count=5,
    signal_coverage_rate=0.85,
    selected_tickers_per_strategy=None,
    data_quality_status="ok",
    ticker_records=None,
) -> dict:
    return create_observation(
        model_version=model_version,
        model_config_hash=cfg_hash,
        intended_execution_session=session,
        universe_count=universe_count,
        valid_ticker_count=valid_ticker_count,
        stale_ticker_count=stale_ticker_count,
        excluded_ticker_count=excluded_ticker_count,
        missing_ticker_count=missing_ticker_count,
        signal_coverage_rate=signal_coverage_rate,
        selected_tickers_per_strategy=selected_tickers_per_strategy or {"Core_Quality_Momentum_V2": ["AAPL"]},
        data_quality_status=data_quality_status,
        ticker_records=ticker_records or [_ticker()],
        metadata={"source": "test"},
    )


# ---------------------------------------------------------------------------
# T01 — observation_key determinism
# ---------------------------------------------------------------------------


class TestObservationKey:
    def test_deterministic(self):
        k1 = make_observation_key(VALID_MODEL, VALID_CFG_HASH, VALID_SESSION)
        k2 = make_observation_key(VALID_MODEL, VALID_CFG_HASH, VALID_SESSION)
        assert k1 == k2

    def test_changes_with_model_version(self):
        k1 = make_observation_key(VALID_MODEL, VALID_CFG_HASH, VALID_SESSION)
        k2 = make_observation_key("other_v2", VALID_CFG_HASH, VALID_SESSION)
        assert k1 != k2

    def test_changes_with_session(self):
        k1 = make_observation_key(VALID_MODEL, VALID_CFG_HASH, VALID_SESSION)
        k2 = make_observation_key(VALID_MODEL, VALID_CFG_HASH, VALID_SESSION_ALT)
        assert k1 != k2

    def test_hex_64_chars(self):
        k = make_observation_key(VALID_MODEL, VALID_CFG_HASH, VALID_SESSION)
        assert len(k) == 64
        assert all(c in "0123456789abcdef" for c in k)


# ---------------------------------------------------------------------------
# T02 — create_observation basics
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
        import uuid
        ev = _create()
        val = uuid.UUID(ev["observation_run_id"])
        assert val.version == 4

    def test_content_hash_present_64_hex(self):
        ev = _create()
        h = ev["content_hash"]
        assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)

    def test_event_hash_present_64_hex(self):
        ev = _create()
        h = ev.get("event_hash")
        assert h and len(h) == 64

    def test_previous_event_hash_is_none_for_first_event(self):
        ev = _create()
        assert ev.get("previous_event_hash") is None

    def test_writes_jsonl_file(self, tmp_ledger):
        _create()
        files = list(tmp_ledger.glob("*_v2b_observations.jsonl"))
        assert len(files) == 1
        lines = [l for l in files[0].read_text().splitlines() if l.strip()]
        assert len(lines) == 1

    def test_index_updated(self, tmp_ledger):
        ev = _create()
        idx = json.loads((tmp_ledger / "v2b_idx.json").read_text())
        assert ev["observation_key"] in idx

    def test_run_level_fields_present(self):
        ev = _create()
        for field in (
            "universe_count", "valid_ticker_count", "stale_ticker_count",
            "excluded_ticker_count", "missing_ticker_count",
            "signal_coverage_rate", "selected_tickers_per_strategy",
            "data_quality_status",
        ):
            assert field in ev, f"Missing run-level field: {field}"


# ---------------------------------------------------------------------------
# T03 — idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_same_content_returns_idempotent_match(self):
        _create()
        ev2 = _create()
        assert ev2["event_type"] == "IDEMPOTENT_MATCH"

    def test_idempotent_does_not_write_new_line(self, tmp_ledger):
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
# T04 — conflict
# ---------------------------------------------------------------------------


def _conflict_pair(session=VALID_SESSION):
    """First call succeeds; second has same key but different payload → ConflictError."""
    common = dict(
        model_version=VALID_MODEL,
        model_config_hash=VALID_CFG_HASH,
        intended_execution_session=session,
        valid_ticker_count=480, stale_ticker_count=5,
        excluded_ticker_count=10, missing_ticker_count=5,
        selected_tickers_per_strategy={"Core_Quality_Momentum_V2": ["AAPL"]},
        data_quality_status="ok",
        ticker_records=[_ticker()],
    )
    ev1 = create_observation(universe_count=500, signal_coverage_rate=0.85, **common)
    with pytest.raises(ConflictError):
        create_observation(universe_count=501, signal_coverage_rate=0.85, **common)
    return ev1


class TestConflict:
    def test_raises_conflict_error(self):
        _conflict_pair()

    def test_conflict_event_written_to_ledger(self, tmp_ledger):
        _conflict_pair()
        files = list(tmp_ledger.glob("*_v2b_observations.jsonl"))
        lines = [l for l in files[0].read_text().splitlines() if l.strip()]
        types = [json.loads(l)["event_type"] for l in lines]
        assert "CONFLICT" in types

    def test_conflict_status_is_conflict(self):
        ev = _conflict_pair()
        assert get_observation_status(ev["observation_key"]) == "CONFLICT"

    def test_conflict_has_hash_chain(self, tmp_ledger):
        ev = _conflict_pair()
        events = get_observation_events(ev["observation_key"])
        created_hash = events[0]["event_hash"]
        conflict_ev = events[1]
        assert conflict_ev["previous_event_hash"] == created_hash


# ---------------------------------------------------------------------------
# T05 — status transitions
# ---------------------------------------------------------------------------


class TestStatusTransitions:
    def test_initial_status_is_created(self):
        ev = _create()
        assert get_observation_status(ev["observation_key"]) == "CREATED"

    def test_created_to_collecting(self):
        ev = _create()
        transition_observation(ev["observation_key"], "COLLECTING")
        assert get_observation_status(ev["observation_key"]) == "COLLECTING"

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

    def test_completed_is_terminal(self):
        ev = _create()
        key = ev["observation_key"]
        transition_observation(key, "COLLECTING")
        transition_observation(key, "COMPLETED")
        with pytest.raises(InvalidTransitionError):
            transition_observation(key, "COLLECTING")

    def test_cancelled_is_terminal(self):
        ev = _create()
        key = ev["observation_key"]
        transition_observation(key, "COLLECTING")
        transition_observation(key, "CANCELLED")
        with pytest.raises(InvalidTransitionError):
            transition_observation(key, "COLLECTING")

    def test_direct_created_to_completed_illegal(self):
        ev = _create()
        with pytest.raises(InvalidTransitionError):
            transition_observation(ev["observation_key"], "COMPLETED")

    def test_transition_event_has_hash_chain(self):
        ev = _create()
        key = ev["observation_key"]
        transition_observation(key, "COLLECTING")
        transition_observation(key, "COMPLETED")
        events = get_observation_events(key)
        assert events[1]["previous_event_hash"] == events[0]["event_hash"]
        assert events[2]["previous_event_hash"] == events[1]["event_hash"]


# ---------------------------------------------------------------------------
# T06 — event retrieval
# ---------------------------------------------------------------------------


class TestGetEvents:
    def test_returns_all_events_in_order(self):
        ev = _create()
        key = ev["observation_key"]
        transition_observation(key, "COLLECTING")
        transition_observation(key, "COMPLETED")
        types = [e["event_type"] for e in get_observation_events(key)]
        assert types == ["OBSERVATION_CREATED", "COLLECTING", "COMPLETED"]

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

    def test_summary_has_required_fields(self):
        _create()
        obs = list_observations()[0]
        required = {
            "observation_key", "status", "model_version",
            "intended_execution_session", "content_hash", "created_at",
        }
        assert required.issubset(obs.keys())


# ---------------------------------------------------------------------------
# T08 — corrupt ledger handling
# ---------------------------------------------------------------------------


class TestCorruptLedger:
    def test_incomplete_last_line_warns_and_skips(self, tmp_ledger):
        ev = _create()
        key = ev["observation_key"]
        files = list(tmp_ledger.glob("*_v2b_observations.jsonl"))
        with open(files[0], "a") as f:
            f.write("{broken json")
        with pytest.warns(UserWarning):
            events = get_observation_events(key)
        assert len(events) >= 1

    def test_mid_file_corruption_raises_corruption_error(self, tmp_ledger):
        ev = _create()
        key = ev["observation_key"]
        transition_observation(key, "COLLECTING")
        files = list(tmp_ledger.glob("*_v2b_observations.jsonl"))
        lines = files[0].read_text().splitlines()
        # Corrupt first line (mid-file), keep the second line valid
        corrupted = "{bad json}\n" + "\n".join(lines[1:]) + "\n"
        files[0].write_text(corrupted)
        with pytest.raises(CorruptionError):
            get_observation_events(key)

    def test_manipulated_creation_event_raises_corruption(self, tmp_ledger):
        """Valid JSON with modified universe_count but stale event_hash → CorruptionError."""
        _create(universe_count=100)
        files = list(tmp_ledger.glob("*_v2b_observations.jsonl"))
        raw_line = files[0].read_text().strip()
        ev = json.loads(raw_line)
        # Tamper with universe_count but leave event_hash unchanged
        ev["universe_count"] = 999
        files[0].write_text(json.dumps(ev) + "\n")
        with pytest.raises(CorruptionError):
            list_observations()

    def test_manipulated_transition_event_raises_corruption(self, tmp_ledger):
        """Tampered transition event (wrong event_hash) → CorruptionError."""
        ev = _create()
        key = ev["observation_key"]
        transition_observation(key, "COLLECTING")
        files = list(tmp_ledger.glob("*_v2b_observations.jsonl"))
        lines = files[0].read_text().splitlines()
        # Parse and tamper with the second line (COLLECTING event)
        transition_ev = json.loads(lines[1])
        transition_ev["note"] = "tampered"  # change content, leave event_hash stale
        files[0].write_text(lines[0] + "\n" + json.dumps(transition_ev) + "\n")
        with pytest.raises(CorruptionError):
            get_observation_events(key)

    def test_broken_hash_chain_raises_corruption(self, tmp_ledger):
        """Wrong previous_event_hash on second event → CorruptionError."""
        ev = _create()
        key = ev["observation_key"]
        transition_observation(key, "COLLECTING")
        files = list(tmp_ledger.glob("*_v2b_observations.jsonl"))
        lines = files[0].read_text().splitlines()
        # Rebuild second event with wrong previous_event_hash
        ev2 = json.loads(lines[1])
        ev2_body = {k: v for k, v in ev2.items() if k != "event_hash"}
        ev2_body["previous_event_hash"] = "0" * 64  # wrong
        import hashlib, json as _json
        new_hash = hashlib.sha256(
            _json.dumps(ev2_body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        ev2_body["event_hash"] = new_hash
        files[0].write_text(lines[0] + "\n" + json.dumps(ev2_body) + "\n")
        with pytest.raises(CorruptionError):
            get_observation_events(key)


# ---------------------------------------------------------------------------
# T09 — fail-closed index
# ---------------------------------------------------------------------------


class TestIndexFailClosed:
    def test_corrupt_index_rebuilds_from_jsonl(self, tmp_ledger):
        ev = _create()
        key = ev["observation_key"]
        # Corrupt the index
        (tmp_ledger / "v2b_idx.json").write_text("{not valid json")
        # Should still find the observation by rebuilding
        status = get_observation_status(key)
        assert status == "CREATED"

    def test_missing_index_rebuilds_from_jsonl(self, tmp_ledger):
        ev = _create()
        key = ev["observation_key"]
        (tmp_ledger / "v2b_idx.json").unlink()
        status = get_observation_status(key)
        assert status == "CREATED"

    def test_concurrent_index_updates_different_months(self, tmp_ledger):
        """Two observations in different months must both appear in the index."""
        ev1 = _create(session="2026-08-11", cfg_hash="a" * 64)
        ev2 = _create(session="2026-09-01", cfg_hash="b" * 64)
        idx = json.loads((tmp_ledger / "v2b_idx.json").read_text())
        assert ev1["observation_key"] in idx
        assert ev2["observation_key"] in idx


# ---------------------------------------------------------------------------
# T10 — structural invariant
# ---------------------------------------------------------------------------


class TestStructuralInvariant:
    def test_assert_order_creation_blocked_passes(self):
        assert_order_creation_blocked()

    def test_module_constant_is_true(self):
        assert ledger._ORDER_CREATION_BLOCKED is True


# ---------------------------------------------------------------------------
# T11 — content hash properties
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_hash_changes_with_payload(self):
        h1 = make_content_hash({"universe_count": 100})
        h2 = make_content_hash({"universe_count": 200})
        assert h1 != h2

    def test_hash_is_deterministic(self):
        p = {"a": 1, "b": [1, 2, 3]}
        assert make_content_hash(p) == make_content_hash(p)

    def test_hash_rejects_nan(self):
        with pytest.raises((ValueError, TypeError)):
            make_content_hash({"score": float("nan")})

    def test_hash_rejects_inf(self):
        with pytest.raises((ValueError, TypeError)):
            make_content_hash({"score": float("inf")})


# ---------------------------------------------------------------------------
# T12 — ticker record schema
# ---------------------------------------------------------------------------


class TestTickerRecord:
    def test_ticker_record_has_all_required_fields(self):
        r = _ticker()
        for field in ("ticker", "raw_factor_inputs", "raw_factor_unavailable_reasons",
                      "factor_availability", "scores", "factor_coverage",
                      "rank", "excluded", "exclusion_reason",
                      "selected_by_strategy", "sector", "value_sector_adjusted",
                      "provenance"):
            assert field in r, f"Missing field: {field}"

    def test_provenance_has_field_level_details(self):
        r = _ticker()
        prov = r["provenance"][0]
        assert "provenance_type" in prov
        assert "source" in prov
        assert "as_of_date" in prov

    def test_excluded_ticker_requires_exclusion_reason(self):
        with pytest.raises(ValidationError):
            make_ticker_record(
                "BAD", excluded=True, exclusion_reason=None,
                provenance=[provenance_entry("unavailable", "none")],
            )

    def test_non_excluded_ticker_must_have_no_reason(self):
        with pytest.raises(ValidationError):
            make_ticker_record(
                "AAPL", excluded=False, exclusion_reason="some_reason",
                provenance=[provenance_entry("point_in_time", "price_history")],
            )

    def test_raw_factor_none_requires_reason(self):
        with pytest.raises(ValidationError):
            make_ticker_record(
                "AAPL",
                raw_factor_inputs={"roe": None},
                raw_factor_unavailable_reasons={},  # missing reason for roe
                provenance=[provenance_entry("point_in_time", "price_history")],
            )

    def test_raw_factor_nan_rejected(self):
        with pytest.raises(ValidationError):
            make_ticker_record(
                "AAPL",
                raw_factor_inputs={"momentum_12_1": float("nan")},
                provenance=[provenance_entry("point_in_time", "price_history")],
            )

    def test_duplicate_ticker_in_records_rejected(self):
        with pytest.raises(ValidationError):
            _create(ticker_records=[_ticker("AAPL"), _ticker("AAPL")])

    def test_selected_by_strategy_is_list(self):
        r = _ticker()
        assert isinstance(r["selected_by_strategy"], list)

    def test_selected_tickers_per_strategy_in_event(self):
        ev = _create(
            selected_tickers_per_strategy={
                "Core_Quality_Momentum_V2": ["AAPL"],
                "Aggressive_Momentum_V2": [],
            }
        )
        sps = ev["selected_tickers_per_strategy"]
        assert "Core_Quality_Momentum_V2" in sps
        assert "Aggressive_Momentum_V2" in sps


# ---------------------------------------------------------------------------
# T13 — strict input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_invalid_date_format_rejected(self):
        with pytest.raises(ValidationError):
            _create(session="not-a-date")

    def test_weekend_rejected(self):
        with pytest.raises(ValidationError):
            _create(session=WEEKEND)

    def test_holiday_rejected(self):
        with pytest.raises(ValidationError):
            _create(session=HOLIDAY)

    def test_invalid_month_rejected(self):
        with pytest.raises(ValidationError):
            _create(session=INVALID_DATE)

    def test_dst_sunday_rejected(self):
        with pytest.raises(ValidationError):
            _create(session=DST_SUNDAY)

    def test_dst_monday_valid(self):
        ev = _create(session=DST_MONDAY)
        assert ev["event_type"] == "OBSERVATION_CREATED"

    def test_empty_model_version_rejected(self):
        with pytest.raises(ValidationError):
            create_observation(
                model_version="",
                model_config_hash=VALID_CFG_HASH,
                intended_execution_session=VALID_SESSION,
                universe_count=100,
                valid_ticker_count=100,
                stale_ticker_count=0,
                excluded_ticker_count=0,
                missing_ticker_count=0,
                signal_coverage_rate=0.9,
                selected_tickers_per_strategy={},
                data_quality_status="ok",
                ticker_records=[],
            )

    def test_short_config_hash_rejected(self):
        with pytest.raises(ValidationError):
            _create(cfg_hash="abc123")  # Not 64 chars

    def test_empty_config_hash_rejected(self):
        with pytest.raises(ValidationError):
            _create(cfg_hash="")

    def test_non_hex_config_hash_rejected(self):
        with pytest.raises(ValidationError):
            _create(cfg_hash="g" * 64)  # 'g' is not hex

    def test_signal_coverage_above_1_rejected(self):
        with pytest.raises(ValidationError):
            _create(signal_coverage_rate=7.0)

    def test_signal_coverage_negative_rejected(self):
        with pytest.raises(ValidationError):
            _create(signal_coverage_rate=-0.1)

    def test_signal_coverage_nan_rejected(self):
        with pytest.raises(ValidationError):
            _create(signal_coverage_rate=float("nan"))

    def test_signal_coverage_inf_rejected(self):
        with pytest.raises(ValidationError):
            _create(signal_coverage_rate=float("inf"))

    def test_negative_universe_count_rejected(self):
        with pytest.raises(ValidationError):
            _create(universe_count=-1)

    def test_float_universe_count_rejected(self):
        with pytest.raises((ValidationError, TypeError)):
            _create(universe_count=1.5)

    def test_nan_score_in_ticker_rejected(self):
        with pytest.raises(ValidationError):
            tickers = [make_ticker_record(
                "AAPL",
                scores={"momentum": float("nan"), "quality": None,
                        "value": None, "safety": None, "composite": None},
                provenance=[provenance_entry("point_in_time", "price_history")],
            )]
            _create(ticker_records=tickers)


# ---------------------------------------------------------------------------
# T14 — timestamp UTC awareness
# ---------------------------------------------------------------------------


class TestTimestampUtc:
    def test_timestamps_are_utc_aware(self):
        ev = _create()
        ts = pd.Timestamp(ev["timestamp_utc"])
        assert ts.tz is not None

    def test_utc_offset_is_zero(self):
        ev = _create()
        ts = pd.Timestamp(ev["timestamp_utc"])
        offset = ts.utcoffset()
        assert offset is not None
        assert offset.total_seconds() == 0.0

    def test_dst_session_date_accepted(self):
        """A valid post-DST session date should be accepted and stored correctly."""
        ev = _create(session=DST_MONDAY)
        assert ev["intended_execution_session"] == DST_MONDAY


# ---------------------------------------------------------------------------
# T15 — session partitioning
# ---------------------------------------------------------------------------


class TestSessionPartitioning:
    def test_august_session_creates_august_file(self, tmp_ledger):
        _create(session="2026-08-11")
        files = list(tmp_ledger.glob("2026-08_v2b_observations.jsonl"))
        assert len(files) == 1

    def test_different_months_create_different_files(self, tmp_ledger):
        _create(session="2026-08-11", cfg_hash="a" * 64)
        _create(session="2026-09-01", cfg_hash="b" * 64)
        assert len(list(tmp_ledger.glob("2026-08_v2b_observations.jsonl"))) == 1
        assert len(list(tmp_ledger.glob("2026-09_v2b_observations.jsonl"))) == 1


# ---------------------------------------------------------------------------
# T16 — unknown key returns None
# ---------------------------------------------------------------------------


class TestGetStatus:
    def test_unknown_key_returns_none(self):
        assert get_observation_status("0" * 64) is None


# ---------------------------------------------------------------------------
# T17 — concurrent identical create (threads)
# ---------------------------------------------------------------------------


class TestConcurrentCreateThreads:
    def test_identical_concurrent_create_single_event_written(self, tmp_ledger):
        """N threads creating identical observation → exactly 1 OBSERVATION_CREATED line."""
        results = []
        errors = []

        def worker():
            try:
                results.append(_create())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Unexpected errors: {errors}"
        files = list(tmp_ledger.glob("*_v2b_observations.jsonl"))
        assert len(files) == 1
        lines = [l for l in files[0].read_text().splitlines() if l.strip()]
        event_types = [json.loads(l)["event_type"] for l in lines]
        assert event_types.count("OBSERVATION_CREATED") == 1

    def test_identical_concurrent_create_same_run_id(self, tmp_ledger):
        """All concurrent identical creates return the same observation_run_id."""
        run_ids = []
        errors = []

        def worker():
            try:
                ev = _create()
                run_ids.append(ev["observation_run_id"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(set(run_ids)) == 1, f"Multiple run_ids: {set(run_ids)}"


# ---------------------------------------------------------------------------
# T18 — concurrent divergent create (threads)
# ---------------------------------------------------------------------------


class TestConcurrentDivergentCreateThreads:
    def test_divergent_concurrent_create_one_created_one_conflict(self, tmp_ledger):
        """Two threads: same key, different payload → 1 CREATED + 1 CONFLICT in file."""
        common = dict(
            model_version=VALID_MODEL,
            model_config_hash=VALID_CFG_HASH,
            intended_execution_session=VALID_SESSION,
            valid_ticker_count=480, stale_ticker_count=5,
            excluded_ticker_count=10, missing_ticker_count=5,
            selected_tickers_per_strategy={"Core_Quality_Momentum_V2": ["AAPL"]},
            data_quality_status="ok",
            ticker_records=[_ticker()],
        )
        successes = []
        conflicts = []
        barrier = threading.Barrier(2)

        def worker_a():
            barrier.wait()
            try:
                ev = create_observation(universe_count=500, signal_coverage_rate=0.85, **common)
                successes.append(ev)
            except ConflictError:
                conflicts.append("conflict_a")

        def worker_b():
            barrier.wait()
            try:
                ev = create_observation(universe_count=501, signal_coverage_rate=0.85, **common)
                successes.append(ev)
            except ConflictError:
                conflicts.append("conflict_b")

        t1 = threading.Thread(target=worker_a)
        t2 = threading.Thread(target=worker_b)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert len(successes) == 1
        assert len(conflicts) == 1

        files = list(tmp_ledger.glob("*_v2b_observations.jsonl"))
        lines = [l for l in files[0].read_text().splitlines() if l.strip()]
        types = [json.loads(l)["event_type"] for l in lines]
        assert types.count("OBSERVATION_CREATED") == 1
        assert types.count("CONFLICT") == 1


# ---------------------------------------------------------------------------
# T19 — concurrent terminal transitions (threads)
# ---------------------------------------------------------------------------


class TestConcurrentTransitionThreads:
    def test_concurrent_terminal_transitions_only_one_wins(self, tmp_ledger):
        """Two threads try COLLECTING→COMPLETED and COLLECTING→FAILED_DATA simultaneously.
        Exactly one must succeed; the other must get InvalidTransitionError."""
        ev = _create()
        key = ev["observation_key"]
        transition_observation(key, "COLLECTING")

        successes = []
        failures = []
        barrier = threading.Barrier(2)

        def transition_completed():
            barrier.wait()
            try:
                transition_observation(key, "COMPLETED")
                successes.append("COMPLETED")
            except InvalidTransitionError:
                failures.append("COMPLETED_blocked")

        def transition_failed_data():
            barrier.wait()
            try:
                transition_observation(key, "FAILED_DATA")
                successes.append("FAILED_DATA")
            except InvalidTransitionError:
                failures.append("FAILED_DATA_blocked")

        t1 = threading.Thread(target=transition_completed)
        t2 = threading.Thread(target=transition_failed_data)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert len(successes) == 1
        assert len(failures) == 1

        # Observation is in exactly one terminal state
        status = get_observation_status(key)
        assert status in {"COMPLETED", "FAILED_DATA"}

        # File has exactly one terminal-state event
        events = get_observation_events(key)
        terminal = [e for e in events if e["event_type"] in {"COMPLETED", "FAILED_DATA"}]
        assert len(terminal) == 1


# ---------------------------------------------------------------------------
# T20 — concurrent creates, different months (threads)
# ---------------------------------------------------------------------------


class TestConcurrentDifferentMonths:
    def test_different_month_creates_both_land_in_index(self, tmp_ledger):
        """Creates for different months update the shared index without losing entries."""
        ev1_holder = []
        ev2_holder = []
        barrier = threading.Barrier(2)

        def create_aug():
            barrier.wait()
            ev1_holder.append(_create(session="2026-08-11", cfg_hash="a" * 64))

        def create_sep():
            barrier.wait()
            ev2_holder.append(_create(session="2026-09-01", cfg_hash="b" * 64))

        t1 = threading.Thread(target=create_aug)
        t2 = threading.Thread(target=create_sep)
        t1.start(); t2.start()
        t1.join(); t2.join()

        idx = json.loads((tmp_ledger / "v2b_idx.json").read_text())
        assert ev1_holder[0]["observation_key"] in idx
        assert ev2_holder[0]["observation_key"] in idx


# ---------------------------------------------------------------------------
# T21 — multiprocessing concurrent identical create
# ---------------------------------------------------------------------------


def _mp_worker_create(args: tuple) -> str:
    """Worker function for multiprocessing test."""
    ledger_dir_str, session, cfg_hash = args
    import modules.v2b_ledger as _ledger
    from pathlib import Path
    _ledger.LEDGER_DIR = Path(ledger_dir_str)
    ev = _ledger.create_observation(
        model_version=VALID_MODEL,
        model_config_hash=cfg_hash,
        intended_execution_session=session,
        universe_count=500,
        valid_ticker_count=480,
        stale_ticker_count=5,
        excluded_ticker_count=10,
        missing_ticker_count=5,
        signal_coverage_rate=0.85,
        selected_tickers_per_strategy={"Core_Quality_Momentum_V2": ["AAPL"]},
        data_quality_status="ok",
        ticker_records=[_ledger.make_ticker_record(
            "AAPL",
            scores={"composite": 70.0},
            provenance=[_ledger.provenance_entry("point_in_time", "price_history", as_of_date=session)],
        )],
    )
    return ev["event_type"]


class TestMultiprocessingCreate:
    def test_processes_identical_create_exactly_one_written(self, tmp_ledger):
        """Four separate processes creating the same observation → 1 CREATED, rest IDEMPOTENT."""
        ctx = multiprocessing.get_context("spawn")
        args = [(str(tmp_ledger), VALID_SESSION, VALID_CFG_HASH)] * 4
        with ctx.Pool(4) as pool:
            event_types = pool.map(_mp_worker_create, args)

        assert event_types.count("OBSERVATION_CREATED") == 1
        assert event_types.count("IDEMPOTENT_MATCH") == 3

        files = list(tmp_ledger.glob("*_v2b_observations.jsonl"))
        assert len(files) == 1
        lines = [l for l in files[0].read_text().splitlines() if l.strip()]
        file_types = [json.loads(l)["event_type"] for l in lines]
        assert file_types.count("OBSERVATION_CREATED") == 1


# ---------------------------------------------------------------------------
# T22 — no V1 portfolio changes
# ---------------------------------------------------------------------------


class TestNoV1SideEffects:
    def test_create_does_not_call_execute_buy(self):
        with mock.patch("modules.portfolio.execute_buy") as m:
            _create()
            m.assert_not_called()

    def test_create_does_not_call_execute_sell(self):
        with mock.patch("modules.portfolio.execute_sell") as m:
            _create()
            m.assert_not_called()

    def test_create_does_not_call_execute_pyramid_fill(self):
        with mock.patch("modules.portfolio.execute_pyramid_fill") as m:
            _create()
            m.assert_not_called()

    def test_v2b_does_not_import_v1_execution_modules(self):
        import sys
        import importlib
        import importlib.util
        v1_forbidden = {
            "modules.portfolio", "modules.orders", "modules.fills", "modules.ledger",
        }
        before = set(sys.modules.keys()) & v1_forbidden
        # Force re-evaluation of create_observation (it already runs, so new imports detected)
        _create()
        after = set(sys.modules.keys()) & v1_forbidden
        new_imports = after - before
        # The v2b_ledger module itself must not import any forbidden modules
        v2b_source = __import__("inspect").getsource(ledger)
        for mod in v1_forbidden:
            assert f"import {mod}" not in v2b_source, \
                f"v2b_ledger imports forbidden V1 module: {mod!r}"

    def test_no_orders_or_fills_files_written(self, tmp_ledger):
        """V1 orders/fills directories must not be created or written to."""
        v1_paths = [
            tmp_ledger.parent / "data_v4" / "orders",
            tmp_ledger.parent / "data_v4" / "fills",
        ]
        _create()
        for p in v1_paths:
            files = list(p.glob("**/*")) if p.exists() else []
            assert not files, f"V1 path was written: {p}"


# ---------------------------------------------------------------------------
# T23 — Issue 1: Crash window — index gap recovery
# ---------------------------------------------------------------------------


class TestCrashWindowRecovery:
    """
    Reproduces Issue 1: crash between JSONL append and index update leaves the
    key absent from the index. All lookup paths must fall back to JSONL scan
    and repair the index entry.
    """

    def _remove_from_idx(self, tmp_ledger: Path, key: str) -> None:
        idx_path = tmp_ledger / "v2b_idx.json"
        idx = json.loads(idx_path.read_text())
        del idx[key]
        idx_path.write_text(json.dumps(idx))

    def test_get_status_finds_via_jsonl_scan_on_index_miss(self, tmp_ledger):
        """get_observation_status must return CREATED even when key is absent from index."""
        ev = _create()
        key = ev["observation_key"]
        self._remove_from_idx(tmp_ledger, key)
        # Before fix: returned None. After fix: scans JSONL and returns "CREATED".
        status = get_observation_status(key)
        assert status == "CREATED"

    def test_get_status_repairs_index_after_jsonl_scan(self, tmp_ledger):
        """After a crash-recovery lookup, the index entry must be restored."""
        ev = _create()
        key = ev["observation_key"]
        self._remove_from_idx(tmp_ledger, key)
        get_observation_status(key)  # Triggers repair
        idx = json.loads((tmp_ledger / "v2b_idx.json").read_text())
        assert key in idx, "Index was not repaired after crash-recovery lookup"

    def test_get_events_finds_via_jsonl_scan_on_index_miss(self, tmp_ledger):
        """get_observation_events must return events via JSONL scan when key absent from index."""
        ev = _create()
        key = ev["observation_key"]
        self._remove_from_idx(tmp_ledger, key)
        events = get_observation_events(key)
        assert len(events) == 1
        assert events[0]["event_type"] == "OBSERVATION_CREATED"

    def test_idempotent_retry_repairs_index(self, tmp_ledger):
        """An idempotent create must repair the index even when the key was missing from it."""
        ev = _create()
        key = ev["observation_key"]
        self._remove_from_idx(tmp_ledger, key)
        # Before fix: IDEMPOTENT_MATCH returned but index not repaired.
        # After fix: IDEMPOTENT_MATCH AND key back in index.
        ev2 = _create()
        assert ev2["event_type"] == "IDEMPOTENT_MATCH"
        idx = json.loads((tmp_ledger / "v2b_idx.json").read_text())
        assert key in idx, "Idempotent retry did not repair the index"

    def test_transition_finds_via_jsonl_scan_on_index_miss(self, tmp_ledger):
        """transition_observation must find the observation via JSONL scan when index is stale."""
        ev = _create()
        key = ev["observation_key"]
        self._remove_from_idx(tmp_ledger, key)
        transition_observation(key, "COLLECTING")
        assert get_observation_status(key) == "COLLECTING"


# ---------------------------------------------------------------------------
# T24 — Issue 2: Corrupt index must not erase existing mappings
# ---------------------------------------------------------------------------


class TestCorruptIndexPreservesExistingMappings:
    """
    Reproduces Issue 2: _update_idx_locked used _load_idx_raw() which returned {}
    on corruption. A new create then saved only its own key, making all prior
    observations invisible.
    """

    def test_corrupt_index_create_preserves_existing_key(self, tmp_ledger):
        """Creating B after index corruption must not erase A's mapping."""
        ev_a = _create(session=VALID_SESSION, cfg_hash="a" * 64)
        key_a = ev_a["observation_key"]

        # Corrupt the index — simulates a partial write or fs corruption
        (tmp_ledger / "v2b_idx.json").write_text("{not valid json")

        # Creating B must rebuild from JSONL (preserving A) then add B
        ev_b = _create(session=VALID_SESSION_ALT, cfg_hash="b" * 64)
        key_b = ev_b["observation_key"]

        idx = json.loads((tmp_ledger / "v2b_idx.json").read_text())
        assert key_a in idx, "Key A was erased when corrupt index was replaced"
        assert key_b in idx, "Key B not in index after recovery"

    def test_corrupt_index_get_status_then_create_preserves_all(self, tmp_ledger):
        """After a get_observation_status repair, a subsequent create still sees both keys."""
        ev_a = _create(session=VALID_SESSION, cfg_hash="a" * 64)
        key_a = ev_a["observation_key"]
        (tmp_ledger / "v2b_idx.json").write_text("{not valid json")

        # Repair via lookup
        get_observation_status(key_a)

        # Now create a new observation — index must already be repaired with A
        ev_b = _create(session=VALID_SESSION_ALT, cfg_hash="b" * 64)
        key_b = ev_b["observation_key"]

        idx = json.loads((tmp_ledger / "v2b_idx.json").read_text())
        assert key_a in idx
        assert key_b in idx


# ---------------------------------------------------------------------------
# T25 — Issue 3: Hash fields are mandatory for record_version="2"
# ---------------------------------------------------------------------------


class TestMandatoryHashFields:
    """
    Reproduces Issue 3: event_hash and previous_event_hash were only checked
    if present. For record_version="2" both fields are mandatory.
    Missing, wrong-type, or wrong-length values must raise CorruptionError.
    Unsupported or missing record_version is fail-closed.
    """

    def _tamper(self, tmp_ledger: Path, modifier) -> None:
        """Read JSONL first line, apply modifier, write back WITHOUT recomputing event_hash."""
        files = list(tmp_ledger.glob("*_v2b_observations.jsonl"))
        raw = files[0].read_text().strip()
        ev = json.loads(raw)
        modifier(ev)
        files[0].write_text(json.dumps(ev) + "\n")

    def test_missing_event_hash_raises_corruption(self, tmp_ledger):
        """event_hash field absent → CorruptionError (was silently skipped before fix)."""
        ev = _create()
        self._tamper(tmp_ledger, lambda e: e.pop("event_hash"))
        with pytest.raises(CorruptionError, match="event_hash"):
            get_observation_events(ev["observation_key"])

    def test_missing_previous_event_hash_raises_corruption(self, tmp_ledger):
        """previous_event_hash field absent → CorruptionError (was silently skipped before fix)."""
        ev = _create()
        self._tamper(tmp_ledger, lambda e: e.pop("previous_event_hash"))
        with pytest.raises(CorruptionError, match="previous_event_hash"):
            get_observation_events(ev["observation_key"])

    def test_missing_record_version_raises_corruption(self, tmp_ledger):
        """Absent record_version is fail-closed: unsupported value raises CorruptionError."""
        ev = _create()
        self._tamper(tmp_ledger, lambda e: e.pop("record_version"))
        with pytest.raises(CorruptionError, match="record_version"):
            get_observation_events(ev["observation_key"])

    def test_unsupported_record_version_raises_corruption(self, tmp_ledger):
        """record_version='1' is not in supported set → fail-closed CorruptionError."""
        ev = _create()
        self._tamper(tmp_ledger, lambda e: e.update({"record_version": "1"}))
        with pytest.raises(CorruptionError, match="record_version"):
            get_observation_events(ev["observation_key"])

    def test_event_hash_wrong_length_raises_corruption(self, tmp_ledger):
        """event_hash with < 64 chars → CorruptionError."""
        ev = _create()
        self._tamper(tmp_ledger, lambda e: e.update({"event_hash": "abc"}))
        with pytest.raises(CorruptionError, match="event_hash"):
            get_observation_events(ev["observation_key"])

    def test_previous_event_hash_wrong_type_raises_corruption(self, tmp_ledger):
        """previous_event_hash as integer → CorruptionError."""
        ev = _create()
        self._tamper(tmp_ledger, lambda e: e.update({"previous_event_hash": 42}))
        with pytest.raises(CorruptionError, match="previous_event_hash"):
            get_observation_events(ev["observation_key"])


# ---------------------------------------------------------------------------
# T26 — Issue 4: Complete ticker schema must be validated
# ---------------------------------------------------------------------------


class TestTickerSchemaCompleteness:
    """
    Reproduces Issue 4: _validate_ticker_records used .get() for all fields
    so a dict with only {"ticker": "AAPL"} was accepted. All 13 documented
    fields must now be required and type-validated.
    """

    def test_minimal_ticker_dict_rejected(self):
        """{"ticker": "AAPL"} alone must be rejected — all required fields must be present."""
        with pytest.raises(ValidationError, match="missing required fields"):
            _create(ticker_records=[{"ticker": "AAPL"}])

    def test_missing_scores_field_rejected(self):
        """Ticker record missing the scores field entirely must be rejected."""
        rec = _ticker()
        del rec["scores"]
        with pytest.raises(ValidationError, match="missing required fields"):
            _create(ticker_records=[rec])

    def test_missing_score_key_rejected(self):
        """scores dict missing required key 'composite' must be rejected."""
        rec = _ticker()
        del rec["scores"]["composite"]
        with pytest.raises(ValidationError, match="composite"):
            _create(ticker_records=[rec])

    def test_bool_as_score_rejected(self):
        """scores value that is bool must be rejected (bool is a subclass of int)."""
        rec = _ticker()
        rec["scores"]["momentum"] = True
        with pytest.raises(ValidationError, match="bool"):
            _create(ticker_records=[rec])

    def test_bool_as_factor_coverage_rejected(self):
        """factor_coverage that is bool must be rejected."""
        rec = _ticker()
        rec["factor_coverage"] = True
        with pytest.raises(ValidationError, match="bool"):
            _create(ticker_records=[rec])

    def test_nan_as_score_rejected_in_validate_path(self):
        """NaN score injected past make_ticker_record must be caught by validate_ticker_records."""
        rec = _ticker()
        rec["scores"]["value"] = float("nan")
        with pytest.raises(ValidationError, match="NaN|finite"):
            _create(ticker_records=[rec])

    def test_rank_zero_rejected(self):
        """rank=0 must be rejected — rank must be a positive integer or None."""
        rec = _ticker()
        rec["rank"] = 0
        with pytest.raises(ValidationError, match="rank"):
            _create(ticker_records=[rec])

    def test_rank_negative_rejected(self):
        """rank=-1 must be rejected."""
        rec = _ticker()
        rec["rank"] = -1
        with pytest.raises(ValidationError, match="rank"):
            _create(ticker_records=[rec])

    def test_rank_bool_rejected(self):
        """rank=True must be rejected even though bool is a subclass of int."""
        rec = _ticker()
        rec["rank"] = True
        with pytest.raises(ValidationError, match="rank"):
            _create(ticker_records=[rec])

    def test_provenance_missing_type_rejected(self):
        """provenance entry without provenance_type must be rejected at create time."""
        rec = _ticker()
        rec["provenance"] = [{"source": "price_history"}]  # missing provenance_type
        with pytest.raises(ValidationError, match="provenance_type"):
            _create(ticker_records=[rec])

    def test_provenance_missing_source_rejected(self):
        """provenance entry without source must be rejected at create time."""
        rec = _ticker()
        rec["provenance"] = [{"provenance_type": "point_in_time"}]  # missing source
        with pytest.raises(ValidationError, match="source"):
            _create(ticker_records=[rec])

    def test_selected_tickers_cross_validation_rejects_unknown_ticker(self):
        """selected_tickers_per_strategy referencing a ticker not in records is rejected
        when ticker_records_represents_all_scored=True."""
        with pytest.raises(ValidationError, match="MSFT"):
            create_observation(
                model_version=VALID_MODEL,
                model_config_hash=VALID_CFG_HASH,
                intended_execution_session=VALID_SESSION,
                universe_count=500,
                valid_ticker_count=480,
                stale_ticker_count=5,
                excluded_ticker_count=10,
                missing_ticker_count=5,
                signal_coverage_rate=0.85,
                selected_tickers_per_strategy={"Core_Quality_Momentum_V2": ["MSFT"]},
                data_quality_status="ok",
                ticker_records=[_ticker("AAPL")],
                ticker_records_represents_all_scored=True,
            )

    def test_selected_tickers_cross_validation_skipped_when_subset(self):
        """ticker_records_represents_all_scored=False must bypass cross-validation."""
        ev = create_observation(
            model_version=VALID_MODEL,
            model_config_hash=VALID_CFG_HASH,
            intended_execution_session=VALID_SESSION,
            universe_count=500,
            valid_ticker_count=480,
            stale_ticker_count=5,
            excluded_ticker_count=10,
            missing_ticker_count=5,
            signal_coverage_rate=0.85,
            selected_tickers_per_strategy={"Core_Quality_Momentum_V2": ["MSFT"]},
            data_quality_status="ok",
            ticker_records=[_ticker("AAPL")],
            ticker_records_represents_all_scored=False,
        )
        assert ev["event_type"] == "OBSERVATION_CREATED"


# ---------------------------------------------------------------------------
# T27 — Issue 1 (Round 4): Wrong index mapping hides valid observation
# ---------------------------------------------------------------------------


class TestStaleIndexMappingRecovery:
    """
    Reproduces Issue 1 (Round 4): a valid YYYYMM value in the index that points
    to the wrong partition (or a non-existent one) was trusted without verification.
    The lookup returned None / [] even though the observation existed in JSONL.
    """

    def _set_idx_mapping(self, tmp_ledger: Path, key: str, yyyymm: str) -> None:
        idx_path = tmp_ledger / "v2b_idx.json"
        idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
        idx[key] = yyyymm
        idx_path.write_text(json.dumps(idx))

    def test_mapping_to_nonexistent_month_repaired(self, tmp_ledger):
        """Index maps key → "2099-01" (no such file) → must fall back and find real partition."""
        ev = _create()
        key = ev["observation_key"]
        self._set_idx_mapping(tmp_ledger, key, "2099-01")
        status = get_observation_status(key)
        assert status == "CREATED"

    def test_mapping_to_wrong_existing_month_repaired(self, tmp_ledger):
        """Index maps key → existing but wrong month → must fall back to JSONL scan."""
        ev_aug = _create(session=VALID_SESSION, cfg_hash="a" * 64)
        # Also create a file for a different month so it exists
        _create(session=VALID_SESSION_ALT, cfg_hash="b" * 64)
        key_aug = ev_aug["observation_key"]
        # Point key_aug at the wrong month (Sept file exists but doesn't have this key)
        self._set_idx_mapping(tmp_ledger, key_aug, "2026-09")
        status = get_observation_status(key_aug)
        assert status == "CREATED"

    def test_stale_mapping_index_repaired_after_lookup(self, tmp_ledger):
        """After stale-mapping recovery, index is updated to the correct partition."""
        ev = _create()
        key = ev["observation_key"]
        self._set_idx_mapping(tmp_ledger, key, "2099-01")
        get_observation_status(key)  # Triggers repair
        idx = json.loads((tmp_ledger / "v2b_idx.json").read_text())
        assert idx.get(key) == "2026-08", f"Index not repaired: {idx.get(key)!r}"

    def test_events_returned_after_stale_mapping_recovery(self, tmp_ledger):
        """get_observation_events must return correct events after stale-mapping repair."""
        ev = _create()
        key = ev["observation_key"]
        self._set_idx_mapping(tmp_ledger, key, "2099-01")
        events = get_observation_events(key)
        assert len(events) == 1
        assert events[0]["event_type"] == "OBSERVATION_CREATED"

    def test_transition_works_after_stale_mapping_recovery(self, tmp_ledger):
        """transition_observation must work after repairing a stale index mapping."""
        ev = _create()
        key = ev["observation_key"]
        self._set_idx_mapping(tmp_ledger, key, "2099-01")
        transition_observation(key, "COLLECTING")
        assert get_observation_status(key) == "COLLECTING"

    def test_key_in_multiple_partitions_raises_corruption(self, tmp_ledger):
        """If a key appears in more than one JSONL partition, CorruptionError is raised.

        The multi-partition check fires during JSONL scan (slow path). To reach the
        scan, the index is pointed at a non-existent month so the fast path fails.
        """
        ev = _create(session=VALID_SESSION, cfg_hash="a" * 64)
        key = ev["observation_key"]
        # Duplicate the event into a second month's JSONL
        src = list(tmp_ledger.glob("*_v2b_observations.jsonl"))[0]
        dup = tmp_ledger / "2026-09_v2b_observations.jsonl"
        dup.write_text(src.read_text())
        # Force the fast path to fail → triggers JSONL scan
        self._set_idx_mapping(tmp_ledger, key, "2099-01")
        # JSONL scan finds key in both "2026-08" and "2026-09" → CorruptionError
        with pytest.raises(CorruptionError, match="multiple partitions"):
            get_observation_status(key)

    def test_valid_mapping_not_disturbed(self, tmp_ledger):
        """A correct index mapping must be used as-is without triggering a scan."""
        ev = _create()
        key = ev["observation_key"]
        # Confirm the index is correct before and after lookup
        idx_before = json.loads((tmp_ledger / "v2b_idx.json").read_text())
        status = get_observation_status(key)
        idx_after = json.loads((tmp_ledger / "v2b_idx.json").read_text())
        assert status == "CREATED"
        assert idx_before == idx_after, "Valid mapping was unnecessarily modified"

    def test_invalid_format_in_index_value_triggers_scan(self, tmp_ledger):
        """Index value that is not a valid YYYYMM string must trigger JSONL scan."""
        ev = _create()
        key = ev["observation_key"]
        self._set_idx_mapping(tmp_ledger, key, "not-a-date")
        status = get_observation_status(key)
        assert status == "CREATED"


# ---------------------------------------------------------------------------
# T28 — Issue 2 (Round 4): ticker_records_represents_all_scored in content hash
# ---------------------------------------------------------------------------


class TestRepresentsAllScoredInContentHash:
    """
    Reproduces Issue 2 (Round 4): ticker_records_represents_all_scored was stored
    in the event but absent from the canonical payload, so True→False produced the
    same content_hash and was silently treated as IDEMPOTENT_MATCH.
    """

    def _common_kwargs(self):
        return dict(
            model_version=VALID_MODEL,
            model_config_hash=VALID_CFG_HASH,
            intended_execution_session=VALID_SESSION,
            universe_count=500,
            valid_ticker_count=480,
            stale_ticker_count=5,
            excluded_ticker_count=10,
            missing_ticker_count=5,
            signal_coverage_rate=0.85,
            selected_tickers_per_strategy={"Core_Quality_Momentum_V2": ["AAPL"]},
            data_quality_status="ok",
            ticker_records=[_ticker()],
            metadata={"source": "test"},
        )

    def test_true_then_false_raises_conflict(self):
        """True → False with same payload must write CONFLICT and raise ConflictError."""
        create_observation(**self._common_kwargs(), ticker_records_represents_all_scored=True)
        with pytest.raises(ConflictError):
            create_observation(**self._common_kwargs(), ticker_records_represents_all_scored=False)

    def test_false_then_true_raises_conflict(self):
        """False → True with same payload must write CONFLICT and raise ConflictError."""
        create_observation(**self._common_kwargs(), ticker_records_represents_all_scored=False)
        with pytest.raises(ConflictError):
            create_observation(**self._common_kwargs(), ticker_records_represents_all_scored=True)

    def test_true_then_true_is_idempotent(self):
        """True → True with identical payload must return IDEMPOTENT_MATCH."""
        create_observation(**self._common_kwargs(), ticker_records_represents_all_scored=True)
        ev2 = create_observation(**self._common_kwargs(), ticker_records_represents_all_scored=True)
        assert ev2["event_type"] == "IDEMPOTENT_MATCH"

    def test_false_then_false_is_idempotent(self):
        """False → False with identical payload must return IDEMPOTENT_MATCH."""
        create_observation(**self._common_kwargs(), ticker_records_represents_all_scored=False)
        ev2 = create_observation(**self._common_kwargs(), ticker_records_represents_all_scored=False)
        assert ev2["event_type"] == "IDEMPOTENT_MATCH"

    def test_content_hash_differs_between_true_and_false(self):
        """content_hash for True must differ from content_hash for False."""
        ev_true = create_observation(**self._common_kwargs(), ticker_records_represents_all_scored=True)
        # Use a different session to avoid conflict
        kw = self._common_kwargs()
        kw["intended_execution_session"] = VALID_SESSION_ALT
        ev_false = create_observation(**kw, ticker_records_represents_all_scored=False)
        assert ev_true["content_hash"] != ev_false["content_hash"]

    def test_event_stores_flag_value(self):
        """The OBSERVATION_CREATED event must store the correct flag value."""
        ev_true = create_observation(**self._common_kwargs(), ticker_records_represents_all_scored=True)
        assert ev_true["ticker_records_represents_all_scored"] is True
        kw = self._common_kwargs()
        kw["intended_execution_session"] = VALID_SESSION_ALT
        ev_false = create_observation(**kw, ticker_records_represents_all_scored=False)
        assert ev_false["ticker_records_represents_all_scored"] is False


# ---------------------------------------------------------------------------
# T29 — Round 5: fast-path duplicate detection (valid index + key in other partition)
# ---------------------------------------------------------------------------


class TestFastPathDuplicateDetection:
    """
    Reproduces the remaining Round 4 issue: _resolve_partition_for_key trusted
    the index fast path (key found in indexed partition) and returned immediately
    without checking other partitions. A valid index + duplicate JSONL entry in
    another month was silently accepted instead of raising CorruptionError.
    """

    def _duplicate_key_to(self, tmp_ledger: Path, src_yyyymm: str, dst_yyyymm: str) -> None:
        src = tmp_ledger / f"{src_yyyymm}_v2b_observations.jsonl"
        dst = tmp_ledger / f"{dst_yyyymm}_v2b_observations.jsonl"
        dst.write_text(src.read_text())

    def test_valid_index_plus_duplicate_raises_on_get_status(self, tmp_ledger):
        """Valid index → 2026-08 + duplicate in 2026-09 → get_observation_status raises."""
        ev = _create(session=VALID_SESSION, cfg_hash="a" * 64)
        key = ev["observation_key"]
        self._duplicate_key_to(tmp_ledger, "2026-08", "2026-09")
        # Verify the index is still correct (not tampered)
        idx = json.loads((tmp_ledger / "v2b_idx.json").read_text())
        assert idx.get(key) == "2026-08", "Precondition: index must point to 2026-08"
        # Fast path finds key in 2026-08, must then find it also in 2026-09 → CorruptionError
        with pytest.raises(CorruptionError, match="multiple partitions"):
            get_observation_status(key)

    def test_valid_index_plus_duplicate_raises_on_get_events(self, tmp_ledger):
        """Valid index + duplicate in other partition → get_observation_events raises."""
        ev = _create(session=VALID_SESSION, cfg_hash="a" * 64)
        key = ev["observation_key"]
        self._duplicate_key_to(tmp_ledger, "2026-08", "2026-09")
        with pytest.raises(CorruptionError, match="multiple partitions"):
            get_observation_events(key)

    def test_valid_index_plus_duplicate_raises_on_transition(self, tmp_ledger):
        """Valid index + duplicate → transition_observation raises before writing any event."""
        ev = _create(session=VALID_SESSION, cfg_hash="a" * 64)
        key = ev["observation_key"]
        self._duplicate_key_to(tmp_ledger, "2026-08", "2026-09")
        with pytest.raises(CorruptionError, match="multiple partitions"):
            transition_observation(key, "COLLECTING")
        # No transition written — both files still have exactly the original CREATED event
        for yyyymm in ("2026-08", "2026-09"):
            p = tmp_ledger / f"{yyyymm}_v2b_observations.jsonl"
            lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
            types = [json.loads(ln)["event_type"] for ln in lines]
            assert types == ["OBSERVATION_CREATED"], (
                f"{yyyymm}: expected only OBSERVATION_CREATED, got {types}"
            )

    def test_single_partition_valid_index_works_normally(self, tmp_ledger):
        """Control: one partition, correct index — fast path must succeed without error."""
        ev = _create()
        key = ev["observation_key"]
        assert get_observation_status(key) == "CREATED"
        assert len(get_observation_events(key)) == 1

    def test_duplicate_detected_when_other_partition_is_alphabetically_earlier(self, tmp_ledger):
        """Duplicate in a file sorted BEFORE the indexed partition must also be detected."""
        ev = _create(session="2026-09-01", cfg_hash="a" * 64)
        key = ev["observation_key"]
        # Duplicate from 2026-09 → 2026-08 (sorts earlier alphabetically)
        self._duplicate_key_to(tmp_ledger, "2026-09", "2026-08")
        idx = json.loads((tmp_ledger / "v2b_idx.json").read_text())
        assert idx.get(key) == "2026-09", "Precondition: index must point to 2026-09"
        with pytest.raises(CorruptionError, match="multiple partitions"):
            get_observation_status(key)

    def test_index_repair_cannot_hide_duplicate(self, tmp_ledger):
        """Repairing the index to one partition must not suppress the duplicate check."""
        ev = _create(session=VALID_SESSION, cfg_hash="a" * 64)
        key = ev["observation_key"]
        self._duplicate_key_to(tmp_ledger, "2026-08", "2026-09")
        # Manually point index to the duplicate partition (2026-09)
        idx_path = tmp_ledger / "v2b_idx.json"
        idx = json.loads(idx_path.read_text())
        idx[key] = "2026-09"
        idx_path.write_text(json.dumps(idx))
        # Fast path: finds key in 2026-09, scans 2026-08 (also has it) → CorruptionError
        with pytest.raises(CorruptionError, match="multiple partitions"):
            get_observation_status(key)


# ---------------------------------------------------------------------------
# T30 — Round 6: create_observation global uniqueness + _rebuild_idx fail-closed
# ---------------------------------------------------------------------------


class TestCreateObservationDuplicateDetection:
    """
    Reproduces the remaining Round 5 issue: create_observation trusted that only
    the expected partition matters and never checked other partitions for the same
    key. A duplicate JSONL entry silently produced IDEMPOTENT_MATCH or CONFLICT
    instead of CorruptionError.

    Also covers _rebuild_idx fail-closed: silent last-write-wins was replaced by
    an immediate CorruptionError when the same key appears in multiple partitions.
    """

    def _duplicate_key_to(self, tmp_ledger: Path, src_yyyymm: str, dst_yyyymm: str) -> None:
        src = tmp_ledger / f"{src_yyyymm}_v2b_observations.jsonl"
        dst = tmp_ledger / f"{dst_yyyymm}_v2b_observations.jsonl"
        dst.write_text(src.read_text())

    # ── create_observation ────────────────────────────────────────────────

    def test_valid_index_duplicate_identical_payload_raises_not_idempotent(self, tmp_ledger):
        """Same key in expected partition + another partition → CorruptionError, not IDEMPOTENT_MATCH."""
        ev = _create(session=VALID_SESSION, cfg_hash="a" * 64)
        key = ev["observation_key"]
        self._duplicate_key_to(tmp_ledger, "2026-08", "2026-09")
        # Before fix: returned IDEMPOTENT_MATCH silently
        with pytest.raises(CorruptionError, match="multiple partitions"):
            _create(session=VALID_SESSION, cfg_hash="a" * 64)

    def test_key_only_in_wrong_partition_create_raises_no_new_event(self, tmp_ledger):
        """Key exists only in a non-canonical partition → CorruptionError, no new event written."""
        ev = _create(session=VALID_SESSION, cfg_hash="a" * 64)
        key = ev["observation_key"]
        # Move: copy to wrong month, then delete from correct month
        self._duplicate_key_to(tmp_ledger, "2026-08", "2026-09")
        (tmp_ledger / "2026-08_v2b_observations.jsonl").unlink()
        # create targets "2026-08"; cross-partition check finds key in "2026-09" → CorruptionError
        with pytest.raises(CorruptionError, match="multiple partitions"):
            _create(session=VALID_SESSION, cfg_hash="a" * 64)
        # "2026-08" must not have been created
        assert not (tmp_ledger / "2026-08_v2b_observations.jsonl").exists()

    def test_duplicate_partition_different_payload_raises_not_conflict(self, tmp_ledger):
        """Duplicate + different universe_count → CorruptionError, not ConflictError."""
        ev = _create(session=VALID_SESSION, cfg_hash="a" * 64)
        self._duplicate_key_to(tmp_ledger, "2026-08", "2026-09")
        # Would normally trigger CONFLICT (different content_hash)
        with pytest.raises(CorruptionError, match="multiple partitions"):
            _create(session=VALID_SESSION, cfg_hash="a" * 64, universe_count=999)

    def test_duplicate_check_does_not_mutate_jsonl_or_index(self, tmp_ledger):
        """CorruptionError must leave JSONL files and the index completely unchanged."""
        ev = _create(session=VALID_SESSION, cfg_hash="a" * 64)
        self._duplicate_key_to(tmp_ledger, "2026-08", "2026-09")
        idx_before = (tmp_ledger / "v2b_idx.json").read_text()
        content_aug = (tmp_ledger / "2026-08_v2b_observations.jsonl").read_text()
        content_sep = (tmp_ledger / "2026-09_v2b_observations.jsonl").read_text()
        with pytest.raises(CorruptionError):
            _create(session=VALID_SESSION, cfg_hash="a" * 64)
        assert (tmp_ledger / "v2b_idx.json").read_text() == idx_before, "Index was modified"
        assert (tmp_ledger / "2026-08_v2b_observations.jsonl").read_text() == content_aug
        assert (tmp_ledger / "2026-09_v2b_observations.jsonl").read_text() == content_sep

    def test_no_conflict_event_written_when_duplicate_found(self, tmp_ledger):
        """A CONFLICT event must NOT be written when a cross-partition duplicate is detected."""
        _create(session=VALID_SESSION, cfg_hash="a" * 64)
        self._duplicate_key_to(tmp_ledger, "2026-08", "2026-09")
        with pytest.raises(CorruptionError):
            _create(session=VALID_SESSION, cfg_hash="a" * 64, universe_count=999)
        # Exactly one event (the original CREATED) in the canonical partition
        lines = [
            ln for ln in
            (tmp_ledger / "2026-08_v2b_observations.jsonl").read_text().splitlines()
            if ln.strip()
        ]
        assert len(lines) == 1
        assert json.loads(lines[0])["event_type"] == "OBSERVATION_CREATED"

    def test_idempotent_create_single_partition_still_works(self, tmp_ledger):
        """Control: idempotent create in one partition must still return IDEMPOTENT_MATCH."""
        _create()
        ev2 = _create()
        assert ev2["event_type"] == "IDEMPOTENT_MATCH"

    def test_conflict_single_partition_still_works(self, tmp_ledger):
        """Control: conflict with changed payload in one partition must still raise ConflictError."""
        _create()
        with pytest.raises(ConflictError):
            _create(universe_count=999)

    def test_concurrent_creates_still_produce_exactly_one_event(self, tmp_ledger):
        """Duplicate check must not break concurrent identical creates."""
        results = []
        errors = []

        def worker():
            try:
                results.append(_create())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Unexpected errors: {errors}"
        files = list(tmp_ledger.glob("*_v2b_observations.jsonl"))
        lines = [ln for ln in files[0].read_text().splitlines() if ln.strip()]
        types = [json.loads(ln)["event_type"] for ln in lines]
        assert types.count("OBSERVATION_CREATED") == 1

    # ── _rebuild_idx fail-closed ──────────────────────────────────────────

    def test_missing_index_with_duplicate_raises_on_rebuild(self, tmp_ledger):
        """Missing index + key in two partitions → _rebuild_idx raises CorruptionError."""
        _create(session=VALID_SESSION, cfg_hash="a" * 64)
        self._duplicate_key_to(tmp_ledger, "2026-08", "2026-09")
        (tmp_ledger / "v2b_idx.json").unlink()
        with pytest.raises(CorruptionError, match="multiple partitions"):
            get_observation_status("0" * 64)  # Triggers _load_idx_or_rebuild → _rebuild_idx

    def test_corrupt_index_with_duplicate_raises_on_rebuild(self, tmp_ledger):
        """Corrupt index + key in two partitions → _rebuild_idx raises CorruptionError."""
        _create(session=VALID_SESSION, cfg_hash="a" * 64)
        self._duplicate_key_to(tmp_ledger, "2026-08", "2026-09")
        (tmp_ledger / "v2b_idx.json").write_text("{not valid json")
        with pytest.raises(CorruptionError, match="multiple partitions"):
            get_observation_status("0" * 64)  # Triggers _load_idx_or_rebuild → _rebuild_idx

    def test_rebuild_idx_single_partition_produces_correct_index(self, tmp_ledger):
        """_rebuild_idx with no duplicates must return a correct single-entry mapping."""
        ev = _create(session=VALID_SESSION, cfg_hash="a" * 64)
        key = ev["observation_key"]
        # Test _rebuild_idx directly — it must find the key in the correct partition.
        # (get_observation_status uses an in-memory rebuild for the fast path and does
        # not guarantee writing the index file; that's verified via T09.)
        idx = ledger._rebuild_idx()
        assert idx.get(key) == "2026-08"
        assert len(idx) == 1
