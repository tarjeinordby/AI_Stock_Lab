"""
V2B.4 Signal Outcome Ledger — behavioral tests.

All tests operate in a temporary directory.  The module-level OUTCOME_DIR is
monkeypatched per test via the `tmp_outcome` fixture so no real data_v4 files
are created or read.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest import mock

import pytest

import modules.v2b_outcome as outcome
from modules.v2b_outcome import (
    GRACE_SESSIONS,
    HOLDING_SESSIONS_TUPLE,
    OUTCOME_DEFINITION_VERSION,
    OUTCOME_TRACKING_START_SESSION,
    STRATEGY_ID,
    ContentConflictError,
    CorruptionError,
    InvalidTransitionError,
    ObservationValidationError,
    _canonical_json,
    _compute_event_hash,
    compute_exit_session,
    create_pending,
    get_outcome_events,
    get_outcome_status,
    list_pending_outcomes,
    make_outcome_key,
    make_pending_content_hash,
    make_terminal_content_hash,
    record_fetch_deferred,
    record_terminal,
    validate_and_rebuild_index,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def tmp_outcome(tmp_path, monkeypatch):
    """Redirect OUTCOME_DIR to a fresh temp directory for every test."""
    monkeypatch.setattr(outcome, "OUTCOME_DIR", tmp_path)
    yield tmp_path


# ── Helper builders ───────────────────────────────────────────────────────────


def _make_pending(
    obs_key="obs_key_001",
    holding_sessions=5,
    entry_session="2026-09-08",
    exit_session="2026-09-14",
    tickers=None,
):
    if tickers is None:
        tickers = ["AAPL", "MSFT"]
    return create_pending(
        observation_key=obs_key,
        strategy_id=STRATEGY_ID,
        holding_sessions=holding_sessions,
        outcome_definition_version=OUTCOME_DEFINITION_VERSION,
        entry_session=entry_session,
        exit_session=exit_session,
        selected_tickers=tickers,
    )


def _ok(obs_key="obs_key_001", n=5):
    return make_outcome_key(obs_key, STRATEGY_ID, n, OUTCOME_DEFINITION_VERSION)


def _minimal_terminal_payload(outcome_key, terminal_type="OUTCOME_RECORDED", tickers=None):
    if tickers is None:
        tickers = ["AAPL", "MSFT"]
    payload = {
        "event_type": terminal_type,
        "outcome_key": outcome_key,
        "observation_key": "obs_key_001",
        "strategy_id": STRATEGY_ID,
        "outcome_definition_version": OUTCOME_DEFINITION_VERSION,
        "holding_sessions": 5,
        "entry_session": "2026-09-08",
        "exit_session": "2026-09-14",
        "selected_tickers": sorted(tickers),
        "unavailable_tickers": [],
        "unavailable_reasons": {},
        "entry_prices": {"AAPL": 220.0, "MSFT": 430.0},
        "exit_prices": {"AAPL": 231.0, "MSFT": 440.0},
        "spy_entry_price": 550.0,
        "spy_exit_price": 557.75,
        "per_ticker_return_pct": {"AAPL": 5.0, "MSFT": 2.326},
        "spy_return_pct": 1.409,
        "portfolio_return_pct": 3.663,
        "portfolio_return_complete": True,
        "forward_alpha_vs_spy": 2.254,
        "hit_rate_vs_spy": 1.0,
        "available_subset_return_pct": 3.663,
        "adjustment_mode": "auto_adjust=True",
        "data_source": "yfinance",
        "fetch_date_range": "2026-09-08/2026-09-15",
        "provider_end_exclusive": "2026-09-15",
        "fetched_at": "2026-09-14T16:00:00+00:00",
        "measurement_date": "2026-09-14",
        "order_creation_blocked": True,
    }
    return payload


# ════════════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════════════


def test_tracking_start_session():
    assert OUTCOME_TRACKING_START_SESSION == "2026-09-08"


def test_grace_sessions():
    assert GRACE_SESSIONS == 5


def test_holding_sessions_tuple():
    assert HOLDING_SESSIONS_TUPLE == (1, 5, 21, 63)


def test_order_creation_blocked_invariant():
    from modules.v2b_outcome import _ORDER_CREATION_BLOCKED
    assert _ORDER_CREATION_BLOCKED is True


# ════════════════════════════════════════════════════════════════════════════════
# compute_exit_session
# ════════════════════════════════════════════════════════════════════════════════


def test_exit_session_n1_same_day():
    assert compute_exit_session("2026-09-08", 1) == "2026-09-08"


def test_exit_session_n5():
    with mock.patch("modules.exchange_calendar.nth_session_after", return_value="2026-09-14") as m:
        result = compute_exit_session("2026-09-08", 5)
    m.assert_called_once_with("2026-09-08", 4)
    assert result == "2026-09-14"


def test_exit_session_n21():
    with mock.patch("modules.exchange_calendar.nth_session_after", return_value="2026-10-06") as m:
        result = compute_exit_session("2026-09-08", 21)
    m.assert_called_once_with("2026-09-08", 20)
    assert result == "2026-10-06"


def test_exit_session_n63():
    with mock.patch("modules.exchange_calendar.nth_session_after", return_value="2026-12-04") as m:
        result = compute_exit_session("2026-09-08", 63)
    m.assert_called_once_with("2026-09-08", 62)
    assert result == "2026-12-04"


def test_exit_session_invalid_n():
    with pytest.raises(ValueError, match="holding_sessions must be one of"):
        compute_exit_session("2026-09-08", 7)


# ════════════════════════════════════════════════════════════════════════════════
# make_outcome_key
# ════════════════════════════════════════════════════════════════════════════════


def test_outcome_key_deterministic():
    k1 = make_outcome_key("obs1", STRATEGY_ID, 5, OUTCOME_DEFINITION_VERSION)
    k2 = make_outcome_key("obs1", STRATEGY_ID, 5, OUTCOME_DEFINITION_VERSION)
    assert k1 == k2


def test_outcome_key_unique_by_holding_sessions():
    k5 = make_outcome_key("obs1", STRATEGY_ID, 5, OUTCOME_DEFINITION_VERSION)
    k21 = make_outcome_key("obs1", STRATEGY_ID, 21, OUTCOME_DEFINITION_VERSION)
    assert k5 != k21


def test_outcome_key_unique_by_obs_key():
    k1 = make_outcome_key("obs1", STRATEGY_ID, 5, OUTCOME_DEFINITION_VERSION)
    k2 = make_outcome_key("obs2", STRATEGY_ID, 5, OUTCOME_DEFINITION_VERSION)
    assert k1 != k2


def test_outcome_key_64_hex():
    k = make_outcome_key("obs1", STRATEGY_ID, 5, OUTCOME_DEFINITION_VERSION)
    assert len(k) == 64
    assert all(c in "0123456789abcdef" for c in k)


def test_outcome_key_includes_strategy_id():
    k1 = make_outcome_key("obs1", "Strategy_A", 5, OUTCOME_DEFINITION_VERSION)
    k2 = make_outcome_key("obs1", "Strategy_B", 5, OUTCOME_DEFINITION_VERSION)
    assert k1 != k2


def test_outcome_key_includes_definition_version():
    k1 = make_outcome_key("obs1", STRATEGY_ID, 5, "event_study_v1")
    k2 = make_outcome_key("obs1", STRATEGY_ID, 5, "event_study_v2")
    assert k1 != k2


# ════════════════════════════════════════════════════════════════════════════════
# make_pending_content_hash
# ════════════════════════════════════════════════════════════════════════════════


def test_pending_content_hash_deterministic():
    h1 = make_pending_content_hash(STRATEGY_ID, OUTCOME_DEFINITION_VERSION, 5, "2026-09-08", "2026-09-14", "2026-09", ["AAPL", "MSFT"])
    h2 = make_pending_content_hash(STRATEGY_ID, OUTCOME_DEFINITION_VERSION, 5, "2026-09-08", "2026-09-14", "2026-09", ["AAPL", "MSFT"])
    assert h1 == h2


def test_pending_content_hash_order_independent():
    h1 = make_pending_content_hash(STRATEGY_ID, OUTCOME_DEFINITION_VERSION, 5, "2026-09-08", "2026-09-14", "2026-09", ["AAPL", "MSFT"])
    h2 = make_pending_content_hash(STRATEGY_ID, OUTCOME_DEFINITION_VERSION, 5, "2026-09-08", "2026-09-14", "2026-09", ["MSFT", "AAPL"])
    assert h1 == h2


def test_pending_content_hash_changes_with_tickers():
    h1 = make_pending_content_hash(STRATEGY_ID, OUTCOME_DEFINITION_VERSION, 5, "2026-09-08", "2026-09-14", "2026-09", ["AAPL", "MSFT"])
    h2 = make_pending_content_hash(STRATEGY_ID, OUTCOME_DEFINITION_VERSION, 5, "2026-09-08", "2026-09-14", "2026-09", ["AAPL", "GOOG"])
    assert h1 != h2


# ════════════════════════════════════════════════════════════════════════════════
# make_terminal_content_hash
# ════════════════════════════════════════════════════════════════════════════════


def test_terminal_content_hash_excludes_timestamps():
    """fetched_at and measurement_date must not affect the hash."""
    ok = _ok()
    p1 = _minimal_terminal_payload(ok)
    p2 = dict(p1)
    p2["fetched_at"] = "2099-01-01T00:00:00+00:00"
    p2["measurement_date"] = "2099-01-01"
    assert make_terminal_content_hash(p1) == make_terminal_content_hash(p2)


def test_terminal_content_hash_excludes_chain_fields():
    ok = _ok()
    p1 = _minimal_terminal_payload(ok)
    p2 = dict(p1)
    p2["event_hash"] = "x" * 64
    p2["previous_event_hash"] = "y" * 64
    assert make_terminal_content_hash(p1) == make_terminal_content_hash(p2)


def test_terminal_content_hash_sensitive_to_returns():
    ok = _ok()
    p1 = _minimal_terminal_payload(ok)
    p2 = dict(p1)
    p2["portfolio_return_pct"] = 99.99
    assert make_terminal_content_hash(p1) != make_terminal_content_hash(p2)


def test_terminal_content_hash_ticker_order_independent():
    ok = _ok()
    p1 = _minimal_terminal_payload(ok, tickers=["AAPL", "MSFT"])
    p2 = dict(p1)
    p2["selected_tickers"] = ["MSFT", "AAPL"]
    assert make_terminal_content_hash(p1) == make_terminal_content_hash(p2)


# ════════════════════════════════════════════════════════════════════════════════
# create_pending
# ════════════════════════════════════════════════════════════════════════════════


def test_create_pending_writes_event():
    ev = _make_pending()
    assert isinstance(ev, dict)
    assert ev["event_type"] == "OUTCOME_PENDING"
    assert ev["order_creation_blocked"] is True
    assert ev["previous_event_hash"] is None
    assert len(ev["event_hash"]) == 64


def test_create_pending_event_hash_valid():
    ev = _make_pending()
    body = {k: v for k, v in ev.items() if k != "event_hash"}
    expected = _compute_event_hash(body)
    assert ev["event_hash"] == expected


def test_create_pending_persisted_to_disk(tmp_outcome):
    ev = _make_pending()
    exit_partition = ev["exit_partition_yyyymm"]
    path = tmp_outcome / f"{exit_partition}_v2b_outcomes.jsonl"
    assert path.exists()
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    assert lines[0]["event_type"] == "OUTCOME_PENDING"


def test_create_pending_index_updated(tmp_outcome):
    ev = _make_pending()
    ok = ev["outcome_key"]
    idx_path = tmp_outcome / "v2b_outcome_idx.json"
    idx = json.loads(idx_path.read_text())
    assert ok in idx
    assert idx[ok] == ev["exit_partition_yyyymm"]


def test_create_pending_idempotent():
    _make_pending()
    result = _make_pending()
    assert result == "IDEMPOTENT_MATCH"


def test_create_pending_content_conflict():
    _make_pending(tickers=["AAPL", "MSFT"])
    with pytest.raises(ContentConflictError):
        _make_pending(tickers=["AAPL", "GOOG"])  # different tickers = conflict


def test_create_pending_tickers_sorted_on_disk(tmp_outcome):
    ev = _make_pending(tickers=["MSFT", "AAPL"])
    assert ev["selected_tickers"] == ["AAPL", "MSFT"]


def test_create_pending_empty_tickers_valid():
    """Empty ticker list is valid — will become UNAVAILABLE later."""
    ev = create_pending(
        observation_key="obs_empty",
        strategy_id=STRATEGY_ID,
        holding_sessions=5,
        outcome_definition_version=OUTCOME_DEFINITION_VERSION,
        entry_session="2026-09-08",
        exit_session="2026-09-14",
        selected_tickers=[],
    )
    assert ev["event_type"] == "OUTCOME_PENDING"
    assert ev["selected_tickers"] == []


def test_create_pending_invalid_ticker_raises():
    with pytest.raises(ObservationValidationError):
        create_pending(
            observation_key="obs_bad",
            strategy_id=STRATEGY_ID,
            holding_sessions=5,
            outcome_definition_version=OUTCOME_DEFINITION_VERSION,
            entry_session="2026-09-08",
            exit_session="2026-09-14",
            selected_tickers=["AAPL", ""],  # empty string invalid
        )


def test_create_pending_duplicate_tickers_raises():
    with pytest.raises(ObservationValidationError):
        create_pending(
            observation_key="obs_dup",
            strategy_id=STRATEGY_ID,
            holding_sessions=5,
            outcome_definition_version=OUTCOME_DEFINITION_VERSION,
            entry_session="2026-09-08",
            exit_session="2026-09-14",
            selected_tickers=["AAPL", "AAPL"],
        )


def test_create_pending_non_string_ticker_raises():
    with pytest.raises(ObservationValidationError):
        create_pending(
            observation_key="obs_bad2",
            strategy_id=STRATEGY_ID,
            holding_sessions=5,
            outcome_definition_version=OUTCOME_DEFINITION_VERSION,
            entry_session="2026-09-08",
            exit_session="2026-09-14",
            selected_tickers=[123],  # type: ignore[list-item]
        )


def test_create_pending_multiple_holding_sessions_independent():
    ev1 = create_pending(
        observation_key="obs_multi",
        strategy_id=STRATEGY_ID,
        holding_sessions=1,
        outcome_definition_version=OUTCOME_DEFINITION_VERSION,
        entry_session="2026-09-08",
        exit_session="2026-09-08",
        selected_tickers=["AAPL"],
    )
    ev5 = create_pending(
        observation_key="obs_multi",
        strategy_id=STRATEGY_ID,
        holding_sessions=5,
        outcome_definition_version=OUTCOME_DEFINITION_VERSION,
        entry_session="2026-09-08",
        exit_session="2026-09-14",
        selected_tickers=["AAPL"],
    )
    assert ev1["outcome_key"] != ev5["outcome_key"]
    assert ev1["holding_sessions"] == 1
    assert ev5["holding_sessions"] == 5


# ════════════════════════════════════════════════════════════════════════════════
# record_fetch_deferred
# ════════════════════════════════════════════════════════════════════════════════


def test_record_fetch_deferred_basic():
    _make_pending()
    ok = _ok()
    missing = [{"symbol": "AAPL", "field": "adjusted_open", "session": "2026-09-08"}]
    ev = record_fetch_deferred(
        outcome_key=ok,
        attempted_at="2026-09-14T16:00:00+00:00",
        attempt_date="2026-09-14",
        missing_price_points=missing,
        source="yfinance",
        detail="missing price",
    )
    assert ev is not None
    assert ev["event_type"] == "OUTCOME_FETCH_DEFERRED"
    assert ev["order_creation_blocked"] is True
    assert ev["previous_event_hash"] is not None  # chains to PENDING


def test_record_fetch_deferred_dedup_same_day_same_missing():
    _make_pending()
    ok = _ok()
    missing = [{"symbol": "AAPL", "field": "adjusted_open", "session": "2026-09-08"}]
    ev1 = record_fetch_deferred(ok, "t1", "2026-09-14", missing, "yfinance", "x")
    ev2 = record_fetch_deferred(ok, "t2", "2026-09-14", missing, "yfinance", "x")
    assert ev1 is not None
    assert ev2 is None  # deduplicated


def test_record_fetch_deferred_not_dedup_different_date():
    _make_pending()
    ok = _ok()
    missing = [{"symbol": "AAPL", "field": "adjusted_open", "session": "2026-09-08"}]
    ev1 = record_fetch_deferred(ok, "t1", "2026-09-14", missing, "yfinance", "x")
    ev2 = record_fetch_deferred(ok, "t2", "2026-09-15", missing, "yfinance", "x")
    assert ev1 is not None
    assert ev2 is not None  # different date → not dedup


def test_record_fetch_deferred_not_dedup_different_missing():
    _make_pending()
    ok = _ok()
    m1 = [{"symbol": "AAPL", "field": "adjusted_open", "session": "2026-09-08"}]
    m2 = [{"symbol": "MSFT", "field": "adjusted_open", "session": "2026-09-08"}]
    ev1 = record_fetch_deferred(ok, "t1", "2026-09-14", m1, "yfinance", "x")
    ev2 = record_fetch_deferred(ok, "t2", "2026-09-14", m2, "yfinance", "x")
    assert ev1 is not None
    assert ev2 is not None


def test_record_fetch_deferred_not_found_raises():
    with pytest.raises(InvalidTransitionError, match="not found"):
        record_fetch_deferred("x" * 64, "t", "2026-09-14", [], "yfinance", "x")


def test_record_fetch_deferred_on_terminal_raises():
    _make_pending()
    ok = _ok()
    payload = _minimal_terminal_payload(ok)
    record_terminal(ok, payload)
    with pytest.raises(InvalidTransitionError, match="terminal state"):
        record_fetch_deferred(ok, "t", "2026-09-14", [], "yfinance", "x")


def test_record_fetch_deferred_hash_chain():
    _make_pending()
    ok = _ok()
    missing = [{"symbol": "AAPL", "field": "adjusted_open", "session": "2026-09-08"}]
    ev = record_fetch_deferred(ok, "t", "2026-09-14", missing, "yfinance", "x")
    assert ev is not None
    # Verify hash chain
    body = {k: v for k, v in ev.items() if k != "event_hash"}
    assert ev["event_hash"] == _compute_event_hash(body)


# ════════════════════════════════════════════════════════════════════════════════
# record_terminal
# ════════════════════════════════════════════════════════════════════════════════


def test_record_terminal_outcome_recorded():
    _make_pending()
    ok = _ok()
    payload = _minimal_terminal_payload(ok)
    ev = record_terminal(ok, payload)
    assert isinstance(ev, dict)
    assert ev["event_type"] == "OUTCOME_RECORDED"
    assert ev["order_creation_blocked"] is True
    assert "terminal_content_hash" in ev


def test_record_terminal_hash_chain():
    _make_pending()
    ok = _ok()
    payload = _minimal_terminal_payload(ok)
    ev = record_terminal(ok, payload)
    body = {k: v for k, v in ev.items() if k != "event_hash"}
    assert ev["event_hash"] == _compute_event_hash(body)
    assert ev["previous_event_hash"] is not None


def test_record_terminal_idempotent():
    _make_pending()
    ok = _ok()
    payload = _minimal_terminal_payload(ok)
    record_terminal(ok, payload)
    result = record_terminal(ok, payload)
    assert result == "IDEMPOTENT_MATCH"


def test_record_terminal_content_conflict():
    _make_pending()
    ok = _ok()
    p1 = _minimal_terminal_payload(ok)
    p2 = dict(p1)
    p2["portfolio_return_pct"] = 99.99
    record_terminal(ok, p1)
    with pytest.raises(ContentConflictError):
        record_terminal(ok, p2)


def test_record_terminal_wrong_type_raises():
    _make_pending()
    ok = _ok()
    p1 = _minimal_terminal_payload(ok, terminal_type="OUTCOME_RECORDED")
    p2 = _minimal_terminal_payload(ok, terminal_type="OUTCOME_INCOMPLETE")
    record_terminal(ok, p1)
    with pytest.raises(InvalidTransitionError):
        record_terminal(ok, p2)


def test_record_terminal_not_found_raises():
    with pytest.raises(InvalidTransitionError, match="not found"):
        payload = _minimal_terminal_payload("x" * 64)
        record_terminal("x" * 64, payload)


def test_record_terminal_non_terminal_event_type_raises():
    _make_pending()
    ok = _ok()
    p = _minimal_terminal_payload(ok)
    p["event_type"] = "OUTCOME_FETCH_DEFERRED"
    with pytest.raises(ValueError, match="not a terminal state"):
        record_terminal(ok, p)


def test_record_terminal_incomplete():
    _make_pending()
    ok = _ok()
    p = _minimal_terminal_payload(ok, terminal_type="OUTCOME_INCOMPLETE")
    p["portfolio_return_pct"] = None
    p["forward_alpha_vs_spy"] = None
    p["hit_rate_vs_spy"] = None
    p["portfolio_return_complete"] = False
    p["unavailable_tickers"] = ["MSFT"]
    p["unavailable_reasons"] = {"MSFT": "price_row_missing: adjusted_open@2026-09-08"}
    ev = record_terminal(ok, p)
    assert ev["event_type"] == "OUTCOME_INCOMPLETE"


def test_record_terminal_unavailable():
    _make_pending()
    ok = _ok()
    p = _minimal_terminal_payload(ok, terminal_type="OUTCOME_UNAVAILABLE")
    p["portfolio_return_pct"] = None
    p["forward_alpha_vs_spy"] = None
    p["hit_rate_vs_spy"] = None
    p["portfolio_return_complete"] = False
    p["entry_prices"] = {}
    p["exit_prices"] = {}
    p["spy_entry_price"] = None
    p["spy_exit_price"] = None
    p["spy_return_pct"] = None
    p["per_ticker_return_pct"] = {}
    p["available_subset_return_pct"] = None
    p["unavailable_tickers"] = ["AAPL", "MSFT"]
    ev = record_terminal(ok, p)
    assert ev["event_type"] == "OUTCOME_UNAVAILABLE"


def test_record_terminal_after_fetch_deferred():
    _make_pending()
    ok = _ok()
    missing = [{"symbol": "AAPL", "field": "adjusted_open", "session": "2026-09-08"}]
    record_fetch_deferred(ok, "t", "2026-09-14", missing, "yfinance", "x")
    # After grace, can still write terminal
    payload = _minimal_terminal_payload(ok, terminal_type="OUTCOME_INCOMPLETE")
    payload["portfolio_return_pct"] = None
    payload["forward_alpha_vs_spy"] = None
    payload["hit_rate_vs_spy"] = None
    payload["portfolio_return_complete"] = False
    payload["unavailable_tickers"] = ["AAPL"]
    ev = record_terminal(ok, payload)
    assert ev["event_type"] == "OUTCOME_INCOMPLETE"


# ════════════════════════════════════════════════════════════════════════════════
# get_outcome_events / get_outcome_status / list_pending_outcomes
# ════════════════════════════════════════════════════════════════════════════════


def test_get_outcome_events_empty():
    assert get_outcome_events("x" * 64) == []


def test_get_outcome_status_none():
    assert get_outcome_status("x" * 64) is None


def test_get_outcome_status_pending():
    _make_pending()
    ok = _ok()
    assert get_outcome_status(ok) == "OUTCOME_PENDING"


def test_get_outcome_status_recorded():
    _make_pending()
    ok = _ok()
    payload = _minimal_terminal_payload(ok)
    record_terminal(ok, payload)
    assert get_outcome_status(ok) == "OUTCOME_RECORDED"


def test_list_pending_outcomes_empty():
    assert list_pending_outcomes() == []


def test_list_pending_outcomes_single():
    _make_pending()
    pending = list_pending_outcomes()
    assert len(pending) == 1
    assert pending[0]["event_type"] == "OUTCOME_PENDING"


def test_list_pending_outcomes_excludes_terminal():
    _make_pending()
    ok = _ok()
    payload = _minimal_terminal_payload(ok)
    record_terminal(ok, payload)
    assert list_pending_outcomes() == []


def test_list_pending_outcomes_multiple_holding_sessions():
    create_pending(
        observation_key="obs_multi2",
        strategy_id=STRATEGY_ID,
        holding_sessions=1,
        outcome_definition_version=OUTCOME_DEFINITION_VERSION,
        entry_session="2026-09-08",
        exit_session="2026-09-08",
        selected_tickers=["AAPL"],
    )
    create_pending(
        observation_key="obs_multi2",
        strategy_id=STRATEGY_ID,
        holding_sessions=5,
        outcome_definition_version=OUTCOME_DEFINITION_VERSION,
        entry_session="2026-09-08",
        exit_session="2026-09-14",
        selected_tickers=["AAPL"],
    )
    pending = list_pending_outcomes()
    assert len(pending) == 2


# ════════════════════════════════════════════════════════════════════════════════
# validate_and_rebuild_index
# ════════════════════════════════════════════════════════════════════════════════


def test_validate_empty_dir():
    idx = validate_and_rebuild_index()
    assert idx == {}


def test_validate_after_write():
    _make_pending()
    ok = _ok()
    idx = validate_and_rebuild_index()
    assert ok in idx


def test_validate_rebuilds_missing_index(tmp_outcome):
    _make_pending()
    ok = _ok()
    # Delete index
    idx_path = tmp_outcome / "v2b_outcome_idx.json"
    idx_path.unlink()
    idx = validate_and_rebuild_index()
    assert ok in idx
    assert idx_path.exists()


def test_validate_detects_broken_event_hash(tmp_outcome):
    _make_pending()
    ok = _ok()
    partition = ok  # find it via index
    idx = json.loads((tmp_outcome / "v2b_outcome_idx.json").read_text())
    part = idx[ok]
    path = tmp_outcome / f"{part}_v2b_outcomes.jsonl"
    # Corrupt the event_hash
    lines = path.read_text().splitlines()
    ev = json.loads(lines[0])
    ev["event_hash"] = "a" * 64
    path.write_text(json.dumps(ev) + "\n")
    with pytest.raises(CorruptionError):
        validate_and_rebuild_index()


def test_validate_detects_broken_previous_hash(tmp_outcome):
    """Test that a broken hash chain (wrong previous_event_hash) is detected."""
    _make_pending()
    ok = _ok()
    missing = [{"symbol": "AAPL", "field": "adjusted_open", "session": "2026-09-08"}]
    record_fetch_deferred(ok, "t", "2026-09-14", missing, "yfinance", "x")
    # Corrupt the second event's previous_event_hash
    idx = json.loads((tmp_outcome / "v2b_outcome_idx.json").read_text())
    part = idx[ok]
    path = tmp_outcome / f"{part}_v2b_outcomes.jsonl"
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    # Tamper the second event: change previous_event_hash but keep event_hash
    lines[1]["previous_event_hash"] = "b" * 64
    # Recompute event_hash to pass hash check
    body = {k: v for k, v in lines[1].items() if k != "event_hash"}
    lines[1]["event_hash"] = _compute_event_hash(body)
    path.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    with pytest.raises(CorruptionError, match="broken chain"):
        validate_and_rebuild_index()


# ════════════════════════════════════════════════════════════════════════════════
# Hash chain correctness
# ════════════════════════════════════════════════════════════════════════════════


def test_hash_chain_pending_to_terminal():
    _make_pending()
    ok = _ok()
    payload = _minimal_terminal_payload(ok)
    record_terminal(ok, payload)

    events = get_outcome_events(ok)
    assert len(events) == 2
    assert events[0]["previous_event_hash"] is None
    assert events[1]["previous_event_hash"] == events[0]["event_hash"]


def test_hash_chain_pending_to_deferred_to_terminal():
    _make_pending()
    ok = _ok()
    missing = [{"symbol": "AAPL", "field": "adjusted_open", "session": "2026-09-08"}]
    record_fetch_deferred(ok, "t", "2026-09-14", missing, "yfinance", "x")
    p = _minimal_terminal_payload(ok, terminal_type="OUTCOME_INCOMPLETE")
    p["portfolio_return_pct"] = None
    p["forward_alpha_vs_spy"] = None
    p["hit_rate_vs_spy"] = None
    p["portfolio_return_complete"] = False
    p["unavailable_tickers"] = ["AAPL"]
    record_terminal(ok, p)

    events = get_outcome_events(ok)
    assert len(events) == 3
    assert events[0]["previous_event_hash"] is None
    assert events[1]["previous_event_hash"] == events[0]["event_hash"]
    assert events[2]["previous_event_hash"] == events[1]["event_hash"]


# ════════════════════════════════════════════════════════════════════════════════
# V1 isolation
# ════════════════════════════════════════════════════════════════════════════════


def test_no_v1_imports():
    """v2b_outcome must not statically import V1 execution modules (AST check)."""
    import ast as _ast

    forbidden_modules = {
        "modules.ledger",
        "modules.orders",
        "modules.portfolio",
        "modules.fills",
        "modules.state",
    }
    source_files = [
        "modules/v2b_outcome.py",
        "modules/v2b_outcome_runner.py",
        "v2b_daily_outcome.py",
    ]
    for fname in source_files:
        src = open(fname).read()
        tree = _ast.parse(src, filename=fname)
        for node in _ast.walk(tree):
            if isinstance(node, _ast.ImportFrom):
                mod = node.module or ""
                if mod in forbidden_modules:
                    pytest.fail(f"{fname}: imports forbidden module {mod}")
            elif isinstance(node, _ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_modules:
                        pytest.fail(f"{fname}: imports forbidden module {alias.name}")


def test_order_creation_blocked_in_pending_event():
    ev = _make_pending()
    assert ev["order_creation_blocked"] is True


def test_order_creation_blocked_in_terminal_event():
    _make_pending()
    ok = _ok()
    payload = _minimal_terminal_payload(ok)
    ev = record_terminal(ok, payload)
    assert ev["order_creation_blocked"] is True


def test_order_creation_blocked_in_fetch_deferred():
    _make_pending()
    ok = _ok()
    ev = record_fetch_deferred(ok, "t", "2026-09-14", [], "yfinance", "x")
    assert ev is not None
    assert ev["order_creation_blocked"] is True


# ════════════════════════════════════════════════════════════════════════════════
# Partitioning
# ════════════════════════════════════════════════════════════════════════════════


def test_partition_by_exit_session(tmp_outcome):
    """Events must land in YYYY-MM_v2b_outcomes.jsonl matching the exit_session month."""
    ev = create_pending(
        observation_key="obs_part",
        strategy_id=STRATEGY_ID,
        holding_sessions=5,
        outcome_definition_version=OUTCOME_DEFINITION_VERSION,
        entry_session="2026-09-08",
        exit_session="2026-10-06",
        selected_tickers=["AAPL"],
    )
    partition = ev["exit_partition_yyyymm"]
    assert partition == "2026-10"
    assert (tmp_outcome / "2026-10_v2b_outcomes.jsonl").exists()


def test_cross_year_partition(tmp_outcome):
    ev = create_pending(
        observation_key="obs_year",
        strategy_id=STRATEGY_ID,
        holding_sessions=63,
        outcome_definition_version=OUTCOME_DEFINITION_VERSION,
        entry_session="2026-10-05",
        exit_session="2027-01-04",
        selected_tickers=["AAPL"],
    )
    assert ev["exit_partition_yyyymm"] == "2027-01"
    assert (tmp_outcome / "2027-01_v2b_outcomes.jsonl").exists()


# ════════════════════════════════════════════════════════════════════════════════
# Runner: _extract_price
# ════════════════════════════════════════════════════════════════════════════════


def test_extract_price_single_ticker():
    """Single-ticker DataFrame (flat columns)."""
    import pandas as pd
    from modules.v2b_outcome_runner import _extract_price

    idx = pd.to_datetime(["2026-09-08", "2026-09-14"])
    df = pd.DataFrame({"Open": [220.0, 225.0], "Close": [221.0, 231.0]}, index=idx)
    assert _extract_price(df, "AAPL", "2026-09-08", "adjusted_open") == pytest.approx(220.0)
    assert _extract_price(df, "AAPL", "2026-09-14", "adjusted_close") == pytest.approx(231.0)


def test_extract_price_multi_ticker():
    """Multi-ticker DataFrame (MultiIndex columns)."""
    import pandas as pd
    from modules.v2b_outcome_runner import _extract_price

    idx = pd.to_datetime(["2026-09-08", "2026-09-14"])
    cols = pd.MultiIndex.from_tuples([("Open", "AAPL"), ("Close", "AAPL"), ("Open", "MSFT"), ("Close", "MSFT")])
    data = [[220.0, 221.0, 430.0, 431.0], [225.0, 231.0, 435.0, 440.0]]
    df = pd.DataFrame(data, index=idx, columns=cols)
    assert _extract_price(df, "AAPL", "2026-09-08", "adjusted_open") == pytest.approx(220.0)
    assert _extract_price(df, "MSFT", "2026-09-14", "adjusted_close") == pytest.approx(440.0)


def test_extract_price_missing_date():
    import pandas as pd
    from modules.v2b_outcome_runner import _extract_price

    idx = pd.to_datetime(["2026-09-08"])
    df = pd.DataFrame({"Open": [220.0], "Close": [221.0]}, index=idx)
    assert _extract_price(df, "AAPL", "2026-09-15", "adjusted_open") is None


def test_extract_price_missing_symbol_in_multi():
    import pandas as pd
    from modules.v2b_outcome_runner import _extract_price

    idx = pd.to_datetime(["2026-09-08"])
    cols = pd.MultiIndex.from_tuples([("Open", "AAPL"), ("Close", "AAPL")])
    df = pd.DataFrame([[220.0, 221.0]], index=idx, columns=cols)
    assert _extract_price(df, "GOOG", "2026-09-08", "adjusted_open") is None


def test_extract_price_none_df():
    from modules.v2b_outcome_runner import _extract_price
    assert _extract_price(None, "AAPL", "2026-09-08", "adjusted_open") is None


def test_extract_price_unknown_field():
    import pandas as pd
    from modules.v2b_outcome_runner import _extract_price

    idx = pd.to_datetime(["2026-09-08"])
    df = pd.DataFrame({"Open": [220.0]}, index=idx)
    assert _extract_price(df, "AAPL", "2026-09-08", "volume") is None


# ════════════════════════════════════════════════════════════════════════════════
# Runner: run_outcome_tracker integration
# ════════════════════════════════════════════════════════════════════════════════


def _build_fake_observation_events(obs_key, entry_session, tickers):
    """Build a minimal list of events that looks like a COMPLETED V2B observation."""
    return [
        {
            "event_type": "OBSERVATION_CREATED",
            "observation_key": obs_key,
            "model_version": "quant_baseline_v2",
            "intended_execution_session": entry_session,
            "selected_tickers_per_strategy": {
                STRATEGY_ID: tickers,
            },
        },
        {
            "event_type": "OBSERVATION_COMPLETED",
            "observation_key": obs_key,
        },
    ]


def _make_fake_df(tickers_and_spy, entry_session, exit_session):
    """Create a fake multi-ticker yfinance DataFrame for testing."""
    import pandas as pd
    all_syms = list(tickers_and_spy)
    idx = pd.to_datetime([entry_session, exit_session])
    cols = pd.MultiIndex.from_tuples(
        [(field, sym) for sym in all_syms for field in ["Open", "Close"]]
    )
    prices = {
        "AAPL": (220.0, 225.0, 221.0, 231.0),  # open_entry, open_exit, close_entry, close_exit
        "MSFT": (430.0, 435.0, 431.0, 440.0),
        "SPY": (550.0, 552.0, 551.0, 557.75),
    }
    data = []
    for date in idx:
        row = []
        for sym in all_syms:
            p = prices.get(sym, (100.0, 101.0, 100.5, 102.0))
            if str(date)[:10] == entry_session:
                row.extend([p[0], p[2]])  # Open, Close for entry
            else:
                row.extend([p[1], p[3]])  # Open, Close for exit
        data.append(row)
    return pd.DataFrame(data, index=idx, columns=cols)


def test_run_outcome_tracker_full_cycle(tmp_outcome, monkeypatch):
    """Full integration: repair + process → OUTCOME_RECORDED."""
    import modules.v2b_outcome_runner as runner

    obs_key = "obs_integration_001"
    entry_session = "2026-09-08"
    exit_session = "2026-09-14"
    tickers = ["AAPL", "MSFT"]

    # Monkeypatch V2B ledger read-only calls
    fake_obs_list = [
        {
            "observation_key": obs_key,
            "status": "COMPLETED",
            "intended_execution_session": entry_session,
        }
    ]
    monkeypatch.setattr(runner, "list_observations", lambda: fake_obs_list)
    monkeypatch.setattr(
        runner, "get_observation_events",
        lambda key: _build_fake_observation_events(key, entry_session, tickers),
    )

    # Monkeypatch exchange_calendar
    monkeypatch.setattr(runner, "is_trading_session", lambda d: True)

    # Monkeypatch compute_exit_session
    def fake_exit(entry, n):
        mapping = {1: entry_session, 5: exit_session, 21: "2026-10-06", 63: "2026-12-04"}
        return mapping[n]
    monkeypatch.setattr(runner, "compute_exit_session", fake_exit)

    # Monkeypatch yfinance
    fake_df = _make_fake_df(tickers + ["SPY"], entry_session, exit_session)
    monkeypatch.setattr(runner, "_fetch_ohlcv_with_retry", lambda *a, **kw: fake_df)

    # Run on exit day (N=5 only matured today)
    exit_code = runner.run_outcome_tracker(exit_session)
    assert exit_code == 0

    # N=5 outcome should be RECORDED
    ok5 = make_outcome_key(obs_key, STRATEGY_ID, 5, OUTCOME_DEFINITION_VERSION)
    assert get_outcome_status(ok5) == "OUTCOME_RECORDED"

    events = get_outcome_events(ok5)
    terminal_ev = events[-1]
    assert terminal_ev["portfolio_return_complete"] is True
    assert terminal_ev["portfolio_return_pct"] is not None
    assert terminal_ev["forward_alpha_vs_spy"] is not None
    assert terminal_ev["hit_rate_vs_spy"] is not None
    assert terminal_ev["order_creation_blocked"] is True


def test_run_outcome_tracker_transport_error_returns_1(tmp_outcome, monkeypatch):
    """Transport failure → exit code 1, PENDING kept."""
    import modules.v2b_outcome_runner as runner

    obs_key = "obs_transport_fail"
    entry_session = "2026-09-08"
    exit_session = "2026-09-08"  # N=1

    monkeypatch.setattr(runner, "list_observations", lambda: [
        {"observation_key": obs_key, "status": "COMPLETED", "intended_execution_session": entry_session}
    ])
    monkeypatch.setattr(runner, "get_observation_events",
        lambda key: _build_fake_observation_events(key, entry_session, ["AAPL"]))
    monkeypatch.setattr(runner, "is_trading_session", lambda d: True)
    monkeypatch.setattr(runner, "compute_exit_session", lambda entry, n: exit_session if n <= 1 else "2099-01-01")
    monkeypatch.setattr(runner, "_fetch_ohlcv_with_retry", lambda *a, **kw: None)  # transport failure

    exit_code = runner.run_outcome_tracker(entry_session)
    assert exit_code == 1

    ok1 = make_outcome_key(obs_key, STRATEGY_ID, 1, OUTCOME_DEFINITION_VERSION)
    assert get_outcome_status(ok1) == "OUTCOME_PENDING"


def test_run_outcome_tracker_missing_prices_before_grace(tmp_outcome, monkeypatch):
    """Missing prices before grace period → FETCH_DEFERRED."""
    import pandas as pd
    import modules.v2b_outcome_runner as runner

    obs_key = "obs_deferred"
    entry_session = "2026-09-08"
    exit_session = "2026-09-08"

    monkeypatch.setattr(runner, "list_observations", lambda: [
        {"observation_key": obs_key, "status": "COMPLETED", "intended_execution_session": entry_session}
    ])
    monkeypatch.setattr(runner, "get_observation_events",
        lambda key: _build_fake_observation_events(key, entry_session, ["AAPL"]))
    monkeypatch.setattr(runner, "is_trading_session", lambda d: True)
    monkeypatch.setattr(runner, "compute_exit_session", lambda entry, n: exit_session if n <= 1 else "2099-01-01")

    # Return empty DataFrame (valid response, but no price rows)
    monkeypatch.setattr(runner, "_fetch_ohlcv_with_retry",
        lambda *a, **kw: pd.DataFrame())

    # sessions_between_count returns 2 < 5 (grace not elapsed)
    monkeypatch.setattr(runner, "sessions_between_count", lambda start, end: 2)

    exit_code = runner.run_outcome_tracker(entry_session)
    assert exit_code == 0  # not a transport error

    ok1 = make_outcome_key(obs_key, STRATEGY_ID, 1, OUTCOME_DEFINITION_VERSION)
    events = get_outcome_events(ok1)
    event_types = [e["event_type"] for e in events]
    assert "OUTCOME_FETCH_DEFERRED" in event_types
    assert get_outcome_status(ok1) == "OUTCOME_PENDING"


def test_run_outcome_tracker_missing_prices_after_grace(tmp_outcome, monkeypatch):
    """Missing prices after grace period → OUTCOME_UNAVAILABLE."""
    import pandas as pd
    import modules.v2b_outcome_runner as runner

    obs_key = "obs_unavailable"
    entry_session = "2026-09-08"
    exit_session = "2026-09-08"

    monkeypatch.setattr(runner, "list_observations", lambda: [
        {"observation_key": obs_key, "status": "COMPLETED", "intended_execution_session": entry_session}
    ])
    monkeypatch.setattr(runner, "get_observation_events",
        lambda key: _build_fake_observation_events(key, entry_session, ["AAPL"]))
    monkeypatch.setattr(runner, "is_trading_session", lambda d: True)
    monkeypatch.setattr(runner, "compute_exit_session", lambda entry, n: exit_session if n <= 1 else "2099-01-01")

    monkeypatch.setattr(runner, "_fetch_ohlcv_with_retry",
        lambda *a, **kw: pd.DataFrame())

    # Grace has elapsed
    monkeypatch.setattr(runner, "sessions_between_count", lambda start, end: 5)

    exit_code = runner.run_outcome_tracker(exit_session)
    assert exit_code == 0

    ok1 = make_outcome_key(obs_key, STRATEGY_ID, 1, OUTCOME_DEFINITION_VERSION)
    assert get_outcome_status(ok1) == "OUTCOME_UNAVAILABLE"


def test_run_outcome_tracker_idempotent_repair(tmp_outcome, monkeypatch):
    """Running the tracker twice produces the same PENDING events (no duplicates)."""
    import modules.v2b_outcome_runner as runner

    obs_key = "obs_idempotent_repair"
    entry_session = "2026-09-08"

    def fake_obs():
        return [{"observation_key": obs_key, "status": "COMPLETED", "intended_execution_session": entry_session}]

    monkeypatch.setattr(runner, "list_observations", fake_obs)
    monkeypatch.setattr(runner, "get_observation_events",
        lambda key: _build_fake_observation_events(key, entry_session, ["AAPL"]))
    monkeypatch.setattr(runner, "is_trading_session", lambda d: True)
    # All exits in future — only repair step runs
    monkeypatch.setattr(runner, "compute_exit_session", lambda entry, n: "2099-01-01")
    monkeypatch.setattr(runner, "_fetch_ohlcv_with_retry", lambda *a, **kw: None)

    today = "2026-09-08"
    runner.run_outcome_tracker(today)
    runner.run_outcome_tracker(today)

    # Should still be exactly 4 PENDINGs (one per holding_sessions)
    pending = list_pending_outcomes()
    assert len(pending) == 4


# ════════════════════════════════════════════════════════════════════════════════
# Aggregate metrics correctness
# ════════════════════════════════════════════════════════════════════════════════


def test_aggregates_null_if_any_ticker_missing():
    """If ANY selected ticker is missing, portfolio_return_pct and derivatives must be null."""
    _make_pending()
    ok = _ok()
    p = _minimal_terminal_payload(ok, terminal_type="OUTCOME_INCOMPLETE")
    # AAPL prices missing
    p["entry_prices"] = {"MSFT": 430.0}
    p["exit_prices"] = {"MSFT": 440.0}
    p["per_ticker_return_pct"] = {"MSFT": 2.326}
    p["unavailable_tickers"] = ["AAPL"]
    p["unavailable_reasons"] = {"AAPL": "price_row_missing"}
    p["portfolio_return_pct"] = None  # must be null per design
    p["forward_alpha_vs_spy"] = None
    p["hit_rate_vs_spy"] = None
    p["portfolio_return_complete"] = False
    p["available_subset_return_pct"] = 2.326  # separate metric — may be non-null
    ev = record_terminal(ok, p)
    assert ev["portfolio_return_pct"] is None
    assert ev["forward_alpha_vs_spy"] is None
    assert ev["hit_rate_vs_spy"] is None
    assert ev["available_subset_return_pct"] == pytest.approx(2.326)


def test_aggregates_null_if_spy_missing():
    """If SPY is missing, portfolio_return_pct, alpha, and hit_rate must be null."""
    _make_pending()
    ok = _ok()
    p = _minimal_terminal_payload(ok, terminal_type="OUTCOME_INCOMPLETE")
    p["spy_entry_price"] = None
    p["spy_exit_price"] = None
    p["spy_return_pct"] = None
    p["portfolio_return_pct"] = None
    p["forward_alpha_vs_spy"] = None
    p["hit_rate_vs_spy"] = None
    p["portfolio_return_complete"] = False
    ev = record_terminal(ok, p)
    assert ev["portfolio_return_pct"] is None
    assert ev["forward_alpha_vs_spy"] is None
    assert ev["hit_rate_vs_spy"] is None


# ════════════════════════════════════════════════════════════════════════════════
# Outcome key format verification
# ════════════════════════════════════════════════════════════════════════════════


def test_outcome_key_format_in_event():
    ev = _make_pending()
    ok = ev["outcome_key"]
    assert len(ok) == 64
    assert all(c in "0123456789abcdef" for c in ok)


def test_pending_content_hash_in_event():
    ev = _make_pending()
    assert "pending_content_hash" in ev
    pch = ev["pending_content_hash"]
    assert len(pch) == 64


def test_terminal_content_hash_in_terminal_event():
    _make_pending()
    ok = _ok()
    payload = _minimal_terminal_payload(ok)
    ev = record_terminal(ok, payload)
    assert "terminal_content_hash" in ev
    assert len(ev["terminal_content_hash"]) == 64
