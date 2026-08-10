"""
V2B Shadow Observation Ledger — immutable append-only event store.

Stores prospective shadow observations of the V2 model scoring process.
Cannot create orders, fills, trades, or touch any V1 state.

File layout: data_v4/v2b_ledger/{YYYY-MM}_v2b_observations.jsonl
             data_v4/v2b_ledger/{YYYY-MM}_v2b_observations.lock
             data_v4/v2b_ledger/v2b_idx.json        (key→file index)

observation_key = SHA-256(model_version + "|" + model_config_hash + "|" + intended_execution_session)

Status machine:
  CREATED → COLLECTING → COMPLETED
                       → FAILED_DATA
                       → FAILED_VALIDATION
                       → CANCELLED
  CREATED → CONFLICT  (same key, different content_hash — raises ConflictError)
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import uuid
import warnings
from datetime import timezone
from pathlib import Path
from typing import Literal

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEDGER_VERSION = "1"
RECORD_VERSION = "1"
LEDGER_DIR = Path("data_v4/v2b_ledger")

# Structural invariant — never overridable
_ORDER_CREATION_BLOCKED: bool = True

ObservationStatus = Literal[
    "CREATED",
    "COLLECTING",
    "COMPLETED",
    "FAILED_DATA",
    "FAILED_VALIDATION",
    "CANCELLED",
    "CONFLICT",
]

EventType = Literal[
    "OBSERVATION_CREATED",
    "COLLECTING",
    "COMPLETED",
    "FAILED_DATA",
    "FAILED_VALIDATION",
    "CANCELLED",
    "CONFLICT",
    "IDEMPOTENT_MATCH",
]

ProvenanceType = Literal["point_in_time", "current_snapshot", "unavailable", "unknown"]

# Valid status transitions
_TRANSITIONS: dict[ObservationStatus, set[ObservationStatus]] = {
    "CREATED": {"COLLECTING", "CONFLICT"},
    "COLLECTING": {"COMPLETED", "FAILED_DATA", "FAILED_VALIDATION", "CANCELLED"},
    "COMPLETED": set(),
    "FAILED_DATA": set(),
    "FAILED_VALIDATION": set(),
    "CANCELLED": set(),
    "CONFLICT": set(),
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConflictError(RuntimeError):
    """Raised when an observation_key is reused with different content."""


class CorruptionError(RuntimeError):
    """Raised on mid-file JSON corruption (fail-closed)."""


class InvalidTransitionError(RuntimeError):
    """Raised on illegal status transition."""


# ---------------------------------------------------------------------------
# Key / hash helpers
# ---------------------------------------------------------------------------


def make_observation_key(
    model_version: str, model_config_hash: str, intended_execution_session: str
) -> str:
    payload = f"{model_version}|{model_config_hash}|{intended_execution_session}"
    return hashlib.sha256(payload.encode()).hexdigest()


def make_content_hash(canonical_payload: dict) -> str:
    serialized = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def provenance_entry(
    provenance_type: ProvenanceType,
    source: str,
    as_of_date: str | None = None,
    note: str | None = None,
) -> dict:
    entry: dict = {"type": provenance_type, "source": source}
    if as_of_date is not None:
        entry["as_of_date"] = as_of_date
    if note is not None:
        entry["note"] = note
    return entry


def _now_iso() -> str:
    return pd.Timestamp.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Per-ticker observation record
# ---------------------------------------------------------------------------


def make_ticker_record(
    ticker: str,
    momentum_score: float | None,
    quality_score: float | None,
    value_score: float | None,
    safety_score: float | None,
    composite_score: float | None,
    factor_coverage: float | None,
    rank: int | None,
    excluded: bool,
    exclusion_reason: str | None,
    sector: str | None,
    value_sector_adjusted: bool,
    provenance: list[dict],
) -> dict:
    return {
        "ticker": ticker,
        "scores": {
            "momentum": momentum_score,
            "quality": quality_score,
            "value": value_score,
            "safety": safety_score,
            "composite": composite_score,
        },
        "factor_coverage": factor_coverage,
        "rank": rank,
        "excluded": excluded,
        "exclusion_reason": exclusion_reason,
        "sector": sector,
        "value_sector_adjusted": value_sector_adjusted,
        "provenance": provenance,
    }


# ---------------------------------------------------------------------------
# File path helpers
# ---------------------------------------------------------------------------


def _ledger_path(yyyymm: str) -> Path:
    return LEDGER_DIR / f"{yyyymm}_v2b_observations.jsonl"


def _lock_path(yyyymm: str) -> Path:
    return LEDGER_DIR / f"{yyyymm}_v2b_observations.lock"


def _idx_path() -> Path:
    return LEDGER_DIR / "v2b_idx.json"


def _yyyymm_for(session_date: str) -> str:
    return session_date[:7]


# ---------------------------------------------------------------------------
# Index helpers (key → YYYY-MM, so we can locate the right JSONL file)
# ---------------------------------------------------------------------------


def _load_idx() -> dict[str, str]:
    p = _idx_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_idx_atomic(idx: dict[str, str]) -> None:
    p = _idx_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(idx, sort_keys=True))
    tmp.replace(p)


# ---------------------------------------------------------------------------
# Safe JSONL reader
# ---------------------------------------------------------------------------


def _read_events_safe(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events: list[dict] = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            is_last = i == len(lines) - 1
            if is_last:
                warnings.warn(
                    f"v2b_ledger: incomplete last line in {path.name} — skipped",
                    stacklevel=2,
                )
            else:
                raise CorruptionError(
                    f"v2b_ledger: mid-file JSON corruption at line {i+1} in {path.name} — fail-closed"
                )
    return events


# ---------------------------------------------------------------------------
# Atomic append helper
# ---------------------------------------------------------------------------


def _append_event_locked(path: Path, lock: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock.touch(exist_ok=True)
    with open(lock, "r") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            line = json.dumps(event, sort_keys=True, separators=(",", ":"), default=str) + "\n"
            tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".ev_")
            try:
                with os.fdopen(tmp_fd, "w") as tf:
                    tf.write(line)
                    tf.flush()
                    os.fsync(tf.fileno())
                # append to main file atomically
                with open(path, "a") as af:
                    af.write(line)
                    af.flush()
                    os.fsync(af.fileno())
            finally:
                try:
                    os.unlink(tmp_name)
                except FileNotFoundError:
                    pass
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------


def create_observation(
    *,
    model_version: str,
    model_config_hash: str,
    intended_execution_session: str,
    universe_count: int,
    signal_coverage: float,
    ticker_records: list[dict],
    metadata: dict | None = None,
) -> dict:
    """
    Open a new shadow observation. Returns the created event dict.

    Idempotency:
      - Same key + same content_hash → returns IDEMPOTENT_MATCH event (no write)
      - Same key + different content_hash → writes CONFLICT event and raises ConflictError
    """
    observation_key = make_observation_key(
        model_version, model_config_hash, intended_execution_session
    )
    yyyymm = _yyyymm_for(intended_execution_session)
    path = _ledger_path(yyyymm)
    lock = _lock_path(yyyymm)

    canonical = {
        "model_version": model_version,
        "model_config_hash": model_config_hash,
        "intended_execution_session": intended_execution_session,
        "universe_count": universe_count,
        "signal_coverage": signal_coverage,
        "ticker_records": ticker_records,
        "metadata": metadata or {},
    }
    content_hash = make_content_hash(canonical)

    # Check for existing observation with this key
    existing = _find_events_for_key(observation_key, path)
    if existing:
        prior = existing[0]
        prior_hash = prior.get("content_hash")
        if prior_hash == content_hash:
            idempotent_event = {
                "event_type": "IDEMPOTENT_MATCH",
                "record_version": RECORD_VERSION,
                "observation_key": observation_key,
                "observation_run_id": prior.get("observation_run_id"),
                "content_hash": content_hash,
                "timestamp_utc": _now_iso(),
                "order_creation_blocked": _ORDER_CREATION_BLOCKED,
            }
            return idempotent_event
        else:
            conflict_event = {
                "event_type": "CONFLICT",
                "record_version": RECORD_VERSION,
                "observation_key": observation_key,
                "observation_run_id": str(uuid.uuid4()),
                "prior_content_hash": prior_hash,
                "new_content_hash": content_hash,
                "timestamp_utc": _now_iso(),
                "order_creation_blocked": _ORDER_CREATION_BLOCKED,
            }
            _append_event_locked(path, lock, conflict_event)
            raise ConflictError(
                f"observation_key {observation_key[:16]}… already exists with different content. "
                f"Prior hash: {prior_hash[:16]}… New hash: {content_hash[:16]}…"
            )

    run_id = str(uuid.uuid4())
    event = {
        "event_type": "OBSERVATION_CREATED",
        "record_version": RECORD_VERSION,
        "ledger_version": LEDGER_VERSION,
        "observation_key": observation_key,
        "observation_run_id": run_id,
        "content_hash": content_hash,
        "timestamp_utc": _now_iso(),
        "order_creation_blocked": _ORDER_CREATION_BLOCKED,
        "model_version": model_version,
        "model_config_hash": model_config_hash,
        "intended_execution_session": intended_execution_session,
        "universe_count": universe_count,
        "signal_coverage": signal_coverage,
        "ticker_records": ticker_records,
        "metadata": metadata or {},
        # point_in_time_fundamentals is never True globally — see per-ticker provenance
        "point_in_time_fundamentals_global": False,
    }

    _append_event_locked(path, lock, event)

    # Update index
    idx = _load_idx()
    idx[observation_key] = yyyymm
    _save_idx_atomic(idx)

    return event


def transition_observation(
    observation_key: str,
    new_status: ObservationStatus,
    note: str | None = None,
) -> dict:
    """
    Append a status transition event. Validates the transition is legal.
    """
    idx = _load_idx()
    yyyymm = idx.get(observation_key)
    if yyyymm is None:
        raise KeyError(f"observation_key not found in index: {observation_key[:16]}…")

    path = _ledger_path(yyyymm)
    lock = _lock_path(yyyymm)

    events = _find_events_for_key(observation_key, path)
    if not events:
        raise KeyError(f"No events found for observation_key: {observation_key[:16]}…")

    current_status = _derive_status(events)
    allowed = _TRANSITIONS.get(current_status, set())
    if new_status not in allowed:
        raise InvalidTransitionError(
            f"Cannot transition from {current_status} → {new_status}. "
            f"Allowed: {sorted(allowed) or 'none (terminal)'}"
        )

    run_id = events[0].get("observation_run_id")
    event: dict = {
        "event_type": new_status,
        "record_version": RECORD_VERSION,
        "observation_key": observation_key,
        "observation_run_id": run_id,
        "timestamp_utc": _now_iso(),
        "order_creation_blocked": _ORDER_CREATION_BLOCKED,
    }
    if note:
        event["note"] = note

    _append_event_locked(path, lock, event)
    return event


def get_observation_status(observation_key: str) -> ObservationStatus | None:
    """Return current status of an observation, or None if not found."""
    idx = _load_idx()
    yyyymm = idx.get(observation_key)
    if yyyymm is None:
        return None
    path = _ledger_path(yyyymm)
    events = _find_events_for_key(observation_key, path)
    if not events:
        return None
    return _derive_status(events)


def get_observation_events(observation_key: str) -> list[dict]:
    """Return all events for an observation_key, in append order."""
    idx = _load_idx()
    yyyymm = idx.get(observation_key)
    if yyyymm is None:
        return []
    path = _ledger_path(yyyymm)
    return _find_events_for_key(observation_key, path)


def list_observations(yyyymm: str | None = None) -> list[dict]:
    """
    Return a list of summary dicts (one per unique observation_key).
    If yyyymm is None, reads all JSONL files in the ledger directory.
    """
    if yyyymm is not None:
        files = [_ledger_path(yyyymm)]
    else:
        files = sorted(LEDGER_DIR.glob("*_v2b_observations.jsonl"))

    by_key: dict[str, list[dict]] = {}
    for path in files:
        try:
            events = _read_events_safe(path)
        except CorruptionError:
            raise
        for ev in events:
            key = ev.get("observation_key")
            if key:
                by_key.setdefault(key, []).append(ev)

    result = []
    for key, evs in by_key.items():
        first = evs[0]
        result.append(
            {
                "observation_key": key,
                "observation_run_id": first.get("observation_run_id"),
                "status": _derive_status(evs),
                "model_version": first.get("model_version"),
                "intended_execution_session": first.get("intended_execution_session"),
                "content_hash": first.get("content_hash"),
                "created_at": first.get("timestamp_utc"),
                "event_count": len(evs),
            }
        )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_events_for_key(observation_key: str, path: Path) -> list[dict]:
    events = _read_events_safe(path)
    return [e for e in events if e.get("observation_key") == observation_key]


def _derive_status(events: list[dict]) -> ObservationStatus:
    """Derive current status from ordered event list."""
    if not events:
        raise ValueError("Cannot derive status from empty event list")
    status_map: dict[str, ObservationStatus] = {
        "OBSERVATION_CREATED": "CREATED",
        "COLLECTING": "COLLECTING",
        "COMPLETED": "COMPLETED",
        "FAILED_DATA": "FAILED_DATA",
        "FAILED_VALIDATION": "FAILED_VALIDATION",
        "CANCELLED": "CANCELLED",
        "CONFLICT": "CONFLICT",
        "IDEMPOTENT_MATCH": "CREATED",  # idempotent — did not change state
    }
    current: ObservationStatus = "CREATED"
    for ev in events:
        et = ev.get("event_type", "")
        if et in status_map:
            candidate = status_map[et]
            # Only update if it's a real transition (not idempotent)
            if et != "IDEMPOTENT_MATCH":
                current = candidate
    return current


# ---------------------------------------------------------------------------
# Structural invariant assertion (callable by tests / callers)
# ---------------------------------------------------------------------------


def assert_order_creation_blocked() -> None:
    """Raises AssertionError if structural invariant is violated."""
    assert _ORDER_CREATION_BLOCKED is True, (
        "V2B ledger structural invariant violated: order_creation_blocked must always be True"
    )
