"""
V2B Shadow Observation Ledger — immutable append-only event store.

Stores prospective shadow observations of the V2 model scoring process.
Cannot create orders, fills, trades, or touch any V1 state.

File layout (all paths relative to this file's repo root):
  data_v4/v2b_ledger/{YYYY-MM}_v2b_observations.jsonl
  data_v4/v2b_ledger/{YYYY-MM}_v2b_observations.lock
  data_v4/v2b_ledger/v2b_idx.json       (observation_key → YYYY-MM)
  data_v4/v2b_ledger/v2b_idx.lock

observation_key = SHA-256(model_version + "|" + model_config_hash + "|" + intended_execution_session)

Each event carries:
  event_hash           = SHA-256(canonical JSON of event WITHOUT event_hash field)
  previous_event_hash  = event_hash of prior event for same observation_key (None for first)

This forms a per-observation tamper-evident chain.

Status machine:
  CREATED → COLLECTING → COMPLETED
                       → FAILED_DATA
                       → FAILED_VALIDATION
                       → CANCELLED
  CREATED → CONFLICT   (same key, different content_hash — raises ConflictError)

Isolation: zero imports from V1 execution modules.
  Does NOT import: modules.portfolio, modules.orders, modules.fills,
                   modules.ledger, modules.state
  Does NOT call:   execute_buy, execute_sell, execute_pyramid_fill
  Does NOT create: orders, fills, trades, or position changes
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
import uuid
import warnings
from contextlib import contextmanager
from datetime import timezone
from pathlib import Path
from typing import Generator, Literal

import pandas as pd

# ---------------------------------------------------------------------------
# Constants — LEDGER_DIR is anchored to the repository root via __file__
# ---------------------------------------------------------------------------

LEDGER_VERSION = "1"
RECORD_VERSION = "2"  # v2: hash chain, strict validation

# Anchor to repo root (modules/ lives one level below repo root)
LEDGER_DIR: Path = Path(__file__).parent.parent / "data_v4" / "v2b_ledger"

# Structural invariant — hardcoded, never overridable by caller
_ORDER_CREATION_BLOCKED: bool = True

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

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

_VALID_PROVENANCE_TYPES: frozenset[str] = frozenset(
    {"point_in_time", "current_snapshot", "unavailable", "unknown"}
)

_TRANSITIONS: dict[str, set[str]] = {
    "CREATED": {"COLLECTING", "CONFLICT"},
    "COLLECTING": {"COMPLETED", "FAILED_DATA", "FAILED_VALIDATION", "CANCELLED"},
    "COMPLETED": set(),
    "FAILED_DATA": set(),
    "FAILED_VALIDATION": set(),
    "CANCELLED": set(),
    "CONFLICT": set(),
}

_STATUS_FROM_EVENT: dict[str, str] = {
    "OBSERVATION_CREATED": "CREATED",
    "COLLECTING": "COLLECTING",
    "COMPLETED": "COMPLETED",
    "FAILED_DATA": "FAILED_DATA",
    "FAILED_VALIDATION": "FAILED_VALIDATION",
    "CANCELLED": "CANCELLED",
    "CONFLICT": "CONFLICT",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConflictError(RuntimeError):
    """Raised when an observation_key is reused with different content."""


class CorruptionError(RuntimeError):
    """Raised on hash mismatch, broken chain, or mid-file JSON corruption."""


class InvalidTransitionError(RuntimeError):
    """Raised on illegal status transition."""


class ValidationError(ValueError):
    """Raised on invalid input to create_observation or make_ticker_record."""


# ---------------------------------------------------------------------------
# Canonical JSON — no default=str, allow_nan=False
# ---------------------------------------------------------------------------


def _canonical_json(obj: object) -> str:
    """Strict canonical JSON for hashing. Raises ValueError on NaN/Inf/unsupported types."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _store_json(obj: object) -> str:
    """JSON for storage (same strict rules as canonical)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------


def _compute_event_hash(event_body: dict) -> str:
    """SHA-256 of the event body (must NOT include the event_hash field itself)."""
    return hashlib.sha256(_canonical_json(event_body).encode()).hexdigest()


def make_observation_key(
    model_version: str,
    model_config_hash: str,
    intended_execution_session: str,
) -> str:
    payload = f"{model_version}|{model_config_hash}|{intended_execution_session}"
    return hashlib.sha256(payload.encode()).hexdigest()


def make_content_hash(canonical_payload: dict) -> str:
    """SHA-256 of canonical JSON of the observation payload. No NaN/Inf permitted."""
    return hashlib.sha256(_canonical_json(canonical_payload).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return pd.Timestamp.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_session_date(s: str) -> None:
    if not isinstance(s, str) or not _DATE_RE.match(s):
        raise ValidationError(
            f"intended_execution_session must be YYYY-MM-DD, got: {s!r}"
        )
    try:
        pd.Timestamp(s)
    except Exception:
        raise ValidationError(f"Invalid date: {s!r}")
    # Validate as a real NYSE session
    from modules.exchange_calendar import CalendarUnavailableError, is_trading_session

    try:
        if not is_trading_session(s):
            raise ValidationError(
                f"{s!r} is not a NYSE trading session (weekend, holiday, or invalid date)"
            )
    except CalendarUnavailableError as exc:
        raise ValidationError(f"Cannot validate NYSE session for {s!r}: {exc}") from exc


def _validate_hex64(value: str, field: str) -> None:
    if not isinstance(value, str) or not _HEX64_RE.match(value):
        raise ValidationError(
            f"{field} must be a 64-character lowercase SHA-256 hex string, got: {value!r}"
        )


def _validate_finite_fraction(value: object, field: str) -> None:
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be a number, got: {type(value).__name__}")
    if not math.isfinite(float(value)):
        raise ValidationError(f"{field} must be a finite number (no NaN/Inf), got: {value!r}")
    fv = float(value)
    if not (0.0 <= fv <= 1.0):
        raise ValidationError(f"{field} must be in [0, 1], got: {fv!r}")


def _validate_nonneg_int(value: object, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"{field} must be a non-negative integer, got: {type(value).__name__}")
    if value < 0:
        raise ValidationError(f"{field} must be >= 0, got: {value!r}")


def _validate_finite_score(value: object, field: str) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be a finite number or None, got: {type(value).__name__}")
    if not math.isfinite(float(value)):
        raise ValidationError(f"{field} must be finite (no NaN/Inf), got: {value!r}")


def _validate_ticker_records(records: list[dict]) -> None:
    if not isinstance(records, list):
        raise ValidationError("ticker_records must be a list")
    seen: set[str] = set()
    for i, rec in enumerate(records):
        ticker = rec.get("ticker")
        if not ticker or not isinstance(ticker, str):
            raise ValidationError(f"ticker_records[{i}].ticker must be a non-empty string")
        if ticker in seen:
            raise ValidationError(f"Duplicate ticker in ticker_records: {ticker!r}")
        seen.add(ticker)
        # Validate scores
        scores = rec.get("scores", {})
        for factor in ("momentum", "quality", "value", "safety", "composite"):
            _validate_finite_score(scores.get(factor), f"ticker {ticker!r} scores.{factor}")
        # Validate factor_coverage
        fc = rec.get("factor_coverage")
        if fc is not None:
            if not isinstance(fc, (int, float)) or not math.isfinite(float(fc)):
                raise ValidationError(
                    f"ticker {ticker!r} factor_coverage must be finite or None, got: {fc!r}"
                )
            if not 0.0 <= float(fc) <= 1.0:
                raise ValidationError(
                    f"ticker {ticker!r} factor_coverage must be in [0, 1], got: {fc!r}"
                )
        # Cross-validate excluded + exclusion_reason
        excluded = rec.get("excluded", False)
        reason = rec.get("exclusion_reason")
        if excluded and not reason:
            raise ValidationError(
                f"ticker {ticker!r}: excluded=True requires a non-empty exclusion_reason"
            )
        if not excluded and reason is not None:
            raise ValidationError(
                f"ticker {ticker!r}: exclusion_reason must be None when excluded=False"
            )
        # Validate provenance types
        for p in rec.get("provenance", []):
            ptype = p.get("provenance_type") or p.get("type")
            if ptype not in _VALID_PROVENANCE_TYPES:
                raise ValidationError(
                    f"ticker {ticker!r}: invalid provenance type {ptype!r}. "
                    f"Must be one of {sorted(_VALID_PROVENANCE_TYPES)}"
                )
        # selected_by_strategy must be a list if present
        sbs = rec.get("selected_by_strategy")
        if sbs is not None and not isinstance(sbs, list):
            raise ValidationError(
                f"ticker {ticker!r}: selected_by_strategy must be a list or None"
            )


def _validate_selected_tickers_per_strategy(value: object) -> None:
    if not isinstance(value, dict):
        raise ValidationError("selected_tickers_per_strategy must be a dict")
    for strategy, tickers in value.items():
        if not isinstance(tickers, list):
            raise ValidationError(
                f"selected_tickers_per_strategy[{strategy!r}] must be a list"
            )


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def provenance_entry(
    provenance_type: ProvenanceType,
    source: str,
    *,
    source_timestamp: str | None = None,
    data_cutoff_at: str | None = None,
    as_of_date: str | None = None,
    note: str | None = None,
) -> dict:
    if provenance_type not in _VALID_PROVENANCE_TYPES:
        raise ValidationError(
            f"Invalid provenance_type {provenance_type!r}. "
            f"Must be one of {sorted(_VALID_PROVENANCE_TYPES)}"
        )
    entry: dict = {"provenance_type": provenance_type, "source": source}
    if source_timestamp is not None:
        entry["source_timestamp"] = source_timestamp
    if data_cutoff_at is not None:
        entry["data_cutoff_at"] = data_cutoff_at
    if as_of_date is not None:
        entry["as_of_date"] = as_of_date
    if note is not None:
        entry["note"] = note
    return entry


# ---------------------------------------------------------------------------
# Schema constructors
# ---------------------------------------------------------------------------


def make_ticker_record(
    ticker: str,
    *,
    raw_factor_inputs: dict | None = None,
    raw_factor_unavailable_reasons: dict[str, str] | None = None,
    factor_availability: dict[str, bool] | None = None,
    scores: dict[str, float | None] | None = None,
    factor_coverage: float | None = None,
    rank: int | None = None,
    excluded: bool = False,
    exclusion_reason: str | None = None,
    selected_by_strategy: list[str] | None = None,
    sector: str | None = None,
    value_sector_adjusted: bool = False,
    provenance: list[dict] | None = None,
) -> dict:
    """
    Construct a per-ticker observation record.

    raw_factor_inputs: dict of factor name → value (None = unavailable).
    raw_factor_unavailable_reasons: dict of factor name → reason string,
        required for every None entry in raw_factor_inputs.
    factor_availability: dict of factor group name → bool.
    scores: dict of {"momentum", "quality", "value", "safety", "composite"} → float|None.
    selected_by_strategy: list of strategy names that selected this ticker.
    provenance: list of provenance_entry() dicts (field-level provenance).
    """
    if not ticker or not isinstance(ticker, str):
        raise ValidationError("ticker must be a non-empty string")

    # Cross-validate excluded / exclusion_reason
    if excluded and not exclusion_reason:
        raise ValidationError(
            f"ticker {ticker!r}: excluded=True requires a non-empty exclusion_reason"
        )
    if not excluded and exclusion_reason is not None:
        raise ValidationError(
            f"ticker {ticker!r}: exclusion_reason must be None when excluded=False"
        )

    raw = raw_factor_inputs or {}
    reasons = raw_factor_unavailable_reasons or {}

    # Enforce: every None raw input must have a reason
    for fname, fval in raw.items():
        if fval is None and fname not in reasons:
            raise ValidationError(
                f"ticker {ticker!r}: raw_factor_inputs[{fname!r}] is None "
                f"but has no entry in raw_factor_unavailable_reasons"
            )
        if fval is not None:
            if not isinstance(fval, (int, float)):
                raise ValidationError(
                    f"ticker {ticker!r}: raw_factor_inputs[{fname!r}] must be numeric or None"
                )
            if not math.isfinite(float(fval)):
                raise ValidationError(
                    f"ticker {ticker!r}: raw_factor_inputs[{fname!r}] must be finite (no NaN/Inf)"
                )

    scores_clean = scores or {}
    for factor, val in scores_clean.items():
        if val is not None:
            if not isinstance(val, (int, float)):
                raise ValidationError(
                    f"ticker {ticker!r}: scores.{factor} must be a finite number or None, "
                    f"got {type(val).__name__}"
                )
            if not math.isfinite(float(val)):
                raise ValidationError(
                    f"ticker {ticker!r}: scores.{factor} must be finite (no NaN/Inf), got {val!r}"
                )

    return {
        "ticker": ticker,
        "raw_factor_inputs": raw,
        "raw_factor_unavailable_reasons": reasons,
        "factor_availability": factor_availability or {},
        "scores": {
            "momentum": scores_clean.get("momentum"),
            "quality": scores_clean.get("quality"),
            "value": scores_clean.get("value"),
            "safety": scores_clean.get("safety"),
            "composite": scores_clean.get("composite"),
        },
        "factor_coverage": factor_coverage,
        "rank": rank,
        "excluded": excluded,
        "exclusion_reason": exclusion_reason,
        "selected_by_strategy": selected_by_strategy or [],
        "sector": sector,
        "value_sector_adjusted": value_sector_adjusted,
        "provenance": provenance or [],
    }


# ---------------------------------------------------------------------------
# File path helpers (all use module-level LEDGER_DIR)
# ---------------------------------------------------------------------------


def _ledger_path(yyyymm: str) -> Path:
    return LEDGER_DIR / f"{yyyymm}_v2b_observations.jsonl"


def _lock_path(yyyymm: str) -> Path:
    return LEDGER_DIR / f"{yyyymm}_v2b_observations.lock"


def _idx_path() -> Path:
    return LEDGER_DIR / "v2b_idx.json"


def _idx_lock_path() -> Path:
    return LEDGER_DIR / "v2b_idx.lock"


def _yyyymm_for(session_date: str) -> str:
    return session_date[:7]


# ---------------------------------------------------------------------------
# Directory fsync (durable directory entry after file create/rename)
# ---------------------------------------------------------------------------


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Lock context managers
# ---------------------------------------------------------------------------


@contextmanager
def _monthly_lock(yyyymm: str) -> Generator[None, None, None]:
    lp = _lock_path(yyyymm)
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.touch(exist_ok=True)
    with open(lp, "r") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


@contextmanager
def _idx_lock_ctx() -> Generator[None, None, None]:
    lp = _idx_lock_path()
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.touch(exist_ok=True)
    with open(lp, "r") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Safe JSONL reader with hash and chain verification
# ---------------------------------------------------------------------------


def _read_events_raw(path: Path, verify_integrity: bool = True) -> list[dict]:
    """
    Read all events from a JSONL file.

    - Incomplete last line → warn + skip (crash-safe truncation)
    - Mid-file JSON error → CorruptionError (fail-closed)
    - event_hash mismatch → CorruptionError
    - Broken previous_event_hash chain → CorruptionError
    """
    if not path.exists():
        return []

    content = path.read_text()
    all_lines = content.splitlines()
    events: list[dict] = []

    for i, raw_line in enumerate(all_lines):
        line = raw_line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            is_last_nonempty = all(not l.strip() for l in all_lines[i + 1 :])
            if is_last_nonempty:
                warnings.warn(
                    f"v2b_ledger: incomplete last line in {path.name} (line {i + 1}) — skipped",
                    UserWarning,
                    stacklevel=4,
                )
                continue
            raise CorruptionError(
                f"v2b_ledger: mid-file JSON parse failure at line {i + 1} in {path.name}"
            )
        if verify_integrity and "event_hash" in ev:
            body = {k: v for k, v in ev.items() if k != "event_hash"}
            computed = _compute_event_hash(body)
            if computed != ev["event_hash"]:
                raise CorruptionError(
                    f"v2b_ledger: event_hash mismatch at line {i + 1} in {path.name}: "
                    f"stored={ev['event_hash'][:16]}… computed={computed[:16]}…"
                )
        events.append(ev)

    if verify_integrity:
        _verify_hash_chains(events, path.name)

    return events


def _verify_hash_chains(events: list[dict], filename: str = "") -> None:
    """Verify that previous_event_hash chains are unbroken for each observation key."""
    by_key: dict[str, list[dict]] = {}
    for ev in events:
        key = ev.get("observation_key")
        if key:
            by_key.setdefault(key, []).append(ev)

    for key, chain in by_key.items():
        prev_hash: str | None = None
        for ev in chain:
            stored_prev = ev.get("previous_event_hash")
            # Only verify chain if the event has previous_event_hash field
            if "previous_event_hash" in ev:
                if stored_prev != prev_hash:
                    raise CorruptionError(
                        f"v2b_ledger: hash chain broken for {key[:16]}… in {filename}: "
                        f"expected previous_event_hash={prev_hash}, got={stored_prev}"
                    )
            if "event_hash" in ev:
                prev_hash = ev["event_hash"]


# ---------------------------------------------------------------------------
# Append helper (caller must hold the monthly lock)
# ---------------------------------------------------------------------------


def _append_line(path: Path, event: dict) -> None:
    """Append one event line to JSONL. Caller MUST hold the monthly lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _store_json(event) + "\n"
    is_new_file = not path.exists()
    with open(path, "a") as af:
        af.write(line)
        af.flush()
        os.fsync(af.fileno())
    if is_new_file:
        _fsync_dir(path.parent)


# ---------------------------------------------------------------------------
# Index I/O
# ---------------------------------------------------------------------------


def _load_idx_raw() -> dict[str, str]:
    """Load index without rebuilding. Returns {} on any error."""
    p = _idx_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _rebuild_idx() -> dict[str, str]:
    """
    Rebuild the index by scanning all JSONL files.
    Raises CorruptionError if any file has mid-file corruption.
    """
    idx: dict[str, str] = {}
    for jsonl_path in sorted(LEDGER_DIR.glob("*_v2b_observations.jsonl")):
        name = jsonl_path.name
        # Extract YYYY-MM from "{YYYY-MM}_v2b_observations.jsonl"
        yyyymm = name[:7]
        events = _read_events_raw(jsonl_path)
        for ev in events:
            if ev.get("event_type") == "OBSERVATION_CREATED":
                key = ev.get("observation_key")
                if key:
                    idx[key] = yyyymm
    return idx


def _load_idx_or_rebuild() -> dict[str, str]:
    """
    Load index. If corrupt or missing but JSONL files exist, rebuild.
    A corrupt index must never make a valid observation appear absent.
    """
    p = _idx_path()
    if p.exists():
        try:
            data = json.loads(p.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        # Index is corrupt — rebuild from JSONL
        return _rebuild_idx()
    # No index file — rebuild if JSONL files exist
    if list(LEDGER_DIR.glob("*_v2b_observations.jsonl")):
        return _rebuild_idx()
    return {}


def _save_idx_atomic(idx: dict[str, str]) -> None:
    """Atomically save index: write to temp, fsync, rename, fsync dir."""
    p = _idx_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(idx, sort_keys=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=p.parent, prefix=".idx_")
    try:
        with os.fdopen(tmp_fd, "w") as tf:
            tf.write(content)
            tf.flush()
            os.fsync(tf.fileno())
        Path(tmp_name).replace(p)
        _fsync_dir(p.parent)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _update_idx_locked(observation_key: str, yyyymm: str) -> None:
    """Thread/process-safe index update: exclusive lock → read → add → save."""
    with _idx_lock_ctx():
        idx = _load_idx_raw()
        idx[observation_key] = yyyymm
        _save_idx_atomic(idx)


# ---------------------------------------------------------------------------
# Event construction helpers (inside lock, so we have latest prev_hash)
# ---------------------------------------------------------------------------


def _build_event(
    event_type: str,
    observation_key: str,
    run_id: str,
    extra: dict,
    prev_hash: str | None,
) -> dict:
    body: dict = {
        "event_type": event_type,
        "record_version": RECORD_VERSION,
        "observation_key": observation_key,
        "observation_run_id": run_id,
        "timestamp_utc": _now_iso(),
        "order_creation_blocked": _ORDER_CREATION_BLOCKED,
        "previous_event_hash": prev_hash,
        **extra,
    }
    event_hash = _compute_event_hash(body)
    return {**body, "event_hash": event_hash}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_events_for_key(observation_key: str, events: list[dict]) -> list[dict]:
    return [e for e in events if e.get("observation_key") == observation_key]


def _derive_status(events: list[dict]) -> ObservationStatus:
    if not events:
        raise ValueError("Cannot derive status from empty event list")
    current: str = "CREATED"
    for ev in events:
        et = ev.get("event_type", "")
        if et in _STATUS_FROM_EVENT and et != "IDEMPOTENT_MATCH":
            current = _STATUS_FROM_EVENT[et]
    return current  # type: ignore[return-value]


def _last_event_hash(key_events: list[dict]) -> str | None:
    for ev in reversed(key_events):
        if "event_hash" in ev:
            return ev["event_hash"]
    return None


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------


def create_observation(
    *,
    model_version: str,
    model_config_hash: str,
    intended_execution_session: str,
    universe_count: int,
    valid_ticker_count: int,
    stale_ticker_count: int,
    excluded_ticker_count: int,
    missing_ticker_count: int,
    signal_coverage_rate: float,
    selected_tickers_per_strategy: dict[str, list[str]],
    data_quality_status: str,
    ticker_records: list[dict],
    metadata: dict | None = None,
) -> dict:
    """
    Open a new shadow observation. Returns the created event dict.

    Idempotency:
      - Same key + same content_hash  → returns IDEMPOTENT_MATCH (no write)
      - Same key + different hash     → writes CONFLICT event, raises ConflictError

    All reads and the write decision happen under an exclusive monthly lock,
    making create_observation linearizable.
    """
    # ── Input validation ──────────────────────────────────────────────────
    if not model_version or not isinstance(model_version, str):
        raise ValidationError("model_version must be a non-empty string")
    _validate_hex64(model_config_hash, "model_config_hash")
    _validate_session_date(intended_execution_session)
    _validate_nonneg_int(universe_count, "universe_count")
    _validate_nonneg_int(valid_ticker_count, "valid_ticker_count")
    _validate_nonneg_int(stale_ticker_count, "stale_ticker_count")
    _validate_nonneg_int(excluded_ticker_count, "excluded_ticker_count")
    _validate_nonneg_int(missing_ticker_count, "missing_ticker_count")
    _validate_finite_fraction(signal_coverage_rate, "signal_coverage_rate")
    _validate_selected_tickers_per_strategy(selected_tickers_per_strategy)
    if not isinstance(data_quality_status, str) or not data_quality_status:
        raise ValidationError("data_quality_status must be a non-empty string")
    _validate_ticker_records(ticker_records)

    # ── Key and content hash ──────────────────────────────────────────────
    observation_key = make_observation_key(
        model_version, model_config_hash, intended_execution_session
    )
    yyyymm = _yyyymm_for(intended_execution_session)
    path = _ledger_path(yyyymm)

    canonical = {
        "model_version": model_version,
        "model_config_hash": model_config_hash,
        "intended_execution_session": intended_execution_session,
        "universe_count": universe_count,
        "valid_ticker_count": valid_ticker_count,
        "stale_ticker_count": stale_ticker_count,
        "excluded_ticker_count": excluded_ticker_count,
        "missing_ticker_count": missing_ticker_count,
        "signal_coverage_rate": signal_coverage_rate,
        "selected_tickers_per_strategy": selected_tickers_per_strategy,
        "data_quality_status": data_quality_status,
        "ticker_records": ticker_records,
        "metadata": metadata or {},
    }
    content_hash = make_content_hash(canonical)

    # ── Atomically read-then-write under the monthly lock ─────────────────
    with _monthly_lock(yyyymm):
        all_events = _read_events_raw(path)
        key_events = _find_events_for_key(observation_key, all_events)

        if key_events:
            prior_creation = key_events[0]
            prior_hash = prior_creation.get("content_hash")

            if prior_hash == content_hash:
                # Idempotent — do not write
                return {
                    "event_type": "IDEMPOTENT_MATCH",
                    "record_version": RECORD_VERSION,
                    "observation_key": observation_key,
                    "observation_run_id": prior_creation.get("observation_run_id"),
                    "content_hash": content_hash,
                    "timestamp_utc": _now_iso(),
                    "order_creation_blocked": _ORDER_CREATION_BLOCKED,
                }

            # Conflict — write CONFLICT event then raise
            conflict_run_id = str(uuid.uuid4())
            prev_hash = _last_event_hash(key_events)
            conflict_ev = _build_event(
                "CONFLICT",
                observation_key,
                conflict_run_id,
                {
                    "prior_content_hash": prior_hash,
                    "new_content_hash": content_hash,
                },
                prev_hash,
            )
            _append_line(path, conflict_ev)
            raise ConflictError(
                f"observation_key {observation_key[:16]}… already exists with different "
                f"content. Prior={prior_hash[:16] if prior_hash else 'None'}… "
                f"New={content_hash[:16]}…"
            )

        # New observation
        run_id = str(uuid.uuid4())
        event_body_extra = {
            "ledger_version": LEDGER_VERSION,
            "content_hash": content_hash,
            "model_version": model_version,
            "model_config_hash": model_config_hash,
            "intended_execution_session": intended_execution_session,
            "universe_count": universe_count,
            "valid_ticker_count": valid_ticker_count,
            "stale_ticker_count": stale_ticker_count,
            "excluded_ticker_count": excluded_ticker_count,
            "missing_ticker_count": missing_ticker_count,
            "signal_coverage_rate": signal_coverage_rate,
            "selected_tickers_per_strategy": selected_tickers_per_strategy,
            "data_quality_status": data_quality_status,
            "ticker_records": ticker_records,
            "metadata": metadata or {},
            # point_in_time_fundamentals_global is NEVER True.
            # Per-ticker provenance tracks which fields are genuinely PIT.
            "point_in_time_fundamentals_global": False,
        }
        event = _build_event(
            "OBSERVATION_CREATED",
            observation_key,
            run_id,
            event_body_extra,
            None,  # First event — no previous_event_hash
        )
        _append_line(path, event)

    # ── Update index AFTER releasing the monthly lock ─────────────────────
    _update_idx_locked(observation_key, yyyymm)

    return event


def transition_observation(
    observation_key: str,
    new_status: ObservationStatus,
    note: str | None = None,
) -> dict:
    """
    Append a status-transition event. Status is read and transition is validated
    under the exclusive monthly lock — no race between concurrent transitions.
    """
    idx = _load_idx_or_rebuild()
    yyyymm = idx.get(observation_key)
    if yyyymm is None:
        raise KeyError(f"observation_key not found: {observation_key[:16]}…")

    path = _ledger_path(yyyymm)

    with _monthly_lock(yyyymm):
        all_events = _read_events_raw(path)
        key_events = _find_events_for_key(observation_key, all_events)

        if not key_events:
            raise KeyError(f"No events found for key: {observation_key[:16]}…")

        current_status = _derive_status(key_events)
        allowed = _TRANSITIONS.get(current_status, set())
        if new_status not in allowed:
            raise InvalidTransitionError(
                f"Cannot transition from {current_status} → {new_status}. "
                f"Allowed: {sorted(allowed) or 'none (terminal state)'}"
            )

        run_id = key_events[0].get("observation_run_id")
        prev_hash = _last_event_hash(key_events)

        extra: dict = {}
        if note:
            extra["note"] = note

        event = _build_event(new_status, observation_key, run_id, extra, prev_hash)
        _append_line(path, event)

    return event


def get_observation_status(observation_key: str) -> ObservationStatus | None:
    """Return current status, rebuilding index if corrupt. Returns None if not found."""
    idx = _load_idx_or_rebuild()
    yyyymm = idx.get(observation_key)
    if yyyymm is None:
        return None
    path = _ledger_path(yyyymm)
    all_events = _read_events_raw(path)
    key_events = _find_events_for_key(observation_key, all_events)
    if not key_events:
        return None
    return _derive_status(key_events)


def get_observation_events(observation_key: str) -> list[dict]:
    """Return all events for a key (in append order). Verifies hash integrity."""
    idx = _load_idx_or_rebuild()
    yyyymm = idx.get(observation_key)
    if yyyymm is None:
        return []
    path = _ledger_path(yyyymm)
    all_events = _read_events_raw(path)
    return _find_events_for_key(observation_key, all_events)


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
    for p in files:
        if not p.exists():
            continue
        for ev in _read_events_raw(p):
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
# Structural invariant assertion
# ---------------------------------------------------------------------------


def assert_order_creation_blocked() -> None:
    """Raises AssertionError if the structural invariant is violated."""
    assert _ORDER_CREATION_BLOCKED is True, (
        "V2B ledger structural invariant violated: order_creation_blocked must always be True"
    )
