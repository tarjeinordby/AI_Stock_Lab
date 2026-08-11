"""
V2B Shadow Observation Ledger — immutable append-only event store.

Stores prospective shadow observations of the V2 model scoring process.
Cannot create orders, fills, trades, or touch any V1 state.

File layout (all paths relative to this file's repo root):
  data_v4/v2b_ledger/{YYYY-MM}_v2b_observations.jsonl
  data_v4/v2b_ledger/{YYYY-MM}_v2b_observations.lock
  data_v4/v2b_ledger/v2b_idx.json       (observation_key → YYYY-MM, rebuildable cache)
  data_v4/v2b_ledger/v2b_idx.lock

observation_key = SHA-256(model_version + "|" + model_config_hash + "|" + intended_execution_session)

Each event carries:
  event_hash           = SHA-256(canonical JSON of event WITHOUT event_hash field)
  previous_event_hash  = event_hash of prior event for same observation_key (None for first)

This forms a per-observation tamper-evident chain.
Both fields are MANDATORY for record_version="2". Missing, wrong-type, wrong-length,
or mismatched values raise CorruptionError. Unsupported record_version is fail-closed.

STATUS MACHINE:
  CREATED → COLLECTING → COMPLETED
                       → FAILED_DATA
                       → FAILED_VALIDATION
                       → CANCELLED
  CREATED → CONFLICT   (same key, different content_hash — raises ConflictError)

LOCK ORDERING (deadlock prevention):
  Rule: monthly lock is ALWAYS acquired and FULLY RELEASED before the index lock is
  acquired. Never hold both simultaneously. Violations are a bug, not a policy choice.

  Correct sequence for create_observation:
    1. acquire monthly lock (per YYYY-MM partition)
    2. read / check / write JSONL event
    3. release monthly lock
    4. acquire index lock
    5. rebuild-or-read index, add new entry, save atomically
    6. release index lock

  Correct sequence for transition_observation:
    1. read index without any lock (safe: reads are atomic at the OS page level)
    2. acquire monthly lock
    3. read / check / write JSONL event
    4. release monthly lock
    (no index update needed for transitions)

  Crash recovery (index gap after JSONL write):
    get_observation_status / get_observation_events scan JSONL files when a key is
    absent from the index, then repair the index entry under the index lock.
    create_observation (idempotent path) always repairs the index before returning.

INDEX CONTRACT:
  v2b_idx.json is a REBUILDABLE CACHE, not the source of truth. The JSONL files
  are always authoritative. An index mapping is only trusted if the mapped partition
  actually contains the observation_key. Any mapping that fails this check (absent,
  invalid format, pointing to a non-existent or wrong-month file) triggers a full
  JSONL scan and atomic index repair.

  _update_idx_locked always rebuilds from JSONL when the index is corrupt (so a
  new create never replaces a corrupt index with only its own key).

  If an observation_key is found in more than one partition during a scan,
  CorruptionError is raised immediately (fail-closed).

RUN-LEVEL COUNTS SEMANTICS:
  universe_count           — total tickers in the scoring universe before any filtering
  valid_ticker_count       — tickers with enough price/fundamental data to score
  stale_ticker_count       — tickers with data too old to trust (still in universe)
  excluded_ticker_count    — tickers excluded by portfolio rules (e.g. below SMA200)
  missing_ticker_count     — tickers in universe with no usable data at all

  WARNING: these categories can overlap (e.g. a ticker can be both valid and excluded).
  Do NOT assume valid + stale + excluded + missing == universe_count.

  ticker_records_represents_all_scored:
    True  — ticker_records contains every ticker that was scored (complete universe)
    False — ticker_records is a subset (e.g. only top-N candidates); cross-validation
            of selected_tickers_per_strategy against records is skipped

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
# Constants — LEDGER_DIR anchored to repo root via __file__
# ---------------------------------------------------------------------------

LEDGER_VERSION = "1"
RECORD_VERSION = "2"  # v2: hash chain, strict field validation

# Anchor to repo root (modules/ lives one level below repo root)
LEDGER_DIR: Path = Path(__file__).parent.parent / "data_v4" / "v2b_ledger"

# Structural invariant — hardcoded, never overridable by caller
_ORDER_CREATION_BLOCKED: bool = True

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_YYYYMM_RE = re.compile(r"^\d{4}-\d{2}$")

# record_version values this code can read. Anything else is fail-closed.
_SUPPORTED_RECORD_VERSIONS: frozenset[str] = frozenset({"2"})

# All fields required in every ticker record passed to create_observation.
_REQUIRED_TICKER_FIELDS: frozenset[str] = frozenset({
    "ticker",
    "raw_factor_inputs",
    "raw_factor_unavailable_reasons",
    "factor_availability",
    "scores",
    "factor_coverage",
    "rank",
    "excluded",
    "exclusion_reason",
    "selected_by_strategy",
    "sector",
    "value_sector_adjusted",
    "provenance",
})

# All factor keys required under scores.
_REQUIRED_SCORE_KEYS: frozenset[str] = frozenset(
    {"momentum", "quality", "value", "safety", "composite"}
)

ObservationStatus = Literal[
    "CREATED",
    "COLLECTING",
    "COMPLETED",
    "FAILED_DATA",
    "FAILED_VALIDATION",
    "CANCELLED",
    "CONFLICT",
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
    """Raised on hash mismatch, broken chain, missing mandatory field, or bad record_version."""


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
    """JSON for on-disk storage (same strict rules)."""
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
    """SHA-256 of canonical JSON of the observation payload. NaN/Inf raise ValueError."""
    return hashlib.sha256(_canonical_json(canonical_payload).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return pd.Timestamp.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Input validation helpers
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
    from modules.exchange_calendar import CalendarUnavailableError, is_trading_session
    try:
        if not is_trading_session(s):
            raise ValidationError(
                f"{s!r} is not a NYSE trading session (weekend, holiday, or invalid date)"
            )
    except CalendarUnavailableError as exc:
        raise ValidationError(f"Cannot validate NYSE session for {s!r}: {exc}") from exc


def _validate_hex64(value: object, field: str) -> None:
    if not isinstance(value, str) or not _HEX64_RE.match(value):
        raise ValidationError(
            f"{field} must be a 64-character lowercase SHA-256 hex string, got: {value!r}"
        )


def _validate_finite_fraction(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be a number (not bool), got: {type(value).__name__}")
    if not math.isfinite(float(value)):
        raise ValidationError(f"{field} must be finite (no NaN/Inf), got: {value!r}")
    fv = float(value)
    if not (0.0 <= fv <= 1.0):
        raise ValidationError(f"{field} must be in [0, 1], got: {fv!r}")


def _validate_nonneg_int(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field} must be a non-negative integer, got: {type(value).__name__}")
    if value < 0:
        raise ValidationError(f"{field} must be >= 0, got: {value!r}")


def _validate_ticker_records(records: list[dict]) -> None:
    if not isinstance(records, list):
        raise ValidationError("ticker_records must be a list")
    seen: set[str] = set()
    for i, rec in enumerate(records):
        _validate_single_ticker_record(rec, i, seen)
        seen.add(rec["ticker"])


def _validate_single_ticker_record(rec: dict, idx: int, seen_tickers: set[str]) -> None:
    # --- Required fields presence ---
    missing = _REQUIRED_TICKER_FIELDS - set(rec.keys())
    if missing:
        label = rec.get("ticker", f"index {idx}")
        raise ValidationError(
            f"ticker_records[{idx}] ({label!r}): missing required fields: {sorted(missing)}"
        )

    ticker = rec["ticker"]
    if not ticker or not isinstance(ticker, str):
        raise ValidationError(f"ticker_records[{idx}].ticker must be a non-empty string")
    if ticker in seen_tickers:
        raise ValidationError(f"Duplicate ticker in ticker_records: {ticker!r}")

    # --- raw_factor_inputs ---
    rfi = rec["raw_factor_inputs"]
    if not isinstance(rfi, dict):
        raise ValidationError(f"ticker {ticker!r}: raw_factor_inputs must be a dict")
    rfur = rec["raw_factor_unavailable_reasons"]
    if not isinstance(rfur, dict):
        raise ValidationError(f"ticker {ticker!r}: raw_factor_unavailable_reasons must be a dict")
    for fname, fval in rfi.items():
        if fval is None:
            if fname not in rfur:
                raise ValidationError(
                    f"ticker {ticker!r}: raw_factor_inputs[{fname!r}]=None "
                    f"requires an entry in raw_factor_unavailable_reasons"
                )
        else:
            if isinstance(fval, bool):
                raise ValidationError(
                    f"ticker {ticker!r}: raw_factor_inputs[{fname!r}] must not be bool"
                )
            if not isinstance(fval, (int, float)):
                raise ValidationError(
                    f"ticker {ticker!r}: raw_factor_inputs[{fname!r}] must be numeric or None"
                )
            if not math.isfinite(float(fval)):
                raise ValidationError(
                    f"ticker {ticker!r}: raw_factor_inputs[{fname!r}] must be finite (no NaN/Inf)"
                )

    # --- factor_availability ---
    fa = rec["factor_availability"]
    if not isinstance(fa, dict):
        raise ValidationError(f"ticker {ticker!r}: factor_availability must be a dict")
    for k, v in fa.items():
        if not isinstance(v, bool):
            raise ValidationError(
                f"ticker {ticker!r}: factor_availability[{k!r}] must be bool, got {type(v).__name__}"
            )

    # --- scores ---
    scores = rec["scores"]
    if not isinstance(scores, dict):
        raise ValidationError(f"ticker {ticker!r}: scores must be a dict")
    missing_score_keys = _REQUIRED_SCORE_KEYS - set(scores.keys())
    if missing_score_keys:
        raise ValidationError(
            f"ticker {ticker!r}: scores missing required keys: {sorted(missing_score_keys)}"
        )
    for sk in _REQUIRED_SCORE_KEYS:
        val = scores[sk]
        if val is not None:
            if isinstance(val, bool):
                raise ValidationError(
                    f"ticker {ticker!r}: scores.{sk} must not be bool"
                )
            if not isinstance(val, (int, float)):
                raise ValidationError(
                    f"ticker {ticker!r}: scores.{sk} must be numeric or None, got {type(val).__name__}"
                )
            if not math.isfinite(float(val)):
                raise ValidationError(
                    f"ticker {ticker!r}: scores.{sk} must be finite (no NaN/Inf), got {val!r}"
                )

    # --- factor_coverage ---
    fc = rec["factor_coverage"]
    if fc is not None:
        if isinstance(fc, bool):
            raise ValidationError(f"ticker {ticker!r}: factor_coverage must not be bool")
        if not isinstance(fc, (int, float)):
            raise ValidationError(f"ticker {ticker!r}: factor_coverage must be numeric or None")
        if not math.isfinite(float(fc)):
            raise ValidationError(
                f"ticker {ticker!r}: factor_coverage must be finite (no NaN/Inf), got {fc!r}"
            )
        if not 0.0 <= float(fc) <= 1.0:
            raise ValidationError(
                f"ticker {ticker!r}: factor_coverage must be in [0, 1], got {fc!r}"
            )

    # --- rank ---
    rank = rec["rank"]
    if rank is not None:
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise ValidationError(f"ticker {ticker!r}: rank must be a positive integer or None")
        if rank <= 0:
            raise ValidationError(f"ticker {ticker!r}: rank must be > 0, got {rank!r}")

    # --- excluded / exclusion_reason ---
    excluded = rec["excluded"]
    if not isinstance(excluded, bool):
        raise ValidationError(f"ticker {ticker!r}: excluded must be bool")
    reason = rec["exclusion_reason"]
    if excluded and not reason:
        raise ValidationError(f"ticker {ticker!r}: excluded=True requires a non-empty exclusion_reason")
    if not excluded and reason is not None:
        raise ValidationError(f"ticker {ticker!r}: exclusion_reason must be None when excluded=False")

    # --- selected_by_strategy ---
    sbs = rec["selected_by_strategy"]
    if not isinstance(sbs, list):
        raise ValidationError(f"ticker {ticker!r}: selected_by_strategy must be a list")
    for s in sbs:
        if not isinstance(s, str) or not s:
            raise ValidationError(
                f"ticker {ticker!r}: selected_by_strategy items must be non-empty strings"
            )

    # --- sector ---
    sector = rec["sector"]
    if sector is not None and not isinstance(sector, str):
        raise ValidationError(f"ticker {ticker!r}: sector must be a string or None")

    # --- value_sector_adjusted ---
    if not isinstance(rec["value_sector_adjusted"], bool):
        raise ValidationError(f"ticker {ticker!r}: value_sector_adjusted must be bool")

    # --- provenance ---
    prov = rec["provenance"]
    if not isinstance(prov, list):
        raise ValidationError(f"ticker {ticker!r}: provenance must be a list")
    for j, p_entry in enumerate(prov):
        if not isinstance(p_entry, dict):
            raise ValidationError(f"ticker {ticker!r}: provenance[{j}] must be a dict")
        ptype = p_entry.get("provenance_type") or p_entry.get("type")
        if ptype not in _VALID_PROVENANCE_TYPES:
            raise ValidationError(
                f"ticker {ticker!r}: provenance[{j}].provenance_type={ptype!r} "
                f"must be one of {sorted(_VALID_PROVENANCE_TYPES)}"
            )
        source = p_entry.get("source")
        if not source or not isinstance(source, str):
            raise ValidationError(
                f"ticker {ticker!r}: provenance[{j}].source must be a non-empty string"
            )


def _validate_selected_tickers_per_strategy(value: object) -> None:
    if not isinstance(value, dict):
        raise ValidationError("selected_tickers_per_strategy must be a dict")
    for strategy, tickers in value.items():
        if not isinstance(tickers, list):
            raise ValidationError(
                f"selected_tickers_per_strategy[{strategy!r}] must be a list"
            )
        for t in tickers:
            if not isinstance(t, str) or not t:
                raise ValidationError(
                    f"selected_tickers_per_strategy[{strategy!r}] items must be non-empty strings"
                )


def _validate_selected_tickers_vs_records(
    selected_tickers_per_strategy: dict[str, list[str]],
    ticker_records: list[dict],
) -> None:
    """
    Cross-validate: every ticker in selected_tickers_per_strategy must appear
    in ticker_records. Only called when ticker_records_represents_all_scored=True.
    """
    all_tickers = {r["ticker"] for r in ticker_records if "ticker" in r}
    for strategy, tickers in selected_tickers_per_strategy.items():
        for t in tickers:
            if t not in all_tickers:
                raise ValidationError(
                    f"selected_tickers_per_strategy[{strategy!r}] references ticker "
                    f"{t!r} which is not present in ticker_records. "
                    f"Set ticker_records_represents_all_scored=False if records are a subset."
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
    """
    Construct a field-level provenance record.

    source_timestamp: ISO 8601+tz when this data was retrieved from the source.
    data_cutoff_at:   YYYY-MM-DD, last date included in this data (e.g. last trading day).
    as_of_date:       YYYY-MM-DD, effective date of the data for the scoring session.
    """
    if provenance_type not in _VALID_PROVENANCE_TYPES:
        raise ValidationError(
            f"Invalid provenance_type {provenance_type!r}. "
            f"Must be one of {sorted(_VALID_PROVENANCE_TYPES)}"
        )
    if not source or not isinstance(source, str):
        raise ValidationError("provenance source must be a non-empty string")
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
# Schema constructor
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
    Construct a per-ticker observation record that passes _validate_single_ticker_record.

    raw_factor_inputs: dict of factor name → value (None = unavailable).
    raw_factor_unavailable_reasons: for every None in raw_factor_inputs, a reason string.
    factor_availability: {"momentum": bool, "quality": bool, "value": bool, "safety": bool}.
    scores: {"momentum", "quality", "value", "safety", "composite"} → float | None.
    selected_by_strategy: list of strategy names that selected this ticker for ordering.
    provenance: list of provenance_entry() dicts with field-level data lineage.
    """
    if not ticker or not isinstance(ticker, str):
        raise ValidationError("ticker must be a non-empty string")

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

    for fname, fval in raw.items():
        if fval is None:
            if fname not in reasons:
                raise ValidationError(
                    f"ticker {ticker!r}: raw_factor_inputs[{fname!r}]=None "
                    f"requires an entry in raw_factor_unavailable_reasons"
                )
        else:
            if isinstance(fval, bool):
                raise ValidationError(
                    f"ticker {ticker!r}: raw_factor_inputs[{fname!r}] must not be bool"
                )
            if not isinstance(fval, (int, float)) or not math.isfinite(float(fval)):
                raise ValidationError(
                    f"ticker {ticker!r}: raw_factor_inputs[{fname!r}] must be finite or None, got {fval!r}"
                )

    scores_clean = scores or {}
    for factor, val in scores_clean.items():
        if val is not None:
            if isinstance(val, bool):
                raise ValidationError(
                    f"ticker {ticker!r}: scores.{factor} must not be bool"
                )
            if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                raise ValidationError(
                    f"ticker {ticker!r}: scores.{factor} must be finite or None, got {val!r}"
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
# File path helpers
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
# Directory fsync
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
# JSONL reader — mandatory hash fields, fail-closed on bad record_version
# ---------------------------------------------------------------------------


def _verify_event_structure(ev: dict, filename: str, line_num: int) -> None:
    """
    Verify structural integrity of a parsed event dict.

    record_version must be in _SUPPORTED_RECORD_VERSIONS (currently {"2"}).
    Missing or unsupported record_version is fail-closed (CorruptionError).

    For record_version="2":
      - observation_key must be a 64-char hex string
      - observation_run_id must be a non-empty string
      - event_hash must be a 64-char hex string (MANDATORY)
      - previous_event_hash must be present (MANDATORY): None for first event, 64-char hex for rest
    """
    rv = ev.get("record_version")
    if rv not in _SUPPORTED_RECORD_VERSIONS:
        raise CorruptionError(
            f"v2b_ledger: unsupported or missing record_version {rv!r} "
            f"at line {line_num} in {filename} — fail-closed"
        )

    key = ev.get("observation_key")
    if not isinstance(key, str) or not _HEX64_RE.match(key):
        raise CorruptionError(
            f"v2b_ledger: missing or invalid observation_key at line {line_num} in {filename}"
        )

    run_id = ev.get("observation_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise CorruptionError(
            f"v2b_ledger: missing or invalid observation_run_id at line {line_num} in {filename}"
        )

    # event_hash is MANDATORY
    event_hash = ev.get("event_hash")
    if not isinstance(event_hash, str) or not _HEX64_RE.match(event_hash):
        raise CorruptionError(
            f"v2b_ledger: missing or invalid event_hash at line {line_num} in {filename} "
            f"(field is mandatory for record_version='2')"
        )

    # previous_event_hash is MANDATORY as a field (value may be None for first event)
    if "previous_event_hash" not in ev:
        raise CorruptionError(
            f"v2b_ledger: missing previous_event_hash at line {line_num} in {filename} "
            f"(field is mandatory for record_version='2')"
        )
    peh = ev["previous_event_hash"]
    if peh is not None and (not isinstance(peh, str) or not _HEX64_RE.match(peh)):
        raise CorruptionError(
            f"v2b_ledger: invalid previous_event_hash at line {line_num} in {filename}: "
            f"must be None or a 64-char hex string, got {peh!r}"
        )


def _read_events_raw(path: Path) -> list[dict]:
    """
    Read all events from a JSONL file with full integrity verification.

    - Incomplete last line → UserWarning + skip (crash-safe truncation)
    - Mid-file JSON error → CorruptionError (fail-closed)
    - Unsupported/missing record_version → CorruptionError
    - Missing mandatory field (event_hash, previous_event_hash) → CorruptionError
    - event_hash value mismatch → CorruptionError
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
            is_last_nonempty = all(not ln.strip() for ln in all_lines[i + 1:])
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

        # Structural field verification (record_version, mandatory hash fields)
        _verify_event_structure(ev, path.name, i + 1)

        # event_hash value verification
        body = {k: v for k, v in ev.items() if k != "event_hash"}
        computed = _compute_event_hash(body)
        if computed != ev["event_hash"]:
            raise CorruptionError(
                f"v2b_ledger: event_hash mismatch at line {i + 1} in {path.name}: "
                f"stored={ev['event_hash'][:16]}… computed={computed[:16]}…"
            )

        events.append(ev)

    _verify_hash_chains(events, path.name)
    return events


def _verify_hash_chains(events: list[dict], filename: str = "") -> None:
    """
    Verify that previous_event_hash chains are unbroken for each observation key.
    All events are assumed to have already passed _verify_event_structure.
    """
    by_key: dict[str, list[dict]] = {}
    for ev in events:
        key = ev.get("observation_key")
        if key:
            by_key.setdefault(key, []).append(ev)

    for key, chain in by_key.items():
        prev_hash: str | None = None
        for ev in chain:
            stored_prev = ev["previous_event_hash"]  # field is mandatory; always present
            if stored_prev != prev_hash:
                raise CorruptionError(
                    f"v2b_ledger: hash chain broken for {key[:16]}… in {filename}: "
                    f"expected previous_event_hash={prev_hash!r}, got={stored_prev!r}"
                )
            prev_hash = ev["event_hash"]  # mandatory; always present


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


def _load_idx_raw_strict() -> dict[str, str] | None:
    """
    Load index without rebuilding.
    Returns {} if file does not exist (genuinely empty / first run).
    Returns None if file exists but is corrupt (parse error or wrong type).
    Never raises.
    """
    p = _idx_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            return data
        return None  # Valid JSON but wrong type
    except Exception:
        return None  # Parse failure


def _load_idx_raw() -> dict[str, str]:
    """Load index without rebuilding. Returns {} on any error (caller handles missing key)."""
    result = _load_idx_raw_strict()
    return result if result is not None else {}


def _rebuild_idx() -> dict[str, str]:
    """
    Rebuild the index by scanning all JSONL files.
    Raises CorruptionError if any file has mid-file corruption or broken hash chain.
    """
    idx: dict[str, str] = {}
    for jsonl_path in sorted(LEDGER_DIR.glob("*_v2b_observations.jsonl")):
        yyyymm = jsonl_path.name[:7]
        events = _read_events_raw(jsonl_path)
        for ev in events:
            if ev.get("event_type") == "OBSERVATION_CREATED":
                key = ev.get("observation_key")
                if key:
                    idx[key] = yyyymm
    return idx


def _load_idx_or_rebuild() -> dict[str, str]:
    """
    Load index. If corrupt, rebuild from JSONL (fail-closed on mid-file corruption).
    A corrupt index entry can never permanently hide a valid observation.
    """
    p = _idx_path()
    if p.exists():
        result = _load_idx_raw_strict()
        if result is not None:
            return result
        # Corrupt — rebuild
        return _rebuild_idx()
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
    """
    Thread/process-safe index update under exclusive idx_lock.

    If the index is corrupt, rebuilds from JSONL FIRST to preserve existing mappings,
    then adds the new entry. A corrupt index must never be replaced by only the new key.

    Lock ordering: NEVER call while holding a monthly lock.
    """
    with _idx_lock_ctx():
        idx = _load_idx_raw_strict()
        if idx is None:
            # Index corrupt — rebuild to preserve all existing entries
            idx = _rebuild_idx()
        idx[observation_key] = yyyymm
        _save_idx_atomic(idx)


def _find_key_in_jsonl(observation_key: str) -> str | None:
    """
    Scan all monthly JSONL files looking for this observation_key.
    Returns the YYYY-MM partition if found in exactly one partition.
    Raises CorruptionError if the key appears in more than one partition.
    Returns None if not found anywhere.
    """
    found_in: list[str] = []
    for p in sorted(LEDGER_DIR.glob("*_v2b_observations.jsonl")):
        yyyymm = p.name[:7]
        events = _read_events_raw(p)
        if any(e.get("observation_key") == observation_key for e in events):
            found_in.append(yyyymm)
    if len(found_in) > 1:
        raise CorruptionError(
            f"v2b_ledger: observation_key {observation_key[:16]}… found in multiple "
            f"partitions: {found_in} — data integrity violation"
        )
    return found_in[0] if found_in else None


def _resolve_partition_for_key(
    observation_key: str,
) -> tuple[str | None, list[dict]]:
    """
    Locate the partition and events for an observation_key, validating the index mapping.

    Fast path: index has a valid YYYYMM mapping AND the mapped partition contains the key.
    Slow path: index is absent, has invalid format, or the mapped partition does not
               contain the key → full JSONL scan, atomically repair the index.

    Returns (yyyymm, key_events).
    yyyymm is None and key_events is [] if the key does not exist anywhere.
    Raises CorruptionError if the key is found in more than one partition.
    """
    idx = _load_idx_or_rebuild()
    cached_yyyymm: str | None = idx.get(observation_key)

    if cached_yyyymm is not None and isinstance(cached_yyyymm, str) and _YYYYMM_RE.match(cached_yyyymm):
        path = _ledger_path(cached_yyyymm)
        all_events = _read_events_raw(path)  # Returns [] if file does not exist
        key_events = _find_events_for_key(observation_key, all_events)
        if key_events:
            return cached_yyyymm, key_events
        # Mapped partition does not contain the key — stale or wrong mapping

    # Full scan — also detects multi-partition duplicates
    real_yyyymm = _find_key_in_jsonl(observation_key)
    if real_yyyymm is None:
        return None, []

    # Repair stale mapping under index lock
    if real_yyyymm != cached_yyyymm:
        _update_idx_locked(observation_key, real_yyyymm)

    path = _ledger_path(real_yyyymm)
    all_events = _read_events_raw(path)
    return real_yyyymm, _find_events_for_key(observation_key, all_events)


# ---------------------------------------------------------------------------
# Event construction helpers
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
        h = ev.get("event_hash")
        if h:
            return h
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
    ticker_records_represents_all_scored: bool = True,
    metadata: dict | None = None,
) -> dict:
    """
    Open a new shadow observation. Returns the created event dict.

    ticker_records_represents_all_scored:
      True  — ticker_records contains every scored ticker; cross-validation of
               selected_tickers_per_strategy is enforced.
      False — ticker_records is a subset (e.g. top-N only); cross-validation
               is skipped. Caller is responsible for consistency.

    Idempotency:
      Same key + same content_hash → IDEMPOTENT_MATCH (no write).
        The index is repaired if the key was missing (crash recovery).
      Same key + different hash   → CONFLICT event written, ConflictError raised.

    Linearizability:
      All read-check-write decisions happen inside an exclusive monthly lock.
      Index update happens AFTER the monthly lock is released (see lock ordering docs).
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
    if ticker_records_represents_all_scored:
        _validate_selected_tickers_vs_records(selected_tickers_per_strategy, ticker_records)

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
        # ticker_records_represents_all_scored changes validation semantics AND is stored
        # in the event — it must be part of canonical so a changed flag triggers CONFLICT.
        "ticker_records_represents_all_scored": ticker_records_represents_all_scored,
        "ticker_records": ticker_records,
        "metadata": metadata or {},
    }
    content_hash = make_content_hash(canonical)

    # ── Atomically read-then-write under the monthly lock ─────────────────
    _event_result: dict | None = None
    _is_idempotent = False

    with _monthly_lock(yyyymm):
        all_events = _read_events_raw(path)
        key_events = _find_events_for_key(observation_key, all_events)

        if key_events:
            prior_creation = key_events[0]
            prior_hash = prior_creation.get("content_hash")

            if prior_hash == content_hash:
                # Idempotent — do not write. Will repair index after lock release.
                _is_idempotent = True
                _event_result = {
                    "event_type": "IDEMPOTENT_MATCH",
                    "record_version": RECORD_VERSION,
                    "observation_key": observation_key,
                    "observation_run_id": prior_creation.get("observation_run_id"),
                    "content_hash": content_hash,
                    "timestamp_utc": _now_iso(),
                    "order_creation_blocked": _ORDER_CREATION_BLOCKED,
                }
            else:
                # Conflict — write CONFLICT event then raise
                conflict_run_id = str(uuid.uuid4())
                prev_hash = _last_event_hash(key_events)
                conflict_ev = _build_event(
                    "CONFLICT",
                    observation_key,
                    conflict_run_id,
                    {"prior_content_hash": prior_hash, "new_content_hash": content_hash},
                    prev_hash,
                )
                _append_line(path, conflict_ev)
                raise ConflictError(
                    f"observation_key {observation_key[:16]}… already exists with different "
                    f"content. Prior={prior_hash[:16] if prior_hash else 'None'}… "
                    f"New={content_hash[:16]}…"
                )
        else:
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
                "ticker_records_represents_all_scored": ticker_records_represents_all_scored,
                "ticker_records": ticker_records,
                "metadata": metadata or {},
                # Never True globally — per-ticker provenance tracks PIT fields.
                "point_in_time_fundamentals_global": False,
            }
            _event_result = _build_event(
                "OBSERVATION_CREATED",
                observation_key,
                run_id,
                event_body_extra,
                None,  # First event for this key
            )
            _append_line(path, _event_result)

    # ── Update index AFTER releasing the monthly lock ─────────────────────
    # Always called: repairs the index gap if the process crashed between
    # _append_line and a prior _update_idx_locked call (idempotent creates
    # also repair the index this way).
    _update_idx_locked(observation_key, yyyymm)

    return _event_result  # type: ignore[return-value]


def transition_observation(
    observation_key: str,
    new_status: ObservationStatus,
    note: str | None = None,
) -> dict:
    """
    Append a status-transition event.

    Status is read and the transition is validated under the exclusive monthly lock,
    so concurrent calls to transition the same observation are serialized correctly.

    Index validation: the mapped partition is verified to contain the key before
    taking the monthly lock. A stale or wrong mapping triggers JSONL scan and repair.
    """
    yyyymm, _ = _resolve_partition_for_key(observation_key)
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
    """
    Return current status. Returns None only if the key genuinely does not exist.

    Index validation: the mapped partition is verified to contain the key. A stale
    or wrong mapping (wrong month, non-existent file) triggers JSONL scan and repair.
    """
    _, key_events = _resolve_partition_for_key(observation_key)
    if not key_events:
        return None
    return _derive_status(key_events)


def get_observation_events(observation_key: str) -> list[dict]:
    """
    Return all events for a key (in append order). Returns [] if not found.

    Index validation: the mapped partition is verified to contain the key. A stale
    or wrong mapping triggers JSONL scan and repair.
    """
    _, key_events = _resolve_partition_for_key(observation_key)
    return key_events


def list_observations(yyyymm: str | None = None) -> list[dict]:
    """Return a summary list (one dict per observation_key)."""
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
        result.append({
            "observation_key": key,
            "observation_run_id": first.get("observation_run_id"),
            "status": _derive_status(evs),
            "model_version": first.get("model_version"),
            "intended_execution_session": first.get("intended_execution_session"),
            "content_hash": first.get("content_hash"),
            "created_at": first.get("timestamp_utc"),
            "event_count": len(evs),
        })
    return result


# ---------------------------------------------------------------------------
# Structural invariant assertion
# ---------------------------------------------------------------------------


def assert_order_creation_blocked() -> None:
    assert _ORDER_CREATION_BLOCKED is True, (
        "V2B ledger structural invariant violated: order_creation_blocked must always be True"
    )
