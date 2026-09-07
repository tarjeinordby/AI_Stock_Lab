"""
V2B.4 — Signal outcome ledger.

Immutable append-only event store for prospective signal-outcome event study.
Records forward returns for Factor_Only_Core_V2 shadow signals over
1, 5, 21, and 63 holding sessions (inclusive — entry_session counts as day 1).

EVENTSTUDY ONLY — not a portfolio simulation:
  - Measures hit-rate and forward alpha against SPY.
  - These overlapping point-observations CANNOT produce Sharpe, Sortino, beta,
    maximum drawdown, or promotion criteria.  See V2B design docs (Del B).
  - order_creation_blocked = True is a structural invariant.

File layout (relative to repo root):
  data_v4/v2b_outcomes/{YYYY-MM}_v2b_outcomes.jsonl  (partitioned by exit_session month)
  data_v4/v2b_outcomes/v2b_outcomes.lock             (single global lock — no deadlock)
  data_v4/v2b_outcomes/v2b_outcome_idx.json          (outcome_key → exit_partition_yyyymm)

Outcome key:
  SHA-256(observation_key + "|" + strategy_id + "|" + str(holding_sessions)
          + "|" + outcome_definition_version)

Holding sessions — inclusive counting:
  N=1: exit_session = entry_session (same-day close)
  N=5: exit_session = 4th NYSE session after entry_session (entry = day 1, exit = day 5)

Locking — single global lock (deadlock-free):
  Held for the complete transaction:
    read events → validate chain → compute transition → append to JSONL
    → fsync JSONL → update in-memory index → save index atomically → release lock

  Startup validation (validate_and_rebuild_index) runs before any write.
  It does not need the global lock because GitHub Actions serialises runs via
  concurrency groups.  After startup, the index is trusted within the lock.

State machine:
  OUTCOME_PENDING → OUTCOME_RECORDED     (terminal: all tickers + SPY complete)
  OUTCOME_PENDING → OUTCOME_INCOMPLETE   (terminal: partial data after grace period)
  OUTCOME_PENDING → OUTCOME_UNAVAILABLE  (terminal: no usable data after grace period)
  Non-terminal helper: OUTCOME_FETCH_DEFERRED  (valid response, missing price rows)

Hash chain (record_version="1"):
  event_hash = SHA-256(canonical_json(event_body_without_event_hash_field))
  previous_event_hash = event_hash of prior event for same outcome_key (null for first)
  Broken chains raise CorruptionError — fail-closed.

Idempotency:
  PENDING  : pending_content_hash compared — IDEMPOTENT_MATCH or ContentConflictError
  Terminals: terminal_content_hash compared — IDEMPOTENT_MATCH or ContentConflictError
  FETCH_DEFERRED: deduplicated by (outcome_key, attempt_date, sorted missing_price_points)

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
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Literal

import pandas as pd

# ── Structural invariant ──────────────────────────────────────────────────────
_ORDER_CREATION_BLOCKED: bool = True

# ── Configuration constants ───────────────────────────────────────────────────

# Pre-registered activation date for event_study_v1.
# Only COMPLETED observations with intended_execution_session >= this date
# are eligible for outcome tracking.  Immutable after V2B.4 activation.
OUTCOME_TRACKING_START_SESSION: str = "2026-09-08"

# Number of NYSE sessions strictly after exit_session that must have elapsed
# before a missing-data PENDING may be terminated as INCOMPLETE or UNAVAILABLE.
GRACE_SESSIONS: int = 5

# Outcome definition version — part of the outcome_key.
# Change this if the measurement definition changes (entry/exit timing, price type, etc.)
OUTCOME_DEFINITION_VERSION: str = "event_study_v1"

# Only strategy tracked in V2B.4.0
STRATEGY_ID: str = "Factor_Only_Core_V2"

# Valid holding session counts
HOLDING_SESSIONS_TUPLE: tuple[int, ...] = (1, 5, 21, 63)

RECORD_VERSION: str = "1"

OUTCOME_DIR: Path = Path(__file__).parent.parent / "data_v4" / "v2b_outcomes"

_HEX64_RE = __import__("re").compile(r"^[0-9a-f]{64}$")
_DATE_RE = __import__("re").compile(r"^\d{4}-\d{2}-\d{2}$")
_YYYYMM_RE = __import__("re").compile(r"^\d{4}-\d{2}$")

# ── Exceptions ────────────────────────────────────────────────────────────────


class CorruptionError(RuntimeError):
    """Broken hash chain, tampered event_hash, invalid JSON, or duplicate key."""


class InvalidTransitionError(RuntimeError):
    """Illegal state machine transition attempted."""


class ContentConflictError(RuntimeError):
    """Same outcome_key produced different semantic content — fail-closed."""


class ObservationValidationError(ValueError):
    """Observation data is structurally invalid (missing/duplicate tickers, etc.)."""


# ── State machine ─────────────────────────────────────────────────────────────

_TERMINAL_STATES: frozenset[str] = frozenset({
    "OUTCOME_RECORDED",
    "OUTCOME_INCOMPLETE",
    "OUTCOME_UNAVAILABLE",
})

_NON_TERMINAL_EVENTS: frozenset[str] = frozenset({
    "OUTCOME_PENDING",
    "OUTCOME_FETCH_DEFERRED",
})

_ALL_EVENT_TYPES: frozenset[str] = _TERMINAL_STATES | _NON_TERMINAL_EVENTS


# ── Canonical JSON ────────────────────────────────────────────────────────────


def _canonical_json(obj: object) -> str:
    """Strict canonical JSON for hashing.  Raises ValueError on NaN/Inf."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _store_json(obj: object) -> str:
    """JSON for on-disk storage — same strict rules as canonical."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


# ── Hash helpers ──────────────────────────────────────────────────────────────


def _compute_event_hash(event_body: dict) -> str:
    """SHA-256 of event body.  Must NOT include the event_hash field itself."""
    return hashlib.sha256(_canonical_json(event_body).encode()).hexdigest()


def make_outcome_key(
    observation_key: str,
    strategy_id: str,
    holding_sessions: int,
    outcome_definition_version: str,
) -> str:
    """Deterministic outcome key.  Unique per strategy and definition version."""
    payload = (
        f"{observation_key}|{strategy_id}|{holding_sessions}|{outcome_definition_version}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def make_pending_content_hash(
    strategy_id: str,
    outcome_definition_version: str,
    holding_sessions: int,
    entry_session: str,
    exit_session: str,
    exit_partition_yyyymm: str,
    selected_tickers: list[str],
) -> str:
    """Content hash for a PENDING event.  Used for idempotency comparison."""
    body = {
        "strategy_id": strategy_id,
        "outcome_definition_version": outcome_definition_version,
        "holding_sessions": holding_sessions,
        "entry_session": entry_session,
        "exit_session": exit_session,
        "exit_partition_yyyymm": exit_partition_yyyymm,
        "selected_tickers": sorted(selected_tickers),
    }
    return hashlib.sha256(_canonical_json(body).encode()).hexdigest()


def make_terminal_content_hash(payload: dict) -> str:
    """
    Content hash for a terminal event (RECORDED / INCOMPLETE / UNAVAILABLE).

    Covers all semantic result fields.  Excludes volatile timestamps
    (fetched_at, measurement_date) and hash chain fields (event_hash,
    previous_event_hash).
    """
    body = {
        "event_type": payload["event_type"],
        "outcome_key": payload["outcome_key"],
        "observation_key": payload["observation_key"],
        "strategy_id": payload["strategy_id"],
        "outcome_definition_version": payload["outcome_definition_version"],
        "holding_sessions": payload["holding_sessions"],
        "entry_session": payload["entry_session"],
        "exit_session": payload["exit_session"],
        "selected_tickers": sorted(payload["selected_tickers"]),
        "unavailable_tickers": sorted(payload.get("unavailable_tickers") or []),
        "unavailable_reasons": payload.get("unavailable_reasons") or {},
        "entry_prices": payload.get("entry_prices") or {},
        "exit_prices": payload.get("exit_prices") or {},
        "spy_entry_price": payload.get("spy_entry_price"),
        "spy_exit_price": payload.get("spy_exit_price"),
        "per_ticker_return_pct": payload.get("per_ticker_return_pct") or {},
        "spy_return_pct": payload.get("spy_return_pct"),
        "portfolio_return_pct": payload.get("portfolio_return_pct"),
        "portfolio_return_complete": payload.get("portfolio_return_complete"),
        "forward_alpha_vs_spy": payload.get("forward_alpha_vs_spy"),
        "hit_rate_vs_spy": payload.get("hit_rate_vs_spy"),
        "available_subset_return_pct": payload.get("available_subset_return_pct"),
        "adjustment_mode": payload.get("adjustment_mode"),
        "data_source": payload.get("data_source"),
        "fetch_date_range": payload.get("fetch_date_range"),
        "provider_end_exclusive": payload.get("provider_end_exclusive"),
    }
    return hashlib.sha256(_canonical_json(body).encode()).hexdigest()


# ── Calendar helpers ──────────────────────────────────────────────────────────


def compute_exit_session(entry_session: str, holding_sessions: int) -> str:
    """
    Compute exit_session using inclusive holding-session counting.

    entry_session counts as day 1, so:
      holding_sessions=1  →  exit = entry_session  (same-day close)
      holding_sessions=5  →  exit = 4th NYSE session after entry_session
      holding_sessions=21 →  exit = 20th NYSE session after entry_session
      holding_sessions=63 →  exit = 62nd NYSE session after entry_session

    Uses the NYSE calendar; raises CalendarUnavailableError if unavailable.
    """
    if holding_sessions not in HOLDING_SESSIONS_TUPLE:
        raise ValueError(
            f"holding_sessions must be one of {HOLDING_SESSIONS_TUPLE}, got {holding_sessions}"
        )
    if holding_sessions == 1:
        return entry_session
    from modules.exchange_calendar import nth_session_after  # noqa: PLC0415
    return nth_session_after(entry_session, holding_sessions - 1)


def _ny_today() -> str:
    """Current date in America/New_York as YYYY-MM-DD."""
    from zoneinfo import ZoneInfo  # noqa: PLC0415
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── File path helpers ─────────────────────────────────────────────────────────


def _ledger_path(exit_partition_yyyymm: str) -> Path:
    return OUTCOME_DIR / f"{exit_partition_yyyymm}_v2b_outcomes.jsonl"


def _global_lock_path() -> Path:
    return OUTCOME_DIR / "v2b_outcomes.lock"


def _idx_path() -> Path:
    return OUTCOME_DIR / "v2b_outcome_idx.json"


def _all_jsonl_paths() -> list[Path]:
    if not OUTCOME_DIR.exists():
        return []
    return sorted(OUTCOME_DIR.glob("????-??_v2b_outcomes.jsonl"))


# ── Global lock ───────────────────────────────────────────────────────────────


@contextmanager
def _global_lock() -> Generator[None, None, None]:
    """Single exclusive lock covering all outcome ledger operations."""
    OUTCOME_DIR.mkdir(parents=True, exist_ok=True)
    lp = _global_lock_path()
    lp.touch(exist_ok=True)
    with open(lp, "r") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


# ── JSONL I/O ─────────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict]:
    """
    Read all events from a JSONL file.
    Incomplete last line is tolerated (skipped with a warning).
    Any non-last-line JSON error raises CorruptionError (fail-closed).
    """
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    events: list[dict] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            events.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            if i == len(lines) - 1:
                import warnings  # noqa: PLC0415
                warnings.warn(
                    f"v2b_outcome: incomplete last line in {path.name} — skipped",
                    stacklevel=2,
                )
            else:
                raise CorruptionError(
                    f"v2b_outcome: JSON parse error at line {i + 1} in {path.name} "
                    f"(non-last line — fail-closed): {exc}"
                ) from exc
    return events


def _append_event_to_disk(event: dict, path: Path) -> None:
    """Append a JSON event line to the JSONL file and fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _store_json(event) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


# ── Index management ──────────────────────────────────────────────────────────


def _load_idx() -> dict[str, str]:
    """Load index from disk.  Returns {} if absent or unreadable."""
    p = _idx_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_idx_atomic(idx: dict[str, str]) -> None:
    """Write index atomically via tempfile + rename and fsync."""
    OUTCOME_DIR.mkdir(parents=True, exist_ok=True)
    p = _idx_path()
    data = _store_json(idx) + "\n"
    fd, tmp_path = tempfile.mkstemp(dir=str(OUTCOME_DIR), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, p)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ── Hash chain verification ───────────────────────────────────────────────────


def _verify_chain(events: list[dict], outcome_key: str, source: str) -> None:
    """
    Verify the hash chain for a sequence of events belonging to outcome_key.
    Raises CorruptionError on any violation.
    """
    prev_hash: str | None = None
    for i, ev in enumerate(events):
        ev_type = ev.get("event_type", "?")

        # event_hash must be present and 64 hex chars
        ev_hash = ev.get("event_hash")
        if not isinstance(ev_hash, str) or not _HEX64_RE.match(ev_hash):
            raise CorruptionError(
                f"v2b_outcome: missing/invalid event_hash at position {i} "
                f"(type={ev_type}) in {source}"
            )

        # previous_event_hash field must be present
        if "previous_event_hash" not in ev:
            raise CorruptionError(
                f"v2b_outcome: missing previous_event_hash field at position {i} "
                f"(type={ev_type}) in {source}"
            )
        peh = ev["previous_event_hash"]
        if i == 0:
            if peh is not None:
                raise CorruptionError(
                    f"v2b_outcome: first event must have previous_event_hash=null "
                    f"at position 0 (type={ev_type}) in {source}, got {peh!r}"
                )
        else:
            if peh != prev_hash:
                raise CorruptionError(
                    f"v2b_outcome: broken chain at position {i} (type={ev_type}) "
                    f"in {source}: expected previous_event_hash={prev_hash!r}, "
                    f"got {peh!r}"
                )

        # Recompute event_hash and compare
        body_for_hash = {k: v for k, v in ev.items() if k != "event_hash"}
        expected_hash = _compute_event_hash(body_for_hash)
        if ev_hash != expected_hash:
            raise CorruptionError(
                f"v2b_outcome: event_hash mismatch at position {i} (type={ev_type}) "
                f"in {source}: stored={ev_hash!r}, computed={expected_hash!r}"
            )

        prev_hash = ev_hash


def _get_chain_tip(events: list[dict]) -> str | None:
    """Return event_hash of the last event in the chain, or None if empty."""
    if not events:
        return None
    return events[-1]["event_hash"]


# ── Status derivation ─────────────────────────────────────────────────────────


def _derive_status(events: list[dict]) -> str | None:
    """
    Derive current status from event list.
    Returns None if no events, 'OUTCOME_PENDING' if pending with no terminal,
    or the terminal state type if one exists.
    """
    if not events:
        return None
    for ev in events:
        et = ev.get("event_type", "")
        if et in _TERMINAL_STATES:
            return et
    return "OUTCOME_PENDING"


# ── Index scan helpers ────────────────────────────────────────────────────────


def _read_all_events_for_key(
    outcome_key: str, idx: dict[str, str]
) -> tuple[list[dict], str | None]:
    """
    Return (events, partition_yyyymm) for outcome_key, or ([], None) if not found.
    Scans from index first; falls back to full JSONL scan on index miss.
    Raises CorruptionError if the key appears in multiple partitions.
    """
    # Fast path via index
    known_partition = idx.get(outcome_key)
    if known_partition:
        path = _ledger_path(known_partition)
        all_events = _read_jsonl(path)
        key_events = [e for e in all_events if e.get("outcome_key") == outcome_key]
        if key_events:
            return key_events, known_partition
        # Index points to wrong partition — fall through to full scan

    # Slow path: scan all partitions
    found_partition: str | None = None
    found_events: list[dict] = []
    for path in _all_jsonl_paths():
        events_in_file = _read_jsonl(path)
        key_events = [e for e in events_in_file if e.get("outcome_key") == outcome_key]
        if key_events:
            part = path.name[:7]  # YYYY-MM prefix
            if found_partition is not None and found_partition != part:
                raise CorruptionError(
                    f"v2b_outcome: outcome_key {outcome_key[:16]}… appears in multiple "
                    f"partitions: {found_partition!r} and {part!r}"
                )
            found_partition = part
            found_events = key_events

    return found_events, found_partition


# ── Startup validation ────────────────────────────────────────────────────────


def validate_and_rebuild_index() -> dict[str, str]:
    """
    Scan all JSONL partition files, verify hash chains, detect duplicate keys
    across partitions, and rebuild the index if it is missing or stale.

    Called once at runner startup before any writes.
    Raises CorruptionError on any integrity violation.
    Returns the validated/rebuilt index dict.
    """
    OUTCOME_DIR.mkdir(parents=True, exist_ok=True)
    new_idx: dict[str, str] = {}
    partition_for_key: dict[str, str] = {}

    for path in _all_jsonl_paths():
        partition = path.name[:7]
        all_events = _read_jsonl(path)

        # Group events by outcome_key
        by_key: dict[str, list[dict]] = {}
        for ev in all_events:
            ok = ev.get("outcome_key")
            if not ok:
                raise CorruptionError(
                    f"v2b_outcome: event missing outcome_key in {path.name}"
                )
            by_key.setdefault(ok, []).append(ev)

        for ok, events in by_key.items():
            # Global uniqueness: same key must not appear in multiple partitions
            if ok in partition_for_key and partition_for_key[ok] != partition:
                raise CorruptionError(
                    f"v2b_outcome: outcome_key {ok[:16]}… appears in both "
                    f"{partition_for_key[ok]!r} and {partition!r}"
                )
            partition_for_key[ok] = partition

            # Verify hash chain
            _verify_chain(events, ok, path.name)
            new_idx[ok] = partition

    # Rebuild index if it differs from current
    current_idx = _load_idx()
    if new_idx != current_idx:
        _save_idx_atomic(new_idx)

    return new_idx


# ── Core API ──────────────────────────────────────────────────────────────────


def create_pending(
    observation_key: str,
    strategy_id: str,
    holding_sessions: int,
    outcome_definition_version: str,
    entry_session: str,
    exit_session: str,
    selected_tickers: list[str],
) -> dict | Literal["IDEMPOTENT_MATCH"]:
    """
    Create an OUTCOME_PENDING event for the given parameters.

    Validates that selected_tickers contains no duplicates and no empty strings.
    sorted(selected_tickers) is used for all hash computations.

    Returns:
      dict                — new PENDING event (written to ledger)
      "IDEMPOTENT_MATCH"  — identical PENDING already exists

    Raises:
      ContentConflictError   — same outcome_key, different semantic content
      InvalidTransitionError — outcome already in a terminal state (unless content matches)
      CorruptionError        — broken chain or invalid data
      ObservationValidationError — invalid tickers
    """
    # Validate tickers
    if not isinstance(selected_tickers, list):
        raise ObservationValidationError(
            f"selected_tickers must be a list, got {type(selected_tickers).__name__}"
        )
    for t in selected_tickers:
        if not isinstance(t, str) or not t.strip():
            raise ObservationValidationError(
                f"selected_tickers contains invalid ticker: {t!r}"
            )
    if len(selected_tickers) != len(set(selected_tickers)):
        raise ObservationValidationError(
            f"selected_tickers contains duplicate tickers: {selected_tickers}"
        )

    exit_partition_yyyymm = exit_session[:7]
    sorted_tickers = sorted(selected_tickers)
    pch = make_pending_content_hash(
        strategy_id, outcome_definition_version, holding_sessions,
        entry_session, exit_session, exit_partition_yyyymm, selected_tickers,
    )
    outcome_key = make_outcome_key(
        observation_key, strategy_id, holding_sessions, outcome_definition_version
    )

    with _global_lock():
        idx = _load_idx()
        existing_events, existing_partition = _read_all_events_for_key(outcome_key, idx)

        if existing_events:
            _verify_chain(existing_events, outcome_key, f"outcome_key={outcome_key[:16]}…")
            first_ev = existing_events[0]
            stored_pch = first_ev.get("pending_content_hash")

            # Compare against the original PENDING event's content hash
            if stored_pch == pch:
                return "IDEMPOTENT_MATCH"
            else:
                raise ContentConflictError(
                    f"outcome_key {outcome_key[:16]}… already exists with different "
                    f"content (strategy/tickers/sessions mismatch). "
                    f"stored_pch={stored_pch!r} != new_pch={pch!r}"
                )

        # New outcome — write PENDING
        created_at = _utc_now_iso()
        event_body: dict = {
            "event_type": "OUTCOME_PENDING",
            "record_version": RECORD_VERSION,
            "outcome_key": outcome_key,
            "observation_key": observation_key,
            "strategy_id": strategy_id,
            "outcome_definition_version": outcome_definition_version,
            "holding_sessions": holding_sessions,
            "entry_session": entry_session,
            "exit_session": exit_session,
            "exit_partition_yyyymm": exit_partition_yyyymm,
            "selected_tickers": sorted_tickers,
            "pending_content_hash": pch,
            "created_at": created_at,
            "order_creation_blocked": True,
            "previous_event_hash": None,
        }
        event_body["event_hash"] = _compute_event_hash(
            {k: v for k, v in event_body.items() if k != "event_hash"}
        )

        path = _ledger_path(exit_partition_yyyymm)
        _append_event_to_disk(event_body, path)

        idx[outcome_key] = exit_partition_yyyymm
        _save_idx_atomic(idx)

    return event_body


def record_fetch_deferred(
    outcome_key: str,
    attempted_at: str,
    attempt_date: str,
    missing_price_points: list[dict],
    source: str,
    detail: str,
) -> dict | None:
    """
    Record that a valid yfinance response was received but price rows were missing.

    Deduplicates by (outcome_key, attempt_date, canonical sorted missing_price_points).
    Returns None if deduplicated (identical event already exists for today).
    Raises CorruptionError / InvalidTransitionError on integrity violations.

    missing_price_points: list of {symbol, field, session} dicts.
    """
    _canonical_missing = sorted(
        missing_price_points,
        key=lambda x: (_canonical_json(x),),
    )
    canonical_mpp_str = _canonical_json(_canonical_missing)

    with _global_lock():
        idx = _load_idx()
        existing_events, partition = _read_all_events_for_key(outcome_key, idx)
        if not existing_events:
            raise InvalidTransitionError(
                f"record_fetch_deferred: outcome_key {outcome_key[:16]}… not found in ledger"
            )
        _verify_chain(existing_events, outcome_key, f"outcome_key={outcome_key[:16]}…")

        status = _derive_status(existing_events)
        if status in _TERMINAL_STATES:
            raise InvalidTransitionError(
                f"record_fetch_deferred: outcome_key {outcome_key[:16]}… is already in "
                f"terminal state {status!r} — FETCH_DEFERRED not allowed"
            )

        # Deduplication check
        for ev in existing_events:
            if ev.get("event_type") != "OUTCOME_FETCH_DEFERRED":
                continue
            if ev.get("attempt_date") != attempt_date:
                continue
            stored_mpp = ev.get("missing_price_points", [])
            if _canonical_json(stored_mpp) == canonical_mpp_str:
                return None  # deduplicated

        # Build and append FETCH_DEFERRED event
        chain_tip = _get_chain_tip(existing_events)
        event_body: dict = {
            "event_type": "OUTCOME_FETCH_DEFERRED",
            "record_version": RECORD_VERSION,
            "outcome_key": outcome_key,
            "strategy_id": existing_events[0].get("strategy_id"),
            "outcome_definition_version": existing_events[0].get("outcome_definition_version"),
            "attempted_at": attempted_at,
            "attempt_date": attempt_date,
            "missing_price_points": _canonical_missing,
            "missing_data_class": "price_row_missing",
            "source": source,
            "detail": detail,
            "order_creation_blocked": True,
            "previous_event_hash": chain_tip,
        }
        event_body["event_hash"] = _compute_event_hash(
            {k: v for k, v in event_body.items() if k != "event_hash"}
        )

        path = _ledger_path(partition)
        _append_event_to_disk(event_body, path)
        # index partition unchanged — no index update needed

    return event_body


def record_terminal(
    outcome_key: str,
    terminal_payload: dict,
) -> dict | Literal["IDEMPOTENT_MATCH"]:
    """
    Write a terminal event (OUTCOME_RECORDED / OUTCOME_INCOMPLETE / OUTCOME_UNAVAILABLE).

    terminal_payload must include 'event_type' and all semantic result fields.
    terminal_content_hash is computed and stored in the event; used for idempotency.

    Returns:
      dict                — written terminal event
      "IDEMPOTENT_MATCH"  — identical terminal event already exists

    Raises:
      ContentConflictError   — same outcome_key + same event_type but different content
      InvalidTransitionError — same outcome_key but different terminal type already exists,
                               or attempt to terminate a non-PENDING outcome
      CorruptionError        — broken chain
    """
    event_type = terminal_payload["event_type"]
    if event_type not in _TERMINAL_STATES:
        raise ValueError(f"record_terminal: event_type {event_type!r} is not a terminal state")

    tch = make_terminal_content_hash(terminal_payload)

    with _global_lock():
        idx = _load_idx()
        existing_events, partition = _read_all_events_for_key(outcome_key, idx)
        if not existing_events:
            raise InvalidTransitionError(
                f"record_terminal: outcome_key {outcome_key[:16]}… not found in ledger"
            )
        _verify_chain(existing_events, outcome_key, f"outcome_key={outcome_key[:16]}…")

        status = _derive_status(existing_events)
        if status in _TERMINAL_STATES:
            # Find the stored terminal event
            terminal_ev = next(e for e in existing_events if e.get("event_type") in _TERMINAL_STATES)
            stored_type = terminal_ev.get("event_type")
            if stored_type != event_type:
                raise InvalidTransitionError(
                    f"record_terminal: outcome_key {outcome_key[:16]}… is already in "
                    f"terminal state {stored_type!r} — cannot transition to {event_type!r}"
                )
            stored_tch = terminal_ev.get("terminal_content_hash")
            if stored_tch == tch:
                return "IDEMPOTENT_MATCH"
            else:
                raise ContentConflictError(
                    f"record_terminal: outcome_key {outcome_key[:16]}… already has "
                    f"{event_type!r} with different content. "
                    f"stored_tch={stored_tch!r} != new_tch={tch!r}"
                )

        if status != "OUTCOME_PENDING":
            raise InvalidTransitionError(
                f"record_terminal: outcome_key {outcome_key[:16]}… has unexpected status "
                f"{status!r} — only OUTCOME_PENDING may transition to terminal"
            )

        chain_tip = _get_chain_tip(existing_events)
        event_body: dict = dict(terminal_payload)
        event_body["record_version"] = RECORD_VERSION
        event_body["outcome_key"] = outcome_key
        event_body["terminal_content_hash"] = tch
        event_body["order_creation_blocked"] = True
        event_body["previous_event_hash"] = chain_tip
        # Remove event_hash if caller accidentally included it
        event_body.pop("event_hash", None)
        event_body["event_hash"] = _compute_event_hash(
            {k: v for k, v in event_body.items() if k != "event_hash"}
        )

        path = _ledger_path(partition)
        _append_event_to_disk(event_body, path)
        # index partition unchanged — no index update needed

    return event_body


# ── Read API ──────────────────────────────────────────────────────────────────


def get_outcome_events(outcome_key: str) -> list[dict]:
    """Return all events for the given outcome_key in append order."""
    idx = _load_idx()
    events, _ = _read_all_events_for_key(outcome_key, idx)
    return events


def get_outcome_status(outcome_key: str) -> str | None:
    """Return current outcome status, or None if outcome_key not found."""
    return _derive_status(get_outcome_events(outcome_key))


def list_pending_outcomes() -> list[dict]:
    """
    Return the initial OUTCOME_PENDING event for every outcome that is still
    in PENDING status (no terminal event written yet).

    Used by the runner to find outcomes that have matured (exit_session passed).
    """
    result: list[dict] = []
    idx = _load_idx()

    for outcome_key, partition in idx.items():
        path = _ledger_path(partition)
        all_events = _read_jsonl(path)
        key_events = [e for e in all_events if e.get("outcome_key") == outcome_key]
        if not key_events:
            continue
        status = _derive_status(key_events)
        if status == "OUTCOME_PENDING":
            first_ev = key_events[0]
            result.append(first_ev)

    return result
