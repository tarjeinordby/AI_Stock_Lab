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
  data_v4/v2b_outcomes/v2b_outcomes.lock             (single global lock)
  data_v4/v2b_outcomes/v2b_outcome_idx.json          (outcome_key → exit_partition_yyyymm)

Outcome key:
  SHA-256(observation_key + "|" + strategy_id + "|" + str(holding_sessions)
          + "|" + outcome_definition_version)

Locking — single global lock, held for full transaction:
  read events → validate chain → compute transition → append to JSONL
  → fsync JSONL → update index → fsync index dir → release lock

  validate_and_rebuild_index() also runs under the global lock.
  No deadlock possible (sequential, not nested).

State machine:
  OUTCOME_PENDING → OUTCOME_RECORDED     (all tickers + SPY complete)
  OUTCOME_PENDING → OUTCOME_INCOMPLETE   (some data after grace period)
  OUTCOME_PENDING → OUTCOME_UNAVAILABLE  (no complete tickers after grace / empty list)
  Non-terminal helper: OUTCOME_FETCH_DEFERRED  (valid response, missing price rows)

Event sequence invariants (per outcome_key):
  [0]   OUTCOME_PENDING           (exactly one, always first)
  [1..] OUTCOME_FETCH_DEFERRED    (zero or more)
  [-1]  terminal event            (zero or one, always last)
  Unknown event_type → CorruptionError

Hash chain (record_version="1"):
  event_hash = SHA-256(canonical_json(event_body_excluding_event_hash))
  previous_event_hash = event_hash of prior event (null for first)
  Broken chains → CorruptionError (fail-closed)

Idempotency:
  PENDING  : pending_content_hash compared
  Terminals: terminal_content_hash compared (includes exit_partition_yyyymm)
  FETCH_DEFERRED: (outcome_key, attempt_date, sorted missing_price_points)

Terminal payload validation (record_terminal):
  Identity fields verified against PENDING.
  State invariants verified per terminal type.
  Schema constants verified: adjustment_mode, data_source, fetch_date_range.
  Metrics recalculated and compared within 1e-9 tolerance.
  First invalid payload → OutcomeValidationError (no write).

JSONL integrity: any invalid non-empty line → CorruptionError (no lenient last-line skip).

V1 isolation: no imports from V1 execution modules.
  Does NOT import: modules.portfolio, modules.orders, modules.fills,
                   modules.ledger, modules.state
  Does NOT call:   execute_buy, execute_sell, execute_pyramid_fill
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator, Literal

# ── Structural invariant ──────────────────────────────────────────────────────
_ORDER_CREATION_BLOCKED: bool = True

# ── Configuration constants ───────────────────────────────────────────────────

OUTCOME_TRACKING_START_SESSION: str = "2026-09-08"
GRACE_SESSIONS: int = 5
OUTCOME_DEFINITION_VERSION: str = "event_study_v1"
STRATEGY_ID: str = "Factor_Only_Core_V2"
HOLDING_SESSIONS_TUPLE: tuple[int, ...] = (1, 5, 21, 63)
RECORD_VERSION: str = "1"

# Schema constants — required in every terminal payload
ADJUSTMENT_MODE: str = "split_and_dividend_adjusted"
DATA_SOURCE: str = "yfinance_daily_ohlcv"

OUTCOME_DIR: Path = Path(__file__).parent.parent / "data_v4" / "v2b_outcomes"

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# ── Exceptions ────────────────────────────────────────────────────────────────


class CorruptionError(RuntimeError):
    """Broken hash chain, tampered event, invalid JSON, or duplicate key."""


class InvalidTransitionError(RuntimeError):
    """Illegal state machine transition attempted."""


class ContentConflictError(RuntimeError):
    """Same outcome_key produced different semantic content — fail-closed."""


class ObservationValidationError(ValueError):
    """Observation data is structurally invalid (missing key, duplicate tickers, etc.)."""


class OutcomeValidationError(ValueError):
    """Terminal payload fails invariant validation — no ledger write performed."""


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
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _store_json(obj: object) -> str:
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
    Content hash for a terminal event.  Covers all semantic result fields.
    Excludes volatile timestamps (fetched_at, measurement_date) and chain fields.
    Includes exit_partition_yyyymm.
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
        "exit_partition_yyyymm": payload.get("exit_partition_yyyymm"),
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
    Compute exit_session (inclusive counting: entry_session = day 1).
    N=1 → same day.  N>1 → (N-1)th NYSE session after entry_session.
    """
    if holding_sessions not in HOLDING_SESSIONS_TUPLE:
        raise ValueError(
            f"holding_sessions must be one of {HOLDING_SESSIONS_TUPLE}, got {holding_sessions}"
        )
    if holding_sessions == 1:
        return entry_session
    from modules.exchange_calendar import nth_session_after  # noqa: PLC0415
    return nth_session_after(entry_session, holding_sessions - 1)


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


# ── Directory fsync ───────────────────────────────────────────────────────────


def _fsync_dir(directory: Path) -> None:
    """fsync directory so rename of atomic-write temp-files is crash-durable."""
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# ── JSONL I/O ─────────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict]:
    """
    Read all events from a JSONL file.  Fail-closed: any invalid non-empty line
    raises CorruptionError.  Empty file returns [].
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
            raise CorruptionError(
                f"v2b_outcome: JSON parse error at line {i + 1} in {path.name} "
                f"(fail-closed — manual recovery required): {exc}"
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
    p = _idx_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_idx_atomic(idx: dict[str, str]) -> None:
    """Write index atomically via tempfile + rename, then fsync the directory."""
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
        _fsync_dir(OUTCOME_DIR)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ── Hash chain verification ───────────────────────────────────────────────────


def _verify_chain(events: list[dict], outcome_key: str, source: str) -> None:
    """Verify SHA-256 hash chain for a sequence of events.  Fail-closed."""
    prev_hash: str | None = None
    for i, ev in enumerate(events):
        et = ev.get("event_type", "?")

        ev_hash = ev.get("event_hash")
        if not isinstance(ev_hash, str) or not _HEX64_RE.match(ev_hash):
            raise CorruptionError(
                f"v2b_outcome: missing/invalid event_hash at position {i} "
                f"(type={et}) in {source}"
            )
        if "previous_event_hash" not in ev:
            raise CorruptionError(
                f"v2b_outcome: missing previous_event_hash field at position {i} "
                f"(type={et}) in {source}"
            )
        peh = ev["previous_event_hash"]
        if i == 0:
            if peh is not None:
                raise CorruptionError(
                    f"v2b_outcome: first event must have previous_event_hash=null "
                    f"at position 0 (type={et}) in {source}, got {peh!r}"
                )
        else:
            if peh != prev_hash:
                raise CorruptionError(
                    f"v2b_outcome: broken chain at position {i} (type={et}) "
                    f"in {source}: expected previous_event_hash={prev_hash!r}, "
                    f"got {peh!r}"
                )

        body_for_hash = {k: v for k, v in ev.items() if k != "event_hash"}
        expected_hash = _compute_event_hash(body_for_hash)
        if ev_hash != expected_hash:
            raise CorruptionError(
                f"v2b_outcome: event_hash mismatch at position {i} (type={et}) "
                f"in {source}: stored={ev_hash!r}, computed={expected_hash!r}"
            )
        prev_hash = ev_hash


def _get_chain_tip(events: list[dict]) -> str | None:
    if not events:
        return None
    return events[-1]["event_hash"]


# ── Event sequence validation ─────────────────────────────────────────────────


def _validate_event_sequence(events: list[dict], source: str) -> None:
    """
    Validate state machine sequence for events belonging to one outcome_key.

    Invariants:
    - First event is exactly OUTCOME_PENDING
    - Middle events are only OUTCOME_FETCH_DEFERRED
    - At most one terminal event; it must be the last event
    - All event_type values are known (_ALL_EVENT_TYPES)
    - All events share the same outcome_key
    """
    if not events:
        return

    for i, ev in enumerate(events):
        et = ev.get("event_type")
        if et not in _ALL_EVENT_TYPES:
            raise CorruptionError(
                f"v2b_outcome: unknown event_type {et!r} at position {i} in {source}"
            )

    if events[0].get("event_type") != "OUTCOME_PENDING":
        raise CorruptionError(
            f"v2b_outcome: first event is {events[0].get('event_type')!r}, "
            f"expected OUTCOME_PENDING in {source}"
        )

    ok = events[0].get("outcome_key")
    for i, ev in enumerate(events):
        if ev.get("outcome_key") != ok:
            raise CorruptionError(
                f"v2b_outcome: mismatched outcome_key at position {i} in {source}"
            )

    terminal_events = [e for e in events if e.get("event_type") in _TERMINAL_STATES]
    if len(terminal_events) > 1:
        raise CorruptionError(
            f"v2b_outcome: multiple terminal events in {source}: "
            f"{[e['event_type'] for e in terminal_events]}"
        )

    if terminal_events:
        if events[-1].get("event_type") not in _TERMINAL_STATES:
            raise CorruptionError(
                f"v2b_outcome: terminal event is not the last event in {source}"
            )
        for ev in events[1:-1]:
            if ev.get("event_type") != "OUTCOME_FETCH_DEFERRED":
                raise CorruptionError(
                    f"v2b_outcome: unexpected {ev.get('event_type')!r} between "
                    f"PENDING and terminal in {source}"
                )
    else:
        for ev in events[1:]:
            if ev.get("event_type") != "OUTCOME_FETCH_DEFERRED":
                raise CorruptionError(
                    f"v2b_outcome: unexpected {ev.get('event_type')!r} after "
                    f"PENDING (no terminal yet) in {source}"
                )


# ── Status derivation ─────────────────────────────────────────────────────────


def _derive_status(events: list[dict]) -> str | None:
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
    Fast path via index; falls back to full scan on index miss.
    Raises CorruptionError if key appears in multiple partitions.
    """
    known_partition = idx.get(outcome_key)
    if known_partition:
        path = _ledger_path(known_partition)
        all_events = _read_jsonl(path)
        key_events = [e for e in all_events if e.get("outcome_key") == outcome_key]
        if key_events:
            return key_events, known_partition

    found_partition: str | None = None
    found_events: list[dict] = []
    for path in _all_jsonl_paths():
        all_events = _read_jsonl(path)
        key_events = [e for e in all_events if e.get("outcome_key") == outcome_key]
        if key_events:
            part = path.name[:7]
            if found_partition is not None and found_partition != part:
                raise CorruptionError(
                    f"v2b_outcome: outcome_key {outcome_key[:16]}… appears in multiple "
                    f"partitions: {found_partition!r} and {part!r}"
                )
            found_partition = part
            found_events = key_events

    return found_events, found_partition


# ── Terminal payload validation ───────────────────────────────────────────────


def _validate_identity_fields(payload: dict, pending_event: dict) -> None:
    """Verify terminal payload identity fields exactly match the original PENDING."""
    for field in (
        "outcome_key",
        "observation_key",
        "strategy_id",
        "outcome_definition_version",
        "holding_sessions",
        "entry_session",
        "exit_session",
        "exit_partition_yyyymm",
    ):
        pval = pending_event.get(field)
        tval = payload.get(field)
        if pval != tval:
            raise OutcomeValidationError(
                f"Terminal payload {field!r} mismatch with PENDING: "
                f"pending={pval!r}, payload={tval!r}"
            )

    pending_tickers = sorted(pending_event.get("selected_tickers", []))
    payload_tickers = sorted(payload.get("selected_tickers", []))
    if pending_tickers != payload_tickers:
        raise OutcomeValidationError(
            f"Terminal payload selected_tickers mismatch: "
            f"pending={pending_tickers}, payload={payload_tickers}"
        )


def _validate_schema_fields(payload: dict, pending_event: dict) -> None:
    """Verify required schema constants are present and correct."""
    adj = payload.get("adjustment_mode")
    if adj != ADJUSTMENT_MODE:
        raise OutcomeValidationError(
            f"adjustment_mode must be {ADJUSTMENT_MODE!r}, got {adj!r}"
        )
    src = payload.get("data_source")
    if src != DATA_SOURCE:
        raise OutcomeValidationError(
            f"data_source must be {DATA_SOURCE!r}, got {src!r}"
        )
    expected_fdr = [pending_event["entry_session"], pending_event["exit_session"]]
    fdr = payload.get("fetch_date_range")
    if fdr != expected_fdr:
        raise OutcomeValidationError(
            f"fetch_date_range must be {expected_fdr!r}, got {fdr!r}"
        )

    # provider_end_exclusive: calendar day after exit_session
    exit_session = pending_event["exit_session"]
    expected_pex = (
        datetime.strptime(exit_session, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")
    pex = payload.get("provider_end_exclusive")
    if pex != expected_pex:
        raise OutcomeValidationError(
            f"provider_end_exclusive must be {expected_pex!r} "
            f"(calendar day after exit_session={exit_session!r}), got {pex!r}"
        )

    # measurement_date: required, must be YYYY-MM-DD
    measurement_date = payload.get("measurement_date")
    if not isinstance(measurement_date, str) or not _DATE_RE.match(measurement_date):
        raise OutcomeValidationError(
            f"measurement_date must be a YYYY-MM-DD string, got {measurement_date!r}"
        )

    # fetched_at: non-null iff selected_tickers is non-empty
    selected_tickers = pending_event.get("selected_tickers", [])
    fetched_at = payload.get("fetched_at")
    if selected_tickers:
        if not isinstance(fetched_at, str) or not fetched_at.strip():
            raise OutcomeValidationError(
                f"fetched_at must be a non-empty ISO timestamp string for "
                f"non-empty selected_tickers, got {fetched_at!r}"
            )
    else:
        if fetched_at is not None:
            raise OutcomeValidationError(
                f"fetched_at must be null for empty selected_tickers, got {fetched_at!r}"
            )


def _check_close(label: str, expected: float, actual: object) -> None:
    """Raise OutcomeValidationError if actual doesn't match expected within 1e-9."""
    if actual is None or not isinstance(actual, (int, float)):
        raise OutcomeValidationError(
            f"{label}: expected finite float {expected:.10f}, got {actual!r}"
        )
    if not math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9):
        raise OutcomeValidationError(
            f"{label}: expected {expected:.10f}, got {float(actual):.10f}"
        )


def _validate_recorded_invariants(payload: dict) -> None:
    """OUTCOME_RECORDED: all tickers + SPY complete, aggregates verified, subset null."""
    selected_tickers = payload.get("selected_tickers", [])
    entry_prices = payload.get("entry_prices", {})
    exit_prices = payload.get("exit_prices", {})

    if not selected_tickers:
        raise OutcomeValidationError(
            "OUTCOME_RECORDED requires non-empty selected_tickers"
        )
    for t in selected_tickers:
        ep = entry_prices.get(t)
        xp = exit_prices.get(t)
        if ep is None or not (isinstance(ep, (int, float)) and ep > 0 and math.isfinite(ep)):
            raise OutcomeValidationError(
                f"OUTCOME_RECORDED: missing/invalid entry_price for {t!r}: {ep!r}"
            )
        if xp is None or not (isinstance(xp, (int, float)) and xp > 0 and math.isfinite(xp)):
            raise OutcomeValidationError(
                f"OUTCOME_RECORDED: missing/invalid exit_price for {t!r}: {xp!r}"
            )
    if set(entry_prices.keys()) != set(selected_tickers):
        raise OutcomeValidationError(
            "OUTCOME_RECORDED: entry_prices keys must equal selected_tickers"
        )
    if set(exit_prices.keys()) != set(selected_tickers):
        raise OutcomeValidationError(
            "OUTCOME_RECORDED: exit_prices keys must equal selected_tickers"
        )

    spy_ep = payload.get("spy_entry_price")
    spy_xp = payload.get("spy_exit_price")
    if spy_ep is None or not (isinstance(spy_ep, (int, float)) and spy_ep > 0 and math.isfinite(spy_ep)):
        raise OutcomeValidationError(
            f"OUTCOME_RECORDED: missing/invalid spy_entry_price: {spy_ep!r}"
        )
    if spy_xp is None or not (isinstance(spy_xp, (int, float)) and spy_xp > 0 and math.isfinite(spy_xp)):
        raise OutcomeValidationError(
            f"OUTCOME_RECORDED: missing/invalid spy_exit_price: {spy_xp!r}"
        )

    if payload.get("unavailable_tickers"):
        raise OutcomeValidationError(
            "OUTCOME_RECORDED: unavailable_tickers must be empty"
        )
    if payload.get("portfolio_return_complete") is not True:
        raise OutcomeValidationError(
            "OUTCOME_RECORDED: portfolio_return_complete must be True"
        )
    if payload.get("available_subset_return_pct") is not None:
        raise OutcomeValidationError(
            "OUTCOME_RECORDED: available_subset_return_pct must be null"
        )

    per_ticker_pct = payload.get("per_ticker_return_pct", {})
    if set(per_ticker_pct.keys()) != set(selected_tickers):
        raise OutcomeValidationError(
            "OUTCOME_RECORDED: per_ticker_return_pct keys must equal selected_tickers"
        )
    for t in selected_tickers:
        expected_r = ((exit_prices[t] - entry_prices[t]) / entry_prices[t]) * 100.0
        _check_close(f"per_ticker_return_pct[{t!r}]", expected_r, per_ticker_pct.get(t))

    expected_spy = ((spy_xp - spy_ep) / spy_ep) * 100.0
    _check_close("spy_return_pct", expected_spy, payload.get("spy_return_pct"))

    expected_pf = sum(
        ((exit_prices[t] - entry_prices[t]) / entry_prices[t]) * 100.0
        for t in selected_tickers
    ) / len(selected_tickers)
    _check_close("portfolio_return_pct", expected_pf, payload.get("portfolio_return_pct"))
    _check_close(
        "forward_alpha_vs_spy",
        expected_pf - expected_spy,
        payload.get("forward_alpha_vs_spy"),
    )

    expected_hr = sum(
        1 for t in selected_tickers
        if ((exit_prices[t] - entry_prices[t]) / entry_prices[t]) * 100.0 > expected_spy
    ) / len(selected_tickers)
    _check_close("hit_rate_vs_spy", expected_hr, payload.get("hit_rate_vs_spy"))


def _validate_incomplete_invariants(payload: dict) -> None:
    """OUTCOME_INCOMPLETE: ≥1 complete ticker, ≥1 missing, aggregates null, subset verified."""
    selected_tickers = payload.get("selected_tickers", [])
    entry_prices = payload.get("entry_prices", {})
    exit_prices = payload.get("exit_prices", {})

    complete_tickers = [
        t for t in selected_tickers if t in entry_prices and t in exit_prices
    ]
    if not complete_tickers:
        raise OutcomeValidationError(
            "OUTCOME_INCOMPLETE: requires at least one complete ticker (entry + exit prices)"
        )

    spy_complete = (
        payload.get("spy_entry_price") is not None
        and payload.get("spy_exit_price") is not None
    )
    missing_tickers = [t for t in selected_tickers if t not in complete_tickers]
    if not missing_tickers and spy_complete:
        raise OutcomeValidationError(
            "OUTCOME_INCOMPLETE: no missing data found — should use OUTCOME_RECORDED"
        )

    # Exact key set checks — no extra or missing keys allowed
    if set(entry_prices.keys()) != set(complete_tickers):
        raise OutcomeValidationError(
            f"OUTCOME_INCOMPLETE: entry_prices keys must be exactly "
            f"{sorted(complete_tickers)}, got {sorted(entry_prices.keys())}"
        )
    if set(exit_prices.keys()) != set(complete_tickers):
        raise OutcomeValidationError(
            f"OUTCOME_INCOMPLETE: exit_prices keys must be exactly "
            f"{sorted(complete_tickers)}, got {sorted(exit_prices.keys())}"
        )

    # unavailable_tickers must exactly match missing selected_tickers
    unavailable = sorted(payload.get("unavailable_tickers") or [])
    if unavailable != sorted(missing_tickers):
        raise OutcomeValidationError(
            f"OUTCOME_INCOMPLETE: unavailable_tickers must exactly match missing "
            f"selected_tickers {sorted(missing_tickers)}, got {unavailable}"
        )

    # unavailable_reasons keys must exactly match unavailable_tickers
    reasons = payload.get("unavailable_reasons") or {}
    if set(reasons.keys()) != set(unavailable):
        raise OutcomeValidationError(
            f"OUTCOME_INCOMPLETE: unavailable_reasons keys must be exactly "
            f"{unavailable}, got {sorted(reasons.keys())}"
        )

    if payload.get("portfolio_return_complete") is not False:
        raise OutcomeValidationError(
            "OUTCOME_INCOMPLETE: portfolio_return_complete must be False"
        )
    for field in ("portfolio_return_pct", "forward_alpha_vs_spy", "hit_rate_vs_spy"):
        if payload.get(field) is not None:
            raise OutcomeValidationError(f"OUTCOME_INCOMPLETE: {field} must be null")

    per_ticker_pct = payload.get("per_ticker_return_pct", {})
    # per_ticker_return_pct keys must be exactly complete_tickers
    if set(per_ticker_pct.keys()) != set(complete_tickers):
        raise OutcomeValidationError(
            f"OUTCOME_INCOMPLETE: per_ticker_return_pct keys must be exactly "
            f"{sorted(complete_tickers)}, got {sorted(per_ticker_pct.keys())}"
        )
    for t in complete_tickers:
        ep, xp = entry_prices[t], exit_prices[t]
        if not (isinstance(ep, (int, float)) and ep > 0 and math.isfinite(ep)):
            raise OutcomeValidationError(
                f"OUTCOME_INCOMPLETE: invalid entry_price for {t!r}: {ep!r}"
            )
        if not (isinstance(xp, (int, float)) and xp > 0 and math.isfinite(xp)):
            raise OutcomeValidationError(
                f"OUTCOME_INCOMPLETE: invalid exit_price for {t!r}: {xp!r}"
            )
        _check_close(
            f"per_ticker_return_pct[{t!r}]",
            ((xp - ep) / ep) * 100.0,
            per_ticker_pct.get(t),
        )

    expected_subset = sum(
        ((exit_prices[t] - entry_prices[t]) / entry_prices[t]) * 100.0
        for t in complete_tickers
    ) / len(complete_tickers)
    _check_close(
        "available_subset_return_pct",
        expected_subset,
        payload.get("available_subset_return_pct"),
    )

    for t, reason in reasons.items():
        if reason != "price_missing_after_grace":
            raise OutcomeValidationError(
                f"OUTCOME_INCOMPLETE: unavailable_reasons[{t!r}] must be "
                f"'price_missing_after_grace', got {reason!r}"
            )


def _validate_unavailable_invariants(payload: dict) -> None:
    """OUTCOME_UNAVAILABLE: no complete selected ticker, all aggregates null."""
    selected_tickers = payload.get("selected_tickers", [])
    entry_prices = payload.get("entry_prices", {})
    exit_prices = payload.get("exit_prices", {})
    per_ticker_pct = payload.get("per_ticker_return_pct", {})

    complete = [t for t in selected_tickers if t in entry_prices and t in exit_prices]
    if complete:
        raise OutcomeValidationError(
            f"OUTCOME_UNAVAILABLE: has complete tickers {complete} — "
            f"should use INCOMPLETE or RECORDED"
        )

    # Exact field checks — prices and returns must be empty
    if entry_prices:
        raise OutcomeValidationError(
            f"OUTCOME_UNAVAILABLE: entry_prices must be empty {{}}, "
            f"got keys {sorted(entry_prices.keys())}"
        )
    if exit_prices:
        raise OutcomeValidationError(
            f"OUTCOME_UNAVAILABLE: exit_prices must be empty {{}}, "
            f"got keys {sorted(exit_prices.keys())}"
        )
    if per_ticker_pct:
        raise OutcomeValidationError(
            f"OUTCOME_UNAVAILABLE: per_ticker_return_pct must be empty {{}}, "
            f"got keys {sorted(per_ticker_pct.keys())}"
        )

    # unavailable_tickers must equal selected_tickers (for non-empty)
    unavailable = sorted(payload.get("unavailable_tickers") or [])
    if selected_tickers and unavailable != sorted(selected_tickers):
        raise OutcomeValidationError(
            f"OUTCOME_UNAVAILABLE: unavailable_tickers must equal selected_tickers "
            f"{sorted(selected_tickers)}, got {unavailable}"
        )

    # unavailable_reasons keys must exactly match unavailable_tickers
    reasons = payload.get("unavailable_reasons") or {}
    if set(reasons.keys()) != set(unavailable):
        raise OutcomeValidationError(
            f"OUTCOME_UNAVAILABLE: unavailable_reasons keys must be exactly "
            f"{unavailable}, got {sorted(reasons.keys())}"
        )

    if payload.get("portfolio_return_complete") is not False:
        raise OutcomeValidationError(
            "OUTCOME_UNAVAILABLE: portfolio_return_complete must be False"
        )
    for field in (
        "portfolio_return_pct",
        "forward_alpha_vs_spy",
        "hit_rate_vs_spy",
        "available_subset_return_pct",
    ):
        if payload.get(field) is not None:
            raise OutcomeValidationError(f"OUTCOME_UNAVAILABLE: {field} must be null")

    for t, reason in reasons.items():
        if reason != "price_missing_after_grace":
            raise OutcomeValidationError(
                f"OUTCOME_UNAVAILABLE: unavailable_reasons[{t!r}] must be "
                f"'price_missing_after_grace', got {reason!r}"
            )


def _validate_terminal_payload(payload: dict, pending_event: dict) -> None:
    """Full validation of a terminal payload.  Raises OutcomeValidationError on any failure."""
    _validate_identity_fields(payload, pending_event)
    _validate_schema_fields(payload, pending_event)
    et = payload["event_type"]
    if et == "OUTCOME_RECORDED":
        _validate_recorded_invariants(payload)
    elif et == "OUTCOME_INCOMPLETE":
        _validate_incomplete_invariants(payload)
    elif et == "OUTCOME_UNAVAILABLE":
        _validate_unavailable_invariants(payload)


# ── Startup validation ────────────────────────────────────────────────────────


def validate_and_rebuild_index() -> dict[str, str]:
    """
    Scan all JSONL partitions under the global lock, verify hash chains and event
    sequences, detect duplicate keys, and rebuild the index if stale.

    Raises CorruptionError on any integrity violation (fail-closed).
    Returns the validated index dict.
    """
    OUTCOME_DIR.mkdir(parents=True, exist_ok=True)

    with _global_lock():
        new_idx: dict[str, str] = {}
        partition_for_key: dict[str, str] = {}

        for path in _all_jsonl_paths():
            partition = path.name[:7]
            all_events = _read_jsonl(path)

            by_key: dict[str, list[dict]] = {}
            for ev in all_events:
                ok = ev.get("outcome_key")
                if not ok:
                    raise CorruptionError(
                        f"v2b_outcome: event missing outcome_key in {path.name}"
                    )
                by_key.setdefault(ok, []).append(ev)

            for ok, events in by_key.items():
                if ok in partition_for_key and partition_for_key[ok] != partition:
                    raise CorruptionError(
                        f"v2b_outcome: outcome_key {ok[:16]}… appears in both "
                        f"{partition_for_key[ok]!r} and {partition!r}"
                    )
                partition_for_key[ok] = partition
                _verify_chain(events, ok, path.name)
                _validate_event_sequence(events, path.name)
                new_idx[ok] = partition

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
    Create an OUTCOME_PENDING event.  Empty selected_tickers is valid.

    Returns "IDEMPOTENT_MATCH" if identical PENDING already exists.

    Raises:
      ObservationValidationError — non-string, empty-string, or duplicate tickers
      ContentConflictError       — same key, different content
      CorruptionError            — broken chain or corrupt data
    """
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
        existing_events, _ = _read_all_events_for_key(outcome_key, idx)

        if existing_events:
            _verify_chain(existing_events, outcome_key, f"outcome_key={outcome_key[:16]}…")
            stored_pch = existing_events[0].get("pending_content_hash")
            if stored_pch == pch:
                return "IDEMPOTENT_MATCH"
            raise ContentConflictError(
                f"outcome_key {outcome_key[:16]}… already exists with different content. "
                f"stored_pch={stored_pch!r} != new_pch={pch!r}"
            )

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
            "created_at": _utc_now_iso(),
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
    missing_tickers: list[str],
    missing_benchmarks: list[str],
    provider_end_exclusive: str,
) -> dict | None:
    """
    Record that a valid yfinance response was received but price rows were missing.

    Schema v3.1: stores missing_tickers (non-SPY selected tickers absent from response),
    missing_benchmarks (e.g. ["SPY"]), and provider_end_exclusive.
    source is hardcoded to DATA_SOURCE ("yfinance_daily_ohlcv").

    Deduplicates by (outcome_key, attempt_date, sorted missing_tickers, sorted missing_benchmarks).
    Returns None if deduplicated.
    """
    canonical_mt = sorted(missing_tickers)
    canonical_mb = sorted(missing_benchmarks)

    with _global_lock():
        idx = _load_idx()
        existing_events, partition = _read_all_events_for_key(outcome_key, idx)
        if not existing_events:
            raise InvalidTransitionError(
                f"record_fetch_deferred: outcome_key {outcome_key[:16]}… not found"
            )
        _verify_chain(existing_events, outcome_key, f"outcome_key={outcome_key[:16]}…")

        status = _derive_status(existing_events)
        if status in _TERMINAL_STATES:
            raise InvalidTransitionError(
                f"record_fetch_deferred: outcome_key {outcome_key[:16]}… is already "
                f"terminal ({status!r}) — FETCH_DEFERRED not allowed"
            )

        # Dedup by (attempt_date, missing_tickers, missing_benchmarks)
        for ev in existing_events:
            if ev.get("event_type") != "OUTCOME_FETCH_DEFERRED":
                continue
            if ev.get("attempt_date") != attempt_date:
                continue
            if (
                sorted(ev.get("missing_tickers", [])) == canonical_mt
                and sorted(ev.get("missing_benchmarks", [])) == canonical_mb
            ):
                return None

        chain_tip = _get_chain_tip(existing_events)
        event_body: dict = {
            "event_type": "OUTCOME_FETCH_DEFERRED",
            "record_version": RECORD_VERSION,
            "outcome_key": outcome_key,
            "strategy_id": existing_events[0].get("strategy_id"),
            "outcome_definition_version": existing_events[0].get("outcome_definition_version"),
            "attempted_at": attempted_at,
            "attempt_date": attempt_date,
            "missing_tickers": canonical_mt,
            "missing_benchmarks": canonical_mb,
            "provider_end_exclusive": provider_end_exclusive,
            "source": DATA_SOURCE,
            "order_creation_blocked": True,
            "previous_event_hash": chain_tip,
        }
        event_body["event_hash"] = _compute_event_hash(
            {k: v for k, v in event_body.items() if k != "event_hash"}
        )

        path = _ledger_path(partition)
        _append_event_to_disk(event_body, path)

    return event_body


def record_terminal(
    outcome_key: str,
    terminal_payload: dict,
) -> dict | Literal["IDEMPOTENT_MATCH"]:
    """
    Write a terminal event (OUTCOME_RECORDED / OUTCOME_INCOMPLETE / OUTCOME_UNAVAILABLE).

    Validates terminal_payload against the original PENDING and state invariants
    before any write.  An invalid payload raises OutcomeValidationError with no write.

    Returns "IDEMPOTENT_MATCH" if identical terminal already exists.

    Raises:
      OutcomeValidationError  — payload fails validation (no write)
      ContentConflictError    — same type but different content
      InvalidTransitionError  — wrong type already written, or no PENDING found
      CorruptionError         — broken chain
    """
    event_type = terminal_payload.get("event_type", "")
    if event_type not in _TERMINAL_STATES:
        raise ValueError(
            f"record_terminal: event_type {event_type!r} is not a terminal state"
        )

    payload_ok = terminal_payload.get("outcome_key")
    if payload_ok != outcome_key:
        raise OutcomeValidationError(
            f"record_terminal: payload outcome_key {payload_ok!r} != argument {outcome_key!r}"
        )

    tch = make_terminal_content_hash(terminal_payload)

    with _global_lock():
        idx = _load_idx()
        existing_events, partition = _read_all_events_for_key(outcome_key, idx)
        if not existing_events:
            raise InvalidTransitionError(
                f"record_terminal: outcome_key {outcome_key[:16]}… not found"
            )
        _verify_chain(existing_events, outcome_key, f"outcome_key={outcome_key[:16]}…")

        status = _derive_status(existing_events)
        if status in _TERMINAL_STATES:
            terminal_ev = next(
                e for e in existing_events if e.get("event_type") in _TERMINAL_STATES
            )
            stored_type = terminal_ev.get("event_type")
            if stored_type != event_type:
                raise InvalidTransitionError(
                    f"record_terminal: outcome_key {outcome_key[:16]}… is already "
                    f"{stored_type!r} — cannot transition to {event_type!r}"
                )
            stored_tch = terminal_ev.get("terminal_content_hash")
            if stored_tch == tch:
                return "IDEMPOTENT_MATCH"
            raise ContentConflictError(
                f"record_terminal: outcome_key {outcome_key[:16]}… already has "
                f"{event_type!r} with different content."
            )

        if status != "OUTCOME_PENDING":
            raise InvalidTransitionError(
                f"record_terminal: unexpected status {status!r} — only PENDING may terminate"
            )

        first_ev = existing_events[0]
        if first_ev.get("event_type") != "OUTCOME_PENDING":
            raise CorruptionError(
                f"record_terminal: first event for {outcome_key[:16]}… is "
                f"{first_ev.get('event_type')!r}, expected OUTCOME_PENDING"
            )

        # Fail-closed guard: INCOMPLETE/UNAVAILABLE from non-empty tickers requires
        # a FETCH_DEFERRED from the same measurement_date (proof that a prior attempt
        # documented the missing prices under the global lock).
        # Empty selected_tickers is the only exception (immediate UNAVAILABLE, no fetch).
        if event_type in ("OUTCOME_INCOMPLETE", "OUTCOME_UNAVAILABLE"):
            _tickers_from_pending = first_ev.get("selected_tickers", [])
            if _tickers_from_pending:
                _measurement_date = terminal_payload.get("measurement_date")
                _deferred = [
                    e for e in existing_events
                    if e.get("event_type") == "OUTCOME_FETCH_DEFERRED"
                    and e.get("attempt_date") == _measurement_date
                ]
                if not _deferred:
                    raise OutcomeValidationError(
                        f"record_terminal: {event_type} for non-empty tickers requires "
                        f"OUTCOME_FETCH_DEFERRED with attempt_date={_measurement_date!r} "
                        f"(none found for {outcome_key[:16]}…)"
                    )
                # All claimed unavailable_tickers must be documented in same-day FETCH_DEFERREDs
                _documented: set[str] = set()
                for _ev in _deferred:
                    _documented.update(_ev.get("missing_tickers", []))
                _unavail_in_payload = set(terminal_payload.get("unavailable_tickers") or [])
                _undocumented = _unavail_in_payload - _documented
                if _undocumented:
                    raise OutcomeValidationError(
                        f"record_terminal: unavailable_tickers {sorted(_undocumented)} "
                        f"not documented in any FETCH_DEFERRED for {outcome_key[:16]}… "
                        f"(documented missing_tickers: {sorted(_documented)})"
                    )

        # Validate payload before any write (raises OutcomeValidationError on failure)
        _validate_terminal_payload(terminal_payload, first_ev)

        chain_tip = _get_chain_tip(existing_events)
        event_body: dict = dict(terminal_payload)
        event_body["record_version"] = RECORD_VERSION
        event_body["outcome_key"] = outcome_key
        event_body["terminal_content_hash"] = tch
        event_body["order_creation_blocked"] = True
        event_body["previous_event_hash"] = chain_tip
        event_body.pop("event_hash", None)
        event_body["event_hash"] = _compute_event_hash(
            {k: v for k, v in event_body.items() if k != "event_hash"}
        )

        path = _ledger_path(partition)
        _append_event_to_disk(event_body, path)

    return event_body


# ── Read API ──────────────────────────────────────────────────────────────────


def get_outcome_events(outcome_key: str) -> list[dict]:
    idx = _load_idx()
    events, _ = _read_all_events_for_key(outcome_key, idx)
    return events


def get_outcome_status(outcome_key: str) -> str | None:
    return _derive_status(get_outcome_events(outcome_key))


def list_pending_outcomes() -> list[dict]:
    """Return the initial PENDING event for every outcome still in PENDING status."""
    result: list[dict] = []
    idx = _load_idx()
    for outcome_key, partition in idx.items():
        path = _ledger_path(partition)
        all_events = _read_jsonl(path)
        key_events = [e for e in all_events if e.get("outcome_key") == outcome_key]
        if not key_events:
            continue
        if _derive_status(key_events) == "OUTCOME_PENDING":
            result.append(key_events[0])
    return result
