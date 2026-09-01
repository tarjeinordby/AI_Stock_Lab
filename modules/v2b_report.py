"""
V2B.3 — Shadow observation reporter.

Reads COMPLETED V2B observations from the ledger, builds structured shadow
reports, sends clearly labeled V2 SHADOW Telegram messages, and appends
append-only report records with content hashes.

SHADOW ONLY:
  order_creation_blocked = True is a structural invariant.
  This module never creates orders, fills, or trades.
  V1 production messages are completely unaffected.
  V2 shadow Telegram messages are labeled "V2 SHADOW — INGEN HANDLER"
  and must never be presented as buy recommendations.

Report storage:
  data_v4/v2b_reports/{YYYY-MM}_v2b_reports.jsonl
  data_v4/v2b_reports/v2b_report_idx.json

Hash chain (record_version="2"):
  Every event carries:
    event_hash           = SHA-256(canonical JSON without event_hash field)
    previous_event_hash  = event_hash of prior event for same report_key (None for first)
  Broken chains, wrong record_version, or tampered event_hash raise CorruptionError.

Telegram outbox lifecycle per report_key:
  REPORT_CREATED → SEND_CLAIMED → SEND_CONFIRMED
                                → SEND_AMBIGUOUS  (crash recovery / uncertain delivery)
                                → SEND_FAILED     (definitive failure, currently reserved)
  Delivery guarantee: at-least-once with explicit AMBIGUOUS status.
  Telegram API provides no idempotency key, so exactly-once is not achievable.

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
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Generator

from modules.v2b_ledger import (
    get_observation_events,
    list_observations,
    make_observation_key,
)

# ── Structural invariant ──────────────────────────────────────────────────────
_ORDER_CREATION_BLOCKED: bool = True

REPORT_VERSION = "shadow_report_v1"
RECORD_VERSION = "2"         # mandatory hash-chain version
SEND_CLAIM_TTL_SECONDS = 300  # 5 minutes

REPORT_DIR: Path = Path(__file__).parent.parent / "data_v4" / "v2b_reports"

_HEX64_RE = __import__("re").compile(r"^[0-9a-f]{64}$")
_SUPPORTED_RECORD_VERSIONS: frozenset[str] = frozenset({"2"})


# ── Exceptions ────────────────────────────────────────────────────────────────

class ObservationNotFoundError(Exception):
    """No COMPLETED observation found for the requested session."""


class ObservationIncompleteError(Exception):
    """Observation exists but has not reached COMPLETED status."""


class ObservationAmbiguousError(Exception):
    """Multiple COMPLETED observations for the same session — fail-closed."""


class ObservationIntegrityError(Exception):
    """Integrity check failed — observation_key mismatch or missing required fields."""


class ReportConflictError(Exception):
    """Same observation_key produced different report content — fail-closed."""


class SafetyBoundaryError(RuntimeError):
    """Raised fail-closed when a V2B safety invariant would be violated."""


class CorruptionError(RuntimeError):
    """Raised on hash mismatch, broken chain, missing mandatory field, or bad record_version."""


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ShadowReportResult:
    status: str                          # CREATED | IDEMPOTENT_MATCH | CONFLICT | FAILED
    telegram_status: str | None = None   # CONFIRMED | AMBIGUOUS | SKIPPED | NONE
    report_key: str | None = None
    observation_key: str | None = None
    intended_execution_session: str | None = None
    telegram_sent: bool = False          # True only if SEND_CONFIRMED
    telegram_skipped: bool = False
    error: str | None = None
    detail: str | None = None


# ── Safety guard ──────────────────────────────────────────────────────────────

def _assert_order_creation_blocked() -> None:
    if not _ORDER_CREATION_BLOCKED:
        raise SafetyBoundaryError(
            "V2B invariant violated: _ORDER_CREATION_BLOCKED must always be True. "
            "This reporter must never create orders, fills, or trades."
        )


# ── Canonical JSON + hashing ──────────────────────────────────────────────────

def _canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def make_report_key(report_content: dict) -> str:
    """SHA-256 of canonical report_content dict. NaN/Inf raise ValueError."""
    return _sha256(_canonical_json(report_content))


def _make_event_hash(event_body: dict) -> str:
    """SHA-256 of event body WITHOUT the event_hash field itself."""
    return _sha256(_canonical_json({k: v for k, v in event_body.items() if k != "event_hash"}))


def _now_iso(_now: Callable[[], datetime] | None = None) -> str:
    dt = _now() if _now is not None else datetime.now(timezone.utc)
    return dt.isoformat()


# ── File path helpers ─────────────────────────────────────────────────────────

def _report_jsonl_path(yyyymm: str) -> Path:
    return REPORT_DIR / f"{yyyymm}_v2b_reports.jsonl"


def _report_lock_path(yyyymm: str) -> Path:
    return REPORT_DIR / f"{yyyymm}_v2b_reports.lock"


def _report_idx_path() -> Path:
    return REPORT_DIR / "v2b_report_idx.json"


def _report_idx_lock_path() -> Path:
    return REPORT_DIR / "v2b_report_idx.lock"


# ── Lock context managers ─────────────────────────────────────────────────────

@contextmanager
def _monthly_report_lock(yyyymm: str) -> Generator[None, None, None]:
    lp = _report_lock_path(yyyymm)
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
    lp = _report_idx_lock_path()
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.touch(exist_ok=True)
    with open(lp, "r") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


# ── Hash chain integrity ──────────────────────────────────────────────────────

def _verify_report_event_structure(ev: dict, filename: str, line_num: int) -> None:
    """
    Verify structural integrity of a parsed report event dict.

    record_version must be "2". Missing or unsupported → CorruptionError.
    report_key must be a 64-char hex string.
    event_hash must be a 64-char hex string (MANDATORY).
    previous_event_hash must be present: None for first event per key, 64-char hex for rest.
    """
    rv = ev.get("record_version")
    if rv not in _SUPPORTED_RECORD_VERSIONS:
        raise CorruptionError(
            f"v2b_report: unsupported or missing record_version {rv!r} "
            f"at line {line_num} in {filename} — fail-closed"
        )

    rk = ev.get("report_key")
    if not isinstance(rk, str) or not _HEX64_RE.match(rk):
        raise CorruptionError(
            f"v2b_report: missing or invalid report_key at line {line_num} in {filename}"
        )

    event_hash = ev.get("event_hash")
    if not isinstance(event_hash, str) or not _HEX64_RE.match(event_hash):
        raise CorruptionError(
            f"v2b_report: missing or invalid event_hash at line {line_num} in {filename} "
            f"(mandatory for record_version='2')"
        )

    if "previous_event_hash" not in ev:
        raise CorruptionError(
            f"v2b_report: missing previous_event_hash at line {line_num} in {filename} "
            f"(mandatory for record_version='2')"
        )
    peh = ev["previous_event_hash"]
    if peh is not None and (not isinstance(peh, str) or not _HEX64_RE.match(peh)):
        raise CorruptionError(
            f"v2b_report: invalid previous_event_hash at line {line_num} in {filename}: "
            f"must be None or a 64-char hex string, got {peh!r}"
        )


def _verify_report_hash_chains(events: list[dict], filename: str = "") -> None:
    """Verify that previous_event_hash chains are unbroken for each report_key."""
    by_key: dict[str, list[dict]] = {}
    for ev in events:
        rk = ev.get("report_key")
        if rk:
            by_key.setdefault(rk, []).append(ev)

    for rk, chain in by_key.items():
        prev_hash: str | None = None
        for ev in chain:
            stored_prev = ev["previous_event_hash"]
            if stored_prev != prev_hash:
                raise CorruptionError(
                    f"v2b_report: hash chain broken for {rk[:16]}… in {filename}: "
                    f"expected previous_event_hash={prev_hash!r}, got={stored_prev!r}"
                )
            prev_hash = ev["event_hash"]


# ── Hardened JSONL reader ─────────────────────────────────────────────────────

def _read_report_events_raw(yyyymm: str) -> list[dict]:
    """
    Read all report events from a monthly JSONL file with full integrity verification.

    - Incomplete last line → UserWarning + skip (crash-safe truncation)
    - Mid-file JSON error → CorruptionError (fail-closed)
    - Missing/wrong record_version → CorruptionError
    - Missing mandatory field (event_hash, previous_event_hash) → CorruptionError
    - event_hash value mismatch → CorruptionError
    - Broken previous_event_hash chain → CorruptionError
    """
    path = _report_jsonl_path(yyyymm)
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
                    f"v2b_report: incomplete last line in {path.name} (line {i + 1}) — skipped",
                    UserWarning,
                    stacklevel=4,
                )
                continue
            raise CorruptionError(
                f"v2b_report: mid-file JSON parse failure at line {i + 1} in {path.name}"
            )

        _verify_report_event_structure(ev, path.name, i + 1)

        body = {k: v for k, v in ev.items() if k != "event_hash"}
        computed = _make_event_hash(body)
        if computed != ev["event_hash"]:
            raise CorruptionError(
                f"v2b_report: event_hash mismatch at line {i + 1} in {path.name}: "
                f"stored={ev['event_hash'][:16]}… computed={computed[:16]}…"
            )

        events.append(ev)

    _verify_report_hash_chains(events, path.name)
    return events


def _last_event_hash_for_key(events: list[dict], report_key: str) -> str | None:
    """Return the event_hash of the last event for a given report_key, or None."""
    last = None
    for ev in events:
        if ev.get("report_key") == report_key:
            last = ev["event_hash"]
    return last


# ── JSONL append ──────────────────────────────────────────────────────────────

def _append_report_event(yyyymm: str, event: dict) -> None:
    """Append one event to the monthly JSONL. Caller MUST hold the monthly lock."""
    path = _report_jsonl_path(yyyymm)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _canonical_json(event) + "\n"
    with open(path, "a") as af:
        af.write(line)
        af.flush()
        os.fsync(af.fileno())


def _build_and_append_event(
    yyyymm: str,
    event_body: dict[str, Any],
    prev_hash: str | None,
) -> dict:
    """
    Add record_version, previous_event_hash, and event_hash to event_body, then append.
    Caller MUST hold the monthly lock.
    Returns the complete event dict.
    """
    event_body["record_version"] = RECORD_VERSION
    event_body["previous_event_hash"] = prev_hash
    event_body["order_creation_blocked"] = _ORDER_CREATION_BLOCKED
    event_body["event_hash"] = _make_event_hash(event_body)
    _append_report_event(yyyymm, event_body)
    return event_body


# ── Index helpers ─────────────────────────────────────────────────────────────

def _load_report_idx_raw() -> dict[str, dict] | None:
    """
    Load report index without rebuilding.
    Returns {} if file does not exist.
    Returns None if file exists but is corrupt.
    Never raises.
    """
    p = _report_idx_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _rebuild_report_idx() -> dict[str, dict]:
    """
    Rebuild the report index by scanning all JSONL files.
    Raises CorruptionError if the same observation_key appears in multiple partitions.
    """
    idx: dict[str, dict] = {}
    for jsonl_path in sorted(REPORT_DIR.glob("*_v2b_reports.jsonl")):
        yyyymm = jsonl_path.name[:7]
        events = _read_report_events_raw(yyyymm)
        for ev in events:
            if ev.get("event_type") == "REPORT_CREATED":
                obs_key = ev.get("observation_key", "")
                rk = ev.get("report_key", "")
                if obs_key:
                    if obs_key in idx and idx[obs_key].get("report_key") != rk:
                        raise CorruptionError(
                            f"v2b_report: observation_key {obs_key[:16]}… found with different "
                            f"report_keys in multiple partitions — data integrity violation"
                        )
                    idx[obs_key] = {"yyyymm": yyyymm, "report_key": rk}
    return idx


def _load_report_idx() -> dict[str, dict]:
    """Load report index. Falls back to rebuild if corrupt. Returns {} on empty."""
    result = _load_report_idx_raw()
    if result is not None:
        return result
    return _rebuild_report_idx()


def _save_report_idx_atomic(idx: dict[str, dict]) -> None:
    """Atomically save report index via temp-file + rename."""
    p = _report_idx_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(idx, sort_keys=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=p.parent, prefix=".ridx_")
    try:
        with os.fdopen(tmp_fd, "w") as tf:
            tf.write(content)
            tf.flush()
            os.fsync(tf.fileno())
        Path(tmp_name).replace(p)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _update_report_idx(observation_key: str, yyyymm: str, report_key: str) -> None:
    """Under exclusive index lock, record (observation_key → yyyymm, report_key)."""
    with _idx_lock_ctx():
        raw = _load_report_idx_raw()
        if raw is None:
            idx = _rebuild_report_idx()
        else:
            idx = raw
        idx[observation_key] = {"yyyymm": yyyymm, "report_key": report_key}
        _save_report_idx_atomic(idx)


def _validate_index_entry(obs_key: str, entry: dict) -> bool:
    """Return True if the index entry is valid and points to a JSONL that contains obs_key."""
    yyyymm = entry.get("yyyymm", "")
    path = _report_jsonl_path(yyyymm)
    if not path.exists():
        return False
    try:
        events = _read_report_events_raw(yyyymm)
    except CorruptionError:
        return False
    return any(
        e.get("event_type") == "REPORT_CREATED" and e.get("observation_key") == obs_key
        for e in events
    )


# ── Observation reader ────────────────────────────────────────────────────────

def read_completed_observation(session: str) -> dict:
    """
    Find the single COMPLETED observation for the given intended_execution_session.

    Returns the full OBSERVATION_CREATED event dict.

    Raises ObservationNotFoundError if no COMPLETED observation exists for the session.
    Raises ObservationAmbiguousError if multiple COMPLETED observations exist (fail-closed).
    Raises ObservationIntegrityError if the event is missing or malformed.

    event_hash integrity is verified by v2b_ledger's _read_events_raw during
    get_observation_events() — tampered events raise CorruptionError there.
    """
    _assert_order_creation_blocked()
    observations = list_observations()
    matching = [
        o for o in observations
        if o.get("intended_execution_session") == session
    ]

    if not matching:
        raise ObservationNotFoundError(
            f"No V2B observation found for session {session!r}. "
            "Run v2b_daily_shadow.py first."
        )

    completed = [o for o in matching if o.get("status") == "COMPLETED"]

    if not completed:
        statuses = [o.get("status") for o in matching]
        raise ObservationIncompleteError(
            f"V2B observation for session {session!r} is not COMPLETED. "
            f"Current status(es): {statuses}"
        )

    if len(completed) > 1:
        keys = [o.get("observation_key", "?")[:16] + "…" for o in completed]
        raise ObservationAmbiguousError(
            f"{len(completed)} COMPLETED observations for session {session!r} — fail-closed. "
            f"Keys: {keys}. Manual review required."
        )

    obs_summary = completed[0]
    obs_key = obs_summary["observation_key"]

    events = get_observation_events(obs_key)
    created_event = next(
        (e for e in events if e["event_type"] == "OBSERVATION_CREATED"),
        None,
    )
    if created_event is None:
        raise ObservationIntegrityError(
            f"OBSERVATION_CREATED event missing for key {obs_key[:16]}…"
        )

    return created_event


def validate_observation_integrity(obs_event: dict, session: str) -> None:
    """
    Verify application-level invariants on the observation event.

    Checks:
    1. intended_execution_session in event matches requested session
    2. observation_key matches make_observation_key(model_version, model_config_hash, session)

    event_hash integrity is already verified by v2b_ledger during reading.
    """
    event_session = obs_event.get("intended_execution_session")
    if event_session != session:
        raise ObservationIntegrityError(
            f"Session mismatch: requested {session!r}, event has {event_session!r}"
        )

    obs_key = obs_event.get("observation_key")
    model_version = obs_event.get("model_version")
    model_config_hash = obs_event.get("model_config_hash")

    if not obs_key or not model_version or not model_config_hash:
        raise ObservationIntegrityError(
            "Missing required fields in OBSERVATION_CREATED event: "
            "observation_key, model_version, or model_config_hash"
        )

    expected_key = make_observation_key(model_version, model_config_hash, session)
    if expected_key != obs_key:
        raise ObservationIntegrityError(
            f"observation_key does not match derivation formula: "
            f"expected={expected_key[:16]}… stored={obs_key[:16]}…"
        )


# ── Report builder ────────────────────────────────────────────────────────────

def build_report_content(obs_event: dict) -> dict:
    """
    Build structured report content from a COMPLETED OBSERVATION_CREATED event.

    agreement_tickers = explicit intersection of factor_only_selected and claude_shadow_selected.
    Raises ValueError if Claude has selected tickers outside the factor selection
    (design invariant from V2B.2 — should never happen in production).
    """
    selected_per_strategy = obs_event.get("selected_tickers_per_strategy", {})
    factor_only_selected = selected_per_strategy.get("Factor_Only_Core_V2", [])
    claude_selected = selected_per_strategy.get("Factor_Plus_Claude_Shadow_V2", [])

    fo_set = set(factor_only_selected)
    cs_set = set(claude_selected)

    # Validate Claude subset invariant — fail-closed if violated
    claude_outside_factor = sorted(cs_set - fo_set)
    if claude_outside_factor:
        raise ValueError(
            f"Claude selected tickers not in factor selection (design invariant violated): "
            f"{claude_outside_factor}"
        )

    # Explicit intersection — correct even if cs_set were not a subset
    agreement_tickers = sorted(fo_set & cs_set)
    factor_only_not_in_claude = sorted(fo_set - cs_set)

    # Factor coverage: mean across all ticker records with valid coverage
    ticker_records = obs_event.get("ticker_records", [])
    factor_coverages = [
        float(r["factor_coverage"])
        for r in ticker_records
        if isinstance(r.get("factor_coverage"), (int, float))
        and r["factor_coverage"] is not None
    ]
    factor_coverage_mean = (
        round(sum(factor_coverages) / len(factor_coverages), 4)
        if factor_coverages else None
    )

    metadata = obs_event.get("metadata", {})
    claude_meta = metadata.get("claude_shadow", {})

    return {
        "observation_key": obs_event.get("observation_key"),
        "observation_run_id": obs_event.get("observation_run_id"),
        "content_hash": obs_event.get("content_hash"),
        "intended_execution_session": obs_event.get("intended_execution_session"),
        "model_version": obs_event.get("model_version", ""),
        "model_config_hash": obs_event.get("model_config_hash", ""),
        "portfolio_version": metadata.get("portfolio_version", ""),
        "portfolio_config_hash": metadata.get("portfolio_config_hash", ""),
        "factor_only_selected": sorted(factor_only_selected),
        "claude_shadow_selected": sorted(claude_selected),
        "agreement_tickers": agreement_tickers,
        "factor_only_not_in_claude": factor_only_not_in_claude,
        "factor_coverage_mean": factor_coverage_mean,
        "stale_tickers": sorted(metadata.get("stale_tickers", [])),
        "universe_count": obs_event.get("universe_count", 0),
        "valid_ticker_count": obs_event.get("valid_ticker_count", 0),
        "stale_ticker_count": obs_event.get("stale_ticker_count", 0),
        "excluded_ticker_count": obs_event.get("excluded_ticker_count", 0),
        "missing_ticker_count": obs_event.get("missing_ticker_count", 0),
        "signal_coverage_rate": obs_event.get("signal_coverage_rate", 0.0),
        "data_quality_status": obs_event.get("data_quality_status", "unknown"),
        "claude_shadow_status": claude_meta.get("status", "not_collected"),
        "claude_ok_count": claude_meta.get("ok_ticker_count", 0),
    }


# ── Report ledger ─────────────────────────────────────────────────────────────

def create_report(obs_event: dict) -> ShadowReportResult:
    """
    Append a shadow report record for the given observation event.

    Verifies under the exclusive monthly lock — does NOT trust the index alone.
    Idempotent: same report_key → IDEMPOTENT_MATCH (no write).
    Conflict: same observation_key, different report_key → ReportConflictError.

    Returns ShadowReportResult with status CREATED or IDEMPOTENT_MATCH.
    """
    _assert_order_creation_blocked()

    obs_key = obs_event.get("observation_key", "")
    obs_run_id = obs_event.get("observation_run_id", "")
    session = obs_event.get("intended_execution_session", "")
    yyyymm = session[:7] if len(session) >= 7 else "unknown"

    report_content = build_report_content(obs_event)
    report_key = make_report_key(report_content)

    with _monthly_report_lock(yyyymm):
        # Under lock: verify directly from JSONL, don't trust index
        events = _read_report_events_raw(yyyymm)
        existing = next(
            (e for e in events
             if e.get("event_type") == "REPORT_CREATED"
             and e.get("observation_key") == obs_key),
            None,
        )

        if existing is not None:
            existing_key = existing.get("report_key")
            if existing_key == report_key:
                _update_report_idx(obs_key, yyyymm, report_key)
                return ShadowReportResult(
                    status="IDEMPOTENT_MATCH",
                    report_key=report_key,
                    observation_key=obs_key,
                    intended_execution_session=session,
                )
            else:
                raise ReportConflictError(
                    f"Report conflict for observation {obs_key[:16]}…: "
                    f"existing report_key={existing_key[:16] if existing_key else '?'}…, "
                    f"new report_key={report_key[:16]}…"
                )

        prev_hash = _last_event_hash_for_key(events, report_key)

        event_body: dict[str, Any] = {
            "event_type": "REPORT_CREATED",
            "report_version": REPORT_VERSION,
            "report_key": report_key,
            "observation_key": obs_key,
            "observation_run_id": obs_run_id,
            "intended_execution_session": session,
            "generated_at": _now_iso(),
            "report_content": report_content,
        }
        _build_and_append_event(yyyymm, event_body, prev_hash)

    _update_report_idx(obs_key, yyyymm, report_key)

    return ShadowReportResult(
        status="CREATED",
        report_key=report_key,
        observation_key=obs_key,
        intended_execution_session=session,
    )


# ── Telegram outbox ───────────────────────────────────────────────────────────
#
# Delivery model: at-least-once with bounded duplicates.
#
# Lifecycle per report_key (append-only):
#   REPORT_CREATED
#   → SEND_CLAIMED(claim_id, expires_at)   — lease acquired before sending
#   → SEND_CONFIRMED(claim_id)             — delivered; no further send needed
#   → SEND_AMBIGUOUS(original_claim_id)    — outcome uncertain; retry allowed
#   → SEND_FAILED(claim_id)               — reserved for definitive failure (unused)
#
# After SEND_AMBIGUOUS, a new SEND_CLAIMED may be written to retry.
# Duplicate risk: if SEND_CONFIRMED was not written after a successful send
# (crash between send() and the lock), the next retry sends again.
# This is the at-least-once guarantee — no manual review required for AMBIGUOUS.
#
# Deterministic reducer (_reduce_send_state):
#   - Processes ALL events in order.
#   - The "active claim" is the LAST SEND_CLAIMED event.
#   - The state is determined by whether the active claim has a terminal event.
#   - An old SEND_AMBIGUOUS for a prior claim does NOT block a newer SEND_CONFIRMED.
#   - Invalid chains (duplicate claim_id, unknown claim_id in terminal, contradictory
#     terminals) raise CorruptionError — fail-closed.

def _reduce_send_state(
    events_for_key: list[dict],
) -> tuple[str | None, dict | None]:
    """
    Deterministic reducer: reconstruct current send state from the full event history
    for a single report_key.

    Returns (status, active_claim_event) where status is:
      None             — no send activity yet
      "SEND_CLAIMED"   — active claim exists, no terminal event yet
      "SEND_CONFIRMED" — active claim has been confirmed (message delivered)
      "SEND_AMBIGUOUS" — active claim was marked ambiguous (retryable)
      "SEND_FAILED"    — active claim definitively failed (reserved)

    The active claim is the LAST SEND_CLAIMED event in the history.
    A terminal event for an OLDER claim has NO effect on the state of a NEWER claim.

    Raises CorruptionError on:
      - SEND_CLAIMED with missing claim_id
      - Duplicate claim_id across SEND_CLAIMED events
      - SEND_CONFIRMED/SEND_FAILED with missing or unknown claim_id
      - SEND_AMBIGUOUS with missing or unknown original_claim_id
      - Contradictory terminals for the same claim_id (e.g. CONFIRMED then AMBIGUOUS)
    """
    claims: dict[str, dict] = {}          # claim_id → SEND_CLAIMED event
    terminals: dict[str, str] = {}        # claim_id → terminal event type
    last_claim_ev: dict | None = None

    for ev in events_for_key:
        et = ev.get("event_type", "")

        if et == "SEND_CLAIMED":
            cid = ev.get("claim_id")
            if not cid:
                raise CorruptionError(
                    "v2b_report: SEND_CLAIMED event missing claim_id"
                )
            if cid in claims:
                raise CorruptionError(
                    f"v2b_report: duplicate claim_id {cid!r} in SEND_CLAIMED events"
                )
            claims[cid] = ev
            last_claim_ev = ev

        elif et in ("SEND_CONFIRMED", "SEND_FAILED"):
            cid = ev.get("claim_id")
            if not cid:
                raise CorruptionError(
                    f"v2b_report: {et} event missing claim_id"
                )
            if cid not in claims:
                raise CorruptionError(
                    f"v2b_report: {et} references unknown claim_id {cid!r} — "
                    "no matching SEND_CLAIMED found"
                )
            existing = terminals.get(cid)
            if existing is not None and existing != et:
                raise CorruptionError(
                    f"v2b_report: contradictory terminals for claim_id {cid!r}: "
                    f"{existing!r} then {et!r}"
                )
            terminals[cid] = et  # idempotent for same type

        elif et == "SEND_AMBIGUOUS":
            cid = ev.get("original_claim_id")
            if not cid:
                raise CorruptionError(
                    "v2b_report: SEND_AMBIGUOUS event missing original_claim_id"
                )
            if cid not in claims:
                raise CorruptionError(
                    f"v2b_report: SEND_AMBIGUOUS references unknown claim_id {cid!r} — "
                    "no matching SEND_CLAIMED found"
                )
            existing = terminals.get(cid)
            if existing is not None and existing != "SEND_AMBIGUOUS":
                raise CorruptionError(
                    f"v2b_report: contradictory terminals for claim_id {cid!r}: "
                    f"{existing!r} then SEND_AMBIGUOUS"
                )
            terminals[cid] = "SEND_AMBIGUOUS"

    if last_claim_ev is None:
        return None, None

    active_cid = last_claim_ev.get("claim_id")
    terminal = terminals.get(active_cid)
    if terminal is not None:
        return terminal, last_claim_ev
    return "SEND_CLAIMED", last_claim_ev


def _is_claim_expired(claim_ev: dict, _now: Callable[[], datetime] | None = None) -> bool:
    """Return True if the SEND_CLAIMED event's lease has expired."""
    expires_at = claim_ev.get("claim_expires_at", "")
    if not expires_at:
        return True
    try:
        exp_dt = datetime.fromisoformat(expires_at)
        now_dt = _now() if _now is not None else datetime.now(timezone.utc)
        return now_dt >= exp_dt
    except Exception:
        return True


def _events_for_report_key(all_events: list[dict], report_key: str) -> list[dict]:
    return [e for e in all_events if e.get("report_key") == report_key]


def _claim_send_slot(
    report_key: str,
    obs_key: str,
    session: str,
    yyyymm: str,
    _now: Callable[[], datetime] | None = None,
) -> str | None:
    """
    Under monthly lock: check send state and write SEND_CLAIMED if safe.

    Returns new claim_id when the slot was successfully claimed.
    Returns None when:
      - state is SEND_CONFIRMED (already sent — do not retry)
      - state is SEND_CLAIMED with a live (non-expired) lease (parallel run active)

    Retryable states that produce a new claim_id:
      - No prior activity (None)
      - SEND_AMBIGUOUS (prior attempt uncertain — at-least-once retry)
      - SEND_FAILED    (prior attempt failed — retry)
      - SEND_CLAIMED but lease expired → write SEND_AMBIGUOUS for expired claim,
        then write new SEND_CLAIMED

    Raises CorruptionError if the event chain is structurally invalid.
    """
    with _monthly_report_lock(yyyymm):
        all_events = _read_report_events_raw(yyyymm)
        key_events = _events_for_report_key(all_events, report_key)
        state, active_claim_ev = _reduce_send_state(key_events)

        if state == "SEND_CONFIRMED":
            return None  # Already delivered — do not re-send

        if state == "SEND_CLAIMED" and active_claim_ev is not None:
            if not _is_claim_expired(active_claim_ev, _now):
                return None  # Another process holds a live lease — do not send
            # Expired lease: close it as AMBIGUOUS, then proceed to new claim
            expired_cid = active_claim_ev.get("claim_id")
            prev_hash = _last_event_hash_for_key(all_events, report_key)
            ambiguous_body: dict[str, Any] = {
                "event_type": "SEND_AMBIGUOUS",
                "report_key": report_key,
                "observation_key": obs_key,
                "intended_execution_session": session,
                "original_claim_id": expired_cid,
                "reason": "claim_expired_with_no_confirmation — possible delivery",
                "resolved_at": _now_iso(_now),
            }
            _build_and_append_event(yyyymm, ambiguous_body, prev_hash)
            # Re-read to get updated chain for the new SEND_CLAIMED
            all_events = _read_report_events_raw(yyyymm)

        # state is None, SEND_AMBIGUOUS, SEND_FAILED, or just closed expired claim
        # All allow a new SEND_CLAIMED
        now_dt = _now() if _now is not None else datetime.now(timezone.utc)
        expires_dt = now_dt + timedelta(seconds=SEND_CLAIM_TTL_SECONDS)
        claim_id = str(uuid.uuid4())
        prev_hash = _last_event_hash_for_key(all_events, report_key)
        claim_body: dict[str, Any] = {
            "event_type": "SEND_CLAIMED",
            "report_key": report_key,
            "observation_key": obs_key,
            "intended_execution_session": session,
            "claim_id": claim_id,
            "claimed_at": now_dt.isoformat(),
            "claim_expires_at": expires_dt.isoformat(),
        }
        _build_and_append_event(yyyymm, claim_body, prev_hash)

    return claim_id


def _confirm_send(
    report_key: str,
    claim_id: str,
    obs_key: str,
    session: str,
    yyyymm: str,
    _now: Callable[[], datetime] | None = None,
) -> None:
    """
    Under monthly lock: write SEND_CONFIRMED for the given claim_id.

    Validates:
      - claim_id must be the active (last) claim in the event history
      - active claim must not already have a terminal event
    Idempotent: if SEND_CONFIRMED already exists for this claim_id, return without writing.
    Rejects stale workers: raises ValueError if claim_id was superseded by a newer claim.
    Raises CorruptionError if the chain is structurally invalid.
    """
    with _monthly_report_lock(yyyymm):
        all_events = _read_report_events_raw(yyyymm)
        key_events = _events_for_report_key(all_events, report_key)
        state, active_claim_ev = _reduce_send_state(key_events)

        active_cid = active_claim_ev.get("claim_id") if active_claim_ev else None

        # Idempotent: same claim_id already confirmed
        if state == "SEND_CONFIRMED" and active_cid == claim_id:
            return

        # Stale or foreign claim_id — this worker was superseded
        if active_cid != claim_id:
            raise ValueError(
                f"_confirm_send: claim_id {claim_id!r} is not the active claim "
                f"(active: {active_cid!r}, state: {state!r}) — "
                "stale or superseded worker rejected"
            )

        # Active claim already has a non-CLAIMED terminal — contradictory confirmation
        if state not in ("SEND_CLAIMED",):
            raise CorruptionError(
                f"_confirm_send: active claim {claim_id!r} already has terminal "
                f"state {state!r} — cannot confirm"
            )

        prev_hash = _last_event_hash_for_key(all_events, report_key)
        confirmed_body: dict[str, Any] = {
            "event_type": "SEND_CONFIRMED",
            "report_key": report_key,
            "observation_key": obs_key,
            "intended_execution_session": session,
            "claim_id": claim_id,
            "confirmed_at": _now_iso(_now),
        }
        _build_and_append_event(yyyymm, confirmed_body, prev_hash)


def _mark_send_ambiguous(
    report_key: str,
    claim_id: str,
    reason: str,
    obs_key: str,
    session: str,
    yyyymm: str,
    _now: Callable[[], datetime] | None = None,
) -> None:
    """
    Under monthly lock: write SEND_AMBIGUOUS for the given claim_id.

    State-aware:
      - claim_id must be the active (last) claim
      - Idempotent: SEND_AMBIGUOUS already written for this claim_id → return
      - Forbidden: SEND_CONFIRMED → AMBIGUOUS (raises ValueError)
      - Stale/superseded claim_id raises ValueError
      - Corrupt chain raises CorruptionError
    """
    with _monthly_report_lock(yyyymm):
        all_events = _read_report_events_raw(yyyymm)
        key_events = _events_for_report_key(all_events, report_key)
        state, active_ev = _reduce_send_state(key_events)

        active_cid = active_ev.get("claim_id") if active_ev else None

        if active_cid != claim_id:
            raise ValueError(
                f"_mark_send_ambiguous: claim_id {claim_id!r} is not the active "
                f"claim (active: {active_cid!r}, state: {state!r}) — "
                "stale or superseded"
            )

        if state == "SEND_CONFIRMED":
            raise ValueError(
                f"_mark_send_ambiguous: claim {claim_id!r} is already SEND_CONFIRMED — "
                "CONFIRMED → AMBIGUOUS is forbidden"
            )

        if state == "SEND_AMBIGUOUS":
            return  # Idempotent

        if state != "SEND_CLAIMED":
            raise CorruptionError(
                f"_mark_send_ambiguous: unexpected state {state!r} for claim {claim_id!r}"
            )

        prev_hash = _last_event_hash_for_key(all_events, report_key)
        body: dict[str, Any] = {
            "event_type": "SEND_AMBIGUOUS",
            "report_key": report_key,
            "observation_key": obs_key,
            "intended_execution_session": session,
            "original_claim_id": claim_id,
            "reason": reason,
            "resolved_at": _now_iso(_now),
        }
        _build_and_append_event(yyyymm, body, prev_hash)


def _reconcile_after_confirm_failure(
    report_key: str,
    claim_id: str,
    reason: str,
    obs_key: str,
    session: str,
    yyyymm: str,
    _now: Callable[[], datetime] | None = None,
) -> str:
    """
    Called when _confirm_send() raised AFTER Telegram was already sent.

    Under monthly lock: reads current ledger state and reconciles:
      - SEND_CONFIRMED already written (write succeeded, exception was post-write):
        return "CONFIRMED" — no SEND_AMBIGUOUS needed
      - SEND_CLAIMED still open (write failed before or during append):
        write SEND_AMBIGUOUS, return "AMBIGUOUS"
      - SEND_AMBIGUOUS already written (concurrent reconciliation):
        return "AMBIGUOUS" (idempotent)
      - Stale/superseded claim_id: raises ValueError (fail-closed)
      - Corrupt chain: raises CorruptionError (fail-closed)

    Returns "CONFIRMED" | "AMBIGUOUS".
    """
    with _monthly_report_lock(yyyymm):
        all_events = _read_report_events_raw(yyyymm)
        key_events = _events_for_report_key(all_events, report_key)
        state, active_ev = _reduce_send_state(key_events)

        active_cid = active_ev.get("claim_id") if active_ev else None

        if active_cid != claim_id:
            raise ValueError(
                f"_reconcile: claim_id {claim_id!r} is not the active claim "
                f"(active: {active_cid!r}, state: {state!r}) — "
                "stale or superseded; cannot reconcile"
            )

        if state == "SEND_CONFIRMED":
            return "CONFIRMED"  # Write succeeded before the exception — no AMBIGUOUS

        if state == "SEND_AMBIGUOUS":
            return "AMBIGUOUS"  # Already reconciled by a concurrent call

        if state != "SEND_CLAIMED":
            raise CorruptionError(
                f"_reconcile: unexpected state {state!r} for claim {claim_id!r}"
            )

        # Write failed before or during append — delivery uncertain
        prev_hash = _last_event_hash_for_key(all_events, report_key)
        body: dict[str, Any] = {
            "event_type": "SEND_AMBIGUOUS",
            "report_key": report_key,
            "observation_key": obs_key,
            "intended_execution_session": session,
            "original_claim_id": claim_id,
            "reason": f"confirm_failed_after_send — {reason}",
            "resolved_at": _now_iso(_now),
        }
        _build_and_append_event(yyyymm, body, prev_hash)
        return "AMBIGUOUS"


def record_telegram_sent(report_key: str, obs_key: str, session: str) -> None:
    """
    Compatibility shim: claim + confirm for an already-sent message (used by tests).
    Uses the full outbox protocol so the event chain remains valid.
    """
    yyyymm = session[:7] if len(session) >= 7 else "unknown"
    claim_id = _claim_send_slot(report_key, obs_key, session, yyyymm)
    if claim_id is not None:
        _confirm_send(report_key, claim_id, obs_key, session, yyyymm)


def telegram_already_sent(report_key: str, yyyymm: str) -> bool:
    """
    Return True if SEND_CONFIRMED exists for this report_key.

    Raises CorruptionError if the event chain is invalid.
    NEVER returns False on corruption — a corrupt ledger must not allow
    a send attempt that could produce a duplicate message.
    """
    # No try/except: CorruptionError must propagate to prevent duplicate send
    events = _read_report_events_raw(yyyymm)
    key_events = _events_for_report_key(events, report_key)
    state, _ = _reduce_send_state(key_events)
    return state == "SEND_CONFIRMED"


# ── Telegram message formatter ────────────────────────────────────────────────

def format_telegram_message(report_content: dict) -> str:
    """
    Format a labeled V2 SHADOW Telegram message.

    Clearly marked "V2 SHADOW — INGEN HANDLER".
    Never a buy recommendation. V1 production messages completely unaffected.
    """
    session = report_content.get("intended_execution_session", "ukjent")
    obs_key = report_content.get("observation_key", "")
    obs_key_short = obs_key[:16] + "…" if len(obs_key) >= 16 else obs_key

    fo_selected = report_content.get("factor_only_selected", [])
    cs_selected = report_content.get("claude_shadow_selected", [])
    agreement = report_content.get("agreement_tickers", [])
    not_in_claude = report_content.get("factor_only_not_in_claude", [])

    fo_count = len(fo_selected)
    cs_count = len(cs_selected)
    agree_count = len(agreement)
    removed_count = len(not_in_claude)

    coverage_mean = report_content.get("factor_coverage_mean")
    coverage_pct = f"{coverage_mean * 100:.1f}%" if coverage_mean is not None else "n/a"

    dq = report_content.get("data_quality_status", "ukjent")
    stale = report_content.get("stale_ticker_count", 0)
    missing = report_content.get("missing_ticker_count", 0)
    excluded = report_content.get("excluded_ticker_count", 0)

    model_v = report_content.get("model_version", "ukjent")
    port_v = report_content.get("portfolio_version", "ukjent")

    claude_status = report_content.get("claude_shadow_status", "not_collected")
    claude_ok = report_content.get("claude_ok_count", 0)

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🔬 V2 SHADOW — INGEN HANDLER",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"Session:     {session}",
        f"Nøkkel:      {obs_key_short}",
        "",
        f"📊 Factor Only — {fo_count} tickere",
    ]

    if fo_selected:
        ticker_str = ", ".join(fo_selected[:10])
        if len(fo_selected) > 10:
            ticker_str += f", … (+{len(fo_selected) - 10})"
        lines.append(f"   {ticker_str}")

    lines.append("")
    if claude_status == "not_collected":
        lines.append("📊 Factor + Claude Shadow — ikke samlet inn")
    else:
        lines.append(f"📊 Factor + Claude Shadow — {cs_count} tickere")
        lines.append(f"   ✅ Enighet:                {agree_count} tickere")
        lines.append(f"   ❌ Fjernet av Claude:       {removed_count} tickere")
        if claude_ok > cs_count:
            lines.append(f"   ℹ️  Claude-svar mottatt:    {claude_ok} (kun utvalgte inkl.)")
        if not_in_claude:
            removed_str = ", ".join(not_in_claude[:5])
            if len(not_in_claude) > 5:
                removed_str += f", … (+{len(not_in_claude) - 5})"
            lines.append(f"   Fjernet: {removed_str}")

    lines += [
        "",
        "⚙️  Datakvalitet",
        f"   Factor coverage: {coverage_pct}",
        f"   Data quality:    {dq}",
        f"   Stale: {stale}  |  Mangler: {missing}  |  Ekskludert: {excluded}",
        "",
        "⚙️  Konfigurasjon",
        f"   Model:     {model_v}",
        f"   Portfolio: {port_v}",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "⚠️  V2 skyggerapport. Ingen ordre er plassert. V1-produksjon er urørt.",
    ]

    return "\n".join(lines)


# ── Top-level entry point ─────────────────────────────────────────────────────

def run_shadow_report(
    session: str,
    *,
    send_telegram_fn: Callable[[str], None] | None = None,
    _now: Callable[[], datetime] | None = None,
) -> ShadowReportResult:
    """
    Full shadow reporting flow for a given intended_execution_session.

    Delivery guarantee: at-least-once with SEND_AMBIGUOUS on uncertainty.
    Telegram API has no idempotency key — exactly-once is not achievable.

    Returns ShadowReportResult; never raises.
    V1 production messages are never modified or replaced.
    """
    _assert_order_creation_blocked()

    try:
        obs_event = read_completed_observation(session)
    except (ObservationNotFoundError, ObservationIncompleteError,
            ObservationAmbiguousError) as exc:
        return ShadowReportResult(
            status="FAILED",
            intended_execution_session=session,
            error=str(exc),
        )

    try:
        validate_observation_integrity(obs_event, session)
    except ObservationIntegrityError as exc:
        return ShadowReportResult(
            status="FAILED",
            observation_key=obs_event.get("observation_key"),
            intended_execution_session=session,
            error=f"Integrity check failed: {exc}",
        )

    try:
        result = create_report(obs_event)
    except ReportConflictError as exc:
        return ShadowReportResult(
            status="CONFLICT",
            observation_key=obs_event.get("observation_key"),
            intended_execution_session=session,
            error=str(exc),
        )

    if send_telegram_fn is None:
        result.telegram_status = "NONE"
        return result

    yyyymm = session[:7] if len(session) >= 7 else "unknown"
    report_key = result.report_key
    obs_key = result.observation_key or ""

    # Outbox protocol: claim → send → confirm/ambiguous
    try:
        claim_id = _claim_send_slot(report_key, obs_key, session, yyyymm, _now)
    except CorruptionError as exc:
        # Corrupt ledger: fail-closed — do not attempt send
        result.detail = f"Corrupt report ledger — Telegram blocked: {exc}"
        result.telegram_status = "FAILED"
        return result

    if claim_id is None:
        # Already confirmed or another process holds a live lease
        result.telegram_skipped = True
        result.telegram_status = "SKIPPED"
        return result

    # Phase 1: Build report content — local, no side-effects; failure is safe to abort
    try:
        report_content = build_report_content(obs_event)
        message = format_telegram_message(report_content)
    except Exception as exc:
        result.detail = f"Report build failed before send: {exc}"
        result.telegram_status = "FAILED"
        return result

    # Phase 2: Send to Telegram — after this point delivery is uncertain on any exception
    send_exc: BaseException | None = None
    try:
        send_telegram_fn(message)
    except Exception as exc:
        send_exc = exc

    if send_exc is not None:
        # Message was NOT sent (exception before delivery). Write SEND_AMBIGUOUS.
        try:
            _mark_send_ambiguous(
                report_key, claim_id, str(send_exc), obs_key, session, yyyymm, _now
            )
        except Exception as mark_exc:
            result.detail = (
                f"Telegram send failed AND AMBIGUOUS write failed — "
                f"send: {send_exc!r}  mark: {mark_exc!r}"
            )
            result.telegram_status = "AMBIGUOUS"
            return result
        result.detail = f"Telegram send failed (AMBIGUOUS): {send_exc}"
        result.telegram_status = "AMBIGUOUS"
        return result

    # Phase 3: Confirm — message was delivered; record it under lock
    try:
        _confirm_send(report_key, claim_id, obs_key, session, yyyymm, _now)
        result.telegram_sent = True
        result.telegram_status = "CONFIRMED"
    except Exception as confirm_exc:
        # _confirm_send threw after Telegram was already sent.
        # Reconcile under lock: read current state and write AMBIGUOUS only if needed.
        try:
            reconciled = _reconcile_after_confirm_failure(
                report_key, claim_id, str(confirm_exc),
                obs_key, session, yyyymm, _now,
            )
            result.telegram_sent = True  # Message WAS delivered to Telegram
            result.telegram_status = reconciled
            if reconciled == "AMBIGUOUS":
                result.detail = (
                    f"Send succeeded but confirm failed — reconciled as AMBIGUOUS: "
                    f"{confirm_exc}"
                )
        except Exception as reconcile_exc:
            result.telegram_sent = True
            result.telegram_status = "AMBIGUOUS"
            result.detail = (
                f"Send succeeded, confirm failed, reconciliation failed — "
                f"confirm: {confirm_exc!r}  reconcile: {reconcile_exc!r}"
            )

    return result
