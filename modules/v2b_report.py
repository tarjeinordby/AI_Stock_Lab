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

Isolation: zero imports from V1 execution modules.
  Does NOT import: modules.portfolio, modules.orders, modules.fills,
                   modules.ledger, modules.state
  Does NOT call:   execute_buy, execute_sell, execute_pyramid_fill
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
REPORT_DIR: Path = Path(__file__).parent.parent / "data_v4" / "v2b_reports"


# ── Exceptions ────────────────────────────────────────────────────────────────

class ObservationNotFoundError(Exception):
    """No observation of any status exists for the requested session."""


class ObservationIncompleteError(Exception):
    """Observation exists for the session but has not reached COMPLETED status."""


class ObservationIntegrityError(Exception):
    """Integrity check failed — observation_key mismatch or missing required fields."""


class ReportConflictError(Exception):
    """Same observation_key produced different report content — fail-closed."""


class SafetyBoundaryError(RuntimeError):
    """Raised fail-closed when a V2B safety invariant would be violated."""


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ShadowReportResult:
    status: str  # CREATED | IDEMPOTENT_MATCH | CONFLICT | FAILED
    report_key: str | None = None
    observation_key: str | None = None
    intended_execution_session: str | None = None
    telegram_sent: bool = False
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
    """SHA-256 of event body excluding the event_hash field."""
    return _sha256(_canonical_json({k: v for k, v in event_body.items() if k != "event_hash"}))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


# ── Index helpers ─────────────────────────────────────────────────────────────

def _load_report_idx() -> dict[str, dict]:
    """Load report index. Returns {} on any error."""
    p = _report_idx_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


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
    """Under exclusive index lock, record the (observation_key → yyyymm, report_key) mapping."""
    with _idx_lock_ctx():
        idx = _load_report_idx()
        idx[observation_key] = {"yyyymm": yyyymm, "report_key": report_key}
        _save_report_idx_atomic(idx)


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


def _read_report_events(yyyymm: str) -> list[dict]:
    """Read all report events from a monthly JSONL file."""
    path = _report_jsonl_path(yyyymm)
    if not path.exists():
        return []
    events = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # Skip truncated last lines (crash-safe)
    return events


def _read_all_report_events() -> list[dict]:
    """Read all report events from all monthly partitions."""
    events = []
    for path in sorted(REPORT_DIR.glob("*_v2b_reports.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


# ── Observation reader ────────────────────────────────────────────────────────

def read_completed_observation(session: str) -> dict:
    """
    Find the COMPLETED observation for the given intended_execution_session.

    Returns the full OBSERVATION_CREATED event dict.

    Raises ObservationNotFoundError if no observation exists for the session.
    Raises ObservationIncompleteError if observation exists but is not COMPLETED.
    Raises ObservationIntegrityError if the event is missing or malformed.

    Note: event_hash integrity is verified by v2b_ledger's _read_events_raw
    during get_observation_events() — tampered events raise CorruptionError there.
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

    Returns a dict with all fields required for the shadow report:
    strategy selections, agreement/disagreement, coverage, quality, and provenance.
    """
    selected_per_strategy = obs_event.get("selected_tickers_per_strategy", {})
    factor_only_selected = selected_per_strategy.get("Factor_Only_Core_V2", [])
    claude_selected = selected_per_strategy.get("Factor_Plus_Claude_Shadow_V2", [])

    fo_set = set(factor_only_selected)
    cs_set = set(claude_selected)

    # Claude selection is always a subset of Factor-Only (V2B.2 design invariant)
    agreement_tickers = sorted(cs_set)
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

    # Check index first (fast path)
    idx = _load_report_idx()
    if obs_key in idx:
        existing = idx[obs_key]
        if existing.get("report_key") == report_key:
            return ShadowReportResult(
                status="IDEMPOTENT_MATCH",
                report_key=report_key,
                observation_key=obs_key,
                intended_execution_session=session,
            )
        else:
            # Different report content for same observation — conflict
            raise ReportConflictError(
                f"Report conflict for observation {obs_key[:16]}…: "
                f"existing report_key={existing.get('report_key', '?')[:16]}…, "
                f"new report_key={report_key[:16]}…"
            )

    # Write under lock
    with _monthly_report_lock(yyyymm):
        # Re-read under lock to avoid TOCTOU
        events = _read_report_events(yyyymm)
        existing_report = next(
            (e for e in events
             if e.get("event_type") == "REPORT_CREATED"
             and e.get("observation_key") == obs_key),
            None,
        )

        if existing_report is not None:
            existing_key = existing_report.get("report_key")
            if existing_key == report_key:
                # Repair index if needed
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

        # Build and append new REPORT_CREATED event
        event_body: dict[str, Any] = {
            "event_type": "REPORT_CREATED",
            "report_version": REPORT_VERSION,
            "report_key": report_key,
            "observation_key": obs_key,
            "observation_run_id": obs_run_id,
            "intended_execution_session": session,
            "generated_at": _now_iso(),
            "order_creation_blocked": _ORDER_CREATION_BLOCKED,
            "report_content": report_content,
        }
        event_body["event_hash"] = _make_event_hash(event_body)
        _append_report_event(yyyymm, event_body)

    _update_report_idx(obs_key, yyyymm, report_key)

    return ShadowReportResult(
        status="CREATED",
        report_key=report_key,
        observation_key=obs_key,
        intended_execution_session=session,
    )


def telegram_already_sent(report_key: str, yyyymm: str) -> bool:
    """Return True if a TELEGRAM_SENT event already exists for this report_key."""
    events = _read_report_events(yyyymm)
    return any(
        e.get("event_type") == "TELEGRAM_SENT" and e.get("report_key") == report_key
        for e in events
    )


def record_telegram_sent(report_key: str, obs_key: str, session: str) -> None:
    """Append a TELEGRAM_SENT event (under lock) to record that the message was sent."""
    yyyymm = session[:7] if len(session) >= 7 else "unknown"
    with _monthly_report_lock(yyyymm):
        # Idempotent: skip if already recorded
        if telegram_already_sent(report_key, yyyymm):
            return
        event_body: dict[str, Any] = {
            "event_type": "TELEGRAM_SENT",
            "report_key": report_key,
            "observation_key": obs_key,
            "intended_execution_session": session,
            "sent_at": _now_iso(),
            "order_creation_blocked": _ORDER_CREATION_BLOCKED,
        }
        event_body["event_hash"] = _make_event_hash(event_body)
        _append_report_event(yyyymm, event_body)


# ── Telegram message formatter ────────────────────────────────────────────────

def format_telegram_message(report_content: dict) -> str:
    """
    Format a labeled V2 SHADOW Telegram message.

    The message is clearly marked "V2 SHADOW — INGEN HANDLER" and must never
    be presented as a buy recommendation or production signal.
    V1 production messages are completely unaffected by this call.
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
) -> ShadowReportResult:
    """
    Full shadow reporting flow for a given intended_execution_session:

    1. Read the COMPLETED observation from the ledger.
    2. Validate observation integrity (key derivation, session match).
    3. Build report content.
    4. Append report record (idempotent).
    5. If send_telegram_fn is provided and not already sent: format and send.
    6. Record telegram sent status (idempotent).

    On ObservationNotFoundError or ObservationIncompleteError: returns
    ShadowReportResult(status="FAILED", error=...) — caller should send a
    shadow-error Telegram if desired, but must not affect V1.

    On ReportConflictError: returns ShadowReportResult(status="CONFLICT", ...)
    — fail-closed, does not send Telegram.

    V1 production messages are never modified or replaced.
    """
    _assert_order_creation_blocked()

    try:
        obs_event = read_completed_observation(session)
    except (ObservationNotFoundError, ObservationIncompleteError) as exc:
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
        return result

    yyyymm = session[:7] if len(session) >= 7 else "unknown"
    report_key = result.report_key

    if telegram_already_sent(report_key, yyyymm):
        result.telegram_skipped = True
        return result

    try:
        report_content = build_report_content(obs_event)
        message = format_telegram_message(report_content)
        send_telegram_fn(message)
        record_telegram_sent(report_key, result.observation_key or "", session)
        result.telegram_sent = True
    except Exception as exc:
        result.detail = f"Telegram send failed: {exc}"

    return result
