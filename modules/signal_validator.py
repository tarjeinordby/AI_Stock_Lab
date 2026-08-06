"""
Signal validation for the next_session_daily_open_v1 execution model.

Six validation layers (all must pass for full execution):
  1. Required fields  — signal_run_id, created_at_utc, date, data_cutoff_at,
                        intended_execution_session, published_at,
                        published_commit_sha, model_version
  1b. Publication status — publication_status must be exactly "published"
  2. Session match    — intended_execution_session == today's NYSE session
  3. Publication      — publication record exists in signal_publications.jsonl
                        with workflow_conclusion == "success"; pub record
                        content_hash validated; key fields cross-validated
                        against signal; finalized_commit_sha must be present
                        (persistence-ack after Phase 4 push)
  4. Ledger integrity — signal_run status == "completed" (sidecar) AND
                        model_version + data_cutoff_at match the JSONL record;
                        ImportError → rejected (fail-closed, not skipped)
  5. Content hash     — signal_content_hash present and matches recomputed hash
                        (missing or empty hash → rejected, no backward compat)

Data quality tier is computed and reported but does NOT block execution.
Data quality thresholds will be proposed and approved before being enforced.

Actions always allowed — rely on position/price data, NOT signal ranking:
  stop_loss           trailing stop, fixed stop (price-based)
  drawdown_protection sell on extreme portfolio drawdown

Actions blocked when any layer fails:
  new_buy             any new position entry
  pyramid_fill        second 40% tranche of an existing partial position
  signal_sells        rank/score/SMA-200 based exits
                      (correlation-based sells also blocked — corr_pairs from signal)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SignalValidationResult:
    is_valid: bool
    failure_mode: str          # "ok" | "missing_fields" | "stale_signal" |
                               # "unpublished" | "workflow_failed" |
                               # "ledger_incomplete" | "ledger_mismatch" | "ledger_error" |
                               # "hash_missing" | "hash_mismatch" | "hash_error" |
                               # "data_quality_reduced"
    reason: str                # Human-readable Norwegian description
    allow_new_buys: bool
    allow_pyramid: bool
    allow_protective_sells: bool   # Always True — never blocked
    signal_run_id: Optional[str]
    intended_session: Optional[str]
    actual_session: str
    generated_at: Optional[str]    # created_at_utc
    published_at: Optional[str]
    published_commit_sha: Optional[str]
    model_version: Optional[str]
    data_cutoff_at: Optional[str]
    data_quality_tier: str         # "normal" | "reduced" | "unknown" (informational only)
    data_quality: dict


# ---------------------------------------------------------------------------
# Content hash
# ---------------------------------------------------------------------------

def compute_signal_content_hash(payload: dict) -> str:
    """SHA-256 of the payload with signal_content_hash excluded."""
    p = {k: v for k, v in payload.items() if k != "signal_content_hash"}
    return hashlib.sha256(
        json.dumps(p, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Data quality tier (informational — no execution blocking)
# ---------------------------------------------------------------------------

def _data_quality_tier(signal: dict) -> tuple[str, dict]:
    """
    Compute data quality tier for reporting. Does not block execution.
    Thresholds will be proposed and approved before enforcement.
    """
    universe_count = signal.get("universe_count") or 0
    valid_count    = signal.get("valid_ticker_count") or 0
    stale_count    = signal.get("stale_ticker_count") or 0
    excluded_count = signal.get("excluded_ticker_count") or 0
    missing_count  = signal.get("missing_ticker_count") or 0

    quality = {
        "universe_count":        universe_count,
        "valid_ticker_count":    valid_count,
        "stale_ticker_count":    stale_count,
        "excluded_ticker_count": excluded_count,
        "missing_ticker_count":  missing_count,
        "signal_coverage_rate":  None,
        "stale_pct":             None,
    }

    if universe_count <= 0:
        return "unknown", quality

    coverage  = valid_count / universe_count
    stale_pct = stale_count / universe_count
    quality["signal_coverage_rate"] = round(coverage, 4)
    quality["stale_pct"]            = round(stale_pct, 4)

    if coverage >= 0.90:
        return "normal", quality
    return "reduced", quality


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = (
    "signal_run_id",
    "created_at_utc",
    "date",
    "data_cutoff_at",
    "intended_execution_session",
    "published_at",
    "published_commit_sha",
    "model_version",
)


def validate_signal(
    signal: dict,
    signal_path: str,
    session_date: str,
    *,
    _now=None,
    _get_publication_fn: Optional[Callable] = None,   # (signal_run_id) → dict|None
    _get_ledger_status_fn: Optional[Callable] = None, # (signal_run_id, year_month) → dict|None
    _get_ledger_record_fn: Optional[Callable] = None, # (signal_run_id, year_month) → dict|None
) -> SignalValidationResult:
    """
    Five-layer signal validation. Never raises — exceptions become failure modes.

    Args:
        signal:       Parsed signal JSON dict (finalized — must have publication fields).
        signal_path:  Path to the signal file (for audit; not used for date logic).
        session_date: Today's NYSE trading session (YYYY-MM-DD, from today_str()).
        _now:         Override UTC now — deterministic testing only.
        _get_publication_fn:  Inject publication record lookup (tests).
        _get_ledger_status_fn: Inject status-sidecar lookup (tests).
        _get_ledger_record_fn: Inject JSONL signal_run record lookup (tests).
    """
    signal_run_id     = signal.get("signal_run_id")
    generated_at      = signal.get("created_at_utc")
    published_at      = signal.get("published_at")
    published_sha     = signal.get("published_commit_sha")
    model_version     = signal.get("model_version")
    intended_session  = signal.get("intended_execution_session")
    data_cutoff_at    = signal.get("data_cutoff_at")

    data_tier, data_quality = _data_quality_tier(signal)

    def _reject(mode: str, reason: str) -> SignalValidationResult:
        return SignalValidationResult(
            is_valid=False,
            failure_mode=mode,
            reason=reason,
            allow_new_buys=False,
            allow_pyramid=False,
            allow_protective_sells=True,
            signal_run_id=signal_run_id,
            intended_session=intended_session,
            actual_session=session_date,
            generated_at=generated_at,
            published_at=published_at,
            published_commit_sha=published_sha,
            model_version=model_version,
            data_cutoff_at=data_cutoff_at,
            data_quality_tier=data_tier,
            data_quality=data_quality,
        )

    def _ok(failure_mode="ok", reason="Signal validert") -> SignalValidationResult:
        return SignalValidationResult(
            is_valid=True,
            failure_mode=failure_mode,
            reason=reason,
            allow_new_buys=True,
            allow_pyramid=True,
            allow_protective_sells=True,
            signal_run_id=signal_run_id,
            intended_session=intended_session,
            actual_session=session_date,
            generated_at=generated_at,
            published_at=published_at,
            published_commit_sha=published_sha,
            model_version=model_version,
            data_cutoff_at=data_cutoff_at,
            data_quality_tier=data_tier,
            data_quality=data_quality,
        )

    # ------------------------------------------------------------------
    # Layer 1: Required fields — all must be non-empty strings
    # ------------------------------------------------------------------
    missing = [f for f in _REQUIRED_FIELDS if not signal.get(f)]
    if missing:
        return _reject(
            "missing_fields",
            f"Obligatoriske felt mangler eller er tomme: {', '.join(missing)}",
        )

    # ------------------------------------------------------------------
    # Layer 1b: publication_status must be exactly "published"
    # ------------------------------------------------------------------
    pub_status = signal.get("publication_status")
    if pub_status != "published":
        return _reject(
            "unpublished",
            f"publication_status er '{pub_status}', forventet 'published'. "
            "Kjør BOT_MODE=finalize_signal etter vellykket git push.",
        )

    # ------------------------------------------------------------------
    # Layer 2: Session match
    # intended_execution_session must equal today's session (required by Layer 1)
    # ------------------------------------------------------------------
    if intended_session != session_date:
        return _reject(
            "stale_signal",
            (
                f"Signal intendert sesjon {intended_session} matcher ikke "
                f"dagens sesjon {session_date}. "
                "Signalet er utdatert — ingen nye kjøp."
            ),
        )

    # ------------------------------------------------------------------
    # Layer 3: Publication record — must exist with workflow_conclusion="success"
    # ------------------------------------------------------------------
    try:
        if _get_publication_fn is not None:
            pub_record = _get_publication_fn(signal_run_id)
        else:
            from modules.publication import get_signal_publication  # noqa: PLC0415
            pub_record = get_signal_publication(signal_run_id)

        if pub_record is None:
            return _reject(
                "unpublished",
                (
                    f"Ingen publikasjonsrecord funnet for {signal_run_id}. "
                    "Signal er ikke bekreftet publisert etter push."
                ),
            )
        conclusion = pub_record.get("workflow_conclusion")
        if conclusion != "success":
            return _reject(
                "workflow_failed",
                (
                    f"Workflow conclusion er '{conclusion}', forventet 'success'. "
                    "Signalet ble ikke publisert etter vellykket push."
                ),
            )

        # Point 2: Validate publication record's own content_hash
        pub_stored_hash = pub_record.get("content_hash", "")
        if pub_stored_hash:
            try:
                from modules.publication import compute_publication_content_hash  # noqa: PLC0415
                pub_expected = compute_publication_content_hash(pub_record)
                if pub_expected != pub_stored_hash:
                    return _reject(
                        "publication_error",
                        "Publikasjonsrecord content_hash er ugyldig — "
                        "recorden kan ha blitt modifisert etter skriving.",
                    )
            except Exception as exc:
                return _reject(
                    "publication_error",
                    f"Kunne ikke verifisere publikasjonsrecord hash: {exc}",
                )

        # Point 3: Cross-validate key fields between pub record and signal
        rec_run_id = pub_record.get("signal_run_id")
        if rec_run_id and rec_run_id != signal_run_id:
            return _reject(
                "publication_error",
                f"Publikasjonsrecord signal_run_id mismatch: "
                f"record={rec_run_id!r}, signal={signal_run_id!r}",
            )

        rec_published_at = pub_record.get("published_at")
        if rec_published_at and published_at and rec_published_at != published_at:
            return _reject(
                "publication_error",
                f"published_at mismatch: record={rec_published_at!r}, "
                f"signal={published_at!r}",
            )

        rec_commit_sha = pub_record.get("commit_sha")
        if rec_commit_sha and published_sha and rec_commit_sha != published_sha:
            return _reject(
                "publication_error",
                f"commit_sha mismatch: record={rec_commit_sha!r}, "
                f"signal published_commit_sha={published_sha!r}",
            )

        # Point 6: Persistence-ack — finalized_commit_sha must be present
        if not pub_record.get("finalized_commit_sha"):
            return _reject(
                "publication_error",
                "Persistence-ack mangler: finalized_commit_sha ikke satt. "
                "Kjør BOT_MODE=ack_publication etter siste push (Phase 4).",
            )

    except Exception as exc:
        return _reject("publication_error", f"Publikasjonssjekk feilet: {exc}")

    # ------------------------------------------------------------------
    # Layer 4: Ledger integrity
    #   4a. Status sidecar must show "completed"
    #   4b. JSONL record must exist and fields must match signal payload
    # ------------------------------------------------------------------
    year_month = (signal.get("date") or session_date)[:7]
    try:
        if _get_ledger_status_fn is not None:
            run_status = _get_ledger_status_fn(signal_run_id, year_month)
        else:
            from modules.ledger import get_signal_run_status  # noqa: PLC0415
            run_status = get_signal_run_status(signal_run_id, year_month)

        if run_status is None:
            return _reject(
                "ledger_incomplete",
                f"Ingen ledger-statussidecar for {signal_run_id} ({year_month})",
            )
        if run_status.get("status") != "completed":
            return _reject(
                "ledger_incomplete",
                f"Ledger-status for {signal_run_id} er "
                f"'{run_status.get('status')}', forventet 'completed'",
            )
    except Exception as exc:
        return _reject("ledger_error", f"Ledger-statussjekk feilet: {exc}")

    # 4b — Cross-validate against JSONL signal_run record
    try:
        if _get_ledger_record_fn is not None:
            run_record = _get_ledger_record_fn(signal_run_id, year_month)
        else:
            from modules.ledger import get_signal_run_record  # noqa: PLC0415
            run_record = get_signal_run_record(signal_run_id, year_month)

        if run_record is None:
            return _reject(
                "ledger_incomplete",
                f"Ingen signal_run JSONL-record funnet for {signal_run_id} ({year_month})",
            )

        # model_version must match
        ledger_model = run_record.get("model_version")
        if ledger_model and ledger_model != model_version:
            return _reject(
                "ledger_mismatch",
                (
                    f"model_version mismatch: signal='{model_version}', "
                    f"ledger='{ledger_model}'"
                ),
            )

        # data_cutoff_at must match canonical_data_cutoff in ledger
        ledger_cutoff = run_record.get("canonical_data_cutoff")
        if ledger_cutoff and ledger_cutoff != data_cutoff_at:
            return _reject(
                "ledger_mismatch",
                (
                    f"data_cutoff_at mismatch: signal='{data_cutoff_at}', "
                    f"ledger='{ledger_cutoff}'"
                ),
            )
    except Exception as exc:
        return _reject("ledger_error", f"Ledger JSONL-kryssvalidering feilet: {exc}")

    # ------------------------------------------------------------------
    # Layer 5: Content hash — required; missing or empty → reject
    # ------------------------------------------------------------------
    stored_hash = signal.get("signal_content_hash", "")
    if not stored_hash:
        return _reject(
            "hash_missing",
            (
                "signal_content_hash mangler eller er tom. "
                "Signalet er ikke fullstendig finalisert."
            ),
        )
    try:
        computed = compute_signal_content_hash(signal)
        if computed != stored_hash:
            return _reject(
                "hash_mismatch",
                (
                    "signal_content_hash mismatch — filen kan ha blitt "
                    "modifisert etter publisering"
                ),
            )
    except Exception as exc:
        return _reject("hash_error", f"Content hash beregning feilet: {exc}")

    # ------------------------------------------------------------------
    # All layers passed — data quality is informational only
    # ------------------------------------------------------------------
    if data_tier == "reduced":
        return _ok(
            failure_mode="data_quality_reduced",
            reason=(
                f"Signal validert — data-kvalitet redusert "
                f"(coverage={data_quality.get('signal_coverage_rate')}). "
                "Terskler for blokkering er ikke ennå godkjent."
            ),
        )
    return _ok()
