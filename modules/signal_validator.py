"""
Signal validation for the next_session_daily_open_v1 execution model.

Six-layer validation (all must pass for full execution):
  1. Required fields  — signal_run_id, created_at_utc, date present
  2. Session match    — intended_execution_session == today's NYSE session
  3. Publication      — publication_status == "published"
  4. Ledger           — signal_run ledger record status == "completed"
  5. Content hash     — signal_content_hash matches recomputed hash (tamper-detect)
  6. Data quality     — coverage/stale thresholds within limits

Actions always allowed — rely on position/price data, NOT signal ranking:
  stop_loss       trailing stop, fixed stop
  drawdown_protection   sell on extreme portfolio drawdown

Actions blocked when signal is invalid or stale:
  new_buy         any new position entry
  pyramid_fill    second 40% tranche of an existing partial position
  signal_sells    rank/score/SMA-200 based exits
                  (correlation-based sells also blocked — corr_pairs from signal)

Note: protective sells (Stop-loss, Trailing stop) proceed even on stale signal
because they are triggered by price, not signal rank.

Data quality thresholds and rationale
--------------------------------------
signal_coverage_rate = valid_ticker_count / max(universe_count, 1)

  NORMAL  >= 0.90  Full execution, all actions permitted.
           Rationale: >=90% of universe has fresh data. Signals representative.

  REDUCED  0.75-0.89  New buys permitted; warn on data gaps.
           Rationale: Meaningful missing data but universe broadly covered.
           We allow buys for all candidates since individual ticker staleness
           is not available in the aggregated signal payload. The degraded
           coverage is logged and included in the Telegram report.

  BLOCKED  < 0.75  No new buys, no pyramid fills.
           Rationale: <75% coverage. Signal may not reflect current conditions.
           Protective stops still run — they depend on price, not coverage.

stale_pct = stale_ticker_count / max(universe_count, 1)

  STALE_BLOCK > 0.25  Block new buys regardless of overall coverage rate.
           Rationale: If >25% of tickers have data older than 5 calendar days,
           momentum factors (mom_12_1, mom_3_1, relative_strength_3m) are
           unreliable. Stale prices produce wrong rankings; the bot could
           enter trends that reversed days ago.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

COVERAGE_NORMAL = 0.90
COVERAGE_REDUCED = 0.75
STALE_BLOCK_PCT = 0.25


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SignalValidationResult:
    is_valid: bool
    failure_mode: str        # "ok" | "missing_fields" | "stale_signal" | "invalid_session" |
                             # "unpublished" | "ledger_incomplete" | "ledger_error" |
                             # "hash_mismatch" | "hash_error" | "calendar_unavailable" |
                             # "data_quality_blocked"
    reason: str              # Human-readable Norwegian description
    allow_new_buys: bool     # False when signal invalid/stale or data quality blocked
    allow_pyramid: bool      # False when signal invalid/stale or data quality blocked
    allow_protective_sells: bool  # Always True — never blocked
    signal_run_id: Optional[str]
    intended_session: Optional[str]  # From signal payload (what signal claims)
    actual_session: str              # Today's computed session (execution context)
    generated_at: Optional[str]      # created_at_utc from signal
    published_at: Optional[str]
    model_version: Optional[str]
    data_quality_tier: str           # "normal" | "reduced" | "blocked" | "unknown"
    data_quality: dict               # Raw counts from signal payload


# ---------------------------------------------------------------------------
# Content hash  (identical pattern to ledger.py _make_content_hash)
# ---------------------------------------------------------------------------

def compute_signal_content_hash(payload: dict) -> str:
    """SHA-256 of the payload with signal_content_hash excluded."""
    p = {k: v for k, v in payload.items() if k != "signal_content_hash"}
    return hashlib.sha256(
        json.dumps(p, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Data quality tier
# ---------------------------------------------------------------------------

def _data_quality_tier(signal: dict) -> tuple[str, dict]:
    universe_count = signal.get("universe_count") or 0
    valid_count    = signal.get("valid_ticker_count") or 0
    stale_count    = signal.get("stale_ticker_count") or 0
    excluded_count = signal.get("excluded_ticker_count") or 0
    missing_count  = signal.get("missing_ticker_count") or 0

    quality = {
        "universe_count":      universe_count,
        "valid_ticker_count":  valid_count,
        "stale_ticker_count":  stale_count,
        "excluded_ticker_count": excluded_count,
        "missing_ticker_count": missing_count,
        "signal_coverage_rate": None,
        "stale_pct":           None,
    }

    if universe_count <= 0:
        return "unknown", quality

    coverage  = valid_count / universe_count
    stale_pct = stale_count / universe_count
    quality["signal_coverage_rate"] = round(coverage, 4)
    quality["stale_pct"]            = round(stale_pct, 4)

    if stale_pct > STALE_BLOCK_PCT or coverage < COVERAGE_REDUCED:
        return "blocked", quality
    if coverage < COVERAGE_NORMAL:
        return "reduced", quality
    return "normal", quality


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------

def validate_signal(
    signal: dict,
    signal_path: str,
    session_date: str,
    *,
    _now=None,
    _get_ledger_status: Optional[Callable] = None,    # (signal_run_id, year_month) → dict|None
    _intended_session_fn: Optional[Callable] = None,  # (date_str) → str  — for tests/injection
) -> SignalValidationResult:
    """
    Six-layer signal validation. Never raises — exceptions become failure modes.

    Args:
        signal:       Parsed signal JSON dict.
        signal_path:  Path to the signal file on disk.
        session_date: Today's NYSE trading session (YYYY-MM-DD, from today_str()).
        _now:         Override UTC now — for deterministic testing only.
        _get_ledger_status:   Injected ledger lookup — for testing only.
        _intended_session_fn: Injected calendar fn — for testing only.
    """
    signal_run_id = signal.get("signal_run_id")
    generated_at  = signal.get("created_at_utc")
    published_at  = signal.get("published_at")
    model_version = signal.get("model_version")
    intended_session = signal.get("intended_execution_session")
    pub_status    = signal.get("publication_status")

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
            model_version=model_version,
            data_quality_tier=data_tier,
            data_quality=data_quality,
        )

    def _ok(allow_new_buys=True, allow_pyramid=True, failure_mode="ok", reason="Signal validert"):
        return SignalValidationResult(
            is_valid=True,
            failure_mode=failure_mode,
            reason=reason,
            allow_new_buys=allow_new_buys,
            allow_pyramid=allow_pyramid,
            allow_protective_sells=True,
            signal_run_id=signal_run_id,
            intended_session=intended_session,
            actual_session=session_date,
            generated_at=generated_at,
            published_at=published_at,
            model_version=model_version,
            data_quality_tier=data_tier,
            data_quality=data_quality,
        )

    # ------------------------------------------------------------------
    # Layer 1: Required fields
    # ------------------------------------------------------------------
    if not signal_run_id:
        return _reject("missing_fields", "signal_run_id mangler fra signal-fil")
    if not generated_at:
        return _reject("missing_fields", "created_at_utc mangler fra signal-fil")
    if not signal.get("date"):
        return _reject("missing_fields", "date mangler fra signal-fil")

    # ------------------------------------------------------------------
    # Layer 2: Session match — use stored field first, fall back to calendar
    # ------------------------------------------------------------------
    if not intended_session:
        try:
            if _intended_session_fn is not None:
                intended_session = _intended_session_fn(signal["date"])
            else:
                from modules.exchange_calendar import (   # noqa: PLC0415
                    CalendarUnavailableError,
                    intended_execution_session as _ies,
                )
                intended_session = _ies(signal["date"])
        except Exception:
            return _reject(
                "calendar_unavailable",
                (
                    "Kan ikke bestemme intended_execution_session — "
                    "exchange_calendars er utilgjengelig og feltet mangler i signal"
                ),
            )

    if intended_session != session_date:
        return _reject(
            "stale_signal",
            (
                f"Signal intendert sesjon {intended_session} matcher ikke "
                f"dagens sesjon {session_date}. "
                f"Signalet er utdatert — ingen nye kjøp."
            ),
        )

    # ------------------------------------------------------------------
    # Layer 3: Publication status
    # ------------------------------------------------------------------
    if pub_status != "published":
        return _reject(
            "unpublished",
            f"publication_status er '{pub_status}', forventet 'published'",
        )

    # ------------------------------------------------------------------
    # Layer 4: Ledger check — run must be completed
    # ------------------------------------------------------------------
    year_month = (signal.get("date") or session_date)[:7]
    try:
        if _get_ledger_status is not None:
            run_status = _get_ledger_status(signal_run_id, year_month)
        else:
            from modules.ledger import get_signal_run_status  # noqa: PLC0415
            run_status = get_signal_run_status(signal_run_id, year_month)

        if run_status is None:
            return _reject(
                "ledger_incomplete",
                f"Ingen ledger-record funnet for {signal_run_id} ({year_month})",
            )
        ledger_stat = run_status.get("status")
        if ledger_stat != "completed":
            return _reject(
                "ledger_incomplete",
                f"Ledger-status for {signal_run_id} er '{ledger_stat}', forventet 'completed'",
            )
    except (ImportError, ModuleNotFoundError):
        pass  # Ledger not available — skip check (backward compat)
    except Exception as exc:
        return _reject("ledger_error", f"Ledger-sjekk feilet: {exc}")

    # ------------------------------------------------------------------
    # Layer 5: Content hash — tamper detection
    # ------------------------------------------------------------------
    stored_hash = signal.get("signal_content_hash", "")
    if stored_hash:
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
    # If no stored hash: older signal (pre-commit-6) — skip check (backward compat)

    # ------------------------------------------------------------------
    # Layer 6: Data quality thresholds
    # ------------------------------------------------------------------
    if data_tier == "blocked":
        coverage = data_quality.get("signal_coverage_rate")
        stale = data_quality.get("stale_pct")
        return _ok(
            allow_new_buys=False,
            allow_pyramid=False,
            failure_mode="data_quality_blocked",
            reason=(
                f"Data-kvalitet blokkerer nye kjøp: "
                f"coverage={coverage}, stale_pct={stale} "
                f"(grenser: coverage>={COVERAGE_REDUCED}, stale<={STALE_BLOCK_PCT})"
            ),
        )

    # All layers passed
    return _ok(
        allow_new_buys=True,
        allow_pyramid=True,
        failure_mode="ok" if data_tier != "reduced" else "data_quality_reduced",
        reason="Signal validert" if data_tier != "reduced" else (
            f"Signal validert — data-kvalitet redusert "
            f"(coverage={data_quality.get('signal_coverage_rate')})"
        ),
    )
