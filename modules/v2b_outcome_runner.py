"""
V2B.4 — Signal outcome runner.

Daily outcome tracking cycle:
  1. Validate / rebuild the outcome index (fail-closed under global lock)
  2. Scan COMPLETED V2B observations >= OUTCOME_TRACKING_START_SESSION
  3. Repair missing PENDING events idempotently (ObservationValidationError propagates)
  4. Process matured PENDINGs (exit_session reached):
       - Empty selected_tickers → OUTCOME_UNAVAILABLE immediately (no fetch, no grace)
       - Fetch adjusted prices via yfinance (3 retries, [2,5]s backoff)
       - Transport failure → collect error, keep PENDING, exit 1 at end
       - Valid response, ALL symbols absent → FETCH_DEFERRED, keep PENDING (provider failure)
       - Valid response, some missing prices:
           → always write FETCH_DEFERRED first
           → if grace not elapsed: keep PENDING
           → if grace elapsed: write INCOMPLETE or UNAVAILABLE
       - All prices present → OUTCOME_RECORDED immediately

Terminal classification (on missing data after grace):
  - No complete ticker pair  → OUTCOME_UNAVAILABLE
  - ≥1 complete ticker       → OUTCOME_INCOMPLETE

Exit codes from run_outcome_tracker():
  0 — all outcomes processed successfully
  1 — one or more transport errors (outcomes kept PENDING for retry)
  CorruptionError / OutcomeValidationError / etc. propagate to caller for exit 2

V1 isolation: no imports from V1 execution modules.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

log = logging.getLogger(__name__)

from modules.v2b_outcome import (  # noqa: E402
    ADJUSTMENT_MODE,
    DATA_SOURCE,
    GRACE_SESSIONS,
    HOLDING_SESSIONS_TUPLE,
    OUTCOME_DEFINITION_VERSION,
    OUTCOME_TRACKING_START_SESSION,
    STRATEGY_ID,
    ObservationValidationError,
    OutcomeValidationError,
    compute_exit_session,
    create_pending,
    list_pending_outcomes,
    make_outcome_key,
    record_fetch_deferred,
    record_terminal,
    validate_and_rebuild_index,
)

from modules.v2b_ledger import (  # noqa: E402
    get_observation_events,
    list_observations,
)

from modules.exchange_calendar import (  # noqa: E402
    is_trading_session,
    sessions_between_count,
)

_MAX_FETCH_ATTEMPTS: int = 3
_RETRY_BACKOFFS: list[float] = [2.0, 5.0]

_FIELD_MAP: dict[str, str] = {
    "adjusted_open": "Open",
    "adjusted_close": "Close",
}


# ── yfinance helpers ──────────────────────────────────────────────────────────


def _fetch_ohlcv_with_retry(
    symbols: list[str],
    provider_start: str,
    provider_end_exclusive: str,
) -> "Optional[object]":
    """
    Download OHLCV data (auto_adjust=True) with up to 3 attempts.
    Returns the DataFrame on success (may be empty), None after 3 transport failures.
    """
    import yfinance as yf  # noqa: PLC0415

    log_symbols = symbols[:10] + (["…"] if len(symbols) > 10 else [])
    for attempt in range(1, _MAX_FETCH_ATTEMPTS + 1):
        try:
            df = yf.download(
                tickers=symbols,
                start=provider_start,
                end=provider_end_exclusive,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            return df
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "yfinance fetch attempt %d/%d failed (symbols=%s): %s",
                attempt, _MAX_FETCH_ATTEMPTS, log_symbols, type(exc).__name__,
            )
            if attempt < _MAX_FETCH_ATTEMPTS:
                time.sleep(_RETRY_BACKOFFS[attempt - 1])

    log.error(
        "yfinance fetch failed after %d attempts (symbols=%s) — keeping PENDING",
        _MAX_FETCH_ATTEMPTS, log_symbols,
    )
    return None


def _extract_price(
    df: "object",
    symbol: str,
    session_date: str,
    field: str,
) -> float | None:
    """
    Extract a single price from a yfinance DataFrame.

    Handles both single-ticker (flat columns) and multi-ticker (MultiIndex
    columns with (field_name, ticker) pairs).  auto_adjust=True maps
    'adjusted_open' → 'Open' and 'adjusted_close' → 'Close'.
    """
    import math as _math  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    if df is None or df.empty:
        return None

    yf_col = _FIELD_MAP.get(field)
    if yf_col is None:
        return None

    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        if (yf_col, symbol) not in cols:
            return None
        series = df[(yf_col, symbol)]
    else:
        if yf_col not in cols:
            return None
        series = df[yf_col]

    # Use boolean mask to handle duplicate-index DataFrames cleanly
    mask = series.index.normalize() == pd.Timestamp(session_date)
    if not mask.any():
        # Fallback: string-prefix match for non-standard index
        mask2 = [str(i)[:10] == session_date for i in series.index]
        if not any(mask2):
            return None
        value = series.iloc[[i for i, m in enumerate(mask2) if m][0]]
    else:
        value = series[mask].iloc[0]

    if value is None or (isinstance(value, float) and _math.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _next_calendar_day(date_str: str) -> str:
    """Return the calendar day after date_str — used as yfinance end (exclusive)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return (dt + timedelta(days=1)).strftime("%Y-%m-%d")


def _compute_return_pct(entry: float, exit_: float) -> float:
    return ((exit_ - entry) / entry) * 100.0


# ── Observation validation (repair step) ─────────────────────────────────────


def _extract_selected_tickers(observation_key: str) -> list[str]:
    """
    Read the V2B observation's OBSERVATION_CREATED event and return
    selected_tickers_per_strategy[STRATEGY_ID].

    Raises ObservationValidationError on any structural problem.
    Explicitly empty list (selected_tickers=[]) is valid — caller handles it.
    """
    events = get_observation_events(observation_key)
    created_ev = next(
        (e for e in events if e.get("event_type") == "OBSERVATION_CREATED"), None
    )
    if created_ev is None:
        raise ObservationValidationError(
            f"Observation {observation_key[:16]}… has no OBSERVATION_CREATED event"
        )

    stps = created_ev.get("selected_tickers_per_strategy")
    if not isinstance(stps, dict):
        raise ObservationValidationError(
            f"Observation {observation_key[:16]}…: "
            f"selected_tickers_per_strategy is missing or not a dict"
        )
    if STRATEGY_ID not in stps:
        raise ObservationValidationError(
            f"Observation {observation_key[:16]}…: "
            f"selected_tickers_per_strategy missing {STRATEGY_ID!r} key"
        )
    tickers = stps[STRATEGY_ID]
    if not isinstance(tickers, list):
        raise ObservationValidationError(
            f"Observation {observation_key[:16]}…: "
            f"selected_tickers_per_strategy[{STRATEGY_ID!r}] is not a list"
        )
    for t in tickers:
        if not isinstance(t, str) or not t.strip():
            raise ObservationValidationError(
                f"Observation {observation_key[:16]}…: invalid ticker {t!r}"
            )
    if len(tickers) != len(set(tickers)):
        raise ObservationValidationError(
            f"Observation {observation_key[:16]}…: duplicate tickers in {STRATEGY_ID}"
        )
    return tickers


# ── Repair step ───────────────────────────────────────────────────────────────


def _repair_pending_for_observation(
    observation_key: str,
    entry_session: str,
) -> None:
    """
    For one COMPLETED observation, ensure PENDING events exist for all
    HOLDING_SESSIONS_TUPLE holding-session counts.  Idempotent.

    ObservationValidationError propagates immediately (no catch/skip).
    Empty ticker list is explicitly valid.
    """
    # entry_session must be a valid NYSE trading session
    if not is_trading_session(entry_session):
        raise ObservationValidationError(
            f"Observation {observation_key[:16]}… entry_session={entry_session!r} "
            f"is not a NYSE trading session"
        )

    selected_tickers = _extract_selected_tickers(observation_key)

    for n in HOLDING_SESSIONS_TUPLE:
        exit_session = compute_exit_session(entry_session, n)
        result = create_pending(
            observation_key=observation_key,
            strategy_id=STRATEGY_ID,
            holding_sessions=n,
            outcome_definition_version=OUTCOME_DEFINITION_VERSION,
            entry_session=entry_session,
            exit_session=exit_session,
            selected_tickers=selected_tickers,
        )
        if result == "IDEMPOTENT_MATCH":
            log.debug(
                "PENDING already exists (idempotent): obs=%s… N=%d",
                observation_key[:16], n,
            )
        else:
            log.info(
                "Created PENDING: obs=%s… N=%d exit=%s tickers=%d",
                observation_key[:16], n, exit_session, len(selected_tickers),
            )


# ── Per-outcome processing ────────────────────────────────────────────────────


def _write_terminal(
    terminal_type: str,
    pending_event: dict,
    complete_tickers: list[str],
    entry_prices: dict[str, float],
    exit_prices: dict[str, float],
    spy_entry_price: float | None,
    spy_exit_price: float | None,
    unavailable_tickers: list[str],
    unavailable_reasons: dict[str, str],
    fetched_at: str | None,
    provider_end_exclusive: str,
    today_str: str,
) -> None:
    """Build the terminal payload, validate, and write to ledger."""
    outcome_key = pending_event["outcome_key"]
    observation_key = pending_event["observation_key"]
    holding_sessions = pending_event["holding_sessions"]
    entry_session = pending_event["entry_session"]
    exit_session = pending_event["exit_session"]
    exit_partition_yyyymm = pending_event["exit_partition_yyyymm"]
    selected_tickers = pending_event.get("selected_tickers", [])

    per_ticker_return_pct: dict[str, float] = {
        t: _compute_return_pct(entry_prices[t], exit_prices[t])
        for t in complete_tickers
    }

    spy_return_pct: float | None = None
    if spy_entry_price is not None and spy_exit_price is not None:
        spy_return_pct = _compute_return_pct(spy_entry_price, spy_exit_price)

    all_complete = (
        set(complete_tickers) == set(selected_tickers)
        and spy_return_pct is not None
    )

    if terminal_type == "OUTCOME_RECORDED":
        portfolio_return_pct: float | None = (
            sum(per_ticker_return_pct.values()) / len(selected_tickers)
            if selected_tickers else None
        )
        forward_alpha: float | None = (
            portfolio_return_pct - spy_return_pct
            if portfolio_return_pct is not None and spy_return_pct is not None
            else None
        )
        hit_rate: float | None = (
            sum(1 for r in per_ticker_return_pct.values() if r > spy_return_pct)
            / len(selected_tickers)
            if selected_tickers and spy_return_pct is not None
            else None
        )
        available_subset: float | None = None  # null in RECORDED per schema
        portfolio_return_complete = bool(all_complete)
    else:
        # INCOMPLETE or UNAVAILABLE
        portfolio_return_pct = None
        forward_alpha = None
        hit_rate = None
        portfolio_return_complete = False
        if complete_tickers:
            available_subset = (
                sum(per_ticker_return_pct.values()) / len(complete_tickers)
            )
        else:
            available_subset = None

    payload: dict = {
        "event_type": terminal_type,
        "outcome_key": outcome_key,
        "observation_key": observation_key,
        "strategy_id": STRATEGY_ID,
        "outcome_definition_version": OUTCOME_DEFINITION_VERSION,
        "holding_sessions": holding_sessions,
        "entry_session": entry_session,
        "exit_session": exit_session,
        "exit_partition_yyyymm": exit_partition_yyyymm,
        "selected_tickers": sorted(selected_tickers),
        "unavailable_tickers": sorted(unavailable_tickers),
        "unavailable_reasons": unavailable_reasons,
        "entry_prices": entry_prices,
        "exit_prices": exit_prices,
        "spy_entry_price": spy_entry_price,
        "spy_exit_price": spy_exit_price,
        "per_ticker_return_pct": per_ticker_return_pct,
        "spy_return_pct": spy_return_pct,
        "portfolio_return_pct": portfolio_return_pct,
        "portfolio_return_complete": portfolio_return_complete,
        "forward_alpha_vs_spy": forward_alpha,
        "hit_rate_vs_spy": hit_rate,
        "available_subset_return_pct": available_subset,
        "adjustment_mode": ADJUSTMENT_MODE,
        "data_source": DATA_SOURCE,
        "fetch_date_range": [entry_session, exit_session],
        "provider_end_exclusive": provider_end_exclusive,
        "fetched_at": fetched_at,
        "measurement_date": today_str,
        "order_creation_blocked": True,
    }

    result = record_terminal(outcome_key=outcome_key, terminal_payload=payload)
    if result == "IDEMPOTENT_MATCH":
        log.debug(
            "%s idempotent: %s… N=%d",
            terminal_type, outcome_key[:16], holding_sessions,
        )
    else:
        log.info(
            "%s: %s… N=%d portfolio_return=%s alpha=%s hit_rate=%s",
            terminal_type, outcome_key[:16], holding_sessions,
            f"{portfolio_return_pct:.4f}%" if portfolio_return_pct is not None else "null",
            f"{forward_alpha:.4f}%" if forward_alpha is not None else "null",
            f"{hit_rate:.2%}" if hit_rate is not None else "null",
        )


def _process_one_pending(
    pending_event: dict,
    today_str: str,
    transport_errors: list[str],
) -> None:
    """
    Process a single matured PENDING event.

    Raises any non-transport error (CorruptionError, OutcomeValidationError, etc.)
    so it propagates to the caller for exit code 2.
    Only actual transport failures (None from fetch) are collected in transport_errors.
    """
    outcome_key = pending_event["outcome_key"]
    observation_key = pending_event["observation_key"]
    holding_sessions = pending_event["holding_sessions"]
    entry_session = pending_event["entry_session"]
    exit_session = pending_event["exit_session"]
    selected_tickers: list[str] = pending_event.get("selected_tickers", [])

    # Guard: only process matured outcomes
    if exit_session > today_str:
        return

    # Empty ticker list → immediate UNAVAILABLE (no fetch, no grace check)
    if not selected_tickers:
        provider_end_exclusive = _next_calendar_day(exit_session)
        _write_terminal(
            terminal_type="OUTCOME_UNAVAILABLE",
            pending_event=pending_event,
            complete_tickers=[],
            entry_prices={},
            exit_prices={},
            spy_entry_price=None,
            spy_exit_price=None,
            unavailable_tickers=[],
            unavailable_reasons={},
            fetched_at=None,
            provider_end_exclusive=provider_end_exclusive,
            today_str=today_str,
        )
        return

    all_symbols = list(selected_tickers) + (
        ["SPY"] if "SPY" not in selected_tickers else []
    )
    provider_end_exclusive = _next_calendar_day(exit_session)
    fetched_at = datetime.now(timezone.utc).isoformat()

    df = _fetch_ohlcv_with_retry(all_symbols, entry_session, provider_end_exclusive)
    if df is None:
        # Transport failure — keep PENDING, record for exit-code 1
        transport_errors.append(
            f"outcome_key={outcome_key[:16]}… N={holding_sessions} "
            f"(obs={observation_key[:16]}…)"
        )
        return

    # Extract prices
    entry_prices: dict[str, float] = {}
    exit_prices: dict[str, float] = {}
    missing_price_points: list[dict] = []

    for ticker in selected_tickers:
        ep = _extract_price(df, ticker, entry_session, "adjusted_open")
        xp = _extract_price(df, ticker, exit_session, "adjusted_close")
        if ep is not None and xp is not None:
            entry_prices[ticker] = ep
            exit_prices[ticker] = xp
        else:
            if ep is None:
                missing_price_points.append(
                    {"symbol": ticker, "field": "adjusted_open", "session": entry_session}
                )
            if xp is None:
                missing_price_points.append(
                    {"symbol": ticker, "field": "adjusted_close", "session": exit_session}
                )

    spy_entry_price = _extract_price(df, "SPY", entry_session, "adjusted_open")
    spy_exit_price = _extract_price(df, "SPY", exit_session, "adjusted_close")
    if spy_entry_price is None:
        missing_price_points.append(
            {"symbol": "SPY", "field": "adjusted_open", "session": entry_session}
        )
    if spy_exit_price is None:
        missing_price_points.append(
            {"symbol": "SPY", "field": "adjusted_close", "session": exit_session}
        )

    complete_tickers = [
        t for t in selected_tickers if t in entry_prices and t in exit_prices
    ]
    all_complete = (
        len(complete_tickers) == len(selected_tickers)
        and spy_entry_price is not None
        and spy_exit_price is not None
    )

    # All symbols absent → provider-level failure (treat like transport failure)
    if not entry_prices and not exit_prices and spy_entry_price is None and spy_exit_price is None:
        log.warning(
            "All symbols absent from valid yfinance response: %s… N=%d exit=%s "
            "— writing FETCH_DEFERRED, keeping PENDING",
            outcome_key[:16], holding_sessions, exit_session,
        )
        record_fetch_deferred(
            outcome_key=outcome_key,
            attempted_at=fetched_at,
            attempt_date=today_str,
            missing_price_points=missing_price_points,
            source="yfinance",
            detail=f"N={holding_sessions} exit={exit_session}: all symbols absent from response",
        )
        return

    if not all_complete:
        # Some prices missing → always write FETCH_DEFERRED first
        record_fetch_deferred(
            outcome_key=outcome_key,
            attempted_at=fetched_at,
            attempt_date=today_str,
            missing_price_points=missing_price_points,
            source="yfinance",
            detail=(
                f"N={holding_sessions} exit={exit_session} "
                f"missing {len(missing_price_points)} price point(s)"
            ),
        )

        # Then check grace
        grace_elapsed = sessions_between_count(exit_session, today_str) >= GRACE_SESSIONS
        if not grace_elapsed:
            return  # keep PENDING

        # Grace elapsed → determine terminal type and build reasons
        unavailable_tickers = [t for t in selected_tickers if t not in complete_tickers]
        unavailable_reasons = {t: "price_missing_after_grace" for t in unavailable_tickers}

        if not complete_tickers:
            terminal_type = "OUTCOME_UNAVAILABLE"
        else:
            terminal_type = "OUTCOME_INCOMPLETE"

        _write_terminal(
            terminal_type=terminal_type,
            pending_event=pending_event,
            complete_tickers=complete_tickers,
            entry_prices=entry_prices,
            exit_prices=exit_prices,
            spy_entry_price=spy_entry_price,
            spy_exit_price=spy_exit_price,
            unavailable_tickers=unavailable_tickers,
            unavailable_reasons=unavailable_reasons,
            fetched_at=fetched_at,
            provider_end_exclusive=provider_end_exclusive,
            today_str=today_str,
        )
        return

    # All prices present → OUTCOME_RECORDED
    _write_terminal(
        terminal_type="OUTCOME_RECORDED",
        pending_event=pending_event,
        complete_tickers=complete_tickers,
        entry_prices=entry_prices,
        exit_prices=exit_prices,
        spy_entry_price=spy_entry_price,
        spy_exit_price=spy_exit_price,
        unavailable_tickers=[],
        unavailable_reasons={},
        fetched_at=fetched_at,
        provider_end_exclusive=provider_end_exclusive,
        today_str=today_str,
    )


# ── Main runner entry point ───────────────────────────────────────────────────


def run_outcome_tracker(today_str: str) -> int:
    """
    Run the full V2B.4 outcome tracking cycle for today_str (YYYY-MM-DD).

    Returns:
      0 — all outcomes processed successfully
      1 — one or more transport errors (outcomes kept PENDING for retry)

    Non-transport errors (CorruptionError, OutcomeValidationError, ObservationValidationError,
    InvalidTransitionError, ContentConflictError, unexpected exceptions) propagate to the
    caller, which maps them to exit code 2.
    """
    log.info("V2B.4 outcome tracker starting for %s", today_str)

    # Step 1: validate / rebuild index (under global lock, fail-closed)
    validate_and_rebuild_index()
    log.debug("Index validated/rebuilt")

    # Step 2: scan COMPLETED observations >= tracking start
    all_obs = list_observations()
    eligible = [
        obs for obs in all_obs
        if obs.get("status") == "COMPLETED"
        and (obs.get("intended_execution_session") or "") >= OUTCOME_TRACKING_START_SESSION
    ]
    log.info("Found %d eligible COMPLETED observations", len(eligible))

    transport_errors: list[str] = []

    # Step 3: repair — ensure PENDING events exist for all holding-session counts
    # ObservationValidationError propagates immediately (no catch/skip)
    for obs in eligible:
        observation_key = obs["observation_key"]
        entry_session = obs.get("intended_execution_session")
        if not entry_session:
            raise ObservationValidationError(
                f"Observation {observation_key[:16]}… has no intended_execution_session"
            )
        _repair_pending_for_observation(
            observation_key=observation_key,
            entry_session=entry_session,
        )

    # Step 4: process matured PENDINGs
    # Only transport errors are collected; all other errors propagate
    pending_outcomes = list_pending_outcomes()
    log.info("Found %d PENDING outcomes to evaluate", len(pending_outcomes))

    for pending_ev in pending_outcomes:
        exit_session = pending_ev.get("exit_session", "")
        if exit_session > today_str:
            continue
        # Intentionally no broad exception handler here.
        # CorruptionError, OutcomeValidationError, ContentConflictError, etc.
        # propagate to the caller for exit code 2.
        _process_one_pending(
            pending_event=pending_ev,
            today_str=today_str,
            transport_errors=transport_errors,
        )

    if transport_errors:
        log.error(
            "V2B.4 outcome tracker finished with %d transport error(s):\n  %s",
            len(transport_errors), "\n  ".join(transport_errors),
        )
        return 1

    log.info("V2B.4 outcome tracker finished successfully")
    return 0
