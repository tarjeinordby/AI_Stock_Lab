"""
V2B.4 — Signal outcome runner.

Coordinates the daily outcome tracking process:
  1. Validate / rebuild the outcome index (fail-closed on corruption)
  2. Scan COMPLETED V2B observations >= OUTCOME_TRACKING_START_SESSION
  3. Repair missing PENDING events idempotently for Factor_Only_Core_V2
  4. Process matured PENDINGs (exit_session reached):
       - Fetch adjusted prices via yfinance (up to 3 retries, backoff [2, 5]s)
       - If all prices present → OUTCOME_RECORDED (may write before grace)
       - If prices missing and grace NOT yet elapsed → OUTCOME_FETCH_DEFERRED
       - If prices missing and grace elapsed → OUTCOME_INCOMPLETE or OUTCOME_UNAVAILABLE

V1 isolation: no imports from V1 execution modules.
  Does NOT import: modules.portfolio, modules.orders, modules.fills,
                   modules.ledger, modules.state
  Does NOT call:   execute_buy, execute_sell, execute_pyramid_fill
  Does NOT create: orders, fills, trades, or position changes
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)

# ── V2B.4 constants (imported from pure ledger module) ───────────────────────
from modules.v2b_outcome import (  # noqa: E402
    GRACE_SESSIONS,
    HOLDING_SESSIONS_TUPLE,
    OUTCOME_DEFINITION_VERSION,
    OUTCOME_TRACKING_START_SESSION,
    STRATEGY_ID,
    ObservationValidationError,
    compute_exit_session,
    create_pending,
    list_pending_outcomes,
    make_outcome_key,
    record_fetch_deferred,
    record_terminal,
    validate_and_rebuild_index,
)

# ── V2B shadow ledger (read-only: observation_key + selected tickers) ────────
from modules.v2b_ledger import (  # noqa: E402
    get_observation_events,
    list_observations,
)

# ── Exchange calendar ─────────────────────────────────────────────────────────
from modules.exchange_calendar import (  # noqa: E402
    is_trading_session,
    sessions_between_count,
)

# ── Retry configuration ───────────────────────────────────────────────────────
_MAX_FETCH_ATTEMPTS: int = 3
_RETRY_BACKOFFS: list[float] = [2.0, 5.0]  # seconds between attempts 1→2 and 2→3

_FIELD_MAP: dict[str, str] = {
    "adjusted_open": "Open",
    "adjusted_close": "Close",
}


# ── yfinance fetch helpers ────────────────────────────────────────────────────


def _fetch_ohlcv_with_retry(
    symbols: list[str],
    provider_start: str,
    provider_end_exclusive: str,
) -> "Optional[object]":  # returns pd.DataFrame | None
    """
    Download OHLCV data from yfinance with up to 3 attempts and
    backoff of [2, 5] seconds between retries.

    Returns the DataFrame on success, None after 3 transport failures.
    Sanitised logging: ticker list is truncated at 10 symbols in log output.
    """
    import yfinance as yf  # local import to isolate yfinance dependency

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
    df: "object",  # pd.DataFrame
    symbol: str,
    session_date: str,
    field: str,
) -> float | None:
    """
    Extract a single price from a yfinance DataFrame.

    Handles both single-ticker (flat columns) and multi-ticker (MultiIndex
    columns with (field_name, ticker) pairs) DataFrames.

    field must be "adjusted_open" or "adjusted_close" — mapped to "Open"/"Close"
    (yfinance auto_adjust=True already adjusts Open and Close).

    Returns None if the symbol, column, or date row is absent.
    """
    import pandas as pd  # noqa: PLC0415

    if df is None or df.empty:
        return None

    yf_col = _FIELD_MAP.get(field)
    if yf_col is None:
        return None

    cols = df.columns
    # MultiIndex check: (field_name, ticker) tuples
    if isinstance(cols, pd.MultiIndex):
        if (yf_col, symbol) not in cols:
            return None
        series = df[(yf_col, symbol)]
    else:
        # Single-ticker download — columns are plain strings
        if yf_col not in cols:
            return None
        series = df[yf_col]

    # Match date — index may be datetime or date
    session_dt = pd.Timestamp(session_date)
    if session_dt not in series.index:
        # Try matching by date (timezone may differ)
        matched = [
            v for idx_val, v in series.items()
            if str(idx_val)[:10] == session_date
        ]
        if not matched:
            return None
        value = matched[0]
    else:
        value = series[session_dt]

    import math  # noqa: PLC0415
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def _next_calendar_day(date_str: str) -> str:
    """Return the calendar day after date_str (YYYY-MM-DD), used as yfinance end."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return (dt + timedelta(days=1)).strftime("%Y-%m-%d")


# ── Observation validation ────────────────────────────────────────────────────


def _extract_selected_tickers(observation_key: str) -> list[str]:
    """
    Read the V2B observation's CREATED event and return
    selected_tickers_per_strategy[STRATEGY_ID].

    Raises ObservationValidationError on structural problems.
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
            f"Observation {observation_key[:16]}…: selected_tickers_per_strategy is missing or not a dict"
        )

    if STRATEGY_ID not in stps:
        raise ObservationValidationError(
            f"Observation {observation_key[:16]}…: selected_tickers_per_strategy has no {STRATEGY_ID!r} key"
        )

    tickers = stps[STRATEGY_ID]
    if not isinstance(tickers, list):
        raise ObservationValidationError(
            f"Observation {observation_key[:16]}…: selected_tickers_per_strategy[{STRATEGY_ID!r}] is not a list"
        )

    for t in tickers:
        if not isinstance(t, str) or not t.strip():
            raise ObservationValidationError(
                f"Observation {observation_key[:16]}…: invalid ticker {t!r} in {STRATEGY_ID}"
            )

    if len(tickers) != len(set(tickers)):
        raise ObservationValidationError(
            f"Observation {observation_key[:16]}…: duplicate tickers in {STRATEGY_ID}: {tickers}"
        )

    return tickers


# ── Repair step ───────────────────────────────────────────────────────────────


def _repair_pending_for_observation(
    observation_key: str,
    entry_session: str,
    today_str: str,
    transport_errors: list[str],
) -> None:
    """
    For one COMPLETED observation, ensure PENDING events exist for all
    HOLDING_SESSIONS_TUPLE holding-session counts.  Idempotent.

    Empty ticker list → ObservationValidationError is logged and skipped.
    """
    try:
        selected_tickers = _extract_selected_tickers(observation_key)
    except ObservationValidationError as exc:
        log.warning("Skipping observation %s…: %s", observation_key[:16], exc)
        return

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
                "Observation %s… N=%d: PENDING already exists (idempotent)",
                observation_key[:16], n,
            )
        else:
            log.info(
                "Created PENDING: obs=%s… N=%d exit=%s tickers=%d",
                observation_key[:16], n, exit_session, len(selected_tickers),
            )


# ── Per-outcome processing ────────────────────────────────────────────────────


def _compute_per_ticker_return(entry: float, exit_: float) -> float:
    return ((exit_ - entry) / entry) * 100.0


def _process_one_pending(
    pending_event: dict,
    today_str: str,
    transport_errors: list[str],
) -> None:
    """
    Process a single matured PENDING event.

    Steps:
      1. Skip if exit_session has not yet been reached
      2. Fetch yfinance prices (entry_session open, exit_session close) for all
         selected_tickers + SPY
      3. On transport failure: add to transport_errors, continue (no FETCH_DEFERRED)
      4. On valid response with all prices → write OUTCOME_RECORDED immediately
      5. On valid response with missing prices and grace not elapsed → FETCH_DEFERRED
      6. On valid response with missing prices and grace elapsed → OUTCOME_INCOMPLETE
         or OUTCOME_UNAVAILABLE
    """
    outcome_key = pending_event["outcome_key"]
    observation_key = pending_event["observation_key"]
    holding_sessions = pending_event["holding_sessions"]
    entry_session = pending_event["entry_session"]
    exit_session = pending_event["exit_session"]
    selected_tickers: list[str] = pending_event.get("selected_tickers", [])
    exit_partition_yyyymm = pending_event["exit_partition_yyyymm"]

    # Don't process until exit_session has been reached
    if exit_session > today_str:
        return

    # All symbols to fetch: selected tickers + SPY as benchmark
    all_symbols = list(selected_tickers) + (["SPY"] if "SPY" not in selected_tickers else [])

    provider_start = entry_session
    provider_end_exclusive = _next_calendar_day(exit_session)

    from datetime import datetime as _dt, timezone as _tz  # noqa: PLC0415
    fetched_at = _dt.now(_tz.utc).isoformat()

    df = _fetch_ohlcv_with_retry(all_symbols, provider_start, provider_end_exclusive)
    if df is None:
        # Transport failure — keep PENDING, record for exit-code
        transport_errors.append(
            f"outcome_key={outcome_key[:16]}… (obs={observation_key[:16]}… N={holding_sessions})"
        )
        return

    # Extract prices
    entry_prices: dict[str, float] = {}
    exit_prices: dict[str, float] = {}
    unavailable_tickers: list[str] = []
    unavailable_reasons: dict[str, str] = {}
    missing_price_points: list[dict] = []

    for ticker in selected_tickers:
        ep = _extract_price(df, ticker, entry_session, "adjusted_open")
        xp = _extract_price(df, ticker, exit_session, "adjusted_close")
        if ep is None:
            missing_price_points.append(
                {"symbol": ticker, "field": "adjusted_open", "session": entry_session}
            )
        if xp is None:
            missing_price_points.append(
                {"symbol": ticker, "field": "adjusted_close", "session": exit_session}
            )
        if ep is not None and xp is not None:
            entry_prices[ticker] = ep
            exit_prices[ticker] = xp
        else:
            unavailable_tickers.append(ticker)
            missing_fields = []
            if ep is None:
                missing_fields.append(f"adjusted_open@{entry_session}")
            if xp is None:
                missing_fields.append(f"adjusted_close@{exit_session}")
            unavailable_reasons[ticker] = "price_row_missing: " + ", ".join(missing_fields)

    # SPY prices
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

    # Grace elapsed check
    grace_elapsed = sessions_between_count(exit_session, today_str) >= GRACE_SESSIONS

    # If any prices are missing (tickers or SPY)
    if missing_price_points:
        if not grace_elapsed:
            # Write FETCH_DEFERRED (idempotent — deduplicated within the ledger)
            from zoneinfo import ZoneInfo  # noqa: PLC0415
            attempt_date = _dt.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
            result = record_fetch_deferred(
                outcome_key=outcome_key,
                attempted_at=fetched_at,
                attempt_date=attempt_date,
                missing_price_points=missing_price_points,
                source="yfinance",
                detail=(
                    f"N={holding_sessions} exit={exit_session} "
                    f"missing {len(missing_price_points)} price point(s)"
                ),
            )
            if result is None:
                log.debug(
                    "FETCH_DEFERRED deduplicated: %s… N=%d",
                    outcome_key[:16], holding_sessions,
                )
            else:
                log.info(
                    "FETCH_DEFERRED: %s… N=%d exit=%s missing=%d",
                    outcome_key[:16], holding_sessions, exit_session,
                    len(missing_price_points),
                )
            return

        # Grace has elapsed — terminate
        # Classify: INCOMPLETE vs UNAVAILABLE
        # UNAVAILABLE: no usable ticker prices at all, or SPY missing (no official signal)
        has_any_ticker_price = bool(entry_prices)  # at least one complete ticker pair
        has_spy = spy_entry_price is not None and spy_exit_price is not None

        if not has_any_ticker_price and not has_spy:
            terminal_type = "OUTCOME_UNAVAILABLE"
        else:
            terminal_type = "OUTCOME_INCOMPLETE"

        _write_terminal(
            outcome_key=outcome_key,
            terminal_type=terminal_type,
            pending_event=pending_event,
            entry_prices=entry_prices,
            exit_prices=exit_prices,
            spy_entry_price=spy_entry_price,
            spy_exit_price=spy_exit_price,
            unavailable_tickers=unavailable_tickers,
            unavailable_reasons=unavailable_reasons,
            fetched_at=fetched_at,
            provider_start=provider_start,
            provider_end_exclusive=provider_end_exclusive,
            today_str=today_str,
        )
        return

    # All prices present (including SPY) → OUTCOME_RECORDED
    _write_terminal(
        outcome_key=outcome_key,
        terminal_type="OUTCOME_RECORDED",
        pending_event=pending_event,
        entry_prices=entry_prices,
        exit_prices=exit_prices,
        spy_entry_price=spy_entry_price,
        spy_exit_price=spy_exit_price,
        unavailable_tickers=[],
        unavailable_reasons={},
        fetched_at=fetched_at,
        provider_start=provider_start,
        provider_end_exclusive=provider_end_exclusive,
        today_str=today_str,
    )


def _write_terminal(
    outcome_key: str,
    terminal_type: str,
    pending_event: dict,
    entry_prices: dict[str, float],
    exit_prices: dict[str, float],
    spy_entry_price: float | None,
    spy_exit_price: float | None,
    unavailable_tickers: list[str],
    unavailable_reasons: dict[str, str],
    fetched_at: str,
    provider_start: str,
    provider_end_exclusive: str,
    today_str: str,
) -> None:
    """Compute return metrics and write terminal event."""
    selected_tickers: list[str] = pending_event.get("selected_tickers", [])
    entry_session = pending_event["entry_session"]
    exit_session = pending_event["exit_session"]
    holding_sessions = pending_event["holding_sessions"]
    observation_key = pending_event["observation_key"]

    # Per-ticker returns (only where both entry and exit prices are available)
    per_ticker_return_pct: dict[str, float] = {}
    for ticker in selected_tickers:
        if ticker in entry_prices and ticker in exit_prices:
            per_ticker_return_pct[ticker] = _compute_per_ticker_return(
                entry_prices[ticker], exit_prices[ticker]
            )

    # SPY return
    spy_return_pct: float | None = None
    if spy_entry_price is not None and spy_exit_price is not None:
        spy_return_pct = _compute_per_ticker_return(spy_entry_price, spy_exit_price)

    # Aggregate metrics — null if ANY selected_ticker or SPY price is missing
    all_tickers_available = set(per_ticker_return_pct.keys()) == set(selected_tickers)
    spy_available = spy_return_pct is not None

    portfolio_return_complete = all_tickers_available and spy_available

    if all_tickers_available and selected_tickers:
        available_subset_return_pct = sum(per_ticker_return_pct.values()) / len(selected_tickers)
    elif per_ticker_return_pct:
        available_subset_return_pct = sum(per_ticker_return_pct.values()) / len(per_ticker_return_pct)
    else:
        available_subset_return_pct = None

    if portfolio_return_complete and selected_tickers:
        portfolio_return_pct: float | None = sum(per_ticker_return_pct.values()) / len(selected_tickers)
        forward_alpha_vs_spy: float | None = portfolio_return_pct - spy_return_pct  # type: ignore[operator]
        hit_rate_vs_spy: float | None = (
            sum(1 for r in per_ticker_return_pct.values() if r > spy_return_pct)  # type: ignore[operator]
            / len(selected_tickers)
        )
    elif portfolio_return_complete and not selected_tickers:
        # Empty ticker list — UNAVAILABLE per design
        portfolio_return_pct = None
        forward_alpha_vs_spy = None
        hit_rate_vs_spy = None
    else:
        portfolio_return_pct = None
        forward_alpha_vs_spy = None
        hit_rate_vs_spy = None

    payload: dict = {
        "event_type": terminal_type,
        "outcome_key": outcome_key,
        "observation_key": observation_key,
        "strategy_id": STRATEGY_ID,
        "outcome_definition_version": OUTCOME_DEFINITION_VERSION,
        "holding_sessions": holding_sessions,
        "entry_session": entry_session,
        "exit_session": exit_session,
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
        "forward_alpha_vs_spy": forward_alpha_vs_spy,
        "hit_rate_vs_spy": hit_rate_vs_spy,
        "available_subset_return_pct": available_subset_return_pct,
        "adjustment_mode": "auto_adjust=True",
        "data_source": "yfinance",
        "fetch_date_range": f"{provider_start}/{provider_end_exclusive}",
        "provider_end_exclusive": provider_end_exclusive,
        "fetched_at": fetched_at,
        "measurement_date": today_str,
        "order_creation_blocked": True,
    }

    result = record_terminal(outcome_key=outcome_key, terminal_payload=payload)
    if result == "IDEMPOTENT_MATCH":
        log.debug("Terminal %s idempotent: %s… N=%d", terminal_type, outcome_key[:16], holding_sessions)
    else:
        log.info(
            "%s: %s… N=%d portfolio_return_pct=%s alpha=%s hit_rate=%s",
            terminal_type, outcome_key[:16], holding_sessions,
            f"{portfolio_return_pct:.4f}%" if portfolio_return_pct is not None else "null",
            f"{forward_alpha_vs_spy:.4f}%" if forward_alpha_vs_spy is not None else "null",
            f"{hit_rate_vs_spy:.2%}" if hit_rate_vs_spy is not None else "null",
        )


# ── Main runner entry point ───────────────────────────────────────────────────


def run_outcome_tracker(today_str: str) -> int:
    """
    Run the full V2B.4 outcome tracking cycle for today_str (YYYY-MM-DD).

    Returns:
      0 — all operations succeeded
      1 — one or more transport errors occurred (outcomes kept PENDING for retry)

    Raises CorruptionError if any hash chain or index integrity check fails
    (fail-closed — exit 2 in the entry point wrapper).
    """
    log.info("V2B.4 outcome tracker starting for %s", today_str)

    # Step 1: Validate / rebuild index (fail-closed)
    validate_and_rebuild_index()
    log.debug("Index validated/rebuilt")

    # Step 2: Scan COMPLETED observations >= OUTCOME_TRACKING_START_SESSION
    all_obs = list_observations()
    eligible = [
        obs for obs in all_obs
        if obs.get("status") == "COMPLETED"
        and (obs.get("intended_execution_session") or "") >= OUTCOME_TRACKING_START_SESSION
    ]
    log.info("Found %d eligible COMPLETED observations", len(eligible))

    transport_errors: list[str] = []

    # Step 3: Repair — ensure PENDING events exist for all holding-session counts
    for obs in eligible:
        observation_key = obs["observation_key"]
        entry_session = obs["intended_execution_session"]
        if not entry_session:
            log.warning("Observation %s… has no intended_execution_session — skipping", observation_key[:16])
            continue
        if not is_trading_session(entry_session):
            log.warning(
                "Observation %s… entry_session=%s is not a trading session — skipping",
                observation_key[:16], entry_session,
            )
            continue
        _repair_pending_for_observation(
            observation_key=observation_key,
            entry_session=entry_session,
            today_str=today_str,
            transport_errors=transport_errors,
        )

    # Step 4: Process matured PENDINGs
    pending_outcomes = list_pending_outcomes()
    log.info("Found %d PENDING outcomes to evaluate", len(pending_outcomes))

    for pending_ev in pending_outcomes:
        exit_session = pending_ev.get("exit_session", "")
        if exit_session > today_str:
            # Not yet matured
            continue
        try:
            _process_one_pending(
                pending_event=pending_ev,
                today_str=today_str,
                transport_errors=transport_errors,
            )
        except Exception as exc:  # noqa: BLE001
            ok = pending_ev.get("outcome_key", "?")[:16]
            log.error(
                "Unexpected error processing outcome %s…: %s — keeping PENDING",
                ok, exc, exc_info=True,
            )
            transport_errors.append(f"unexpected_error: outcome_key={ok}…")

    if transport_errors:
        log.error(
            "V2B.4 outcome tracker finished with %d error(s):\n  %s",
            len(transport_errors), "\n  ".join(transport_errors),
        )
        return 1

    log.info("V2B.4 outcome tracker finished successfully")
    return 0
