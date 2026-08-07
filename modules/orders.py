"""
Persistent order state for the next_session_daily_open_v1 execution model.

Lifecycle:
  pending_price → settling  (fill annotated, portfolio not yet persisted)
  settling → executed       (after portfolio + fill ledger both durably saved)
  pending_price → expired   (session passed without fill)
  pending_price → failed_price  (permanent market/price failure, not retried)

The order_id is deterministic:
  make_order_id(signal_id, portfolio_id, portfolio_version, ticker, session_date, action)
  — same inputs always yield the same order_id (idempotency key).

signal_id is a stable per-candidate identifier derived from (signal_run_id, strategy, ticker, action).
portfolio_id and portfolio_version ensure two portfolios consuming the same signal get different
order_ids, preventing cross-portfolio confusion.

Legacy fallback: _make_legacy_order_id() matches order_ids created by code prior to this scheme.
get_or_create_order() checks both new and legacy formats to prevent double-fills on upgrade.

Storage: STATE_DIR/orders.jsonl — append-only snapshots. Each update
appends a full order snapshot; the current state is the last record
per order_id when the file is read.

Crash recovery: a corrupt last line (truncated mid-write) is repaired on disk
under an exclusive lock on the first load_orders() call after the crash.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os

from modules.state import STATE_DIR, now_utc

ORDERS_FILE = f"{STATE_DIR}/orders.jsonl"
_ORDERS_LOCK_FILE = f"{STATE_DIR}/orders.jsonl.lock"

# Statuses
PENDING_PRICE = "pending_price"
SETTLING = "settling"       # in-flight: trade filled, portfolio not yet persisted
EXECUTED = "executed"
EXPIRED = "expired"
FAILED_PRICE = "failed_price"
CANCELLED = "cancelled"

TERMINAL = frozenset([EXECUTED, EXPIRED, FAILED_PRICE, CANCELLED])


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def make_order_id(signal_id, portfolio_id: str, portfolio_version: str,
                  ticker: str, session_date: str, action: str) -> str:
    """Deterministic order ID: same inputs always produce the same 16-char hex string.

    Uses signal_id (per-candidate, not per-run), portfolio_id, and portfolio_version
    so that two portfolios consuming the same signal get DIFFERENT order_ids, preventing
    cross-portfolio confusion while still being idempotent across reruns.

    signal_id=None is treated as "" for hashing (safe for safety-action orders).
    """
    sid = signal_id if signal_id is not None else ""
    raw = f"{sid}|{portfolio_id}|{portfolio_version}|{ticker}|{session_date}|{action}"
    return "ord-" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def _make_legacy_order_id(signal_run_id: str, ticker: str, strategy: str,
                          session_date: str, action: str) -> str:
    """Old v1 order ID format — for backward-compat lookup only.

    Matches order_ids created before the signal_id+portfolio_id scheme was introduced.
    Used exclusively in get_or_create_order() to prevent double-fills on upgrade.
    """
    raw = f"{signal_run_id}|{ticker}|{strategy}|{session_date}|{action}"
    return "ord-" + hashlib.sha256(raw.encode()).hexdigest()[:12]



def make_trade_id(order_id: str, timestamp_iso: str = "") -> str:
    """Tie a trade fill back to its order.
    Omit timestamp_iso for a deterministic ID (one fill per order) — enables crash recovery.
    """
    raw = order_id if not timestamp_iso else f"{order_id}|{timestamp_iso}"
    return "trd-" + hashlib.sha256(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Order construction
# ---------------------------------------------------------------------------

# Sentinel: distinguishes "caller did not pass signal_id" from "caller passed None"
_SIGNAL_ID_UNSET = object()


def build_order(
    signal_run_id: str,
    ticker: str,
    strategy: str,
    session_date: str,
    action: str,
    target_value: float,
    reason: str,
    signal_price: float | None,
    execution_version: str,
    pyramid_remaining: float = 0.0,
    portfolio_id: str = "",
    portfolio_version: str = "",
    signal_id=_SIGNAL_ID_UNSET,
    action_origin: str | None = None,
) -> dict:
    """Build a new order dict (not yet persisted).

    signal_id: stable content-addressed signal identifier (ledger's signal_id for the candidate).
    Omit to default to signal_run_id (backward compat).
    Pass None explicitly for safety-action orders that have no associated signal.
    action_origin: "signal" | "portfolio_safety" | None.
      "portfolio_safety" marks stop-loss / drawdown protection sells that run without a valid signal.
    """
    _signal_id = signal_run_id if signal_id is _SIGNAL_ID_UNSET else signal_id
    return {
        "order_id": make_order_id(_signal_id, portfolio_id, portfolio_version, ticker, session_date, action),
        "signal_id": _signal_id,
        "signal_run_id": signal_run_id,
        "portfolio_id": portfolio_id,
        "portfolio_version": portfolio_version,
        "ticker": ticker,
        "strategy": strategy,
        "intended_execution_session": session_date,
        "action": action,
        "action_origin": action_origin,
        "target_value": round(float(target_value), 2),
        "pyramid_remaining": round(float(pyramid_remaining), 2),
        "reason": reason,
        "signal_price": signal_price,
        "execution_version": execution_version,
        "status": PENDING_PRICE,
        "created_at": now_utc().isoformat(),
        "updated_at": now_utc().isoformat(),
        "attempted_at": None,
        "failure_reason": None,
        "trade_id": None,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _append(record: dict) -> None:
    os.makedirs(os.path.dirname(ORDERS_FILE), exist_ok=True)
    # Touch lock file before opening content file to guarantee lock exists
    open(_ORDERS_LOCK_FILE, "a").close()
    try:
        with open(_ORDERS_LOCK_FILE, "rb") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                with open(ORDERS_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
    except OSError as exc:
        raise RuntimeError(f"Kunne ikke skrive ordre til {ORDERS_FILE}: {exc}") from exc


def _truncate_corrupt_last_line() -> None:
    """Remove the corrupt last line from orders.jsonl under exclusive lock + fsync.

    Called after detecting a corrupt last line during load_orders(). Re-reads and
    re-validates the last line under the EXCLUSIVE lock before truncating — this closes
    the race window between the shared-lock read (in load_orders) and the exclusive-lock
    truncation: if another process appended a valid record in between, we skip truncation.
    """
    lock_dir = os.path.dirname(_ORDERS_LOCK_FILE)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)
    open(_ORDERS_LOCK_FILE, "a").close()
    with open(_ORDERS_LOCK_FILE, "rb") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            with open(ORDERS_FILE, "r+b") as f:
                content = f.read()
                if not content:
                    return

                # Re-verify under exclusive lock: find and re-check the last non-empty line
                stripped = content.rstrip(b"\n")
                last_nl = stripped.rfind(b"\n")
                last_line = stripped[last_nl + 1:] if last_nl >= 0 else stripped

                if not last_line:
                    return  # Nothing to truncate

                try:
                    json.loads(last_line.decode("utf-8"))
                    return  # Valid now — another process appended a good record, skip truncation
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass  # Still corrupt — proceed with truncation

                # Truncate to end of last valid line (keep the trailing \n of previous record)
                truncate_at = last_nl + 1 if last_nl >= 0 else 0
                f.seek(truncate_at)
                f.truncate()
                f.flush()
                os.fsync(f.fileno())
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def load_orders() -> dict:
    """Return {order_id: order} with the latest state per order.

    Since each update appends a full snapshot, the last record for a given
    order_id overrides earlier ones.

    Acquires a shared lock on read to prevent reading a partially-written record.
    Mid-history corruption (non-last line) is fail-closed — raises RuntimeError.
    A corrupt last line is repaired: truncated from disk under exclusive lock + fsync,
    then skipped gracefully (crash during write is the expected cause).
    """
    orders: dict = {}
    if not os.path.exists(ORDERS_FILE):
        return orders
    lock_dir = os.path.dirname(_ORDERS_LOCK_FILE)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)
    open(_ORDERS_LOCK_FILE, "a").close()
    with open(_ORDERS_LOCK_FILE, "rb") as lf:
        fcntl.flock(lf, fcntl.LOCK_SH)
        try:
            with open(ORDERS_FILE, encoding="utf-8") as f:
                raw_lines = f.readlines()
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)

    corrupt_last_line = False
    for i, raw_line in enumerate(raw_lines):
        line = raw_line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            orders[rec["order_id"]] = rec
        except json.JSONDecodeError:
            has_more = any(l.strip() for l in raw_lines[i + 1:])
            if has_more:
                raise RuntimeError(
                    f"Ordre-ledger er korrupt (linje {i + 1}/{len(raw_lines)}): "
                    f"{ORDERS_FILE}. "
                    "Korrupt JSON midt i filen — kan ikke repareres automatisk."
                ) from None
            # Last line only — crash during write. Repair disk state under exclusive lock.
            corrupt_last_line = True

    if corrupt_last_line:
        _truncate_corrupt_last_line()

    return orders


def save_order(
    order: dict,
    *,
    status: str | None = None,
    failure_reason: str | None = None,
    trade_id: str | None = None,
) -> dict:
    """
    Persist an order snapshot with optional field overrides.
    Returns the updated order dict.
    """
    updated = dict(order)
    updated["updated_at"] = now_utc().isoformat()
    updated["attempted_at"] = now_utc().isoformat()
    if status is not None:
        updated["status"] = status
    if failure_reason is not None:
        updated["failure_reason"] = failure_reason
    if trade_id is not None:
        updated["trade_id"] = trade_id
    _append(updated)
    return updated


# ---------------------------------------------------------------------------
# Order queries
# ---------------------------------------------------------------------------

def get_or_create_order(
    orders: dict,
    signal_run_id: str,
    ticker: str,
    strategy: str,
    session_date: str,
    action: str,
    target_value: float,
    reason: str,
    signal_price: float | None,
    execution_version: str,
    pyramid_remaining: float = 0.0,
    portfolio_id: str = "",
    portfolio_version: str = "",
    signal_id=_SIGNAL_ID_UNSET,
    action_origin: str | None = None,
) -> tuple[dict, bool]:
    """Return (order, is_new).

    Lookup order by new-format order_id first (signal_id + portfolio_id + portfolio_version).
    Falls back to legacy-format order_id (signal_run_id + ticker + strategy) only if the
    stored order's portfolio_id is empty or matches the current portfolio_id — prevents
    a legacy order for portfolio_A from blocking portfolio_B.
    Returns (existing, False) if found; (new, True) if not — caller must persist the new order.
    """
    _signal_id = signal_run_id if signal_id is _SIGNAL_ID_UNSET else signal_id
    order_id = make_order_id(_signal_id, portfolio_id, portfolio_version, ticker, session_date, action)
    if order_id in orders:
        return orders[order_id], False
    # Legacy fallback: check v1 order_id format to prevent double-fills on upgrade.
    # Portfolio-safe: only match if the stored portfolio_id is empty or matches ours.
    legacy_id = _make_legacy_order_id(signal_run_id, ticker, strategy, session_date, action)
    if legacy_id in orders and legacy_id != order_id:
        legacy_order = orders[legacy_id]
        stored_pid = legacy_order.get("portfolio_id", "")
        if stored_pid == "" or stored_pid == portfolio_id:
            return legacy_order, False
    order = build_order(
        signal_run_id, ticker, strategy, session_date, action,
        target_value, reason, signal_price, execution_version,
        pyramid_remaining=pyramid_remaining,
        portfolio_id=portfolio_id,
        portfolio_version=portfolio_version,
        signal_id=signal_id,
        action_origin=action_origin,
    )
    return order, True


def reconcile_settling_orders(orders: dict, strategy_name: str, state: dict) -> list:
    """Reconcile SETTLING orders using fill event WAL + commit_intent + portfolio state.

    Primary recovery path (WAL-based orders with commit_intent):
    - commit_intent found + current portfolio hash == intent hash:
        - persisted event present → EXECUTED
        - no persisted event → crash after portfolio save, before mark → reconstruct
    - commit_intent found + hashes do NOT match → crash before portfolio save → PENDING_PRICE
    - no commit_intent → legacy path (fill events + ticker-presence check)

    Legacy path (orders without commit_intent — pre-Round4 code):
    - fill event present + portfolio reflects fill → crash after save → reconstruct
    - fill event present + portfolio does NOT reflect fill → crash before save → PENDING_PRICE
    - no fill event + portfolio reflects fill → very old code (no WAL) → EXECUTED
    - no fill event + no portfolio fill → crash before save → PENDING_PRICE

    Raises RuntimeError if the fill ledger cannot be read (fail-closed — caller must alert).
    Returns the list of reconciled orders.
    """
    from modules.fills import (  # noqa: PLC0415
        compute_portfolio_state_hash,
        load_fill_events,
        mark_fill_persisted,
    )

    # Raises RuntimeError on ledger read failure — do not catch; let it propagate fail-closed
    fill_events_by_order, commit_intents = load_fill_events()

    # Build map: order_id → commit_intent (for this strategy)
    order_to_intent: dict[str, dict] = {}
    for ci in commit_intents:
        if ci.get("strategy") != strategy_name:
            continue
        for fill_ref in ci.get("fills", []):
            oid = fill_ref.get("order_id")
            if oid:
                # Last commit_intent wins (most recent batch attempt)
                order_to_intent[oid] = ci

    # Compute current portfolio hash once for all orders in this strategy
    current_hash = compute_portfolio_state_hash(state)
    positions = state.get("positions", {})

    reconciled = []

    for order in list(orders.values()):
        if order.get("status") != SETTLING:
            continue
        if order.get("strategy") != strategy_name:
            continue

        order_id = order["order_id"]
        ticker = order.get("ticker")
        action = order.get("action")

        order_fill_events = fill_events_by_order.get(order_id, [])
        filling_ev = next((e for e in order_fill_events if e.get("status") == "filling"), None)
        persisted_ev = next((e for e in order_fill_events if e.get("status") == "persisted"), None)
        has_filling = filling_ev is not None

        commit_intent = order_to_intent.get(order_id)

        if commit_intent is not None:
            # ---------------------------------------------------------------
            # New WAL-based path: commit_intent proves intended state
            # ---------------------------------------------------------------
            intent_hash = commit_intent.get("post_portfolio_state_hash", "")
            commit_id = commit_intent.get("commit_id")

            if current_hash != intent_hash:
                # Portfolio hash doesn't match intent — crash happened before portfolio save
                updated = save_order(
                    order, status=PENDING_PRICE,
                    failure_reason=(
                        "crash-recovery: commit_intent hash mismatch — "
                        "portfolio not saved, queued for retry"
                    ),
                )
            elif persisted_ev is not None:
                # Portfolio matches + persisted event exists → fully committed → EXECUTED
                updated = save_order(order, status=EXECUTED)
            elif has_filling:
                # Portfolio matches intent but persisted marker missing — crash after save,
                # before mark_fill_persisted. Reconstruct now.
                mark_fill_persisted(
                    order_id,
                    filling_ev.get("fill_attempt_id", ""),
                    filling_ev.get("content_hash", ""),
                    commit_id=commit_id,
                    post_portfolio_state_hash=intent_hash,
                )
                updated = save_order(order, status=EXECUTED)
            else:
                # commit_intent exists but no filling event — should not happen in normal flow
                updated = save_order(
                    order, status=PENDING_PRICE,
                    failure_reason=(
                        "crash-recovery: commit_intent utan filling-event — queued for retry"
                    ),
                )

        else:
            # ---------------------------------------------------------------
            # Legacy path: no commit_intent — use fill events + portfolio state
            # Isolated clearly from the new WAL path above.
            # ---------------------------------------------------------------

            # Portfolio state check (authoritative fallback for legacy orders)
            if action == "BUY":
                portfolio_has_fill = ticker in positions
            elif action == "SELL":
                portfolio_has_fill = ticker not in positions
            elif action == "PYRAMID_FILL":
                pos = positions.get(ticker, {})
                portfolio_has_fill = bool(pos) and not pos.get("is_partial", True)
            else:
                portfolio_has_fill = False

            # Strict: persisted marker must reference the filling event's fill_attempt_id
            fill_persisted = (
                filling_ev is not None
                and persisted_ev is not None
                and (
                    not persisted_ev.get("fill_attempt_id")
                    or not filling_ev.get("fill_attempt_id")
                    or persisted_ev.get("fill_attempt_id") == filling_ev.get("fill_attempt_id")
                )
            )

            if fill_persisted:
                # Fill WAL confirmed persisted → portfolio definitely on disk → EXECUTED
                updated = save_order(order, status=EXECUTED)
            elif order_fill_events and has_filling and portfolio_has_fill:
                # Fill event exists but not marked persisted — crash after portfolio save,
                # before mark_fill_persisted. Reconstruct: mark persisted now, EXECUTED.
                filling_events = [e for e in order_fill_events if e.get("status") == "filling"]
                last_filling = filling_events[-1]
                mark_fill_persisted(
                    order_id,
                    last_filling.get("fill_attempt_id", ""),
                    last_filling.get("content_hash", ""),
                )
                updated = save_order(order, status=EXECUTED)
            elif not order_fill_events and portfolio_has_fill:
                # No fill events (legacy system before fills.jsonl) — portfolio is authoritative
                updated = save_order(order, status=EXECUTED)
            else:
                # Portfolio does not reflect fill → crash before portfolio save → retry
                updated = save_order(
                    order, status=PENDING_PRICE,
                    failure_reason="crash-recovery: fill not persisted to portfolio, queued for retry",
                )

        orders[order_id] = updated
        reconciled.append(updated)

    return reconciled


def recover_settling_orders(orders: dict) -> list:
    """Removed: blindly converting SETTLING → FAILED_PRICE is incorrect.

    Use reconcile_settling_orders() instead, which uses the fill-event WAL and
    portfolio state to correctly classify crash scenarios as EXECUTED or PENDING_PRICE.
    FAILED_PRICE is reserved for permanent price failures, not transient crashes.
    """
    raise NotImplementedError(
        "recover_settling_orders() has been removed. "
        "Use reconcile_settling_orders(orders, strategy_name, state) instead."
    )


def get_pending_for_session(orders: dict, session_date: str, strategy: str | None = None) -> list:
    """
    Return all PENDING_PRICE orders for the given session (optionally filtered by strategy).
    These are candidates for same-session retry.
    """
    return [
        o for o in orders.values()
        if o["status"] == PENDING_PRICE
        and o["intended_execution_session"] == session_date
        and (strategy is None or o["strategy"] == strategy)
    ]


def expire_stale_orders(orders: dict, current_session_date: str, *, _now=None) -> list:
    """
    Transition PENDING_PRICE orders to EXPIRED when their session is definitively over.

    Past sessions (intended < current_session_date) are always expired.
    Current session (intended == current_session_date) is expired only once the exchange
    has actually closed — this handles early-close days (e.g. day after Thanksgiving).
    If the calendar is unavailable for the current-session check the order is left as
    PENDING_PRICE (fail-open: do not expire prematurely).

    Args:
        orders: mutable {order_id: order} dict — modified in place.
        current_session_date: today's intended execution session (YYYY-MM-DD).
        _now: override for current UTC time; used in tests only.
    """
    from modules.state import now_utc as _now_utc
    now = _now if _now is not None else _now_utc()

    expired = []
    for order_id, order in list(orders.items()):
        if order["status"] != PENDING_PRICE:
            continue

        intended = order["intended_execution_session"]

        if intended < current_session_date:
            failure_reason = f"Session {intended} ended without fill"

        elif intended == current_session_date:
            # Expire only if the session has already closed (respects early-close days)
            try:
                from modules.exchange_calendar import (  # noqa: PLC0415
                    CalendarUnavailableError,
                    session_close_utc,
                )
                close_utc = session_close_utc(intended)
                if now < close_utc:
                    continue  # session still open — retry window remains
                failure_reason = (
                    f"Session {intended} closed at {close_utc.isoformat()} without fill"
                )
            except Exception:
                # Calendar unavailable — fail open, don't expire current session
                continue

        else:
            continue  # future session — never expire

        updated = save_order(order, status=EXPIRED, failure_reason=failure_reason)
        orders[order_id] = updated
        expired.append(updated)

    return expired
