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


def make_candidate_signal_id(signal_run_id: str, strategy: str, ticker: str,
                              action: str = "BUY") -> str:
    """Stable per-candidate signal identifier: one per (signal_run_id, strategy, ticker, action).

    Two portfolios consuming the same signal get the SAME signal_id, but different order_ids
    (because portfolio_id differs). This allows tracing a recommendation across portfolios.
    """
    raw = f"{signal_run_id}|{strategy}|{ticker}|{action}"
    return "sig-" + hashlib.sha256(raw.encode()).hexdigest()[:12]


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
) -> dict:
    """Build a new order dict (not yet persisted).

    signal_id: stable content-addressed signal identifier (e.g. signal_content_hash).
    Omit to default to signal_run_id (backward compat).
    Pass None explicitly for safety-action orders that have no associated signal.
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

    Called after detecting a corrupt last line during load_orders(). Prevents the next
    append from writing after a corrupt partial record (which would leave the file in a
    permanently broken mid-file state on the next load).
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
                if content.endswith(b"\n"):
                    # Complete line (ends with \n) but invalid JSON — remove entire last line
                    last_nl = len(content) - 1
                    prev_nl = content.rfind(b"\n", 0, last_nl)
                    truncate_at = prev_nl + 1  # 0 if prev_nl == -1 (single corrupt line)
                else:
                    # File truncated mid-write (no trailing \n) — truncate at last valid \n
                    last_nl = content.rfind(b"\n")
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
) -> tuple[dict, bool]:
    """Return (order, is_new).

    Lookup order by new-format order_id first (signal_id + portfolio_id + portfolio_version).
    Falls back to legacy-format order_id (signal_run_id + ticker + strategy) to prevent
    double-fills when upgrading from old code that used a different key scheme.
    Returns (existing, False) if found; (new, True) if not — caller must persist the new order.
    """
    _signal_id = signal_run_id if signal_id is _SIGNAL_ID_UNSET else signal_id
    order_id = make_order_id(_signal_id, portfolio_id, portfolio_version, ticker, session_date, action)
    if order_id in orders:
        return orders[order_id], False
    # Legacy fallback: check v1 order_id format to prevent double-fills on upgrade
    legacy_id = _make_legacy_order_id(signal_run_id, ticker, strategy, session_date, action)
    if legacy_id in orders and legacy_id != order_id:
        return orders[legacy_id], False
    order = build_order(
        signal_run_id, ticker, strategy, session_date, action,
        target_value, reason, signal_price, execution_version,
        pyramid_remaining=pyramid_remaining,
        portfolio_id=portfolio_id,
        portfolio_version=portfolio_version,
        signal_id=signal_id,
    )
    return order, True


def reconcile_settling_orders(orders: dict, strategy_name: str, state: dict) -> list:
    """Reconcile SETTLING orders using fill event WAL + portfolio state.

    Called after loading strategy state so we can tell whether the portfolio was saved
    before a crash. Uses the fills.jsonl WAL as the primary source; falls back to
    portfolio state check for legacy orders (before fills.jsonl existed).

    Crash-before-portfolio-save → PENDING_PRICE (not FAILED_PRICE — transient, retry).
    Crash-after-portfolio-save → reconstructed as EXECUTED (portfolio is authoritative).

    Returns the list of reconciled orders.
    """
    reconciled = []
    positions = state.get("positions", {})

    for order in list(orders.values()):
        if order.get("status") != SETTLING:
            continue
        if order.get("strategy") != strategy_name:
            continue

        order_id = order["order_id"]
        ticker = order.get("ticker")
        action = order.get("action")

        # Portfolio state check (used as fallback if no fill events exist)
        if action == "BUY":
            portfolio_has_fill = ticker in positions
        elif action == "SELL":
            portfolio_has_fill = ticker not in positions
        elif action == "PYRAMID_FILL":
            pos = positions.get(ticker, {})
            portfolio_has_fill = bool(pos) and not pos.get("is_partial", True)
        else:
            portfolio_has_fill = False

        # Fill event WAL (most authoritative — available from fills.jsonl)
        try:
            from modules.fills import (  # noqa: PLC0415
                get_fill_events_for_order,
                is_fill_persisted,
                mark_fill_persisted,
            )
            fill_persisted = is_fill_persisted(order_id)
            fill_events = get_fill_events_for_order(order_id)
        except Exception:
            fill_persisted = False
            fill_events = []

        if fill_persisted:
            # Fill WAL confirmed persisted → portfolio definitely on disk → EXECUTED
            updated = save_order(order, status=EXECUTED)
        elif fill_events and portfolio_has_fill:
            # Fill event exists but not marked persisted — crash after portfolio save,
            # before mark_fill_persisted. Reconstruct: mark persisted now, EXECUTED.
            try:
                mark_fill_persisted(order_id)
            except Exception:
                pass
            updated = save_order(order, status=EXECUTED)
        elif not fill_events and portfolio_has_fill:
            # No fill events (legacy system before fills.jsonl) — portfolio is authoritative
            updated = save_order(order, status=EXECUTED)
        else:
            # Portfolio does not reflect fill → crash happened before portfolio save.
            # Not FAILED_PRICE (crash is transient) — reset to PENDING_PRICE for retry.
            updated = save_order(
                order, status=PENDING_PRICE,
                failure_reason="crash-recovery: fill not persisted to portfolio, queued for retry",
            )

        orders[order_id] = updated
        reconciled.append(updated)

    return reconciled


def recover_settling_orders(orders: dict) -> list:
    """
    Crash recovery: convert any SETTLING orders to FAILED_PRICE.

    SETTLING orders represent fills that were annotated but whose portfolio state
    was never persisted (crash between fill and save_strategy_state). On the next
    run they are unknown — conservative recovery marks them FAILED_PRICE so the
    position is not double-credited.

    Returns the list of recovered orders.
    """
    recovered = []
    for order_id, order in list(orders.items()):
        if order.get("status") == SETTLING:
            updated = save_order(
                order,
                status=FAILED_PRICE,
                failure_reason="crash-recovery: settling order not finalized",
            )
            orders[order_id] = updated
            recovered.append(updated)
    return recovered


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
