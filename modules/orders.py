"""
Persistent order state for the next_session_daily_open_v1 execution model.

Lifecycle:
  pending_price → executed  (fill succeeded)
  pending_price → expired   (session passed without fill)
  pending_price → failed_price  (permanent failure, not retried)

The order_id is deterministic: same (signal_run_id, ticker, strategy,
session_date, action) always yields the same order_id — this is the
idempotency key. Running execute twice for the same session cannot
create duplicate orders or double-fills.

Storage: STATE_DIR/orders.jsonl — append-only snapshots. Each update
appends a full order snapshot; the current state is the last record
per order_id when the file is read.
"""

import hashlib
import json
import os

from modules.state import STATE_DIR, now_utc

ORDERS_FILE = f"{STATE_DIR}/orders.jsonl"

# Statuses
PENDING_PRICE = "pending_price"
EXECUTED = "executed"
EXPIRED = "expired"
FAILED_PRICE = "failed_price"
CANCELLED = "cancelled"

TERMINAL = frozenset([EXECUTED, EXPIRED, FAILED_PRICE, CANCELLED])


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def make_order_id(signal_run_id: str, ticker: str, strategy: str,
                  session_date: str, action: str) -> str:
    """
    Deterministic order ID — identical inputs always produce the same 16-char hex string.
    This serves as both the primary key and the idempotency guard: if the bot runs
    execute twice for the same session, get_or_create returns the existing order.
    """
    raw = f"{signal_run_id}|{ticker}|{strategy}|{session_date}|{action}"
    return "ord-" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def make_trade_id(order_id: str, timestamp_iso: str) -> str:
    """Tie a trade fill back to its order. Not deterministic — unique per fill."""
    raw = f"{order_id}|{timestamp_iso}"
    return "trd-" + hashlib.sha256(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Order construction
# ---------------------------------------------------------------------------

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
) -> dict:
    """Build a new order dict (not yet persisted)."""
    return {
        "order_id": make_order_id(signal_run_id, ticker, strategy, session_date, action),
        "signal_run_id": signal_run_id,
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
    with open(ORDERS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_orders() -> dict:
    """
    Return {order_id: order} with the latest state per order.
    Since each update appends a full snapshot, the last record for a given
    order_id overrides earlier ones.
    """
    orders: dict = {}
    if not os.path.exists(ORDERS_FILE):
        return orders
    with open(ORDERS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                orders[rec["order_id"]] = rec
            except Exception:
                pass
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
) -> tuple[dict, bool]:
    """
    Return (order, is_new). If an order with the same deterministic ID already
    exists in `orders`, return it (is_new=False). Otherwise build a new one
    (is_new=True) — the caller is responsible for persisting it.
    """
    order_id = make_order_id(signal_run_id, ticker, strategy, session_date, action)
    if order_id in orders:
        return orders[order_id], False
    order = build_order(
        signal_run_id, ticker, strategy, session_date, action,
        target_value, reason, signal_price, execution_version,
        pyramid_remaining=pyramid_remaining,
    )
    return order, True


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
