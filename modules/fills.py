"""
Write-ahead fill/trade event ledger for crash-consistent portfolio persistence.

Sequence for every fill:
  1. execute_buy/sell/pyramid → trade data known (in-memory state mutated)
  2. write_fill_event(..., status="filling")  ← WAL entry BEFORE portfolio disk write
  3. save_strategy_state_atomic(...)           ← portfolio persisted to disk
  4. mark_fill_persisted(order_id)            ← append "persisted" event to WAL
  5. save_order(EXECUTED)                     ← terminal status

Recovery via reconcile_settling_orders:
  - fill_event "persisted" → portfolio on disk → EXECUTED
  - fill_event "filling" + position in portfolio → crash after save, before mark → reconstruct
  - fill_event "filling" + no position in portfolio → crash before portfolio save → PENDING_PRICE retry
  - no fill_event + position in portfolio → legacy system (no WAL) → EXECUTED
  - no fill_event + no position → legacy crash before save → PENDING_PRICE retry
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

FILLS_FILE = Path("data_v4/ledger/fills.jsonl")
_FILLS_LOCK_FILE = Path("data_v4/ledger/fills.jsonl.lock")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def make_fill_id(order_id: str) -> str:
    """Deterministic fill ID — one fill per order, enables idempotent crash recovery."""
    return "fill-" + hashlib.sha256(order_id.encode()).hexdigest()[:12]


def _append_record(record: dict) -> None:
    FILLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _FILLS_LOCK_FILE.touch(exist_ok=True)
    try:
        with open(_FILLS_LOCK_FILE, "rb") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                with open(FILLS_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
    except OSError as exc:
        raise RuntimeError(f"Kunne ikke skrive fill-event til {FILLS_FILE}: {exc}") from exc


def write_fill_event(
    order_id: str,
    trade_id: str,
    signal_id,
    signal_run_id,
    portfolio_id: str,
    strategy: str,
    ticker: str,
    action: str,
    session_date: str,
    shares: float,
    price: float,
    value: float,
    cost: float,
    reason: str,
    execution_version: str,
    cash_before: float,
    cash_after: float,
    status: str = "filling",
) -> dict:
    """Write fill event to WAL before portfolio state is saved to disk.

    status='filling' until mark_fill_persisted() is called after save_strategy_state().
    """
    fill_id = make_fill_id(order_id)
    record = {
        "fill_id": fill_id,
        "order_id": order_id,
        "trade_id": trade_id,
        "signal_id": signal_id,
        "signal_run_id": signal_run_id,
        "portfolio_id": portfolio_id,
        "strategy": strategy,
        "ticker": ticker,
        "action": action,
        "session_date": session_date,
        "shares": round(float(shares), 6),
        "price": round(float(price), 2),
        "value": round(float(value), 2),
        "cost": round(float(cost), 2),
        "reason": reason,
        "execution_version": execution_version,
        "cash_before": round(float(cash_before), 2),
        "cash_after": round(float(cash_after), 2),
        "status": status,
        "written_at": _utc_now(),
    }
    _append_record(record)
    return record


def mark_fill_persisted(order_id: str) -> None:
    """Append a 'persisted' status event after portfolio state is durably saved."""
    fill_id = make_fill_id(order_id)
    _append_record({
        "fill_id": fill_id,
        "order_id": order_id,
        "status": "persisted",
        "written_at": _utc_now(),
    })


def load_fill_events() -> dict[str, list[dict]]:
    """Return {order_id: [events_in_order]} from fills.jsonl. Tolerates corrupt last line."""
    result: dict[str, list[dict]] = {}
    if not FILLS_FILE.exists():
        return result
    try:
        with open(FILLS_FILE, encoding="utf-8") as f:
            lines = f.readlines()
        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
                oid = rec.get("order_id")
                if oid:
                    result.setdefault(oid, []).append(rec)
            except json.JSONDecodeError:
                pass  # tolerate corrupt last line
    except OSError:
        return {}
    return result


def get_fill_events_for_order(order_id: str) -> list[dict]:
    """Return all fill events for a given order_id (chronological)."""
    return load_fill_events().get(order_id, [])


def is_fill_persisted(order_id: str) -> bool:
    """Return True if the fill for this order was durably persisted (portfolio on disk)."""
    return any(e.get("status") == "persisted" for e in get_fill_events_for_order(order_id))
