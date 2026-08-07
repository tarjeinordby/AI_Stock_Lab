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

Integrity:
  - Every record includes a content_hash (SHA-256 of all fields except content_hash itself).
  - Reads acquire LOCK_SH; writes acquire LOCK_EX + fsync.
  - Mid-file corruption is fail-closed (RuntimeError).
  - Corrupt last line is repaired under LOCK_EX with re-verification (no race).
  - A "persisted" marker is only valid if a corresponding integrity-validated "filling" event exists.
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

# Required fields for "filling" events — validated on load
_FILL_SCHEMA_FIELDS = frozenset({
    "fill_id", "order_id", "trade_id",
    "portfolio_id", "portfolio_version", "strategy", "ticker", "action",
    "intended_execution_session", "actual_execution_session",
    "shares", "execution_price", "execution_version",
    "cash_before", "cash_after", "status", "written_at",
})


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def make_fill_id(order_id: str) -> str:
    """Deterministic fill ID — one fill per order, enables idempotent crash recovery."""
    return "fill-" + hashlib.sha256(order_id.encode()).hexdigest()[:12]


def _make_content_hash(record: dict) -> str:
    r = {k: v for k, v in record.items() if k != "content_hash"}
    return hashlib.sha256(
        json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _append_record(record: dict) -> None:
    FILLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _FILLS_LOCK_FILE.touch(exist_ok=True)
    record["content_hash"] = _make_content_hash(record)
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
    *,
    order_id: str,
    trade_id: str,
    signal_id,
    signal_run_id,
    portfolio_id: str,
    portfolio_version: str,
    strategy: str,
    ticker: str,
    action: str,
    intended_execution_session: str,
    actual_execution_session: str,
    shares: float,
    execution_price: float,
    execution_price_source: str = "next_session_daily_open_v1",
    execution_price_timestamp: str | None = None,
    execution_price_interval: str = "1d",
    gross_execution_price: float | None = None,
    gross_execution_value: float | None = None,
    total_execution_cost: float | None = None,
    net_cash_effect: float | None = None,
    slippage_bps: int = 0,
    slippage_amount: float = 0.0,
    commission_amount: float = 0.0,
    reason: str = "",
    execution_version: str = "",
    cash_before: float = 0.0,
    cash_after: float = 0.0,
    status: str = "filling",
) -> dict:
    """Write fill event to WAL before portfolio state is saved to disk.

    status='filling' until mark_fill_persisted() is called after save_strategy_state().
    All monetary fields are required to be present for full crash-recovery reconstruction.
    """
    fill_id = make_fill_id(order_id)
    _shares = round(float(shares), 6)
    _price = round(float(execution_price), 4)
    _gross_price = round(float(gross_execution_price if gross_execution_price is not None else execution_price), 4)
    _gross_value = round(float(gross_execution_value if gross_execution_value is not None else _shares * _price), 2)
    _commission = round(float(commission_amount), 4)
    _total_cost = round(float(total_execution_cost if total_execution_cost is not None else _commission), 4)
    _net = round(float(net_cash_effect if net_cash_effect is not None else _gross_value - _commission), 2)

    record = {
        "fill_id": fill_id,
        "order_id": order_id,
        "trade_id": trade_id,
        "signal_id": signal_id,
        "signal_run_id": signal_run_id,
        "portfolio_id": portfolio_id,
        "portfolio_version": portfolio_version,
        "strategy": strategy,
        "ticker": ticker,
        "action": action,
        "intended_execution_session": intended_execution_session,
        "actual_execution_session": actual_execution_session,
        "shares": _shares,
        "execution_price": _price,
        "execution_price_source": execution_price_source,
        "execution_price_timestamp": execution_price_timestamp,
        "execution_price_interval": execution_price_interval,
        "gross_execution_price": _gross_price,
        "slippage_bps": int(slippage_bps),
        "slippage_amount": round(float(slippage_amount), 4),
        "commission_amount": _commission,
        "gross_execution_value": _gross_value,
        "total_execution_cost": _total_cost,
        "net_cash_effect": _net,
        "reason": reason,
        "execution_version": execution_version,
        "cash_before": round(float(cash_before), 2),
        "cash_after": round(float(cash_after), 2),
        "status": status,
        "written_at": _utc_now(),
        "content_hash": "",  # filled by _append_record
    }
    _append_record(record)
    return record


def mark_fill_persisted(order_id: str) -> None:
    """Append a 'persisted' status event after portfolio state is durably saved.

    Raises RuntimeError if the WAL write fails — caller must NOT write EXECUTED on failure.
    """
    fill_id = make_fill_id(order_id)
    _append_record({
        "fill_id": fill_id,
        "order_id": order_id,
        "status": "persisted",
        "written_at": _utc_now(),
        "content_hash": "",  # filled by _append_record
    })


def _repair_fills_last_line() -> None:
    """Truncate corrupt last line from fills.jsonl under LOCK_EX with re-verification.

    Acquires LOCK_EX, re-reads the file to confirm the last line is still corrupt
    (guards against another process having appended a valid record since our LOCK_SH read),
    then truncates. This ensures detection + truncation are atomic under one exclusive lock.
    """
    _FILLS_LOCK_FILE.touch(exist_ok=True)
    try:
        with open(_FILLS_LOCK_FILE, "rb") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                with open(FILLS_FILE, "r+b") as f:
                    content = f.read()
                    if not content:
                        return

                    # Find the last non-empty line under the exclusive lock
                    stripped = content.rstrip(b"\n")
                    last_nl = stripped.rfind(b"\n")
                    last_line = stripped[last_nl + 1:] if last_nl >= 0 else stripped

                    if not last_line:
                        return

                    try:
                        json.loads(last_line.decode("utf-8"))
                        return  # Valid now — another process wrote a good record
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass  # Still corrupt — proceed

                    # Truncate to end of last valid line
                    truncate_at = last_nl + 1 if last_nl >= 0 else 0
                    f.seek(truncate_at)
                    f.truncate()
                    f.flush()
                    os.fsync(f.fileno())
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
    except OSError as exc:
        raise RuntimeError(f"Kunne ikke reparere fills.jsonl: {exc}") from exc


def load_fill_events() -> dict[str, list[dict]]:
    """Return {order_id: [events_in_order]} from fills.jsonl.

    - Acquires LOCK_SH for reading.
    - Validates content_hash for records that include it.
    - Mid-file corruption raises RuntimeError (fail-closed).
    - Corrupt last line is repaired under LOCK_EX with re-verification.
    - OSError on open raises RuntimeError (no silent empty return).
    - Validates required schema fields for 'filling' events.
    """
    result: dict[str, list[dict]] = {}
    if not FILLS_FILE.exists():
        return result

    FILLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _FILLS_LOCK_FILE.touch(exist_ok=True)

    try:
        with open(_FILLS_LOCK_FILE, "rb") as lf:
            fcntl.flock(lf, fcntl.LOCK_SH)
            try:
                with open(FILLS_FILE, encoding="utf-8") as f:
                    raw = f.read()
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
    except OSError as exc:
        raise RuntimeError(f"Fill-ledger kan ikke leses ({FILLS_FILE}): {exc}") from exc

    lines = raw.splitlines()
    corrupt_last = False

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        is_last = not any(ln.strip() for ln in lines[i + 1:])

        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            if is_last:
                corrupt_last = True
                continue
            raise RuntimeError(
                f"Fill-ledger korrupt JSON midt i filen (linje {i + 1}/{len(lines)}): "
                f"{FILLS_FILE} — fail-closed: {exc}"
            ) from exc

        # Content hash validation — only for records that carry the field
        stored_hash = rec.get("content_hash", "")
        if stored_hash:
            computed = _make_content_hash(rec)
            if stored_hash != computed:
                if is_last:
                    corrupt_last = True
                    continue
                raise RuntimeError(
                    f"Fill-ledger integritets-feil (linje {i + 1}): "
                    f"content_hash mismatch — forventet {computed[:8]}…, "
                    f"lagret {stored_hash[:8]}… — fail-closed"
                )

        # Schema validation for "filling" events
        if rec.get("status") == "filling":
            missing = [fld for fld in _FILL_SCHEMA_FIELDS if fld not in rec]
            if missing:
                if is_last:
                    corrupt_last = True
                    continue
                raise RuntimeError(
                    f"Fill-ledger skjema-feil (linje {i + 1}): "
                    f"mangler felt {missing} — fail-closed"
                )

        oid = rec.get("order_id")
        if oid:
            result.setdefault(oid, []).append(rec)

    if corrupt_last:
        _repair_fills_last_line()

    return result


def get_fill_events_for_order(order_id: str) -> list[dict]:
    """Return all fill events for a given order_id (chronological)."""
    return load_fill_events().get(order_id, [])


def is_fill_persisted(order_id: str) -> bool:
    """Return True if the fill for this order was durably persisted.

    Requires both:
    - A valid, integrity-checked "filling" event (the authoritative trade record)
    - A "persisted" marker appended after save_strategy_state()
    """
    events = load_fill_events().get(order_id, [])
    has_filling = any(e.get("status") == "filling" for e in events)
    if not has_filling:
        return False  # Orphaned persisted marker — not valid
    return any(e.get("status") == "persisted" for e in events)


def project_fills_to_trades(trades_df: "pd.DataFrame") -> "pd.DataFrame":
    """Replay persisted fill events into trades_df — idempotent, deduplicated by trade_id.

    Called at startup in run_execute() after load_trades() to recover any trade rows
    that were lost when the process crashed after EXECUTED but before save_csv(trades_df).

    Raises RuntimeError if the fill ledger is unreadable (caller should alert + decide).
    """
    import pandas as pd  # noqa: PLC0415

    events_by_order = load_fill_events()  # Raises RuntimeError if ledger is corrupt

    existing_trade_ids: set[str] = set()
    if not trades_df.empty and "trade_id" in trades_df.columns:
        existing_trade_ids = set(trades_df["trade_id"].dropna().astype(str))

    new_rows = []
    for order_events in events_by_order.values():
        filling = next((e for e in order_events if e.get("status") == "filling"), None)
        if filling is None:
            continue
        if not any(e.get("status") == "persisted" for e in order_events):
            continue  # Only project fully-persisted fills

        trade_id = filling.get("trade_id")
        if not trade_id or trade_id in existing_trade_ids:
            continue  # Already present in trades.csv

        price = filling.get("execution_price", filling.get("price", 0.0))
        commission = filling.get("commission_amount", filling.get("cost", 0.0))
        gross_value = filling.get(
            "gross_execution_value",
            round(float(filling.get("shares", 0)) * float(price), 2),
        )
        session = filling.get("actual_execution_session", filling.get("session_date", ""))

        row = {
            "date": session,
            "timestamp_utc": filling.get("execution_price_timestamp", filling.get("written_at", "")),
            "strategy": filling.get("strategy", ""),
            "action": filling.get("action", ""),
            "ticker": filling.get("ticker", ""),
            "shares": filling.get("shares", 0.0),
            "price": price,
            "value": gross_value,
            "cost": commission,
            "reason": filling.get("reason", ""),
            "order_id": filling.get("order_id"),
            "trade_id": trade_id,
            "signal_run_id": filling.get("signal_run_id"),
            "execution_version": filling.get("execution_version", ""),
            "gross_execution_price": filling.get("gross_execution_price", price),
            "commission_amount": commission,
            "execution_session": session,
            "execution_price_source": filling.get("execution_price_source", ""),
            "order_status": "filled",
            "slippage_bps": filling.get("slippage_bps", 0),
            "slippage_amount": filling.get("slippage_amount", 0.0),
            "gross_execution_value": gross_value,
            "total_execution_cost": filling.get("total_execution_cost", commission),
            "net_cash_effect": filling.get("net_cash_effect", gross_value - float(commission)),
            "_recovered_from_wal": True,
        }
        new_rows.append(row)
        existing_trade_ids.add(trade_id)

    if new_rows:
        recovered_df = pd.DataFrame(new_rows)
        trades_df = pd.concat([trades_df, recovered_df], ignore_index=True)
        print(f"WAL-gjenoppbygging: {len(new_rows)} handelrad(er) gjenopprettet fra fills.jsonl")

    return trades_df
