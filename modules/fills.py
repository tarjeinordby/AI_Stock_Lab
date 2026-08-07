"""
Write-ahead fill/trade event ledger for crash-consistent portfolio persistence.

Sequence for every fill:
  1. execute_buy/sell/pyramid → trade data known (in-memory state mutated)
  2. write_fill_event(..., status="filling")  ← WAL entry BEFORE portfolio disk write
  3. compute_portfolio_state_hash(state)      ← canonical hash of cash+positions
  4. write_commit_intent(...)                 ← intent written BEFORE portfolio save
  5. save_strategy_state_atomic(...)          ← portfolio persisted to disk
  6. verify: reload + hash must equal pre-save hash
  7. mark_fill_persisted(order_id, fill_attempt_id, filling_content_hash,
                         commit_id, post_portfolio_state_hash) ← persisted event
  8. save_order(EXECUTED)                     ← terminal status

Recovery via reconcile_settling_orders:
  - commit_intent found + portfolio hash matches intent hash:
      - persisted event present → EXECUTED
      - no persisted event → reconstruct (write persisted, EXECUTED)
  - commit_intent found + portfolio hash does NOT match → PENDING_PRICE (crash before save)
  - no commit_intent → legacy path (fill_event + portfolio state check)
  - no fill_event + position in portfolio → legacy system (no WAL) → EXECUTED
  - no fill_event + no position → legacy crash before save → PENDING_PRICE retry

Integrity:
  - Every record includes a mandatory content_hash (SHA-256 of all fields except content_hash).
  - Missing or empty content_hash → fail-closed (no legacy path).
  - fill_attempt_id is a UUID per write — unique even when retrying at the same price/session.
  - Persisted marker cross-references fill_attempt_id AND filling_content_hash.
  - Resolver requires exactly one filling event matching both ID and hash.
  - action must be one of {BUY, SELL, PYRAMID_FILL}.
  - Cash direction validated: BUY/PYRAMID_FILL cash_after ≤ cash_before; SELL cash_after ≥ cash_before.
  - Cash arithmetic validated: cash_after ≈ cash_before + net_cash_effect (±$0.02 tolerance).
  - fill_id validated: must equal make_fill_id(order_id).
  - Reads acquire LOCK_SH; writes acquire LOCK_EX + fsync.
  - Mid-file corruption is fail-closed (RuntimeError).
  - Corrupt last line is repaired under LOCK_EX with re-verification (no race).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

FILLS_FILE = Path("data_v4/ledger/fills.jsonl")
_FILLS_LOCK_FILE = Path("data_v4/ledger/fills.jsonl.lock")

_VALID_ACTIONS = frozenset({"BUY", "SELL", "PYRAMID_FILL"})
_VALID_STATUSES = frozenset({"filling", "persisted", "commit_intent"})

# Required fields for "filling" events — validated on load
_FILL_SCHEMA_FIELDS = frozenset({
    "fill_id", "fill_attempt_id", "order_id", "trade_id",
    "signal_id", "signal_run_id",
    "portfolio_id", "portfolio_version", "strategy", "ticker", "action",
    "intended_execution_session", "actual_execution_session",
    "execution_price", "execution_price_source", "execution_price_timestamp",
    "execution_price_interval", "execution_version",
    "gross_execution_price", "shares",
    "slippage_bps", "slippage_amount", "commission_amount",
    "gross_execution_value", "total_execution_cost", "net_cash_effect",
    "cash_before", "cash_after", "reason", "status", "written_at",
})

# Required fields for "persisted" events — validated on load
_PERSISTED_SCHEMA_FIELDS = frozenset({
    "fill_id", "fill_attempt_id", "filling_content_hash",
    "order_id", "post_portfolio_state_hash", "status", "written_at",
})

# Required fields for "commit_intent" events — validated on load
_COMMIT_INTENT_SCHEMA_FIELDS = frozenset({
    "commit_id", "strategy", "portfolio_id", "portfolio_version",
    "post_portfolio_state_hash", "fills", "status", "written_at",
})


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def make_fill_id(order_id: str) -> str:
    """Deterministic fill ID — one fill per order, enables idempotent crash recovery."""
    return "fill-" + hashlib.sha256(order_id.encode()).hexdigest()[:12]


def compute_portfolio_state_hash(state: dict) -> str:
    """Full SHA-256 of canonical cash + positions using real portfolio field names.

    Covers all economically-relevant fields: shares, avg_price, last_price,
    highest_price, is_partial, pyramid_remaining_value, pyramid_min_price.
    Uses full SHA-256 (64 hex chars) for tamper detection.
    """
    canonical = {
        "cash": round(float(state.get("cash", 0)), 2),
        "positions": {
            ticker: {
                "avg_price": round(float(pos.get("avg_price", 0)), 4),
                "highest_price": round(float(pos.get("highest_price", 0)), 4),
                "is_partial": bool(pos.get("is_partial", False)),
                "last_price": round(float(pos.get("last_price", 0)), 4),
                "pyramid_min_price": round(float(pos.get("pyramid_min_price", 0)), 4),
                "pyramid_remaining_value": round(float(pos.get("pyramid_remaining_value", 0)), 2),
                "shares": round(float(pos.get("shares", 0)), 6),
            }
            for ticker, pos in sorted(state.get("positions", {}).items())
        },
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


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

    Returns the record dict with fill_attempt_id and content_hash set.
    fill_attempt_id is a UUID — unique per write attempt even with identical price/session.
    net_cash_effect fallback is action-aware: negative for BUY/PYRAMID_FILL, positive for SELL.
    """
    fill_id = make_fill_id(order_id)
    # UUID per write — unique even if retrying at same price and session
    fa_id = "fa-" + uuid.uuid4().hex[:12]
    _shares = round(float(shares), 6)
    _price = round(float(execution_price), 4)
    _gross_price = round(float(gross_execution_price if gross_execution_price is not None else execution_price), 4)
    _gross_value = round(float(gross_execution_value if gross_execution_value is not None else _shares * _price), 2)
    _commission = round(float(commission_amount), 4)
    _total_cost = round(float(total_execution_cost if total_execution_cost is not None else _commission), 4)
    # Action-aware net_cash_effect fallback: BUY/PYRAMID_FILL is cash-out (negative)
    if net_cash_effect is not None:
        _net = round(float(net_cash_effect), 2)
    elif action in ("BUY", "PYRAMID_FILL"):
        _net = round(-(_gross_value + _commission), 2)
    else:  # SELL
        _net = round(_gross_value - _commission, 2)

    record = {
        "fill_id": fill_id,
        "fill_attempt_id": fa_id,
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


def write_commit_intent(
    *,
    strategy: str,
    portfolio_id: str,
    portfolio_version: str,
    post_portfolio_state_hash: str,
    fills: list[dict],
) -> dict:
    """Write a commit_intent event before the portfolio save.

    The commit_intent proves what we INTENDED to persist. On recovery:
    - If current portfolio hash matches post_portfolio_state_hash → save completed.
    - If not → crash before save → retry fills as PENDING_PRICE.

    fills: list of {order_id, fill_attempt_id, filling_content_hash} dicts.
    Returns the written record dict with commit_id and content_hash set.
    """
    commit_id = "ci-" + uuid.uuid4().hex[:12]
    record = {
        "commit_id": commit_id,
        "strategy": strategy,
        "portfolio_id": portfolio_id,
        "portfolio_version": portfolio_version,
        "post_portfolio_state_hash": post_portfolio_state_hash,
        "fills": fills,
        "status": "commit_intent",
        "written_at": _utc_now(),
        "content_hash": "",  # filled by _append_record
    }
    _append_record(record)
    return record


def mark_fill_persisted(
    order_id: str,
    fill_attempt_id: str,
    filling_content_hash: str,
    post_portfolio_state_hash: str | None = None,
    commit_id: str | None = None,
) -> None:
    """Append a 'persisted' marker after portfolio state is durably saved and verified.

    References the exact filling attempt by fill_attempt_id + filling_content_hash.
    References the commit_intent by commit_id (if provided).
    Raises RuntimeError if the WAL write fails — caller must NOT write EXECUTED on failure.
    """
    fill_id = make_fill_id(order_id)
    _append_record({
        "fill_id": fill_id,
        "fill_attempt_id": fill_attempt_id,
        "filling_content_hash": filling_content_hash,
        "order_id": order_id,
        "commit_id": commit_id,
        "post_portfolio_state_hash": post_portfolio_state_hash,
        "status": "persisted",
        "written_at": _utc_now(),
        "content_hash": "",  # filled by _append_record
    })


def _repair_fills_last_line() -> None:
    """Truncate corrupt last line from fills.jsonl under LOCK_EX with re-verification."""
    _FILLS_LOCK_FILE.touch(exist_ok=True)
    try:
        with open(_FILLS_LOCK_FILE, "rb") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                with open(FILLS_FILE, "r+b") as f:
                    content = f.read()
                    if not content:
                        return

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

                    truncate_at = last_nl + 1 if last_nl >= 0 else 0
                    f.seek(truncate_at)
                    f.truncate()
                    f.flush()
                    os.fsync(f.fileno())
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
    except OSError as exc:
        raise RuntimeError(f"Kunne ikke reparere fills.jsonl: {exc}") from exc


def load_fill_events() -> tuple[dict[str, list[dict]], list[dict]]:
    """Return (events_by_order, commit_intents) from fills.jsonl.

    events_by_order: {order_id: [events_in_order]}
    commit_intents: list of commit_intent records (in chronological order)

    Integrity checks:
    - content_hash MANDATORY for all records — missing or empty → fail-closed.
    - filling events: full schema validation, action/cash/arithmetic checks.
    - persisted events: schema validation, filling_content_hash must be non-empty.
    - fill_id validated: must equal make_fill_id(order_id) for filling events.
    - Cash arithmetic: cash_after ≈ cash_before + net_cash_effect (±$0.02).
    - Mid-file corruption raises RuntimeError (fail-closed).
    - Corrupt last line is repaired under LOCK_EX with re-verification.
    - OSError on open raises RuntimeError (no silent empty return).
    """
    result: dict[str, list[dict]] = {}
    commit_intents: list[dict] = []
    if not FILLS_FILE.exists():
        return result, commit_intents

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

        # content_hash MANDATORY — missing or empty is always fail-closed (no legacy path,
        # no "corrupt last line" repair). Hash mismatch on last line is repairable.
        stored_hash = rec.get("content_hash", "")
        if not stored_hash:
            raise RuntimeError(
                f"Fill-ledger integritets-feil (linje {i + 1}): "
                f"content_hash mangler eller er tom — fail-closed (ingen legacy-sti)"
            )
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

        status = rec.get("status", "")

        # commit_intent events — collect separately (no order_id)
        if status == "commit_intent":
            missing_ci = [fld for fld in _COMMIT_INTENT_SCHEMA_FIELDS if fld not in rec]
            if missing_ci:
                raise RuntimeError(
                    f"Fill-ledger skjema-feil commit_intent (linje {i + 1}): "
                    f"mangler felt {missing_ci} — fail-closed"
                )
            if not rec.get("post_portfolio_state_hash"):
                raise RuntimeError(
                    f"Fill-ledger commit_intent (linje {i + 1}): "
                    f"post_portfolio_state_hash mangler/tom — fail-closed"
                )
            commit_intents.append(rec)
            continue

        # Schema and semantic validation for "filling" events — all are fail-closed
        if status == "filling":
            missing = [fld for fld in _FILL_SCHEMA_FIELDS if fld not in rec]
            if missing:
                raise RuntimeError(
                    f"Fill-ledger skjema-feil (linje {i + 1}): "
                    f"mangler felt {missing} — fail-closed"
                )

            # fill_id must equal make_fill_id(order_id)
            order_id_val = rec.get("order_id", "")
            expected_fill_id = make_fill_id(order_id_val)
            if rec.get("fill_id") != expected_fill_id:
                raise RuntimeError(
                    f"Fill-ledger fill_id-feil (linje {i + 1}): "
                    f"forventet {expected_fill_id!r}, faktisk {rec.get('fill_id')!r} — fail-closed"
                )

            action = rec.get("action", "")
            if action not in _VALID_ACTIONS:
                raise RuntimeError(
                    f"Fill-ledger ugyldig action '{action}' (linje {i + 1}): "
                    f"må være ett av {sorted(_VALID_ACTIONS)} — fail-closed"
                )

            shares = rec.get("shares", 0)
            price = rec.get("execution_price", 0)
            if not (isinstance(shares, (int, float)) and shares > 0):
                raise RuntimeError(
                    f"Fill-ledger ugyldige shares={shares!r} (linje {i + 1}): "
                    f"må være positiv numerisk — fail-closed"
                )
            if not (isinstance(price, (int, float)) and price > 0):
                raise RuntimeError(
                    f"Fill-ledger ugyldig execution_price={price!r} (linje {i + 1}): "
                    f"må være positiv numerisk — fail-closed"
                )

            cash_before = float(rec.get("cash_before", 0))
            cash_after = float(rec.get("cash_after", 0))
            net_cash_effect = float(rec.get("net_cash_effect", 0))
            tolerance = 0.02

            # Cash direction check
            if action in ("BUY", "PYRAMID_FILL"):
                if cash_after > cash_before + tolerance:
                    raise RuntimeError(
                        f"Fill-ledger cash-feil (linje {i + 1}): "
                        f"{action} men cash_after ({cash_after}) > cash_before ({cash_before}) "
                        f"— fail-closed"
                    )
            elif action == "SELL":
                if cash_after < cash_before - tolerance:
                    raise RuntimeError(
                        f"Fill-ledger cash-feil (linje {i + 1}): "
                        f"SELL men cash_after ({cash_after}) < cash_before ({cash_before}) "
                        f"— fail-closed"
                    )

            # Cash arithmetic check: cash_after ≈ cash_before + net_cash_effect
            expected_cash_after = round(cash_before + net_cash_effect, 2)
            if abs(cash_after - expected_cash_after) > tolerance:
                raise RuntimeError(
                    f"Fill-ledger cash-aritmetikk-feil (linje {i + 1}): "
                    f"cash_before={cash_before} + net_cash_effect={net_cash_effect} "
                    f"= {expected_cash_after}, men cash_after={cash_after} — fail-closed"
                )

        # Schema validation for "persisted" events
        elif status == "persisted":
            missing_p = [fld for fld in _PERSISTED_SCHEMA_FIELDS if fld not in rec]
            if missing_p:
                raise RuntimeError(
                    f"Fill-ledger skjema-feil persisted (linje {i + 1}): "
                    f"mangler felt {missing_p} — fail-closed"
                )
            if not rec.get("filling_content_hash"):
                raise RuntimeError(
                    f"Fill-ledger integritets-feil (linje {i + 1}): "
                    f"filling_content_hash mangler/tom — fail-closed"
                )

        oid = rec.get("order_id")
        if oid:
            result.setdefault(oid, []).append(rec)

    # Duplicate fill_attempt_id with different content → tamper detected
    fa_id_to_hash: dict[str, str] = {}
    for oid, events in result.items():
        for ev in events:
            if ev.get("status") == "filling":
                fa_id = ev.get("fill_attempt_id", "")
                ch = ev.get("content_hash", "")
                if fa_id and fa_id in fa_id_to_hash and fa_id_to_hash[fa_id] != ch:
                    raise RuntimeError(
                        f"Fill-ledger: duplicate fill_attempt_id {fa_id!r} med ulik innhold "
                        f"— tamper detected — fail-closed"
                    )
                if fa_id:
                    fa_id_to_hash[fa_id] = ch

    if corrupt_last:
        _repair_fills_last_line()

    return result, commit_intents


def get_fill_events_for_order(order_id: str) -> list[dict]:
    """Return all fill events for a given order_id (chronological)."""
    events_by_order, _ = load_fill_events()
    return events_by_order.get(order_id, [])


def is_fill_persisted(order_id: str) -> bool:
    """Return True if the fill for this order was durably persisted.

    Strict resolver:
    - Requires a valid filling event (status=filling, fill_attempt_id set).
    - Requires a persisted marker whose fill_attempt_id AND filling_content_hash match.
    - Orphaned markers (no filling event) or mismatched IDs/hashes → False.
    """
    events_by_order, _ = load_fill_events()
    events = events_by_order.get(order_id, [])
    filling = next((e for e in events if e.get("status") == "filling"), None)
    if filling is None:
        return False
    persisted = next((e for e in events if e.get("status") == "persisted"), None)
    if persisted is None:
        return False
    # Strict cross-reference: fill_attempt_id must match
    p_fa = persisted.get("fill_attempt_id")
    f_fa = filling.get("fill_attempt_id")
    if p_fa and f_fa and p_fa != f_fa:
        return False
    # Also verify filling_content_hash if present
    ref_hash = persisted.get("filling_content_hash")
    if ref_hash and filling.get("content_hash") != ref_hash:
        return False
    return True


def _find_filling_by_attempt_id(
    order_events: list[dict], ref_fa_id: str, order_id: str
) -> dict:
    """Strict resolver: find exactly one filling event matching ref_fa_id.

    Raises RuntimeError if:
    - No filling event matches ref_fa_id.
    - Multiple filling events match ref_fa_id with different content (tamper detected).
    """
    matching = [
        e for e in order_events
        if e.get("status") == "filling" and e.get("fill_attempt_id") == ref_fa_id
    ]
    if not matching:
        raise RuntimeError(
            f"Ingen filling-event for fill_attempt_id={ref_fa_id} (order {order_id}) — "
            f"persisted marker er foreldreløs — fail-closed"
        )
    if len(matching) > 1:
        hashes = {e.get("content_hash") for e in matching}
        if len(hashes) > 1:
            raise RuntimeError(
                f"Fill-ledger: duplicate fill_attempt_id={ref_fa_id} (order {order_id}) "
                f"med ulik innhold — tamper detected — fail-closed"
            )
    return matching[0]


def project_fills_to_trades(trades_df: "pd.DataFrame") -> "pd.DataFrame":
    """Replay persisted fill events into trades_df — idempotent, deduplicated by trade_id.

    Uses fill_attempt_id in the persisted marker to find the exact filling event.
    Raises RuntimeError if:
    - The fill ledger is unreadable.
    - A persisted marker references a fill_attempt_id with no matching filling event.
    - The filling_content_hash in the persisted marker does not match the filling event.
    - Multiple filling events have the same fill_attempt_id but different content (tamper).
    """
    import pandas as pd  # noqa: PLC0415

    events_by_order, _ = load_fill_events()  # Raises RuntimeError if ledger is corrupt

    existing_trade_ids: set[str] = set()
    if not trades_df.empty and "trade_id" in trades_df.columns:
        existing_trade_ids = set(trades_df["trade_id"].dropna().astype(str))

    new_rows = []
    for order_id, order_events in events_by_order.items():
        persisted = next((e for e in order_events if e.get("status") == "persisted"), None)
        if persisted is None:
            continue

        # Find the exact filling event referenced by this persisted marker
        ref_fa_id = persisted.get("fill_attempt_id")
        if not ref_fa_id:
            raise RuntimeError(
                f"Persisted marker for order {order_id} mangler fill_attempt_id — "
                f"kan ikke projisere til trades — fail-closed"
            )

        filling = _find_filling_by_attempt_id(order_events, ref_fa_id, order_id)

        # Cross-verify the filling event's content hash if persisted marker has it
        ref_hash = persisted.get("filling_content_hash")
        if ref_hash and filling.get("content_hash") != ref_hash:
            raise RuntimeError(
                f"filling_content_hash mismatch for fill_attempt_id={ref_fa_id} "
                f"(order {order_id}): forventet {ref_hash[:8]}…, "
                f"faktisk {filling.get('content_hash', '')[:8]}… — fail-closed"
            )

        trade_id = filling.get("trade_id")
        if not trade_id or trade_id in existing_trade_ids:
            continue

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
