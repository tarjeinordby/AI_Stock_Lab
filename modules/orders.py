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
FAILED_RECONCILIATION = "failed_reconciliation"  # terminal: hash-mismatch / versionless legacy
CANCELLED = "cancelled"

TERMINAL = frozenset([EXECUTED, EXPIRED, FAILED_PRICE, FAILED_RECONCILIATION, CANCELLED])


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


def _resolve_ci_filling_strict(
    order_id: str,
    order_fill_events: list[dict],
    commit_intent: dict,
    expected_version: int,
) -> tuple[dict, dict]:
    """Shared strict validator: find and validate the filling referenced by commit_intent.

    Enforces the same integrity guarantees for both the persisted path and reconstruction:
    - Exactly one fill_ref for order_id in commit_intent.fills
    - Exactly one filling event matching fill_attempt_id
    - content_hash link is intact
    - filling.record_version is exactly expected_version (type-strict int check)

    Returns (fill_ref, filling_event) on success.
    Raises RuntimeError (fail-closed) on any failure.
    """
    _refs = [f for f in commit_intent.get("fills", []) if f.get("order_id") == order_id]
    if len(_refs) != 1:
        raise RuntimeError(
            f"commit_intent {commit_intent.get('commit_id')!r} har {len(_refs)} "
            f"fill_ref(s) for ordre {order_id} — forventet nøyaktig 1 — fail-closed"
        )
    fill_ref = _refs[0]
    ref_fa_id = fill_ref.get("fill_attempt_id", "")
    ref_hash = fill_ref.get("filling_content_hash", "")

    _fillings = [
        e for e in order_fill_events
        if e.get("status") == "filling" and e.get("fill_attempt_id") == ref_fa_id
    ]
    if len(_fillings) != 1:
        raise RuntimeError(
            f"commit_intent {commit_intent.get('commit_id')!r} refererer "
            f"filling-event {ref_fa_id!r} for ordre {order_id} — "
            f"fant {len(_fillings)} (forventet nøyaktig 1) — fail-closed"
        )
    filling = _fillings[0]

    if ref_hash and filling.get("content_hash") != ref_hash:
        raise RuntimeError(
            f"content_hash-mismatch for filling {ref_fa_id!r} (ordre {order_id}) "
            f"— filling er manipulert — fail-closed"
        )

    _fv = filling.get("record_version")
    if not (type(_fv) is int and _fv == expected_version):
        raise RuntimeError(
            f"filling for ordre {order_id} (fa_id={ref_fa_id!r}) har "
            f"record_version={_fv!r} — krever nøyaktig int {expected_version} "
            f"— mixed-version chain er ikke tillatt — fail-closed"
        )

    return fill_ref, filling


def reconcile_settling_orders(orders: dict, strategy_name: str, state: dict) -> list:
    """Reconcile SETTLING orders using fill event WAL + commit_intent + portfolio state.

    Batch-atomic two-phase model:
    A commit_intent represents a single portfolio save covering ALL its fill_refs.
    SETTLING orders are grouped by their commit_intent. If any fill_ref in a batch
    fails validation, the entire batch is fail-closed (no partial reconciliation).

    PHASE 1 — preflight (read/validate only, no writes):
    - Group SETTLING orders by commit_id
    - Validate commit_intent version for the batch
    - Validate filling version and content_hash for every fill_ref in the batch
    - Validate existing persisted chains (resolve_fill strict) where present
    - Classify batch hash: post / pre / no_pre / neither
    - If any check fails → mark entire batch invalid

    PHASE 2 — apply results (writes only if entire batch valid):
    - Invalid batch: all orders → FAILED_RECONCILIATION, no writes
    - hash=post: reconstruct (mark_fill_persisted) where needed, then EXECUTED
    - hash=pre or no_pre: all orders → PENDING_PRICE
    - Neither: batch already marked invalid in Phase 1 → FAILED_RECONCILIATION

    No commit_intent path (versionless legacy or crash before CI was written):
    - Versionless records (no record_version) → FAILED_RECONCILIATION + manual_review
    - Any persisted marker without commit_intent → FAILED_RECONCILIATION + manual_review
    - Strict fill events only, no persisted → PENDING_PRICE (retry)

    Raises RuntimeError if the fill ledger cannot be read (fail-closed — caller must alert).
    Also raises if any manual_review cases detected (_failed_recs non-empty).
    Returns the list of reconciled orders.
    """
    from modules.fills import (  # noqa: PLC0415
        _FILL_RECORD_VERSION as _FRV,
        compute_portfolio_state_hash,
        load_fill_events,
        mark_fill_persisted,
        resolve_fill,
    )

    # Raises RuntimeError on ledger read failure — do not catch; let it propagate fail-closed
    fill_events_by_order, commit_intents = load_fill_events()
    _cid_to_ci: dict[str, dict] = {ci["commit_id"]: ci for ci in commit_intents}

    # Build order_id → commit_intent map (for this strategy; last CI per order_id wins)
    order_to_intent: dict[str, dict] = {}
    for ci in commit_intents:
        if ci.get("strategy") != strategy_name:
            continue
        for fill_ref in ci.get("fills", []):
            oid = fill_ref.get("order_id")
            if oid:
                order_to_intent[oid] = ci

    current_hash = compute_portfolio_state_hash(state)

    reconciled = []
    _failed_recs: list[str] = []

    # -----------------------------------------------------------------------
    # Group SETTLING orders: batch (has CI) vs. no-CI path
    # -----------------------------------------------------------------------
    batch_by_cid: dict[str, list[dict]] = {}  # cid → [order, ...]
    settling_no_ci: list[dict] = []

    for order in list(orders.values()):
        if order.get("status") != SETTLING:
            continue
        if order.get("strategy") != strategy_name:
            continue
        oid = order["order_id"]
        ci = order_to_intent.get(oid)
        if ci is None:
            settling_no_ci.append(order)
        else:
            batch_by_cid.setdefault(ci["commit_id"], []).append(order)

    # =========================================================================
    # GLOBAL OVERLAP PRE-CHECK — before Phase 1.
    # An "active" CI has ≥1 SETTLING order in batch_by_cid. Two active CIs that
    # share any order_id across their fill_refs form an overlapping pair. All CIs
    # in the connected component are marked invalid before any Phase 1 write.
    # Historical/superseded CIs (not in batch_by_cid) are never counted.
    # =========================================================================
    batch_preflight: dict[str, dict] = {}  # cid → result

    # 0a. Collect all order_ids referenced by each active CI's fill_refs
    _active_cid_oids: dict[str, set[str]] = {}
    for _cid in batch_by_cid:
        _active_cid_oids[_cid] = {
            _fr.get("order_id")
            for _fr in _cid_to_ci[_cid].get("fills", [])
            if _fr.get("order_id")
        }

    # 0b. Invert: order_id → set of active cids that reference it
    _oid_to_active_cids: dict[str, set[str]] = {}
    for _cid, _oids in _active_cid_oids.items():
        for _oid in _oids:
            _oid_to_active_cids.setdefault(_oid, set()).add(_cid)

    # 0c. Contested = order_id in more than one active CI
    _contested_oids: dict[str, set[str]] = {
        _oid: _cids
        for _oid, _cids in _oid_to_active_cids.items()
        if len(_cids) > 1
    }

    # 0d. Transitive closure: BFS/DFS over CI adjacency (shared contested order)
    if _contested_oids:
        _adjacency: dict[str, set[str]] = {_cid: set() for _cid in batch_by_cid}
        for _cids in _contested_oids.values():
            _cid_list = list(_cids)
            for _i, _ca in enumerate(_cid_list):
                for _cb in _cid_list[_i + 1:]:
                    _adjacency[_ca].add(_cb)
                    _adjacency[_cb].add(_ca)

        _visited_overlap: set[str] = set()
        for _start in list(_adjacency):
            if _start in _visited_overlap:
                continue
            _component: set[str] = set()
            _stack = [_start]
            while _stack:
                _cur = _stack.pop()
                if _cur in _component:
                    continue
                _component.add(_cur)
                _stack.extend(_adjacency[_cur] - _component)
            _visited_overlap.update(_component)
            if len(_component) <= 1:
                continue
            # Multi-CI component: every CI in it is fail-closed
            for _cid in _component:
                _contested_here = sorted(
                    _oid for _oid in _active_cid_oids[_cid] if _oid in _contested_oids
                )
                _other_cids = sorted(_component - {_cid})
                batch_preflight[_cid] = {
                    "valid": False,
                    "fail_reason": (
                        f"aktiv CI-overlap: commit_intent {_cid!r} deler "
                        f"order_id(s) {_contested_here!r} med {_other_cids!r} "
                        f"— overlappende aktive commit_intents — fail-closed"
                    ),
                    "hash_result": None,
                    "filling_by_order": {},
                    "needs_reconstruction": set(),
                }

    # =========================================================================
    # PHASE 1: Preflight — validate every batch completely before any write.
    # Outcome is determined here; Phase 2 only executes what Phase 1 decided.
    # CIs pre-marked by the overlap check are skipped here.
    # =========================================================================
    for cid, batch_orders in batch_by_cid.items():
        if cid in batch_preflight:  # pre-marked invalid by overlap detection
            continue
        ci = _cid_to_ci[cid]
        res: dict = {
            "valid": True,
            "fail_reason": None,
            "hash_result": None,   # "post" | "pre" | "neither"
            "filling_by_order": {},  # oid → (fill_ref, filling_event)
            "needs_reconstruction": set(),  # oids without persisted markers
            "executed_oids": set(),  # fill_refs whose order is EXECUTED with valid chain
        }

        # 1a. Commit_intent version must be exactly int 2
        _ci_rv = ci.get("record_version")
        if not (type(_ci_rv) is int and _ci_rv == _FRV):
            res["valid"] = False
            res["fail_reason"] = (
                f"commit_intent {cid!r} har record_version={_ci_rv!r} "
                f"— krever nøyaktig int {_FRV} — versjonsløs CI kan ikke autorisere EXECUTED"
            )

        # 1b. Validate ALL fill_refs in CI — not just batch_orders (SETTLING).
        # Missing orders, wrong statuses, or invalid chains all block the entire batch.
        if res["valid"]:
            for fill_ref in ci.get("fills", []):
                oid = fill_ref.get("order_id")
                if not oid:
                    res["valid"] = False
                    res["fail_reason"] = (
                        f"commit_intent {cid!r} har fill_ref uten order_id — fail-closed"
                    )
                    break
                order_events = fill_events_by_order.get(oid, [])

                # Strict filling validation (version, content_hash, exactly-one match)
                try:
                    fill_ref_val, filling = _resolve_ci_filling_strict(
                        oid, order_events, ci, _FRV
                    )
                    res["filling_by_order"][oid] = (fill_ref_val, filling)
                except RuntimeError as _ve:
                    res["valid"] = False
                    res["fail_reason"] = str(_ve)
                    break

                # Look up the order record — must exist in the orders dict
                order_rec = orders.get(oid)
                if order_rec is None:
                    res["valid"] = False
                    res["fail_reason"] = (
                        f"fill_ref for ordre {oid!r} i commit_intent {cid!r} "
                        f"mangler fra orders-dicten — fail-closed (manual_review)"
                    )
                    break

                order_status = order_rec.get("status")

                if order_status == SETTLING:
                    # Detect overlapping CIs: SETTLING order mapped to a different CI
                    _mapped_ci = order_to_intent.get(oid)
                    if _mapped_ci is not None and _mapped_ci.get("commit_id") != cid:
                        res["valid"] = False
                        res["fail_reason"] = (
                            f"ordre {oid!r} er SETTLING men er mappet til "
                            f"commit_intent {_mapped_ci.get('commit_id')!r} "
                            f"— overlappende commit_intents — fail-closed"
                        )
                        break
                    # Validate existing persisted chain if present
                    _p_markers = [
                        e for e in order_events if e.get("status") == "persisted"
                    ]
                    if _p_markers:
                        try:
                            resolve_fill(
                                oid, order_events, commit_intents,
                                expected_commit_id=cid, strict=True,
                            )
                        except RuntimeError as _re:
                            res["valid"] = False
                            res["fail_reason"] = str(_re)
                            break
                    else:
                        res["needs_reconstruction"].add(oid)

                elif order_status == EXECUTED:
                    # EXECUTED is only valid if a strict persisted chain exists for this CI
                    _p_markers = [
                        e for e in order_events if e.get("status") == "persisted"
                    ]
                    if not _p_markers:
                        res["valid"] = False
                        res["fail_reason"] = (
                            f"ordre {oid!r} er EXECUTED men mangler persisted-marker "
                            f"— inkonsistent tilstand — fail-closed"
                        )
                        break
                    try:
                        resolve_fill(
                            oid, order_events, commit_intents,
                            expected_commit_id=cid, strict=True,
                        )
                    except RuntimeError as _re:
                        res["valid"] = False
                        res["fail_reason"] = str(_re)
                        break
                    # EXECUTED with valid chain: already done — Phase 2 will not touch it
                    res["executed_oids"].add(oid)

                else:
                    # Unexpected status: PENDING_PRICE, EXPIRED, FAILED_PRICE, etc.
                    res["valid"] = False
                    res["fail_reason"] = (
                        f"fill_ref for ordre {oid!r} har status {order_status!r} "
                        f"— forventet SETTLING eller EXECUTED — fail-closed"
                    )
                    break

        # 1c. Hash classification with consistency matrix.
        # pre-hash is only safe for retry when NO fill_ref in the batch has a
        # persisted marker and NO fill_ref's order is EXECUTED.  A persisted
        # marker or EXECUTED order asserts the portfolio WAS saved to post-hash;
        # current==pre contradicts that (external revert?) → fail-closed.
        # Strict v2 CIs always have pre_hash (enforced by load_fill_events);
        # the explicit "no pre_hash" branch is defense-in-depth.
        if res["valid"]:
            _post_h = ci.get("post_portfolio_state_hash", "")
            _pre_h = ci.get("pre_portfolio_state_hash", "")
            _settling_oids = {order["order_id"] for order in batch_orders}
            if current_hash == _post_h:
                res["hash_result"] = "post"
            elif _pre_h and current_hash == _pre_h:
                _settling_with_persisted = _settling_oids - res["needs_reconstruction"]
                if _settling_with_persisted or res["executed_oids"]:
                    res["valid"] = False
                    res["fail_reason"] = (
                        f"portfolio er på pre-hash men batch {cid!r} inneholder "
                        f"persisted-records {sorted(_settling_with_persisted)!r} "
                        f"og/eller EXECUTED-ordre {sorted(res['executed_oids'])!r} "
                        f"— selvmotsigende tilstand (portfolio revertert?) "
                        f"— fail-closed (manual_review)"
                    )
                else:
                    res["hash_result"] = "pre"
            elif not _pre_h:
                res["valid"] = False
                res["fail_reason"] = (
                    f"commit_intent {cid!r} mangler pre_portfolio_state_hash "
                    f"— strict v2-skjema krever pre_hash — fail-closed"
                )
            else:
                res["valid"] = False
                res["fail_reason"] = (
                    f"portfolio hash matcher verken "
                    f"pre ({_pre_h[:8] if _pre_h else '?'}…) "
                    f"eller post ({_post_h[:8] if _post_h else '?'}…)"
                )

        batch_preflight[cid] = res

    # =========================================================================
    # PHASE 2: Apply results — writes only for valid batches.
    # Iteration order does NOT affect outcomes (decided entirely in Phase 1).
    # =========================================================================
    for cid, batch_orders in batch_by_cid.items():
        ci = _cid_to_ci[cid]
        res = batch_preflight[cid]
        commit_id = ci["commit_id"]
        _post_h = ci.get("post_portfolio_state_hash", "")
        _pre_h = ci.get("pre_portfolio_state_hash", "")

        if not res["valid"]:
            # Entire batch fail-closed: no writes, every order → FAILED_RECONCILIATION
            for order in batch_orders:
                oid = order["order_id"]
                _failed_recs.append(oid)
                updated = save_order(
                    order, status=FAILED_RECONCILIATION,
                    failure_reason=(
                        f"failed_reconciliation: batch {commit_id!r} — "
                        f"{res['fail_reason']} — fail-closed"
                    ),
                )
                orders[oid] = updated
                reconciled.append(updated)
            continue

        hash_result = res["hash_result"]
        for order in batch_orders:
            oid = order["order_id"]
            _, ci_filling = res["filling_by_order"][oid]

            if hash_result == "post":
                # Portfolio was saved. Write persisted marker only for orders that need it.
                if oid in res["needs_reconstruction"]:
                    mark_fill_persisted(
                        oid,
                        ci_filling.get("fill_attempt_id", ""),
                        ci_filling.get("content_hash", ""),
                        commit_id=commit_id,
                        post_portfolio_state_hash=_post_h,
                    )
                updated = save_order(order, status=EXECUTED)

            elif hash_result == "pre":
                # Crash before portfolio save — safe to retry (filling is strict v2)
                updated = save_order(
                    order, status=PENDING_PRICE,
                    failure_reason=(
                        "crash-recovery: portfolio i pre-fill tilstand (current==pre) "
                        "— queued for retry"
                    ),
                )

            else:  # "no_pre"
                # Old commit_intent (no pre_hash stored) — retry conservatively
                updated = save_order(
                    order, status=PENDING_PRICE,
                    failure_reason=(
                        "crash-recovery: commit_intent hash mismatch "
                        "— portfolio not saved, queued for retry"
                    ),
                )

            orders[oid] = updated
            reconciled.append(updated)

    # -----------------------------------------------------------------------
    # No-commit_intent path: versionless / orphaned persisted / strict-no-CI
    # -----------------------------------------------------------------------
    for order in settling_no_ci:
        oid = order["order_id"]
        order_fill_events = fill_events_by_order.get(oid, [])

        _has_versionless = any(
            e.get("record_version") is None
            for e in order_fill_events
            if e.get("status") in ("filling", "persisted")
        )
        _has_persisted = any(e.get("status") == "persisted" for e in order_fill_events)

        if _has_versionless or _has_persisted:
            # Versionless records or any persisted marker without commit_intent:
            # never auto-authorize EXECUTED — terminal manual_review.
            _failed_recs.append(oid)
            updated = save_order(
                order, status=FAILED_RECONCILIATION,
                failure_reason=(
                    "manual_review: versjonsløs SETTLING-ordre uten commit_intent "
                    "— aldri auto-autoriser EXECUTED"
                ),
            )
        else:
            # Strict fill events only, no persisted, no CI: crash before CI written → retry
            updated = save_order(
                order, status=PENDING_PRICE,
                failure_reason="crash-recovery: filling uten commit_intent — queued for retry",
            )

        orders[oid] = updated
        reconciled.append(updated)

    if _failed_recs:
        raise RuntimeError(
            f"Reconcile failed_reconciliation/manual_review for ordre {_failed_recs}: "
            f"hash-mismatch eller versjonsløs legacy-ordre — fail-closed (Telegram required)"
        )

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


def check_pending_price_guard(
    orders: dict,
    strategy_name: str,
    session_date: str,
    *,
    send_telegram_fn=None,
) -> None:
    """Block same-day retry if any PENDING_PRICE order has a persisted fill record.

    A persisted marker means the order already touched the portfolio at some point.
    Whether the fill chain is valid (complete) or corrupt (broken/missing CI),
    re-executing would risk a double-fill. Both cases write FAILED_RECONCILIATION to
    the order ledger, update orders[oid] in-place, then raise RuntimeError (fail-closed).

    Writing the terminal status before raising ensures the next run will not see the
    order as PENDING_PRICE, preventing repeated Telegram alerts for the same order.

    Pass send_telegram_fn=send_telegram to notify on block; omit in tests.
    """
    from modules.fills import load_fill_events, resolve_fill  # noqa: PLC0415

    fill_events, commit_intents = load_fill_events()
    for order in get_pending_for_session(orders, session_date, strategy_name):
        oid = order["order_id"]
        events = fill_events.get(oid, [])
        if not any(e.get("status") == "persisted" for e in events):
            continue  # No persisted marker — safe to retry normally

        ticker = order.get("ticker", "?")
        chain_err = None
        try:
            resolve_fill(oid, events, commit_intents, strict=True)
        except RuntimeError as _e:
            chain_err = _e

        if chain_err is None:
            failure_reason = (
                f"gyldig strict persisted fill-chain funnet under PENDING_PRICE "
                f"— portfolio allerede lagret — selvmotsigende tilstand — fail-closed"
            )
            msg = (
                f"⚠️ KRITISK: PENDING_PRICE-ordre {oid!r} ({ticker}) "
                f"har gyldig strict persisted fill-chain — stopper retry — manual_review\n"
                f"Strategi: {strategy_name}"
            )
        else:
            failure_reason = (
                f"persisted-record funnet men korrupt/ufullstendig fill-chain "
                f"— fail-closed: {chain_err}"
            )
            msg = (
                f"⚠️ KRITISK: PENDING_PRICE-ordre {oid!r} ({ticker}) "
                f"har persisted-record men korrupt/ufullstendig fill-chain "
                f"— stopper retry — manual_review\n"
                f"Strategi: {strategy_name}\n"
                f"Feil: {chain_err}"
            )

        # Write FAILED_RECONCILIATION BEFORE raising — next run won't see order as pending
        orders[oid] = save_order(order, status=FAILED_RECONCILIATION, failure_reason=failure_reason)

        if send_telegram_fn is not None:
            send_telegram_fn(msg)

        if chain_err is None:
            raise RuntimeError(
                f"PENDING_PRICE-ordre {oid!r} har gyldig strict persisted fill-chain "
                f"— aldri re-execute — fail-closed"
            )
        else:
            raise RuntimeError(
                f"PENDING_PRICE-ordre {oid!r} har persisted-record men korrupt fill-chain "
                f"— aldri re-execute — fail-closed: {chain_err}"
            ) from chain_err


def build_execution_stats(orders: dict, session_date: str, strategy_name: str) -> dict:
    """Build execution statistics for a strategy's session cohort from the persistent order ledger.

    Cohort definition: all orders where
        intended_execution_session == session_date AND strategy == strategy_name.

    Each order_id is counted exactly once — load_orders() already deduplicates by order_id
    (last record wins in the append-only JSONL), so retried orders update the same order,
    not the denominator.

    fill_rate = (executed + settling) / cohort_size.
    SETTLING counts as executed because the transition to EXECUTED completes within the same
    run, before this function is called in normal flow.

    recommendations (candidates_count) = number of signal candidates for the strategy from
    the most recent validated signal-run. Defined externally and passed via the result dict
    (key: "recommendations"). Included here for documentation only; not computed here.
    """
    cohort = [
        o for o in orders.values()
        if o.get("intended_execution_session") == session_date
        and o.get("strategy") == strategy_name
    ]

    stats: dict = {
        "cohort_size": len(cohort),
        "executed": 0,
        "settling": 0,
        "pending_price": 0,
        "failed_price": 0,
        "failed_reconciliation": 0,
        "expired": 0,
        "cancelled": 0,
        "missing_execution_price": 0,  # PENDING_PRICE with "no execution price" in failure_reason
        "buy_created": 0,
        "buy_executed": 0,
        "sell_created": 0,
        "sell_executed": 0,
        "pyramid_created": 0,
        "pyramid_executed": 0,
    }

    for o in cohort:
        status = o.get("status", "")
        action = o.get("action", "")
        filled = status in (EXECUTED, SETTLING)

        if status == EXECUTED:
            stats["executed"] += 1
        elif status == SETTLING:
            stats["settling"] += 1
        elif status == PENDING_PRICE:
            stats["pending_price"] += 1
            fr = o.get("failure_reason") or ""
            if "no execution price" in fr:
                stats["missing_execution_price"] += 1
        elif status == FAILED_PRICE:
            stats["failed_price"] += 1
        elif status == FAILED_RECONCILIATION:
            stats["failed_reconciliation"] += 1
        elif status == EXPIRED:
            stats["expired"] += 1
        elif status == CANCELLED:
            stats["cancelled"] += 1

        if action == "BUY":
            stats["buy_created"] += 1
            if filled:
                stats["buy_executed"] += 1
        elif action == "SELL":
            stats["sell_created"] += 1
            if filled:
                stats["sell_executed"] += 1
        elif action == "PYRAMID_FILL":
            stats["pyramid_created"] += 1
            if filled:
                stats["pyramid_executed"] += 1

    n = stats["cohort_size"]
    ex = stats["executed"] + stats["settling"]
    stats["fill_rate"] = (ex / n) if n > 0 else None
    bc = stats["buy_created"]
    stats["buy_fill_rate"] = (stats["buy_executed"] / bc) if bc > 0 else None
    sc = stats["sell_created"]
    stats["sell_fill_rate"] = (stats["sell_executed"] / sc) if sc > 0 else None
    pc = stats["pyramid_created"]
    stats["pyramid_fill_rate"] = (stats["pyramid_executed"] / pc) if pc > 0 else None

    return stats


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
