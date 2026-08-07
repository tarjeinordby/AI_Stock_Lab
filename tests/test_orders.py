"""
Tests for modules/orders.py — persistent order state, idempotency, and retry logic.

Key invariants:
  - make_order_id is deterministic: same inputs → same ID
  - get_or_create_order returns (existing, False) when order_id already in orders
  - get_or_create_order returns (new, True) for fresh inputs
  - Orders in TERMINAL state are never re-filled
  - expire_stale_orders transitions old PENDING_PRICE → EXPIRED and leaves current session alone
  - save_order persists a JSONL snapshot; load_orders picks up last-write-wins state
  - Same order cannot be filled twice (EXECUTED is terminal)
"""

import json
import os
import tempfile

import pandas as pd
import pytest

import modules.orders as orders_mod
from modules.orders import (
    CANCELLED,
    EXECUTED,
    EXPIRED,
    FAILED_PRICE,
    PENDING_PRICE,
    SETTLING,
    TERMINAL,
    build_order,
    expire_stale_orders,
    get_or_create_order,
    get_pending_for_session,
    load_orders,
    make_candidate_signal_id,
    make_order_id,
    make_trade_id,
    reconcile_settling_orders,
    save_order,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SESSION = "2026-08-06"
STRATEGY = "quant_baseline_v1"
TICKER = "AAPL"
SIGNAL_RUN_ID = "run-abc123"
EXEC_VERSION = "v1"


def _order(**overrides):
    kwargs = dict(
        signal_run_id=SIGNAL_RUN_ID,
        ticker=TICKER,
        strategy=STRATEGY,
        session_date=SESSION,
        action="BUY",
        target_value=5000.0,
        reason="test",
        signal_price=100.0,
        execution_version=EXEC_VERSION,
    )
    kwargs.update(overrides)
    return build_order(**kwargs)


def _orders_file(tmp_path):
    return str(tmp_path / "orders.jsonl")


def _with_file(tmp_path, monkeypatch):
    path = _orders_file(tmp_path)
    monkeypatch.setattr(orders_mod, "ORDERS_FILE", path)
    monkeypatch.setattr(orders_mod, "_ORDERS_LOCK_FILE", path + ".lock")
    return path


# ---------------------------------------------------------------------------
# make_order_id: determinism
# ---------------------------------------------------------------------------

class TestMakeOrderId:
    """New key format: (signal_id, portfolio_id, portfolio_version, ticker, session, action)."""

    def test_same_inputs_same_id(self):
        a = make_order_id("sig-abc", "port-abc", "v1", TICKER, SESSION, "BUY")
        b = make_order_id("sig-abc", "port-abc", "v1", TICKER, SESSION, "BUY")
        assert a == b

    def test_different_action_different_id(self):
        buy = make_order_id("sig-abc", "port-abc", "v1", TICKER, SESSION, "BUY")
        sell = make_order_id("sig-abc", "port-abc", "v1", TICKER, SESSION, "SELL")
        assert buy != sell

    def test_different_ticker_different_id(self):
        aapl = make_order_id("sig-abc", "port-abc", "v1", "AAPL", SESSION, "BUY")
        msft = make_order_id("sig-abc", "port-abc", "v1", "MSFT", SESSION, "BUY")
        assert aapl != msft

    def test_different_session_different_id(self):
        today = make_order_id("sig-abc", "port-abc", "v1", TICKER, "2026-08-06", "BUY")
        tomorrow = make_order_id("sig-abc", "port-abc", "v1", TICKER, "2026-08-07", "BUY")
        assert today != tomorrow

    def test_different_signal_id_different_id(self):
        a = make_order_id("sig-aaa", "port-abc", "v1", TICKER, SESSION, "BUY")
        b = make_order_id("sig-bbb", "port-abc", "v1", TICKER, SESSION, "BUY")
        assert a != b

    def test_different_portfolio_id_different_id(self):
        """Two portfolios consuming same signal → different order_ids."""
        a = make_order_id("sig-same", "port-A", "v1", TICKER, SESSION, "BUY")
        b = make_order_id("sig-same", "port-B", "v1", TICKER, SESSION, "BUY")
        assert a != b

    def test_id_starts_with_ord_prefix(self):
        oid = make_order_id("sig-abc", "port-abc", "v1", TICKER, SESSION, "BUY")
        assert oid.startswith("ord-")

    def test_id_has_fixed_length(self):
        # "ord-" + 12 hex chars = 16 chars total
        oid = make_order_id("sig-abc", "port-abc", "v1", TICKER, SESSION, "BUY")
        assert len(oid) == 16

    def test_signal_id_none_treated_as_empty(self):
        """signal_id=None (safety actions) maps to same key as signal_id=''."""
        a = make_order_id(None, "port-abc", "v1", TICKER, SESSION, "SELL")
        b = make_order_id("", "port-abc", "v1", TICKER, SESSION, "SELL")
        assert a == b


# ---------------------------------------------------------------------------
# make_trade_id
# ---------------------------------------------------------------------------

class TestMakeTradeId:
    def test_starts_with_trd_prefix(self):
        tid = make_trade_id("ord-abc123456789", "2026-08-06T10:00:00Z")
        assert tid.startswith("trd-")

    def test_different_timestamps_different_ids(self):
        a = make_trade_id("ord-abc123456789", "2026-08-06T10:00:00Z")
        b = make_trade_id("ord-abc123456789", "2026-08-06T10:00:01Z")
        assert a != b


# ---------------------------------------------------------------------------
# build_order: initial state
# ---------------------------------------------------------------------------

class TestBuildOrder:
    def test_status_is_pending_price(self):
        o = _order()
        assert o["status"] == PENDING_PRICE

    def test_order_id_matches_make_order_id(self):
        # _order() uses default signal_id (=signal_run_id), portfolio_id="", portfolio_version=""
        o = _order()
        expected = make_order_id(SIGNAL_RUN_ID, "", "", TICKER, SESSION, "BUY")
        assert o["order_id"] == expected

    def test_trade_id_is_none(self):
        o = _order()
        assert o["trade_id"] is None

    def test_intended_session_matches_input(self):
        o = _order(session_date=SESSION)
        assert o["intended_execution_session"] == SESSION


# ---------------------------------------------------------------------------
# get_or_create_order: idempotency
# ---------------------------------------------------------------------------

class TestGetOrCreateOrder:
    def _call(self, existing_orders=None, **overrides):
        kwargs = dict(
            orders=existing_orders or {},
            signal_run_id=SIGNAL_RUN_ID,
            ticker=TICKER,
            strategy=STRATEGY,
            session_date=SESSION,
            action="BUY",
            target_value=5000.0,
            reason="test",
            signal_price=100.0,
            execution_version=EXEC_VERSION,
        )
        kwargs.update(overrides)
        return get_or_create_order(**kwargs)

    def test_creates_new_order_when_absent(self):
        order, is_new = self._call()
        assert is_new is True
        assert order["status"] == PENDING_PRICE

    def test_returns_existing_order_when_present(self):
        existing = _order()
        oid = existing["order_id"]
        order, is_new = self._call(existing_orders={oid: existing})
        assert is_new is False
        assert order["order_id"] == oid

    def test_same_inputs_return_same_order_id(self):
        o1, _ = self._call()
        o2, _ = self._call()
        assert o1["order_id"] == o2["order_id"]

    def test_existing_executed_order_returned_unchanged(self):
        executed = _order()
        executed["status"] = EXECUTED
        oid = executed["order_id"]
        order, is_new = self._call(existing_orders={oid: executed})
        assert is_new is False
        assert order["status"] == EXECUTED

    def test_existing_expired_order_returned_not_recreated(self):
        expired = _order()
        expired["status"] = EXPIRED
        oid = expired["order_id"]
        order, is_new = self._call(existing_orders={oid: expired})
        assert is_new is False
        assert order["status"] == EXPIRED


# ---------------------------------------------------------------------------
# save_order / load_orders: JSONL persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_then_load_roundtrip(self, tmp_path, monkeypatch):
        _with_file(tmp_path, monkeypatch)
        o = _order()
        save_order(o)
        loaded = load_orders()
        assert o["order_id"] in loaded

    def test_last_write_wins(self, tmp_path, monkeypatch):
        _with_file(tmp_path, monkeypatch)
        o = _order()
        save_order(o)
        updated = save_order(o, status=EXECUTED, trade_id="trd-xyz")
        loaded = load_orders()
        assert loaded[o["order_id"]]["status"] == EXECUTED
        assert loaded[o["order_id"]]["trade_id"] == "trd-xyz"

    def test_load_missing_file_returns_empty(self, tmp_path, monkeypatch):
        _with_file(tmp_path, monkeypatch)
        loaded = load_orders()
        assert loaded == {}

    def test_save_order_returns_updated_dict(self, tmp_path, monkeypatch):
        _with_file(tmp_path, monkeypatch)
        o = _order()
        result = save_order(o, status=EXECUTED, trade_id="trd-abc")
        assert result["status"] == EXECUTED
        assert result["trade_id"] == "trd-abc"

    def test_each_save_appends_new_line(self, tmp_path, monkeypatch):
        path = _with_file(tmp_path, monkeypatch)
        o = _order()
        save_order(o)
        save_order(o, status=EXECUTED)
        lines = [l for l in open(path).readlines() if l.strip()]
        assert len(lines) == 2

    def test_multiple_orders_persist_independently(self, tmp_path, monkeypatch):
        _with_file(tmp_path, monkeypatch)
        o1 = _order(ticker="AAPL")
        o2 = _order(ticker="MSFT")
        save_order(o1)
        save_order(o2)
        loaded = load_orders()
        assert len(loaded) == 2


# ---------------------------------------------------------------------------
# TERMINAL state prevents re-fill
# ---------------------------------------------------------------------------

class TestTerminalStates:
    def test_executed_is_terminal(self):
        assert EXECUTED in TERMINAL

    def test_expired_is_terminal(self):
        assert EXPIRED in TERMINAL

    def test_failed_price_is_terminal(self):
        assert FAILED_PRICE in TERMINAL

    def test_cancelled_is_terminal(self):
        assert CANCELLED in TERMINAL

    def test_pending_price_is_not_terminal(self):
        assert PENDING_PRICE not in TERMINAL

    def test_executed_order_returned_by_get_or_create_as_not_new(self):
        executed = _order()
        executed["status"] = EXECUTED
        oid = executed["order_id"]
        order, is_new = get_or_create_order(
            orders={oid: executed},
            signal_run_id=SIGNAL_RUN_ID,
            ticker=TICKER,
            strategy=STRATEGY,
            session_date=SESSION,
            action="BUY",
            target_value=5000.0,
            reason="test",
            signal_price=100.0,
            execution_version=EXEC_VERSION,
        )
        assert is_new is False
        assert order["status"] == EXECUTED


# ---------------------------------------------------------------------------
# get_pending_for_session
# ---------------------------------------------------------------------------

class TestGetPendingForSession:
    def _make_orders(self):
        o1 = _order(ticker="AAPL", session_date="2026-08-06")
        o2 = _order(ticker="MSFT", session_date="2026-08-06")
        o3 = _order(ticker="NVDA", session_date="2026-08-07")
        o4 = _order(ticker="TSLA", session_date="2026-08-06")
        o4["status"] = EXECUTED
        return {o["order_id"]: o for o in [o1, o2, o3, o4]}

    def test_returns_only_pending_for_session(self):
        all_orders = self._make_orders()
        pending = get_pending_for_session(all_orders, "2026-08-06")
        assert all(o["status"] == PENDING_PRICE for o in pending)
        assert all(o["intended_execution_session"] == "2026-08-06" for o in pending)
        assert len(pending) == 2  # AAPL and MSFT (TSLA is EXECUTED)

    def test_different_session_not_returned(self):
        all_orders = self._make_orders()
        pending = get_pending_for_session(all_orders, "2026-08-07")
        tickers = [o["ticker"] for o in pending]
        assert "AAPL" not in tickers
        assert "NVDA" in tickers

    def test_strategy_filter_works(self):
        o1 = _order(ticker="AAPL", strategy="quant_baseline_v1")
        o2 = _order(ticker="MSFT", strategy="momentum_v2")
        orders = {o["order_id"]: o for o in [o1, o2]}
        pending = get_pending_for_session(orders, SESSION, strategy="quant_baseline_v1")
        assert len(pending) == 1
        assert pending[0]["ticker"] == "AAPL"

    def test_empty_when_no_pending(self):
        o = _order()
        o["status"] = EXECUTED
        orders = {o["order_id"]: o}
        assert get_pending_for_session(orders, SESSION) == []


# ---------------------------------------------------------------------------
# expire_stale_orders
# ---------------------------------------------------------------------------

class TestExpireStaleOrders:
    # Use a fixed "now" before NYSE close (20:00 UTC = 16:00 ET) so tests are
    # deterministic regardless of when they are run.
    _BEFORE_CLOSE = pd.Timestamp("2026-08-06T18:00:00", tz="UTC")  # 14:00 ET — market open
    _AFTER_CLOSE  = pd.Timestamp("2026-08-06T21:00:00", tz="UTC")  # 17:00 ET — after close

    def test_old_pending_becomes_expired(self, tmp_path, monkeypatch):
        _with_file(tmp_path, monkeypatch)
        old = _order(session_date="2026-08-05")
        orders = {old["order_id"]: old}
        expired = expire_stale_orders(orders, "2026-08-06", _now=self._BEFORE_CLOSE)
        assert len(expired) == 1
        assert orders[old["order_id"]]["status"] == EXPIRED

    def test_current_session_not_expired_while_open(self, tmp_path, monkeypatch):
        """Current session is not expired while the exchange is still open."""
        _with_file(tmp_path, monkeypatch)
        current = _order(session_date="2026-08-06")
        orders = {current["order_id"]: current}
        expired = expire_stale_orders(orders, "2026-08-06", _now=self._BEFORE_CLOSE)
        assert len(expired) == 0
        assert orders[current["order_id"]]["status"] == PENDING_PRICE

    def test_current_session_expired_after_close(self, tmp_path, monkeypatch):
        """Current session is expired once the exchange has closed (handles early close)."""
        _with_file(tmp_path, monkeypatch)
        current = _order(session_date="2026-08-06")
        orders = {current["order_id"]: current}
        expired = expire_stale_orders(orders, "2026-08-06", _now=self._AFTER_CLOSE)
        assert len(expired) == 1
        assert orders[current["order_id"]]["status"] == EXPIRED
        assert "2026-08-06" in expired[0]["failure_reason"]

    def test_already_executed_not_touched(self, tmp_path, monkeypatch):
        _with_file(tmp_path, monkeypatch)
        old_exec = _order(session_date="2026-08-05")
        old_exec["status"] = EXECUTED
        orders = {old_exec["order_id"]: old_exec}
        expired = expire_stale_orders(orders, "2026-08-06", _now=self._BEFORE_CLOSE)
        assert len(expired) == 0
        assert orders[old_exec["order_id"]]["status"] == EXECUTED

    def test_expired_orders_persisted_to_file(self, tmp_path, monkeypatch):
        path = _with_file(tmp_path, monkeypatch)
        old = _order(session_date="2026-08-04")
        save_order(old)
        orders = load_orders()
        expire_stale_orders(orders, "2026-08-06", _now=self._BEFORE_CLOSE)
        reloaded = load_orders()
        assert reloaded[old["order_id"]]["status"] == EXPIRED

    def test_failure_reason_set_on_expiry(self, tmp_path, monkeypatch):
        _with_file(tmp_path, monkeypatch)
        old = _order(session_date="2026-08-05")
        orders = {old["order_id"]: old}
        expired = expire_stale_orders(orders, "2026-08-06", _now=self._BEFORE_CLOSE)
        assert expired[0]["failure_reason"] is not None
        assert "2026-08-05" in expired[0]["failure_reason"]

    def test_multiple_stale_orders_all_expired(self, tmp_path, monkeypatch):
        _with_file(tmp_path, monkeypatch)
        o1 = _order(ticker="AAPL", session_date="2026-08-04")
        o2 = _order(ticker="MSFT", session_date="2026-08-05")
        o3 = _order(ticker="NVDA", session_date="2026-08-06")  # current — session still open
        orders = {o["order_id"]: o for o in [o1, o2, o3]}
        expired = expire_stale_orders(orders, "2026-08-06", _now=self._BEFORE_CLOSE)
        assert len(expired) == 2
        assert orders[o3["order_id"]]["status"] == PENDING_PRICE


# ---------------------------------------------------------------------------
# make_trade_id: determinism (Point 3)
# ---------------------------------------------------------------------------

class TestMakeTradeIdDeterminism:
    def test_deterministic_without_timestamp(self):
        """make_trade_id without timestamp is deterministic — same order → same trade."""
        a = make_trade_id("ord-abc123456789")
        b = make_trade_id("ord-abc123456789")
        assert a == b

    def test_no_timestamp_starts_with_trd(self):
        assert make_trade_id("ord-abc123456789").startswith("trd-")

    def test_different_orders_different_ids_without_timestamp(self):
        a = make_trade_id("ord-aaa111222333")
        b = make_trade_id("ord-bbb444555666")
        assert a != b

    def test_no_timestamp_differs_from_with_timestamp(self):
        """Deterministic (no timestamp) and timestamped forms produce different IDs."""
        a = make_trade_id("ord-abc123456789")
        b = make_trade_id("ord-abc123456789", "2026-08-06T10:00:00Z")
        assert a != b


# ---------------------------------------------------------------------------
# Point 1: Pyramid fill idempotency — behavioral tests
# ---------------------------------------------------------------------------

class TestPyramidFillIdempotency:
    """Pyramid idempotency: order ledger checked BEFORE portfolio mutation."""

    def test_second_get_or_create_returns_existing_order(self, tmp_path, monkeypatch):
        """Getting or creating the same pyramid order twice returns the same order (is_new=False)."""
        _with_file(tmp_path, monkeypatch)
        orders = {}
        kwargs = dict(
            orders=orders, signal_run_id="run-001", ticker="AAPL",
            strategy=STRATEGY, session_date=SESSION, action="PYRAMID_FILL",
            target_value=2000.0, reason="pyramid", signal_price=155.0,
            execution_version=EXEC_VERSION,
        )
        order1, is_new1 = get_or_create_order(**kwargs)
        assert is_new1
        saved = save_order(order1)
        orders[order1["order_id"]] = saved

        order2, is_new2 = get_or_create_order(**kwargs)
        assert not is_new2
        assert order2["order_id"] == order1["order_id"]

    def test_executed_pyramid_order_blocks_second_fill(self, tmp_path, monkeypatch):
        """When pyramid order is EXECUTED, get_or_create returns it as terminal — no second fill."""
        _with_file(tmp_path, monkeypatch)
        orders = {}
        order, _ = get_or_create_order(
            orders=orders, signal_run_id="run-001", ticker="AAPL",
            strategy=STRATEGY, session_date=SESSION, action="PYRAMID_FILL",
            target_value=2000.0, reason="test", signal_price=155.0,
            execution_version=EXEC_VERSION,
        )
        s1 = save_order(order)
        s2 = save_order(s1, status="settling", trade_id="trd-pyr")
        s3 = save_order(s2, status=EXECUTED)
        orders[order["order_id"]] = s3

        # Second run: same inputs — get_or_create must return EXECUTED, is_new=False
        order2, is_new2 = get_or_create_order(
            orders=orders, signal_run_id="run-001", ticker="AAPL",
            strategy=STRATEGY, session_date=SESSION, action="PYRAMID_FILL",
            target_value=2000.0, reason="test", signal_price=155.0,
            execution_version=EXEC_VERSION,
        )
        assert not is_new2
        assert order2["status"] == EXECUTED
        assert order2["status"] in TERMINAL  # callers must skip fill for TERMINAL orders

    def test_settling_pyramid_order_blocks_second_fill(self, tmp_path, monkeypatch):
        """SETTLING pyramid order (in-flight fill) must block a second fill attempt."""
        from modules.orders import SETTLING
        _with_file(tmp_path, monkeypatch)
        orders = {}
        order, _ = get_or_create_order(
            orders=orders, signal_run_id="run-001", ticker="AAPL",
            strategy=STRATEGY, session_date=SESSION, action="PYRAMID_FILL",
            target_value=2000.0, reason="test", signal_price=155.0,
            execution_version=EXEC_VERSION,
        )
        settling = save_order(order, status=SETTLING, trade_id="trd-pyr")
        orders[order["order_id"]] = settling

        order2, is_new2 = get_or_create_order(
            orders=orders, signal_run_id="run-001", ticker="AAPL",
            strategy=STRATEGY, session_date=SESSION, action="PYRAMID_FILL",
            target_value=2000.0, reason="test", signal_price=155.0,
            execution_version=EXEC_VERSION,
        )
        assert not is_new2
        assert order2["status"] == SETTLING
        # Callers must check: status in TERMINAL or status == SETTLING → skip fill
        from modules.orders import SETTLING as S
        assert order2["status"] == S

    def test_pyramid_order_deterministic_id(self):
        """PYRAMID_FILL and BUY for same ticker/session get different order IDs."""
        buy_id = make_order_id("sig-abc", "port-abc", "v1", TICKER, SESSION, "BUY")
        pyr_id = make_order_id("sig-abc", "port-abc", "v1", TICKER, SESSION, "PYRAMID_FILL")
        assert buy_id != pyr_id

    def test_pyramid_order_same_inputs_same_id(self):
        """Same pyramid fill inputs always produce the same order ID."""
        a = make_order_id("sig-pyr", "port-abc", "v1", TICKER, SESSION, "PYRAMID_FILL")
        b = make_order_id("sig-pyr", "port-abc", "v1", TICKER, SESSION, "PYRAMID_FILL")
        assert a == b


# ---------------------------------------------------------------------------
# Point 3: Crash recovery — portfolio-reconciled (not blind FAILED_PRICE)
# ---------------------------------------------------------------------------

class TestCrashReconciliation:
    """_reconcile_settling_orders uses portfolio state to classify SETTLING orders."""

    def _make_orders_with_settling(self, tmp_path, monkeypatch, action="BUY", ticker="AAPL"):
        _with_file(tmp_path, monkeypatch)
        orders = {}
        order, _ = get_or_create_order(
            orders=orders, signal_run_id="run-001", ticker=ticker,
            strategy=STRATEGY, session_date=SESSION, action=action,
            target_value=5000.0, reason="test", signal_price=100.0,
            execution_version=EXEC_VERSION,
        )
        s = save_order(order)
        s = save_order(s, status="settling", trade_id="trd-crash")
        orders[order["order_id"]] = s
        return orders, order["order_id"]

    def test_reconcile_buy_with_position_marks_executed(self, tmp_path, monkeypatch):
        """SETTLING BUY + ticker in portfolio → crash after save → EXECUTED."""
        orders, oid = self._make_orders_with_settling(tmp_path, monkeypatch, "BUY", "AAPL")
        state = {"positions": {"AAPL": {"shares": 10}}, "cash": 5000.0}
        reconciled = reconcile_settling_orders(orders, STRATEGY, state)
        assert len(reconciled) == 1
        assert orders[oid]["status"] == EXECUTED

    def test_reconcile_buy_without_position_marks_pending_price(self, tmp_path, monkeypatch):
        """SETTLING BUY + ticker NOT in portfolio → crash before save → PENDING_PRICE retry."""
        orders, oid = self._make_orders_with_settling(tmp_path, monkeypatch, "BUY", "AAPL")
        state = {"positions": {}, "cash": 10000.0}
        reconciled = reconcile_settling_orders(orders, STRATEGY, state)
        assert len(reconciled) == 1
        assert orders[oid]["status"] == PENDING_PRICE
        assert "crash-recovery" in orders[oid]["failure_reason"]

    def test_reconcile_sell_without_position_marks_executed(self, tmp_path, monkeypatch):
        """SETTLING SELL + ticker NOT in portfolio → crash after save → EXECUTED."""
        orders, oid = self._make_orders_with_settling(tmp_path, monkeypatch, "SELL", "AAPL")
        state = {"positions": {}, "cash": 15000.0}
        reconciled = reconcile_settling_orders(orders, STRATEGY, state)
        assert orders[oid]["status"] == EXECUTED

    def test_reconcile_sell_with_position_marks_pending_price(self, tmp_path, monkeypatch):
        """SETTLING SELL + ticker still in portfolio → crash before save → PENDING_PRICE retry."""
        orders, oid = self._make_orders_with_settling(tmp_path, monkeypatch, "SELL", "AAPL")
        state = {"positions": {"AAPL": {"shares": 10}}, "cash": 10000.0}
        reconciled = reconcile_settling_orders(orders, STRATEGY, state)
        assert orders[oid]["status"] == PENDING_PRICE

    def test_reconcile_pyramid_fill_not_partial_marks_executed(self, tmp_path, monkeypatch):
        """SETTLING PYRAMID_FILL + is_partial=False → fill succeeded → EXECUTED."""
        orders, oid = self._make_orders_with_settling(tmp_path, monkeypatch, "PYRAMID_FILL", "AAPL")
        state = {"positions": {"AAPL": {"shares": 15, "is_partial": False, "pyramid_remaining_value": 0.0}}}
        reconciled = reconcile_settling_orders(orders, STRATEGY, state)
        assert orders[oid]["status"] == EXECUTED

    def test_reconcile_only_targets_matching_strategy(self, tmp_path, monkeypatch):
        """Reconciliation does not touch SETTLING orders for other strategies."""
        _with_file(tmp_path, monkeypatch)
        orders = {}
        o = build_order("run-001", "AAPL", "other_strategy", SESSION, "BUY",
                        5000.0, "test", 100.0, EXEC_VERSION)
        s = save_order(o)
        s = save_order(s, status="settling", trade_id="trd-x")
        orders[o["order_id"]] = s

        state = {"positions": {}, "cash": 10000.0}
        reconciled = reconcile_settling_orders(orders, STRATEGY, state)
        assert len(reconciled) == 0  # different strategy — not touched
        assert orders[o["order_id"]]["status"] == "settling"


# ---------------------------------------------------------------------------
# Point 4: load_orders read lock + fail-closed mid-file corruption
# ---------------------------------------------------------------------------

class TestLoadOrdersHardening:
    def test_mid_file_corruption_raises(self, tmp_path, monkeypatch):
        """Corrupt JSON in the middle of the file raises RuntimeError (fail-closed)."""
        path = _with_file(tmp_path, monkeypatch)
        o1 = _order(ticker="AAPL")
        o2 = _order(ticker="MSFT")
        save_order(o1)
        # Inject corrupt line between two valid records
        with open(path, "a") as f:
            f.write('{"order_id": "bad-json-here\n')  # broken
        save_order(o2)
        with pytest.raises(RuntimeError, match="korrupt"):
            load_orders()

    def test_corrupt_last_line_skipped_gracefully(self, tmp_path, monkeypatch):
        """A truncated last line (crash during write) is skipped without error."""
        path = _with_file(tmp_path, monkeypatch)
        o = _order()
        save_order(o)
        with open(path, "a") as f:
            f.write('{"order_id": "truncated\n')  # crash-truncated final line
        loaded = load_orders()
        assert o["order_id"] in loaded
        assert len(loaded) == 1

    def test_lock_file_created_on_load(self, tmp_path, monkeypatch):
        """load_orders creates the lock file (needed for subsequent exclusive writes)."""
        path = _with_file(tmp_path, monkeypatch)
        save_order(_order())
        load_orders()
        assert os.path.exists(path + ".lock")

    def test_terminal_status_preserved_before_corrupt_last_line(self, tmp_path, monkeypatch):
        """A corrupt last line must not cause loss of a previously-recorded EXECUTED status."""
        path = _with_file(tmp_path, monkeypatch)
        o = _order()
        save_order(o)
        save_order(o, status=EXECUTED, trade_id="trd-abc")
        with open(path, "a") as f:
            f.write('{"order_id": "corrupt\n')
        loaded = load_orders()
        assert loaded[o["order_id"]]["status"] == EXECUTED


# ---------------------------------------------------------------------------
# Points 2 & 6: signal_id separation and portfolio_version
# ---------------------------------------------------------------------------

class TestTraceabilityFields:
    def test_build_order_signal_id_defaults_to_signal_run_id(self):
        """Without explicit signal_id, order.signal_id falls back to signal_run_id."""
        o = build_order("run-abc", "AAPL", STRATEGY, SESSION, "BUY",
                        5000.0, "test", 100.0, EXEC_VERSION)
        assert o["signal_id"] == "run-abc"
        assert o["signal_run_id"] == "run-abc"

    def test_build_order_explicit_signal_id_differs_from_run_id(self):
        """Explicit signal_id (e.g. content hash) is stored separately from signal_run_id."""
        from modules.orders import _SIGNAL_ID_UNSET
        o = build_order("run-abc", "AAPL", STRATEGY, SESSION, "BUY",
                        5000.0, "test", 100.0, EXEC_VERSION, signal_id="hash-xyz")
        assert o["signal_id"] == "hash-xyz"
        assert o["signal_run_id"] == "run-abc"
        assert o["signal_id"] != o["signal_run_id"]

    def test_build_order_explicit_none_signal_id_for_safety_actions(self):
        """Safety-action orders can have signal_id=None (no associated signal)."""
        o = build_order("safety-2026-08-06", "AAPL", STRATEGY, SESSION, "SELL",
                        0.0, "stop-loss", 100.0, EXEC_VERSION, signal_id=None)
        assert o["signal_id"] is None
        assert o["signal_run_id"] == "safety-2026-08-06"

    def test_get_or_create_order_passes_signal_id(self, tmp_path, monkeypatch):
        """get_or_create_order forwards signal_id to build_order."""
        _with_file(tmp_path, monkeypatch)
        orders = {}
        order, is_new = get_or_create_order(
            orders=orders, signal_run_id="run-001", ticker="AAPL",
            strategy=STRATEGY, session_date=SESSION, action="BUY",
            target_value=5000.0, reason="test", signal_price=100.0,
            execution_version=EXEC_VERSION, signal_id="content-hash-abc",
        )
        assert is_new
        assert order["signal_id"] == "content-hash-abc"
        assert order["signal_run_id"] == "run-001"


# ---------------------------------------------------------------------------
# Point 2: Per-candidate signal_id and new order_id key scheme
# ---------------------------------------------------------------------------

class TestOrderIdWithSignalId:
    """make_order_id now uses signal_id + portfolio_id + portfolio_version (not strategy)."""

    def test_two_portfolios_same_signal_different_order_id(self):
        """Two portfolios consuming same signal for same ticker → different order_ids."""
        sig_id = "sig-abc123abc123"
        id_a = make_order_id(sig_id, "port-A", "v1", TICKER, SESSION, "BUY")
        id_b = make_order_id(sig_id, "port-B", "v1", TICKER, SESSION, "BUY")
        assert id_a != id_b

    def test_two_signals_same_portfolio_different_order_id(self):
        """Two different signals in the same portfolio → different order_ids."""
        id_a = make_order_id("sig-run-001", "port-X", "v1", TICKER, SESSION, "BUY")
        id_b = make_order_id("sig-run-002", "port-X", "v1", TICKER, SESSION, "BUY")
        assert id_a != id_b

    def test_rerun_same_portfolio_same_order_id(self):
        """Rerunning with same signal_id and portfolio → same order_id (idempotent)."""
        sig_id = "sig-stable123456"
        a = make_order_id(sig_id, "port-Y", "v1", TICKER, SESSION, "BUY")
        b = make_order_id(sig_id, "port-Y", "v1", TICKER, SESSION, "BUY")
        assert a == b

    def test_make_candidate_signal_id_stable(self):
        """Same (signal_run_id, strategy, ticker, action) → same signal_id."""
        a = make_candidate_signal_id("run-001", STRATEGY, TICKER, "BUY")
        b = make_candidate_signal_id("run-001", STRATEGY, TICKER, "BUY")
        assert a == b

    def test_make_candidate_signal_id_different_ticker(self):
        a = make_candidate_signal_id("run-001", STRATEGY, "AAPL", "BUY")
        b = make_candidate_signal_id("run-001", STRATEGY, "MSFT", "BUY")
        assert a != b

    def test_make_candidate_signal_id_different_strategy(self):
        a = make_candidate_signal_id("run-001", "strat_A", TICKER, "BUY")
        b = make_candidate_signal_id("run-001", "strat_B", TICKER, "BUY")
        assert a != b

    def test_make_candidate_signal_id_different_action(self):
        a = make_candidate_signal_id("run-001", STRATEGY, TICKER, "BUY")
        b = make_candidate_signal_id("run-001", STRATEGY, TICKER, "SELL")
        assert a != b

    def test_make_candidate_signal_id_starts_with_sig_prefix(self):
        sid = make_candidate_signal_id("run-001", STRATEGY, TICKER, "BUY")
        assert sid.startswith("sig-")

    def test_make_candidate_signal_id_has_fixed_length(self):
        # "sig-" + 12 hex chars = 16 chars total
        sid = make_candidate_signal_id("run-001", STRATEGY, TICKER, "BUY")
        assert len(sid) == 16

    def test_legacy_order_found_prevents_duplicate(self, tmp_path, monkeypatch):
        """Legacy v1 order_id (old make_order_id format) found via fallback → is_new=False."""
        from modules.orders import _make_legacy_order_id
        _with_file(tmp_path, monkeypatch)
        # Simulate an order created by OLD code: order_id uses old hash format
        # (signal_run_id | ticker | strategy | session | action)
        legacy_id = _make_legacy_order_id(SIGNAL_RUN_ID, TICKER, STRATEGY, SESSION, "BUY")
        old_order = {
            "order_id": legacy_id,
            "signal_run_id": SIGNAL_RUN_ID,
            "signal_id": SIGNAL_RUN_ID,
            "portfolio_id": "",
            "portfolio_version": "",
            "ticker": TICKER,
            "strategy": STRATEGY,
            "intended_execution_session": SESSION,
            "action": "BUY",
            "target_value": 5000.0,
            "pyramid_remaining": 0.0,
            "reason": "test",
            "signal_price": 100.0,
            "execution_version": EXEC_VERSION,
            "status": PENDING_PRICE,
            "created_at": "2026-08-06T00:00:00Z",
            "updated_at": "2026-08-06T00:00:00Z",
            "attempted_at": None,
            "failure_reason": None,
            "trade_id": None,
        }
        save_order(old_order)
        orders = {legacy_id: old_order}

        # New-style call with per-candidate signal_id and portfolio_id
        new_sig_id = make_candidate_signal_id(SIGNAL_RUN_ID, STRATEGY, TICKER, "BUY")
        order, is_new = get_or_create_order(
            orders=orders, signal_run_id=SIGNAL_RUN_ID, ticker=TICKER,
            strategy=STRATEGY, session_date=SESSION, action="BUY",
            target_value=5000.0, reason="test", signal_price=100.0,
            execution_version=EXEC_VERSION, signal_id=new_sig_id,
            portfolio_id="port-abc", portfolio_version="v1",
        )
        assert not is_new  # legacy order found — no duplicate created
        assert order["order_id"] == legacy_id


# ---------------------------------------------------------------------------
# Point 3a: Fill event WAL (modules/fills.py)
# ---------------------------------------------------------------------------

class TestFillEventWAL:
    """Fill events written before portfolio save; marked persisted after."""

    def _patch_fills(self, tmp_path, monkeypatch):
        import modules.fills as fills_mod
        fills_path = str(tmp_path / "fills.jsonl")
        lock_path = str(tmp_path / "fills.jsonl.lock")
        monkeypatch.setattr(fills_mod, "FILLS_FILE", __import__("pathlib").Path(fills_path))
        monkeypatch.setattr(fills_mod, "_FILLS_LOCK_FILE", __import__("pathlib").Path(lock_path))
        return fills_path

    def test_write_fill_event_creates_file(self, tmp_path, monkeypatch):
        from modules.fills import write_fill_event
        self._patch_fills(tmp_path, monkeypatch)
        rec = write_fill_event(
            order_id="ord-abc123456789", trade_id="trd-x", signal_id="sig-y",
            signal_run_id="run-001", portfolio_id="port-a", strategy=STRATEGY,
            ticker=TICKER, action="BUY", session_date=SESSION,
            shares=10.0, price=150.0, value=1500.0, cost=0.0, reason="test",
            execution_version="v1", cash_before=10000.0, cash_after=8500.0,
        )
        assert rec["status"] == "filling"
        assert rec["fill_id"].startswith("fill-")

    def test_is_fill_persisted_false_before_mark(self, tmp_path, monkeypatch):
        from modules.fills import write_fill_event, is_fill_persisted
        self._patch_fills(tmp_path, monkeypatch)
        oid = "ord-abc123456789"
        write_fill_event(
            order_id=oid, trade_id="trd-x", signal_id=None, signal_run_id=None,
            portfolio_id="", strategy=STRATEGY, ticker=TICKER, action="BUY",
            session_date=SESSION, shares=1.0, price=100.0, value=100.0, cost=0.0,
            reason="t", execution_version="v1", cash_before=1000.0, cash_after=900.0,
        )
        assert not is_fill_persisted(oid)

    def test_is_fill_persisted_true_after_mark(self, tmp_path, monkeypatch):
        from modules.fills import write_fill_event, mark_fill_persisted, is_fill_persisted
        self._patch_fills(tmp_path, monkeypatch)
        oid = "ord-abc123456789"
        write_fill_event(
            order_id=oid, trade_id="trd-x", signal_id=None, signal_run_id=None,
            portfolio_id="", strategy=STRATEGY, ticker=TICKER, action="BUY",
            session_date=SESSION, shares=1.0, price=100.0, value=100.0, cost=0.0,
            reason="t", execution_version="v1", cash_before=1000.0, cash_after=900.0,
        )
        mark_fill_persisted(oid)
        assert is_fill_persisted(oid)

    def test_make_fill_id_deterministic(self):
        from modules.fills import make_fill_id
        a = make_fill_id("ord-abc123456789")
        b = make_fill_id("ord-abc123456789")
        assert a == b
        assert a.startswith("fill-")

    def test_reconcile_uses_fill_event_persisted(self, tmp_path, monkeypatch):
        """SETTLING + fill_event 'persisted' → EXECUTED regardless of portfolio state."""
        import modules.fills as fills_mod
        self._patch_fills(tmp_path, monkeypatch)
        _with_file(tmp_path, monkeypatch)

        from modules.fills import write_fill_event, mark_fill_persisted
        orders = {}
        order, _ = get_or_create_order(
            orders=orders, signal_run_id="run-001", ticker=TICKER,
            strategy=STRATEGY, session_date=SESSION, action="BUY",
            target_value=5000.0, reason="test", signal_price=100.0,
            execution_version=EXEC_VERSION,
        )
        s = save_order(order)
        s = save_order(s, status="settling", trade_id="trd-crash")
        orders[order["order_id"]] = s

        write_fill_event(
            order_id=order["order_id"], trade_id="trd-crash", signal_id=None,
            signal_run_id="run-001", portfolio_id="", strategy=STRATEGY,
            ticker=TICKER, action="BUY", session_date=SESSION,
            shares=5.0, price=100.0, value=500.0, cost=0.0, reason="test",
            execution_version=EXEC_VERSION, cash_before=10000.0, cash_after=9500.0,
        )
        mark_fill_persisted(order["order_id"])

        # Portfolio does NOT have position (to test fill WAL takes precedence)
        state = {"positions": {}, "cash": 10000.0}
        reconciled = reconcile_settling_orders(orders, STRATEGY, state)
        assert len(reconciled) == 1
        assert orders[order["order_id"]]["status"] == EXECUTED

    def test_reconcile_crash_before_portfolio_gives_pending_retry(self, tmp_path, monkeypatch):
        """SETTLING + fill_event 'filling' (not persisted) + no position → PENDING_PRICE."""
        self._patch_fills(tmp_path, monkeypatch)
        _with_file(tmp_path, monkeypatch)

        from modules.fills import write_fill_event
        orders = {}
        order, _ = get_or_create_order(
            orders=orders, signal_run_id="run-001", ticker=TICKER,
            strategy=STRATEGY, session_date=SESSION, action="BUY",
            target_value=5000.0, reason="test", signal_price=100.0,
            execution_version=EXEC_VERSION,
        )
        s = save_order(order)
        s = save_order(s, status="settling", trade_id="trd-x")
        orders[order["order_id"]] = s

        # WAL entry exists but not marked persisted (crash before portfolio save)
        write_fill_event(
            order_id=order["order_id"], trade_id="trd-x", signal_id=None,
            signal_run_id="run-001", portfolio_id="", strategy=STRATEGY,
            ticker=TICKER, action="BUY", session_date=SESSION,
            shares=5.0, price=100.0, value=500.0, cost=0.0, reason="test",
            execution_version=EXEC_VERSION, cash_before=10000.0, cash_after=9500.0,
        )

        # Portfolio does NOT have position (portfolio save was not completed)
        state = {"positions": {}, "cash": 10000.0}
        reconciled = reconcile_settling_orders(orders, STRATEGY, state)
        assert len(reconciled) == 1
        assert orders[order["order_id"]]["status"] == PENDING_PRICE

    def test_reconcile_crash_after_portfolio_save_reconstructs(self, tmp_path, monkeypatch):
        """SETTLING + fill_event 'filling' + position in portfolio → mark persisted + EXECUTED."""
        self._patch_fills(tmp_path, monkeypatch)
        _with_file(tmp_path, monkeypatch)

        from modules.fills import write_fill_event, is_fill_persisted
        orders = {}
        order, _ = get_or_create_order(
            orders=orders, signal_run_id="run-001", ticker=TICKER,
            strategy=STRATEGY, session_date=SESSION, action="BUY",
            target_value=5000.0, reason="test", signal_price=100.0,
            execution_version=EXEC_VERSION,
        )
        s = save_order(order)
        s = save_order(s, status="settling", trade_id="trd-y")
        orders[order["order_id"]] = s

        # WAL entry exists but not marked persisted (crash after save, before mark)
        write_fill_event(
            order_id=order["order_id"], trade_id="trd-y", signal_id=None,
            signal_run_id="run-001", portfolio_id="", strategy=STRATEGY,
            ticker=TICKER, action="BUY", session_date=SESSION,
            shares=5.0, price=100.0, value=500.0, cost=0.0, reason="test",
            execution_version=EXEC_VERSION, cash_before=10000.0, cash_after=9500.0,
        )

        # Portfolio HAS position (portfolio save completed before crash)
        state = {"positions": {TICKER: {"shares": 5}}, "cash": 9500.0}
        reconciled = reconcile_settling_orders(orders, STRATEGY, state)
        assert len(reconciled) == 1
        assert orders[order["order_id"]]["status"] == EXECUTED
        # Fill event should now be marked persisted (reconstruction)
        assert is_fill_persisted(order["order_id"])


# ---------------------------------------------------------------------------
# Point 3b: Atomic portfolio save + strict load
# ---------------------------------------------------------------------------

class TestAtomicPortfolioSave:
    def test_save_strategy_state_uses_atomic_replace(self):
        """Source check: save_strategy_state uses os.replace (atomic) and os.fsync."""
        with open("modules/state.py") as f:
            src = f.read()
        assert "os.replace" in src
        assert ".tmp" in src
        assert "os.fsync" in src

    def test_load_strategy_state_raises_on_corrupt_json(self, tmp_path, monkeypatch):
        """Corrupt portfolio JSON raises RuntimeError (not silent empty dict)."""
        import modules.state as state_mod
        from modules.state import load_strategy_state
        (tmp_path / "s1.json").write_text('{"bad": json here}')
        monkeypatch.setattr(state_mod, "STATE_DIR", str(tmp_path))
        with pytest.raises(RuntimeError, match="korrupt"):
            load_strategy_state("s1")

    def test_load_strategy_state_raises_on_empty_file(self, tmp_path, monkeypatch):
        """Empty portfolio file raises RuntimeError (not silent new state)."""
        import modules.state as state_mod
        from modules.state import load_strategy_state
        (tmp_path / "s2.json").write_text("")
        monkeypatch.setattr(state_mod, "STATE_DIR", str(tmp_path))
        with pytest.raises(RuntimeError, match="tom"):
            load_strategy_state("s2")

    def test_load_strategy_state_missing_file_returns_default(self, tmp_path, monkeypatch):
        """Missing file (first run) returns initial state (not an error)."""
        import modules.state as state_mod
        from modules.state import load_strategy_state
        monkeypatch.setattr(state_mod, "STATE_DIR", str(tmp_path))
        state = load_strategy_state("s3")
        assert "cash" in state
        assert state["cash"] > 0


# ---------------------------------------------------------------------------
# Point 4: Repair corrupt last ledger line on disk
# ---------------------------------------------------------------------------

class TestLoadOrdersTruncateLastLine:
    """load_orders truncates a corrupt last line from disk under exclusive lock."""

    def test_corrupt_last_line_removed_from_disk(self, tmp_path, monkeypatch):
        """After load_orders, the corrupt last line is removed from the file on disk."""
        path = _with_file(tmp_path, monkeypatch)
        o = _order()
        save_order(o)
        with open(path, "ab") as f:
            f.write(b'{"order_id": "bad-partial')  # no \n, crash mid-write
        # Load → repairs file
        loaded = load_orders()
        assert o["order_id"] in loaded
        # Verify file is clean on disk
        with open(path) as f:
            lines = [l for l in f.readlines() if l.strip()]
        assert len(lines) == 1  # only the valid save line remains
        for line in lines:
            json.loads(line)  # must parse without error

    def test_new_append_after_corrupt_last_line_succeeds(self, tmp_path, monkeypatch):
        """After load_orders truncates corrupt line, next save lands correctly."""
        path = _with_file(tmp_path, monkeypatch)
        o = _order(ticker="AAPL")
        save_order(o)
        with open(path, "ab") as f:
            f.write(b'{"partial":')  # crash mid-write
        # Load → truncates corrupt line
        load_orders()
        # Next save must succeed
        o2 = _order(ticker="MSFT")
        save_order(o2)
        loaded = load_orders()
        assert o["order_id"] in loaded
        assert o2["order_id"] in loaded

    def test_terminal_history_preserved_after_disk_truncation(self, tmp_path, monkeypatch):
        """Truncating corrupt last line must not lose earlier EXECUTED status."""
        path = _with_file(tmp_path, monkeypatch)
        o = _order()
        save_order(o)
        save_order(o, status=EXECUTED, trade_id="trd-xxx")
        with open(path, "ab") as f:
            f.write(b'{"corrupt":')
        loaded = load_orders()
        assert loaded[o["order_id"]]["status"] == EXECUTED

    def test_mid_file_corruption_still_raises(self, tmp_path, monkeypatch):
        """Corrupt line in the middle of file still raises RuntimeError (fail-closed)."""
        path = _with_file(tmp_path, monkeypatch)
        o1 = _order(ticker="AAPL")
        o2 = _order(ticker="MSFT")
        save_order(o1)
        with open(path, "a") as f:
            f.write('{"order_id": "bad-json-mid\n')  # broken mid-file
        save_order(o2)
        with pytest.raises(RuntimeError, match="korrupt"):
            load_orders()


# ---------------------------------------------------------------------------
# Point 1: Execution timing — execute.yml and DST-safe UTC schedule
# ---------------------------------------------------------------------------

class TestExecutionTiming:
    def test_execute_yml_uses_cron_not_workflow_run(self):
        """execute.yml must trigger via schedule cron, not workflow_run from premarket."""
        try:
            with open(".github/workflows/execute.yml") as f:
                content = f.read()
        except FileNotFoundError:
            pytest.skip("execute.yml not found — run from repo root")
        assert "cron:" in content
        assert "40 14" in content  # 14:40 UTC
        assert "workflow_run:" not in content

    def test_14_40_utc_after_nyse_open_in_edt(self):
        """14:40 UTC = 10:40 ET (UTC-4 EDT/summer) — after NYSE open at 09:30 ET."""
        from datetime import datetime, timezone, timedelta
        ET_EDT = timezone(timedelta(hours=-4))
        execute_utc = datetime(2026, 7, 1, 14, 40, tzinfo=timezone.utc)  # Summer day
        execute_et = execute_utc.astimezone(ET_EDT)
        nyse_open_et = datetime(2026, 7, 1, 9, 30, tzinfo=ET_EDT)
        assert execute_et > nyse_open_et, "14:40 UTC must be after 09:30 ET in EDT"

    def test_14_40_utc_after_nyse_open_in_est(self):
        """14:40 UTC = 09:40 ET (UTC-5 EST/winter) — after NYSE open at 09:30 ET."""
        from datetime import datetime, timezone, timedelta
        ET_EST = timezone(timedelta(hours=-5))
        execute_utc = datetime(2026, 1, 15, 14, 40, tzinfo=timezone.utc)  # Winter day
        execute_et = execute_utc.astimezone(ET_EST)
        nyse_open_et = datetime(2026, 1, 15, 9, 30, tzinfo=ET_EST)
        assert execute_et > nyse_open_et, "14:40 UTC must be after 09:30 ET in EST"

    def test_holiday_gives_graceful_skip_not_exception(self):
        """is_trading_session() returning False → graceful return, not CalendarUnavailableError.

        Source check: run_execute must separate the holiday check from CalendarUnavailableError.
        """
        with open("stock_bot.py") as f:
            src = f.read()
        fn_start = src.find("def run_execute(")
        fn_end = src.find("\ndef ", fn_start + 10)
        fn_body = src[fn_start:fn_end]
        # Non-trading day (holiday/weekend) must produce graceful skip, not an exception
        assert "SKIPPED_NON_TRADING_SESSION" in fn_body, (
            "run_execute must log SKIPPED_NON_TRADING_SESSION on holiday"
        )
        assert "if not trading_today" in fn_body, (
            "run_execute must guard on 'if not trading_today' to return gracefully"
        )

    def test_calendar_unavailable_error_propagates(self):
        """CalendarUnavailableError from is_trading_session → fail-closed (raised in run_execute)."""
        with open("stock_bot.py") as f:
            src = f.read()
        fn_start = src.find("def run_execute(")
        fn_end = src.find("\ndef ", fn_start + 10)
        fn_body = src[fn_start:fn_end]
        # CalendarUnavailableError must be caught and re-raised (not swallowed)
        assert "except CalendarUnavailableError" in fn_body
        assert "raise" in fn_body
