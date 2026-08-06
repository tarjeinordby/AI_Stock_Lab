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
    TERMINAL,
    build_order,
    expire_stale_orders,
    get_or_create_order,
    get_pending_for_session,
    load_orders,
    make_order_id,
    make_trade_id,
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
    return path


# ---------------------------------------------------------------------------
# make_order_id: determinism
# ---------------------------------------------------------------------------

class TestMakeOrderId:
    def test_same_inputs_same_id(self):
        a = make_order_id(SIGNAL_RUN_ID, TICKER, STRATEGY, SESSION, "BUY")
        b = make_order_id(SIGNAL_RUN_ID, TICKER, STRATEGY, SESSION, "BUY")
        assert a == b

    def test_different_action_different_id(self):
        buy = make_order_id(SIGNAL_RUN_ID, TICKER, STRATEGY, SESSION, "BUY")
        sell = make_order_id(SIGNAL_RUN_ID, TICKER, STRATEGY, SESSION, "SELL")
        assert buy != sell

    def test_different_ticker_different_id(self):
        aapl = make_order_id(SIGNAL_RUN_ID, "AAPL", STRATEGY, SESSION, "BUY")
        msft = make_order_id(SIGNAL_RUN_ID, "MSFT", STRATEGY, SESSION, "BUY")
        assert aapl != msft

    def test_different_session_different_id(self):
        today = make_order_id(SIGNAL_RUN_ID, TICKER, STRATEGY, "2026-08-06", "BUY")
        tomorrow = make_order_id(SIGNAL_RUN_ID, TICKER, STRATEGY, "2026-08-07", "BUY")
        assert today != tomorrow

    def test_different_signal_run_id_different_id(self):
        a = make_order_id("run-aaa", TICKER, STRATEGY, SESSION, "BUY")
        b = make_order_id("run-bbb", TICKER, STRATEGY, SESSION, "BUY")
        assert a != b

    def test_id_starts_with_ord_prefix(self):
        oid = make_order_id(SIGNAL_RUN_ID, TICKER, STRATEGY, SESSION, "BUY")
        assert oid.startswith("ord-")

    def test_id_has_fixed_length(self):
        # "ord-" + 12 hex chars = 16 chars total
        oid = make_order_id(SIGNAL_RUN_ID, TICKER, STRATEGY, SESSION, "BUY")
        assert len(oid) == 16


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
        o = _order()
        expected = make_order_id(SIGNAL_RUN_ID, TICKER, STRATEGY, SESSION, "BUY")
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
