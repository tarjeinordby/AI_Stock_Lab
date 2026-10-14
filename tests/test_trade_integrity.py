"""Integrity of trades.csv: every row carries its order/trade IDs, WAL recovery never
duplicates a row, and rejected buys get an accurate order status."""

import copy
import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest

if "anthropic" not in sys.modules:
    sys.modules["anthropic"] = MagicMock()

STRATEGY = "Max_Return_AI"
SESSION = "2026-10-14"


@pytest.fixture
def harness(tmp_path, monkeypatch):
    import modules.fills as fm
    import modules.orders as om
    import stock_bot as sb
    from modules.versioning import PORTFOLIO_VERSION

    monkeypatch.setattr(om, "ORDERS_FILE", str(tmp_path / "orders.jsonl"))
    monkeypatch.setattr(om, "_ORDERS_LOCK_FILE", str(tmp_path / "orders.jsonl.lock"))
    monkeypatch.setattr(fm, "FILLS_FILE", tmp_path / "fills.jsonl")
    monkeypatch.setattr(fm, "_FILLS_LOCK_FILE", tmp_path / "fills.jsonl.lock")

    store = {STRATEGY: {
        "strategy": STRATEGY, "portfolio_id": "p-int", "portfolio_version": PORTFOLIO_VERSION,
        "cash": 10000.0, "positions": {}, "highest_portfolio_value": 10000.0,
        "weekly_meta": {"iso_week": "", "buys_this_week": 0}, "cooldowns": {},
        "last_execution_date": None,
    }}
    monkeypatch.setattr(sb, "load_strategy_state", lambda n: copy.deepcopy(store[n]))
    monkeypatch.setattr(sb, "save_strategy_state", lambda n, s: store.__setitem__(n, copy.deepcopy(s)))
    monkeypatch.setattr(sb, "current_portfolio_value", lambda s, *a, **k: (10000.0, 0.0, s))
    monkeypatch.setattr(sb, "check_sub_sector_concentration", lambda *a, **k: (False, "", 0))
    monkeypatch.setattr(sb, "check_correlation_against_held", lambda *a, **k: (False, "", 0.0))
    monkeypatch.setattr(sb, "send_telegram", lambda *a, **k: None)

    def run(weight, open_price, signal_price=100.0):
        monkeypatch.setattr(sb, "build_target_weights", lambda *a, **k: {"AAPL": weight})
        candidate = {"ticker": "AAPL", "price": signal_price, "rank": 1, "strategy_score": 80.0,
                     "sector": "Technology", "above_sma200": True, "days_to_earnings": None}
        signal = {"strategies": {STRATEGY: {"candidates": [candidate]}},
                  "regime": {"regime": "bullish"}}
        exec_cache = {"AAPL": open_price} if open_price is not None else {}
        orders = {}
        _, trades_df = sb.run_strategy_execution(
            STRATEGY, signal, pd.DataFrame(), {}, exec_cache,
            orders=orders, signal_run_id="run-int", session_date=SESSION,
            execution_version="exec_next_open_v1",
        )
        buy_orders = [o for o in orders.values() if o["action"] == "BUY" and o["ticker"] == "AAPL"]
        return trades_df, buy_orders

    return run


# --- Duplicate bug -------------------------------------------------------------

class TestTradeRowsCarryIds:

    def test_filled_buy_row_has_order_and_trade_id(self, harness):
        trades_df, buy_orders = harness(weight=0.10, open_price=100.0)
        rows = trades_df[trades_df["ticker"] == "AAPL"]
        assert len(rows) == 1
        assert len(buy_orders) == 1
        assert rows.iloc[0]["order_id"] == buy_orders[0]["order_id"]
        assert rows.iloc[0]["trade_id"] == buy_orders[0]["trade_id"]
        assert pd.notna(rows.iloc[0]["signal_run_id"])

    def test_wal_recovery_after_normal_run_does_not_duplicate(self, harness):
        """Reproduces the 19 duplicate pairs: a normal run followed by WAL projection."""
        from modules.fills import project_fills_to_trades
        trades_df, _ = harness(weight=0.10, open_price=100.0)
        assert len(trades_df[trades_df["ticker"] == "AAPL"]) == 1
        projected = project_fills_to_trades(trades_df)
        assert len(projected[projected["ticker"] == "AAPL"]) == 1

    def test_wal_recovery_skips_legacy_row_without_ids(self, tmp_path, monkeypatch):
        """A pre-fix row with no trade_id must still be recognised as the same trade."""
        import modules.fills as fm
        from modules.fills import (
            compute_portfolio_state_hash, mark_fill_persisted, project_fills_to_trades,
            write_commit_intent, write_fill_event,
        )
        monkeypatch.setattr(fm, "FILLS_FILE", tmp_path / "fills.jsonl")
        monkeypatch.setattr(fm, "_FILLS_LOCK_FILE", tmp_path / "fills.jsonl.lock")
        rec = write_fill_event(
            order_id="ord-legacy", trade_id="trd-legacy", signal_id=None, signal_run_id="r",
            portfolio_id="", portfolio_version="", strategy="s1", ticker="ABNB", action="PYRAMID_FILL",
            intended_execution_session="2026-08-24", actual_execution_session="2026-08-24",
            shares=6.193968, execution_price=189.0, execution_price_timestamp="2026-08-24T13:30:00Z",
            commission_amount=1.17, reason="pyr", execution_version="v1",
            cash_before=3876.61, cash_after=2704.78,
        )
        ci = write_commit_intent(
            strategy="s1", portfolio_id="", portfolio_version="",
            pre_portfolio_state_hash=compute_portfolio_state_hash({"cash": 1.0, "positions": {}}),
            post_portfolio_state_hash="a" * 64,
            fills=[{"order_id": "ord-legacy", "fill_attempt_id": rec["fill_attempt_id"],
                    "filling_content_hash": rec["content_hash"]}],
        )
        mark_fill_persisted("ord-legacy", rec["fill_attempt_id"], rec["content_hash"],
                            post_portfolio_state_hash="a" * 64, commit_id=ci["commit_id"])
        legacy = pd.DataFrame([{
            "date": "2026-08-24", "strategy": "s1", "action": "BUY", "ticker": "ABNB",
            "shares": 6.193968, "price": 189.0, "order_id": None, "trade_id": None,
            "execution_session": "2026-08-24",
        }])
        projected = project_fills_to_trades(legacy)
        assert len(projected) == 1

    def test_pyramid_fill_row_labelled_pyramid_fill(self):
        from modules.portfolio import execute_pyramid_fill
        state = {"cash": 5000.0, "positions": {"AAPL": {
            "shares": 6.0, "avg_price": 100.0, "is_partial": True, "buy_date": "2026-01-02",
            "pyramid_remaining_value": 600.0, "pyramid_min_price": 102.0, "highest_price": 100.0,
        }}}
        _, df, trade = execute_pyramid_fill(state, pd.DataFrame(), STRATEGY, "AAPL", {"AAPL": 110.0})
        assert trade["action"] == "PYRAMID_FILL"
        assert df.iloc[-1]["action"] == "PYRAMID_FILL"


# --- Order-status bug ----------------------------------------------------------

class TestRejectedBuyStatus:

    def test_below_min_position_is_cancelled_not_missing_price(self, harness):
        """Reproduces the 114 orders from 2026-10-14: price existed, order was just too small."""
        from modules.orders import CANCELLED
        trades_df, buy_orders = harness(weight=0.05, open_price=100.0)  # 10000*0.05*0.6 = $300
        assert trades_df.empty or len(trades_df[trades_df["ticker"] == "AAPL"]) == 0
        assert buy_orders[0]["status"] == CANCELLED
        assert buy_orders[0]["failure_code"] == "below_min_position"

    def test_open_gap_is_cancelled_with_gap_code(self, harness):
        from modules.orders import CANCELLED
        _, buy_orders = harness(weight=0.10, open_price=106.0, signal_price=100.0)
        assert buy_orders[0]["status"] == CANCELLED
        assert buy_orders[0]["failure_code"] == "open_gap_too_large"

    def test_truly_missing_price_stays_pending(self, harness):
        from modules.orders import PENDING_PRICE
        _, buy_orders = harness(weight=0.10, open_price=None)
        assert buy_orders[0]["status"] == PENDING_PRICE
        assert buy_orders[0]["failure_code"] == "missing_execution_price"
