import copy
import inspect
import sys
from unittest.mock import MagicMock

import pandas as pd

if "anthropic" not in sys.modules:
    sys.modules["anthropic"] = MagicMock()

from modules.reconcile import reconcile_strategy

CREATED = "2026-10-13T11:32:36+00:00"


def _trades(*rows):
    base = {"order_id": "o", "trade_id": "t", "cost": 0.5}
    return pd.DataFrame([{**base, **r} for r in rows])


BUY = {"timestamp_utc": "2026-10-14T18:50:57+00:00", "strategy": "S", "action": "BUY",
       "ticker": "ABBV", "shares": 2.0, "price": 250.0, "value": 500.0, "net_cash_effect": -500.5}


def _state(cash, positions):
    return {"created_at": CREATED, "cash": cash, "positions": positions}


def test_matching_state_is_ok():
    r = reconcile_strategy(_trades(BUY), _state(9499.5, {"ABBV": {"shares": 2.0, "last_price": 250.0}}), "S")
    assert r["ok"] and abs(r["cash_diff"]) < 1e-9 and not r["share_diffs"]


def test_duplicate_row_is_flagged():
    dup = dict(BUY, trade_id="t2")
    r = reconcile_strategy(_trades(BUY, dup), _state(9499.5, {"ABBV": {"shares": 2.0, "last_price": 250.0}}), "S")
    assert not r["ok"]
    assert r["deviation_usd"] > 1.0
    assert r["share_diffs"][0]["ticker"] == "ABBV"


def test_trades_before_portfolio_creation_are_ignored():
    old = dict(BUY, timestamp_utc="2026-06-11T07:38:07+00:00", ticker="ROKU")
    r = reconcile_strategy(_trades(old, BUY), _state(9499.5, {"ABBV": {"shares": 2.0}}), "S")
    assert r["ok"]


def test_split_row_changes_shares_not_cash():
    split = {"timestamp_utc": "2026-10-15T15:00:00+00:00", "strategy": "S", "action": "SPLIT",
             "ticker": "ABBV", "shares": 2.0, "price": 0.0, "value": 0.0, "cost": 0.0,
             "net_cash_effect": 0.0}
    r = reconcile_strategy(_trades(BUY, split), _state(9499.5, {"ABBV": {"shares": 4.0}}), "S")
    assert r["ok"]


def test_cash_only_deviation_under_one_dollar_is_ok():
    r = reconcile_strategy(_trades(BUY), _state(9499.9, {"ABBV": {"shares": 2.0}}), "S")
    assert r["ok"]
    r = reconcile_strategy(_trades(BUY), _state(9501.0, {"ABBV": {"shares": 2.0}}), "S")
    assert not r["ok"]


def test_post_execute_reconciliation_alerts_on_deviation(monkeypatch):
    import stock_bot as sb
    states = {s: _state(10000.0, {}) for s in sb.STRATEGIES}
    first = next(iter(sb.STRATEGIES))
    states[first] = _state(9000.0, {})  # $1000 missing with no trades
    monkeypatch.setattr(sb, "load_trades", lambda: pd.DataFrame())
    monkeypatch.setattr(sb, "load_strategy_state", lambda s: copy.deepcopy(states[s]))
    sent = []
    results = sb.run_post_execute_reconciliation(send_fn=sent.append)
    assert len(sent) == 1 and "AVSTEMMING" in sent[0] and first in sent[0]
    assert sum(not r["ok"] for r in results) == 1


def test_post_execute_reconciliation_silent_when_ok(monkeypatch):
    import stock_bot as sb
    monkeypatch.setattr(sb, "load_trades", lambda: pd.DataFrame())
    monkeypatch.setattr(sb, "load_strategy_state", lambda s: _state(10000.0, {}))
    sent = []
    sb.run_post_execute_reconciliation(send_fn=sent.append)
    assert sent == []


def test_run_execute_reconciles_after_saving_trades():
    import stock_bot as sb
    src = inspect.getsource(sb.run_execute)
    assert src.index("save_csv(trades_df, TRADES_FILE)") < src.index("run_post_execute_reconciliation()")


def test_split_is_recorded_and_not_applied_twice(monkeypatch):
    import stock_bot as sb
    strat = next(iter(sb.STRATEGIES))
    store = {s: {"positions": {}} for s in sb.STRATEGIES}
    store[strat] = {"positions": {"NVDA": {"shares": 3.0, "avg_price": 900.0}}}
    monkeypatch.setattr(sb, "load_strategy_state", lambda s: copy.deepcopy(store[s]))
    monkeypatch.setattr(sb, "save_strategy_state", lambda s, st: store.__setitem__(s, copy.deepcopy(st)))
    splits = [{"ticker": "NVDA", "ratio": 10}]
    _, df = sb.apply_split_adjustments(splits, pd.DataFrame(), "2026-10-15")
    _, df = sb.apply_split_adjustments(splits, df, "2026-10-15")
    assert store[strat]["positions"]["NVDA"]["shares"] == 30.0
    rows = df[df["action"] == "SPLIT"]
    assert len(rows) == 1 and rows.iloc[0]["shares"] == 27.0 and rows.iloc[0]["net_cash_effect"] == 0.0
