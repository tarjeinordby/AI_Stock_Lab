"""
Behavioral tests for the execution model (modules/portfolio.py).

Key invariants:
  - execute_buy/sell/pyramid_fill require a validated entry in execution_price_cache.
  - If the ticker is absent from execution_price_cache, no fill occurs:
      - cash is unchanged
      - positions are unchanged
      - trades_df is unchanged
      - the function returns (state, trades_df, None)
  - Trade records for successful fills contain execution metadata:
      execution_session, execution_price_source, order_status="filled"
  - execution_price_cache and valuation_price_cache are independent:
      benchmark and MTM functions use valuation, not execution prices.
"""

import pandas as pd
import pytest

from modules.portfolio import (
    EXECUTION_PRICE_SOURCE,
    execute_buy,
    execute_pyramid_fill,
    execute_sell,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INITIAL_CASH = 100_000.0


def _make_state(cash=INITIAL_CASH, positions=None):
    """Minimal strategy state dict."""
    return {
        "cash": float(cash),
        "positions": positions or {},
        "cooldowns": {},
        "highest_portfolio_value": float(cash),
        "weekly_meta": {"buys_this_week": 0},
    }


def _make_candidate(ticker, price=100.0, score=7.5, rank=1):
    return {
        "ticker": ticker,
        "price": price,
        "strategy_score": score,
        "rank": rank,
        "sector": "Technology",
    }


def _empty_trades():
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# execute_buy: no fill when ticker missing from execution_price_cache
# ---------------------------------------------------------------------------

class TestExecuteBuyNoFill:
    def test_returns_none_trade_when_ticker_absent_from_execution_cache(self):
        state = _make_state()
        _, _, trade = execute_buy(
            state, _empty_trades(), "test_strat", "AAPL",
            target_value=5000.0, reason="test",
            candidate=_make_candidate("AAPL"),
            execution_price_cache={},   # AAPL not present
        )
        assert trade is None

    def test_cash_unchanged_when_no_fill(self):
        state = _make_state(cash=50_000.0)
        state_after, _, _ = execute_buy(
            state, _empty_trades(), "test_strat", "AAPL",
            target_value=5000.0, reason="test",
            candidate=_make_candidate("AAPL"),
            execution_price_cache={},
        )
        assert state_after["cash"] == 50_000.0

    def test_positions_unchanged_when_no_fill(self):
        state = _make_state()
        state_after, _, _ = execute_buy(
            state, _empty_trades(), "test_strat", "AAPL",
            target_value=5000.0, reason="test",
            candidate=_make_candidate("AAPL"),
            execution_price_cache={},
        )
        assert "AAPL" not in state_after["positions"]

    def test_trades_df_unchanged_when_no_fill(self):
        state = _make_state()
        trades = _empty_trades()
        _, trades_after, _ = execute_buy(
            state, trades, "test_strat", "AAPL",
            target_value=5000.0, reason="test",
            candidate=_make_candidate("AAPL"),
            execution_price_cache={},
        )
        assert len(trades_after) == 0

    def test_signal_price_cannot_substitute_for_execution_price(self):
        """
        Candidate has a price (signal_price=100). This must NOT be used
        as a fill price when the ticker is absent from execution_price_cache.
        """
        state = _make_state()
        candidate = _make_candidate("AAPL", price=100.0)
        state_after, _, trade = execute_buy(
            state, _empty_trades(), "test_strat", "AAPL",
            target_value=5000.0, reason="test",
            candidate=candidate,
            execution_price_cache={},   # AAPL not present despite candidate.price=100
        )
        assert trade is None
        assert "AAPL" not in state_after["positions"]
        assert state_after["cash"] == INITIAL_CASH


# ---------------------------------------------------------------------------
# execute_buy: successful fill
# ---------------------------------------------------------------------------

class TestExecuteBuyFill:
    def test_fill_at_execution_price_not_signal_price(self):
        """Fill must happen at the open price (102.0), not signal price (100.0).
        Uses a 2% gap — below the 2.5% half-size threshold and far from the 5% skip threshold.
        """
        state = _make_state()
        candidate = _make_candidate("AAPL", price=100.0)   # signal price (yesterday's close)
        execution_price_cache = {"AAPL": 102.0}            # open price (2% gap, within limits)
        _, _, trade = execute_buy(
            state, _empty_trades(), "test_strat", "AAPL",
            target_value=5000.0, reason="test",
            candidate=candidate,
            execution_price_cache=execution_price_cache,
        )
        assert trade is not None
        assert trade["price"] == 102.0   # filled at open price, not signal price

    def test_trade_record_contains_execution_metadata(self):
        state = _make_state()
        _, _, trade = execute_buy(
            state, _empty_trades(), "test_strat", "AAPL",
            target_value=5000.0, reason="test",
            candidate=_make_candidate("AAPL", price=100.0),
            execution_price_cache={"AAPL": 101.0},   # 1% gap — no filtering
        )
        assert trade is not None
        assert trade["execution_price_source"] == EXECUTION_PRICE_SOURCE
        assert trade["order_status"] == "filled"
        assert "execution_session" in trade

    def test_cash_decreases_by_filled_amount(self):
        state = _make_state(cash=50_000.0)
        state_after, _, trade = execute_buy(
            state, _empty_trades(), "test_strat", "AAPL",
            target_value=5000.0, reason="test",
            candidate=_make_candidate("AAPL"),
            execution_price_cache={"AAPL": 100.0},
        )
        assert trade is not None
        assert state_after["cash"] < 50_000.0

    def test_position_created_after_fill(self):
        state = _make_state()
        state_after, _, trade = execute_buy(
            state, _empty_trades(), "test_strat", "AAPL",
            target_value=5000.0, reason="test",
            candidate=_make_candidate("AAPL"),
            execution_price_cache={"AAPL": 100.0},
        )
        assert trade is not None
        assert "AAPL" in state_after["positions"]


# ---------------------------------------------------------------------------
# execute_sell: no fill when ticker missing from execution_price_cache
# ---------------------------------------------------------------------------

class TestExecuteSellNoFill:
    def _state_with_position(self, ticker="AAPL", shares=50.0, avg_price=100.0):
        state = _make_state(cash=10_000.0)
        state["positions"][ticker] = {
            "shares": shares, "avg_price": avg_price, "last_price": avg_price,
            "highest_price": avg_price, "market_value": shares * avg_price,
            "buy_date": "2026-07-01", "is_partial": False,
            "pyramid_remaining_value": 0.0, "pyramid_min_price": 0.0,
            "buy_score": 7.0,
        }
        return state

    def test_returns_none_trade_when_ticker_absent(self):
        state = self._state_with_position()
        _, _, trade = execute_sell(
            state, _empty_trades(), "test_strat", "AAPL",
            reason="Stop-loss", execution_price_cache={},
        )
        assert trade is None

    def test_cash_unchanged_when_no_fill(self):
        state = self._state_with_position()
        original_cash = state["cash"]
        state_after, _, _ = execute_sell(
            state, _empty_trades(), "test_strat", "AAPL",
            reason="Stop-loss", execution_price_cache={},
        )
        assert state_after["cash"] == original_cash

    def test_position_retained_when_no_fill(self):
        state = self._state_with_position()
        state_after, _, _ = execute_sell(
            state, _empty_trades(), "test_strat", "AAPL",
            reason="Stop-loss", execution_price_cache={},
        )
        assert "AAPL" in state_after["positions"]

    def test_trades_df_unchanged_when_no_fill(self):
        state = self._state_with_position()
        trades = _empty_trades()
        _, trades_after, _ = execute_sell(
            state, trades, "test_strat", "AAPL",
            reason="Stop-loss", execution_price_cache={},
        )
        assert len(trades_after) == 0


# ---------------------------------------------------------------------------
# execute_sell: successful fill
# ---------------------------------------------------------------------------

class TestExecuteSellFill:
    def _state_with_position(self, ticker="AAPL", shares=50.0, avg_price=100.0):
        state = _make_state(cash=10_000.0)
        state["positions"][ticker] = {
            "shares": shares, "avg_price": avg_price, "last_price": avg_price,
            "highest_price": avg_price, "market_value": shares * avg_price,
            "buy_date": "2026-07-01", "is_partial": False,
            "pyramid_remaining_value": 0.0, "pyramid_min_price": 0.0,
            "buy_score": 7.0,
        }
        return state

    def test_fill_uses_execution_price(self):
        state = self._state_with_position(shares=50.0, avg_price=100.0)
        _, _, trade = execute_sell(
            state, _empty_trades(), "test_strat", "AAPL",
            reason="Trailing stop", execution_price_cache={"AAPL": 120.0},
        )
        assert trade is not None
        assert trade["price"] == 120.0

    def test_trade_record_contains_execution_metadata(self):
        state = self._state_with_position()
        _, _, trade = execute_sell(
            state, _empty_trades(), "test_strat", "AAPL",
            reason="Stop-loss", execution_price_cache={"AAPL": 85.0},
        )
        assert trade is not None
        assert trade["execution_price_source"] == EXECUTION_PRICE_SOURCE
        assert trade["order_status"] == "filled"
        assert "execution_session" in trade

    def test_position_removed_after_fill(self):
        state = self._state_with_position()
        state_after, _, trade = execute_sell(
            state, _empty_trades(), "test_strat", "AAPL",
            reason="Stop-loss", execution_price_cache={"AAPL": 85.0},
        )
        assert trade is not None
        assert "AAPL" not in state_after["positions"]

    def test_cash_increases_after_fill(self):
        state = self._state_with_position(shares=50.0, avg_price=100.0)
        original_cash = state["cash"]
        state_after, _, trade = execute_sell(
            state, _empty_trades(), "test_strat", "AAPL",
            reason="Stop-loss", execution_price_cache={"AAPL": 85.0},
        )
        assert trade is not None
        assert state_after["cash"] > original_cash


# ---------------------------------------------------------------------------
# execute_pyramid_fill: no fill when ticker missing from execution_price_cache
# ---------------------------------------------------------------------------

class TestExecutePyramidFillNoFill:
    def _partial_state(self, ticker="AAPL", buy_date="2026-07-01", days_back=10):
        from modules.state import today_str
        import datetime
        today = datetime.date.today()
        early = (today - datetime.timedelta(days=days_back)).isoformat()
        state = _make_state(cash=20_000.0)
        state["positions"][ticker] = {
            "shares": 30.0, "avg_price": 100.0, "last_price": 102.0,
            "highest_price": 105.0, "market_value": 3060.0,
            "buy_date": early, "is_partial": True,
            "pyramid_remaining_value": 2000.0, "pyramid_min_price": 102.0,
            "buy_score": 7.0,
        }
        return state

    def test_returns_none_when_ticker_absent(self):
        state = self._partial_state()
        _, _, fill = execute_pyramid_fill(
            state, _empty_trades(), "test_strat", "AAPL",
            execution_price_cache={},
        )
        assert fill is None

    def test_cash_unchanged_when_no_fill(self):
        state = self._partial_state()
        original_cash = state["cash"]
        state_after, _, _ = execute_pyramid_fill(
            state, _empty_trades(), "test_strat", "AAPL",
            execution_price_cache={},
        )
        assert state_after["cash"] == original_cash

    def test_position_stays_partial_when_no_fill(self):
        state = self._partial_state()
        state_after, _, _ = execute_pyramid_fill(
            state, _empty_trades(), "test_strat", "AAPL",
            execution_price_cache={},
        )
        assert state_after["positions"]["AAPL"]["is_partial"] is True
