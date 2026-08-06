"""
Tests for modules/market_data.py:

  prefetch_open_prices  — valuation cache (open prices with signal-price fallback)
  prefetch_execution_prices — execution cache (strict session-date validation, no fallback)

All yfinance I/O is mocked.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from modules.market_data import prefetch_execution_prices, prefetch_open_prices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _single_df(open_prices):
    """Flat DataFrame (single-ticker yf.download response)."""
    return pd.DataFrame(
        {"Open": open_prices, "Close": [p + 1 for p in open_prices]},
        index=pd.date_range("2026-08-05", periods=len(open_prices)),
    )


def _multi_df(ticker_open_map):
    """MultiIndex DataFrame (multi-ticker yf.download response, group_by='ticker')."""
    dates = pd.date_range("2026-08-05", periods=2)
    tuples = []
    data_rows = [[], []]
    for ticker, opens in ticker_open_map.items():
        tuples.append((ticker, "Open"))
        tuples.append((ticker, "Close"))
        data_rows[0] += [opens[0], opens[0] + 1]
        data_rows[1] += [opens[1], opens[1] + 1]
    mi = pd.MultiIndex.from_tuples(tuples)
    return pd.DataFrame(data_rows, index=dates, columns=mi)


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

class TestPrefetchOpenPricesEmptyInput:
    def test_empty_tickers_returns_empty_dict_without_network_call(self):
        with patch("modules.market_data.yf.download") as mock_dl:
            result = prefetch_open_prices([])
        assert result == {}
        mock_dl.assert_not_called()


# ---------------------------------------------------------------------------
# Single-ticker (flat DataFrame from yf.download)
# ---------------------------------------------------------------------------

class TestPrefetchOpenPricesSingleTicker:
    def test_returns_last_open_price(self):
        df = _single_df([100.0, 105.0])
        with patch("modules.market_data.yf.download", return_value=df):
            result = prefetch_open_prices(["AAPL"])
        assert result["AAPL"] == 105.0

    def test_single_row_df(self):
        df = _single_df([99.5])
        with patch("modules.market_data.yf.download", return_value=df):
            result = prefetch_open_prices(["AAPL"])
        assert result["AAPL"] == 99.5

    def test_zero_open_price_triggers_fallback(self):
        df = _single_df([0.0, 0.0])
        with patch("modules.market_data.yf.download", return_value=df):
            result = prefetch_open_prices(["AAPL"], fallback_prices={"AAPL": 150.0})
        assert result["AAPL"] == 150.0

    def test_negative_open_price_triggers_fallback(self):
        df = _single_df([100.0, -5.0])
        with patch("modules.market_data.yf.download", return_value=df):
            result = prefetch_open_prices(["AAPL"], fallback_prices={"AAPL": 120.0})
        assert result["AAPL"] == 120.0


# ---------------------------------------------------------------------------
# Multi-ticker (MultiIndex DataFrame)
# ---------------------------------------------------------------------------

class TestPrefetchOpenPricesMultiTicker:
    def test_extracts_open_for_each_ticker(self):
        df = _multi_df({"AAPL": [100.0, 105.0], "MSFT": [200.0, 210.0]})
        with patch("modules.market_data.yf.download", return_value=df):
            result = prefetch_open_prices(["AAPL", "MSFT"])
        assert result["AAPL"] == 105.0
        assert result["MSFT"] == 210.0

    def test_missing_ticker_in_multiindex_uses_fallback(self):
        df = _multi_df({"AAPL": [100.0, 105.0]})
        with patch("modules.market_data.yf.download", return_value=df):
            result = prefetch_open_prices(
                ["AAPL", "NVDA"],
                fallback_prices={"NVDA": 900.0},
            )
        assert result["AAPL"] == 105.0
        assert result["NVDA"] == 900.0

    def test_no_fallback_for_missing_ticker_absent_from_result(self):
        df = _multi_df({"AAPL": [100.0, 105.0]})
        with patch("modules.market_data.yf.download", return_value=df):
            result = prefetch_open_prices(["AAPL", "NVDA"])
        assert "AAPL" in result
        assert "NVDA" not in result


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------

class TestPrefetchOpenPricesFallback:
    def test_empty_df_uses_fallback_for_all(self):
        with patch("modules.market_data.yf.download", return_value=pd.DataFrame()):
            result = prefetch_open_prices(
                ["AAPL", "MSFT"],
                fallback_prices={"AAPL": 150.0, "MSFT": 300.0},
            )
        assert result["AAPL"] == 150.0
        assert result["MSFT"] == 300.0

    def test_none_df_uses_fallback(self):
        with patch("modules.market_data.yf.download", return_value=None):
            result = prefetch_open_prices(["AAPL"], fallback_prices={"AAPL": 200.0})
        assert result["AAPL"] == 200.0

    def test_fallback_price_of_zero_not_used(self):
        with patch("modules.market_data.yf.download", return_value=pd.DataFrame()):
            result = prefetch_open_prices(["AAPL"], fallback_prices={"AAPL": 0.0})
        assert "AAPL" not in result

    def test_fallback_price_of_none_not_used(self):
        with patch("modules.market_data.yf.download", return_value=pd.DataFrame()):
            result = prefetch_open_prices(["AAPL"], fallback_prices={"AAPL": None})
        assert "AAPL" not in result

    def test_no_fallback_provided_missing_ticker_absent(self):
        with patch("modules.market_data.yf.download", return_value=pd.DataFrame()):
            result = prefetch_open_prices(["AAPL"])
        assert "AAPL" not in result


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestPrefetchOpenPricesErrorHandling:
    def test_yf_exception_falls_back_gracefully(self):
        with patch("modules.market_data.yf.download", side_effect=Exception("network error")):
            result = prefetch_open_prices(["AAPL"], fallback_prices={"AAPL": 180.0})
        assert result["AAPL"] == 180.0

    def test_yf_exception_no_fallback_returns_empty(self):
        with patch("modules.market_data.yf.download", side_effect=Exception("timeout")):
            result = prefetch_open_prices(["AAPL"])
        assert result == {}

    def test_partial_exception_does_not_lose_successful_chunks(self):
        good_df = _single_df([110.0, 115.0])
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("chunk 1 failed")
            return good_df

        with patch("modules.market_data.yf.download", side_effect=side_effect):
            result = prefetch_open_prices(
                ["FAIL", "AAPL"],
                fallback_prices={"FAIL": 50.0},
                chunk_size=1,
            )
        assert result["AAPL"] == 115.0
        assert result["FAIL"] == 50.0


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

class TestPrefetchOpenPricesChunking:
    def test_chunk_size_one_calls_download_once_per_ticker(self):
        df = _single_df([100.0, 105.0])
        with patch("modules.market_data.yf.download", return_value=df) as mock_dl:
            prefetch_open_prices(["AAPL", "MSFT", "NVDA"], chunk_size=1)
        assert mock_dl.call_count == 3

    def test_chunk_size_larger_than_tickers_calls_download_once(self):
        df = _multi_df({"AAPL": [100.0, 105.0], "MSFT": [200.0, 210.0]})
        with patch("modules.market_data.yf.download", return_value=df) as mock_dl:
            prefetch_open_prices(["AAPL", "MSFT"], chunk_size=80)
        assert mock_dl.call_count == 1

    def test_results_from_all_chunks_are_merged(self):
        responses = [_single_df([100.0, 105.0]), _single_df([200.0, 210.0])]
        call_count = [0]

        def side_effect(**kwargs):
            r = responses[call_count[0]]
            call_count[0] += 1
            return r

        with patch("modules.market_data.yf.download", side_effect=side_effect):
            result = prefetch_open_prices(["AAPL", "MSFT"], chunk_size=1)
        assert result["AAPL"] == 105.0
        assert result["MSFT"] == 210.0


# ---------------------------------------------------------------------------
# Open price is used as fill price (semantic test)
# ---------------------------------------------------------------------------

class TestPrefetchOpenPricesSemantics:
    def test_open_not_close_is_returned(self):
        """Open price (105) must be returned, not Close (106)."""
        df = pd.DataFrame(
            {"Open": [100.0, 105.0], "Close": [101.0, 106.0]},
            index=pd.date_range("2026-08-05", periods=2),
        )
        with patch("modules.market_data.yf.download", return_value=df):
            result = prefetch_open_prices(["AAPL"])
        assert result["AAPL"] == 105.0
        assert result["AAPL"] != 106.0

    def test_download_uses_daily_interval(self):
        """Verify the download call uses interval='1d' (not 1m intraday)."""
        df = _single_df([100.0])
        with patch("modules.market_data.yf.download", return_value=df) as mock_dl:
            prefetch_open_prices(["AAPL"])
        _, kwargs = mock_dl.call_args
        assert kwargs.get("interval") == "1d"

    def test_download_uses_two_day_period(self):
        """period='2d' ensures today's bar is included even mid-session."""
        df = _single_df([100.0])
        with patch("modules.market_data.yf.download", return_value=df) as mock_dl:
            prefetch_open_prices(["AAPL"])
        _, kwargs = mock_dl.call_args
        assert kwargs.get("period") == "2d"


# ===========================================================================
# prefetch_execution_prices — strict session-date validation, no fallback
# ===========================================================================

SESSION = "2026-08-06"   # a known trading day (Thursday)
FRIDAY  = "2026-08-07"   # signal date
MONDAY  = "2026-08-10"   # next trading session after Friday


def _session_df(session_date, open_price, ticker=None):
    """
    Single-row DataFrame whose index date matches session_date.
    If ticker is given, wraps in a MultiIndex (multi-ticker yf response).
    """
    idx = pd.DatetimeIndex([pd.Timestamp(session_date)])
    flat = pd.DataFrame({"Open": [open_price], "Close": [open_price + 1.0]}, index=idx)
    if ticker is None:
        return flat
    mi = pd.MultiIndex.from_tuples([(ticker, "Open"), (ticker, "Close")])
    return pd.DataFrame([[open_price, open_price + 1.0]], index=idx, columns=mi)


def _two_day_df(yesterday, today, open_prices, ticker=None):
    """
    Two-row DataFrame: yesterday's row then today's session row.
    Used to verify that only the session_date row is used.
    """
    idx = pd.DatetimeIndex([pd.Timestamp(yesterday), pd.Timestamp(today)])
    flat = pd.DataFrame({"Open": open_prices, "Close": [p + 1 for p in open_prices]}, index=idx)
    if ticker is None:
        return flat
    mi = pd.MultiIndex.from_tuples([(ticker, "Open"), (ticker, "Close")])
    data = [[open_prices[0], open_prices[0] + 1], [open_prices[1], open_prices[1] + 1]]
    return pd.DataFrame(data, index=idx, columns=mi)


class TestPrefetchExecutionPricesEmptyInput:
    def test_empty_tickers_returns_empty_dict_without_network_call(self):
        with patch("modules.market_data.yf.download") as mock_dl:
            result = prefetch_execution_prices([], SESSION)
        assert result == {}
        mock_dl.assert_not_called()


class TestPrefetchExecutionPricesSessionValidation:
    def test_returns_open_price_when_session_date_matches(self):
        df = _session_df(SESSION, 105.0)
        with patch("modules.market_data.yf.download", return_value=df):
            result = prefetch_execution_prices(["AAPL"], SESSION)
        assert result["AAPL"] == 105.0

    def test_rejects_row_when_date_does_not_match_session(self):
        """Yesterday's open row must NEVER be used as a fill price."""
        yesterday = "2026-08-05"
        df = _session_df(yesterday, 99.0)   # only yesterday's row
        with patch("modules.market_data.yf.download", return_value=df):
            result = prefetch_execution_prices(["AAPL"], SESSION)
        assert "AAPL" not in result

    def test_selects_correct_row_when_multiple_rows_present(self):
        """With 2 days of data, only the row matching session_date is used."""
        yesterday = "2026-08-05"
        df = _two_day_df(yesterday, SESSION, [99.0, 107.0])
        with patch("modules.market_data.yf.download", return_value=df):
            result = prefetch_execution_prices(["AAPL"], SESSION)
        assert result["AAPL"] == 107.0      # today's open
        assert result["AAPL"] != 99.0       # not yesterday's open

    def test_yesterday_open_cannot_be_used_as_fill(self):
        """If today's session row is missing, ticker is excluded — no stale fill."""
        yesterday = "2026-08-05"
        df = _session_df(yesterday, 120.0)  # only yesterday's bar
        with patch("modules.market_data.yf.download", return_value=df):
            result = prefetch_execution_prices(["AAPL"], SESSION)
        assert "AAPL" not in result


class TestPrefetchExecutionPricesNoFallback:
    def test_no_fallback_applied_when_session_row_missing(self):
        """Signal price must never become a fill price."""
        df = _session_df("2026-08-05", 99.0)  # wrong date
        with patch("modules.market_data.yf.download", return_value=df):
            result = prefetch_execution_prices(
                ["AAPL"],
                SESSION,
                # no fallback_prices parameter — the function accepts none
            )
        assert "AAPL" not in result

    def test_empty_download_produces_empty_result(self):
        with patch("modules.market_data.yf.download", return_value=pd.DataFrame()):
            result = prefetch_execution_prices(["AAPL"], SESSION)
        assert result == {}

    def test_exception_produces_empty_result(self):
        with patch("modules.market_data.yf.download", side_effect=Exception("network")):
            result = prefetch_execution_prices(["AAPL"], SESSION)
        assert result == {}


class TestPrefetchExecutionPricesFridayToMonday:
    def test_monday_session_uses_monday_open_not_friday_open(self):
        """
        Signal generated Friday; intended_execution_session → Monday.
        prefetch_execution_prices must use Monday's Open, not Friday's.
        """
        friday_open = 100.0
        monday_open = 103.0
        df = _two_day_df(FRIDAY, MONDAY, [friday_open, monday_open])
        with patch("modules.market_data.yf.download", return_value=df):
            result = prefetch_execution_prices(["AAPL"], MONDAY)
        assert result["AAPL"] == monday_open
        assert result["AAPL"] != friday_open

    def test_friday_open_rejected_when_session_is_monday(self):
        """Only Friday data available on Monday → no fill."""
        df = _session_df(FRIDAY, 100.0)
        with patch("modules.market_data.yf.download", return_value=df):
            result = prefetch_execution_prices(["AAPL"], MONDAY)
        assert "AAPL" not in result


class TestPrefetchExecutionPricesMultiTicker:
    def test_extracts_correct_open_for_each_ticker(self):
        idx = pd.DatetimeIndex([pd.Timestamp(SESSION)])
        mi = pd.MultiIndex.from_tuples([
            ("AAPL", "Open"), ("AAPL", "Close"),
            ("MSFT", "Open"), ("MSFT", "Close"),
        ])
        df = pd.DataFrame([[105.0, 106.0, 210.0, 211.0]], index=idx, columns=mi)
        with patch("modules.market_data.yf.download", return_value=df):
            result = prefetch_execution_prices(["AAPL", "MSFT"], SESSION)
        assert result["AAPL"] == 105.0
        assert result["MSFT"] == 210.0

    def test_ticker_not_in_multiindex_excluded_without_fallback(self):
        df = _session_df(SESSION, 105.0, ticker="AAPL")
        with patch("modules.market_data.yf.download", return_value=df):
            result = prefetch_execution_prices(["AAPL", "NVDA"], SESSION)
        assert result["AAPL"] == 105.0
        assert "NVDA" not in result


class TestPrefetchExecutionPricesDownloadParams:
    def test_uses_five_day_period(self):
        """period=5d is required to handle Monday execution of Friday signals."""
        df = _session_df(SESSION, 100.0)
        with patch("modules.market_data.yf.download", return_value=df) as mock_dl:
            prefetch_execution_prices(["AAPL"], SESSION)
        _, kwargs = mock_dl.call_args
        assert kwargs.get("period") == "5d"

    def test_uses_daily_interval(self):
        df = _session_df(SESSION, 100.0)
        with patch("modules.market_data.yf.download", return_value=df) as mock_dl:
            prefetch_execution_prices(["AAPL"], SESSION)
        _, kwargs = mock_dl.call_args
        assert kwargs.get("interval") == "1d"


class TestExecutionVsValuationCachesSeparation:
    def test_execution_cache_has_no_fallback_valuation_cache_does(self):
        """
        execution_price_cache: ticker absent when open price unavailable
        valuation_price_cache: ticker present via signal-price fallback
        """
        empty = pd.DataFrame()
        with patch("modules.market_data.yf.download", return_value=empty):
            exec_cache = prefetch_execution_prices(["AAPL"], SESSION)
            val_cache = prefetch_open_prices(["AAPL"], fallback_prices={"AAPL": 150.0})
        assert "AAPL" not in exec_cache      # no fill without valid session open
        assert val_cache["AAPL"] == 150.0    # valuation uses signal fallback

    def test_two_caches_are_independent_objects(self):
        df = _session_df(SESSION, 105.0)
        with patch("modules.market_data.yf.download", return_value=df):
            exec_cache = prefetch_execution_prices(["AAPL"], SESSION)
            val_cache = prefetch_open_prices(["AAPL"], fallback_prices={"AAPL": 99.0})
        assert exec_cache is not val_cache
        exec_cache["NEW"] = 1.0
        assert "NEW" not in val_cache
