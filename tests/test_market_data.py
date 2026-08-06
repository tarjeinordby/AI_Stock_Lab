"""
Tests for modules/market_data.py — prefetch_open_prices (MOO execution model).

prefetch_open_prices is the entry point for execute-mode fill prices.
It batch-fetches today's daily Open bar for all tickers and falls back
to the signal's yesterday-close prices when a ticker's data is unavailable.

All yfinance I/O is mocked.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from modules.market_data import prefetch_open_prices


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
