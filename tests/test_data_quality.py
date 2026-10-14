"""
Tests for modules/market_data.run_data_quality_checks.

Large single-day moves must be FLAGGED, never silently overwritten: a real
+17% earnings gap is information the technicals and the stop-loss need to see.
"""

import pandas as pd

from modules.market_data import run_data_quality_checks


def _df(closes, volumes=None):
    volumes = volumes if volumes is not None else [1_000_000] * len(closes)
    return pd.DataFrame(
        {"Close": closes, "Volume": volumes},
        index=pd.date_range("2026-08-03", periods=len(closes), freq="B"),
    )


class TestLargeMovesAreFlaggedNotReplaced:
    def test_large_up_move_keeps_last_close(self):
        # ARM 2026-08-03 -> 2026-08-04: 239.06 -> 280.56 (+17.4%)
        md = {"ARM": _df([239.06, 280.56])}
        md, report = run_data_quality_checks(md)
        assert float(md["ARM"]["Close"].iloc[-1]) == 280.56

    def test_large_up_move_is_flagged(self):
        md = {"ARM": _df([239.06, 280.56])}
        _, report = run_data_quality_checks(md)
        flags = report["price_flags"]
        assert [f["ticker"] for f in flags] == ["ARM"]
        assert flags[0]["pct_change"] == round(280.56 / 239.06 - 1, 4)
        assert flags[0]["last_price"] == 280.56
        assert flags[0]["prev_price"] == 239.06

    def test_large_down_move_below_split_threshold_keeps_close(self):
        md = {"XYZ": _df([100.0, 75.0])}  # -25%
        md, report = run_data_quality_checks(md)
        assert float(md["XYZ"]["Close"].iloc[-1]) == 75.0
        assert report["price_flags"][0]["pct_change"] == -0.25

    def test_input_dataframe_is_not_mutated(self):
        original = _df([100.0, 130.0])
        md = {"XYZ": original}
        md, _ = run_data_quality_checks(md)
        assert md["XYZ"] is original
        assert float(original["Close"].iloc[-1]) == 130.0

    def test_small_move_not_flagged(self):
        md = {"XYZ": _df([100.0, 110.0])}
        _, report = run_data_quality_checks(md)
        assert report["price_flags"] == []

    def test_move_exactly_at_threshold_not_flagged(self):
        md = {"XYZ": _df([100.0, 115.0])}
        _, report = run_data_quality_checks(md)
        assert report["price_flags"] == []


class TestOtherChecksUnchanged:
    def test_split_drop_detected_and_close_kept(self):
        md = {"XYZ": _df([100.0, 50.0])}
        md, report = run_data_quality_checks(md)
        assert float(md["XYZ"]["Close"].iloc[-1]) == 50.0
        assert report["split_detected"][0]["ratio"] == 2
        assert report["price_flags"] == []

    def test_zero_volume_streak_excluded(self):
        md = {"XYZ": _df([10, 10, 10, 10], volumes=[100, 0, 0, 0])}
        md, report = run_data_quality_checks(md)
        assert "XYZ" not in md
        assert report["volume_excluded"] == [{"ticker": "XYZ", "zero_days": 3}]

    def test_benchmarks_skipped(self):
        md = {"SPY": _df([100.0, 150.0]), "QQQ": _df([100.0, 50.0])}
        md, report = run_data_quality_checks(md)
        assert report == {"price_flags": [], "volume_excluded": [], "split_detected": []}
        assert float(md["SPY"]["Close"].iloc[-1]) == 150.0
