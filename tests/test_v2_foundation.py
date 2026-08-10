"""
V2 Foundation tests — behavioral tests for quant_baseline_v2.

All tests verify observable behavior, not source code text or function existence.
Tests pass without modifying any V1 code or data.
"""

import math

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(n_days: int, start: float = 100.0, daily_ret: float = 0.001) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    prices = [start * ((1 + daily_ret) ** i) for i in range(n_days)]
    return pd.DataFrame(
        {"Adj Close": prices, "Close": prices, "Volume": [1_000_000] * n_days},
        index=dates,
    )


def _is_null(x) -> bool:
    if x is None:
        return True
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


# ---------------------------------------------------------------------------
# A. V1 isolation
# ---------------------------------------------------------------------------

class TestV1Isolation:

    def test_v1_model_version_unchanged_after_v2_import(self):
        from modules.versioning import MODEL_VERSION
        import modules.v2_strategy
        from modules.versioning import MODEL_VERSION as after
        assert MODEL_VERSION == "quant_baseline_v1"
        assert after == "quant_baseline_v1"

    def test_v1_portfolio_version_unchanged_after_v2_import(self):
        from modules.versioning import PORTFOLIO_VERSION
        import modules.v2_strategy
        from modules.versioning import PORTFOLIO_VERSION as after
        assert PORTFOLIO_VERSION == "risk_parity_pyramid_v1"
        assert after == "risk_parity_pyramid_v1"

    def test_v1_strategies_dict_not_contaminated(self):
        from modules.scoring import STRATEGIES
        import modules.v2_strategy
        v1_expected = {"Max_Return_AI", "Quality_Momentum_AI", "Balanced_AI",
                       "Low_Risk_AI", "MegaCap_AI", "AI_Sentiment_AI"}
        for name in v1_expected:
            assert name in STRATEGIES
        for name in STRATEGIES:
            assert not name.endswith("_V2"), f"V2 name leaked into V1 STRATEGIES: {name}"

    def test_v1_build_target_weights_is_separate_function(self):
        from modules.portfolio import build_target_weights
        from modules.v2_strategy import build_target_weights_v2
        assert build_target_weights is not build_target_weights_v2

    def test_v1_yaml_entries_still_present(self):
        from modules.versioning import get_model_registry, get_portfolio_registry
        assert "quant_baseline_v1" in get_model_registry()
        assert "risk_parity_pyramid_v1" in get_portfolio_registry()


# ---------------------------------------------------------------------------
# B. Order ID isolation
# ---------------------------------------------------------------------------

class TestOrderIdIsolation:

    def test_v2_names_do_not_overlap_with_v1(self):
        from modules.v2_strategy import V2_STRATEGY_IDS
        from modules.scoring import STRATEGIES as V1
        assert set(V2_STRATEGY_IDS) & set(V1.keys()) == set()

    def test_all_v2_names_end_with_v2(self):
        from modules.v2_strategy import V2_STRATEGY_IDS
        for name in V2_STRATEGY_IDS:
            assert name.endswith("_V2"), f"{name!r} must end with _V2"

    def test_v2_strategy_ids_frozenset_matches_dict(self):
        from modules.v2_strategy import V2_STRATEGY_IDS, V2_STRATEGIES
        assert set(V2_STRATEGY_IDS) == set(V2_STRATEGIES.keys())

    def test_is_v2_strategy_false_for_v1_names(self):
        from modules.v2_strategy import is_v2_strategy
        from modules.scoring import STRATEGIES as V1
        for name in V1:
            assert not is_v2_strategy(name)

    def test_is_v2_strategy_true_for_v2_names(self):
        from modules.v2_strategy import is_v2_strategy, V2_STRATEGY_IDS
        for name in V2_STRATEGY_IDS:
            assert is_v2_strategy(name)


# ---------------------------------------------------------------------------
# C. Strategy uniqueness
# ---------------------------------------------------------------------------

class TestStrategyUniqueness:

    def test_strategy_ids_are_unique(self):
        from modules.v2_strategy import V2_STRATEGIES
        ids = [cfg["strategy_id"] for cfg in V2_STRATEGIES.values()]
        assert len(ids) == len(set(ids))

    def test_exactly_three_non_shadow_strategies(self):
        from modules.v2_strategy import V2_STRATEGIES
        ordinary = [n for n, c in V2_STRATEGIES.items() if not c.get("shadow_only")]
        assert set(ordinary) == {
            "Core_Quality_Momentum_V2",
            "Aggressive_Momentum_V2",
            "Defensive_Quality_V2",
        }

    def test_no_balanced_v2(self):
        from modules.v2_strategy import V2_STRATEGIES
        for name in V2_STRATEGIES:
            assert "Balanced" not in name

    def test_strategies_have_distinct_weights(self):
        from modules.v2_strategy import V2_STRATEGIES
        core = V2_STRATEGIES["Core_Quality_Momentum_V2"]["base_weights"]
        aggr = V2_STRATEGIES["Aggressive_Momentum_V2"]["base_weights"]
        defn = V2_STRATEGIES["Defensive_Quality_V2"]["base_weights"]
        assert core != aggr
        assert core != defn
        assert aggr != defn
        # Ordering invariants
        assert aggr["momentum"] > core["momentum"] > defn["momentum"]
        assert defn["safety"] > core["safety"] > aggr["safety"]

    def test_factor_weights_sum_to_one(self):
        from modules.v2_strategy import V2_STRATEGIES
        for name, cfg in V2_STRATEGIES.items():
            total = sum(cfg["base_weights"].values())
            assert abs(total - 1.0) < 1e-9, f"{name}: weights sum to {total}"

    def test_shadow_strategies_marked_shadow_only(self):
        from modules.v2_strategy import V2_STRATEGIES
        for name in ("Factor_Only_Core_V2", "Factor_Plus_Claude_Shadow_V2"):
            assert V2_STRATEGIES[name]["shadow_only"] is True

    def test_claude_shadow_strategy_flagged(self):
        from modules.v2_strategy import V2_STRATEGIES
        assert V2_STRATEGIES["Factor_Plus_Claude_Shadow_V2"]["claude_shadow"] is True
        assert V2_STRATEGIES["Factor_Only_Core_V2"]["claude_shadow"] is False


# ---------------------------------------------------------------------------
# D. Missing feature handling
# ---------------------------------------------------------------------------

class TestMissingFeatureHandling:

    def test_empty_fundamentals_quality_returns_null_score(self):
        from modules.v2_strategy import build_quality_factor
        result = build_quality_factor({"AAPL": {}, "MSFT": {}})
        for _, row in result.iterrows():
            assert row["quality_available"] is False or row["quality_available"] == False
            assert _is_null(row.get("quality_score"))

    def test_empty_fundamentals_value_returns_null_score(self):
        from modules.v2_strategy import build_value_factor
        result = build_value_factor({"AAPL": {}, "MSFT": {}})
        for _, row in result.iterrows():
            assert row["value_available"] is False or row["value_available"] == False

    def test_partial_fundamentals_correct_coverage(self):
        from modules.v2_strategy import build_quality_factor
        result = build_quality_factor({"TICK": {"returnOnEquity": 0.15}})
        row = result.iloc[0]
        assert row["quality_roe_available"] is True or row["quality_roe_available"] == True
        assert row["quality_gross_margin_available"] is False or row["quality_gross_margin_available"] == False
        assert abs(row["quality_coverage"] - 0.20) < 0.01

    def test_all_missing_quality_coverage_is_zero(self):
        from modules.v2_strategy import build_quality_factor
        result = build_quality_factor({"X": {}})
        assert result.iloc[0]["quality_coverage"] == 0.0

    def test_insufficient_price_history_momentum_unavailable(self):
        from modules.v2_strategy import build_momentum_factor
        df = _make_price_df(10)
        result = build_momentum_factor({"TICK": df}, as_of=df.index[-1])
        row = result.iloc[0]
        assert row["momentum_available"] is False or row["momentum_available"] == False
        assert _is_null(row.get("momentum_score"))


# ---------------------------------------------------------------------------
# E. Factor coverage computation
# ---------------------------------------------------------------------------

class TestFactorCoverage:

    def _full_info(self):
        return {
            "returnOnEquity": 0.20, "grossMargins": 0.40, "debtToEquity": 0.5,
            "earningsGrowth": 0.10, "freeCashflow": 500_000_000,
            "totalRevenue": 5_000_000_000, "marketCap": 10_000_000_000,
            "forwardPE": 20.0, "enterpriseValue": 12_000_000_000, "ebitda": 1_000_000_000,
        }

    def test_all_factors_available_gives_coverage_one(self):
        from modules.v2_strategy import (
            build_momentum_factor, build_quality_factor,
            build_value_factor, build_safety_factor, build_v2_factor_scores,
        )
        df = _make_price_df(300)
        as_of = df.index[-1]
        tickers = ["AAPL", "MSFT", "GOOG"]
        price_data = {t: df.copy() for t in tickers}
        fundamentals = {t: self._full_info() for t in tickers}

        combined = build_v2_factor_scores(
            build_momentum_factor(price_data, as_of),
            build_quality_factor(fundamentals),
            build_value_factor(fundamentals),
            build_safety_factor(price_data, as_of),
        )
        for _, row in combined.iterrows():
            assert row["factor_coverage"] == 1.0, f"Got {row['factor_coverage']}"

    def test_no_fundamentals_gives_coverage_half(self):
        from modules.v2_strategy import (
            build_momentum_factor, build_quality_factor,
            build_value_factor, build_safety_factor, build_v2_factor_scores,
        )
        df = _make_price_df(300)
        as_of = df.index[-1]
        tickers = ["A", "B"]
        price_data = {t: df.copy() for t in tickers}
        fundamentals = {t: {} for t in tickers}

        combined = build_v2_factor_scores(
            build_momentum_factor(price_data, as_of),
            build_quality_factor(fundamentals),
            build_value_factor(fundamentals),
            build_safety_factor(price_data, as_of),
        )
        for _, row in combined.iterrows():
            assert abs(row["factor_coverage"] - 0.5) < 0.01, f"Got {row['factor_coverage']}"


# ---------------------------------------------------------------------------
# F. Capped allocation
# ---------------------------------------------------------------------------

class TestCappedAllocation:

    def test_single_candidate_capped(self):
        from modules.v2_strategy import build_target_weights_v2
        result = build_target_weights_v2({"AAPL": 80.0}, max_position_weight=0.12)
        assert "AAPL" in result
        assert result["AAPL"] <= 0.12 + 1e-9

    def test_two_equal_candidates_both_capped(self):
        from modules.v2_strategy import build_target_weights_v2
        result = build_target_weights_v2({"A": 80.0, "B": 80.0}, max_position_weight=0.15)
        assert result["A"] <= 0.15 + 1e-9
        assert result["B"] <= 0.15 + 1e-9
        assert abs(sum(result.values()) - 0.30) < 1e-9

    def test_ten_candidates_surplus_becomes_cash(self):
        from modules.v2_strategy import build_target_weights_v2
        scores = {f"T{i}": 50.0 for i in range(10)}
        result = build_target_weights_v2(scores, max_position_weight=0.08)
        assert abs(sum(result.values()) - 0.80) < 1e-9
        for t, w in result.items():
            assert w <= 0.08 + 1e-9, f"{t}={w:.6f} exceeds cap"

    def test_skewed_scores_cap_never_exceeded(self):
        from modules.v2_strategy import build_target_weights_v2
        scores = {"TOP": 99.0, "A": 10.0, "B": 10.0, "C": 10.0, "D": 10.0}
        cap = 0.20
        result = build_target_weights_v2(scores, max_position_weight=cap)
        for t, w in result.items():
            assert w <= cap + 1e-9, f"{t}={w:.6f} exceeds cap {cap}"

    def test_exposure_applied_and_cap_still_respected(self):
        from modules.v2_strategy import build_target_weights_v2
        result = build_target_weights_v2({"A": 50.0, "B": 50.0}, max_position_weight=0.12, exposure=0.70)
        for w in result.values():
            assert w <= 0.12 + 1e-9

    def test_three_candidates_correct_total(self):
        from modules.v2_strategy import build_target_weights_v2
        result = build_target_weights_v2({"X": 70.0, "Y": 70.0, "Z": 70.0}, max_position_weight=0.12)
        assert abs(sum(result.values()) - 0.36) < 1e-9
        for w in result.values():
            assert w <= 0.12 + 1e-9

    def test_empty_returns_empty(self):
        from modules.v2_strategy import build_target_weights_v2
        assert build_target_weights_v2({}, max_position_weight=0.12) == {}


# ---------------------------------------------------------------------------
# G. No look-ahead
# ---------------------------------------------------------------------------

class TestNoLookahead:

    def test_momentum_unavailable_with_insufficient_history(self):
        from modules.v2_strategy import build_momentum_factor
        df = _make_price_df(10)
        result = build_momentum_factor({"TICK": df}, as_of=df.index[-1])
        assert result.iloc[0]["momentum_available"] == False

    def test_momentum_respects_as_of_cutoff(self):
        from modules.v2_strategy import build_momentum_factor
        df = _make_price_df(300)
        # Using first 20 rows as as_of — should be unavailable
        result = build_momentum_factor({"TICK": df}, as_of=df.index[19])
        assert result.iloc[0]["momentum_available"] == False

    def test_safety_respects_as_of_cutoff(self):
        from modules.v2_strategy import build_safety_factor
        df = _make_price_df(300)
        result_early = build_safety_factor({"TICK": df}, as_of=df.index[5])
        # Only 5 days — should be unavailable (< 30 required)
        assert result_early.iloc[0]["safety_available"] == False

    def test_friday_next_session_is_not_friday(self):
        from modules.exchange_calendar import intended_execution_session
        next_sess = intended_execution_session("2026-08-07")  # Friday
        next_ts = pd.Timestamp(next_sess)
        assert next_ts.weekday() != 4  # next session is Monday, not Friday
        assert next_ts > pd.Timestamp("2026-08-07")

    def test_holiday_next_session_is_after_holiday(self):
        from modules.exchange_calendar import intended_execution_session
        next_sess = intended_execution_session("2025-07-04")  # Independence Day
        assert pd.Timestamp(next_sess) > pd.Timestamp("2025-07-04")

    def test_monday_next_session_is_different_calendar_date(self):
        from modules.exchange_calendar import nth_session_after
        next_sess = nth_session_after("2026-08-10", 1)
        assert pd.Timestamp(next_sess).date() > pd.Timestamp("2026-08-10").date()


# ---------------------------------------------------------------------------
# H. Claude shadow blocked from order creation
# ---------------------------------------------------------------------------

class TestClaudeShadowBlocked:

    def test_shadow_unavailable_always_blocked(self):
        from modules.v2_claude_shadow import shadow_unavailable
        assert shadow_unavailable()["order_creation_blocked"] is True

    def test_validated_output_always_blocked(self):
        from modules.v2_claude_shadow import validate_claude_shadow_output
        raw = {
            "signal_direction": "bullish", "evidence_strength": "strong",
            "guidance_change": "raised", "estimate_revision_direction": "upward",
            "margin_trend": "improving", "earnings_quality": "high",
            "capital_allocation_quality": "high", "thesis_risks": ["macro"],
            "catalyst_strength": "strong", "uncertainty": "low",
            "source_ids": ["earnings-q4"], "source_published_at": ["2026-01-30"],
            "model_id": "claude-sonnet-5", "prompt_version": "shadow_v1",
            "generated_at": "2026-08-10T12:00:00Z", "data_cutoff_at": "2026-08-10T09:00:00Z",
        }
        result = validate_claude_shadow_output(raw)
        assert result["order_creation_blocked"] is True

    def test_assert_raises_when_not_blocked(self):
        from modules.v2_claude_shadow import assert_shadow_cannot_create_order
        with pytest.raises(RuntimeError, match="order_creation_blocked"):
            assert_shadow_cannot_create_order({"order_creation_blocked": False})

    def test_assert_passes_when_blocked(self):
        from modules.v2_claude_shadow import assert_shadow_cannot_create_order
        assert_shadow_cannot_create_order({"order_creation_blocked": True})

    def test_none_input_becomes_unavailable(self):
        from modules.v2_claude_shadow import validate_claude_shadow_output
        result = validate_claude_shadow_output(None)
        assert result["signal_direction"] == "unavailable"
        assert result["order_creation_blocked"] is True

    def test_unknown_enum_becomes_unavailable(self):
        from modules.v2_claude_shadow import validate_claude_shadow_output
        result = validate_claude_shadow_output({"signal_direction": "mega_bullish"})
        assert result["signal_direction"] == "unavailable"

    def test_shadow_content_hash_deterministic(self):
        from modules.v2_claude_shadow import validate_claude_shadow_output
        raw = {"signal_direction": "neutral", "evidence_strength": "weak"}
        r1 = validate_claude_shadow_output(raw)
        r2 = validate_claude_shadow_output(raw)
        assert r1["content_hash"] == r2["content_hash"]


# ---------------------------------------------------------------------------
# I. Evaluation metrics
# ---------------------------------------------------------------------------

class TestEvaluationMetrics:

    def _returns(self, n=300, mu=0.0003, sigma=0.01, seed=42):
        rng = np.random.default_rng(seed)
        return pd.Series(rng.normal(mu, sigma, n))

    def test_compute_metrics_basic_fields(self):
        from modules.v2_evaluation import compute_metrics
        r = self._returns()
        m = compute_metrics(r, {"SPY": self._returns(mu=0.0002), "QQQ": self._returns(mu=0.00025)},
                            "Core_V2", "in_sample")
        assert m.sharpe_ratio is not None
        assert m.sortino_ratio is not None
        assert m.max_drawdown is not None and m.max_drawdown < 0
        assert m.annualized_return is not None
        assert m.annualized_volatility is not None and m.annualized_volatility > 0

    def test_insufficient_data_gives_false_evidence(self):
        from modules.v2_evaluation import compute_metrics
        r = self._returns(n=50)
        m = compute_metrics(r, {}, "test", "validation")
        assert m.sufficient_evidence is False
        assert m.evidence_note

    def test_compare_returns_sorted_dataframe(self):
        from modules.v2_evaluation import compute_metrics, compare_strategies
        m1 = compute_metrics(self._returns(mu=0.0005), {}, "High", "in_sample")
        m2 = compute_metrics(self._returns(mu=0.0001), {}, "Low", "in_sample")
        df = compare_strategies([m1, m2])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df.iloc[0]["strategy"] == "High"

    def test_factor_coverage_stored_in_metrics(self):
        from modules.v2_evaluation import compute_metrics
        m = compute_metrics(self._returns(), {}, "Core_V2", "in_sample", factor_coverage=0.75)
        assert m.factor_coverage == 0.75

    def test_survivorship_bias_note_always_present(self):
        from modules.v2_evaluation import EvaluationMetrics
        m = EvaluationMetrics(strategy_name="test", period="in_sample")
        assert m.survivorship_bias_note
        assert "survivorship" in m.survivorship_bias_note.lower()

    def test_point_in_time_fundamentals_default_false(self):
        from modules.v2_evaluation import EvaluationMetrics
        m = EvaluationMetrics(strategy_name="test", period="in_sample")
        assert m.point_in_time_fundamentals is False


# ---------------------------------------------------------------------------
# J. Backtest integrity
# ---------------------------------------------------------------------------

class TestBacktestIntegrity:

    def test_three_ordinary_strategies_have_distinct_score_columns(self):
        from modules.v2_strategy import V2_STRATEGIES
        ordinary = {n: c for n, c in V2_STRATEGIES.items() if not c.get("shadow_only")}
        cols = [c["score_column"] for c in ordinary.values()]
        assert len(cols) == len(set(cols))

    def test_aggressive_has_highest_momentum_weight(self):
        from modules.v2_strategy import V2_STRATEGIES
        ordinary = {n: c for n, c in V2_STRATEGIES.items() if not c.get("shadow_only")}
        max_name = max(ordinary, key=lambda n: ordinary[n]["base_weights"]["momentum"])
        assert max_name == "Aggressive_Momentum_V2"

    def test_defensive_has_highest_safety_weight(self):
        from modules.v2_strategy import V2_STRATEGIES
        ordinary = {n: c for n, c in V2_STRATEGIES.items() if not c.get("shadow_only")}
        max_name = max(ordinary, key=lambda n: ordinary[n]["base_weights"]["safety"])
        assert max_name == "Defensive_Quality_V2"

    def test_missing_fundamentals_null_not_fifty(self):
        from modules.v2_strategy import build_quality_factor, build_value_factor
        qual = build_quality_factor({"NOSCORE": {}})
        val = build_value_factor({"NOSCORE": {}})
        assert _is_null(qual.iloc[0].get("quality_score"))
        assert _is_null(val.iloc[0].get("value_score"))


# ---------------------------------------------------------------------------
# K. Evaluation period definitions
# ---------------------------------------------------------------------------

class TestEvaluationPeriods:

    def test_oos_is_locked(self):
        from modules.v2_evaluation import EVALUATION_PERIODS
        assert EVALUATION_PERIODS["out_of_sample"].get("locked") is True

    def test_in_sample_ends_before_validation(self):
        from modules.v2_evaluation import EVALUATION_PERIODS
        assert (pd.Timestamp(EVALUATION_PERIODS["in_sample"]["end"])
                < pd.Timestamp(EVALUATION_PERIODS["validation"]["start"]))

    def test_validation_ends_before_oos(self):
        from modules.v2_evaluation import EVALUATION_PERIODS
        assert (pd.Timestamp(EVALUATION_PERIODS["validation"]["end"])
                < pd.Timestamp(EVALUATION_PERIODS["out_of_sample"]["start"]))

    def test_benchmarks_include_spy_and_qqq(self):
        from modules.v2_evaluation import BENCHMARKS
        assert "SPY" in BENCHMARKS
        assert "QQQ" in BENCHMARKS

    def test_promotion_requires_oos_period(self):
        from modules.v2_evaluation import PROMOTION_CRITERIA
        assert PROMOTION_CRITERIA["min_out_of_sample_months"] >= 3


# ---------------------------------------------------------------------------
# L. V2 version constants and registry
# ---------------------------------------------------------------------------

class TestV2VersionConstants:

    def test_v2_model_version_distinct_from_v1(self):
        from modules.v2_strategy import V2_MODEL_VERSION
        from modules.versioning import MODEL_VERSION
        assert V2_MODEL_VERSION != MODEL_VERSION
        assert "v2" in V2_MODEL_VERSION.lower()

    def test_v2_portfolio_version_distinct_from_v1(self):
        from modules.v2_strategy import V2_PORTFOLIO_VERSION
        from modules.versioning import PORTFOLIO_VERSION
        assert V2_PORTFOLIO_VERSION != PORTFOLIO_VERSION
        assert "v2" in V2_PORTFOLIO_VERSION.lower()

    def test_v2_entries_present_in_yaml_registries(self):
        from modules.versioning import get_model_registry, get_portfolio_registry
        from modules.v2_strategy import V2_MODEL_VERSION, V2_PORTFOLIO_VERSION
        assert V2_MODEL_VERSION in get_model_registry()
        assert V2_PORTFOLIO_VERSION in get_portfolio_registry()
