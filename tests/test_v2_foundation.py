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
# K. Period definitions (4-period structure)
# ---------------------------------------------------------------------------

class TestPeriodDefinitions:

    def test_four_periods_defined(self):
        from modules.v2_evaluation import EVALUATION_PERIODS
        expected = {"in_sample", "validation", "retrospective_holdout", "prospective_out_of_sample"}
        assert set(EVALUATION_PERIODS.keys()) == expected

    def test_prospective_oos_is_locked(self):
        from modules.v2_evaluation import EVALUATION_PERIODS
        assert EVALUATION_PERIODS["prospective_out_of_sample"].get("locked") is True

    def test_retrospective_holdout_is_locked(self):
        from modules.v2_evaluation import EVALUATION_PERIODS
        assert EVALUATION_PERIODS["retrospective_holdout"].get("locked") is True

    def test_in_sample_ends_before_validation(self):
        from modules.v2_evaluation import EVALUATION_PERIODS
        assert (EVALUATION_PERIODS["in_sample"]["end"]
                < EVALUATION_PERIODS["validation"]["start"])

    def test_validation_ends_before_retrospective(self):
        from modules.v2_evaluation import EVALUATION_PERIODS
        assert (EVALUATION_PERIODS["validation"]["end"]
                < EVALUATION_PERIODS["retrospective_holdout"]["start"])

    def test_retrospective_ends_before_prospective(self):
        from modules.v2_evaluation import EVALUATION_PERIODS
        assert (EVALUATION_PERIODS["retrospective_holdout"]["end"]
                < EVALUATION_PERIODS["prospective_out_of_sample"]["start"])

    def test_prospective_oos_starts_after_registration(self):
        from modules.v2_evaluation import EVALUATION_PERIODS, MODEL_REGISTRATION_DATE
        assert EVALUATION_PERIODS["prospective_out_of_sample"]["start"] > MODEL_REGISTRATION_DATE

    def test_prospective_oos_end_is_none(self):
        from modules.v2_evaluation import EVALUATION_PERIODS
        assert EVALUATION_PERIODS["prospective_out_of_sample"]["end"] is None

    def test_model_registration_date_is_2026_08_10(self):
        from modules.v2_evaluation import MODEL_REGISTRATION_DATE
        assert MODEL_REGISTRATION_DATE == pd.Timestamp("2026-08-10")

    def test_get_prospective_oos_months_before_lock(self):
        from modules.v2_evaluation import get_prospective_oos_months
        months = get_prospective_oos_months(as_of=pd.Timestamp("2026-08-05"))
        assert months == 0

    def test_get_prospective_oos_months_after_lock(self):
        from modules.v2_evaluation import get_prospective_oos_months
        months = get_prospective_oos_months(as_of=pd.Timestamp("2027-02-11"))
        assert months >= 6

    def test_benchmarks_include_spy_and_qqq(self):
        from modules.v2_evaluation import BENCHMARKS
        assert "SPY" in BENCHMARKS
        assert "QQQ" in BENCHMARKS

    def test_promotion_criteria_key_is_prospective_oos_months(self):
        from modules.v2_evaluation import PROMOTION_CRITERIA
        assert "min_prospective_oos_months" in PROMOTION_CRITERIA
        assert PROMOTION_CRITERIA["min_prospective_oos_months"] >= 6


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


# ---------------------------------------------------------------------------
# M. Config / code parity
# ---------------------------------------------------------------------------

class TestConfigCodeParity:

    def test_safety_beta_data_source_mentions_spy(self):
        from modules.versioning import get_model_registry
        safety = get_model_registry()["quant_baseline_v2"]["config"]["factors"]["safety"]
        assert "SPY" in safety["beta_data_source"]

    def test_safety_min_overlap_days_matches_code(self):
        from modules.versioning import get_model_registry
        safety = get_model_registry()["quant_baseline_v2"]["config"]["factors"]["safety"]
        # Code: len(common_idx) >= 60 in build_safety_factor
        assert safety["min_overlap_days"] == 60

    def test_value_sector_min_group_size_matches_code(self):
        from modules.versioning import get_model_registry
        from modules.v2_strategy import _SECTOR_MIN_GROUP_SIZE
        value = get_model_registry()["quant_baseline_v2"]["config"]["factors"]["value"]
        assert value["sector_min_group_size"] == _SECTOR_MIN_GROUP_SIZE

    def test_value_sector_fallback_mentions_cross_sectional(self):
        from modules.versioning import get_model_registry
        value = get_model_registry()["quant_baseline_v2"]["config"]["factors"]["value"]
        assert "cross_sectional" in value["sector_fallback"]


# ---------------------------------------------------------------------------
# N. NaN / inf / non-numeric handling
# ---------------------------------------------------------------------------

class TestNaNInfHandling:

    def test_nan_roe_gives_quality_unavailable(self):
        from modules.v2_strategy import build_quality_factor
        import math
        result = build_quality_factor({"TICK": {"returnOnEquity": float("nan")}})
        row = result.iloc[0]
        assert row["quality_roe_available"] is False or row["quality_roe_available"] == False

    def test_inf_roe_gives_quality_unavailable(self):
        from modules.v2_strategy import build_quality_factor
        result = build_quality_factor({"TICK": {"returnOnEquity": float("inf")}})
        row = result.iloc[0]
        assert row["quality_roe_available"] is False or row["quality_roe_available"] == False

    def test_neg_inf_gives_quality_unavailable(self):
        from modules.v2_strategy import build_quality_factor
        result = build_quality_factor({"TICK": {"returnOnEquity": float("-inf")}})
        row = result.iloc[0]
        assert row["quality_roe_available"] is False or row["quality_roe_available"] == False

    def test_non_numeric_string_gives_quality_unavailable(self):
        from modules.v2_strategy import build_quality_factor
        result = build_quality_factor({"TICK": {"returnOnEquity": "N/A"}})
        row = result.iloc[0]
        assert row["quality_roe_available"] is False or row["quality_roe_available"] == False

    def test_nan_mktcap_gives_value_fcf_yield_unavailable(self):
        from modules.v2_strategy import build_value_factor
        result = build_value_factor({"TICK": {
            "freeCashflow": 1_000_000, "marketCap": float("nan")
        }})
        row = result.iloc[0]
        assert row["value_fcf_yield_available"] is False or row["value_fcf_yield_available"] == False

    def test_inf_fpe_gives_earnings_yield_unavailable(self):
        from modules.v2_strategy import build_value_factor
        result = build_value_factor({"TICK": {"forwardPE": float("inf")}})
        row = result.iloc[0]
        assert row["value_earnings_yield_available"] is False or row["value_earnings_yield_available"] == False

    def test_all_nan_gives_null_composite_not_fifty(self):
        from modules.v2_strategy import build_quality_factor
        result = build_quality_factor({"TICK": {
            "returnOnEquity": float("nan"),
            "grossMargins": float("nan"),
            "debtToEquity": float("nan"),
            "earningsGrowth": float("nan"),
            "freeCashflow": float("nan"),
            "totalRevenue": float("nan"),
        }})
        row = result.iloc[0]
        assert row["quality_available"] == False
        assert _is_null(row.get("quality_score"))


# ---------------------------------------------------------------------------
# O. Empty universe
# ---------------------------------------------------------------------------

class TestEmptyUniverse:

    def test_all_empty_inputs_returns_stable_schema(self):
        from modules.v2_strategy import build_v2_factor_scores, EMPTY_SCORE_SCHEMA
        result = build_v2_factor_scores(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == EMPTY_SCORE_SCHEMA
        assert len(result) == 0

    def test_empty_price_data_dict_returns_empty_momentum(self):
        from modules.v2_strategy import build_momentum_factor
        result = build_momentum_factor({}, as_of=pd.Timestamp("2026-01-01"))
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_empty_fundamentals_dict_returns_empty_quality(self):
        from modules.v2_strategy import build_quality_factor
        result = build_quality_factor({})
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# P. Timezone handling
# ---------------------------------------------------------------------------

class TestTimezone:

    def test_tz_aware_as_of_with_naive_momentum_no_type_error(self):
        from modules.v2_strategy import build_momentum_factor
        df = _make_price_df(300)  # naive DatetimeIndex
        as_of = pd.Timestamp("2022-12-31", tz="UTC")
        # Must not raise TypeError comparing tz-aware vs tz-naive
        result = build_momentum_factor({"TICK": df}, as_of=as_of)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_tz_aware_as_of_with_naive_safety_no_type_error(self):
        from modules.v2_strategy import build_safety_factor
        df = _make_price_df(300)
        as_of = pd.Timestamp("2022-12-31", tz="US/Eastern")
        result = build_safety_factor({"TICK": df}, as_of=as_of)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_tz_aware_as_of_momentum_correct_cutoff(self):
        from modules.v2_strategy import build_momentum_factor
        df = _make_price_df(300)
        # as_of pointing to the last date in the series — data should be available
        last = df.index[-1]
        as_of_utc = pd.Timestamp(last).tz_localize("UTC")
        result = build_momentum_factor({"TICK": df}, as_of=as_of_utc)
        assert result.iloc[0]["momentum_available"] == True


# ---------------------------------------------------------------------------
# Q. Momentum history threshold (252 days)
# ---------------------------------------------------------------------------

class TestMomentumHistoryThreshold:

    def test_exactly_252_days_gives_available(self):
        from modules.v2_strategy import build_momentum_factor
        df = _make_price_df(252)
        result = build_momentum_factor({"TICK": df}, as_of=df.index[-1])
        assert result.iloc[0]["momentum_available"] == True

    def test_251_days_gives_unavailable(self):
        from modules.v2_strategy import build_momentum_factor
        df = _make_price_df(251)
        result = build_momentum_factor({"TICK": df}, as_of=df.index[-1])
        assert result.iloc[0]["momentum_available"] == False

    def test_300_days_gives_available(self):
        from modules.v2_strategy import build_momentum_factor
        df = _make_price_df(300)
        result = build_momentum_factor({"TICK": df}, as_of=df.index[-1])
        assert result.iloc[0]["momentum_available"] == True


# ---------------------------------------------------------------------------
# R. Promotion criteria (real SPY comparison)
# ---------------------------------------------------------------------------

class TestCheckPromotionCriteria:

    def _make_metrics(self, sharpe=0.80, drawdown=-0.15, alpha_frac=0.60,
                      strategy_name="Core_V2", period="prospective_out_of_sample"):
        from modules.v2_evaluation import EvaluationMetrics
        m = EvaluationMetrics(strategy_name=strategy_name, period=period)
        m.sharpe_ratio = sharpe
        m.max_drawdown = drawdown
        m._positive_alpha_months_fraction = alpha_frac
        m.factor_coverage = 0.85
        m.survivorship_bias_note = "survivorship bias present — current universe only"
        m.sufficient_evidence = True
        return m

    def test_all_criteria_pass_with_strong_strategy(self):
        from modules.v2_evaluation import check_promotion_criteria
        strat = self._make_metrics(sharpe=1.20, drawdown=-0.12, alpha_frac=0.62)
        spy = self._make_metrics(sharpe=0.90, drawdown=-0.20, strategy_name="SPY")
        # NOT promoting (backtest_not_blocking_promotion always False in V2A)
        results = check_promotion_criteria(strat, spy, oos_months=8, min_factor_coverage=0.6)
        assert results["min_prospective_oos_months"] is True
        assert results["min_sharpe_above_spy"] is True
        assert results["max_drawdown_ratio_vs_spy"] is True
        assert results["min_positive_alpha_months_fraction"] is True
        assert results["must_report_survivorship_bias"] is True
        assert results["must_report_factor_coverage"] is True
        # Promotion still blocked due to backtest constraint
        assert results["backtest_not_blocking_promotion"] is False

    def test_insufficient_oos_months_fails(self):
        from modules.v2_evaluation import check_promotion_criteria
        strat = self._make_metrics(sharpe=1.20)
        spy = self._make_metrics(sharpe=0.90, strategy_name="SPY")
        results = check_promotion_criteria(strat, spy, oos_months=3)
        assert results["min_prospective_oos_months"] is False

    def test_sharpe_too_close_to_spy_fails(self):
        from modules.v2_evaluation import check_promotion_criteria
        strat = self._make_metrics(sharpe=0.95)
        spy = self._make_metrics(sharpe=0.90, strategy_name="SPY")
        # Difference is only 0.05, below threshold of 0.10
        results = check_promotion_criteria(strat, spy, oos_months=8)
        assert results["min_sharpe_above_spy"] is False

    def test_worse_drawdown_than_spy_fails(self):
        from modules.v2_evaluation import check_promotion_criteria
        strat = self._make_metrics(drawdown=-0.35)
        spy = self._make_metrics(drawdown=-0.20, strategy_name="SPY")
        # abs(strat_dd) / abs(spy_dd) = 0.35/0.20 = 1.75 > 1.25
        results = check_promotion_criteria(strat, spy, oos_months=8)
        assert results["max_drawdown_ratio_vs_spy"] is False

    def test_low_alpha_fraction_fails(self):
        from modules.v2_evaluation import check_promotion_criteria
        strat = self._make_metrics(sharpe=1.20, alpha_frac=0.45)
        spy = self._make_metrics(sharpe=0.90, strategy_name="SPY")
        results = check_promotion_criteria(strat, spy, oos_months=8)
        assert results["min_positive_alpha_months_fraction"] is False

    def test_missing_sharpe_fails_sharpe_criterion(self):
        from modules.v2_evaluation import check_promotion_criteria
        strat = self._make_metrics()
        strat.sharpe_ratio = None
        spy = self._make_metrics(strategy_name="SPY")
        results = check_promotion_criteria(strat, spy, oos_months=8)
        assert results["min_sharpe_above_spy"] is False


# ---------------------------------------------------------------------------
# S. Backtest status
# ---------------------------------------------------------------------------

class TestBacktestStatus:

    def test_full_four_factor_backtest_unsupported(self):
        from modules.v2_evaluation import BACKTEST_STATUS
        assert BACKTEST_STATUS["full_four_factor"] == "unsupported"

    def test_momentum_safety_backtest_supported(self):
        from modules.v2_evaluation import BACKTEST_STATUS
        assert BACKTEST_STATUS["momentum_and_safety_only"] == "supported"

    def test_promotion_blocked(self):
        from modules.v2_evaluation import BACKTEST_STATUS
        assert BACKTEST_STATUS["promotion_blocked"] is True

    def test_run_v2_backtest_raises_not_implemented(self):
        from modules.v2_evaluation import run_v2_backtest
        with pytest.raises(NotImplementedError):
            run_v2_backtest()

    def test_run_v2_backtest_with_args_raises_not_implemented(self):
        from modules.v2_evaluation import run_v2_backtest
        with pytest.raises(NotImplementedError):
            run_v2_backtest("some_arg", strategy="Core_V2")


# ---------------------------------------------------------------------------
# T. Hardened Claude shadow provenance
# ---------------------------------------------------------------------------

class TestClaudeShadowHardened:

    def _valid_raw(self):
        return {
            "signal_direction": "bullish",
            "evidence_strength": "strong",
            "guidance_change": "raised",
            "estimate_revision_direction": "upward",
            "margin_trend": "improving",
            "earnings_quality": "high",
            "capital_allocation_quality": "high",
            "thesis_risks": ["macro risk"],
            "catalyst_strength": "strong",
            "uncertainty": "low",
            "source_ids": ["s1", "s2"],
            "source_published_at": ["2026-08-01T09:00:00+00:00", "2026-08-05T14:30:00+00:00"],
            "model_id": "claude-sonnet-5",
            "prompt_version": "shadow_v1",
            "generated_at": "2026-08-10T12:00:00+00:00",
            "data_cutoff_at": "2026-08-09T16:00:00+00:00",
        }

    def test_valid_provenance_gives_provenance_valid_true(self):
        from modules.v2_claude_shadow import validate_claude_shadow_output
        result = validate_claude_shadow_output(self._valid_raw())
        assert result["provenance_valid"] is True
        assert result["provenance_failures"] == []

    def test_missing_generated_at_fails_provenance(self):
        from modules.v2_claude_shadow import validate_claude_shadow_output
        raw = self._valid_raw()
        raw.pop("generated_at")
        result = validate_claude_shadow_output(raw)
        assert result["provenance_valid"] is False
        assert any("generated_at" in f for f in result["provenance_failures"])

    def test_missing_timezone_on_generated_at_fails(self):
        from modules.v2_claude_shadow import validate_claude_shadow_output
        raw = self._valid_raw()
        raw["generated_at"] = "2026-08-10T12:00:00"  # no timezone
        result = validate_claude_shadow_output(raw)
        assert result["provenance_valid"] is False
        assert any("generated_at" in f for f in result["provenance_failures"])

    def test_generated_at_before_cutoff_fails(self):
        from modules.v2_claude_shadow import validate_claude_shadow_output
        raw = self._valid_raw()
        raw["generated_at"] = "2026-08-08T00:00:00+00:00"  # before data_cutoff_at
        raw["data_cutoff_at"] = "2026-08-09T16:00:00+00:00"
        result = validate_claude_shadow_output(raw)
        assert result["provenance_valid"] is False
        assert any("before data_cutoff_at" in f for f in result["provenance_failures"])

    def test_parallel_list_length_mismatch_fails(self):
        from modules.v2_claude_shadow import validate_claude_shadow_output
        raw = self._valid_raw()
        raw["source_ids"] = ["s1", "s2", "s3"]  # 3 ids but 2 timestamps
        result = validate_claude_shadow_output(raw)
        assert result["provenance_valid"] is False
        assert any("source_ids length" in f for f in result["provenance_failures"])

    def test_source_published_after_cutoff_fails(self):
        from modules.v2_claude_shadow import validate_claude_shadow_output
        raw = self._valid_raw()
        # source published AFTER data_cutoff_at
        raw["source_published_at"] = ["2026-08-10T20:00:00+00:00", "2026-08-05T14:30:00+00:00"]
        raw["data_cutoff_at"] = "2026-08-09T16:00:00+00:00"
        result = validate_claude_shadow_output(raw)
        assert result["provenance_valid"] is False
        assert any("after data_cutoff_at" in f for f in result["provenance_failures"])

    def test_source_published_missing_timezone_fails(self):
        from modules.v2_claude_shadow import validate_claude_shadow_output
        raw = self._valid_raw()
        raw["source_published_at"] = ["2026-08-01T09:00:00", "2026-08-05T14:30:00+00:00"]
        result = validate_claude_shadow_output(raw)
        assert result["provenance_valid"] is False
        assert any("source_published_at[0]" in f for f in result["provenance_failures"])

    def test_shadow_unavailable_has_provenance_fields(self):
        from modules.v2_claude_shadow import shadow_unavailable
        result = shadow_unavailable("not_requested")
        assert "provenance_valid" in result
        assert "provenance_failures" in result
        assert result["provenance_valid"] is False
        assert isinstance(result["provenance_failures"], list)

    def test_schema_dict_exported(self):
        from modules.v2_claude_shadow import CLAUDE_SHADOW_JSON_SCHEMA
        assert isinstance(CLAUDE_SHADOW_JSON_SCHEMA, dict)
        assert "properties" in CLAUDE_SHADOW_JSON_SCHEMA
        assert CLAUDE_SHADOW_JSON_SCHEMA["properties"]["order_creation_blocked"]["const"] is True


# ---------------------------------------------------------------------------
# U. Config hash validation
# ---------------------------------------------------------------------------

class TestConfigHashValidation:

    def test_v2_model_config_hash_validates(self):
        from modules.versioning import validate_model_config_hash
        # Should not raise
        validate_model_config_hash("quant_baseline_v2")

    def test_v2_portfolio_config_hash_validates(self):
        from modules.versioning import validate_portfolio_config_hash
        validate_portfolio_config_hash("risk_managed_factor_v2")

    def test_v1_model_config_hash_still_validates(self):
        from modules.versioning import validate_model_config_hash
        validate_model_config_hash("quant_baseline_v1")

    def test_v1_portfolio_config_hash_still_validates(self):
        from modules.versioning import validate_portfolio_config_hash
        validate_portfolio_config_hash("risk_parity_pyramid_v1")

    def test_v2_model_hash_is_not_placeholder(self):
        from modules.versioning import get_model_registry
        entry = get_model_registry()["quant_baseline_v2"]
        h = entry.get("model_config_hash", "")
        assert h not in ("", "placeholder", "PLACEHOLDER", None, "n/a")
        assert len(str(h)) == 64  # SHA-256 hex digest

    def test_v2_portfolio_hash_is_not_placeholder(self):
        from modules.versioning import get_portfolio_registry
        entry = get_portfolio_registry()["risk_managed_factor_v2"]
        h = entry.get("portfolio_config_hash", "")
        assert h not in ("", "placeholder", "PLACEHOLDER", None, "n/a")
        assert len(str(h)) == 64


# ---------------------------------------------------------------------------
# V. Sector-adjusted value factor (Round 3)
# ---------------------------------------------------------------------------

class TestSectorAdjustedValue:
    """Regression: sector adjustment must actually change value_raw, not just set a flag."""

    def _make_fundamentals(self, n: int = 8) -> dict:
        result = {}
        for i in range(n):
            result[f"T{i:02d}"] = {
                "freeCashflow": 1_000_000 * (i + 1),
                "marketCap": 20_000_000 * (i + 1),
                "forwardPE": 15.0 + i,
                "enterpriseValue": 25_000_000 * (i + 1),
                "ebitda": 2_000_000 * (i + 1),
            }
        return result

    def test_sector_adjustment_changes_value_raw(self):
        from modules.v2_strategy import build_value_factor
        fundamentals = self._make_fundamentals(8)
        sector_map = {f"T{i:02d}": "Technology" for i in range(8)}

        result_no = build_value_factor(fundamentals, sector_map=None)
        result_with = build_value_factor(fundamentals, sector_map=sector_map)

        no_raw = result_no.set_index("ticker")["value_raw"].dropna()
        with_raw = result_with.set_index("ticker")["value_raw"].dropna()
        common = no_raw.index.intersection(with_raw.index)
        diffs = (no_raw.loc[common] - with_raw.loc[common]).abs()
        assert diffs.max() > 1e-10, "Sector adjustment must change value_raw for at least one ticker"

    def test_sector_adjusted_flag_true_for_large_sector(self):
        from modules.v2_strategy import build_value_factor
        fundamentals = self._make_fundamentals(8)
        sector_map = {f"T{i:02d}": "Technology" for i in range(8)}
        result = build_value_factor(fundamentals, sector_map=sector_map)
        assert "value_sector_adjusted" in result.columns
        n_adjusted = int(result["value_sector_adjusted"].sum())
        assert n_adjusted >= 5, f"Expected >= 5 adjusted tickers, got {n_adjusted}"

    def test_small_sector_not_adjusted(self):
        from modules.v2_strategy import build_value_factor
        # Only 4 tickers in sector — below _SECTOR_MIN_GROUP_SIZE (5)
        fundamentals = self._make_fundamentals(4)
        sector_map = {f"T{i:02d}": "Technology" for i in range(4)}
        result = build_value_factor(fundamentals, sector_map=sector_map)
        assert int(result["value_sector_adjusted"].sum()) == 0, \
            "Sectors with fewer than 5 tickers must not be adjusted"

    def test_without_sector_map_all_flags_false(self):
        from modules.v2_strategy import build_value_factor
        fundamentals = self._make_fundamentals(8)
        result = build_value_factor(fundamentals, sector_map=None)
        assert "value_sector_adjusted" in result.columns
        assert int(result["value_sector_adjusted"].sum()) == 0

    def test_single_sector_lower_ev_ebitda_scores_higher(self):
        """Lower EV/EBITDA → higher ev_ebitda_inv → should score higher on value."""
        from modules.v2_strategy import build_value_factor
        # All same except enterpriseValue — T00 cheapest, T05 most expensive
        fundamentals = {}
        for i in range(6):
            fundamentals[f"T{i:02d}"] = {
                "freeCashflow": 1_000_000,
                "marketCap": 20_000_000,
                "forwardPE": 15.0,
                "enterpriseValue": 10_000_000 * (i + 1),
                "ebitda": 2_000_000,
            }
        sector_map = {f"T{i:02d}": "Technology" for i in range(6)}
        result = build_value_factor(fundamentals, sector_map=sector_map)
        scored = result.dropna(subset=["value_score"]).set_index("ticker")
        if "T00" in scored.index and "T05" in scored.index:
            assert scored.at["T00", "value_score"] > scored.at["T05", "value_score"], \
                "Lower EV/EBITDA ticker must have higher value_score"


# ---------------------------------------------------------------------------
# W. Timezone normalization (Round 3)
# ---------------------------------------------------------------------------

class TestTimezoneNormalization:
    """Regression: must tz_convert to UTC before stripping timezone."""

    def test_normalize_converts_et_to_utc_naive(self):
        from modules.v2_strategy import _normalize_df_index
        # 2024-01-15 16:00 America/New_York = 21:00 UTC (January = EST = UTC-5)
        idx = pd.DatetimeIndex(["2024-01-15 16:00:00"]).tz_localize("America/New_York")
        df = pd.DataFrame({"Close": [100.0]}, index=idx)
        result = _normalize_df_index(df)
        assert result.index.tz is None
        assert result.index[0].hour == 21, (
            f"16:00 ET must become 21:00 UTC-naive (EST=UTC-5), got hour={result.index[0].hour}"
        )

    def test_normalize_naive_index_unchanged(self):
        from modules.v2_strategy import _normalize_df_index
        idx = pd.DatetimeIndex(["2024-01-15 16:00:00"])
        df = pd.DataFrame({"Close": [100.0]}, index=idx)
        result = _normalize_df_index(df)
        assert result.index[0].hour == 16, "Naive index must not be modified"

    def test_utc_aware_cutoff_excludes_row_after_cutoff(self):
        """16:00 ET row (= 21:00 UTC-naive) must be excluded by a 14:00 UTC-naive cutoff."""
        from modules.v2_strategy import _normalize_df_index
        idx = pd.DatetimeIndex(["2024-01-15 16:00:00"]).tz_localize("America/New_York")
        df = pd.DataFrame({"Close": [100.0]}, index=idx)
        df_norm = _normalize_df_index(df)
        cutoff = pd.Timestamp("2024-01-15 14:00:00")
        filtered = df_norm[df_norm.index <= cutoff]
        assert len(filtered) == 0, \
            "Row at 21:00 UTC-naive must be excluded by 14:00 UTC-naive cutoff"

    def test_get_prospective_oos_months_utc_aware(self):
        from modules.v2_evaluation import get_prospective_oos_months
        as_of = pd.Timestamp("2027-02-11", tz="UTC")
        months = get_prospective_oos_months(as_of=as_of)
        assert months >= 6, f"Expected >= 6 months, got {months}"

    def test_get_prospective_oos_months_et_aware(self):
        from modules.v2_evaluation import get_prospective_oos_months
        as_of = pd.Timestamp("2027-02-11 12:00:00").tz_localize("America/New_York")
        months = get_prospective_oos_months(as_of=as_of)
        assert months >= 6, f"Expected >= 6 months, got {months}"

    def test_get_prospective_oos_months_before_window_tz_aware(self):
        from modules.v2_evaluation import get_prospective_oos_months
        # Day before prospective OOS window opens
        as_of = pd.Timestamp("2026-08-10", tz="UTC")
        assert get_prospective_oos_months(as_of=as_of) == 0

    def test_dst_transition_winter_vs_summer(self):
        """UTC offset differs across DST — normalization must reflect actual UTC moment."""
        from modules.v2_strategy import _normalize_df_index
        # 2024-01-15 = EST (UTC-5): 16:00 ET → 21:00 UTC
        idx_winter = pd.DatetimeIndex(["2024-01-15 16:00:00"]).tz_localize("America/New_York")
        # 2024-07-15 = EDT (UTC-4): 16:00 ET → 20:00 UTC
        idx_summer = pd.DatetimeIndex(["2024-07-15 16:00:00"]).tz_localize("America/New_York")
        df_w = _normalize_df_index(pd.DataFrame({"Close": [100.0]}, index=idx_winter))
        df_s = _normalize_df_index(pd.DataFrame({"Close": [100.0]}, index=idx_summer))
        assert df_w.index[0].hour == 21, "Winter (EST): 16:00 ET must become 21:00 UTC-naive"
        assert df_s.index[0].hour == 20, "Summer (EDT): 16:00 ET must become 20:00 UTC-naive"


# ---------------------------------------------------------------------------
# X. Evaluator hardening (Round 3)
# ---------------------------------------------------------------------------

class TestEvaluatorHardening:
    """Regression: QQQ alignment, rank column, activity_data, promotion period gate."""

    def _make_returns(self, n: int = 300, seed: int = 0) -> pd.Series:
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2023-01-01", periods=n, freq="B")
        return pd.Series(rng.normal(0.0005, 0.01, n), index=idx)

    def test_qqq_no_overlap_gives_none(self):
        from modules.v2_evaluation import compute_metrics
        r = self._make_returns(300)
        # QQQ on completely non-overlapping dates
        qqq_idx = pd.date_range("2020-01-01", periods=300, freq="B")
        qqq_r = pd.Series([0.001] * 300, index=qqq_idx)
        spy_r = pd.Series([0.001] * 300, index=r.index)
        m = compute_metrics(r, {"SPY": spy_r, "QQQ": qqq_r}, "S", "in_sample")
        assert m.excess_return_vs_qqq is None, "Non-overlapping QQQ dates must yield None"

    def test_qqq_aligned_to_strategy_dates(self):
        from modules.v2_evaluation import compute_metrics
        r = self._make_returns(300)
        # QQQ spans a much wider window — only overlap should be used
        qqq_idx = pd.date_range("2022-01-01", periods=700, freq="B")
        qqq_r = pd.Series([0.001] * 700, index=qqq_idx)
        spy_r = pd.Series([0.001] * 300, index=r.index)
        m = compute_metrics(r, {"SPY": spy_r, "QQQ": qqq_r}, "S", "in_sample")
        assert m.excess_return_vs_qqq is not None, "Overlapping QQQ dates must produce a value"

    def test_insufficient_evidence_never_ranks_above_sufficient(self):
        from modules.v2_evaluation import EvaluationMetrics, compare_strategies
        a = EvaluationMetrics("High_Sharpe_Insuf", "in_sample")
        a.sufficient_evidence = False
        a.sharpe_ratio = 9.0

        b = EvaluationMetrics("Low_Sharpe_Suf", "in_sample")
        b.sufficient_evidence = True
        b.sharpe_ratio = 1.0

        df = compare_strategies([a, b])
        row_a = df[df["strategy"] == "High_Sharpe_Insuf"].iloc[0]
        row_b = df[df["strategy"] == "Low_Sharpe_Suf"].iloc[0]
        assert row_b["rank"] == 1, f"Sufficient-evidence strategy must rank 1, got {row_b['rank']}"
        rank_a = row_a["rank"]
        assert rank_a is None or (isinstance(rank_a, float) and math.isnan(rank_a)), \
            f"Insufficient-evidence must have rank=None/NaN, got {rank_a}"

    def test_rank_column_present(self):
        from modules.v2_evaluation import EvaluationMetrics, compare_strategies
        m = EvaluationMetrics("X", "in_sample")
        m.sufficient_evidence = True
        m.sharpe_ratio = 1.5
        df = compare_strategies([m])
        assert "rank" in df.columns

    def test_activity_data_populates_fields(self):
        from modules.v2_evaluation import compute_metrics
        r = self._make_returns(300)
        spy_r = pd.Series([0.001] * 300, index=r.index)
        activity = {
            "avg_positions": 12.0,
            "avg_sector_concentration": 0.25,
            "estimated_annual_turnover": 0.50,
        }
        m = compute_metrics(r, {"SPY": spy_r}, "S", "in_sample", activity_data=activity)
        assert m.avg_positions == 12.0
        assert m.avg_sector_concentration == 0.25
        assert m.estimated_annual_turnover == 0.50
        # Cost estimate: turnover × 5 bps one-way × 2 = 5.0 bps
        assert m.estimated_annual_cost_bps == pytest.approx(5.0, rel=1e-6)

    def test_activity_fields_none_when_absent(self):
        from modules.v2_evaluation import compute_metrics
        r = self._make_returns(300)
        spy_r = pd.Series([0.001] * 300, index=r.index)
        m = compute_metrics(r, {"SPY": spy_r}, "S", "in_sample")
        assert m.avg_positions is None
        assert m.avg_sector_concentration is None
        assert m.estimated_annual_turnover is None
        assert m.estimated_annual_cost_bps is None

    def test_promotion_blocked_for_retrospective_holdout(self):
        from modules.v2_evaluation import EvaluationMetrics, check_promotion_criteria
        strat = EvaluationMetrics("S", "retrospective_holdout")
        strat.sharpe_ratio = 2.0
        spy = EvaluationMetrics("SPY", "retrospective_holdout")
        spy.sharpe_ratio = 1.0
        spy.max_drawdown = -0.20
        result = check_promotion_criteria(strat, spy, oos_months=12)
        assert result["period_is_prospective_oos"] is False
        assert result["all_passed"] is False
        assert len(result.get("failures", [])) >= 1

    def test_promotion_blocked_for_in_sample(self):
        from modules.v2_evaluation import EvaluationMetrics, check_promotion_criteria
        strat = EvaluationMetrics("S", "in_sample")
        strat.sharpe_ratio = 5.0
        spy = EvaluationMetrics("SPY", "in_sample")
        spy.sharpe_ratio = 1.0
        spy.max_drawdown = -0.10
        result = check_promotion_criteria(strat, spy, oos_months=12)
        assert result.get("all_passed") is False
        assert result.get("period_is_prospective_oos") is False
