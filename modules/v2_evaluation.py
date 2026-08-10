"""
V2 evaluation framework — metrics, period definitions, strategy comparison.

Pre-registered evaluation protocol. Period splits must not be changed
after out-of-sample data has been observed.

V2A: computes metrics from return series. Does not run backtests itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Registration date — the day the V2 model was first registered
# ---------------------------------------------------------------------------

MODEL_REGISTRATION_DATE = pd.Timestamp("2026-08-10")

# ---------------------------------------------------------------------------
# Pre-registered period definitions (locked — do not change after OOS observed)
#
# Four periods:
#   in_sample              2015-01-01 – 2020-12-31  (model development)
#   validation             2021-01-01 – 2022-12-31  (hyperparameter sanity check)
#   retrospective_holdout  2023-01-01 – 2026-08-09  (observed before lock; not for tuning)
#   prospective_oos        2026-08-11+               (truly OOS; not yet observed at lock)
# ---------------------------------------------------------------------------

EVALUATION_PERIODS = {
    "in_sample": {
        "start": pd.Timestamp("2015-01-01"),
        "end": pd.Timestamp("2020-12-31"),
        "description": "Model development period. Parameters may only be set using this window.",
        "locked": False,
    },
    "validation": {
        "start": pd.Timestamp("2021-01-01"),
        "end": pd.Timestamp("2022-12-31"),
        "description": "Validation period. No parameter tuning after observing this data.",
        "locked": False,
    },
    "retrospective_holdout": {
        "start": pd.Timestamp("2023-01-01"),
        "end": pd.Timestamp("2026-08-09"),
        "description": (
            "Retrospective holdout. Observed before model lock; not used for tuning. "
            "Reported for context only — not for promotion decisions."
        ),
        "locked": True,
    },
    "prospective_out_of_sample": {
        "start": pd.Timestamp("2026-08-11"),
        "end": None,  # Open-ended; updated to current date at evaluation time
        "description": (
            "Prospective OOS. Truly out-of-sample — no data observed before model lock. "
            "Primary window for promotion decisions."
        ),
        "locked": True,
    },
}

BENCHMARKS = ["SPY", "QQQ"]
RISK_FREE_RATE_ANNUAL = 0.04
TRADING_DAYS_PER_YEAR = 252
EST_TRADE_COST_BPS = 5  # one-way; 10 bps round-trip

PROMOTION_CRITERIA = {
    "min_prospective_oos_months": 6,
    "min_sharpe_above_spy": 0.10,          # strategy Sharpe must exceed SPY Sharpe by this
    "max_drawdown_ratio_vs_spy": 1.25,     # max_drawdown / spy_max_drawdown must be <= this
    "min_positive_alpha_months_fraction": 0.50,
    "must_report_survivorship_bias": True,
    "must_report_factor_coverage": True,
    "promotion_blocked_if_backtest_unsupported": True,
}

# ---------------------------------------------------------------------------
# Backtest status — explains why full backtest is not supported
# ---------------------------------------------------------------------------

BACKTEST_STATUS = {
    "full_four_factor": "unsupported",
    "full_four_factor_reason": (
        "Point-in-time fundamentals not available. Using yfinance .info "
        "(current snapshot) would introduce look-ahead bias for quality and value factors. "
        "Survivorship bias present: universe is current S&P 500 members only."
    ),
    "momentum_and_safety_only": "supported",
    "momentum_and_safety_only_caveat": (
        "Momentum and safety factors use price data only — no fundamental look-ahead. "
        "Survivorship bias still present; treat results as directional only."
    ),
    "promotion_blocked": True,
    "promotion_blocked_reason": (
        "Promotion requires >= 6 months of prospective OOS data from a compliant "
        "paper portfolio run. Backtest results alone are insufficient for promotion."
    ),
}


def run_v2_backtest(*args, **kwargs) -> None:
    """
    Full V2 backtest is not supported.

    Fundamental factors (quality, value) require point-in-time data that is not
    available via yfinance. Running a backtest with current-snapshot fundamentals
    would introduce look-ahead bias. See BACKTEST_STATUS for details.
    """
    raise NotImplementedError(
        "Full V2 backtest is unsupported: point-in-time fundamentals not available. "
        "Use BACKTEST_STATUS['momentum_and_safety_only'] for a limited price-only backtest."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_prospective_oos_months(as_of: pd.Timestamp | None = None) -> int:
    """
    Return the number of complete calendar months in the prospective OOS window.

    The prospective OOS window starts 2026-08-11 (the day after model lock).
    Returns 0 if as_of is before the window start.
    Timezone-aware as_of is converted to UTC-naive before comparison.
    """
    if as_of is None:
        as_of = pd.Timestamp.today()
    # Normalize timezone-aware as_of to UTC-naive
    if hasattr(as_of, "tzinfo") and as_of.tzinfo is not None:
        as_of = as_of.tz_convert("UTC").tz_localize(None)
    window_start = EVALUATION_PERIODS["prospective_out_of_sample"]["start"]
    if as_of < window_start:
        return 0
    delta = as_of - window_start
    return max(0, int(delta.days // 30))


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluationMetrics:
    strategy_name: str
    period: str

    annualized_return: Optional[float] = None
    total_return: Optional[float] = None
    excess_return_vs_spy: Optional[float] = None
    excess_return_vs_qqq: Optional[float] = None

    annualized_volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    calmar_ratio: Optional[float] = None
    beta: Optional[float] = None
    downside_capture: Optional[float] = None

    hit_rate: Optional[float] = None

    # These three fields require position-level data (trade log), not just return series
    avg_positions: Optional[float] = None          # requires position-level data
    avg_sector_concentration: Optional[float] = None  # requires position-level data
    estimated_annual_turnover: Optional[float] = None  # requires position-level data
    estimated_annual_cost_bps: Optional[float] = None  # requires position-level data

    factor_coverage: Optional[float] = None
    claude_availability: float = 0.0   # always 0.0 in V2A — Claude cannot create orders
    survivorship_bias_note: str = "survivorship bias present — current universe only"
    point_in_time_fundamentals: bool = False
    data_quality: str = "limited"

    factor_only_sharpe: Optional[float] = None
    claude_shadow_sharpe: Optional[float] = None

    _positive_alpha_months_fraction: Optional[float] = None  # fraction of months strategy beat SPY

    sufficient_evidence: bool = False
    evidence_note: str = ""


def compute_metrics(
    returns: pd.Series,
    benchmark_returns: dict[str, pd.Series],
    strategy_name: str,
    period: str,
    factor_coverage: float | None = None,
    risk_free_rate: float = RISK_FREE_RATE_ANNUAL,
    activity_data: dict | None = None,
) -> EvaluationMetrics:
    """
    Compute evaluation metrics from a daily return series.

    returns: pd.Series of daily portfolio returns (not cumulative)
    benchmark_returns: {"SPY": series, "QQQ": series}

    activity_data: optional dict with position-level fields that cannot be
      computed from returns alone. Supported keys:
        avg_positions, avg_sector_concentration, estimated_annual_turnover,
        estimated_annual_cost_bps, claude_availability.
      When absent, these fields remain None (not fabricated as 0).

    claude_availability is always 0.0 in V2A (Claude not connected for orders).
    """
    m = EvaluationMetrics(strategy_name=strategy_name, period=period)
    m.claude_availability = 0.0  # V2A: Claude is never available for order creation

    if returns is None or len(returns) < 20:
        m.evidence_note = "insufficient_evidence: fewer than 20 observations"
        return m

    r = returns.dropna()
    n = len(r)

    total_ret = float((1 + r).prod() - 1)
    years = n / TRADING_DAYS_PER_YEAR
    m.total_return = total_ret
    m.annualized_return = float((1 + total_ret) ** (1 / years) - 1) if years > 0 else None

    daily_std = float(r.std())
    m.annualized_volatility = daily_std * math.sqrt(TRADING_DAYS_PER_YEAR)

    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_daily = r - daily_rf
    if daily_std > 0:
        m.sharpe_ratio = float(excess_daily.mean() / daily_std * math.sqrt(TRADING_DAYS_PER_YEAR))

    neg_r = r[r < daily_rf]
    if len(neg_r) >= 5:
        ds = float(neg_r.std()) * math.sqrt(TRADING_DAYS_PER_YEAR)
        ann_exc = float(excess_daily.mean()) * TRADING_DAYS_PER_YEAR
        if ds > 0:
            m.sortino_ratio = ann_exc / ds

    cumulative = (1 + r).cumprod()
    drawdowns = cumulative / cumulative.cummax() - 1
    m.max_drawdown = float(drawdowns.min())

    if m.max_drawdown and m.max_drawdown < 0 and m.annualized_return is not None:
        m.calmar_ratio = float(m.annualized_return / abs(m.max_drawdown))

    m.hit_rate = float((r > 0).mean())

    spy_r = benchmark_returns.get("SPY")
    if spy_r is not None and len(spy_r) >= 20:
        spy_r = spy_r.dropna().reindex(r.index).dropna()
        aligned = r.reindex(spy_r.index).dropna()
        if len(aligned) >= 20:
            spy_ann = float((1 + spy_r).prod() ** (TRADING_DAYS_PER_YEAR / len(spy_r)) - 1)
            m.excess_return_vs_spy = (m.annualized_return or 0.0) - spy_ann
            spy_var = float(spy_r.var())
            if spy_var > 0:
                cov = float(np.cov(aligned.values, spy_r.reindex(aligned.index).fillna(0).values)[0, 1])
                m.beta = cov / spy_var
            spy_down = spy_r < 0
            if spy_down.sum() >= 5:
                strat_down = aligned.reindex(spy_r[spy_down].index).dropna()
                spy_down_al = spy_r[spy_down].reindex(strat_down.index).dropna()
                if len(spy_down_al) >= 5 and float(spy_down_al.mean()) != 0:
                    m.downside_capture = float(strat_down.mean() / spy_down_al.mean())

            # Monthly alpha fraction: fraction of months strategy outperformed SPY
            if isinstance(r.index, pd.DatetimeIndex):
                strat_monthly = (1 + r).resample("ME").prod() - 1
                spy_monthly = (1 + spy_r).resample("ME").prod() - 1
                common_months = strat_monthly.index.intersection(spy_monthly.index)
                if len(common_months) >= 2:
                    strat_m = strat_monthly.reindex(common_months)
                    spy_m = spy_monthly.reindex(common_months)
                    m._positive_alpha_months_fraction = float((strat_m > spy_m).mean())

    qqq_r = benchmark_returns.get("QQQ")
    if qqq_r is not None and len(qqq_r) >= 20:
        # Align QQQ to strategy dates (same as SPY alignment — no cross-period contamination)
        qqq_aligned = qqq_r.dropna().reindex(r.index).dropna()
        strat_for_qqq = r.reindex(qqq_aligned.index).dropna()
        qqq_aligned = qqq_aligned.reindex(strat_for_qqq.index).dropna()
        if len(qqq_aligned) >= 20:
            years_qqq = len(qqq_aligned) / TRADING_DAYS_PER_YEAR
            if years_qqq > 0:
                qqq_total = float((1 + qqq_aligned).prod() - 1)
                qqq_ann = float((1 + qqq_total) ** (1 / years_qqq) - 1)
                m.excess_return_vs_qqq = (m.annualized_return or 0.0) - qqq_ann
        # else: no overlap → excess_return_vs_qqq remains None

    m.factor_coverage = factor_coverage
    m.sufficient_evidence = n >= TRADING_DAYS_PER_YEAR and m.sharpe_ratio is not None
    if not m.sufficient_evidence:
        m.evidence_note = f"limited: {n} days of data (need {TRADING_DAYS_PER_YEAR}+)"

    # Activity fields — only populated when position-level data is provided
    m.claude_availability = 0.0  # V2A: Claude not connected for order creation
    if activity_data:
        m.avg_positions = activity_data.get("avg_positions")
        m.avg_sector_concentration = activity_data.get("avg_sector_concentration")
        m.estimated_annual_turnover = activity_data.get("estimated_annual_turnover")
        explicit_cost = activity_data.get("estimated_annual_cost_bps")
        if explicit_cost is not None:
            m.estimated_annual_cost_bps = explicit_cost
        elif m.estimated_annual_turnover is not None:
            # Estimate: turnover × round-trip cost (2 × one-way 5 bps = 10 bps)
            m.estimated_annual_cost_bps = m.estimated_annual_turnover * EST_TRADE_COST_BPS * 2
        if "claude_availability" in activity_data:
            m.claude_availability = activity_data["claude_availability"]

    return m


def _metrics_to_row(m: EvaluationMetrics, rank) -> dict:
    return {
        "rank": rank,
        "strategy": m.strategy_name,
        "period": m.period,
        "sufficient_evidence": m.sufficient_evidence,
        "evidence_note": m.evidence_note,
        "ann_return": m.annualized_return,
        "excess_vs_spy": m.excess_return_vs_spy,
        "excess_vs_qqq": m.excess_return_vs_qqq,
        "volatility": m.annualized_volatility,
        "sharpe": m.sharpe_ratio,
        "sortino": m.sortino_ratio,
        "max_drawdown": m.max_drawdown,
        "calmar": m.calmar_ratio,
        "beta": m.beta,
        "downside_capture": m.downside_capture,
        "hit_rate": m.hit_rate,
        "avg_positions": m.avg_positions,
        "avg_sector_concentration": m.avg_sector_concentration,
        "estimated_annual_turnover": m.estimated_annual_turnover,
        "estimated_annual_cost_bps": m.estimated_annual_cost_bps,
        "claude_availability": m.claude_availability,
        "factor_coverage": m.factor_coverage,
        "positive_alpha_months_fraction": m._positive_alpha_months_fraction,
        "survivorship_bias": m.survivorship_bias_note,
        "point_in_time_fundamentals": m.point_in_time_fundamentals,
    }


def compare_strategies(results: list[EvaluationMetrics]) -> pd.DataFrame:
    """
    Build a comparison table with explicit rank column.

    Sufficient-evidence strategies are ranked by Sharpe (descending).
    Insufficient-evidence strategies are listed last with rank=None,
    regardless of their Sharpe value. A strategy with Sharpe 9.0 but
    sufficient_evidence=False never outranks a strategy with Sharpe 1.0
    and sufficient_evidence=True.
    """
    sufficient = sorted(
        [m for m in results if m.sufficient_evidence],
        key=lambda m: (m.sharpe_ratio or float("-inf")),
        reverse=True,
    )
    insufficient = [m for m in results if not m.sufficient_evidence]

    rows = []
    for rank_idx, m in enumerate(sufficient, start=1):
        rows.append(_metrics_to_row(m, rank=rank_idx))
    for m in insufficient:
        rows.append(_metrics_to_row(m, rank=None))

    return pd.DataFrame(rows)


def check_promotion_criteria(
    strategy_metrics: EvaluationMetrics,
    spy_metrics: EvaluationMetrics,
    oos_months: int,
    min_factor_coverage: float = 0.6,
) -> dict[str, bool]:
    """
    Check all promotion criteria against actual SPY benchmark metrics.

    Promotion is blocked if V2 backtest is unsupported (see BACKTEST_STATUS).
    All criteria must pass for promotion.
    """
    results: dict[str, bool | list] = {}
    failures: list[str] = []

    # 0. Period must be prospective_out_of_sample — only this counts for promotion
    if strategy_metrics.period != "prospective_out_of_sample":
        results["period_is_prospective_oos"] = False
        results["all_passed"] = False
        results["failures"] = [
            f"Promotion requires period='prospective_out_of_sample'; "
            f"got '{strategy_metrics.period}'. Retrospective or in-sample data "
            "cannot substitute for prospective out-of-sample evidence."
        ]
        return results
    results["period_is_prospective_oos"] = True

    # 1. Minimum prospective OOS data
    results["min_prospective_oos_months"] = (
        oos_months >= PROMOTION_CRITERIA["min_prospective_oos_months"]
    )

    # 2. Sharpe ratio must exceed SPY Sharpe by the threshold
    strat_sharpe = strategy_metrics.sharpe_ratio
    spy_sharpe = spy_metrics.sharpe_ratio
    if strat_sharpe is not None and spy_sharpe is not None:
        results["min_sharpe_above_spy"] = (
            (strat_sharpe - spy_sharpe) >= PROMOTION_CRITERIA["min_sharpe_above_spy"]
        )
    else:
        results["min_sharpe_above_spy"] = False

    # 3. Max drawdown must not be too much worse than SPY's
    strat_dd = strategy_metrics.max_drawdown
    spy_dd = spy_metrics.max_drawdown
    if strat_dd is not None and spy_dd is not None and spy_dd < 0:
        results["max_drawdown_ratio_vs_spy"] = (
            abs(strat_dd) / abs(spy_dd) <= PROMOTION_CRITERIA["max_drawdown_ratio_vs_spy"]
        )
    else:
        results["max_drawdown_ratio_vs_spy"] = False

    # 4. Fraction of months with positive alpha vs SPY
    alpha_frac = strategy_metrics._positive_alpha_months_fraction
    if alpha_frac is not None:
        results["min_positive_alpha_months_fraction"] = (
            alpha_frac >= PROMOTION_CRITERIA["min_positive_alpha_months_fraction"]
        )
    else:
        results["min_positive_alpha_months_fraction"] = False

    # 5. Survivorship bias disclosed
    results["must_report_survivorship_bias"] = bool(strategy_metrics.survivorship_bias_note)

    # 6. Factor coverage disclosed
    results["must_report_factor_coverage"] = (
        strategy_metrics.factor_coverage is not None
        and strategy_metrics.factor_coverage >= min_factor_coverage
    )

    # 7. Backtest support gate — promotion blocked if backtest unsupported
    results["backtest_not_blocking_promotion"] = not BACKTEST_STATUS.get("promotion_blocked", False)

    # Collect failures for reporting
    for k, v in results.items():
        if isinstance(v, bool) and not v:
            failures.append(k)

    results["all_passed"] = all(
        v for k, v in results.items()
        if isinstance(v, bool) and k != "all_passed"
    )
    results["failures"] = failures
    return results
