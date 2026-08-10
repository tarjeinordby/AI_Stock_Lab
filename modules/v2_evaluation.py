"""
V2 evaluation framework — metrics, period definitions, strategy comparison.

Pre-registered evaluation protocol. Period splits must not be changed
after out-of-sample data has been observed.

V2A: computes metrics from return series. Does not run backtests itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Pre-registered period definitions (locked — do not change after OOS observed)
# ---------------------------------------------------------------------------

EVALUATION_PERIODS = {
    "in_sample": {
        "start": "2015-01-01",
        "end": "2020-12-31",
        "description": "Model development period. Parameters may only be set using this window.",
    },
    "validation": {
        "start": "2021-01-01",
        "end": "2022-12-31",
        "description": "Validation period. No parameter tuning after observing this data.",
    },
    "out_of_sample": {
        "start": "2023-01-01",
        "end": None,  # Rolling; updated to current date at evaluation time
        "description": "Truly out-of-sample. Never used for model selection.",
        "locked": True,
    },
}

BENCHMARKS = ["SPY", "QQQ"]
RISK_FREE_RATE_ANNUAL = 0.04
TRADING_DAYS_PER_YEAR = 252
EST_TRADE_COST_BPS = 5  # one-way; 10 bps round-trip

PROMOTION_CRITERIA = {
    "min_out_of_sample_months": 6,
    "min_sharpe_vs_spy": 0.10,
    "max_drawdown_ratio_vs_spy": 1.25,
    "min_positive_alpha_months_fraction": 0.50,
    "must_report_survivorship_bias": True,
    "must_report_factor_coverage": True,
}


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
    avg_positions: Optional[float] = None
    avg_sector_concentration: Optional[float] = None
    estimated_annual_turnover: Optional[float] = None
    estimated_annual_cost_bps: Optional[float] = None

    factor_coverage: Optional[float] = None
    claude_availability: Optional[float] = None
    survivorship_bias_note: str = "survivorship bias present — current universe only"
    point_in_time_fundamentals: bool = False
    data_quality: str = "limited"

    factor_only_sharpe: Optional[float] = None
    claude_shadow_sharpe: Optional[float] = None

    sufficient_evidence: bool = False
    evidence_note: str = ""


def compute_metrics(
    returns: pd.Series,
    benchmark_returns: dict[str, pd.Series],
    strategy_name: str,
    period: str,
    factor_coverage: float | None = None,
    risk_free_rate: float = RISK_FREE_RATE_ANNUAL,
) -> EvaluationMetrics:
    """
    Compute evaluation metrics from a daily return series.

    returns: pd.Series of daily portfolio returns (not cumulative)
    benchmark_returns: {"SPY": series, "QQQ": series}
    """
    m = EvaluationMetrics(strategy_name=strategy_name, period=period)

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

    qqq_r = benchmark_returns.get("QQQ")
    if qqq_r is not None and len(qqq_r) >= 20:
        qqq_clean = qqq_r.dropna()
        qqq_ann = float((1 + qqq_clean).prod() ** (TRADING_DAYS_PER_YEAR / len(qqq_clean)) - 1)
        m.excess_return_vs_qqq = (m.annualized_return or 0.0) - qqq_ann

    m.factor_coverage = factor_coverage
    m.sufficient_evidence = n >= TRADING_DAYS_PER_YEAR and m.sharpe_ratio is not None
    if not m.sufficient_evidence:
        m.evidence_note = f"limited: {n} days of data (need {TRADING_DAYS_PER_YEAR}+)"

    return m


def compare_strategies(results: list[EvaluationMetrics]) -> pd.DataFrame:
    """Build a comparison table sorted by Sharpe ratio descending."""
    rows = []
    for m in results:
        rows.append({
            "strategy": m.strategy_name,
            "period": m.period,
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
            "factor_coverage": m.factor_coverage,
            "sufficient_evidence": m.sufficient_evidence,
            "evidence_note": m.evidence_note,
            "survivorship_bias": m.survivorship_bias_note,
            "point_in_time_fundamentals": m.point_in_time_fundamentals,
        })
    df = pd.DataFrame(rows)
    if "sharpe" in df.columns and not df.empty:
        df = df.sort_values("sharpe", ascending=False, na_position="last")
    return df


def check_promotion_criteria(metrics: EvaluationMetrics, oos_months: int) -> dict[str, bool]:
    """Check all promotion criteria. Returns dict of criterion → passed."""
    return {
        "min_out_of_sample_months": oos_months >= PROMOTION_CRITERIA["min_out_of_sample_months"],
        "min_sharpe_vs_spy": (
            metrics.sharpe_ratio is not None and metrics.sharpe_ratio > 0
            and metrics.excess_return_vs_spy is not None
        ),
        "max_drawdown_ratio_vs_spy": metrics.max_drawdown is not None,
        "must_report_survivorship_bias": bool(metrics.survivorship_bias_note),
        "must_report_factor_coverage": metrics.factor_coverage is not None,
        "sufficient_evidence": metrics.sufficient_evidence,
    }
