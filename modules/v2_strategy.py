"""
Quant Baseline V2 — strategy definitions, factor model, and capped allocation.

V2A: research/paper foundation only. Not active in production workflows.

Key differences from V1:
  - Four factors: momentum, quality, value, safety (replaces sentiment in base weights)
  - Explicit feature_available flags — no silent median imputation
  - factor_coverage field on each scored ticker
  - Correct capped-weight allocation (no renormalization above the cap)
  - Three clearly differentiated strategies (no Balanced_V2)
  - MegaCap as universe filter, not scoring penalty
  - Claude shadow logged but blocked from order creation

V2 strategy names end in _V2 — they cannot collide with V1 idempotency keys.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# V2 version constants
# ---------------------------------------------------------------------------

V2_MODEL_VERSION = "quant_baseline_v2"
V2_PORTFOLIO_VERSION = "risk_managed_factor_v2"

# Strategy names with _V2 suffix — cannot collide with V1 strategy names
V2_STRATEGY_IDS: frozenset[str] = frozenset({
    "Core_Quality_Momentum_V2",
    "Aggressive_Momentum_V2",
    "Defensive_Quality_V2",
    "Factor_Only_Core_V2",
    "Factor_Plus_Claude_Shadow_V2",
})


def is_v2_strategy(strategy_name: str) -> bool:
    return strategy_name in V2_STRATEGY_IDS


# ---------------------------------------------------------------------------
# V2 strategy configuration
# Pre-registered weights. NOT optimized against history.
# Factor weights sum to 1.0 for each strategy.
# ---------------------------------------------------------------------------

V2_STRATEGIES: dict[str, dict[str, Any]] = {
    "Core_Quality_Momentum_V2": {
        "strategy_id": "CQM_V2",
        "description": (
            "Core strategy. Balanced exposure to long-term momentum and company quality, "
            "with moderate value and risk components. Designed for steady factor exposure "
            "without chasing either pure momentum or pure defensive posture."
        ),
        "base_weights": {"momentum": 0.35, "quality": 0.30, "value": 0.20, "safety": 0.15},
        "score_column": "score_core_v2",
        "max_positions": 12,
        "max_position_weight": 0.12,
        "max_new_buys_per_week": 4,
        "buy_top_n": 25,
        "min_score_percentile": 0.70,
        "sell_rank_threshold": 110,
        "stop_loss": -0.18,
        "trailing_stop": -0.25,
        "buyback_cooldown_days": 10,
        "exposure": {
            "explosive": 0.95, "bullish": 0.90,
            "neutral": 0.65, "defensive": 0.35, "unknown": 0.55,
        },
        "shadow_only": False,
        "claude_shadow": False,
    },
    "Aggressive_Momentum_V2": {
        "strategy_id": "AGM_V2",
        "description": (
            "Higher momentum exposure. Accepts higher short-term volatility in exchange "
            "for stronger factor tilt toward price trend. Retains quality floor and "
            "risk controls. Clearly distinct from Core: momentum weight 50% vs 35%."
        ),
        "base_weights": {"momentum": 0.50, "quality": 0.25, "value": 0.15, "safety": 0.10},
        "score_column": "score_aggressive_v2",
        "max_positions": 10,
        "max_position_weight": 0.15,
        "max_new_buys_per_week": 5,
        "buy_top_n": 20,
        "min_score_percentile": 0.75,
        "sell_rank_threshold": 120,
        "stop_loss": -0.22,
        "trailing_stop": -0.28,
        "buyback_cooldown_days": 7,
        "exposure": {
            "explosive": 1.00, "bullish": 1.00,
            "neutral": 0.70, "defensive": 0.35, "unknown": 0.60,
        },
        "shadow_only": False,
        "claude_shadow": False,
    },
    "Defensive_Quality_V2": {
        "strategy_id": "DQV_V2",
        "description": (
            "Prioritizes profitability, balance-sheet strength, and downside protection. "
            "Lowest momentum weight of the three (15%). Safety factor at 25% — highest "
            "of all three. Clearly distinct from Core and Aggressive."
        ),
        "base_weights": {"momentum": 0.15, "quality": 0.40, "value": 0.20, "safety": 0.25},
        "score_column": "score_defensive_v2",
        "max_positions": 15,
        "max_position_weight": 0.09,
        "max_new_buys_per_week": 3,
        "buy_top_n": 30,
        "min_score_percentile": 0.65,
        "sell_rank_threshold": 100,
        "stop_loss": -0.14,
        "trailing_stop": -0.20,
        "buyback_cooldown_days": 14,
        "exposure": {
            "explosive": 0.80, "bullish": 0.75,
            "neutral": 0.55, "defensive": 0.30, "unknown": 0.50,
        },
        "shadow_only": False,
        "claude_shadow": False,
    },
    # Shadow control variants — same base weights as Core, shadow_only=True
    "Factor_Only_Core_V2": {
        "strategy_id": "FOC_V2",
        "description": "Control: Core_Quality_Momentum_V2 without any Claude signal. Shadow only.",
        "base_weights": {"momentum": 0.35, "quality": 0.30, "value": 0.20, "safety": 0.15},
        "score_column": "score_core_v2",
        "shadow_only": True,
        "claude_shadow": False,
        "max_positions": 12,
        "max_position_weight": 0.12,
        "max_new_buys_per_week": 4,
        "buy_top_n": 25,
        "min_score_percentile": 0.70,
        "sell_rank_threshold": 110,
        "stop_loss": -0.18,
        "trailing_stop": -0.25,
        "buyback_cooldown_days": 10,
        "exposure": {
            "explosive": 0.95, "bullish": 0.90,
            "neutral": 0.65, "defensive": 0.35, "unknown": 0.55,
        },
    },
    "Factor_Plus_Claude_Shadow_V2": {
        "strategy_id": "FCS_V2",
        "description": (
            "Same as Factor_Only_Core_V2 but Claude shadow analysis is logged "
            "alongside factor scores. Claude output CANNOT create orders. "
            "Allows direct measurement of Claude signal contribution."
        ),
        "base_weights": {"momentum": 0.35, "quality": 0.30, "value": 0.20, "safety": 0.15},
        "score_column": "score_core_v2",
        "shadow_only": True,
        "claude_shadow": True,
        "max_positions": 12,
        "max_position_weight": 0.12,
        "max_new_buys_per_week": 4,
        "buy_top_n": 25,
        "min_score_percentile": 0.70,
        "sell_rank_threshold": 110,
        "stop_loss": -0.18,
        "trailing_stop": -0.25,
        "buyback_cooldown_days": 10,
        "exposure": {
            "explosive": 0.95, "bullish": 0.90,
            "neutral": 0.65, "defensive": 0.35, "unknown": 0.55,
        },
    },
}

# ---------------------------------------------------------------------------
# Factor computation helpers
# ---------------------------------------------------------------------------

def _winsorize_rank(series: pd.Series, low: float = 0.02, high: float = 0.98) -> pd.Series:
    """Winsorize at low/high percentile, then convert to [0, 100] percentile rank."""
    if series.isna().all():
        return series
    lo = series.quantile(low)
    hi = series.quantile(high)
    clipped = series.clip(lower=lo, upper=hi)
    return clipped.rank(pct=True, na_option="keep") * 100.0


def build_momentum_factor(price_data: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> pd.DataFrame:
    """
    Compute momentum factor from price data.

    Sub-factors (all skip last month to reduce reversal noise):
      ret_12_1: return from 12 months ago to 1 month ago  (weight 0.40)
      ret_6_1:  return from 6 months ago to 1 month ago   (weight 0.35)
      ret_3_1:  return from 3 months ago to 1 month ago   (weight 0.25)

    Only uses data up to as_of (no look-ahead).
    Missing sub-factors are skipped; weights renormalized over available sub-factors.
    Unavailable tickers get momentum_score=None, not median-fill.
    """
    DAYS_1M = 21
    DAYS_3M = 63
    DAYS_6M = 126
    DAYS_12M = 252

    rows = []
    for ticker, df in price_data.items():
        if df is None or df.empty:
            rows.append({
                "ticker": ticker,
                "ret_12_1": None, "ret_6_1": None, "ret_3_1": None,
                "momentum_raw": None, "momentum_available": False,
            })
            continue

        df = df[df.index <= as_of].copy()
        close_col = "Adj Close" if "Adj Close" in df.columns else "Close"

        if close_col not in df.columns or len(df) < DAYS_12M + DAYS_1M:
            rows.append({
                "ticker": ticker,
                "ret_12_1": None, "ret_6_1": None, "ret_3_1": None,
                "momentum_raw": None, "momentum_available": False,
            })
            continue

        prices = df[close_col].dropna()
        if len(prices) < DAYS_12M + DAYS_1M:
            rows.append({
                "ticker": ticker,
                "ret_12_1": None, "ret_6_1": None, "ret_3_1": None,
                "momentum_raw": None, "momentum_available": False,
            })
            continue

        p_1m = prices.iloc[-DAYS_1M] if len(prices) >= DAYS_1M else None
        p_3m = prices.iloc[-DAYS_3M] if len(prices) >= DAYS_3M else None
        p_6m = prices.iloc[-DAYS_6M] if len(prices) >= DAYS_6M else None
        p_12m = prices.iloc[-DAYS_12M] if len(prices) >= DAYS_12M else None

        def _ret(p_end, p_start):
            if p_end is None or p_start is None or p_start == 0:
                return None
            return float(p_end / p_start) - 1.0

        r12 = _ret(p_1m, p_12m)
        r6 = _ret(p_1m, p_6m)
        r3 = _ret(p_1m, p_3m)

        weights = {"ret_12_1": 0.40, "ret_6_1": 0.35, "ret_3_1": 0.25}
        vals = {"ret_12_1": r12, "ret_6_1": r6, "ret_3_1": r3}
        w_sum = sum(w for k, w in weights.items() if vals[k] is not None)
        if w_sum <= 0:
            rows.append({
                "ticker": ticker,
                "ret_12_1": r12, "ret_6_1": r6, "ret_3_1": r3,
                "momentum_raw": None, "momentum_available": False,
            })
            continue

        mom_raw = sum(vals[k] * w / w_sum for k, w in weights.items() if vals[k] is not None)
        rows.append({
            "ticker": ticker,
            "ret_12_1": r12, "ret_6_1": r6, "ret_3_1": r3,
            "momentum_raw": mom_raw, "momentum_available": True,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    avail = result["momentum_available"] == True
    if avail.sum() > 1:
        result.loc[avail, "momentum_score"] = _winsorize_rank(result.loc[avail, "momentum_raw"])
    else:
        result["momentum_score"] = None
    result.loc[~avail, "momentum_score"] = None
    return result


def build_quality_factor(fundamentals: dict[str, dict]) -> pd.DataFrame:
    """
    Compute quality factor from fundamentals (yfinance .info — NOT point-in-time).

    Sub-factors: roe (0.25), gross_margin (0.25), debt_equity_inv (0.20),
                 earnings_growth (0.20), fcf_margin (0.10)

    Missing values tracked via feature_available flags.
    NEVER imputes missing values with median and treats them as real.
    """
    rows = []
    for ticker, info in fundamentals.items():
        if not info:
            rows.append(_quality_unavailable(ticker))
            continue

        roe = info.get("returnOnEquity")
        gm = info.get("grossMargins")
        de = info.get("debtToEquity")
        eg = info.get("earningsGrowth")
        fcf_m = info.get("freeCashflow")
        rev = info.get("totalRevenue")

        fcf_margin = None
        if fcf_m is not None and rev and float(rev) > 0:
            try:
                fcf_margin = float(fcf_m) / float(rev)
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        de_inv = None
        if de is not None:
            try:
                de_inv = 1.0 / (1.0 + min(abs(float(de)), 5.0))
            except (TypeError, ValueError):
                pass

        features = {
            "roe": roe,
            "gross_margin": gm,
            "debt_equity_inv": de_inv,
            "earnings_growth": eg,
            "fcf_margin": fcf_margin,
        }
        weights = {"roe": 0.25, "gross_margin": 0.25, "debt_equity_inv": 0.20,
                   "earnings_growth": 0.20, "fcf_margin": 0.10}
        available = {k: (v is not None) for k, v in features.items()}
        coverage = sum(available.values()) / len(available)
        w_sum = sum(w for k, w in weights.items() if available[k])

        if w_sum <= 0:
            quality_raw = None
        else:
            quality_raw = sum(float(features[k]) * w / w_sum
                              for k, w in weights.items() if available[k])

        rows.append({
            "ticker": ticker,
            **{f"quality_{k}_available": v for k, v in available.items()},
            "quality_raw": quality_raw,
            "quality_coverage": coverage,
            "quality_available": quality_raw is not None,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    avail = result["quality_available"] == True
    if avail.sum() > 1:
        result.loc[avail, "quality_score"] = _winsorize_rank(result.loc[avail, "quality_raw"])
    else:
        result["quality_score"] = None
    result.loc[~avail, "quality_score"] = None
    return result


def _quality_unavailable(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "quality_roe_available": False,
        "quality_gross_margin_available": False,
        "quality_debt_equity_inv_available": False,
        "quality_earnings_growth_available": False,
        "quality_fcf_margin_available": False,
        "quality_raw": None,
        "quality_coverage": 0.0,
        "quality_available": False,
    }


def build_value_factor(fundamentals: dict[str, dict]) -> pd.DataFrame:
    """
    Compute value factor from fundamentals (yfinance .info — NOT point-in-time).

    Sub-factors: fcf_yield (0.40), earnings_yield (0.35), ev_ebitda_inv (0.25)

    Missing values tracked. NOT point-in-time.
    """
    rows = []
    for ticker, info in fundamentals.items():
        if not info:
            rows.append(_value_unavailable(ticker))
            continue

        fcf = info.get("freeCashflow")
        mktcap = info.get("marketCap")
        fpe = info.get("forwardPE")
        ev = info.get("enterpriseValue")
        ebitda = info.get("ebitda")

        fcf_yield = None
        if fcf is not None and mktcap and float(mktcap) > 0:
            try:
                fcf_yield = float(fcf) / float(mktcap)
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        earnings_yield = None
        if fpe is not None:
            try:
                fpe_val = float(fpe)
                if fpe_val > 0:
                    earnings_yield = 1.0 / fpe_val
            except (TypeError, ValueError):
                pass

        ev_ebitda_inv = None
        if ev is not None and ebitda and float(ebitda) > 0:
            try:
                ratio = float(ev) / float(ebitda)
                if ratio > 0:
                    ev_ebitda_inv = 1.0 / ratio
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        features = {
            "fcf_yield": fcf_yield,
            "earnings_yield": earnings_yield,
            "ev_ebitda_inv": ev_ebitda_inv,
        }
        weights = {"fcf_yield": 0.40, "earnings_yield": 0.35, "ev_ebitda_inv": 0.25}
        available = {k: (v is not None) for k, v in features.items()}
        coverage = sum(available.values()) / len(available)
        w_sum = sum(w for k, w in weights.items() if available[k])

        if w_sum <= 0:
            value_raw = None
        else:
            value_raw = sum(float(features[k]) * w / w_sum
                            for k, w in weights.items() if available[k])

        rows.append({
            "ticker": ticker,
            **{f"value_{k}_available": v for k, v in available.items()},
            "value_raw": value_raw,
            "value_coverage": coverage,
            "value_available": value_raw is not None,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    avail = result["value_available"] == True
    if avail.sum() > 1:
        result.loc[avail, "value_score"] = _winsorize_rank(result.loc[avail, "value_raw"])
    else:
        result["value_score"] = None
    result.loc[~avail, "value_score"] = None
    return result


def _value_unavailable(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "value_fcf_yield_available": False,
        "value_earnings_yield_available": False,
        "value_ev_ebitda_inv_available": False,
        "value_raw": None,
        "value_coverage": 0.0,
        "value_available": False,
    }


def build_safety_factor(price_data: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> pd.DataFrame:
    """
    Compute safety/risk factor from price data (always available from prices).

    Sub-factors: vol60_inv (0.30), downside_vol_inv (0.25), drawdown_inv (0.15),
                 log_volume (0.10). Beta excluded here (needs SPY reference series).

    Only uses data up to as_of (no look-ahead).
    """
    rows = []
    for ticker, df in price_data.items():
        if df is None or df.empty:
            rows.append({"ticker": ticker, "safety_raw": None, "safety_available": False})
            continue

        df = df[df.index <= as_of].copy()
        close_col = "Adj Close" if "Adj Close" in df.columns else "Close"

        if close_col not in df.columns or len(df) < 30:
            rows.append({"ticker": ticker, "safety_raw": None, "safety_available": False})
            continue

        prices = df[close_col].dropna()
        if len(prices) < 30:
            rows.append({"ticker": ticker, "safety_raw": None, "safety_available": False})
            continue

        returns = prices.pct_change().dropna()
        returns_60 = returns.tail(60) if len(returns) >= 60 else returns
        prices_1y = prices.tail(252) if len(prices) >= 252 else prices

        vol60 = returns_60.std() * math.sqrt(252) if len(returns_60) >= 20 else None

        neg_returns = returns_60[returns_60 < 0]
        downside_vol = (neg_returns.std() * math.sqrt(252)) if len(neg_returns) >= 10 else vol60

        max_dd = None
        if len(prices_1y) >= 20:
            roll_max = prices_1y.cummax()
            dd = float((prices_1y / roll_max - 1).min())
            if not math.isnan(dd):
                max_dd = abs(dd)

        avg_vol = None
        if "Volume" in df.columns:
            vol_series = df["Volume"].dropna().tail(60)
            if len(vol_series) >= 20:
                avg_vol = float(vol_series.mean())

        EPSILON = 1e-6
        features = {
            "vol60_inv": (1.0 / (vol60 + EPSILON)) if vol60 is not None else None,
            "downside_vol_inv": (1.0 / (downside_vol + EPSILON)) if downside_vol is not None else None,
            "drawdown_inv": (1.0 / (max_dd + EPSILON)) if max_dd is not None else None,
            "log_volume": math.log(avg_vol + 1.0) if avg_vol is not None else None,
        }
        weights = {"vol60_inv": 0.30, "downside_vol_inv": 0.25, "drawdown_inv": 0.15, "log_volume": 0.10}
        available = {k: (v is not None) for k, v in features.items()}
        w_sum = sum(w for k, w in weights.items() if available[k])

        if w_sum <= 0:
            rows.append({"ticker": ticker, "safety_raw": None, "safety_available": False})
            continue

        safety_raw = sum(float(features[k]) * w / w_sum for k, w in weights.items() if available[k])
        rows.append({"ticker": ticker, "safety_raw": safety_raw, "safety_available": True})

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    avail = result["safety_available"] == True
    if avail.sum() > 1:
        result.loc[avail, "safety_score"] = _winsorize_rank(result.loc[avail, "safety_raw"])
    else:
        result["safety_score"] = None
    result.loc[~avail, "safety_score"] = None
    return result


def build_v2_factor_scores(
    momentum_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    value_df: pd.DataFrame,
    safety_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all four factor DataFrames and compute per-strategy composite scores.

    factor_coverage = fraction of 4 factors with a real score (0.0 – 1.0).
    Missing factors produce None in the composite — not 50.0 neutral fill.
    """
    df = momentum_df[["ticker", "momentum_score", "momentum_available"]].copy()
    df = df.merge(
        quality_df[["ticker", "quality_score", "quality_available", "quality_coverage"]],
        on="ticker", how="outer",
    )
    df = df.merge(
        value_df[["ticker", "value_score", "value_available", "value_coverage"]],
        on="ticker", how="outer",
    )
    df = df.merge(
        safety_df[["ticker", "safety_score", "safety_available"]],
        on="ticker", how="outer",
    )

    avail_cols = ["momentum_available", "quality_available", "value_available", "safety_available"]
    for c in avail_cols:
        if c not in df.columns:
            df[c] = False
    df["factor_coverage"] = df[avail_cols].fillna(False).sum(axis=1) / 4.0

    factor_score_col = {
        "momentum": "momentum_score",
        "quality": "quality_score",
        "value": "value_score",
        "safety": "safety_score",
    }
    avail_col = {
        "momentum": "momentum_available",
        "quality": "quality_available",
        "value": "value_available",
        "safety": "safety_available",
    }

    # Compute composite for each strategy (deduplicate by score_column)
    seen_cols: set[str] = set()
    for strat_cfg in V2_STRATEGIES.values():
        col = strat_cfg["score_column"]
        if col in seen_cols:
            continue
        seen_cols.add(col)
        weights = strat_cfg["base_weights"]
        scores = []
        for row in df.itertuples():
            row_vals = []
            row_wts = []
            for factor, w in weights.items():
                is_avail = getattr(row, avail_col[factor], False)
                s = getattr(row, factor_score_col[factor], None)
                if is_avail and s is not None and not (isinstance(s, float) and math.isnan(s)):
                    row_vals.append(float(s) * w)
                    row_wts.append(w)
            if row_wts:
                w_total = sum(row_wts)
                scores.append(sum(row_vals) / w_total * sum(weights.values()))
            else:
                scores.append(None)
        df[col] = scores

    return df


# ---------------------------------------------------------------------------
# Correct capped-weight allocation (fixes the renormalization-above-cap bug)
# ---------------------------------------------------------------------------

def build_target_weights_v2(
    scores: dict[str, float],
    max_position_weight: float,
    exposure: float = 1.0,
) -> dict[str, float]:
    """
    Correct capped-weight allocation.

    Algorithm:
    1. Proportional initial weights from scores.
    2. Iteratively cap at max_position_weight and redistribute surplus
       only to positions still below the cap.
    3. Any surplus that cannot be distributed without breaching the cap
       stays uninvested (caller treats 1 - sum(weights) as cash).
    4. Apply regime exposure multiplier after capping.

    Invariant: all(w <= max_position_weight for w in result.values())
    """
    if not scores:
        return {}

    cap = max_position_weight
    total_raw = sum(scores.values())
    if total_raw <= 0:
        return {}

    weights = {t: s / total_raw for t, s in scores.items()}
    n = len(weights)

    for _ in range(n):
        capped = {t: min(w, cap) for t, w in weights.items()}
        surplus = sum(weights[t] - capped[t] for t in weights)
        if surplus < 1e-10:
            break

        under = {t: w for t, w in capped.items() if w < cap - 1e-12}
        if not under:
            break  # surplus cannot be invested without breaching cap → becomes cash

        total_under = sum(under.values())
        if total_under <= 0:
            break

        for t in under:
            capped[t] += surplus * (capped[t] / total_under)
            capped[t] = min(capped[t], cap)

        weights = capped

    # Apply cap one final time: if the loop broke on "no under positions",
    # weights still holds uncapped proportional values. Cap here ensures the
    # invariant holds regardless of which exit path the loop took.
    return {t: min(w, cap) * exposure for t, w in weights.items()}
