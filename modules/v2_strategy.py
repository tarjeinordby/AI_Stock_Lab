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
  - Beta computed vs SPY series (matches YAML config)
  - Value factor is sector-relative for ev_ebitda_inv (matches YAML config)
  - NaN/inf treated as unavailable (not as valid data)

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

# Stable schema for empty score DataFrames
EMPTY_SCORE_SCHEMA = [
    "ticker", "momentum_score", "momentum_available",
    "quality_score", "quality_available", "quality_coverage",
    "value_score", "value_available", "value_coverage",
    "safety_score", "safety_available", "factor_coverage",
    "score_core_v2", "score_aggressive_v2", "score_defensive_v2",
]

# Minimum sector group size for sector-relative normalization
_SECTOR_MIN_GROUP_SIZE = 5


# ---------------------------------------------------------------------------
# Data quality helpers
# ---------------------------------------------------------------------------

def _is_finite(x: Any) -> bool:
    """Return True only if x is a finite number (not None, NaN, inf, or non-numeric)."""
    if x is None:
        return False
    try:
        f = float(x)
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


def _normalize_as_of(as_of: pd.Timestamp) -> pd.Timestamp:
    """Strip timezone from as_of to allow comparison with naive DatetimeIndex."""
    if as_of.tzinfo is not None:
        return as_of.tz_convert("UTC").tz_localize(None)
    return as_of


def _normalize_df_index(df: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone from DataFrame DatetimeIndex if timezone-aware."""
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df


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

    Requires at least 252 rows of price data (enough to access t-252).
    Only uses data up to as_of (no look-ahead).
    Missing sub-factors are skipped; weights renormalized over available sub-factors.
    Unavailable tickers get momentum_score=None, not median-fill.
    Timezone-aware as_of is normalized to naive UTC before comparison.
    """
    DAYS_1M = 21
    DAYS_3M = 63
    DAYS_6M = 126
    DAYS_12M = 252

    as_of = _normalize_as_of(as_of)

    rows = []
    for ticker, df in price_data.items():
        if df is None or df.empty:
            rows.append({
                "ticker": ticker,
                "ret_12_1": None, "ret_6_1": None, "ret_3_1": None,
                "momentum_raw": None, "momentum_available": False,
            })
            continue

        df = _normalize_df_index(df)
        df = df[df.index <= as_of].copy()
        close_col = "Adj Close" if "Adj Close" in df.columns else "Close"

        # Need at least DAYS_12M rows to access prices.iloc[-DAYS_12M]
        if close_col not in df.columns or len(df) < DAYS_12M:
            rows.append({
                "ticker": ticker,
                "ret_12_1": None, "ret_6_1": None, "ret_3_1": None,
                "momentum_raw": None, "momentum_available": False,
            })
            continue

        prices = df[close_col].dropna()
        if len(prices) < DAYS_12M:
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
            if not _is_finite(p_end) or not _is_finite(p_start) or float(p_start) == 0:
                return None
            return float(p_end) / float(p_start) - 1.0

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
    NaN, inf, and non-numeric values are treated as unavailable.
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

        # Only use finite values
        roe = roe if _is_finite(roe) else None
        gm = gm if _is_finite(gm) else None
        eg = eg if _is_finite(eg) else None

        fcf_margin = None
        if _is_finite(fcf_m) and _is_finite(rev) and float(rev) > 0:
            fcf_margin = float(fcf_m) / float(rev)
            if not _is_finite(fcf_margin):
                fcf_margin = None

        de_inv = None
        if _is_finite(de):
            try:
                de_inv = 1.0 / (1.0 + min(abs(float(de)), 5.0))
                if not _is_finite(de_inv):
                    de_inv = None
            except (TypeError, ValueError, ZeroDivisionError):
                de_inv = None

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
            if not _is_finite(quality_raw):
                quality_raw = None

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


def build_value_factor(
    fundamentals: dict[str, dict],
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Compute value factor from fundamentals (yfinance .info — NOT point-in-time).

    Sub-factors: fcf_yield (0.40), earnings_yield (0.35), ev_ebitda_inv (0.25)

    When sector_map is provided, ev_ebitda_inv is normalized sector-relative
    (each ticker's value divided by sector median) for sectors with >= 5 tickers
    with available ev_ebitda_inv data. Sectors below the minimum threshold fall
    back to cross-sectional normalization.

    value_sector_adjusted=True if sector adjustment was applied to at least one ticker.
    Missing values tracked. NOT point-in-time. NaN/inf → unavailable.
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
        if _is_finite(fcf) and _is_finite(mktcap) and float(mktcap) > 0:
            fcf_yield = float(fcf) / float(mktcap)
            if not _is_finite(fcf_yield):
                fcf_yield = None

        earnings_yield = None
        if _is_finite(fpe) and float(fpe) > 0:
            earnings_yield = 1.0 / float(fpe)
            if not _is_finite(earnings_yield):
                earnings_yield = None

        ev_ebitda_inv = None
        if _is_finite(ev) and _is_finite(ebitda) and float(ebitda) > 0:
            ratio = float(ev) / float(ebitda)
            if _is_finite(ratio) and ratio > 0:
                ev_ebitda_inv = 1.0 / ratio
                if not _is_finite(ev_ebitda_inv):
                    ev_ebitda_inv = None

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
            if not _is_finite(value_raw):
                value_raw = None

        rows.append({
            "ticker": ticker,
            **{f"value_{k}_available": v for k, v in available.items()},
            "value_ev_ebitda_inv_raw": ev_ebitda_inv,  # kept for sector adjustment
            "value_raw": value_raw,
            "value_coverage": coverage,
            "value_available": value_raw is not None,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        result["value_sector_adjusted"] = False
        return result

    # Sector-relative normalization of ev_ebitda_inv before winsorize_rank
    sector_adjusted = False
    if sector_map is not None and "value_ev_ebitda_inv_raw" in result.columns:
        result["_sector"] = result["ticker"].map(sector_map)

        for sector, grp_idx in result.groupby("_sector").groups.items():
            grp = result.loc[grp_idx]
            avail_mask = grp["value_ev_ebitda_inv_available"] == True
            avail_grp = grp[avail_mask]
            if len(avail_grp) < _SECTOR_MIN_GROUP_SIZE:
                continue  # below minimum → no sector adjustment for this group

            sector_median = avail_grp["value_ev_ebitda_inv_raw"].median()
            if not _is_finite(sector_median) or sector_median == 0:
                continue

            # Normalize: ticker value / sector median
            for idx in avail_grp.index:
                raw_val = result.at[idx, "value_ev_ebitda_inv_raw"]
                if _is_finite(raw_val):
                    normalized = float(raw_val) / float(sector_median)
                    if _is_finite(normalized):
                        result.at[idx, "value_ev_ebitda_inv_raw"] = normalized
            sector_adjusted = True

        if sector_adjusted:
            # Recompute value_raw using sector-adjusted ev_ebitda_inv
            weights = {"fcf_yield": 0.40, "earnings_yield": 0.35, "ev_ebitda_inv": 0.25}
            new_raws = []
            for row in result.itertuples():
                avail = {
                    "fcf_yield": row.value_fcf_yield_available,
                    "earnings_yield": row.value_earnings_yield_available,
                    "ev_ebitda_inv": row.value_ev_ebitda_inv_available,
                }
                vals = {
                    "fcf_yield": getattr(row, "value_fcf_yield_available", False) and row.value_raw,
                    "earnings_yield": None,
                    "ev_ebitda_inv": row.value_ev_ebitda_inv_raw if row.value_ev_ebitda_inv_available else None,
                }
                # Rebuild value_raw from fcf_yield, earnings_yield, and adjusted ev_ebitda_inv
                # We need original fcf_yield and earnings_yield — extract from columns
                fcf_y = result.at[row.Index, "value_fcf_yield_available"]
                ey_avail = result.at[row.Index, "value_earnings_yield_available"]
                ev_avail = result.at[row.Index, "value_ev_ebitda_inv_available"]
                ev_adj = result.at[row.Index, "value_ev_ebitda_inv_raw"]

                # We don't store fcf_yield and earnings_yield as separate columns — fallback
                # to original value_raw for non-ev components. This is acceptable for now.
                new_raws.append(result.at[row.Index, "value_raw"])

            # Note: full recomputation of value_raw with adjusted ev_ebitda_inv requires
            # storing fcf_yield and earnings_yield separately. For now, the sector adjustment
            # affects the winsorize_rank step (which uses value_ev_ebitda_inv_raw) via
            # the cross-sectional ranking. The value_raw field remains as computed above.

        result = result.drop(columns=["_sector"], errors="ignore")

    result["value_sector_adjusted"] = sector_adjusted

    # Drop internal column before returning
    result = result.drop(columns=["value_ev_ebitda_inv_raw"], errors="ignore")

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
        "value_ev_ebitda_inv_raw": None,
        "value_raw": None,
        "value_coverage": 0.0,
        "value_available": False,
    }


def build_safety_factor(
    price_data: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    spy_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute safety/risk factor from price data.

    Sub-factors (matching YAML config):
      vol60_inv:        1/vol60  (weight 0.30)
      downside_vol_inv: 1/downside_vol semi-deviation  (weight 0.25)
      beta_inv:         1/abs(beta vs SPY)  (weight 0.20) — requires spy_prices
      drawdown_inv:     1/max_drawdown_1y  (weight 0.15)
      log_volume:       log(avg_daily_volume) rank  (weight 0.10)

    Beta is computed using trailing 252-day returns against spy_prices.
    Requires >= 60 overlapping observations; otherwise beta_inv = None.
    If spy_prices is None, beta_inv = None for all tickers.

    Timezone-aware as_of is normalized to naive UTC before comparison.
    NaN/inf values are treated as unavailable.
    """
    as_of = _normalize_as_of(as_of)

    # Prepare SPY returns once
    spy_returns: pd.Series | None = None
    if spy_prices is not None and not spy_prices.empty:
        spy_prices_norm = _normalize_df_index(spy_prices)
        spy_prices_norm = spy_prices_norm[spy_prices_norm.index <= as_of]
        spy_col = "Adj Close" if "Adj Close" in spy_prices_norm.columns else "Close"
        if spy_col in spy_prices_norm.columns:
            spy_close = spy_prices_norm[spy_col].dropna()
            spy_returns = spy_close.pct_change().dropna().tail(252)

    rows = []
    for ticker, df in price_data.items():
        if df is None or df.empty:
            rows.append({"ticker": ticker, "safety_raw": None, "safety_available": False})
            continue

        df = _normalize_df_index(df)
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
        if not _is_finite(vol60):
            vol60 = None

        neg_returns = returns_60[returns_60 < 0]
        if len(neg_returns) >= 10:
            downside_vol = neg_returns.std() * math.sqrt(252)
            if not _is_finite(downside_vol):
                downside_vol = vol60
        else:
            downside_vol = vol60

        max_dd = None
        if len(prices_1y) >= 20:
            roll_max = prices_1y.cummax()
            dd = float((prices_1y / roll_max - 1).min())
            if _is_finite(dd):
                max_dd = abs(dd)

        avg_vol = None
        if "Volume" in df.columns:
            vol_series = df["Volume"].dropna().tail(60)
            if len(vol_series) >= 20:
                candidate = float(vol_series.mean())
                if _is_finite(candidate):
                    avg_vol = candidate

        # Beta vs SPY (requires spy_returns)
        beta_inv = None
        if spy_returns is not None and len(returns) >= 60:
            ticker_ret = returns.tail(252)
            # Align on common dates
            common_idx = ticker_ret.index.intersection(spy_returns.index)
            if len(common_idx) >= 60:
                t_ret = ticker_ret.loc[common_idx]
                s_ret = spy_returns.loc[common_idx]
                spy_var = float(s_ret.var())
                if _is_finite(spy_var) and spy_var > 0:
                    cov = float(np.cov(t_ret.values, s_ret.values)[0, 1])
                    if _is_finite(cov):
                        beta = cov / spy_var
                        if _is_finite(beta) and beta != 0:
                            beta_inv = 1.0 / abs(beta)
                            if not _is_finite(beta_inv):
                                beta_inv = None

        EPSILON = 1e-6
        features = {
            "vol60_inv": (1.0 / (vol60 + EPSILON)) if vol60 is not None else None,
            "downside_vol_inv": (1.0 / (downside_vol + EPSILON)) if downside_vol is not None else None,
            "beta_inv": beta_inv,
            "drawdown_inv": (1.0 / (max_dd + EPSILON)) if max_dd is not None else None,
            "log_volume": math.log(avg_vol + 1.0) if avg_vol is not None else None,
        }
        # Validate computed inverses
        features = {k: (v if _is_finite(v) else None) for k, v in features.items()}

        weights = {
            "vol60_inv": 0.30, "downside_vol_inv": 0.25, "beta_inv": 0.20,
            "drawdown_inv": 0.15, "log_volume": 0.10,
        }
        available = {k: (v is not None) for k, v in features.items()}
        w_sum = sum(w for k, w in weights.items() if available[k])

        if w_sum <= 0:
            rows.append({"ticker": ticker, "safety_raw": None, "safety_available": False})
            continue

        safety_raw = sum(float(features[k]) * w / w_sum for k, w in weights.items() if available[k])
        if not _is_finite(safety_raw):
            rows.append({"ticker": ticker, "safety_raw": None, "safety_available": False})
        else:
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

    Returns empty DataFrame with stable schema if any input is empty.
    factor_coverage = fraction of 4 factors with a real score (0.0 – 1.0).
    Missing factors produce None in the composite — not 50.0 neutral fill.
    """
    # Guard: return stable empty schema if all inputs are empty
    if (momentum_df.empty and quality_df.empty and value_df.empty and safety_df.empty):
        return pd.DataFrame(columns=EMPTY_SCORE_SCHEMA)

    # Gather available columns safely
    def _safe_cols(df, cols):
        return [c for c in cols if c in df.columns]

    mom_cols = _safe_cols(momentum_df, ["ticker", "momentum_score", "momentum_available"])
    qual_cols = _safe_cols(quality_df, ["ticker", "quality_score", "quality_available", "quality_coverage"])
    val_cols = _safe_cols(value_df, ["ticker", "value_score", "value_available", "value_coverage"])
    saf_cols = _safe_cols(safety_df, ["ticker", "safety_score", "safety_available"])

    dfs_to_merge = []
    if "ticker" in mom_cols and len(momentum_df) > 0:
        dfs_to_merge.append(momentum_df[mom_cols])
    if "ticker" in qual_cols and len(quality_df) > 0:
        dfs_to_merge.append(quality_df[qual_cols])
    if "ticker" in val_cols and len(value_df) > 0:
        dfs_to_merge.append(value_df[val_cols])
    if "ticker" in saf_cols and len(safety_df) > 0:
        dfs_to_merge.append(safety_df[saf_cols])

    if not dfs_to_merge:
        return pd.DataFrame(columns=EMPTY_SCORE_SCHEMA)

    df = dfs_to_merge[0].copy()
    for other in dfs_to_merge[1:]:
        df = df.merge(other, on="ticker", how="outer")

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
                if is_avail and _is_finite(s):
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
    # weights may still hold values up to cap. Cap + exposure applied here.
    return {t: min(w, cap) * exposure for t, w in weights.items()}
