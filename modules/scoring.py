import numpy as np
import pandas as pd

from modules.state import safe_float
from modules.universe import MEGACAP_TICKERS

# ============================================================
# STRATEGY CONFIGURATION — single source of truth
# ============================================================

STRATEGIES = {
    "Max_Return_AI": {
        "description": "Mest aggressiv. Jakter høyest mulig avkastning med mer risiko.",
        "score_column": "score_max_return",
        "weights": {"momentum": 0.45, "quality": 0.30, "value": 0.20, "sentiment": 0.05},
        "max_positions": 10,
        "max_position_weight": 0.15,
        "max_new_buys_per_week": 5,
        "buy_top_n": 20,
        "min_score_percentile": 0.80,
        "sell_rank_threshold": 120,
        "stop_loss": -0.22,
        "trailing_stop": -0.28,
        "buyback_cooldown_days": 7,
        "exposure": {
            "explosive": 1.00,
            "bullish": 1.00,
            "neutral": 0.70,
            "defensive": 0.40,
            "unknown": 0.60,
        },
    },
    "Quality_Momentum_AI": {
        "description": "Sterk momentum, men med mer kvalitets- og risikokontroll.",
        "score_column": "score_quality_momentum",
        "weights": {"momentum": 0.35, "quality": 0.40, "value": 0.20, "sentiment": 0.05},
        "max_positions": 10,
        "max_position_weight": 0.13,
        "max_new_buys_per_week": 4,
        "buy_top_n": 20,
        "min_score_percentile": 0.78,
        "sell_rank_threshold": 110,
        "stop_loss": -0.18,
        "trailing_stop": -0.25,
        "buyback_cooldown_days": 10,
        "exposure": {
            "explosive": 0.95,
            "bullish": 0.90,
            "neutral": 0.65,
            "defensive": 0.35,
            "unknown": 0.55,
        },
    },
    "Balanced_AI": {
        "description": "Balanserer momentum, trend, risiko og stabilitet.",
        "score_column": "score_balanced",
        "weights": {"momentum": 0.35, "quality": 0.40, "value": 0.20, "sentiment": 0.05},
        "max_positions": 10,
        "max_position_weight": 0.12,
        "max_new_buys_per_week": 4,
        "buy_top_n": 18,
        "min_score_percentile": 0.75,
        "sell_rank_threshold": 110,
        "stop_loss": -0.18,
        "trailing_stop": -0.25,
        "buyback_cooldown_days": 10,
        "exposure": {
            "explosive": 0.90,
            "bullish": 0.85,
            "neutral": 0.60,
            "defensive": 0.30,
            "unknown": 0.50,
        },
    },
    "Low_Risk_AI": {
        "description": "Prøver å slå indeks med lavere svingninger og mindre drawdown.",
        "score_column": "score_low_risk",
        "weights": {"momentum": 0.25, "quality": 0.45, "value": 0.25, "sentiment": 0.05},
        "max_positions": 12,
        "max_position_weight": 0.10,
        "max_new_buys_per_week": 3,
        "buy_top_n": 25,
        "min_score_percentile": 0.70,
        "sell_rank_threshold": 100,
        "stop_loss": -0.15,
        "trailing_stop": -0.22,
        "buyback_cooldown_days": 10,
        "exposure": {
            "explosive": 0.80,
            "bullish": 0.75,
            "neutral": 0.50,
            "defensive": 0.25,
            "unknown": 0.45,
        },
    },
    "MegaCap_AI": {
        "description": "Fokuserer på store, robuste kvalitetsselskaper.",
        "score_column": "score_megacap",
        "weights": {"momentum": 0.30, "quality": 0.45, "value": 0.20, "sentiment": 0.05},
        "max_positions": 8,
        "max_position_weight": 0.15,
        "max_new_buys_per_week": 3,
        "buy_top_n": 25,
        "min_score_percentile": 0.65,
        "sell_rank_threshold": 100,
        "stop_loss": -0.18,
        "trailing_stop": -0.25,
        "buyback_cooldown_days": 10,
        "exposure": {
            "explosive": 0.90,
            "bullish": 0.85,
            "neutral": 0.55,
            "defensive": 0.30,
            "unknown": 0.50,
        },
    },
    "AI_Sentiment_AI": {
        "description": "Tester earnings-quality-signal (beat/miss-historikk) som differensiator.",
        "score_column": "score_ai_sentiment",
        "weights": {"momentum": 0.30, "quality": 0.45, "value": 0.15, "sentiment": 0.10},
        "max_positions": 10,
        "max_position_weight": 0.13,
        "max_new_buys_per_week": 4,
        "buy_top_n": 20,
        "min_score_percentile": 0.75,
        "sell_rank_threshold": 110,
        "stop_loss": -0.18,
        "trailing_stop": -0.25,
        "buyback_cooldown_days": 7,
        "exposure": {
            "explosive": 0.95,
            "bullish": 0.90,
            "neutral": 0.60,
            "defensive": 0.30,
            "unknown": 0.50,
        },
    },
}

TOP_CANDIDATES_TO_SAVE = 150
TOP_CANDIDATES_TO_REPORT = 5

# ============================================================
# DYNAMIC REGIME WEIGHTS
# ============================================================

REGIME_WEIGHTS = {
    "explosive": {"momentum": 0.45, "quality": 0.35, "value": 0.15, "sentiment": 0.05},
    "bullish":   {"momentum": 0.35, "quality": 0.40, "value": 0.20, "sentiment": 0.05},
    "neutral":   {"momentum": 0.25, "quality": 0.45, "value": 0.25, "sentiment": 0.05},
    "defensive": {"momentum": 0.15, "quality": 0.50, "value": 0.30, "sentiment": 0.05},
    "unknown":   {"momentum": 0.30, "quality": 0.40, "value": 0.25, "sentiment": 0.05},
}

_MACRO_INVERTED_ADJ = {"momentum": -0.10, "quality": +0.10, "value": 0.0, "sentiment": 0.0}


def get_regime_weights(regime, macro_inverted=False):
    """
    Return pure regime-based weights (used for display and backtest).
    If macro_inverted (yield curve < -0.5%): shift -10% momentum → +10% quality.
    """
    w = dict(REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["unknown"]))
    if macro_inverted:
        for k in _MACRO_INVERTED_ADJ:
            w[k] = max(0.0, w[k] + _MACRO_INVERTED_ADJ[k])
        total = sum(w.values())
        if total > 0:
            w = {k: v / total for k, v in w.items()}
    return w


def get_effective_weights(strategy_weights, regime, macro_inverted=False):
    """
    Blend each strategy's base weights 50/50 with the regime-based weights.
    Preserves per-strategy differentiation (e.g. AI_Sentiment_AI's 30% sentiment)
    while incorporating regime direction (more momentum in bull, more quality in bear).
    """
    rw = get_regime_weights(regime, macro_inverted)
    blended = {k: 0.50 * strategy_weights.get(k, 0) + 0.50 * rw.get(k, 0) for k in rw}
    total = sum(blended.values())
    if total <= 0:
        return strategy_weights
    return {k: v / total for k, v in blended.items()}


# ============================================================
# FACTOR SCORING HELPERS
# ============================================================

def percentile_rank(series, higher_is_better=True):
    s = pd.Series(series).astype(float)
    if not higher_is_better:
        s = -s
    return s.rank(pct=True)


def _quality_sub_score(roe, gross_margin, debt_equity, earnings_growth):
    parts = []

    if roe is not None:
        if roe > 0.20:
            parts.append(1.00)
        elif roe > 0.15:
            parts.append(0.85)
        elif roe > 0.10:
            parts.append(0.65)
        elif roe > 0.0:
            parts.append(0.40)
        else:
            parts.append(0.10)

    if gross_margin is not None:
        if gross_margin > 0.60:
            parts.append(1.00)
        elif gross_margin > 0.40:
            parts.append(0.80)
        elif gross_margin > 0.25:
            parts.append(0.60)
        elif gross_margin > 0.10:
            parts.append(0.35)
        else:
            parts.append(0.10)

    if debt_equity is not None:
        if debt_equity < 0.30:
            parts.append(1.00)
        elif debt_equity < 0.70:
            parts.append(0.80)
        elif debt_equity < 1.00:
            parts.append(0.60)
        elif debt_equity < 1.50:
            parts.append(0.35)
        else:
            parts.append(0.10)

    if earnings_growth is not None:
        if earnings_growth > 0.30:
            parts.append(1.00)
        elif earnings_growth > 0.20:
            parts.append(0.85)
        elif earnings_growth > 0.10:
            parts.append(0.65)
        elif earnings_growth > 0.0:
            parts.append(0.40)
        else:
            parts.append(0.10)

    return sum(parts) / len(parts) if parts else None


def _value_sub_score(forward_pe, peg_ratio, price_to_sales):
    parts = []

    if forward_pe is not None and forward_pe > 0:
        if 8 <= forward_pe <= 15:
            parts.append(1.00)
        elif 15 < forward_pe <= 25:
            parts.append(0.75)
        elif 25 < forward_pe <= 35:
            parts.append(0.50)
        elif forward_pe < 8:
            parts.append(0.30)  # too low may signal problems
        else:
            parts.append(0.20)

    if peg_ratio is not None and peg_ratio > 0:
        if peg_ratio < 1.0:
            parts.append(1.00)
        elif peg_ratio < 1.5:
            parts.append(0.80)
        elif peg_ratio < 2.0:
            parts.append(0.60)
        elif peg_ratio < 3.0:
            parts.append(0.35)
        else:
            parts.append(0.15)

    if price_to_sales is not None and price_to_sales > 0:
        if price_to_sales < 2:
            parts.append(1.00)
        elif price_to_sales < 5:
            parts.append(0.75)
        elif price_to_sales < 10:
            parts.append(0.50)
        elif price_to_sales < 20:
            parts.append(0.30)
        else:
            parts.append(0.10)

    return sum(parts) / len(parts) if parts else None


# ============================================================
# MAIN SCORING ENGINE
# ============================================================

def build_score_table(analyzed_stocks, fundamentals, sentiment_scores, earnings_data,
                      regime="bullish", macro_inverted=False):
    df = pd.DataFrame(analyzed_stocks)
    if df.empty:
        return df

    df = df.set_index("ticker")

    # ---- Momentum factor (0-100) ----
    for col in ["mom_12_1", "mom_6_1", "mom_3_1", "ret_1m"]:
        if col in df.columns:
            lo, hi = df[col].quantile(0.02), df[col].quantile(0.98)
            df[col] = df[col].clip(lo, hi)

    df["rank_mom_12_1"] = percentile_rank(df["mom_12_1"], True)
    df["rank_mom_6_1"] = percentile_rank(df["mom_6_1"], True)
    df["rank_mom_3_1"] = percentile_rank(df["mom_3_1"], True)
    df["rank_ret_1m"] = percentile_rank(df["ret_1m"], True)
    df["rank_rs"] = percentile_rank(df["relative_strength_3m"].fillna(0), True)

    df["momentum_score"] = (
        0.40 * df["rank_mom_12_1"]
        + 0.25 * df["rank_mom_6_1"]
        + 0.20 * df["rank_rs"]
        + 0.15 * df["rank_ret_1m"]
    ) * 100

    # ---- Quality factor (0-100) ----
    quality_raw = []
    for ticker in df.index:
        f = fundamentals.get(ticker, {})
        q = _quality_sub_score(
            f.get("roe"),
            f.get("gross_margin"),
            f.get("debt_equity"),
            f.get("earnings_growth"),
        )
        quality_raw.append(q)
    df["quality_raw"] = quality_raw
    q_median = df["quality_raw"].median()
    df["quality_score"] = df["quality_raw"].fillna(q_median) * 100

    # ---- Value/Growth factor (0-100) ----
    value_raw = []
    for ticker in df.index:
        f = fundamentals.get(ticker, {})
        v = _value_sub_score(
            f.get("forward_pe"),
            f.get("peg_ratio"),
            f.get("price_to_sales"),
        )
        value_raw.append(v)
    df["value_raw"] = value_raw
    v_median = df["value_raw"].median()
    df["value_score"] = df["value_raw"].fillna(v_median) * 100

    # ---- Sector data ----
    df["sector"] = [
        (fundamentals.get(t) or {}).get("sector", "Unknown")
        for t in df.index
    ]

    # ---- Sentiment factor (0-100) ----
    sent_raw = []
    for ticker in df.index:
        sent = sentiment_scores.get(ticker, {})
        earn = earnings_data.get(ticker, {})

        sent_score = safe_float(sent.get("score"), 0.5)
        earn_bonus = safe_float(earn.get("earnings_bonus"), 0)
        earn_contrib = 0.5 + earn_bonus / 100.0  # center at 0.5

        combined = max(0.0, min(1.0, 0.25 * sent_score + 0.75 * earn_contrib))
        sent_raw.append(combined)
    df["sentiment_raw"] = sent_raw
    s_median = df["sentiment_raw"].median()
    df["sentiment_score"] = df["sentiment_raw"].fillna(s_median) * 100

    # ---- Earnings flags ----
    df["earnings_soon"] = [
        (earnings_data.get(t) or {}).get("earnings_soon", False) for t in df.index
    ]
    df["next_earnings"] = [
        (earnings_data.get(t) or {}).get("next_earnings") for t in df.index
    ]
    df["days_to_earnings"] = [
        (earnings_data.get(t) or {}).get("days_to_earnings") for t in df.index
    ]

    # ---- Per-strategy scores (regime-adjusted weights) ----
    for strat_name, config in STRATEGIES.items():
        w = get_effective_weights(config["weights"], regime, macro_inverted)
        col = config["score_column"]

        df[col] = (
            w["momentum"] * df["momentum_score"]
            + w["quality"] * df["quality_score"]
            + w["value"] * df["value_score"]
            + w["sentiment"] * df["sentiment_score"]
        )

        # MegaCap: heavily penalize non-megacap stocks
        if strat_name == "MegaCap_AI":
            df.loc[~df["is_megacap"], col] *= 0.10

        # Global penalties
        df.loc[~df["above_sma200"], col] *= 0.45
        df.loc[df["overbought"], col] *= 0.90
        df.loc[df["very_weak_rsi"], col] *= 0.70
        df.loc[df["vol60"] > 1.20, col] *= 0.80

    return df


def make_strategy_candidates(score_df, strategy_name, config):
    col = config["score_column"]
    df = score_df.copy().sort_values(col, ascending=False)
    df["strategy_score"] = df[col]
    df["rank"] = range(1, len(df) + 1)
    df["score_percentile"] = df[col].rank(pct=True)

    candidates = df.head(TOP_CANDIDATES_TO_SAVE).reset_index().to_dict(orient="records")
    return candidates
