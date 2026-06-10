"""
Correlation analysis — detects highly-correlated positions and filters candidates.

Usage:
  Signal mode  : compute_correlation_matrix(market_data, tickers) → include pairs in payload
  Execute mode : check_correlation_against_held() in buy loop
                 filter_correlated_sells()  for existing positions
  Weekly report: format_correlation_heatmap()
"""

import numpy as np
import pandas as pd

CORRELATION_THRESHOLD = 0.85
HEATMAP_MIN_THRESHOLD = 0.70
LOOKBACK_DAYS = 60


def compute_correlation_matrix(market_data, tickers):
    """
    Compute 60-day daily return correlations for given tickers.
    Returns pd.DataFrame (tickers × tickers), empty if insufficient data.
    """
    returns = {}
    for ticker in tickers:
        if ticker not in market_data or ticker in ("SPY", "QQQ"):
            continue
        try:
            close = market_data[ticker]["Close"].tail(LOOKBACK_DAYS + 5)
            ret = close.pct_change().dropna()
            if len(ret) >= 40:
                returns[ticker] = ret
        except Exception:
            pass

    if len(returns) < 2:
        return pd.DataFrame()

    df = pd.DataFrame(returns).dropna()
    if len(df) < 30:
        return pd.DataFrame()

    return df.corr()


def find_correlated_pairs(corr_matrix, threshold=HEATMAP_MIN_THRESHOLD):
    """
    Return list of {ticker_a, ticker_b, correlation} for all pairs >= threshold.
    Sorted descending by correlation.
    """
    if corr_matrix.empty:
        return []

    pairs = []
    tickers = corr_matrix.columns.tolist()
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]
            try:
                corr = float(corr_matrix.loc[a, b])
                if not np.isnan(corr) and corr >= threshold:
                    pairs.append({"ticker_a": a, "ticker_b": b, "correlation": round(corr, 4)})
            except Exception:
                pass

    return sorted(pairs, key=lambda x: -x["correlation"])


def check_correlation_against_held(ticker, held_tickers, corr_pairs, threshold=CORRELATION_THRESHOLD):
    """
    Check if ticker is highly correlated (>= threshold) with any held ticker.
    Returns (is_correlated, most_correlated_held_ticker, correlation).
    """
    best_corr = 0.0
    best_match = None

    for pair in corr_pairs:
        if pair["correlation"] < threshold:
            break  # sorted descending — can stop early
        a, b = pair["ticker_a"], pair["ticker_b"]
        if a == ticker and b in held_tickers and pair["correlation"] > best_corr:
            best_corr, best_match = pair["correlation"], b
        elif b == ticker and a in held_tickers and pair["correlation"] > best_corr:
            best_corr, best_match = pair["correlation"], a

    return (True, best_match, best_corr) if best_match else (False, None, 0.0)


def filter_correlated_sells(positions, candidates_by_ticker, corr_pairs, threshold=CORRELATION_THRESHOLD):
    """
    Among currently held positions, find pairs with correlation >= threshold.
    For each pair: flag the lower-scored one for selling.

    Returns list of {sell_ticker, keep_ticker, correlation, reason}.
    """
    if not positions or not corr_pairs:
        return []

    held = set(positions.keys())
    to_sell = []
    flagged = set()

    for pair in corr_pairs:
        if pair["correlation"] < threshold:
            break
        a, b = pair["ticker_a"], pair["ticker_b"]
        if a not in held or b not in held:
            continue
        if a in flagged or b in flagged:
            continue

        score_a = (candidates_by_ticker.get(a) or {}).get("strategy_score") or 0
        score_b = (candidates_by_ticker.get(b) or {}).get("strategy_score") or 0

        sell, keep = (b, a) if score_a >= score_b else (a, b)

        to_sell.append({
            "sell_ticker": sell,
            "keep_ticker": keep,
            "correlation": pair["correlation"],
            "reason": f"Korr {pair['correlation']:.0%} med {keep} — selger svakere ({sell})",
        })
        flagged.add(sell)

    return to_sell


def format_correlation_heatmap(corr_pairs, held_tickers):
    """
    Format notable correlation pairs for the weekly Telegram report.
    Shows pairs where BOTH tickers are held and correlation >= 70%.
    """
    if not corr_pairs or not held_tickers:
        return ""

    held_set = set(held_tickers)
    notable = [
        p for p in corr_pairs
        if p["ticker_a"] in held_set
        and p["ticker_b"] in held_set
        and p["correlation"] >= HEATMAP_MIN_THRESHOLD
    ]

    if not notable:
        return ""

    lines = ["\n🔗 KORRELASJONER I PORTEFØLJE (≥70%):"]
    for p in notable[:15]:
        a, b, corr = p["ticker_a"], p["ticker_b"], p["correlation"]
        filled = int(corr * 16)
        bar = "█" * filled + "░" * (16 - filled)
        flag = " ⚠️" if corr >= CORRELATION_THRESHOLD else ""
        lines.append(f"  {a:6} ↔ {b:6}: {corr:.0%} [{bar}]{flag}")

    return "\n".join(lines)
