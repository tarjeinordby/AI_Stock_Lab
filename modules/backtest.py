"""
Backtesting engine — simulates all 6 strategies over 2015-2025.

Caveats (documented in results):
  - Survivorship bias: uses the current universe (stocks that still trade today)
  - No historical fundamentals/sentiment → quality/value/sentiment fixed at neutral (50.0)
  - Transaction costs applied via estimate_trade_cost()
  - Monthly rebalancing; live system rebalances daily
  - MegaCap_AI treated as a standard strategy (no megacap bias flag in backtest)
"""

import math
import os

import numpy as np
import pandas as pd
import yfinance as yf

from modules.portfolio import MIN_POSITION_VALUE, build_target_weights, estimate_trade_cost
from modules.scoring import STRATEGIES, get_regime_weights, percentile_rank
from modules.state import DATA_DIR, START_CAPITAL, now_utc, save_json

BACKTEST_DIR = f"{DATA_DIR}/backtest"
BACKTEST_RESULTS_FILE = f"{BACKTEST_DIR}/results.json"

# Representative universe — covers all sectors, avoids 600-ticker download
BACKTEST_UNIVERSE = [
    "SPY", "QQQ",
    # Tech megacaps
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO",
    "TSLA", "ORCL", "AMD", "CRM", "ADBE", "QCOM", "CSCO",
    "INTC", "TXN", "MU", "AMAT", "LRCX",
    # Finance
    "JPM", "V", "MA", "BAC", "GS", "MS", "BRK-B",
    # Healthcare
    "LLY", "JNJ", "PFE", "UNH", "ABBV",
    # Consumer
    "COST", "WMT", "HD", "MCD", "KO", "PEP", "NKE",
    # Energy
    "XOM", "CVX",
    # Industrials
    "BA", "CAT", "HON", "GE",
    # Telecom/Media
    "DIS", "NFLX", "T", "VZ",
    # International
    "TSM", "ASML", "NVO",
    # High-growth watchlist (may not have 10Y history)
    "PLTR", "SHOP", "SNOW", "DDOG", "NET", "CRWD",
    "PANW", "ZS", "COIN", "ROKU", "UBER", "ABNB",
    "DELL", "APP", "MDB", "CELH", "SOFI",
    # Autos
    "F", "GM",
]


def _ensure_dir():
    os.makedirs(BACKTEST_DIR, exist_ok=True)


def _download_data(tickers, start="2014-01-01", end="2025-12-31"):
    """
    Download daily OHLCV for all tickers. Returns {ticker: df}.
    Uses period="max" + post-filter (avoids SSL issues with start/end on some systems).
    """
    out = {}
    chunk_size = 60
    total = len(tickers)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    for i in range(0, total, chunk_size):
        chunk = tickers[i:i + chunk_size]
        n_chunk = i // chunk_size + 1
        n_total = math.ceil(total / chunk_size)
        print(f"  Chunk {n_chunk}/{n_total} ({len(chunk)} tickere)...")
        try:
            raw = yf.download(
                tickers=chunk,
                period="max",
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                progress=False,
                threads=True,
            )
            if raw is None or raw.empty:
                continue
            for ticker in chunk:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        if ticker not in raw.columns.get_level_values(0):
                            continue
                        df = raw[ticker].copy()
                    else:
                        df = raw.copy()
                    df = df[["Close", "Volume"]].dropna()
                    # Filter to requested date range
                    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
                    if len(df) >= 200:
                        out[ticker] = df
                except Exception:
                    pass
        except Exception as e:
            print(f"  Chunk {n_chunk} feil: {e}")

    print(f"  Hentet data for {len(out)}/{total} tickere")
    return out


def _get_price_at(close_series, date):
    """Return last available close price on or before date."""
    try:
        avail = close_series.index[close_series.index <= date]
        if avail.empty:
            return None
        return float(close_series.loc[avail[-1]])
    except Exception:
        return None


def _detect_regime(spy_close, date):
    """Detect market regime using SPY data up to given date (no VIX in backtest)."""
    try:
        data = spy_close.loc[spy_close.index <= date].tail(260)
        if len(data) < 200:
            return "unknown"
        sma200 = float(data.rolling(200).mean().iloc[-1])
        sma50 = float(data.rolling(50).mean().iloc[-1])
        price = float(data.iloc[-1])
        ret_3m = (price / float(data.iloc[-min(63, len(data))])) - 1 if len(data) >= 63 else 0
        if price > sma200 and price > sma50 and ret_3m > 0.05:
            return "explosive"
        if price > sma200 and ret_3m > 0:
            return "bullish"
        if price > sma200 or price > sma50:
            return "neutral"
        return "defensive"
    except Exception:
        return "unknown"


def _compute_technicals(price_data, tickers, date):
    """
    Compute momentum-only technical scores for all tickers at given date.
    Returns list of dicts compatible with build_target_weights.
    """
    scored = []
    spy_close = price_data.get("SPY", pd.DataFrame()).get("Close", pd.Series())

    spy_r3 = None
    if not spy_close.empty:
        spy_hist = spy_close.loc[spy_close.index <= date]
        if len(spy_hist) >= 63:
            spy_r3 = float(spy_hist.iloc[-1]) / float(spy_hist.iloc[-63]) - 1

    for ticker in tickers:
        if ticker in ("SPY", "QQQ") or ticker not in price_data:
            continue
        try:
            df = price_data[ticker]
            close = df["Close"].loc[df.index <= date]
            if len(close) < 260:
                continue

            price = float(close.iloc[-1])
            if price < 5:
                continue

            # Dollar volume filter (relaxed for historical data)
            vol = df["Volume"].loc[df.index <= date].tail(20)
            if float(vol.mean()) * price < 5_000_000:
                continue

            def r(n):
                if len(close) <= n:
                    return None
                prev = float(close.iloc[-n])
                return (price / prev - 1) if prev > 0 else None

            ret_1m = r(21) or 0.0
            ret_3m = r(63) or 0.0
            ret_6m = r(126) or 0.0

            p_1m = float(close.iloc[-21]) if len(close) > 21 else price
            p_6m = float(close.iloc[-126]) if len(close) > 126 else price
            p_12m = float(close.iloc[-252]) if len(close) > 252 else price

            mom_12_1 = (p_1m / p_12m - 1) if p_12m > 0 else 0.0
            mom_6_1 = (p_1m / p_6m - 1) if p_6m > 0 else 0.0

            sma200 = float(close.rolling(200).mean().iloc[-1])
            sma50 = float(close.rolling(50).mean().iloc[-1])
            above_sma200 = price > sma200

            daily_ret = close.pct_change().dropna()
            vol60 = float(daily_ret.tail(60).std() * (252 ** 0.5))
            vol60 = max(0.10, min(1.50, vol60))

            # RSI (simplified)
            delta = close.diff().tail(15)
            g = float(delta.clip(lower=0).mean())
            l = float((-delta.clip(upper=0)).mean())
            rsi = (100 - 100 / (1 + g / l)) if l > 0 else 50.0

            rel_str = (ret_3m - spy_r3) if spy_r3 is not None else ret_3m

            scored.append({
                "ticker": ticker,
                "price": price,
                "ret_1m": ret_1m,
                "ret_3m": ret_3m,
                "ret_6m": ret_6m,
                "ret_12m": r(252) or 0.0,
                "mom_12_1": mom_12_1,
                "mom_6_1": mom_6_1,
                "mom_3_1": ret_3m,
                "relative_strength_3m": rel_str,
                "vol60": vol60,
                "vol20": vol60,
                "above_sma200": above_sma200,
                "above_sma50": price > sma50,
                "overbought": rsi > 75,
                "very_weak_rsi": rsi < 30,
                "is_megacap": False,
                "sector": "Unknown",
                # Neutral fundamentals/sentiment (no historical data)
                "quality_score": 50.0,
                "value_score": 50.0,
                "sentiment_score": 50.0,
                "score_percentile": 0.5,
            })
        except Exception:
            pass

    if len(scored) < 10:
        return []

    # Compute momentum factor scores
    df_s = pd.DataFrame(scored).set_index("ticker")
    df_s["rank_mom_12_1"] = percentile_rank(df_s["mom_12_1"], True)
    df_s["rank_mom_6_1"] = percentile_rank(df_s["mom_6_1"], True)
    df_s["rank_mom_3_1"] = percentile_rank(df_s["mom_3_1"], True)
    df_s["rank_ret_1m"] = percentile_rank(df_s["ret_1m"], True)
    df_s["rank_rs"] = percentile_rank(df_s["relative_strength_3m"].fillna(0), True)

    df_s["momentum_score"] = (
        0.40 * df_s["rank_mom_12_1"]
        + 0.25 * df_s["rank_mom_6_1"]
        + 0.20 * df_s["rank_rs"]
        + 0.15 * df_s["rank_ret_1m"]
    ) * 100

    return df_s.reset_index().to_dict(orient="records")


def _build_candidates(scored, strategy_name, config, regime):
    """Build scored candidate list for one strategy at one point in time."""
    if not scored:
        return []

    df = pd.DataFrame(scored).set_index("ticker")
    weights = get_regime_weights(regime, macro_inverted=False)
    col = config["score_column"]

    df[col] = (
        weights["momentum"] * df["momentum_score"]
        + weights["quality"] * df["quality_score"]
        + weights["value"] * df["value_score"]
        + weights["sentiment"] * df["sentiment_score"]
    )

    # Standard penalties
    df.loc[~df["above_sma200"], col] *= 0.45
    df.loc[df["overbought"], col] *= 0.90
    df.loc[df["very_weak_rsi"], col] *= 0.70
    df.loc[df["vol60"] > 1.20, col] *= 0.80

    df["strategy_score"] = df[col]
    df["score_percentile"] = df[col].rank(pct=True)
    df["rank"] = df[col].rank(ascending=False)

    return df.sort_values(col, ascending=False).reset_index().to_dict(orient="records")


def _simulate_strategy(price_data, spy_close, rebalance_dates, strategy_name, config):
    """
    Simulate one strategy over all rebalancing dates.
    Returns list of {date: str, value: float} monthly snapshots.
    """
    tickers = [t for t in price_data if t not in ("SPY", "QQQ")]
    portfolio = {"cash": float(START_CAPITAL), "positions": {}}
    monthly_values = []

    for date in rebalance_dates:
        try:
            # Score with data available up to this date (no look-ahead)
            scored = _compute_technicals(price_data, tickers, date)
            regime = _detect_regime(spy_close, date)
            candidates = _build_candidates(scored, strategy_name, config, regime) if scored else []

            # Update market values of held positions
            for ticker, pos in list(portfolio["positions"].items()):
                p = _get_price_at(price_data[ticker]["Close"], date) if ticker in price_data else None
                if p and p > 0:
                    pos["last_price"] = p
                    pos["market_value"] = pos["shares"] * p

            total = portfolio["cash"] + sum(
                p.get("market_value", 0) for p in portfolio["positions"].values()
            )

            if candidates:
                target_weights = build_target_weights(candidates, config, regime, portfolio_drawdown=0.0)
                target_set = set(target_weights.keys())

                # Sell positions no longer in target
                for ticker in list(portfolio["positions"].keys()):
                    if ticker not in target_set:
                        p = _get_price_at(price_data[ticker]["Close"], date) if ticker in price_data else None
                        if p and p > 0:
                            value = portfolio["positions"][ticker]["shares"] * p
                            cost = estimate_trade_cost(value, "SELL")
                            portfolio["cash"] += value - cost
                        else:
                            portfolio["cash"] += portfolio["positions"][ticker].get("market_value", 0)
                        del portfolio["positions"][ticker]

                # Re-valuate after sells
                total = portfolio["cash"] + sum(
                    p.get("market_value", 0) for p in portfolio["positions"].values()
                )

                # Buy new positions
                for ticker, weight in sorted(target_weights.items(), key=lambda x: -x[1]):
                    if ticker in portfolio["positions"]:
                        continue
                    if len(portfolio["positions"]) >= config["max_positions"]:
                        break
                    if ticker not in price_data:
                        continue
                    p = _get_price_at(price_data[ticker]["Close"], date)
                    if not p or p <= 0:
                        continue
                    target_val = min(total * weight, portfolio["cash"] * 0.99)
                    if target_val < MIN_POSITION_VALUE:
                        continue
                    cost = estimate_trade_cost(target_val, "BUY")
                    if target_val + cost > portfolio["cash"]:
                        continue
                    shares = target_val / p
                    portfolio["cash"] -= target_val + cost
                    portfolio["positions"][ticker] = {
                        "shares": shares,
                        "avg_price": p,
                        "last_price": p,
                        "market_value": shares * p,
                    }

            # End-of-month valuation
            final = portfolio["cash"]
            for ticker, pos in portfolio["positions"].items():
                p = _get_price_at(price_data[ticker]["Close"], date) if ticker in price_data else None
                if p and p > 0:
                    pos["last_price"] = p
                    pos["market_value"] = pos["shares"] * p
                final += pos.get("market_value", 0)

            monthly_values.append({"date": str(date.date()), "value": round(final, 2)})

        except Exception as e:
            last = monthly_values[-1]["value"] if monthly_values else float(START_CAPITAL)
            monthly_values.append({"date": str(date.date()), "value": last})
            print(f"  Feil {strategy_name} {date.date()}: {e}")

    return monthly_values


def _compute_stats(monthly_values, spy_monthly_values, strategy_name):
    """Compute summary statistics from monthly value series."""
    if len(monthly_values) < 6:
        return {}

    values = pd.Series([v["value"] for v in monthly_values])
    dates = pd.to_datetime([v["date"] for v in monthly_values])
    spy_vals = pd.Series([v["value"] for v in spy_monthly_values[:len(monthly_values)]])

    returns = values.pct_change().dropna()
    spy_returns = spy_vals.pct_change().dropna()

    total_return = float(values.iloc[-1] / values.iloc[0] - 1)
    spy_total = float(spy_vals.iloc[-1] / spy_vals.iloc[0] - 1)

    # Sharpe (annualized from monthly returns)
    sharpe = 0.0
    if len(returns) >= 12 and float(returns.std()) > 0:
        sharpe = float(returns.mean() / returns.std() * math.sqrt(12))

    # Max drawdown
    cum = values / values.iloc[0]
    peak = cum.expanding().max()
    max_drawdown = float(((cum / peak) - 1).min())

    # Win rate
    win_rate = float((returns > 0).mean()) if len(returns) > 0 else 0.0

    # Annual returns
    df_yr = pd.DataFrame({"value": values.values, "year": dates.year})
    yearly = {}
    for yr, grp in df_yr.groupby("year"):
        if len(grp) >= 2:
            yearly[int(yr)] = round(float(grp["value"].iloc[-1] / grp["value"].iloc[0] - 1), 4)

    best = max(yearly.items(), key=lambda x: x[1]) if yearly else (None, None)
    worst = min(yearly.items(), key=lambda x: x[1]) if yearly else (None, None)

    return {
        "strategy": strategy_name,
        "total_return": round(total_return, 4),
        "spy_total_return": round(spy_total, 4),
        "alpha": round(total_return - spy_total, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_drawdown, 4),
        "win_rate": round(win_rate, 4),
        "num_months": int(len(returns)),
        "best_year": {"year": best[0], "return": best[1]} if best[0] else None,
        "worst_year": {"year": worst[0], "return": worst[1]} if worst[0] else None,
        "yearly_returns": yearly,
    }


def run_backtest():
    """
    Simulate all 6 strategies from 2015-2025.
    Saves to data_v3/backtest/results.json and prints summary table.
    """
    _ensure_dir()

    print("=" * 60)
    print("BACKTEST MODE  (2015-01-01  →  2025-12-31)")
    print("=" * 60)
    print("Merk: Survivorship bias. Ingen historiske fundamentals/sentiment.")
    print()

    # Download historical price data
    print("1. Laster ned prisdata (10 år)...")
    price_data = _download_data(BACKTEST_UNIVERSE, start="2014-01-01", end="2025-12-31")

    if "SPY" not in price_data:
        raise RuntimeError("SPY-data mangler — kan ikke kjøre backtest.")

    spy_close = price_data["SPY"]["Close"]

    # Monthly rebalancing dates (first trading day of each month)
    print("\n2. Bygger rebalanseringskalender...")
    monthly_starts = pd.date_range(start="2015-01-01", end="2025-12-01", freq="MS")
    rebalance_dates = []
    for d in monthly_starts:
        avail = spy_close.index[spy_close.index >= d]
        if not avail.empty:
            rebalance_dates.append(avail[0])
    print(f"   {len(rebalance_dates)} rebalanseringsdatoer")

    # SPY buy-and-hold benchmark (constant shares from start)
    spy_start_price = _get_price_at(spy_close, rebalance_dates[0])
    spy_monthly = []
    for date in rebalance_dates:
        p = _get_price_at(spy_close, date)
        val = (START_CAPITAL * p / spy_start_price) if (p and spy_start_price) else START_CAPITAL
        spy_monthly.append({"date": str(date.date()), "value": round(val, 2)})

    # Simulate each strategy
    print("\n3. Simulerer strategier (månedlig rebalansering)...")
    all_stats = {}
    all_series = {}

    for name, config in STRATEGIES.items():
        print(f"   {name}...")
        series = _simulate_strategy(price_data, spy_close, rebalance_dates, name, config)
        all_series[name] = series
        stats = _compute_stats(series, spy_monthly, name)
        all_stats[name] = stats

        t = stats.get("total_return", 0) * 100
        a = stats.get("alpha", 0) * 100
        sh = stats.get("sharpe", 0)
        mdd = stats.get("max_drawdown", 0) * 100
        wr = stats.get("win_rate", 0) * 100
        print(f"   → {t:+.1f}% | α {a:+.1f}% | Sharpe {sh:.2f} | MaxDD {mdd:.1f}% | WinR {wr:.0f}%")

    # Save results
    payload = {
        "generated_at_utc": now_utc().isoformat(),
        "backtest_period": {"start": "2015-01-01", "end": "2025-12-31"},
        "universe_size": len([t for t in price_data if t not in ("SPY", "QQQ")]),
        "caveats": [
            "Survivorship bias: kun aksjer som eksisterer i dag",
            "Ingen historiske fundamentals/sentiment — neutral 50.0 brukt",
            "Månedlig rebalansering (live-system rebalanserer daglig)",
        ],
        "spy_benchmark": {
            "total_return": round(float(spy_close.iloc[-1] / spy_close.iloc[0] - 1), 4),
            "monthly_values": spy_monthly,
        },
        "strategies": all_stats,
        "monthly_series": all_series,
    }

    save_json(payload, BACKTEST_RESULTS_FILE)

    # Summary table
    print("\n" + "=" * 72)
    print(f"{'Strategi':<24} {'Total':>8} {'Alpha':>8} {'Sharpe':>7} {'MaxDD':>8} {'Win%':>6}")
    print("-" * 72)
    for name, s in sorted(all_stats.items(), key=lambda x: -x[1].get("total_return", 0)):
        t = s.get("total_return", 0) * 100
        a = s.get("alpha", 0) * 100
        sh = s.get("sharpe", 0)
        mdd = s.get("max_drawdown", 0) * 100
        wr = s.get("win_rate", 0) * 100
        print(f"{name:<24} {t:>+7.1f}%  {a:>+7.1f}%  {sh:>6.2f}  {mdd:>7.1f}%  {wr:>5.0f}%")
    spy_total = float(spy_close.iloc[-1] / spy_close.iloc[0] - 1) * 100
    print(f"{'SPY (buy & hold)':<24} {spy_total:>+7.1f}%")
    print("=" * 72)
    print(f"\nResultater lagret: {BACKTEST_RESULTS_FILE}")
