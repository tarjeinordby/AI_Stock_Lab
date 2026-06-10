import math

import numpy as np
import pandas as pd

from modules.market_data import MIN_AVG_DOLLAR_VOLUME, MIN_PRICE
from modules.state import pct_change
from modules.universe import MEGACAP_TICKERS


def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def analyze_stock(ticker, df, spy_ret_3m=None):
    try:
        close = df["Close"]
        volume = df["Volume"]
        price = float(close.iloc[-1])

        if price < MIN_PRICE:
            return None

        avg_volume_20 = float(volume.tail(20).mean())
        avg_dollar_volume = avg_volume_20 * price
        if avg_dollar_volume < MIN_AVG_DOLLAR_VOLUME:
            return None

        if len(close) < 252:
            return None

        sma50 = float(close.rolling(50).mean().iloc[-1])
        sma100 = float(close.rolling(100).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1])

        rsi = float(calculate_rsi(close).iloc[-1])

        price_1m = float(close.iloc[-21])
        price_3m = float(close.iloc[-63])
        price_6m = float(close.iloc[-126])
        price_12m = float(close.iloc[-252])

        ret_1m = pct_change(price, price_1m)
        ret_3m = pct_change(price, price_3m)
        ret_6m = pct_change(price, price_6m)
        ret_12m = pct_change(price, price_12m)
        mom_12_1 = pct_change(price_1m, price_12m)
        mom_6_1 = pct_change(price_1m, price_6m)
        mom_3_1 = pct_change(price_1m, price_3m)

        if None in [ret_1m, ret_3m, ret_6m, ret_12m, mom_12_1, mom_6_1, mom_3_1]:
            return None

        daily_returns = close.pct_change().dropna()
        vol60 = float(daily_returns.tail(60).std() * math.sqrt(252))
        vol20 = float(daily_returns.tail(20).std() * math.sqrt(252))

        high_52w = float(close.tail(252).max())
        low_52w = float(close.tail(252).min())

        relative_strength_3m = None
        if spy_ret_3m is not None and ticker not in ("SPY", "QQQ"):
            relative_strength_3m = ret_3m - spy_ret_3m

        trend_score = 0
        if price > sma50:
            trend_score += 1
        if price > sma100:
            trend_score += 1
        if price > sma200:
            trend_score += 2
        if sma50 > sma100 > sma200:
            trend_score += 2
        if price > high_52w * 0.90:
            trend_score += 1

        return {
            "ticker": ticker,
            "price": price,
            "ret_1m": ret_1m,
            "ret_3m": ret_3m,
            "ret_6m": ret_6m,
            "ret_12m": ret_12m,
            "mom_12_1": mom_12_1,
            "mom_6_1": mom_6_1,
            "mom_3_1": mom_3_1,
            "relative_strength_3m": relative_strength_3m,
            "vol60": vol60,
            "vol20": vol20,
            "rsi": rsi,
            "distance_from_high": pct_change(price, high_52w),
            "distance_from_low": pct_change(price, low_52w),
            "drawdown_3m": pct_change(price, float(close.tail(63).max())),
            "drawdown_6m": pct_change(price, float(close.tail(126).max())),
            "trend_score": trend_score,
            "above_sma200": price > sma200,
            "above_sma50": price > sma50,
            "healthy_rsi": 45 <= rsi <= 72,
            "overbought": rsi > 78,
            "very_weak_rsi": rsi < 38,
            "near_high": price > high_52w * 0.90,
            "avg_dollar_volume": avg_dollar_volume,
            "sma50": sma50,
            "sma100": sma100,
            "sma200": sma200,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "is_megacap": ticker in MEGACAP_TICKERS,
        }
    except Exception as e:
        print(f"Teknisk analysefeil {ticker}: {e}")
        return None
