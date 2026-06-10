from datetime import datetime

import pandas as pd
import yfinance as yf

from modules.state import safe_float, today_str

EARNINGS_ALERT_DAYS = 14


def _get_next_earnings_date(t):
    try:
        cal = t.calendar
        if cal is None:
            return None
        if isinstance(cal, pd.DataFrame) and cal.empty:
            return None
        if isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.index:
                vals = cal.loc["Earnings Date"].values
                if len(vals) > 0:
                    return str(vals[0])[:10]
            # Some yfinance versions have dates as column headers
            for col in cal.columns:
                val = cal[col].iloc[0] if not cal[col].empty else None
                if val is not None:
                    return str(col)[:10]
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date", [])
            if ed:
                return str(ed[0])[:10]
    except Exception:
        pass
    return None


def _get_beat_history(t):
    try:
        hist = t.earnings_history
        if hist is None or (hasattr(hist, "empty") and hist.empty):
            return None, None
        hist = hist.dropna(subset=["epsActual", "epsEstimate"])
        hist = hist.tail(4)
        if len(hist) < 2:
            return None, None
        beats = (hist["epsActual"] > hist["epsEstimate"]).sum()
        beat_rate = float(beats) / len(hist)
        last = hist.iloc[-1]
        actual = safe_float(last.get("epsActual"))
        estimate = safe_float(last.get("epsEstimate"))
        if actual is not None and estimate is not None and estimate != 0:
            last_surprise = (actual - estimate) / abs(estimate)
        else:
            last_surprise = None
        return beat_rate, last_surprise
    except Exception:
        return None, None


def fetch_earnings_data(ticker):
    try:
        t = yf.Ticker(ticker)
        next_earnings = _get_next_earnings_date(t)
        beat_rate, last_surprise_pct = _get_beat_history(t)

        # Earnings score bonus
        bonus = 0
        if beat_rate is not None and last_surprise_pct is not None:
            if beat_rate > 0.75 and last_surprise_pct > 0.05:
                bonus = 15
            elif beat_rate < 0.50 or last_surprise_pct < -0.05:
                bonus = -10

        # Days to next earnings
        days_to_earnings = None
        if next_earnings:
            try:
                ned = datetime.strptime(next_earnings[:10], "%Y-%m-%d")
                tod = datetime.strptime(today_str(), "%Y-%m-%d")
                days_to_earnings = (ned - tod).days
            except Exception:
                pass

        earnings_soon = (
            days_to_earnings is not None and 0 <= days_to_earnings <= EARNINGS_ALERT_DAYS
        )

        return {
            "beat_rate": beat_rate,
            "last_surprise_pct": last_surprise_pct,
            "earnings_bonus": bonus,
            "next_earnings": next_earnings,
            "days_to_earnings": days_to_earnings,
            "earnings_soon": earnings_soon,
        }
    except Exception as e:
        print(f"Earnings feil {ticker}: {e}")
        return _empty_earnings()


def _empty_earnings():
    return {
        "beat_rate": None,
        "last_surprise_pct": None,
        "earnings_bonus": 0,
        "next_earnings": None,
        "days_to_earnings": None,
        "earnings_soon": False,
    }


def fetch_earnings_bulk(tickers):
    result = {}
    print(f"Henter earnings-data for {len(tickers)} tickere...")
    for i, ticker in enumerate(tickers):
        result[ticker] = fetch_earnings_data(ticker)
        if (i + 1) % 20 == 0:
            print(f"  Earnings: {i+1}/{len(tickers)}")
    print(f"Earnings komplett for {len(result)} tickere")
    return result
