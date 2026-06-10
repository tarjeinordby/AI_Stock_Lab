import json
import os
import re
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

import anthropic

from modules.state import STATE_DIR, load_json, safe_float, save_json, today_str

EARNINGS_ALERT_DAYS = 14
EARNINGS_ANALYSIS_CACHE_FILE = f"{STATE_DIR}/earnings_analysis_cache.json"
EARNINGS_ANALYSIS_MODEL = "claude-opus-4-5"


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


# ============================================================
# DEEP EARNINGS ANALYSIS (Claude Opus + web_search)
# ============================================================

def _cache_is_fresh_analysis(entry):
    fetched_at = entry.get("fetched_at") if entry else None
    if not fetched_at:
        return False
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(fetched_at)).total_seconds()
        return age < 24 * 3600
    except Exception:
        return False


def _parse_analysis_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _fetch_deep_analysis(client, ticker, days_to_earnings, next_earnings_date):
    """Use Claude Opus + web_search to analyse an upcoming earnings report."""
    from modules.sentiment import run_claude_web_search

    prompt = (
        f"{ticker} reports earnings in {days_to_earnings} days (around {next_earnings_date}). "
        "Search for: analyst consensus estimates (EPS and revenue), recent guidance, "
        "key risks, and any pre-announcement news. "
        "Based on what you find, give a deep analysis of what to expect. "
        "Respond ONLY with a JSON object (no markdown): "
        '{"outlook": "2-3 sentence summary of what to expect", '
        '"risk_factors": ["risk1", "risk2", "risk3"], '
        '"recommendation": "one of: Hold / Reduce before earnings / Wait for report / Buy the dip"}'
    )
    try:
        raw = run_claude_web_search(client, EARNINGS_ANALYSIS_MODEL, prompt, max_tokens=768)
        parsed = _parse_analysis_json(raw)
        if parsed and "outlook" in parsed:
            return {
                "outlook": str(parsed.get("outlook", ""))[:600],
                "risk_factors": (parsed.get("risk_factors") or [])[:4],
                "recommendation": str(parsed.get("recommendation", "Hold"))[:80],
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
    except Exception as e:
        print(f"  Deep earnings feil {ticker}: {e}")
    return None


def fetch_deep_earnings_analysis(tickers_earnings):
    """
    Run deep Claude Opus earnings analysis for tickers with upcoming earnings.

    tickers_earnings: dict {ticker: earnings_data} — only processes tickers
    where earnings_soon=True (days_to_earnings <= 14).

    Returns {ticker: {outlook, risk_factors, recommendation}}.
    Cached for 24 hours. Falls back gracefully if no API key.
    """
    if not tickers_earnings:
        return {}

    soon = {
        t: e for t, e in tickers_earnings.items()
        if e.get("earnings_soon") and e.get("days_to_earnings") is not None
    }
    if not soon:
        return {}

    cache = load_json(EARNINGS_ANALYSIS_CACHE_FILE, {})
    result = {}
    to_fetch = []

    for ticker, e in soon.items():
        entry = cache.get(ticker)
        if entry and _cache_is_fresh_analysis(entry):
            result[ticker] = entry
        else:
            to_fetch.append((ticker, e))

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ANTHROPIC_API_KEY mangler — hopper over deep earnings-analyse")
        return result

    if to_fetch:
        print(f"Deep earnings-analyse (Claude Opus): {len(to_fetch)} tickere...")
        client = anthropic.Anthropic(api_key=api_key)

        for ticker, e in to_fetch:
            days = e.get("days_to_earnings", "?")
            date = e.get("next_earnings", "?")
            analysis = _fetch_deep_analysis(client, ticker, days, date)
            if analysis:
                cache[ticker] = analysis
                result[ticker] = analysis
                rec = analysis.get("recommendation", "")
                print(f"  {ticker} ({days}d): {rec} — {analysis.get('outlook', '')[:60]}")
            else:
                print(f"  {ticker}: ingen analyse hentet")

        save_json(cache, EARNINGS_ANALYSIS_CACHE_FILE)

    print(f"Deep earnings-analyse komplett for {len(result)} tickere")
    return result
