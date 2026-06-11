"""
LAG 3 — Earnings Opus-analyse.
Henter historiske earnings-data fra yfinance (ingen web_search).
Kjøres når earnings er innen 14 dager for en posisjon.
Hard-cap: 120s totalt, 30s per kall.
"""
import json
import os
import re
import signal
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

import anthropic

from modules.state import STATE_DIR, load_json, safe_float, save_json, today_str

EARNINGS_ALERT_DAYS = 14
EARNINGS_ANALYSIS_CACHE_FILE = f"{STATE_DIR}/earnings_analysis_cache.json"
EARNINGS_ANALYSIS_MODEL = "claude-opus-4-5"

PER_CALL_TIMEOUT = 30   # seconds — Anthropic SDK timeout
TOTAL_TIMEOUT    = 120  # seconds — hard wall-clock cap via SIGALRM

# Approx cost per token (USD) — Opus 4.5
_IN_CPT  = 15.00 / 1_000_000
_OUT_CPT = 75.00 / 1_000_000


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout("Earnings analysis total timeout")


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
# DEEP EARNINGS ANALYSIS (Claude Opus, yfinance-data)
# ============================================================

def _cache_is_fresh_analysis(entry):
    if not entry or "fetched_at" not in entry:
        return False
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(entry["fetched_at"])).total_seconds()
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


def _fetch_earnings_yf_data(ticker):
    """Henter historisk earnings-data fra yfinance for Opus-prompten."""
    t = yf.Ticker(ticker)
    out = {}

    # Historiske earnings siste 8 kvartaler
    try:
        hist = t.earnings_history
        if hist is not None and not (hasattr(hist, "empty") and hist.empty):
            hist = hist.dropna(subset=["epsActual", "epsEstimate"]).tail(8)
            records = []
            for idx, row in hist.iterrows():
                actual   = safe_float(row.get("epsActual"))
                estimate = safe_float(row.get("epsEstimate"))
                if actual is not None and estimate is not None and estimate != 0:
                    surprise = round((actual - estimate) / abs(estimate) * 100, 1)
                else:
                    surprise = None
                quarter = str(idx)[:7] if idx else "?"
                records.append({"q": quarter, "est": estimate, "actual": actual, "surprise_pct": surprise})
            out["history"] = records
        else:
            out["history"] = []
    except Exception:
        out["history"] = []

    # Revenue trend (kvartalsvise)
    try:
        fins = t.quarterly_financials
        if fins is not None and not (hasattr(fins, "empty") and fins.empty):
            for key in ("Total Revenue", "Revenue"):
                if key in fins.index:
                    rev_row = fins.loc[key].dropna()
                    out["revenue_q_b"] = [round(safe_float(v, 0) / 1e9, 2) for v in rev_row.values[:4]]
                    break
        if "revenue_q_b" not in out:
            out["revenue_q_b"] = []
    except Exception:
        out["revenue_q_b"] = []

    # Analyst-konsensus neste kvartal
    try:
        info = t.info or {}
        out["forward_eps"]     = safe_float(info.get("forwardEps"))
        out["revenue_est_b"]   = round(safe_float(info.get("revenueEstimate"), 0) / 1e9, 2) if info.get("revenueEstimate") else None
        out["forward_pe"]      = safe_float(info.get("forwardPE"))
        out["current_price"]   = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        # Last quarter guidance (52w guidance if available)
        out["52w_guidance"]    = info.get("targetMeanPrice")
    except Exception:
        out["forward_eps"]   = None
        out["revenue_est_b"] = None
        out["forward_pe"]    = None
        out["current_price"] = None
        out["52w_guidance"]  = None

    return out


def _build_earnings_prompt(ticker, days_to_earnings, next_earnings_date, yf_data):
    # History table
    hist = yf_data.get("history", [])
    if hist:
        hist_lines = []
        for r in hist:
            surprise = f"{r['surprise_pct']:+.1f}%" if r.get("surprise_pct") is not None else "N/A"
            hist_lines.append(f"  {r['q']}: est ${r['est']}, actual ${r['actual']}, surprise {surprise}")
        hist_str = "\n".join(hist_lines)
    else:
        hist_str = "  No historical data"

    rev = yf_data.get("revenue_q_b", [])
    rev_str = " → ".join(f"${v}B" for v in rev) if rev else "No data"

    fwd_eps  = yf_data.get("forward_eps")
    rev_est  = yf_data.get("revenue_est_b")
    fwd_str  = f"EPS consensus ${fwd_eps}" if fwd_eps else "No data"
    rev_est_str = f"Revenue consensus ${rev_est}B" if rev_est else "No data"

    return (
        f"You are a senior analyst preparing for {ticker} earnings "
        f"in {days_to_earnings} days (around {next_earnings_date}).\n\n"
        f"Historical performance vs estimates (last 8 quarters):\n{hist_str}\n\n"
        f"Revenue trend (quarterly): {rev_str}\n"
        f"{fwd_str}\n"
        f"{rev_est_str}\n\n"
        "Assess the upcoming earnings:\n"
        'JSON only: {"beat_probability": <0.0-1.0>, "expected_reaction": "<string>", '
        '"key_metrics_to_watch": ["<m1>", "<m2>", "<m3>"], '
        '"position_recommendation": "hold/reduce_before/add_before", '
        '"reasoning": "<max 200 chars>", "risk_level": "low/medium/high"}'
    )


def _fetch_deep_analysis(client, ticker, days_to_earnings, next_earnings_date):
    """Kaller Claude Opus med yfinance-data for å analysere kommende earnings."""
    yf_data = _fetch_earnings_yf_data(ticker)
    prompt  = _build_earnings_prompt(ticker, days_to_earnings, next_earnings_date, yf_data)
    try:
        response = client.messages.create(
            model=EARNINGS_ANALYSIS_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        raw    = response.content[0].text if response.content else ""
        parsed = _parse_analysis_json(raw)
        if not parsed or "position_recommendation" not in parsed:
            return None, 0.0

        usage   = getattr(response, "usage", None)
        in_tok  = getattr(usage, "input_tokens",  len(prompt) // 4) if usage else len(prompt) // 4
        out_tok = getattr(usage, "output_tokens", 100)               if usage else 100
        cost    = in_tok * _IN_CPT + out_tok * _OUT_CPT

        return {
            "beat_probability":         max(0.0, min(1.0, safe_float(parsed.get("beat_probability"), 0.5))),
            "expected_reaction":         str(parsed.get("expected_reaction", ""))[:120],
            "key_metrics_to_watch":      (parsed.get("key_metrics_to_watch") or [])[:3],
            "position_recommendation":   str(parsed.get("position_recommendation", "hold"))[:30],
            "reasoning":                 str(parsed.get("reasoning", ""))[:200],
            "risk_level":                str(parsed.get("risk_level", "medium"))[:10],
            "fetched_at":                datetime.now(timezone.utc).isoformat(),
        }, cost

    except anthropic.RateLimitError:
        print(f"  {ticker}: 429 rate limit — hopper over earnings-analyse")
        return None, 0.0
    except anthropic.APITimeoutError:
        print(f"  {ticker}: timeout ({PER_CALL_TIMEOUT}s) — hopper over")
        return None, 0.0
    except _Timeout:
        raise
    except Exception as e:
        print(f"  Deep earnings feil {ticker}: {str(e)[:120]}")
        return None, 0.0


def fetch_deep_earnings_analysis(tickers_earnings):
    """
    Lag 3: Claude Opus-analyse for aksjer med earnings innen 14 dager.
    tickers_earnings: {ticker: earnings_data} — kun tickers med earnings_soon=True behandles.
    Returnerer (analyses_dict, total_cost_usd).
    """
    if not tickers_earnings:
        return {}, 0.0

    soon = {
        t: e for t, e in tickers_earnings.items()
        if e.get("earnings_soon") and e.get("days_to_earnings") is not None
    }
    if not soon:
        return {}, 0.0

    cache    = load_json(EARNINGS_ANALYSIS_CACHE_FILE, {})
    result   = {}
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
        return result, 0.0

    total_cost = 0.0

    if to_fetch:
        print(f"Deep earnings [Opus]: {len(to_fetch)} tickere...")
        client = anthropic.Anthropic(api_key=api_key, timeout=PER_CALL_TIMEOUT)

        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(TOTAL_TIMEOUT)
        try:
            for ticker, e in to_fetch:
                days = e.get("days_to_earnings", "?")
                date = e.get("next_earnings", "?")
                analysis, cost = _fetch_deep_analysis(client, ticker, days, date)
                total_cost += cost
                if analysis:
                    cache[ticker] = analysis
                    result[ticker] = analysis
                    rec = analysis.get("position_recommendation", "")
                    print(f"  {ticker} ({days}d): {rec} (p={analysis.get('beat_probability', '?'):.0%}) — {analysis.get('reasoning', '')[:60]}")
                else:
                    print(f"  {ticker}: ingen analyse hentet")
        except _Timeout:
            print(f"Earnings: total timeout ({TOTAL_TIMEOUT}s) — avbryter")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        save_json(cache, EARNINGS_ANALYSIS_CACHE_FILE)

    print(f"Deep earnings [Opus] komplett: {len(result)} tickere"
          + (f" — estimert kostnad: ${total_cost:.4f}" if total_cost > 0 else ""))
    return result, total_cost
