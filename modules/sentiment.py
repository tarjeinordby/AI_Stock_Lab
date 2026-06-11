"""
LAG 1 — Daglig Haiku-sentiment.
Henter gratis strukturdata fra yfinance, kaller claude-haiku (ingen web_search).
Hard-cap: 60s totalt, 8s per kall, maks 10 kall.
"""
import json
import os
import re
import signal
from datetime import datetime, timezone

import anthropic
import pandas as pd
import yfinance as yf

from modules.state import SENTIMENT_CACHE_FILE, load_json, safe_float, save_json

SENTIMENT_MODEL = "claude-haiku-4-5-20251001"
CACHE_MAX_AGE_HOURS = 23
MAX_SENTIMENT_CALLS = 10
PER_CALL_TIMEOUT = 8    # seconds — Anthropic SDK timeout
TOTAL_TIMEOUT = 60      # seconds — hard wall-clock cap via SIGALRM

# Approx cost per token (USD) — Haiku 4.5
_IN_CPT  = 0.80 / 1_000_000
_OUT_CPT = 4.00 / 1_000_000

_NEUTRAL = {"raw_score": 0.0, "score": 0.5, "summary": "Ingen data", "key_factors": []}


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout("Sentiment total timeout")


def _cache_is_fresh(entry):
    if not entry or "fetched_at" not in entry:
        return False
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(entry["fetched_at"])).total_seconds()
        return age < CACHE_MAX_AGE_HOURS * 3600
    except Exception:
        return False


def _get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key, timeout=PER_CALL_TIMEOUT)


def _fetch_yf_data(ticker):
    """Hent gratis data fra yfinance for å bygge prompt."""
    t = yf.Ticker(ticker)
    out = {}

    # Siste 5 nyhetstitler
    try:
        news = t.news or []
        titles = []
        for n in news[:5]:
            title = (n.get("content") or {}).get("title") or n.get("title", "")
            if title:
                titles.append(str(title)[:120])
        out["news"] = titles
    except Exception:
        out["news"] = []

    # Analyst-anbefalinger (siste 3 måneder)
    try:
        rec = t.recommendations
        if rec is not None and not (hasattr(rec, "empty") and rec.empty):
            totals = {}
            for col in ("strongBuy", "buy", "hold", "sell", "strongSell"):
                if col in rec.columns:
                    totals[col] = int(rec[col].tail(3).sum())
            out["ratings"] = totals
        else:
            out["ratings"] = {}
    except Exception:
        out["ratings"] = {}

    # Earnings beat/miss de siste 4 kvartalene
    try:
        hist = t.earnings_history
        if hist is not None and not (hasattr(hist, "empty") and hist.empty):
            hist = hist.dropna(subset=["epsActual", "epsEstimate"]).tail(4)
            track = []
            for _, row in hist.iterrows():
                actual = safe_float(row.get("epsActual"))
                est = safe_float(row.get("epsEstimate"))
                if actual is not None and est is not None:
                    track.append("beat" if actual > est else "miss")
            out["earnings_track"] = track
        else:
            out["earnings_track"] = []
    except Exception:
        out["earnings_track"] = []

    # Netto insider-kjøp/-salg siste 90 dager (USD)
    try:
        ins = t.insider_transactions
        if ins is not None and not (hasattr(ins, "empty") and ins.empty):
            ins = ins.copy()
            if "startDate" in ins.columns:
                ins["startDate"] = pd.to_datetime(ins["startDate"], errors="coerce", utc=True)
                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=90)
                ins = ins[ins["startDate"] > cutoff]
            buy_val = sell_val = 0.0
            if "value" in ins.columns and "transactionDescription" in ins.columns:
                desc = ins["transactionDescription"].fillna("")
                buy_val = safe_float(ins.loc[desc.str.contains("Buy|Purchase", case=False, na=False), "value"].sum(), 0.0)
                sell_val = safe_float(ins.loc[desc.str.contains("Sale|Sell", case=False, na=False), "value"].sum(), 0.0)
            out["insider_net_usd"] = int(buy_val - sell_val)
        else:
            out["insider_net_usd"] = 0
    except Exception:
        out["insider_net_usd"] = 0

    # Kurs vs 52-ukers høy
    try:
        fi = t.fast_info
        current = safe_float(getattr(fi, "last_price", None))
        high_52w = safe_float(getattr(fi, "year_high", None))
        if current and high_52w and high_52w > 0:
            out["pct_from_52w_high"] = round((current / high_52w - 1) * 100, 1)
        else:
            out["pct_from_52w_high"] = None
    except Exception:
        out["pct_from_52w_high"] = None

    return out


def _build_prompt(ticker, yf_data):
    news_str = "\n".join(f"- {t}" for t in yf_data.get("news", [])) or "No recent news"

    r = yf_data.get("ratings", {})
    ratings_str = (
        f"strongBuy={r.get('strongBuy', 0)}, buy={r.get('buy', 0)}, "
        f"hold={r.get('hold', 0)}, sell={r.get('sell', 0)}, "
        f"strongSell={r.get('strongSell', 0)}"
    ) if r else "No data"

    track = yf_data.get("earnings_track", [])
    track_str = " → ".join(track) if track else "No data"

    insider = yf_data.get("insider_net_usd", 0)
    insider_str = (
        f"Net buy ${insider:,}" if insider > 0
        else f"Net sell ${abs(insider):,}" if insider < 0
        else "No activity"
    )

    pct = yf_data.get("pct_from_52w_high")
    high_str = f"{pct:+.1f}% from 52w high" if pct is not None else "No data"

    return (
        f"You are a senior equity analyst. Analyze this data for {ticker}:\n\n"
        f"News (last 7 days):\n{news_str}\n\n"
        f"Analyst ratings (30d): {ratings_str}\n"
        f"Earnings track record (last 4Q): {track_str}\n"
        f"Insider activity (90d): {insider_str}\n"
        f"Current price vs 52w high: {high_str}\n\n"
        "Rate sentiment for a 6-12 month investor.\n"
        'JSON only: {"score": <-1.0 to 1.0>, "summary": "<max 80 chars>", "key_factors": ["<factor1>", "<factor2>"]}'
    )


def _parse_json(text):
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


def _fetch_single(client, ticker):
    """Returnerer (data_dict | None, error_str | None, cost_usd)."""
    yf_data = _fetch_yf_data(ticker)
    prompt = _build_prompt(ticker, yf_data)
    try:
        response = client.messages.create(
            model=SENTIMENT_MODEL,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text if response.content else ""
        parsed = _parse_json(raw)
        if not parsed or "score" not in parsed:
            return None, "Ugyldig JSON-respons", 0.0

        raw_score = max(-1.0, min(1.0, safe_float(parsed.get("score"), 0.0)))
        score = round((raw_score + 1.0) / 2.0, 4)

        usage = getattr(response, "usage", None)
        in_tok  = getattr(usage, "input_tokens",  len(prompt) // 4) if usage else len(prompt) // 4
        out_tok = getattr(usage, "output_tokens", 50)               if usage else 50
        cost = in_tok * _IN_CPT + out_tok * _OUT_CPT

        return {
            "raw_score": round(raw_score, 4),
            "score": score,
            "summary": str(parsed.get("summary", ""))[:80],
            "key_factors": (parsed.get("key_factors") or [])[:2],
            "source": "claude",
        }, None, cost

    except anthropic.RateLimitError:
        print(f"  {ticker}: 429 rate limit — fallback 0.5")
        return None, "RateLimitError", 0.0
    except anthropic.APITimeoutError:
        print(f"  {ticker}: timeout ({PER_CALL_TIMEOUT}s) — fallback 0.5")
        return None, "APITimeoutError", 0.0
    except _Timeout:
        raise
    except Exception as e:
        print(f"  {ticker}: feil — {str(e)[:120]}")
        return None, str(e)[:120], 0.0


def fetch_sentiment_bulk(tickers):
    """
    Lag 1: Daglig Haiku-sentiment med yfinance-data (ingen web_search).
    Returnerer (scores_dict, total_cost_usd).
    scores_dict: {ticker: {raw_score, score, summary, key_factors, source, ...}}
    """
    if not tickers:
        return {}, 0.0

    api_key_set = bool(os.environ.get("ANTHROPIC_API_KEY", ""))
    print(f"ANTHROPIC_API_KEY satt: {api_key_set}")

    cache = load_json(SENTIMENT_CACHE_FILE, {})
    result = {}
    to_fetch = []

    for ticker in tickers:
        entry = cache.get(ticker)
        if entry and _cache_is_fresh(entry):
            if entry.get("source") == "fallback" and api_key_set:
                to_fetch.append(ticker)
            else:
                result[ticker] = entry
        else:
            to_fetch.append(ticker)

    if len(to_fetch) > MAX_SENTIMENT_CALLS:
        skipped = to_fetch[MAX_SENTIMENT_CALLS:]
        to_fetch = to_fetch[:MAX_SENTIMENT_CALLS]
        for ticker in skipped:
            entry = cache.get(ticker)
            if entry:
                result[ticker] = entry
    else:
        skipped = []

    skip_msg = f", {len(skipped)} hoppet over (cap={MAX_SENTIMENT_CALLS})" if skipped else ""
    print(f"Sentiment [Haiku]: {len(result)} fra cache, {len(to_fetch)} hentes nå{skip_msg}")

    client = _get_client()
    if not client and to_fetch:
        print("ANTHROPIC_API_KEY mangler — nøytral score (0.5) for alle")
        for ticker in to_fetch:
            entry = {**_NEUTRAL, "source": "fallback", "error": "ANTHROPIC_API_KEY mangler",
                     "fetched_at": datetime.now(timezone.utc).isoformat()}
            cache[ticker] = entry
            result[ticker] = entry
        save_json(cache, SENTIMENT_CACHE_FILE)
        return result, 0.0
    elif not client:
        return result, 0.0

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(TOTAL_TIMEOUT)
    total_cost = 0.0
    errors = []
    try:
        for ticker in to_fetch:
            data, err, cost = _fetch_single(client, ticker)
            total_cost += cost
            if data is None:
                data = {**_NEUTRAL, "source": "fallback"}
                if err:
                    data["error"] = err
                    errors.append(f"{ticker}: {err}")
            data["fetched_at"] = datetime.now(timezone.utc).isoformat()
            cache[ticker] = data
            result[ticker] = data
            tag = "Haiku ✓" if data.get("source") == "claude" else "fallback"
            print(f"  {ticker}: {data['score']:.2f} ({tag}) — {data.get('summary', '')[:60]}")
    except _Timeout:
        print(f"Sentiment: total timeout ({TOTAL_TIMEOUT}s) — avbryter")
        for ticker in to_fetch:
            if ticker not in result:
                entry = {**_NEUTRAL, "source": "fallback", "error": "total timeout",
                         "fetched_at": datetime.now(timezone.utc).isoformat()}
                cache[ticker] = entry
                result[ticker] = entry
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    save_json(cache, SENTIMENT_CACHE_FILE)

    claude_n = sum(1 for t in to_fetch if result.get(t, {}).get("source") == "claude")
    fallback_n = len(to_fetch) - claude_n
    print(f"Sentiment komplett: {claude_n} Haiku, {fallback_n} fallback"
          + (f" ({len(skipped)} hoppet over)" if skipped else "")
          + f" — estimert kostnad: ${total_cost:.4f}")
    if errors:
        print(f"  Feil ({len(errors)}): {'; '.join(errors[:3])}")

    return result, total_cost
