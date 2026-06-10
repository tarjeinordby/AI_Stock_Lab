import json
import os
from datetime import datetime, timezone

import requests

from modules.state import SENTIMENT_CACHE_FILE, load_json, safe_float, save_json

PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
PERPLEXITY_MODEL = "llama-3.1-sonar-large-128k-online"
CACHE_MAX_AGE_HOURS = 24

_NEUTRAL = {"raw_score": 0.0, "score": 0.5, "summary": "Ingen data", "key_events": []}


def _cache_is_fresh(entry):
    if not entry or "fetched_at" not in entry:
        return False
    try:
        fetched = datetime.fromisoformat(entry["fetched_at"])
        age = datetime.now(timezone.utc) - fetched
        return age.total_seconds() < CACHE_MAX_AGE_HOURS * 3600
    except Exception:
        return False


def _fetch_single(ticker):
    if not PERPLEXITY_API_KEY:
        return None

    prompt = (
        f"What are the most important news for {ticker} in the last 7 days? "
        "Focus on: earnings, guidance, analyst ratings, major contracts, "
        "regulatory issues, competitive threats. "
        "Rate the overall sentiment for a 3-12 month investor: "
        "score from -1.0 (very negative) to +1.0 (very positive). "
        'Respond ONLY in JSON: {"score": float, "summary": string, "key_events": [string]}'
    )

    try:
        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": PERPLEXITY_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
            },
            timeout=30,
        )

        if resp.status_code != 200:
            print(f"Perplexity HTTP {resp.status_code} for {ticker}")
            return None

        content = resp.json()["choices"][0]["message"]["content"]
        start = content.find("{")
        end = content.rfind("}") + 1
        if start < 0 or end <= start:
            return None

        parsed = json.loads(content[start:end])
        raw = max(-1.0, min(1.0, safe_float(parsed.get("score"), 0.0)))
        return {
            "raw_score": raw,
            "score": (raw + 1.0) / 2.0,  # normalize to [0, 1]
            "summary": str(parsed.get("summary", ""))[:500],
            "key_events": (parsed.get("key_events") or [])[:5],
        }
    except Exception as e:
        print(f"Sentiment feil {ticker}: {e}")
        return None


def fetch_sentiment_bulk(tickers):
    cache = load_json(SENTIMENT_CACHE_FILE, {})
    result = {}
    to_fetch = []

    for ticker in tickers:
        entry = cache.get(ticker)
        if entry and _cache_is_fresh(entry):
            result[ticker] = entry
        else:
            to_fetch.append(ticker)

    print(f"Sentiment: {len(result)} fra cache, {len(to_fetch)} hentes nå")

    if not PERPLEXITY_API_KEY:
        print("PERPLEXITY_API_KEY mangler — bruker nøytral score (0.5) for alle")
        for ticker in to_fetch:
            entry = dict(_NEUTRAL)
            entry["fetched_at"] = datetime.now(timezone.utc).isoformat()
            cache[ticker] = entry
            result[ticker] = entry
        save_json(cache, SENTIMENT_CACHE_FILE)
        return result

    for i, ticker in enumerate(to_fetch):
        data = _fetch_single(ticker) or dict(_NEUTRAL)
        data["fetched_at"] = datetime.now(timezone.utc).isoformat()
        cache[ticker] = data
        result[ticker] = data

        if (i + 1) % 10 == 0:
            print(f"  Sentiment: {i+1}/{len(to_fetch)}")
            save_json(cache, SENTIMENT_CACHE_FILE)

    save_json(cache, SENTIMENT_CACHE_FILE)
    print(f"Sentiment komplett for {len(result)} tickere")
    return result
