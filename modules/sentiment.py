import json
import os
import re
import signal
import time
from datetime import datetime, timezone

import anthropic

from modules.state import SENTIMENT_CACHE_FILE, load_json, safe_float, save_json

SENTIMENT_MODEL = "claude-sonnet-4-5"
CACHE_MAX_AGE_HOURS = 24
MAX_SENTIMENT_CALLS = 5
PER_CALL_TIMEOUT = 10   # seconds — passed to Anthropic SDK
TOTAL_TIMEOUT = 60      # seconds — hard wall-clock cap for entire module

_NEUTRAL = {"raw_score": 0.0, "score": 0.5, "summary": "Ingen data", "key_events": []}
_WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}


class _SentimentTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _SentimentTimeout("Sentiment total timeout")


def _cache_is_fresh(entry):
    if not entry or "fetched_at" not in entry:
        return False
    try:
        fetched = datetime.fromisoformat(entry["fetched_at"])
        age = datetime.now(timezone.utc) - fetched
        return age.total_seconds() < CACHE_MAX_AGE_HOURS * 3600
    except Exception:
        return False


def _get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key, timeout=PER_CALL_TIMEOUT)


def _run_web_search(client, prompt):
    messages = [{"role": "user", "content": prompt}]
    last_text = ""

    for _ in range(5):
        response = client.messages.create(
            model=SENTIMENT_MODEL,
            max_tokens=512,
            tools=[_WEB_SEARCH_TOOL],
            messages=messages,
        )

        text_parts = [
            b.text for b in response.content
            if getattr(b, "type", "") == "text" and hasattr(b, "text")
        ]
        if text_parts:
            last_text = " ".join(text_parts).strip()

        if response.stop_reason == "end_turn":
            return last_text

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = [
                {"type": "tool_result", "tool_use_id": b.id, "content": ""}
                for b in response.content
                if getattr(b, "type", "") == "tool_use"
            ]
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
        else:
            break

    return last_text


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
    prompt = (
        f"Search for the most important news about {ticker} stock in the last 7 days. "
        "Focus on: earnings, guidance, analyst upgrades/downgrades, major contracts, "
        "regulatory issues, and competitive threats. "
        "Based on what you find, rate overall investor sentiment on a scale from "
        "0.0 (very negative) to 1.0 (very positive), where 0.5 is neutral. "
        "Respond ONLY with a JSON object: "
        '{"score": float, "summary": "1-2 sentence assessment", "key_events": ["event1", "event2"]}'
    )
    try:
        raw = _run_web_search(client, prompt)
        parsed = _parse_json(raw)
        if not parsed or "score" not in parsed:
            return None, "Ugyldig JSON-respons fra Claude"
        score = max(0.0, min(1.0, safe_float(parsed.get("score"), 0.5)))
        return {
            "raw_score": round((score * 2.0) - 1.0, 4),
            "score": round(score, 4),
            "summary": str(parsed.get("summary", ""))[:500],
            "key_events": (parsed.get("key_events") or [])[:5],
            "source": "claude",
        }, None
    except anthropic.RateLimitError:
        print(f"  {ticker}: 429 rate limit — fallback 0.5")
        return None, "RateLimitError"
    except anthropic.APITimeoutError:
        print(f"  {ticker}: timeout ({PER_CALL_TIMEOUT}s) — fallback 0.5")
        return None, "APITimeoutError"
    except _SentimentTimeout:
        raise  # let total-timeout propagate
    except Exception as e:
        err = str(e)[:120]
        print(f"  {ticker}: feil — {err}")
        return None, err


def fetch_sentiment_bulk(tickers):
    """
    Fetch news sentiment for tickers using Claude Sonnet + web_search.
    Hard-capped at TOTAL_TIMEOUT seconds total and MAX_SENTIMENT_CALLS API calls.
    Returns {ticker: {raw_score, score, summary, key_events, source, error?}}.
    """
    if not tickers:
        return {}

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

    cache_count = len(result)
    skip_msg = f", {len(skipped)} hoppet over (cap={MAX_SENTIMENT_CALLS})" if skipped else ""
    print(f"Sentiment: {cache_count} fra cache, {len(to_fetch)} hentes nå{skip_msg}")

    client = _get_client()
    if not client and to_fetch:
        print("ANTHROPIC_API_KEY mangler — bruker nøytral score (0.5) for alle")
        for ticker in to_fetch:
            entry = {
                **_NEUTRAL,
                "source": "fallback",
                "error": "ANTHROPIC_API_KEY mangler",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            cache[ticker] = entry
            result[ticker] = entry
        save_json(cache, SENTIMENT_CACHE_FILE)
        return result
    elif not client:
        return result

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(TOTAL_TIMEOUT)
    errors = []
    try:
        for ticker in to_fetch:
            data, err = _fetch_single(client, ticker)
            if data is None:
                data = {**_NEUTRAL, "source": "fallback"}
                if err:
                    data["error"] = err
                    errors.append(f"{ticker}: {err}")
            data["fetched_at"] = datetime.now(timezone.utc).isoformat()
            cache[ticker] = data
            result[ticker] = data
            src_tag = "Claude ✓" if data.get("source") == "claude" else "fallback"
            print(f"  {ticker}: {data['score']:.2f} ({src_tag}) — {data.get('summary', '')[:60]}")
    except _SentimentTimeout:
        print(f"Sentiment: total timeout ({TOTAL_TIMEOUT}s) nådd — avbryter")
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

    claude_count = sum(1 for t in to_fetch if result.get(t, {}).get("source") == "claude")
    fallback_count = len(to_fetch) - claude_count
    print(f"Sentiment komplett: {claude_count} Claude, {fallback_count} fallback"
          + (f" ({len(skipped)} hoppet over)" if skipped else ""))
    if errors:
        print(f"  Feil ({len(errors)}): {'; '.join(errors[:3])}")

    return result
