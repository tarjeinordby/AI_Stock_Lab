import json
import os
import re
from datetime import datetime, timezone

import anthropic

from modules.state import SENTIMENT_CACHE_FILE, load_json, safe_float, save_json

SENTIMENT_MODEL = "claude-sonnet-4-5"
CACHE_MAX_AGE_HOURS = 24

_NEUTRAL = {"raw_score": 0.0, "score": 0.5, "summary": "Ingen data", "key_events": []}

_WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}


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
    return anthropic.Anthropic(api_key=api_key)


def run_claude_web_search(client, model, prompt, max_tokens=1024):
    """
    Run Claude with web_search_20250305 tool, handling the agentic loop.
    Returns the final text response.
    """
    messages = [{"role": "user", "content": prompt}]
    tools = [_WEB_SEARCH_TOOL]
    last_text = ""

    for _ in range(10):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            tools=tools,
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
            # Add assistant turn (includes tool_use blocks)
            messages.append({"role": "assistant", "content": response.content})

            # For web_search_20250305, results are filled server-side.
            # Pass back empty tool_result to continue the loop.
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
    """Extract first JSON object from text."""
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
        raw = run_claude_web_search(client, SENTIMENT_MODEL, prompt, max_tokens=512)
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
    except Exception as e:
        err = str(e)[:120]
        print(f"  Sentiment feil {ticker}: {err}")
        return None, err


def fetch_sentiment_bulk(tickers):
    """
    Fetch news sentiment for tickers using Claude Sonnet + web_search.
    Returns {ticker: {raw_score, score, summary, key_events, source, error?}}.
    source is 'claude' when fetched via API, 'fallback' when using neutral 0.5.
    Results cached for 24 hours.
    """
    if not tickers:
        return {}

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

    errors = []
    for i, ticker in enumerate(to_fetch):
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

        if (i + 1) % 5 == 0:
            save_json(cache, SENTIMENT_CACHE_FILE)

    save_json(cache, SENTIMENT_CACHE_FILE)

    claude_count = sum(1 for t in to_fetch if result.get(t, {}).get("source") == "claude")
    fallback_count = len(to_fetch) - claude_count
    print(f"Sentiment komplett: {claude_count} Claude, {fallback_count} fallback")
    if errors:
        print(f"  Feil ({len(errors)}): {'; '.join(errors[:3])}")

    return result
