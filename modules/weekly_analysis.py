"""
LAG 2 — Ukentlig Sonnet-analyse av posisjoner vi eier.
Kjøres kun mandager i execute-modus.
Henter finansdata fra yfinance (ingen web_search).
Hard-cap: 90s totalt, 15s per kall.
"""
import json
import os
import re
import signal
from datetime import datetime, timezone

import anthropic
import pandas as pd
import yfinance as yf

from modules.state import WEEKLY_ANALYSIS_CACHE_FILE, load_json, safe_float, save_json

WEEKLY_MODEL = "claude-sonnet-4-5"
CACHE_MAX_AGE_HOURS = 6 * 24   # 6 dager — trygt over en uke
PER_CALL_TIMEOUT = 15
TOTAL_TIMEOUT = 90

# Approx cost per token (USD) — Sonnet 4.5
_IN_CPT  = 3.00 / 1_000_000
_OUT_CPT = 15.00 / 1_000_000

_NEUTRAL_ANALYSIS = {
    "thesis_intact": True,
    "conviction": "low",
    "action": "hold",
    "reasoning": "Ingen data tilgjengelig",
    "risks": [],
    "catalysts": [],
    "source": "fallback",
}


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout("Weekly analysis total timeout")


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
    """Henter finansdata til Sonnet-prompten."""
    t = yf.Ticker(ticker)
    out = {}

    # Siste 10 nyheter
    try:
        news = t.news or []
        titles = []
        for n in news[:10]:
            title = (n.get("content") or {}).get("title") or n.get("title", "")
            if title:
                titles.append(str(title)[:120])
        out["news"] = titles
    except Exception:
        out["news"] = []

    # Kvartalsvise inntekter (siste 4 kvartaler)
    try:
        fins = t.quarterly_financials
        if fins is not None and not (hasattr(fins, "empty") and fins.empty):
            rev_row = None
            for key in ("Total Revenue", "Revenue"):
                if key in fins.index:
                    rev_row = fins.loc[key]
                    break
            if rev_row is not None:
                cols = rev_row.dropna().index[:4]
                vals = [safe_float(rev_row[c]) for c in cols]
                out["revenue_q"] = [round(v / 1e9, 2) if v else None for v in vals]
            else:
                out["revenue_q"] = []

            gp_row = None
            for key in ("Gross Profit",):
                if key in fins.index:
                    gp_row = fins.loc[key]
                    break
            rev_vals = out.get("revenue_q", [])
            if gp_row is not None and rev_vals:
                gp_cols = gp_row.dropna().index[:4]
                rev_cols = rev_row.dropna().index[:4] if rev_row is not None else []
                margins = []
                for i, c in enumerate(gp_cols[:len(rev_cols)]):
                    gp = safe_float(gp_row.get(c))
                    rv = safe_float(rev_row.get(c)) if rev_row is not None else None
                    if gp and rv and rv != 0:
                        margins.append(round(gp / rv * 100, 1))
                out["gross_margin_q"] = margins
            else:
                out["gross_margin_q"] = []
        else:
            out["revenue_q"] = []
            out["gross_margin_q"] = []
    except Exception:
        out["revenue_q"] = []
        out["gross_margin_q"] = []

    # EPS trend (kvartalsvise, siste 4)
    try:
        hist = t.earnings_history
        if hist is not None and not (hasattr(hist, "empty") and hist.empty):
            hist = hist.dropna(subset=["epsActual"]).tail(4)
            out["eps_q"] = [safe_float(row.get("epsActual")) for _, row in hist.iterrows()]
        else:
            out["eps_q"] = []
    except Exception:
        out["eps_q"] = []

    # Analyst price targets
    try:
        info = t.info or {}
        target_mean = safe_float(info.get("targetMeanPrice"))
        target_low  = safe_float(info.get("targetLowPrice"))
        target_high = safe_float(info.get("targetHighPrice"))
        current     = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        out["analyst_target_mean"] = target_mean
        out["analyst_target_low"]  = target_low
        out["analyst_target_high"] = target_high
        out["current_price"]       = current
        out["short_pct"]           = round(safe_float(info.get("shortPercentOfFloat"), 0.0) * 100, 1)
        prev_inst = safe_float(info.get("heldPercentInstitutions"))
        out["inst_held_pct"]       = round(prev_inst * 100, 1) if prev_inst else None
    except Exception:
        out["analyst_target_mean"] = None
        out["analyst_target_low"]  = None
        out["analyst_target_high"] = None
        out["current_price"]       = None
        out["short_pct"]           = None
        out["inst_held_pct"]       = None

    return out


def _build_prompt(ticker, yf_data, pos):
    avg_price   = safe_float(pos.get("avg_price"))
    last_price  = safe_float(pos.get("last_price") or pos.get("avg_price"))
    ret_pct     = ((last_price / avg_price) - 1) * 100 if avg_price and last_price and avg_price != 0 else None

    news_str = "\n".join(f"- {t}" for t in yf_data.get("news", [])) or "No recent news"

    rev = yf_data.get("revenue_q", [])
    rev_str = " → ".join(f"${v}B" for v in rev if v) if rev else "No data"

    margins = yf_data.get("gross_margin_q", [])
    margin_str = " → ".join(f"{m}%" for m in margins) if margins else "No data"

    eps = yf_data.get("eps_q", [])
    eps_str = " → ".join(f"${e:.2f}" for e in eps if e is not None) if eps else "No data"

    tm = yf_data.get("analyst_target_mean")
    tl = yf_data.get("analyst_target_low")
    th = yf_data.get("analyst_target_high")
    target_str = f"avg ${tm}" + (f", range ${tl}-${th}" if tl and th else "") if tm else "No data"

    short_str  = f"{yf_data.get('short_pct')}%" if yf_data.get("short_pct") is not None else "No data"
    inst_str   = f"{yf_data.get('inst_held_pct')}%" if yf_data.get("inst_held_pct") is not None else "No data"

    buy_str    = f"${avg_price:.2f}" if avg_price else "N/A"
    curr_str   = f"${last_price:.2f}" if last_price else "N/A"
    ret_str    = f"{ret_pct:+.1f}%" if ret_pct is not None else "N/A"

    return (
        f"You are a portfolio manager reviewing a position in {ticker}.\n\n"
        f"Financial trend (last 4 quarters):\n"
        f"Revenue growth: {rev_str}\n"
        f"Gross margin trend: {margin_str}\n"
        f"EPS trend: {eps_str}\n\n"
        f"Recent developments:\n{news_str}\n\n"
        f"Analyst targets: {target_str}\n"
        f"Short interest: {short_str}\n"
        f"Institutional ownership: {inst_str}\n\n"
        f"We bought at {buy_str}, currently at {curr_str} ({ret_str}).\n\n"
        "Assess: Is our investment thesis still intact?\n"
        'JSON only: {"thesis_intact": true/false, "conviction": "high/medium/low", '
        '"action": "hold/add/reduce/sell", "reasoning": "<max 150 chars>", '
        '"risks": ["<risk1>", "<risk2>"], "catalysts": ["<cat1>", "<cat2>"]}'
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


def _fetch_single(client, ticker, pos):
    """Returnerer (data_dict | None, error_str | None, cost_usd)."""
    yf_data = _fetch_yf_data(ticker)
    prompt  = _build_prompt(ticker, yf_data, pos)
    try:
        response = client.messages.create(
            model=WEEKLY_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw    = response.content[0].text if response.content else ""
        parsed = _parse_json(raw)
        if not parsed or "action" not in parsed:
            return None, "Ugyldig JSON-respons", 0.0

        usage   = getattr(response, "usage", None)
        in_tok  = getattr(usage, "input_tokens",  len(prompt) // 4) if usage else len(prompt) // 4
        out_tok = getattr(usage, "output_tokens", 80)               if usage else 80
        cost    = in_tok * _IN_CPT + out_tok * _OUT_CPT

        return {
            "thesis_intact": bool(parsed.get("thesis_intact", True)),
            "conviction":    str(parsed.get("conviction", "low"))[:20],
            "action":        str(parsed.get("action", "hold"))[:20],
            "reasoning":     str(parsed.get("reasoning", ""))[:150],
            "risks":         (parsed.get("risks") or [])[:2],
            "catalysts":     (parsed.get("catalysts") or [])[:2],
            "source":        "claude",
        }, None, cost

    except anthropic.RateLimitError:
        print(f"  {ticker}: 429 rate limit — fallback")
        return None, "RateLimitError", 0.0
    except anthropic.APITimeoutError:
        print(f"  {ticker}: timeout ({PER_CALL_TIMEOUT}s) — fallback")
        return None, "APITimeoutError", 0.0
    except _Timeout:
        raise
    except Exception as e:
        print(f"  {ticker}: feil — {str(e)[:120]}")
        return None, str(e)[:120], 0.0


def fetch_weekly_analysis(held_positions):
    """
    Lag 2: Ukentlig Sonnet-gjennomgang av posisjoner.
    held_positions: {ticker: {avg_price, last_price, ...}} — kun eide aksjer.
    Returnerer (analyses_dict, total_cost_usd).
    analyses_dict: {ticker: {thesis_intact, conviction, action, reasoning, risks, catalysts}}
    """
    if not held_positions:
        return {}, 0.0

    cache  = load_json(WEEKLY_ANALYSIS_CACHE_FILE, {})
    result = {}
    to_fetch = []

    for ticker, pos in held_positions.items():
        entry = cache.get(ticker)
        if entry and _cache_is_fresh(entry):
            result[ticker] = entry
        else:
            to_fetch.append((ticker, pos))

    print(f"Weekly [Sonnet]: {len(result)} fra cache, {len(to_fetch)} hentes nå")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and to_fetch:
        print("ANTHROPIC_API_KEY mangler — hopper over ukentlig analyse")
        for ticker, _ in to_fetch:
            entry = {**_NEUTRAL_ANALYSIS, "fetched_at": datetime.now(timezone.utc).isoformat()}
            cache[ticker] = entry
            result[ticker] = entry
        save_json(cache, WEEKLY_ANALYSIS_CACHE_FILE)
        return result, 0.0

    client = _get_client()
    if not client:
        return result, 0.0

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(TOTAL_TIMEOUT)
    total_cost = 0.0
    errors = []
    try:
        for ticker, pos in to_fetch:
            data, err, cost = _fetch_single(client, ticker, pos)
            total_cost += cost
            if data is None:
                data = {**_NEUTRAL_ANALYSIS, "source": "fallback"}
                if err:
                    data["error"] = err
                    errors.append(f"{ticker}: {err}")
            data["fetched_at"] = datetime.now(timezone.utc).isoformat()
            cache[ticker] = data
            result[ticker] = data
            tag = "Sonnet ✓" if data.get("source") == "claude" else "fallback"
            action = data.get("action", "hold")
            conviction = data.get("conviction", "low")
            print(f"  {ticker}: {action} ({conviction}) [{tag}] — {data.get('reasoning', '')[:60]}")
    except _Timeout:
        print(f"Weekly: total timeout ({TOTAL_TIMEOUT}s) — avbryter")
        for ticker, _ in to_fetch:
            if ticker not in result:
                entry = {**_NEUTRAL_ANALYSIS, "source": "fallback", "error": "total timeout",
                         "fetched_at": datetime.now(timezone.utc).isoformat()}
                cache[ticker] = entry
                result[ticker] = entry
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    save_json(cache, WEEKLY_ANALYSIS_CACHE_FILE)

    claude_n   = sum(1 for t in to_fetch if result.get(t[0], {}).get("source") == "claude")
    fallback_n = len(to_fetch) - claude_n
    print(f"Weekly komplett: {claude_n} Sonnet, {fallback_n} fallback"
          + f" — estimert kostnad: ${total_cost:.4f}")
    if errors:
        print(f"  Feil ({len(errors)}): {'; '.join(errors[:3])}")

    return result, total_cost
