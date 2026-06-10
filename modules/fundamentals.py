from datetime import datetime, timezone

import yfinance as yf

from modules.state import (
    FUNDAMENTALS_CACHE_FILE,
    load_json,
    safe_float,
    save_json,
)

CACHE_MAX_AGE_DAYS = 7


def _cache_is_fresh(entry):
    if not entry or "fetched_at" not in entry:
        return False
    try:
        fetched = datetime.fromisoformat(entry["fetched_at"])
        age = datetime.now(timezone.utc) - fetched
        return age.days < CACHE_MAX_AGE_DAYS
    except Exception:
        return False


def _parse_info(info):
    roe = safe_float(info.get("returnOnEquity"))
    gross_margin = safe_float(info.get("grossMargins"))

    # yfinance returns debtToEquity as a percentage (150 = 1.5x ratio)
    raw_de = safe_float(info.get("debtToEquity"))
    debt_equity = raw_de / 100.0 if raw_de is not None else None

    earnings_growth = safe_float(info.get("earningsGrowth"))
    revenue_growth = safe_float(info.get("revenueGrowth"))
    forward_pe = safe_float(info.get("forwardPE"))
    peg_ratio = safe_float(info.get("pegRatio"))
    price_to_sales = safe_float(info.get("priceToSalesTrailing12Months"))
    sector = info.get("sector") or "Unknown"
    market_cap = safe_float(info.get("marketCap"))

    return {
        "roe": roe,
        "gross_margin": gross_margin,
        "debt_equity": debt_equity,
        "earnings_growth": earnings_growth,
        "revenue_growth": revenue_growth,
        "forward_pe": forward_pe,
        "peg_ratio": peg_ratio,
        "price_to_sales": price_to_sales,
        "sector": sector,
        "market_cap": market_cap,
    }


def _empty_fundamentals():
    return {
        "roe": None,
        "gross_margin": None,
        "debt_equity": None,
        "earnings_growth": None,
        "revenue_growth": None,
        "forward_pe": None,
        "peg_ratio": None,
        "price_to_sales": None,
        "sector": "Unknown",
        "market_cap": None,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def fetch_fundamentals_bulk(tickers):
    cache = load_json(FUNDAMENTALS_CACHE_FILE, {})
    result = {}
    to_fetch = []

    for ticker in tickers:
        entry = cache.get(ticker)
        if entry and _cache_is_fresh(entry):
            result[ticker] = entry
        else:
            to_fetch.append(ticker)

    print(f"Fundamentals: {len(result)} fra cache, {len(to_fetch)} hentes nå")

    for i, ticker in enumerate(to_fetch):
        try:
            info = yf.Ticker(ticker).info
            parsed = _parse_info(info)
            parsed["fetched_at"] = datetime.now(timezone.utc).isoformat()
        except Exception as e:
            print(f"  Fundamentals feil {ticker}: {e}")
            parsed = _empty_fundamentals()

        cache[ticker] = parsed
        result[ticker] = parsed

        if (i + 1) % 50 == 0:
            print(f"  Fundamentals: {i+1}/{len(to_fetch)} hentet, lagrer cache...")
            save_json(cache, FUNDAMENTALS_CACHE_FILE)

    save_json(cache, FUNDAMENTALS_CACHE_FILE)
    print(f"Fundamentals komplett for {len(result)} tickere")
    return result
