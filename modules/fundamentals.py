from datetime import datetime, timezone

import pandas as pd
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
    free_cashflow = safe_float(info.get("freeCashflow"))
    fcf_positive = (free_cashflow > 0) if free_cashflow is not None else None

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
        "free_cashflow": free_cashflow,
        "fcf_positive": fcf_positive,
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
        "free_cashflow": None,
        "fcf_positive": None,
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


# ── INSIDER SIGNAL ────────────────────────────────────────────

_INSIDER_BUY_THRESHOLD = 200_000   # netto insider-kjøp i USD (90 dager)


def _fetch_insider_signal(ticker):
    """Returnerer (netto_usd, er_signifikant_kjøp) fra Form 4-data."""
    try:
        ins = yf.Ticker(ticker).insider_transactions
        if ins is None or (hasattr(ins, "empty") and ins.empty):
            return 0, False
        ins = ins.copy()
        if "startDate" in ins.columns:
            ins["startDate"] = pd.to_datetime(ins["startDate"], errors="coerce", utc=True)
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=90)
            ins = ins[ins["startDate"] > cutoff]
        if "value" not in ins.columns or "transactionDescription" not in ins.columns:
            return 0, False
        desc = ins["transactionDescription"].fillna("")
        buy_val  = float(ins.loc[desc.str.contains("Buy|Purchase", case=False, na=False), "value"].sum() or 0)
        sell_val = float(ins.loc[desc.str.contains("Sale|Sell",    case=False, na=False), "value"].sum() or 0)
        net = int(buy_val - sell_val)
        return net, net > _INSIDER_BUY_THRESHOLD
    except Exception:
        return 0, False


def fetch_insider_bulk(tickers):
    """
    Hent insider-signal for en liten liste tickere (topp-kandidater + eide).
    Ingen cache — kalles for 20-30 tickere.
    Returnerer {ticker: {"insider_net_90d": int, "insider_buying": bool}}.
    """
    result = {}
    buyers = []
    for ticker in tickers:
        net, is_buying = _fetch_insider_signal(ticker)
        result[ticker] = {"insider_net_90d": net, "insider_buying": is_buying}
        if is_buying:
            buyers.append(f"{ticker} (${net:,})")
    if buyers:
        print(f"Insider-kjøp ✓: {', '.join(buyers)}")
    else:
        print(f"Insider-signal: ingen signifikante kjøp blant {len(tickers)} tickere")
    return result
