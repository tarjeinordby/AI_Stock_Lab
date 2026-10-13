import json
import os
import io
import requests
import pandas as pd

EXTRA_WATCHLIST = [
    "PLTR", "ARM", "SMCI", "TSM", "ASML", "NVO", "SHOP", "SE",
    "COIN", "RBLX", "U", "SNOW", "MDB", "DDOG", "NET", "CRWD",
    "CELH", "ELF", "TOST", "APP", "HOOD", "SOFI", "MSTR",
    "UBER", "ABNB", "PANW", "ZS", "OKTA", "BILL", "ROKU",
    "DELL", "HIMS", "IONQ", "RGTI", "QBTS",
]

MEGACAP_TICKERS = {
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "AVGO",
    "TSLA", "BRK-B", "LLY", "JPM", "V", "MA", "NFLX", "COST",
    "ORCL", "AMD", "CRM", "ADBE", "QCOM", "CSCO", "PEP", "KO",
    "MCD", "WMT", "HD", "UNH", "ABBV", "NVO", "ASML", "TSM",
}

_STATIC_FILE = "data_v4/universe_tickers.json"
_MIN_UNIVERSE_SIZE = 200


def clean_ticker(ticker):
    return str(ticker).strip().replace(".", "-")


def _load_static_universe():
    """Load tickers from the committed static file. Returns (sp500, nasdaq100) lists."""
    if not os.path.exists(_STATIC_FILE):
        return [], []
    try:
        with open(_STATIC_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("sp500", []), data.get("nasdaq100", [])
    except Exception as e:
        print(f"Kunne ikke lese static universe-fil: {e}")
        return [], []


def fetch_sp500_tickers():
    """Fetch S&P 500 tickers from Wikipedia with browser headers."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
        }
        r = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers,
            timeout=20,
        )
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
        df = tables[0]
        return [clean_ticker(x) for x in df["Symbol"].tolist()]
    except Exception as e:
        print(f"Kunne ikke hente S&P 500 fra Wikipedia: {e}")
        return []


def fetch_nasdaq100_tickers():
    """Fetch Nasdaq 100 tickers from Wikipedia with browser headers."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
        }
        r = requests.get(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            headers=headers,
            timeout=20,
        )
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
        for table in tables:
            cols = [str(c).lower() for c in table.columns]
            if "ticker" in cols:
                col = table.columns[cols.index("ticker")]
                return [clean_ticker(x) for x in table[col].dropna().tolist()]
            if "symbol" in cols:
                col = table.columns[cols.index("symbol")]
                return [clean_ticker(x) for x in table[col].dropna().tolist()]
        return []
    except Exception as e:
        print(f"Kunne ikke hente Nasdaq 100 fra Wikipedia: {e}")
        return []


def build_universe():
    """
    Build the trading universe.

    Primary source: data_v4/universe_tickers.json (static, committed to repo).
    Fallback: Wikipedia scraping with browser headers.
    Always merges in EXTRA_WATCHLIST regardless of source.

    Alerts if universe falls below _MIN_UNIVERSE_SIZE — indicates the static
    file is missing and Wikipedia scraping also failed.
    """
    sp500, nasdaq100 = _load_static_universe()
    source = "static"

    if len(sp500) < _MIN_UNIVERSE_SIZE:
        print(
            f"Static universe insufficient ({len(sp500)} S&P 500 tickers) — "
            "prøver Wikipedia..."
        )
        sp500 = fetch_sp500_tickers()
        nasdaq100 = fetch_nasdaq100_tickers()
        source = "wikipedia"

    universe = sorted(set(sp500 + nasdaq100 + EXTRA_WATCHLIST + ["SPY", "QQQ"]))

    if len(universe) < _MIN_UNIVERSE_SIZE:
        print(
            f"⚠️ ADVARSEL: Univers er uventet lite ({len(universe)} tickere) — "
            f"static fil mangler og Wikipedia-henting feilet. "
            f"Commit data_v4/universe_tickers.json eller refresh manuelt."
        )

    print(
        f"Univers: {len(universe)} tickere "
        f"(S&P500={len(sp500)}, NDX={len(nasdaq100)}, "
        f"watchlist={len(EXTRA_WATCHLIST)}, kilde={source})"
    )
    return universe


def refresh_universe_file():
    """
    Fetch fresh S&P 500 + Nasdaq 100 from Wikipedia and update the static file.
    Run manually (~quarterly) via BOT_MODE=refresh_universe.
    """
    sp500 = fetch_sp500_tickers()
    nasdaq100 = fetch_nasdaq100_tickers()

    if len(sp500) < _MIN_UNIVERSE_SIZE:
        raise RuntimeError(
            f"Refresh feilet: kun {len(sp500)} S&P 500 tickere hentet — "
            "sjekk Wikipedia-tilgang"
        )

    from modules.state import today_str
    payload = {
        "sp500": sorted(set(sp500)),
        "nasdaq100": sorted(set(nasdaq100)),
        "extra_watchlist": sorted(EXTRA_WATCHLIST),
        "updated": today_str(),
        "note": "Static fallback — refresh manually ~quarterly via BOT_MODE=refresh_universe.",
    }
    with open(_STATIC_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    total = len(set(sp500 + nasdaq100 + list(EXTRA_WATCHLIST)))
    print(f"Universe oppdatert: {total} unike tickere → {_STATIC_FILE}")
    return payload
