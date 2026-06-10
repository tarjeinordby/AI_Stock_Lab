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


def clean_ticker(ticker):
    return str(ticker).strip().replace(".", "-")


def fetch_sp500_tickers():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        return [clean_ticker(x) for x in df["Symbol"].tolist()]
    except Exception as e:
        print(f"Kunne ikke hente S&P 500: {e}")
        return []


def fetch_nasdaq100_tickers():
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url)
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
        print(f"Kunne ikke hente Nasdaq 100: {e}")
        return []


def build_universe():
    sp500 = fetch_sp500_tickers()
    nasdaq100 = fetch_nasdaq100_tickers()
    universe = sorted(set(sp500 + nasdaq100 + EXTRA_WATCHLIST + ["SPY", "QQQ"]))
    print(
        f"Univers: {len(universe)} tickere "
        f"(S&P500={len(sp500)}, NDX={len(nasdaq100)}, watchlist={len(EXTRA_WATCHLIST)})"
    )
    return universe
