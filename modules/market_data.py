import pandas as pd
import yfinance as yf

MIN_PRICE = 10
MIN_AVG_DOLLAR_VOLUME = 25_000_000


def extract_ticker_data(raw, ticker):
    try:
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker not in raw.columns.get_level_values(0):
                return None
            df = raw[ticker].copy()
        else:
            df = raw.copy()
        needed = ["Open", "High", "Low", "Close", "Volume"]
        for col in needed:
            if col not in df.columns:
                return None
        df = df[needed].dropna()
        if len(df) < 260:
            return None
        return df
    except Exception:
        return None


def download_daily_data(tickers, chunk_size=80):
    all_data = {}
    total_chunks = (len(tickers) + chunk_size - 1) // chunk_size
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        chunk_num = i // chunk_size + 1
        print(f"Henter prisdata chunk {chunk_num}/{total_chunks} ({len(chunk)} tickere)...")
        try:
            raw = yf.download(
                tickers=chunk,
                period="2y",
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            for ticker in chunk:
                df = extract_ticker_data(raw, ticker)
                if df is not None and not df.empty:
                    all_data[ticker] = df
        except Exception as e:
            print(f"Nedlastingsfeil chunk {chunk_num}: {e}")
    print(f"Prisdata hentet for {len(all_data)}/{len(tickers)} tickere")
    return all_data


def get_latest_price(ticker, fallback_price=None):
    try:
        data = yf.download(
            tickers=ticker,
            period="5d",
            interval="1m",
            auto_adjust=False,
            progress=False,
            prepost=False,
        )
        if data is not None and not data.empty:
            close = data["Close"].squeeze().dropna()
            if not close.empty:
                price = float(close.iloc[-1])
                if price > 0:
                    return price
    except Exception as e:
        print(f"Intraday pris feilet for {ticker}: {e}")
    return fallback_price


def get_price_cached(ticker, fallback_price, price_cache):
    if ticker in price_cache:
        return price_cache[ticker]
    price = get_latest_price(ticker, fallback_price)
    price_cache[ticker] = price
    return price


def _get_latest_close_from_raw(raw, ticker):
    """Extract latest close price for a ticker from a yf.download result."""
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker not in raw.columns.get_level_values(0):
                return None
            close = raw[ticker]["Close"].dropna()
        else:
            close = raw["Close"].squeeze().dropna()
        if close.empty:
            return None
        return float(close.iloc[-1])
    except Exception:
        return None


def compute_premarket_moves(tickers, prev_prices, chunk_size=40):
    """
    Fetch pre-market prices (including pre/post session) and compute
    % change vs prev_prices (yesterday's close from signal).

    Returns {ticker: pct_change} for all tickers with data.
    """
    if not tickers:
        return {}

    moves = {}
    print(f"Henter pre-market priser for {len(tickers)} tickere...")

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        try:
            raw = yf.download(
                tickers=chunk,
                period="1d",
                interval="1m",
                auto_adjust=False,
                prepost=True,
                group_by="ticker",
                progress=False,
            )
            if raw is None or raw.empty:
                continue

            for ticker in chunk:
                prev = prev_prices.get(ticker)
                if not prev or prev <= 0:
                    continue
                current = _get_latest_close_from_raw(raw, ticker)
                if current and current > 0:
                    moves[ticker] = (current / prev) - 1

        except Exception as e:
            print(f"  Pre-market chunk feil: {e}")

    flagged = {t: m for t, m in moves.items() if abs(m) > 0.04}
    if flagged:
        print(f"Pre-market bevegelser >4%: {len(flagged)} tickere")
        for t, m in sorted(flagged.items(), key=lambda x: -abs(x[1])):
            arrow = "↑" if m > 0 else "↓"
            print(f"  {t}: {m:+.1%} {arrow}")
    else:
        print("Pre-market: ingen bevegelser >4%")

    return moves


def get_vix():
    try:
        data = yf.download(
            tickers="^VIX",
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if data is not None and not data.empty:
            close = data["Close"].squeeze().dropna()  # squeeze: DataFrame→Series
            if not close.empty:
                vix = float(close.iloc[-1])
                print(f"VIX: {vix:.1f}")
                return vix
    except Exception as e:
        print(f"VIX feil: {e}")
    return None
