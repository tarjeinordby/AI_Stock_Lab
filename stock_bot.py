import os
import math
import warnings
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf


warnings.simplefilter(action="ignore", category=FutureWarning)


# =========================
# TELEGRAM
# =========================

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        raise ValueError("Mangler TELEGRAM_BOT_TOKEN eller TELEGRAM_CHAT_ID i GitHub Secrets.")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    # Telegram har maksgrense per melding. Vi splitter hvis teksten blir lang.
    chunks = split_message(message, max_length=3800)

    for chunk in chunks:
        payload = {
            "chat_id": CHAT_ID,
            "text": chunk
        }

        response = requests.post(url, json=payload)

        if response.status_code == 200:
            print("Melding sendt til Telegram ✅")
        else:
            print("Feil ved sending:", response.text)
            response.raise_for_status()


def split_message(message, max_length=3800):
    lines = message.split("\n")
    chunks = []
    current = ""

    for line in lines:
        if len(current) + len(line) + 1 > max_length:
            chunks.append(current)
            current = line
        else:
            current += "\n" + line if current else line

    if current:
        chunks.append(current)

    return chunks


# =========================
# UNIVERS: S&P 500 + NASDAQ 100
# =========================

EXTRA_WATCHLIST = [
    "PLTR", "ARM", "SMCI", "TSM", "ASML", "NVO", "SHOP", "SE",
    "COIN", "RBLX", "U", "SNOW", "MDB", "DDOG", "NET", "CRWD",
    "CELH", "ELF", "TOST", "APP", "HOOD", "SOFI"
]


def clean_ticker(ticker):
    # Yahoo Finance bruker BRK-B i stedet for BRK.B
    return str(ticker).strip().replace(".", "-")


def fetch_sp500_tickers():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        tickers = [clean_ticker(x) for x in df["Symbol"].tolist()]
        return tickers
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
                ticker_col = table.columns[cols.index("ticker")]
                return [clean_ticker(x) for x in table[ticker_col].dropna().tolist()]

            if "symbol" in cols:
                symbol_col = table.columns[cols.index("symbol")]
                return [clean_ticker(x) for x in table[symbol_col].dropna().tolist()]

        return []

    except Exception as e:
        print(f"Kunne ikke hente Nasdaq 100: {e}")
        return []


def build_universe():
    sp500 = fetch_sp500_tickers()
    nasdaq100 = fetch_nasdaq100_tickers()

    universe = sorted(set(sp500 + nasdaq100 + EXTRA_WATCHLIST + ["SPY", "QQQ"]))

    print(f"Antall tickere i univers: {len(universe)}")
    return universe


# =========================
# DATAHENTING
# =========================

def download_market_data(tickers, chunk_size=80):
    all_data = {}

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]

        print(f"Henter data for {len(chunk)} tickere...")

        try:
            raw = yf.download(
                tickers=chunk,
                period="1y",
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False
            )

            for ticker in chunk:
                df = extract_ticker_data(raw, ticker)

                if df is not None and not df.empty:
                    all_data[ticker] = df

        except Exception as e:
            print(f"Feil ved nedlasting av chunk: {e}")

    print(f"Data hentet for {len(all_data)} tickere")
    return all_data


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

        if len(df) < 220:
            return None

        return df

    except Exception:
        return None


# =========================
# INDIKATORER
# =========================

def calculate_rsi(close, period=14):
    delta = close.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def pct_change(current, previous):
    if previous is None or previous == 0 or math.isnan(previous):
        return None

    return ((current / previous) - 1) * 100


def safe_round(value, digits=2):
    if value is None:
        return None

    try:
        if math.isnan(value):
            return None

        return round(float(value), digits)

    except Exception:
        return None


# =========================
# AKSJEANALYSE
# =========================

def analyze_stock(ticker, df, spy_3m_return):
    try:
        close = df["Close"]
        volume = df["Volume"]

        price = float(close.iloc[-1])

        if price < 10:
            return None

        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1])

        rsi = float(calculate_rsi(close).iloc[-1])

        price_1m_ago = float(close.iloc[-21])
        price_3m_ago = float(close.iloc[-63])
        price_6m_ago = float(close.iloc[-126])

        ret_1m = pct_change(price, price_1m_ago)
        ret_3m = pct_change(price, price_3m_ago)
        ret_6m = pct_change(price, price_6m_ago)

        high_52w = float(close.max())
        low_52w = float(close.min())

        distance_from_high = pct_change(price, high_52w)
        distance_from_low = pct_change(price, low_52w)

        avg_volume_20 = float(volume.tail(20).mean())
        latest_volume = float(volume.iloc[-1])
        volume_ratio = latest_volume / avg_volume_20 if avg_volume_20 > 0 else 1

        avg_dollar_volume = avg_volume_20 * price

        # Filtrer bort illikvide aksjer
        if avg_dollar_volume < 50_000_000:
            return None

        daily_returns = close.pct_change().dropna()

        volatility_60d = float(daily_returns.tail(60).std() * math.sqrt(252) * 100)

        drawdown_3m = pct_change(price, float(close.tail(63).max()))

        relative_strength_3m = None
        if spy_3m_return is not None and ticker not in ["SPY", "QQQ"]:
            relative_strength_3m = ret_3m - spy_3m_return

        trend_score = 0

        if price > sma20:
            trend_score += 1

        if price > sma50:
            trend_score += 2

        if price > sma200:
            trend_score += 2

        if sma20 > sma50 > sma200:
            trend_score += 3
        elif sma50 > sma200:
            trend_score += 1

        healthy_rsi = 45 <= rsi <= 70
        overbought = rsi > 75
        weak_rsi = rsi < 40

        near_high = distance_from_high is not None and distance_from_high > -12

        return {
            "ticker": ticker,
            "price": price,
            "ret_1m": ret_1m,
            "ret_3m": ret_3m,
            "ret_6m": ret_6m,
            "relative_strength_3m": relative_strength_3m,
            "rsi": rsi,
            "volatility_60d": volatility_60d,
            "volume_ratio": volume_ratio,
            "distance_from_high": distance_from_high,
            "distance_from_low": distance_from_low,
            "drawdown_3m": drawdown_3m,
            "trend_score": trend_score,
            "healthy_rsi": healthy_rsi,
            "overbought": overbought,
            "weak_rsi": weak_rsi,
            "near_high": near_high,
            "avg_dollar_volume": avg_dollar_volume,
            "above_sma50": price > sma50,
            "above_sma200": price > sma200,
            "sma20_above_sma50_above_sma200": sma20 > sma50 > sma200
        }

    except Exception as e:
        print(f"Analysefeil for {ticker}: {e}")
        return None


# =========================
# STRATEGIER
# =========================

def score_momentum_ai(stock):
    score = 0

    score += stock["ret_1m"] * 0.25
    score += stock["ret_3m"] * 0.55
    score += stock["ret_6m"] * 0.30

    if stock["relative_strength_3m"] is not None:
        score += stock["relative_strength_3m"] * 0.70

    score += stock["trend_score"] * 2.0

    if stock["near_high"]:
        score += 6

    if stock["overbought"]:
        score -= 5

    if not stock["above_sma50"]:
        score -= 10

    if stock["ret_1m"] < -5:
        score -= 8

    return score


def score_quality_momentum_ai(stock):
    score = 0

    score += stock["ret_3m"] * 0.35
    score += stock["ret_6m"] * 0.45

    if stock["relative_strength_3m"] is not None:
        score += stock["relative_strength_3m"] * 0.45

    score += stock["trend_score"] * 2.5

    if stock["healthy_rsi"]:
        score += 8

    if stock["near_high"]:
        score += 5

    # Straffer for høy volatilitet
    score -= stock["volatility_60d"] * 0.25

    if stock["overbought"]:
        score -= 4

    if not stock["above_sma200"]:
        score -= 15

    return score


def score_aggressive_ai(stock):
    score = 0

    score += stock["ret_1m"] * 0.65
    score += stock["ret_3m"] * 0.65
    score += stock["ret_6m"] * 0.20

    if stock["relative_strength_3m"] is not None:
        score += stock["relative_strength_3m"] * 0.60

    score += stock["volume_ratio"] * 4
    score += stock["trend_score"] * 1.5

    if stock["near_high"]:
        score += 7

    # Aggressive_AI tåler mer volatilitet, men ikke total svakhet
    if not stock["above_sma50"]:
        score -= 12

    if stock["ret_1m"] < 0:
        score -= 5

    return score


def score_low_risk_ai(stock):
    score = 0

    score += stock["ret_3m"] * 0.25
    score += stock["ret_6m"] * 0.30

    if stock["relative_strength_3m"] is not None:
        score += stock["relative_strength_3m"] * 0.30

    score += stock["trend_score"] * 3.0

    if stock["healthy_rsi"]:
        score += 8

    if stock["near_high"]:
        score += 4

    # Hovedpoenget: lavere volatilitet og mindre drawdown
    score -= stock["volatility_60d"] * 0.65

    if stock["drawdown_3m"] is not None:
        score += stock["drawdown_3m"] * 0.35  # drawdown er negativ, så dette straffer store fall

    if not stock["above_sma200"]:
        score -= 20

    return score


def score_balanced_ai(stock):
    score = 0

    score += stock["ret_1m"] * 0.15
    score += stock["ret_3m"] * 0.40
    score += stock["ret_6m"] * 0.35

    if stock["relative_strength_3m"] is not None:
        score += stock["relative_strength_3m"] * 0.45

    score += stock["trend_score"] * 2.5

    if stock["healthy_rsi"]:
        score += 7

    if stock["near_high"]:
        score += 5

    score -= stock["volatility_60d"] * 0.30

    if stock["overbought"]:
        score -= 4

    if not stock["above_sma50"]:
        score -= 8

    if not stock["above_sma200"]:
        score -= 12

    return score


STRATEGIES = {
    "Momentum_AI": score_momentum_ai,
    "Quality_Momentum_AI": score_quality_momentum_ai,
    "Aggressive_AI": score_aggressive_ai,
    "Low_Risk_AI": score_low_risk_ai,
    "Balanced_AI": score_balanced_ai
}


# =========================
# RANGERING
# =========================

def rank_stocks(analyzed_stocks):
    rankings = {}

    for strategy_name, strategy_function in STRATEGIES.items():
        scored = []

        for stock in analyzed_stocks:
            try:
                score = strategy_function(stock)
                item = stock.copy()
                item["strategy_score"] = score
                scored.append(item)
            except Exception:
                continue

        scored = sorted(scored, key=lambda x: x["strategy_score"], reverse=True)
        rankings[strategy_name] = scored

    return rankings


# =========================
# MELDING
# =========================

def format_stock_line(stock, rank):
    return (
        f"{rank}. {stock['ticker']} | "
        f"${safe_round(stock['price'], 2)} | "
        f"Score {safe_round(stock['strategy_score'], 1)} | "
        f"1M {safe_round(stock['ret_1m'], 1)}% | "
        f"3M {safe_round(stock['ret_3m'], 1)}% | "
        f"6M {safe_round(stock['ret_6m'], 1)}% | "
        f"RS {safe_round(stock['relative_strength_3m'], 1)}% | "
        f"RSI {safe_round(stock['rsi'], 0)} | "
        f"Vol {safe_round(stock['volatility_60d'], 0)}%"
    )


def format_market_context(spy, qqq):
    lines = []

    lines.append("MARKEDSKONTEKST")

    if spy:
        lines.append(
            f"SPY: 1M {safe_round(spy['ret_1m'], 1)}% | "
            f"3M {safe_round(spy['ret_3m'], 1)}% | "
            f"6M {safe_round(spy['ret_6m'], 1)}% | "
            f"Trend {spy['trend_score']}/8"
        )

    if qqq:
        lines.append(
            f"QQQ: 1M {safe_round(qqq['ret_1m'], 1)}% | "
            f"3M {safe_round(qqq['ret_3m'], 1)}% | "
            f"6M {safe_round(qqq['ret_6m'], 1)}% | "
            f"Trend {qqq['trend_score']}/8"
        )

    return "\n".join(lines)


def build_message(rankings, analyzed_count, universe_count, spy, qqq):
    today = datetime.utcnow().strftime("%Y-%m-%d")

    message = ""
    message += "📊 AI STOCK LAB v2\n"
    message += f"Dato: {today}\n"
    message += "Rapport: Pre-market watchlist\n"
    message += f"Univers: {universe_count} tickere\n"
    message += f"Analysert etter filtre: {analyzed_count} tickere\n\n"

    message += format_market_context(spy, qqq)
    message += "\n\n"

    strategy_titles = {
        "Momentum_AI": "🏆 Momentum_AI — sterkest relativ styrke",
        "Quality_Momentum_AI": "🧠 Quality_Momentum_AI — sterk trend + lavere risiko",
        "Aggressive_AI": "🔥 Aggressive_AI — høy momentum / høyere risiko",
        "Low_Risk_AI": "🛡 Low_Risk_AI — jevnere aksjer med lavere volatilitet",
        "Balanced_AI": "⚖️ Balanced_AI — samlet beste kandidater"
    }

    for strategy_name, stocks in rankings.items():
        message += "━━━━━━━━━━━━━━\n"
        message += strategy_titles.get(strategy_name, strategy_name)
        message += "\n"
        message += "━━━━━━━━━━━━━━\n"

        top = stocks[:5]

        if not top:
            message += "Ingen kandidater.\n\n"
            continue

        for i, stock in enumerate(top, start=1):
            message += format_stock_line(stock, i)
            message += "\n"

        message += "\n"

    message += "⚠️ Dette er ikke kjøpsordre. Kun beslutningsstøtte.\n"
    message += "Sjekk alltid nyheter, earnings, risiko og egen portefølje før handel.\n"
    message += "Neste steg: paper tracking for å måle strategiene mot SPY/QQQ."

    return message


# =========================
# KJØR BOT
# =========================

def run_bot():
    universe = build_universe()
    market_data = download_market_data(universe)

    spy = None
    qqq = None

    if "SPY" in market_data:
        spy = analyze_stock("SPY", market_data["SPY"], None)

    if "QQQ" in market_data:
        spy_3m = spy["ret_3m"] if spy else None
        qqq = analyze_stock("QQQ", market_data["QQQ"], spy_3m)

    spy_3m_return = spy["ret_3m"] if spy else None

    analyzed_stocks = []

    for ticker, df in market_data.items():
        if ticker in ["SPY", "QQQ"]:
            continue

        result = analyze_stock(ticker, df, spy_3m_return)

        if result:
            analyzed_stocks.append(result)

    rankings = rank_stocks(analyzed_stocks)

    message = build_message(
        rankings=rankings,
        analyzed_count=len(analyzed_stocks),
        universe_count=len(universe),
        spy=spy,
        qqq=qqq
    )

    send_telegram(message)


if __name__ == "__main__":
    run_bot()
