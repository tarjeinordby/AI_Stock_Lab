import os
import requests
import yfinance as yf
from datetime import datetime


# =========================
# TELEGRAM
# =========================

BOT_TOKEN = os.getenv("REMOVED")
CHAT_ID = os.getenv("7386922633")


def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        raise ValueError("Mangler TELEGRAM_BOT_TOKEN eller TELEGRAM_CHAT_ID i GitHub Secrets.")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        print("Melding sendt til Telegram ✅")
    else:
        print("Feil ved sending:", response.text)
        response.raise_for_status()


# =========================
# AKSJELISTE
# =========================

tickers = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "AMD",
    "PLTR",
    "AVGO"
]


# =========================
# HJELPEFUNKSJONER
# =========================

def safe_float(value):
    try:
        if hasattr(value, "iloc"):
            return float(value.iloc[0])
        return float(value)
    except Exception:
        return None


# =========================
# ANALYSE
# =========================

def analyze_stock(ticker):
    data = yf.download(
        ticker,
        period="1y",
        interval="1d",
        progress=False,
        auto_adjust=True
    )

    if data.empty:
        return None

    data["SMA50"] = data["Close"].rolling(50).mean()
    data["SMA200"] = data["Close"].rolling(200).mean()

    latest = data.iloc[-1]

    price = safe_float(latest["Close"])
    sma50 = safe_float(latest["SMA50"])
    sma200 = safe_float(latest["SMA200"])

    if price is None or sma50 is None or sma200 is None:
        return None

    score = 0
    reasons = []

    if price > sma50:
        score += 1
        reasons.append("✅ Pris over 50-dagers snitt")
    else:
        reasons.append("❌ Pris under 50-dagers snitt")

    if price > sma200:
        score += 1
        reasons.append("✅ Pris over 200-dagers snitt")
    else:
        reasons.append("❌ Pris under 200-dagers snitt")

    if sma50 > sma200:
        score += 1
        reasons.append("✅ 50-dagers snitt over 200-dagers snitt")
    else:
        reasons.append("❌ 50-dagers snitt under 200-dagers snitt")

    if score == 3:
        conclusion = "STERK WATCHLIST"
    elif score == 2:
        conclusion = "MULIG WATCHLIST"
    elif score == 1:
        conclusion = "SVAK / VENT"
    else:
        conclusion = "UNNGÅ FORELØPIG"

    return {
        "ticker": ticker,
        "price": round(price, 2),
        "score": score,
        "conclusion": conclusion,
        "reasons": reasons
    }


# =========================
# KJØR BOT
# =========================

def run_bot():
    today = datetime.now().strftime("%Y-%m-%d")

    message = f"📊 AI Stock Alert\nDato: {today}\n\n"

    results = []

    for ticker in tickers:
        result = analyze_stock(ticker)
        if result:
            results.append(result)

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    for r in results:
        message += "━━━━━━━━━━━━━━\n"
        message += f"{r['ticker']}\n"
        message += f"Pris: ${r['price']}\n"
        message += f"Score: {r['score']}/3\n"
        message += f"Konklusjon: {r['conclusion']}\n"

        for reason in r["reasons"]:
            message += f"{reason}\n"

        message += "\n"

    message += "⚠️ Ikke kjøpsordre. Kun beslutningsstøtte."

    send_telegram(message)


if __name__ == "__main__":
    run_bot()
