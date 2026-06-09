import os
import json
import math
import warnings
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf


warnings.simplefilter(action="ignore", category=FutureWarning)


# ============================================================
# AI STOCK LAB v3
# Multi-strategy long-term paper portfolio lab
#
# Modes:
#   BOT_MODE=signal   -> builds and saves signal snapshot silently
#   BOT_MODE=execute  -> updates paper portfolios and sends Telegram report
#
# Goal:
#   Test multiple long-term portfolio strategies against SPY/QQQ.
# ============================================================


# =========================
# GLOBAL CONFIG
# =========================

START_CAPITAL = 10_000.0

DATA_DIR = "data_v3"
STATE_DIR = f"{DATA_DIR}/state"
SIGNALS_DIR = f"{DATA_DIR}/signals"
REPORTS_DIR = f"{DATA_DIR}/reports"

TRADES_FILE = f"{DATA_DIR}/trades.csv"
PERFORMANCE_FILE = f"{DATA_DIR}/performance.csv"
BENCHMARK_FILE = f"{STATE_DIR}/benchmarks.json"

MIN_PRICE = 10
MIN_AVG_DOLLAR_VOLUME = 25_000_000

TOP_CANDIDATES_TO_SAVE = 150
TOP_CANDIDATES_TO_REPORT = 5

OSLO = ZoneInfo("Europe/Oslo")
NY = ZoneInfo("America/New_York")


# =========================
# STRATEGY CONFIG
# =========================

STRATEGIES = {
    "Max_Return_AI": {
        "description": "Mest aggressiv. Jakter høyest mulig avkastning med mer risiko.",
        "score_column": "score_max_return",
        "max_positions": 15,
        "max_position_weight": 0.12,
        "max_new_buys_per_week": 5,
        "buy_top_n": 20,
        "min_score_percentile": 0.80,
        "sell_rank_threshold": 120,
        "stop_loss": -0.22,
        "trailing_stop": -0.35,
        "buyback_cooldown_days": 7,
        "exposure": {
            "bullish": 1.00,
            "neutral": 0.70,
            "defensive": 0.40,
            "unknown": 0.60
        }
    },
    "Quality_Momentum_AI": {
        "description": "Sterk momentum, men med mer kvalitets- og risikokontroll.",
        "score_column": "score_quality_momentum",
        "max_positions": 15,
        "max_position_weight": 0.10,
        "max_new_buys_per_week": 4,
        "buy_top_n": 20,
        "min_score_percentile": 0.78,
        "sell_rank_threshold": 110,
        "stop_loss": -0.20,
        "trailing_stop": -0.30,
        "buyback_cooldown_days": 10,
        "exposure": {
            "bullish": 0.95,
            "neutral": 0.65,
            "defensive": 0.35,
            "unknown": 0.55
        }
    },
    "Balanced_AI": {
        "description": "Balanserer momentum, trend, risiko og stabilitet.",
        "score_column": "score_balanced",
        "max_positions": 15,
        "max_position_weight": 0.09,
        "max_new_buys_per_week": 4,
        "buy_top_n": 18,
        "min_score_percentile": 0.75,
        "sell_rank_threshold": 110,
        "stop_loss": -0.20,
        "trailing_stop": -0.30,
        "buyback_cooldown_days": 10,
        "exposure": {
            "bullish": 0.90,
            "neutral": 0.60,
            "defensive": 0.30,
            "unknown": 0.50
        }
    },
    "Low_Risk_AI": {
        "description": "Prøver å slå indeks med lavere svingninger og mindre drawdown.",
        "score_column": "score_low_risk",
        "max_positions": 15,
        "max_position_weight": 0.08,
        "max_new_buys_per_week": 3,
        "buy_top_n": 25,
        "min_score_percentile": 0.70,
        "sell_rank_threshold": 100,
        "stop_loss": -0.16,
        "trailing_stop": -0.24,
        "buyback_cooldown_days": 10,
        "exposure": {
            "bullish": 0.75,
            "neutral": 0.50,
            "defensive": 0.25,
            "unknown": 0.45
        }
    },
    "MegaCap_AI": {
        "description": "Fokuserer på store, mer robuste kvalitetsselskaper.",
        "score_column": "score_megacap",
        "max_positions": 12,
        "max_position_weight": 0.11,
        "max_new_buys_per_week": 3,
        "buy_top_n": 25,
        "min_score_percentile": 0.65,
        "sell_rank_threshold": 100,
        "stop_loss": -0.18,
        "trailing_stop": -0.28,
        "buyback_cooldown_days": 10,
        "exposure": {
            "bullish": 0.85,
            "neutral": 0.55,
            "defensive": 0.30,
            "unknown": 0.50
        }
    }
}


MEGACAP_TICKERS = {
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "AVGO",
    "TSLA", "BRK-B", "LLY", "JPM", "V", "MA", "NFLX", "COST",
    "ORCL", "AMD", "CRM", "ADBE", "QCOM", "CSCO", "PEP", "KO",
    "MCD", "WMT", "HD", "UNH", "ABBV", "NVO", "ASML", "TSM"
}


# =========================
# TELEGRAM
# =========================

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


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


def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram secrets mangler. Hopper over sending.")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    for chunk in split_message(message):
        payload = {
            "chat_id": CHAT_ID,
            "text": chunk
        }

        response = requests.post(url, json=payload, timeout=30)

        if response.status_code == 200:
            print("Melding sendt til Telegram ✅")
        else:
            print("Telegram-feil:", response.text)
            response.raise_for_status()


# =========================
# TIME / FILE HELPERS
# =========================

def now_utc():
    return datetime.now(timezone.utc)


def now_oslo():
    return now_utc().astimezone(OSLO)


def now_ny():
    return now_utc().astimezone(NY)


def today_str():
    return now_oslo().strftime("%Y-%m-%d")


def ensure_dirs():
    for path in [DATA_DIR, STATE_DIR, SIGNALS_DIR, REPORTS_DIR]:
        os.makedirs(path, exist_ok=True)


def read_csv_or_empty(path, columns):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame(columns=columns)

    return pd.DataFrame(columns=columns)


def save_csv(df, path):
    ensure_dirs()
    df.to_csv(path, index=False)


def to_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [to_json_safe(v) for v in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        if math.isnan(float(obj)):
            return None
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj


def save_json(payload, path):
    ensure_dirs()
    with open(path, "w") as f:
        json.dump(to_json_safe(payload), f, indent=2, sort_keys=True)


def load_json(path, default):
    if not os.path.exists(path):
        return default

    with open(path, "r") as f:
        return json.load(f)


# =========================
# UNIVERSE
# =========================

EXTRA_WATCHLIST = [
    "PLTR", "ARM", "SMCI", "TSM", "ASML", "NVO", "SHOP", "SE",
    "COIN", "RBLX", "U", "SNOW", "MDB", "DDOG", "NET", "CRWD",
    "CELH", "ELF", "TOST", "APP", "HOOD", "SOFI", "MSTR",
    "UBER", "ABNB", "PANW", "ZS", "OKTA", "BILL", "ROKU",
    "DELL", "HIMS", "IONQ", "RGTI", "QBTS"
]


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

    print(f"Univers: {len(universe)} tickere")
    return universe


# =========================
# MARKET DATA
# =========================

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

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        print(f"Henter daglige data for {len(chunk)} tickere...")

        try:
            raw = yf.download(
                tickers=chunk,
                period="2y",
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
            print(f"Nedlastingsfeil chunk: {e}")

    print(f"Data hentet for {len(all_data)} tickere")
    return all_data


def get_latest_price(ticker, fallback_price=None):
    try:
        data = yf.download(
            tickers=ticker,
            period="5d",
            interval="1m",
            auto_adjust=False,
            progress=False,
            prepost=False
        )

        if data is not None and not data.empty:
            close = data["Close"].dropna()

            if not close.empty:
                price = float(close.iloc[-1])

                if price > 0:
                    return price

    except Exception as e:
        print(f"Intraday pris feilet for {ticker}: {e}")

    return fallback_price


# =========================
# INDICATORS
# =========================

def safe_float(x, default=None):
    try:
        if x is None:
            return default

        value = float(x)

        if math.isnan(value):
            return default

        return value

    except Exception:
        return default


def safe_round(x, digits=2):
    value = safe_float(x)

    if value is None:
        return "N/A"

    return round(value, digits)


def pct_change(current, previous):
    try:
        if previous is None or previous == 0:
            return None

        if math.isnan(previous):
            return None

        return (current / previous) - 1

    except Exception:
        return None


def calculate_rsi(close, period=14):
    delta = close.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def percentile_rank(series, higher_is_better=True):
    s = pd.Series(series).astype(float)

    if not higher_is_better:
        s = -s

    return s.rank(pct=True)


# =========================
# STOCK ANALYSIS
# =========================

def analyze_stock(ticker, df, spy_ret_3m=None):
    try:
        close = df["Close"]
        volume = df["Volume"]

        price = float(close.iloc[-1])

        if price < MIN_PRICE:
            return None

        avg_volume_20 = float(volume.tail(20).mean())
        avg_dollar_volume = avg_volume_20 * price

        if avg_dollar_volume < MIN_AVG_DOLLAR_VOLUME:
            return None

        sma50 = float(close.rolling(50).mean().iloc[-1])
        sma100 = float(close.rolling(100).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1])

        rsi = float(calculate_rsi(close).iloc[-1])

        price_1m = float(close.iloc[-21])
        price_3m = float(close.iloc[-63])
        price_6m = float(close.iloc[-126])
        price_12m = float(close.iloc[-252])

        ret_1m = pct_change(price, price_1m)
        ret_3m = pct_change(price, price_3m)
        ret_6m = pct_change(price, price_6m)
        ret_12m = pct_change(price, price_12m)

        mom_12_1 = pct_change(price_1m, price_12m)
        mom_6_1 = pct_change(price_1m, price_6m)
        mom_3_1 = pct_change(price_1m, price_3m)

        if None in [ret_1m, ret_3m, ret_6m, ret_12m, mom_12_1, mom_6_1, mom_3_1]:
            return None

        daily_returns = close.pct_change().dropna()
        vol60 = float(daily_returns.tail(60).std() * math.sqrt(252))
        vol20 = float(daily_returns.tail(20).std() * math.sqrt(252))

        high_52w = float(close.tail(252).max())
        low_52w = float(close.tail(252).min())

        distance_from_high = pct_change(price, high_52w)
        distance_from_low = pct_change(price, low_52w)

        drawdown_3m = pct_change(price, float(close.tail(63).max()))
        drawdown_6m = pct_change(price, float(close.tail(126).max()))

        relative_strength_3m = None

        if spy_ret_3m is not None and ticker not in ["SPY", "QQQ"]:
            relative_strength_3m = ret_3m - spy_ret_3m

        trend_score = 0

        if price > sma50:
            trend_score += 1

        if price > sma100:
            trend_score += 1

        if price > sma200:
            trend_score += 2

        if sma50 > sma100 > sma200:
            trend_score += 2

        if price > high_52w * 0.90:
            trend_score += 1

        above_sma200 = price > sma200
        healthy_rsi = 45 <= rsi <= 72
        overbought = rsi > 78
        very_weak_rsi = rsi < 38
        near_high = price > high_52w * 0.90

        return {
            "ticker": ticker,
            "price": price,
            "ret_1m": ret_1m,
            "ret_3m": ret_3m,
            "ret_6m": ret_6m,
            "ret_12m": ret_12m,
            "mom_12_1": mom_12_1,
            "mom_6_1": mom_6_1,
            "mom_3_1": mom_3_1,
            "relative_strength_3m": relative_strength_3m,
            "vol60": vol60,
            "vol20": vol20,
            "rsi": rsi,
            "distance_from_high": distance_from_high,
            "distance_from_low": distance_from_low,
            "drawdown_3m": drawdown_3m,
            "drawdown_6m": drawdown_6m,
            "trend_score": trend_score,
            "above_sma200": above_sma200,
            "healthy_rsi": healthy_rsi,
            "overbought": overbought,
            "very_weak_rsi": very_weak_rsi,
            "near_high": near_high,
            "avg_dollar_volume": avg_dollar_volume,
            "sma50": sma50,
            "sma100": sma100,
            "sma200": sma200,
            "is_megacap": ticker in MEGACAP_TICKERS
        }

    except Exception as e:
        print(f"Analysefeil {ticker}: {e}")
        return None


# =========================
# SCORE ENGINE
# =========================

def build_score_table(analyzed_stocks):
    df = pd.DataFrame(analyzed_stocks)

    if df.empty:
        return df

    df = df.set_index("ticker")

    factor_cols = [
        "mom_12_1", "mom_6_1", "mom_3_1", "ret_1m", "ret_3m", "ret_6m",
        "relative_strength_3m", "vol60", "drawdown_6m", "trend_score"
    ]

    for col in factor_cols:
        if col in df.columns:
            lower = df[col].quantile(0.02)
            upper = df[col].quantile(0.98)
            df[col] = df[col].clip(lower, upper)

    df["rank_mom_12_1"] = percentile_rank(df["mom_12_1"], True)
    df["rank_mom_6_1"] = percentile_rank(df["mom_6_1"], True)
    df["rank_mom_3_1"] = percentile_rank(df["mom_3_1"], True)
    df["rank_ret_1m"] = percentile_rank(df["ret_1m"], True)
    df["rank_ret_3m"] = percentile_rank(df["ret_3m"], True)
    df["rank_rs"] = percentile_rank(df["relative_strength_3m"].fillna(0), True)
    df["rank_trend"] = percentile_rank(df["trend_score"], True)
    df["rank_low_vol"] = percentile_rank(df["vol60"], False)
    df["rank_low_drawdown"] = percentile_rank(df["drawdown_6m"], True)

    df["momentum_core"] = (
        0.45 * df["rank_mom_12_1"] +
        0.25 * df["rank_mom_6_1"] +
        0.15 * df["rank_mom_3_1"] +
        0.15 * df["rank_rs"]
    )

    df["short_momentum"] = (
        0.55 * df["rank_ret_1m"] +
        0.30 * df["rank_ret_3m"] +
        0.15 * df["rank_rs"]
    )

    df["stability"] = (
        0.65 * df["rank_low_vol"] +
        0.35 * df["rank_low_drawdown"]
    )

    df["rsi_quality"] = np.where(
        df["healthy_rsi"],
        1.0,
        np.where(df["overbought"], 0.35, np.where(df["very_weak_rsi"], 0.20, 0.60))
    )

    df["quality_proxy"] = (
        0.40 * df["rank_trend"] +
        0.35 * df["stability"] +
        0.25 * df["rsi_quality"]
    )

    df["score_max_return"] = (
        0.50 * df["momentum_core"] +
        0.25 * df["short_momentum"] +
        0.15 * df["rank_trend"] +
        0.10 * df["rank_rs"]
    )

    df["score_quality_momentum"] = (
        0.42 * df["momentum_core"] +
        0.25 * df["quality_proxy"] +
        0.20 * df["rank_trend"] +
        0.13 * df["stability"]
    )

    df["score_balanced"] = (
        0.35 * df["momentum_core"] +
        0.25 * df["quality_proxy"] +
        0.20 * df["rank_trend"] +
        0.20 * df["stability"]
    )

    df["score_low_risk"] = (
        0.20 * df["momentum_core"] +
        0.20 * df["quality_proxy"] +
        0.20 * df["rank_trend"] +
        0.40 * df["stability"]
    )

    df["score_megacap"] = (
        0.30 * df["momentum_core"] +
        0.30 * df["quality_proxy"] +
        0.20 * df["rank_trend"] +
        0.20 * df["stability"]
    )

    df.loc[~df["is_megacap"], "score_megacap"] *= 0.10

    score_cols = [
        "score_max_return",
        "score_quality_momentum",
        "score_balanced",
        "score_low_risk",
        "score_megacap"
    ]

    for col in score_cols:
        df.loc[~df["above_sma200"], col] *= 0.45
        df.loc[df["overbought"], col] *= 0.90
        df.loc[df["very_weak_rsi"], col] *= 0.70
        df.loc[df["vol60"] > 1.20, col] *= 0.80

        df[col] = df[col] * 100

    return df


# =========================
# MARKET REGIME
# =========================

def detect_market_regime(spy, qqq):
    if spy is None or qqq is None:
        return {
            "regime": "unknown",
            "reason": "Mangler full benchmark-data"
        }

    spy_good = spy["above_sma200"] and spy["ret_3m"] > 0
    qqq_good = qqq["above_sma200"] and qqq["ret_3m"] > 0

    if spy_good and qqq_good:
        return {
            "regime": "bullish",
            "reason": "SPY og QQQ er over 200SMA og har positiv 3M trend"
        }

    if spy["above_sma200"] or qqq["above_sma200"]:
        return {
            "regime": "neutral",
            "reason": "Markedet er blandet"
        }

    return {
        "regime": "defensive",
        "reason": "SPY/QQQ viser svak trend"
    }


# =========================
# SIGNAL MODE
# =========================

def make_strategy_candidates(score_df, strategy_name, config):
    score_col = config["score_column"]

    df = score_df.copy()
    df = df.sort_values(score_col, ascending=False)
    df["strategy_score"] = df[score_col]
    df["rank"] = range(1, len(df) + 1)
    df["score_percentile"] = df[score_col].rank(pct=True)

    candidates = df.head(TOP_CANDIDATES_TO_SAVE).reset_index().to_dict(orient="records")

    return candidates


def save_signal_snapshot(score_df, regime, spy, qqq, universe_count, analyzed_count):
    date = today_str()
    path = f"{SIGNALS_DIR}/{date}_signal.json"

    strategies_payload = {}

    for strategy_name, config in STRATEGIES.items():
        candidates = make_strategy_candidates(score_df, strategy_name, config)

        strategies_payload[strategy_name] = {
            "description": config["description"],
            "score_column": config["score_column"],
            "candidates": candidates
        }

    payload = {
        "created_at_utc": now_utc().isoformat(),
        "created_at_oslo": now_oslo().isoformat(),
        "created_at_ny": now_ny().isoformat(),
        "date": date,
        "universe_count": universe_count,
        "analyzed_count": analyzed_count,
        "regime": regime,
        "spy": spy,
        "qqq": qqq,
        "strategies": strategies_payload
    }

    save_json(payload, path)

    save_json(
        {
            "last_signal_file": path,
            "updated_at_utc": now_utc().isoformat()
        },
        f"{STATE_DIR}/_global.json"
    )

    return path


def run_signal():
    universe = build_universe()
    market_data = download_daily_data(universe)

    spy = None
    qqq = None

    if "SPY" in market_data:
        spy = analyze_stock("SPY", market_data["SPY"], None)

    spy_3m = spy["ret_3m"] if spy else None

    if "QQQ" in market_data:
        qqq = analyze_stock("QQQ", market_data["QQQ"], spy_3m)

    analyzed = []

    for ticker, df in market_data.items():
        if ticker in ["SPY", "QQQ"]:
            continue

        result = analyze_stock(ticker, df, spy_3m)

        if result:
            analyzed.append(result)

    score_df = build_score_table(analyzed)

    if score_df.empty:
        raise ValueError("Ingen aksjer ble analysert. Sjekk datakilde.")

    regime = detect_market_regime(spy, qqq)

    signal_file = save_signal_snapshot(
        score_df=score_df,
        regime=regime,
        spy=spy,
        qqq=qqq,
        universe_count=len(universe),
        analyzed_count=len(analyzed)
    )

    print(f"Signal lagret: {signal_file}")
    print("Signal-modus er stille. Ingen Telegram-melding sendt.")


# =========================
# PORTFOLIO STATE
# =========================

def strategy_state_file(strategy_name):
    return f"{STATE_DIR}/{strategy_name}.json"


def initial_strategy_state(strategy_name):
    return {
        "strategy": strategy_name,
        "created_at": now_utc().isoformat(),
        "cash": START_CAPITAL,
        "positions": {},
        "highest_portfolio_value": START_CAPITAL,
        "weekly_meta": {
            "iso_week": "",
            "buys_this_week": 0
        },
        "cooldowns": {},
        "last_execution_date": None
    }


def load_strategy_state(strategy_name):
    return load_json(strategy_state_file(strategy_name), initial_strategy_state(strategy_name))


def save_strategy_state(strategy_name, state):
    save_json(state, strategy_state_file(strategy_name))


def reset_week_if_needed(state):
    current_week = now_oslo().strftime("%G-W%V")

    if state["weekly_meta"].get("iso_week") != current_week:
        state["weekly_meta"] = {
            "iso_week": current_week,
            "buys_this_week": 0
        }

    return state


def trading_days_between(date_str):
    if not date_str:
        return 999

    try:
        start = pd.Timestamp(date_str)
        end = pd.Timestamp(today_str())
        return len(pd.bdate_range(start, end)) - 1
    except Exception:
        return 999


def load_latest_signal():
    global_state = load_json(f"{STATE_DIR}/_global.json", {})
    signal_path = global_state.get("last_signal_file")

    if signal_path and os.path.exists(signal_path):
        with open(signal_path, "r") as f:
            return json.load(f), signal_path

    files = sorted([f for f in os.listdir(SIGNALS_DIR) if f.endswith("_signal.json")])

    if not files:
        raise FileNotFoundError("Fant ingen signalfil. Kjør signal først.")

    signal_path = f"{SIGNALS_DIR}/{files[-1]}"

    with open(signal_path, "r") as f:
        return json.load(f), signal_path


# =========================
# TRADING HELPERS
# =========================

def load_trades():
    return read_csv_or_empty(
        TRADES_FILE,
        [
            "date", "timestamp_utc", "strategy", "action", "ticker",
            "shares", "price", "value", "cost", "reason"
        ]
    )


def load_performance():
    return read_csv_or_empty(
        PERFORMANCE_FILE,
        [
            "date", "strategy", "portfolio_value", "cash",
            "positions_value", "return_pct", "num_positions",
            "spy_return_pct", "qqq_return_pct"
        ]
    )


def estimate_trade_cost(value, side):
    cost = abs(value) * 0.001

    if side == "SELL":
        cost += abs(value) * 0.00003

    return cost


def get_price_cached(ticker, fallback_price, price_cache):
    if ticker in price_cache:
        return price_cache[ticker]

    price = get_latest_price(ticker, fallback_price)
    price_cache[ticker] = price

    return price


def current_portfolio_value(state, candidates_by_ticker, price_cache):
    positions_value = 0.0

    for ticker, pos in state["positions"].items():
        fallback = pos.get("last_price", pos.get("avg_price"))
        candidate_price = candidates_by_ticker.get(ticker, {}).get("price", fallback)
        price = get_price_cached(ticker, candidate_price, price_cache)

        if price is None:
            price = fallback

        shares = float(pos["shares"])
        value = shares * price

        pos["last_price"] = price
        pos["market_value"] = value
        pos["highest_price"] = max(float(pos.get("highest_price", price)), price)

        positions_value += value

    total = float(state["cash"]) + positions_value

    return total, positions_value, state


def should_sell_position(ticker, pos, candidate, config):
    if candidate is None:
        return True, "Falt ut av lagret kandidatunivers"

    price = candidate.get("live_price", candidate.get("price", pos.get("last_price", pos.get("avg_price"))))
    avg_price = float(pos.get("avg_price", price))
    highest_price = float(pos.get("highest_price", avg_price))

    if price is None or avg_price <= 0:
        return False, ""

    ret_from_entry = (price / avg_price) - 1
    ret_from_high = (price / highest_price) - 1 if highest_price > 0 else 0

    if ret_from_entry <= config["stop_loss"]:
        return True, f"Stop-loss {safe_round(ret_from_entry * 100, 1)}%"

    if ret_from_high <= config["trailing_stop"]:
        return True, f"Trailing stop {safe_round(ret_from_high * 100, 1)}% fra topp"

    if not candidate.get("above_sma200", False):
        return True, "Under 200-dagers snitt"

    rank = int(candidate.get("rank", 999))
    score = float(candidate.get("strategy_score", 0))

    if rank > config["sell_rank_threshold"] and score < 45:
        return True, "Kraftig svekket ranking"

    return False, ""


def execute_sell(state, trades_df, strategy_name, ticker, reason, candidates_by_ticker, price_cache):
    pos = state["positions"].get(ticker)

    if not pos:
        return state, trades_df, None

    fallback = pos.get("last_price", pos.get("avg_price"))
    signal_price = candidates_by_ticker.get(ticker, {}).get("price", fallback)
    price = get_price_cached(ticker, signal_price, price_cache)

    if price is None or price <= 0:
        return state, trades_df, None

    shares = float(pos["shares"])
    value = shares * price
    cost = estimate_trade_cost(value, "SELL")
    net_value = value - cost

    state["cash"] += net_value

    trade = {
        "date": today_str(),
        "timestamp_utc": now_utc().isoformat(),
        "strategy": strategy_name,
        "action": "SELL",
        "ticker": ticker,
        "shares": round(shares, 6),
        "price": round(price, 2),
        "value": round(value, 2),
        "cost": round(cost, 2),
        "reason": reason
    }

    trades_df = pd.concat([trades_df, pd.DataFrame([trade])], ignore_index=True)

    del state["positions"][ticker]

    state["cooldowns"][ticker] = today_str()

    return state, trades_df, trade


def execute_buy(state, trades_df, strategy_name, ticker, target_value, reason, candidate, price_cache):
    signal_price = candidate.get("price")
    price = get_price_cached(ticker, signal_price, price_cache)

    if price is None or price <= 0:
        return state, trades_df, None

    if signal_price:
        gap = (price / signal_price) - 1

        if gap > 0.05:
            return state, trades_df, None

        if gap > 0.025:
            target_value *= 0.50

    target_value = min(target_value, state["cash"])

    if target_value < 100:
        return state, trades_df, None

    cost = estimate_trade_cost(target_value, "BUY")
    total_needed = target_value + cost

    if total_needed > state["cash"]:
        target_value = state["cash"] * 0.995
        cost = estimate_trade_cost(target_value, "BUY")
        total_needed = target_value + cost

    shares = target_value / price

    if shares <= 0:
        return state, trades_df, None

    state["cash"] -= total_needed

    state["positions"][ticker] = {
        "shares": shares,
        "avg_price": price,
        "last_price": price,
        "highest_price": price,
        "market_value": shares * price,
        "buy_date": today_str()
    }

    state["weekly_meta"]["buys_this_week"] += 1

    trade = {
        "date": today_str(),
        "timestamp_utc": now_utc().isoformat(),
        "strategy": strategy_name,
        "action": "BUY",
        "ticker": ticker,
        "shares": round(shares, 6),
        "price": round(price, 2),
        "value": round(target_value, 2),
        "cost": round(cost, 2),
        "reason": reason
    }

    trades_df = pd.concat([trades_df, pd.DataFrame([trade])], ignore_index=True)

    return state, trades_df, trade


def build_target_weights(candidates, config, regime_name):
    if not candidates:
        return {}

    df = pd.DataFrame(candidates).copy()

    df = df[
        (df["above_sma200"] == True) &
        (df["strategy_score"] > 0)
    ].copy()

    if df.empty:
        return {}

    df = df.sort_values("strategy_score", ascending=False).head(config["buy_top_n"])

    df = df[df["score_percentile"] >= config["min_score_percentile"]].copy()

    if df.empty:
        return {}

    df["vol60"] = df["vol60"].clip(lower=0.20, upper=1.50)
    df["raw_weight"] = (df["strategy_score"] ** 1.25) / df["vol60"]

    if df["raw_weight"].sum() <= 0:
        return {}

    df["target_weight"] = df["raw_weight"] / df["raw_weight"].sum()
    df["target_weight"] = df["target_weight"].clip(upper=config["max_position_weight"])

    if df["target_weight"].sum() > 0:
        df["target_weight"] = df["target_weight"] / df["target_weight"].sum()

    exposure = config["exposure"].get(regime_name, config["exposure"].get("unknown", 0.50))
    df["target_weight"] *= exposure

    return dict(zip(df["ticker"], df["target_weight"]))


# =========================
# BENCHMARK
# =========================

def load_benchmark_state():
    return load_json(
        BENCHMARK_FILE,
        {
            "created_at": now_utc().isoformat(),
            "SPY": {},
            "QQQ": {}
        }
    )


def save_benchmark_state(state):
    save_json(state, BENCHMARK_FILE)


def update_benchmarks(price_cache):
    state = load_benchmark_state()

    for symbol in ["SPY", "QQQ"]:
        fallback = state.get(symbol, {}).get("last_price")
        price = get_price_cached(symbol, fallback, price_cache)

        if price is None:
            continue

        if not state.get(symbol):
            state[symbol] = {}

        if not state[symbol].get("start_price"):
            state[symbol]["start_price"] = price
            state[symbol]["start_date"] = today_str()

        state[symbol]["last_price"] = price
        state[symbol]["updated_at"] = now_utc().isoformat()

    save_benchmark_state(state)

    return state


def benchmark_return(benchmark_state, symbol):
    b = benchmark_state.get(symbol, {})
    start = b.get("start_price")
    last = b.get("last_price")

    if not start or not last:
        return None

    return (last / start) - 1


# =========================
# EXECUTION ENGINE
# =========================

def run_strategy_execution(strategy_name, signal, trades_df, price_cache):
    config = STRATEGIES[strategy_name]
    state = load_strategy_state(strategy_name)
    state = reset_week_if_needed(state)

    strategy_payload = signal["strategies"][strategy_name]
    candidates = strategy_payload["candidates"]
    candidates_by_ticker = {x["ticker"]: x for x in candidates}

    regime_name = signal["regime"]["regime"]

    total_before, positions_value_before, state = current_portfolio_value(
        state=state,
        candidates_by_ticker=candidates_by_ticker,
        price_cache=price_cache
    )

    sells = []
    buys = []

    # Sells: kun ved tydelig svekkelse
    for ticker, pos in list(state["positions"].items()):
        candidate = candidates_by_ticker.get(ticker)
        sell, reason = should_sell_position(ticker, pos, candidate, config)

        if sell:
            state, trades_df, trade = execute_sell(
                state=state,
                trades_df=trades_df,
                strategy_name=strategy_name,
                ticker=ticker,
                reason=reason,
                candidates_by_ticker=candidates_by_ticker,
                price_cache=price_cache
            )

            if trade:
                sells.append(trade)

    total_now, positions_value_now, state = current_portfolio_value(
        state=state,
        candidates_by_ticker=candidates_by_ticker,
        price_cache=price_cache
    )

    target_weights = build_target_weights(candidates, config, regime_name)
    target_tickers = list(target_weights.keys())

    # Buys: kontrollert, ikke daytrading
    for ticker in target_tickers:
        if ticker in state["positions"]:
            continue

        if len(state["positions"]) >= config["max_positions"]:
            break

        if state["weekly_meta"]["buys_this_week"] >= config["max_new_buys_per_week"]:
            break

        last_sell = state["cooldowns"].get(ticker)

        if trading_days_between(last_sell) < config["buyback_cooldown_days"]:
            continue

        candidate = candidates_by_ticker.get(ticker)

        if not candidate:
            continue

        target_value = total_now * target_weights[ticker]

        state, trades_df, trade = execute_buy(
            state=state,
            trades_df=trades_df,
            strategy_name=strategy_name,
            ticker=ticker,
            target_value=target_value,
            reason=f"Rank {candidate.get('rank')}, score {safe_round(candidate.get('strategy_score'), 1)}",
            candidate=candidate,
            price_cache=price_cache
        )

        if trade:
            buys.append(trade)

    total_after, positions_value_after, state = current_portfolio_value(
        state=state,
        candidates_by_ticker=candidates_by_ticker,
        price_cache=price_cache
    )

    state["highest_portfolio_value"] = max(
        float(state.get("highest_portfolio_value", START_CAPITAL)),
        total_after
    )

    state["last_execution_date"] = today_str()

    save_strategy_state(strategy_name, state)

    return_pct = (total_after / START_CAPITAL) - 1

    result = {
        "strategy": strategy_name,
        "description": config["description"],
        "portfolio_value": total_after,
        "cash": state["cash"],
        "positions_value": positions_value_after,
        "return_pct": return_pct,
        "num_positions": len(state["positions"]),
        "positions": state["positions"],
        "buys": buys,
        "sells": sells,
        "top_candidates": candidates[:TOP_CANDIDATES_TO_REPORT]
    }

    return result, trades_df


def run_execute():
    signal, signal_path = load_latest_signal()

    trades_df = load_trades()
    performance_df = load_performance()
    price_cache = {}

    benchmark_state = update_benchmarks(price_cache)
    spy_ret = benchmark_return(benchmark_state, "SPY")
    qqq_ret = benchmark_return(benchmark_state, "QQQ")

    results = []

    for strategy_name in STRATEGIES.keys():
        result, trades_df = run_strategy_execution(
            strategy_name=strategy_name,
            signal=signal,
            trades_df=trades_df,
            price_cache=price_cache
        )

        results.append(result)

        performance_row = {
            "date": today_str(),
            "strategy": strategy_name,
            "portfolio_value": round(result["portfolio_value"], 2),
            "cash": round(result["cash"], 2),
            "positions_value": round(result["positions_value"], 2),
            "return_pct": round(result["return_pct"] * 100, 2),
            "num_positions": result["num_positions"],
            "spy_return_pct": round(spy_ret * 100, 2) if spy_ret is not None else None,
            "qqq_return_pct": round(qqq_ret * 100, 2) if qqq_ret is not None else None
        }

        performance_df = pd.concat([performance_df, pd.DataFrame([performance_row])], ignore_index=True)

    save_csv(trades_df, TRADES_FILE)
    save_csv(performance_df, PERFORMANCE_FILE)

    message = build_execute_message(
        results=results,
        signal=signal,
        signal_path=signal_path,
        spy_ret=spy_ret,
        qqq_ret=qqq_ret
    )

    send_telegram(message)


# =========================
# REPORTING
# =========================

def format_pct(x):
    if x is None:
        return "N/A"

    return f"{safe_round(x * 100, 2)}%"


def build_execute_message(results, signal, signal_path, spy_ret, qqq_ret):
    message = ""
    message += "🧠 AI STOCK LAB v3 — MULTI-STRATEGY\n"
    message += f"Dato: {today_str()}\n"
    message += f"Marked: {signal['regime']['regime'].upper()} — {signal['regime']['reason']}\n"
    message += f"Signalfil: {os.path.basename(signal_path)}\n\n"

    message += "🏁 LEADERBOARD\n"

    leaderboard = sorted(results, key=lambda x: x["return_pct"], reverse=True)

    for i, result in enumerate(leaderboard, start=1):
        message += (
            f"{i}. {result['strategy']}: "
            f"${safe_round(result['portfolio_value'], 2)} "
            f"({safe_round(result['return_pct'] * 100, 2)}%) | "
            f"{result['num_positions']} posisjoner | "
            f"Cash ${safe_round(result['cash'], 0)}\n"
        )

    message += f"SPY: {format_pct(spy_ret)}\n"
    message += f"QQQ: {format_pct(qqq_ret)}\n\n"

    message += "📌 DAGENS HANDLINGER\n"

    any_action = False

    for result in results:
        buys = result["buys"]
        sells = result["sells"]

        if buys or sells:
            any_action = True
            message += f"\n{result['strategy']}\n"

            if buys:
                message += "KJØP:\n"

                for trade in buys:
                    message += (
                        f"- {trade['ticker']} | "
                        f"${trade['value']} @ ${trade['price']} | "
                        f"{trade['reason']}\n"
                    )

            if sells:
                message += "SELG:\n"

                for trade in sells:
                    message += (
                        f"- {trade['ticker']} | "
                        f"${trade['value']} @ ${trade['price']} | "
                        f"{trade['reason']}\n"
                    )

    if not any_action:
        message += "Ingen kjøp/salg i dag. Strategiene holder posisjonene.\n"

    message += "\n📊 STRATEGI-DETALJER\n"

    for result in leaderboard:
        message += "\n"
        message += f"{result['strategy']}\n"
        message += f"{result['description']}\n"
        message += (
            f"Verdi: ${safe_round(result['portfolio_value'], 2)} | "
            f"Avkastning: {safe_round(result['return_pct'] * 100, 2)}% | "
            f"Cash: ${safe_round(result['cash'], 0)}\n"
        )

        if result["positions"]:
            holdings = []

            for ticker, pos in sorted(result["positions"].items()):
                value = pos.get("market_value", 0)
                avg = pos.get("avg_price", 0)
                last = pos.get("last_price", avg)
                pnl = (last / avg - 1) if avg else 0

                holdings.append(
                    f"{ticker}({safe_round(pnl * 100, 1)}%)"
                )

            message += "Hold: " + ", ".join(holdings[:15]) + "\n"
        else:
            message += "Hold: ingen\n"

        top = result["top_candidates"][:3]

        if top:
            message += "Topp 3 nå: "

            parts = []

            for candidate in top:
                parts.append(
                    f"{candidate['ticker']} "
                    f"score {safe_round(candidate['strategy_score'], 1)}"
                )

            message += ", ".join(parts) + "\n"

    message += "\n⚠️ Dette er paper trading, ikke ekte ordre.\n"
    message += "Målet er å teste hvilke modeller som over tid slår SPY/QQQ."

    return message


# =========================
# MAIN
# =========================

def main():
    ensure_dirs()

    mode = os.environ.get("BOT_MODE", "signal").strip().lower()

    print(f"BOT_MODE={mode}")
    print(f"Oslo time: {now_oslo().isoformat()}")
    print(f"New York time: {now_ny().isoformat()}")

    if mode == "signal":
        run_signal()
    elif mode == "execute":
        run_execute()
    else:
        raise ValueError(f"Ukjent BOT_MODE: {mode}")


if __name__ == "__main__":
    main()
