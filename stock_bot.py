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
# AI PORTFOLIO LAB v2
# Long-term paper portfolio manager
#
# Modes:
#   BOT_MODE=signal   -> creates signal snapshot / candidates
#   BOT_MODE=execute  -> updates paper portfolio from latest signal
#
# Goal:
#   Long-term outperformance vs SPY / QQQ with controlled turnover.
# ============================================================


# =========================
# CONFIG
# =========================

START_CAPITAL = 10_000.0

MAX_POSITIONS = 12
MAX_POSITION_WEIGHT = 0.12
MAX_NEW_BUYS_PER_WEEK = 4
BUYBACK_COOLDOWN_DAYS = 10

TARGET_VOL_NORMAL = 0.16
TARGET_VOL_DEFENSIVE = 0.08

HARD_STOP_LOSS = -0.18
TRAILING_STOP_FROM_HIGH = -0.25

MIN_PRICE = 10
MIN_AVG_DOLLAR_VOLUME = 25_000_000

TOP_CANDIDATES_TO_SAVE = 75
TOP_CANDIDATES_TO_REPORT = 12

MONDAY_REBALANCE_ONLY = True

DATA_DIR = "data"
STATE_DIR = f"{DATA_DIR}/state"
SIGNALS_DIR = f"{DATA_DIR}/signals"
REPORTS_DIR = f"{DATA_DIR}/reports"

STATE_FILE = f"{STATE_DIR}/portfolio_state.json"
TRADES_FILE = f"{DATA_DIR}/trades.csv"
PERFORMANCE_FILE = f"{DATA_DIR}/performance.csv"

OSLO = ZoneInfo("Europe/Oslo")
NY = ZoneInfo("America/New_York")


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
# FILE HELPERS
# =========================

def ensure_dirs():
    for path in [DATA_DIR, STATE_DIR, SIGNALS_DIR, REPORTS_DIR]:
        os.makedirs(path, exist_ok=True)


def now_utc():
    return datetime.now(timezone.utc)


def now_oslo():
    return now_utc().astimezone(OSLO)


def now_ny():
    return now_utc().astimezone(NY)


def today_str():
    return now_oslo().strftime("%Y-%m-%d")


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


def load_state():
    ensure_dirs()

    if not os.path.exists(STATE_FILE):
        return {
            "created_at": now_utc().isoformat(),
            "cash": START_CAPITAL,
            "positions": {},
            "highest_portfolio_value": START_CAPITAL,
            "weekly_meta": {
                "iso_week": "",
                "buys_this_week": 0
            },
            "cooldowns": {},
            "last_signal_file": None,
            "last_execution_date": None
        }

    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(state):
    ensure_dirs()

    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


# =========================
# UNIVERSE
# =========================

EXTRA_WATCHLIST = [
    "PLTR", "ARM", "SMCI", "TSM", "ASML", "NVO", "SHOP", "SE",
    "COIN", "RBLX", "U", "SNOW", "MDB", "DDOG", "NET", "CRWD",
    "CELH", "ELF", "TOST", "APP", "HOOD", "SOFI", "MSTR",
    "UBER", "ABNB", "PANW", "ZS", "OKTA", "BILL", "ROKU"
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
    """
    Brukes i execute-modus.
    Prøver intraday 1m først. Hvis det feiler, bruker den fallback fra signal.
    """
    try:
        data = yf.download(
            tickers=ticker,
            period="5d",
            interval="1m",
            auto_adjust=False,
            progress=False,
            prepost=False
        )

        if data is not None and not data.empty and "Close" in data.columns:
            price = float(data["Close"].dropna().iloc[-1])
            if price > 0:
                return price

    except Exception as e:
        print(f"Intraday pris feilet for {ticker}: {e}")

    return fallback_price


# =========================
# INDICATORS
# =========================

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


# =========================
# SIGNAL ENGINE
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

        # 12-1 momentum: bruker pris for 1 måned siden mot 12 måneder siden
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
            "avg_dollar_volume": avg_dollar_volume,
            "sma50": sma50,
            "sma100": sma100,
            "sma200": sma200
        }

    except Exception as e:
        print(f"Analysefeil {ticker}: {e}")
        return None


def percentile_rank(series, higher_is_better=True):
    s = pd.Series(series).astype(float)

    if not higher_is_better:
        s = -s

    return s.rank(pct=True)


def build_score_table(analyzed_stocks):
    df = pd.DataFrame(analyzed_stocks)

    if df.empty:
        return df

    df = df.set_index("ticker")

    # Robust clipping for å hindre at én ekstrem aksje dominerer alt
    factor_cols = [
        "mom_12_1", "mom_6_1", "mom_3_1", "relative_strength_3m",
        "ret_6m", "ret_3m", "vol60", "drawdown_6m", "trend_score"
    ]

    for col in factor_cols:
        if col in df.columns:
            lower = df[col].quantile(0.02)
            upper = df[col].quantile(0.98)
            df[col] = df[col].clip(lower, upper)

    # Momentum sleeve
    df["rank_mom_12_1"] = percentile_rank(df["mom_12_1"], True)
    df["rank_mom_6_1"] = percentile_rank(df["mom_6_1"], True)
    df["rank_mom_3_1"] = percentile_rank(df["mom_3_1"], True)
    df["rank_rs"] = percentile_rank(df["relative_strength_3m"].fillna(0), True)

    df["momentum_rank"] = (
        0.45 * df["rank_mom_12_1"] +
        0.25 * df["rank_mom_6_1"] +
        0.15 * df["rank_mom_3_1"] +
        0.15 * df["rank_rs"]
    )

    # Trend sleeve
    df["trend_rank"] = percentile_rank(df["trend_score"], True)

    # Low risk / stability sleeve
    df["vol_rank"] = percentile_rank(df["vol60"], False)
    df["drawdown_rank"] = percentile_rank(df["drawdown_6m"], True)

    df["stability_rank"] = (
        0.65 * df["vol_rank"] +
        0.35 * df["drawdown_rank"]
    )

    # Quality proxy siden vi foreløpig ikke har ekte fundamentals
    # Denne favoriserer aksjer med sterk trend, lavere volatilitet, og sunn RSI.
    df["rsi_quality"] = np.where(
        df["healthy_rsi"],
        1.0,
        np.where(df["overbought"], 0.35, np.where(df["very_weak_rsi"], 0.20, 0.60))
    )

    df["quality_proxy_rank"] = (
        0.40 * df["trend_rank"] +
        0.35 * df["stability_rank"] +
        0.25 * df["rsi_quality"]
    )

    # Total ensemble score
    df["score"] = (
        0.45 * df["momentum_rank"] +
        0.25 * df["quality_proxy_rank"] +
        0.20 * df["trend_rank"] +
        0.10 * df["stability_rank"]
    )

    # Straffer direkte
    df.loc[~df["above_sma200"], "score"] *= 0.55
    df.loc[df["overbought"], "score"] *= 0.90
    df.loc[df["vol60"] > 1.20, "score"] *= 0.80

    df = df.sort_values("score", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    df["score_percentile"] = df["score"].rank(pct=True)

    return df


# =========================
# REGIME DETECTION
# =========================

def detect_market_regime(spy, qqq):
    """
    Enkel regimesjekk:
    - bullish: SPY og QQQ over 200 SMA og positiv 3M
    - neutral: blandet
    - defensive: svakt marked
    """
    if spy is None or qqq is None:
        return {
            "regime": "unknown",
            "max_gross_exposure": 0.70,
            "target_vol": TARGET_VOL_DEFENSIVE
        }

    spy_good = spy["above_sma200"] and spy["ret_3m"] > 0
    qqq_good = qqq["above_sma200"] and qqq["ret_3m"] > 0

    if spy_good and qqq_good:
        return {
            "regime": "bullish",
            "max_gross_exposure": 1.00,
            "target_vol": TARGET_VOL_NORMAL
        }

    if spy["above_sma200"] or qqq["above_sma200"]:
        return {
            "regime": "neutral",
            "max_gross_exposure": 0.70,
            "target_vol": 0.12
        }

    return {
        "regime": "defensive",
        "max_gross_exposure": 0.40,
        "target_vol": TARGET_VOL_DEFENSIVE
    }


# =========================
# SIGNAL MODE
# =========================

def save_signal_snapshot(score_df, regime, spy, qqq):
    ensure_dirs()

    date = today_str()
    path = f"{SIGNALS_DIR}/{date}_signal.json"

    candidates = score_df.head(TOP_CANDIDATES_TO_SAVE).reset_index().to_dict(orient="records")

    payload = {
        "created_at_utc": now_utc().isoformat(),
        "created_at_oslo": now_oslo().isoformat(),
        "created_at_ny": now_ny().isoformat(),
        "date": date,
        "regime": regime,
        "spy": spy,
        "qqq": qqq,
        "candidates": candidates
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    state = load_state()
    state["last_signal_file"] = path
    save_state(state)

    return path


def build_signal_message(score_df, regime, analyzed_count, universe_count, spy, qqq):
    message = ""
    message += "📈 AI PORTFOLIO LAB v2 — SIGNAL\n"
    message += f"Dato: {today_str()}\n"
    message += f"Marked: {regime['regime'].upper()}\n"
    message += f"Maks eksponering: {int(regime['max_gross_exposure'] * 100)}%\n"
    message += f"Univers: {universe_count} tickere | Analysert: {analyzed_count}\n\n"

    if spy:
        message += (
            f"SPY: 3M {safe_round(spy['ret_3m'] * 100, 1)}% | "
            f"6M {safe_round(spy['ret_6m'] * 100, 1)}% | "
            f"Over 200SMA: {'ja' if spy['above_sma200'] else 'nei'}\n"
        )

    if qqq:
        message += (
            f"QQQ: 3M {safe_round(qqq['ret_3m'] * 100, 1)}% | "
            f"6M {safe_round(qqq['ret_6m'] * 100, 1)}% | "
            f"Over 200SMA: {'ja' if qqq['above_sma200'] else 'nei'}\n"
        )

    message += "\n"
    message += "🏆 TOPP KANDIDATER\n"

    top = score_df.head(TOP_CANDIDATES_TO_REPORT).reset_index()

    for i, row in enumerate(top.itertuples(), start=1):
        message += (
            f"{i}. {row.ticker} | "
            f"Score {safe_round(row.score, 3)} | "
            f"3M {safe_round(row.ret_3m * 100, 1)}% | "
            f"6M {safe_round(row.ret_6m * 100, 1)}% | "
            f"RS {safe_round(row.relative_strength_3m * 100 if row.relative_strength_3m is not None else None, 1)}% | "
            f"Vol {safe_round(row.vol60 * 100, 0)}% | "
            f"RSI {safe_round(row.rsi, 0)}\n"
        )

    message += "\n"
    message += "Neste steg: execute-kjøringen bruker dette signalet til paper-portefølje.\n"
    message += "Dette er ikke ekte kjøpsordre."

    return message


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
        qqq=qqq
    )

    print(f"Signal lagret: {signal_file}")

    message = build_signal_message(
        score_df=score_df,
        regime=regime,
        analyzed_count=len(analyzed),
        universe_count=len(universe),
        spy=spy,
        qqq=qqq
    )

    send_telegram(message)


# =========================
# EXECUTION / PORTFOLIO
# =========================

def load_latest_signal():
    state = load_state()
    signal_path = state.get("last_signal_file")

    if not signal_path or not os.path.exists(signal_path):
        files = sorted([f for f in os.listdir(SIGNALS_DIR) if f.endswith("_signal.json")])

        if not files:
            raise FileNotFoundError("Fant ingen signalfil. Kjør signal først.")

        signal_path = f"{SIGNALS_DIR}/{files[-1]}"

    with open(signal_path, "r") as f:
        return json.load(f), signal_path


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


def load_trades():
    return read_csv_or_empty(
        TRADES_FILE,
        ["date", "timestamp_utc", "action", "ticker", "shares", "price", "value", "cost", "reason"]
    )


def load_performance():
    return read_csv_or_empty(
        PERFORMANCE_FILE,
        ["date", "portfolio_value", "cash", "positions_value", "return_pct", "num_positions", "spy_return_pct", "qqq_return_pct"]
    )


def estimate_trade_cost(value, side):
    # Enkel realistisk kostnadsmodell for paper.
    # 0.10% total friksjon på kjøp/salg.
    base_cost = abs(value) * 0.001

    # Litt ekstra på salg for fees.
    if side == "SELL":
        base_cost += abs(value) * 0.00003

    return base_cost


def current_portfolio_value(state, candidates_by_ticker):
    positions_value = 0.0
    updated_positions = {}

    for ticker, pos in state["positions"].items():
        fallback = pos.get("last_price", pos.get("avg_price"))
        price = candidates_by_ticker.get(ticker, {}).get("price", fallback)

        if price is None:
            price = fallback

        shares = float(pos["shares"])
        value = shares * price

        pos["last_price"] = price
        pos["market_value"] = value

        updated_positions[ticker] = pos
        positions_value += value

    state["positions"] = updated_positions

    total = float(state["cash"]) + positions_value

    return total, positions_value, state


def build_target_weights(candidates, regime, state):
    df = pd.DataFrame(candidates)

    if df.empty:
        return {}

    df = df.sort_values("score", ascending=False).head(50)

    # Bare aksjer med god trend og positiv score
    df = df[
        (df["above_sma200"] == True) &
        (df["score"] > 0)
    ].copy()

    if df.empty:
        return {}

    # Primære kandidater
    df = df.head(MAX_POSITIONS).copy()

    # Score-vol sizing
    df["vol60"] = df["vol60"].clip(lower=0.20, upper=1.50)
    df["raw_weight"] = (df["score"] ** 1.25) / df["vol60"]

    if df["raw_weight"].sum() <= 0:
        return {}

    df["target_weight"] = df["raw_weight"] / df["raw_weight"].sum()

    # Maks posisjonsvekt
    df["target_weight"] = df["target_weight"].clip(upper=MAX_POSITION_WEIGHT)

    # Normaliser etter cap
    if df["target_weight"].sum() > 0:
        df["target_weight"] = df["target_weight"] / df["target_weight"].sum()

    # Markedsregime bestemmer total eksponering
    max_exposure = regime.get("max_gross_exposure", 0.70)
    df["target_weight"] *= max_exposure

    return dict(zip(df["ticker"], df["target_weight"]))


def should_hard_sell(ticker, pos, candidate):
    if candidate is None:
        return True, "Ikke lenger i kandidatlisten"

    price = candidate.get("price", pos.get("last_price", pos.get("avg_price")))
    avg_price = pos.get("avg_price", price)
    highest_price = pos.get("highest_price", avg_price)

    if price and highest_price:
        pos["highest_price"] = max(highest_price, price)

    ret_from_entry = (price / avg_price) - 1 if avg_price else 0
    ret_from_high = (price / pos.get("highest_price", highest_price)) - 1 if pos.get("highest_price") else 0

    if ret_from_entry <= HARD_STOP_LOSS:
        return True, f"Hard stop-loss {safe_round(ret_from_entry * 100, 1)}%"

    if ret_from_high <= TRAILING_STOP_FROM_HIGH:
        return True, f"Trailing stop {safe_round(ret_from_high * 100, 1)}% fra topp"

    if not candidate.get("above_sma200", False):
        return True, "Under 200-dagers snitt"

    if candidate.get("rank", 999) > 75 and candidate.get("score", 0) < 0.50:
        return True, "Falt langt ned i ranking"

    return False, ""


def execute_sell(state, trades_df, ticker, reason, candidates_by_ticker):
    pos = state["positions"].get(ticker)

    if not pos:
        return state, trades_df, None

    fallback_price = pos.get("last_price", pos.get("avg_price"))
    signal_price = candidates_by_ticker.get(ticker, {}).get("price", fallback_price)
    price = get_latest_price(ticker, signal_price)

    shares = float(pos["shares"])
    value = shares * price
    cost = estimate_trade_cost(value, "SELL")
    net_value = value - cost

    state["cash"] += net_value

    trade = {
        "date": today_str(),
        "timestamp_utc": now_utc().isoformat(),
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


def execute_buy(state, trades_df, ticker, target_value, reason, candidate):
    signal_price = candidate.get("price")
    price = get_latest_price(ticker, signal_price)

    if price is None or price <= 0:
        return state, trades_df, None

    # Gap guard: ikke jag aksjer som har åpnet altfor høyt over signalpris
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


def update_benchmark_state(state, signal):
    candidates = signal["candidates"]
    by_ticker = {x["ticker"]: x for x in candidates}

    spy_price = None
    qqq_price = None

    # Hent fra signal hvis mulig, ellers yfinance
    for symbol in ["SPY", "QQQ"]:
        if symbol not in state:
            state[symbol] = {}

    for symbol in ["SPY", "QQQ"]:
        try:
            price = get_latest_price(symbol, None)

            if price is None:
                price = by_ticker.get(symbol, {}).get("price")

            if not state[symbol].get("start_price") and price:
                state[symbol]["start_price"] = price

            if price:
                state[symbol]["last_price"] = price

        except Exception:
            pass

    return state


def benchmark_return(state, symbol):
    b = state.get(symbol, {})
    start = b.get("start_price")
    last = b.get("last_price")

    if not start or not last:
        return None

    return (last / start) - 1


def run_execute():
    signal, signal_path = load_latest_signal()
    state = load_state()
    state = reset_week_if_needed(state)

    candidates = signal["candidates"]
    candidates_by_ticker = {x["ticker"]: x for x in candidates}

    trades_df = load_trades()
    performance_df = load_performance()

    regime = signal["regime"]
    is_monday = now_oslo().weekday() == 0

    total_before, positions_value_before, state = current_portfolio_value(state, candidates_by_ticker)

    sells = []
    buys = []
    holds = []

    # 1. Hard sells kan skje alle dager
    for ticker, pos in list(state["positions"].items()):
        candidate = candidates_by_ticker.get(ticker)
        sell, reason = should_hard_sell(ticker, pos, candidate)

        if sell:
            state, trades_df, trade = execute_sell(
                state=state,
                trades_df=trades_df,
                ticker=ticker,
                reason=reason,
                candidates_by_ticker=candidates_by_ticker
            )

            if trade:
                sells.append(trade)

    # Oppdater verdi etter salg
    total_now, positions_value_now, state = current_portfolio_value(state, candidates_by_ticker)

    # 2. Bygg target weights
    target_weights = build_target_weights(candidates, regime, state)

    # 3. Kjøp / rebalanser
    current_positions = set(state["positions"].keys())
    target_tickers = list(target_weights.keys())

    # Mandag: mer aktiv rebalansering
    # Andre dager: kun kjøp nye superkandidater hvis vi har plass og ukentlig buy-budget
    for ticker in target_tickers:
        if ticker in state["positions"]:
            holds.append(ticker)
            continue

        if len(state["positions"]) >= MAX_POSITIONS:
            break

        if state["weekly_meta"]["buys_this_week"] >= MAX_NEW_BUYS_PER_WEEK:
            break

        candidate = candidates_by_ticker.get(ticker)

        if not candidate:
            continue

        # Anti flip-flop: ikke kjøp tilbake rett etter salg
        last_sell = state["cooldowns"].get(ticker)

        if trading_days_between(last_sell) < BUYBACK_COOLDOWN_DAYS:
            continue

        # Kjøpsstrenghet:
        # Mandag: topp 12 ok.
        # Tirs-fre: bare topp 5 / svært høy score.
        rank = candidate.get("rank", 999)
        score_percentile = candidate.get("score_percentile", 0)

        if is_monday:
            if rank > 12:
                continue
        else:
            if rank > 5 and score_percentile < 0.97:
                continue

        target_weight = target_weights[ticker]
        target_value = total_now * target_weight

        state, trades_df, trade = execute_buy(
            state=state,
            trades_df=trades_df,
            ticker=ticker,
            target_value=target_value,
            reason=f"Rank {rank}, score {safe_round(candidate.get('score'), 3)}",
            candidate=candidate
        )

        if trade:
            buys.append(trade)

    # 4. Oppdater priser/verdi
    total_after, positions_value_after, state = current_portfolio_value(state, candidates_by_ticker)

    state["highest_portfolio_value"] = max(
        float(state.get("highest_portfolio_value", START_CAPITAL)),
        total_after
    )

    state["last_execution_date"] = today_str()
    state = update_benchmark_state(state, signal)

    spy_ret = benchmark_return(state, "SPY")
    qqq_ret = benchmark_return(state, "QQQ")

    return_pct = (total_after / START_CAPITAL) - 1

    performance_row = {
        "date": today_str(),
        "portfolio_value": round(total_after, 2),
        "cash": round(state["cash"], 2),
        "positions_value": round(positions_value_after, 2),
        "return_pct": round(return_pct * 100, 2),
        "num_positions": len(state["positions"]),
        "spy_return_pct": round(spy_ret * 100, 2) if spy_ret is not None else None,
        "qqq_return_pct": round(qqq_ret * 100, 2) if qqq_ret is not None else None
    }

    performance_df = pd.concat([performance_df, pd.DataFrame([performance_row])], ignore_index=True)

    save_csv(trades_df, TRADES_FILE)
    save_csv(performance_df, PERFORMANCE_FILE)
    save_state(state)

    message = build_execute_message(
        state=state,
        buys=buys,
        sells=sells,
        holds=holds,
        total_after=total_after,
        positions_value=positions_value_after,
        return_pct=return_pct,
        spy_ret=spy_ret,
        qqq_ret=qqq_ret,
        signal=signal,
        signal_path=signal_path
    )

    send_telegram(message)


def build_execute_message(state, buys, sells, holds, total_after, positions_value, return_pct, spy_ret, qqq_ret, signal, signal_path):
    message = ""
    message += "🧠 AI PORTFOLIO LAB v2 — EXECUTE\n"
    message += f"Dato: {today_str()}\n"
    message += f"Marked: {signal['regime']['regime'].upper()}\n"
    message += f"Signalfil: {os.path.basename(signal_path)}\n\n"

    message += "📊 PORTFØLJE\n"
    message += f"Verdi: ${safe_round(total_after, 2)} ({safe_round(return_pct * 100, 2)}%)\n"
    message += f"Cash: ${safe_round(state['cash'], 2)}\n"
    message += f"Investert: ${safe_round(positions_value, 2)}\n"
    message += f"Posisjoner: {len(state['positions'])}/{MAX_POSITIONS}\n"

    if spy_ret is not None:
        message += f"SPY siden start: {safe_round(spy_ret * 100, 2)}%\n"

    if qqq_ret is not None:
        message += f"QQQ siden start: {safe_round(qqq_ret * 100, 2)}%\n"

    message += "\n"

    message += "🟢 KJØP\n"
    if buys:
        for t in buys:
            message += f"- {t['ticker']} | ${t['value']} @ ${t['price']} | {t['reason']}\n"
    else:
        message += "- Ingen nye kjøp\n"

    message += "\n"

    message += "🔴 SELG\n"
    if sells:
        for t in sells:
            message += f"- {t['ticker']} | ${t['value']} @ ${t['price']} | {t['reason']}\n"
    else:
        message += "- Ingen salg\n"

    message += "\n"

    message += "🟡 HOLD\n"
    if state["positions"]:
        for ticker, pos in sorted(state["positions"].items()):
            value = pos.get("market_value", 0)
            avg = pos.get("avg_price", 0)
            last = pos.get("last_price", avg)
            pnl = (last / avg - 1) if avg else 0

            message += (
                f"- {ticker}: ${safe_round(value, 2)} | "
                f"P/L {safe_round(pnl * 100, 1)}%\n"
            )
    else:
        message += "- Ingen posisjoner\n"

    message += "\n"
    message += "⚠️ Paper trading, ikke ekte ordre.\n"
    message += "Målet er å teste en langsiktig porteføljemodell mot SPY/QQQ."

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
