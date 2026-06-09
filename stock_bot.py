import os
import math
import warnings
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf


warnings.simplefilter(action="ignore", category=FutureWarning)


# =========================
# KONFIGURASJON
# =========================

START_CAPITAL = 10_000
MAX_POSITIONS = 10
BUY_TOP_N = 10
HOLD_TOP_N = 20
MIN_CASH_TO_BUY = 100

DATA_DIR = "data"

PORTFOLIO_FILE = f"{DATA_DIR}/paper_portfolios.csv"
TRADES_FILE = f"{DATA_DIR}/trades.csv"
PERFORMANCE_FILE = f"{DATA_DIR}/performance.csv"
CASH_FILE = f"{DATA_DIR}/cash.csv"
BENCHMARK_FILE = f"{DATA_DIR}/benchmark_state.csv"


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
        raise ValueError("Mangler TELEGRAM_BOT_TOKEN eller TELEGRAM_CHAT_ID i GitHub Secrets.")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    for chunk in split_message(message):
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


# =========================
# FILER / DATA
# =========================

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def read_csv_or_empty(path, columns):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame(columns=columns)

    return pd.DataFrame(columns=columns)


def save_csv(df, path):
    ensure_data_dir()
    df.to_csv(path, index=False)


# =========================
# UNIVERS
# =========================

EXTRA_WATCHLIST = [
    "PLTR", "ARM", "SMCI", "TSM", "ASML", "NVO", "SHOP", "SE",
    "COIN", "RBLX", "U", "SNOW", "MDB", "DDOG", "NET", "CRWD",
    "CELH", "ELF", "TOST", "APP", "HOOD", "SOFI"
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
    if previous is None or previous == 0:
        return None

    try:
        if math.isnan(previous):
            return None
    except Exception:
        pass

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

        if ret_1m is None or ret_3m is None or ret_6m is None:
            return None

        high_52w = float(close.max())
        low_52w = float(close.min())

        distance_from_high = pct_change(price, high_52w)
        distance_from_low = pct_change(price, low_52w)

        avg_volume_20 = float(volume.tail(20).mean())
        latest_volume = float(volume.iloc[-1])
        volume_ratio = latest_volume / avg_volume_20 if avg_volume_20 > 0 else 1

        avg_dollar_volume = avg_volume_20 * price

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

    score -= stock["volatility_60d"] * 0.65

    if stock["drawdown_3m"] is not None:
        score += stock["drawdown_3m"] * 0.35

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

        for rank, item in enumerate(scored, start=1):
            item["rank"] = rank

        rankings[strategy_name] = scored

    return rankings


# =========================
# PAPER PORTFOLIO
# =========================

def load_cash():
    columns = ["strategy", "cash"]
    cash_df = read_csv_or_empty(CASH_FILE, columns)

    existing = set(cash_df["strategy"].tolist()) if not cash_df.empty else set()

    rows = []

    for strategy in STRATEGIES.keys():
        if strategy not in existing:
            rows.append({
                "strategy": strategy,
                "cash": START_CAPITAL
            })

    if rows:
        cash_df = pd.concat([cash_df, pd.DataFrame(rows)], ignore_index=True)

    return cash_df


def get_cash(cash_df, strategy):
    row = cash_df[cash_df["strategy"] == strategy]

    if row.empty:
        return START_CAPITAL

    return float(row.iloc[0]["cash"])


def set_cash(cash_df, strategy, cash):
    if strategy in cash_df["strategy"].values:
        cash_df.loc[cash_df["strategy"] == strategy, "cash"] = cash
    else:
        cash_df = pd.concat([
            cash_df,
            pd.DataFrame([{"strategy": strategy, "cash": cash}])
        ], ignore_index=True)

    return cash_df


def load_portfolio():
    columns = [
        "strategy", "ticker", "shares", "buy_price", "buy_date",
        "current_price", "market_value"
    ]

    return read_csv_or_empty(PORTFOLIO_FILE, columns)


def load_trades():
    columns = [
        "date", "strategy", "action", "ticker", "price",
        "shares", "value", "reason"
    ]

    return read_csv_or_empty(TRADES_FILE, columns)


def load_performance():
    columns = [
        "date", "strategy", "portfolio_value", "cash",
        "positions_value", "return_pct", "num_positions"
    ]

    return read_csv_or_empty(PERFORMANCE_FILE, columns)


def update_current_prices(portfolio_df, price_map):
    if portfolio_df.empty:
        return portfolio_df

    for idx, row in portfolio_df.iterrows():
        ticker = row["ticker"]

        if ticker in price_map:
            price = price_map[ticker]
            shares = float(row["shares"])
            portfolio_df.at[idx, "current_price"] = price
            portfolio_df.at[idx, "market_value"] = shares * price

    return portfolio_df


def run_paper_portfolios(rankings, price_map):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    portfolio_df = load_portfolio()
    trades_df = load_trades()
    performance_df = load_performance()
    cash_df = load_cash()

    portfolio_df = update_current_prices(portfolio_df, price_map)

    report = {}

    new_trades = []

    for strategy, ranked_stocks in rankings.items():
        top_buy = ranked_stocks[:BUY_TOP_N]
        top_hold = ranked_stocks[:HOLD_TOP_N]

        buy_tickers = [x["ticker"] for x in top_buy if x["strategy_score"] > 0 and x["above_sma200"]]
        hold_tickers = [x["ticker"] for x in top_hold if x["strategy_score"] > 0 and x["above_sma200"]]

        strategy_portfolio = portfolio_df[portfolio_df["strategy"] == strategy].copy()
        cash = get_cash(cash_df, strategy)

        buys = []
        sells = []
        holds = []

        # SELG
        for _, pos in strategy_portfolio.iterrows():
            ticker = pos["ticker"]
            should_sell = ticker not in hold_tickers

            if should_sell:
                price = price_map.get(ticker)

                if price is None:
                    continue

                shares = float(pos["shares"])
                value = shares * price
                cash += value

                sells.append(ticker)

                new_trades.append({
                    "date": today,
                    "strategy": strategy,
                    "action": "SELL",
                    "ticker": ticker,
                    "price": round(price, 2),
                    "shares": round(shares, 6),
                    "value": round(value, 2),
                    "reason": f"Ikke lenger topp {HOLD_TOP_N} / svakere trend"
                })

                portfolio_df = portfolio_df[
                    ~((portfolio_df["strategy"] == strategy) & (portfolio_df["ticker"] == ticker))
                ]

        # OPPDATER ETTER SALG
        strategy_portfolio = portfolio_df[portfolio_df["strategy"] == strategy].copy()
        owned_tickers = set(strategy_portfolio["ticker"].tolist())

        for ticker in owned_tickers:
            if ticker in hold_tickers:
                holds.append(ticker)

        # KJØP
        positions_count = len(owned_tickers)

        positions_value = float(strategy_portfolio["market_value"].sum()) if not strategy_portfolio.empty else 0
        total_equity = cash + positions_value

        target_position_value = total_equity / MAX_POSITIONS

        for ticker in buy_tickers:
            if positions_count >= MAX_POSITIONS:
                break

            if ticker in owned_tickers:
                continue

            if cash < MIN_CASH_TO_BUY:
                break

            price = price_map.get(ticker)

            if price is None or price <= 0:
                continue

            buy_value = min(target_position_value, cash)
            shares = buy_value / price

            if buy_value < MIN_CASH_TO_BUY:
                continue

            cash -= buy_value
            positions_count += 1
            owned_tickers.add(ticker)
            buys.append(ticker)

            new_row = {
                "strategy": strategy,
                "ticker": ticker,
                "shares": shares,
                "buy_price": price,
                "buy_date": today,
                "current_price": price,
                "market_value": buy_value
            }

            portfolio_df = pd.concat([portfolio_df, pd.DataFrame([new_row])], ignore_index=True)

            new_trades.append({
                "date": today,
                "strategy": strategy,
                "action": "BUY",
                "ticker": ticker,
                "price": round(price, 2),
                "shares": round(shares, 6),
                "value": round(buy_value, 2),
                "reason": f"Topp {BUY_TOP_N} i {strategy}"
            })

        # OPPDATER CASH
        cash_df = set_cash(cash_df, strategy, cash)

        # PERFORMANCE
        strategy_portfolio = portfolio_df[portfolio_df["strategy"] == strategy].copy()
        positions_value = float(strategy_portfolio["market_value"].sum()) if not strategy_portfolio.empty else 0
        portfolio_value = cash + positions_value
        return_pct = ((portfolio_value / START_CAPITAL) - 1) * 100

        performance_row = {
            "date": today,
            "strategy": strategy,
            "portfolio_value": round(portfolio_value, 2),
            "cash": round(cash, 2),
            "positions_value": round(positions_value, 2),
            "return_pct": round(return_pct, 2),
            "num_positions": len(strategy_portfolio)
        }

        performance_df = pd.concat([performance_df, pd.DataFrame([performance_row])], ignore_index=True)

        report[strategy] = {
            "portfolio_value": portfolio_value,
            "cash": cash,
            "positions_value": positions_value,
            "return_pct": return_pct,
            "num_positions": len(strategy_portfolio),
            "buys": buys,
            "sells": sells,
            "holds": holds,
            "top_candidates": [x["ticker"] for x in top_buy[:5]]
        }

    if new_trades:
        trades_df = pd.concat([trades_df, pd.DataFrame(new_trades)], ignore_index=True)

    portfolio_df = update_current_prices(portfolio_df, price_map)

    save_csv(portfolio_df, PORTFOLIO_FILE)
    save_csv(trades_df, TRADES_FILE)
    save_csv(performance_df, PERFORMANCE_FILE)
    save_csv(cash_df, CASH_FILE)

    return report


# =========================
# BENCHMARK
# =========================

def update_benchmarks(price_map):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    benchmark_df = read_csv_or_empty(
        BENCHMARK_FILE,
        ["benchmark", "start_date", "start_price", "shares"]
    )

    performance_df = load_performance()

    benchmark_report = {}

    for benchmark in ["SPY", "QQQ"]:
        price = price_map.get(benchmark)

        if price is None:
            continue

        existing = benchmark_df[benchmark_df["benchmark"] == benchmark]

        if existing.empty:
            shares = START_CAPITAL / price

            benchmark_df = pd.concat([
                benchmark_df,
                pd.DataFrame([{
                    "benchmark": benchmark,
                    "start_date": today,
                    "start_price": price,
                    "shares": shares
                }])
            ], ignore_index=True)

        else:
            shares = float(existing.iloc[0]["shares"])

        value = shares * price
        return_pct = ((value / START_CAPITAL) - 1) * 100

        performance_df = pd.concat([
            performance_df,
            pd.DataFrame([{
                "date": today,
                "strategy": benchmark,
                "portfolio_value": round(value, 2),
                "cash": 0,
                "positions_value": round(value, 2),
                "return_pct": round(return_pct, 2),
                "num_positions": 1
            }])
        ], ignore_index=True)

        benchmark_report[benchmark] = {
            "value": value,
            "return_pct": return_pct
        }

    save_csv(benchmark_df, BENCHMARK_FILE)
    save_csv(performance_df, PERFORMANCE_FILE)

    return benchmark_report


# =========================
# RAPPORT
# =========================

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


def build_portfolio_message(report, benchmark_report, rankings, analyzed_count, universe_count, spy, qqq):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    title_map = {
        "Momentum_AI": "🏆 Momentum_AI",
        "Quality_Momentum_AI": "🧠 Quality_Momentum_AI",
        "Aggressive_AI": "🔥 Aggressive_AI",
        "Low_Risk_AI": "🛡 Low_Risk_AI",
        "Balanced_AI": "⚖️ Balanced_AI"
    }

    explanation_map = {
        "Momentum_AI": "Jakter aksjer med sterkest fart og relativ styrke.",
        "Quality_Momentum_AI": "Ser etter sterk trend, men straffer ekstrem risiko.",
        "Aggressive_AI": "Tar høyere risiko for å fange eksplosive bevegelser.",
        "Low_Risk_AI": "Prioriterer lavere volatilitet og jevnere trend.",
        "Balanced_AI": "Kombinerer momentum, trend og risiko."
    }

    message = ""
    message += "📊 AI PORTFOLIO MANAGER v1\n"
    message += f"Dato: {today}\n"
    message += "Rapport: Paper portfolio kjøp/hold/selg\n"
    message += f"Univers: {universe_count} tickere\n"
    message += f"Analysert etter filtre: {analyzed_count} tickere\n\n"

    message += format_market_context(spy, qqq)
    message += "\n\n"

    # Leaderboard
    leaderboard = sorted(report.items(), key=lambda x: x[1]["return_pct"], reverse=True)

    message += "🏁 LEADERBOARD\n"
    for i, (strategy, data) in enumerate(leaderboard, start=1):
        message += (
            f"{i}. {strategy}: "
            f"${safe_round(data['portfolio_value'], 2)} "
            f"({safe_round(data['return_pct'], 2)}%) | "
            f"{data['num_positions']} posisjoner\n"
        )

    for benchmark, data in benchmark_report.items():
        message += (
            f"{benchmark}: "
            f"${safe_round(data['value'], 2)} "
            f"({safe_round(data['return_pct'], 2)}%)\n"
        )

    message += "\n"

    # Strategier
    for strategy, data in report.items():
        message += "━━━━━━━━━━━━━━\n"
        message += f"{title_map.get(strategy, strategy)}\n"
        message += f"{explanation_map.get(strategy, '')}\n"
        message += "━━━━━━━━━━━━━━\n"

        message += (
            f"Verdi: ${safe_round(data['portfolio_value'], 2)} "
            f"({safe_round(data['return_pct'], 2)}%)\n"
        )
        message += f"Cash: ${safe_round(data['cash'], 2)} | Posisjoner: {data['num_positions']}/{MAX_POSITIONS}\n"

        buys = data["buys"]
        sells = data["sells"]
        holds = data["holds"]
        top_candidates = data["top_candidates"]

        message += f"Topp kandidater nå: {', '.join(top_candidates) if top_candidates else 'Ingen'}\n"

        if buys:
            message += f"KJØP: {', '.join(buys)}\n"
        else:
            message += "KJØP: ingen\n"

        if holds:
            message += f"HOLD: {', '.join(holds[:10])}\n"
        else:
            message += "HOLD: ingen\n"

        if sells:
            message += f"SELG: {', '.join(sells)}\n"
        else:
            message += "SELG: ingen\n"

        # Vis topp 3 med litt data
        top_ranked = rankings[strategy][:3]
        message += "Topp 3 detaljer:\n"

        for stock in top_ranked:
            message += (
                f"- {stock['ticker']}: "
                f"Score {safe_round(stock['strategy_score'], 1)} | "
                f"3M {safe_round(stock['ret_3m'], 1)}% | "
                f"RS {safe_round(stock['relative_strength_3m'], 1)}% | "
                f"RSI {safe_round(stock['rsi'], 0)} | "
                f"Vol {safe_round(stock['volatility_60d'], 0)}%\n"
            )

        message += "\n"

    message += "⚠️ Dette er paper trading, ikke ekte ordre.\n"
    message += "Målet er å teste hvilke strategier som faktisk slår SPY/QQQ over tid.\n"
    message += "Ikke bruk dette til ekte kjøp/salg før strategiene har bevist seg over tid."

    return message


# =========================
# KJØR BOT
# =========================

def run_bot():
    ensure_data_dir()

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

    price_map = {}

    for ticker, df in market_data.items():
        try:
            price_map[ticker] = float(df["Close"].iloc[-1])
        except Exception:
            pass

    report = run_paper_portfolios(rankings, price_map)
    benchmark_report = update_benchmarks(price_map)

    message = build_portfolio_message(
        report=report,
        benchmark_report=benchmark_report,
        rankings=rankings,
        analyzed_count=len(analyzed_stocks),
        universe_count=len(universe),
        spy=spy,
        qqq=qqq
    )

    send_telegram(message)


if __name__ == "__main__":
    run_bot()
