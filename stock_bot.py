import os
import warnings

import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)

from modules.earnings import fetch_earnings_bulk
from modules.fundamentals import fetch_fundamentals_bulk
from modules.market_data import compute_premarket_moves, download_daily_data, get_price_cached, get_vix
from modules.macro import get_macro_status
from modules.portfolio import (
    build_target_weights,
    current_portfolio_value,
    execute_buy,
    execute_pyramid_fill,
    execute_sell,
)
from modules.reporting import (
    build_full_message,
    send_telegram,
)
from modules.risk import (
    drawdown_protection_message,
    get_portfolio_drawdown,
    should_sell_position,
    spy_is_recovering,
)
from modules.scoring import STRATEGIES, TOP_CANDIDATES_TO_REPORT, build_score_table, make_strategy_candidates
from modules.sentiment import fetch_sentiment_bulk
from modules.state import (
    GLOBAL_STATE_FILE,
    PERFORMANCE_FILE,
    SIGNALS_DIR,
    START_CAPITAL,
    TRADES_FILE,
    ensure_dirs,
    load_benchmark_state,
    load_json,
    load_latest_signal,
    load_performance,
    load_strategy_state,
    load_trades,
    now_ny,
    now_oslo,
    now_utc,
    reset_week_if_needed,
    safe_float,
    safe_round,
    save_benchmark_state,
    save_csv,
    save_json,
    save_strategy_state,
    today_str,
    trading_days_between,
)
from modules.technicals import analyze_stock
from modules.universe import build_universe
from modules.fundamentals import FUNDAMENTALS_CACHE_FILE


# ============================================================
# MARKET REGIME
# ============================================================

def detect_market_regime(spy_data, qqq_data, vix):
    if spy_data is None or qqq_data is None:
        return {"regime": "unknown", "reason": "Mangler benchmark-data", "vix": vix}

    spy_bull = spy_data["above_sma200"] and spy_data["ret_3m"] > 0
    qqq_bull = qqq_data["above_sma200"] and qqq_data["ret_3m"] > 0

    if spy_bull and qqq_bull:
        if vix is not None and vix < 15:
            return {
                "regime": "explosive",
                "reason": f"SPY+QQQ over 200SMA, begge positiv 3M, VIX={vix:.1f}<15",
                "vix": vix,
            }
        return {
            "regime": "bullish",
            "reason": "SPY og QQQ over 200SMA med positiv 3M trend",
            "vix": vix,
        }

    if spy_data["above_sma200"] or qqq_data["above_sma200"]:
        return {
            "regime": "neutral",
            "reason": "Blandet marked — én av SPY/QQQ over 200SMA",
            "vix": vix,
        }

    return {
        "regime": "defensive",
        "reason": "SPY og QQQ begge under 200SMA",
        "vix": vix,
    }


# ============================================================
# BENCHMARK HELPERS
# ============================================================

def update_benchmarks(price_cache):
    state = load_benchmark_state()
    for symbol in ("SPY", "QQQ"):
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


# ============================================================
# SIGNAL MODE
# ============================================================

def run_signal():
    print("=" * 60)
    print("SIGNAL MODE")
    print("=" * 60)

    universe = build_universe()
    market_data = download_daily_data(universe)

    # Benchmark analysis
    spy_data, qqq_data = None, None
    if "SPY" in market_data:
        spy_data = analyze_stock("SPY", market_data["SPY"], None)
    spy_3m = spy_data["ret_3m"] if spy_data else None
    if "QQQ" in market_data:
        qqq_data = analyze_stock("QQQ", market_data["QQQ"], spy_3m)

    # Technical analysis for all stocks
    analyzed = []
    for ticker, df in market_data.items():
        if ticker in ("SPY", "QQQ"):
            continue
        result = analyze_stock(ticker, df, spy_3m)
        if result:
            analyzed.append(result)

    print(f"Teknisk analyse: {len(analyzed)} aksjer analysert")

    if not analyzed:
        raise ValueError("Ingen aksjer analysert. Sjekk datakilde.")

    all_tickers = [a["ticker"] for a in analyzed]

    # Fundamentals for all tickers (cached, 7 days)
    fundamentals = fetch_fundamentals_bulk(all_tickers)

    # Preliminary scoring (neutral 0.5 sentiment) to identify top candidates
    neutral_sentiment = {t: {"score": 0.5, "raw_score": 0.0} for t in all_tickers}
    neutral_earnings = {t: {"earnings_bonus": 0, "earnings_soon": False} for t in all_tickers}
    prelim_df = build_score_table(analyzed, fundamentals, neutral_sentiment, neutral_earnings)

    # Identify tickers needing real sentiment/earnings:
    # owned positions + top 20 per strategy (union)
    sentiment_tickers = set()
    for strat_name, config in STRATEGIES.items():
        strat_state = load_strategy_state(strat_name)
        sentiment_tickers.update(strat_state.get("positions", {}).keys())

    for strat_name, config in STRATEGIES.items():
        col = config["score_column"]
        if col in prelim_df.columns:
            top20 = prelim_df[col].nlargest(20).index.tolist()
            sentiment_tickers.update(top20)

    # Filter to tickers that are actually in our analyzed set
    sentiment_tickers = [t for t in sentiment_tickers if t in all_tickers]
    print(f"Sentiment + earnings fetch for {len(sentiment_tickers)} tickere")

    earnings_data = fetch_earnings_bulk(sentiment_tickers)
    sentiment_scores = fetch_sentiment_bulk(sentiment_tickers)

    # Expand to full universe (others get neutral)
    full_earnings = {t: neutral_earnings[t] for t in all_tickers}
    full_earnings.update(earnings_data)

    full_sentiment = {t: neutral_sentiment[t] for t in all_tickers}
    full_sentiment.update(sentiment_scores)

    # Final scoring with all 4 factors
    score_df = build_score_table(analyzed, fundamentals, full_sentiment, full_earnings)

    if score_df.empty:
        raise ValueError("Score-tabellen er tom. Sjekk datakilde.")

    # Market regime
    vix = get_vix()
    regime = detect_market_regime(spy_data, qqq_data, vix)
    print(f"Markedsregime: {regime['regime'].upper()} — {regime['reason']}")

    # Build signal payload
    strategies_payload = {}
    for strat_name, config in STRATEGIES.items():
        candidates = make_strategy_candidates(score_df, strat_name, config)
        strategies_payload[strat_name] = {
            "description": config["description"],
            "score_column": config["score_column"],
            "candidates": candidates,
        }

    payload = {
        "created_at_utc": now_utc().isoformat(),
        "created_at_oslo": now_oslo().isoformat(),
        "created_at_ny": now_ny().isoformat(),
        "date": today_str(),
        "universe_count": len(universe),
        "analyzed_count": len(analyzed),
        "regime": regime,
        "spy": spy_data,
        "qqq": qqq_data,
        "strategies": strategies_payload,
    }

    date = today_str()
    signal_path = f"{SIGNALS_DIR}/{date}_signal.json"
    save_json(payload, signal_path)
    save_json(
        {"last_signal_file": signal_path, "updated_at_utc": now_utc().isoformat()},
        GLOBAL_STATE_FILE,
    )

    print(f"Signal lagret: {signal_path}")
    print("Signal-modus stille — ingen Telegram-melding.")


# ============================================================
# EXECUTE MODE — per strategy
# ============================================================

def run_strategy_execution(
    strategy_name, signal, trades_df, price_cache, premarket_flags=None, macro_mult=1.0
):
    if premarket_flags is None:
        premarket_flags = {}
    config = STRATEGIES[strategy_name]
    state = load_strategy_state(strategy_name)
    state = reset_week_if_needed(state)

    strategy_payload = signal["strategies"].get(strategy_name, {})
    candidates = strategy_payload.get("candidates", [])
    candidates_by_ticker = {c["ticker"]: c for c in candidates}

    regime_name = signal.get("regime", {}).get("regime", "unknown")

    # Valuation before any trades
    total_before, _, state = current_portfolio_value(state, candidates_by_ticker, price_cache)

    # Portfolio drawdown from peak
    drawdown = get_portfolio_drawdown(state, total_before)
    dd_msg = drawdown_protection_message(drawdown)
    if dd_msg:
        print(f"  [{strategy_name}] {dd_msg}")

    sells = []
    buys = []

    # Evaluate sells
    for ticker, pos in list(state["positions"].items()):
        candidate = candidates_by_ticker.get(ticker)
        sell, reason = should_sell_position(ticker, pos, candidate, config)
        if sell:
            state, trades_df, trade = execute_sell(
                state, trades_df, strategy_name, ticker, reason,
                candidates_by_ticker, price_cache
            )
            if trade:
                sells.append(trade)
                print(f"  [{strategy_name}] SELG {ticker}: {reason}")

    # Re-valuate after sells
    total_now, _, state = current_portfolio_value(state, candidates_by_ticker, price_cache)
    drawdown_now = get_portfolio_drawdown(state, total_now)

    # Spy recovery check (required to buy when in drawdown protection)
    spy_candidate = candidates_by_ticker.get("SPY", signal.get("spy"))
    in_drawdown_protection = drawdown_now <= -0.20
    can_buy = (not in_drawdown_protection) or spy_is_recovering(spy_candidate)

    target_weights = build_target_weights(
        candidates, config, regime_name,
        portfolio_drawdown=drawdown_now, macro_mult=macro_mult,
    )
    target_tickers = list(target_weights.keys())

    # Pyramid fills for existing partial positions
    pyramid_fills = []
    for ticker in list(state["positions"].keys()):
        state, trades_df, fill = execute_pyramid_fill(
            state, trades_df, strategy_name, ticker, candidates_by_ticker, price_cache
        )
        if fill:
            pyramid_fills.append(fill)
            print(f"  [{strategy_name}] PYRAMID {ticker}: {fill['reason']}")

    # Count sector positions for correlation filter
    def _sector_counts():
        counts = {}
        for t in state["positions"]:
            sector = candidates_by_ticker.get(t, {}).get("sector", "Unknown")
            counts[sector] = counts.get(sector, 0) + 1
        return counts

    # Evaluate new buys
    if can_buy:
        for ticker in target_tickers:
            if ticker in state["positions"]:
                continue
            if len(state["positions"]) >= config["max_positions"]:
                break
            if state["weekly_meta"]["buys_this_week"] >= config["max_new_buys_per_week"]:
                break

            cooldown_date = state["cooldowns"].get(ticker)
            if trading_days_between(cooldown_date) < config["buyback_cooldown_days"]:
                continue

            candidate = candidates_by_ticker.get(ticker)
            if not candidate:
                continue

            # Earnings blackout: skip if earnings within 3 trading days
            days_to_earnings = candidate.get("days_to_earnings")
            if days_to_earnings is not None and 0 <= days_to_earnings <= 3:
                print(f"  [{strategy_name}] SKIP {ticker}: earnings om {days_to_earnings}d (blackout)")
                continue

            # Correlation filter: max 2 positions per sector per strategy
            sector = candidate.get("sector", "Unknown")
            sector_counts = _sector_counts()
            if sector_counts.get(sector, 0) >= 2:
                print(f"  [{strategy_name}] SKIP {ticker}: sektorbegrensning {sector} ({sector_counts.get(sector)}≥2)")
                continue

            target_value_full = total_now * target_weights[ticker]
            # Pyramid: buy 60% now, queue 40% for fill after 5 days + >2% gain
            initial_value = target_value_full * 0.60
            pyramid_remaining = target_value_full * 0.40
            reason = f"Rank #{candidate.get('rank')}, score {safe_round(candidate.get('strategy_score'), 1)}"

            pm_move = premarket_flags.get(ticker)
            state, trades_df, trade = execute_buy(
                state, trades_df, strategy_name, ticker, initial_value,
                reason, candidate, price_cache,
                premarket_move=pm_move,
                pyramid_remaining=pyramid_remaining,
            )
            if trade:
                buys.append(trade)
                print(f"  [{strategy_name}] KJØP 60% {ticker}: {trade['reason']}")

    # Build sector map for held positions (for reporting)
    sector_map = {}
    for ticker in state["positions"]:
        sector_map[ticker] = candidates_by_ticker.get(ticker, {}).get("sector", "Unknown")

    # Final valuation
    total_after, positions_value_after, state = current_portfolio_value(
        state, candidates_by_ticker, price_cache
    )

    state["highest_portfolio_value"] = max(
        float(state.get("highest_portfolio_value", START_CAPITAL)),
        total_after,
    )
    state["last_execution_date"] = today_str()
    save_strategy_state(strategy_name, state)

    return_pct = (total_after / START_CAPITAL) - 1

    # Pre-market flags relevant to this strategy (held positions + new buys)
    relevant_tickers = set(state["positions"].keys()) | {t["ticker"] for t in buys}
    strategy_pm_flags = {t: m for t, m in premarket_flags.items() if t in relevant_tickers}

    return {
        "strategy": strategy_name,
        "description": config["description"],
        "portfolio_value": total_after,
        "cash": state["cash"],
        "positions_value": positions_value_after,
        "return_pct": return_pct,
        "num_positions": len(state["positions"]),
        "positions": state["positions"],
        "buys": buys + pyramid_fills,
        "sells": sells,
        "top_candidates": candidates[:TOP_CANDIDATES_TO_REPORT],
        "top_candidates_full": candidates[:50],
        "drawdown": drawdown_now,
        "premarket_flags": strategy_pm_flags,
        "sector_map": sector_map,
    }, trades_df


# ============================================================
# EXECUTE MODE — orchestrator
# ============================================================

def run_execute():
    print("=" * 60)
    print("EXECUTE MODE")
    print("=" * 60)

    signal, signal_path = load_latest_signal()
    print(f"Bruker signal: {signal_path}")

    trades_df = load_trades()
    performance_df = load_performance()
    price_cache = {}

    benchmark_state = update_benchmarks(price_cache)
    spy_ret = benchmark_return(benchmark_state, "SPY")
    qqq_ret = benchmark_return(benchmark_state, "QQQ")

    print(f"SPY siden start: {(spy_ret or 0)*100:.2f}%  |  QQQ: {(qqq_ret or 0)*100:.2f}%")

    # Macro filter (FRED 10Y/2Y yield spread, no API key)
    macro = get_macro_status()
    macro_mult = macro.get("exposure_mult", 1.0)

    # --- Pre-market filter ---
    # Collect held tickers + top candidates from signal as prev_prices reference
    prev_prices = {}
    pm_tickers = set()

    for strat_name in STRATEGIES:
        strat_state = load_strategy_state(strat_name)
        held = strat_state.get("positions", {})
        pm_tickers.update(held.keys())
        payload = signal.get("strategies", {}).get(strat_name, {})
        for c in payload.get("candidates", [])[:30]:
            t = c.get("ticker")
            if t:
                pm_tickers.add(t)
                if t not in prev_prices and c.get("price"):
                    prev_prices[t] = c["price"]
        # For held positions, use last_price as prev_close if not in candidates
        for t, pos in held.items():
            if t not in prev_prices:
                prev_prices[t] = pos.get("last_price") or pos.get("avg_price")

    all_pm_moves = compute_premarket_moves(list(pm_tickers), prev_prices)
    premarket_flags = {t: m for t, m in all_pm_moves.items() if abs(m) > 0.04}

    results = []
    drawdown_warnings = []

    for strategy_name in STRATEGIES.keys():
        print(f"\nKjører {strategy_name}...")
        result, trades_df = run_strategy_execution(
            strategy_name, signal, trades_df, price_cache,
            premarket_flags=premarket_flags,
            macro_mult=macro_mult,
        )
        results.append(result)

        if result["drawdown"] <= -0.20:
            msg = drawdown_protection_message(result["drawdown"])
            if msg:
                drawdown_warnings.append(f"{strategy_name}: {msg}")

        performance_row = {
            "date": today_str(),
            "strategy": strategy_name,
            "portfolio_value": round(result["portfolio_value"], 2),
            "cash": round(result["cash"], 2),
            "positions_value": round(result["positions_value"], 2),
            "return_pct": round(result["return_pct"] * 100, 2),
            "num_positions": result["num_positions"],
            "spy_return_pct": round(spy_ret * 100, 2) if spy_ret is not None else None,
            "qqq_return_pct": round(qqq_ret * 100, 2) if qqq_ret is not None else None,
        }
        performance_df = pd.concat(
            [performance_df, pd.DataFrame([performance_row])], ignore_index=True
        )

    save_csv(trades_df, TRADES_FILE)
    save_csv(performance_df, PERFORMANCE_FILE)

    # Load fundamentals cache for weekly sector report
    fundamentals_cache = load_json(FUNDAMENTALS_CACHE_FILE, {})

    message = build_full_message(
        results=results,
        signal=signal,
        signal_path=signal_path,
        spy_ret=spy_ret,
        qqq_ret=qqq_ret,
        drawdown_warnings=drawdown_warnings,
        fundamentals_cache=fundamentals_cache,
        macro=macro,
    )

    send_telegram(message)


# ============================================================
# MAIN
# ============================================================

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
