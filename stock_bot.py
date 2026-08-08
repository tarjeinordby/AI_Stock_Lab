import json
import os
import warnings

import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)

from modules.earnings import fetch_deep_earnings_analysis, fetch_earnings_bulk
from modules.fundamentals import fetch_fundamentals_bulk, fetch_insider_bulk
from modules.exchange_calendar import CalendarUnavailableError, intended_execution_session
from modules.orders import (
    EXECUTED, EXPIRED, FAILED_PRICE, PENDING_PRICE, SETTLING, TERMINAL,  # noqa: F401 (FAILED_PRICE used in legacy print)
    expire_stale_orders,
    get_or_create_order,
    get_pending_for_session,
    load_orders,

    make_trade_id,
    reconcile_settling_orders,

    save_order,
)
from modules.versioning import PORTFOLIO_VERSION
from modules.market_data import compute_premarket_moves, download_daily_data, get_price_cached, get_vix, prefetch_execution_prices, prefetch_open_prices, run_data_quality_checks
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
from modules.correlation import (
    check_correlation_against_held,
    check_sub_sector_concentration,
    compute_correlation_matrix,
    filter_correlated_sells,
    find_correlated_pairs,
    SUB_SECTOR_LABELS,
)
from modules.scoring import (
    STRATEGIES,
    TOP_CANDIDATES_TO_REPORT,
    build_score_table,
    get_regime_weights,
    make_strategy_candidates,
    passes_quality_filter,
)
from modules.reporting import is_monday
from modules.sentiment import fetch_sentiment_bulk
from modules.weekly_analysis import fetch_weekly_analysis
from modules.state import (
    GLOBAL_STATE_FILE,
    PERFORMANCE_FILE,
    SIGNALS_DIR,
    STATE_DIR,
    START_CAPITAL,
    TRADES_FILE,
    SignalNotPublishedError,
    ensure_dirs,
    load_benchmark_state,
    load_json,
    load_latest_signal,
    load_performance,
    load_strategy_state,
    load_trades,
    is_market_open,
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
from modules.ledger import (
    LedgerWriteError,
    build_all_ledger_records,
    build_signal_run_record,
    check_ledger_file_sizes,
    compute_input_manifest,
    compute_universe_hash,
    get_signal_run_status,
    make_signal_run_id,
    update_signal_run_status,
    write_feature_snapshots,
    write_signal_records,
    write_signal_run,
)
from modules.versioning import EXECUTION_VERSION, MODEL_VERSION, get_model_config_hash


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
# PRE-MARKET MODE
# ============================================================

def run_premarket():
    print("=" * 60)
    print("PRE-MARKET MODE")
    print("=" * 60)

    signal, signal_path = load_latest_signal()
    print(f"Bruker signal: {signal_path}")

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
        for t, pos in held.items():
            if t not in prev_prices:
                prev_prices[t] = pos.get("last_price") or pos.get("avg_price")

    all_moves = compute_premarket_moves(list(pm_tickers), prev_prices)

    result = {
        "fetched_at_utc": now_utc().isoformat(),
        "moves": {t: round(m, 6) for t, m in all_moves.items()},
        "flagged": {t: round(m, 6) for t, m in all_moves.items() if abs(m) > 0.04},
    }

    out_path = f"{STATE_DIR}/premarket_prices.json"
    save_json(result, out_path)
    print(f"Pre-market priser lagret: {out_path}")


# ============================================================
# SPLIT ADJUSTMENTS
# ============================================================

def apply_split_adjustments(split_detected):
    """
    For confirmed splits, adjust shares and avg_price in all strategy states.
    Returns list of adjustment records for logging.
    """
    adjustments = []
    for split in split_detected:
        ratio = split.get("ratio")
        if not ratio:
            continue
        ticker = split["ticker"]
        for strat_name in STRATEGIES:
            state = load_strategy_state(strat_name)
            pos = state["positions"].get(ticker)
            if not pos:
                continue
            old_shares = pos.get("shares", 0)
            old_avg = pos.get("avg_price", 0)
            pos["shares"] = round(old_shares * ratio, 6)
            pos["avg_price"] = round(old_avg / ratio, 6)
            if pos.get("pyramid_min_price"):
                pos["pyramid_min_price"] = round(pos["pyramid_min_price"] / ratio, 6)
            save_strategy_state(strat_name, state)
            adjustments.append({
                "ticker": ticker, "strategy": strat_name, "ratio": ratio,
                "old_shares": old_shares, "new_shares": pos["shares"],
            })
            print(f"  Split {ticker} {ratio}:1 [{strat_name}]: {old_shares:.2f} → {pos['shares']:.2f} aksjer")
    return adjustments


# ============================================================
# SIGNAL MODE
# ============================================================

def run_signal():
    print("=" * 60)
    print("SIGNAL MODE")
    print("=" * 60)

    universe = build_universe()
    market_data = download_daily_data(universe)
    market_data, quality_report = run_data_quality_checks(market_data)

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

    # Fundamentals for all tickers (cached, 7 days) — inkluderer nå FCF
    fundamentals = fetch_fundamentals_bulk(all_tickers)

    # Quality filter: fjern lavkvalitet-tickere fra scoring
    # Eide posisjoner beholdes alltid (kan fortsatt selges)
    held_set = set()
    for sn in STRATEGIES:
        held_set.update(load_strategy_state(sn).get("positions", {}).keys())
    before_count = len(analyzed)
    analyzed = [a for a in analyzed if passes_quality_filter(a["ticker"], fundamentals, held_set)]
    filtered_count = before_count - len(analyzed)
    print(f"Quality filter: {before_count} → {len(analyzed)} tickere ({filtered_count} filtrert bort)")

    all_tickers = [a["ticker"] for a in analyzed]

    # Preliminary scoring (neutral data, default weights) to identify top candidates
    neutral_sentiment = {t: {"score": 0.5, "raw_score": 0.0} for t in all_tickers}
    neutral_earnings = {t: {"earnings_bonus": 0, "earnings_soon": False} for t in all_tickers}
    prelim_df = build_score_table(analyzed, fundamentals, neutral_sentiment, neutral_earnings)

    # Identify tickers needing real sentiment/earnings + insider fetch.
    # Eide aksjer prioriteres først, deretter topp-kandidater per strategi.
    seen = set()
    sentiment_tickers = []
    for strat_name in STRATEGIES:
        strat_state = load_strategy_state(strat_name)
        for t in strat_state.get("positions", {}).keys():
            if t in all_tickers and t not in seen:
                sentiment_tickers.append(t)
                seen.add(t)

    for strat_name, config in STRATEGIES.items():
        col = config["score_column"]
        if col in prelim_df.columns:
            for t in prelim_df[col].nlargest(20).index.tolist():
                if t in all_tickers and t not in seen:
                    sentiment_tickers.append(t)
                    seen.add(t)
    print(f"Sentiment + earnings + insider fetch for {len(sentiment_tickers)} tickere")

    # Insider-signal for topp-kandidater (ingen cache, liten subset)
    insider_data = fetch_insider_bulk(sentiment_tickers)
    for ticker, insider in insider_data.items():
        if ticker in fundamentals:
            fundamentals[ticker].update(insider)

    earnings_data = fetch_earnings_bulk(sentiment_tickers)
    sentiment_scores, sentiment_cost = fetch_sentiment_bulk(sentiment_tickers)

    # Layer 3: Opus earnings deep-dive for tickers with earnings within 14 days
    deep_earnings, earnings_ai_cost = fetch_deep_earnings_analysis(earnings_data)
    signal_ai_cost = sentiment_cost + earnings_ai_cost

    # Expand to full universe (others get neutral)
    full_earnings = {t: neutral_earnings[t] for t in all_tickers}
    full_earnings.update(earnings_data)

    full_sentiment = {t: neutral_sentiment[t] for t in all_tickers}
    full_sentiment.update(sentiment_scores)

    # Market regime + macro (must be before final scoring for dynamic weights)
    vix = get_vix()
    regime = detect_market_regime(spy_data, qqq_data, vix)
    print(f"Markedsregime: {regime['regime'].upper()} — {regime['reason']}")

    macro = get_macro_status()
    macro_inverted = macro.get("inverted", False)
    active_weights = get_regime_weights(regime["regime"], macro_inverted)
    print(f"Aktive vekter: Mom {active_weights['momentum']:.0%} | "
          f"Kval {active_weights['quality']:.0%} | "
          f"Verdi {active_weights['value']:.0%} | "
          f"Sent {active_weights['sentiment']:.0%}")

    # Final scoring with regime-adjusted weights
    score_df = build_score_table(
        analyzed, fundamentals, full_sentiment, full_earnings,
        regime=regime["regime"], macro_inverted=macro_inverted,
    )

    if score_df.empty:
        raise ValueError("Score-tabellen er tom. Sjekk datakilde.")

    # Correlation matrix for all analyzed tickers (used in execute-mode filtering)
    corr_matrix = compute_correlation_matrix(market_data, all_tickers)
    corr_pairs = find_correlated_pairs(corr_matrix, threshold=0.70)
    if corr_pairs:
        high_corr = [p for p in corr_pairs if p["correlation"] >= 0.75]
        print(f"Korrelasjonspar ≥70%: {len(corr_pairs)} | ≥75%: {len(high_corr)}")

    # Build signal payload
    strategies_payload = {}
    for strat_name, config in STRATEGIES.items():
        candidates = make_strategy_candidates(score_df, strat_name, config)
        strategies_payload[strat_name] = {
            "description": config["description"],
            "score_column": config["score_column"],
            "candidates": candidates,
        }

    date = today_str()
    year_month = date[:7]

    # ----------------------------------------------------------------
    # Ledger write — must succeed before signal file is saved.
    # Failure → status=failed, Telegram alert, no signal file.
    # ----------------------------------------------------------------
    model_cfg_hash = get_model_config_hash()
    universe_hash = compute_universe_hash(universe)  # full requested universe (pre-quality-filter)
    input_manifest = compute_input_manifest(market_data, quality_report, date)
    signal_run_id = make_signal_run_id(
        date,
        MODEL_VERSION,
        model_cfg_hash,
        universe_hash,
        input_manifest["canonical_data_cutoff"],
        input_manifest["input_manifest_hash"],
    )

    # File size drift warning — non-blocking
    large_files = check_ledger_file_sizes(year_month)
    for fname, size_mb in large_files.items():
        send_telegram(f"⚠️ LEDGER-STØRRELSE: {fname} er {size_mb:.1f} MB (grense: 50 MB)")
        print(f"⚠️ Ledger file size warning: {fname} = {size_mb:.1f} MB")

    existing = get_signal_run_status(signal_run_id, year_month)
    if existing and existing.get("status") == "completed":
        print(f"Ledger: run {signal_run_id} allerede fullført (idempotent retry).")
        # Blocker 1: embed ledger signal_ids into candidates on completed-run retry.
        # Fail closed if any candidate lacks exactly one matching signal_record.
        from modules.ledger import get_signal_records_for_run  # noqa: PLC0415
        signal_records_retry = get_signal_records_for_run(signal_run_id, year_month)
        sig_id_map_retry = {
            (sr["strategy_id"], sr["ticker"]): sr["signal_id"]
            for sr in signal_records_retry
        }
        for strat_name, strat_data in strategies_payload.items():
            for c in strat_data.get("candidates", []):
                sid = sig_id_map_retry.get((strat_name, c.get("ticker", "")))
                if sid is None:
                    raise RuntimeError(
                        f"Idempotent retry: ingen signal_record for "
                        f"{strat_name}/{c.get('ticker')} i run {signal_run_id} — fail-closed"
                    )
                c["signal_id"] = sid
    else:
        run_record = build_signal_run_record(
            signal_run_id=signal_run_id,
            intended_session=date,
            model_version=MODEL_VERSION,
            model_config_hash=model_cfg_hash,
            universe_hash=universe_hash,
            canonical_data_cutoff=input_manifest["canonical_data_cutoff"],
            input_manifest_hash=input_manifest["input_manifest_hash"],
            universe_count=len(universe),
            analyzed_count=len(analyzed),
            observed_min_data_timestamp=input_manifest.get("observed_min_data_timestamp"),
            observed_max_data_timestamp=input_manifest.get("observed_max_data_timestamp"),
            valid_ticker_count=input_manifest.get("valid_ticker_count", 0),
            stale_ticker_count=input_manifest.get("stale_ticker_count", 0),
            excluded_ticker_count=input_manifest.get("excluded_ticker_count", 0),
        )
        write_signal_run(run_record, year_month)
        update_signal_run_status(signal_run_id, "pending", year_month)

        analyzed_by_ticker = score_df.reset_index().to_dict(orient="records")
        try:
            feature_snapshots, signal_records = build_all_ledger_records(
                signal_run_id=signal_run_id,
                analyzed_by_ticker=analyzed_by_ticker,
                strategies_payload=strategies_payload,
                fundamentals=fundamentals,
                sentiment_scores=full_sentiment,
                full_earnings=full_earnings,
                macro=macro,
                regime=regime["regime"],
                model_version=MODEL_VERSION,
                model_config_hash=model_cfg_hash,
                strategies_config=STRATEGIES,
            )
            n_fs = write_feature_snapshots(feature_snapshots, year_month)
            n_sr = write_signal_records(signal_records, year_month)
            update_signal_run_status(
                signal_run_id, "completed", year_month,
                n_feature_snapshots=n_fs, n_signal_records=n_sr,
            )
            print(f"Ledger: {n_fs} feature snapshots, {n_sr} signal records skrevet.")

            # Embed the ledger's authoritative signal_id into each candidate so execute can
            # read candidate["signal_id"] directly — no separate ID system in execute.
            sig_id_map = {
                (sr["strategy_id"], sr["ticker"]): sr["signal_id"]
                for sr in signal_records
            }
            for strat_name, strat_data in strategies_payload.items():
                for c in strat_data.get("candidates", []):
                    c["signal_id"] = sig_id_map.get((strat_name, c.get("ticker", "")))
        except LedgerWriteError as exc:
            update_signal_run_status(
                signal_run_id, "failed", year_month, error=str(exc)
            )
            send_telegram(f"⚠️ LEDGER-FEIL — signal ikke lagret: {exc}")
            raise

    # Compute intended_execution_session for this signal (next NYSE session after today)
    try:
        _next_session = intended_execution_session(date)
    except CalendarUnavailableError:
        _next_session = None
        print("ADVARSEL: Kan ikke beregne intended_execution_session — børskalender utilgjengelig")

    payload = {
        "created_at_utc": now_utc().isoformat(),
        "created_at_oslo": now_oslo().isoformat(),
        "created_at_ny": now_ny().isoformat(),
        "date": date,
        "signal_run_id": signal_run_id,
        # --- Session metadata (required by signal_validator) ---
        "intended_execution_session": _next_session,
        "model_version": MODEL_VERSION,
        # --- Data cutoff (required by signal_validator, cross-validated with ledger) ---
        "data_cutoff_at": input_manifest["canonical_data_cutoff"],
        # --- Data quality counts ---
        "valid_ticker_count": input_manifest.get("valid_ticker_count", 0),
        "stale_ticker_count": input_manifest.get("stale_ticker_count", 0),
        "excluded_ticker_count": input_manifest.get("excluded_ticker_count", 0),
        "missing_ticker_count": 0,
        # --- Publication fields: set to draft; populated by run_finalize_signal() ---
        "publication_status": "draft",
        "published_at": None,
        "published_commit_sha": None,
        "signal_content_hash": "",
        # --- Signal content ---
        "universe_count": len(universe),
        "analyzed_count": len(analyzed),
        "regime": regime,
        "spy": spy_data,
        "qqq": qqq_data,
        "strategies": strategies_payload,
        "earnings_analysis": deep_earnings,
        "active_weights": active_weights,
        "corr_pairs": corr_pairs,
        "sentiment_scores": {
            t: {k: v for k, v in d.items() if k in ("score", "source", "error", "summary")}
            for t, d in sentiment_scores.items()
        },
        "quality_report": quality_report,
        "signal_ai_cost_usd": round(signal_ai_cost, 6),
        "quality_filtered_count": filtered_count,
    }

    signal_path = f"{SIGNALS_DIR}/{date}_signal.json"
    save_json(payload, signal_path)
    save_json(
        {"last_signal_file": signal_path, "updated_at_utc": now_utc().isoformat()},
        GLOBAL_STATE_FILE,
    )

    print(f"Signal (draft) lagret: {signal_path}")
    print("Venter på finalisering etter vellykket git push (BOT_MODE=finalize_signal).")


# ============================================================
# FINALIZE SIGNAL — run after successful git push in signal.yml
# ============================================================

def run_finalize_signal():
    """
    Called as BOT_MODE=finalize_signal after a successful git push.

    Reads the draft signal written by run_signal(), updates publication fields,
    recomputes signal_content_hash, saves the finalized signal in place,
    and appends a publication record to signal_publications.jsonl.

    SIGNAL_COMMIT_SHA env var must be set to the SHA of the push that
    committed the draft signal.
    """
    print("=" * 60)
    print("FINALIZE SIGNAL MODE")
    print("=" * 60)

    commit_sha = os.environ.get("SIGNAL_COMMIT_SHA", "").strip()
    if not commit_sha:
        raise RuntimeError(
            "SIGNAL_COMMIT_SHA ikke satt. "
            "Kjøres kun fra signal.yml etter vellykket git push."
        )

    global_state = load_json(GLOBAL_STATE_FILE, {})
    signal_path = global_state.get("last_signal_file")
    if not signal_path or not os.path.exists(signal_path):
        raise FileNotFoundError(
            f"Ingen signalfil å finalisere: {signal_path!r}"
        )

    with open(signal_path, "r") as f:
        payload = json.load(f)

    if payload.get("publication_status") == "published":
        print(f"Signal allerede publisert: {signal_path} — ingen handling.")
        return

    published_at = now_utc().isoformat()
    payload["publication_status"] = "published"
    payload["published_at"] = published_at
    payload["published_commit_sha"] = commit_sha

    from modules.signal_validator import compute_signal_content_hash  # noqa: PLC0415
    payload["signal_content_hash"] = compute_signal_content_hash(payload)

    save_json(payload, signal_path)
    print(f"Signal finalisert: {signal_path}")
    print(f"  published_commit_sha: {commit_sha}")
    print(f"  published_at:         {published_at}")
    print(f"  signal_content_hash:  {payload['signal_content_hash'][:16]}...")

    from modules.publication import write_signal_publication  # noqa: PLC0415
    signal_run_id = payload.get("signal_run_id", "unknown")
    write_signal_publication(
        signal_run_id=signal_run_id,
        signal_path=signal_path,
        commit_sha=commit_sha,
        published_at=published_at,
        workflow_conclusion="success",
    )
    print(f"Publikasjonsrecord (signal_publication) skrevet for {signal_run_id}")
    print(f"  Venter på Phase 4 push → BOT_MODE=ack_publication for persistence-ack")


def run_ack_publication():
    """
    Called as BOT_MODE=ack_publication after the final git push (Phase 4).

    Appends a publication_ack record with finalized_commit_sha — the SHA of
    the commit that actually contains the finalized signal file. This is the
    verifiable persistence acknowledgment required by signal_validator.py Layer 3.

    FINAL_COMMIT_SHA env var must be set to the SHA of the Phase 4 push.
    """
    print("=" * 60)
    print("ACK PUBLICATION MODE")
    print("=" * 60)

    final_sha = os.environ.get("FINAL_COMMIT_SHA", "").strip()
    if not final_sha:
        raise RuntimeError(
            "FINAL_COMMIT_SHA ikke satt. "
            "Kjøres kun fra signal.yml etter Phase 4 (final) push."
        )

    global_state = load_json(GLOBAL_STATE_FILE, {})
    signal_path = global_state.get("last_signal_file")
    if not signal_path or not os.path.exists(signal_path):
        raise FileNotFoundError(
            f"Ingen signalfil registrert for ack: {signal_path!r}"
        )

    with open(signal_path, "r") as f:
        payload = json.load(f)

    signal_run_id = payload.get("signal_run_id")
    if not signal_run_id:
        raise ValueError("signal_run_id mangler i signalfilen — kan ikke skrive ack")

    from modules.publication import write_publication_ack  # noqa: PLC0415
    write_publication_ack(signal_run_id=signal_run_id, finalized_commit_sha=final_sha)
    print(f"Persistence-ack skrevet for {signal_run_id}")
    print(f"  finalized_commit_sha: {final_sha}")


# ============================================================
# EXECUTE MODE — per strategy
# ============================================================


def _write_fill_event_for_order(order: dict, trade: dict, cash_before: float,
                                cash_after: float, execution_version: str) -> dict:
    """Append a fill event to the WAL and return the written record (including fill_attempt_id).

    action is taken from order["action"] — the authoritative source. trade.get("action")
    may be "BUY" even for a PYRAMID_FILL trade, so we must not use it.

    order["intended_execution_session"] is the authority for session. trade["execution_session"]
    must equal it — any mismatch is fail-closed (session integrity guard).

    execution_price_timestamp is set to the NYSE session open (09:30 ET) for the intended
    execution session — not the runner processing time.
    """
    from modules.fills import write_fill_event  # noqa: PLC0415
    price = float(trade.get("price", 0))
    commission = float(trade.get("cost", 0))

    # order.intended_execution_session is the authority — validate trade session matches
    intended_session = order.get("intended_execution_session", "")
    actual_session = trade.get("execution_session", trade.get("date", ""))
    if actual_session != intended_session:
        raise RuntimeError(
            f"Session-mismatch for ordre {order.get('order_id')}: "
            f"intended={intended_session!r} ≠ trade_session={actual_session!r} "
            "— fyll-event avvist fail-closed"
        )

    # Fail-closed: calendar must be available; no fallback to runner timestamp (Bug 7 fix).
    # Use intended_session (authoritative) for the timestamp lookup.
    from modules.exchange_calendar import session_open_utc  # noqa: PLC0415
    try:
        open_ts = session_open_utc(intended_session)
        execution_price_timestamp = open_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as exc:
        raise RuntimeError(
            f"Kan ikke bestemme execution_price_timestamp for session {intended_session!r}: {exc} "
            "— fyll-event avvist fail-closed"
        ) from exc

    return write_fill_event(
        order_id=order["order_id"],
        trade_id=make_trade_id(order["order_id"]),
        signal_id=order.get("signal_id"),
        signal_run_id=order.get("signal_run_id"),
        portfolio_id=order.get("portfolio_id", ""),
        portfolio_version=order.get("portfolio_version", ""),
        strategy=order.get("strategy", ""),
        ticker=trade.get("ticker", ""),
        action=order["action"],  # authoritative: PYRAMID_FILL must not be overridden by trade
        intended_execution_session=intended_session,
        actual_execution_session=actual_session,
        shares=float(trade.get("shares", 0)),
        execution_price=price,
        execution_price_source=trade.get("execution_price_source", "next_session_daily_open_v1"),
        execution_price_timestamp=execution_price_timestamp,
        gross_execution_price=trade.get("gross_execution_price", price),
        gross_execution_value=trade.get("gross_execution_value"),
        total_execution_cost=trade.get("total_execution_cost"),
        net_cash_effect=trade.get("net_cash_effect"),
        slippage_bps=trade.get("slippage_bps", 0),
        slippage_amount=float(trade.get("slippage_amount", 0.0)),
        commission_amount=commission,
        reason=trade.get("reason", ""),
        execution_version=execution_version,
        cash_before=cash_before,
        cash_after=cash_after,
    )


def _annotate_trade(trade, order_id, orders, signal_run_id, execution_version):
    """Add traceability fields to a trade dict and persist the order as SETTLING.

    SETTLING is an intermediate status: trade filled, portfolio not yet persisted.
    The caller must call save_order(order, status=EXECUTED) after save_strategy_state()
    to maintain crash-consistent ordering: EXECUTED only after portfolio is on disk.

    trade_id is deterministic (no timestamp) — enables idempotent crash reconciliation.
    """
    trade_id = make_trade_id(order_id)
    trade["order_id"] = order_id
    trade["trade_id"] = trade_id
    # Full traceability chain: signal_run_id → order_id → trade_id
    trade["signal_run_id"] = signal_run_id
    trade["execution_version"] = execution_version
    # Explicit execution cost fields (zero in paper model — model assumption, not omission)
    trade["gross_execution_price"] = trade["price"]
    trade["commission_amount"] = trade["cost"]
    order = orders.get(order_id, {})
    updated = save_order(order, status=SETTLING, trade_id=trade_id)
    orders[order_id] = updated
    return trade_id


def run_strategy_execution(
    strategy_name, signal, trades_df, valuation_price_cache, execution_price_cache,
    orders, signal_run_id, session_date, execution_version,
    signal_id=None,
    premarket_flags=None, macro_mult=1.0, corr_pairs=None,
    market_open=True,
    allow_new_buys: bool = True,
    allow_pyramid: bool = True,
    allow_signal_sells: bool = True,
):
    if premarket_flags is None:
        premarket_flags = {}
    if corr_pairs is None:
        corr_pairs = []
    config = STRATEGIES[strategy_name]
    state = load_strategy_state(strategy_name)
    state = reset_week_if_needed(state)

    # Reconcile SETTLING orders — raises RuntimeError if fill ledger is unreadable (fail-closed)
    reconciled = reconcile_settling_orders(orders, strategy_name, state)
    if reconciled:
        ex = sum(1 for r in reconciled if r["status"] == EXECUTED)
        pp = sum(1 for r in reconciled if r["status"] == PENDING_PRICE)
        print(f"  [{strategy_name}] Crash-recovery: {ex} EXECUTED, {pp} PENDING_PRICE")

    strategy_payload = signal["strategies"].get(strategy_name, {})
    candidates = strategy_payload.get("candidates", [])
    candidates_by_ticker = {c["ticker"]: c for c in candidates}

    regime_name = signal.get("regime", {}).get("regime", "unknown")

    # Valuation before any trades (uses valuation cache — fallback to signal price allowed)
    total_before, _, state = current_portfolio_value(state, candidates_by_ticker, valuation_price_cache)

    # Portfolio drawdown from peak
    drawdown = get_portfolio_drawdown(state, total_before)
    dd_msg = drawdown_protection_message(drawdown)
    if dd_msg:
        print(f"  [{strategy_name}] {dd_msg}")

    sells = []
    buys = []
    correlation_log = []
    orders_created = 0
    orders_executed = 0
    orders_pending_price = 0
    orders_failed_price = 0
    buy_orders_created = 0
    buy_orders_executed = 0
    sell_orders_created = 0
    sell_orders_executed = 0
    # Tracks (order_id, fill_attempt_id, filling_content_hash) for each SETTLING fill
    settling_fills: list[tuple[str, str, str]] = []
    # Capture pre-fill hash (before any trades mutate state) for commit_intent pre/post recovery
    from modules.fills import compute_portfolio_state_hash as _compute_hash  # noqa: PLC0415
    _pre_fill_hash = _compute_hash(state)
    portfolio_id = state.get("portfolio_id", strategy_name)
    portfolio_version = PORTFOLIO_VERSION  # schema version from versioning.py, not created_at

    # --- Retry pending orders from today's session (same-day retry) ---
    for order in get_pending_for_session(orders, session_date, strategy_name):
        ticker = order["ticker"]
        if order["action"] == "SELL" and ticker in state["positions"]:
            cash_before = state["cash"]
            state, trades_df, trade = execute_sell(
                state, trades_df, strategy_name, ticker, order["reason"],
                execution_price_cache,
            )
            if trade:
                _fill_rec = _write_fill_event_for_order(order, trade, cash_before, state["cash"], execution_version)
                _annotate_trade(trade, order["order_id"], orders, signal_run_id, execution_version)
                settling_fills.append((order["order_id"], _fill_rec["fill_attempt_id"], _fill_rec["content_hash"]))
                sells.append(trade)
                orders_executed += 1
                sell_orders_executed += 1
                print(f"  [{strategy_name}] RETRY-SELG {ticker}: {order['reason']}")
            else:
                updated = save_order(
                    orders[order["order_id"]],
                    status=PENDING_PRICE,
                    failure_reason="no execution price for session (retry attempt)",
                )
                orders[order["order_id"]] = updated
                orders_pending_price += 1
        elif order["action"] == "BUY" and ticker not in state["positions"]:
            candidate = candidates_by_ticker.get(ticker)
            if candidate:
                cash_before = state["cash"]
                state, trades_df, trade = execute_buy(
                    state, trades_df, strategy_name, ticker,
                    order["target_value"], order["reason"], candidate,
                    execution_price_cache,
                    premarket_move=premarket_flags.get(ticker),
                    pyramid_remaining=order.get("pyramid_remaining", 0.0),
                )
                if trade:
                    _fill_rec = _write_fill_event_for_order(order, trade, cash_before, state["cash"], execution_version)
                    _annotate_trade(trade, order["order_id"], orders, signal_run_id, execution_version)
                    settling_fills.append((order["order_id"], _fill_rec["fill_attempt_id"], _fill_rec["content_hash"]))
                    buys.append(trade)
                    orders_executed += 1
                    buy_orders_executed += 1
                    print(f"  [{strategy_name}] RETRY-KJØP {ticker}: {order['reason']}")
                else:
                    updated = save_order(
                        orders[order["order_id"]],
                        status=PENDING_PRICE,
                        failure_reason="no execution price for session (retry attempt)",
                    )
                    orders[order["order_id"]] = updated
                    orders_pending_price += 1

    # --- Helper: get or create order and mark outcome ---
    def _sell_with_order(ticker, reason, signal_price=None, action_origin="signal"):
        nonlocal orders_created, orders_executed, orders_pending_price
        nonlocal sell_orders_created, sell_orders_executed
        # Portfolio-safety sells (stop-loss, drawdown protection) have no signal_id.
        # Signal-based sells use the candidate's ledger signal_id if available.
        sell_signal_id = (
            None if action_origin == "portfolio_safety"
            else candidates_by_ticker.get(ticker, {}).get("signal_id")
        )
        order, is_new = get_or_create_order(
            orders, signal_run_id, ticker, strategy_name, session_date, "SELL",
            target_value=0.0, reason=reason, signal_price=signal_price,
            execution_version=execution_version,
            portfolio_id=portfolio_id, portfolio_version=portfolio_version,
            signal_id=sell_signal_id, action_origin=action_origin,
        )
        if order["status"] in TERMINAL:
            return None   # already handled
        if is_new:
            # Persist PENDING_PRICE creation event BEFORE the fill attempt so the
            # full lifecycle (created → executed/pending) is always in the JSONL.
            saved = save_order(order)
            orders[order["order_id"]] = saved
            orders_created += 1
            sell_orders_created += 1
        cash_before = state["cash"]
        state2, trades2, trade = execute_sell(
            state, trades_df, strategy_name, ticker, reason, execution_price_cache
        )
        if trade:
            _fill_rec = _write_fill_event_for_order(order, trade, cash_before, state2["cash"], execution_version)
            _annotate_trade(trade, order["order_id"], orders, signal_run_id, execution_version)
            settling_fills.append((order["order_id"], _fill_rec["fill_attempt_id"], _fill_rec["content_hash"]))
            orders_executed += 1
            sell_orders_executed += 1
        else:
            updated = save_order(
                orders[order["order_id"]],
                status=PENDING_PRICE,
                failure_reason="no execution price for session",
            )
            orders[order["order_id"]] = updated
            orders_pending_price += 1
        return trade, state2, trades2

    # Correlation-based sells: require valid signal (corr_pairs come from signal)
    if allow_signal_sells:
        corr_sell_candidates = filter_correlated_sells(
            state["positions"], candidates_by_ticker, corr_pairs
        )
        for cs in corr_sell_candidates:
            ticker = cs["sell_ticker"]
            if ticker not in state["positions"]:
                continue
            pos = state["positions"][ticker]
            result = _sell_with_order(ticker, cs["reason"], signal_price=pos.get("last_price"))
            if result is None:
                continue
            trade, state, trades_df = result
            if trade:
                sells.append(trade)
                log_msg = (
                    f"❌ {cs['sell_ticker']} solgt — "
                    f"{cs['correlation']:.0%} korrelert med {cs['keep_ticker']}"
                )
                correlation_log.append(log_msg)
                print(f"  [{strategy_name}] KORR-SELG {ticker}: {cs['reason']}")

    # Evaluate normal sells — protective stops always run; signal-based exits require valid signal
    _PROTECTIVE = ("Stop-loss", "Trailing stop")
    for ticker, pos in list(state["positions"].items()):
        candidate = candidates_by_ticker.get(ticker)
        # Inject current session price so stop-loss uses today's open, not stale signal price.
        # execution_price_cache is the authoritative fill price; valuation_price_cache as fallback.
        current_price = execution_price_cache.get(ticker) or valuation_price_cache.get(ticker)
        if current_price is not None:
            candidate = dict(candidate) if candidate else {"ticker": ticker}
            candidate["price"] = current_price
        sell, reason = should_sell_position(ticker, pos, candidate, config)
        if not sell:
            continue
        is_protective = any(reason.startswith(p) for p in _PROTECTIVE)
        if not market_open and not is_protective:
            print(f"  [{strategy_name}] SKIP SELG {ticker}: markedet stengt ({reason})")
            continue
        if not allow_signal_sells and not is_protective:
            print(f"  [{strategy_name}] SKIP SELG {ticker}: signal ugyldig, kun stop-loss tillatt ({reason})")
            continue
        _origin = "portfolio_safety" if is_protective else "signal"
        result = _sell_with_order(ticker, reason, signal_price=pos.get("last_price"),
                                  action_origin=_origin)
        if result is None:
            continue
        trade, state, trades_df = result
        if trade:
            sells.append(trade)
            print(f"  [{strategy_name}] SELG {ticker}: {reason}")

    # Re-valuate after sells (valuation cache)
    total_now, _, state = current_portfolio_value(state, candidates_by_ticker, valuation_price_cache)
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

    # Pyramid fills — blocked when signal is invalid (require fresh ranking to confirm thesis)
    pyramid_fills = []
    if allow_pyramid:
        for ticker in list(state["positions"].keys()):
            pos = state["positions"].get(ticker, {})
            if not pos.get("is_partial"):
                continue  # not a partial position — no pyramid pending

            pyr_remaining_val = float(pos.get("pyramid_remaining_value", 0.0))

            # Idempotency check FIRST — before any portfolio mutation
            pyr_signal_id = candidates_by_ticker.get(ticker, {}).get("signal_id")
            pyr_order, pyr_is_new = get_or_create_order(
                orders, signal_run_id, ticker, strategy_name, session_date,
                "PYRAMID_FILL",
                target_value=pyr_remaining_val,
                reason=f"Pyramid fill: {ticker}",
                signal_price=pos.get("last_price"),
                execution_version=execution_version,
                portfolio_id=portfolio_id, portfolio_version=portfolio_version,
                signal_id=pyr_signal_id,
            )
            # Terminal or in-flight orders must never trigger another fill
            if pyr_order["status"] in TERMINAL or pyr_order["status"] == SETTLING:
                continue

            if pyr_is_new:
                saved = save_order(pyr_order)
                orders[pyr_order["order_id"]] = saved
                orders_created += 1

            # NOW execute portfolio mutation — order ledger already committed
            pyr_cash_before = state["cash"]
            state, trades_df, fill = execute_pyramid_fill(
                state, trades_df, strategy_name, ticker, execution_price_cache
            )
            if fill:
                _fill_rec = _write_fill_event_for_order(pyr_order, fill, pyr_cash_before, state["cash"], execution_version)
                _annotate_trade(fill, pyr_order["order_id"], orders, signal_run_id, execution_version)
                settling_fills.append((pyr_order["order_id"], _fill_rec["fill_attempt_id"], _fill_rec["content_hash"]))
                pyramid_fills.append(fill)
                orders_executed += 1
                print(f"  [{strategy_name}] PYRAMID {ticker}: {fill['reason']}")
            else:
                # Execution price unavailable or fill conditions not met — keep as PENDING_PRICE
                updated = save_order(
                    orders[pyr_order["order_id"]],
                    status=PENDING_PRICE,
                    failure_reason="pyramid fill conditions not met or no execution price",
                )
                orders[pyr_order["order_id"]] = updated
                orders_pending_price += 1
    elif list(state["positions"].values()):
        print(f"  [{strategy_name}] PYRAMID blokkert: signal ugyldig")

    # Count sector positions for correlation filter
    def _sector_counts():
        counts = {}
        for t in state["positions"]:
            sector = candidates_by_ticker.get(t, {}).get("sector", "Unknown")
            counts[sector] = counts.get(sector, 0) + 1
        return counts

    # Evaluate new buys
    if not market_open:
        print(f"  [{strategy_name}] Markedet er stengt, ingen kjøp utføres")
    if not allow_new_buys:
        print(f"  [{strategy_name}] Kjøp blokkert: signal ugyldig eller data-kvalitet for lav")

    if allow_new_buys and can_buy and market_open:
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

            # Sector filter: max 2 positions per sector per strategy
            sector = candidate.get("sector", "Unknown")
            sector_counts = _sector_counts()
            if sector_counts.get(sector, 0) >= 2:
                print(f"  [{strategy_name}] SKIP {ticker}: sektorbegrensning {sector} ({sector_counts.get(sector)}≥2)")
                continue

            held_set = set(state["positions"].keys())

            # Sub-sector concentration filter: max 3 per sub-sector
            is_concentrated, sub_sector, sub_count = check_sub_sector_concentration(
                ticker, held_set
            )
            if is_concentrated:
                sub_label = SUB_SECTOR_LABELS.get(sub_sector, sub_sector)
                log_msg = f"❌ {ticker} ekskludert — {sub_count} {sub_label} allerede"
                correlation_log.append(log_msg)
                print(f"  [{strategy_name}] SKIP {ticker}: {sub_count} {sub_label} allerede")
                continue

            # Correlation filter: skip if >75% correlated with an existing position
            is_corr, corr_with, corr_val = check_correlation_against_held(
                ticker, held_set, corr_pairs
            )
            if is_corr:
                log_msg = f"❌ {ticker} ekskludert — {corr_val:.0%} korrelert med {corr_with}"
                correlation_log.append(log_msg)
                print(f"  [{strategy_name}] SKIP {ticker}: {log_msg}")
                continue

            target_value_full = total_now * target_weights[ticker]
            # Pyramid: buy 60% now, queue 40% for fill after 5 days + >2% gain
            initial_value = target_value_full * 0.60
            pyramid_remaining = target_value_full * 0.40
            reason = f"Rank #{candidate.get('rank')}, score {safe_round(candidate.get('strategy_score'), 1)}"

            pm_move = premarket_flags.get(ticker)
            signal_price = candidate.get("price")

            buy_signal_id = candidate.get("signal_id")  # ledger's authoritative signal_id
            order, is_new = get_or_create_order(
                orders, signal_run_id, ticker, strategy_name, session_date, "BUY",
                target_value=initial_value, reason=reason, signal_price=signal_price,
                execution_version=execution_version, pyramid_remaining=pyramid_remaining,
                portfolio_id=portfolio_id, portfolio_version=portfolio_version,
                signal_id=buy_signal_id,
            )
            if order["status"] in TERMINAL:
                continue   # already executed or expired in a previous run

            if is_new:
                # Persist PENDING_PRICE creation event BEFORE the fill attempt so the
                # full lifecycle (created → executed/pending) is always in the JSONL.
                saved = save_order(order)
                orders[order["order_id"]] = saved
                orders_created += 1
                buy_orders_created += 1

            buy_cash_before = state["cash"]
            state, trades_df, trade = execute_buy(
                state, trades_df, strategy_name, ticker, initial_value,
                reason, candidate, execution_price_cache,
                premarket_move=pm_move,
                pyramid_remaining=pyramid_remaining,
            )

            if trade:
                _fill_rec = _write_fill_event_for_order(order, trade, buy_cash_before, state["cash"], execution_version)
                _annotate_trade(trade, order["order_id"], orders, signal_run_id, execution_version)
                settling_fills.append((order["order_id"], _fill_rec["fill_attempt_id"], _fill_rec["content_hash"]))
                buys.append(trade)
                orders_executed += 1
                buy_orders_executed += 1
                print(f"  [{strategy_name}] KJØP 60% {ticker}: {trade['reason']}")
            else:
                updated = save_order(
                    orders[order["order_id"]],
                    status=PENDING_PRICE,
                    failure_reason="no execution price for session",
                )
                orders[order["order_id"]] = updated
                orders_pending_price += 1

    # Build sector map for held positions (for reporting)
    sector_map = {}
    for ticker in state["positions"]:
        sector_map[ticker] = candidates_by_ticker.get(ticker, {}).get("sector", "Unknown")

    # Final valuation (valuation cache)
    total_after, positions_value_after, state = current_portfolio_value(
        state, candidates_by_ticker, valuation_price_cache
    )

    state["highest_portfolio_value"] = max(
        float(state.get("highest_portfolio_value", START_CAPITAL)),
        total_after,
    )
    state["last_execution_date"] = today_str()

    # Compute portfolio state hash before save — verified after atomic write
    from modules.fills import (  # noqa: PLC0415
        compute_portfolio_state_hash,
        mark_fill_persisted,
        write_commit_intent,
    )
    pre_save_hash = compute_portfolio_state_hash(state)

    # Write commit_intent BEFORE portfolio save — proves what we intended to persist.
    # On crash recovery: if portfolio hash matches this intent → save completed.
    portfolio_id = state.get("portfolio_id", strategy_name)
    portfolio_version = state.get("portfolio_version", "")
    fills_for_intent = [
        {"order_id": oid, "fill_attempt_id": fa_id, "filling_content_hash": ch}
        for oid, fa_id, ch in settling_fills
    ]
    commit_intent_rec = None
    if fills_for_intent:
        try:
            commit_intent_rec = write_commit_intent(
                strategy=strategy_name,
                portfolio_id=portfolio_id,
                portfolio_version=portfolio_version,
                pre_portfolio_state_hash=_pre_fill_hash,
                post_portfolio_state_hash=pre_save_hash,
                fills=fills_for_intent,
            )
        except RuntimeError as exc:
            send_telegram(
                f"⚠️ KRITISK: commit_intent WAL feilet for {strategy_name}: {exc} "
                "— kjøring avbrutt"
            )
            raise

    save_strategy_state(strategy_name, state)

    # Verify: reload portfolio from disk and confirm hash matches in-memory state
    post_save_state = load_strategy_state(strategy_name)
    post_save_hash = compute_portfolio_state_hash(post_save_state)
    if post_save_hash != pre_save_hash:
        send_telegram(
            f"⚠️ KRITISK: Portfolio state hash mismatch etter save for {strategy_name}: "
            f"forventet {pre_save_hash!r}, faktisk {post_save_hash!r} — kjøring avbrutt"
        )
        raise RuntimeError(
            f"Portfolio state hash mismatch for {strategy_name}: "
            f"pre={pre_save_hash!r} post={post_save_hash!r}"
        )

    # Mark fills as persisted in WAL + finalize SETTLING → EXECUTED (fail-closed per fill)
    # EXECUTED is only written if mark_fill_persisted succeeds and hash matches.
    # A WAL write failure leaves the order as SETTLING for reconcile_settling_orders on restart.
    _commit_id = commit_intent_rec.get("commit_id") if commit_intent_rec else None
    for oid, fa_id, ch in settling_fills:
        if oid not in orders or orders[oid].get("status") != SETTLING:
            continue
        try:
            mark_fill_persisted(
                oid, fa_id, ch,
                post_portfolio_state_hash=pre_save_hash,
                commit_id=_commit_id,
            )
        except RuntimeError as exc:
            send_telegram(
                f"⚠️ KRITISK: Fill-WAL feilet for {oid}: {exc} "
                "— ordre forblir SETTLING, kjøring avbrutt"
            )
            raise  # fail-closed: order stays SETTLING, next run reconciles
        updated = save_order(orders[oid], status=EXECUTED)
        orders[oid] = updated

    return_pct = (total_after / START_CAPITAL) - 1

    # Pre-market flags relevant to this strategy (held positions + new buys)
    relevant_tickers = set(state["positions"].keys()) | {t["ticker"] for t in buys}
    strategy_pm_flags = {t: m for t, m in premarket_flags.items() if t in relevant_tickers}

    buy_fill_rate = (buy_orders_executed / buy_orders_created) if buy_orders_created > 0 else None
    sell_fill_rate = (sell_orders_executed / sell_orders_created) if sell_orders_created > 0 else None
    fill_rate = (orders_executed / orders_created) if orders_created > 0 else None

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
        "correlation_log": correlation_log,
        "candidates_count": len(candidates),
        "orders_created": orders_created,
        "orders_executed": orders_executed,
        "orders_pending_price": orders_pending_price,
        "orders_failed_price": orders_failed_price,
        "buy_orders_created": buy_orders_created,
        "buy_orders_executed": buy_orders_executed,
        "sell_orders_created": sell_orders_created,
        "sell_orders_executed": sell_orders_executed,
        "buy_fill_rate": buy_fill_rate,
        "sell_fill_rate": sell_fill_rate,
        "fill_rate": fill_rate,
    }, trades_df


# ============================================================
# EXECUTE MODE — orchestrator
# ============================================================

def run_execute():
    print("=" * 60)
    print("EXECUTE MODE")
    print("=" * 60)

    # Try to load signal — failure allows protective sells but blocks new buys/pyramids
    signal = None
    signal_path = None
    signal_load_error = None
    try:
        signal, signal_path = load_latest_signal()
    except (FileNotFoundError, SignalNotPublishedError) as exc:
        signal_load_error = str(exc)
        print(f"Signal ikke tilgjengelig: {exc}")

    # Determine session_date via NYSE calendar.
    # CalendarUnavailableError = calendar cannot be validated (fail-closed, alert + raise).
    # is_trading_session() returning False = known holiday/weekend (graceful SKIPPED exit).
    from modules.exchange_calendar import CalendarUnavailableError, is_trading_session  # noqa: PLC0415
    ny_date = now_ny().strftime("%Y-%m-%d")
    try:
        trading_today = is_trading_session(ny_date)
    except CalendarUnavailableError as exc:
        send_telegram(f"⚠️ KRITISK: NYSE-kalender utilgjengelig — execute avbrutt: {exc}")
        raise

    if not trading_today:
        print(
            f"{ny_date} er ikke en NYSE-handelsdag (helligdag/helg) "
            "— ingen kjøring (SKIPPED_NON_TRADING_SESSION)"
        )
        return

    session_date = ny_date

    if signal is not None:
        print(f"Bruker signal: {signal_path}")
        from modules.signal_validator import validate_signal  # noqa: PLC0415
        signal_validation = validate_signal(signal, signal_path, session_date)
        print(f"Signal-validering: {signal_validation.failure_mode} — {signal_validation.reason}")
        if not signal_validation.is_valid or not signal_validation.allow_new_buys:
            from modules.reporting import build_stale_signal_message  # noqa: PLC0415
            send_telegram(build_stale_signal_message(signal_validation))
            if signal_validation.failure_mode == "calendar_unavailable":
                print("⚠️ KRITISK: Exchange calendar utilgjengelig — execution fail-closed, ingen nye kjøp")
    else:
        # No signal available — run protective sells only
        from modules.signal_validator import SignalValidationResult  # noqa: PLC0415
        signal_validation = SignalValidationResult(
            is_valid=False,
            failure_mode="unpublished",
            reason=signal_load_error or "Signal ikke tilgjengelig",
            allow_new_buys=False,
            allow_pyramid=False,
            allow_protective_sells=True,
            signal_run_id=None,
            intended_session=None,
            actual_session=session_date,
            generated_at=None,
            published_at=None,
            published_commit_sha=None,
            model_version=None,
            data_cutoff_at=None,
            data_quality_tier="unknown",
            data_quality={},
        )
        from modules.reporting import build_stale_signal_message  # noqa: PLC0415
        send_telegram(build_stale_signal_message(signal_validation))
        print("⚠️ Ingen signal — kun beskyttende stop-loss/trailing-stop kjøres")
        signal = {
            "strategies": {name: {"candidates": []} for name in STRATEGIES},
            "regime": {},
            "corr_pairs": [],
            "active_weights": None,
            "sentiment_scores": {},
            "quality_report": {},
            "signal_run_id": f"safety-{session_date}",
            "signal_id": None,       # no real signal — nullable for safety-action orders
            "action_origin": "portfolio_safety",
            "earnings_analysis": {},
        }
        signal_path = "no-signal"

    trades_df = load_trades()
    performance_df = load_performance()

    # WAL trade recovery: project any persisted fill events into trades_df
    # Idempotent — skips trade_ids already present in trades.csv
    # Fail-closed: any WAL failure stops execution before any state mutation.
    try:
        from modules.fills import project_fills_to_trades  # noqa: PLC0415
        trades_df = project_fills_to_trades(trades_df)
    except RuntimeError as exc:
        send_telegram(f"⚠️ WAL-gjenoppbygging feilet — kjøring avbrutt: {exc}")
        raise  # fail-closed: stop before any portfolio mutation

    # Macro filter (FRED 10Y/2Y yield spread, no API key)
    macro = get_macro_status()
    macro_mult = macro.get("exposure_mult", 1.0)

    # Load precomputed correlation pairs, active weights, sentiment and quality from signal
    corr_pairs = signal.get("corr_pairs", [])
    active_weights = signal.get("active_weights")
    sentiment_scores = signal.get("sentiment_scores", {})
    quality_report = signal.get("quality_report", {})

    # Apply split adjustments before running strategies
    split_detected = quality_report.get("split_detected", [])
    if split_detected:
        print(f"Mulige splits oppdaget: {[s['ticker'] for s in split_detected]}")
        apply_split_adjustments(split_detected)

    # --- MOO execution model: collect all tickers + yesterday's closes ---
    # fallback_prices = yesterday's close from signal (open price fallback + premarket reference)
    fallback_prices = {}
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
                if t not in fallback_prices and c.get("price"):
                    fallback_prices[t] = c["price"]
        for t, pos in held.items():
            if t not in fallback_prices:
                fallback_prices[t] = pos.get("last_price") or pos.get("avg_price")

    all_tickers = pm_tickers | {"SPY", "QQQ"}

    # session_date and signal_validation set above (before price prefetch)

    # Execution cache: session-validated open prices — no fallback, no fill if missing
    execution_price_cache = prefetch_execution_prices(list(pm_tickers), session_date)

    # Valuation cache: open prices with signal-price fallback — used for MTM, drawdown, benchmarks
    valuation_price_cache = prefetch_open_prices(list(all_tickers), fallback_prices)

    benchmark_state = update_benchmarks(valuation_price_cache)
    spy_ret = benchmark_return(benchmark_state, "SPY")
    qqq_ret = benchmark_return(benchmark_state, "QQQ")

    print(f"SPY siden start: {(spy_ret or 0)*100:.2f}%  |  QQQ: {(qqq_ret or 0)*100:.2f}%")

    # --- Pre-market filter ---
    # Compare yesterday's close (fallback_prices) vs pre-market price to detect gaps
    all_pm_moves = compute_premarket_moves(list(pm_tickers), fallback_prices)
    premarket_flags = {t: m for t, m in all_pm_moves.items() if abs(m) > 0.04}

    market_open = is_market_open()
    if not market_open:
        print("Markedet er stengt — stop-loss kan fortsatt utføres, kjøp hoppes over")

    # --- Order state: load, crash-recovery, expire stale, share across all strategies ---
    signal_run_id = signal.get("signal_run_id", "unknown")
    # signal_id: stable content-addressed identifier (content hash for real signals, None for safety)
    signal_id = signal.get("signal_id")
    if signal_id is None:
        signal_id = signal.get("signal_content_hash")  # real signal has this field
    orders = load_orders()
    # SETTLING orders are reconciled per-strategy inside run_strategy_execution, which checks
    # portfolio state to distinguish crash-before-save (FAILED_PRICE) from crash-after-save (EXECUTED).
    expired = expire_stale_orders(orders, session_date)
    if expired:
        print(f"Utløpte ordre fra tidligere sesjoner: {len(expired)}")

    results = []
    drawdown_warnings = []

    for strategy_name in STRATEGIES.keys():
        print(f"\nKjører {strategy_name}...")
        result, trades_df = run_strategy_execution(
            strategy_name, signal, trades_df, valuation_price_cache, execution_price_cache,
            orders=orders,
            signal_run_id=signal_run_id,
            signal_id=signal_id,
            session_date=session_date,
            execution_version=EXECUTION_VERSION,
            premarket_flags=premarket_flags,
            macro_mult=macro_mult,
            corr_pairs=corr_pairs,
            market_open=market_open,
            allow_new_buys=signal_validation.allow_new_buys,
            allow_pyramid=signal_validation.allow_pyramid,
            allow_signal_sells=signal_validation.is_valid,
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

    earnings_analysis = signal.get("earnings_analysis", {})

    # Layer 2: Weekly Sonnet thesis review — kun mandager
    weekly_ai_cost = 0.0
    weekly_analyses = {}
    if is_monday():
        held_positions = {}
        for strat_name in STRATEGIES:
            strat_state = load_strategy_state(strat_name)
            for ticker, pos in strat_state.get("positions", {}).items():
                if ticker not in held_positions:
                    held_positions[ticker] = pos
        if held_positions:
            weekly_analyses, weekly_ai_cost = fetch_weekly_analysis(held_positions)

    # Total AI cost this run (signal cost from yesterday's signal + today's execute costs)
    signal_ai_cost = safe_float(signal.get("signal_ai_cost_usd"), 0.0)
    total_ai_cost = signal_ai_cost + weekly_ai_cost

    message = build_full_message(
        results=results,
        signal=signal,
        signal_path=signal_path,
        spy_ret=spy_ret,
        qqq_ret=qqq_ret,
        drawdown_warnings=drawdown_warnings,
        fundamentals_cache=fundamentals_cache,
        macro=macro,
        earnings_analysis=earnings_analysis,
        active_weights=active_weights,
        corr_pairs=corr_pairs,
        sentiment_scores=sentiment_scores,
        quality_report=quality_report,
        weekly_analyses=weekly_analyses,
        ai_cost_usd=total_ai_cost,
        signal_validation=signal_validation,
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
    elif mode == "finalize_signal":
        run_finalize_signal()
    elif mode == "ack_publication":
        run_ack_publication()
    elif mode == "execute":
        run_execute()
    elif mode == "premarket":
        run_premarket()
    elif mode == "backtest":
        from modules.backtest import run_backtest
        run_backtest()
    else:
        raise ValueError(f"Ukjent BOT_MODE: {mode}")


if __name__ == "__main__":
    main()
