import os

import requests

from modules.state import (
    PERFORMANCE_FILE,
    START_CAPITAL,
    load_performance,
    now_oslo,
    safe_float,
    safe_round,
    today_str,
)

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

REGIME_LABELS = {
    "explosive": "EKSPLOSIVT 🚀",
    "bullish": "BULL 📈",
    "neutral": "NØYTRALT ➡️",
    "defensive": "BEAR 🛡️",
    "unknown": "UKJENT ❓",
}


def format_pct(x, decimals=2):
    v = safe_float(x)
    if v is None:
        return "N/A"
    sign = "+" if v > 0 else ""
    return f"{sign}{round(v * 100, decimals)}%"


def format_pct_pts(x, decimals=2):
    v = safe_float(x)
    if v is None:
        return "N/A"
    sign = "+" if v > 0 else ""
    return f"{sign}{round(v * 100, decimals)}pp"


def is_monday():
    return now_oslo().weekday() == 0


def is_first_monday_of_month():
    today = now_oslo()
    return today.weekday() == 0 and today.day <= 7


def split_message(text, max_length=3800):
    lines = text.split("\n")
    chunks, current = [], ""
    for line in lines:
        if len(current) + len(line) + 1 > max_length:
            chunks.append(current)
            current = line
        else:
            current += ("\n" + line) if current else line
    if current:
        chunks.append(current)
    return chunks


def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram secrets mangler — skriver melding til stdout:\n")
        print("─" * 60)
        print(message)
        print("─" * 60)
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for chunk in split_message(message):
        try:
            resp = requests.post(url, json={"chat_id": CHAT_ID, "text": chunk}, timeout=30)
            if resp.status_code == 200:
                print("Melding sendt til Telegram ✅")
            else:
                print(f"Telegram-feil: {resp.text}")
        except Exception as e:
            print(f"Telegram exception: {e}")


# ============================================================
# DAILY REPORT
# ============================================================

def _build_leaderboard(results, spy_ret, qqq_ret):
    lines = ["🏁 LEADERBOARD (avkastning siden start)"]
    sorted_results = sorted(results, key=lambda r: r["return_pct"], reverse=True)

    for i, r in enumerate(sorted_results, 1):
        alpha = (r["return_pct"] - spy_ret) if spy_ret is not None else None
        alpha_str = f" | α {format_pct(alpha)}" if alpha is not None else ""
        lines.append(
            f"{i}. {r['strategy']}: {format_pct(r['return_pct'])}"
            f"{alpha_str} | {r['num_positions']}pos | ${safe_round(r['cash'], 0)} cash"
        )

    lines.append(f"SPY: {format_pct(spy_ret)}  |  QQQ: {format_pct(qqq_ret)}")
    return "\n".join(lines)


def _build_data_quality_section(quality_report):
    """Format data quality alerts. Only shown if issues exist."""
    if not quality_report:
        return ""
    price_flags = quality_report.get("price_flags", [])
    volume_excluded = quality_report.get("volume_excluded", [])
    split_detected = quality_report.get("split_detected", [])
    total = len(price_flags) + len(volume_excluded) + len(split_detected)
    if total == 0:
        return ""

    lines = [f"\n⚠️ DATAKVALITET: {total} aksjer flagget i dag"]
    for f in price_flags:
        sign = "+" if f["pct_change"] > 0 else ""
        lines.append(f"  {f['ticker']}: {sign}{f['pct_change']:.0%} (mistenkelig)")
    for v in volume_excluded:
        lines.append(f"  {v['ticker']}: 0 volum {v['zero_days']} dager")
    for s in split_detected:
        ratio_str = f" — {s['ratio']}:1 split?" if s.get("ratio") else ""
        lines.append(f"  {s['ticker']}: {s['pct_drop']:.0%}{ratio_str}")
    return "\n".join(lines)


def _build_sentiment_summary(sentiment_scores, held_tickers=None):
    """Format sentiment line: 🤖 Sentiment: NVDA +0.7 (Claude) | TSM +0.4 (Claude) | ARM 0.5 (fallback)"""
    if not sentiment_scores:
        return ""

    if held_tickers is None:
        held_tickers = set()

    held, others = [], []
    for ticker, data in sentiment_scores.items():
        score = safe_float(data.get("score"), 0.5)
        source = data.get("source", "fallback")
        entry = (ticker, score, source)
        (held if ticker in held_tickers else others).append(entry)

    held.sort(key=lambda x: -x[1])
    others.sort(key=lambda x: -x[1])
    shown = (held + others)[:12]

    claude_count = sum(1 for _, _, s in shown if s == "claude")

    if claude_count == 0:
        return "🤖 Sentiment: ingen API-nøkkel — alle fallback (0.5)"

    parts = []
    for ticker, score, source in shown:
        sign = "+" if score > 0.5 else ""
        label = "Claude" if source == "claude" else "fallback"
        parts.append(f"{ticker} {sign}{score:.1f} ({label})")

    return "🤖 Sentiment: " + " | ".join(parts)


def _build_correlation_alerts(results):
    """Collect unique correlation/concentration exclusion messages across all strategies."""
    seen = set()
    lines = []
    for r in results:
        for msg in r.get("correlation_log", []):
            if msg not in seen:
                seen.add(msg)
                lines.append(f"  {msg}")
    if not lines:
        return ""
    return "\n🚫 FILTRERT UT:\n" + "\n".join(lines)


def _fmt_rate(executed, created):
    """Format fill rate as 'ex/n (pct)' or '0/0 (N/A)' when cohort is empty."""
    if created == 0:
        return "0/0 (N/A)"
    return f"{executed}/{created} ({executed / created:.0%})"


def _build_actions(results):
    lines = ["📌 DAGENS HANDLINGER"]
    any_action = False

    for r in results:
        buys = r.get("buys", [])
        sells = r.get("sells", [])
        if not buys and not sells:
            continue
        any_action = True
        lines.append(f"\n{r['strategy']}:")
        if buys:
            lines.append("  KJØP:")
            for t in buys:
                lines.append(
                    f"  + {t['ticker']} ${t['value']} @ ${t['price']} — {t['reason']}"
                )
        if sells:
            lines.append("  SELG:")
            for t in sells:
                lines.append(
                    f"  - {t['ticker']} ${t['value']} @ ${t['price']} — {t['reason']}"
                )

    if not any_action:
        lines.append("Ingen kjøp/salg i dag.")

    # ── Execution statistics ──────────────────────────────────────────────────
    # Aggregated from the persistent order ledger (via exec_stats in each result).
    # Cohort = orders where intended_execution_session == session_date AND strategy.
    # Each order_id counted once regardless of how many reruns touched it.
    # recommendations = candidates_count per strategy (sum = total, no dedup across strategies).

    def _sum(key):
        return sum(r.get("exec_stats", {}).get(key, 0) for r in results)

    total_n       = _sum("cohort_size")
    total_ex      = _sum("executed")  # terminal EXECUTED only; SETTLING shown separately
    total_pending = _sum("pending_price")
    total_fp      = _sum("failed_price")
    total_fr      = _sum("failed_reconciliation")
    total_expired = _sum("expired")
    total_cancel  = _sum("cancelled")
    total_miss    = _sum("missing_execution_price")
    total_buy_n   = _sum("buy_created")
    total_buy_ex  = _sum("buy_executed")
    total_sell_n  = _sum("sell_created")
    total_sell_ex = _sum("sell_executed")
    total_pyr_n   = _sum("pyramid_created")
    total_pyr_ex  = _sum("pyramid_executed")

    # recommendations: per strategy from signal candidates (sum = cross-strategy total)
    rec_parts = [
        f"{r.get('recommendations', 0)} ({r['strategy']})" for r in results
    ]
    total_rec = sum(r.get("recommendations", 0) for r in results)
    rec_str = " + ".join(rec_parts) + f" = {total_rec} totalt" if len(results) > 1 else str(total_rec)

    # expired_this_run: orders expired by expire_stale_orders() today (may be from prior sessions)
    total_expired_run = sum(r.get("expired_this_run", 0) for r in results)

    status_parts = []
    if total_pending > 0:
        status_parts.append(f"{total_pending} venter pris")
    if total_miss > 0:
        status_parts.append(f"{total_miss} mangler exekveringspris")
    if total_fp > 0:
        status_parts.append(f"{total_fp} feilet (pris)")
    if total_fr > 0:
        status_parts.append(f"{total_fr} feilet (reconcile)")
    if total_expired > 0:
        status_parts.append(f"{total_expired} utløpt")
    if total_cancel > 0:
        status_parts.append(f"{total_cancel} kansellert")
    status_str = "  " + " | ".join(status_parts) if status_parts else ""

    total_settling = _sum("settling")
    total_unclassified = _sum("unclassified_status") + _sum("unclassified_action")
    total_safety = _sum("safety_created")

    # fill_rate = terminal EXECUTED only / cohort_size (SETTLING shown separately)
    fill_rate_str = _fmt_rate(total_ex, total_n)

    stat_lines = [
        f"\n📋 ORDREUTFØRELSE (sesjonskohort)",
        f"  📡 Anbefalinger: {rec_str}",
        f"  Opprettede: {total_n}",
        f"  Fylt (EXECUTED): {fill_rate_str}",
    ]
    if total_settling > 0:
        stat_lines.append(f"  Under filling (SETTLING): {total_settling}")
    if status_str:
        stat_lines.append(status_str)
    if total_safety > 0:
        stat_lines.append(f"  herav sikkerhetssalg: {total_safety}")
    if total_unclassified > 0:
        stat_lines.append(f"  ⚠️ Ukjent status/action: {total_unclassified} (manual_review)")
    stat_lines += [
        f"  KJØP:    {_fmt_rate(total_buy_ex, total_buy_n)}",
        f"  SELG:    {_fmt_rate(total_sell_ex, total_sell_n)}",
        f"  PYRAMID: {_fmt_rate(total_pyr_ex, total_pyr_n)}",
    ]
    if total_expired_run > 0:
        stat_lines.append(
            f"  🗑️ Utløpt i dag (fra tidl. sesjoner): {total_expired_run}"
        )

    lines.extend(stat_lines)

    return "\n".join(lines)


def _build_premarket_section(results):
    """Collect all pre-market flags across strategies and format a section."""
    # Aggregate: ticker -> {move, held_by, candidate_for}
    aggregated = {}
    for r in results:
        strat = r["strategy"]
        held = set(r.get("positions", {}).keys())
        flags = r.get("premarket_flags", {})

        for ticker, move in flags.items():
            if ticker not in aggregated:
                aggregated[ticker] = {"move": move, "held_by": [], "candidate_for": []}
            if ticker in held:
                aggregated[ticker]["held_by"].append(strat)
            else:
                aggregated[ticker]["candidate_for"].append(strat)

    if not aggregated:
        return ""

    lines = ["\n🌅 PRE-MARKET (>4% fra gårsdagens slutt)"]
    for ticker, info in sorted(aggregated.items(), key=lambda x: -abs(x[1]["move"])):
        move = info["move"]
        arrow = "↑" if move > 0 else "↓"
        held = info["held_by"]
        cand = info["candidate_for"]

        parts = []
        if held:
            parts.append(f"holdt: {', '.join(set(held))}")
        if cand:
            parts.append(f"kandidat: {', '.join(set(cand))} → kjøpsstørrelse halvert")

        detail = "  |  ".join(parts) if parts else ""
        lines.append(f"  {ticker}: {move:+.1%} {arrow}  {detail}")

    return "\n".join(lines)


def _build_earnings_alerts(results, earnings_analysis=None):
    if earnings_analysis is None:
        earnings_analysis = {}

    # Collect positions with upcoming earnings — split blackout (≤3d) from alert (4-14d)
    blackout = {}
    alert = {}
    for r in results:
        candidates_by_ticker = {c["ticker"]: c for c in r.get("top_candidates_full", [])}
        for ticker in r.get("positions", {}):
            c = candidates_by_ticker.get(ticker, {})
            days = c.get("days_to_earnings")
            if days is None:
                continue
            date = c.get("next_earnings", "?")
            if days <= 3:
                bucket = blackout
            elif c.get("earnings_soon"):
                bucket = alert
            else:
                continue
            if ticker not in bucket:
                bucket[ticker] = {"date": date, "days": days, "strategies": []}
            bucket[ticker]["strategies"].append(r["strategy"])

    lines = []
    if blackout:
        lines.append("\n🚫 EARNINGS BLACKOUT (≤3 dager — ikke kjøp):")
        for ticker, info in sorted(blackout.items(), key=lambda x: x[1].get("days") or 999):
            strats = ", ".join(set(info["strategies"]))
            lines.append(f"  ⚠️ {ticker}: {info['date']} (om {info['days']}d) — {strats}")
            analysis = earnings_analysis.get(ticker)
            if analysis:
                lines.append(f"    🤖 {analysis.get('recommendation', '')}: {analysis.get('outlook', '')[:120]}")
    if alert:
        lines.append("\n⚠️ EARNINGS SNART (4-14 dager):")
        for ticker, info in sorted(alert.items(), key=lambda x: x[1].get("days") or 999):
            strats = ", ".join(set(info["strategies"]))
            lines.append(f"  {ticker}: {info['date']} (om {info['days']}d) — {strats}")
            analysis = earnings_analysis.get(ticker)
            if analysis:
                risks = " | ".join(analysis.get("risk_factors", [])[:2])
                lines.append(f"    🤖 {analysis.get('recommendation', '')}: {analysis.get('outlook', '')[:120]}")
                if risks:
                    lines.append(f"    Risiko: {risks}")

    return "\n".join(lines) if lines else ""


def _build_positions_summary(results):
    lines = ["\n📊 POSISJONER"]
    sorted_results = sorted(results, key=lambda r: r["return_pct"], reverse=True)

    for r in sorted_results:
        lines.append(f"\n{r['strategy']} (${safe_round(r['portfolio_value'], 0)}):")
        positions = r.get("positions", {})
        sector_map = r.get("sector_map", {})
        if positions:
            holdings = []
            for ticker, pos in sorted(positions.items()):
                avg = pos.get("avg_price", 0)
                last = pos.get("last_price", avg)
                pnl = (last / avg - 1) if avg else 0
                partial_mark = "(P)" if pos.get("is_partial") else ""
                holdings.append(f"{ticker}{partial_mark}({format_pct(pnl, 1)})")
            lines.append("  " + ", ".join(holdings[:15]))

            # Sector exposure — compact one-liner
            sector_exposure = {}
            total_val = r.get("portfolio_value", 1) or 1
            for ticker, pos in positions.items():
                sector = sector_map.get(ticker, "?")
                mv = pos.get("market_value", 0)
                sector_exposure[sector] = sector_exposure.get(sector, 0) + mv
            if sector_exposure:
                sector_str = " | ".join(
                    f"{s[:6]}:{mv/total_val:.0%}"
                    for s, mv in sorted(sector_exposure.items(), key=lambda x: -x[1])
                )
                lines.append(f"  [{sector_str}]")
        else:
            lines.append("  (ingen posisjoner)")

        top = r.get("top_candidates", [])[:3]
        if top:
            parts = [
                f"{c['ticker']} {safe_round(c['strategy_score'], 1)}"
                for c in top
            ]
            lines.append(f"  Topp 3: {', '.join(parts)}")

    return "\n".join(lines)


def build_daily_report(results, signal, signal_path, spy_ret, qqq_ret, drawdown_warnings,
                       macro=None, earnings_analysis=None, active_weights=None,
                       sentiment_scores=None, quality_report=None, ai_cost_usd=0.0):
    regime = signal.get("regime", {})
    regime_name = regime.get("regime", "unknown")
    regime_label = REGIME_LABELS.get(regime_name, regime_name.upper())
    vix = regime.get("vix")
    vix_str = f" | VIX {vix:.1f}" if vix else ""

    lines = [
        f"🧠 AI STOCK LAB v4",
        f"Dato: {today_str()}",
        f"Marked: {regime_label}{vix_str}",
        f"  {regime.get('reason', '')}",
    ]

    # Macro status
    if macro:
        mult = macro.get("exposure_mult", 1.0)
        mult_str = f" → eksponering ×{mult:.0%}" if mult < 1.0 else ""
        lines.append(f"📉 10Y/2Y: {macro.get('status', 'N/A')}{mult_str}")

    # Active factor weights
    if active_weights:
        mom = round(active_weights.get("momentum", 0) * 100)
        qual = round(active_weights.get("quality", 0) * 100)
        val = round(active_weights.get("value", 0) * 100)
        sent = round(active_weights.get("sentiment", 0) * 100)
        lines.append(f"📊 Aktive vekter: Mom {mom}% | Kval {qual}% | Verdi {val}% | Sent {sent}%")

    # Sentiment source summary
    if sentiment_scores is not None:
        held_tickers = set()
        for r in results:
            held_tickers.update(r.get("positions", {}).keys())
        sent_line = _build_sentiment_summary(sentiment_scores, held_tickers)
        if sent_line:
            lines.append(sent_line)

    lines.append("")
    lines.append(_build_leaderboard(results, spy_ret, qqq_ret))
    lines.append("")
    premarket_block = _build_premarket_section(results)
    if premarket_block:
        lines.append(premarket_block)

    lines.append(_build_actions(results))
    corr_block = _build_correlation_alerts(results)
    if corr_block:
        lines.append(corr_block)
    earnings_block = _build_earnings_alerts(results, earnings_analysis=earnings_analysis)
    if earnings_block:
        lines.append(earnings_block)

    if drawdown_warnings:
        lines.append("\n🔴 DRAWDOWN ADVARSLER:")
        for w in drawdown_warnings:
            lines.append(f"  {w}")

    quality_block = _build_data_quality_section(quality_report)
    if quality_block:
        lines.append(quality_block)

    lines.append(_build_positions_summary(results))
    lines.append("\n💡 Faktiske handler bør vente til 09:35 ET for bedre spread og likviditet.")
    lines.append("⚠️ Paper trading — ikke ekte ordre.")
    if ai_cost_usd and ai_cost_usd > 0:
        lines.append(f"💰 AI-kostnad i dag: ~${ai_cost_usd:.2f}")

    return "\n".join(lines)


# ============================================================
# WEEKLY REPORT (Mondays)
# ============================================================

def build_weekly_report(results, fundamentals_cache=None, corr_pairs=None, weekly_analyses=None):
    lines = ["\n📅 UKENTLIG DEEP-DIVE"]

    for r in results:
        positions = r.get("positions", {})
        if not positions:
            continue

        best_ticker, best_pnl = None, -999
        worst_ticker, worst_pnl = None, 999

        for ticker, pos in positions.items():
            avg = pos.get("avg_price", 0)
            last = pos.get("last_price", avg)
            pnl = (last / avg - 1) if avg else 0
            if pnl > best_pnl:
                best_pnl, best_ticker = pnl, ticker
            if pnl < worst_pnl:
                worst_pnl, worst_ticker = pnl, ticker

        lines.append(f"\n{r['strategy']}:")
        if best_ticker:
            lines.append(f"  Beste:   {best_ticker} {format_pct(best_pnl)}")
        if worst_ticker:
            lines.append(f"  Dårligste: {worst_ticker} {format_pct(worst_pnl)}")

        # Sector exposure
        sector_exposure = {}
        total_val = r.get("portfolio_value", 1)
        for ticker, pos in positions.items():
            sector = "Unknown"
            if fundamentals_cache and ticker in fundamentals_cache:
                sector = fundamentals_cache[ticker].get("sector", "Unknown")
            mv = pos.get("market_value", 0)
            sector_exposure[sector] = sector_exposure.get(sector, 0) + mv

        if sector_exposure:
            lines.append("  Sektor-eksponering:")
            for sector, val in sorted(sector_exposure.items(), key=lambda x: -x[1]):
                pct = val / total_val if total_val > 0 else 0
                lines.append(f"    {sector}: {format_pct(pct, 1)}")

    # Weekly AI thesis review (Layer 2 — Sonnet)
    if weekly_analyses:
        lines.append("\n🧠 Ukentlig tese-gjennomgang (Sonnet):")
        action_icons = {"hold": "⏸", "add": "➕", "reduce": "⬇️", "sell": "🔴"}
        conviction_icons = {"high": "●●●", "medium": "●●○", "low": "●○○"}
        for ticker, wa in sorted(weekly_analyses.items()):
            if wa.get("source") == "fallback":
                continue
            action     = wa.get("action", "hold")
            conviction = wa.get("conviction", "low")
            intact     = "✓" if wa.get("thesis_intact") else "✗"
            icon       = action_icons.get(action, "•")
            conv_str   = conviction_icons.get(conviction, conviction)
            reasoning  = wa.get("reasoning", "")[:80]
            lines.append(f"  {icon} {ticker} [{intact} tese | {conv_str}]: {reasoning}")
            risks = wa.get("risks", [])
            cats  = wa.get("catalysts", [])
            if risks:
                lines.append(f"    Risiko: {' | '.join(risks)}")
            if cats:
                lines.append(f"    Katalysator: {' | '.join(cats)}")

    # Correlation heatmap across all held positions
    if corr_pairs:
        from modules.correlation import format_correlation_heatmap
        all_held = set()
        for r in results:
            all_held.update(r.get("positions", {}).keys())
        heatmap = format_correlation_heatmap(corr_pairs, all_held)
        if heatmap:
            lines.append(heatmap)

    return "\n".join(lines)


# ============================================================
# MONTHLY REPORT (first Monday of month)
# ============================================================

def build_monthly_report(results, spy_ret, qqq_ret):
    lines = ["\n📆 MÅNEDLIG OPPSUMMERING"]
    lines.append(f"SPY siden start: {format_pct(spy_ret)}")
    lines.append(f"QQQ siden start: {format_pct(qqq_ret)}")
    lines.append("")

    sorted_results = sorted(results, key=lambda r: r["return_pct"], reverse=True)

    lines.append("Strategi-rangering vs SPY:")
    for r in sorted_results:
        alpha = (r["return_pct"] - spy_ret) if spy_ret is not None else None
        alpha_str = f"  α {format_pct(alpha)}" if alpha is not None else ""
        val = safe_round(r["portfolio_value"], 2)
        lines.append(
            f"  {r['strategy']}: ${val} ({format_pct(r['return_pct'])}){alpha_str}"
        )

    lines.append("")
    lines.append("Faktor-snitt for nåværende posisjoner:")
    for r in sorted_results:
        top = r.get("top_candidates_full", [])
        held_tickers = set(r.get("positions", {}).keys())
        held = [c for c in top if c.get("ticker") in held_tickers]
        if not held:
            continue

        def avg(key):
            vals = [safe_float(c.get(key)) for c in held if safe_float(c.get(key)) is not None]
            return sum(vals) / len(vals) if vals else None

        m = avg("momentum_score")
        q = avg("quality_score")
        v = avg("value_score")
        s = avg("sentiment_score")

        parts = []
        if m is not None:
            parts.append(f"Mom={safe_round(m, 1)}")
        if q is not None:
            parts.append(f"Kval={safe_round(q, 1)}")
        if v is not None:
            parts.append(f"Val={safe_round(v, 1)}")
        if s is not None:
            parts.append(f"Sent={safe_round(s, 1)}")

        lines.append(f"  {r['strategy']}: {', '.join(parts)}")

    return "\n".join(lines)


# ============================================================
# STALE / INVALID SIGNAL REPORT
# ============================================================

def build_stale_signal_message(validation) -> str:
    """
    Build a Telegram alert for a stale or invalid signal.
    validation is a SignalValidationResult (imported lazily to avoid circular import).
    """
    blocked_buys = sum(
        len(r.get("buys", [])) == 0 and not r.get("orders_created", 0)
        for r in getattr(validation, "_results", [])
    )
    tier = getattr(validation, "data_quality_tier", "unknown")
    dq = getattr(validation, "data_quality", {})
    coverage = dq.get("signal_coverage_rate")
    stale_pct = dq.get("stale_pct")

    mode = getattr(validation, "failure_mode", "unknown")
    is_fully_invalid = not getattr(validation, "is_valid", True)
    header = "⛔ SIGNAL AVVIST" if is_fully_invalid else "⚠️ BEGRENSET KJØRING"

    lines = [
        header,
        f"Signal-run ID:     {getattr(validation, 'signal_run_id', 'ukjent') or 'ukjent'}",
        f"Intendert sesjon:  {getattr(validation, 'intended_session', 'ukjent') or 'ukjent'}",
        f"Dagens sesjon:     {getattr(validation, 'actual_session', today_str())}",
        f"Generert (UTC):    {getattr(validation, 'generated_at', 'ukjent') or 'ukjent'}",
        f"Publisert (UTC):   {getattr(validation, 'published_at', 'ukjent') or 'ukjent'}",
        f"Feilmodus:         {mode}",
        f"Årsak:             {getattr(validation, 'reason', '')}",
    ]

    if coverage is not None or stale_pct is not None:
        lines.append(
            f"Data-kvalitet:     tier={tier}, "
            f"coverage={coverage}, stale_pct={stale_pct}"
        )

    lines += [
        "",
        "✅ Tillatte handlinger: stop-loss, trailing stop, drawdown-beskyttelse",
        "❌ Blokkerte handlinger: nye kjøp, pyramidering"
        + (", signal-baserte salg" if is_fully_invalid else ""),
    ]

    return "\n".join(lines)


# ============================================================
# FULL MESSAGE ASSEMBLY
# ============================================================

def build_full_message(results, signal, signal_path, spy_ret, qqq_ret,
                       drawdown_warnings, fundamentals_cache=None, macro=None,
                       earnings_analysis=None, active_weights=None, corr_pairs=None,
                       sentiment_scores=None, quality_report=None,
                       weekly_analyses=None, ai_cost_usd=0.0,
                       signal_validation=None):
    # Prefix restricted-execution banner when signal is invalid/stale
    prefix = ""
    if signal_validation is not None and (
        not signal_validation.is_valid or not signal_validation.allow_new_buys
    ):
        mode = signal_validation.failure_mode
        prefix = f"⚠️ BEGRENSET KJØRING ({mode}) — kun risikohåndtering\n\n"

    msg = prefix + build_daily_report(
        results, signal, signal_path, spy_ret, qqq_ret, drawdown_warnings,
        macro=macro, earnings_analysis=earnings_analysis, active_weights=active_weights,
        sentiment_scores=sentiment_scores, quality_report=quality_report,
        ai_cost_usd=ai_cost_usd,
    )

    if is_monday():
        msg += "\n" + build_weekly_report(
            results, fundamentals_cache, corr_pairs=corr_pairs,
            weekly_analyses=weekly_analyses,
        )

    if is_first_monday_of_month():
        msg += "\n" + build_monthly_report(results, spy_ret, qqq_ret)

    return msg
