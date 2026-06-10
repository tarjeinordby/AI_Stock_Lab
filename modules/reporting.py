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

    return "\n".join(lines)


def _build_earnings_alerts(results):
    # Collect all positions with upcoming earnings across all strategies
    soon = {}
    for r in results:
        candidates_by_ticker = {c["ticker"]: c for c in r.get("top_candidates_full", [])}
        for ticker in r.get("positions", {}):
            c = candidates_by_ticker.get(ticker, {})
            if c.get("earnings_soon"):
                days = c.get("days_to_earnings")
                date = c.get("next_earnings", "?")
                if ticker not in soon:
                    soon[ticker] = {"date": date, "days": days, "strategies": []}
                soon[ticker]["strategies"].append(r["strategy"])

    if not soon:
        return ""

    lines = ["\n⚠️ EARNINGS DENNE UKEN / 14 DAGER:"]
    for ticker, info in sorted(soon.items(), key=lambda x: x[1].get("days") or 999):
        strats = ", ".join(set(info["strategies"]))
        lines.append(f"  {ticker}: {info['date']} (om {info['days']}d) — {strats}")

    return "\n".join(lines)


def _build_positions_summary(results):
    lines = ["\n📊 POSISJONER"]
    sorted_results = sorted(results, key=lambda r: r["return_pct"], reverse=True)

    for r in sorted_results:
        lines.append(f"\n{r['strategy']} (${safe_round(r['portfolio_value'], 0)}):")
        positions = r.get("positions", {})
        if positions:
            holdings = []
            for ticker, pos in sorted(positions.items()):
                avg = pos.get("avg_price", 0)
                last = pos.get("last_price", avg)
                pnl = (last / avg - 1) if avg else 0
                holdings.append(f"{ticker}({format_pct(pnl, 1)})")
            lines.append("  " + ", ".join(holdings[:15]))
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


def build_daily_report(results, signal, signal_path, spy_ret, qqq_ret, drawdown_warnings):
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
        "",
    ]

    lines.append(_build_leaderboard(results, spy_ret, qqq_ret))
    lines.append("")
    lines.append(_build_actions(results))
    earnings_block = _build_earnings_alerts(results)
    if earnings_block:
        lines.append(earnings_block)

    if drawdown_warnings:
        lines.append("\n🔴 DRAWDOWN ADVARSLER:")
        for w in drawdown_warnings:
            lines.append(f"  {w}")

    lines.append(_build_positions_summary(results))
    lines.append("\n⚠️ Paper trading — ikke ekte ordre.")

    return "\n".join(lines)


# ============================================================
# WEEKLY REPORT (Mondays)
# ============================================================

def build_weekly_report(results, fundamentals_cache=None):
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
# FULL MESSAGE ASSEMBLY
# ============================================================

def build_full_message(results, signal, signal_path, spy_ret, qqq_ret,
                       drawdown_warnings, fundamentals_cache=None):
    msg = build_daily_report(results, signal, signal_path, spy_ret, qqq_ret, drawdown_warnings)

    if is_monday():
        msg += "\n" + build_weekly_report(results, fundamentals_cache)

    if is_first_monday_of_month():
        msg += "\n" + build_monthly_report(results, spy_ret, qqq_ret)

    return msg
