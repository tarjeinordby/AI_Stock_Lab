"""Reconcile trades.csv against each strategy's saved portfolio state.

Replays every trade since the portfolio was created (state["created_at"]) from
START_CAPITAL and compares the resulting cash and share counts with the state file.
SPLIT rows change share counts without touching cash.
"""

import pandas as pd

from modules.state import START_CAPITAL

ALERT_THRESHOLD_USD = 1.0
_SHARE_TOLERANCE = 1e-6


def _cash_effect(row):
    if pd.notna(row.get("net_cash_effect")):
        return float(row["net_cash_effect"])
    value, cost = float(row.get("value") or 0.0), float(row.get("cost") or 0.0)
    return (value - cost) if row["action"] == "SELL" else -(value + cost)


def reconcile_strategy(trades_df, state, strategy):
    created = pd.Timestamp(state["created_at"])
    if created.tzinfo is None:
        created = created.tz_localize("UTC")

    rows = trades_df[trades_df["strategy"] == strategy].copy() if not trades_df.empty else trades_df
    if not rows.empty:
        rows["_ts"] = pd.to_datetime(rows["timestamp_utc"], utc=True, format="mixed")
        rows = rows[rows["_ts"] >= created].sort_values("_ts", kind="stable")

    cash = START_CAPITAL
    shares, last_price = {}, {}
    for _, r in rows.iterrows():
        ticker, action = r["ticker"], r["action"]
        if action == "SPLIT":
            shares[ticker] = shares.get(ticker, 0.0) + float(r["shares"])
            continue
        cash += _cash_effect(r)
        last_price[ticker] = float(r["price"])
        if action == "SELL":
            shares[ticker] = 0.0
        else:
            shares[ticker] = shares.get(ticker, 0.0) + float(r["shares"])

    positions = state.get("positions", {})
    cash_diff = float(state["cash"]) - cash
    share_diffs = []
    for ticker in sorted(set(shares) | set(positions)):
        replayed = shares.get(ticker, 0.0)
        saved = float(positions.get(ticker, {}).get("shares", 0.0))
        if abs(saved - replayed) > _SHARE_TOLERANCE:
            pos = positions.get(ticker, {})
            price = float(pos.get("last_price") or pos.get("avg_price") or last_price.get(ticker) or 0.0)
            share_diffs.append({
                "ticker": ticker, "state": saved, "trades": replayed,
                "usd": abs(saved - replayed) * price,
            })

    deviation_usd = abs(cash_diff) + sum(d["usd"] for d in share_diffs)
    return {
        "strategy": strategy,
        "trades_replayed": len(rows),
        "cash_state": float(state["cash"]),
        "cash_trades": cash,
        "cash_diff": cash_diff,
        "share_diffs": share_diffs,
        "deviation_usd": deviation_usd,
        "ok": deviation_usd <= ALERT_THRESHOLD_USD,
    }


def reconcile_all(trades_df, strategies, load_state):
    return [reconcile_strategy(trades_df, load_state(s), s) for s in strategies]


def format_reconciliation(results):
    lines = []
    for r in results:
        tag = "OK" if r["ok"] else "AVVIK"
        line = (f"{tag:5} {r['strategy']}: kontant diff {r['cash_diff']:+.2f} USD, "
                f"{r['trades_replayed']} handler avspilt")
        for d in r["share_diffs"]:
            line += (f"\n      {d['ticker']}: state {d['state']:.6f} vs trades {d['trades']:.6f}"
                     f" (~{d['usd']:.2f} USD)")
        lines.append(line)
    return "\n".join(lines)
