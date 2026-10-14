"""One-off cleanup of data_v4/trades.csv after the 2026-10-14 reconciliation.

1. Drop the 19 duplicate copies (rows without IDs whose trade also exists as an
   ID'd row restored from the fill ledger). The ID'd copy is kept.
2. Move the 18 test trades made on 2026-06-11 before the portfolios were created
   (08:42:29 UTC) to data_v4/archive/trades_test_2026-06-11.csv.
3. Backfill order_id/trade_id (and traceability fields) for the 2026-10-14 buys
   from data_v4/state/orders.jsonl.
4. Give the remaining rows from before the order system existed stable IDs
   prefixed "legacy-".

Idempotent: running it on an already-cleaned file changes nothing.
"""

import hashlib
import json
import os
import sys

import pandas as pd

TRADES = "data_v4/trades.csv"
ARCHIVE = "data_v4/archive/trades_test_2026-06-11.csv"
ORDERS = "data_v4/state/orders.jsonl"
PORTFOLIOS_CREATED = pd.Timestamp("2026-06-11T08:42:29Z")


def main(dry_run=False):
    t = pd.read_csv(TRADES)
    cols = list(t.columns)
    before = len(t)
    ts = pd.to_datetime(t["timestamp_utc"], utc=True, format="mixed")

    # 2. test trades
    test_mask = ts < PORTFOLIOS_CREATED
    test_rows = t[test_mask]

    # 1. duplicates: same strategy/side/ticker/date/shares, keep the row that has IDs
    side = t["action"].where(t["action"] == "SELL", "BUY")
    key = pd.DataFrame({"s": t["strategy"], "side": side, "tk": t["ticker"],
                        "d": t["date"], "sh": t["shares"].round(6)})
    in_group = key.duplicated(keep=False)
    group_has_id = t["order_id"].notna().groupby([key[c] for c in key.columns]).transform("any")
    dup_mask = in_group & t["order_id"].isna() & group_has_id & ~test_mask

    kept = t[~test_mask & ~dup_mask].copy()

    # 3. backfill 2026-10-14 IDs from the order ledger
    last = {}
    with open(ORDERS) as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                last[o["order_id"]] = o
    executed = {}
    for o in last.values():
        if o.get("status") == "executed" and o.get("trade_id"):
            k = (o["strategy"], o["action"], o["ticker"], o.get("intended_execution_session"))
            executed.setdefault(k, []).append(o)
    backfilled = 0
    for i, r in kept[kept["order_id"].isna()].iterrows():
        k = (r["strategy"], r["action"], r["ticker"], r["execution_session"])
        matches = executed.get(k, [])
        if len(matches) == 1:
            o = matches[0]
            kept.at[i, "order_id"] = o["order_id"]
            kept.at[i, "trade_id"] = o["trade_id"]
            kept.at[i, "signal_run_id"] = o.get("signal_run_id")
            kept.at[i, "execution_version"] = o.get("execution_version")
            kept.at[i, "gross_execution_price"] = r["price"]
            kept.at[i, "commission_amount"] = r["cost"]
            backfilled += 1
        elif len(matches) > 1:
            sys.exit(f"Ambiguous order match for row {i}: {k}")

    # 4. legacy IDs for the rest
    legacy = 0
    for i, r in kept[kept["order_id"].isna()].iterrows():
        raw = f"{r['strategy']}|{r['action']}|{r['ticker']}|{r['date']}|{r['shares']}|{r['timestamp_utc']}"
        h = hashlib.sha256(raw.encode()).hexdigest()[:12]
        kept.at[i, "order_id"] = f"legacy-ord-{h}"
        kept.at[i, "trade_id"] = f"legacy-trd-{h}"
        legacy += 1

    assert kept["order_id"].notna().all() and kept["trade_id"].notna().all()
    assert not kept["trade_id"].duplicated().any(), "trade_id must be unique"

    print(f"rader før: {before}")
    print(f"  duplikater fjernet:        {int(dup_mask.sum())}")
    print(f"  testhandler arkivert:      {len(test_rows)}")
    print(f"  ID-er fylt inn fra ordrer: {backfilled}")
    print(f"  legacy-ID-er gitt:         {legacy}")
    print(f"rader etter: {len(kept)}")

    if dry_run:
        return kept
    os.makedirs(os.path.dirname(ARCHIVE), exist_ok=True)
    if len(test_rows):
        if os.path.exists(ARCHIVE):
            prev = pd.read_csv(ARCHIVE)
            test_rows = pd.concat([prev, test_rows]).drop_duplicates()
        test_rows.to_csv(ARCHIVE, index=False)
    kept[cols].to_csv(TRADES, index=False)
    return kept


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
