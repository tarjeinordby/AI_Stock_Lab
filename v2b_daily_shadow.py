"""
V2B.2 — Daily shadow collection entry point.

SHADOW ONLY. Never creates orders, fills, or trades.
order_creation_blocked = True is a structural invariant.

Called by .github/workflows/v2b_shadow.yml. Exits 0 cleanly for non-trading days.
Prints a labeled V2 SHADOW preview; does NOT send to Telegram.

V1 production workflows and Telegram messages are completely unaffected.
"""

from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

warnings.simplefilter(action="ignore", category=FutureWarning)

from modules.exchange_calendar import CalendarUnavailableError, is_trading_session
from modules.v2b_shadow_runner import (
    SessionRejectedError,
    format_shadow_preview,
    run_shadow_observation,
)
from modules.versioning import get_model_config_hash, get_portfolio_config_hash
from modules.v2_strategy import V2_MODEL_VERSION, V2_PORTFOLIO_VERSION

# ── Determine as_of_date ──────────────────────────────────────────────────────
# Historical backfill is NOT supported in V2B.2 — no point-in-time data source.
# Only today's date in America/New_York is accepted as a manual as_of_date.
_NY_TZ = ZoneInfo("America/New_York")
_TODAY_NY = datetime.now(_NY_TZ).strftime("%Y-%m-%d")

_ENV_AS_OF_DATE = os.environ.get("V2B_AS_OF_DATE", "").strip()
if _ENV_AS_OF_DATE and _ENV_AS_OF_DATE != _TODAY_NY:
    print(
        f"ERROR: V2B_AS_OF_DATE={_ENV_AS_OF_DATE!r} is not today's date in "
        f"America/New_York ({_TODAY_NY!r}).\n"
        "Historical backfill is not supported in V2B.2 — a point-in-time data\n"
        "source is required for any date other than today.\n"
        "Remove V2B_AS_OF_DATE or set it to today's date in America/New_York."
    )
    sys.exit(1)

AS_OF_DATE = _TODAY_NY

print(f"V2B shadow collection — as_of_date: {AS_OF_DATE}")
print("SHADOW ONLY: order_creation_blocked = True")

# ── Session guard ─────────────────────────────────────────────────────────────
try:
    if not is_trading_session(AS_OF_DATE):
        print(f"{AS_OF_DATE} is not a NYSE trading session. Exiting cleanly.")
        sys.exit(0)
except CalendarUnavailableError as exc:
    print(f"NYSE calendar unavailable: {exc}. Exiting cleanly.")
    sys.exit(0)

# ── Data cutoff timestamp ──────────────────────────────────────────────────────
DATA_CUTOFF_AT = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

# ── Fetch data ────────────────────────────────────────────────────────────────
# Data fetching uses yfinance utilities (not V1 execution modules).
# No execute_buy/sell/pyramid_fill is called anywhere in this script.

try:
    from modules.universe import build_universe
    from modules.market_data import download_daily_data
    from modules.fundamentals import fetch_fundamentals_bulk

    print("Fetching universe...")
    tickers = build_universe()
    # SPY is always included for beta calculation
    if "SPY" not in tickers:
        tickers = ["SPY"] + tickers

    print(f"Downloading price data for {len(tickers)} tickers...")
    price_data = download_daily_data(tickers)
    spy_prices = price_data.pop("SPY", None)

    print("Fetching fundamentals...")
    fundamentals = fetch_fundamentals_bulk(tickers)

    # Sector map from yfinance info
    sector_map = {
        t: info.get("sector", "Unknown")
        for t, info in fundamentals.items()
        if isinstance(info, dict) and info.get("sector")
    }

except Exception as exc:
    print(f"Data fetch failed: {exc}")
    sys.exit(1)

# ── Config hashes ─────────────────────────────────────────────────────────────
model_config_hash = get_model_config_hash(V2_MODEL_VERSION)
portfolio_config_hash = get_portfolio_config_hash(V2_PORTFOLIO_VERSION)

if not model_config_hash:
    print(f"model_config_hash not found for {V2_MODEL_VERSION!r}. Aborting.")
    sys.exit(1)

# ── Run shadow observation ─────────────────────────────────────────────────────
print("Running V2B shadow observation...")
try:
    result = run_shadow_observation(
        as_of_date=AS_OF_DATE,
        price_data=price_data,
        fundamentals=fundamentals,
        data_cutoff_at=DATA_CUTOFF_AT,
        spy_prices=spy_prices,
        sector_map=sector_map,
        claude_outputs=None,   # Phase 2: Claude shadow not yet wired
        metadata={"source": "v2b_daily_shadow.py", "github_actions": True},
        model_config_hash=model_config_hash,
        portfolio_config_hash=portfolio_config_hash,
    )
except SessionRejectedError as exc:
    print(f"Session rejected: {exc}. Exiting cleanly.")
    sys.exit(0)
except Exception as exc:
    print(f"Shadow observation failed unexpectedly: {exc}")
    sys.exit(1)

# ── Print labeled preview (NOT sent to Telegram) ───────────────────────────────
print()
print(format_shadow_preview(result))
print()

if result.status in ("COMPLETED", "IDEMPOTENT_MATCH"):
    print(f"V2B shadow collection finished: {result.status}")
    sys.exit(0)
else:
    print(f"V2B shadow collection ended with: {result.status} — {result.error}")
    sys.exit(1)
