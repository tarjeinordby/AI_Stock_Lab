"""
V2B.3 — Daily shadow report entry point.

SHADOW ONLY. Never creates orders, fills, or trades.
order_creation_blocked = True is a structural invariant.

Called by .github/workflows/v2b_shadow.yml after v2b_daily_shadow.py.
Reads the COMPLETED V2B observation for today's intended_execution_session,
builds a structured shadow report, and sends a clearly labeled
"V2 SHADOW — INGEN HANDLER" Telegram message.

V1 production workflows and Telegram messages are completely unaffected.
On any V2 error: a shadow-error message is sent (if Telegram is available),
but exit code is 0 so V1 is never blocked.
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

warnings.simplefilter(action="ignore", category=FutureWarning)

from modules.exchange_calendar import (
    CalendarUnavailableError,
    is_trading_session,
    intended_execution_session as _calendar_next_session,
)
from modules.reporting import send_telegram
from modules.v2b_report import (
    ObservationIncompleteError,
    ObservationNotFoundError,
    ReportConflictError,
    format_telegram_message,
    run_shadow_report,
)

# ── Determine today's date in America/New_York ────────────────────────────────
_NY_TZ = ZoneInfo("America/New_York")
TODAY_NY = datetime.now(_NY_TZ).strftime("%Y-%m-%d")

print(f"V2B shadow report — date: {TODAY_NY}")
print("SHADOW ONLY: order_creation_blocked = True")

# ── Session guard ─────────────────────────────────────────────────────────────
try:
    if not is_trading_session(TODAY_NY):
        print(f"{TODAY_NY} is not a NYSE trading session. No report needed. Exiting cleanly.")
        sys.exit(0)
    INTENDED_SESSION = _calendar_next_session(TODAY_NY)
except CalendarUnavailableError as exc:
    print(f"NYSE calendar unavailable: {exc}. Exiting cleanly.")
    sys.exit(0)

print(f"Intended execution session: {INTENDED_SESSION}")

# ── Run shadow report ─────────────────────────────────────────────────────────
print("Running V2B shadow report...")

result = run_shadow_report(INTENDED_SESSION, send_telegram_fn=send_telegram)

if result.status == "FAILED":
    print(f"Shadow report not available: {result.error}")
    # Send a shadow-error notice (not a V1 message)
    send_telegram(
        f"⚠️ V2 SHADOW — rapport utilgjengelig\n"
        f"Session: {INTENDED_SESSION}\n"
        f"Årsak: {result.error}\n"
        "V1-produksjon er urørt."
    )
    # Exit 0 — V1 must never be blocked by a V2 failure
    sys.exit(0)

if result.status == "CONFLICT":
    print(f"Shadow report conflict (fail-closed): {result.error}")
    send_telegram(
        f"⚠️ V2 SHADOW — rapportkonflikt (fail-closed)\n"
        f"Session: {INTENDED_SESSION}\n"
        f"Detalj: {result.error}\n"
        "Ingen V2-rapport sendt. V1-produksjon er urørt."
    )
    sys.exit(1)

if result.telegram_sent:
    print(f"V2B shadow report sent to Telegram: {result.status}")
elif result.telegram_skipped:
    print(f"V2B shadow Telegram already sent (idempotent): {result.status}")
else:
    print(f"V2B shadow report {result.status}: {result.detail or 'no Telegram configured'}")

print(f"Report key:  {result.report_key}")
print(f"Observation: {result.observation_key}")
print("V2B shadow report complete.")
sys.exit(0)
