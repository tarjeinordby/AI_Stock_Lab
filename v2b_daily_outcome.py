"""
V2B.4 — Daily outcome tracker entry point.

Invoked from GitHub Actions job 2 (outcome-tracker) after job 1
(shadow-collection-report) completes.

Exit codes:
  0 — all outcomes processed successfully (commit outcomes)
  1 — one or more transport errors (retry tomorrow; do NOT commit)
  2 — integrity failure (CorruptionError) or calendar unavailable
  3 — not a trading session today (soft skip — do NOT commit)

V1 isolation: this file does NOT import from V1 execution modules.
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def main() -> int:
    from zoneinfo import ZoneInfo
    from datetime import datetime

    today_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    # ── Calendar check ────────────────────────────────────────────────────────
    try:
        from modules.exchange_calendar import is_trading_session
    except Exception as exc:
        log.error("Cannot import exchange_calendar: %s", exc)
        return 2

    if not is_trading_session(today_str):
        log.info("Not a trading session (%s) — V2B.4 outcome tracker: skip", today_str)
        return 3

    # ── Run outcome tracker ───────────────────────────────────────────────────
    try:
        from modules.v2b_outcome import CorruptionError
        from modules.v2b_outcome_runner import run_outcome_tracker
    except Exception as exc:
        log.error("Import error: %s", exc)
        return 2

    try:
        exit_code = run_outcome_tracker(today_str)
    except CorruptionError as exc:
        log.error("INTEGRITY FAILURE — fail-closed: %s", exc)
        return 2
    except Exception as exc:
        log.error("Unexpected error in outcome tracker: %s", exc, exc_info=True)
        return 2

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
