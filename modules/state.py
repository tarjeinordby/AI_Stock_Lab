import os
import json
import math
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from modules.versioning import PORTFOLIO_VERSION as _PORTFOLIO_VERSION

OSLO = ZoneInfo("Europe/Oslo")
NY = ZoneInfo("America/New_York")

DATA_DIR = "data_v4"
STATE_DIR = f"{DATA_DIR}/state"
SIGNALS_DIR = f"{DATA_DIR}/signals"
REPORTS_DIR = f"{DATA_DIR}/reports"
TRADES_FILE = f"{DATA_DIR}/trades.csv"
PERFORMANCE_FILE = f"{DATA_DIR}/performance.csv"
BENCHMARK_FILE = f"{STATE_DIR}/benchmarks.json"
FUNDAMENTALS_CACHE_FILE = f"{STATE_DIR}/fundamentals_cache.json"
SENTIMENT_CACHE_FILE = f"{STATE_DIR}/sentiment_cache.json"
WEEKLY_ANALYSIS_CACHE_FILE = f"{STATE_DIR}/weekly_analysis_cache.json"
GLOBAL_STATE_FILE = f"{STATE_DIR}/_global.json"
ORDERS_FILE = f"{STATE_DIR}/orders.jsonl"

START_CAPITAL = 10_000.0


def now_utc():
    return datetime.now(timezone.utc)


def now_oslo():
    return now_utc().astimezone(OSLO)


def now_ny():
    return now_utc().astimezone(NY)


def today_str():
    return now_oslo().strftime("%Y-%m-%d")


def is_market_open():
    """True if NYSE trading window is active: Mon–Fri, 09:35–15:55 ET."""
    now = now_ny()
    if now.weekday() >= 5:          # lørdag=5, søndag=6
        return False
    open_time  = now.replace(hour=9,  minute=35, second=0, microsecond=0)
    close_time = now.replace(hour=15, minute=55, second=0, microsecond=0)
    return open_time <= now < close_time


def ensure_dirs():
    for path in [DATA_DIR, STATE_DIR, SIGNALS_DIR, REPORTS_DIR]:
        os.makedirs(path, exist_ok=True)


def to_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        if math.isnan(float(obj)):
            return None
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def save_json(payload, path):
    ensure_dirs()
    with open(path, "w") as f:
        json.dump(to_json_safe(payload), f, indent=2, sort_keys=True)


def load_json(path, default=None):
    if default is None:
        default = {}
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Feil ved lasting av {path}: {e}")
        return default


def read_csv_or_empty(path, columns):
    if not os.path.exists(path):
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Kan ikke lese CSV {path}: {exc}") from exc


def save_csv(df, path):
    ensure_dirs()
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    parent = os.path.dirname(os.path.abspath(path))
    fd = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def safe_float(x, default=None):
    try:
        if x is None:
            return default
        value = float(x)
        if math.isnan(value):
            return default
        return value
    except Exception:
        return default


def safe_round(x, digits=2):
    value = safe_float(x)
    if value is None:
        return "N/A"
    return round(value, digits)


def pct_change(current, previous):
    try:
        if previous is None or previous == 0:
            return None
        if math.isnan(float(previous)):
            return None
        return (float(current) / float(previous)) - 1
    except Exception:
        return None


def load_trades():
    return read_csv_or_empty(
        TRADES_FILE,
        ["date", "timestamp_utc", "strategy", "action", "ticker",
         "shares", "price", "value", "cost", "reason"]
    )


def load_performance():
    return read_csv_or_empty(
        PERFORMANCE_FILE,
        ["date", "strategy", "portfolio_value", "cash",
         "positions_value", "return_pct", "num_positions",
         "spy_return_pct", "qqq_return_pct"]
    )


def strategy_state_file(strategy_name):
    return f"{STATE_DIR}/{strategy_name}.json"


def initial_strategy_state(strategy_name):
    return {
        "strategy": strategy_name,
        "created_at": now_utc().isoformat(),
        "portfolio_id": str(uuid.uuid4()),
        "portfolio_version": _PORTFOLIO_VERSION,
        "cash": START_CAPITAL,
        "positions": {},
        "highest_portfolio_value": START_CAPITAL,
        "weekly_meta": {"iso_week": "", "buys_this_week": 0},
        "cooldowns": {},
        "last_execution_date": None,
    }


def load_strategy_state(strategy_name):
    path = strategy_state_file(strategy_name)
    if not os.path.exists(path):
        return initial_strategy_state(strategy_name)
    try:
        with open(path, "r") as f:
            content = f.read()
    except OSError as exc:
        raise RuntimeError(f"Kan ikke lese porteføljefil {path}: {exc}") from exc
    if not content.strip():
        raise RuntimeError(
            f"Porteføljefil for '{strategy_name}' er tom: {path}. "
            "Slett filen for å starte på nytt med initial kapital."
        )
    try:
        state = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Porteføljefil for '{strategy_name}' er korrupt: {path}. "
            f"JSON-feil: {exc}. Slett eller reparer filen manuelt."
        ) from exc
    # Backward-compat migrations
    if "highest_portfolio_value" not in state:
        state["highest_portfolio_value"] = START_CAPITAL
    if "cooldowns" not in state:
        state["cooldowns"] = {}
    if "weekly_meta" not in state:
        state["weekly_meta"] = {"iso_week": "", "buys_this_week": 0}
    if "portfolio_id" not in state:
        state["portfolio_id"] = str(uuid.uuid4())
    if "portfolio_version" not in state:
        state["portfolio_version"] = _PORTFOLIO_VERSION
    return state


def save_strategy_state(strategy_name, state):
    """Atomically save portfolio state: temp file → fsync → os.replace → fsync parent.

    Crash between write and os.replace leaves a .tmp file (harmless) but never
    produces a partially-written portfolio JSON at the canonical path.
    """
    path = strategy_state_file(strategy_name)
    ensure_dirs()
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(to_json_safe(state), f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    parent = os.path.dirname(os.path.abspath(path))
    fd = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def reset_week_if_needed(state):
    current_week = now_oslo().strftime("%G-W%V")
    if state["weekly_meta"].get("iso_week") != current_week:
        state["weekly_meta"] = {"iso_week": current_week, "buys_this_week": 0}
    return state


def trading_days_between(date_str):
    if not date_str:
        return 999
    try:
        start = pd.Timestamp(date_str)
        end = pd.Timestamp(today_str())
        return len(pd.bdate_range(start, end)) - 1
    except Exception:
        return 999


class SignalNotPublishedError(Exception):
    """Raised when the signal file exists but is not yet finalized/published."""


def load_latest_signal():
    """
    Load the most recently registered signal file.

    Fail-closed: no directory scan fallback. The signal must have been
    explicitly registered in _global.json by run_finalize_signal() with
    publication_status="published". If the file is missing or unpublished,
    raise rather than silently falling back to an older or unvalidated signal.
    """
    global_state = load_json(GLOBAL_STATE_FILE, {})
    signal_path = global_state.get("last_signal_file")

    if not signal_path:
        raise FileNotFoundError(
            "Ingen signalfil registrert i _global.json. "
            "Kjør BOT_MODE=signal etterfulgt av BOT_MODE=finalize_signal."
        )
    if not os.path.exists(signal_path):
        raise FileNotFoundError(
            f"Signalfil ikke funnet på disk: {signal_path}. "
            "Sjekk at data_v4/signals/ er korrekt committed og pushet."
        )

    with open(signal_path, "r") as f:
        signal = json.load(f)

    if signal.get("publication_status") != "published":
        raise SignalNotPublishedError(
            f"Signal {signal_path!r} er ikke publisert "
            f"(status={signal.get('publication_status')!r}). "
            "Kjør BOT_MODE=finalize_signal etter vellykket git push."
        )

    return signal, signal_path


def load_benchmark_state():
    return load_json(BENCHMARK_FILE, {
        "created_at": now_utc().isoformat(),
        "SPY": {},
        "QQQ": {},
    })


def save_benchmark_state(state):
    save_json(state, BENCHMARK_FILE)


