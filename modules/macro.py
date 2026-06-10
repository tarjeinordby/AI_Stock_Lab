from datetime import datetime, timezone
from io import StringIO

import pandas as pd
import requests

from modules.state import STATE_DIR, load_json, save_json

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y"
MACRO_CACHE_FILE = f"{STATE_DIR}/macro_cache.json"
INVERSION_THRESHOLD = -0.5  # percent (not decimal)


def _cache_is_fresh(cache):
    fetched_at = cache.get("fetched_at")
    if not fetched_at:
        return False
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(fetched_at)).total_seconds()
        return age < 24 * 3600
    except Exception:
        return False


def _build_status(spread):
    if spread is None:
        return "Ingen data", 1.0, False
    inverted = spread < INVERSION_THRESHOLD
    exposure_mult = 0.80 if inverted else 1.0
    if spread < -1.0:
        label = f"Kraftig invertert ({spread:.2f}%)"
    elif spread < INVERSION_THRESHOLD:
        label = f"Invertert ({spread:.2f}%)"
    elif spread < 0:
        label = f"Svakt negativ ({spread:.2f}%)"
    else:
        label = f"Normal ({spread:.2f}%)"
    return label, exposure_mult, inverted


def get_macro_status():
    """
    Fetch 10Y-2Y Treasury yield spread from FRED (no API key required).
    Returns dict: spread, date, inverted, exposure_mult, status.
    If spread < -0.5%: exposure_mult = 0.80 (reduce all exposure by 20%).
    Cached for 24 hours.
    """
    cache = load_json(MACRO_CACHE_FILE, {})
    if _cache_is_fresh(cache):
        return cache

    spread, date = None, None
    try:
        resp = requests.get(FRED_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.columns = ["date", "spread"]
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce")
        df = df.dropna()
        if not df.empty:
            last = df.iloc[-1]
            spread = float(last["spread"])
            date = str(last["date"])
    except Exception as e:
        print(f"FRED API feil: {e} — bruker nøytral makro (mult=1.0)")

    label, exposure_mult, inverted = _build_status(spread)

    result = {
        "spread": spread,
        "date": date,
        "inverted": inverted,
        "exposure_mult": exposure_mult,
        "status": label,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }

    save_json(result, MACRO_CACHE_FILE)
    print(f"10Y/2Y rentespread: {label} | multiplikator: {exposure_mult:.0%}")
    return result
