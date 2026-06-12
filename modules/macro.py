import datetime as dt
from datetime import datetime, timezone
from io import StringIO

import pandas as pd
import requests

from modules.state import STATE_DIR, load_json, save_json

MACRO_CACHE_FILE = f"{STATE_DIR}/macro_cache.json"
INVERSION_THRESHOLD = -0.5  # percent (not decimal)

_TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/daily-treasury-rates.csv/{year}/all"
    "?type=daily_treasury_yield_curve&field_tdr_date_value={year}&download=true"
)


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


def _fetch_treasury_spread():
    """Fetch 10Y-2Y spread from Treasury.gov (no API key needed)."""
    year = dt.date.today().year
    url = _TREASURY_URL.format(year=year)
    resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df.columns = [c.strip() for c in df.columns]
    df["2 Yr"] = pd.to_numeric(df["2 Yr"], errors="coerce")
    df["10 Yr"] = pd.to_numeric(df["10 Yr"], errors="coerce")
    df = df.dropna(subset=["2 Yr", "10 Yr"])
    if df.empty:
        return None, None
    last = df.iloc[0]  # newest row is first
    spread = float(last["10 Yr"]) - float(last["2 Yr"])
    date = str(last["Date"]).strip()
    return spread, date


def get_macro_status():
    """
    Fetch 10Y-2Y Treasury yield spread from Treasury.gov (no API key required).
    Returns dict: spread, date, inverted, exposure_mult, status.
    If spread < -0.5%: exposure_mult = 0.80 (reduce all exposure by 20%).
    Cached for 24 hours.
    """
    cache = load_json(MACRO_CACHE_FILE, {})
    if _cache_is_fresh(cache):
        return cache

    spread, date = None, None
    try:
        spread, date = _fetch_treasury_spread()
    except Exception as e:
        print(f"Treasury.gov feil: {e} — bruker nøytral makro (mult=1.0)")

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
