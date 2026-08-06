"""
Immutable signal ledger — three append-only record types.

Record types (monthly JSONL files):
  signal_run       — one per daily analysis run
  feature_snapshot — one per ticker per run (shared across all strategies)
  signal_record    — one per strategy×ticker per run

ID scheme:
  signal_run_id       = "run-{date}-{sha256(date|model|hash8)[:12]}"
  feature_snapshot_id = "fs-{sha256(run_id|ticker|input_hash)[:12]}"
  signal_id           = "sig-{sha256(run_id|model|strategy|ticker|snapshot_id)[:12]}"

portfolio_version and execution_version are NOT part of signal_id — the same
signal record can be consumed by multiple portfolios or execution models.

Write order per signal run (enforced by stock_bot.py):
  1. write_signal_run(status="pending")
  2. write_feature_snapshots(...)     ← may raise LedgerWriteError
  3. write_signal_records(...)        ← may raise LedgerWriteError
  4. update_signal_run_status("completed")
  5. [caller saves signal file]

If any step fails: update_signal_run_status("failed"), Telegram alert, no signal file.

Atomicity: per-record-type lock file + fcntl.flock(LOCK_EX)
Idempotency: sidecar index (*.idx.json) for O(1) duplicate checks
Partial write: _heal_partial_last_line truncates at last complete '\\n' on each open
Crash recovery: _needs_index_rebuild detects when index is older than JSONL data

Not committed to git — data_v4/ledger/ is in .gitignore.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
LEDGER_DIR = Path(__file__).parent.parent / "data_v4" / "ledger"

_RECORD_TYPES = ("signal_run", "feature_snapshot", "signal_record")
_ID_FIELDS = {
    "signal_run": "signal_run_id",
    "feature_snapshot": "feature_snapshot_id",
    "signal_record": "signal_id",
}


class LedgerWriteError(Exception):
    """
    Raised when ledger write fails or a partial duplicate is detected.
    Caller must NOT save the signal file when this is raised.
    """


# ---------------------------------------------------------------------------
# Deterministic ID computation
# ---------------------------------------------------------------------------

def make_signal_run_id(
    intended_session: str,
    model_version: str,
    model_config_hash: str,
    run_type: str = "scheduled",
) -> str:
    """
    Retry-stable run ID: same session + model + hash → same ID.
    Changed model or new day → new ID.
    """
    key = f"{intended_session}|{model_version}|{model_config_hash[:8]}|{run_type}"
    h = hashlib.sha256(key.encode()).hexdigest()[:12]
    return f"run-{intended_session}-{h}"


def make_feature_snapshot_id(
    signal_run_id: str, ticker: str, input_hash: str
) -> str:
    """Deterministic snapshot ID — same input data → same ID."""
    key = f"{signal_run_id}|{ticker}|{input_hash}"
    return "fs-" + hashlib.sha256(key.encode()).hexdigest()[:12]


def make_signal_id(
    signal_run_id: str,
    model_version: str,
    strategy_id: str,
    ticker: str,
    feature_snapshot_id: str,
) -> str:
    """
    Deterministic signal ID. portfolio_version and execution_version are excluded —
    the signal can be consumed by multiple portfolios without new IDs.
    """
    key = f"{signal_run_id}|{model_version}|{strategy_id}|{ticker}|{feature_snapshot_id}"
    return "sig-" + hashlib.sha256(key.encode()).hexdigest()[:12]


def compute_input_hash(feature_dict: dict) -> str:
    """SHA-256[:16] of canonical JSON — detects any change in input data."""
    canonical = json.dumps(
        feature_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _safe(val: Any) -> Any:
    """Convert NaN/inf/numpy scalars to JSON-safe Python types."""
    if val is None:
        return None
    if isinstance(val, bool):
        return bool(val)
    try:
        import math
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        if isinstance(val, int):
            return int(val)
        return round(f, 8)
    except (TypeError, ValueError):
        return str(val) if not isinstance(val, str) else val


def _get_git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _make_content_hash(record: dict) -> str:
    """SHA-256 of the full record minus the content_hash field itself."""
    r = {k: v for k, v in record.items() if k != "content_hash"}
    return hashlib.sha256(
        json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _jsonl_path(record_type: str, year_month: str) -> Path:
    return LEDGER_DIR / f"{year_month}_{record_type}s.jsonl"


def _index_path(record_type: str, year_month: str) -> Path:
    return LEDGER_DIR / f"{year_month}_{record_type}s.idx.json"


def _lock_path(record_type: str, year_month: str) -> Path:
    return LEDGER_DIR / f"{year_month}_{record_type}s.lock"


def _status_path(year_month: str) -> Path:
    return LEDGER_DIR / f"{year_month}_signal_run_status.json"


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def _load_index(index_path: Path, record_type: str, year_month: str) -> dict:
    if index_path.exists():
        try:
            return json.loads(index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "schema_version": 1,
        "record_type": record_type,
        "year_month": year_month,
        "ids": {},
        "count": 0,
        "last_updated": _utc_now(),
    }


def _save_index_atomic(index_path: Path, index: dict) -> None:
    tmp = index_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, ensure_ascii=True), encoding="utf-8")
    tmp.rename(index_path)


def _rebuild_index_from_jsonl(
    jsonl_path: Path, record_type: str, year_month: str, id_field: str
) -> dict:
    """Full JSONL scan to rebuild a missing or stale sidecar index."""
    index = _load_index(Path("/dev/null"), record_type, year_month)
    if not jsonl_path.exists():
        return index
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rid = rec.get(id_field)
                if rid:
                    index["ids"][rid] = True
            except json.JSONDecodeError:
                pass
    index["count"] = len(index["ids"])
    return index


def _needs_index_rebuild(index_path: Path, jsonl_path: Path) -> bool:
    """
    True if index is missing, or JSONL is newer than index (crash between JSONL
    write and index update — index is stale).
    """
    if not index_path.exists():
        return jsonl_path.exists()
    if not jsonl_path.exists():
        return False
    return jsonl_path.stat().st_mtime > index_path.stat().st_mtime + 1.0


# ---------------------------------------------------------------------------
# Partial write detection / healing
# ---------------------------------------------------------------------------

def _heal_partial_last_line(path: Path) -> int:
    """
    Truncate the file at the last complete newline.
    Returns number of bytes removed (0 if file was clean).
    """
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with open(path, "rb") as f:
        f.seek(-1, 2)
        last = f.read(1)
    if last == b"\n":
        return 0
    with open(path, "rb") as f:
        content = f.read()
    last_nl = content.rfind(b"\n")
    if last_nl < 0:
        with open(path, "r+b") as f:
            f.truncate(0)
        return len(content)
    removed = len(content) - (last_nl + 1)
    with open(path, "r+b") as f:
        f.truncate(last_nl + 1)
    return removed


# ---------------------------------------------------------------------------
# Core append — atomic, idempotent, fail-closed
# ---------------------------------------------------------------------------

def _append_batch(
    records: list[dict],
    record_type: str,
    id_field: str,
    year_month: str,
) -> int:
    """
    Append records to the monthly JSONL file.

    Returns:
      > 0  — records written
        0  — all records already present (idempotent retry, no-op)

    Raises LedgerWriteError:
      — partial duplicate (some IDs exist, some new — inconsistent state)
      — any I/O error

    Caller must NOT save the signal file if this raises.
    """
    if not records:
        return 0

    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    jpath = _jsonl_path(record_type, year_month)
    ipath = _index_path(record_type, year_month)
    lpath = _lock_path(record_type, year_month)
    lpath.touch(exist_ok=True)

    new_ids = [r[id_field] for r in records]

    with open(lpath, "rb") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            _heal_partial_last_line(jpath)

            if _needs_index_rebuild(ipath, jpath):
                index = _rebuild_index_from_jsonl(jpath, record_type, year_month, id_field)
                _save_index_atomic(ipath, index)
            else:
                index = _load_index(ipath, record_type, year_month)

            existing = {rid for rid in new_ids if index["ids"].get(rid)}
            if existing:
                if len(existing) == len(new_ids):
                    return 0  # full idempotent retry
                raise LedgerWriteError(
                    f"Partial duplicate in {record_type}: "
                    f"{len(existing)}/{len(new_ids)} IDs already exist. "
                    f"First duplicate: {next(iter(existing))}"
                )

            now = _utc_now()
            lines = []
            for rec in records:
                rec.setdefault("written_at", now)
                rec["content_hash"] = _make_content_hash(rec)
                lines.append(
                    json.dumps(rec, sort_keys=True, ensure_ascii=True) + "\n"
                )

            with open(jpath, "a", encoding="utf-8") as df:
                df.writelines(lines)
                df.flush()
                os.fsync(df.fileno())

            for rid in new_ids:
                index["ids"][rid] = True
            index["count"] = len(index["ids"])
            index["last_updated"] = _utc_now()
            _save_index_atomic(ipath, index)

            return len(records)

        except LedgerWriteError:
            raise
        except Exception as exc:
            raise LedgerWriteError(
                f"Ledger write failed for {record_type}: {exc}"
            ) from exc
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Signal run status (mutable — separate from append-only JSONL)
# ---------------------------------------------------------------------------

def _load_run_status(year_month: str) -> dict:
    p = _status_path(year_month)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_run_status(year_month: str, status_map: dict) -> None:
    p = _status_path(year_month)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(status_map, ensure_ascii=True), encoding="utf-8")
    tmp.rename(p)


def get_signal_run_status(signal_run_id: str, year_month: str) -> dict | None:
    return _load_run_status(year_month).get(signal_run_id)


def update_signal_run_status(
    signal_run_id: str,
    status: str,
    year_month: str,
    *,
    n_feature_snapshots: int = 0,
    n_signal_records: int = 0,
    error: str | None = None,
) -> None:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    status_map = _load_run_status(year_month)
    existing = status_map.get(signal_run_id, {})
    existing.update({
        "status": status,
        "updated_at": _utc_now(),
    })
    if status == "completed":
        existing["completed_at"] = _utc_now()
        existing["n_feature_snapshots"] = n_feature_snapshots
        existing["n_signal_records"] = n_signal_records
    if error:
        existing["error"] = error
    status_map[signal_run_id] = existing
    _save_run_status(year_month, status_map)


# ---------------------------------------------------------------------------
# Public write functions
# ---------------------------------------------------------------------------

def write_signal_run(run_record: dict, year_month: str) -> int:
    """
    Append the initial signal_run record (status="pending").
    Idempotent: returns 0 if already written (same signal_run_id).
    """
    return _append_batch([run_record], "signal_run", "signal_run_id", year_month)


def write_feature_snapshots(snapshots: list[dict], year_month: str) -> int:
    """
    Append feature_snapshot records. Raises LedgerWriteError on failure.
    Returns count of newly written records (0 = idempotent retry).
    """
    return _append_batch(snapshots, "feature_snapshot", "feature_snapshot_id", year_month)


def write_signal_records(records: list[dict], year_month: str) -> int:
    """
    Append signal_record records. Raises LedgerWriteError on failure.
    Returns count of newly written records (0 = idempotent retry).
    """
    return _append_batch(records, "signal_record", "signal_id", year_month)


# ---------------------------------------------------------------------------
# Public read functions
# ---------------------------------------------------------------------------

def record_id_exists(
    record_id: str, record_type: str, year_month: str
) -> bool:
    """O(1) check via sidecar index (no JSONL scan required)."""
    id_field = _ID_FIELDS[record_type]
    ipath = _index_path(record_type, year_month)
    jpath = _jsonl_path(record_type, year_month)
    if _needs_index_rebuild(ipath, jpath):
        index = _rebuild_index_from_jsonl(jpath, record_type, year_month, id_field)
        _save_index_atomic(ipath, index)
    else:
        index = _load_index(ipath, record_type, year_month)
    return bool(index["ids"].get(record_id))


def check_ledger_integrity(
    record_type: str, year_month: str
) -> tuple[int, int]:
    """
    Scan JSONL and count (valid_records, corrupt_lines).
    Corrupt = line that cannot be parsed as JSON.
    """
    jpath = _jsonl_path(record_type, year_month)
    if not jpath.exists():
        return 0, 0
    ok, bad = 0, 0
    with open(jpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                ok += 1
            except json.JSONDecodeError:
                bad += 1
    return ok, bad


# ---------------------------------------------------------------------------
# Builder functions (called from stock_bot.py)
# ---------------------------------------------------------------------------

def build_signal_run_record(
    signal_run_id: str,
    intended_session: str,
    model_version: str,
    model_config_hash: str,
    universe_count: int,
    analyzed_count: int,
    attempt_number: int = 1,
    run_type: str = "scheduled",
    first_attempt_at: str | None = None,
) -> dict:
    now = _utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "signal_run",
        "written_at": now,
        "content_hash": "",
        "signal_run_id": signal_run_id,
        "intended_signal_session": intended_session,
        "model_version": model_version,
        "model_config_hash": model_config_hash,
        "run_type": run_type,
        "first_attempt_at": first_attempt_at or now,
        "latest_attempt_at": now,
        "attempt_number": attempt_number,
        "universe_count": universe_count,
        "analyzed_count": analyzed_count,
        "git_commit": _get_git_commit(),
    }


def build_all_ledger_records(
    signal_run_id: str,
    analyzed_by_ticker: list[dict],
    strategies_payload: dict,
    fundamentals: dict,
    sentiment_scores: dict,
    full_earnings: dict,
    macro: dict,
    regime: str,
    model_version: str,
    model_config_hash: str,
    strategies_config: dict,
) -> tuple[list[dict], list[dict]]:
    """
    Build feature_snapshots and signal_records from in-memory scoring data.

    analyzed_by_ticker: score_df.reset_index().to_dict(orient="records")
    strategies_payload: {strat_name: {"candidates": [...], ...}}
    strategies_config: scoring.STRATEGIES (for per-strategy params)

    Returns:
      feature_snapshots  — one per ticker (deduplicated)
      signal_records     — one per (strategy, ticker) from strategies_payload
    """
    git_commit = _get_git_commit()
    now = _utc_now()
    macro_inverted = bool(macro.get("inverted", False))

    # --- Feature snapshots: one per ticker in analyzed_by_ticker ---
    snapshots: list[dict] = []
    snapshot_id_by_ticker: dict[str, str] = {}

    for row in analyzed_by_ticker:
        ticker = str(row["ticker"])
        f = fundamentals.get(ticker) or {}
        sent = sentiment_scores.get(ticker) or {}
        earn = full_earnings.get(ticker) or {}

        features = {
            # Price / technical
            "price": _safe(row.get("price")),
            "sma200": _safe(row.get("sma200")),
            "above_sma200": bool(row.get("above_sma200", False)),
            "vol60": _safe(row.get("vol60")),
            "rsi": _safe(row.get("rsi")),
            "overbought": bool(row.get("overbought", False)),
            "very_weak_rsi": bool(row.get("very_weak_rsi", False)),
            "is_megacap": bool(row.get("is_megacap", False)),
            # Momentum raw
            "mom_12_1": _safe(row.get("mom_12_1")),
            "mom_6_1": _safe(row.get("mom_6_1")),
            "mom_3_1": _safe(row.get("mom_3_1")),
            "ret_1m": _safe(row.get("ret_1m")),
            "relative_strength_3m": _safe(row.get("relative_strength_3m")),
            # Computed factor scores
            "momentum_score": _safe(row.get("momentum_score")),
            "quality_score": _safe(row.get("quality_score")),
            "value_score": _safe(row.get("value_score")),
            "sentiment_score": _safe(row.get("sentiment_score")),
            # Fundamentals
            "roe": _safe(f.get("roe")),
            "gross_margin": _safe(f.get("gross_margin")),
            "debt_equity": _safe(f.get("debt_equity")),
            "earnings_growth": _safe(f.get("earnings_growth")),
            "forward_pe": _safe(f.get("forward_pe")),
            "peg_ratio": _safe(f.get("peg_ratio")),
            "price_to_sales": _safe(f.get("price_to_sales")),
            "sector": str(f.get("sector") or "Unknown"),
            "market_cap": _safe(f.get("market_cap")),
            "fcf_positive": f.get("fcf_positive"),
            # Insider
            "insider_buying": bool(row.get("insider_buying", False)),
            # Sentiment / earnings
            "news_score": _safe(sent.get("score")),
            "earnings_bonus": _safe(earn.get("earnings_bonus")),
            "earnings_soon": bool(earn.get("earnings_soon", False)),
            "next_earnings": earn.get("next_earnings"),
            "days_to_earnings": _safe(earn.get("days_to_earnings")),
            # Macro (shared across tickers — affects scoring weights)
            "market_regime": regime,
            "macro_inverted": macro_inverted,
            "macro_spread_10y_2y": _safe(macro.get("spread")),
        }

        input_hash = compute_input_hash(features)
        snapshot_id = make_feature_snapshot_id(signal_run_id, ticker, input_hash)
        snapshot_id_by_ticker[ticker] = snapshot_id

        snapshots.append({
            "schema_version": SCHEMA_VERSION,
            "record_type": "feature_snapshot",
            "written_at": now,
            "content_hash": "",
            "feature_snapshot_id": snapshot_id,
            "signal_run_id": signal_run_id,
            "ticker": ticker,
            "input_hash": input_hash,
            "features": features,
            "data_quality": {
                "fundamentals_complete": all(
                    _safe(f.get(k)) is not None
                    for k in ("roe", "gross_margin", "debt_equity")
                ),
                "sentiment_available": sent.get("score") is not None,
                "earnings_available": earn.get("earnings_bonus") is not None,
            },
        })

    # --- Signal records: one per (strategy, ticker) from strategies_payload ---
    signal_records: list[dict] = []

    for strat_name, strat_data in strategies_payload.items():
        strat_cfg = strategies_config.get(strat_name, {})
        buy_top_n = strat_cfg.get("buy_top_n", 20)
        min_pct = strat_cfg.get("min_score_percentile", 0.70)

        for candidate in strat_data.get("candidates", []):
            ticker = str(candidate.get("ticker", ""))
            snapshot_id = snapshot_id_by_ticker.get(ticker)
            if snapshot_id is None:
                continue

            signal_id = make_signal_id(
                signal_run_id, model_version, strat_name, ticker, snapshot_id
            )
            rank = candidate.get("rank", 999)
            score_pct = _safe(candidate.get("score_percentile", 0))
            score_col = strat_cfg.get("score_column", "")
            score = _safe(candidate.get(score_col) or candidate.get("strategy_score"))

            if rank <= buy_top_n and score_pct is not None and score_pct >= min_pct:
                recommendation = "BUY_CANDIDATE"
            else:
                recommendation = "MONITOR"

            signal_records.append({
                "schema_version": SCHEMA_VERSION,
                "record_type": "signal_record",
                "written_at": now,
                "content_hash": "",
                "signal_id": signal_id,
                "signal_run_id": signal_run_id,
                "feature_snapshot_id": snapshot_id,
                "ticker": ticker,
                "strategy_id": strat_name,
                "model_version": model_version,
                "model_config_hash": model_config_hash,
                "score": score,
                "rank": rank,
                "score_percentile": score_pct,
                "recommendation": recommendation,
                "component_scores": {
                    "momentum": _safe(candidate.get("momentum_score")),
                    "quality": _safe(candidate.get("quality_score")),
                    "value": _safe(candidate.get("value_score")),
                    "sentiment": _safe(candidate.get("sentiment_score")),
                },
                "penalties_applied": {
                    "below_sma200": not bool(candidate.get("above_sma200", True)),
                    "overbought": bool(candidate.get("overbought", False)),
                    "very_weak_rsi": bool(candidate.get("very_weak_rsi", False)),
                    "high_vol": (_safe(candidate.get("vol60")) or 0) > 1.20,
                    "non_megacap_in_megacap": (
                        strat_name == "MegaCap_AI"
                        and not bool(candidate.get("is_megacap", False))
                    ),
                },
                "bonus_applied": {
                    "insider_buying": bool(candidate.get("insider_buying", False)),
                },
                "git_commit": git_commit,
            })

    return snapshots, signal_records
