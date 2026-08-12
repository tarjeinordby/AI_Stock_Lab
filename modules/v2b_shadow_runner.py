"""
V2B.2 — Daily shadow observation runner.

Records V2 factor scores to the V2B ledger for prospective OOS evaluation.

SHADOW ONLY:
  order_creation_blocked = True is a structural invariant.
  This module never creates orders, fills, or trades.
  No V1 execution modules are imported.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from modules.exchange_calendar import (
    CalendarUnavailableError,
    is_trading_session,
    intended_execution_session as _calendar_next_session,
)
from modules.v2_claude_shadow import shadow_unavailable, validate_claude_shadow_output
from modules.v2_strategy import (
    V2_MODEL_VERSION,
    V2_PORTFOLIO_VERSION,
    V2_STRATEGIES,
    build_momentum_factor,
    build_quality_factor,
    build_safety_factor,
    build_value_factor,
    build_v2_factor_scores,
)
from modules.v2b_ledger import (
    ConflictError,
    CorruptionError,
    create_observation,
    get_observation_status,
    provenance_entry,
    make_ticker_record,
    transition_observation,
)
from modules.versioning import get_model_config_hash, get_portfolio_config_hash

# ── Structural invariant ──────────────────────────────────────────────────────
# Never set to False. Any code path that would set this to False violates the
# V2B boundary and must be rejected fail-closed.
_ORDER_CREATION_BLOCKED: bool = True

# Shadow strategies recorded by this runner
_FACTOR_ONLY = "Factor_Only_Core_V2"
_CLAUDE_SHADOW = "Factor_Plus_Claude_Shadow_V2"

# Max calendar-day gap between last price close and as_of_date
_DEFAULT_MAX_PRICE_STALENESS_DAYS: int = 5

# Minimum fraction of 4 factors needed for a ticker to count as "valid"
_MIN_FACTOR_COVERAGE: float = 0.25

_DATA_QUALITY_OK_COVERAGE: float = 0.90
_DATA_QUALITY_DEGRADED_COVERAGE: float = 0.70


# ── Exceptions ────────────────────────────────────────────────────────────────

class SessionRejectedError(Exception):
    """as_of_date is not a NYSE trading session (weekend, holiday, or calendar error)."""


class StaleDataError(Exception):
    """Price or fundamental data is too old or temporally invalid."""


class SafetyBoundaryError(RuntimeError):
    """Raised fail-closed when a V2B safety invariant would be violated."""


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ShadowRunResult:
    status: str  # COMPLETED | FAILED_DATA | FAILED_VALIDATION | IDEMPOTENT_MATCH | CONFLICT
    observation_key: str | None
    intended_execution_session: str | None
    quant_ticker_count: int = 0
    claude_ticker_count: int = 0
    claude_available: bool = False
    error: str | None = None
    detail: str | None = None


# ── Safety guard ──────────────────────────────────────────────────────────────

def _assert_order_creation_blocked() -> None:
    """Fail-closed check. Called at entry of every public function."""
    if not _ORDER_CREATION_BLOCKED:
        raise SafetyBoundaryError(
            "V2B invariant violated: _ORDER_CREATION_BLOCKED must always be True. "
            "This runner must never create orders, fills, or trades."
        )


# ── Session validation ────────────────────────────────────────────────────────

def validate_session(as_of_date: str) -> str:
    """
    Validate as_of_date is a NYSE trading session.

    Returns the intended_execution_session (next NYSE session after as_of_date).
    Raises SessionRejectedError for weekends, holidays, or calendar failures.
    """
    _assert_order_creation_blocked()
    try:
        if not is_trading_session(as_of_date):
            raise SessionRejectedError(
                f"as_of_date {as_of_date!r} is not a NYSE trading session "
                "(weekend, holiday, or market closed)"
            )
        return _calendar_next_session(as_of_date)
    except SessionRejectedError:
        raise
    except CalendarUnavailableError as exc:
        raise SessionRejectedError(f"NYSE calendar unavailable: {exc}") from exc


# ── Data validation ───────────────────────────────────────────────────────────

def validate_data_cutoff(data_cutoff_at: str, as_of_date: str) -> None:
    """
    Validate data_cutoff_at:
    - Must be a timezone-aware ISO 8601 datetime string
    - Must not be after the end of as_of_date in UTC (future data rejected)

    Raises StaleDataError on any violation.
    """
    _assert_order_creation_blocked()
    if not isinstance(data_cutoff_at, str) or not data_cutoff_at.strip():
        raise StaleDataError("data_cutoff_at must be a non-empty ISO 8601 string")

    try:
        normalized = data_cutoff_at.strip().replace("Z", "+00:00")
        cutoff_dt = datetime.fromisoformat(normalized)
    except (ValueError, AttributeError) as exc:
        raise StaleDataError(
            f"data_cutoff_at {data_cutoff_at!r} is not valid ISO 8601: {exc}"
        ) from exc

    if cutoff_dt.tzinfo is None:
        raise StaleDataError(
            f"data_cutoff_at {data_cutoff_at!r} must include timezone info (e.g. +00:00)"
        )

    # Reject if cutoff is after end-of-day on as_of_date
    as_of_eod = datetime.fromisoformat(f"{as_of_date}T23:59:59+00:00")
    cutoff_utc = cutoff_dt.astimezone(timezone.utc)
    as_of_utc = as_of_eod.astimezone(timezone.utc)

    if cutoff_utc > as_of_utc:
        raise StaleDataError(
            f"data_cutoff_at ({data_cutoff_at}) is after end-of-day on as_of_date "
            f"({as_of_date}) — refusing to use data from the future"
        )


def _find_stale_tickers(
    price_data: dict[str, pd.DataFrame],
    as_of_date: str,
    max_days: int,
) -> set[str]:
    """Return tickers whose most recent price is more than max_days before as_of_date."""
    stale: set[str] = set()
    as_of_ts = pd.Timestamp(as_of_date)
    for ticker, df in price_data.items():
        if df is None or df.empty:
            continue
        try:
            last_date = df.index.max()
            if last_date.tzinfo is not None:
                last_date = last_date.tz_convert("UTC").tz_localize(None)
            gap = (as_of_ts - last_date).days
            if gap > max_days:
                stale.add(ticker)
        except Exception:
            stale.add(ticker)
    return stale


# ── Factor computation ────────────────────────────────────────────────────────

def _is_finite(x: Any) -> bool:
    if x is None:
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _compute_factor_scores(
    price_data: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    as_of_date: str,
    spy_prices: pd.DataFrame | None,
    sector_map: dict[str, str] | None,
) -> pd.DataFrame:
    as_of_ts = pd.Timestamp(as_of_date)
    mom_df = build_momentum_factor(price_data, as_of_ts)
    qual_df = build_quality_factor(fundamentals)
    val_df = build_value_factor(fundamentals, sector_map)
    saf_df = build_safety_factor(price_data, as_of_ts, spy_prices)
    return build_v2_factor_scores(mom_df, qual_df, val_df, saf_df)


# ── Ticker record construction ────────────────────────────────────────────────

def _ticker_provenance(
    ticker: str,
    has_price: bool,
    has_fundamentals: bool,
    as_of_date: str,
    data_cutoff_at: str,
) -> list[dict]:
    entries = []
    if has_price:
        entries.append(provenance_entry(
            "point_in_time",
            "yfinance_daily_prices",
            as_of_date=as_of_date,
            data_cutoff_at=as_of_date,
            note="momentum and safety factors",
        ))
    else:
        entries.append(provenance_entry(
            "unavailable",
            "yfinance_daily_prices",
            note="price data not available for this ticker",
        ))
    if has_fundamentals:
        entries.append(provenance_entry(
            "current_snapshot",
            "yfinance_info",
            source_timestamp=data_cutoff_at,
            data_cutoff_at=as_of_date,
            note="quality and value factors — NOT point-in-time",
        ))
    else:
        entries.append(provenance_entry(
            "unavailable",
            "yfinance_info",
            note="fundamentals not available for this ticker",
        ))
    return entries


def _build_ticker_records(
    scores_df: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    sector_map: dict[str, str] | None,
    as_of_date: str,
    data_cutoff_at: str,
    selected_factor_only: list[str],
    selected_claude: list[str],
) -> list[dict]:
    if scores_df.empty:
        return []

    # Assign rank by score_core_v2 (rank 1 = best)
    df = scores_df.copy()
    if "score_core_v2" in df.columns:
        df["_rank"] = (
            df["score_core_v2"]
            .rank(ascending=False, na_option="bottom", method="first")
            .astype(int)
        )
    else:
        df["_rank"] = len(df)

    records = []
    selected_fo_set = set(selected_factor_only)
    selected_cs_set = set(selected_claude)

    for _, row in df.iterrows():
        ticker = str(row["ticker"])
        has_price = ticker in price_data and price_data[ticker] is not None and not price_data[ticker].empty
        has_fund = ticker in fundamentals and bool(fundamentals[ticker])

        strategies_selected = []
        if ticker in selected_fo_set:
            strategies_selected.append(_FACTOR_ONLY)
        if ticker in selected_cs_set:
            strategies_selected.append(_CLAUDE_SHADOW)

        factor_avail = {
            "momentum": bool(row.get("momentum_available", False)),
            "quality": bool(row.get("quality_available", False)),
            "value": bool(row.get("value_available", False)),
            "safety": bool(row.get("safety_available", False)),
        }

        def _score(col: str) -> float | None:
            v = row.get(col)
            return float(v) if _is_finite(v) else None

        composite = _score("score_core_v2")
        coverage = float(row.get("factor_coverage", 0.0))
        rank_val = int(row["_rank"])
        vsadj = bool(row.get("value_sector_adjusted", False))
        sector = sector_map.get(ticker) if sector_map else None

        # raw_factor_inputs: factor scores (0-100); None entries need unavailable reasons
        raw_inputs: dict[str, Any] = {}
        raw_reasons: dict[str, str] = {}
        for factor, score_col, avail_col in [
            ("momentum", "momentum_score", "momentum_available"),
            ("quality", "quality_score", "quality_available"),
            ("value", "value_score", "value_available"),
            ("safety", "safety_score", "safety_available"),
        ]:
            avail = bool(row.get(avail_col, False))
            s = row.get(score_col)
            if avail and _is_finite(s):
                raw_inputs[factor] = float(s)
            else:
                raw_inputs[factor] = None
                raw_reasons[factor] = "factor unavailable or score non-finite"

        provenance = _ticker_provenance(ticker, has_price, has_fund, as_of_date, data_cutoff_at)

        try:
            rec = make_ticker_record(
                ticker,
                raw_factor_inputs=raw_inputs,
                raw_factor_unavailable_reasons=raw_reasons,
                factor_availability=factor_avail,
                scores={
                    "momentum": raw_inputs.get("momentum"),
                    "quality": raw_inputs.get("quality"),
                    "value": raw_inputs.get("value"),
                    "safety": raw_inputs.get("safety"),
                    "composite": composite,
                },
                factor_coverage=coverage,
                rank=rank_val,
                excluded=False,
                exclusion_reason=None,
                selected_by_strategy=strategies_selected,
                sector=sector,
                value_sector_adjusted=vsadj,
                provenance=provenance,
            )
            records.append(rec)
        except Exception:
            # Skip tickers whose records fail validation (data quality guard)
            pass

    return records


# ── Data quality label ────────────────────────────────────────────────────────

def _data_quality_label(valid_count: int, total_count: int, stale_count: int) -> str:
    if total_count == 0:
        return "no_data"
    coverage = valid_count / total_count
    if stale_count > 0:
        return "degraded_stale"
    if coverage >= _DATA_QUALITY_OK_COVERAGE:
        return "ok"
    if coverage >= _DATA_QUALITY_DEGRADED_COVERAGE:
        return "degraded"
    return "poor"


# ── Observation completion ────────────────────────────────────────────────────

def _advance_to_completed(observation_key: str) -> str:
    """
    Advance a CREATED or COLLECTING observation to COMPLETED.
    Used for crash recovery when get IDEMPOTENT_MATCH on an incomplete observation.
    Returns the final status string.
    """
    current = get_observation_status(observation_key)
    if current in ("COMPLETED", "FAILED_DATA", "FAILED_VALIDATION", "CANCELLED", "CONFLICT"):
        return current

    if current == "CREATED":
        transition_observation(observation_key, "COLLECTING")
    transition_observation(observation_key, "COMPLETED")
    return "COMPLETED"


# ── Main entry point ──────────────────────────────────────────────────────────

def run_shadow_observation(
    as_of_date: str,
    price_data: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    data_cutoff_at: str,
    spy_prices: pd.DataFrame | None = None,
    sector_map: dict[str, str] | None = None,
    claude_outputs: dict[str, dict] | None = None,
    metadata: dict | None = None,
    *,
    model_version: str = V2_MODEL_VERSION,
    model_config_hash: str | None = None,
    portfolio_version: str = V2_PORTFOLIO_VERSION,
    portfolio_config_hash: str | None = None,
    max_price_staleness_days: int = _DEFAULT_MAX_PRICE_STALENESS_DAYS,
) -> ShadowRunResult:
    """
    Run a daily V2B shadow observation.

    Accepted parameters:
        as_of_date:               Signal generation date ("YYYY-MM-DD"). Must be NYSE session.
        price_data:               ticker → DataFrame with Adj Close/Close and Volume columns.
        fundamentals:             ticker → yfinance .info dict.
        data_cutoff_at:          ISO 8601+tz timestamp when data was retrieved. Must not be future.
        spy_prices:               SPY daily prices for beta calculation (optional).
        sector_map:               ticker → sector string for sector-relative value (optional).
        claude_outputs:           ticker → raw Claude shadow output dict (optional).
        metadata:                 Caller-supplied observation context (optional).
        model_version:            V2 model version (default: quant_baseline_v2).
        model_config_hash:        64-char hex SHA-256 of model config. If omitted, loaded
                                  from versioning.py.
        portfolio_version:        V2 portfolio version (default: risk_managed_factor_v2).
        portfolio_config_hash:    64-char hex SHA-256 of portfolio config (optional).
        max_price_staleness_days: Max calendar days between last close and as_of_date.

    Safety invariants (enforced fail-closed):
    - _ORDER_CREATION_BLOCKED is always True
    - No V1 execution functions are called
    - No orders, fills, trades, cash, or positions are modified
    """
    # ── Safety ────────────────────────────────────────────────────────────────
    _assert_order_creation_blocked()

    # ── Config hashes ─────────────────────────────────────────────────────────
    if not model_config_hash:
        model_config_hash = get_model_config_hash(model_version)
    if not model_config_hash:
        return ShadowRunResult(
            status="FAILED_VALIDATION",
            observation_key=None,
            intended_execution_session=None,
            error=f"model_config_hash is required but could not be loaded for {model_version!r}",
        )

    if not portfolio_config_hash:
        portfolio_config_hash = get_portfolio_config_hash(portfolio_version) or ""

    # ── Session validation ─────────────────────────────────────────────────────
    # Raises SessionRejectedError — caller handles exit 0 for non-trading days
    intended_session = validate_session(as_of_date)

    # ── Data cutoff validation ─────────────────────────────────────────────────
    try:
        validate_data_cutoff(data_cutoff_at, as_of_date)
    except StaleDataError as exc:
        return ShadowRunResult(
            status="FAILED_DATA",
            observation_key=None,
            intended_execution_session=intended_session,
            error=str(exc),
        )

    # ── Ticker universe ────────────────────────────────────────────────────────
    all_tickers = sorted(set(list(price_data) + list(fundamentals)))
    if not all_tickers:
        return ShadowRunResult(
            status="FAILED_DATA",
            observation_key=None,
            intended_execution_session=intended_session,
            error="No tickers provided — cannot create shadow observation",
        )

    # ── Staleness check ────────────────────────────────────────────────────────
    stale_tickers = _find_stale_tickers(price_data, as_of_date, max_price_staleness_days)
    if len(stale_tickers) == len([t for t in all_tickers if t in price_data]):
        # Every ticker with price data is stale — treat as data failure
        return ShadowRunResult(
            status="FAILED_DATA",
            observation_key=None,
            intended_execution_session=intended_session,
            error=f"All {len(stale_tickers)} ticker(s) with price data are stale "
                  f"(last close > {max_price_staleness_days} days before {as_of_date})",
        )

    # ── Factor computation ────────────────────────────────────────────────────
    # Exclude stale tickers from price-derived factor computation
    clean_price = {t: df for t, df in price_data.items() if t not in stale_tickers}
    scores_df = _compute_factor_scores(
        clean_price, fundamentals, as_of_date, spy_prices, sector_map
    )

    if scores_df.empty:
        return ShadowRunResult(
            status="FAILED_DATA",
            observation_key=None,
            intended_execution_session=intended_session,
            error="Factor computation produced no scored tickers",
        )

    # ── Strategy selection ────────────────────────────────────────────────────
    buy_top_n = V2_STRATEGIES[_FACTOR_ONLY].get("buy_top_n", 25)
    if "score_core_v2" in scores_df.columns:
        valid_scores = scores_df.dropna(subset=["score_core_v2"]).sort_values(
            "score_core_v2", ascending=False
        )
    else:
        valid_scores = scores_df.head(0)
    selected_factor_only = valid_scores["ticker"].head(buy_top_n).tolist()

    # ── Claude shadow validation ───────────────────────────────────────────────
    validated_claude: dict[str, dict] = {}
    if claude_outputs:
        try:
            cutoff_norm = data_cutoff_at.strip().replace("Z", "+00:00")
            cutoff_utc = datetime.fromisoformat(cutoff_norm).astimezone(timezone.utc)
        except Exception:
            cutoff_utc = None

        for ticker, raw_output in claude_outputs.items():
            validated = validate_claude_shadow_output(raw_output)

            # Reject if provenance invalid
            if not validated.get("provenance_valid"):
                failures = validated.get("provenance_failures", [])
                validated_claude[ticker] = shadow_unavailable(
                    f"provenance_invalid: {'; '.join(str(f) for f in failures)}"
                )
                continue

            # Reject if Claude's own data_cutoff_at postdates runner's data cutoff.
            # This means Claude used data that is newer than our cutoff — reject.
            if cutoff_utc is not None:
                claude_cutoff_str = validated.get("data_cutoff_at")
                if claude_cutoff_str:
                    try:
                        claude_cutoff_utc = datetime.fromisoformat(
                            claude_cutoff_str.replace("Z", "+00:00")
                        ).astimezone(timezone.utc)
                        if claude_cutoff_utc > cutoff_utc:
                            validated_claude[ticker] = shadow_unavailable(
                                "claude_data_cutoff_after_runner_cutoff"
                            )
                            continue
                    except (ValueError, AttributeError):
                        pass

            validated_claude[ticker] = validated

    # Tickers with genuinely valid Claude output (provenance_valid=True)
    claude_ok_tickers = [t for t, v in validated_claude.items() if v.get("provenance_valid")]
    claude_available = len(claude_ok_tickers) > 0
    # Factor_Plus_Claude_Shadow_V2 is only recorded when valid Claude output exists
    selected_claude = selected_factor_only if claude_available else []

    # ── Count statistics ──────────────────────────────────────────────────────
    scored_tickers = set(scores_df["ticker"].tolist()) if not scores_df.empty else set()
    total_count = len(all_tickers)
    valid_count = int(
        (scores_df["factor_coverage"] >= _MIN_FACTOR_COVERAGE).sum()
        if "factor_coverage" in scores_df.columns else 0
    )
    stale_count = len(stale_tickers)
    missing_count = len(set(all_tickers) - scored_tickers)
    excluded_count = 0
    signal_coverage = valid_count / total_count if total_count > 0 else 0.0
    dq_status = _data_quality_label(valid_count, total_count, stale_count)

    represents_all = (missing_count == 0 and stale_count == 0)

    # ── Build ticker records ──────────────────────────────────────────────────
    ticker_records = _build_ticker_records(
        scores_df,
        price_data,
        fundamentals,
        sector_map,
        as_of_date,
        data_cutoff_at,
        selected_factor_only,
        selected_claude,
    )

    # ── selected_tickers_per_strategy ─────────────────────────────────────────
    selected_per_strategy: dict[str, list[str]] = {_FACTOR_ONLY: selected_factor_only}
    if claude_available:
        selected_per_strategy[_CLAUDE_SHADOW] = selected_claude

    # ── Observation metadata ──────────────────────────────────────────────────
    obs_metadata: dict[str, Any] = dict(metadata or {})
    obs_metadata["runner_version"] = "v2b_shadow_runner_v1"
    obs_metadata["as_of_date"] = as_of_date
    obs_metadata["data_cutoff_at"] = data_cutoff_at
    obs_metadata["portfolio_version"] = portfolio_version
    obs_metadata["portfolio_config_hash"] = portfolio_config_hash or ""
    obs_metadata["stale_tickers"] = sorted(stale_tickers)
    obs_metadata["claude_shadow"] = {
        "available": claude_available,
        "ok_ticker_count": len(claude_ok_tickers),
        "per_ticker": {t: v for t, v in validated_claude.items()},
    }

    # ── Create observation (idempotent) ───────────────────────────────────────
    try:
        event = create_observation(
            model_version=model_version,
            model_config_hash=model_config_hash,
            intended_execution_session=intended_session,
            universe_count=total_count,
            valid_ticker_count=valid_count,
            stale_ticker_count=stale_count,
            excluded_ticker_count=excluded_count,
            missing_ticker_count=missing_count,
            signal_coverage_rate=signal_coverage,
            selected_tickers_per_strategy=selected_per_strategy,
            data_quality_status=dq_status,
            ticker_records=ticker_records,
            ticker_records_represents_all_scored=represents_all,
            metadata=obs_metadata,
        )
    except ConflictError as exc:
        return ShadowRunResult(
            status="CONFLICT",
            observation_key=None,
            intended_execution_session=intended_session,
            quant_ticker_count=len(ticker_records),
            claude_available=claude_available,
            error=str(exc),
        )
    except (CorruptionError, Exception) as exc:
        return ShadowRunResult(
            status="FAILED_VALIDATION",
            observation_key=None,
            intended_execution_session=intended_session,
            error=f"Observation creation failed: {exc}",
        )

    obs_key = event["observation_key"]

    # ── Handle idempotent match (including crash recovery) ───────────────────
    if event["event_type"] == "IDEMPOTENT_MATCH":
        current_status = get_observation_status(obs_key)
        if current_status not in ("COMPLETED", "FAILED_DATA", "FAILED_VALIDATION",
                                   "CANCELLED", "CONFLICT"):
            # Crash recovery: observation exists but was not completed
            _advance_to_completed(obs_key)
            return ShadowRunResult(
                status="COMPLETED",
                observation_key=obs_key,
                intended_execution_session=intended_session,
                quant_ticker_count=len(ticker_records),
                claude_ticker_count=len(claude_ok_tickers),
                claude_available=claude_available,
            )
        return ShadowRunResult(
            status="IDEMPOTENT_MATCH",
            observation_key=obs_key,
            intended_execution_session=intended_session,
            quant_ticker_count=len(ticker_records),
            claude_ticker_count=len(claude_ok_tickers),
            claude_available=claude_available,
        )

    # ── Transition new observation to COMPLETED ───────────────────────────────
    try:
        transition_observation(obs_key, "COLLECTING")
        transition_observation(obs_key, "COMPLETED")
    except Exception as exc:
        try:
            transition_observation(obs_key, "FAILED_VALIDATION", note=str(exc))
        except Exception:
            pass
        return ShadowRunResult(
            status="FAILED_VALIDATION",
            observation_key=obs_key,
            intended_execution_session=intended_session,
            quant_ticker_count=len(ticker_records),
            claude_available=claude_available,
            error=str(exc),
        )

    return ShadowRunResult(
        status="COMPLETED",
        observation_key=obs_key,
        intended_execution_session=intended_session,
        quant_ticker_count=len(ticker_records),
        claude_ticker_count=len(claude_ok_tickers),
        claude_available=claude_available,
    )


# ── Telegram preview (V2 SHADOW — never sent automatically) ──────────────────

def format_shadow_preview(result: ShadowRunResult) -> str:
    """
    Format a labeled V2 SHADOW preview. NOT sent to production Telegram.
    Clearly marked to prevent confusion with V1 production messages.
    For logging or future manual inspection only.
    """
    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🔬 V2 SHADOW — NOT A PRODUCTION SIGNAL",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"Status:  {result.status}",
        f"Session: {result.intended_execution_session}",
        f"Tickers: {result.quant_ticker_count}",
        f"Claude:  {'yes' if result.claude_available else 'no'} "
        f"({result.claude_ticker_count} ticker(s))",
    ]
    if result.error:
        lines.append(f"Error:   {result.error}")
    if result.observation_key:
        lines.append(f"Key:     {result.observation_key[:16]}…")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("⚠  This is a shadow observation. No orders were created.")
    return "\n".join(lines)
