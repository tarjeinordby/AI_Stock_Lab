# V2B Shadow Observation Ledger

**Module:** `modules/v2b_ledger.py`
**Status:** V2B.1 — active
**Registered:** 2026-08-11

---

## Purpose

The V2B shadow observation ledger is an immutable append-only event store for prospective shadow observations of the V2 model's scoring process. It records what the model saw (factor scores, coverage, rankings, exclusions) on each execution session without ever influencing live orders or positions.

**Absolute invariant:** `order_creation_blocked = True` is hardcoded and never overridable.

---

## File Layout

```
data_v4/v2b_ledger/
  {YYYY-MM}_v2b_observations.jsonl   — append-only event log, one JSON per line
  {YYYY-MM}_v2b_observations.lock    — exclusive file lock for concurrent writers
  v2b_idx.json                       — observation_key → YYYY-MM index
```

One JSONL file per calendar month, partitioned by `intended_execution_session`.

---

## Observation Key

```
observation_key = SHA-256(model_version + "|" + model_config_hash + "|" + intended_execution_session)
```

The key uniquely identifies one intended scoring run. It is deterministic — the same model, config, and session always produce the same key.

---

## Event Schema

Every event written to the JSONL has these common fields:

| Field | Type | Description |
|-------|------|-------------|
| `event_type` | string | One of the event types below |
| `record_version` | string | Schema version ("1") |
| `observation_key` | string | SHA-256 hex, 64 chars |
| `observation_run_id` | UUID4 | Unique per attempt (same across events for one observation) |
| `timestamp_utc` | ISO 8601+tz | When the event was written |
| `order_creation_blocked` | bool | Always `true` — structural invariant |

`OBSERVATION_CREATED` events additionally carry:

| Field | Type | Description |
|-------|------|-------------|
| `ledger_version` | string | Ledger schema version |
| `content_hash` | string | SHA-256 of canonical JSON payload |
| `model_version` | string | e.g. `"quant_baseline_v2"` |
| `model_config_hash` | string | SHA-256 of model config at time of observation |
| `intended_execution_session` | string | `YYYY-MM-DD` of the scoring session |
| `universe_count` | int | Number of tickers in the scored universe |
| `signal_coverage` | float | Fraction of universe with complete signals |
| `ticker_records` | array | Per-ticker observation records (see below) |
| `metadata` | object | Caller-supplied context |
| `point_in_time_fundamentals_global` | bool | Always `false` — see Per-Ticker Provenance |

---

## Per-Ticker Record

Each element of `ticker_records` contains:

```json
{
  "ticker": "AAPL",
  "scores": {
    "momentum": 75.0,
    "quality": 60.0,
    "value": 50.0,
    "safety": 80.0,
    "composite": 70.0
  },
  "factor_coverage": 0.75,
  "rank": 1,
  "excluded": false,
  "exclusion_reason": null,
  "sector": "Technology",
  "value_sector_adjusted": true,
  "provenance": [
    {"type": "point_in_time", "source": "price_history", "as_of_date": "2026-08-11"},
    {"type": "current_snapshot", "source": "yfinance", "note": "quality factor"}
  ]
}
```

### Provenance Types

| Type | Meaning |
|------|---------|
| `point_in_time` | Data available as-of the execution session date (price-derived factors) |
| `current_snapshot` | yfinance current values — not point-in-time; introduces look-ahead in backtests |
| `unavailable` | Factor could not be computed for this ticker |
| `unknown` | Data provenance not determined |

**`point_in_time_fundamentals_global` is always `false`.** Quality and value factors use yfinance `.info` current snapshots, not Compustat/FactSet point-in-time data. The ledger records per-field provenance to track which sub-factors are genuinely PIT and which are not.

---

## Status Machine

```
CREATED ──► COLLECTING ──► COMPLETED
         │             ├──► FAILED_DATA
         │             ├──► FAILED_VALIDATION
         │             └──► CANCELLED
         └──► CONFLICT (terminal — raises ConflictError)
```

All terminal states (COMPLETED, FAILED_DATA, FAILED_VALIDATION, CANCELLED, CONFLICT) accept no further transitions.

---

## Idempotency and Conflict

| Scenario | Result |
|----------|--------|
| Same `observation_key` + same `content_hash` | Returns `IDEMPOTENT_MATCH` event; no new line written |
| Same `observation_key` + different `content_hash` | Writes `CONFLICT` event; raises `ConflictError` |
| New `observation_key` | Writes `OBSERVATION_CREATED` event |

The content hash covers the full canonical payload (model metadata, universe_count, signal_coverage, all ticker_records, and metadata). Any change to any field produces a different hash and triggers a conflict on replay.

---

## Concurrency and Persistence

- **File locking:** `fcntl.LOCK_EX` on a dedicated `.lock` file guards every append.
- **fsync:** Each append flushes to disk before releasing the lock.
- **Atomic writes:** Events are written as complete lines; the lock is held for the duration of the append.
- **Corrupt last line:** Incomplete JSON on the last line of a JSONL file (e.g., from an interrupted write) triggers a warning and is skipped.
- **Mid-file corruption:** Any non-last-line JSON parse failure raises `CorruptionError` immediately — fail-closed.

---

## V1 / V2B Boundary

The V2B ledger:
- Does **not** import any V1 module (`modules/ledger.py`, `modules/orders.py`, `modules/fills.py`, `modules/portfolio.py`, `modules/state.py`, etc.)
- Does **not** read or write V1 state, signals, positions, orders, fills, or cash
- Does **not** call `execute_buy`, `execute_sell`, or `execute_pyramid_fill`
- Cannot influence any V1 production workflow

---

## Future: GitHub Actions Persistence

The JSONL files in `data_v4/v2b_ledger/` are designed to be committed to the repository after each scoring session via a GitHub Actions workflow. This provides:
- Tamper-evident history (git SHA chain over immutable events)
- Durable audit trail for the prospective OOS period (2026-08-11+)
- Easy review of per-session observations as pull request diffs

When the Actions workflow is implemented, each run should commit only the delta lines appended in that session, using an atomic append pattern identical to the local file-locking approach.

---

## API Summary

```python
from modules.v2b_ledger import (
    create_observation,      # Open a new shadow observation
    transition_observation,  # Advance status (COLLECTING → COMPLETED, etc.)
    get_observation_status,  # Current status for a key
    get_observation_events,  # All events for a key (in append order)
    list_observations,       # Summary list for a YYYY-MM (or all months)
    make_observation_key,    # SHA-256(model_version|config_hash|session)
    make_content_hash,       # SHA-256 of canonical JSON payload
    make_ticker_record,      # Construct a per-ticker observation dict
    provenance_entry,        # Construct a provenance dict
    assert_order_creation_blocked,  # Verify structural invariant
)
```

### Exceptions

| Exception | When raised |
|-----------|-------------|
| `ConflictError` | Same key, different content hash |
| `CorruptionError` | Mid-file JSON parse failure (fail-closed) |
| `InvalidTransitionError` | Illegal status transition |
