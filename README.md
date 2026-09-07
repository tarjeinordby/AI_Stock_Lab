# AI Stock Lab

A personal systematic trading system and quantitative research platform, running daily in production since 2026.

---

## What it is

AI Stock Lab is a fully automated equity signal generation and execution system built on a multi-factor quantitative model. It runs on a daily GitHub Actions schedule, selects positions across a US large-cap universe, and manages a live portfolio with structured risk controls.

Alongside the live system, a parallel **shadow research layer (V2B)** runs prospective out-of-sample studies on next-generation strategies — collecting signals and tracking their forward returns without touching production state. The two systems are architecturally isolated by design, not just convention.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│               V1 — Production               │
│  Factor scoring → Execution → Fill ledger   │
└─────────────────────────────────────────────┘
          │ read-only observation
          ▼
┌─────────────────────────────────────────────┐
│            V2B — Shadow Research            │
│  Signal ledger → Outcome tracking → Report  │
│                                             │
│  Cannot: create orders · fills · trades     │
│          modify cash · positions · V1 state │
└─────────────────────────────────────────────┘
```

---

## Factor model

Four factors, pre-registered weights, no look-ahead optimization:

| Factor | Captures |
|--------|----------|
| Momentum | Price trend over 12 months, adjusted for 1-month reversal |
| Quality | Return on equity, earnings stability |
| Value | Sector-relative EV/EBITDA inverse |
| Safety | Beta vs SPY, low-volatility tilt |

Scores are computed with explicit `feature_available` flags — no silent median imputation on missing data.

---

## V2B — Signal-outcome event study

V2B runs a **prospective event study**, not a backtest. Each signal observation is committed to an append-only ledger on the day it is generated, then resolved to a forward return at the exit session.

- Holding periods: **1, 5, 21, and 63 NYSE sessions**
- Benchmark: SPY (forward alpha and hit-rate)
- Outcome states: `OUTCOME_RECORDED` / `OUTCOME_INCOMPLETE` / `OUTCOME_UNAVAILABLE`
- Grace period: 5 NYSE sessions after exit before declaring data permanently absent
- Aggregated returns are only computed when all tickers **and** SPY are complete — partial data produces a separate available-subset return rather than corrupting the aggregate

Every terminal event is validated against the original `OUTCOME_PENDING` record before any disk write. An invalid payload raises `OutcomeValidationError` — no partial commits.

---

## Engineering notes

**Immutable event ledger with SHA-256 hash chains**
Each event stores a hash of its own body and the hash of the preceding event. A broken chain raises `CorruptionError` and halts the process — fail-closed, no silent recovery.

**Single global lock**
All ledger operations (read → validate → transition → write → fsync) run under one `fcntl.LOCK_EX` lock. No partial writes, no torn reads.

**V1 isolation enforced structurally**
V2B modules have no static imports from V1 execution modules. This is verified by an AST-based static analysis test that runs in CI on every push.

**Provider failure vs. permanent absence**
If an entire yfinance response is empty, the outcome stays `PENDING` and retries the next day. Only partial data (some tickers missing, benchmark present) triggers the grace-period countdown.

---

## Automation

| Workflow | Schedule | Purpose |
|----------|----------|---------|
| `execute.yml` | 21:30 UTC Mon–Fri | Signal generation, order placement, fill recording |
| `premarket.yml` | 13:00 UTC Mon–Fri | Pre-market data collection |
| `signal.yml` | Daily | Factor scoring, universe filtering, strategy ranking |
| `v2b_shadow.yml` | 21:30 UTC Mon–Fri | Shadow collection + outcome tracking (two independent jobs) |

All secrets (Telegram, Anthropic API) are injected as GitHub Secrets at runtime. No credentials are stored in the repository.

---

## Stack

Python 3.12 · pandas · numpy · yfinance · anthropic · exchange_calendars · PyYAML · pytest

**1 269 tests · 0 failures**

---

## Development phases

- [x] **V1** — Live factor-based portfolio with daily execution and Telegram reporting
- [x] **V2B.1–3** — Shadow observation ledger, shadow reporting
- [ ] **V2B.4** — Signal-outcome event study *(under independent review)*
- [ ] **Del B** — Statistical evaluation and V2 promotion decision
