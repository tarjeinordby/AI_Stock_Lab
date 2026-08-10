# Quant Baseline V2 — Evaluation Protocol

**Pre-registered:** 2026-08-10
**Status:** V2A research
**Warning:** Do not adjust model parameters after observing out-of-sample data.

---

## Period Definitions

The model was registered on **2026-08-10**. Four distinct periods apply:

| Period | Start | End | `is_truly_oos` | Purpose |
|--------|-------|-----|----------------|---------|
| `in_sample` | 2015-01-01 | 2020-12-31 | No | Model development — weights were set here |
| `validation` | 2021-01-01 | 2022-12-31 | No | Sanity check only — no tuning after viewing |
| `retrospective_holdout` | 2023-01-01 | 2026-08-09 | **No** | Observed before registration — context only |
| `prospective_out_of_sample` | **2026-08-11** | Rolling | **Yes** | Only period valid for promotion decisions |

**Critical distinctions:**

- `retrospective_holdout` (2023-01-01 – 2026-08-09): This data existed and was observable
  when the model was registered on 2026-08-10. It was **not** used to set factor weights,
  but it cannot be considered truly out-of-sample because it was available to the researcher.
  Results from this period are reported for context only and **cannot substitute for
  prospective OOS evidence** in promotion decisions.

- `prospective_out_of_sample`: Starts 2026-08-11 — the first NYSE session after model
  registration. **Only this period counts toward promotion.** Earliest possible promotion
  date: **2027-02-11** (6 months after registration).

**Rule:** Factor weights (pre-registered 2026-08-10) cannot be changed based on
any observed results. To change weights, create a new model version under a new
`model_version` key with a new registration date.

---

## Strategies Under Evaluation

1. Core_Quality_Momentum_V2 (Factor Only)
2. Aggressive_Momentum_V2
3. Defensive_Quality_V2
4. Factor_Only_Core_V2 (shadow control)
5. Factor_Plus_Claude_Shadow_V2 (shadow, Claude unavailable in V2A)
6. SPY (benchmark)
7. QQQ (benchmark)
8. MegaCap comparison portfolio (optional)

---

## Required Metrics

For each strategy and each period:

### Return Metrics
- Annualized return
- Total return
- Excess annualized return vs SPY (aligned to strategy dates)
- Excess annualized return vs QQQ (aligned to strategy dates)

### Risk Metrics
- Annualized volatility
- Sharpe ratio (risk-free = 4% annual, updated to current T-bill when available)
- Sortino ratio
- Maximum drawdown
- Calmar ratio (annualized return / |max drawdown|)
- Beta vs SPY
- Downside capture ratio vs SPY

### Activity Metrics (requires position-level data — None until live runner)
- Average number of positions
- Estimated annual turnover (% of portfolio value)
- Estimated annual cost (turnover × 10 bps round-trip)
- Sector concentration (average HHI)

### Data Quality Metrics
- Factor coverage (per ticker, averaged across portfolio)
- Claude availability (0.0 in V2A — not yet connected)
- Point-in-time fundamentals: False for quality/value factors
- Survivorship bias: always "present — current universe only"

---

## Insufficient Evidence — No Ranking

When `sufficient_evidence=False` (fewer than 252 trading days, failed Sharpe
computation, etc.):

- The strategy is listed last in `compare_strategies()` output with `rank=None`
- A strategy with `sufficient_evidence=False` and Sharpe 9.0 **must not** rank above
  a strategy with `sufficient_evidence=True` and Sharpe 1.0
- Results are labeled `insufficient_evidence` — no winner is declared
- Do not extrapolate from short periods or incomplete data

---

## Backtest Limitations

**Full four-factor backtest: UNSUPPORTED**

A backtest using quality and value factors requires point-in-time fundamental data
(e.g., Compustat or FactSet). yfinance `.info` returns current values only. Using
current fundamentals for historical periods introduces look-ahead bias. Running
`run_v2_backtest()` raises `NotImplementedError`.

**Momentum + safety only backtest: SUPPORTED**

Both factors are computed entirely from price data, which is available as-of any
historical date. Survivorship bias still applies. Results are directional only.
This variant can be run without look-ahead bias.

**Promotion is always blocked** (`BACKTEST_STATUS["promotion_blocked"] = True`)
until 6 months of prospective paper-trading data are available.

---

## Comparison: Factor-Only vs Claude-Shadow

1. Factor_Only_Core_V2 and Factor_Plus_Claude_Shadow_V2 use identical factor scores
2. Any difference in logged signal quality is attributed to Claude
3. Claude cannot create orders — comparison is signal-quality only in V2A
4. Minimum 3 months of shadow logs required before drawing conclusions

---

## Cost Assumptions

- One-way transaction cost: 5 basis points (0.05%)
- Round-trip cost: 10 basis points
- No market impact modeled (paper trading, small capital)
- No financing costs (fully invested or cash, no leverage)

---

## Promotion Criteria

V2 can only be promoted to live production when **ALL** of the following are met:

1. **Period:** Results must be from `prospective_out_of_sample` only
2. **Minimum OOS data:** ≥ 6 months of prospective paper trading (earliest: 2027-02-11)
3. **Sufficient evidence:** `sufficient_evidence=True` for the strategy
4. **Sharpe vs SPY:** Strategy Sharpe exceeds SPY Sharpe by ≥ 0.10 in the same period
5. **Drawdown discipline:** Max drawdown ≤ 1.25 × SPY max drawdown
6. **Alpha months:** ≥ 50% of months with positive alpha vs SPY
7. **Survivorship bias:** Disclosed and noted in all results
8. **Factor coverage:** ≥ 60% average factor coverage
9. **Point-in-time fundamentals:** Must be True (blocks all current V2A promotion)
10. **No overfitting:** Factor weights unchanged from 2026-08-10

**Winner selection:** Highest `prospective_out_of_sample` Sharpe among strategies
meeting all criteria. If no strategy passes all criteria, promotion is deferred.

---

## Anti-Overfitting Rules

1. Factor weights are fixed as of 2026-08-10. Do not change them.
2. Do not use retrospective_holdout results for model selection.
3. If performance is disappointing, create a new model version — do not modify
   `quant_baseline_v2` weights.
4. Report all three strategies — do not cherry-pick the best performer.
5. `prospective_out_of_sample` data must never be used to tune parameters.
