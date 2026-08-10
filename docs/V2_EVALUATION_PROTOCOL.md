# Quant Baseline V2 — Evaluation Protocol

**Pre-registered:** 2026-08-10
**Status:** V2A research
**Warning:** Do not adjust model parameters after observing out-of-sample data.

---

## Period Definitions

| Period | Start | End | Purpose |
|--------|-------|-----|---------|
| In-sample | 2015-01-01 | 2020-12-31 | Model development only |
| Validation | 2021-01-01 | 2022-12-31 | Parameter validation (no tuning after viewing) |
| Out-of-sample | 2023-01-01 | Rolling | Never used for parameter selection |

**Rule:** Factor weights (pre-registered 2026-08-10) cannot be changed based on
validation or out-of-sample results. A new model version must be created and
registered under a new `model_version` key.

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
- Excess annualized return vs SPY
- Excess annualized return vs QQQ

### Risk Metrics
- Annualized volatility
- Sharpe ratio (risk-free = 4% annual, updated to current T-bill when available)
- Sortino ratio
- Maximum drawdown
- Calmar ratio (annualized return / |max drawdown|)
- Beta vs SPY
- Downside capture ratio vs SPY

### Activity Metrics
- Average number of positions
- Estimated annual turnover (% of portfolio value)
- Estimated annual cost (turnover × 10 bps round-trip)
- Hit rate (fraction of positive-return trading days)
- Sector concentration (average HHI)

### Data Quality Metrics
- Factor coverage (per ticker, averaged across portfolio)
- Claude availability (0.0 in V2A — not yet connected)
- Point-in-time fundamentals: False for quality/value
- Survivorship bias: always noted as "present"

---

## Comparison: Factor-Only vs Claude-Shadow

The primary purpose of the shadow design is to isolate Claude's contribution.
Comparison methodology:

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
- No tax drag (paper trading)

---

## Promotion Criteria

V2 can only be promoted to live production when ALL of the following are met:

1. **Minimum out-of-sample period:** 6 months of live paper trading
2. **Evidence quality:** `sufficient_evidence=True` for all three strategies
3. **Survivorship bias:** Explicitly reported in all results
4. **Factor coverage:** Reported per-strategy; not hidden
5. **Sharpe vs SPY:** Core strategy Sharpe > SPY Sharpe in out-of-sample period
6. **Drawdown discipline:** Max drawdown does not exceed SPY's by more than 25%
7. **Independent review:** Results reviewed before promotion
8. **No overfitting:** Factor weights unchanged from pre-registration date

**Winner selection:** The strategy with the highest out-of-sample Sharpe ratio
(NOT in-sample Sharpe) advances. If no strategy meets the drawdown criterion,
promotion is deferred.

---

## Anti-Overfitting Rules

1. Factor weights are fixed as of 2026-08-10. Do not change them.
2. Do not observe out-of-sample results before finalizing model parameters.
3. If validation results are disappointing, create a new model version with
   new weights — do not modify `quant_baseline_v2`.
4. Do not mine the out-of-sample period for patterns to "fix" the model.
5. Report all three strategies — do not cherry-pick the best performer.

---

## Reporting Insufficient Evidence

When data is insufficient (< 252 trading days, missing factor data, etc.):

- Report `evidence_note` from `EvaluationMetrics`
- Do not produce a winner ranking based on incomplete data
- Label results `insufficient_evidence` in all outputs
- Do not extrapolate from short periods
