# Quant Baseline V2 — Design Document

**Status:** Research / Paper Trading (V2A foundation)
**Branch:** feature/quant-baseline-v2
**Date:** 2026-08-10

---

## Scope

V2A establishes the research foundation: strategy definitions, factor model,
correct portfolio allocation, Claude shadow schema, and evaluation protocol.
V2 is not active in production. It cannot affect V1 orders or fills.

---

## Why V2?

V1 has three known limitations this research aims to address:

1. **No explicit risk factor in scoring.** V1 uses risk-parity position sizing,
   but risk characteristics do not enter the signal score itself. A high-volatility
   stock with strong momentum gets the same score as a low-volatility stock with
   the same momentum and quality profile.

2. **Position weight cap bug.** `build_target_weights()` clips weights at
   `max_position_weight`, then renormalizes to 100%. The renormalization can push
   weights back above the cap. V2 uses a correct iterative redistribution algorithm
   in `build_target_weights_v2()`.

3. **Silent median imputation for missing fundamentals.** V1 fills missing
   quality and value scores with the cross-sectional median and treats them as
   real observations. This overstates confidence in low-data stocks. V2 tracks
   `feature_available` flags and reports `factor_coverage` explicitly.

---

## Three Strategies

### A. Core_Quality_Momentum_V2
**Factor weights:** Momentum 35%, Quality 30%, Value 20%, Safety 15%

The reference portfolio. Balanced exposure across all four factors.
Not tilted strongly toward any single characteristic. Designed to be the
baseline against which Aggressive and Defensive are compared.

### B. Aggressive_Momentum_V2
**Factor weights:** Momentum 50%, Quality 25%, Value 15%, Safety 10%

Stronger momentum tilt. Accepts higher short-term volatility in pursuit of
trend-following returns. Retains minimum quality and safety floors to avoid
speculative positions. Clearly distinct from Core: momentum weight 50% vs 35%,
safety weight 10% vs 15%.

### C. Defensive_Quality_V2
**Factor weights:** Momentum 15%, Quality 40%, Value 20%, Safety 25%

Prioritizes balance-sheet strength, profitability, and downside protection.
Lowest momentum exposure of the three strategies. Highest safety weight (25%).
Clearly distinct from Core and Aggressive. Designed to outperform in drawdown
environments and underperform in strong bull markets.

### Why these three?

The weights are simple, round numbers chosen *before* running any backtest.
They are not the result of optimization. The key differentiation:

| Factor    | Core | Aggressive | Defensive |
|-----------|------|------------|-----------|
| Momentum  | 0.35 | **0.50**   | **0.15**  |
| Quality   | 0.30 | 0.25       | **0.40**  |
| Value     | 0.20 | 0.15       | 0.20      |
| Safety    | 0.15 | **0.10**   | **0.25**  |
| **Sum**   | 1.00 | 1.00       | 1.00      |

No Balanced_V2 is created — the Core strategy already fills this role.

---

## Four Factors

### Momentum
**Sub-factors:** 12-1 month return (40%), 6-1 month return (35%), 3-1 month return (25%)

All momentum sub-factors skip the most recent month to reduce short-term reversal noise
(standard practice in academic momentum literature). The 12-1 weight is highest because
it captures the most robust and well-documented momentum anomaly. The shorter-term
components add responsiveness to more recent trend changes.

**Data:** Daily price returns from yfinance. **Always available.**

### Quality
**Sub-factors:** ROE (25%), Gross Margin (25%), Debt/Equity inverted (20%),
Earnings Growth (20%), FCF Margin (10%)

Quality captures profitability, efficiency, and financial health. Debt/equity is
inverted (lower leverage → higher score) with a 5× cap before inversion to prevent
extreme leverage from dominating. FCF margin has lowest weight because it is least
frequently available.

**Data:** yfinance `.info` — current fundamentals. **NOT point-in-time.**
**Limitation:** Historical backtests using this data have look-ahead bias.
All backtest results with quality data must be labeled `limited/unsupported`.

### Value
**Sub-factors:** FCF yield (40%), Earnings yield / 1×PE (35%), 1/(EV/EBITDA) (25%)

FCF-based value metrics are preferred over pure earnings multiples because they
are less susceptible to accounting distortions. EV/EBITDA is sector-relative
(high EV/EBITDA in capital-light software is different from the same ratio in
manufacturing). Earnings yield is 1/forward_PE (higher = cheaper).

**Data:** yfinance `.info` — current fundamentals. **NOT point-in-time.**
**Same look-ahead limitation as Quality.**

### Safety
**Sub-factors:** 1/vol60 (30%), 1/downside_vol (25%), 1/max_drawdown_1y (15%),
log(avg_volume) rank (10%)

Safety captures pure risk characteristics derived entirely from price data.
Lower volatility, lower downside deviation, lower drawdown, and higher liquidity
all improve the safety score. All sub-factors are always computable from price history.

**Data:** Daily price returns from yfinance. **Always available.**

---

## MegaCap Filter

MegaCap is not a scoring penalty for non-megacap stocks. It is an optional
universe filter: when enabled, only companies with market cap ≥ $50B are
considered. This filter is applied as a pre-screening step before any scoring.

This design avoids the common mistake of implicitly penalizing non-megacap
stocks through a scoring adjustment that has no fundamental justification.

---

## Claude Shadow

Two shadow variants exist (not separate portfolios):

- **Factor_Only_Core_V2:** Pure quantitative signal. No Claude input.
- **Factor_Plus_Claude_Shadow_V2:** Same factor signal + Claude analysis logged.
  Claude output **cannot create orders** in V2A (`order_creation_blocked=True`).

The shadow design allows direct measurement: any difference in logged signal
quality between Factor_Only and Factor_Plus is attributable solely to the
Claude analysis layer.

**What Claude currently analyses (V1 baseline):**
- Limited yfinance data and news headlines
- Does not read full primary sources (SEC filings, transcripts)
- Opus for earnings analysis, Sonnet for weekly analysis
- Small direct weight in V1 total score

**V2 Claude output schema** (see `modules/v2_claude_shadow.py`):
Structured JSON with fields: `signal_direction`, `evidence_strength`,
`guidance_change`, `estimate_revision_direction`, `margin_trend`,
`earnings_quality`, `capital_allocation_quality`, `thesis_risks`,
`catalyst_strength`, `uncertainty`, `source_ids`, `source_published_at`,
`model_id`, `prompt_version`, `generated_at`, `data_cutoff_at`.

Missing or invalid fields → `"unavailable"`. Never fabricated neutral values.

---

## What Is Active vs. Not Active

| Component | Status in V2A |
|-----------|---------------|
| Core_Quality_Momentum_V2 paper fills | Not active |
| Aggressive_Momentum_V2 paper fills | Not active |
| Defensive_Quality_V2 paper fills | Not active |
| Factor_Only_Core_V2 shadow log | Not active |
| Factor_Plus_Claude_Shadow_V2 shadow log | Not active |
| V2 signal generation | Not active |
| V2 in signal.yml / execute.yml | Not active |
| V1 production workflows | **Unchanged** |
| V1 orders / fills / state | **Unchanged** |

V2 can only be invoked by explicit `mode="paper"` call. It cannot be triggered
by the production GitHub Actions workflows.

---

## Known Limitations (V2A)

1. Fundamental data (quality, value) is not point-in-time — historical backtests
   with these factors have look-ahead bias.
2. Backtest universe has survivorship bias (current universe, not historical).
3. Claude analysis is limited to yfinance data and headline text — no primary sources.
4. No live execution yet — paper only.
