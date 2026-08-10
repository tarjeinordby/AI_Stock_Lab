# Quant Baseline V2 — Data Audit

**Status:** V2A research
**Date:** 2026-08-10

---

## Available Data Sources

| Source | Feature | Point-in-Time | Frequency | Timestamp | Honest Backtest? |
|--------|---------|---------------|-----------|-----------|------------------|
| yfinance daily OHLCV | Prices, returns, volume | Yes (adjusted) | Daily | Download date | Yes |
| yfinance .info | Fundamentals (ROE, margins, PE, etc.) | **No** | Snapshot | Download date | **No** |
| Treasury.gov | 10Y-2Y yield curve | Yes | Daily | Publication date | Yes |
| yfinance .insider | Insider transactions | Yes (reported date) | Event | Report date | Limited |
| yfinance .calendar | Earnings calendar | Yes | Event | Announcement date | Limited |

---

## Factor Coverage — What Is Available

### Momentum Factor
- **Features:** 12-1 return, 6-1 return, 3-1 return
- **Data source:** yfinance adjusted daily close
- **Point-in-time:** Yes — price data is available as-of any historical date
- **Always available:** Yes, for tickers with >= 1 year of history
- **Backtest use:** Honest

### Safety Factor
- **Features:** vol60, downside_vol, max_drawdown_1y, avg_volume
- **Data source:** yfinance daily close and volume
- **Point-in-time:** Yes — computed from price history available at signal date
- **Always available:** Yes, for tickers with >= 30 days of history
- **Backtest use:** Honest

### Quality Factor
- **Features:** ROE, gross_margin, debt/equity, earnings_growth, FCF_margin
- **Data source:** yfinance `.info` (snapshot — current values only)
- **Point-in-time:** **No** — yfinance returns today's fundamentals, not historical values
- **Backtest use:** **Unsupported / Limited**
  - A 2018 backtest using 2026 fundamentals has look-ahead bias
  - Companies that went bankrupt between 2018-2026 are excluded (survivorship)
  - Quality scores in historical backtests must be marked `limited`

### Value Factor
- **Features:** FCF yield, earnings yield (1/PE), 1/(EV/EBITDA)
- **Data source:** yfinance `.info` (snapshot — current values only)
- **Point-in-time:** **No** — same limitation as Quality
- **Backtest use:** **Unsupported / Limited**

---

## Missing Data — What Is Not Available

| Feature | Status | Notes |
|---------|--------|-------|
| Historical point-in-time fundamentals | Missing | Would require Compustat, FactSet, or similar |
| SEC filings (10-K, 10-Q, transcripts) | Not connected | Interface defined in v2_claude_shadow.py but no data |
| Earnings call transcripts | Not connected | Required for honest earnings quality scoring |
| Analyst estimates history | Not connected | Required for estimate revision direction |
| Short interest | Not connected | Useful for sentiment/risk |
| Options market data | Not connected | Useful for implied volatility / uncertainty |
| News full-text | Not connected | Headlines only via yfinance |

---

## Survivorship Bias

The current universe consists of stocks that are **currently trading**.
Companies that were delisted, acquired, or went bankrupt between the backtest
start date and today are excluded. This systematically biases historical
performance upward.

**Impact:** All backtest results must include the note:
> "Survivorship bias present — current universe only. Historical performance
> is overstated."

**Remediation (not yet implemented):** Use a historical universe membership
database (e.g., CRSP, S&P constituent history) to include all stocks that were
in the investment universe at each historical date.

---

## Data Policy

V2 enforces the following data policy:

1. Missing fundamental data → `feature_available=False`, score = `null`
2. Null scores excluded from factor computation (not filled with median)
3. `factor_coverage` = fraction of 4 factors with a real score, reported per ticker
4. All analysis uses only data available at `data_cutoff_at` timestamp
5. Point-in-time fundamentals = `False` for all yfinance `.info` data
6. Backtest results with non-point-in-time data → labeled `limited/unsupported`
