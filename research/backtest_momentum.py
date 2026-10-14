"""
Honest backtest 2010-2025 of ONE pre-registered strategy (research/backtest_spec.md).

Research only. Nothing in modules/ imports this file, and the rules below must
not be changed after results have been seen.

  Universe : historical S&P 500 members (research/data/sp500_membership/components.csv)
  Signal   : (0.5 * mom_12_1 + 0.5 * mom_6_1) / 12m volatility, at month-end close
  Portfolio: top 10, equal weight, trade at next session's close, monthly
  Costs    : 0.10 % commission + 5 bps slippage per side, +0.003 % on sells
  Controls : SPY, RSP (equal-weight S&P 500 ETF), equal-weight of members with data

Run:  python3 research/backtest_momentum.py
Writes research/backtest_report.md and research/output/backtest_*.csv.
Prices are cached in research/cache/ (git-ignored).
"""

from __future__ import annotations

import math
import os
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMBERSHIP_CSV = os.path.join(ROOT, "research", "data", "sp500_membership", "components.csv")
CACHE = os.path.join(ROOT, "research", "cache", "sp500_prices_2008_2026.pkl")
OUT_DIR = os.path.join(ROOT, "research", "output")
REPORT = os.path.join(ROOT, "research", "backtest_report.md")

PRICE_START, PRICE_END = "2008-06-01", "2026-01-06"
FIRST_SIGNAL, LAST_SIGNAL = "2009-12-01", "2025-11-30"
EVAL_START, EVAL_END = "2010-01-01", "2025-12-31"

TOP_N = 10
LOOKBACK_12, LOOKBACK_6, SKIP = 252, 126, 21
MIN_RETURNS = 200
COMMISSION, SLIPPAGE, SELL_FEE = 0.0010, 0.0005, 0.00003
BUY_COST = COMMISSION + SLIPPAGE
SELL_COST = COMMISSION + SLIPPAGE + SELL_FEE


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_membership() -> pd.DataFrame:
    m = pd.read_csv(MEMBERSHIP_CSV, parse_dates=["date"]).sort_values("date")
    m["tickers"] = m["tickers"].str.split(",")
    return m.reset_index(drop=True)


def members_on(membership: pd.DataFrame, day: pd.Timestamp) -> list[str]:
    i = membership["date"].searchsorted(day, side="right") - 1
    return list(membership.at[i, "tickers"]) if i >= 0 else []


def yf_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")


def load_prices(tickers: list[str]) -> pd.DataFrame:
    """Adjusted daily closes, columns = membership tickers (not yf symbols)."""
    if os.path.exists(CACHE):
        cached = pd.read_pickle(CACHE)
        if set(tickers) <= set(cached.columns):
            return cached
    frames = []
    syms = sorted({yf_symbol(t) for t in tickers})
    for i in range(0, len(syms), 100):
        chunk = syms[i:i + 100]
        print(f"  yfinance {i + len(chunk)}/{len(syms)}")
        raw = yf.download(chunk, start=PRICE_START, end=PRICE_END, auto_adjust=True,
                          progress=False, group_by="column", threads=True)
        close = raw["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(chunk[0])
        frames.append(close)
    px = pd.concat(frames, axis=1)
    back = {yf_symbol(t): t for t in tickers}
    px = px.rename(columns=back)
    px = px.reindex(columns=sorted(set(tickers)))
    px = px.dropna(how="all", axis=0)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    px.to_pickle(CACHE)
    return px


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

def month_end_sessions(days: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = pd.Series(days, index=days)
    ends = s.groupby([days.year, days.month]).max()
    return [d for d in ends if pd.Timestamp(FIRST_SIGNAL) <= d <= pd.Timestamp(LAST_SIGNAL)]


def scores_on(px: pd.DataFrame, rets: pd.DataFrame, i: int, universe: list[str]) -> pd.Series:
    if i < LOOKBACK_12:
        return pd.Series(dtype=float)
    cols = [t for t in universe if t in px.columns]
    p_t = px.iloc[i][cols]
    p_skip = px.iloc[i - SKIP][cols]
    p_12 = px.iloc[i - LOOKBACK_12][cols]
    p_6 = px.iloc[i - LOOKBACK_6][cols]
    window = rets.iloc[i - LOOKBACK_12 + 1:i + 1][cols]
    n_ret = window.notna().sum()
    vol = window.std(ddof=1) * math.sqrt(252)
    mom12 = p_skip / p_12 - 1
    mom6 = p_skip / p_6 - 1
    score = (0.5 * mom12 + 0.5 * mom6) / vol
    ok = p_t.notna() & p_skip.notna() & p_12.notna() & p_6.notna() & (n_ret >= MIN_RETURNS) & (vol > 0)
    return score[ok].dropna()


def simulate(px: pd.DataFrame, targets: dict[pd.Timestamp, list[str]], start_value: float = 1.0):
    """
    Daily simulation. ``targets`` maps execution day -> list of names to hold
    in equal weight from that day's close. Returns (daily values, trade log).
    """
    days = px.index
    first_exec = min(targets)
    shares: dict[str, float] = {}
    last_px: dict[str, float] = {}
    cash = start_value
    values, trades = {}, []
    started = False
    for d in days[days >= first_exec]:
        row = px.loc[d]
        for t in shares:
            p = row.get(t)
            if p is not None and not np.isnan(p):
                last_px[t] = float(p)
        value = cash + sum(shares[t] * last_px[t] for t in shares)
        if d in targets:
            names = targets[d]
            target_val = {t: 0.0 for t in shares}
            # Value after costs solves V' = V - costs(V'); iterate (converges fast).
            post = value
            for _ in range(20):
                per = post / len(names) if names else 0.0
                tv = dict(target_val)
                for t in names:
                    tv[t] = per
                buys = sells = 0.0
                for t, v in tv.items():
                    cur = shares.get(t, 0.0) * last_px.get(t, float(row.get(t, np.nan)))
                    diff = v - cur
                    if diff > 0:
                        buys += diff
                    else:
                        sells += -diff
                cost = buys * BUY_COST + sells * SELL_COST
                post = value - cost
            per = post / len(names) if names else 0.0
            new_shares = {}
            for t in names:
                p = float(row[t])
                last_px[t] = p
                new_shares[t] = per / p
            trades.append({"day": d, "buys": buys, "sells": sells, "cost": cost,
                           "value_before": value, "names": ",".join(names)})
            shares = new_shares
            last_px = {t: last_px[t] for t in shares}
            cash = 0.0 if names else post
            value = post
            started = True
        if started:
            values[d] = value
    return pd.Series(values), pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(v: pd.Series) -> dict:
    v = v[(v.index >= EVAL_START) & (v.index <= EVAL_END)]
    r = v.pct_change().dropna()
    years = len(r) / 252
    cagr = (v.iloc[-1] / v.iloc[0]) ** (1 / years) - 1
    sharpe = r.mean() / r.std(ddof=1) * math.sqrt(252)
    dd = (v / v.cummax() - 1).min()
    return {"cagr": cagr, "vol": r.std(ddof=1) * math.sqrt(252), "sharpe": sharpe,
            "max_dd": dd, "total": v.iloc[-1] / v.iloc[0] - 1}


def calendar_returns(v: pd.Series) -> pd.Series:
    v = v[(v.index >= pd.Timestamp(EVAL_START) - pd.Timedelta(days=10)) & (v.index <= EVAL_END)]
    year_end = v.groupby(v.index.year).last()
    first = v[v.index >= EVAL_START].iloc[0]
    prev = pd.Series([first] + list(year_end.values[:-1]), index=year_end.index)
    out = year_end / prev - 1
    return out[out.index >= int(EVAL_START[:4])]


def annual_turnover(trades: pd.DataFrame, v: pd.Series) -> pd.Series:
    t = trades.copy()
    t["year"] = pd.DatetimeIndex(t["day"]).year
    t["traded"] = (t["buys"] + t["sells"]) / 2
    avg_val = v.groupby(v.index.year).mean()
    return (t.groupby("year")["traded"].sum() / avg_val).dropna()


def annual_cost_drag(trades: pd.DataFrame, v: pd.Series) -> pd.Series:
    t = trades.copy()
    t["year"] = pd.DatetimeIndex(t["day"]).year
    avg_val = v.groupby(v.index.year).mean()
    return (t.groupby("year")["cost"].sum() / avg_val).dropna()


def buy_and_hold(px: pd.Series, start: pd.Timestamp) -> pd.Series:
    p = px[px.index >= start].ffill()
    return p / p.iloc[0] * (1 - BUY_COST)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    membership = load_membership()
    window = membership[(membership["date"] >= "2008-01-01") & (membership["date"] <= EVAL_END)]
    prev = membership[membership["date"] < "2008-01-01"].tail(1)
    all_tickers = sorted(set().union(*[set(t) for t in pd.concat([prev, window])["tickers"]]))
    print(f"{len(all_tickers)} historiske medlemmer 2008–2025")

    px = load_prices(all_tickers + ["SPY", "RSP"])
    bench = px[["SPY", "RSP"]].copy()
    px = px.drop(columns=["SPY", "RSP"], errors="ignore")
    # Never let a stray non-trading row in: use SPY's sessions as the calendar.
    days = bench["SPY"].dropna().index
    px = px.reindex(days)
    bench = bench.reindex(days)
    rets = px.pct_change(fill_method=None)

    signal_days = month_end_sessions(days)
    strat_targets, ew_targets, selections, coverage = {}, {}, [], []
    for sd in signal_days:
        i = days.get_loc(sd)
        exec_day = days[i + 1]
        members = members_on(membership, sd)
        have = [t for t in members if t in px.columns and not np.isnan(px.at[sd, t])]
        coverage.append({"signal_day": sd, "members": len(members), "with_price": len(have)})
        sc = scores_on(px, rets, i, members)
        top = list(sc.sort_values(ascending=False).head(TOP_N).index)
        # Executed at next close: drop names without a price that day.
        top = [t for t in top if not np.isnan(px.at[exec_day, t])]
        ew = [t for t in have if not np.isnan(px.at[exec_day, t])]
        strat_targets[exec_day] = top
        ew_targets[exec_day] = ew
        selections.append({"signal_day": sd.date(), "exec_day": exec_day.date(),
                           "eligible": len(sc), "picks": ",".join(top)})

    strat_v, strat_tr = simulate(px, strat_targets)
    ew_v, ew_tr = simulate(px, ew_targets)
    start = strat_v.index[strat_v.index >= EVAL_START][0]
    spy_v = buy_and_hold(bench["SPY"], start)
    rsp_v = buy_and_hold(bench["RSP"], start)
    series = {"Strategi (momentum/vol topp 10)": strat_v,
              "SPY": spy_v,
              "RSP (likevektet S&P 500 ETF)": rsp_v,
              "Likevekt historiske medlemmer m/data": ew_v}

    pd.DataFrame(selections).to_csv(os.path.join(OUT_DIR, "backtest_selections.csv"), index=False)
    pd.DataFrame({k: v for k, v in series.items()}).to_csv(os.path.join(OUT_DIR, "backtest_daily_values.csv"))

    cov = pd.DataFrame(coverage)
    cov["year"] = pd.DatetimeIndex(cov["signal_day"]).year
    cov["missing_share"] = 1 - cov["with_price"] / cov["members"]
    missing_by_year = cov.groupby("year")["missing_share"].mean()
    distinct = distinct_missing_by_year(membership, px)
    write_report(series, strat_tr, ew_tr, missing_by_year, distinct, selections)


def distinct_missing_by_year(membership: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """For each year: distinct members at any time that year, and how many have
    no yfinance price at all during that year."""
    rows = []
    for y in range(int(EVAL_START[:4]), int(EVAL_END[:4]) + 1):
        a, b = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
        rows_y = membership[(membership["date"] >= a) & (membership["date"] <= b)]
        sets = [set(members_on(membership, a))] + [set(t) for t in rows_y["tickers"]]
        names = set().union(*sets)
        yr = px[(px.index >= a) & (px.index <= b)]
        no_data = [t for t in names if t not in yr.columns or yr[t].notna().sum() == 0]
        rows.append({"year": y, "distinct_members": len(names), "no_data": len(no_data),
                     "share": len(no_data) / len(names), "examples": ",".join(sorted(no_data)[:8])})
    return pd.DataFrame(rows).set_index("year")


def _pct(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.{nd}f} %"


def write_report(series, strat_tr, ew_tr, missing_by_year, distinct, selections) -> None:
    names = list(series)
    m = {k: metrics(v) for k, v in series.items()}
    cal = pd.DataFrame({k: calendar_returns(v) for k, v in series.items()})
    to_strat = annual_turnover(strat_tr, series[names[0]])
    to_ew = annual_turnover(ew_tr, series[names[3]])
    cal.to_csv(os.path.join(OUT_DIR, "backtest_calendar_returns.csv"))
    s, spy, rsp, ew = (m[k] for k in names)

    L = [
        "# Backtest 2010–2025: momentum/volatilitet, topp 10 av historisk S&P 500",
        "",
        f"_Generert {date.today().isoformat()} av `research/backtest_momentum.py`. "
        "Regler forhåndsregistrert i `research/backtest_spec.md` (committet før første kjøring). "
        "Én strategi, ingen varianter prøvd, ingen parametre justert._",
        "",
        "## Hovedtall (2010-01-04 – 2025-12-31, etter kostnader, kontantrente 0 %)",
        "",
        "| | CAGR | Volatilitet | Sharpe | Maks drawdown | Total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k in names:
        x = m[k]
        L.append(f"| {k} | {_pct(x['cagr'])} | {x['vol'] * 100:.1f} % | {x['sharpe']:.2f} | "
                 f"{_pct(x['max_dd'])} | {_pct(x['total'], 0)} |")
    L += [
        "",
        f"Gjennomsnittlig årlig turnover (énveis): strategi **{to_strat.mean():.0%}**, "
        f"likevekt medlemmer {to_ew.mean():.0%}. "
        f"Handelskostnadene trakk strategien ned med i snitt "
        f"**{annual_cost_drag(strat_tr, series[names[0]]).mean():.2%} per år** "
        f"(likevekt medlemmer: {annual_cost_drag(ew_tr, series[names[3]]).mean():.2%}).",
        "",
        "## Ærlig tolkning",
        "",
        *interpretation(m, names, missing_by_year),
        "",
        "## Avkastning per kalenderår",
        "",
        "| År | " + " | ".join(names) + " | Turnover strategi | Mangler kursdata |",
        "|---|" + "---:|" * (len(names) + 2),
    ]
    for y in cal.index:
        L.append(f"| {y} | " + " | ".join(_pct(cal.at[y, k]) for k in names)
                 + f" | {to_strat.get(y, float('nan')):.0%} | {missing_by_year.get(y, float('nan')):.1%} |")
    L += [
        "",
        "## Manglende kursdata i yfinance",
        "",
        "yfinance har bare kurser for selskaper som fortsatt er børsnotert under samme ticker. "
        "To typer mangler:",
        "",
        "1. **Selskaper som er borte** (kjøpt opp, konkurs, tatt av børs). De kan aldri bli valgt, "
        "og mange hadde dårlig avkastning før de forsvant (overlevelsesskjevhet). Det gjør "
        "backtesten for **optimistisk**, både for strategien og for kontroll 3 — men ikke for "
        "SPY og RSP, som er ekte fond.",
        "2. **Selskaper som har byttet ticker** (f.eks. BK → BNY, ABC → COR, ANTM → ELV). "
        "Historikken finnes i yfinance under det nye navnet, men medlemslisten bruker det gamle. "
        "Dette er ikke overlevelsesskjevhet, men gjør universet mindre enn det burde vært. "
        "Spesifikasjonen sier at tickere brukes slik kilden oppgir dem, så dette er ikke rettet "
        "(en navnemapping lagt til etter å ha sett resultatene ville vært en endring etter fasit).",
        "",
        "En ny nedlasting av alle manglende tickere (25 om gangen, uten parallellitet) ga 0 "
        "ekstra treff, så hullene skyldes ikke midlertidige nedlastingsfeil.",
        "",
        "«Andel per måned» = snitt over årets 12 signaldager av andelen medlemmer uten kurs den dagen. "
        "«Unike medlemmer» = alle som var medlem en eller annen gang i året.",
        "",
        "| År | Andel per måned | Unike medlemmer | Uten data hele året | Andel | Eksempler |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for y, r in distinct.iterrows():
        L.append(f"| {y} | {missing_by_year.get(y, float('nan')):.1%} | {r['distinct_members']} | "
                 f"{r['no_data']} | {r['share']:.1%} | {r['examples']} |")
    L += [
        "",
        "## Andre begrensninger",
        "",
        "- **Gjenbrukte tickere:** yfinance knytter en ticker til dagens selskap. Hvis en ticker "
        "tidligere tilhørte et annet selskap, kan feil kurshistorikk bli brukt for perioden før.",
        "- **Medlemslisten** er satt sammen av én privatperson fra Clenow-data + Wikipedia; "
        "ikke offisiell S&P-data. Forfatteren skriver selv at listen kan mangle noen navn.",
        "- **Utførelse** til sluttkurs dagen etter signalet, uten markedspåvirkning utover 5 bps.",
        "- **Én historisk periode** (2010–2025) som var uvanlig god for amerikanske storselskaper "
        "og for momentum i teknologi.",
        "- Momentum-faktoren var kjent og publisert lenge før 2010 (Jegadeesh & Titman 1993), så "
        "dette er ikke en «oppdaget» strategi, men regelen er heller ikke uavhengig av at vi vet "
        "at momentum har fungert historisk.",
        "",
        "## Filer",
        "",
        "- `research/output/backtest_daily_values.csv` — daglig verdi for alle fire serier",
        "- `research/output/backtest_calendar_returns.csv` — årlig avkastning",
        "- `research/output/backtest_selections.csv` — valgte aksjer hver måned",
        "",
    ]
    with open(REPORT, "w") as fh:
        fh.write("\n".join(L))
    print(f"Skrev {REPORT}")


def interpretation(m, names, missing_by_year) -> list[str]:
    s, spy, rsp, ew = (m[k] for k in names)
    out = [
        f"- Strategien ga {_pct(s['cagr'])} per år mot {_pct(spy['cagr'])} for SPY og "
        f"{_pct(rsp['cagr'])} for RSP. Mot likevekt av de samme (overlevende) medlemmene: "
        f"{_pct(ew['cagr'])}.",
        f"- Den mest rettferdige sammenligningen er strategi mot kontroll 3 (samme "
        f"overlevelsesskjevhet): differanse {_pct(s['cagr'] - ew['cagr'])} per år.",
        f"- Sharpe {s['sharpe']:.2f} mot {spy['sharpe']:.2f} (SPY). Maks drawdown "
        f"{_pct(s['max_dd'])} mot {_pct(spy['max_dd'])} (SPY).",
        f"- Overlevelsesskjevhet: i snitt manglet {missing_by_year.mean():.1%} av medlemmene "
        "kursdata på signaldagene, mest i de tidlige årene. Den reelle avkastningen ville "
        "sannsynligvis vært lavere enn tallene over.",
        "- 16 år med én strategi gir fortsatt stor usikkerhet: standardfeilen på årlig "
        "meravkastning er omtrent tracking error / √16.",
    ]
    return out


if __name__ == "__main__":
    main()
