"""
Factor information coefficient (IC) report on the signal files we already have.

Research only: reads data_v4/signals/ and yfinance, writes
research/factor_ic_report.md and research/output/factor_ic_*.csv.
Nothing in modules/ imports this file.

For each signal date t (anchored to the NYSE session the signal's prices come
from) and each factor f, IC_t = Spearman rank correlation across stocks between
f and the forward total return close(t) -> close(t+h), h in {5, 21} sessions.

Run:  python3 research/factor_ic.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from modules.exchange_calendar import last_completed_session  # noqa: E402

SIGNAL_GLOB = os.path.join(ROOT, "data_v4", "signals", "*_signal.json")
OUT_DIR = os.path.join(ROOT, "research", "output")
REPORT = os.path.join(ROOT, "research", "factor_ic_report.md")
HORIZONS = (5, 21)

FACTORS = [
    # component scores (0-100)
    "momentum_score", "quality_score", "value_score", "trend_score", "sentiment_score",
    # raw inputs
    "mom_12_1", "mom_6_1", "mom_3_1", "ret_1m", "relative_strength_3m",
    "distance_from_high", "rsi", "vol20", "vol60",
    # strategy composites
    "score_max_return", "score_quality_momentum", "score_balanced",
    "score_low_risk", "score_megacap", "score_ai_sentiment",
]


def load_signals() -> pd.DataFrame:
    """One row per (signal file, ticker). ``session`` is a first guess; see anchor_sessions."""
    rows = []
    for path in sorted(glob.glob(SIGNAL_GLOB)):
        d = json.load(open(path))
        session = last_completed_session(pd.Timestamp(d["created_at_utc"]))
        seen = set()
        for strat in d["strategies"].values():
            for c in strat["candidates"]:
                if c["ticker"] in seen:
                    continue
                seen.add(c["ticker"])
                row = {"file": os.path.basename(path), "created_at_utc": d["created_at_utc"],
                       "session": session, "ticker": c["ticker"], "signal_price": c.get("price"),
                       "universe_count": d.get("universe_count")}
                for f in FACTORS:
                    row[f] = c.get(f)
                rows.append(row)
    return pd.DataFrame(rows)


def _match_rate(g: pd.DataFrame, close: pd.DataFrame, ts) -> float:
    px = close.loc[ts, g["ticker"]].to_numpy(dtype=float)
    sp = g["signal_price"].to_numpy(dtype=float)
    ok = ~np.isnan(px) & ~np.isnan(sp)
    return float(np.mean(np.abs(sp[ok] / px[ok] - 1) < 0.005)) if ok.any() else 0.0


def anchor_sessions(sig: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """
    Set ``session`` to the trading day whose close the signal prices actually
    match. The file name and the run time are NOT reliable: several runs used
    data that was one session old, and some early runs were made intraday.

    Candidates: the two sessions before last_completed_session(created_at),
    that session itself, and the session in progress at creation time.
    If no candidate matches >= 80 % of prices (intraday run), use the session
    in progress (prices taken before its close; no look-ahead) and mark it.
    """
    days = close.dropna(how="all").index
    out = []
    for f, g in sig.groupby("file"):
        guess = pd.Timestamp(g["session"].iloc[0])
        i = days.get_indexer([guess])[0]
        cands = [days[j] for j in range(max(i - 2, 0), min(i + 2, len(days)))]
        rates = {ts: _match_rate(g, close, ts) for ts in cands}
        best = max(rates, key=rates.get)
        g = g.copy()
        if rates[best] >= 0.8:
            g["session"], g["anchor"] = best.strftime("%Y-%m-%d"), "close_match"
        else:
            created = pd.Timestamp(g["created_at_utc"].iloc[0]).tz_convert("America/New_York")
            nxt = days[days.normalize() >= created.normalize().tz_localize(None)]
            g["session"], g["anchor"] = nxt[0].strftime("%Y-%m-%d"), "intraday"
        g["match_rate"] = rates[best]
        g["labelled_session"] = guess.strftime("%Y-%m-%d")
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    # Two files can map to the same session (re-runs, stale data): keep the latest run.
    latest = df.groupby("session")["created_at_utc"].transform("max")
    return df[df["created_at_utc"] == latest].reset_index(drop=True)


def download_prices(tickers, start, end) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = yf.download(sorted(tickers), start=start, end=end, auto_adjust=False,
                      progress=False, group_by="column", threads=True)
    return raw["Adj Close"], raw["Close"]


def spearman(x: pd.Series, y: pd.Series) -> float:
    m = x.notna() & y.notna()
    if m.sum() < 5:
        return float("nan")
    rx, ry = x[m].rank(), y[m].rank()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def newey_west_t(series: pd.Series, lags: int) -> float:
    """t-stat of the mean with Newey-West (Bartlett) standard errors."""
    x = series.dropna().to_numpy()
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1 - k / (lags + 1)
        var += 2 * w * (e[k:] @ e[:-k]) / n
    if var <= 0:
        return float("nan")
    return float(x.mean() / math.sqrt(var / n))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    raw_sig = load_signals()
    tickers = set(raw_sig["ticker"])
    start = pd.Timestamp(raw_sig["session"].min()) - pd.Timedelta(days=10)
    adj, close = download_prices(tickers, start.strftime("%Y-%m-%d"),
                                 (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    # Signal prices are split/dividend-adjusted as of the run date; compare
    # against the adjusted close (tiny later dividend drift stays < 0.5 %).
    sig = anchor_sessions(raw_sig, adj)
    sessions = sorted(sig["session"].unique())
    trading_days = adj.dropna(how="all").index
    last_px_day = trading_days[-1]

    files = sig.groupby("file").first().reset_index()
    files[["file", "created_at_utc", "labelled_session", "session", "anchor", "match_rate"]].to_csv(
        os.path.join(OUT_DIR, "factor_ic_anchors.csv"), index=False)
    match_rate = files

    ic_rows, obs_rows = [], []
    for s in sessions:
        ts = pd.Timestamp(s)
        if ts not in trading_days:
            continue
        i = trading_days.get_loc(ts)
        day = sig[sig["session"] == s].set_index("ticker")
        for h in HORIZONS:
            if i + h > len(trading_days) - 1:
                continue
            t1 = trading_days[i + h]
            fwd = adj.loc[t1, day.index] / adj.loc[ts, day.index] - 1
            # Cross-sectionally demeaned is irrelevant for rank IC; keep raw.
            obs_rows.append({"session": s, "h": h, "n": int(fwd.notna().sum()),
                             "universe_count": int(day["universe_count"].iloc[0])})
            for f in FACTORS:
                vals = pd.to_numeric(day[f], errors="coerce")
                ic_rows.append({"session": s, "h": h, "factor": f,
                                "ic": spearman(vals, fwd), "n": int((vals.notna() & fwd.notna()).sum())})

    ic = pd.DataFrame(ic_rows)
    obs = pd.DataFrame(obs_rows)
    ic.to_csv(os.path.join(OUT_DIR, "factor_ic_by_date.csv"), index=False)

    summary = []
    for (h, f), g in ic.groupby(["h", "factor"]):
        g = g.sort_values("session")
        s = g["ic"].dropna()
        nonoverlap = g.iloc[::h]["ic"].dropna()
        summary.append({
            "h": h, "factor": f, "dates": len(s), "mean_ic": s.mean(),
            "median_ic": s.median(), "std_ic": s.std(ddof=1),
            "hit_rate": (s > 0).mean(),
            "t_naive": s.mean() / (s.std(ddof=1) / math.sqrt(len(s))) if len(s) > 2 and s.std() > 0 else float("nan"),
            "t_newey_west": newey_west_t(s, lags=h - 1),
            "nonoverlap_n": len(nonoverlap), "nonoverlap_mean_ic": nonoverlap.mean(),
            "mean_n_stocks": g["n"].mean(),
        })
    summ = pd.DataFrame(summary)
    summ.to_csv(os.path.join(OUT_DIR, "factor_ic_summary.csv"), index=False)
    write_report(summ, obs, sig, files, last_px_day)


def _summary_lines(summ: pd.DataFrame) -> list[str]:
    out = []
    for h in HORIZONS:
        p = summ[summ["h"] == h].sort_values("mean_ic", ascending=False)
        comp = p[p["factor"].str.startswith("score_")]["mean_ic"]
        top = ", ".join(f"`{r.factor}` {r.mean_ic:+.2f}" for r in p.head(3).itertuples())
        bot = ", ".join(f"`{r.factor}` {r.mean_ic:+.2f}" for r in p.tail(3).itertuples())
        out.append(f"- **{h} dager:** høyest snitt-IC {top}; lavest {bot}. "
                   f"De seks strategiscorene ligger på {comp.min():+.2f} til {comp.max():+.2f}.")
    out.append("- Med 2–17 uavhengige perioder er nesten ingenting av dette skilt fra null "
               "med rimelig sikkerhet. Les tallene som hypoteser, ikke som funn.")
    return out


def _independent(o: pd.DataFrame, h: int) -> int:
    if o.empty:
        return 0
    span = len(pd.bdate_range(o["session"].min(), o["session"].max()))
    return max(1, round(span / h))


def _fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{nd}f}"


def write_report(summ, obs, sig, files, last_px_day) -> None:
    by_session = sig.groupby("session").agg(n=("ticker", "size"), universe=("universe_count", "first"))
    small = by_session[by_session["universe"] <= 37]
    stale = int(((files["anchor"] == "close_match")
                 & (files["session"] < files["labelled_session"])).sum())
    lines = [
        "# Faktor-IC-rapport (forskning, ikke beslutningsgrunnlag)",
        "",
        f"_Generert {date.today().isoformat()} av `research/factor_ic.py`. "
        f"Kurser fra yfinance til og med {last_px_day.date()}._",
        "",
        "## ⚠️ Les dette først",
        "",
        f"- **Universet var bare 37 aksjer fram til oktober 2026.** Etter kvalitetsfilteret "
        f"var det bare **{int(small['n'].min())}–{int(small['n'].max())} aksjer** per dag å rangere "
        f"({len(small)} signaldager). Det store universet (544 aksjer) kom først "
        f"med signalet 13. oktober (kurser fra {by_session[by_session['universe'] > 37].index.min()}), "
        f"og det har ennå ikke 5 dager med fremtidig avkastning å måle mot. **Alle tallene under gjelder altså det lille universet.**",
        "- **Overlappende perioder.** Signalene er daglige, men avkastningen måles 5 og 21 "
        "handelsdager fram. To nabodager deler 20 av 21 dager med avkastning, så de er nesten "
        "samme observasjon. Antall *uavhengige* observasjoner er omtrent antall dager delt på "
        "horisonten (se kolonnen «uavh. n»). Vanlig t-verdi («t naiv») overdriver sikkerheten "
        "kraftig; «t NW» (Newey–West) korrigerer delvis, men er også upålitelig med så få "
        "uavhengige perioder.",
        "- **Kort og spesiell periode.** Juni og august–oktober 2026, med et opphold "
        "23. juni–5. august da systemet ikke kjørte. Ett markedsregime.",
        "- **Utvalgsskjevhet.** Bare aksjer som passerte kvalitetsfilteret er med, og filteret "
        "bruker selv noen av faktorene (bl.a. over 200-dagers snitt).",
        "- **Mange faktorer testet** (20 stk × 2 horisonter). Med så mange tester vil noen se "
        "«signifikante» ut ved en tilfeldighet.",
        "",
        "**Konklusjon på forhånd:** Disse dataene kan ikke bevise at noen faktor virker eller "
        "ikke virker. De kan bare vise om noe er grovt feil (f.eks. en faktor med sterkt "
        "*negativ* IC over hele perioden).",
        "",
        "## Kort oppsummering",
        "",
        *_summary_lines(summ),
        "",
        "## Metode",
        "",
        "- Signaldato = den NYSE-sesjonen hvis sluttkurs signalprisene faktisk stemmer med "
        "(innenfor 0,5 %). Filnavn og kjøretidspunkt er ikke pålitelige, se under. "
        "Ved flere filer for samme sesjon brukes den siste.",
        f"- {int((files['anchor'] == 'close_match').sum())} av {len(files)} filer (etter at dubletter for samme sesjon er fjernet) ble forankret "
        f"på sluttkurs-treff; {int((files['anchor'] == 'intraday').sum())} var kjørt midt i "
        "handelsdagen og forankres til den dagens sluttkurs (faktorene ble målt før stengning, "
        "så det er ingen fremtidslekkasje).",
        f"- **Funn:** {stale} signalfiler bruker kurser som er én eller flere sesjoner eldre enn "
        "`last_completed_session` på kjøretidspunktet (dvs. nyeste handelsdag manglet i "
        "yfinance-dataene). Dette gjelder særlig kjøringene fra 29. september og utover. "
        "Se `research/output/factor_ic_anchors.csv`.",
        "- Fremtidig avkastning = justert sluttkurs (utbytte og splitt) fra sesjon t til t+h.",
        "- IC_t = Spearman-rangkorrelasjon på tvers av aksjer samme dag. Rapportert: snitt over dager.",
        "- «Ikke-overlapp»: bare hver h-te dag, så periodene ikke overlapper.",
        "",
    ]
    for h in HORIZONS:
        o = obs[obs["h"] == h]
        lines += [
            f"## Horisont {h} handelsdager",
            "",
            f"{len(o)} signaldager med fullført {h}-dagers avkastning "
            f"({o['session'].min() if len(o) else '–'} – {o['session'].max() if len(o) else '–'}), "
            f"gjennomsnittlig {o['n'].mean():.1f} aksjer per dag. "
            f"Omtrent {_independent(o, h)} uavhengige {h}-dagers perioder.",
            "",
            "| Faktor | Snitt-IC | Median | Std | Andel > 0 | t naiv | t NW | uavh. n | IC uavh. |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        part = summ[summ["h"] == h].sort_values("mean_ic", ascending=False)
        for _, r in part.iterrows():
            lines.append(
                f"| `{r['factor']}` | {_fmt(r['mean_ic'])} | {_fmt(r['median_ic'])} | "
                f"{r['std_ic']:.3f} | {r['hit_rate']:.0%} | {_fmt(r['t_naive'], 1)} | "
                f"{_fmt(r['t_newey_west'], 1)} | {r['nonoverlap_n']} | {_fmt(r['nonoverlap_mean_ic'])} |")
        lines.append("")
    lines += [
        "## Filer",
        "",
        "- `research/output/factor_ic_by_date.csv` — IC per dag, faktor og horisont",
        "- `research/output/factor_ic_summary.csv` — tabellene over",
        "",
    ]
    with open(REPORT, "w") as fh:
        fh.write("\n".join(lines))
    print(f"Skrev {REPORT}")


if __name__ == "__main__":
    main()
