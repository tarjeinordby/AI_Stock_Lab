# Backtest 2010–2025: momentum/volatilitet, topp 10 av historisk S&P 500

_Generert 2026-10-15 av `research/backtest_momentum.py`. Regler forhåndsregistrert i `research/backtest_spec.md` (committet før første kjøring). Én strategi, ingen varianter prøvd, ingen parametre justert._

## Hovedtall (2010-01-04 – 2025-12-31, etter kostnader, kontantrente 0 %)

| | CAGR | Volatilitet | Sharpe | Maks drawdown | Total |
|---|---:|---:|---:|---:|---:|
| Strategi (momentum/vol topp 10) | +14.3 % | 22.8 % | 0.70 | -42.6 % | +745 % |
| SPY | +13.9 % | 17.2 % | 0.84 | -33.7 % | +702 % |
| RSP (likevektet S&P 500 ETF) | +12.1 % | 18.0 % | 0.72 | -39.0 % | +520 % |
| Likevekt historiske medlemmer m/data | +13.3 % | 18.2 % | 0.78 | -39.8 % | +636 % |

Gjennomsnittlig årlig turnover (énveis): strategi **562%**, likevekt medlemmer 36%. Handelskostnadene trakk strategien ned med i snitt **1.70% per år** (likevekt medlemmer: 0.11%).

## Ærlig tolkning

- Strategien ga +14.3 % per år mot +13.9 % for SPY og +12.1 % for RSP. Mot likevekt av de samme (overlevende) medlemmene: +13.3 %.
- Den mest rettferdige sammenligningen er strategi mot kontroll 3 (samme overlevelsesskjevhet): differanse +1.0 % per år.
- Sharpe 0.70 mot 0.84 (SPY). Maks drawdown -42.6 % mot -33.7 % (SPY).
- Overlevelsesskjevhet: i snitt manglet 20.1% av medlemmene kursdata på signaldagene, mest i de tidlige årene. Den reelle avkastningen ville sannsynligvis vært lavere enn tallene over.
- 16 år med én strategi gir fortsatt stor usikkerhet: standardfeilen på årlig meravkastning er omtrent tracking error / √16.

## Avkastning per kalenderår

| År | Strategi (momentum/vol topp 10) | SPY | RSP (likevektet S&P 500 ETF) | Likevekt historiske medlemmer m/data | Turnover strategi | Mangler kursdata |
|---|---:|---:|---:|---:|---:|---:|
| 2010 | +9.0 % | +13.1 % | +19.4 % | +21.7 % | 562% | 34.2% |
| 2011 | +9.1 % | +1.9 % | -0.7 % | +2.5 % | 686% | 33.3% |
| 2012 | +8.4 % | +16.0 % | +17.2 % | +19.3 % | 468% | 31.3% |
| 2013 | +43.4 % | +32.3 % | +35.5 % | +37.1 % | 698% | 29.3% |
| 2014 | +13.3 % | +13.5 % | +14.1 % | +14.8 % | 477% | 28.3% |
| 2015 | +7.8 % | +1.2 % | -2.7 % | -1.5 % | 550% | 26.4% |
| 2016 | +10.3 % | +12.0 % | +14.5 % | +16.3 % | 611% | 23.4% |
| 2017 | +31.7 % | +21.7 % | +18.5 % | +20.6 % | 601% | 20.4% |
| 2018 | -14.2 % | -4.6 % | -7.8 % | -7.3 % | 568% | 18.6% |
| 2019 | +18.9 % | +31.2 % | +28.9 % | +30.8 % | 596% | 15.8% |
| 2020 | -4.7 % | +18.3 % | +12.7 % | +12.5 % | 528% | 12.8% |
| 2021 | +11.1 % | +28.7 % | +29.4 % | +30.0 % | 629% | 10.9% |
| 2022 | +10.8 % | -18.2 % | -11.6 % | -11.2 % | 484% | 8.2% |
| 2023 | +24.5 % | +26.2 % | +13.7 % | +15.8 % | 477% | 5.7% |
| 2024 | +43.9 % | +24.9 % | +12.8 % | +12.3 % | 576% | 4.2% |
| 2025 | +20.2 % | +17.7 % | +11.2 % | +11.0 % | 481% | 2.9% |

## Manglende kursdata i yfinance

yfinance har bare kurser for selskaper som fortsatt er børsnotert under samme ticker. To typer mangler:

1. **Selskaper som er borte** (kjøpt opp, konkurs, tatt av børs). De kan aldri bli valgt, og mange hadde dårlig avkastning før de forsvant (overlevelsesskjevhet). Det gjør backtesten for **optimistisk**, både for strategien og for kontroll 3 — men ikke for SPY og RSP, som er ekte fond.
2. **Selskaper som har byttet ticker** (f.eks. BK → BNY, ABC → COR, ANTM → ELV). Historikken finnes i yfinance under det nye navnet, men medlemslisten bruker det gamle. Dette er ikke overlevelsesskjevhet, men gjør universet mindre enn det burde vært. Spesifikasjonen sier at tickere brukes slik kilden oppgir dem, så dette er ikke rettet (en navnemapping lagt til etter å ha sett resultatene ville vært en endring etter fasit).

En ny nedlasting av alle manglende tickere (25 om gangen, uten parallellitet) ga 0 ekstra treff, så hullene skyldes ikke midlertidige nedlastingsfeil.

«Andel per måned» = snitt over årets 12 signaldager av andelen medlemmer uten kurs den dagen. «Unike medlemmer» = alle som var medlem en eller annen gang i året.

| År | Andel per måned | Unike medlemmer | Uten data hele året | Andel | Eksempler |
|---|---:|---:|---:|---:|---|
| 2010 | 34.2% | 513 | 182 | 35.5% | AABA,ABC,ACS,AET,AGN,AKS,ALTR,ANDV |
| 2011 | 33.3% | 515 | 175 | 34.0% | AABA,ABC,AET,AGN,AKS,ALTR,ANDV,ANRZQ |
| 2012 | 31.3% | 515 | 164 | 31.8% | AABA,ABC,ADT,AET,AGN,ALTR,ALXN,ANDV |
| 2013 | 29.3% | 516 | 156 | 30.2% | AABA,ABC,ADS,ADT,AET,AGN,ALTR,ALXN |
| 2014 | 28.3% | 513 | 149 | 29.0% | AABA,ABC,ADS,ADT,AET,AGN,ALTR,ALXN |
| 2015 | 26.4% | 529 | 149 | 28.2% | AABA,ABC,ADS,ADT,AET,AGN,ALTR,ALXN |
| 2016 | 23.4% | 533 | 133 | 25.0% | AABA,ABC,ADS,ADT,AET,AGN,ALXN,ANDV |
| 2017 | 20.4% | 533 | 116 | 21.8% | AABA,ABC,ADS,AET,AGN,ALXN,ANDV,ANSS |
| 2018 | 18.6% | 529 | 105 | 19.8% | ABC,ABMD,ADS,AET,AGN,ALXN,ANDV,ANSS |
| 2019 | 15.8% | 534 | 89 | 16.7% | ABC,ABMD,ADS,AGN,ALXN,ANSS,ANTM,APC |
| 2020 | 12.8% | 525 | 72 | 13.7% | ABC,ABMD,ADS,AGN,ALXN,ANSS,ANTM,ARNC |
| 2021 | 10.9% | 526 | 63 | 12.0% | ABC,ABMD,ALXN,ANSS,ANTM,ATVI,AVB,BK |
| 2022 | 8.2% | 527 | 52 | 9.9% | ABC,ABMD,ANSS,ANTM,ATVI,AVB,BK,BLL |
| 2023 | 5.7% | 522 | 34 | 6.5% | ABC,ANSS,ATVI,AVB,BK,CDAY,CMA,CTLT |
| 2024 | 4.2% | 523 | 26 | 5.0% | ANSS,AVB,BK,CDAY,CMA,CTLT,CTRA,DAY |
| 2025 | 2.9% | 524 | 18 | 3.4% | ANSS,AVB,BK,CTRA,DAY,DFS,EA,EQR |

## Andre begrensninger

- **Gjenbrukte tickere:** yfinance knytter en ticker til dagens selskap. Hvis en ticker tidligere tilhørte et annet selskap, kan feil kurshistorikk bli brukt for perioden før.
- **Medlemslisten** er satt sammen av én privatperson fra Clenow-data + Wikipedia; ikke offisiell S&P-data. Forfatteren skriver selv at listen kan mangle noen navn.
- **Utførelse** til sluttkurs dagen etter signalet, uten markedspåvirkning utover 5 bps.
- **Én historisk periode** (2010–2025) som var uvanlig god for amerikanske storselskaper og for momentum i teknologi.
- Momentum-faktoren var kjent og publisert lenge før 2010 (Jegadeesh & Titman 1993), så dette er ikke en «oppdaget» strategi, men regelen er heller ikke uavhengig av at vi vet at momentum har fungert historisk.

## Filer

- `research/output/backtest_daily_values.csv` — daglig verdi for alle fire serier
- `research/output/backtest_calendar_returns.csv` — årlig avkastning
- `research/output/backtest_selections.csv` — valgte aksjer hver måned
