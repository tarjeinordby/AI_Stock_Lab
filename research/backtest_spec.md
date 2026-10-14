# Forhåndsregistrert backtest-spesifikasjon (låst før første kjøring)

Skrevet og committet 2026-10-15 **før** backtesten ble kjørt. Ingen parametre
endres etter at resultatene er sett. Én strategi, ingen varianter.

## Datakilder

- **Historisk S&P 500-medlemskap:** GitHub `fja05680/sp500`, commit
  `a2430f2af0c79ddf0748e91de11bdeb1616ab5a7` (2026-09-07), fil
  `S&P 500 Historical Components & Changes (Updated).csv` (lagret som
  `research/data/sp500_membership/components.csv`, SHA-256
  `36326709d46d6cd25834de5df457b16f5f96fad3a06b9beac28f7b88aa0b0d54`).
  Opprinnelig fra Andreas Clenow (*Trading Evolved*, 1996–2019), deretter
  oppdatert fra Wikipedia sine endringslister. Hver rad er en dato og hele
  medlemslisten fra den datoen. Medlemskap på dag d = siste rad med dato ≤ d.
- **Kurser:** yfinance, daglig, `auto_adjust=True` (justert for utbytte og
  splitt = totalavkastning). Ticker `BRK.B` → `BRK-B` (punktum → bindestrek).

## Strategi

- **Signaldag:** siste NYSE-handelsdag i hver måned, des. 2009 – nov. 2025.
- **Handel:** til sluttkurs **neste** handelsdag (én dags forsinkelse, ingen
  fremtidslekkasje).
- **Univers:** aksjer som er S&P 500-medlemmer på signaldagen og har kurs i
  yfinance på dag t, t−21, t−126 og t−252, og minst 200 gyldige daglige
  avkastninger de siste 252 dagene.
- **Score:** `(0,5 · (P[t−21]/P[t−252] − 1) + 0,5 · (P[t−21]/P[t−126] − 1)) / σ₁₂`
  der `σ₁₂` = standardavvik av daglige avkastninger siste 252 dager · √252.
  (t i handelsdager.)
- **Portefølje:** topp 10 etter score, likevekt (10 % hver), rebalansert helt
  tilbake til likevekt hver måned.
- **Kostnader:** kommisjon 0,10 % + slippage 0,05 % per side, pluss salgsgebyr
  0,003 % av salgsverdi (som i `docs/evaluation_plan.md`). Kjøp 0,15 %, salg 0,153 %.
- **Kontanter:** 0 % rente.
- **Aksje som mister kurs midt i en periode** (data slutter): verdien fryses på
  siste kurs og selges til den kursen ved neste rebalansering.

## Kontroller

1. **SPY** kjøp og hold (én kjøpskostnad).
2. **RSP** (Invesco S&P 500 Equal Weight ETF) kjøp og hold — et ekte,
   investerbart likevektet S&P 500 uten overlevelsesskjevhet.
3. **Likevektet historiske medlemmer med yfinance-data**, samme regler,
   rebalansering og kostnader som strategien. Har samme overlevelsesskjevhet
   som strategien, så forskjellen strategi − (3) er den mest rettferdige.

## Periode og mål

- Avkastning måles fra sluttkurs første handelsdag i 2010 til siste handelsdag 2025.
- Rapporteres: årlig avkastning per kalenderår, CAGR, Sharpe (rente 0, daglig,
  √252), maks drawdown, årlig turnover (énveis: (kjøp+salg)/2 / snittverdi),
  og andel historiske medlemmer uten kursdata i yfinance per år.
