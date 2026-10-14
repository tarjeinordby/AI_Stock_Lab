# Faktor-IC-rapport (forskning, ikke beslutningsgrunnlag)

_Generert 2026-10-15 av `research/factor_ic.py`. Kurser fra yfinance til og med 2026-10-14._

## ⚠️ Les dette først

- **Universet var bare 37 aksjer fram til oktober 2026.** Etter kvalitetsfilteret var det bare **23–26 aksjer** per dag å rangere (50 signaldager). Det store universet (544 aksjer) kom først med signalet 13. oktober (kurser fra 2026-10-12), og det har ennå ikke 5 dager med fremtidig avkastning å måle mot. **Alle tallene under gjelder altså det lille universet.**
- **Overlappende perioder.** Signalene er daglige, men avkastningen måles 5 og 21 handelsdager fram. To nabodager deler 20 av 21 dager med avkastning, så de er nesten samme observasjon. Antall *uavhengige* observasjoner er omtrent antall dager delt på horisonten (se kolonnen «uavh. n»). Vanlig t-verdi («t naiv») overdriver sikkerheten kraftig; «t NW» (Newey–West) korrigerer delvis, men er også upålitelig med så få uavhengige perioder.
- **Kort og spesiell periode.** Juni og august–oktober 2026, med et opphold 23. juni–5. august da systemet ikke kjørte. Ett markedsregime.
- **Utvalgsskjevhet.** Bare aksjer som passerte kvalitetsfilteret er med, og filteret bruker selv noen av faktorene (bl.a. over 200-dagers snitt).
- **Mange faktorer testet** (20 stk × 2 horisonter). Med så mange tester vil noen se «signifikante» ut ved en tilfeldighet.

**Konklusjon på forhånd:** Disse dataene kan ikke bevise at noen faktor virker eller ikke virker. De kan bare vise om noe er grovt feil (f.eks. en faktor med sterkt *negativ* IC over hele perioden).

## Kort oppsummering

- **5 dager:** høyest snitt-IC `sentiment_score` +0.09, `mom_6_1` +0.09, `mom_12_1` +0.09; lavest `rsi` -0.00, `value_score` -0.00, `ret_1m` -0.06. De seks strategiscorene ligger på +0.06 til +0.07.
- **21 dager:** høyest snitt-IC `sentiment_score` +0.27, `mom_6_1` +0.25, `momentum_score` +0.17; lavest `value_score` -0.14, `rsi` -0.19, `ret_1m` -0.21. De seks strategiscorene ligger på +0.04 til +0.05.
- Med 2–17 uavhengige perioder er nesten ingenting av dette skilt fra null med rimelig sikkerhet. Les tallene som hypoteser, ikke som funn.

## Metode

- Signaldato = den NYSE-sesjonen hvis sluttkurs signalprisene faktisk stemmer med (innenfor 0,5 %). Filnavn og kjøretidspunkt er ikke pålitelige, se under. Ved flere filer for samme sesjon brukes den siste.
- 50 av 51 filer (etter at dubletter for samme sesjon er fjernet) ble forankret på sluttkurs-treff; 1 var kjørt midt i handelsdagen og forankres til den dagens sluttkurs (faktorene ble målt før stengning, så det er ingen fremtidslekkasje).
- **Funn:** 10 signalfiler bruker kurser som er én eller flere sesjoner eldre enn `last_completed_session` på kjøretidspunktet (dvs. nyeste handelsdag manglet i yfinance-dataene). Dette gjelder særlig kjøringene fra 29. september og utover. Se `research/output/factor_ic_anchors.csv`.
- Fremtidig avkastning = justert sluttkurs (utbytte og splitt) fra sesjon t til t+h.
- IC_t = Spearman-rangkorrelasjon på tvers av aksjer samme dag. Rapportert: snitt over dager.
- «Ikke-overlapp»: bare hver h-te dag, så periodene ikke overlapper.

## Horisont 5 handelsdager

49 signaldager med fullført 5-dagers avkastning (2026-06-11 – 2026-10-07), gjennomsnittlig 24.0 aksjer per dag. Omtrent 17 uavhengige 5-dagers perioder.

| Faktor | Snitt-IC | Median | Std | Andel > 0 | t naiv | t NW | uavh. n | IC uavh. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `sentiment_score` | +0.094 | +0.122 | 0.284 | 63% | +2.3 | +1.4 | 10 | +0.080 |
| `mom_6_1` | +0.094 | +0.123 | 0.283 | 69% | +2.3 | +1.2 | 10 | +0.093 |
| `mom_12_1` | +0.089 | +0.123 | 0.369 | 67% | +1.7 | +0.9 | 10 | +0.072 |
| `distance_from_high` | +0.072 | +0.100 | 0.249 | 61% | +2.0 | +1.1 | 10 | +0.043 |
| `momentum_score` | +0.071 | +0.103 | 0.320 | 65% | +1.6 | +0.8 | 10 | +0.060 |
| `score_megacap` | +0.070 | +0.056 | 0.274 | 61% | +1.8 | +1.0 | 10 | +0.056 |
| `score_low_risk` | +0.063 | +0.054 | 0.270 | 59% | +1.6 | +0.9 | 10 | +0.071 |
| `score_ai_sentiment` | +0.061 | +0.064 | 0.274 | 59% | +1.6 | +0.8 | 10 | +0.058 |
| `score_quality_momentum` | +0.060 | +0.057 | 0.273 | 57% | +1.5 | +0.8 | 10 | +0.058 |
| `score_balanced` | +0.060 | +0.057 | 0.273 | 57% | +1.5 | +0.8 | 10 | +0.058 |
| `score_max_return` | +0.057 | +0.064 | 0.273 | 57% | +1.5 | +0.8 | 10 | +0.057 |
| `trend_score` | +0.050 | +0.115 | 0.266 | 65% | +1.3 | +0.7 | 10 | +0.057 |
| `relative_strength_3m` | +0.046 | +0.007 | 0.282 | 54% | +1.1 | +0.6 | 10 | +0.036 |
| `quality_score` | +0.046 | +0.044 | 0.196 | 65% | +1.6 | +1.0 | 10 | +0.045 |
| `mom_3_1` | +0.046 | -0.047 | 0.288 | 47% | +1.1 | +0.6 | 10 | +0.044 |
| `vol20` | +0.030 | +0.057 | 0.311 | 55% | +0.7 | +0.4 | 10 | +0.078 |
| `vol60` | +0.002 | +0.027 | 0.296 | 53% | +0.1 | +0.0 | 10 | +0.002 |
| `rsi` | -0.003 | +0.073 | 0.279 | 55% | -0.1 | -0.0 | 10 | -0.030 |
| `value_score` | -0.003 | -0.000 | 0.320 | 47% | -0.1 | -0.0 | 10 | +0.036 |
| `ret_1m` | -0.058 | -0.022 | 0.249 | 45% | -1.6 | -0.9 | 10 | -0.073 |

## Horisont 21 handelsdager

34 signaldager med fullført 21-dagers avkastning (2026-06-11 – 2026-09-15), gjennomsnittlig 24.1 aksjer per dag. Omtrent 3 uavhengige 21-dagers perioder.

| Faktor | Snitt-IC | Median | Std | Andel > 0 | t naiv | t NW | uavh. n | IC uavh. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `sentiment_score` | +0.274 | +0.315 | 0.172 | 94% | +9.3 | +5.5 | 2 | +0.303 |
| `mom_6_1` | +0.247 | +0.327 | 0.326 | 68% | +4.4 | +1.6 | 2 | +0.181 |
| `momentum_score` | +0.168 | +0.361 | 0.347 | 65% | +2.8 | +1.0 | 2 | +0.059 |
| `mom_12_1` | +0.162 | +0.364 | 0.436 | 59% | +2.2 | +0.8 | 2 | +0.109 |
| `vol60` | +0.150 | +0.164 | 0.157 | 85% | +5.6 | +2.9 | 2 | -0.047 |
| `trend_score` | +0.088 | +0.137 | 0.241 | 65% | +2.1 | +0.7 | 2 | +0.085 |
| `mom_3_1` | +0.047 | +0.038 | 0.194 | 56% | +1.4 | +0.8 | 2 | -0.145 |
| `score_ai_sentiment` | +0.047 | +0.024 | 0.229 | 59% | +1.2 | +0.5 | 2 | -0.089 |
| `score_megacap` | +0.047 | +0.023 | 0.235 | 56% | +1.2 | +0.5 | 2 | -0.098 |
| `score_max_return` | +0.043 | +0.026 | 0.227 | 53% | +1.1 | +0.4 | 2 | -0.097 |
| `score_quality_momentum` | +0.042 | +0.011 | 0.224 | 53% | +1.1 | +0.4 | 2 | -0.098 |
| `score_balanced` | +0.042 | +0.011 | 0.224 | 53% | +1.1 | +0.4 | 2 | -0.098 |
| `score_low_risk` | +0.039 | +0.015 | 0.210 | 56% | +1.1 | +0.4 | 2 | -0.106 |
| `distance_from_high` | +0.023 | -0.021 | 0.218 | 47% | +0.6 | +0.2 | 2 | -0.061 |
| `quality_score` | +0.004 | +0.059 | 0.223 | 59% | +0.1 | +0.0 | 2 | -0.068 |
| `vol20` | -0.028 | -0.036 | 0.241 | 44% | -0.7 | -0.3 | 2 | -0.097 |
| `relative_strength_3m` | -0.044 | -0.096 | 0.185 | 27% | -1.4 | -0.8 | 2 | -0.208 |
| `value_score` | -0.142 | -0.313 | 0.328 | 41% | -2.5 | -0.9 | 2 | -0.043 |
| `rsi` | -0.191 | -0.182 | 0.245 | 24% | -4.5 | -3.1 | 2 | -0.130 |
| `ret_1m` | -0.207 | -0.242 | 0.208 | 18% | -5.8 | -3.5 | 2 | -0.206 |

## Filer

- `research/output/factor_ic_by_date.csv` — IC per dag, faktor og horisont
- `research/output/factor_ic_summary.csv` — tabellene over
