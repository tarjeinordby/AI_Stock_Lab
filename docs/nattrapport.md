# Nattrapport — branch `natt-steg1` (2026-10-15)

Alt er gjort på branchen `natt-steg1`, laget fra oppdatert `main`. **Ingenting er
pushet.** Ingenting under `data_v4/` eller `data_v3/` er endret, ingen eksisterende
workflow-filer er endret, og ingen strategiregler, vekter eller terskler er endret.

**Testene:** `python3 -m pytest -q` → **1207 passed** (var 1179 før; 28 nye tester).

---

## Kort oppsummert

| # | Oppgave | Status | Commit |
|---|---|---|---|
| 1 | Datakvalitetssjekk flagger i stedet for å bytte kurs + ARM | ✅ ferdig | `3739e56` |
| 2 | GitHub-workflow som kjører pytest | ✅ ferdig (ikke testet på GitHub, siden ingenting er pushet) | `516de8b` |
| 3 | Ny kapitalfordelingsfunksjon (ikke koblet inn) | ✅ ferdig | `dfe10ad` |
| 4 | Utkast til evalueringsplan | ✅ ferdig, merket UTKAST | `e3e9afb` |
| 5 | Faktorrapport (IC) | ✅ ferdig | `8e65c31` |
| 6 | Ærlig backtest 2010–2025 | ✅ ferdig | `755912c` (spesifikasjon) + `0be92f9` |

---

## 1. Datakvalitetssjekken og ARM

**Hva er endret:** Før byttet `run_data_quality_checks` ut dagens sluttkurs med
gårsdagens hvis en aksje beveget seg mer enn 15 % på én dag. Det betyr at et ekte,
stort kursfall ble skjult for én dag. Nå blir bevegelsen bare **flagget** i
rapporten, og den ekte kursen beholdes. Splitt-sjekken og null-volum-sjekken er
uendret. Navnet `price_adjusted` i signal-ledgeren er beholdt så det gamle formatet
fortsatt stemmer (bare forklaringen i koden er oppdatert).

**ARM −37,6 %: Nei, dette forklarer det ikke.** ARM ble aldri flagget av sjekken.
Det egentlige svaret er at **systemet ikke kjørte fra 23. juni til 5. august**.
ARM ble kjøpt 19. juni til 439,75 og passerte stop-loss-grensen (−18 % / −22 %)
allerede 24.–26. juni. Men stop-loss sjekkes bare når systemet kjører, så første
gang regelen kunne slå til var 5. august, da kursen var 274,58 (−37,6 %).
Detaljer: `docs/arm_stop_loss_investigation.md`.

## 2. Test-workflow

`.github/workflows/tests.yml` kjører `pytest` på push og pull request. Den hopper
over push-commits som bare endrer `data/`, `data_v3/` eller `data_v4/` (det er de
daglige bot-commitene), så den ikke kjører mange ganger om dagen for ingenting.
Den kan ikke ødelegge noe: den har bare lesetilgang og pusher ingenting.

## 3. Ny kapitalfordeling (`modules/capital_allocation.py`)

Funksjonen `allocate_capital` fordeler pengene i forhold til score, men:
- ingen aksje får mer enn `max_weight` (f.eks. 40 %),
- ingen sektor får mer enn `sector_cap` **av hele kapitalen**,
- det som ikke får plass hos en aksje/sektor som er full, går videre til de andre,
- kontanter blir bare igjen hvis grensene gjør det umulig å investere alt
  (f.eks. 2 aksjer med maks 40 % = maks 80 %).

Testen du ba om: score 90/8/2 med maks 40 % per aksje → **40 % / 40 % / 20 %, 100 % investert** ✅.
19 tester totalt, inkludert en tilfeldig test som sjekker grensene på 300 ulike oppsett.
**Ikke koblet inn i live-koden** — ingen andre filer importerer den.

## 4. Evalueringsplan (`docs/evaluation_plan.md`)

Tydelig merket **UTKAST — IKKE GODKJENT**. Inneholder: SPY og likevektet univers
som kontroller, totalavkastning med utbytte, kontantrente 0 %, kostnader
(0,10 % + salgsgebyr + 5 bps per side), formler for Sharpe og drawdown,
beslutning etter 12 måneder, ærlig forbehold om usikkerhet og om at det er prøvd
ca. 15–30 varianter, og regelen om at parameterendring starter klokken på nytt.

## 5. Faktorrapport (`research/factor_ic_report.md`)

**Hva den viser:** Momentum (6-1, 12-1) og AI-sentiment hadde positiv sammenheng med
avkastningen de neste 5 og 21 dagene; 1-måneds avkastning og RSI hadde negativ.
De seks strategiscorene hadde svak sammenheng (IC rundt +0,04 til +0,07).

**Men:** universet var bare 37 aksjer (23–26 etter filter) fram til oktober, og
med overlappende perioder finnes det bare 2–17 uavhengige observasjoner. **Dette
kan ikke brukes som bevis for noe.** Det står tydelig i rapporten.

**⚠️ Viktig funn underveis:** 10 av de siste signalfilene (fra 29. september) bruker
kurser som er **én handelsdag gamle**. Eksempel: signalet merket 13. oktober bruker
sluttkursene fra 12. oktober. Signal-ledgeren bekrefter det
(`observed_max_data_timestamp` er én sesjon bak). Det ser ut til at yfinance ikke
alltid har dagens kurs klar når jobben kjører etter midnatt UTC, og
foreldelsessjekken slår bare til ved mer enn 5 dagers alder. **Dette er ikke rettet**
(det er live-kode og var ikke en del av oppdraget).

## 6. Backtest 2010–2025 (`research/backtest_report.md`)

Reglene ble skrevet ned og committet **før** første kjøring (`research/backtest_spec.md`),
og ble ikke endret etterpå. Kilde for historisk S&P 500-medlemskap: GitHub
`fja05680/sp500` (låst til én bestemt versjon, dokumentert i spesifikasjonen).

| | Årlig avkastning | Sharpe | Maks fall |
|---|---:|---:|---:|
| **Strategien** (momentum/volatilitet, topp 10) | +14,3 % | 0,70 | −42,6 % |
| SPY | +13,9 % | 0,84 | −33,7 % |
| RSP (ekte likevektet S&P 500-fond) | +12,1 % | 0,72 | −39,0 % |
| Likevekt av de samme medlemmene med data | +13,3 % | 0,78 | −39,8 % |

**Hva betyr det på enkelt norsk:** Strategien tjente litt mer enn SPY, men med mye
større svingninger og større fall, så risikojustert (Sharpe) var den **dårligere**
enn å bare eie SPY. Den byttet ut ca. 5–6 ganger porteføljen per år, og
handelskostnadene spiste ca. 1,7 % per år.

Og tallene er trolig **for gode**: 20 % av medlemmene i snitt (34 % i 2010, 3 % i 2025)
mangler kursdata i yfinance. Mange av dem er selskaper som gikk dårlig og
forsvant, og de kunne aldri bli valgt. Noen er bare selskaper som har byttet
ticker (f.eks. BK → BNY). Dette er dokumentert per år i rapporten.

**Konklusjon:** Denne enkle momentumstrategien slo ikke SPY på en meningsfull måte
etter risiko og kostnader i 2010–2025.

---

## Det du må ta stilling til / godkjenne

1. **Oppgave 1 (live-kode):** Er det greit at store kursbevegelser nå bare flagges?
   Dette endrer hvilke kurser signalet ser på dager med >15 % bevegelse. Krever din
   godkjenning og uavhengig gjennomgang før merge.
2. **Oppgave 2:** Vil du ha test-workflowen? Den kjører først når branchen pushes.
3. **Oppgave 3:** Skal den nye kapitalfordelingen kobles inn i live-koden (eget steg)?
   Hvilke verdier for maks per aksje og sektorgrense?
4. **Oppgave 4:** Les utkastet til evalueringsplan og godkjenn eller send rettelser.
   Spesielt: beslutningsregelen etter 12 måneder (punkt 5) og salgsgebyr-satsen.
5. **Nytt funn — gamle kurser i signalet:** Vil du at jeg undersøker og retter at
   signalet siden 29. september ofte bruker gårsdagens kurser?
6. **Backtest-resultatet:** Bør det påvirke planen (f.eks. mer vekt på SPY/likevekt
   som standard)?
7. **Merge:** Ingenting er merget eller pushet. Si fra når/om branchen skal pushes.

## Forbehold

- Workflowen i oppgave 2 er bare sjekket lokalt (gyldig YAML), ikke kjørt på GitHub.
- Kildene i oppgave 5 og 6 (yfinance, fja05680/sp500) er gratis og ikke offisielle.
- Forskningsrapportene laster ned nye kurser hver gang de kjøres, så tallene kan
  endre seg litt hvis yfinance retter data.
