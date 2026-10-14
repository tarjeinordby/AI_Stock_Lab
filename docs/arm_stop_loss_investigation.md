# Hvorfor ble ARM solgt på −37,6 %? (undersøkelse 2026-10-15)

**Kort svar: Nei, erstatningen av sluttkurs i `run_data_quality_checks` forklarer det ikke.
Årsaken er at systemet ikke kjørte mellom 23. juni og 5. august 2026.**

## Fakta

| | Dato | Kurs |
|---|---|---|
| Kjøp (4 strategier) | 2026-06-19 | 439,75 |
| Stop-loss-sjekk (signal) | 2026-08-05 | sluttkurs 274,58 → 274,58 / 439,75 − 1 = **−37,56 %** |
| Salg (utført) | 2026-08-06 | 289,14 (−34,2 % fra kjøp) |

Stop-loss-grensene er −15 % til −22 % (`modules/scoring.py`). ARM passerte −18 %
(≈ 360,6) allerede 24. juni (sluttkurs 359,08) og −22 % (≈ 343,0) 26. juni (334,27).

## Hvorfor ikke datakvalitetssjekken

- Ingen signalfil for perioden 2026-06-11 – 2026-08-07 flagget ARM i `quality_report`.
- ARM hadde bare én dag over 15 % i perioden: 2026-08-04 (+17,4 %, 239,06 → 280,56).
  Det fantes ingen signalkjøring den dagen, og 5. august var bevegelsen −2,1 %, så
  den gamle erstatningslogikken ble aldri brukt på ARM.
- Kursen som utløste salget (274,58) er den ekte sluttkursen fra yfinance.

## Faktisk årsak

Git-loggen viser ingen Signal/Execute-kjøringer fra 2026-06-23 til 2026-08-05.
Stop-loss sjekkes bare når systemet kjører, så tapet vokste i seks uker uten at
noen regel kunne slå til. Første kjøring etter pausen fant −37,6 % og solgte.

## Hva endringen likevel fikser

Den gamle koden kunne ha skjult et ekte kursfall på 15–40 % på selve dagen det
skjedde (sluttkursen ble satt tilbake til gårsdagens), slik at stop-loss ville
kommet én dag for sent. Nå flagges bevegelsen, men kursen beholdes.
