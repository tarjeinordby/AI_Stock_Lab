# Evalueringsplan — UTKAST

> **⚠️ UTKAST — IKKE GODKJENT.** Skrevet 2026-10-15 (natt-steg1) som forslag.
> Ingenting her gjelder før eieren har godkjent dokumentet. Når det er godkjent,
> settes datoen nedenfor, og fra da regnes klokken.
>
> **Godkjent dato:** _(ikke satt)_ · **Klokken starter:** _(første NYSE-sesjon etter godkjenning)_

Formålet er å avgjøre, på forhånd og uten å se på resultatene, *hvordan* vi skal
bedømme om en strategi er bedre enn å bare kjøpe indeksen.

---

## 1. Hva som sammenlignes

Hver strategi sammenlignes med to kontroller som løper i samme periode, med
samme startkapital og samme kostnadsmodell:

| Kontroll | Definisjon |
|---|---|
| **SPY** | Kjøp og hold SPY. Utbytte reinvesteres. Ingen rebalansering. |
| **Likevektet univers** | Lik vekt i alle aksjer i strategiens univers den dagen, rebalansert månedlig (samme dag som strategiene). Belastes kostnader på samme måte som strategiene. |

Likevektet univers er den viktigste kontrollen: den skiller *aksjevalg* fra
*hvilket univers vi har valgt*. En strategi som bare slår SPY fordi universet
er tungt i teknologi, har ikke vist noen evne til å velge aksjer.

## 2. Avkastning

- **Totalavkastning med utbytte.** Kurser justert for utbytte og splitter
  (yfinance `auto_adjust=True` eller `Adj Close`). For porteføljer med egne
  posisjoner: kontantutbytte krediteres på eks-dato.
- **Kontantrente: 0 %.** Kontanter gir ingen avkastning. Dette gjør det
  ugunstig å holde mye kontanter, og det er med vilje.
- Daglig porteføljeverdi `V_t` registreres etter NYSE-stengning.
  Daglig avkastning `r_t = V_t / V_{t-1} − 1`.

## 3. Kostnader (gjelder strategier og likevektet kontroll)

| Post | Sats | Når |
|---|---|---|
| Kommisjon | 0,10 % av handelsverdi | kjøp og salg |
| Slippage | 5 bps (0,05 %) av handelsverdi | kjøp og salg |
| Salgsgebyr (SEC § 31 + FINRA TAF) | fast antatt 0,003 % av salgsverdi | kun salg |

Kostnad per handel: kjøp `0,15 %`, salg `0,153 %`. Én rundtur ≈ `0,303 %`.
Salgsgebyrets faktiske sats endres av SEC flere ganger i året; vi bruker en fast,
litt konservativ verdi for å holde regnestykket stabilt. *(Satsen bør
verifiseres mot SEC sin gjeldende kunngjøring før godkjenning.)*

SPY-kontrollen belastes én kjøpskostnad ved start og ingen senere.

## 4. Formler

La `r_t` være daglige avkastninger, `N` antall handelsdager.

- **Årlig avkastning (CAGR):** `(V_N / V_0)^(252 / N) − 1`
- **Annualisert volatilitet:** `σ = std(r_t) · √252` (utvalgsstandardavvik, ddof = 1)
- **Sharpe-ratio:** `S = mean(r_t) / std(r_t) · √252`. Med kontantrente 0 % er
  meravkastningen lik avkastningen.
- **Drawdown:** `DD_t = V_t / max_{s≤t} V_s − 1`. **Maks drawdown** = `min_t DD_t`.
- **Meravkastning mot kontroll:** `a_t = r_t^strategi − r_t^kontroll`.
  Informasjonsratio `IR = mean(a_t) / std(a_t) · √252`.
- **Turnover (årlig):** sum av |handelsverdi| / (2 · gjennomsnittlig V), annualisert.

## 5. Beslutning etter 12 måneder

Beslutningen tas tidligst **12 måneder** etter at klokken startet. Før det
rapporteres tallene, men ingen strategi forfremmes, legges ned eller endres på
grunn av resultatene.

Etter 12 måneder (forslag til regel — skal godkjennes):

1. En strategi **beholdes** bare hvis den har høyere totalavkastning etter
   kostnader enn **begge** kontrollene, *og* ikke dårligere maks drawdown enn
   SPY med mer enn 5 prosentpoeng.
2. Hvis ingen strategi oppfyller (1), er standardbeslutningen å **bytte til
   SPY/likevekt** — ikke å justere strategiene og prøve igjen.
3. Resultatet skrives ned i en beslutningslogg med dato, tall og begrunnelse,
   *før* neste periode starter.

## 6. Parameterendring starter klokken på nytt

Enhver endring av strategiregler, vekter, terskler, univers eller
kostnadsmodell for en strategi **nullstiller dens klokke**. Den gamle perioden
arkiveres og telles ikke med i neste vurdering. Rene feilrettinger som ikke
endrer hvilke handler som skulle vært gjort (f.eks. logging, rapportering)
nullstiller ikke klokken; det må dokumenteres i commit-meldingen hvorfor.

Dette er den viktigste regelen i dokumentet: uten den kan vi justere oss fram
til et godt resultat ved en tilfeldighet.

## 7. Ærlig forbehold om statistisk usikkerhet

- **12 måneder er kort.** Med typisk årlig volatilitet rundt 20 % og en
  meravkastning-volatilitet (tracking error) på 8–10 % mot indeksen, er
  standardfeilen på ett års meravkastning omtrent like stor som
  tracking error, altså ±8–10 prosentpoeng. En strategi som slår SPY med
  5 prosentpoeng på ett år kan fullt ut skyldes flaks. For å skille en reell
  fordel på 3 % per år fra null med rimelig sikkerhet trengs ofte 10–20 år.
- **Mange varianter er prøvd.** Prosjektet har så langt kjørt 6 parallelle
  strategier (Max_Return, Quality_Momentum, Balanced, Low_Risk, MegaCap,
  AI_Sentiment), 2 registrerte modellversjoner (`quant_baseline_v1`, `_v2`),
  4 porteføljeversjoner, V2B-skyggevarianter med og uten Claude, og flere
  runder med endrede vekter og regler før versjonene ble frosset
  (se `config/` og git-historikken). Et forsiktig anslag er 15–30 varianter. Når mange varianter
  prøves, vil den beste nesten alltid se bra ut ved en tilfeldighet.
  *Antall varianter som er prøvd skal oppgis sammen med hvert resultat.*
- Av disse grunnene skal resultatet etter 12 måneder brukes til å
  **forkaste** dårlige strategier, ikke som bevis for at en strategi er god.

---

*Dette dokumentet erstatter ikke `docs/V2_EVALUATION_PROTOCOL.md` før det er
godkjent.*
