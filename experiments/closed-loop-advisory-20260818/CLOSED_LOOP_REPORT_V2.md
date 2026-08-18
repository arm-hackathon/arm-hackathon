# AEOLUS Habitat V2 — Closed-Loop Advisory Experiment v2 (Broadened Roster)

**Date:** 2026-08-18
**Status:** Development evidence only. Not qualification. Not deployment. No learned actuator authority.
**Preregistration:** `preregistration-v2.json`, self-SHA-256 `0b7e3e7be5413d0093ff51d6065b314adbcaffeb25442107b384e9e4f7795712` (member T13 corrected to T12 before any v2 outcome; documented in-file)
**Results:** `paired-v2-results.json` (immutable, written once); 238 runs total

## What changed from v1

All 17 held-out clusters (contingency, dormant, EVA-transition, occupied modes), fault members T01/T07/T12, repetitions R01+R02 = 102 fault pairs, plus 17 HEALTHY no-harm pairs. Resource-consumption deltas (battery/oxygen/sorbent) recorded per run.

## Headline result

- **Fault pairs: 78 better, 24 equal, 0 worse.** Pre-registered majority rule met.
- 72 of the 78 wins were driven to **exactly zero** threshold exceedance; 96/102 advised fault runs had zero exceedance overall.
- All 24 equal pairs were dormant scenarios where control already had zero exceedance and the adviser correctly never intervened.
- **Zero terminal regressions.** Strict trace replay and shadow-physics digest equality held in all 238 runs. 793 proposals, all VALID, **81 overridden by HMC** (authority exercised).

## Healthy no-harm check — honest read

- Pre-registered letter: advised exceedance must be exactly 0 in all 17 healthy pairs. **Met in 16/17.**
- The miss: `eva_transition/high/balanced-initial-state` — advised 0.0378. But canonical HMC in that same scenario scored **8.354**: the adviser reduced exceedance by 99.5%, it just didn't reach zero.
- Also notable: canonical HMC exceeds warning thresholds in 13 of 17 healthy scenarios (up to 20.33) — warning-level excursions are normal in this simulator's canonical policy; the adviser eliminated them in 12 of those 13.

## Resource cost — flagged per preregistration

The safety improvement is not free. Where the adviser intervenes it consumes more:

- Battery: median **+757 Wh** per pair (max +1541 Wh)
- Oxygen store: median **+1.97 mol** (max +3.80)
- CO2 sorbent: median **+6.04 mol** (max +7.64)

Many pairs flagged "more of all three resources." All advised runs still stayed above the 0.2 resource-warning floor (included in the zero exceedances), but a resource-constrained scenario could make this trade-off binding. Dormant/equal pairs consumed exactly zero extra.

## Claim boundary (unchanged)

Development evidence: a forecast-advised HMC outperformed canonical HMC on physical safety metrics across the full held-out roster, at a measurable resource cost. This is **not** qualification (no sealed corpus/CAL gate), **not** missing-sensor-robust (no availability masks), and **not** deployment authorization.

## Next gates

1. **Resume the availability-aware FIT/CAL corpus** (stopped at 1,856/3,744, ~33h) to qualify an availability-masked version of this adviser — now justified by demonstrated closed-loop value.
2. Optionally: resource-weighted risk functional (v3 preregistration) to trade safety vs consumables explicitly.
