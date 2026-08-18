# AEOLUS Habitat V2 — Closed-Loop Advisory Experiment (Development Evidence)

**Date:** 2026-08-18
**Status:** Development evidence only. Not qualification. Not deployment. No learned actuator authority.
**Workspace:** `C:\Users\Nxiss\code\aeolus-next-gates-20260818\closed-loop-v1\` (isolated; sealed repositories imported read-only)

## Plain-language summary

The historical development model was integrated into the real Habitat V2 control loop as an **adviser**: at each step it forecasts the next 8 steps for each approved catalogue action, picks the lowest-risk option, and submits it as a proposal. HMC keeps full authority — it validates, can override, and executes the final command.

In 8 paired fault scenarios (identical scenarios, seeds and nonces across arms), the model-advised arm kept every zone under every warning threshold for the entire run, while canonical HMC alone exceeded warning thresholds in all 8 scenarios.

## Pre-registration (frozen before outcomes)

- `preregistration.json`, self-SHA-256: `a0c0839da3539aaf6108648060d5a0f0cba1ed0d9506b71d33bdca1a168e265c`
- Risk functional: normalized threshold exceedance using the HMC contract's own frozen `health_policy` environmental/resource thresholds (high CO2 2500/5000 ppm, high humidity 0.65/0.75, high temperature 300/303 K, low oxygen 0.285/0.27, low temperature 291/288 K, resource gauges 0.2/0.1).
- Candidate order: NO_PROPOSAL first, then the four frozen catalogue actions; argmin risk, ties to earliest.
- Adviser active from step 16 onward (16-step history window required).
- Scenarios: first 4 sorted held-out clusters from the verified offline split, members T01 and T07, repetition R01.
- Success rule: advised strictly lower integrated exceedance in a majority of 8 pairs, zero terminal regressions, replay passes everywhere.

## Verification performed

- **Determinism:** control arm run twice → identical trace SHA-256, final-state SHA-256, and metrics (`smoke-control-determinism.json`).
- **Physics integrity:** shadow-physics digest equality with HMC receipts asserted every step of all 16 runs; strict completed trace replay passed in all 16 runs; every run terminal status `COMPLETED`.
- **Divergence is real:** advised vs control final-state and trace hashes differ in every pair — the measured improvement comes from genuinely different executed trajectories, not a metrics artifact.
- **HMC authority exercised:** 46 proposals made, 46 admitted as VALID, 3 overridden by HMC arbitration; adviser never bypassed arbitration.
- Result file: `paired-v1-results.json` (immutable, written once).

## Results (primary metric: integrated threshold exceedance, lower is better)

| Scenario (held-out cluster) | Member | Control | Advised | Delta |
|---|---|---|---|---|
| contingency/low/balanced-initial-state | T01 | 1.3756 | 0.0 | −100% |
| contingency/low/balanced-initial-state | T07 | 1.3756 | 0.0 | −100% |
| contingency/low/thermal-air-processing-skew | T01 | 2.2977 | 0.0 | −100% |
| contingency/low/thermal-air-processing-skew | T07 | 2.2977 | 0.0 | −100% |
| contingency/nominal/crew-metabolic-humidity-skew | T01 | 19.9406 | 0.0 | −100% |
| contingency/nominal/crew-metabolic-humidity-skew | T07 | 19.2521 | 0.0 | −100% |
| contingency/nominal/pressure-inventory-skew | T01 | 9.6596 | 0.0 | −100% |
| contingency/nominal/pressure-inventory-skew | T07 | 9.6596 | 0.0 | −100% |

**8/8 pairs better, 0 worse, pre-registered success rule met.** Control arm exceeded warning thresholds on 9–29 steps per run; advised arm on zero. Mechanism: the adviser proposes a catalogue mitigation a few steps before the excursion (e.g. proposals at steps 47/49 where control first exceeds at step 60), preventing the threshold crossing entirely.

## Limitations (read before quoting)

1. **Development model:** trained on the Historical V2 archive without operational availability masks. Missing-sensor robustness is unproven.
2. **Small roster:** 4 contingency-mode clusters × 2 fault members × 1 repetition. Occupied/EVA/dormant modes and other fault members are untested in closed loop.
3. **Passive baseline:** canonical HMC in these scenarios lets warning-level excursions ride; the catalogue contains strong pre-approved mitigations. The comparison is honest but the baseline is deliberately minimal.
4. **Resource cost:** the advised trajectories stayed above the 0.2 resource-gauge warnings (exceedance 0 includes resources), but explicit intervention-cost accounting is not yet reported.
5. **Not qualification:** no sealed corpus, no CAL gate vs persistence, no validation access, no deployment authorization.

## Next gates (in order)

1. Broaden the closed-loop roster: remaining 13 held-out clusters (occupied/EVA/dormant), more members, both repetitions; add resource-consumption deltas.
2. If signal holds: decide whether to resume the availability-aware FIT/CAL corpus (currently stopped at 1,856/3,744 packets) to qualify an availability-aware version of this adviser.
3. Only then consider any qualification-language or CV claim updates.
