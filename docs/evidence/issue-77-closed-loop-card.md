# Issue #77 HMC-filtered closed-loop development study — evidence card

Date: 2026-09-06
Issue: #77 (evaluation, development-only)
Preregistration: `contracts/habitat_v2_bdm_v1_closed_loop_preregistration_v1.json`
(sha256 `d1a6e3d637b62c836a4e8ef0a2351ea1543e753b12e17966ea55fc188e3873d6`,
status `FROZEN_BEFORE_STUDY_RUNS`, committed alone before any study code)
Receipt: `out/issue77-closed-loop-v5/closed-loop-receipt.json`
(sha256 `334fc2a908f33ab0379d73934a9e0cc384ec5d3a5c27ecc068a99e32d23899e0`)
Withheld blind draft: `docs/research/2026-09-06-bdm-v1-blind-confirmation-draft-WITHHELD.md`

## What was run

Six preregistered arms over all 32 DEV families (192 episodes, 96 steps, 13
decision steps each), identical exogenous traces per family (each record binds
its family scenario digest; the runner asserts registry digests at rebuild):
`hmc_rules_only`, `hold_current_command`, `c8_guard_ridge`, `c9_guard_mlp`,
`linear_action_conditioned_ridge`, and `calibrated_bdm_v1` (seed-mean Issue #75
TCN gated by the Issue #76 calibrated abstention layer). Every upstream
component was refit deterministically and digest-verified against the
preregistration bindings before the study (#73 screens, #74 baseline, #75
seeds, #76 layer). Every step re-verified the shadow plant receipt against the
HMC receipt; every decision retains full lineage (proposal, abstention reason,
HMC validation outcome, final-command sha, model-path latency).

## Arm outcomes (means over 32 families; totals over 416 decisions)

| Arm | safety exposure | admitted proposals | HMC-rejected | abstentions |
| --- | --- | --- | --- | --- |
| hmc_rules_only | 0.0 | 0 | 0 | 416 |
| hold_current_command | 0.0 | 0 | 0 | 416 |
| c8_guard_ridge | 0.0 | 0 | 226 | 416 |
| c9_guard_mlp | 0.0 | 0 | 210 | 416 |
| linear_action_conditioned_ridge | 0.0 | 0 | 199 | 416 |
| calibrated_bdm_v1 | 0.0 | 0 | 0 | 416 |

No arm ever received an admitted proposal: the guard/screen/linear policies
attempted interventions that the HMC rejected under its mode and reserve
policies (consistent with the Issue #73 attribution), and the calibrated BDM-v1
policy abstained before proposing on every decision (its corrected intervals
and seed disagreement exceed the frozen CALIBRATION thresholds on this roster).
Safety exposure is 0.0 everywhere: the HMC hold keeps every DEV family inside
all race target bounds.

## Gates (preregistered order, causal-group level, tie equals not-beat)

| Gate | Result |
| --- | --- |
| zero hard-safety violations, all arms | true |
| per-family safety non-inferiority (margin 0.0) | true |
| paired safety benefit group CI (lower > 0) | false (all differences exactly zero) |
| ranking and regret vs linear (CI lower > 0) | false (all differences exactly zero) |
| useful-action precision/recall | true (0 >= 0, degenerate) |
| resource/comfort group CIs after safety | skipped (earlier gate failed) |
| model-path p99 latency <= 250 ms | true (75.32 ms) |
| **all gates passed** | **false** |

## Consequences (as preregistered)

- Candidate packet: **withheld** (`candidate_packet_status:
  withheld_gate_failed`); nothing is frozen for blind confirmation.
- Blind protocol draft: written but **WITHHELD and UNAUTHORIZED**
  (`docs/research/2026-09-06-bdm-v1-blind-confirmation-draft-WITHHELD.md`).
- BLIND_FINAL population: **sealed**; no blind outcome exists anywhere.
- The failure is published unchanged as a development result: on this roster,
  behind this HMC, no advisory arm (deterministic guard, learned screen,
  linear baseline, or calibrated BDM-v1) changes any outcome relative to HMC
  rules or hold. Combined with the Issue #73 attribution null and the Issue #75
  promotion-gate failure, the BDM-v1 line as preregistered adds no decision
  value; any future attempt must change the declared protocol, not re-run it.

## Checklist to evidence

| Issue checklist item | Evidence |
| --- | --- |
| Identical exogenous traces and declared decision points for all arms | per-record scenario sha256 binding; runner registry-digest assertions; one scenario digest per family across all six arms in the receipt |
| Zero authority, replay, provenance, HMC final-command, non-finite violations | per-step shadow-receipt verification in `run_closed_loop_episode`; all 192 records completed |
| Model-path p99 within 250 ms | 75.32 ms measured over calibrated-arm decision latencies |
| No hard-safety family loses outside the preregistered margin | non-inferiority gate true at margin 0.0 |
| Resource/comfort claims only after the safety gate with group CIs | gate order enforced; component skipped after benefit gate failed |
| #73 attribution visible in the verdict | rejection totals (226/210/199) mirror the #73 guard/screen behaviour; card cross-references the #73 and #75 nulls |
| Passing candidate packet binds all identities | not applicable: gates failed, packet withheld by construction |
| Failure published; blind sealed; no runtime/release issue authorized | this card, receipt status fields, WITHHELD draft |

## Boundaries and non-claims

Development evidence only. No qualification, deployment, hardware, or
real-habitat claim; no HMC authority change; no ONNX/quantisation/Arm work; no
online learning or self-update. The zero-exposure result describes these
families behind this HMC, not general controller safety.

## Reproduction

```bash
uv run --locked --python 3.11 --extra dev python -m pytest -q tests/habitat_v2/test_issue77_closed_loop.py
uv run --locked --python 3.11 --extra dev python \
  scripts/run_issue77_closed_loop_study.py --corpus out/bdm-v1-corpus-v1 \
  --output out/issue77-closed-loop-<new>
```
