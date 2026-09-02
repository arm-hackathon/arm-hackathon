# Issue #56 V4 Model — Final Protocol And Conclusive Result

Status: **DEVELOPMENT EVIDENCE ONLY — ADVISORY MODEL — NOT QUALIFIED OR DEPLOYABLE**
Line status: **CLOSED — conclusive result reached at protocol revision v10**

This document is the single consolidated presentation of the Issue #56 V4
model-study protocol in its final form. The machine-readable protocol is
[`contracts/habitat_v2_forecast_issue_56_v4_model_preregistration_v10.json`](../contracts/habitat_v2_forecast_issue_56_v4_model_preregistration_v10.json);
every claim below is bound to SHA-256 receipts listed in
[Final result receipts](#final-result-receipts).

## Model / protocol correspondence

| Model | Protocol | Contract file |
|---|---|---|
| V1 | V1 protocol | `habitat_v2_forecast_issue_56_preregistration_v1.json` |
| V2 | V2 protocol | `habitat_v2_forecast_issue_56_v2_preregistration_v1.json` |
| V3 (frozen baseline) | V3 protocol | `habitat_v2_forecast_issue_56_v3_preregistration_v2.json` |
| **V4 (final)** | **V4 protocol — final revision v10** | `habitat_v2_forecast_issue_56_v4_model_preregistration_v10.json` |

The `_v10` suffix is the revision counter of the single V4 model-study
protocol lineage; revisions v1–v9 were superseded iterations on the way to the
conclusive result.

## Final protocol terms (revision v10)

- Population: 32 scenario families in 16 paired-sensor condition groups;
  20 TRAIN / 6 VALIDATION / 6 EVALUATION families under split
  `issue56_v4_model_split_v8`; fixture revision 2 (all operating conditions
  start at the O2 upper bound so every condition group produces
  boundary-crossing events).
- Candidates: `c8_o2_excess_guard` and `c9_o2_guard_statistical`; Stage B
  replays every Stage A passer (`replay_all_stage_a_passers_v1`).
- Selection contract `context_gated_select_v2_statistical_dormant`:
  - mechanistic guard — in an occupied O2-excess profile (mean primary O2 at
    or above 0.285), propose `normal-dormant-v1` once, suppressed once dormant
    is the current command;
  - statistical dormant admission (`c9` only) — in any operating mode, admit
    dormant when the calibrated upper-bound screen passes and predicted
    relative safety exposure is negative, with the same repeat suppression;
  - critical HMC health always abstains.
- Stage A gates: zero authority/replay/provenance/admission violations,
  useful-action count ≥ 16, abstention ≤ 0.8, latency p99 ≤ 250 ms,
  dangerous-event recall ≥ 0.98.
- Superiority criterion (per evaluation family, vs the frozen V3 baseline):
  no family may lose, at least four of six families must win outright,
  admissions and aggregate paired safety exposure no worse than V3, and zero
  HMC mismatches. A family wins when V4's episode safety exposure is strictly
  lower than V3's.
- Authority boundary: HMC remains the sole final-command, plant-step, and
  replay authority; the model is advisory-only; protected final-suite access
  is prohibited; V3 artifacts are immutable.

## Protocol lineage (revisions of the V4 model-study protocol)

| Rev | Status | Outcome |
|---|---|---|
| v1 | superseded | first development study, negative (historical, immutable) |
| v2 | superseded (corpus contract still load-bearing) | mask correction; corpus schema bound to this revision |
| v3 | superseded | stage A/B machinery; negative |
| v4 | superseded | extended roster; negative |
| v5 | superseded | context gating against dormant collapse; negative |
| v6 | superseded | split redesign; found the admission ceiling ([evidence](evidence/issue-56-action-risk-v4-model-v6.md)) |
| v7 | superseded | early-intervention gate; found the training-data asymmetry ([evidence](evidence/issue-56-action-risk-v4-model-v7.md)) |
| v8 | superseded | fixture revision 2; aggregate superiority, not per-family ([evidence](evidence/issue-56-action-risk-v4-model-v8.md)) |
| v9 | superseded (rollback tag `v9-perfamily-win`) | per-family criterion; 4 wins / 2 ties ([evidence](evidence/issue-56-action-risk-v4-model-v9.md)) |
| **v10** | **FINAL** | **6 wins / 0 ties / 0 losses** ([evidence](evidence/issue-56-action-risk-v4-model-v10.md)) |

## Final result (v10, candidate `c9_o2_guard_statistical`)

| Family | Condition | V3 exposure | V4 exposure | Δ rel | Outcome |
|---|---|---:|---:|---:|---|
| g0006 sensor-b | high_load/laboratory | 5.007e-04 | 1.907e-04 | −61.9% | WIN |
| g0006 sensor-a | high_load/laboratory | 5.007e-04 | 1.907e-04 | −61.9% | WIN |
| g0001 sensor-b | nominal/fan | 4.292e-04 | 1.907e-04 | −55.6% | WIN |
| g0001 sensor-a | nominal/fan | 2.861e-04 | 1.907e-04 | −33.3% | WIN |
| g0011 sensor-b | eva/cooling | 1.144e-03 | 5.722e-04 | −50.0% | WIN |
| g0011 sensor-a | eva/cooling | 1.144e-03 | 6.199e-04 | −45.8% | WIN |

Six wins, zero ties, zero losses; admissions 6 vs 2; aggregate paired safety
exposure −3.417e-04 (strictly better); zero HMC mismatches and zero emergency
overrides; all preregistered gates passed. `c8_o2_excess_guard` also meets the
criterion with the V9 profile (4 wins / 2 ties / 0 losses).

## Final result receipts

- Source commit: `b0ffad1e21f7baa6edb021b4fc17fcf0c012fde9` (clean worktree)
- Protocol SHA-256: `e750689158c7311a57e6b5bc648cce11045034f7229ded5ff3ceae3c43c08dbb`
- Corpus manifest SHA-256: `19a61a8d5e084bc7b62fae174b93661722ebcb87d187594b3f16650d7b021c8b`
- Corpus samples SHA-256: `81ea37872cee1d30a0198517ee5f45c6162e99895f506c24466dec5d8c6fce98`
- Study results SHA-256: `e09b8be308612fc91f603dcd343c304799598290c9dcef9c56882e706d519565`
- Frozen V3 baseline refit matches frozen artifact: true (both candidate replays)
- Rollback point: tag `v9-perfamily-win` → commit `d2c3b69` (the V9 result,
  without the statistical-dormant clause)

## Reproduction

```bash
uv run --locked --python 3.11 --extra dev python scripts/build_action_risk_v4_corpus.py \
  --output out/issue56-v4-corpus-v10-full-01 \
  --families 32 --split-protocol issue56_v4_model_split_v8

uv run --locked --python 3.11 --extra dev python scripts/verify_action_risk_v4_corpus.py \
  --corpus out/issue56-v4-corpus-v10-full-01 \
  --split-protocol issue56_v4_model_split_v8

uv run --locked --python 3.11 --extra dev python scripts/run_action_risk_v4_study_v3.py \
  --corpus out/issue56-v4-corpus-v10-full-01 \
  --output out/issue56-v4-study-v10-full-01 \
  --protocol-version v10
```

## Boundaries

This result is simulator development evidence only. It does not establish
qualification, certification, deployment readiness, hardware performance,
real-world safety, or authority to control equipment. The V4 model remains
advisory-only; HMC remains the sole final-command, plant-step, and replay
authority; any runtime integration requires separate review per the protocol's
artifact clause.
