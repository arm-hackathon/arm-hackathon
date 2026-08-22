# Issue #53: Missing/Broken Sensors — Dropout-Robust Forecast Lane

Status: draft; ready for local review
Issue: https://github.com/arm-hackathon/arm-hackathon/issues/53
Design base: `261f50f2bff2689ecf47b2fdcc7ec345fa03bf78` (`design/issue-52-long-horizon-actions`, PR #60 ready)
Normative appendix: `2026-08-22-issue-53-missing-sensors-plan.md`
Preregistration: `contracts/habitat_v2_forecast_issue_53_preregistration_v1.json` (new, byte-frozen on plan publish)
Preregistration SHA-256: `A96245F6E717BC83B44438F9D02DBAAA42FA5DED14D3A160FD47A0F4D393D76A`
Stacks on: `ben/habitat-v2-hmc-v1@843a5c1` via Issue #52 lane

## Short design note

**The problem in plain words:** The Issue #52 forecaster is correct to abstain when its latest observation is incomplete — `PersistenceForecaster` and `ActionConditionedLinearForecaster` at `src/aeolus/habitat_v2/forecast_issue52.py:1550` / `1414` return `ABSTAIN` if `not np.all(history.available_mask[-1])`. That keeps the system safe because HMC can still accept, modify, or reject to hold (`src/aeolus/habitat_v2/hmc.py:770`), but it means one broken `ChannelSample` at `src/aeolus/habitat_v2/instrumentation.py:79` (`UNAVAILABLE/MISSING`) switches the learned lane off entirely. The job for #53 is to keep that safety while letting the model say: *"sensor 3 is silent — here is my best 32-step forecast anyway, and here is how unsure I am; if I am too unsure I will still hand back."*

**What this lane adds (without touching frozen Issue #52 evidence):**

* **Dropout-collected corpus (≈33h quiet background).** Reuses the Issue #52 offline kernel at `src/aeolus/habitat_v2/forecast_issue52_rollout.py:418` (`RolloutCheckpoint`, `RolloutResult` at `530`, `ForecastHistory` at `845`). Labels remain primary environmental telemetry + unique resource gauges from `instrumentation.py:79` (`primary_telemetry` / `operational_resource_gauges`); dropout is an *observation-only* mask applied after `instrument_v5_operational_measurement` at `src/aeolus/habitat_v2/instrumentation.py:724` and before `ObservationRecord` (`src/aeolus/habitat_v2/forecast_issue52.py:557`). Truth at `src/aeolus/habitat_v2/state.py` and evaluator `fault_receipt` truth are never mutated. Each sample carries `dropout_mask` + `dropout_config_sha256` in its content-addressed evidence; the simulator remains deterministic and replayable via `SHA256(seed||family||decision||step||descriptor)` masks.

* **Dropout-aware forecaster (development evidence only).** The existing history already stores `available_mask` (`src/aeolus/habitat_v2/forecast_issue52.py:576`, `622`). The new forecaster consumes `float32[16,W] + bool[16,W]` with a mask-aware feature path (`_feature_matrix_masked`): missing cells imputed from last-available or manifest `nominal` (`TargetDescriptor:284`), mask and time-since-observed appended as auxiliary channels. No future mask or hidden truth leaks into history. Outputs remain `float32[32,W]` mean + two 90% interval bounds (same shape) per Issue #52 §13. Intervals are calibrated *conditional on* `k = missing count on latest observation`, so uncertainty visibly widens with `k`.

* **Abstention that depends on calibrated doubt, not on any missing bit.** The current `all(available)` gate becomes `if expected_coverage_at_k < threshold or max_normalized_uncertainty > limit: ABSTAIN`. Threshold frozen on VALIDATION before FINAL. Outcome remains one of `ABSTAINED/WARMUP_NO_PROPOSAL/INVALID_OUTPUT/TIMEOUT...` at `src/aeolus/habitat_v2/forecast_issue52.py:104`, each counted separately. HMC still owns every plant step (`HMC.step`, `propose`/`arbitrate`/`preflight` at `src/aeolus/habitat_v2/hmc.py`); the learned lane has no plant handle, no capability token, and no bypass.

* **Honest per-k evaluation.** The primary forecast metric is unchanged (normalized MAE over horizons 9–32, same aggregation as Issue #52 §16). We report it at `k=0` (complete), `k=1`, `k=3`, and a full `0…6` sweep — point ratio vs the frozen `k=0` baseline and 95% family-bootstrap intervals. Coverage of the 90% intervals at each `k` and abstention PR (precision/recall of handing back when oracle error is high) are co-primary for this lane. The current model’s  `k≥1` abstention rate = 100% is the explicit baseline the new model must beat *without* increasing safety-bound exposure or dangerous-crossing false rates.

* **Written record of what it cannot do.** Alongside the numbers, `docs/evidence/issue-53-dropout-card.md` documents retained vs lost capability (per-head, per-channel, burst vs independent dropout, full-zone correlated loss), with a hard “do not deploy” note and rollback (`disable dropout lane → frozen Issue #52 abstaining lane → HMC hold`).

**Frozen boundary:** The Issue #52 model, its `DE4744E1...0702A3B` preregistration, its dataset, and its traces remain byte-identical. This lane adds new manifests, new dropout-aware datasets, and a new artifact identity (`parent_artifact_sha256` binds the frozen parent). The rule-based HMC controller remains authoritative for every action, always.

## Expected delta and frozen decision rule

Reuse Issue #52’s primary gates (forecast NMAE ≤0.90 point & <0.98 upper vs best non-neural baseline, horizons 1–8 non-inferior, zero authority/replay/provenance violations) and add dropout-specific gates frozen before collection:

* At `k=1`: NMAE vs `k=0` counterpart ≤1.15 point, ≤1.25 upper; coverage ≥ 85% (target 90%).
* At `k=3`: NMAE vs `k=0` counterpart ≤1.40 point, ≤1.60 upper; coverage ≥ 80%; abstention recall on high-error decisions ≥0.80 with precision ≥0.60.
* Safety-non-regression still dominates: total safety-bound exposure mean diff ≤0 and upper ≤0; dangerous-crossing recall diff ≥ −0.02, false-crossing diff ≤ +0.01 — at every `k`.

Family count and roster are not guessed. Same Issue #52 pilot/power method (§11) but with paired log-ratios at `k=3` as the power driver; caps remain 384 families / 2,000,000 transitions. If pilot variance at `k=3` busts caps, publish a negative pilot result.

## Delivery sequence

1. Land this design + normative plan + `habitat_v2_forecast_issue_53_preregistration_v1.json` on `design/issue-53-missing-sensors`.
2. Freeze contracts/manifests/schemas for masked history, dropout config, dataset and artifact.
3. Run pilot ≤32 families / ≤12k transitions to estimate variance at `k=0/1/3` and freeze ranking/abstention formulas if needed.
4. Collect full dropout corpus (≈33h deterministic background) and publish `dataset_manifest_sha256`.
5. Train mask-aware linear baseline first, then smallest MLP that passes VALIDATION gates; calibrate per-k intervals on VALIDATION.
6. One sealed FINAL invocation on held-out FINAL families; compute per-k degradation + abstention PR.
7. Publish honest measurement table (1 vs 3) + dropout card + rollback.
