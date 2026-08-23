# Issue #53 Dropout Capability and Limitation Card

Date: 2026-08-23
Lane: `design/issue-53-missing-sensors`
Preregistration: `contracts/habitat_v2_forecast_issue_53_preregistration_v1.json`
Preregistration SHA-256, LF-normalized: `A96245F6E717BC83B44438F9D02DBAAA42FA5DED14D3A160FD47A0F4D393D76A`
Status: **QUALIFIED forecast lane - do not deploy as a plant-control lane**

HMC remains the sole proposal, arbitration, preflight, capability, plant-step,
and replay authority.

This card records the implementation boundary and the evidence boundary. The
sealed model version recorded below passed every preregistered Issue #53 gate;
qualification does not grant plant-step, capability, safety, or HMC authority.

## What Exists

The lane adds a deterministic, observation-only dropout view over the frozen
Issue #52 history contract in
`src/aeolus/habitat_v2/forecast_issue53_dropout.py`.

* Masks are derived from SHA-256 inputs bound to seed, family, decision step, history offset, and descriptor.
* Native unavailable cells remain unavailable; protected resource gauges are never newly dropped when `resource_gauge_dropout=false`.
* Independent, correlated per-zone burst, and mixed modes are represented in the config contract.
* Forward-fill plus descriptor nominal imputation produces finite features; mask density, age, and mask-aware slope are explicit features.
* Labels remain complete and are not derived from masked inputs.
* The collector reuses `build_offline_checkpoint`, `rollout_catalogue`, and `training_samples_from_rollouts` and writes `samples.jsonl` with a content digest.
* `DropoutAwareLinearForecaster.fit_for_scenario()` fits TRAIN samples only; `calibrate()` accepts VALIDATION samples only.
* `abstention_pr()` requires externally supplied truth-backed oracle errors rather than using the model's own errors to define its oracle.

## Evidence Status

The local smoke run generated one replay-backed family with 48 serialized
candidate/missingness samples and 384 candidate transitions. It verified
serialization and deterministic replay only; it was not used as a
qualification measurement.

The bounded pilot then used 32 whole families and 1,536 serialized samples
(12,288 candidate transitions). Its final committed pilot report was
`NOT QUALIFIED` only because it was a non-sealed validation run; all substantive
forecast, coverage, abstention, safety, replay, provenance, authority, and
latency gates were true. Pilot report SHA-256:
`0dbe11b3a73d821e657004b9b28d325470ad18b2c9c9c1fed7afd43aa71e4cba`.
Pilot dataset identity SHA-256:
`225cc05037ea68028174c807da9cd0946a56580baf558e97735ca03989253047`.
Pilot samples SHA-256:
`48057386abce6cef4711a42013e07e8b1079a66b98692b15227808e8f4e39350`.

The frozen qualification corpus contains 384 whole families, 18,432 serialized
samples, and 147,456 candidate transitions, with 269 TRAIN, 58 VALIDATION, and
57 FINAL families. The sealed model was fit on TRAIN, calibrated on VALIDATION,
and evaluated exactly once on FINAL. Every gate passed for source commit
`b812129cb6ffb0d3566b6542036ce5b29fcf6161`.

An earlier sealed invocation for model source
`dccf0b37c4ab3447c7a2c1844622bcf1b575999e` remains immutable and
`NOT QUALIFIED` because its `k=3` safety exposure upper 95% bound was positive.
The qualified model above is a new model version and does not overwrite that
negative result.

Sealed bundle: retained in the isolated qualification environment; the bundle
is not part of this repository.

* Model artifact: `issue53-dropout-linear-25d407138a448069-cal-3905844701ead27a`
* Artifact SHA-256: `9ffd50bf2127c5d8f47c687e435f89677f10dc640e58c0374b9730f048a0b186`
* Report SHA-256: `67bcfefeec58e03be68f5af87331d35b692799a28931f33361d591d3cba18c89`
* Sealed lock SHA-256: `a373ffa8d3dcba4de8618bbad0da4ec3de97f47b8a39b7e982532734da117cdc`
* Dataset SHA-256: `07f7e7c5cdc12043e84eac40c4e984afc88bd6e7498ce828b6e04a56c8d2ff29`
* Samples SHA-256: `cd50e8b76b5f3a8b127126b0d471ce82e8e40e5e9f48985cbf1ee58d27ebf9d7`

## Qualification Record

The status is `QUALIFIED` for the sealed model version above because every step
below has an artifact or measurement digest and the preregistered gates pass.

1. **Freeze provenance.** Complete. The Issue #52 parent
   preregistration digest, Issue #53 preregistration digest, dropout config
   digest, scenario digest, HMC contract digest, source commit, and runner
   identity. Abort if Issue #52 bytes changed.
2. **Run the bounded pilot.** Complete. The pilot used no more than 32 whole families and
   12,288 physical candidate transitions with the collector; validate the
   config, dataset manifest, samples digest, family split, serialized sample
   digests, tensor shapes, missingness views, and replay binding. Publish
   actual runtime, storage, infeasibility rate, and variance. Do not train
   during collection.
3. **Run the qualified corpus.** Complete. The frozen family roster was collected within the
   384-family / 2,000,000-transition caps on the isolated runner. Publish the
   dataset manifest SHA-256 and samples SHA-256. Stop on failed validation,
   duplicate/clone families, leakage, seed reuse, cap violation, or replay
   mismatch.
4. **Fit and calibrate once.** Complete. The dropout-aware model was fit on
   TRAIN only; normalizers were fit on TRAIN only; per-k intervals and
   abstention thresholds were calibrated on VALIDATION only; the artifact and
   all digest bindings were frozen before FINAL.
5. **Pass VALIDATION selection.** Complete. Forecast NMAE, per-k degradation,
   interval coverage, abstention behavior, observability/action
   identifiability, safety, replay, provenance, and latency gates. Publish a
   negative result rather than loosening thresholds.
6. **Run one sealed FINAL.** Complete. The frozen artifact was evaluated once
   on FINAL families, including the `k=0,1,3,6` sweep, external truth-backed
   `oracle_errors` for abstention PR, and all safety non-regression metrics.
7. **Publish results.** Complete. `issue-53-measurements.md` records actual
   FINAL values, artifact and dataset digests, retained capabilities, and lost
   capabilities.
8. **Keep deployment blocked.** Active. Qualification of the forecast lane does not
   grant it plant-step, capability, or HMC authority.

## Capability Boundaries

These are implementation capabilities, not qualification claims:

1. The feature path can produce a forecast from histories with missing latest-row channels, subject to finite features and uncertainty limits.
2. Missingness is carried as data rather than silently imputed away.
3. Intervals can be conditioned on latest-row missing count after VALIDATION calibration.
4. The deterministic collector can generate complete labeled replay samples without mutating plant truth.

## Limitations and Stop Conditions

1. Qualification covers the preregistered independent dropout setting only. It does not qualify correlated burst mode, mixed mode, higher dropout rates, or other out-of-distribution patterns.
2. Resource gauges are anchors by default. Gauge dropout and adversarial safety-channel dropout are outside current evidence.
3. The qualified model is forecast-only and does not rank, select, or execute plant actions; HMC retains all proposal, arbitration, preflight, capability, plant-step, and replay authority.
4. A missing input never permits the learned lane to bypass HMC, alter a plant state, mint capability, or weaken a safety limit.
5. The collector's 33-hour estimate is a plan estimate, not an observed runtime result; deployment-target performance requires a separate gate.

## Rollback

Disable the dropout lane and its artifact reference. Fall back to the frozen
Issue #52 lane, which abstains on an incomplete latest observation. If that lane
also abstains, HMC retains its existing hold or emergency behavior. No HMC,
physics, actuator, safety, or Issue #52 contract changes are required for
rollback.

## Evidence Bundle

* `contracts/habitat_v2_forecast_issue_53_preregistration_v1.json`
* `scripts/collect_issue53_dropout_dataset.py`
* `src/aeolus/habitat_v2/forecast_issue53_dropout.py`
* `tests/habitat_v2/test_forecast_issue53_dropout.py`
* `docs/evidence/issue-53-measurements.md`

**Bottom line:** the independent dropout-aware forecast lane is measured and
qualified for the frozen Issue #53 contract. It is not a plant-control lane and
must not bypass HMC or the separate safety-core review.
