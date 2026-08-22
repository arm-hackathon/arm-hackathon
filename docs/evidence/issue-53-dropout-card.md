# Issue #53 Dropout Capability and Limitation Card

Date: 2026-08-22
Lane: `design/issue-53-missing-sensors`
Preregistration: `contracts/habitat_v2_forecast_issue_53_preregistration_v1.json`
Preregistration SHA-256, LF-normalized: `6DFA3E084F1585FB696511C54AB676356406496DDF7A639C6D10721A0D3F41B3`
Status: **development evidence only - NOT QUALIFIED - do not deploy**

Ben is offline and all Issue #53 approval and authorization flags remain false.
HMC remains the sole proposal, arbitration, preflight, capability, plant-step,
and replay authority.

This card records the implementation boundary and the evidence boundary. It
does not claim that the model has passed the preregistered gates.

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
candidate/missingness samples and 384 candidate transitions. It verified serialization and deterministic
replay only. The required full corpus, trained artifact, sealed FINAL run, per-k
measurements, safety non-regression report, and deployment qualification do not
exist in this worktree.

## Capability Boundaries

These are implementation capabilities, not qualification claims:

1. The feature path can produce a forecast from histories with missing latest-row channels, subject to finite features and uncertainty limits.
2. Missingness is carried as data rather than silently imputed away.
3. Intervals can be conditioned on latest-row missing count after VALIDATION calibration.
4. The deterministic collector can generate complete labeled replay samples without mutating plant truth.

## Limitations and Stop Conditions

1. No trained model has passed the Issue #53 NMAE, coverage, abstention, safety, replay, provenance, split, or latency gates.
2. Correlated burst mode is implemented and contract-tested but not qualified. Do not treat it as a whole-zone outage solver.
3. Resource gauges are anchors by default. Gauge dropout and adversarial safety-channel dropout are outside current evidence.
4. Rates above the preregistered pilot setting, burst lengths beyond the configured maximum, and other OOD patterns are unmeasured.
5. A missing input never authorizes the learned lane to bypass HMC, alter a plant state, mint capability, or weaken a safety limit.
6. The collector's 33-hour estimate is a plan estimate, not an observed runtime result.

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

**Bottom line:** the dropout-aware implementation and replay-backed collector
exist, but Issue #53 is not trained, measured, qualified, or deployable.
