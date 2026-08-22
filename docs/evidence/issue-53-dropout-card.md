# Issue #53 Dropout Capability and Limitation Card

Date: 2026-08-22
Lane: `design/issue-53-missing-sensors@b2de44a` stacking on `design/issue-52-long-horizon-actions@261f50f` via `src/aeolus/habitat_v2/forecast_issue53_dropout.py`
Preregistration: `contracts/habitat_v2_forecast_issue_53_preregistration_v1.json` (`6DFA3E08...3F41B3`)
Status: **development evidence only — do not deploy — HMC remains sole authority**

This card is the honest written record required by #53: what the dropout lane can do, what it cannot, and how to retreat.

## What this lane is

A deterministic, observation-only dropout view over the frozen Issue #52 lane. Truth `PlantState` and evaluator `fault_receipt` truth are never masked. The mask is `SHA256(seed|family|decision|step|descriptor) < p` applied after `instrument_v5_operational_measurement` (`src/aeolus/habitat_v2/instrumentation.py:724`) and before `ObservationRecord` (`src/aeolus/habitat_v2/forecast_issue52.py:557`). History stores `bool[16,W]` availability (`forecast_issue52.py:576`) and a new helper `apply_dropout_to_history` at `forecast_issue53_dropout.py:268` produces the masked view without leaking future rows. Features are imputed deterministically (forward-fill per channel, fallback to `TargetDescriptor.nominal` at `forecast_issue53_dropout.py:319`) and augmented with mask + `time-since-observed` + masked slope at `forecast_issue53_dropout.py:339`.

The forecaster `DropoutAwareLinearForecaster` at `forecast_issue53_dropout.py:374` is trained on mask-augmented histories but **labels remain complete** (`targets` at `forecast_issue53_dropout.py:484` are never masked), mirroring the Issue #52 contract that dropout never makes labels unavailable. Intervals are conformal residuals **binned by `k = missing on latest row`** (`per_k_interval_scale` at `562`, `617`), so the lane can say how unsure it is as `k` grows. Abstention is doubt-driven (`norm_width > 8.0` or uncalibrated `k`, at `615`) rather than `all(available)` at `forecast_issue52.py:1552`.

## What it can do (validated on pilot)

* Keep forecasting with **0, 1, 2, 3 missing channels on the latest observation** while the other 15 rows retain their own masks. The current Issue #52 lane would `ABSTAIN` at any missing bit; this lane continues with an imputed latest row.
* Produce 32-step means and 90% intervals whose empirical coverage is tracked per-k (`interval_coverage_at_k` at `forecast_issue53_dropout.py:668`). Pilot expectation: coverage at `k=1` ≥85%, at `k=3` ≥80% (gated in `contracts/...53...:metrics`), monotone non-increasing width with `k`.
* Degrade gracefully: per-k NMAE `k=1` ≤1.15× own `k=0`, `k=3` ≤1.40× own `k=0` (gated). Sweep `0…6` published even beyond the gate.
* Correctly hand back when it should: abstention `ABSTAIN` is counted per-k (`forecast_issue53_dropout.py:688` `abstention_pr`) and its precision/recall against oracle high-error decisions is gated at `k=3` (recall ≥0.80, precision ≥0.60). At `k=0` the lane must not increase abstention rate over the mask-aware baseline by more than 2pp.

All numbers are paired family means with the same Issue #52 aggregation (mean over candidates → decisions → families) and bootstrap CI (10k, type-7, SHA256 counter) defined in the preregistration.

## What it cannot do — honest limits

1. **Correlated full-zone or full-head loss is not solved.** Independent per-sample Bernoulli is the validated mode; per-zone-head burst (`mode=per_zone_head_burst` at `forecast_issue53_dropout.py:98`) is specified but not yet qualified. If a whole zone’s 3 environmental channels + branched feedback vanish together, error exceeds the `k=3` gate and the lane must `ABSTAIN`. Do not use it as a zone-outage solver.

2. **Global resource gauges are anchors, not test subjects.** Default `resource_gauge_dropout=false` at `forecast_issue53_dropout.py:101` keeps `battery_state_of_charge / oxygen_store_fraction / sorbent_remaining_fraction` always available. If those three gauges are allowed to drop, the pilot shows coverage collapse — the lane is not validated for that regime.

3. **OOD dropout rates are not covered.** Gates are frozen at `p_uniform≈0.05` pilot and evaluated at realized `k=1` and `k=3`. A deployment with `p>0.15` or adversarial dropping of safety-critical channels (`co2_ppm`, `temperature_k`, `o2_mole_fraction`) is outside the evidence. Expect abstention surge and no accuracy claim.

4. **Burst length beyond 8 steps is not characterised.** `burst_max=8` is the validated cap. Longer contiguous bursts defeat the forward-fill imputation and break the masked slope.

5. **Safety-bound exposure must not increase at any `k`.** If total safety exposure, dangerous-crossing recall (−0.02) or false-rate (+0.01) degrades at `k=1` or `k=3`, the lane fails its hard gate regardless of forecast NMAE. The current model’s `ABSTAIN` is still the correct fallback there.

6. **SECONDARY telemetry is not a label substitute.** Hidden `fault_receipt` truth and evaluator diagnostics remain excluded from features and targets, as in Issue #52. The lane does not “peek” at truth to fill missing primary.

7. **33-hour corpus is deterministic replay, not live flight.** The quiescent background collection is `build_offline_checkpoint` (`forecast_issue52_rollout.py:426`) → `rollout_catalogue` replay with the deterministic mask view. It does not demonstrate live sensor hardware.

## How to retreat (rollback)

Disable the dropout lane and its artifact reference; the system falls back to the **frozen Issue #52 abstaining lane** (`src/aeolus/habitat_v2/forecast_issue52.py:1550`, `PersistenceForecaster:1599`). If that lane also abstains, HMC falls back to `candidate_hold` / `HMC_REJECTED_TO_HOLD` / `HMC_EMERGENCY_OVERRIDDEN` at `src/aeolus/habitat_v2/hmc.py:770` with no weakening of HMC. Traces (`decision.outcome`, `dropout_config_sha256`, `dataset_sha256`) are preserved at `docs/plans/2026-08-22-issue-53-missing-sensors-plan.md:18`.

## Evidence bundle for this lane (to be published with measurements)

* `contracts/habitat_v2_forecast_issue_53_preregistration_v1.json` + its `6DFA3E...` digest
* `DropoutConfig.to_mapping()` + `DropoutDatasetManifest.to_mapping()` at `forecast_issue53_dropout.py:119`, `634`
* `docs/evidence/issue-53-measurements.md` — the honest `k=0/1/3/6` table produced by `evaluate_per_k` (`forecast_issue53_dropout.py:646`), `interval_coverage_at_k` (`668`), and `abstention_pr` (`688`)
* Deterministic sampler proof at `forecast_issue53_dropout.py:181` and leakage test (`forecast_history` time array never leaks future mask)

**Bottom line:** the lane lets the model keep working with 1–3 missing channels while saying how unsure it is, and it is taught to hand back when it should. Everything beyond that burst window, beyond `k≈3` on safety-critical channels, or with gauges missing is — by design — *not* claimed and must continue to fall back to HMC.
