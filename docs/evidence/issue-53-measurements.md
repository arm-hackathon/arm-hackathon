# Issue #53 Honest Measurements - 1 vs 3 Sensors Missing

Status: **NOT QUALIFIED**. This file is a sealed-run template plus a local
collector smoke record. It contains no claimed FINAL metric values.

The Issue #53 preregistration remains authoritative:
`contracts/habitat_v2_forecast_issue_53_preregistration_v1.json`.
The dropout lane is development evidence only; HMC remains the sole proposal,
arbitration,
preflight, capability, plant-step, and replay authority.

## Run Identity

* Dropout config: `DropoutConfig(p_uniform=0.05, mode=independent, resource_gauge_dropout=false, max_missing_per_row=6, seed=530053)`
* Dropout config SHA-256: `c288962aa6d5d53018e866ba19b29d951b21bbef8a7c4463b181692472535f7c` (canonical config digest)
* Issue #53 preregistration SHA-256, LF-normalized: `A96245F6E717BC83B44438F9D02DBAAA42FA5DED14D3A160FD47A0F4D393D76A`
* Frozen Issue #52 parent preregistration SHA-256, LF-normalized: `DE4744E127D2946A43D623EC90D3289B0A3735C99E62C8CECCD87768E0702A3B`
* Dataset manifest: produced by `build_dropout_dataset_manifest`; no qualification manifest is published here
* Full corpus: **NOT RUN**; the preregistration target is up to 384 whole families and up to 2,000,000 candidate transitions
* Sealed FINAL evaluation: **NOT RUN**

## Local Smoke Record

The collector was executed locally for one family to verify the replay and
serialization path only:

* Command: `python scripts/collect_issue53_dropout_dataset.py --families 1 --output <temporary-directory>`
* Replay output: 12 candidates x 4 fixed latest-row missingness views = 48 serialized samples, 384 candidate transitions
* Output files: `dropout_config.json`, `dataset_manifest.json`, `samples.jsonl`
* Result: deterministic replay-backed samples were written; this is not a pilot
  measurement and is not a qualification corpus

## Qualification Checklist

This lane may be marked `QUALIFIED` only after the following evidence exists
and all preregistered gates pass:

1. Frozen provenance record: Issue #52 parent preregistration digest, Issue #53
   preregistration digest, dropout config digest, scenario digest, HMC
   contract digest, source commit, and isolated runner identity.
2. Bounded pilot: at most 32 whole families and 12,288 physical candidate
   transitions, with the dataset manifest, samples digest, split, sample
   hashes, replay binding, actual runtime/storage, infeasibility rate, and
   variance report.
3. Full corpus: the frozen roster within 384 families and 2,000,000 physical
   candidate transitions, validated for clone families, sample coverage,
   leakage, replay determinism, and cap compliance.
4. Frozen artifact: model fitted on TRAIN only; normalizers fitted on TRAIN
   only; per-k intervals and abstention thresholds calibrated on VALIDATION
   only; artifact digest and all contract bindings recorded before FINAL.
5. VALIDATION report: forecast NMAE, per-k degradation, coverage, abstention,
   observability/action identifiability, safety, replay, provenance, split,
   and latency evidence.
6. One sealed FINAL invocation: the frozen artifact evaluated once on FINAL
   families, including the required `k=0..6` sweep, truth-backed
   `oracle_errors` for abstention PR, and safety non-regression.
7. Publication: this table filled with measured values and the dropout card
   updated with retained and lost capabilities. Deployment remains blocked
   because HMC retains all plant-step and capability authority.

All seven items are currently incomplete, so the status is `NOT QUALIFIED`.

## Forecasting Degradation - NMAE Horizons 9-32

Gate: `k=1` vs own `k=0` <=1.15 point / <=1.25 upper; `k=3` <=1.40 / <=1.60.
The required sealed FINAL values are unmeasured.

| k | NMAE mean | vs own k=0 point | 95% upper | n families | Pass |
|---|---:|---:|---:|---:|---|
| 0 | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | NOT RUN |
| 1 | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | NOT RUN |
| 2 | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | NOT RUN |
| 3 | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | NOT RUN |
| 4 | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | NOT RUN |
| 5 | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | NOT RUN |
| 6 | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | NOT RUN |

No degradation claim is made. Values must come from a single sealed FINAL
invocation after TRAIN fitting and VALIDATION calibration are frozen.

## Interval Coverage

Gate: empirical 90% coverage >=85% at `k=1` and >=80% at `k=3`.

| k | Coverage | Mean interval width (normalized) | n cells | Pass |
|---|---:|---:|---:|---|
| 0 | UNMEASURED | UNMEASURED | UNMEASURED | NOT RUN |
| 1 | UNMEASURED | UNMEASURED | UNMEASURED | NOT RUN |
| 3 | UNMEASURED | UNMEASURED | UNMEASURED | NOT RUN |
| 5 | UNMEASURED | UNMEASURED | UNMEASURED | NOT RUN |
| 6 | UNMEASURED | UNMEASURED | UNMEASURED | NOT RUN |

`DropoutAwareLinearForecaster.calibrate()` now accepts VALIDATION samples only
and bins normalized residual scales by latest-row missing count. This code path
has regression coverage; no FINAL coverage result exists.

## Abstention Quality

Gate at `k=3`: recall >=0.80 and precision >=0.60 against externally supplied,
truth-backed oracle errors. The oracle must not be derived from the model's own
predictions. `abstention_pr()` therefore requires an aligned `oracle_errors`
sequence.

| k | Abstention rate | Precision | Recall | Threshold | Pass |
|---|---:|---:|---:|---:|---|
| 0 | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | NOT RUN |
| 1 | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | NOT RUN |
| 3 | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | NOT RUN |
| >=5 | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | NOT RUN |

## Safety Non-Regression

The hard safety gates are also unmeasured. They must be evaluated at each
required `k` using the same eligibility and family aggregation as Issue #52.

| Metric | k=1 point | k=1 upper | k=3 point | k=3 upper | Gate |
|---|---:|---:|---:|---:|---|
| Safety-bound exposure | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | NOT RUN |
| Dangerous-crossing recall | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | NOT RUN |
| Dangerous-crossing false rate | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | NOT RUN |

## What This File Proves

* The deterministic collector executes the existing Issue #52 offline replay kernel and writes content-addressed masked samples.
* Dropout masking is observation-only; labels and replay truth are not changed.
* The implementation has contract and regression coverage for deterministic masking, burst/cap behavior, validation-only calibration, external-oracle abstention, and sample serialization.

## What This File Does Not Prove

* It does not prove the 33-hour corpus was collected.
* It does not prove a trained artifact passed any forecast, calibration, abstention, safety, replay, or latency gate.
* It does not permit deployment or soften the frozen Issue #52 abstention behavior.

## Reproduction Entry Points

```python
from aeolus.habitat_v2.forecast_issue53_dropout import (
    DropoutConfig,
    DropoutAwareLinearForecaster,
    abstention_pr,
    apply_dropout_to_history,
    evaluate_per_k,
    interval_coverage_at_k,
)

config = DropoutConfig(p_uniform=0.05, mode="independent", seed=530053)
# Fit on TRAIN samples, calibrate on VALIDATION samples, then run one sealed
# FINAL evaluation. Supply truth-backed oracle_errors to abstention_pr.
```
