# Issue #53 Honest Measurements - 1 vs 3 Sensors Missing

Status: **QUALIFIED** for the preregistered Issue #53 forecast lane. Deployment
of the learned lane remains blocked: HMC is still the sole proposal,
arbitration, preflight, capability, plant-step, and replay authority.

The Issue #53 preregistration remains authoritative:
`contracts/habitat_v2_forecast_issue_53_preregistration_v1.json`.
The qualified lane remains forecast-only; HMC remains the sole proposal,
arbitration, preflight, capability, plant-step, and replay authority.

## Run Identity

* Dropout config: `DropoutConfig(p_uniform=0.05, mode=independent, resource_gauge_dropout=false, max_missing_per_row=6, seed=530053)`
* Dropout config SHA-256: `c288962aa6d5d53018e866ba19b29d951b21bbef8a7c4463b181692472535f7c` (canonical config digest)
* Issue #53 preregistration SHA-256, LF-normalized: `A96245F6E717BC83B44438F9D02DBAAA42FA5DED14D3A160FD47A0F4D393D76A`
* Frozen Issue #52 parent preregistration SHA-256, LF-normalized: `DE4744E127D2946A43D623EC90D3289B0A3735C99E62C8CECCD87768E0702A3B`
* Scenario SHA-256: `cdd737b02566c077ddc12116b6f93b0af5fb78a531a29a9762de96720b78025e`
* HMC contract SHA-256: `9f4d269ad8d073d6370f5239d8a78f2541db3001097a460447a8feb84fee2414`
* Forecast target manifest SHA-256: `c3104cd72cfc3fb4d1043f4a5a483d696fcd4fb4f7d3045352a4ed0390d66cd6`
* Full corpus dataset SHA-256: `07f7e7c5cdc12043e84eac40c4e984afc88bd6e7498ce828b6e04a56c8d2ff29`
* Full corpus samples SHA-256: `cd50e8b76b5f3a8b127126b0d471ce82e8e40e5e9f48985cbf1ee58d27ebf9d7`
* Full corpus: **RUN**; 384 whole families, 18,432 serialized samples, and 147,456 candidate transitions
* Family split: 269 TRAIN, 58 VALIDATION, and 57 FINAL families; all whole-family isolated
* Model artifact: `issue53-dropout-linear-25d407138a448069-cal-3905844701ead27a`
* Model source commit: `b812129cb6ffb0d3566b6542036ce5b29fcf6161`
* Sealed FINAL evaluation: **QUALIFIED**; exactly one sealed invocation for this model version
* Sealed bundle: `C:\Users\Yarik\AppData\Local\Temp\opencode\issue53-full-committed\`
* Artifact SHA-256: `9ffd50bf2127c5d8f47c687e435f89677f10dc640e58c0374b9730f048a0b186`
* Report SHA-256: `67bcfefeec58e03be68f5af87331d35b692799a28931f33361d591d3cba18c89`
* Sealed lock SHA-256: `a373ffa8d3dcba4de8618bbad0da4ec3de97f47b8a39b7e982532734da117cdc`

## Collection Record

The collector was executed locally for one family to verify the replay and
serialization path only:

* Command: `python scripts/collect_issue53_dropout_dataset.py --families 1 --output <temporary-directory>`
* Replay output: 12 candidates x 4 fixed latest-row missingness views = 48 serialized samples, 384 candidate transitions
* Output files: `dropout_config.json`, `dataset_manifest.json`, `samples.jsonl`
* Result: deterministic replay-backed samples were written before the full run;
  this smoke result was not used as a qualification measurement.

The frozen full corpus was then collected and validated:

* 384 families, with 48 serialized samples and 384 candidate transitions per family
* 12,912 TRAIN samples, 2,784 VALIDATION samples, and 2,736 FINAL samples
* 103,296 TRAIN candidate transitions, 22,272 VALIDATION candidate transitions,
  and 21,888 FINAL candidate transitions
* Dataset identity and sample identity are bound into the sealed artifact and lock

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
   families, including the required `k=0,1,3,6` sweep, truth-backed
   `oracle_errors` for abstention PR, and safety non-regression.
7. Publication: this table filled with measured values and the dropout card
   updated with retained and lost capabilities. Deployment remains blocked
   because HMC retains all plant-step and capability authority.

All seven items are complete for the sealed model version `b812129`; the
preregistered gates passed and the status is `QUALIFIED`.

## Forecasting Degradation - NMAE Horizons 9-32

Gate: `k=1` vs own `k=0` <=1.15 point / <=1.25 upper; `k=3` <=1.40 / <=1.60.
All values below are from the single sealed FINAL invocation. The required
honest-measurement sweep is `k=0,1,3,6`; `k=2,4,5` were not preregistered
reporting cells.

| k | NMAE mean | vs own k=0 point | 95% upper | n families | Pass |
|---|---:|---:|---:|---:|---|
| 0 | 0.0044940554 | 1.0000000000 | 1.0000000000 | 57 | BASELINE |
| 1 | 0.0046535786 | 1.0354964996 | 1.0799847101 | 57 | PASS |
| 3 | 0.0047612128 | 1.0594468526 | 1.1140854143 | 57 | PASS |
| 6 | 0.0049288130 | 1.0967406055 | 1.1763024238 | 57 | OBSERVED |

The primary learned-vs-best-frozen-baseline ratio was `0.0676093777` with
95% CI `[0.0673356776, 0.0679119442]`. The short-horizon `h1-h8` ratio was
`0.2922806056` with 95% CI `[0.2905266963, 0.2940693242]`; both passed.

## Interval Coverage

Gate: empirical 90% coverage >=85% at `k=1` and >=80% at `k=3`.

| k | Coverage | Mean interval width (normalized) | n cells | Pass |
|---|---:|---:|---:|---|
| 0 | 0.9863750812 | 0.0843281849 | 590,976 | OBSERVED |
| 1 | 0.9872651343 | 0.0882365557 | 590,976 | PASS |
| 3 | 0.9873158978 | 0.0893255063 | 590,976 | PASS |
| 6 | 0.9871399177 | 0.0920815946 | 590,976 | OBSERVED |

`DropoutAwareLinearForecaster.calibrate()` accepted VALIDATION samples only and
the sealed artifact retained the calibrated per-k interval scales. Coverage
passed the preregistered `k=1` and `k=3` gates.

## Abstention Quality

Gate at `k=3`: recall >=0.80 and precision >=0.60 against externally supplied,
truth-backed oracle errors. The oracle must not be derived from the model's own
predictions. `abstention_pr()` therefore requires an aligned `oracle_errors`
sequence.

| k | Abstention rate | Precision | Recall | Threshold | Pass |
|---|---:|---:|---:|---:|---|
| 0 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0672489559 | OBSERVED |
| 1 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0672489559 | OBSERVED |
| 3 | 0.3859649123 | 0.8636363636 | 0.9500000000 | 0.0672489559 | PASS |
| 6 | 0.3508771930 | 1.0000000000 | 0.6666666667 | 0.0672489559 | OBSERVED |

The threshold column is the validation `k=0` oracle NMAE P90. The selected
sealed artifact also uses calibrated `k=3` risk and width limits; no external
oracle was derived from the model's own FINAL predictions.

## Safety Non-Regression

The hard safety gates were evaluated at `k=0`, `k=1`, and `k=3` using the same
eligibility and family aggregation as Issue #52.

| Metric | k=1 measured | k=3 measured | Gate |
|---|---:|---:|---|
| Safety-bound exposure difference | -0.0333854554 (upper -0.0329748103) | -0.0328044252 (upper -0.0323830697) | PASS; point and upper <= 0 |
| Dangerous-crossing recall difference | +0.0025553663 | +0.0051107325 | PASS; >= -0.02 |
| Dangerous-crossing false-rate difference | 0.0000000000 | 0.0000000000 | PASS; <= +0.01 |

At `k=0`, exposure difference was `-0.0354073050` with upper 95% bound
`-0.0350336670`; dangerous-crossing recall and false-rate differences were
both `0.0000000000`.

## Systems And Provenance

* Authority violations: `0`; provenance/split violations: `0`; replay failures: `0`
* Non-finite committed states: `0`; invalid outputs: `0`; timeout rate: `0.0`
* All-12 inference p99: `71.0433 ms` against the `250 ms` deadline
* `k=0` abstention-rate delta: `0.0000000000` against the `0.02` maximum
* Artifact, report, lock, dataset, and source commit bindings matched; sealed
  source worktree was clean
* Every gate in the sealed report was `true`, including dataset completeness,
  replay, provenance, authority, safety, forecast, abstention, coverage, and latency

## What This File Proves

* The deterministic collector executes the existing Issue #52 offline replay kernel and writes content-addressed masked samples.
* Dropout masking is observation-only; labels and replay truth are not changed.
* The implementation has contract and regression coverage for deterministic masking, burst/cap behavior, validation-only calibration, external-oracle abstention, and sample serialization.
* The frozen 384-family corpus, TRAIN/VALIDATION/FINAL isolation, sealed artifact,
  and one sealed FINAL invocation passed every preregistered gate for the model
  version above.

## What This File Does Not Prove

* It does not qualify correlated burst mode, resource-gauge dropout, adversarial
  safety-channel dropout, higher dropout rates, or other out-of-distribution patterns.
* It does not grant the learned lane plant-step, capability, safety, or HMC authority.
* It does not permit deployment without the separate HMC and safety-core reviews.

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
# The sealed artifact was fit on TRAIN, calibrated on VALIDATION, and evaluated
# once on FINAL. Supply truth-backed oracle_errors to abstention_pr.
```
