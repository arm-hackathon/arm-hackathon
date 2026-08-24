# Issue #54 Distillation Capability and Limitation Card

Date: 2026-08-24
Lane: `research/issue-54-model-distillation`
Preregistration: `contracts/habitat_v2_forecast_issue_54_preregistration_v1.json`
Preregistration SHA-256, LF-normalized: `E16BEFB588A43F131128056932BBFE5CAA707C87309A828A33C91E1C412D5246`
Status: **DEVELOPMENT EVIDENCE ONLY - research study, not qualified, not deployable**

HMC remains the sole proposal, arbitration, preflight, capability, plant-step,
and replay authority. No student model has any actuator authority. This study
measures how faithfully small "student" models copy the frozen teachers'
predictions; it does not qualify any model for plant control.

## What Exists

The study trains progressively smaller student models that copy the frozen MLP
and ridge teachers' 8-step / 408-output forecasts, then evaluates three
behavioural properties on held-out FINAL families:

* prediction accuracy: family-mean NMAE ratio versus the teacher (with 95% CI);
* action-ranking agreement: top-1 agreement and Kendall tau-b between the
  student and teacher full rankings of the 4 catalogue actions;
* safety-margin closeness: mean absolute difference of normalized safety
  exposure between student and teacher per candidate per decision.

Implementation artifacts:

* `src/aeolus/habitat_v2/forecast_issue54_distillation.py` - sample contract,
  pure-NumPy GELU+MSE+Adam student trainer, ridge student fitter, corpus
  manifest/validation, bootstrap NMAE-ratio statistics, and evaluation.
* `scripts/run_issue54_distillation_study.py` - deterministic collector +
  qualifier entry point (pilot and full modes).
* `tests/habitat_v2/test_forecast_issue54_distillation.py` - 29 tests.
* Frozen teachers: `artifacts/demo-only/habitat-v2-forecast/action-aware-mlp-v1.npz`
  (SHA-256 `a80628fb298ae2f68fb600ecc70922dfddb39e2560207bbd13463e2d4596ecdd`)
  and `action-aware-ridge.npz` (SHA-256
  `0de4b5cdb6ec2b47be260a06f924d8eb00f1def16d5ae668b3ab5191251f29df`).

## Corpus

The preregistration names three corpus options. The frozen forecast contract
bundle (`_validate_bundle` in `projection.py`) pins the development scenario
SHA, so the `committed_corpus` option (a 288-example ridge corpus in the
horizon-32 / 12-candidate Issue #52 format) does not convert to the Issue #54
horizon-8 / 4-action contract without re-projection, and `synthetic_varied_seed`
would require bypassing the frozen bundle check. This run therefore uses
`fresh_pipeline`: the frozen development scenario extended to 48 steps (timeline
repeated by `extend_scenario_for_issue52`), with 32 whole-family variations of
the sensor-model random seed, 3 anchor steps (16, 24, 32), and all 4 catalogue
actions per anchor = 384 distillation samples per teacher.

* Family split (SHA-256 `issue54-split-v1` ordering, 60/20/20): 19 TRAIN, 7
  VALIDATION, 6 FINAL whole families; all minimums from the preregistration are
  met (`TRAIN >= 10`, `VALIDATION >= 5`, `FINAL >= 5`).
* MLP corpus manifest SHA-256: `405cb457c077310bfdf7ff260253130ee63dff39c6071cb286461f52191f231d`
* Ridge corpus manifest SHA-256: `47407af5b0ab826c0dd7324277d3b4e71509deaab0268cd7935ec8ee056cfe4c`
* Samples digest (MLP): `940546929cb82ca689ce08f2141fde82b4299a7d9462dea9d5784ecc343b3edf`
* Hard gates: 0 authority violations, 0 provenance/split violations, 0 replay
  failures; whole-family isolation enforced by `validate_samples`.

## Deliberate Deviation From the Literal Preregistration Text

The preregistration says the student distillation loss is MSE against the
teacher prediction and does not mention target normalization. The raw teacher
predictions span roughly 0.02 to 72,000 (mixed SI units: temperature K, pressure
Pa, CO2 ppm, mole fractions, fractions, airflow m3/s, and resource gauges), so a
raw MSE loss is dominated by the largest-magnitude channels. The frozen MLP
teacher artifact itself was trained with per-target standardization
(`target_mean`/`target_std` in the NPZ), and the ridge teacher's dual-form fit
centers targets. The student MLP trainer therefore standardizes teacher targets
to zero mean / unit variance per output dimension before MSE training and
de-standardizes at inference. This is recorded here as a deliberate, disclosed
implementation choice within the "MSE distillation loss" spec; it is the
difference between students that learn (ratios 0.75-4.5) and students that emit
garbage (ratios in the thousands).

## How Small Is Safe - Result

The measured curve (see `issue-54-measurements.md`) is monotonic in both
directions for both teachers: fidelity degrades as the student shrinks.

* The **linear (ridge) student** is the best student for both teachers and is
  the only student that passes the preregistered primary gate on both teachers
  (`nmae_ratio` point <= 1.5 and upper 95% CI < 2.0): MLP 0.753 (CI 0.699-0.823),
  ridge 0.962 (CI 0.924-0.991).
* The largest MLP student (`sanity-2.1m`, which matches the MLP teacher's
  architecture) is borderline: ridge-teacher ratio 1.409 (CI 1.370-1.443) passes
  the point threshold, MLP-teacher ratio 1.549 (CI 1.420-1.718) does not.
* All smaller MLP students fail the primary gate: `medium-500k` 1.64-3.44,
  `small-100k` 2.39-3.68, `tiny-25k` 3.32-4.50.
* Safety-margin closeness is excellent for every student (mean absolute
  safety-exposure difference 0.00004-0.00044, gate maximum 0.5), because the
  simulated trajectories stay well inside bounds for both teacher and student.
* Ranking agreement degrades with size: Kendall tau from 0.78-0.90 (linear) to
  ~0.07-0.45 (sanity) and effectively random for `tiny-25k`.

## Capability Boundaries

1. The study measures distillation fidelity on the frozen development fixture
   extended to 48 steps. It does not qualify any student for deployment.
2. Students are forecast-only regression models; none may propose, arbitrate,
   preflight, or execute plant actions.
3. The corpus is generated from the frozen dev scenario with varied sensor
   seeds; it is not a production, validation, or final-suite dataset.
4. Only the `fresh_pipeline` corpus option was measured; the other two
   preregistered options were assessed and found incompatible with the frozen
   Issue #54 contract as checked in (see Corpus section).

## Limitations and Stop Conditions

1. The `sanity-2.1m` MLP-teacher student (ratio 1.549, CI upper 1.718) is a
   negative result: even a student with the teacher's exact architecture does
   not meet the point threshold on the MLP teacher with this corpus size and
   trainer.
2. `tiny-25k` students (ratio 3.3-4.5, tau ~ -0.03 to 0.07) are not safe for
   action ranking; their rankings are effectively random.
3. Results are limited to the frozen 4-action catalogue, 8-step horizon, and
   60 s cadence of the Issue #54 forecast contract.
4. Bootstrap statistics resample semantic families; with 6 FINAL families the
   CIs are informative but not a substitute for a larger FINAL roster.

## Rollback

The study adds no HMC, physics, actuator, safety, or contract changes. Removing
the study is as simple as deleting the Issue #54 module, script, tests, and
evidence docs; no rollback of any plant-control path is needed.

## Evidence Bundle

* `contracts/habitat_v2_forecast_issue_54_preregistration_v1.json`
* `docs/plans/2026-08-24-issue-54-model-distillation-design.md`
* `docs/plans/2026-08-24-issue-54-model-distillation-plan.md`
* `src/aeolus/habitat_v2/forecast_issue54_distillation.py`
* `scripts/run_issue54_distillation_study.py`
* `tests/habitat_v2/test_forecast_issue54_distillation.py`
* `docs/evidence/issue-54-measurements.md`

**Bottom line:** distillation works only for the linear student at this corpus
size; every MLP student, including one matching the teacher's architecture,
degrades forecast accuracy by at least 40% on at least one teacher. The study
answers "how small is safe" honestly: below ~1.3M parameters with this trainer
and corpus, none of the tested students meet the preregistered accuracy gate.
