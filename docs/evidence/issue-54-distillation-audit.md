# Issue #54 Distillation Audit

Date: 2026-08-24
Status: **CORRECTED FULL RUN VALIDATED; DEVELOPMENT EVIDENCE ONLY**
Preregistration: `contracts/habitat_v2_forecast_issue_54_preregistration_v1.json`
Preregistration SHA-256, LF-normalized: `E16BEFB588A43F131128056932BBFE5CAA707C87309A828A33C91E1C412D5246`

## Scope

This audit covers the original Issue #54 implementation introduced in commit
`e2a1d77`. It does not alter the frozen preregistration, the frozen teacher
artifacts, the HMC binding, or any plant-control path. The corrected code remains
development evidence only and does not establish hardware or deployment safety.

## Findings

1. **Decision identity was under-specified.** The original collector generated
   three anchors per family but stored only `family_id`. The evaluator therefore
   grouped all candidates from all anchors into one family group. The corrected
   schema binds each row to `decision_id = family_id|anchor=NNNN`, and validates
   every decision has the exact candidate roster once.

2. **The old denominator was wrong.** The original report described six FINAL
   decisions, although 6 FINAL families x 3 anchors yields 18 decisions. The
   corrected evaluator ranks by decision while retaining family as the NMAE
   bootstrap unit.

3. **NMAE normalization was not TRAIN-only.** The original runner used fixed
   descriptor arrays. The corrected runner derives a finite, positive
   per-horizon/target `P95(TRAIN truth) - P05(TRAIN truth)` scale, records its
   shape and digest, and uses it only for NMAE.

4. **Stochastic reporting was incomplete.** The original runner trained MLP
   students only with the default seed. The corrected runner reports separate
   rows for `540054`, `540055`, and `540056`, including validation-selected epoch
   and validation MSE. The deterministic linear student is reported once with
   seed `null`; no FINAL-based seed selection is performed.

5. **Candidate runs were not paired on the same pre-anchor sensor stream.** The
   original HMC reset nonce included `action_id`, so candidate-specific sensor
   randomness could contaminate causal comparisons. The corrected nonce is keyed
   by family and anchor only. Each collected trace is also parsed and replayed
   before the sample is accepted.

6. **Truth and output completeness was not fail-closed.** The corrected sample
   constructor and corpus validator reject non-finite truth, malformed arrays,
   duplicate decision/candidate rows, missing decisions, missing candidates,
   roster drift, manifest digest drift, and incomplete prediction maps.

7. **The ranking metric required an explicit protocol choice.** Issue #54 samples contain
   51 point targets and no forecast intervals. The frozen Issue #52
   `score_trajectory` ranker expects its own target manifest and interval-bearing
   `ForecastTrajectory` values, so the original simplified nominal/bound score
   was not equivalent to the preregistered metric. The corrected implementation
   labels its helper as
   `issue54-simplified-nominal-point-bound-v1`. The declared addendum approves
   this metric for the Track A lane; it does not silently claim `score_trajectory`
   compliance.

8. **The trainer required an explicit loss-normalization choice.** The MLP
   trainer standardizes each teacher output dimension before MSE optimization.
   This is documented in the original capability card, but the frozen
   preregistration states MSE against raw teacher predictions. The protocol
   addendum approves this normalization for the corrected run.

9. **The target and action fixtures differ from the frozen Issue #52 lane.** The
   collector uses the Track A `float32[8,51]` layout (including pressure, oxygen,
   and airflow) and the four `normal-*` forecast fixture actions. The frozen
   Issue #54 text describes a `3 * zone_count + 3` target width and the Issue #52
   catalogue implementation contains 12 candidates. This is another required
   addendum choice, not a detail that can be silently inferred from the old
   tables.

## Required Pre-Run Declaration

The following choices had to be declared before a corrected full run could be
treated as Issue #54 evidence:

- whether `(family_id, anchor_step)` is the decision identity and updates the
  denominator to 96 total decisions and 18 FINAL decisions for the current
  32-family / 3-anchor design;
- whether the 51-target distillation lane uses a formally approved ranking
  metric, or is re-projected to the frozen 27-target Issue #52 ranker with
  interval-bearing trajectories;
- whether the Track A 51-target / four-action fixture is the declared Issue #54
  evaluation contract, or whether the corpus must be regenerated against the
  frozen 27-target / 12-candidate Issue #52 lane;
- the TRAIN-only P95-P05 scale formula and the treatment of unsupported slots;
- whether per-output teacher-target standardization is part of the declared MSE
  loss;
- the reporting rule for the three MLP seeds and the deterministic linear
  student; and
- whether the `fresh_pipeline` option alone is the declared corpus option.

These choices were declared in the machine-readable addendum below before the
full run. The corrected results are now current development evidence; the old
v1 measurements remain historical provenance only.

## Declared Protocol Addendum

The pre-run choices are now declared in the machine-readable addendum
`contracts/habitat_v2_forecast_issue_54_distillation_addendum_v1.json`, committed
before the corrected full run. In summary:

- `fresh_pipeline` is the sole measured corpus option: 32 whole families, anchor
  steps `(16, 24, 32)`, four Track A `normal-*` actions, 96 total decisions, and
  18 FINAL decisions.
- `(family_id, anchor_step)` is the decision identity; family remains the
  bootstrap unit.
- The Track A `float32[8,51]` point-target lane and its four-action fixture are
  the declared Issue #54 contract for this run.
- `issue54-simplified-nominal-point-bound-v1` is approved for this lane by the
  addendum. It is explicitly not the incompatible frozen Issue #52
  interval-bearing `score_trajectory` contract.
- NMAE uses finite positive TRAIN-only per-horizon/target `P95 - P05` scales;
  unsupported slots reject the run.
- MLP targets use per-output TRAIN standardization for optimization and are
  de-standardized before evaluation. Seeds `540054`, `540055`, and `540056` are
  all reported; the linear student is reported once with seed `null`.
- HMC remains the sole authority and the frozen v1 preregistration remains
  byte-identical.

## Corrected Full Run

The declared full run completed in `out/issue54-full-evidence-1` using commit
`a3802d7` and the frozen teacher/HMC artifacts. It collected 384 samples per
teacher, trained every declared MLP seed, and emitted 24 student MLP artifacts.
The independent post-run check verified all roster counts, all 24 artifact
SHA-256 values, both manifest digests, both sample digests, finite result
metrics, and the addendum identity.

- MLP manifest: `6ba2d3ec6118c7a23f44d1ab70460a54f2f03287b1a121f63be18414f7ab8946`
- Ridge manifest: `5769871fc5ce1682b8ac86c986f265e6c845bce56508db820a9730d892228406`
- MLP samples: `aa944440767465eb7b595c4343105417fd6ba02378204d1b43036e4eff7ae5b9`
- Ridge samples: `558da30f049008ebee9ce025c9edb9a6b6ff6ca16951bff2f1777e8d3c75eeb1`
- Results JSON: `44abf8167d2fe6116a71d835a68714dd5ec074f8cbda5dc49fe506bec2149252`
- TRAIN scale shape/digest: `8x51` / `25c9aa1dc0a441c45ca2d13e81d9ee9d725f1de2e759ecdd854e3f10f19ffc83`

## Verification Performed

- `36 passed` in `tests/habitat_v2/test_forecast_issue54_distillation.py`.
- `1113 passed` in the full locked test suite `tests/`.
- Ruff, Python compilation, `uv lock --check`, and `git diff --check` pass.
- The corrected pilot smoke run completed in
  `out/issue54-audit-pilot-5`: 48 samples per teacher, 12 decision IDs, strict
  trace replay, manifest validation, TRAIN-only scale derivation, all three MLP
  seeds, and finite result JSON with hashes for every saved student MLP.
  Its ignored output is not evidence: it has one FINAL family and uses the
  addendum-approved ranking metric.
- The corrected full evidence run completed and passed the independent artifact
  validation described above. Raw outputs remain ignored by policy; the
  committed measurements and capability card record their identities.
