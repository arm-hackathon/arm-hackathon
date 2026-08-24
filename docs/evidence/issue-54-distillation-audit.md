# Issue #54 Distillation Audit

Date: 2026-08-24
Status: **CORRECTIONS IMPLEMENTED; FRESH DECLARED RERUN REQUIRED**
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

7. **The ranking metric remains a protocol boundary.** Issue #54 samples contain
   51 point targets and no forecast intervals. The frozen Issue #52
   `score_trajectory` ranker expects its own target manifest and interval-bearing
   `ForecastTrajectory` values, so the original simplified nominal/bound score
   was not equivalent to the preregistered metric. The corrected implementation
   labels its helper as
   `issue54-simplified-nominal-point-bound-v1` and records
   `NON_PREREGISTERED_METRIC_REQUIRES_ADDENDUM`; it does not silently claim
   `score_trajectory` compliance.

8. **The trainer has a disclosed but non-literal loss normalization.** The MLP
   trainer standardizes each teacher output dimension before MSE optimization.
   This is documented in the original capability card, but the frozen
   preregistration states MSE against raw teacher predictions. A corrected run
   must declare whether this normalization is an approved interpretation or a
   protocol amendment.

9. **The target and action fixtures differ from the frozen Issue #52 lane.** The
   collector uses the Track A `float32[8,51]` layout (including pressure, oxygen,
   and airflow) and the four `normal-*` forecast fixture actions. The frozen
   Issue #54 text describes a `3 * zone_count + 3` target width and the Issue #52
   catalogue implementation contains 12 candidates. This is another required
   addendum choice, not a detail that can be silently inferred from the old
   tables.

## Required Pre-Run Declaration

Before a corrected full run is treated as Issue #54 evidence, declare an
addendum that fixes:

- whether `(family_id, anchor_step)` is the decision identity and updates the
  decision denominator to 18 for the current 32-family / 3-anchor design;
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

Until that declaration and a fresh run exist, the numerical tables in
`issue-54-measurements.md` and the capability claims in
`issue-54-distillation-card.md` are historical provenance, not current results.

## Verification Performed

- `35 passed` in `tests/habitat_v2/test_forecast_issue54_distillation.py`.
- `1112 passed` in the full locked test suite `tests/`.
- Ruff, Python compilation, `uv lock --check`, and `git diff --check` pass.
- The final corrected pilot smoke run completed in
  `out/issue54-audit-pilot-5`: 48 samples per teacher, 12 decision IDs, strict
  trace replay, manifest validation, TRAIN-only scale derivation, all three MLP
  seeds, and finite result JSON with hashes for every saved student MLP.
  Its ignored output is not evidence: it has one FINAL family and uses the
  explicitly non-preregistered ranking metric.
- No corrected full corpus or publishable full-run numerical evidence was
  generated by this audit.
