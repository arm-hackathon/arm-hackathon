# Issue #56: Calibrated Action-Conditioned Safety Risk

Status: implementation approved by the repository owner; development evidence only
Issue: https://github.com/arm-hackathon/arm-hackathon/issues/56
Preregistration: `contracts/habitat_v2_forecast_issue_56_preregistration_v1.json`

## Problem

The Issue #55 v2 race exposed a specific limitation in the existing adviser. The
point forecaster improved comfort, but its eight-step point-ranking gate admitted
rare actions with very large later safety exposure. The scorer has no learned
estimate of crossing probability, no calibrated upper safety estimate, and no
action-conditioned risk target. A lower count of violating steps did not imply
lower severity.

This lane adds a forecast-only safety-risk estimator. It does not replace the
existing forecaster, alter the four-action catalogue, or change HMC.

## Design

At each eligible decision, the offline collector uses the exact 16-step verified
operational history and one frozen catalogue command. The feature vector contains
the last projected numeric row, the window mean, the last-minus-first delta, the
27-field proposed command, and a bias term. It contains no hidden simulator truth,
future measurements, fault labels, arbitration output, or reserve audit state.

The label is produced only offline by cloning the deterministic plant checkpoint
and repeating the candidate command for 32 transitions. The evaluator records:

- whether any target crossed a frozen bound;
- total normalized safety exposure; and
- maximum normalized crossing severity.

The risk model is a small normalized multi-output ridge regressor. Exposure and
maximum severity are fit in `log1p` space; the event output is clipped to a
probability score. Normalizers and coefficients are fit on TRAIN families only.
Validation families provide fixed absolute-residual P90 calibration offsets. The
calibrated upper exposure and crossing probability are used by deterministic
ranking code.

## Runtime boundary

The model receives only a `ForecastHistory` and a frozen catalogue action. It has
no plant object, HMC handle, capability issuer, emergency template, or `step`
reference. It can return one action ID or abstain. A standard advisory proposal is
bound to the current verified snapshot; HMC reparses, arbitrates, preflights, and
executes the final command.

The hard risk gates are intentionally conservative and frozen before comparative
evaluation: calibrated upper exposure must be at most `0.5`, and crossing
probability must be below `0.5`. If every action fails, the adviser abstains.

## Relationship to existing lanes

- Issue #51 changes scenario difficulty; this lane reuses the existing Issue #55
  development roster.
- PR #50/#59 study ensembles and disagreement; this lane has one deterministic
  ridge risk model and no ensemble.
- Issue #52 owns the 32-step forecast and 12-schedule planning contract; this lane
  uses the existing four-action catalogue only and predicts risk labels, not full
  trajectories.
- Issue #53 owns missing-sensor handling; this lane requires the existing complete
  verified history and abstains on unavailable inputs.
- Issue #54 owns teacher-copying and compression; this lane trains against true
  offline risk labels rather than teacher predictions.

## Non-claims

The result is deterministic simulator development evidence. Passing tests or a
safety comparison does not qualify, certify, or deploy a controller and does not
establish hardware or real-world safety.
