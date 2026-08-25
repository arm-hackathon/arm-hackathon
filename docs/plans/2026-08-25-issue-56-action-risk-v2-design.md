# Issue #56 V2: Calibrated Risk-Filtered Point Ranking

Status: declared before implementation; development evidence only
Parent result: `docs/evidence/issue-56-action-risk-measurements.md`
Preregistration: `contracts/habitat_v2_forecast_issue_56_v2_preregistration_v1.json`

## Motivation

V1 preserved the HMC boundary and replayed cleanly, but its unconstrained ridge
event score was called a probability and its validation residual calibration
made every EVALUATION candidate fail the exposure gate. The result was 78/78
abstentions. V2 keeps that negative result immutable and corrects the model and
evaluation design rather than lowering thresholds after seeing EVALUATION.

## Model

V2 separates event occurrence from event severity:

- An event head predicts a sigmoid probability of any safety-bound crossing.
- A conditional severity head predicts `log1p(exposure)` only for positive events.
- A conditional maximum head predicts `log1p(maximum crossing)` only for positive
  events.
- Validation-only monotonic event calibration and positive-severity residual
  quantiles produce conservative upper estimates.

The feature vector retains the V1 verified numeric history summary and candidate
command, and adds only public observable projections: current safety margins,
operating-mode and health one-hots, alarm counts, candidate-minus-current command
delta, and normalized decision position. No hidden plant truth enters runtime.

## Runtime-aligned labels

The primary `effect_4` counterfactual applies an action for the four transitions
until the next decision cadence, then holds the verified pre-decision command for
the remaining 28 transitions. This isolates the action's one-cadence effect
without pretending to know which future policy proposal will be selected. The
`persistent_32` label remains as a declared stress track for a constant-command
counterfactual.

## Runtime policy

V2 does not use risk as its comfort objective. It first filters candidates using
upper event probability, upper expected exposure, and upper maximum crossing.
Among candidates that pass, the existing Issue #55 point model score chooses the
action. If no candidate passes, it abstains. HMC still validates, arbitrates,
preflights, issues capability, steps, and replays every final command.

## Non-vacuity

Validation must retain at least one candidate on at least 10% of declared
decision rows. If this fails, the run stops before EVALUATION and is reported as a
calibration failure. Abstaining forever cannot qualify as a useful result.

## Scope boundaries

This is not an ensemble, scenario-hardening, long-horizon catalogue, missing
sensor, or distillation lane. Existing V1 files and evidence remain unchanged.
The V2 runner compares rules, the existing point adviser, risk-only, and the
risk-filtered point adviser using paired family-level metrics.
