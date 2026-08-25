# Issue #56 Action-Risk Adviser Capability And Limitation Card

Date: 2026-08-25
Lane: `research/action-risk-adviser`
Status: **DEVELOPMENT EVIDENCE ONLY - NOT QUALIFIED OR DEPLOYABLE**

## Capability

- Builds a 610-value action-conditioned feature vector from a complete verified
  16-step operational history and one frozen four-action command.
- Produces offline true-plant labels for 32-step safety exposure, maximum
  normalized crossing, and any-crossing event.
- Fits a deterministic normalized multi-output ridge model on TRAIN families and
  calibrates upper risk estimates on VALIDATION families.
- Uses deterministic hard risk gates and ranking code that can abstain rather
  than emit a proposal.
- Binds any emitted proposal to the current HMC snapshot and standard proposal
  contract; HMC remains the sole final-command and plant-step authority.
- Passed the full development path with zero authority, replay, provenance,
  proposal-admission, and non-finite violations.

## Retained Limitations

1. The EVALUATION model abstained on all 78 decisions. No useful action-selection
   gain was demonstrated.
2. The six-family EVALUATION result is development evidence, not statistical
   qualification, certification, hardware validation, or deployment proof.
3. The corpus reuses the Issue #55 v2 development family matrix. It does not
   qualify unseen faults, correlated sensor failures, higher disturbance rates,
   hardware behavior, or real-world conditions.
4. The crossing score is a calibrated regression score, not a formal probability
   guarantee. The Brier score and upper-exposure coverage are descriptive.
5. Infeasible counterfactual labels are excluded from fitting; this run does not
   establish behavior for a future catalogue with materially different dynamic
   feasibility.
6. The model does not own HMC policy, emergency action templates, capability
   issuance, safety checks, plant stepping, or replay.

## Safety Boundary

`actuator_authority=false` is enforced in the model object. HMC remains the sole
proposal arbitration, preflight, capability, final-command, plant-step, and
replay authority. The oracle, hidden plant state, and future labels never enter
the runtime feature vector.

## Decision

The lane is retained as a fail-closed research implementation and honest
negative result. It must not be enabled as a deployed or qualified controller.
Any future attempt to reduce abstention requires a new preregistered calibration
and safety review rather than post-result threshold tuning.
