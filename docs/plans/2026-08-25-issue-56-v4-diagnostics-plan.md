# Issue #56 V4 Diagnostic and Model-Improvement Plan

## Status

This is a pre-model diagnostic protocol draft. It is not authorization to train,
export, quantize, integrate, or select a learned model. The machine-readable
contract is `contracts/habitat_v2_forecast_issue_56_v4_diagnostics_preregistration_v1.json`.
The contract remains `PRE_MODEL_PROTOCOL_DRAFT_PENDING_AUTHORIZATION` until the
required human scope review is recorded.

The implementation is on `research/action-risk-v4`. V3 and its historical
evidence remain immutable.

## Why this slice comes first

The clean V3 evidence showed a safety-preserving but mostly abstaining
`risk_filtered_point_v3` arm: 2 proposals and 76 abstentions across 78
evaluation decisions. The `risk_only_v3` arm proposed 54 times, but HMC changed
52 requested commands. Before changing a model, the study must distinguish:

- all candidate screening behavior;
- the action actually selected by the adviser;
- the command requested from HMC;
- the command emitted by HMC; and
- the command confirmed by the plant receipt.

The V4 diagnostic module provides those immutable record types and metrics
without changing HMC, V3 inference, V3 thresholds, or the telemetry projection.

## Implemented diagnostic boundary

`src/aeolus/habitat_v2/forecast_issue56_action_risk_v4_diagnostics.py` provides:

- `V4CandidateObservation` for candidate-level rejection/label diagnostics;
- `V4ExecutedObservation` for selected, requested, final, executed, and HMC
  disposition identities;
- separate candidate and executed-action metric functions;
- equal-weight condition-group aggregation;
- deterministic SHA-256 condition-group bootstrap;
- order-independent observation manifests; and
- a complete provenance identity-set validator.

Every row is content-addressed. Final and executed command identities must
match. An abstention cannot carry a selected action or requested command. A
proposal cannot be marked as an abstention. Candidate and executed metrics use
different input types so the V3 reporting ambiguity cannot be repeated by
accident.

## Frozen population and data boundary

- The statistical unit is a two-family condition group.
- The 32-family development roster remains paired by sensor variant.
- Every evaluation stratum must be declared and supported before a clean run.
- Future plant state is label-only.
- Hidden fault truth, future measurements, schedules, seeds, internal noise or
  bias state, reserve audit state, and HMC arbitration outcomes are not runtime
  model inputs.
- Protected final-suite/validation data is out of scope.

Policy-conditioned development histories may be collected only through approved
development scenarios and deterministic replay. Counterfactual label branches
must be independently strict-replayed, not merely shadow-stepped.

## Required gates before model work

1. Ben's participation and exact scope must be recorded.
2. The project owner must confirm that this is an authorized separate research
   lane relative to the repository's learned-model stop boundary.
3. A new learned-model protocol must freeze features, targets, architecture
   candidates, seeds, calibration, thresholds, evaluation strata, non-vacuity
   gates, latency, HMC compatibility metrics, and artifact rules before training.
4. V3 evidence must remain historical; no V3 threshold may be tuned from its
   evaluation behavior.

## Next implementation phases after authorization

### Phase 1: diagnostic integration

Adapt the V3 runner or create a separate V4 runner to emit the new observation
records. Add independent serialized-artifact verification for provenance,
replay, authority, finiteness, and missing/failed episodes.

### Phase 2: development corpus

Generate policy-conditioned development histories from the approved development
roster. Validate family/condition disjointness, positive label support, strict
counterfactual replay, feature leakage, and write-once content-addressed output.

### Phase 3: preregistered model comparison

Only after a new protocol is approved, compare the frozen V3 baseline against
predeclared model candidates. Candidate improvements may include coherent
multi-horizon hazard structure, past-only temporal features, and grouped
calibration, but no architecture is selected by this draft.

### Phase 4: clean evaluation

Run from a clean source tree and a new ignored output directory. Report safety,
useful-action coverage, abstention, selected/executed false-safe behavior,
HMC mismatch/disposition, calibration, latency, provenance, and replay gates.
Publish a failed utility or calibration result rather than repairing it with
evaluation-derived thresholds.

## What this plan does not prove

This diagnostic slice does not qualify a model, establish deployment readiness,
validate hardware, change HMC authority, or demonstrate real-world safety. It
only makes the next simulator development study auditable.
