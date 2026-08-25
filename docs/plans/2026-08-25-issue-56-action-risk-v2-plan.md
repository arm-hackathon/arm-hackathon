# Issue #56 V2 Implementation Plan

## Branch and control

- Branch: `research/action-risk-v2`
- Parent commit: `9145bb870c4f435923a5017dd2a0f17714c69a7b`
- Delivery: local commit only unless explicitly requested otherwise
- V1 remains immutable and is not reinterpreted

## Work sequence

1. Freeze the V2 contract and design before model code or EVALUATION access.
2. Add runtime-aligned `effect_4` and declared `persistent_32` label tracks.
3. Add observable-only feature extensions and content-addressed samples.
4. Implement sigmoid event, positive-only severity, and positive-only maximum
   heads with validation-only calibration.
5. Implement direct feature inference, strict artifact loaders, model identity,
   source commit binding, and dirty-worktree refusal.
6. Add risk-only and risk-filtered point ranking. Reuse the frozen Issue #55
   point model only as a comparator and utility scorer.
7. Add validation non-vacuity checks before EVALUATION.
8. Add paired rules, point, risk-only, and risk-filtered point episodes with
   complete HMC trace and replay evidence.
9. Add deterministic paired whole-family bootstrap intervals and calibration
   diagnostics.
10. Add focused tests for labels, probability semantics, calibration, feature
    provenance, loaders, non-vacuity, ranking, HMC binding, and replay.
11. Run a smoke corpus in a fresh ignored directory. Smoke output is not evidence.
12. Freeze TRAIN/VALIDATION artifacts, pass non-vacuity, and run the declared
    32-family EVALUATION exactly once.
13. Publish measurements and limitation card, including failed gates and any
    residual abstention or utility limitations.
14. Run the full repository verification suite and create a local commit only.

## Required stop conditions

Stop before EVALUATION if a hidden feature is detected, a label track cannot be
replayed, calibration uses EVALUATION, validation coverage is vacuous, source
identity is missing, or an HMC/replay boundary fails. Never repair a failed gate
with an EVALUATION-derived threshold.
