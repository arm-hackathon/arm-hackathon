# Issue #56 V3 Support Revision Plan

## Document control

- Branch: `research/action-risk-v2`
- Parent implementation: `e586733`
- Status: `RESEARCH_STUDY_DEV_EVIDENCE_ONLY`
- Historical protocol: `contracts/habitat_v2_forecast_issue_56_v3_preregistration_v1.json`
- Support protocol: `contracts/habitat_v2_forecast_issue_56_v3_preregistration_v2.json`

## Reason for the revision

The first V3 development split preserved sensor-pair grouping but left the
validation and evaluation label support vacuous for the declared dangerous-event
diagnostics. That result is a split-support failure, not evidence that the
runtime policy is safe or useful. V1 and the first V3 preregistration remain
historical and are not pooled with this revision.

The support revision changes the development split before the comparative rerun.
It does not inspect or use protected final-suite data, and it does not change a
threshold after observing comparative behavior.

## Fixed support design

- The roster is the canonical 32-family Issue #55 development matrix.
- Consecutive sensor variants form one condition group and cannot cross a split.
- The 16 condition groups are assigned by the fixed roster labels in the support
  contract: 10 TRAIN groups, 3 VALIDATION groups, and 3 EVALUATION groups.
- The resulting family counts are 20 TRAIN, 6 VALIDATION, and 6 EVALUATION.
- Smoke runs select complete sensor pairs from each declared split and are marked
  `SMOKE_PATH_ONLY`; they are not comparative evidence.

## Fail-closed behavior

- Calibration refuses a horizon whose validation labels do not contain both event
  classes.
- Evaluation diagnostics refuse a horizon with no positive event support.
- Validation non-vacuity is checked before evaluation episodes are executed.
- Comparative evaluation refuses a dirty source worktree.
- HMC remains the sole final-command, plant-step, and replay authority.

## Execution sequence

1. Commit the support contract, implementation, runner, tests, and this plan.
2. Run the exact 32-family command once from the clean committed tree into a new
   ignored output directory.
3. Record source, contract, manifest, model, calibration, trace, episode, and
   result hashes in a V3 evidence record.
4. Run the complete locked repository verification suite against the final
   review tree.
5. Publish only simulator development evidence. Do not claim qualification,
   deployment, hardware validation, or real-world safety.

## Stop conditions

Stop before or during comparative evaluation if source identity is dirty or
missing, a split crosses sensor pairs, a horizon lacks required support, a
feature uses hidden or future truth, a proposal bypasses HMC, a trace fails
strict replay, a provenance digest fails, or a hard gate is false. A failed
utility or calibration result must be published honestly and cannot be repaired
with evaluation-derived thresholds.
