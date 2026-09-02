# Issue #56 V4 Model V5 Development Study

Date: 2026-08-31
Branch: `research/action-risk-v4-corpus`
Source commit: `2046b941213c04111499db92b7a493b9613924c0`
Status: **DEVELOPMENT EVIDENCE ONLY - NEGATIVE - NOT QUALIFIED OR DEPLOYABLE**

## Protocol And Boundary

This is the V5 development study declared in
`contracts/habitat_v2_forecast_issue_56_v4_model_preregistration_v5.json`,
which added context-gated selection (`context_gated_select_v1`) on top of the
V4 protocol revision lineage. It uses the complete 32-family development
roster: 20 TRAIN, 6 VALIDATION, and 6 EVALUATION families grouped into 16
paired sensor condition groups under the V3 support-stratified split. No
protected final-suite data was accessed. V3 artifacts remain unchanged.

The V4 model is advisory-only. HMC remains the sole final-command, plant-step,
and replay authority.

## Reproduction And Receipts

The corpus was built from a clean source tree at the protocol V5 commit and
independently verified with strict serialized-trace replay:

```bash
uv run --locked --python 3.11 --extra dev python scripts/build_action_risk_v4_corpus.py \
  --output out/issue56-v4-corpus-v5-full-01 \
  --families 32 --resume

uv run --locked --python 3.11 --extra dev python scripts/verify_action_risk_v4_corpus.py \
  --corpus out/issue56-v4-corpus-v5-full-01
```

Corpus receipt: `out/issue56-v4-corpus-v5-full-01`.

- Samples: `1,664` (`1,040` TRAIN, `312` VALIDATION, `312` EVALUATION)
- Replayable traces: `1,696`
- Corpus manifest SHA-256: `492959f96fbfbc2b41e6e786556c151bddb98c323c96e41fa4e0cc48f17ceb20`
- Samples SHA-256: `25b0234abf82d5d45d978ebb57c98f8ab90e7fde440e63960cfee7494561ce7a`
- Trace-manifest SHA-256: `77f2a504b9eac61012348f687f2d6c1f96236249cc008876f70bf6b937471051`
- Independent strict replay: passed

## Study Runs

Two full study attempts under protocol V5 did not complete:

- `out/issue56-v4-study-v5-full-01` fitted the four registered Stage A
  candidate models (`c0_v3_refit`, `c5_action_conditioned_ridge`,
  `c6_action_conditioned_temporal`, `c7_action_conditioned_cumulative`) but
  was interrupted before Stage A metrics and `results.json` were written. No
  Stage A gate verdicts exist for this run.
- `out/issue56-v4-study-v5-full-02` was interrupted before producing output.

A follow-up focused Stage B harness (`out/issue56-v4-stageB-c0_v3_refit`)
reused the `c0_v3_refit` model fitted by the interrupted run and replayed only
the Stage B HMC episodes. This harness ran with an out-of-protocol, locally
relaxed superiority variant and a stubbed latency record; it is exploratory,
not a preregistered V5 result, and its relaxed gate was not adopted. Even so,
it did not achieve superiority:

- Admitted proposals: V4 `2`, V3 `2` (tie, not an exceed)
- HMC mismatch count: `0`; emergency overrides: `0`
- Paired safety-exposure point difference (V4 в€’ V3): `+3.9736e-05`,
  bootstrap CI `[0.0, +8.7420e-05]` (10,000 resamples, seed 560057)
- `safety_no_worse_than_v3`: `false` under the registered `0.0` maximum
- Episodes SHA-256: `dcb70645ee7b7d29bced3c96f8e31d34121640b1f76490d06ca5e839cd9f2cea`

## Structural Finding

Corpus label analysis (decision-step `remaining_metric.crossing_event`) shows
that crossing events occur only in condition groups `g0000`вЂ“`g0003`, which are
exactly the families under the `nominal_occupied` operating condition
(family indices 0вЂ“7). The other twelve condition groups produce no crossing
events at any of the 13 decision steps, with or without injected plant faults.

Under the V3 support-stratified split, EVALUATION contains exactly one
eventful condition group (`g0001`). The frozen V3 risk-filtered point arm
already admits `2` proposals there and abstains on the remaining `76`
decisions, so no V4 candidate can exceed V3's admitted proposal count on this
split. Superiority over V3 is therefore structurally unreachable under the V5
evaluation population, independent of model quality.

## Limitations And Decision

- The result is simulator development evidence only; it does not establish qualification, certification, deployment readiness, hardware performance, real-world safety, or authority to control equipment.
- The interrupted runs mean protocol V5 has no completed preregistered study receipt; the focused Stage B harness is explicitly out-of-protocol.
- No post-result threshold tuning, protected-data access, V3 artifact change, or HMC authority change occurred. The in-place relaxed-gate amendment drafted against the V5 contract was rejected and reverted; preregistered gates are only changed by a new protocol version.

Decision: retain V5 as an honest fail-closed negative development result. A
successor protocol V6 is preregistered with a redesigned evaluation split that
places two eventful condition groups in EVALUATION so superiority over V3
becomes decidable.
