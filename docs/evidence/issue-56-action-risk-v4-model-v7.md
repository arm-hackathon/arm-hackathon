# Issue #56 V4 Model V7 Development Study

Date: 2026-08-31
Branch: `research/action-risk-v4-corpus`
Source commit: `b7b4084a52ab91e9fa67feff506250a9d317fc7b`
Status: **DEVELOPMENT EVIDENCE ONLY - NEGATIVE - NOT QUALIFIED OR DEPLOYABLE**

## Protocol And Boundary

This is the V7 development study declared in
`contracts/habitat_v2_forecast_issue_56_v4_model_preregistration_v7.json`
(protocol SHA-256
`688699bac3445875dbbe81aac9d8ff7023ac924e9a6cd655ca8b3c178842bb1b`).
Protocol V7 revised the superiority criterion after the V6 finding that the
frozen V3 arm already takes every genuinely safety-improving proposal on the
evaluation population: superiority now requires equal-or-more admitted
proposals, strictly better paired safety exposure (point difference < 0 with
bootstrap CI upper в‰¤ 0), and zero HMC mismatches. It also dropped the
`minimum_distinct_selected_actions` Stage A gate, which V6 proved structurally
unreachable (the fixture's only useful intervention action is
`normal-dormant-v1`). Split, candidate roster, and context gates are unchanged
from V6. No protected final-suite data was accessed. V3 artifacts remain
unchanged.

The V4 model is advisory-only. HMC remains the sole final-command, plant-step,
and replay authority.

## Reproduction And Receipts

```bash
uv run --locked --python 3.11 --extra dev python scripts/build_action_risk_v4_corpus.py \
  --output out/issue56-v4-corpus-v7-full-01 \
  --families 32 --split-protocol issue56_v4_model_split_v6

uv run --locked --python 3.11 --extra dev python scripts/verify_action_risk_v4_corpus.py \
  --corpus out/issue56-v4-corpus-v7-full-01 \
  --split-protocol issue56_v4_model_split_v6

uv run --locked --python 3.11 --extra dev python scripts/run_action_risk_v4_study_v3.py \
  --corpus out/issue56-v4-corpus-v7-full-01 \
  --output out/issue56-v4-study-v7-full-01 \
  --protocol-version v7
```

Corpus receipt: `out/issue56-v4-corpus-v7-full-01`.

- Samples: `1,664` (`1,040` TRAIN, `312` VALIDATION, `312` EVALUATION)
- Replayable traces: `1,696`
- Corpus manifest SHA-256: `663d1bd14c5584e175c5b3417b5c6ba827038ef4250f40536b39f99f06a77457`
- Samples SHA-256: `bf361f3b2652749bece30ec6441382b42344454d2867f8327e9172b439d32718` (byte-identical to the V6 corpus samples; only the source-identity binding changed with the new commit)
- Trace-manifest SHA-256: `77f2a504b9eac61012348f687f2d6c1f96236249cc008876f70bf6b937471051`
- Independent strict replay: passed

Study receipt: `out/issue56-v4-study-v7-full-01`.

- Results SHA-256: `537e60ca3c3e8c01f623cac9bdcedb5b59ac34839a021d588ca4f495f968a80e`
- Frozen V3 baseline refit matches frozen artifact: `true`
- Overall status: `DEVELOPMENT_EVIDENCE`
- All preregistered gates passed: `false`
- Outperforms V3: `false`

## Candidate Results (Stage A)

| Candidate | Status | Useful actions | Distinct actions | Dangerous recall | Abstention rate |
|---|---|---:|---:|---:|---:|
| `c0_v3_refit` | PASS | 16 | 1 | 1.0 | 0.577 |
| `c5_action_conditioned_ridge` | FAIL | 0 | 3 | 1.0 | 0.731 |
| `c6_action_conditioned_temporal` | FAIL | 0 | 3 | 1.0 | 0.744 |
| `c7_action_conditioned_cumulative` | FAIL | 3 | 3 | 1.0 | 0.692 |

`c0_v3_refit` is the first candidate in the Issue #56 V4 lineage to pass every
Stage A gate (under the V7 gate set, which no longer requires multiple
distinct selected actions). Stage B therefore ran under the
`stage_a_gate_passer` trigger rather than the fallback rule.

## Stage B (HMC Replay) And Comparison With V3

- Admitted proposals: V4 `4`, V3 `4` (equal, satisfies at-least clause)
- HMC mismatch count: `0`; emergency overrides: `0`
- Paired safety-exposure point difference (V4 в€’ V3): `+3.1789e-05`,
  bootstrap CI `[0.0, +7.9473e-05]` (10,000 resamples, seed 560057)
- Strictly-negative point difference: `false`; CI upper within maximum: `false`
- Superiority achieved: `false`

Proposal timing per eventful evaluation family (decision step of the admitted
`normal-dormant-v1` proposal):

| Family | V3 step | V4 step |
|---|---:|---:|
| g0003 sensor-a | 48 | 48 |
| g0001 sensor-a | 36 | 48 |
| g0001 sensor-b | 48 | 52 |
| g0003 sensor-b | 48 | 48 |

V4 never proposes earlier than V3; it proposes later in two of four eventful
families. The later dormant proposals leave the plant in the faulted regime
longer, which is exactly the `+3.18e-05` paired safety deficit.

## Structural Finding: Training-Data Asymmetry

The frozen V3 baseline is refit on the V3 split's TRAIN population, which
contains two eventful condition groups (`g0002`, `g0003`). Under any split
that keeps superiority decidable вЂ” i.e., places at least two eventful groups
in EVALUATION вЂ” TRAIN can hold at most one eventful group, because the fixture
has only four eventful groups in total and VALIDATION requires one for
threshold calibration (2 + 2 + 1 = 5 > 4). The V4 refit therefore always
learns event risk from strictly less eventful data than the frozen V3
baseline, and `c0`'s fault detection is measurably slower (proposing at 48/52
where V3 proposes at 36/48). The early-intervention superiority clause is thus
unreachable for the V3-architecture refit under any split, and the
action-conditioned candidates that might detect faster do not clear the
useful-action gate.

Combined with the V6 finding (each eventful family admits exactly one
genuinely safety-improving proposal, and frozen V3 already takes all four),
both available routes to superiority вЂ” more admitted proposals, or equal
admissions with strictly better safety вЂ” are structurally unreachable on this
fixture for any split and any candidate in the registered roster.

## Limitations And Decision

- The result is simulator development evidence only; it does not establish qualification, certification, deployment readiness, hardware performance, real-world safety, or authority to control equipment.
- The impossibility argument covers the registered candidate roster, the 32-family roster, and the frozen-V3 comparison as preregistered; it does not rule out superiority under a changed fixture (more eventful condition groups) or a changed baseline definition, both of which would require separate authorization.
- No post-result threshold tuning, protected-data access, V3 artifact change, or HMC authority change occurred.

Decision: retain V7 as an honest fail-closed negative development result. The
Issue #56 V4 model line has now produced seven consecutive protocol revisions
(v1вЂ“v7) without demonstrating superiority over the frozen V3 evidence, and the
V6/V7 structural analyses show why the fixture cannot support such a
demonstration. Recommend closing the line and keeping V4 as advisory-only
development evidence unless a separately authorized fixture or baseline change
is approved.
