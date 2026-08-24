# Issue #54 Distillation Capability and Limitation Card

Date: 2026-08-24
Lane: `research/issue-54-model-distillation`
Preregistration: `contracts/habitat_v2_forecast_issue_54_preregistration_v1.json`
Preregistration SHA-256, LF-normalized:
`E16BEFB588A43F131128056932BBFE5CAA707C87309A828A33C91E1C412D5246`
Protocol addendum:
`contracts/habitat_v2_forecast_issue_54_distillation_addendum_v1.json`
Status: **CORRECTED FULL DEVELOPMENT EVIDENCE ONLY - NOT QUALIFIED OR DEPLOYABLE**

The original v1 measurements are retained only in Git history and are
superseded by the corrected full-run record in
`docs/evidence/issue-54-measurements.md`. The frozen preregistration, frozen
teachers, HMC binding, physics, and actuator paths remain unchanged.

HMC remains the sole proposal, arbitration, preflight, capability, plant-step,
and replay authority. Students are forecast-only advisory models with
`actuator_authority=false`.

## Corrected Full Run

The declared `fresh_pipeline` run used 32 whole-family scenario variations,
anchors `(16, 24, 32)`, and the four Track A `normal-*` candidates:

- 384 samples per teacher and 96 total decisions.
- 19 TRAIN, 7 VALIDATION, and 6 FINAL families.
- 18 FINAL decisions; family is the bootstrap unit.
- Every HMC trace was parsed and strictly replayed before acceptance.
- All three MLP seeds (`540054`, `540055`, `540056`) were reported separately.
- 24 MLP student artifacts were written and their recorded SHA-256 values were
  independently matched.
- TRAIN-only NMAE scales are positive finite `float64[8,51]` values with digest
  `25c9aa1dc0a441c45ca2d13e81d9ee9d725f1de2e759ecdd854e3f10f19ffc83`.

Run identities:

- MLP manifest: `6ba2d3ec6118c7a23f44d1ab70460a54f2f03287b1a121f63be18414f7ab8946`.
- Ridge manifest: `5769871fc5ce1682b8ac86c986f265e6c845bce56508db820a9730d892228406`.
- MLP samples: `aa944440767465eb7b595c4343105417fd6ba02378204d1b43036e4eff7ae5b9`.
- Ridge samples: `558da30f049008ebee9ce025c9edb9a6b6ff6ca16951bff2f1777e8d3c75eeb1`.
- Results JSON: `44abf8167d2fe6116a71d835a68714dd5ec074f8cbda5dc49fe506bec2149252`.

## Results

The primary accuracy gate is NMAE ratio point `<= 1.5` and upper 95% CI `< 2.0`.
The ranges below cover all three declared MLP seeds; `linear` is deterministic.

| Teacher | Student | Params | NMAE ratio range | Upper CI range | Tau-b range | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MLP | linear | 1,278,264 | 0.881 | 1.008 | 0.963 | PASS |
| MLP | sanity-2.1m | 2,102,936 | 0.886-0.933 | 1.031-1.074 | 0.556-0.889 | PASS |
| MLP | medium-500k | 496,148 | 1.007-1.157 | 1.206-1.332 | 0.463-0.815 | PASS |
| MLP | small-100k | 99,556 | 1.018-1.186 | 1.163-1.373 | 0.296-0.593 | PASS |
| MLP | tiny-25k | 25,195 | 1.148-1.239 | 1.257-1.436 | -0.333--0.204 | PASS accuracy, FAIL ranking |
| Ridge | linear | 3,577,344 | 0.937 | 0.996 | 1.000 | PASS |
| Ridge | sanity-2.1m | 1,835,608 | 1.163-1.171 | 1.238-1.250 | 0.889-0.963 | PASS |
| Ridge | medium-500k | 459,208 | 1.260-1.299 | 1.335-1.346 | 0.907-0.981 | PASS |
| Ridge | small-100k | 92,168 | 1.723-1.764 | 1.779-1.820 | 0.556-0.796 | FAIL accuracy |
| Ridge | tiny-25k | 18,760 | 2.269-2.324 | 2.357-2.448 | -0.167--0.037 | FAIL accuracy and ranking |

Safety-exposure differences stayed below `0.0007` for every row, well below the
numeric `0.5` closeness gate. This benign development fixture does not establish
safety under untested conditions.

## Interpretation

- The linear student is the strongest cross-teacher baseline: accuracy ratios
  `0.881` and `0.937`, with Kendall tau-b `0.963` and `1.000`.
- MLP students at every tested size pass the declared accuracy gate in this
  larger corrected corpus, including the `tiny-25k` MLP.
- Accuracy is not sufficient for action ranking: the `tiny-25k` MLP has tau-b
  from `-0.333` to `-0.204` and top-1 agreement only `0.056-0.167`.
- Ridge `medium-500k` retains strong ranking agreement and passes accuracy;
  ridge `small-100k` and `tiny-25k` fail the accuracy gate.
- Seed variation is material for ranking, so all declared seeds are reported and
  no seed is selected using FINAL results.

## Capability Boundaries

1. This is a deterministic simulation development study on the frozen fixture,
   not hardware validation, qualification, certification, or deployment proof.
2. No student may propose, arbitrate, preflight, or execute plant actions.
3. The result applies only to the declared 51-target, 8-step, 4-action Track A
   lane, corpus, teachers, and trainer.
4. The approved ranking metric is the declared
   `issue54-simplified-nominal-point-bound-v1`, not the incompatible frozen
   Issue #52 interval-bearing `score_trajectory` contract.
5. The accuracy and safety results do not authorize replacing HMC or deploying a
   student model.

## Reproduction and Verification

```bash
uv run --locked --python 3.11 --extra dev python scripts/run_issue54_distillation_study.py --output out/issue54-full-evidence-N
```

Use a new ignored output directory for each run. The full corrected run and
post-run artifact audit were followed by:

- `36 passed` focused distillation tests.
- `1113 passed` full locked test suite, including the addendum-integrity test.
- Ruff, compilation, `uv lock --check`, and `git diff --check`.

The evidence answers the Issue #54 research question for this declared lane:
small models can copy forecast accuracy surprisingly well, but tiny models can
lose action-ranking information even when their NMAE remains within the primary
accuracy gate. This is an honest development finding, not a safety certification.
