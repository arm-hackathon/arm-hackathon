# Issue #56 V3 Support-Revision Measurements

Date: 2026-08-25
Branch: `research/action-risk-v2`
Source commit: `0829bb8b35cbd62e22b40e2b2394fc7bb21984e8`
Status: **DEVELOPMENT EVIDENCE ONLY - NOT QUALIFIED OR DEPLOYABLE**

## Protocol

This is the V3 support revision declared in
`contracts/habitat_v2_forecast_issue_56_v3_preregistration_v2.json`. It follows
the original V3 split-support audit without rewriting or pooling the historical
V1/V2/V3 results. The fixed 32-family Issue #55 development roster is grouped
into 16 consecutive sensor pairs and split into 20 TRAIN, 6 VALIDATION, and 6
EVALUATION families. No protected final-suite data was inspected or used.

The study uses the existing HMC, physics, safety, capability, trace, and replay
surfaces unchanged. The model is forecast-only and has `actuator_authority=false`;
HMC remains the sole final-command, plant-step, and replay authority.

## Run Identity

The clean comparative run was executed once from the committed source tree:

```bash
uv run --locked --python 3.11 --extra dev python scripts/run_action_risk_v3.py \
  --output out/issue56-v3-evaluation-20260825-clean-a \
  --families 32
```

Runtime: CPython 3.11.15 on Windows AMD64. The receipt records
`source_worktree_dirty: false`.

- Preregistration SHA-256, LF-normalized: `1fb290338124a63fee6790dd852e1c8629f7ea1ea1d27c1c4f12c14ec1397272`
- Manifest SHA-256: `0b5a20a0c76f30aeb2021f88173791e6af6f685089243935ab0fbd8311e2206b`
- Samples SHA-256: `ab801e82ef1ca188800f6de579de2c392e0bdb3681d79e90251268a8ea74088b`
- Model SHA-256: `e977ccb6b4298c5793838621bd819df50f46926ca2c2b73664ea9da232e4fdb8`
- Calibration SHA-256: `c8764f05c01194860ecacde8280abd97b008a0c7a1fd82ff4feb06cdf70f26bf`
- Validation non-vacuity SHA-256: `099f88a1950e9d6d321451fcc8992530ccf5155eeecb09e846f283f83201acb0`
- Episodes SHA-256: `53122a2936c8fc067f9359dce3d51ee0094d7c83827c21407d1759e3e8113531`
- Aggregate SHA-256: `834fa2d329d725c71e153fcf5a2059796d045f393b292446a1489721ba638015`
- Results SHA-256: `dadb6ffd7a1add2ea993af23db0e9e54aa2aa7740bbca9a59c61772eee4106de`

The 24 control traces are written under the ignored run directory and their
individual SHA-256 identities are included in the hashed episode records.

## Support And Calibration

- Samples: 1,664 total: 1,040 TRAIN, 312 VALIDATION, and 312 EVALUATION.
- Each family supplied 13 decision steps and four catalogue actions, or 52
  action-conditioned samples.
- Evaluation label support was positive at every horizon: 6 positives at 4
  steps, 14 at 16 steps, 56 at 32 steps, and 68 at the remaining horizon.
- Validation non-vacuity retained 71 of 78 validation decision groups, coverage
  `0.9102564103`, and passed the declared 10% minimum.
- Evaluation diagnostics retained 211 of 312 candidate samples, with 68
  dangerous samples available for the diagnostic denominator.
- Dangerous-event recall was `1.0`; selected-action false-safe rate was `0.0`;
  the unfiltered reference false-safe rate was `0.2179487179`.
- Inference p99 was `0.079223 ms`, below the declared `250 ms` limit.

Remaining-horizon calibration diagnostics were descriptive: Brier score
`0.0164496743`, positive conditional upper coverage `1.0`, and positive mean
absolute exposure error `18.9756258123`. These metrics do not constitute a
formal probability guarantee.

## Paired Evaluation Episodes

Six EVALUATION families were replayed under four common-window arms. Each arm
committed 576 steps; all 24 traces passed strict replay.

| Arm | Safety exposure mean | Violation steps mean | Comfort mean | Resource mean | Proposals | Abstentions | HMC mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|
| rules-only common window | 0.0002900759 | 24.3333 | 150.5755 | 0.0118442 | 0 | 78 | 0 |
| point-model common window | 20.4109312 | 16.3333 | 133.0236 | 0.0149528 | 2 | 76 | 0 |
| risk-only V3 | 0.0001430511 | 12.0000 | 121.9344 | 0.0125642 | 54 | 24 | 52 |
| risk-filtered point V3 | 0.0001668930 | 14.0000 | 125.7443 | 0.0124922 | 2 | 76 | 0 |

The primary paired comparison is risk-filtered point V3 versus rules-only:

- Safety-exposure point difference: `-0.0001231829`
- Deterministic paired-bootstrap 95% interval: `[-0.0002702077, 0.0]`
- Safety gate: passed
- Violation-step point difference: `-10.3333`, interval upper bound `0.0`
- Comfort point difference: `-24.8312`, interval upper bound `0.0`
- Resource-composite point difference: `+0.0006480`, interval upper bound `+0.0013680`

The 52 risk-only HMC mismatches mean the requested risk-only proposal differed
from the HMC-arbitrated final command on those cycles. This is recorded as a
requested-to-final mismatch metric; it is not counted as an authority violation
because the HMC arbitration path remained authoritative and all proposals were
admitted through the declared contract.

## Hard Gates

All values below are from `results.json` and were not repaired after evaluation:

- Authority violations: `0`
- Replay failures: `0`
- Provenance violations: `0`
- Non-finite metrics: `0`
- Proposal-admission failures: `0`
- Validation non-vacuity: passed
- Evaluation diagnostic support: passed
- Risk-filtered safety point and upper interval: passed
- Crossing-recall difference: `+1.0`, passed
- False-safe-rate difference: `-0.2179487179`, passed
- Inference p99 latency: `0.079223 ms`, passed
- Overall result: `all_gates_passed: true`

## Interpretation And Limitations

- The support revision demonstrates a reproducible development path with
  non-vacuous validation and evaluation labels on this simulator corpus.
- The risk-filtered point arm passes the preregistered safety gate and remains
  mostly abstaining: 2 proposals and 76 abstentions across the six evaluation
  families. This is not evidence of broad useful action selection.
- The risk-only arm has materially more HMC arbitration modification, so its
  requested-command behavior must not be conflated with executed-command
  behavior.
- The corpus reuses the Issue #55 V2 development family matrix and does not
  establish generalization to unseen faults, hardware, or real environments.
- The results are simulator development evidence only. They do not establish
  qualification, certification, deployment readiness, hardware performance,
  real-world safety, or authority to control equipment.
- Any future reduction in abstention or change in thresholds requires a new
  preregistered protocol and safety review; this result must not be tuned
  post hoc.

## Reproduction

Use a clean checkout at source commit `0829bb8` and a new ignored output path:

```bash
uv run --locked --python 3.11 --extra dev python scripts/run_action_risk_v3.py \
  --output out/<new-issue56-v3-run> \
  --families 32
```

The runner refuses comparative evaluation from a dirty worktree and refuses to
reuse an existing output directory.
