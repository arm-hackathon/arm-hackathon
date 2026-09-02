# Issue #56 V4 Model V2 Development Study

Date: 2026-08-29
Branch: `research/action-risk-v4-corpus`
Source commit: `f6fa742be54181afae5712829296d56ad694edcb`
Status: **DEVELOPMENT EVIDENCE ONLY - NEGATIVE - NOT QUALIFIED OR DEPLOYABLE**

## Protocol And Boundary

This is the separately authorized V2 development study declared in
`contracts/habitat_v2_forecast_issue_56_v4_model_preregistration_v2.json`. It
uses the complete 32-family development roster: 20 TRAIN, 6 VALIDATION, and 6
EVALUATION families grouped into 16 paired sensor conditions. No protected
final-suite data was accessed. V3 artifacts and historical V4 V1 artifacts
remain unchanged.

The V4 model is advisory-only. HMC remains the sole final-command, plant-step,
and replay authority. This run is `offline_model_only`; it issued no HMC
proposals or plant steps, so HMC mismatch and proposal-rate metrics are
unavailable rather than passing or failing.

## Reproduction And Receipts

The corpus was built through the resumable builder from the clean source tree
and independently verified with strict serialized-trace replay:

```bash
uv run --locked --python 3.11 --extra dev python scripts/build_action_risk_v4_corpus.py \
  --output out/issue56-v4-corpus-v2-full-20260829-01 \
  --families 32 --resume

uv run --locked --python 3.11 --extra dev python scripts/verify_action_risk_v4_corpus.py \
  --corpus out/issue56-v4-corpus-v2-full-20260829-01

uv run --locked --python 3.11 --extra dev python scripts/run_action_risk_v4_model.py \
  --corpus out/issue56-v4-corpus-v2-full-20260829-01 \
  --output out/issue56-v4-model-v2-full-20260829-01
```

Corpus receipt: `out/issue56-v4-corpus-v2-full-20260829-01`.

- Samples: `1,664` (`1,040` TRAIN, `312` VALIDATION, `312` EVALUATION)
- Replayable traces: `1,696`
- Corpus manifest SHA-256: `906ac9a59095a7047f2d0cba2b11060cec54a18eed1b25a6b02ef5a4ca92d148`
- Corpus results SHA-256: `6e437e25f912918ea3172c13c6fa421348760b0007d9aa27d002086f41049940`
- Samples SHA-256: `25b0234abf82d5d45d978ebb57c98f8ab90e7fde440e63960cfee7494561ce7a`
- Trace-manifest SHA-256: `77f2a504b9eac61012348f687f2d6c1f96236249cc008876f70bf6b937471051`
- Independent strict replay: passed

Model receipt: `out/issue56-v4-model-v2-full-20260829-01`.

- Model manifest SHA-256: `9ec1d23bf128488ce55be0465d343b5d00f4d9d86ab29092267740d1048842ce`
- Model results SHA-256: `579bc609c21bc3a04cc4ba4fcd51d7f0a3013b93310eaaa389c6779503d8522c`
- Candidate count: `5`
- Overall status: `DEVELOPMENT_EVIDENCE`
- All preregistered gates passed: `false`

The runner uses a trace-free projection after strict family replay, so model
fitting and evaluation do not retain serialized future traces in memory. This
changes memory behavior only; it does not widen the model input projection or
alter corpus labels.

## Candidate Results

| Candidate | Status | Useful actions | Distinct actions | Selected | Abstention rate | Dangerous recall | Selected false-safe rate |
|---|---|---:|---:|---:|---:|---:|---:|
| `c0_v3_refit` | FAIL | 12 | 1 | 64 | 0.1794872 | 1.0 | 0.0 |
| `c1_shared_hazard_ridge` | FAIL | 1 | 1 | 53 | 0.3205128 | 1.0 | 0.0 |
| `c2_shared_hazard_temporal` | FAIL_CLOSED | N/A | N/A | N/A | N/A | N/A | N/A |
| `c3_small_shared_mlp` | FAIL_CLOSED | N/A | N/A | N/A | N/A | N/A | N/A |
| `c4_advantage_ranker` | FAIL_CLOSED | N/A | N/A | N/A | N/A | N/A | N/A |

`c0` and `c1` passed the dangerous-event recall, selected-action false-safe,
authority, replay, provenance, finite-metric, abstention, and latency checks.
Both failed the registered minimum-useful-action gate of `16` and minimum
distinct-selected-action gate of `2`. `c2`, `c3`, and `c4` failed closed during
validation calibration because no registered threshold could meet the recall
target.

## Comparison With Frozen V3

The frozen V3 receipt is
`out/issue56-v3-evaluation-20260825-clean-a/results.json`, with its evidence
record in `docs/evidence/issue-56-action-risk-v3-support-revision.md`. V3's
risk-filtered point arm selected `2` proposals and abstained on `76` decisions
across six EVALUATION families, while passing its declared safety comparison.

The V4 V2 run cannot support a direct superiority claim against that V3 result:
V4 is an offline 32-family development study with no HMC execution, whereas the
frozen V3 receipt contains HMC-replayed evaluation episodes. The valid V4
conclusion is narrower: the corrected V4 protocol completed successfully, but
none of its five registered candidates passed all gates, and it does not replace
the frozen V3 evidence.

## Limitations And Decision

- The result is simulator development evidence only; it does not establish qualification, certification, deployment readiness, hardware performance, real-world safety, or authority to control equipment.
- The EVALUATION population contains only three condition groups, so the grouped bootstrap intervals are descriptive and not qualification evidence.
- HMC-dependent metrics are unavailable because the authorized study path is offline-only.
- No post-result threshold tuning, protected-data access, V3 artifact change, or HMC authority change occurred.

Decision: retain V4 V2 as an honest fail-closed negative development result. Do
not enable it as a deployed or qualified controller. Any attempt to reduce
abstention or change calibration requires a new preregistered protocol and
safety review.
