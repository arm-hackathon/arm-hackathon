# Issue #56 V4 Model V6 Development Study

Date: 2026-08-31
Branch: `research/action-risk-v4-corpus`
Source commit: `8e286f41d5e2009e5f8a3001147d38a827bc88e2`
Status: **DEVELOPMENT EVIDENCE ONLY - NEGATIVE - NOT QUALIFIED OR DEPLOYABLE**

## Protocol And Boundary

This is the V6 development study declared in
`contracts/habitat_v2_forecast_issue_56_v4_model_preregistration_v6.json`
(protocol SHA-256
`b200103a0ebe4711b15b0a35208ccba53ad9f59159ffa3943c6999a0b3564165`).
Protocol V6 redesigned the evaluation split after the V5 negative result:
condition group `g0003` (TRAIN) was swapped with `g0005` (EVALUATION) so
EVALUATION holds two eventful condition groups (`g0001`, `g0003`) while TRAIN
keeps `g0002` and VALIDATION keeps `g0000`. Split counts (20/6/6 families),
candidate roster, context gates, and the strict superiority gate were carried
over unchanged from V5. No protected final-suite data was accessed. V3
artifacts remain unchanged.

The V4 model is advisory-only. HMC remains the sole final-command, plant-step,
and replay authority.

## Reproduction And Receipts

```bash
uv run --locked --python 3.11 --extra dev python scripts/build_action_risk_v4_corpus.py \
  --output out/issue56-v4-corpus-v6-full-01 \
  --families 32 --split-protocol issue56_v4_model_split_v6

uv run --locked --python 3.11 --extra dev python scripts/verify_action_risk_v4_corpus.py \
  --corpus out/issue56-v4-corpus-v6-full-01 \
  --split-protocol issue56_v4_model_split_v6

uv run --locked --python 3.11 --extra dev python scripts/run_action_risk_v4_study_v3.py \
  --corpus out/issue56-v4-corpus-v6-full-01 \
  --output out/issue56-v4-study-v6-full-01 \
  --protocol-version v6
```

Corpus receipt: `out/issue56-v4-corpus-v6-full-01`.

- Samples: `1,664` (`1,040` TRAIN, `312` VALIDATION, `312` EVALUATION)
- Replayable traces: `1,696`
- Corpus manifest SHA-256: `46d8aab2d30aa6744d536d27b79ff60d5bddca194e37388be9a3a0d90b31a4d5`
- Samples SHA-256: `bf361f3b2652749bece30ec6441382b42344454d2867f8327e9172b439d32718`
- Trace-manifest SHA-256: `77f2a504b9eac61012348f687f2d6c1f96236249cc008876f70bf6b937471051` (byte-identical traces to V5; only sample split labels changed)
- Independent strict replay: passed

Study receipt: `out/issue56-v4-study-v6-full-01`.

- Results SHA-256: `36dee8df9e30bab9d4cb1b9d8aa84e50b368f5cf3285496aa4bddba270770efd`
- Episodes SHA-256: `624b517a26df847236178d54acbd7bc3917099e8a84b2ba30c31923ebbfc8ad9`
- Frozen V3 baseline refit matches frozen artifact: `true`
- Overall status: `DEVELOPMENT_EVIDENCE`
- All preregistered gates passed: `false`
- Outperforms V3: `false`

## Candidate Results (Stage A)

| Candidate | Status | Useful actions | Distinct actions | Dangerous recall | Abstention rate |
|---|---|---:|---:|---:|---:|
| `c0_v3_refit` | FAIL | 16 | 1 | 1.0 | 0.577 |
| `c5_action_conditioned_ridge` | FAIL | 0 | 3 | 1.0 | 0.731 |
| `c6_action_conditioned_temporal` | FAIL | 0 | 3 | 1.0 | 0.744 |
| `c7_action_conditioned_cumulative` | FAIL | 3 | 3 | 1.0 | 0.692 |

The redesigned evaluation population raised `c0`'s useful-action count to the
registered minimum of `16` (V5 observed `7` under the old split), confirming
the split was the binding constraint on usefulness. `c0` still fails the
minimum-distinct-selected-action gate (`1 < 2`): it collapses onto
`normal-dormant-v1`. The action-conditioned candidates spread over three
actions but find almost no useful selections. No candidate passed all Stage A
gates, so Stage B ran `c0_v3_refit` under the registered rule
`stage_a_passer_else_best_safety_passing_usefulness`.

## Stage B (HMC Replay) And Comparison With V3

- Admitted proposals: V4 `4`, V3 `4` (tie, not an exceed)
- HMC mismatch count: `0`; emergency overrides: `0`
- Paired safety-exposure point difference (V4 в€’ V3): `+3.1789e-05`,
  bootstrap CI `[0.0, +7.9473e-05]` (10,000 resamples, seed 560057)
- Paired comfort deviation difference: `+6.1648`; resource composite: `-9.5993e-05`
- `safety_no_worse_than_v3`: `false`; `more_admitted_proposals_than_v3`: `false`
- Superiority achieved: `false`

Both arms admit exactly one proposal in each eventful evaluation family and
abstain everywhere else. V3 proposes `normal-dormant-v1` at decision steps
36/48/48/48 across the four eventful families; V4 proposes the same action at
48/48/52/48. The two later V4 proposals leave the plant unprotected longer,
which accounts for the small positive paired safety difference.

## Structural Finding

Counterfactual labels show that in every eventful evaluation family, all 13
decision steps have at least one action (`normal-dormant-v1`) that improves
remaining safety exposure relative to the no-proposal hold вЂ” 52 improving
decisions in total. Despite this, every replayed arm (rules-only, point model,
frozen V3, V4) admits exactly one proposal per eventful family: once the
dormant command is accepted during the step-32вЂ“80 fault window, the residual
risk is small and no further proposal passes the predicted-improvement screen
within the decision horizon (decisions end at step 64, before the fault window
closes). The admitted-proposal count on this evaluation population is
therefore capped at `4`, and the frozen V3 arm already achieves `4`. The
registered superiority clause `admitted_proposal_count_must_exceed_v3` is thus
structurally unreachable for any policy that proposes only genuinely
safety-improving actions; inflating admissions would require vacuous proposals
that the eligibility screen (strict predicted improvement versus hold) exists
to prevent.

This finding rests on the replayed episodes and counterfactual labels above;
the claim that no second intervention per family can improve safety is
inferred from those observations and the fixture physics, not directly
measured for every possible two-proposal schedule.

## Limitations And Decision

- The result is simulator development evidence only; it does not establish qualification, certification, deployment readiness, hardware performance, real-world safety, or authority to control equipment.
- The V6 split achieved its design goal (eventful evaluation capacity doubled; V3 admissions rose 2 в†’ 4 and c0 useful actions rose to 16), but V4 matches rather than exceeds V3.
- A model that proposes earlier than V3 (e.g., at steps 28вЂ“32 when the fault horizon first becomes visible) could achieve the same 4 admissions with strictly less safety exposure; the registered gate does not credit that because it requires strictly more admissions.
- No post-result threshold tuning, protected-data access, V3 artifact change, or HMC authority change occurred.

Decision: retain V6 as an honest fail-closed negative development result. Any
successor protocol that revises the superiority criterion (for example,
crediting equal admissions with strictly better paired safety and earlier
intervention timing) or closes the V4 model line requires separate explicit
authorization.
