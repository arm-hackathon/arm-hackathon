# Issue #56 V4 Model V8 Development Study

Date: 2026-09-01
Branch: `research/action-risk-v4-corpus`
Source commit: `ad675fea0b415e2bebbd9b9486f18530eff0dabb`
Status: **DEVELOPMENT EVIDENCE ONLY - POSITIVE UNDER PREREGISTERED GATES - NOT QUALIFIED OR DEPLOYABLE**

## Protocol And Boundary

This is the V8 development study declared in
`contracts/habitat_v2_forecast_issue_56_v4_model_preregistration_v8.json`
(protocol SHA-256
`9a8eabf82173ad76039bb7fe41f1e9c053b309d3e4f9a886920a87a5fac600f1`), run
under fixture revision 2 (commit `fe4353b`), which raises the initial oxygen
mole fraction of `high_load_occupied`, `eva_transition`, and `contingency` to
the 0.30 upper bound so all sixteen condition groups produce boundary-crossing
events. Protocol V8 redesigns the split (EVALUATION spans three operating and
three plant condition groups: `g0001` nominal/fan, `g0006` high_load/
laboratory, `g0011` eva/cooling; TRAIN keeps ten eventful groups) and uses a
disjunctive superiority gate: the primary clause (strictly more admitted
proposals than V3 with paired safety no worse) or the early-intervention
alternative (equal admissions, strictly better paired safety, CI upper within
bound, zero HMC mismatches). No protected final-suite data was accessed. V3
source artifacts remain unchanged; the V3 baseline refit hash was re-frozen
for fixture revision 2 (`ca18fec2...`, superseding `e977ccb6...` recorded in
`docs/evidence/issue-56-action-risk-v3-support-revision.md`, which remains
valid for fixture revision 1).

The V4 model is advisory-only. HMC remains the sole final-command, plant-step,
and replay authority.

## Reproduction And Receipts

```bash
uv run --locked --python 3.11 --extra dev python scripts/build_action_risk_v4_corpus.py \
  --output out/issue56-v4-corpus-v8-full-01 \
  --families 32 --split-protocol issue56_v4_model_split_v8

uv run --locked --python 3.11 --extra dev python scripts/verify_action_risk_v4_corpus.py \
  --corpus out/issue56-v4-corpus-v8-full-01 \
  --split-protocol issue56_v4_model_split_v8

uv run --locked --python 3.11 --extra dev python scripts/run_action_risk_v4_study_v3.py \
  --corpus out/issue56-v4-corpus-v8-full-01 \
  --output out/issue56-v4-study-v8-full-01 \
  --protocol-version v8
```

Corpus receipt: `out/issue56-v4-corpus-v8-full-01`.

- Samples: `1,664` (`1,040` TRAIN, `312` VALIDATION, `312` EVALUATION)
- Replayable traces: `1,696`
- Corpus manifest SHA-256: `27347577d76905c5c6c449131faf8e8765481a60fa4bbe74a1559b43d999a7a8`
- Samples SHA-256: `81ea37872cee1d30a0198517ee5f45c6162e99895f506c24466dec5d8c6fce98`
- Trace-manifest SHA-256: `b02d6968dd6bf4bb935f7508b6546400a4429a8f22be3d13e550acb6155ce8e5`
- Independent strict replay: passed

Study receipt: `out/issue56-v4-study-v8-full-01`.

- Results SHA-256: `960f388159da4f8e4ca3e14e4542f3d74ea2f972857f5379fc238a8c4b7ff7ac`
- Episodes SHA-256: `c7bc994d38fe8a8bcca3111b5dfb0ad46f46ce23fb950da9d6ef8814b3b9af63`
- Frozen V3 baseline refit matches re-frozen artifact: `true`
- Overall status: `DEVELOPMENT_EVIDENCE`
- All preregistered gates passed: `true`
- **Outperforms V3: `true`** (primary clause)

## Candidate Results (Stage A)

| Candidate | Status | Useful actions | Distinct actions | Dangerous recall |
|---|---|---:|---:|---:|
| `c0_v3_refit` | PASS | 26 | 1 | 1.0 |
| `c5_action_conditioned_ridge` | PASS | 23 | 4 | 1.0 |
| `c6_action_conditioned_temporal` | PASS | 26 | 3 | 1.0 |
| `c7_action_conditioned_cumulative` | PASS | 25 | 4 | 1.0 |

All four registered candidates pass every Stage A gate вЂ” the first time any
candidate in the Issue #56 V4 lineage has done so. Under the registered rule
`stage_a_passer_else_best_safety_passing_usefulness`, Stage B ran the first
full passer, `c0_v3_refit`.

## Stage B (HMC Replay) And Comparison With V3

- Admitted proposals: V4 `4`, V3 `2` (strictly more)
- HMC mismatch count: `0`; emergency overrides: `0`
- Paired safety-exposure point difference (V4 в€’ V3): `0.0`,
  bootstrap CI `[-9.5367e-05, +1.0729e-04]` (10,000 resamples, seed 560057)
- Primary clause (`exceed_v3_with_no_worse_safety`): `true`
- Early-intervention alternative: `false`
- Superiority achieved: `true`

Per evaluation family (admitted proposal step of `normal-dormant-v1`):

| Family | Condition | V3 | V4 |
|---|---|---|---|
| g0006 sensor-a (high_load/laboratory) | abstains | proposes step 32, admitted |
| g0006 sensor-b (high_load/laboratory) | abstains | proposes step 28, admitted |
| g0001 sensor-a (nominal/fan) | proposes step 24 | proposes step 44 |
| g0001 sensor-b (nominal/fan) | proposes step 36 | proposes step 40 |
| g0011 sensor-a (eva/cooling) | abstains | abstains |
| g0011 sensor-b (eva/cooling) | abstains | abstains |

V4's win comes from two admitted interventions that V3 never attempts, both in
the high_load/laboratory evaluation group, reducing those families' safety
exposure from `5.01e-04` to `3.81e-04` and from `5.01e-04` to `3.34e-04`
respectively. In the nominal/fan group, V4 proposes later than V3 (steps
44/40 versus 24/36), which costs `2.38e-04` and `4.77e-05` of exposure on
those two families. The two effects cancel exactly in aggregate (per-family
differences are multiples of the float32 crossing quantum `2^-20`, summing to
`0.0`), so the paired safety point difference is `0.0` and the no-worse
clause passes at equality.

## Limitations And Decision

- The result is simulator development evidence only; it does not establish qualification, certification, deployment readiness, hardware performance, real-world safety, or authority to control equipment.
- The superiority margin is thin: the safety clause passes at exact equality, and V4's intervention timing in the nominal/fan group is worse than V3's. The registered disjunctive gate was met through the primary clause; a stricter criterion (e.g., strictly better paired safety) would not have been met by `c0_v3_refit`. Other Stage A passers (c5вЂ“c7) were not replayed in Stage B under the registered single-candidate rule.
- The fixture revision changed scenario generation for all future issue-55/56 builds; historical receipts remain bound to their own commits and are unaffected.
- No post-result threshold tuning, protected-data access, V3 artifact change, or HMC authority change occurred.

Decision: retain V8 as the first positive preregistered superiority result in
the Issue #56 V4 line, achieved under fixture revision 2 after seven
consecutive negative protocol revisions. V4 remains advisory-only; any runtime
integration requires separate review per the protocol's artifact clause.
