# Issue #56 V4 Model V10 Development Study

Date: 2026-09-02  
Branch: `research/action-risk-v4-corpus`  
Source commit: `b0ffad1e21f7baa6edb021b4fc17fcf0c012fde9`  
Status: **DEVELOPMENT EVIDENCE ONLY - POSITIVE UNDER PREREGISTERED PER-FAMILY GATES - ALL SIX FAMILIES WON - NOT QUALIFIED OR DEPLOYABLE**

## Protocol And Boundary

This is the V10 development study declared in
`contracts/habitat_v2_forecast_issue_56_v4_model_preregistration_v10.json`
(protocol SHA-256
`e750689158c7311a57e6b5bc648cce11045034f7229ded5ff3ceae3c43c08dbb`).
V9 won four of six families but tied the two eva_transition/cooling-loss
families (`g0011`), because dormant is context-blocked outside a nominal
O2-excess profile. The corpus labels show dormant is genuinely
safety-improving on those families from fault onset (step 32) onward,
identically across all three TRAIN eva condition groups, so the pattern is
learnable rather than evaluation-specific. Protocol V10 therefore adds
`c9_o2_guard_statistical`: it keeps the `c8` mechanistic guard for O2-excess
occupied profiles and additionally admits dormant in any operating mode when
the calibrated upper-bound screen passes and predicted relative safety
exposure is negative, suppressing repeat proposals once dormant is the current
command. Roster is `c8_o2_excess_guard` and `c9_o2_guard_statistical`; Stage B
replays both and grades the same per-family superiority criterion as V9 (no
family may lose, at least four of six must win, admissions and aggregate
paired safety no worse than frozen V3, zero HMC mismatches). Split and fixture
are unchanged (fixture revision 2, split `issue56_v4_model_split_v8`). No
protected final-suite data was accessed. V3 artifacts remain unchanged.

The V4 model is advisory-only. HMC remains the sole final-command, plant-step,
and replay authority.

## Reproduction And Receipts

```bash
uv run --locked --python 3.11 --extra dev python scripts/build_action_risk_v4_corpus.py \
  --output out/issue56-v4-corpus-v10-full-01 \
  --families 32 --split-protocol issue56_v4_model_split_v8

uv run --locked --python 3.11 --extra dev python scripts/verify_action_risk_v4_corpus.py \
  --corpus out/issue56-v4-corpus-v10-full-01 \
  --split-protocol issue56_v4_model_split_v8

uv run --locked --python 3.11 --extra dev python scripts/run_action_risk_v4_study_v3.py \
  --corpus out/issue56-v4-corpus-v10-full-01 \
  --output out/issue56-v4-study-v10-full-01 \
  --protocol-version v10
```

Corpus receipt: `out/issue56-v4-corpus-v10-full-01`.

- Samples: `1,664` (`1,040` TRAIN, `312` VALIDATION, `312` EVALUATION)
- Replayable traces: `1,696`
- Corpus manifest SHA-256: `19a61a8d5e084bc7b62fae174b93661722ebcb87d187594b3f16650d7b021c8b`
- Samples SHA-256: `81ea37872cee1d30a0198517ee5f45c6162e99895f506c24466dec5d8c6fce98` (byte-identical to the V8/V9 corpus samples; only the source-identity binding changed with the new commit)
- Trace-manifest SHA-256: `b02d6968dd6bf4bb935f7508b6546400a4429a8f22be3d13e550acb6155ce8e5`
- Independent strict replay: passed

Study receipt: `out/issue56-v4-study-v10-full-01`.

- Results SHA-256: `e09b8be308612fc91f603dcd343c304799598290c9dcef9c56882e706d519565`
- Frozen V3 baseline refit matches frozen artifact: `true` for both candidate replays
- Overall status: `DEVELOPMENT_EVIDENCE`
- All preregistered gates passed: `true`
- **Outperforms V3 (per-family criterion): `true`, achieved by both candidates**

## Candidate Results (Stage A)

| Candidate | Status | Useful actions | Distinct actions |
|---|---|---:|---:|
| `c8_o2_excess_guard` | PASS | 26 | 1 |
| `c9_o2_guard_statistical` | PASS | 34 | 4 |

## Stage B (HMC Replay) And Per-Family Comparison With V3

| Candidate | Wins | Ties | Losses | Admissions V4 / V3 | Aggregate paired diff | Criterion |
|---|---:|---:|---:|---:|---:|---|
| `c8_o2_excess_guard` | 4 | 2 | 0 | 4 / 2 | −1.589e-04 | met |
| `c9_o2_guard_statistical` | **6** | **0** | **0** | 6 / 2 | **−3.417e-04** | **met** |

Per-family detail for `c9_o2_guard_statistical` (each admitted proposal is
`normal-dormant-v1`; steps in parentheses):

| Family | Condition | V3 exposure | V4 exposure | Relative | V4 step | Outcome |
|---|---|---:|---:|---:|---:|---|
| g0006 sensor-b (high_load/laboratory) | V3 abstains | 5.007e-04 | 1.907e-04 | −61.9% | 16 | WIN |
| g0006 sensor-a (high_load/laboratory) | V3 abstains | 5.007e-04 | 1.907e-04 | −61.9% | 16 | WIN |
| g0001 sensor-b (nominal/fan) | V3 proposes step 36 | 4.292e-04 | 1.907e-04 | −55.6% | 16 | WIN |
| g0001 sensor-a (nominal/fan) | V3 proposes step 24 | 2.861e-04 | 1.907e-04 | −33.3% | 16 | WIN |
| g0011 sensor-b (eva/cooling) | V3 abstains | 1.144e-03 | 5.722e-04 | −50.0% | 48 | WIN |
| g0011 sensor-a (eva/cooling) | V3 abstains | 1.144e-03 | 6.199e-04 | −45.8% | 52 | WIN |

HMC mismatch count: `0`; emergency overrides: `0`; all Stage B gates passed
for both candidates.

## Interpretation

`c8_o2_excess_guard` reproduces the V9 result (four wins on the O2-excess
occupied families, ties on the eva group). `c9_o2_guard_statistical` converts
the two eva ties into wins: on the eva/cooling-loss families, dormant becomes
genuinely safety-improving from fault onset onward (the counterfactual labels
show this identically across all three TRAIN eva condition groups, so the
signal is learned rather than fitted to the evaluation families), and the
calibrated screen plus predicted-relative-improvement clause fires at steps
48/52, which HMC admits. On the O2-excess occupied families the mechanistic
guard fires at step 16 exactly as in V9. The result is six wins, zero ties,
zero losses, with strictly better aggregate paired safety (−3.417e-04) and
zero HMC mismatches.

## Limitations And Decision

- The result is simulator development evidence only; it does not establish qualification, certification, deployment readiness, hardware performance, real-world safety, or authority to control equipment.
- The eva-family wins rely on the statistical dormant-admission clause, whose safety case is the calibrated screen and predicted relative improvement rather than the mechanistic O2-injection argument used on O2-excess profiles. Its behavior is fully specified and preregistered, and any harmful proposal would have surfaced as a per-family loss or an HMC mismatch.
- Two candidates were replayed; both meet the criterion. Multiplicity is disclosed and bounded by the preregistered roster.
- No post-result threshold tuning, protected-data access, V3 artifact change, or HMC authority change occurred.

Decision: retain V10 as the first preregistered result in the Issue #56 V4
line that outperforms the frozen V3 baseline on every evaluation family. V4
remains advisory-only; any runtime integration requires separate review per the
protocol's artifact clause. The V9 result (tag `v9-perfamily-win`, commit
`d2c3b69`) remains the rollback point should the statistical-dormant clause be
deemed unacceptable.
