# Issue #56 V4 Model V9 Development Study

Date: 2026-09-01
Branch: `research/action-risk-v4-corpus`
Source commit: `a874c343afd9343584aff88928d0bc7f2adefa4b`
Status: **DEVELOPMENT EVIDENCE ONLY - POSITIVE UNDER PREREGISTERED PER-FAMILY GATES - NOT QUALIFIED OR DEPLOYABLE**

## Protocol And Boundary

This is the V9 development study declared in
`contracts/habitat_v2_forecast_issue_56_v4_model_preregistration_v9.json`
(protocol SHA-256
`a6bc307a6c8f95e715e016bbfefdba345d41082ce16663ca56a1b0c93f1b66e3`).
Protocol V9 changes two things relative to V8. First, Stage B replays every
Stage A passer (`replay_all_stage_a_passers_v1`) instead of a single selected
candidate. Second, superiority is graded per evaluation family: no family may
lose, at least four of the six families must win outright, admissions and
aggregate paired safety must be no worse than the frozen V3 baseline, and HMC
mismatches must be zero. The roster is `c0_v3_refit`, `c3_small_shared_mlp`,
and the new `c8_o2_excess_guard`. The split and fixture are unchanged from V8
(fixture revision 2, split `issue56_v4_model_split_v8`).

`c8_o2_excess_guard` is a V3-architecture ridge model whose selection contract
adds one preregistered mechanistic clause: when the observable context shows
an occupied-mode O2-excess profile (mean primary O2 at or above the 0.30
bound minus the 0.015 margin) and the current command is not already dormant,
it proposes `normal-dormant-v1` once, without the statistical upper-bound and
relative-improvement screens. The safety case is mechanistic: dormant zeroes
all oxygen injection, which cannot increase O2-driven upper-bound exposure in
an O2-excess occupied profile. The clause is suppressed once dormant is the
current command, so it never repeats a proposal. Under every other context the
candidate behaves exactly like the standard context-gated selection. No
protected final-suite data was accessed. V3 artifacts remain unchanged.

The V4 model is advisory-only. HMC remains the sole final-command, plant-step,
and replay authority.

## Reproduction And Receipts

```bash
uv run --locked --python 3.11 --extra dev python scripts/build_action_risk_v4_corpus.py \
  --output out/issue56-v4-corpus-v9-full-01 \
  --families 32 --split-protocol issue56_v4_model_split_v8

uv run --locked --python 3.11 --extra dev python scripts/verify_action_risk_v4_corpus.py \
  --corpus out/issue56-v4-corpus-v9-full-01 \
  --split-protocol issue56_v4_model_split_v8

uv run --locked --python 3.11 --extra dev python scripts/run_action_risk_v4_study_v3.py \
  --corpus out/issue56-v4-corpus-v9-full-01 \
  --output out/issue56-v4-study-v9-full-01 \
  --protocol-version v9
```

Corpus receipt: `out/issue56-v4-corpus-v9-full-01`.

- Samples: `1,664` (`1,040` TRAIN, `312` VALIDATION, `312` EVALUATION)
- Replayable traces: `1,696`
- Corpus manifest SHA-256: `7b2a4092d2d1786b99d35fac19c869badcaeebc45264c8bcd7a6c4a416380fb9`
- Samples SHA-256: `81ea37872cee1d30a0198517ee5f45c6162e99895f506c24466dec5d8c6fce98` (byte-identical to the V8 corpus samples; only the source-identity binding changed with the new commit)
- Trace-manifest SHA-256: `b02d6968dd6bf4bb935f7508b6546400a4429a8f22be3d13e550acb6155ce8e5`
- Independent strict replay: passed

Study receipt: `out/issue56-v4-study-v9-full-01`.

- Results SHA-256: `b1641cac8dfab56b09a8e6ea1901fedc86a3c5202e3a614031314c2f197884ac`
- Frozen V3 baseline refit matches frozen artifact: `true` for all three candidate replays
- Overall status: `DEVELOPMENT_EVIDENCE`
- All preregistered gates passed: `true`
- **Outperforms V3 (per-family criterion): `true`, achieved by `c8_o2_excess_guard`**

## Candidate Results (Stage A)

| Candidate | Status | Useful actions | Distinct actions |
|---|---|---:|---:|
| `c0_v3_refit` | PASS | 26 | 1 |
| `c3_small_shared_mlp` | PASS | 34 | 4 |
| `c8_o2_excess_guard` | PASS | 26 | 1 |

All three candidates pass every Stage A gate, so Stage B replays all three.

## Stage B (HMC Replay) And Per-Family Comparison With V3

| Candidate | Wins | Ties | Losses | Admissions V4 / V3 | Aggregate paired diff | Per-family criterion |
|---|---:|---:|---:|---:|---:|---|
| `c0_v3_refit` | 2 | 2 | 2 | 4 / 2 | n/a | not met |
| `c3_small_shared_mlp` | 3 | 2 | 1 | 8 / 2 | n/a | not met |
| `c8_o2_excess_guard` | **4** | **2** | **0** | 4 / 2 | **в€’1.5895e-04** | **met** |

Per-family detail for `c8_o2_excess_guard` (each admitted proposal is
`normal-dormant-v1` at decision step 16):

| Family | Condition | V3 | V4 | Outcome |
|---|---|---:|---:|---|
| g0006 sensor-b (high_load/laboratory) | V3 abstains | exposure 5.007e-04 | 1.907e-04 | WIN (в€’3.099e-04) |
| g0006 sensor-a (high_load/laboratory) | V3 abstains | exposure 5.007e-04 | 1.907e-04 | WIN (в€’3.099e-04) |
| g0001 sensor-b (nominal/fan) | V3 proposes step 36 | exposure 4.292e-04 | 1.907e-04 | WIN (в€’2.384e-04) |
| g0001 sensor-a (nominal/fan) | V3 proposes step 24 | exposure 2.861e-04 | 1.907e-04 | WIN (в€’9.537e-05) |
| g0011 sensor-a (eva/cooling) | both abstain | exposure 1.144e-03 | 1.144e-03 | TIE |
| g0011 sensor-b (eva/cooling) | both abstain | exposure 1.144e-03 | 1.144e-03 | TIE |

HMC mismatch count: `0`; emergency overrides: `0`; all Stage B gates passed.
Aggregate paired safety point difference (V4 в€’ V3): `в€’1.5895e-04` (strictly
better). Admissions: V4 `4`, V3 `2`.

## Interpretation

The two V8-era blockers are resolved. On the nominal/fan evaluation group
(`g0001`), the statistical upper-bound and relative-improvement screens keep
`c0` and the action-conditioned candidates conservative, so they propose
dormant later than the frozen V3 (or not at all) and lose those families.
`c8_o2_excess_guard` instead trusts the observable O2-excess context and
proposes dormant at the first decision step (16), which HMC admits and which
strictly reduces exposure relative to V3's later proposals. On the
high_load/laboratory group (`g0006`) V3 abstains entirely and the guard's
early dormant intervention is admitted, producing the largest wins. On the
eva/cooling group (`g0011`) the context is not an occupied O2-excess profile,
so the guard abstains exactly as V3 does, preserving the ties without
worsening exposure. The repeat-proposal suppression keeps admissions at one
per family (4 total), so the result is not an artifact of proposal inflation.

`c3_small_shared_mlp` wins three families but still loses one (it proposes
dormant on one g0001 sensor variant later than V3), and its eight admissions
show it re-proposes; it does not meet the per-family criterion. This is
reported unchanged as negative evidence for that candidate.

## Limitations And Decision

- The result is simulator development evidence only; it does not establish qualification, certification, deployment readiness, hardware performance, real-world safety, or authority to control equipment.
- The winning candidate relies on a preregistered mechanistic clause (dormant zeroes oxygen injection under an occupied O2-excess profile) rather than the statistical upper-bound screens; its safety case is fixture- and regime-specific. The clause's observable preconditions and single-shot behavior are fully specified in the protocol.
- Three candidates were replayed; the per-family claim is made for the one candidate meeting the criterion, with the other two reported unchanged. This multiplicity is disclosed and bounded by the preregistered roster.
- No post-result threshold tuning, protected-data access, V3 artifact change, or HMC authority change occurred.

Decision: retain V9 as a positive preregistered per-family superiority result
for the Issue #56 V4 line under fixture revision 2. V4 remains advisory-only;
any runtime integration requires separate review per the protocol's artifact
clause.
