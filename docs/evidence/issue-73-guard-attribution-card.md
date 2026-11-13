# Issue #73 guard-attribution ablation — evidence card

Date: 2026-09-06
Issue: #73 (evaluation, development-only)
Preregistration: `contracts/habitat_v2_bdm_v1_ablation_preregistration_v1.json`
(sha256 `f81ba92fadc7c8a8975df20eb32ade88014bdbbaee37b4a6099f7e58f6527418`,
status `FROZEN_BEFORE_STUDY_RUNS`, committed before any study run)
Study receipt: `out/issue73-ablations-v2/ablation-receipt.json`
(sha256 `b22f477a0580fdd9bc635083d5a39bf9f4d8d6378f60f002f32dd4d5d1d13a8b`)

## What was run

Five preregistered arms over all 32 DEV families (160 episodes, 96 steps each,
13 decision steps): `hmc_rules_only`, `hold_current_command`,
`c8_guard_only`, `c9_learned_screen_without_guard`, `c9_full`. Statistical
screens (ridge and 16-hidden-unit MLP on the declared 33-dim observable
recipe) were fit on TRAIN corpus samples only
(`out/bdm-v1-corpus-v1`, digest `ecec280680a0ba3c984c7182dc288241014b46b3c0687f8bcc91e3322510c219`)
and never on DEV or beyond. Every episode record binds its family scenario
digest; each family shows exactly one scenario digest across all five arms,
proving identical exogenous traces per family.

## Results (stated plainly)

- Safety exposure is 0.0 in every arm on every DEV family: the HMC hold keeps
  all fresh families inside all race target bounds, so no arm can show a
  safety difference.
- Admitted proposals: zero in every arm. The guard attempted proposals on 6
  families and the statistical screens on 20 (learned-only) and 22 (hybrid)
  families; the HMC rejected every attempt under its mode/reserve policies,
  and rejections are recorded per record (`rejected_proposal_count`), never
  fatal.
- Attribution credits (paired group-level differences versus hold):
  guard credit 0.0, learned credit 0.0, hybrid credit 0.0, each with
  degenerate zero-width bootstrap CIs at causal-group level.
- Null learned contribution is retained as the result: on the fresh roster
  neither the deterministic O2-excess guard nor the learned statistical
  screen adds any admitted intervention beyond HMC hold. The Issue #56
  V4/V10 wins do not transfer to these families, and no part of the
  deterministic guard's behaviour is relabelled as learned-model progress.

## Checklist to evidence

| Issue checklist item | Evidence |
| --- | --- |
| Ablation contract frozen before study run | preregistration committed in `3268ebf` with `frozen_before_study_runs: true`; runner refuses unfrozen prereg |
| Identical scenario families and exogenous traces per arm | receipt records one scenario sha256 per family across all five arms; runner asserts registry digests at rebuild |
| Learned-only, guard-only, hybrid reported separately | receipt `tables` and `records` per arm; this card |
| CIs at independent condition groups | `bdm_v1_evaluation.comparison_table` bootstrap over causal groups (10 000 resamples, seed `issue73-bootstrap-v1`); unit tests reject smaller units |
| Zero HMC final-command / replay / provenance / non-finite mismatches | episode runner verifies shadow plant receipt digests against HMC receipts each step and rejects non-admitted bookkeeping; all 160 records passed |
| Negative or null learned contribution retained plainly | this card's results section; receipt stores the zero credits |
| No blind population opened | study partition DEV only; screens fit on TRAIN only; BLIND_FINAL absent from corpus and receipt |

## Boundaries and non-claims

Development evidence only. No model promotion, no qualification or deployment
claim, no HMC authority change, no blind access. Historical Issue #56 V4/V10
artifacts and results remain untouched historical evidence. The null result
does not imply the guard or screen are useless in other operating regimes;
it states only that on these fresh development families, behind this HMC,
they changed nothing.

## Reproduction

```bash
uv run --locked --python 3.11 --extra dev python -m pytest -q tests/habitat_v2/test_issue73_ablations.py
uv run --locked --python 3.11 --extra dev python \
  scripts/run_issue73_ablations.py --corpus out/bdm-v1-corpus-v1 \
  --output out/issue73-ablations-<new>
```
