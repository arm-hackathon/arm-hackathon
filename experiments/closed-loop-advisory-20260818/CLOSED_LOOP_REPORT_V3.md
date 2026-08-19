# Closed-loop report v3 — ensemble adviser with disagreement penalty

Development evidence only. Not qualification, deployment, or safety proof.
Preregistration: `preregistration-v3.json` (frozen before any ensemble-arm run,
SHA-256/16 `bf686d9801a5a363`). Scoring rule, roster, and success criteria were
fixed before outcomes were observed.

## What changed vs v2

The adviser is now a **five-member ensemble** of the frozen action-aware MLP
(base checkpoint + four seeds trained on the identical frozen outer-train
split). Candidates are ranked by:

    penalized_risk(c) = trajectory_risk(mean_member_prediction(c)) * (1 + 2.0 * disagreement(c))

where disagreement is the mean member standard deviation across the 8x51
target grid, normalized by the frozen target standard deviations. On the
frozen outer holdout the ensemble mean scores NMAE 0.1049 (single model:
0.1146) and disagreement correlates with error at r = 0.653
(`ensemble_eval.json` in the training archive).

## Determinism guard

Before the campaign, one v2 control scenario was re-run twice: both reruns
reproduced the frozen v2 integrated exceedance exactly
(1.3756209121704082 == 1.3756209121704082 == frozen record). The v2 control
and single-model arms therefore stand as valid paired comparisons.

## Results (119 ensemble runs vs frozen v2 arms, identical seeds/scenarios)

**Fault pairs (102):**

| comparison | better | equal | worse |
|---|---|---|---|
| single vs control (v2, frozen) | 78 | 24 | 0 |
| **ensemble vs control** | **78** | **24** | **0** |
| ensemble vs single | 0 | 102 | 0 |

- **P1 (no dominated regression): PASS** — zero scenarios where the ensemble
  is worse than both control and single-model arms.
- **P2 (match or beat the single-model record): PASS** — identical 78/24/0.
- Fault runs at zero exceedance: control 24/102, single 96/102, ensemble 96/102.
- Healthy no-harm: 16/17 zero for both advisers; the same EVA-transition
  healthy case scores 0.0378 in both arms (unchanged).

**Behavior did change, outcomes did not:**

| measure | single (v2) | ensemble (v3) |
|---|---|---|
| proposals made | 793 | 1,248 |
| HMC overrides | 81 | 138 |
| median battery delta vs control | +756.6 Wh | +768.9 Wh |
| median oxygen delta | +1.84 mol | +1.87 mol |
| median sorbent delta | +6.04 mol | +5.74 mol |

The disagreement penalty re-ranked candidates often enough to change 455
additional proposals (and 57 additional HMC overrides), yet realized
exceedance was identical in all 102 fault pairs.

## Interpretation (honest)

The penalty is **behavior-active but outcome-neutral on the v2 roster**.
The roster's fault scenarios have a clearly dominant safe action; both the
single model and the ensemble find it, so penalizing uncertain forecasts
changes *which* actions get proposed near ties but not *what happens*. The
safety-relevant outcome ceiling on this roster was already reached by v2
(96/102 fault runs at zero exceedance, zero worse-than-control outcomes).

This is a real result, not a failure: it says the v2 roster cannot
distinguish uncertainty-aware ranking from point-estimate ranking. The
conditions where the disagreement penalty should pay — confident-but-wrong
single-model forecasts, near-tie candidate risks, shifted telemetry
distributions — are exactly the conditions the roster lacks.

## Next gates (not yet done)

1. **Adversarial/near-tie scenario family:** scenarios designed so the
   single model is confidently wrong (distribution-shifted telemetry,
   ambiguous candidate risks). The drift monitor's z-score tooling is the
   natural generator. Only on such a roster can the penalty demonstrate
   value. Success criteria must be frozen before that run, as here.
2. **Risk-coverage reporting:** with disagreement as a certified
   uncertainty proxy (r = 0.653), report advisory coverage vs realized
   exceedance as a selective-prediction curve.
3. Ensemble deployment cost: 5x forward passes per candidate; fine for the
   simulator's cadence, but it must be stated if this pattern ever moves.

## Reproduction

    python run_ensemble_paired.py --smoke --output smoke-v3.json   # determinism guard
    python run_ensemble_paired.py --output paired-v3-ensemble-results.json

Inputs: `action-aware-mlp-v1.pt`, `ensemble/seed-2026081{9,2*}..pt` (hashes in
`preregistration-v3.json`), frozen v2 results for pairing. Torch required;
the sealed `pyproject.toml` is unchanged and the depot remains NumPy-only on
main.
