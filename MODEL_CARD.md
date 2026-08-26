# Model Card — `action_aware_mlp_v1`

Advisory atmosphere forecaster for the Habitat V2 simulation. This model
**proposes; it never commands**. The deterministic Habitat Management
Computer (HMC) is the sole actuator authority in every code path that uses
this model.

## Identity

| Field | Value |
| --- | --- |
| Artifact (NumPy, main branch) | `artifacts/demo-only/habitat-v2-forecast/action-aware-mlp-v1.npz` |
| Artifact SHA-256 | `a80628fb298ae2f68fb600ecc70922dfddb39e2560207bbd13463e2d4596ecdd` |
| Source checkpoint (Torch) | `experiments/closed-loop-advisory-20260818/action-aware-mlp-v1.pt` (qualification branch) |
| Training run | `full-v1-20260818-a` |
| Training seed | `20260818` |

## Intended use

- Forecast the next 8 steps of per-zone atmosphere (51 targets: temperature,
  CO2, O2, humidity and related fields across 8 zones) from the last 16 steps
  of operational telemetry plus one candidate action.
- Enable counterfactual action comparison inside the closed-loop advisory
  harness: the adviser scores every catalogue action's predicted future and
  proposes the lowest-risk one to HMC.

**Out of scope:** hardware control, production building/spacecraft control,
deployment claims of any kind. This is a research simulation model.

## Architecture and training

- MLP: input 3,132 → 512 → 512 → 256 → output 408 (= 8 steps × 51 targets),
  GELU activations.
- Optimizer: AdamW, learning rate 1e-3, weight decay 1e-4, batch size 128,
  maximum 80 epochs, early-stopping patience 12.
- Feature/target normalization statistics are stored inside the artifact
  (`feature_mean/std`, `target_mean/std`) and verified at load.

## Training data

Historical V2 archive — see `CORPUS_DATASHEET.md`. 4,680 simulation packets,
23,400 windowed examples, 60 scenario clusters. Manifest SHA-256
`7c35a2da3a9f902c2994b8b29332f8fafed40b81c49204c3546a75d2b3f76659`.

## Evaluation

- **Held-out generalization:** 17 outer-held-out clusters / 6,630 examples
  (clusters the model never saw in training).
  Normalized MAE **0.1146**; action-blind twin **0.2880** — conditioning on
  the candidate action reduces normalized error by **60.2%**.
  Raw RMSE 34.56 (action-aware) vs 55.67 (action-blind).
- **Closed-loop paired campaign** (238 runs, pre-registered; merged PR #40):
  102 fault pairs — **78 safer / 24 equal / 0 worse** on pre-registered
  threshold-exceedance metrics; 72 advised fault runs finished with zero
  exceedance. HMC overrode 81 of 793 proposals.
- **Demo scenario:** canonical HMC integrated exceedance 19.94 (29 steps
  above the CO2 warning line); model-advised 0.0.

## Limitations and failure modes

- **Complete telemetry only for this artifact.** The model was trained without
  availability masks; it must not forecast from missing/broken sensors. The
  advisory harness enforces abstention in that case (verified guard, PR #41).
  A distinct Issue #53 dropout-aware successor now exists and is qualified only
  for its frozen independent-dropout forecast lane; its evidence does not
  change this artifact's input contract or grant either model plant authority.
  See `docs/evidence/issue-53-dropout-card.md`.
- **Point predictions.** No calibrated uncertainty; HMC treats every
  forecast identically regardless of input novelty.
- **Greedy one-step scoring** over a 4-action catalogue; not a general
  planner.
- **Resource cost.** Advised runs consumed a median +757 Wh battery,
  +1.97 mol O2, +6.04 mol sorbent relative to control; all runs stayed
  above resource floors, but scarcity scenarios are an open limitation.
- **Simulation-only.** All metrics are against the Habitat V2 simulator,
  not physical hardware.
- **Development evidence only** — not qualification, certification, or
  deployment readiness.
