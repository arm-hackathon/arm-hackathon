# Model Card — `action_aware_mlp_v1`

Advisory atmosphere forecaster for the Habitat V2 simulation. The model
supplies forecasts; the historical adviser used them to propose actions. The
model **never commands**. The deterministic Habitat Management Computer (HMC)
is the sole actuator authority in every code path that uses this model.

## Identity

| Field | Value |
| --- | --- |
| Artifact (NumPy, main branch) | `artifacts/demo-only/habitat-v2-forecast/action-aware-mlp-v1.npz` |
| Artifact SHA-256 | `a80628fb298ae2f68fb600ecc70922dfddb39e2560207bbd13463e2d4596ecdd` |
| Source checkpoint (Torch) | Historical archive; SHA-256 `873cb77bb82a06b4c862a13275b55133c3ef26c969d3055a799c80dcd98854a6`; see the [evidence index](docs/evidence/closed-loop-advisory-historical-index.md#exact-archived-file-manifest) |
| Recorded training run | `full-v1-20260818-a` |
| Recorded training seed | `20260818` |

## Intended use

- Forecast the next 8 steps of 51 targets: 48 per-zone values (6 channels
  across 8 zones) plus 3 global resource fractions, from the last 16 steps of
  operational telemetry plus one candidate action.
- Enable counterfactual action comparison inside the closed-loop advisory
  harness: the adviser scores every catalogue action's predicted future and
  proposes the lowest-risk one to HMC.

**Out of scope:** hardware control, production building/spacecraft control,
deployment claims of any kind. This is a research simulation model.

## Architecture and training

- MLP: input 3,132 → 512 → 512 → 256 → output 408 (= 8 steps × 51 targets),
  GELU activations.
- Recorded training metadata: AdamW, learning rate 1e-3, weight decay 1e-4,
  batch size 128, maximum 80 epochs, early-stopping patience 12.
- Feature/target normalization statistics are stored inside the artifact
  (`feature_mean/std`, `target_mean/std`) and verified at load.

## Training data

Historical materials report 4,680 simulation packets, 23,400 windowed examples,
and 60 scenario clusters; see `CORPUS_DATASHEET.md`. Recorded manifest SHA-256:
`7c35a2da3a9f902c2994b8b29332f8fafed40b81c49204c3546a75d2b3f76659`.

## Evaluation

- **Historically reported held-out generalization:** 17 outer-held-out clusters
  / 6,630 examples, described as excluded from training. Normalized MAE
  **0.1146**; action-blind twin **0.2880** — a reported 60.2% error reduction
  from action conditioning. Reported raw RMSE was 34.56 (action-aware) versus
  55.67 (action-blind).
- **Recorded closed-loop V2 campaign** (238 total runs = 119 paired scenarios;
  merged PR #40): across 102 fault pairs, **78 safer / 24 equal / 0 worse** on
  the declared threshold-exceedance metrics. **96/102** advised fault runs
  finished with zero exceedance; 72 of the 78 improved pairs were driven to
  zero. HMC modified or replaced 81 of 793 proposals. The archived plan labels
  itself frozen before outcomes, but plan and result first entered Git
  together, so repository history does not independently establish that
  chronology.
- **Recorded historical V1 demo pair:** canonical HMC integrated threshold
  exceedance 19.94 across 29 steps; model-advised 0.0. A later checked-in replay
  artifact preserves the displayed step sequence.

The training and evaluation statements above are recorded historical
development results. The [historical evidence
index](docs/evidence/closed-loop-advisory-historical-index.md) identifies the
archived bytes, static consistency checks, and missing material that prevents
independent checking of every metric or a fresh-clone rerun of the complete
campaign from current `main`.

## Limitations and failure modes

- **Complete telemetry only for this artifact.** The model was trained without
  availability masks; it must not forecast from missing/broken sensors. The
  historical harness implements complete-telemetry abstention, as identified
  in the [archived authority and availability
  surfaces](docs/evidence/closed-loop-advisory-historical-index.md#archived-authority-and-availability-surfaces).
  The retained V3 result reports zero unavailable-input abstentions and does
  not itself test the degraded-input path. A distinct Issue #53 dropout-aware
  successor now exists and is qualified only for its frozen independent-dropout
  forecast lane; its evidence does not change this artifact's input contract
  or grant either model plant authority. See
  `docs/evidence/issue-53-dropout-card.md`.
- **Point predictions.** No calibrated uncertainty; the historical adviser did
  not use a calibrated novelty or uncertainty signal when selecting proposals.
- **Greedy one-step scoring** over a 4-action catalogue; not a general
  planner.
- **Resource cost.** Across the 102 fault pairs, median advised-minus-control
  deltas were +757 Wh battery, +1.97 mol O2, and +6.04 mol sorbent. The
  historical report states that all runs stayed above resource floors, but the
  raw V2 step records are absent; scarcity scenarios are an open limitation.
- **Simulation-only.** All metrics are against the Habitat V2 simulator,
  not physical hardware.
- **Development evidence only** — not qualification, certification, or
  deployment readiness.
