# Issue #54 Design Note — Model Distillation Study

## Issue

https://github.com/arm-hackathon/arm-hackathon/issues/54 — *Distillation study: how small can the model get and still be safe?*

## Design date

2026-08-24

## Branch

`research/issue-54-model-distillation` (created from `origin/main@6ae8641`)

## Preregistration

`contracts/habitat_v2_forecast_issue_54_preregistration_v1.json` (byte-frozen on plan publish)
SHA-256: `E16BEFB588A43F131128056932BBFE5CAA707C87309A828A33C91E1C412D5246`

## What this study is

Train progressively smaller "student" models that learn by **copying the frozen teachers' predictions** — not the raw data — and measure how three behavioural properties degrade as the parameter count drops:

1. **Prediction accuracy** (NMAE vs ground truth, ratio to teacher).
2. **Action-ranking agreement** (does the student rank the four catalogue candidates the same way the teacher does?).
3. **Safety-margin closeness** (how close is the student's `safety_exposure` to the teacher's?).

The output is an honest "how small is safe" curve and a written capability/limitation card.

## Teachers (frozen)

| Teacher | Artifact | Params | Window | Release tier |
| --- | --- | --- | --- | --- |
| MLP | `action-aware-mlp-v1.npz` (`a80628fb…`) | ~2.1M | 16-step | `DEVELOPMENT_EVIDENCE_ONLY` |
| Ridge | `action-aware-ridge.npz` (`0de4b5cd…`) | ~3.58M | 4-step | `DEMO_ONLY_PERMANENTLY_EXCLUDED` |

Both are frozen, provenance-bound, `actuator_authority=False`. They are never modified.

## Students

Pure-NumPy MLPs (GELU, Adam, MSE distillation loss) adapted from `detector.py:train_temporal_mlp_detector` (lines 352–517), trained to match teacher predictions:

| ID | Architecture (MLP teacher) | Approx params |
| --- | --- | --- |
| `sanity-2.1m` | 3132→512→512→256→408 | ~2.1M |
| `medium-500k` | 3132→140→408 | ~500K |
| `small-100k` | 3132→28→408 | ~100K |
| `tiny-25k` | 3132→7→408 | ~25K |
| `linear` | ridge regression (teacher predictions as targets) | varies |

Each size is trained for **both** teachers. The `sanity-2.1m` student matches the MLP teacher's architecture — a distillation sanity check.

## Corpora (three options, all tried, best recorded)

1. **Fresh pipeline corpus** — generate diverse scenario families (mode/fault/seed variants), run each teacher's forecast demo per family, collect `(input, teacher_pred, ground_truth)` triples.
2. **Committed 288-example ridge corpus** — run both teachers on the existing 288-sample corpus to produce teacher predictions.
3. **Synthetic varied-seed corpus** — vary the development scenario's `random_seed` and initial conditions to generate diverse inputs.

All corpora are family-disjoint split into TRAIN / VALIDATION / FINAL (SHA-256 hash split, 60/20/20).

## Safety boundaries

- Students are `DEVELOPMENT_EVIDENCE_ONLY`, `actuator_authority=False`.
- HMC remains sole actuator authority; students never drive actuation.
- No mutation of frozen teacher artifacts or their provenance.
- No final-suite/validation data access; all corpora derived from the development scenario and deterministic variants.
- Preregistration frozen before any experiments.

## What this study does NOT prove

- It does not qualify any student model for deployment.
- It does not change HMC authority or safety logic.
- It does not claim that distillation agreement implies safety in any real system.
- It is simulator evidence only, not hardware validation.
