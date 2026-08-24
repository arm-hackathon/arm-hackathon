# Issue #54 Full Implementation, Training, and Evaluation Plan

## 1. Document control

- Issue: https://github.com/arm-hackathon/arm-hackathon/issues/54 — *Distillation study: how small can the model get and still be safe?*
- Design date: 2026-08-24
- Branch for this plan: `research/issue-54-model-distillation` (created from `origin/main@6ae8641`)
- Short note: `docs/plans/2026-08-24-issue-54-model-distillation-design.md`
- Machine-readable preregistration: `contracts/habitat_v2_forecast_issue_54_preregistration_v1.json` (byte-frozen on plan publish)
- Preregistration SHA-256: `E16BEFB588A43F131128056932BBFE5CAA707C87309A828A33C91E1C412D5246`
- This-lane status: `RESEARCH_STUDY_DEV_EVIDENCE_ONLY`

This appendix is normative. The short note is the plain-English entry point. The qualification runbook is in `docs/evidence/issue-54-distillation-card.md` and `docs/evidence/issue-54-measurements.md`.

## 2. Scope and context

Issue #54 is a **research study**, not a qualification or deployment. It asks: as a forecast model is compressed via distillation, how do prediction accuracy, action-ranking agreement, and safety-margin closeness degrade? The frozen teachers (MLP ~2.1M and ridge ~3.58M) are never modified. Students learn by copying teacher predictions — not ground truth — and are evaluated against ground truth for the accuracy metric and against the teacher for ranking and safety.

**Frozen evidence rule:** The Issue #52/#53 models, their preregistrations, their artifacts, and the HMC binding (`contracts/habitat_v2_forecast_hmc_binding_v2.json`, `aa37ae63…`) remain byte-identical. This lane adds **new** distillation corpora, **new** student artifacts, and **new** evidence. HMC remains in charge of every action, always.

## 3. Goals and non-goals

Goals (exact acceptance language from #54):

1. Train student models of sizes ~2.1M, 500K, 100K, 25K that copy the current model's predictions.
2. Measure how the loss of internal numbers affects: prediction accuracy, action-ranking agreement, safety-margin closeness.
3. Produce a "how small is safe" curve.
4. Publish an honest written record of what the model cannot do alongside what it can.

Non-goals:

1. No HMC authority change, no new emergency template, no capability minting by learned code.
2. No deployment, no qualification claim, no "closed-issue" claim through this plan.
3. No mutation of frozen teacher artifacts or their provenance.
4. No online or production reinforcement learning.
5. No final-suite/validation data access.
6. No safety-limit, reserve-limit, emergency-threshold, or `HMCContract` change.

## 4. Teacher models (frozen)

### MLP teacher

- Artifact: `artifacts/demo-only/habitat-v2-forecast/action-aware-mlp-v1.npz`
- SHA-256: `a80628fb298ae2f68fb600ecc70922dfddb39e2560207bbd13463e2d4596ecdd`
- Architecture: 3132→512→512→256→408, GELU, float32
- Params: ~2.1M
- Window: 16 steps (16×194 + 27 + 1 = 3132 features)
- Horizon: 8 steps, 51 targets
- Release tier: `DEVELOPMENT_EVIDENCE_ONLY`
- Inference: `src/aeolus/habitat_v2/forecast/live_mlp_demo.py:NumpyMlpPredictor`

### Ridge teacher

- Artifact: `artifacts/demo-only/habitat-v2-forecast/action-aware-ridge.npz`
- SHA-256: `0de4b5cdb6ec2b47be260a06f924d8eb00f1def16d5ae668b3ab5191251f29df`
- Architecture: 8767→408, ridge regression, float64
- Params: ~3.58M
- Window: 4 steps (4×(194+167×5+4+4+287×4) + 27 = 8767 features)
- Horizon: 8 steps, 51 targets
- Release tier: `DEMO_ONLY_PERMANENTLY_EXCLUDED`
- Inference: `src/aeolus/habitat_v2/forecast/baselines.py:DirectRidgeModel`

## 5. Student architectures

### Neural students (pure-NumPy MLP, GELU, Adam, MSE distillation loss)

Adapted from `src/aeolus/detector.py:train_temporal_mlp_detector` (lines 352–517). Key changes:

- GELU activation (exact erf, matching `live_mlp_demo.py:_gelu_exact`) instead of ReLU.
- MSE loss (`||student_pred - teacher_pred||²`) instead of cross-entropy.
- Linear output (no softmax) — regression to `(8, 51)` targets.
- Variable hidden layer widths per student size.

| ID | Hidden layers (MLP teacher, input 3132) | Hidden layers (ridge teacher, input 8767) | Output | Approx params |
| --- | --- | --- | --- | --- |
| `sanity-2.1m` | [512, 512, 256] | [200] | 408 | ~2.1M / ~1.83M |
| `medium-500k` | [140] | [50] | 408 | ~500K / ~459K |
| `small-100k` | [28] | [10] | 408 | ~100K / ~92K |
| `tiny-25k` | [7] | [2] | 408 | ~25K / ~18K |

### Linear student

Ridge regression (`fit_direct_ridge` pattern from `baselines.py`) with teacher predictions as distillation targets. This is the extreme small / non-neural case.

### Training hyperparameters (frozen)

- Optimizer: Adam (β₁=0.9, β₂=0.999, ε=1e-8)
- Learning rate: 0.001
- L2 penalty: 1e-4
- Epochs: 200 (validation selection every 5 epochs)
- Seeds: 540054, 540055, 540056
- Initialization: `numpy.default_rng(seed)`, He-normal scaling
- Distillation loss: MSE between student prediction and teacher prediction (NOT ground truth)
- Normalizers: fit on TRAIN only

## 6. Distillation corpora (three options, all tried)

### Option A — Fresh pipeline corpus

Generate diverse scenario families (mode/fault/seed variants, mirroring the Issue 52/53 collector pattern), run each teacher's forecast demo per family at the anchor step, collect `(input, teacher_pred, ground_truth)` triples. Each family produces one decision with 4 catalogue candidates.

### Option B — Committed 288-example ridge corpus

Run both teachers on the existing 288-sample corpus (already committed) to produce teacher predictions. Use these as distillation targets.

### Option C — Synthetic varied-seed corpus

Vary the development scenario's `sensor_model.random_seed` and initial conditions deterministically to generate diverse inputs. Run each teacher on each variant.

### Split

All corpora are family-disjoint split into TRAIN / VALIDATION / FINAL using:
- `SHA256("issue54-split-v1|" + family_id)`
- Proportions: 60% TRAIN, 20% VALIDATION, 20% FINAL
- Largest-remainder allocation
- Whole-family isolation preserved

## 7. Evaluation metrics

### 7.1 Prediction accuracy degradation (primary)

- For each FINAL sample: `NMAE(model vs ground_truth)` using the same scale as `evaluation.py` (p95-p05 of TRAIN targets).
- Family-level aggregate: mean over candidates within a family.
- Ratio: `NMAE_student / NMAE_teacher` per family, then equal-weight family mean.
- Bootstrap 95% CI (10k resamples, SHA-256 counter sampler).

### 7.2 Action-ranking agreement (co-primary)

For each FINAL decision (4 candidates), both teacher and student produce 4 candidate trajectories. Rank with `forecast_issue52.py:rank_candidates` using `score_trajectory`:

- **Top-1 agreement**: fraction of decisions where the student's top-ranked candidate equals the teacher's top-ranked candidate.
- **Kendall tau-b**: mean Kendall tau-b between student and teacher full rankings over all decisions.

### 7.3 Safety-margin closeness (co-primary)

For each candidate per decision, compute `safety_exposure` from `score_trajectory` (line 1694: how much predicted bounds cross safety bounds). The safety-margin closeness is the mean absolute difference of `safety_exposure` between student and teacher across all candidates and decisions.

### 7.4 Hard gates

- `authority_violation_count`: 0
- `provenance_or_split_violation_count`: 0
- `replay_failure_count`: 0

## 8. Statistics

- Bootstrap: 10,000 resamples, SHA-256 counter sampler, equal-tailed percentile CI at 2.5% and 97.5%.
- Resampling unit: semantic scenario family.
- Point estimator: ratio of equal-weight family aggregate means.
- Seed: 540054.

## 9. Required tests

- Distillation trainer determinism (same seed → identical weights).
- Distillation loss decreases monotonically on TRAIN.
- Student prediction shape and dtype contract (float32, (8, 51)).
- Action-ranking agreement metric correctness on a synthetic example.
- Safety-exposure difference metric correctness on a synthetic example.
- Corpus validation: content-addressed, family-disjoint split, no leakage.
- Preregistration binding: digest matches frozen value.
- Teacher inference: MLP and ridge produce the same predictions as the live demo.
- Authority invariant: all students have `actuator_authority=False`.

## 10. Phased execution

**Phase 0 — Contracts+plan:** freeze preregistration, publish design note and normative plan.

**Phase 1 — Implementation:** `src/aeolus/habitat_v2/forecast_issue54_distillation.py` — regression MLP trainer (GELU+MSE+Adam), student architecture builder, distillation corpus builder, evaluator (NMAE/ranking/safety). Tests.

**Phase 2 — Corpus generation:** `scripts/collect_issue54_distillation_dataset.py` generates all 3 corpus variants for both teachers. Content-addressed, validated.

**Phase 3 — Student training:** `scripts/qualify_issue54_distillation.py` trains all 5 sizes × 2 teachers × 3 corpora = 30 students. Selects on VALIDATION.

**Phase 4 — Evaluation + curve:** Evaluate FINAL for all students. Produce "how small is safe" curve data.

**Phase 5 — Honest write-up:** `docs/evidence/issue-54-distillation-card.md` (capability/limitation card) and `docs/evidence/issue-54-measurements.md` (measurements table).

**Phase 6 — Tests + CI:** Full test suite, Ruff, compileall, lock check.

**Phase 7 — Commit:** Commit all changes. No push, no PR.

## 11. Stop conditions

Stop and publish a negative result if:
- Corpus generation exceeds family/transition caps.
- Distillation loss does not decrease for any student size.
- Any authority/replay/provenance gate fails.
- Non-finite metrics persist after debugging.

Precedence: authority, safety, leakage/provenance, replay/determinism, forecast quality, then ranking agreement.
