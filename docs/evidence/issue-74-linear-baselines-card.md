# Issue #74 linear baselines — evidence card

Date: 2026-09-06
Issue: #74 (model, development-only)
Receipt: `out/issue74-baselines-v1/baselines-receipt.json`
(sha256 `f0f3ac0a4850a5f7a580fec8c1f73c42c156988d9d68ffe1ab1519e2126d6e8f`)
Corpus binding: `ecec280680a0ba3c984c7182dc288241014b46b3c0687f8bcc91e3322510c219`
(fit partition TRAIN, 4160 samples; evaluation partition DEV, 1664 samples)

## What was fit

Three baselines, closed-form and deterministic, on TRAIN only:

- `action_agnostic_ridge` (digest `9c5879d63285…`): ridge over the declared
  45-dim observable context recipe;
- `action_conditioned_ridge` (digest `551c08940877…`): context plus the
  31-dim candidate action encoding (one-hot + command vector);
- `controlled_linear_state_space` (digest `879a1f6de07b…`): per-horizon
  linear transition maps from the 43-dim checkpoint observed state plus the
  27-dim candidate command, with a linear delta head on the same controlled
  input.

Every recipe input is an Issue #70 declared corpus field; the module checks
`RECIPE_INPUT_FIELDS ⊆ FEATURE_FIELD_NAMES` at import and tests re-assert it.
Labels are the corpus true-plant trajectory rows (horizons 4/16/32) and the
five action-minus-hold decision targets. Nothing in the module imports the
HMC or plant stepper; baselines predict only (test-enforced).

## Results on DEV (group-level, causal groups)

| Baseline | mean ranking corr | mean regret | mean exposure error | useful precision / recall |
| --- | --- | --- | --- | --- |
| action_agnostic_ridge | 0.0000 | 0.211726 | 1.269410 | 0.0 / 0.0 |
| action_conditioned_ridge | 0.2839 | 0.000000 | 1.275192 | 0.0 / 0.0 |
| controlled_linear_state_space | 0.2839 | 0.000000 | 1.277485 | 0.0 / 0.0 |

Learning curve (TRAIN fraction → DEV mean regret for the action-conditioned
ridge): 0.25 → 0.0, 0.5 → 0.0, 0.75 → 0.0, 1.0 → 0.0.

Interpretation, stated plainly: on this roster the linear baselines extract
almost no decision signal. True action-minus-hold safety-exposure deltas are
zero for most candidate/decision pairs (the plant stays in bounds under hold
and under every catalogue action), so finite-catalogue regret is zero for
action-aware models and only the action-agnostic model ever "chooses wrong";
ranking correlation is weak (0.28) and no baseline ever predicts a useful
(precision/recall 0.0). The bar BDM-v1 must beat in Issue #75 is therefore
close to the null: meaningful improvement must show up as admitted useful
interventions with safety non-inferiority, not as regression metrics on
degenerate deltas. All negative results are retained.

## Checklist to evidence

| Issue checklist item | Evidence |
| --- | --- |
| Features in the accepted causal input manifest; no hidden truth or future labels as runtime features | import-time `RECIPE_INPUT_FIELDS ⊆ FEATURE_FIELD_NAMES` check; `test_recipe_reads_only_declared_fields`; recipe uses only last-window observables, gauges, achieved feedback, mode, candidate encoding |
| Fits reproducible from declared code, data, split, scaler, seed identities | receipt binds corpus digest, recipe sha256, ridge lambda, seed `issue74-baselines-v1`, split counts, and per-baseline weight digests; `test_ridge_fit_is_deterministic…`, `test_state_space_fit_is_deterministic` |
| Same label definitions and independent group unit as BDM-v1 | corpus labels (Issue #72 72c) and `bdm_v1_evaluation` causal-group statistics |
| Numerical stability and non-finite failure paths tested | `test_non_finite_inputs_fail_closed`, prediction finiteness guard in `predict_sample`, ridge gram regularization |
| All development metrics and negative results reported | table above plus receipt `evaluations`/`tables`/`learning_curve_mean_regret`; null precision/recall stated |
| Baseline never issues a final command or steps the plant | `test_baselines_never_touch_the_plant` (no HMC/physics imports in module); module is pure prediction mathematics |

## Boundaries and non-claims

Development evidence only: no neural candidate, no blind evaluation, no
actuator authority, no online learning, no ONNX/quantisation/release work.
Runtime figures (fit 1.83 s, evaluate 0.86 s) are local development-machine
measurements, not hardware claims.

## Reproduction

```bash
uv run --locked --python 3.11 --extra dev python -m pytest -q tests/habitat_v2/test_issue74_baselines.py
uv run --locked --python 3.11 --extra dev python \
  scripts/fit_issue74_baselines.py --corpus out/bdm-v1-corpus-v1 \
  --output out/issue74-baselines-<new>
```
