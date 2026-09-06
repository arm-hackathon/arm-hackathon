# Issue #75 BDM-v1 causal TCN adviser — evidence card

Date: 2026-09-06
Issue: #75 (model, development-only)
Preregistration: `contracts/habitat_v2_bdm_v1_tcn_preregistration_v1.json`
(sha256 `789c41f2500e144fa18fa713e4943854caa012ecd2a30269090a73245c756b4a`,
status `FROZEN_BEFORE_FINAL_RUNS`, committed alone before any training code)
Receipt: `out/issue75-bdm-v2/bdm-v1-receipt.json`
(sha256 `ba6f97ca6c1e7aec3b9543362713a1563c91d69b97ba0c4aedd1268ffea46c2a`)

## What was built and run

A pure-NumPy causal dilated TCN adviser exactly as preregistered: 16x240
input assembled only from Issue #70 declared corpus fields (environment
channels, explicit mask and staleness, requested/achieved commands, gauges,
mode, retained dispositions, candidate encoding), 32 channels, kernel 3,
dilations 1/2/4/8 (receptive field 31), 632 heads (trajectory, trajectory
quantiles, delta, delta quantiles), 53,240 parameters under the 100,000 cap.
Manual backprop is validated by finite-difference tests; causality, explicit
missingness, action identity, and forbidden-field exclusion are test-enforced;
the module imports neither the HMC nor the plant stepper.

Five preregistered seeds were fit on TRAIN corpus samples only
(`ecec280680a0ba3c984c7182dc288241014b46b3c0687f8bcc91e3322510c219`), with
DEV used only for early stopping and evaluation. All five seeds stopped early
(6-8 epochs, best epochs 1-3): the DEV delta-MSE never improved materially
after the first epochs, consistent with the degenerate action-minus-hold
labels on this roster. The three Issue #74 baselines were refit inside the
study and their weight digests verified equal to the bound values before any
paired comparison.

## Results: the promotion gate FAILED; the negative is retained

Pooled (seed-mean) BDM-v1 versus the refit `action_conditioned_ridge` on DEV
at causal-group level:

| Metric | BDM-v1 pooled | linear baseline | paired difference (95% CI) |
| --- | --- | --- | --- |
| action ranking | 0.1776 | 0.2839 | -0.1063 [-0.2761, -0.0311] (worse) |
| finite-catalogue regret | 0.313736 | 0.000000 | +0.3137 [0.0000, 0.8546] (not beneficial) |
| safety-exposure error | (see receipt) | (see receipt) | CI not beneficial |
| useful-action coverage (recall) | 0.0 | 0.0 | CI not beneficial |
| action-value error | 9.364213 | 5.131881 | worse |
| trajectory MAE (secondary) | 11183.752544 | 263.267153 | far worse |

Per-seed trajectory MAE: 11317.79, 11042.17, 11322.02, 11319.80, 11322.89 (all
reported in the receipt). Per-seed ranking correlations: 0.2465, 0.1236,
-0.0039, 0.1344, 0.2715 (all
reported; none exceeds the baseline's 0.2839). No seed or the pooled model
predicts any improving action (precision/recall 0.0), matching the baselines'
null on coverage while being strictly worse on ranking, regret, and value
error.

Promotion gate (preregistered): paired CIs must exclude zero in the
beneficial direction for ranking, regret, exposure error, and coverage.
Result: `promotion_gate_passed: false` on every component. **BDM-v1 does not
beat Issue #74's frozen linear bar; the model is not promoted.** Per the
preregistration and the issue checklist the negative result is retained and
published unchanged. The temporal-convolutional architecture, as
preregistered, adds no decision value over linear action-conditioned
dynamics on these families; any future BDM line must change the declared
protocol (features, labels, or operating regime), not re-run this one.

## Checklist to evidence

| Issue checklist item | Evidence |
| --- | --- |
| Budget, receptive field, horizons, loss, seeds, stopping frozen before final runs | preregistration committed alone at `f9d1d5a` with `frozen_before_final_runs: true`; runner fails closed otherwise (`test_runner_refuses_unfrozen_preregistration`); parameter count asserted equal to prereg |
| Tests prove causal windows, explicit missingness, action identity, forbidden-field exclusion | `tests/habitat_v2/test_issue75_bdm.py`: no-lookahead cache test, mask-flip sensitivity, candidate-encoding sensitivity, declared-field subset check, finite-difference backprop |
| Every declared seed reported | receipt `seeds` lists all five with digests, epochs run, best epoch; per-seed metrics in `evaluations` |
| Primary verdict uses ranking/regret/action-value/safety exposure/coverage; MAE secondary | receipt `evaluations` and `tables`; this card's table; MAE reported only as secondary |
| If BDM-v1 cannot beat #74 the negative is retained and the model is not promoted | gate false; this card states it plainly; no promotion path exists in code |
| Model proposes or abstains only; HMC sole authority | module has no HMC/plant imports (test-enforced); runner never commands or steps the plant |
| No blind population opened | TRAIN fits, DEV evaluation only; BLIND_FINAL absent from corpus partitions used |

## Boundaries and non-claims

Development evidence only. No qualification, deployment, hardware, or
real-habitat claim; no ONNX/quantisation/Arm work; no HMC authority change.
The negative result concerns this preregistered architecture, loss, and
corpus on these families; it is not a general statement about temporal
convolutional networks.

## Reproduction

```bash
uv run --locked --python 3.11 --extra dev python -m pytest -q tests/habitat_v2/test_issue75_bdm.py
uv run --locked --python 3.11 --extra dev python \
  scripts/train_issue75_bdm.py --corpus out/bdm-v1-corpus-v1 \
  --baselines-receipt out/issue74-baselines-v1/baselines-receipt.json \
  --output out/issue75-bdm-<new>
```
