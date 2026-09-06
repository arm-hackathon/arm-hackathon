# Issue #76 calibrated selective prediction — evidence card

Date: 2026-09-06
Issue: #76 (evaluation, development-only)
Frozen contract: `contracts/habitat_v2_bdm_v1_abstention_thresholds_v1.json`
(sha256 `dcdec9e0b247b8cedfc3d98cc96d6abdb0f4ee562cba3c2d4635d8204c1bdb65`,
status `FROZEN_BEFORE_CLOSED_LOOP_STUDY`)
Receipt: `out/issue76-abstention-v3/abstention-receipt.json`
Bound BDM-v1 receipt: `ba6f97ca6c1e7aec3b9543362713a1563c91d69b97ba0c4aedd1268ffea46c2a`
(the five seeds were retrained deterministically and their digests asserted
equal before calibration; no weight blobs are committed)

## What was fit

On the CALIBRATION partition only (12 causal groups / 1248 samples):
conformal additive offsets per horizon (4/16/32) and quantile (p10/p50/p90)
for the pooled seed-mean TCN quantile heads, and four abstention thresholds
as declared CALIBRATION statistics (max observed staleness, min observed mask
fraction, 95th percentile of cross-seed delta disagreement, 95th percentile of
corrected interval width). Abstention reasons are deterministic and
receipt-visible: `INVALID_INPUT` (fail closed), `STALE_OBSERVATIONS`,
`SEED_DISAGREEMENT`, `WIDE_INTERVAL`.

## Results

- Marginal interval coverage: delta targets 0.850 against nominal 0.800;
  trajectory coverage by horizon 0.831 / 0.829 / 0.829 (in-sample conformal
  guarantee, slightly conservative).
- Coverage by stratum: useful_opportunity 0.791, no_action 0.973,
  sensor_failure 0.977, actuator_failure 0.923, compound 1.000. The failure
  strata each carry a single causal group in CALIBRATION, so their coverage
  figures are descriptive only, not stratum-level guarantees.
- Abstention rate 7.93%: 63 `SEED_DISAGREEMENT`, 36 `WIDE_INTERVAL`, zero
  invalid or stale triggers on CALIBRATION (consistent with the 95th-percentile
  threshold definition). Risk-coverage curve and multipliers are in the receipt.
- Useful-opportunity recall 0.0 and abstained-useful rate 0.0 because the
  CALIBRATION partition contains no truly improving candidate (the same
  degenerate action-minus-hold structure as TRAIN/DEV); the near-total
  abstention guard therefore passes trivially (max allowed 0.5).
- Uncertainty is empirical calibration evidence only: no formal safety
  guarantee, no HMC authority change, propose-or-abstain only.

## Checklist to evidence

| Issue checklist item | Evidence |
| --- | --- |
| Calibration data disjoint from fitting, architecture selection, blind custody | CALIBRATION partition only; TRAIN fits and DEV model choice happened in #75; BLIND_FINAL absent (corpus rejects it fail-closed) |
| Coverage and reliability by horizon and stratum, not aggregate only | receipt `coverage.trajectory_by_horizon` and `coverage.delta_by_stratum`; table above |
| Invalid/non-finite input fails closed before proposal | `INVALID_INPUT` reason first in `abstention_reason`; `test_abstention_reason_priority_and_triggers` |
| OOD and disagreement probes trigger the declared abstention path | `STALE_OBSERVATIONS` / `SEED_DISAGREEMENT` / `WIDE_INTERVAL` reasons with unit tests; CALIBRATION trigger counts in receipt |
| Risk-versus-coverage and useful-opportunity curves; near-total abstention cannot count as success | `risk_coverage_curve` and `useful_opportunity_recall` in receipt; `abstention_guard` with declared 0.5 cap |
| Thresholds and code identities frozen before any later closed-loop study | frozen contract above with layer offsets, thresholds, receipt sha, and module source sha |
| Uncertainty described as empirical calibration evidence, not a formal safety guarantee | this card and the module docstring |

## Boundaries and non-claims

Development evidence only. The calibrated layer does not make the non-promoted
Issue #75 candidate promotable; it merely documents how its quantile heads
behave after conformal correction and when the policy abstains. No blind
access, no online recalibration, no model self-update, no deployment claim.

## Reproduction

```bash
uv run --locked --python 3.11 --extra dev python -m pytest -q tests/habitat_v2/test_issue76_abstention.py
uv run --locked --python 3.11 --extra dev python \
  scripts/calibrate_issue76_abstention.py --corpus out/bdm-v1-corpus-v1 \
  --bdm-receipt out/issue75-bdm-v2/bdm-v1-receipt.json \
  --output out/issue76-abstention-<new>
```
