# Habitat V2 forecast demo artifact

This directory contains the compact action-aware ridge model used by the local
Habitat V2 live forecast demonstration.

## Scope

- Release tier: `DEMO_ONLY_PERMANENTLY_EXCLUDED`
- Qualification evidence: `false`
- Actuator authority: `false`
- Training data: simulator-generated and permanently excluded from D2
  qualification
- Runtime role: forecast candidate-action outcomes only
- Controller role: none; deterministic HMC remains the sole command and
  actuator authority

The model is evaluated at completed simulator step 16 from issued operational
history covering steps 13 through 16. It forecasts eight future simulator
states for each catalogue action before steps 17 through 24 are produced. The
operator-selected catalogue action is separately proposed to HMC. Model output
never selects, modifies, approves, or executes that action.

## Bound evidence

- `action-aware-ridge.npz`: saved model artifact
- `training-report.json`: bounded outer-holdout comparison
- `training-receipt.json`: dataset, model, report, and authority identities

## Trained development MLP artifact (2026-08-18)

- `action-aware-mlp-v1.npz`: the action-aware MLP from the verified Historical
  V2 development training run `full-v1-20260818-a` (23,400 examples, 60
  clusters; outer holdout 17 clusters / 6,630 examples; held-out normalized
  MAE 0.1146 vs 0.2880 action-blind). Converted from the torch checkpoint to
  pure NumPy for dependency-free inference; conversion cross-checked against
  torch (max abs diff 2.1e-07 on a fixed-seed forward pass).
- Release tier: `DEVELOPMENT_EVIDENCE_ONLY`; no actuator authority; no
  availability masks (missing-sensor robustness unproven); not D2
  qualification evidence.
- Closed-loop evidence: merged PR #40 (238-run paired campaign,
  pre-registered scoring, HMC sole authority throughout).

SHA-256 for `action-aware-mlp-v1.npz`:

```text
a80628fb298ae2f68fb600ecc70922dfddb39e2560207bbd13463e2d4596ecdd
```

Source checkpoint SHA-256 (local training evidence, not committed):

```text
873cb77bb82a06b4c862a13275b55133c3ef26c969d3055a799c80dcd98854a6
```

(The full hash set is recorded in the training run's `artifact_hashes.json`;
the committed npz is the judge-runnable artifact.)

The `model-v1/...` paths inside the preserved training report and receipt are
historical paths from the original local training output. This committed demo
package deliberately copies only the selected action-aware ridge to
`action-aware-ridge.npz`; the action-blind ridge and neural residual MLP are not
included here. The selected copy is bound by the same byte length and SHA-256
recorded in those preserved files.

SHA-256 for `action-aware-ridge.npz`:

```text
a6e4ef34fc837bb6539a84e20d015bbd7bbfe4e9fd5a6fc74e3f0217bd978d9a
```

The training report identifies this ridge as the lowest-NMAE model in the
bounded demo comparison. That result is not a production, physical-habitat,
D2-qualification, deployment, or learned-control claim.

## Run and validate on Arm64 Linux

Requirements: native Arm64 Linux, Python 3.11, and Git. From the repository
root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . \
  "numpy==2.4.6" \
  "pytest==9.1.1" \
  "ruff==0.12.12"
export PYTHONPATH=src
python -m pytest -q tests/habitat_v2/test_forecast_*.py
python scripts/run_habitat_v2_live_forecast.py \
  --output out/habitat-v2-live-forecast-demo/manual-arm64-run
python scripts/verify_habitat_v2_live_forecast_demo.py \
  --report out/habitat-v2-live-forecast-demo/manual-arm64-run
```

The output directory must not already exist. Open its `index.html` to inspect
the recorded browser report. This HTML replays outputs from the preceding
Python inference run; it does not run the model itself.

For reproducible hosted evidence, the workflow at
`.github/workflows/habitat-v2-live-forecast-arm64.yml` runs the same checks on
GitHub's native `ubuntu-24.04-arm` runner, rejects any non-Arm64 machine, binds
the integration commit, and uploads the environment receipt plus hash
manifest. Do not describe a run as native Arm64 unless that receipt records
`native_arm64: true` and `platform_machine: aarch64` or `arm64`.
