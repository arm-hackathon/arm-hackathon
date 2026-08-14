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
python scripts/run_habitat_v2_live_forecast_demo.py \
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
