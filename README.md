# Project AEOLUS — `alex/ai-2` experimental branch

> **A**irflow and **E**nvironmental **O**bservation **L**aboratory for
> **U**ser-defined **S**cenarios
>
> Deterministic habitat simulation, realistically imperfect synthetic
> telemetry, and family-held-out fault prediction.

> [!IMPORTANT]
> `alex/ai-2` branches from `main` to make the simulation more complex and test
> prediction models. It preserves AEOLUS contracts and selectively adapts useful
> ideas from legacy `alex/ai`; it does not merge that branch wholesale.

AEOLUS is a local research simulation in abstract CO2 and airflow units. It
does not model real spacecraft equipment, life-support limits, or a general
fluid system. It must not control real environmental or safety-critical
equipment.

## Implemented experiment

The branch retains deterministic replay, explicit fault profiles, the
standalone visualiser, corpus-v2 integrity checks, topology hashes, and the
frozen `model_input_v1 float32[24]` contract from `main`.

Protocol v3 adds a strict development-to-final decision path:

- closed-schema v9 measurement noise, bias and drift, including controller-facing
  CO2 readout effects;
- 360 training and 120 validation scenario families used for every learned
  candidate choice and rule-grid calibration;
- a separately generated 180-family final suite, never used for selection;
- softmax and compact `temporal_summary_v1` MLP candidates, a 216-point
  validation-only rule calibration, strict JSON loading and FP32 ONNX parity;
- a hash-bound policy whose candidate receipt, calibration receipt, validation
  comparison and outcome are replayed before final evaluation; and
- a final report that cannot overwrite an earlier report or reselect a method.

Both learned candidates consume exact `float32[10,24]` corpus-v2 windows. The
selected MLP summarizes last value, mean, population standard deviation, slope
and maximum absolute first difference for 24 channels and three safe
residual/request ratios, then applies a `135 → 16 ReLU → 4 softmax` network.

Predictions contain confidence and probabilities for these exact classes:

```text
nominal
gradual_primary_fan_degradation
blocked_path
frozen_sensor
```

Fault truth, schedules, hidden effectiveness, connection health, seeds and
measurement state never enter telemetry or model features.

## Current measured result

The frozen v3 final result is negative: the calibrated rule baseline remains
preferred over the validation-selected temporal MLP.

| Final-suite evidence | Temporal MLP | Calibrated rules |
|---|---:|---:|
| Macro-F1 | 0.5754744477098027 | 0.642588422763726 |
| Nominal false-alarm rate | 38.5698% | 0.5631% |
| Overall median detection latency | 9 ticks | 10 ticks |
| Scored windows | 8,000 | 8,000 |

The MLP's median detection latency is 11.1% lower, below the fixed 20%
latency condition, while macro-F1 is lower and nominal false alarms are much
higher. `ai_advantage_demonstrated` is therefore `false` and
`rule_baseline` remains preferred. This is a negative result, not a tuned AI
claim.

The 8,000 windows are correlated, stride-one observations from **180** held-out
scenario families. The family—not a window—is the independent replay unit. No
confidence interval, independent-window, alert-burden, wall-clock latency or
deployment claim follows from this result. Detection latency means simulator
ticks from observable onset to first correct causal label.

Protocol v3's final suite is fresh but uses the declared synthetic operating
profiles; it is not OOD stress, hardware-in-the-loop or physical evidence. No
INT8 artifact, Arm benchmark, production controller or autonomous actuator
command is implemented or claimed. See the full
[protocol v3 acceptance record](docs/protocol-v3-acceptance.md).

### Protocol v4 development follow-up

A fresh v4 development cycle separated model fitting, internal calibration and
single-use validation seed clusters. It compared the temporal MLP, two compact
causal TCN training policies and calibrated deterministic rules under
stride-one operational alert metrics.

The balanced TCN achieved the best cluster macro-F1 (`0.704253`), but its
false-alert burden was `121.622` episodes per 1,000 eligible ticks against a
fixed ceiling of `10.0`. Gating and lower class weights reduced alerts only by
reducing fault-detection quality. No learned candidate passed, so
`selected_candidate` is null, `rule_baseline` remains the bounded fallback, and
the response layer stays disconnected.

This is development evidence, not a fresh final-suite result. Two corrected
canonical runs were byte-identical across manifests, corpus, model artifacts and
the complete report. See the
[v4 development outcome](docs/evidence/v4-development-outcome.md) and
[machine reproduction receipt](docs/evidence/v4-reproduction-verification.json).

## Schema-v9 measurement semantics

Every scenario requires exactly:

```json
"telemetry": {
  "airflow_noise_fraction": 0.0,
  "airflow_bias_fraction": 0.0,
  "airflow_drift_fraction": 0.0,
  "actuator_position_noise_fraction": 0.0,
  "co2_sensor_noise_fraction": 0.0,
  "co2_sensor_bias_fraction": 0.0,
  "co2_sensor_drift_fraction": 0.0
}
```

All values must be finite fractions in `0.0..1.0`; missing, unknown and v8
input fails closed. Hand-written scenarios use zero values, preserving their
inherited numerical traces.

Bias is fixed per entity. Noise varies per tick. Drift linearly interpolates
deterministic samples at 20-tick anchors and remains bounded. Airflow effects
are scaled by connection capacity; CO2 effects are scaled by the controller's
upper threshold. Observations are clamped and airflow request/delivery/residual
invariants are recomputed consistently.

CO2 measurement happens before control. Frozen faults hold the latent sensor
value, after which bias, drift and readout noise still apply. This avoids an
unrealistic exactly constant signature while preserving replay determinism and
leaving latent CO2 mass untouched by the measurement calculation itself.

## Reproduce locally

Install dependencies and run all tests:

```bash
uv sync --locked --python 3.11 --extra dev
uv run --locked --python 3.11 --extra dev python -m pytest
```

Use the staged v3 procedure in the
[protocol v3 acceptance record](docs/protocol-v3-acceptance.md#reproduction).
It generates a development corpus, selects and freezes a policy, creates a
separate final corpus, and evaluates the final suite once. All generated output
belongs under a new ignored `out/` directory; the protocol rejects pre-existing
final reports rather than overwriting evidence.

The older `sweep-v2` command path remains in the codebase only for historical
comparison. Its inspected IID and stress metrics are not current selection or
final-evaluation evidence.

Run rolling inference on a compatible scenario:

```bash
PYTHONPATH=src uv run python -m aeolus.detector predict \
  artifacts/aeolus_fault_detector.json scenarios/standard_habitat.json
```

Each line contains `end_tick`, `label`, `confidence` and all class
probabilities. Loading fails closed on format, transform, shape, vocabulary,
non-finite parameters, selector hash or topology hash.

Generate and visualise one replay:

```bash
PYTHONPATH=src uv run python -m aeolus \
  scenarios/primary_fan_degradation.json out/degradation.jsonl
PYTHONPATH=src uv run python -m aeolus.visualise \
  out/degradation.jsonl out/degradation.html
```

On Windows PowerShell, set `$env:PYTHONPATH = "src"` before module commands.
Generated scenarios and corpora remain ignored under `out/`.

## Repository fixtures

| File | Purpose |
|---|---|
| `scenarios/standard_habitat.json` | Healthy zero-noise schema-v9 reference. |
| `scenarios/high_demand_healthy.json` | Healthy high-demand reference. |
| `scenarios/primary_fan_degradation.json` | Gradual Cabin A path degradation. |
| `scenarios/blocked_path.json` | Sudden Cabin B path blockage. |
| `scenarios/frozen_sensor_healthy.json` | Healthy frozen-sensor counterfactual. |
| `scenarios/frozen_sensor.json` | Frozen lab sensor while truth evolves. |
| `scenarios/families.json` | Small historical corpus-v2 contract fixture. |
| `scenarios/sweep-v1.json` | Legacy three-way experimental sweep. |
| `scenarios/sweep-v2.json` | Historical inspected IID/OOD experiment; not current final evidence. |
| `scenarios/sweep-v3-development.json` | Current train/validation-only policy-selection suite. |
| `scenarios/sweep-v3-final.json` | Current fresh final-only evaluation suite. |

## Deferred work

- improve learned calibration/generalisation without using test or stress data
  for selection;
- quantify whether a hybrid learned-plus-rule system adds value;
- only after FP32 value is established, evaluate INT8 and benchmark on a
  declared Arm target;
- defer governor logic and redundant airflow hardware to a separate slice.

See [simulation rules](docs/simulation-rules.md), the
[telemetry contract](docs/telemetry-contract.md), and the historical
[corpus-v2 acceptance receipt](docs/corpus-v2-acceptance.md).

## Licence

[MIT](LICENSE)
