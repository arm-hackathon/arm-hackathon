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

This experiment adds:

- closed-schema v9 telemetry settings for airflow noise, bias and drift;
  actuator-position noise; and CO2 sensor noise, bias and drift;
- SHA-256-derived uniform innovations in `[-1,1)` plus bounded piecewise-linear
  drift between deterministic 20-tick anchors;
- downstream CO2 readout effects: a frozen sensor holds its latent reading
  while bias, drift and readout noise continue, and the measured value drives
  the local controller;
- `sweep-v2`: 840 paired scenario families split into train, validation, IID
  test and separately reported OOD stress evidence;
- a softmax baseline and compact `temporal_summary_v1` MLP, selected using
  validation evidence only;
- a robust rule baseline selected from a committed 216-point validation grid;
- stride-one causal latency, an attainable predeclared advantage policy, strict
  JSON loading, FP32 ONNX export and Python/ONNX parity evidence;
- a one-command deterministic experiment runner.

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

## Measured result

The temporal MLP won validation selection over softmax (`0.5884` versus
`0.4639` macro-F1). It did **not** beat the calibrated rules on the locked IID
test and degrades further under OOD stress.

| Evidence | Temporal MLP | Calibrated rules |
|---|---:|---:|
| IID test macro-F1 | 0.5765 | 0.6410 |
| IID nominal false alarms | 35.36% | 2.53% |
| IID median causal latency | 9 ticks | 11 ticks |
| OOD stress macro-F1 | 0.3085 | 0.4386 |
| OOD stress nominal false alarms | 66.38% | 0.51% |

The learned model is 18.2% faster by median IID latency, below the fixed 20%
latency-win threshold, and the false-alarm regression is material. Therefore
`ai_advantage_demonstrated` is `false` and `rule_baseline` remains preferred.
This is a fair negative result, not a tuned AI claim.

The sweep contains 360 training, 120 validation, 180 IID test and 180 stress
families. It generates 38,640 windows: 37,348 scored windows plus 1,292
transition windows retained for causal history but excluded from fitting and
classification metrics. Python and FP32 ONNX probabilities agree within
`1.257e-6`, below the `1e-5` bound.

Committed compact evidence:

- `artifacts/aeolus_fault_detector.json` — selected transform, normalization,
  MLP and contract metadata;
- `artifacts/aeolus_fault_detector.onnx` — FP32 transform and MLP graph;
- `artifacts/aeolus_fault_metrics.json` — candidate selection, calibrated rule
  parameters, IID/stress metrics, latency, conclusion, sizes and parity.

No INT8 artifact or Arm performance result is implemented or claimed.

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
uv sync --extra dev
uv run --extra dev python -m pytest
```

Run the complete experiment from an empty output directory:

```bash
PYTHONPATH=src uv run --extra ml python -m aeolus.experiment \
  scenarios/sweep-v2.json out/experiment-v2 artifacts
```

The command generates sweep scenarios, corpus-v2 evidence, both candidates,
validation selection, rule calibration, locked IID/stress evaluation and final
artifacts. The accepted family-manifest hash is:

```text
28db9bed90ab18a8f7b970a80dd72fdb3ecae316157b4b9e3819c2c7471f8465
```

Individual stages remain available:

```bash
PYTHONPATH=src uv run python -m aeolus.sweep \
  scenarios/sweep-v2.json out/sweep-v2
PYTHONPATH=src uv run python -m aeolus.corpus \
  --v2 out/corpus-v2 out/sweep-v2/families.json
PYTHONPATH=src uv run --extra ml python -m aeolus.detector train \
  out/corpus-v2/corpus.jsonl out/sweep-v2/families.json \
  28db9bed90ab18a8f7b970a80dd72fdb3ecae316157b4b9e3819c2c7471f8465 \
  artifacts/aeolus_fault_detector.json \
  artifacts/aeolus_fault_detector.onnx \
  artifacts/aeolus_fault_metrics.json
```

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
| `scenarios/sweep-v2.json` | Fair IID plus OOD stress experiment. |

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
