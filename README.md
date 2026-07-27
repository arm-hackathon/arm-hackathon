# ICARUS

> Deterministic multi-zone habitat air-circulation simulation with explicit
> requested-versus-delivered airflow and reproducible gradual fan degradation.

ICARUS is a local simulation. It models abstract CO₂ mass and airflow units in
a hub-layout habitat; it does not model real spacecraft equipment, life-support
limits, or a general fluid system.

## Current slice

The repository currently contains:

- a validated schema-v7 scenario graph with one air-processing bay;
- deterministic seeded CO₂ sources, occupancy profiles, proportional control
  and rate-limited actuators;
- shared-capacity airflow allocation with mass-conserving mixed return air;
- deterministic fault profiles: gradual primary-fan degradation, sudden
  blocked path and frozen sensor;
- JSONL replay traces, an allowlisted model-feature projection, a standalone
  HTML visualiser and a leakage-safe labelled window corpus;
- a streaming rule-baseline fault detector and an evaluation harness that
  grades any labeller on accuracy, confusion and detection latency;
- tests for replay determinism, fault semantics, mass conservation, airflow
  invariants, telemetry boundaries, corpus leakage and detector behaviour.

```text
CO₂ sources → sensor → controller → actuator position
                                      │
                                      v
                    requested loop airflow
                                      │
        static path health + hidden degradation effectiveness
                                      │
                                      v
                 delivered loop airflow → shared processing → scrubbed return
```

`requested_airflow` is derived from configured loop capacity and the measured
actuator position. `delivered_airflow` is physical flow after static path health,
degradation and shared-capacity allocation. `airflow_residual` is their
non-negative difference.

## Scenarios

| File | Purpose |
|---|---|
| `scenarios/standard_habitat.json` | Healthy reference scenario. It declares no fault profiles. |
| `scenarios/high_demand_healthy.json` | Healthy high-demand control and delivery scenario with enough shared capacity to isolate controller demand. |
| `scenarios/primary_fan_degradation.json` | High-demand scenario with a gradual primary-fan degradation on `cabin_a_to_processing`. |
| `scenarios/blocked_path.json` | The same high-demand habitat with a sudden blockage on `cabin_b_to_processing` from tick 30. |
| `scenarios/frozen_sensor.json` | The same habitat with the lab sensor frozen from tick 30 while lab demand steps down at tick 41. |

All five are schema-v7 JSON and replay deterministically from their declared
seeds.

## Run locally

```bash
# Full test suite
uv run --extra dev python -m pytest

# Generate a replay and a self-contained local report
mkdir -p out
PYTHONPATH=src uv run python -m icarus \
  scenarios/primary_fan_degradation.json \
  out/primary_fan_degradation.jsonl
PYTHONPATH=src uv run python -m icarus.visualise \
  out/primary_fan_degradation.jsonl \
  out/primary_fan_degradation.html

# Generate the labelled window corpus and its manifest
PYTHONPATH=src uv run python -m icarus.corpus out/corpus scenarios/*.json

# Grade the rule baseline against the corpus
PYTHONPATH=src uv run python -m icarus.evaluate \
  out/corpus/corpus.jsonl scenarios/*.json
```

On Windows PowerShell, set `PYTHONPATH` for the session before running the same commands:

```powershell
$env:PYTHONPATH = "src"
```

The HTML report plots generated and sensed CO₂, actuator response, requested
and delivered airflow, airflow residual, shared capacity and captured CO₂.
Generated traces and reports belong in `out/`, not Git.

## Replay, telemetry and corpus boundary

Trace records contain observable simulation outputs only. They do not expose
fault effectiveness, connection health, random seed or source-noise state.
`icarus.trace.model_feature_row()` is a separate, strict allowlist for any
future model-facing consumer; visualiser fields do not expand that model
feature set.

`icarus.corpus` builds the labelled window corpus for the future fault
classifier. Every feature row is exactly `model_feature_row()` output and
labels come from declared fault profiles, never from telemetry, so the corpus
carries no hidden fault truth. Corpus output is a generated artifact and
belongs in `out/`, not Git.

## Deliberately out of scope

- model training, ONNX export, quantisation and inference;
- a governor, redundant fan or automatic recovery mechanism;
- Arm benchmarks or hardware claims;
- dashboard, API, database, cloud, MQTT or hardware integration;
- real-world environmental-control or safety claims.

See [`docs/simulation-rules.md`](docs/simulation-rules.md) for the schema,
physics contract and validation rules, and
[`docs/telemetry-contract.md`](docs/telemetry-contract.md) for the observable
telemetry boundary.

## Licence

[MIT](LICENSE)
