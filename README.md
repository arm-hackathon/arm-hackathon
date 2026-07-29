# Project AEOLUS

> **A**irflow and **E**nvironmental **O**bservation **L**aboratory for
> **U**ser-defined **S**cenarios
>
> Deterministic habitat environmental simulation with replayable traces.

AEOLUS is a local simulation. It models abstract CO₂ mass and airflow units in
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
| `scenarios/frozen_sensor_healthy.json` | Fault-free paired reference for the frozen-sensor demand transition. It differs from `frozen_sensor.json` only in `fault_profiles`. |
| `scenarios/frozen_sensor.json` | The same habitat with the lab sensor frozen from tick 30 while lab demand steps down at tick 41. |
| `scenarios/families.json` | Gate-2 family manifest. It binds paired references to the frozen `model_input_v1` topology/selector hashes and one evaluation split. |

All scenario files are schema-v7 JSON and replay deterministically from their
declared seeds. The current three Gate-2 families are test-only fixtures: they
prove the contract but are not a train/validation/test corpus.

## Run locally

```bash
# Full test suite
uv run --extra dev python -m pytest

# Generate a replay and a self-contained local report
mkdir -p out
PYTHONPATH=src uv run python -m aeolus \
  scenarios/primary_fan_degradation.json \
  out/primary_fan_degradation.jsonl
PYTHONPATH=src uv run python -m aeolus.visualise \
  out/primary_fan_degradation.jsonl \
  out/primary_fan_degradation.html

# Generate the labelled window corpus and its manifest
PYTHONPATH=src uv run python -m aeolus.corpus out/corpus scenarios/*.json

# Grade the rule baseline against the corpus
PYTHONPATH=src uv run python -m aeolus.evaluate \
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
`aeolus.trace.model_feature_row()` is a separate, strict allowlist for any
future model-facing consumer; visualiser fields do not expand that model
feature set.

`aeolus.corpus` builds the labelled window corpus for the future fault
classifier. Every feature row is exactly `model_feature_row()` output and
labels come from declared fault profiles, never from telemetry, so the corpus
carries no hidden fault truth. Corpus output is a generated artifact and
belongs in `out/`, not Git.

## Results so far

Every claimed number below is reproducible from the commands in this README.

| Date | What was measured | Result | What it represents |
|---|---|---|---|
| 2026-07-27 | Fault-detection quality of the rule baseline on corpus v1 (115 windows from 5 scenario runs) | **111/115 windows (96.5%)**, zero false alarms on nominal runs; all 4 misses are onset-boundary windows. Detection latency 10 / 5 / 10 ticks (degradation / blocked / frozen). | The performance floor for fault detection in AEOLUS: what the simplest hand-written rules achieve on the first, small corpus. It is the bar the learned classifier must beat — not a model result and not a deployment claim. |

Read the latency column with the accuracy: the baseline is accurate but
structurally slow, because a rule cannot fire until a fault fills its
persistence window. Beating this baseline means *matching* its accuracy
while firing *sooner*.

### Next steps

Each upcoming slice adds its own row to this table, measured by the same
harness and discipline:

1. **Scenario sweep** — generate training volume: the same five scenario
   types across seeds, fault starts, fault strengths and targets, with
   demand shapes varied across every class so labels cannot be inferred
   from occupancy patterns.
2. **Temporal classifier** — trained on swept runs and graded on runs it
   never saw (train/validation/test split by run, never by window).
   Success = baseline-matching accuracy with lower detection latency.
3. **FP32 ONNX + INT8 quantisation on Arm64** — model size, latency,
   throughput and accuracy delta, FP32 versus INT8, on the declared Azure
   Arm64 target. This is the core optimisation evidence.
4. **Safety governor + redundant fan** — orchestrated versus untreated
   fault runs: recovery time and CO₂ exposure with and without
   intervention.
5. **Reproducibility packaging** — one-command reproduction of every row
   in this table, plus the demonstration video.

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
