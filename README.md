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
- a family-bound corpus-v2 contract with paired observable-onset labels and
  transition-aware evaluation, alongside the streaming rule baseline;
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

# Generate the historical v1 labelled window corpus and its manifest
PYTHONPATH=src uv run python -m aeolus.corpus out/corpus \
  scenarios/standard_habitat.json scenarios/high_demand_healthy.json \
  scenarios/primary_fan_degradation.json scenarios/blocked_path.json \
  scenarios/frozen_sensor.json

# Grade the historical rule baseline against corpus v1
PYTHONPATH=src uv run python -m aeolus.evaluate \
  out/corpus/corpus.jsonl scenarios/standard_habitat.json \
  scenarios/high_demand_healthy.json scenarios/primary_fan_degradation.json \
  scenarios/blocked_path.json scenarios/frozen_sensor.json

# Generate the Gate-2 observable-labelled corpus-v2 fixture
PYTHONPATH=src uv run python -m aeolus.corpus \
  --v2 out/corpus-v2 scenarios/families.json

# Grade only rows that match manifest-derived held-out test-family evidence
PYTHONPATH=src uv run python -m aeolus.evaluate \
  --v2 out/corpus-v2/corpus.jsonl scenarios/families.json --split test
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

`aeolus.corpus` builds labelled windows for the future fault classifier. Corpus
v1 feature rows are exactly `model_feature_row()` output and retain their
historical declared-profile labels. Corpus v2 instead uses paired
`model_input_v1` traces to label first observable onset, persists the frozen
selector/topology metadata, and excludes onset-straddling windows from scored
metrics. Evaluation rejects rows whose exact schema, complete reference/fault
family streams and window inventories, float32-narrowed features, feature
values against their recomputed paired replay, family split, role, onset or
label disagree with manifest-derived evidence. Corpus output is a generated
artifact and belongs in `out/`, not Git.

## Results so far

Every claimed number below is reproducible from the commands in this README.

| Date | What was measured | Result | What it represents |
|---|---|---|---|
| 2026-07-27 | Fault-detection quality of the rule baseline on corpus v1 (115 windows from 5 scenario runs) | **111/115 windows (96.5%)**, zero false alarms on nominal runs; all 4 misses are onset-boundary windows. Detection latency 10 / 5 / 10 ticks (degradation / blocked / frozen). | Historical corpus-v1 baseline only; its labels use declared starts and are not comparable to corpus-v2 scoring. |
| 2026-07-29 | Gate-2 corpus-v2 contract fixture (3 paired families, 138 windows) | **134/134 scored correct**, 4 transition-excluded; observable onset 21 / 30 / 31 ticks; latency 10 / 9 / 9 ticks (blocked / frozen / degradation). | A contract-validation fixture, not a model-performance or generalisation claim. It proves family splits, observable labels, and transition-aware scoring before the scenario sweep. |

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
2. **Temporal classifier** — trained on swept families and graded on families
   it never saw (train/validation/test split by family, never by window).
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
