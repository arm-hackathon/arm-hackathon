# ICARUS

> **Deterministic habitat air-circulation simulation with replayable traces, built for the Arm Create: AI Optimization Challenge 2026 Physical AI track.**

ICARUS is a simulation and research prototype. It models circulation and CO2 removal in an abstract habitat layout, then writes a deterministic JSONL trace for every run.

It does **not** control spacecraft, life-support equipment, or any safety-critical hardware.

## Current simulation slice

The current implementation is a user-editable scenario graph, not a hard-coded two-room demo.

```text
scenario JSON → validation → fixed-tick habitat plant → JSONL replay trace
```

The reference scenario contains:

- two crew cabins with CO2 generation
- a laboratory with no CO2 generation
- one air-processing bay that captures CO2
- paired, directed circulation paths between each zone and the processing bay

The current graph supports this hub layout only. It is not a general fluid solver.

## What it does

- Loads and validates versioned JSON scenario files.
- Runs a fixed, deterministic 120-tick simulation.
- Calculates airflow from each path's configured maximum airflow and health.
- Models CO2 generation in zones and removal in the air-processing bay.
- Writes one JSONL trace row per tick with zone CO2, captured CO2, connection airflow, and connection health.
- Produces byte-identical records and traces for repeated runs of the same scenario.
- Rejects invalid scenario graphs with clear errors.

All values are abstract simulation units. They are not real spacecraft measurements, flow rates, or safety thresholds.

## Run it

Python 3.10+ is required. The runtime has no third-party dependencies. `pytest` is needed only for tests.

```bash
# Run the test suite
python -m pytest

# Run the reference habitat scenario and write a replay trace
PYTHONPATH=src python -m icarus \
  scenarios/standard_habitat.json \
  traces/standard_habitat.jsonl
```

With `uv`, the equivalent is:

```bash
uv run --extra dev python -m pytest
uv run --extra dev python -m icarus \
  scenarios/standard_habitat.json \
  traces/standard_habitat.jsonl
```

A successful run prints the source scenario, tick count, trace path, and final zone/captured-CO2 state. The standard scenario produces 120 trace rows.

## Repository layout

```text
src/icarus/
├── config.py       # scenario graph parsing and validation
├── plant.py        # one fixed-tick circulation and CO2-removal step
├── scenario.py     # deterministic scenario runner
├── trace.py        # JSONL trace writer
└── __main__.py     # command-line entry point

scenarios/          # versioned scenario files
tests/              # config, plant, scenario, and trace coverage
docs/               # simulation rules and scope
traces/             # generated or checked-in replay traces
```

## Scope and constraints

The simulation models circulation and CO2 removal only. Oxygen, pressure, temperature, humidity, trace contaminants, fire, leaks, external vacuum, hardware integration, and real telemetry are out of scope.

The implementation deliberately does not yet contain:

- fault profiles that change connection health during a run
- AI or ML fault detection
- ONNX inference, quantization, or Arm benchmarking
- a safety governor, redundant fan, or automatic actuator command
- a dashboard, API, MQTT, database, Docker deployment, or cloud service

Those are future slices, not present-tense claims.

## Further detail

[`docs/simulation-rules.md`](docs/simulation-rules.md) is the source of truth for the graph format, tick order, validation rules, determinism guarantees, and deliberate exclusions.

## License

[MIT](LICENSE)
