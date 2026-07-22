# ICARUS

> **Closed-loop fault response for a simulated habitat ventilation system, optimised for Arm.**
>
> Submission in progress for the [Arm Create: AI Optimization Challenge 2026](https://arm-ai-optimization-challenge.devpost.com/) · **Physical AI track**

ICARUS is a simulated multi-zone habitat ventilation controller. It ingests telemetry from a circulation plant, uses a compact AI model to detect a degrading primary fan, and applies a deterministic safety governor to issue a bounded virtual command to a healthy redundant fan.

```text
simulated sensors → compact AI inference → safety governor → virtual actuator command
       ↑                                                               │
       └──────────── changed ventilation-plant state and replay ───────┘
```

This is a simulation and research prototype. It makes no claim to control real spacecraft, life-support equipment, or certified safety-critical systems.

## The demo

The first scenario is deliberately narrow:

1. A deterministic two-zone habitat ventilation plant begins in a nominal state.
2. The primary circulation fan degrades, reducing actual airflow.
3. Telemetry records airflow, CO₂/air-quality proxy, temperature, commanded fan speed, actual fan output, and tracking residual.
4. A compact ONNX model scores the telemetry window for a fault.
5. The safety governor either commands a bounded boost to a healthy redundant fan or hands control back when telemetry/model output is invalid.
6. The simulator applies that decision and writes a replay trace proving the resulting plant-state change.

The point is not an animated dashboard. The point is a visible, testable loop: **fault → inference → safe decision → virtual action → recovery**.

## Why Physical AI

The Arm Create Physical AI track accepts systems using real or simulated sensor data that produce control signals, anomaly detection, alerting, or actuator decisions for a physical system. ICARUS uses simulated environmental and actuator telemetry, performs local fault inference, and produces a virtual ventilation command under hard safety bounds.

## Arm optimisation evidence

ICARUS will run inference on a declared Arm64 target. The project will publish a reproducible benchmark comparing:

- FP32 ONNX inference against an INT8-quantised path
- model artefact size
- p50 and p95 inference latency
- inference throughput / control-loop deadline behaviour
- fault-detection quality and false alarms
- target specification and exact benchmark commands

No Raspberry Pi, NEON, Arm Performix, memory, latency, or sub-200 ms claim will be made until it has an attached measurement receipt from the declared target.

## Planned first vertical slice

The first runnable slice contains only the proof loop:

```text
icarus/
├── simulation/       # deterministic two-zone ventilation plant
├── scenarios/        # nominal, primary fan degradation, invalid sensor
├── model/            # synthetic-data training, FP32 ONNX export, inference
├── control/          # safety governor and bounded virtual actuation
├── traces/           # JSONL replay writer
├── tests/            # scenario, safety, replay, and model-path tests
└── benchmarks/       # added after the local loop is green
```

It intentionally excludes a web dashboard, API, database, MQTT, hardware integration, and topology model. Those can wait until there is an actual system worth displaying.

## Acceptance conditions for the first slice

- Nominal scenario produces no fault command.
- Primary-fan degradation is injected deterministically and detected by the ONNX model.
- Governor commands the redundant fan within its safe range, never above 80%.
- Invalid telemetry produces `HAND_BACK`, not an autonomous command.
- The replay trace is deterministic for a fixed scenario seed.
- The trace demonstrates that the command changes simulated plant state and restores airflow above the scenario floor.

## Status

**Simulation and control foundation.** The repository contains a configurable,
deterministic habitat simulator, concentration-driven setpoints, rate-limited
actuator movement, occupancy profiles, correlated per-zone CO₂ variation,
shared fan capacity and mixed return air. It also provides JSONL replay traces,
a 60-tick unrecorded warm-up, a standalone trace visualiser and automated
tests. It does not yet contain fault injection, the ONNX detector, the safety
governor, Arm benchmark results, or hardware deployment.

## Intended usage

The current simulator and trace visualiser can be run with:

```bash
python -m pip install -e .
python -m pytest
python -m icarus scenarios/standard_habitat.json traces/standard_habitat.jsonl
python -m icarus.visualise traces/standard_habitat.jsonl out/standard_habitat.html
```

The generated HTML report is self-contained and can be opened directly in a
browser. It plots occupancy, generated and sensed CO₂, actuator setpoints and
actual positions, tracking residuals, power, requested and allocated airflow,
shared capacity, connection health and cumulative captured CO₂.

## Safety and scope

- Simulation only. No live plant, client, or production telemetry.
- No real actuator command is emitted.
- The current proportional controller enforces bounded commands; the planned
  safety governor will add a handback path.
- This is not a certified environmental-control or safety system.

## Licence

[MIT](LICENSE)
