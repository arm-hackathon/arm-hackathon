# ICARUS

> **Closed-loop fault response for a simulated habitat ventilation system, optimized for Arm.**
>
> Submission in progress for the [Arm Create: AI Optimization Challenge 2026](https://arm-ai-optimization-challenge.devpost.com/) · **Physical AI track**

ICARUS is a simulated two-zone habitat ventilation controller. It ingests telemetry from a circulation plant, uses a compact AI model to detect a degrading primary fan, and applies a deterministic safety governor to issue a bounded virtual command to a healthy redundant fan.

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
3. Telemetry records airflow, CO2/air-quality proxy, temperature, commanded fan speed, actual fan output, and tracking residual.
4. A compact ONNX model scores the telemetry window for a fault.
5. The safety governor either commands a bounded boost to a healthy redundant fan or hands control back when telemetry/model output is invalid.
6. The simulator applies that decision and writes a replay trace proving the resulting plant-state change.

The point is not an animated dashboard. The point is a visible, testable loop: **fault → inference → safe decision → virtual action → recovery**.

## Why Physical AI

The Arm Create Physical AI track accepts systems using real or simulated sensor data that produce control signals, anomaly detection, alerting, or actuator decisions for a physical system. ICARUS uses simulated environmental and actuator telemetry, performs local fault inference, and produces a virtual ventilation command under hard safety bounds.

## Arm optimization evidence

ICARUS will run inference on a declared Arm64 target. The project will publish a reproducible benchmark comparing:

- FP32 ONNX inference against an INT8-quantized path
- model artifact size
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

**Planning and scaffold stage.** The repository currently does not yet contain the simulator, model, benchmark results, or a hardware deployment. The README is a contract for what will be built and measured, not evidence that it already exists.

## Intended usage

Once the first slice lands, a clean environment should be able to run:

```bash
python -m pytest
python -m icarus primary_fan_degradation out/degradation.jsonl
```

Expected result: a non-empty replay trace and summary showing injected degradation, model detection, safety-governed redundant-fan command, and recovery.

## Safety and scope

- Simulation only. No live plant, client, or production telemetry.
- No real actuator command is emitted.
- The governor enforces bounded commands and a handback path.
- This is not a certified environmental-control or safety system.

## License

[MIT](LICENSE)
