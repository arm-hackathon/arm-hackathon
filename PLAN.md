# ICARUS Plan

## End goal

Build an Arm-powered orchestration layer for otherwise independent ventilation
actuators. The system must detect a local equipment fault, understand its
effect on the wider habitat, and coordinate a bounded response using healthy
equipment before environmental conditions become unsafe.

The final demonstration must show:

```text
isolated actuator degrades
→ system-wide AI identifies the fault and affected zones
→ deterministic governor activates healthy redundant capacity
→ airflow recovers and CO₂ exposure is reduced
```

AI provides diagnosis and confidence. Deterministic safety logic retains
control of every actuator command.

## Core objectives

1. **System awareness:** combine telemetry from independent actuators, sensors
   and the ventilation topology into one coherent system state.
2. **Fault diagnosis:** distinguish normal demand from fan degradation,
   blockage and invalid sensor data.
3. **Safe orchestration:** coordinate healthy actuators without exceeding
   declared command limits; hand control back when evidence is invalid.
4. **Measurable benefit:** prove that orchestration improves airflow recovery
   and environmental outcomes over isolated local control.
5. **Arm optimisation:** demonstrate reproducible, efficient local inference on
   Arm64 hardware.

## What must be implemented

### 1. Actuator and plant model

- Preserve local CO₂-driven controllers for each zone.
- Separate commanded output, actual output, health, airflow and power.
- Add shared ducts or capacity constraints so actuator behaviour has
  system-wide consequences.
- Add a healthy redundant fan or alternative airflow path.

### 2. Deterministic scenarios

- Nominal operation.
- Gradual primary-fan degradation.
- Blocked airflow path.
- Invalid or frozen sensor.
- Identical fault runs with orchestration enabled and disabled.

### 3. Telemetry and topology

- Record CO₂, actuator command, output, airflow, power, health and validity.
- Derive command-tracking residuals and rolling trends.
- Map each actuator to the ducts and zones it affects.
- Keep every run seeded, replayable and testable.

### 4. AI diagnosis

- Generate labelled telemetry windows from the simulator.
- Train a compact temporal classifier and compare it with rule-based and
  threshold baselines.
- Export FP32 and INT8 ONNX models.
- Report fault class, confidence, detection latency and false alarms.

### 5. Safety governor

- Require persistent model confidence before intervening.
- Select healthy redundant capacity using topology and current demand.
- Cap autonomous commands at 80%.
- Produce `HAND_BACK` for invalid telemetry, invalid inference or insufficient
  healthy capacity.
- Record the reason for every decision.

### 6. Evidence and presentation

- Plot fault injection, detection, commands, airflow and CO₂ recovery together.
- Measure recovery time and CO₂ exposure with and without orchestration.
- Benchmark FP32 versus INT8 model size, latency, throughput, memory and
  detection quality on a declared Arm64 target.
- Provide one-command reproduction, raw benchmark results and a demonstration
  video under three minutes.

## Completion criteria

The project is complete when the nominal run produces no intervention, each
declared fault is detected and classified, invalid data causes `HAND_BACK`, the
backup command remains within its bound, and the orchestrated run measurably
outperforms the same untreated fault. All claims must be backed by replay
traces, automated tests and Arm benchmark evidence.

## Scope guard

Do not prioritise a dashboard, database, cloud service, detailed fluid
dynamics, physical sensors or general building-management features until the
closed-loop orchestration demonstration and Arm measurements are complete.
