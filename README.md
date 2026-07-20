# HVAC Intelligence Layer — Closed-Loop AI Control on Arm

> Submission for the [Arm Create: AI Optimization Challenge 2026](https://arm-ai-optimization-challenge.devpost.com/) — **Physical AI track**

An on-device AI system for commercial HVAC environments. It ingests real-time actuator telemetry — motor torque, power draw, position, temperature — detects faults and anomalies using a quantized model running on an Arm-powered Raspberry Pi, and **closes the loop** by issuing corrective actuator commands directly: adjusting fan speeds, valve positions, and damper setpoints to compensate before a fault cascades into a building-wide problem.

Sensor → AI inference → actuator command → physical effect → repeat.

---

## The Problem

HVAC energy demand is rising — climate change is driving more cooling load, and datacentre growth is compounding it. Modern smart actuators expose rich telemetry, but two costly failure modes persist:

1. **Silent faults** — a degrading actuator goes undetected until the building overheats
2. **Slow recovery** — even when a fault is found, compensating healthy devices manually takes time, and the airflow impact across the building is rarely understood quickly

Existing BMS systems alert. They don’t autonomously compensate.

---

## Solution

A closed-loop AI controller running entirely on an Arm edge board, sitting between the actuator network and the BMS:

```
Actuator Telemetry  (position, torque, power, setpoint, PCB temp, tag)
        │
        ▼
  Ingestion & Normalisation  ←── per-device calibration, tag resolution
        │
        ▼
   AI Agent  ────────────────────── (Raspberry Pi 5, ONNX Runtime INT8)
       │                                    │
       ├─ Anomaly Detection                  │
       ├─ Airflow Impact Modelling           │
       └─ Corrective Action Selection        │
        │                                    │
        ▼                                    ▼
  Actuator Commands              Benchmarks (Arm Performix)
  (fan speed, valve %, damper)   latency, throughput, INT8 vs FP32
        │
        ▼
  Physical Environment Changes
        │
        ▼  (feedback loop)
  Actuator Telemetry  ────────────────▲
```

---

## Device Inputs

| Input | Notes |
|---|---|
| Motor torque | Mechanical stress / obstruction detection |
| Power draw | Efficiency baseline + anomaly |
| Setpoint & actual position | Tracking error — key fault signal |
| Movement direction | Unexpected reversal detection |
| Internal PCB / box temperature | Thermal fault indicator |
| Tag / device ID | Building topology mapping |

---

## AI Agent — Closed Loop

Runs on a fixed cadence (default 5 s) on the Arm edge board. Each tick:

1. **Ingest** — read normalised telemetry from all registered devices
2. **Detect** — anomaly model flags devices with abnormal torque, power, tracking error, or temperature
3. **Diagnose** — graph model estimates downstream airflow impact of any flagged fault
4. **Act** — issue corrective setpoint commands to healthy neighbouring devices to compensate
5. **Log** — structured JSON record: observation, classification, action, rationale

### Actuator Command Space

| Actuator | Command | Range |
|---|---|---|
| Circulation fan | Speed setpoint | 0–100% |
| Damper / valve | Position setpoint | 0–100% open |
| Bypass damper | Open / close | Binary + %|

All commands are **hard-bounded in firmware** — the agent cannot exceed physical actuator limits.

---

## Arm Optimisation Story

The core optimisation is a **measured before/after comparison** of the anomaly detection model on Arm hardware:

| Stage | Model | Runtime | Target metric |
|---|---|---|---|
| Baseline | FP32 ONNX autoencoder | ONNX Runtime (default) | Latency (ms), memory (MB) |
| Optimised | INT8 quantized ONNX | ONNX Runtime + Arm NEON | Latency (ms), memory (MB) |

- Quantization via `onnxruntime.quantization` — post-training static INT8
- Benchmarked with **Arm Performix** on Raspberry Pi 5 (Cortex-A76)
- Target: full sense → detect → act loop under **200 ms** at INT8
- Fallback to deterministic threshold rules if inference exceeds latency budget
- All benchmark results stored in `benchmarks/` with raw Performix output

---

## Scope — Air First

v1 targets air systems — simpler to prototype and demo. Water systems (pumps, valves, heat exchangers) are a natural v2 extension using the same agent architecture.

---

## Repo Structure

```
arm-hackathon/
├── ingestion/        # Device polling, normalisation, tag resolution
├── agent/
│   ├── detector.py   # ONNX anomaly detection model (FP32 + INT8)
│   ├── graph.py      # Airflow impact model across device topology
│   ├── controller.py # Corrective action selection + command dispatch
│   ├── baseline.py   # Deterministic rule-based fallback controller
│   └── logger.py     # Structured JSON rationale logging
├── actuation/        # Actuator command dispatch + hard bound enforcement
├── storage/          # InfluxDB interface + time-series helpers
├── api/              # Local REST/MQTT API for BMS / integrators
├── evaluation/       # Scenario runner, metrics, comparison report
├── params/           # Device registry + calibration files
├── scenarios/        # Scenario definitions (nominal, fault, degradation)
├── benchmarks/       # Arm Performix results — FP32 vs INT8 comparison
├── tests/
│   ├── test_detector.py
│   ├── test_actuation_bounds.py
│   └── test_agent_vs_baseline.py
└── requirements.txt
```

---

## Scenarios

| Scenario | Fault Injected | What It Tests |
|---|---|---|
| Nominal | None | Steady-state control, latency baseline |
| Single device fault | One actuator tracking error | Detection + neighbour compensation |
| Cascading airflow impact | Faulted device starves downstream zones | Graph diagnosis + multi-device response |
| Slow degradation | Torque creeps up over hours | Trend detection before failure |
| Device dropout | Telemetry stops arriving | Safe fallback, deterministic controller takes over |

---

## Getting Started

```bash
git clone https://github.com/akurkar07/arm-hackathon.git
cd arm-hackathon
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Run a scenario
python -m evaluation.runner --scenario scenarios/nominal.json

# Run INT8 vs FP32 benchmark
python -m benchmarks.run --device rpi5
```

**Target hardware:** Raspberry Pi 5 (Arm Cortex-A76) running Linux + InfluxDB  
**Python:** 3.10+ | **Key deps:** `onnxruntime`, `onnx`, `numpy`, `influxdb-client`, `paho-mqtt`

---

## License

MIT — see [LICENSE](LICENSE)

---

> ⚠️ **Prototype only.** Not a certified building management or safety system. Device thresholds, airflow models, and benchmark targets are placeholders pending real hardware validation.
