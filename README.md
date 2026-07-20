# HVAC Intelligence Layer — Belimo Device Diagnostics on Arm

> Submission for the [Arm Create: AI Optimization Challenge 2026](https://arm-ai-optimization-challenge.devpost.com/) — **Physical AI track**

A smart integration and diagnostics layer for Belimo HVAC actuators. Low-level device data — motor torque, power draw, setpoint, position, temperature — is ingested, normalised, and fed to an on-device AI agent running on an Arm-powered Raspberry Pi. The agent detects device faults early, predicts their downstream effect on building-wide airflow, and exposes clean, building-level insight that third-party integrators can actually use.

---

## The Problem

HVAC energy demand is rising — climate change is driving more cooling load, and datacentre growth is compounding it. Belimo sells best-in-class individual actuators, but the raw device data they expose is low-level and fragmented. Third-party integrators face real complexity turning it into useful whole-building insight.

Two costly failure modes result:
1. **Silent faults** — a broken actuator goes undetected until an engineer notices the building is too hot
2. **Slow diagnosis** — even when a fault is found, locating which device caused it and understanding the airflow impact across the building takes time

---

## Solution

Build a layer that sits between Belimo devices and building management systems, running locally on an Arm edge board:

```
Belimo Devices  (actuator position, torque, power, setpoint, PCB temp, tag)
       │
       ▼
  Ingestion & Normalisation  ←── per-device calibration, tag resolution
       │
       ▼
   AI Agent (on Arm / Raspberry Pi + InfluxDB)
       │  ├─ Fault Detection   → flag anomalous device behaviour early
       │  └─ Airflow Diagnosis → model downstream impact on building airflow
       ▼
  Building-Level API / Dashboard
       │  clean, normalised data ready for BMS or third-party integrators
       ▼
  Alerts & Reports
```

---

## Device Inputs

| Input | Source | Notes |
|---|---|---|
| Internal PCB / box temperature | Belimo device telemetry | Thermal fault indicator |
| Motor torque | Device telemetry | Mechanical stress / obstruction |
| Power draw | Device telemetry | Efficiency baseline + anomaly |
| Setpoint & actual position | Device telemetry | Tracking error detection |
| Movement direction | Device telemetry | Unexpected reversals |
| Tag / device ID | Configuration | Building topology mapping |

Normalised position data is the key output — exposed locally via Raspberry Pi and stored in **InfluxDB** for time-series analysis.

---

## AI Agent

Runs on a fixed cadence on the Arm edge board. Each tick:

1. **Ingest** — pull latest telemetry from all registered devices
2. **Normalise** — apply per-device calibration and tag context
3. **Detect** — flag devices with anomalous torque, power, tracking error, or temperature
4. **Diagnose** — model airflow impact of any flagged fault across the building graph
5. **Log** — structured JSON decision record per tick

### Arm Optimisation Focus

- Inference runs fully **on-device** (Raspberry Pi 5 / Cortex-A class), no cloud dependency
- Anomaly detection model quantized to **INT8** for Arm NEON pipelines
- Benchmarked with **Arm Performix** for latency and throughput
- Target: full ingest → detect → diagnose loop under **200 ms** per tick
- Fallback to deterministic threshold rules if inference exceeds latency budget

---

## Scope — Air First

The concept applies to both air and water systems. v1 targets **air** — easier to prototype, simulate, and demo in a hackathon setting. Water systems are a natural v2 extension.

---

## Repo Structure

```
arm-hackathon/
├── ingestion/        # Belimo device polling, normalisation, tag resolution
├── agent/            # AI fault detector + airflow diagnosis model
├── storage/          # InfluxDB interface + time-series helpers
├── api/              # Local REST/MQTT API for BMS / integrators
├── evaluation/       # Scenario runner, metrics, comparison report
├── params/           # Device registry + calibration files
├── scenarios/        # Scenario definitions (nominal, fault, degradation)
├── tests/            # Fault detection, normalisation, agent baseline tests
├── benchmarks/       # Arm Performix results + latency logs
└── requirements.txt
```

---

## Scenarios

| Scenario | What Changes | What It Tests |
|---|---|---|
| Nominal | All devices healthy | Baseline normalisation, latency |
| Single device fault | One actuator torque/position anomaly | Early fault detection |
| Cascading airflow impact | Faulted device starves downstream zones | Building-level diagnosis |
| Slow degradation | Torque creeps up over hours | Trend detection before failure |
| Device dropout | Telemetry stops arriving | Safe fallback, alert escalation |

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
```

**Target hardware:** Raspberry Pi 5 (Arm Cortex-A76) running Linux + InfluxDB  
**Python:** 3.10+

---

## License

MIT — see [LICENSE](LICENSE)

---

> ⚠️ **Prototype only.** Not a certified building management or safety system. Device thresholds and airflow models are placeholders pending real Belimo device data.
