# Division of Labour

> Arm Create: AI Optimization Challenge 2026 — submission deadline **14 August 2026**

Three contributors, 25 days. This document assigns ownership of every module and milestone. Each owner is accountable for their area being tested and merged on time. If you're blocked, raise it immediately — don't sit on it.

---

## Owners at a Glance

| Module | Folder | Owner | Due |
|---|---|---|---|
| Telemetry emulator | `ingestion/emulator.py` | Ben | 23 Jul |
| Device normalisation | `ingestion/normaliser.py` | Ben | 23 Jul |
| InfluxDB interface | `storage/` | MS-Mesh | 25 Jul |
| REST / MQTT API | `api/` | MS-Mesh | 30 Jul |
| Grafana dashboard config | `api/grafana/` | MS-Mesh | 3 Aug |
| Anomaly detection model | `agent/detector.py` | Alex | 27 Jul |
| ONNX export + INT8 quantization | `agent/detector.py` | Alex | 30 Jul |
| Airflow graph model | `agent/graph.py` | Alex | 1 Aug |
| Closed-loop controller | `agent/controller.py` | Alex | 3 Aug |
| Deterministic fallback | `agent/baseline.py` | Alex | 3 Aug |
| Structured logger | `agent/logger.py` | Shared | 27 Jul |
| Actuation dispatch + bounds | `actuation/` | Ben | 28 Jul |
| Scenario definitions | `scenarios/` | Shared | 30 Jul |
| Scenario runner + metrics | `evaluation/` | MS-Mesh | 4 Aug |
| Arm Performix benchmarks | `benchmarks/` | Alex | 6 Aug |
| Tests | `tests/` | Shared | Rolling |
| Devpost write-up + video | — | Alex (lead) | 12 Aug |

---

## Alex — [@akurkar07](https://github.com/akurkar07)

**Focus: AI core, Arm optimisation story, submission**

The highest-weighted judging criterion (40 pts) is technical implementation. This area is Alex's primary responsibility.

### Tasks

- [ ] `agent/detector.py` — build and train a small ONNX autoencoder on normal emulator telemetry
- [ ] Export model to ONNX FP32; record baseline latency and memory on Arm hardware
- [ ] Apply post-training static INT8 quantization via `onnxruntime.quantization`; record delta
- [ ] `agent/graph.py` — topology graph of devices; estimate downstream airflow impact of a flagged fault
- [ ] `agent/controller.py` — select corrective setpoint commands for healthy neighbours; emit bounded commands
- [ ] `agent/baseline.py` — deterministic threshold fallback when inference exceeds latency budget
- [ ] `benchmarks/` — Arm Performix run: FP32 vs INT8, latency (ms), throughput (inferences/s), memory (MB)
- [ ] Devpost submission text (all four sections) and demo video (≤3 min showing fault inject → detect → compensate)

### Exit gates
- Autoencoder detects injected faults in all five emulator scenarios
- INT8 model shows measurable latency improvement over FP32 on Arm hardware
- Full sense → detect → act loop runs under 200 ms at INT8

---

## Ben — [@bbeennyy860-cyber](https://github.com/bbeennyy860-cyber)

**Focus: telemetry pipeline and actuation dispatch**

Everything depends on the emulator existing first. Ben's work unblocks the rest of the team.

### Tasks

- [ ] `ingestion/emulator.py` — generates a configurable stream of actuator telemetry for N devices
  - Normal operation baseline (torque, power, position, setpoint, PCB temp, tag)
  - Injectable fault modes: tracking error, torque spike, power anomaly, device dropout
  - Deterministic seed for reproducible scenarios
- [ ] `ingestion/normaliser.py` — per-device calibration, tag resolution, unit standardisation
- [ ] `actuation/dispatcher.py` — receives setpoint commands from the controller; enforces hard bounds before dispatch
- [ ] `actuation/bounds.py` — declares and validates per-actuator min/max limits
- [ ] Unit tests for emulator fault injection and actuation bounds enforcement

### Exit gates
- Emulator runs standalone and outputs a valid telemetry stream to stdout and InfluxDB
- All five fault modes injectable via scenario config
- Dispatcher rejects any command exceeding declared actuator bounds

---

## MS-Mesh — [@MS-Mesh](https://github.com/MS-Mesh)

**Focus: storage, API, evaluation, demo infrastructure**

MS-Mesh builds the layer that makes results visible — InfluxDB, the API, Grafana, and the scenario runner that produces the evidence judges will see.

### Tasks

- [ ] `storage/influx.py` — InfluxDB client wrapper; write telemetry, agent decisions, and benchmark results
- [ ] `storage/schema.py` — define measurement names, tags, and field keys
- [ ] `api/server.py` — lightweight REST API (FastAPI); expose current device states, active faults, agent actions
- [ ] `api/mqtt.py` — MQTT publish/subscribe bridge for actuator command dispatch (optional but good for demo)
- [ ] `api/grafana/` — Grafana dashboard JSON config showing telemetry, fault flags, and corrective actions in real time
- [ ] `evaluation/runner.py` — loads a scenario JSON, runs the full pipeline, outputs pass/fail + metrics CSV
- [ ] `evaluation/metrics.py` — time-to-detect, time-to-correct, max excursion, loop latency
- [ ] Unit tests for storage writes and scenario runner output

### Exit gates
- InfluxDB receives telemetry and agent decisions during a full scenario run
- Grafana dashboard shows fault injection and agent response in real time
- Scenario runner produces a reproducible metrics CSV for all five scenarios

---

## Shared Responsibilities

### `agent/logger.py`
Structured JSON logger used by the agent each tick. Fields: `timestamp`, `device_id`, `observation`, `anomaly_score`, `fault_flag`, `action`, `rationale`, `loop_latency_ms`. Anyone can stub this first — Alex or Ben, whoever starts on the agent first.

### `scenarios/*.json`
Five scenario definition files (nominal, single fault, cascading, degradation, dropout). Each needs: device count, fault type, fault onset time, deterministic seed, expected detection window. Write these together in one session — they need to be consistent across all three modules.

### `tests/`
Every module owner writes tests for their own code. Shared test: `test_agent_vs_baseline.py` — runs AI controller and deterministic baseline over the same scenario, compares outcomes. Alex leads this once both controllers exist.

### `requirements.txt`
Owners add their own dependencies as they go. Alex owns the final version before submission.

---

## Timeline

```
Wk 1  (21–27 Jul)   Emulator + normaliser (Ben)  |  Detector model (Alex)  |  InfluxDB (MS-Mesh)
Wk 2  (28 Jul–3 Aug) Actuation dispatch (Ben)     |  Graph + controller (Alex) | API + Grafana (MS-Mesh)
Wk 3  (4–10 Aug)    Scenario runner (MS-Mesh)     |  Benchmarks (Alex)      |  Integration + tests (All)
Wk 4  (11–14 Aug)   Buffer + bug fixes (All)      |  Devpost + video (Alex) |  Final review (All)
```

---

## Integration Points

These are the three moments where the three streams connect. Plan a sync call at each one.

1. **23 Jul** — Emulator is running. Alex and MS-Mesh can start consuming the telemetry stream.
2. **3 Aug** — Controller is issuing commands. MS-Mesh wires the full pipeline into the scenario runner.
3. **6 Aug** — All scenarios pass. Alex runs Performix benchmarks on the complete system.

---

> Last updated 2026-07-20. Update this file when ownership changes — don't let it go stale.
