# Space Habitat Ventilation AI

> ⚠️ **Simulation only — not a certified life-support design.**  
> This project is a research/demo build. It must not be presented as real ECLSS design, human-safety guidance, flight software, or a substitute for certified engineering review.

A **physical AI** system for automated ventilation control in a pressurised space habitat. Onboard sensors feed real-time environmental readings (CO₂, O₂, pressure, temperature, humidity) to an AI agent that continuously reasons about cabin state and issues bounded control commands to actuators — fans, scrubbers, and O₂ supply — to keep the environment within safe operating ranges.

---

## Table of Contents

- [Concept](#concept)
- [System Architecture](#system-architecture)
- [Hardware & Sensing Layer](#hardware--sensing-layer)
- [AI Agent](#ai-agent)
- [Actuator Layer](#actuator-layer)
- [Scenarios & Testing](#scenarios--testing)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Parameter Registry](#parameter-registry)
- [References](#references)
- [Disclaimer](#disclaimer)

---

## Concept

The core idea is a closed-loop physical AI controller: sensors observe the real (or emulated) cabin environment, an AI agent interprets those readings and decides on control actions, and actuators respond — all in a continuous feedback cycle. The AI does **not** run an internal physics model; it reasons directly from sensor data, mimicking how a real embedded controller would operate.

### What "Physical AI" Means Here

| Traditional Simulation | Physical AI (This Project) |
|---|---|
| Ground truth from an ODE model | Ground truth from sensors / emulated hardware |
| Controller reads synthetic state | Controller reads live or emulated sensor streams |
| Physics runs in software | Physics happens in hardware (or a hardware-in-the-loop emulator) |
| Evaluate against simulation trace | Evaluate against real sensor logs and actuator response |

### Success Criteria

- AI agent maintains cabin CO₂, O₂, pressure, temperature, and humidity within target ranges under normal and fault conditions
- Clear separation between sensing, reasoning, and actuation layers
- All actuator commands are bounded and traceable to agent rationale
- System degrades gracefully and alerts on faults (sensor dropout, fan failure, scrubber loss)
- Full run logs: sensor readings, agent decisions, actuator states, and outcome classification

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SPACE HABITAT CABIN                      │
│                                                             │
│   [CO₂ Sensor] [O₂ Sensor] [Pressure] [Temp] [Humidity]   │
└──────────────────────┬──────────────────────────────────────┘
                       │ sensor readings (real-time stream)
                       ▼
            ┌──────────────────┐
            │   Sensor Layer   │  sampling, calibration, fault detection
            └────────┬─────────┘
                     │ normalised observation vector
                     ▼
            ┌──────────────────┐
            │    AI Agent      │  interprets state → selects action → logs rationale
            └────────┬─────────┘
                     │ bounded control commands
                     ▼
            ┌──────────────────┐
            │  Actuator Layer  │  fans, scrubbers, O₂ supply valves
            └────────┬─────────┘
                     │ physical effect on cabin environment
                     ▼
            ┌──────────────────┐
            │  Evaluation &    │  target tracking, fault logs, run report
            │  Logging Layer   │
            └──────────────────┘
```

### Layer Responsibilities

| Layer | Owns | Must NOT own |
|---|---|---|
| Sensor layer | Raw readings, calibration, noise flagging, dropout detection | AI policy or actuator commands |
| AI agent | State interpretation, action selection, rationale logging | Direct hardware writes; bypassing actuator bounds |
| Actuator layer | Executing bounded commands, reporting actuator state/faults | Overriding AI commands or modifying sensor data |
| Evaluation | Run pass/fail, metrics, replay logs | Silently correcting invalid sensor or actuator data |

---

## Hardware & Sensing Layer

The system is designed around low-cost, readily available sensors for rapid prototyping. All sensors communicate over I²C or UART and are read by a central microcontroller (e.g. Raspberry Pi or Arduino).

| Variable | Target Sensor | Interface | Notes |
|---|---|---|---|
| CO₂ concentration | MH-Z19C / SCD40 | UART / I²C | NDIR-based; requires warm-up period |
| O₂ concentration | ME2-O2 electrochemical | Analog/UART | Calibrate at ambient before use |
| Cabin pressure | BMP390 / LPS22HB | I²C | ±0.5 hPa typical accuracy |
| Temperature | SHT40 / BME688 | I²C | Co-locate with humidity sensor |
| Humidity | SHT40 / BME688 | I²C | Relative humidity |

### Sensor Contract

- Readings are **timestamped** at acquisition, not at agent ingestion
- Dropout (no reading within `sample_timeout`) is surfaced to the agent as a distinct `SENSOR_FAULT` observation — never silently filled
- Calibration offsets stored in `params/sensor_calibration.json`; never hardcoded

---

## AI Agent

The agent operates on a fixed control cadence (e.g. every 5 seconds). At each tick it:

1. **Observes** — receives the latest normalised sensor vector
2. **Classifies state** — maps observations to one of: `NOMINAL`, `WARNING`, `ALERT`, `FAULT`
3. **Selects action** — chooses actuator setpoints from a bounded action space
4. **Logs rationale** — records the observation, classification, chosen action, and reason in structured JSON

### Action Space

| Actuator | Command Range | Unit |
|---|---|---|
| Circulation fan speed | 0 – 100 | % of max RPM |
| CO₂ scrubber duty cycle | 0 – 100 | % |
| O₂ supply valve | 0 – 100 | % open |
| Dehumidifier duty cycle | 0 – 100 | % |

### Agent Modes

- **Reactive (baseline):** Rule-based thresholds — deterministic, fully auditable, used as the baseline to beat
- **Learned (AI):** Trained or prompted model (e.g. RL policy or LLM reasoning agent) that must be compared against the reactive baseline on identical scenarios before any improvement is claimed

### Rationale Log Schema

```json
{
  "tick": 142,
  "timestamp": "2026-07-20T14:00:00Z",
  "observations": { "co2_ppm": 1450, "o2_pct": 20.1, "pressure_hpa": 1012.3, "temp_c": 22.5, "humidity_pct": 55.2 },
  "state_class": "WARNING",
  "action": { "fan_pct": 80, "scrubber_pct": 90, "o2_valve_pct": 10, "dehumidifier_pct": 0 },
  "rationale": "CO₂ above 1400 ppm threshold; increasing scrubber and fan. O₂ nominal — valve held low.",
  "active_faults": []
}
```

---

## Actuator Layer

Actuators are controlled via GPIO / PWM from the host microcontroller. Each actuator has:

- A **hard saturation limit** enforced in firmware — the agent cannot exceed it regardless of command
- A **state reporter** that feeds back actual position/speed to the evaluation layer
- A **fault flag** raised on unexpected state (e.g. fan not responding to command)

---

## Scenarios & Testing

| Scenario | What Changes | What It Tests |
|---|---|---|
| Nominal operation | Stable occupancy, all actuators healthy | Steady-state control, baseline logging |
| Crew activity spike | Occupancy or activity increases sharply | Agent response speed, CO₂ and O₂ coupling |
| Fan failure | Fan stops responding to commands | Fault detection, graceful degradation, alerting |
| Scrubber degradation | Scrubber efficiency drops gradually | CO₂ drift detection, adaptive scrubber + fan response |
| Sensor dropout | One or more sensors stop reporting | Fault classification, safe fallback behaviour |
| Biased sensor | Sensor reports plausible but wrong values | Anomaly detection, cross-sensor consistency checks |
| Slow leak | Gradual pressure/O₂ loss | Pressure coupling, O₂ supply response, alert escalation |

### Minimum Evidence per Run

- Scenario ID, hardware/firmware revision, sensor calibration file version, random seed (if any)
- Full time-series log: sensor readings, agent observations, state classifications, actions, active faults
- Outcome: `PASS` / `CONTROLLED DEGRADATION` / `AGENT FAILURE` / `SENSOR FAULT`
- AI runs must include structured rationale — never free text only
- AI performance must be compared to the reactive baseline on the same scenario before claiming improvement

---

## Repository Structure

```
arm-hackathon/
├── README.md
├── docs/
│   └── base-plan-v0.1.pdf         # Origin design document
├── params/
│   ├── registry.json               # Parameter registry with units + sources
│   └── sensor_calibration.json     # Per-sensor offsets and thresholds
├── scenarios/
│   ├── nominal.json
│   ├── crew_spike.json
│   └── fan_failure.json
├── sensing/
│   ├── __init__.py
│   ├── reader.py                   # Sensor polling + timestamping
│   ├── calibration.py              # Offset correction + unit conversion
│   └── fault_detector.py          # Dropout, out-of-range, and consistency checks
├── agent/
│   ├── __init__.py
│   ├── baseline.py                 # Reactive rule-based controller
│   ├── ai_agent.py                 # AI controller (RL / LLM / other)
│   └── logger.py                   # Structured rationale + run log writer
├── actuation/
│   ├── __init__.py
│   ├── controller.py               # GPIO/PWM command dispatch + saturation
│   └── state_reporter.py          # Actuator feedback + fault flags
├── evaluation/
│   ├── runner.py                   # Scenario runner
│   └── metrics.py                  # Target tracking, fault stats, comparison report
├── tests/
│   ├── test_sensor_faults.py
│   ├── test_actuator_bounds.py
│   └── test_agent_baseline.py
└── requirements.txt
```

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/akurkar07/arm-hackathon.git
cd arm-hackathon

# Install dependencies
pip install -r requirements.txt

# Run unit tests
python -m pytest tests/

# Start a scenario run (hardware connected or emulated)
python -m evaluation.runner --scenario scenarios/nominal.json
```

> **Prerequisites:** Python 3.10+, `numpy`, `pandas`, `RPi.GPIO` (or GPIO emulator for desktop dev), plus any sensor-specific libraries.

---

## Parameter Registry

All tunable values live in `params/registry.json`. No value enters a release without:

| Field | Description |
|---|---|
| `unit` | SI unit or % |
| `value` | Current value or `"TBD"` |
| `source` | Datasheet, measurement, or `"synthetic"` |
| `confidence` | `high` / `medium` / `low` / `synthetic` |
| `notes` | Context, limits, calibration cadence |

Key parameters to populate before first hardware run:

| Parameter | Unit | Status |
|---|---|---|
| `co2_warning_threshold` | ppm | TBD |
| `co2_alert_threshold` | ppm | TBD |
| `o2_min_threshold` | % | TBD |
| `pressure_nominal` | hPa | TBD |
| `control_cadence` | s | TBD |
| `fan_max_rpm` | RPM | TBD |
| `scrubber_max_flow` | L/min | TBD |

---

## References

- [NASA NTRS 19930018529 — Environmental Control and Life Support System](https://ntrs.nasa.gov/citations/19930018529)
- [NASA NTRS 20170006211 — Environmental Control and Life Support Systems](https://ntrs.nasa.gov/citations/20170006211)
- [SCD40 CO₂ Sensor Datasheet — Sensirion](https://sensirion.com/products/catalog/SCD40/)
- [MH-Z19C CO₂ Sensor Datasheet — Winsen](https://www.winsen-sensor.com/sensors/co2-sensor/mh-z19c.html)

---

## Disclaimer

This is a **simulation/prototype-only** research and demo project built for a hackathon. No results should be interpreted as guidance for real spacecraft environmental control, crew safety, or certified life-support design. All sensor thresholds and hardware numbers are placeholders pending team review and must not be used in any safety-critical application.
