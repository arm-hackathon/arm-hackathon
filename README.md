# Space Habitat Ventilation AI

> Submission for the [Arm Create: AI Optimization Challenge 2026](https://arm-ai-optimization-challenge.devpost.com/) — **Physical AI track**

An AI ventilation controller for a pressurised space habitat, running fully on an Arm-powered edge device. Sensors feed real-time CO₂, O₂, pressure, temperature, and humidity readings to an on-device AI agent that classifies cabin state and issues bounded commands to fans, scrubbers, and O₂ supply — with all inference running locally, optimised for Arm.

---

## Why This Fits the Physical AI Track

The Arm AI Optimization Challenge asks for projects that demonstrate **AI optimization on Arm-powered platforms** in real-world, physical contexts — robotics, embedded devices, sensors, and edge environments. This project delivers:

- **On-device inference** — the AI agent runs entirely on an Arm64/Cortex-M/A-class board (e.g. Raspberry Pi 5, Jetson Orin Nano), no cloud call needed
- **Measurable optimization** — latency, model size, and inference throughput benchmarked with [Arm Performix](https://developer.arm.com/)
- **Real physical loop** — sensors → AI reasoning → actuator commands → environment change → repeat
- **Safety-critical context** — demonstrates that edge AI can operate reliably under fault conditions

---

## System Overview

```
Sensors (CO₂ / O₂ / P / T / RH)
         │
         ▼
   Sensor Layer  ←—— calibration, dropout detection
         │
         ▼
    AI Agent  ←——— runs on Arm device (quantized model / rule-hybrid)
         │           classifies state → selects action → logs rationale
         ▼
  Actuator Layer →—— fan speed, scrubber duty, O₂ valve (PWM/GPIO)
         │
         ▼
  Evaluation & Logging
```

| State Variable | Sensor | Interface |
|---|---|---|
| CO₂ (ppm) | SCD40 / MH-Z19C | I²C / UART |
| O₂ (%) | ME2-O2 | Analog |
| Pressure (hPa) | BMP390 | I²C |
| Temperature (°C) | SHT40 | I²C |
| Humidity (% RH) | SHT40 | I²C |

---

## AI Agent

The agent runs on a fixed cadence (configurable, default 5 s). Each tick:

1. **Observe** — read normalised sensor vector
2. **Classify** — `NOMINAL` / `WARNING` / `ALERT` / `FAULT`
3. **Act** — output bounded actuator setpoints (0–100% per channel)
4. **Log** — structured JSON rationale per decision

### Arm Optimization Focus

- Model quantized to **INT8** for fast inference on Arm Cortex / NEON pipelines
- Benchmarked with **Arm Performix** for tokens/sec and latency
- Fallback to a deterministic rule-based controller if inference exceeds latency budget
- Target: full sense→decide→act loop under **100 ms** on target hardware

---

## Repo Structure

```
arm-hackathon/
├── agent/            # AI controller + baseline rule controller
├── sensing/          # Sensor polling, calibration, fault detection
├── actuation/        # GPIO/PWM dispatch + actuator state reporting
├── evaluation/       # Scenario runner, metrics, comparison report
├── params/           # Parameter registry + sensor calibration files
├── scenarios/        # Scenario definitions (nominal, fault, spike)
├── tests/            # Sensor fault, actuator bounds, agent baseline tests
├── benchmarks/       # Arm Performix results + latency logs
└── requirements.txt
```

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

**Target hardware:** Raspberry Pi 5 / any Arm64 or Cortex-A board running Linux.  
**Python:** 3.10+

---

## Scenarios

| Scenario | Fault Injected | Tests |
|---|---|---|
| Nominal | None | Steady-state control, latency baseline |
| Crew activity spike | CO₂ / O₂ load increase | Response speed and coupling |
| Fan failure | Fan unresponsive | Fault detection, graceful degradation |
| Sensor dropout | Sensor stops reporting | Safe fallback, fault classification |
| Slow pressure leak | Gradual O₂ / pressure loss | Alert escalation, O₂ valve response |

---

## License

MIT — see [LICENSE](LICENSE)

---

> ⚠️ **Simulation / prototype only.** Not a certified life-support design. Thresholds and hardware values are placeholders; do not use in any safety-critical application.
