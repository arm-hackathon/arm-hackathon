# Space Habitat Ventilation AI

> **⚠️ Simulation only — not a certified life-support design.**  
> This project simulates a hypothetical habitat. It must not be presented as real ECLSS design, human-safety guidance, flight software, or a substitute for certified engineering review.

A physics-led simulator using ordinary differential equations (ODEs) to model the atmosphere of a pressurised space habitat. An AI ventilation controller reads simulated sensor observations and proposes bounded control actions; the simulator remains the sole source of environmental truth.

---

## Table of Contents

- [Objective](#objective)
- [Architecture](#architecture)
- [System Model](#system-model)
- [Build Sequence](#build-sequence)
- [Scenarios](#scenarios)
- [Verification Gates](#verification-gates)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Parameter Registry](#parameter-registry)
- [References](#references)
- [Disclaimer](#disclaimer)

---

## Objective

Create an interactive simulation in which an AI ventilation controller must keep a virtual pressurised cabin within defined environmental targets while faults and crew-driven loads occur. The project demonstrates **closed-loop reasoning, traceability, and failure handling** — not operational suitability.

### Success Criteria

- Deterministic, replayable ODE simulation of the selected habitat scope
- Clear separation between physical state, noisy sensors, AI decisions, and actuator effects
- Scenario runs exposing normal operation, degraded ventilation, increased crew load, and component failures
- Mass-balance and safety-invariant tests pass before the AI layer is evaluated
- Team-readable report/dashboard showing state over time, events, actions, and outcome reasoning

---

## Architecture

```
Scenario + parameter set
        |
        v
Physics engine (ODE derivative + numerical integrator) ----> event log / ground-truth trace
        |                                                              ^
        v                                                              |
Sensor model (sampling, delay, noise, dropout)                         |
        |                                                              |
        v                                                              |
AI controller / deterministic baseline ---------> bounded actuator commands
        |
        v
Evaluation layer (targets, invariants, scenario score, replay)
```

| Layer | Owns | Must NOT own |
|---|---|---|
| Physics engine | State evolution, units, event application, ground truth | AI policy, UI formatting, safety claims |
| Sensor model | What the controller can observe and when | Changing ground truth to make controller look better |
| AI / rules controller | Action selection and rationale from observations | Writing directly to state or bypassing actuator constraints |
| Evaluator | Scenario pass/fail, metrics, replay evidence | Silently repairing invalid simulation output |
| UI / report | Readable traces and comparison views | Embedding hidden simulation logic |

---

## System Model

The simulator state is a time-indexed vector `y(t)`. Each derivative is computed from known sources, sinks, flows, actuator settings, and faults.

| State / Input | Meaning | v1 Treatment |
|---|---|---|
| `n_CO2` | Amount of CO₂ in cabin air | Crew generation; ventilation removal; scrubber removal |
| `n_O2` | Amount of O₂ in cabin air | Crew consumption; generator supply; leak/outflow |
| `P` | Cabin pressure | Derived from total gas, temperature, volume, and leaks |
| `T` | Cabin temperature | Simplified heat load and ventilation/HVAC coupling |
| `w` | Water-vapour / humidity | Crew moisture load; dehumidification coupling |
| `u(t)` | Actuator command vector | Fan/flow setpoint, scrubber setting, O₂ supply; bounded |
| `d(t)` | Disturbance and fault vector | Crew count/activity, leak, fan degradation, scrubber loss |

### Representative Conservation Equation

```
d(n_CO2)/dt = crew_CO2_generation(t) + inflow_CO2(t) - outflow_CO2(t) - scrubber_removal(t)
```

Concentration is derived from gas amount and cabin volume. All parameters must have explicit units and documented sources — no hard-coded plausible-looking values.

### Integrator Contract

- Derivative as a pure function: `f(time, state, controls, disturbances, parameters) → state_derivative`
- Configurable numerical solver — start with [`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html); record solver, tolerances, and time grid in each run artifact
- Controller observations evaluated on a fixed cadence even if the ODE solver uses adaptive internal steps
- Non-negativity and conservation validated after every integration window; run labelled **invalid** if any invariant breaks

---

## Build Sequence

| Phase | Deliverable | Exit Gate |
|---|---|---|
| **P0 — Contract** | Parameter registry, scenario schema, equations, assumptions | Every required parameter has a cited source or explicit `TBD`; team approves scope |
| **P1 — Plant** | Deterministic single-zone ODE simulator | Conservation / bounds / replay tests pass |
| **P2 — Baseline** | Rules controller + observability trace | Baseline handles nominal scenario; failure modes visible |
| **P3 — AI** | AI adapter with bounded actions and rationale | AI compared against baseline on held-out fault scenarios |
| **P4 — Fidelity** | Optional zones, transfer flows, richer sensors | v1 regression-green; new complexity earns a measured benefit |

### v1 Scope

**Included:** One well-mixed pressurised cabin; CO₂, O₂, pressure, temperature, and humidity state variables; crew loads, ventilation flow, removal/generation equipment, bounded leak/fault events; deterministic rules baseline.

**Excluded:** CFD / duct geometry / spatial turbulence; real-life habitat-safe thresholds or certification-grade parameter values; direct hardware connection; training an AI model from scratch.

---

## Scenarios

| Scenario | What Changes | What It Proves |
|---|---|---|
| Nominal operation | Stable crew load and healthy actuators | Plant stability, controller cadence, report baseline |
| Crew-load step | Activity/occupancy increases at a known time | Response to a changing source term |
| Fan degradation | Available flow falls gradually or suddenly | Detection, bounded mitigation, escalation reporting |
| Scrubber degradation | Removal capacity decreases | CO₂ mass balance and controller adaptation |
| Sensor fault | Delay, dropout, or biased reading | Separation of observable state from ground truth |
| Leak event | Additional outflow begins | Pressure/gas coupling and invalid-state protection |

### Minimum Evidence per Run

- Scenario and parameter-set identifiers, code revision, solver/tolerance settings, random seed
- Time series for state, observed sensors, actuator commands, active faults, and invariant status
- Outcome line: `pass` / `controlled degradation` / `invalid simulation` / `controller failure`
- For AI runs: action rationale in structured fields — never only free text

---

## Verification Gates

| Gate | Required Proof | Failure Handling |
|---|---|---|
| Mass balance | Closed-system test: no unexplained creation or loss | Block controller evaluation; fix plant equations first |
| Physical bounds | Amounts, concentrations, capacities within declared bounds | Mark run invalid; no clipping to hide errors |
| Determinism | Same params + seed reproduces same trace within tolerance | Record solver/version drift and investigate |
| Event timing | Fault/command changes at exact scheduled control boundary | Fix scheduler/event ordering before scoring AI |
| Baseline comparison | AI compared to deterministic controller over same scenarios | Do not claim improvement without paired traces |
| Replayability | Fresh process reproduces a saved run from artifact bundle | Treat result as anecdotal, not evidence |

---

## Repository Structure

```
arm-hackathon/
├── README.md
├── docs/
│   └── base-plan-v0.1.pdf        # Source design document
├── params/
│   └── registry.csv              # Versioned parameter registry (Appendix A)
├── scenarios/
│   ├── nominal.json
│   └── fan_degradation.json
├── simulator/
│   ├── __init__.py
│   ├── physics.py                # ODE derivative (pure function)
│   ├── integrator.py             # solve_ivp wrapper + invariant checks
│   ├── sensor_model.py           # Noise, delay, dropout
│   └── actuators.py              # Bounded actuator model
├── controllers/
│   ├── baseline.py               # Deterministic rules controller
│   └── ai_agent.py               # AI controller adapter
├── evaluation/
│   ├── runner.py                 # Scenario runner + report
│   └── metrics.py                # Time-above-target, excursion, invariant failures
├── tests/
│   ├── test_mass_balance.py
│   ├── test_bounds.py
│   └── test_determinism.py
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

# Run the P1 plant tests (must be green before adding a controller)
python -m pytest tests/

# Run the nominal scenario
python -m evaluation.runner --scenario scenarios/nominal.json
```

> **Prerequisites:** Python 3.10+, `scipy`, `numpy`, `pandas`, `matplotlib` (or similar plotting library).

---

## Parameter Registry

All parameters live in `params/registry.csv`. No parameter enters a release scenario without the following metadata:

| Parameter | Unit | Value / Range | Source | Confidence | Notes |
|---|---|---|---|---|---|
| `cabin_volume` | m³ | TBD | TBD | TBD | Defined model geometry |
| `crew_CO2_generation` | TBD | TBD | TBD | TBD | Function of activity/load |
| `crew_O2_consumption` | TBD | TBD | TBD | TBD | Function of activity/load |
| `scrubber_capacity` | TBD | TBD | TBD | TBD | Virtual equipment model |
| `fan_flow_capacity` | TBD | TBD | TBD | TBD | Virtual actuator limit |
| `leak_rate` | TBD | TBD | TBD | TBD | Scenario-specific disturbance |

---

## References

- [NASA NTRS 19930018529 — Environmental Control and Life Support System](https://ntrs.nasa.gov/citations/19930018529) — ECLSS subsystem decomposition
- [NASA NTRS 20170006211 — Environmental Control and Life Support Systems](https://ntrs.nasa.gov/citations/20170006211) — Ventilation/environmental-control framing
- [SciPy `solve_ivp` API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) — ODE integration interface

---

## Disclaimer

This is a **simulation-only** research/demo project. Parameter values are synthetic unless explicitly cited. No results should be interpreted as guidance for real spacecraft environmental control, crew safety, or certified life-support design. All thresholds and hardware numbers are placeholders pending team review.
