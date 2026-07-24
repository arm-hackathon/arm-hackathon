# Changelog

## 2026-07-24 — scenario and replay validation hardening

- Bumped the scenario format from version 5 to version 6 because unknown
  scenario fields now fail validation rather than being ignored.
- Added strict unknown-field validation across scenario blocks, zones,
  connections and occupancy periods.
- Tightened visualiser input validation for replay tick sequence, stable entity
  ids, airflow bounds and connection health.

## 2026-07-22 — `model-improvement` compared with `main`

### Simulation and control

- Added proportional CO₂ control that converts each zone's sensor concentration into a bounded actuator setpoint.
- Added rate-limited actuator dynamics: actual position now moves gradually towards its setpoint, with direction, movement time, tracking residual and power use recorded.
- Changed airflow to use actual actuator position rather than the requested setpoint.
- Reworked the plant around CO₂ mass and air volume, deriving concentration for sensing and reporting.
- Connected all room loops through a shared processing stream and a 24-unit fan capacity. Competing requests are scaled proportionally, extracted air is mixed, 50% of its CO₂ is scrubbed, and the remainder is redistributed by allocated airflow.
- Made both legs of each loop matter: the weaker outbound or return connection limits its airflow.
- Preserved simultaneous extraction and exact mass conservation so zone update order does not affect results.

### Dynamic scenario

- Upgraded the scenario format from version 1 to version 5.
- Added required `control`, `actuator`, `simulation` and `air_system` configuration blocks, with validation for finite values, bounds and valid ranges.
- Added per-zone source variation, correlation and occupancy schedules. Randomness is seeded and reproducible, and is independent of graph iteration order.
- Added stronger, correlated variation for the cabins and laboratory so demand visibly changes over time.
- Added a genuine 60-tick warm-up. The full plant and controller run before recording, then the visible tick and captured-CO₂ counter are reset while the settled physical state is retained.
- Kept the measured trace at ticks 1–120 and changed its initial state from zero to realistic post-warm-up concentrations.

### Trace and visualisation

- Added a dependency-free British-English visualiser at `python -m icarus.visualise <trace> <report>`.
- The self-contained HTML report shows occupancy, source mass, concentration, requested and allocated airflow, shared capacity, actuator response, power, connection health and captured CO₂, with responsive charts and hover values.
- Added strict trace validation with line-specific errors for malformed or non-finite data.
- Expanded trace records with actuator and system sections.
- Renamed ambiguous CO₂ fields to distinguish mass from concentration, and added source mass, occupancy multiplier, requested airflow, capacity scaling and actuator telemetry.
- Updated the CLI summary to report zone CO₂ concentrations.
- Regenerated `traces/standard_habitat.jsonl` using the new schema and dynamics.

### Documentation and repository

- Added `PLAN.md` describing the objective, required work, completion criteria and project scope.
- Updated the README to describe the current simulation and control foundation and corrected the usage commands.
- Rewrote the simulation rules for the version 5 model.
- Standardised repository prose and public project terminology on British English.
- Added `out/` to `.gitignore` for generated visualisation reports.

### Tests and verification

- Expanded the test suite from 58 to 100 passing tests.
- Added dedicated controller, actuator and visualiser tests, and extended configuration, plant, scenario and trace coverage.
- Added checks for rate limiting, gradual closing, bounds, occupancy transitions, deterministic noise and traces, shared-capacity contention, mixed return coupling, return-path bottlenecks, warm-up behaviour and full-run mass conservation.
- In the standard measured run, peak concentrations are approximately 0.276 in Cabin A, 0.298 in Cabin B and 0.158 in the lab, against a 0.30 cabin ceiling. Shared capacity is constrained for 73 of 120 ticks.

### Still to implement

- Fault injection, recovery and degraded-operation demonstrations.
- A supervisory optimisation or AI policy, including any ONNX deployment path.
- Redundant-fan or higher-level plant coordination.
- Arm hardware deployment, performance measurements and evidence for the final pitch.
