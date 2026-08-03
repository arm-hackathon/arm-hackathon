# Changelog

## 2026-08-03 — bounded recovery response (development evidence)

- Added a pure override hook to `step_habitat` (`override_commands`) that lets
  an external decision maker issue bounded per-zone commands while the default
  path stays byte-identical to the previous AEOLUS behaviour; conserved
  contract that only the physics engine changes plant state.
- Added `src/aeolus/response.py`: the `BoundedRecoveryGovernor`, a causal,
  observable-only decision maker over exact `model_input_v1` windows with
  declared `ResponseSettings`, four bounded rules (proportional demand,
  frozen-sensor hold, degraded-loop spare-capacity release, rate/energy
  bounds), and per-tick structured rationale.
- Added `run_governed_scenario` with warm-up window seeding so the governor's
  first measured command matches the baseline observation basis.
- Added `scenarios/sweep-response.json` (v3 development, 129 families across
  two operating profiles) and `src/aeolus/response_evidence.py`, a hashed
  evidence harness reporting time-above-ceiling, max excursion, response
  latency, energy and invariant violations for both controllers.
- Response evidence: 117/129 fault families at exact baseline parity; 128/129
  within the one-tick causality margin (one family exceeded it at +2 ticks;
  the baseline is same-tick by design, so the causal governor reports the
  margin explicitly); healthy-reference runs within the margin on all 129;
  zero invariant violations under both controllers; median energy overhead
  0.00%. Evidence-driven iteration measured that boosting or unconditionally
  throttling degraded loops both worsen time-above-ceiling; recorded in
  `docs/bounded-response.md`.
- Guarded the v2 `run_experiment` path against v3 development sweep layouts
  (raise a clear error instead of a silent split failure).
- Full suite: 339 tests passed at commit time.

## 2026-08-03 — response evidence review fixes (`yarofix2`)

- Fixed a self-contradictory conclusion: the aggregate reported one fault
  family beyond the one-tick causality margin (max excess 2 ticks), but the
  conclusion hard-coded "within the margin on all N" and claimed zero
  invariant violations regardless of the aggregate. The conclusion now
  derives both the within-margin counts and the violation counts from the
  aggregate. Regenerated receipt: 128/129 within the margin (1 beyond at +2),
  healthy 129/129, 0/0 invariant violations.
- Scoped `response_latency_ticks` to actions on the zone loop that the fault
  profile actually degrades; rate limits or holds on unrelated healthy loops
  no longer count as a response. The affected-zone set is derived from the
  fault profiles (connection faults map to the served command zone, sensor
  faults to the frozen zone), falling back to all commanded zones only when
  no match exists. Median response latency re-measured at 67 ticks.
- Fixed documentation drift: branch references (`ben/bounded-response`) now
  read `yarofix2`, the README deferred-work bullet no longer defers the
  implemented governor, and the README margin table reports 128/129 instead
  of 129/129.
- Full suite: 343 tests passed at review-fix commit time.

## 2026-08-03 — protocol-v3 frozen final evaluation

- Retired the inspected v2 test and stress partitions as current decision
  inputs. Added a 360-family train / 120-family validation development suite and
  a fresh 180-family final suite with role-specific split validation.
- Added a strict policy that captures validation-only candidate selection, rule
  calibration, model/rule comparison and ONNX parity. Final evaluation replays
  candidate training, selection, ONNX parity and calibration from verified
  development evidence before it evaluates final rows.
- Final evidence is negative: temporal MLP macro-F1 `0.5754744477098027` versus
  rule macro-F1 `0.642588422763726`; nominal false-alarm rate `38.5698%` versus
  `0.5631%`; median detection latency 9 versus 10 simulator ticks. The frozen
  preferred method remains `rule_baseline` and no AI advantage is claimed.
- Documented that 8,000 scored windows are correlated observations within 180
  final scenario families. No independent-window uncertainty, OOD robustness,
  wall-clock performance, hardware or deployment claim is made.

## 2026-08-01 — historical schema-v9 temporal experiment (superseded for final evidence)

- Added deterministic airflow drift and controller-facing CO2 sensor noise,
  bias and drift. Frozen faults now hold latent readings while downstream
  readout effects continue, removing the exact-constant shortcut.
- Added `sweep-v2` with 360 train, 120 validation, 180 IID test and 180 OOD
  stress families sharing fixed primary distributions and disjoint seeds.
- Added `temporal_summary_v1`, a deterministic 135-16-4 NumPy MLP, validation
  selection against softmax, and an embedded FP32 ONNX transform/model graph.
- Added validation-only calibration for the 216-point robust rule grid and
  stride-one causal latency. Replaced the impossible advantage target with a
  fixed error-reduction-or-latency policy.
- Added `aeolus.experiment` for one-command sweep, corpus, training, evaluation
  and artifact export.
- The honest IID result remains negative: temporal-MLP macro-F1 0.5765 versus
  calibrated-rule 0.6410, with 35.36% versus 2.53% nominal false alarms. The
  model is faster by median latency (9 versus 11 ticks) but misses the fixed
  20% threshold. Stress evidence also favours rules. No INT8 or Arm claim is
  made.

## 2026-08-01 — schema-v8 noise and experimental fault detector

- Added required schema-v8 telemetry settings and deterministic SHA-256-derived
  actuator/airflow measurement noise and bias without mutating physical state
  or expanding the replay/model feature allowlists.
- Added `sweep-v1` generation for 558 family-held-out training, validation and
  test families, with paired scenarios differing only in `fault_profiles`.
- Added deterministic class-balanced four-class softmax training over
  `float32[10,24]` corpus-v2 windows, strict JSON loading, rolling prediction,
  and FP32 ONNX export with contract metadata and probability parity checks.
- Added reproducible model and metrics artifacts. Held-out evidence does
  not demonstrate an AI advantage: macro-F1 is 0.5503 versus 0.9872 for the
  rule baseline, so the recorded preferred method remains the rule.
- No INT8 or Arm performance claim is made.

## 2026-07-27 — rule baseline and evaluation harness

- Added `aeolus.baseline.RuleBaseline`: a streaming rule detector over
  model-feature windows. Zero-variance sensor runs mark frozen sensors;
  isolated residual loss (one loop above its sisters, persistent) marks a
  delivery fault; a remembered onset jump separates blockage from gradual
  degradation. Shared-capacity contention does not false-fire because faults
  must be isolated, not merely elevated.
- Added `aeolus.evaluate`: one harness grades any window labeller (rules
  today, the classifier later) on accuracy, per-class support, confusion and
  detection latency against declared fault starts.
- On corpus v1 the baseline scores 111/115 windows (all four misses are
  onset-boundary windows) with detection latencies of 10/5/10 ticks for
  degradation/blocked/frozen — the bar the classifier must beat.

## 2026-07-27 — blocked-path and frozen-sensor faults plus labelled corpus

- Added two schema-v7 fault profile types: `blocked_path` (a sudden step loss
  of delivery effectiveness on an outbound loop) and `frozen_sensor` (a zone
  sensor that holds its first frozen reading while the truth evolves).
- Added `scenarios/blocked_path.json` and `scenarios/frozen_sensor.json`,
  paired with the high-demand baseline habitat; the frozen-sensor scenario
  steps lab demand down after the freeze so the held reading diverges from
  observable reality.
- Added `aeolus.corpus`: a leakage-safe labelled window corpus over all five
  shipped scenarios. Features are exactly the `model_feature_row()` allowlist,
  labels come from declared fault profiles at each window's final tick, and
  regeneration is byte-identical.
- Documented the new fault semantics and the corpus boundary in
  `docs/simulation-rules.md` and `docs/telemetry-contract.md`.
- Aligned `PLAN.md` with the converged design: health and fault truth are
  deliberately hidden rather than recorded, metered airflow is a declared
  abstraction, and capacity-contention and orchestration-pair scenarios are
  explicitly deferred to the governor slice.

## 2026-07-25 — schema-v7 simulation convergence

- Converged current `main` into the advanced deterministic simulation foundation.
- Added closed-schema v7 `fault_profiles` with a validated,
  deterministic `gradual_primary_fan_degradation` profile.
- Separated controller-requested airflow from physical delivered airflow and
  recorded an explicit airflow residual. Static path health, fault
  effectiveness and shared capacity now reduce delivery only.
- Added healthy standard, healthy high-demand and gradual-primary-fan
  degradation scenarios.
- Preserved deterministic warm-up and replay behaviour, mass conservation,
  paired loop delivery and the standalone visualiser.
- Added strict trace/model telemetry boundaries: health and fault truth remain
  hidden, while model-facing data uses an explicit allowlist.
- Updated visualisation, documentation, repository ignores and regression tests
  for the v7 trace contract.

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

- Added a dependency-free British-English visualiser at `python -m aeolus.visualise <trace> <report>`.
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
