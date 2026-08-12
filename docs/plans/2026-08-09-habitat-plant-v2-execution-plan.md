# AEOLUS Habitat Plant V2: Decision-Complete Execution Plan

Date: 2026-08-09
Status: execution candidate pending one bounded independent review
Canonical implementation base: `ben/recovery-policy-candidate` at `89ff12499e395cb732e2cd175af98e021cfdb831`
Planned branch: `ben/habitat-plant-v2-foundation`
Version impact of the first code slice: minor, from 0.2.3 to 0.3.0

## 1. Plain-language objective

Build a new, deterministic habitat simulator that is physically rich enough to make future action-conditioned prediction meaningful.

When it runs, two occupied habitat zones compete for shared ventilation, carbon-dioxide removal, humidity removal, oxygen supply, cooling and electrical power. The simulator records what each zone contains, what each device actually did, which resources were consumed and whether every physical accounting rule remained true.

We know the first slice works when:

1. the same scenario produces byte-identical trace files;
2. every gas, water and energy change has an explicit source, sink or transfer;
3. no action can exceed actuator, processing, inventory or power limits;
4. changing an action changes future CO2, temperature or humidity in the physically expected direction;
5. V1 tests and accepted deterministic-recovery evidence remain untouched;
6. a real command runs the reference Habitat V2 scenario and writes a validated JSONL trace.

## 2. Why this precedes the model

A learned model cannot be evaluated honestly against a world whose variables, units, causal actions and hidden parameters are still changing. Habitat V2 therefore blocks model training until its physical contract, trace contract, parameter ledger, deterministic baselines and observability tests are frozen.

“Perfect” is not an attainable engineering gate. The replacement gate is measurable qualification:

- conservation and invariant tests pass;
- parameter provenance is declared;
- reference and stress scenarios behave plausibly under sensitivity checks;
- strong deterministic baselines are implemented;
- identical observable histories do not hide materially incompatible futures without uncertainty or abstention handling;
- one bounded independent review accepts the frozen candidate.

## 3. Repository truth and preservation boundary

The current repository has a dependent PR stack:

- PR 21 contains integrated bounded-recovery work;
- PR 22 adds the accepted blind deterministic-recovery verification;
- PR 23 archives a rejected temporal early-risk model.

Habitat V2 must branch from PR 22’s head, `ben/recovery-policy-candidate`, not from `main` and not from PR 23. This preserves the accepted deterministic safety baseline while excluding the rejected learned candidate.

V1 remains immutable for evidence and replay purposes:

- existing scenario versions 9 and 10 remain accepted by `aeolus.config`;
- existing `aeolus.plant`, recovery supervisor, traces and evidence are not widened;
- V1’s abstract units are not relabelled as SI units;
- existing artifacts and decision records are not regenerated;
- Habitat V2 receives separate package, schema and trace identities.

## 4. Options considered

### Option A: widen the existing V1 plant in place

Advantages:
- least new package structure;
- existing runner and visualiser could be reused directly.

Rejected because:
- V1’s abstract-unit equations cannot be silently treated as physical SI quantities;
- changing state and trace fields would invalidate reproducibility assumptions;
- recovery evidence would become difficult to interpret against changed plant semantics;
- a large conditional maze would be needed to preserve schema versions 9 and 10.

### Option B: add `aeolus.habitat_v2` beside V1

Advantages:
- clean physical and schema boundary;
- accepted V1 evidence remains replayable;
- V2 can use SI units, explicit inventories and stricter contracts from the start;
- Yaro’s frontend can consume one stable versioned JSON contract;
- comparison between V1 and V2 remains possible without pretending they are the same plant.

Chosen.

### Option C: create a separate repository

Advantages:
- complete isolation;
- no legacy package constraints.

Rejected because:
- splits project history and team review;
- duplicates packaging, CI and documentation;
- complicates frontend integration and comparisons;
- loses useful deterministic safety infrastructure.

## 5. V2 system boundary

### 5.1 Initial topology

The smallest credible reference plant has:

- `crew_cabin`: occupied sleeping/living zone;
- `work_airlock`: occupied work and EVA-transition zone;
- `utility`: central air-processing and shared-resource bay.

Only the first two zones hold independently controlled environmental state. The utility bay owns processing equipment and resources. A third occupied zone is deferred until sensitivity analysis shows that two zones cannot create the required allocation and forecasting difficulty.

The engine must not hard-code those names. Scenario configuration defines occupied zones, while the first schema deliberately allows exactly two occupied zones and one utility bay. This constrains the validation burden without baking demonstration labels into physics.

### 5.2 State

Each occupied zone stores:

- carbon-dioxide amount, mol;
- oxygen amount, mol;
- inert-gas amount, mol;
- water-vapour amount, mol;
- lumped zone temperature, K;
- actual recirculation airflow, m3/s.

The utility state stores:

- sorbent loading, mol CO2;
- condensed-water inventory, mol H2O;
- oxygen-store inventory, mol O2;
- battery energy, Wh;
- actual scrubber and condenser duty;
- cumulative external heat rejected, J;
- cumulative external heat received, J.

Derived values are never independently mutable:

- total pressure from the ideal-gas relation;
- CO2 concentration in ppm;
- O2 partial pressure;
- water-vapour partial pressure;
- relative humidity from saturation vapour pressure;
- remaining sorbent, oxygen and battery fractions.

### 5.3 Exogenous loads

For each step and occupied zone, the scenario supplies:

- CO2 generation, mol/s;
- O2 consumption, mol/s;
- water-vapour generation, mol/s;
- sensible heat, W.

These represent occupancy and activity. They are causal scenario inputs, recorded in traces and available to deterministic baselines only if an equivalent operational signal would exist. Future model contracts must distinguish known schedules from hidden realised metabolic variation.

### 5.4 Commanded actions

The deterministic command boundary accepts:

- commanded recirculation airflow per occupied zone, m3/s;
- scrubber duty, 0..1;
- condenser duty, 0..1;
- cooling thermal-removal request per occupied zone, W, with electrical demand derived from a declared coefficient of performance;
- oxygen injection per occupied zone, mol/s.

The physics engine rate-limits actual airflow and processing duty. It rejects malformed, non-finite, out-of-range or resource-infeasible commands. It never silently clips or rescales a command at the authority boundary.

A later planner may choose only from a finite, versioned catalogue of prevalidated command plans. The learned model will never create raw commands.

## 6. Deterministic physical rules

### 6.1 Units and constants

V2 uses SI units at every public boundary. Constants and equations are classified in a provenance ledger as:

- physical constant;
- publicly sourced requirement or range;
- physics-derived quantity;
- engineering assumption;
- stress-test range.

No engineering assumption may be described as a NASA or flight value.

### 6.2 Initial gas inventory

For each zone:

`n_total = pressure_pa * volume_m3 / (R * temperature_k)`

Water-vapour moles are derived from configured relative humidity and saturation vapour pressure. CO2 and O2 moles are derived from declared fractions. Remaining moles become inert gas. Initial fractions must be non-negative and sum to less than one after water vapour.

### 6.3 Occupant source and sink update

Over one step of duration `dt`:

- add `CO2_rate * dt` to zone CO2;
- subtract `O2_rate * dt` from zone O2;
- add `H2O_rate * dt` to zone water vapour;
- add sensible heat to the thermal budget.

A step that would consume more oxygen than exists fails closed as an invalid physical state.

### 6.4 Recirculation, mixing and return

Each occupied zone is well mixed. For actual airflow `q` and volume `V`, the exchanged fraction over a step is:

`f = 1 - exp(-q * dt / V)`

The engine removes fraction `f` of every gas species from each zone simultaneously. Extracted streams mix in the utility bay. Processing then removes bounded CO2 and water. Remaining gas is returned in proportion to each zone’s extracted total gas amount. This preserves ordering independence and species accounting.

No gas disappears through “airflow”. Only an explicit scrubber, condenser, leak fault or external source/sink can change total species inventory.

### 6.5 CO2 scrubbing

CO2 captured in one step is the minimum of:

- CO2 present in the mixed extracted stream;
- configured maximum removal rate × actual scrubber duty × `dt`;
- remaining sorbent capacity.

Captured CO2 increments sorbent loading. Saturated sorbent cannot remove additional CO2.

### 6.6 Humidity removal and passive condensation

Active condenser removal is bounded by:

- water vapour present in the mixed extracted stream;
- configured maximum removal rate × actual condenser duty × `dt`.

Removed water increments condensed-water inventory.

After the thermal update, vapour exceeding saturation at the new temperature condenses passively. Passive condensate also increments condensed-water inventory. Relative humidity therefore cannot exceed 100% in accepted post-step state.

### 6.7 Oxygen supply

Oxygen injection is bounded by:

- per-zone and total injection-rate limits;
- available oxygen-store inventory;
- shared electrical feasibility where injection power is configured.

The first slice models stored oxygen, not electrolysis. Oxygen generation and hydrogen accounting are deferred until they can be added without an untracked mass sink.

### 6.8 Temperature and zone-thermal closure

The first slice uses a declared grey-box lumped thermal model, not CFD.

For each occupied zone:

- occupant sensible heat is an explicit source;
- commanded cooling is an explicit sink;
- passive conductance to a declared sink temperature is an explicit sink/source;
- recirculation transfers heat through the mixed return stream.

Recirculation heat exchange uses `rho_air * cp_air * q * (T_mixed - T_zone)` and is equal-and-opposite across the two-zone mixed stream. The zone temperature change is net heat divided by declared lumped thermal capacity.

Cooling commands declare thermal removal from each occupied zone. Cooling electrical demand is `cooling_removed_w / cooling_coefficient_of_performance`; the coefficient of performance is a positive engineering-assumption parameter. The external utility loop receives both the heat removed from the occupied zones and the cooling device's electrical energy.

Every step emits an SI-normalised zone-thermal receipt in joules:

- `metabolic_heat_added_j` per zone;
- signed `recirculation_heat_added_j` per zone, whose system sum must be zero within tolerance;
- `cooling_heat_removed_j` per zone;
- `passive_heat_rejected_j` per zone when the zone is warmer than the declared sink;
- `passive_heat_received_j` per zone when the declared sink heats the zone;
- `zone_thermal_energy_delta_j = thermal_capacity_j_per_k * (T_next - T_previous)` per zone.

For each zone, the residual is:

`zone_thermal_residual_j = zone_thermal_energy_delta_j - (metabolic_heat_added_j + recirculation_heat_added_j + passive_heat_received_j - passive_heat_rejected_j - cooling_heat_removed_j)`

The system residual is the sum of the zone residuals. Both must satisfy `abs(residual) <= max(1e-6 J, 1e-10 * max(1 J, receipt energy scale))`.

Utility equipment does not invisibly heat occupied zones in the first slice. All served electrical load energy, battery conversion losses, cooling heat removed and passive heat rejected are counted once in `external_heat_rejected_j`. Passive heat entering from the sink is counted separately in `external_heat_received_j`. Curtailed generation is electrical energy never accepted by the plant and is not counted as plant heat. Injected oxygen is assumed to arrive at the receiving zone temperature, so it carries no net sensible-heat term in this first contract.

### 6.9 Electrical power, battery and energy closure

Total electrical demand is the sum of:

- fixed utility load;
- fan demand derived from actual airflow;
- scrubber demand;
- condenser demand;
- cooling demand;
- oxygen-injection demand.

Every power term is converted over the step using `energy_wh = power_w * dt_seconds / 3600`. Scenario-supplied generation serves demand first. Remaining deficit draws from the battery, bounded by stored energy, discharge efficiency and maximum discharge power. Surplus enters the battery charger, bounded by capacity and maximum charge power. Surplus that cannot be accepted is explicitly recorded as curtailed generation. Charge and discharge efficiencies are declared in `(0, 1]`.

Every step emits a non-negative electrical receipt in Wh:

- `generation_wh`;
- each served device-load term and `served_load_wh` total;
- `battery_charge_input_wh` at the electrical bus;
- `battery_charge_stored_wh` added to storage;
- `battery_withdrawn_wh` removed from storage;
- `battery_bus_output_wh` delivered after discharge loss;
- `charge_conversion_loss_wh`;
- `discharge_conversion_loss_wh`;
- `curtailed_generation_wh`;
- signed `battery_energy_delta_wh`.

The identities are:

`battery_charge_stored_wh = charge_efficiency * battery_charge_input_wh`

`battery_bus_output_wh = discharge_efficiency * battery_withdrawn_wh`

`battery_energy_delta_wh = battery_charge_stored_wh - battery_withdrawn_wh`

`generation_wh + battery_withdrawn_wh = served_load_wh + battery_charge_stored_wh + curtailed_generation_wh + charge_conversion_loss_wh + discharge_conversion_loss_wh`

The electrical residual is the left side minus the right side and must satisfy `abs(residual) <= max(1e-12 Wh, 1e-10 * max(1 Wh, receipt energy scale))`. Charge and discharge are mutually exclusive within a step. Conversion losses are converted to joules and counted once in external utility heat rejection. Served electrical load energy is also counted once as utility heat. Cooling electrical load is included in served load, while the separately removed zone heat is also rejected externally, matching a heat-pump balance.

A command whose required demand cannot be served by generation plus the battery's efficiency-adjusted energy and discharge-power limits fails before plant state is advanced. The engine never makes an infeasible action look feasible by silently reducing it. Battery capacity and charge-power limits do not make an otherwise safe command fail when generation is exogenous; they produce explicit `curtailed_generation_wh`. Exact accepted boundary cases must still close both receipts.

### 6.10 Determinism and step atomicity

A step is a pure transformation of validated configuration, state, load and command. Validation occurs before mutation. The implementation uses immutable dataclasses. The same inputs produce the same output and accounting receipt.

## 7. Schema and integration architecture

### 7.1 Package

Create:

- `src/aeolus/habitat_v2/__init__.py`
- `src/aeolus/habitat_v2/config.py`
- `src/aeolus/habitat_v2/physics.py`
- `src/aeolus/habitat_v2/runner.py`
- `src/aeolus/habitat_v2/trace.py`
- `src/aeolus/habitat_v2/__main__.py`

V1 modules remain unchanged except package version metadata and documentation links.

### 7.2 Scenario schema

Use a separate top-level identity:

`"schema_version": "aeolus_habitat_v2_scenario_v1"`

The parser is closed-schema and rejects unknown fields. The reference file is:

- `scenarios/habitat_v2_reference.json`

It declares topology, initial conditions, equipment limits, electrical limits, deterministic run length, fixed time step and a piecewise-constant load/action timeline. Every numeric field includes its unit in the field name.

The scenario is frozen for a run after parsing. Runtime UI edits must create a new validated scenario document and run identity.

After strict parsing, the validator emits one normalised scenario object containing only accepted fields and canonical numeric values. Canonical scenario bytes are UTF-8 JSON produced with sorted object keys, compact separators and `allow_nan=False`. JSON object key order and source-file path are therefore not semantic inputs. Ordered arrays remain ordered because timeline and topology order can be meaningful. `scenario_sha256` is the lowercase SHA-256 of those canonical bytes.

Habitat V2 declares one immutable equation identity for this slice:

`"equation_contract_revision": "aeolus_habitat_v2_equations_v1"`

The deterministic `run_id` is the lowercase SHA-256 of canonical compact JSON containing exactly:

- `scenario_sha256`;
- `scenario_schema_version`;
- `trace_schema_version`;
- `equation_contract_revision`.

No timestamp, random UUID, file path, host value or mutable package-build metadata participates in run identity.

### 7.3 Trace schema

Use:

`"schema_version": "aeolus_habitat_v2_trace_v1"`

Each JSONL row contains:

- an identical immutable lineage object containing `run_id`, `scenario_sha256`, scenario schema version, trace schema version and equation-contract revision;
- step/time;
- per-zone observable environmental telemetry;
- commanded and actual actions;
- utility resource state;
- exogenous realised loads;
- per-step accounting receipt;
- invariant status.

Hidden future fault truth, latent degradation parameters and labels are excluded from the observable telemetry projection. A separate evaluator-only state may hold them later.

The first slice exposes only deterministic truth telemetry. No model feature allowlist is frozen until observability analysis is complete.

The trace writer uses sorted-key, compact, finite-only JSON serialisation. Before writing, and again when loading, the trace validator recomputes lineage from the parsed scenario and declared contract constants. Every row must contain exactly the same lineage and must match the recomputed scenario digest and run ID. Stale, path-derived, malformed or row-varying lineage fails closed.

### 7.4 CLI

Add:

`python -m aeolus.habitat_v2 <scenario.json> <trace.jsonl>`

The command validates the scenario, refuses to overwrite an existing trace, runs the deterministic scenario and prints final CO2 ppm, temperature, relative humidity, pressure and resource summaries.

The existing `python -m aeolus` V1 command remains byte-for-byte compatible.

## 8. TDD implementation sequence for the first vertical slice

All production code follows a demonstrated failing test first.

### RED group 1: physical configuration and initial state

Tests:

- closed-schema parser rejects unknown or missing fields;
- exactly two occupied zones plus one utility bay are required;
- non-finite and out-of-range SI values are rejected;
- initial gas inventory satisfies the ideal-gas equation;
- derived humidity, partial pressures and gas fractions are finite and physical.

GREEN:

- immutable config/state dataclasses;
- strict JSON parser;
- initial-state constructor and derived-value helpers.

### RED group 2: one-step species and thermal accounting

Tests:

- zero-flow source-only step changes only declared source/sink inventories;
- simultaneous recirculation conserves inert gas and oxygen in the absence of explicit O2 consumption/injection;
- CO2 generation equals airborne increase plus capture increase;
- water generation equals vapour increase plus condensate increase;
- cross-zone recirculation is independent of zone iteration order;
- with only recirculation and no external source or sink, total occupied-zone lumped thermal energy is conserved, inter-zone transfer sums to zero and the result is independent of zone iteration order;
- each zone and system thermal receipt closes within the predeclared SI residual tolerance;
- cooling changes occupied-zone thermal energy by exactly the declared removed heat and increments external heat rejection by removed heat plus served cooling electrical energy;
- passive condensation prevents post-step supersaturation while conserving water.

GREEN:

- atomic physical step;
- mixed-stream processing;
- thermal and condensation calculations;
- explicit accounting receipt.

### RED group 3: bounds, actuator lag and shared resources

Tests:

- malformed and non-finite commands fail closed;
- airflow and duty slew rates are enforced;
- sorbent saturation stops capture;
- oxygen store cannot go negative;
- battery cannot go negative or exceed capacity;
- electrical infeasibility raises before any state change;
- charging and discharging with non-unit efficiencies close the electrical receipt after W-to-Wh conversion, with each conversion loss recorded once;
- exact battery energy/discharge-power boundaries are accepted with closed receipts and the first amount beyond either limit fails before state advance;
- battery capacity and charge-power boundaries produce explicit curtailment with closed receipts rather than hidden loss; malformed or impossible storage states fail before state advance;
- shared flow and processing limits cannot be exceeded.

GREEN:

- command validator;
- actuator dynamics;
- processing and power constraints.

### RED group 4: runner, trace and reproducibility

Tests:

- strict reference scenario loads and runs;
- trace rows validate against exact topology and field allowlists;
- semantically identical scenario objects differing only in JSON object-key order produce the same canonical scenario digest, run ID and byte-identical trace;
- changing one valid load, action-timeline, equipment or initial-condition value changes both scenario digest and run ID;
- a trace row with a stale or mismatched scenario digest, run ID or equation-contract revision is rejected;
- two executions of the same validated scenario and contract revision produce byte-identical JSONL output;
- writer refuses overwrite;
- CLI returns zero and creates a non-empty valid trace;
- increasing scrubber duty lowers future CO2 versus a paired command sequence;
- increasing condenser duty lowers future humidity;
- allocating more cooling lowers future temperature;
- increased zone airflow changes cross-zone environmental coupling.

GREEN:

- scenario runner;
- trace writer/validator;
- V2 CLI;
- checked-in reference scenario.

## 9. First-slice acceptance gates

The local candidate is accepted only if all are true:

1. focused V2 tests pass;
2. complete existing V1 suite passes unchanged;
3. `git diff --check` passes;
4. package imports from outside the source checkout;
5. two reference CLI runs create byte-identical traces;
6. trace validation recomputes and accepts the canonical scenario digest, equation-contract revision and deterministic run ID for every row;
7. species, water, zone-thermal and electrical accounting residuals remain within their predeclared SI tolerances;
8. no post-step state contains NaN, infinity, negative inventory, relative humidity above 1 or resource overflow;
9. one bounded independent reviewer accepts the exact frozen diff, or one named correction is made and targeted tests plus the full suite are rerun;
10. Ben completes the five-question comprehension gate before any non-trivial PR is opened.

No remote push or PR is part of this first local execution unless Ben reviews the resulting diff and comprehension gate.

## 10. Subsequent simulator phases before model work

### Phase 2: plant completeness and rules rewrite

- add scenario operating modes: occupied, EVA transition, contingency and dormant;
- add deterministic fault primitives: scrubber degradation, filter/dust loading, leakage, thermal-control degradation, sensor bias/drift/dropout, power loss and delayed maintenance;
- add deterministic safety invariants and emergency actions;
- add a finite, versioned candidate-plan catalogue;
- rewrite `docs/simulation-rules.md` into a V1 historical section plus a separate Habitat V2 rules document;
- complete the source/assumption ledger and sensitivity table;
- have the rules and equations reviewed before freezing them.

### Phase 3: neutral baseline and observability gate

- define causal operational telemetry and hidden evaluator state;
- run hidden-state aliasing tests to detect identical histories with incompatible futures;
- implement persistence, linear extrapolation, autoregressive/statistical and handwritten forecasting baselines;
- estimate an oracle/headroom ceiling;
- freeze train, validation and untouched whole-scenario final splits;
- predeclare forecast, calibration, dangerous-transition and closed-loop metrics.

If observability is inadequate, expose defensible operational telemetry, require uncertainty/abstention or abandon the task. Do not compensate with privileged hidden state.

### Phase 4: forecast-only learned model

Only after Phases 2 and 3 pass:

- train one compact action-conditioned temporal model for CO2, temperature and humidity trajectories;
- compare against every frozen non-neural baseline using identical histories and candidate actions;
- measure uncertainty calibration, timing and OOD abstention;
- retain forecast-only shadow mode unless all admission gates pass.

### Phase 5: bounded plan ranking

Only after forecast qualification:

- deterministic code scores finite approved candidate plans using predicted exposure, resource use, wear, reserve use and uncertainty;
- deterministic planner selects a plan;
- deterministic governor independently validates and executes;
- learned model has no actuator, override, handback or bypass authority;
- evaluate untouched whole scenarios against governor-only and non-neural forecast-planner arms.

### Phase 6: Arm optimisation

Only after a model earns admission:

- freeze FP32 artifact and target contract;
- export and quantify exact operator support;
- compare FP32 versus INT8 quality, latency, memory, throughput and available energy/resource measurements;
- label host, virtual and physical-hardware evidence separately;
- reject optimisation if quality or safety gates regress.

## 11. Model admission gate

Model implementation is blocked until all of the following are frozen and accepted:

- Habitat V2 scenario schema;
- physical equations and assumptions ledger;
- trace and causal telemetry contract;
- deterministic action catalogue and governor boundary;
- deterministic baseline implementations;
- observability/headroom report;
- leakage-safe split manifest;
- neutral evaluation contract;
- accepted simulator review receipt.

Passing simulator unit tests alone does not open the model gate.

## 12. Team interfaces and ownership

- Ben owns Habitat V2 backend, physical simulator, model and evaluation decisions.
- Yaro owns the website, frontend and simulation visualisation. The backend supplies versioned validated scenario and trace contracts rather than frontend-specific physics.
- Alex contributes architecture reasoning and bounded review. Review findings are advisory until they identify a violated frozen contract or accepted invariant. Ben remains the primary implementation owner.

Public project documentation should describe normal human team ownership and must not expose internal agent orchestration.

## 13. Version and release impact

The first vertical slice is a backwards-compatible user-facing capability and therefore has `minor` impact:

- current version: 0.2.3;
- candidate version: 0.3.0;
- authoritative sources: `pyproject.toml` and the mirrored `aeolus.__version__` until a later cleanup makes one dynamic;
- changelog: required;
- package artifact: local wheel and source distribution if the build tooling is available;
- clean-install smoke test: required before PR;
- external registry publication: out of scope without explicit approval.

## 14. Execution gate

Approved now:

- create an isolated local worktree from `89ff124`;
- add the plan and V2 first vertical slice;
- run RED-GREEN tests, full suite, real CLI, trace and package smoke checks;
- freeze exact candidate bytes;
- obtain one bounded independent review;
- make at most one finding-specific correction autonomously.

Blocked now:

- editing or regenerating accepted V1 evidence;
- model training or integration;
- direct neural commands;
- frontend work;
- remote push, PR or merge before Ben’s code-comprehension gate;
- external package publication;
- claims of NASA fidelity, flight qualification, AI advantage or Arm performance.
