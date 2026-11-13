# Issue 92: research interface for the fixed-topology batched plant step

Date: 2026-11-13
Status: interface, supported-domain validation, scalar adapter and FP64 NumPy reference oracle written; Torch kernel and everything downstream of it not started and waiting for issue 91 to merge
Issue: 92 (depends on issue 91 for batch layout, validation helpers and the airflow backend)
Interface version: `aeolus_habitat_v2_batched_plant_research_interface_v1` (`INTERFACE_VERSION`)
Version impact: minor, for the release owner to decide. This lane changes no version, lock or source-binding file.

Audience: repository engineers and reviewers who know Habitat V2 but have not followed this lane. Every statement about the scalar plant below was checked against `src/aeolus/habitat_v2/physics.py`, `actuators.py` and `faults.py` and cites function names. This document records no test result and no measured number; those belong in the pull request for the commit that was actually run.

## 1. Status and scope

### What exists now

| Piece | Where | What it is |
| --- | --- | --- |
| Research interface | `src/aeolus/habitat_v2/batched_plant_contract.py` | Frozen dataclasses `PlantParameters`, `PlantStateArrays`, `AchievedControl`, `Disturbance`; the zone order; outcome codes; `EXCLUDED_MECHANISMS`; predeclared tolerances. It contains no numerical kernel, no Torch import and no airflow solve. |
| Supported-domain validation | same module: `require_supported_scenario`, `require_supported_dt`, `validate_step_inputs`, dataclass `__post_init__` checks | Fail-closed type, dtype, shape, finiteness and range checks. Raises `BatchedPlantDomainError`. |
| Scalar adapter | `src/aeolus/habitat_v2/batched_plant_reference.py`: `scalar_reference_step`, `disturbance_for_step` | Runs the authoritative scalar plant for one world and one step and reports that step's state, achieved control and disturbance in the research interface. |
| FP64 NumPy reference oracle | same module: `step_arrays`, `rollout_arrays` | An array transcription of the thermal, species and battery update over any leading batch shape, used to check the interface against the scalar plant. |
| Tests | `tests/habitat_v2/` modules whose names contain `batched_plant`, with shared fixtures in `_batched_plant_helpers.py` | Contract, domain, scalar-parity, rollout and batch-independence checks. They read only checked-in development scenarios and write nothing to disk. |

### What is not done

These items wait for issue 91 to merge and are not started:

- the Torch kernel that implements this interface;
- rollouts that obtain airflow and fan power from issue 91's airflow backend instead of from the scalar solver;
- the benchmark CLI;
- the DGX receipt, and any other hardware measurement.

### What the NumPy reference is and is not

`step_arrays` and `rollout_arrays` are a test oracle. They are not the batched backend and they are not a port of the plant. They exist so that the interface (field set, units, update order, infeasibility policy, tolerances) can be exercised against `physics.py` before a real kernel exists, and so that the future kernel has an independent FP64 array result to be compared with in addition to the scalar plant.

The axis convention used here (zone axis last, per-world arrays of shape `batch_shape`, plain `float64` NumPy arrays) is provisional. When issue 91 publishes its layout and validation helpers, this module adopts that layout; section 9 records what does and does not overlap. The field names, units, domains, update order, excluded-mechanism list and tolerances are the parts intended to survive that change.

## 2. Authority boundary

- `physics.py` together with the deterministic HMC remains the sole authority for canonical simulation and replay. Nothing in this lane changes `physics.py`, the HMC, a scenario, a contract or a trace schema.
- A research rollout is not an HMC episode. It is an open-loop application of a declared sequence of achieved controls and disturbances. It has no proposal, arbitration, preflight, command digest, trace row or receipt validation.
- A result produced through this interface must be replayed through the authoritative path (scalar plant plus deterministic HMC) before any claim is made about an admitted intervention.
- Every input is simulator truth for evaluator-side counterfactual work. `PlantStateArrays` holds true species inventories and temperatures; `AchievedControl` holds effective delivery and ability values, which carry fault effect; `Disturbance` holds realised loads. None of these is a model-facing projection, and nothing here may be used as a model feature. The hidden-truth rules in `AGENTS.md` apply unchanged.
- Nothing here is hardware validation or a statement about physical performance.

## 3. Verified map of one scalar plant step (scenario schema v5)

The authoritative step is `_advance_one_step_canonical`, reached through `advance_one_step` (timeline command) or `advance_one_step_with_command` (external command). The table lists its operations in execution order for `aeolus_habitat_v2_scenario_v5`.

"Pre" means the value in the `PlantState` passed in. "Post-N" means the value produced by row N of the same step. The last column says where the operation lives in this lane:

- **research step**: computed inside the research step (today by `step_arrays`, later by the Torch kernel);
- **reference path input**: computed only by the scalar plant; its result enters the research step as a named input field;
- **excluded**: stays on the authoritative path and has no counterpart in the research step.

| # | Operation and `physics.py` function | Formula | State read | Research lane |
| --- | --- | --- | --- | --- |
| 0 | Step bound and segment lookup (`_segment_for_step`) | reject `step >= steps`; pick the timeline segment with `start_step <= step < end_step` | pre step index | excluded (the research state has no step index; `disturbance_for_step` does the lookup for the adapter) |
| 1 | Species sources and oxygen-load check (first loop of `_advance_one_step_canonical`) | per zone: `co2 += co2_generation * dt`, `o2 -= o2_consumption * dt`, `water += water_vapor_generation * dt`; raise `InfeasibleActionError` if `o2_consumption * dt > o2` | pre species | research step (loads are `Disturbance` inputs; code `INFEASIBLE_OXYGEN_LOAD`) |
| 2 | Command validation (`validate_external_command`, schema and digest checks) | closed field set, topology, bounds, canonical bytes, SHA-256 | none | excluded |
| 3 | Fan slew (`_slew`) | move `actual_fan_speed_fraction` toward the requested fraction by at most `speed_slew_fraction_per_s * dt` | pre fan actuator state | excluded (reaches the research step only through row 7) |
| 4 | Physical fault evaluation (`faults.physical_fault_effects`) | profiles active when `start_step <= emitted_step < end_step`, with `emitted_step = step + 1`; linear multiplier from start to end value | pre step index, pre damper positions | excluded (its effect arrives through rows 7 and 10) |
| 5 | Damper slew or jam | a jammed damper keeps its previous position; otherwise `_slew` toward the request by `damper_slew_fraction_per_s * dt` | pre damper positions | excluded (reaches the research step only through row 7) |
| 6 | Effective fan speed | `effective = achieved_fan_speed * fan_speed_multiplier` | post-3, post-4 | excluded (reaches the research step only through row 7) |
| 7 | Air network solve (`air_network.solve_air_network` with `_air_network_spec`) | operating point at the effective fan speed, achieved damper positions and fault-scaled branch supply resistance | post-4, post-5, post-6 | reference path input: `zone_airflow_m3_s`, `fan_electrical_power_w` |
| 8 | Scrubber and condenser duty slew (`_slew`) | toward the requested duty by `scrubber_duty_slew_per_s * dt` and `condenser_duty_slew_per_s * dt` | pre duty actuator state | reference path input: `achieved_scrubber_duty`, `achieved_condenser_duty` |
| 9 | Cooling and oxygen achievement with cap projection (`actuators.achieve_actuator_state`) | per-zone slew by `cooling_slew_w_per_s * dt` and `oxygen_slew_mol_s2 * dt`; if the `fsum` total oxygen exceeds `oxygen_injection_max_total_mol_s`, reduce the largest zones first to one representable float below the cap, never below `max(0, current - slew)` | pre cooling and oxygen actuator state | reference path input: `achieved_cooling_removed_w`, `achieved_oxygen_injection_mol_s` |
| 10 | Effective delivery and abilities | `effective_cooling = achieved_cooling * cooling_delivery_multiplier` and `effective_oxygen = achieved_oxygen * oxygen_delivery_multiplier` per zone (multiplier 1 when no fault is active); abilities are the scrubber and condenser multipliers | post-4, post-9 | reference path input: `effective_cooling_delivery_w`, `effective_oxygen_delivery_mol_s`, `scrubber_capture_ability`, `condenser_removal_ability` |
| 11 | Recirculation with capture and active condensing (`_recirculate`) | `exchange = 1 - exp(-airflow * dt / volume)`; each species is extracted by that fraction into a pool; `captured = min(pool_co2, scrubber_max * duty * ability * dt, sorbent)`; `condensed = min(pool_water, condenser_max * duty * ability * dt)`; the pool is returned to each zone in proportion to that zone's total extracted moles; if nothing was extracted the zones are returned unchanged with zero capture | post-1 species; post-7 airflow; post-8 duties; post-10 abilities; pre sorbent | research step |
| 12 | Oxygen injection and store check | `injected = effective_oxygen * dt` per zone; raise `InfeasibleActionError` if the sorted-zone total exceeds `oxygen_store_mol`; add to zone oxygen | post-11 species; post-10 delivery; pre oxygen store | research step (code `INFEASIBLE_OXYGEN_STORE`) |
| 13 | Recirculation heat (`_recirculation_heat_transfer_j`) | `T_mix = sum(airflow * T) / sum(airflow)`; `q = density * cp * airflow * (T_mix - T) * dt`; the sum of all `q` is subtracted from the last zone in sorted order so the transfers close exactly; all zero when total airflow is zero | pre temperatures (rows 1, 11 and 12 change species only); post-7 airflow | research step |
| 14 | Zone thermal balance (`_apply_zone_thermal_balance`) | explicit Euler: `T_next = T + (sensible_heat * dt + q_recirculation + passive_received - passive_rejected - effective_cooling * dt) / thermal_capacity`, with signed passive heat `conductance * (T_sink - T) * dt` split into received and rejected parts | pre temperatures; post-13; post-10 cooling | research step |
| 15 | Passive condensation (`_apply_passive_condensation`) | `saturated = p_sat(T_next) * volume / (R * T_next)`; `condensed = max(0, water - saturated)`; no latent heat is returned to the zone | post-14 temperature; post-11 water vapour | research step |
| 16 | Electrical balance and infeasibility (`_electrical_balance`) | `served = (base_load + fan_power + scrubber_power_full * achieved_scrubber_duty + condenser_power_full * achieved_condenser_duty + sum(achieved_cooling) / COP + oxygen_power_per_mol_s * sum(achieved_oxygen)) * dt / 3600`. If `generation >= served`: charge input `min(surplus, max_charge * hours, (capacity - battery) / charge_efficiency)`, the rest is curtailed. Otherwise withdraw `(served - generation) / discharge_efficiency`; raise `InfeasibleActionError` if that exceeds `min(battery, max_discharge * hours) + max(1e-12, 1e-10 * max(1, served))` | pre battery; post-7 fan power; post-8 duties; post-9 achieved cooling and oxygen | research step (code `INFEASIBLE_ELECTRICAL`) |
| 17 | External heat | received: sum of passive received heat. Rejected: sum of passive rejected heat and effective cooling heat, plus `(served_load + charge_loss + discharge_loss) * 3600` | post-14, post-16 | research step |
| 18 | Species accounting receipt | per-species residual of after minus before minus declared sources and sinks, with tolerance `max(1e-12, 1e-10 * scale)` | pre and post-15 species | excluded (receipt content; the research step returns its own per-world diagnostics instead) |
| 19 | Utility update | `sorbent -= captured`; `captured_co2 += captured`; `condensed_water += condensed + sum(passive condensation)`; `oxygen_store -= total injected`; battery and heat accumulators advance; achieved and effective actuator values are stored as the next actuator state | pre utility; post-11, 12, 15, 16, 17 | research step for the seven store and accumulator fields; excluded for actuator state, which stays on the reference path |
| 20 | Receipt assembly and operational feedback (`_operational_feedback_truth`, `instrumentation.measure_operational_feedback`) | measurement of the new state at `step + 1` against the previous feedback record | post-19 state; pre `last_operational_feedback` | excluded |

Three points in this map are easy to get wrong and are the reason the map was written:

1. Faults are evaluated at the emitted step, `step + 1`, not at the step index of the input state.
2. Recirculation heat and the thermal balance read pre-step temperatures, while passive condensation reads the new temperature.
3. Electrical load follows achieved actuation, while species and heat follow effective delivery. A degraded cooler draws full power for the cooling it was asked to achieve and delivers less of it.

All three infeasibility checks raise before the scalar plant returns a new state, so an infeasible scalar step leaves the state unchanged. They are reached in the order load, store, electrical. On the scalar path other errors can occur between them (`ScenarioValidationError` from row 2, `ValueError` from row 9); those are not infeasibility and have no research-step code.

## 4. The frozen interface

All arrays are `numpy.ndarray` of dtype `float64` (exact type; subclasses are rejected), finite, and are not mutated by the step. Per-zone arrays have shape `batch_shape + (8,)`. Per-world arrays have shape `batch_shape`. `batch_shape` may be empty.

### Zone order

The zone axis is last and follows `SUPPORTED_ZONE_IDS`, which is `sorted()` of the fixed topology's zone ids and is the order the scalar plant iterates in:

| Index | Zone id |
| --- | --- |
| 0 | `air_processing_bay` |
| 1 | `airlock_suitport` |
| 2 | `common_galley` |
| 3 | `crew_quarters_a` |
| 4 | `crew_quarters_b` |
| 5 | `equipment_power_bay` |
| 6 | `hygiene_medical` |
| 7 | `laboratory` |

The scalar plant applies the recirculation-heat closure correction (row 13) to the last entry, `laboratory`. A backend that reorders zones changes that result.

### Timestep

`dt_seconds` is one real number shared by the batch (not a `bool`), finite, with `1.0 <= dt_seconds <= 600.0` (`SUPPORTED_DT_SECONDS`).

### PlantParameters

Shared by every world in a batch. Built from a scenario with `PlantParameters.from_scenario`.

| Field | Shape | Unit | Domain |
| --- | --- | --- | --- |
| `zone_ids` | tuple of 8 strings | none | exactly `SUPPORTED_ZONE_IDS` |
| `volume_m3` | `(8,)` | m3 | `> 0` |
| `thermal_capacity_j_per_k` | `(8,)` | J/K | `> 0` |
| `passive_thermal_conductance_w_per_k` | `(8,)` | W/K | `>= 0` |
| `sink_temperature_k` | `(8,)` | K | `150 <= x <= 330` |
| `air_density_kg_m3` | scalar | kg/m3 | `> 0` |
| `air_specific_heat_j_kg_k` | scalar | J/(kg K) | `> 0` |
| `scrubber_max_co2_mol_s` | scalar | mol/s | `>= 0` |
| `condenser_max_water_mol_s` | scalar | mol/s | `>= 0` |
| `oxygen_injection_max_total_mol_s` | scalar | mol/s | `>= 0` |
| `cooling_max_thermal_w_per_zone` | scalar | W | `>= 0` |
| `base_load_w` | scalar | W | `>= 0` |
| `scrubber_power_w_full` | scalar | W | `>= 0` |
| `condenser_power_w_full` | scalar | W | `>= 0` |
| `cooling_coefficient_of_performance` | scalar | dimensionless | `> 0` |
| `oxygen_injection_power_w_per_mol_s` | scalar | W per mol/s | `>= 0` |
| `battery_capacity_wh` | scalar | Wh | `> 0` |
| `battery_charge_efficiency` | scalar | fraction | `0 < x <= 1` |
| `battery_discharge_efficiency` | scalar | fraction | `0 < x <= 1` |
| `battery_max_charge_w` | scalar | W | `>= 0` |
| `battery_max_discharge_w` | scalar | W | `>= 0` |

Slew rates, fan and duct parameters, sorbent capacity and feedback parameters are deliberately absent. They belong to mechanisms that stay on the reference path.

### PlantStateArrays

Thermal, gas and store state. Actuator state and `last_operational_feedback` are not carried.

| Field | Shape | Unit | Domain |
| --- | --- | --- | --- |
| `co2_mol` | zone | mol | `>= 0` |
| `o2_mol` | zone | mol | `>= 0` |
| `water_vapor_mol` | zone | mol | `>= 0` |
| `inert_mol` | zone | mol | `>= 0` |
| `temperature_k` | zone | K | `200 <= x <= 332` (`SUPPORTED_TEMPERATURE_K`) |
| `co2_sorbent_remaining_mol` | world | mol | `>= 0` |
| `captured_co2_mol` | world | mol | `>= 0` |
| `condensed_water_mol` | world | mol | `>= 0` |
| `oxygen_store_mol` | world | mol | `>= 0` |
| `battery_energy_wh` | world | Wh | `-1e-6 <= x <= battery_capacity_wh + 1e-6` (`BATTERY_STATE_SLACK_WH`) |
| `external_heat_rejected_j` | world | J | `>= 0` |
| `external_heat_received_j` | world | J | `>= 0` |

The battery slack exists because the scalar plant can leave the battery marginally outside `[0, capacity]` and then keeps stepping: `_electrical_balance` accepts a discharge up to `max(1e-12, 1e-10 * max(1, served_load_wh))` beyond the available energy, and a headroom-limited charge can round one unit in the last place above capacity. A state the plant produces has to be a legal next input.

The temperature band is narrower than the stated validity range of the Murphy-Koop liquid-water saturation formula (123 K to 332 K) and wider than the scenario validator's initial-temperature range (250 K to 330 K).

### AchievedControl

| Field | Shape | Unit | Domain |
| --- | --- | --- | --- |
| `zone_airflow_m3_s` | zone | m3/s | `>= 0` |
| `achieved_cooling_removed_w` | zone | W | `0 <= x <= cooling_max_thermal_w_per_zone` |
| `effective_cooling_delivery_w` | zone | W | `0 <= x <= achieved_cooling_removed_w` |
| `achieved_oxygen_injection_mol_s` | zone | mol/s | `>= 0`; zone total `<= oxygen_injection_max_total_mol_s * (1 + 1e-12)` |
| `effective_oxygen_delivery_mol_s` | zone | mol/s | `0 <= x <= achieved_oxygen_injection_mol_s` |
| `fan_electrical_power_w` | world | W | `>= 0` |
| `achieved_scrubber_duty` | world | fraction | `0 <= x <= 1` |
| `achieved_condenser_duty` | world | fraction | `0 <= x <= 1` |
| `scrubber_capture_ability` | world | fraction | `0 <= x <= 1` |
| `condenser_removal_ability` | world | fraction | `0 <= x <= 1` |

The `1e-12` relative slack on the oxygen total (`OXYGEN_CAP_RELATIVE_SLACK`) exists because `achieve_actuator_state` projects below the cap using `math.fsum`, and a plain sum of the same eight values can land a few units in the last place above it.

### Disturbance

| Field | Shape | Unit | Domain |
| --- | --- | --- | --- |
| `co2_generation_mol_s` | zone | mol/s | `>= 0` |
| `o2_consumption_mol_s` | zone | mol/s | `>= 0` |
| `water_vapor_generation_mol_s` | zone | mol/s | `>= 0` |
| `sensible_heat_w` | zone | W | `>= 0` |
| `generation_w` | world | W | `>= 0` |

The step looks nothing up by step index and draws no random numbers. A stochastic disturbance must be sampled by the caller, per world, and passed in.

### Requested, HMC-final, achieved and effective

| Term | Meaning | Where it lives on the scalar path | Accepted by the research interface |
| --- | --- | --- | --- |
| Requested | The command in a timeline segment or supplied externally | `command` in `_advance_one_step_canonical` after `validate_external_command` | no |
| HMC-final | The command the deterministic HMC emits after arbitration and preflight. From the plant's point of view it is still a requested command | passed to `advance_one_step_with_command` | no |
| Achieved | What the actuator reached after slew limits and the oxygen cap projection | `actual_*` fields of `UtilityState`; `achieved_*` in the step receipt | yes: `achieved_*` fields |
| Effective | Achieved value after physical-fault multipliers; what reaches the air | `effective_*` fields of `UtilityState`; `effective_*` in the step receipt | yes: `effective_*` fields, the two abilities, and the air-network solution |

Only achieved and effective values are accepted because turning a requested command into an achieved one needs the previous actuator state, the slew limits, the cap projection, the fault schedule and the air-network solve. Those are exactly the mechanisms that stay authoritative on the scalar path (section 6). `AchievedControl` has no field that could hold a requested or HMC-final command, and `validate_step_inputs` rejects any other type, so a requested command cannot be handed to the step by mistake. If a caller treated a requested value as achieved it would skip slew and fault effects silently; the interface is shaped to make that hard, but it cannot detect it.

Which value drives what:

| Quantity | Driven by |
| --- | --- |
| Fan electrical load | `fan_electrical_power_w` (air-network solution at the effective fan speed) |
| Scrubber and condenser electrical load | `achieved_scrubber_duty`, `achieved_condenser_duty` |
| Cooling electrical load | sum over zones of `achieved_cooling_removed_w`, divided by the coefficient of performance |
| Oxygen injection electrical load | sum over zones of `achieved_oxygen_injection_mol_s` |
| Recirculation exchange and recirculation heat | `zone_airflow_m3_s` |
| CO2 capture limit | `achieved_scrubber_duty * scrubber_capture_ability` |
| Active condensing limit | `achieved_condenser_duty * condenser_removal_ability` |
| Heat removed from each zone | `effective_cooling_delivery_w` |
| Oxygen added to each zone and drawn from the store | `effective_oxygen_delivery_mol_s` |

## 5. Supported domain and fail-closed behaviour

### Supported scenarios

`require_supported_scenario` raises `BatchedPlantDomainError` unless all of the following hold:

1. the argument is exactly a `Scenario` (not a mapping, not a subclass) and `validate_contract_identities()` passes, so a scenario edited after validation is refused;
2. `scenario_schema_version` is `aeolus_habitat_v2_scenario_v5`;
3. the sorted zone ids equal `SUPPORTED_ZONE_IDS`;
4. the air network has exactly one branch per supported zone;
5. `dt_seconds` is inside `SUPPORTED_DT_SECONDS`.

Scenario schemas v1 to v4 and any other topology are rejected. `PlantParameters.from_scenario`, `disturbance_for_step` and `scalar_reference_step` all call this check first.

### Construction checks

Each interface dataclass checks itself on construction:

- every array is exactly `numpy.ndarray`, dtype `float64`, finite;
- the batch has at least one world (no zero-length batch axis);
- the first per-zone field has at least one axis and eight entries on its last axis; every other field has the shape that follows from it;
- `PlantParameters`: `zone_ids` equals `SUPPORTED_ZONE_IDS`; the four zone arrays have shape `(8,)` and the ranges in section 4; each equipment scalar is a real number (not `bool`), finite and inside its range.

`PlantStateArrays.from_plant_states` additionally requires a nonempty tuple of exact `PlantState` values that each have the supported eight zones.

### `validate_step_inputs`

Called by `step_arrays` before any arithmetic. In order:

1. `parameters`, `state`, `control` and `disturbance` are exactly the four interface types;
2. `dt_seconds` is inside the supported range;
3. state, control and disturbance share one batch shape;
4. the construction checks of all four values are repeated, because the arrays they hold are writable;
5. every state field is inside its domain from `state_field_domain`: species, stores and heat ledgers `>= 0`, `temperature_k` inside `SUPPORTED_TEMPERATURE_K`, battery inside `[0, capacity]` widened by `BATTERY_STATE_SLACK_WH`;
6. every control field is `>= 0`;
7. both duties and both abilities are `<= 1`;
8. `achieved_cooling_removed_w <= cooling_max_thermal_w_per_zone`;
9. `effective_cooling_delivery_w <= achieved_cooling_removed_w` elementwise;
10. `effective_oxygen_delivery_mol_s <= achieved_oxygen_injection_mol_s` elementwise;
11. the zone total of achieved oxygen injection is within the cap plus slack;
12. every disturbance field is `>= 0`.

A violation anywhere in the batch raises for the whole call. Out-of-domain input is a caller error, not a per-world outcome.

There is no upper bound on airflow, fan power, loads or generation. They are accepted as given; a world they drive out of the state domain is reported with code 4 below.

### Per-world outcomes

In-domain inputs that the scalar plant would refuse with `InfeasibleActionError` are not rejected by validation. They are reported per world.

| Code | Name | Condition | Scalar plant message fragment |
| --- | --- | --- | --- |
| 0 `FEASIBLE` | `feasible` | none of the below | none |
| 1 `INFEASIBLE_OXYGEN_LOAD` | `oxygen_load_exceeds_zone_inventory` | `o2_consumption * dt > o2_mol` in any zone | "oxygen load exceeds inventory" |
| 2 `INFEASIBLE_OXYGEN_STORE` | `oxygen_injection_exceeds_store` | zone total of `effective_oxygen_delivery * dt` exceeds `oxygen_store_mol` | "oxygen injection exceeds stored oxygen" |
| 3 `INFEASIBLE_ELECTRICAL` | `electrical_demand_exceeds_supply` | required battery withdrawal exceeds the available withdrawal plus the scalar plant's tolerance | "electrical demand exceeds" |
| 4 `OUTSIDE_SUPPORTED_DOMAIN` | `result_outside_supported_domain` | the step's result for this world would be nonfinite or outside `state_field_domain` (for example a temperature outside the supported band) | none: not a scalar-plant refusal |

Code 4 is a research-domain outcome with no scalar counterpart. The scalar plant has no temperature band: in the same situation it either continues outside the validated range or fails with a math error in the saturation formula. Reporting it per world keeps the step's output a legal next input and stops one world from aborting a batch. A comparison with the scalar plant must treat code 4 worlds as outside the comparison, and report their count.

Policy:

- Order of evaluation. When more than one condition holds, the reported code is the first one the scalar plant would reach: load, then store, then electrical. Code 4 is assigned only when none of the three applies.
- State held. A world with a nonzero code keeps its input state exactly, as the scalar plant does when it raises before the state advances.
- Diagnostics zero. Every `StepDiagnostics` field other than `infeasible_code` is zero for a held world.
- Isolation. Other worlds in the batch are unaffected.

The adapter maps scalar exceptions to codes by the message fragments above. An `InfeasibleActionError` with any other message is re-raised. For an infeasible scalar step the adapter returns `control=None`, because the scalar plant raised before it finished resolving actuation.

### Rollout

`rollout_arrays` applies a declared sequence and fails closed unless:

- `controls` and `disturbances` are tuples (exact type), nonempty and of equal length;
- the length is at most `REFERENCE_ROLLOUT_MAX_STEPS`, which is 240;
- every step passes `validate_step_inputs`.

Freeze semantics. A world that becomes infeasible at sequence index `k` gets `first_infeasible_step = k` and the code of that step, and is frozen at its last feasible state for the rest of the horizon. Later controls and disturbances are not applied to it, and its code is not overwritten. A world that stays feasible has `first_infeasible_step = -1`. `steps_applied` counts the steps that actually advanced each world.

Two consequences worth knowing:

- Later controls and disturbances for a frozen world are still validated, so they must be in-domain even though they are not applied.
- The state a step returns is always inside the input domain, so a rollout cannot fail validation on its own output. A world that would leave the domain is frozen with code 4 like any other held world.

## 6. Excluded mechanisms

This reproduces `EXCLUDED_MECHANISMS`. Each mechanism must be resolved where it stays before values enter the interface.

| Key | What is excluded | Where it stays |
| --- | --- | --- |
| `requested_command_slew` | Requested commands and rate limits for fan, dampers, scrubber, condenser, cooling and oxygen | `physics.py` (`_slew`) and `actuators.py`; the interface accepts achieved values only |
| `oxygen_total_cap_projection` | The `fsum`/`nextafter` projection of achieved oxygen injection onto the equipment cap | `actuators.achieve_actuator_state` |
| `fault_schedule_evaluation` | Evaluation of fault profiles | `faults.physical_fault_effects`; only the per-step effect enters, as effective delivery and ability values |
| `air_network_solve` | The fan and duct operating-point solve | The scalar solver, or issue 91's airflow backend after it merges; per-zone airflow and fan electrical power are inputs |
| `damper_jam` | Holding a jammed damper at its previous position | Part of achieved damper position on the reference path |
| `sensor_and_feedback_models` | Sensor noise, sensor faults, operational feedback and seeds | Measurement side; never enters the plant step |
| `hmc_arbitration_and_preflight` | HMC proposal, arbitration, preflight and final-command digests | Outside the numerical step, on the authoritative path |
| `scenario_parsing_hashing_trace_writing` | Closed-schema parsing, canonical bytes, hashing, accounting-receipt validation and JSONL trace writing | The authoritative path |
| `legacy_scenario_schemas` | Scenario schema v1 to v4 and any topology other than the fixed eight zones | Rejected by `require_supported_scenario` |
| `per_world_plant_parameters` | Parameters that differ between worlds of one batch | Not available in this interface version; parameters are shared |

## 7. Predeclared tolerances

A candidate value is accepted against a reference value when

`|candidate - reference| <= absolute + relative * |reference|`

per variable, per element. The tables below are `reference_tolerances(precision, horizon)` in the contract module, written as `absolute, relative`. "Rollout" means a horizon of at most `REFERENCE_ROLLOUT_MAX_STEPS`.

### FP64

| Variable | One step | Rollout |
| --- | --- | --- |
| `co2_mol` | 1e-12, 1e-12 | 1e-9, 1e-10 |
| `o2_mol` | 1e-12, 1e-12 | 1e-9, 1e-10 |
| `water_vapor_mol` | 1e-12, 1e-12 | 1e-9, 1e-10 |
| `inert_mol` | 1e-12, 1e-12 | 1e-9, 1e-10 |
| `temperature_k` | 1e-10, 0 | 1e-8, 0 |
| `co2_sorbent_remaining_mol` | 1e-12, 1e-12 | 1e-9, 1e-10 |
| `captured_co2_mol` | 1e-12, 1e-12 | 1e-9, 1e-10 |
| `condensed_water_mol` | 1e-12, 1e-12 | 1e-9, 1e-10 |
| `oxygen_store_mol` | 1e-12, 1e-12 | 1e-9, 1e-10 |
| `battery_energy_wh` | 1e-9, 1e-12 | 1e-7, 1e-10 |
| `external_heat_rejected_j` | 1e-6, 1e-12 | 1e-4, 1e-10 |
| `external_heat_received_j` | 1e-6, 1e-12 | 1e-4, 1e-10 |

### FP32 (provisional)

| Variable | One step | Rollout |
| --- | --- | --- |
| the four species fields | 1e-5, 2e-6 | 1e-3, 1e-4 |
| `temperature_k` | 2e-4, 0 | 2e-2, 0 |
| `co2_sorbent_remaining_mol`, `captured_co2_mol`, `condensed_water_mol`, `oxygen_store_mol` | 1e-5, 2e-6 | 1e-3, 1e-4 |
| `battery_energy_wh` | 1e-3, 2e-6 | 1e-1, 1e-4 |
| `external_heat_rejected_j`, `external_heat_received_j` | 1e-1, 2e-6 | 1e1, 1e-4 |

Rules:

- These values were fixed before any backend comparison was run. They must not be loosened to make a comparison pass. A change is a recorded interface revision with a new `INTERFACE_VERSION`.
- Rationale. FP64 one-step values are 1e-12 absolute plus 1e-12 relative for inventories. FP64 rollout values are one hundred times the one-step values over at most 240 steps, which is deliberately tighter than 240 accumulated worst-case one-step errors. FP32 one-step values are a small multiple of float32 resolution at the magnitudes of the development fixture; FP32 rollout values use a 1e-4 relative term.
- The tables must be able to tell a real step from no step. A first draft of the FP32 one-step table used a 1e-4 relative term, and review showed that a candidate returning its input unchanged would have passed it for the large inventories. It was tightened before any FP32 backend existed, and a test now requires an unchanged-state candidate to fail every variable that physically changes on the development fixture, for both precisions.
- The tables are read-only mappings and `Tolerance` rejects negative or nonfinite terms. Compare with `numpy.all(error <= tolerance.allowed(reference))` in float64 so that a NaN error fails.
- The FP32 values are provisional until measured on real hardware. No FP32 kernel exists yet, so nothing has been compared against them.
- The NumPy reference is held to the same FP64 table against the scalar plant. It gets no separate, looser allowance. It is not expected to be bit-identical to the scalar plant, because NumPy's `exp`, `log` and `tanh` need not round the same way as the `math` module functions the scalar plant uses.
- Decision and threshold disagreements are reported separately from numeric error. A candidate that returns a different `infeasible_code` from the reference, or a different `first_infeasible_step` in a rollout, has a decision disagreement even if every state value is inside tolerance. It must be reported as a count of disagreeing worlds with their codes, and must not be averaged into, or hidden by, a numeric error figure. The same applies to the other threshold branches of the step: the charge or discharge branch of the electrical balance, the zero-extraction and zero-airflow branches, and whether passive condensation occurs.
- `reference_tolerances` raises `BatchedPlantDomainError` for an unknown precision or horizon. There is no default table.

## 8. Reproducibility declaration

Declaration for the FP64 NumPy reference: on one platform and one NumPy build, the result for a world is bit-identical regardless of batch size, the world's position in the batch, which other worlds are in the batch, and how the batch is chunked.

Why this holds by construction:

- Zone sums use `zone_sum`, a fixed left-to-right accumulation over the zone axis in sorted zone order. `numpy.sum` is avoided there because its pairwise or vectorised accumulation can round differently depending on memory layout.
- No operation reduces across worlds. The only reductions are over the zone axis of a single world.
- Infeasible worlds are handled by selection (`numpy.where`), so a neighbour's outcome never changes another world's arithmetic.
- There is no shared mutable state, no cache and no step counter. Inputs are not mutated. Parameters are shared but read-only in use.
- There is no random number generation anywhere in the step.

Limits of the declaration:

- It rests on NumPy's elementwise `exp`, `log` and `tanh` returning the same result for the same input regardless of where the element sits in an array. That is what the batch-independence tests check on the platform they run on; it is not a documented NumPy guarantee across builds, CPU feature sets or non-contiguous layouts.
- It is not a cross-platform or cross-version bit-identity claim.
- The scalar plant's zone sums use Python's built-in `sum`. On the locked Python 3.11 that is a plain left-to-right sum, which `zone_sum` reproduces. From Python 3.12 the built-in uses compensated summation for floats, so the scalar plant itself may round differently there and a threshold decision could move by one unit in the last place. The tolerances cover the numeric effect; a decision disagreement at such a point would be reported as one.

The Torch kernel does not exist yet. Its declared reproducibility tolerance (bit-identical, or a stated bound, and under which device, dtype and batch-layout conditions) is to be set when the kernel exists and has been measured. Nothing in this document should be read as a claim about it.

## 9. What this does not prove, and open items

### What this does not prove

- It does not show that any batched backend exists, is correct or is fast. There is no Torch kernel, no benchmark and no hardware result.
- Agreement between the NumPy reference and the scalar plant, where tests show it, covers the checked-in development scenarios, the in-memory stress variant built from the development fixture, and seeded synthetic batches inside the declared domain. It is not a proof over the whole domain.
- It says nothing about closed-loop behaviour. The research rollout is open loop, with achieved controls supplied for every step.
- It does not validate the scalar plant's physics. The scalar plant is taken as the authority, including its modelling choices (explicit Euler, no latent heat from condensation, fixed-density recirculation).
- It makes no claim about admitted interventions, HMC behaviour, forecast quality, hardware, or any final or held-out scenario. No such scenario was opened for this lane.

### Acceptance checklist

The items below paraphrase the issue's acceptance list as this lane understood it; the issue text governs. Status values: done, partly, not started (blocked on 91), hardware unverified. "Done" means written and covered by tests in this lane. It is not a statement that the tests passed on any commit; that is reported in the pull request.

| Item | Status | Note |
| --- | --- | --- |
| Scalar plant step mapped and checked against source | done | Section 3 |
| Research interface frozen: fields, shapes, units, domains, zone order, timestep | done | Axis convention provisional until issue 91's layout is published |
| Achieved and effective actuation only; no requested command field | done | Section 4 |
| Airflow and fan electrical power as explicit inputs | done | Supplied by the scalar solver today |
| Supported-domain validation that fails closed | done | Section 5 |
| Excluded mechanisms declared | done | Section 6 |
| Tolerances predeclared before any backend comparison | done | FP32 values provisional |
| Per-world infeasibility flag with state held | done | Three scalar refusal codes plus code 4 for a result outside the supported domain |
| FP64 reference agrees with the scalar plant over one step and over a bounded rollout | partly | Implemented for the NumPy oracle only; no kernel has been compared |
| Reproducibility declaration | partly | Declared for the NumPy reference; the kernel's is not set |
| Batched Torch kernel implementing the interface | not started (blocked on 91) | |
| Rollouts through issue 91's airflow backend | not started (blocked on 91) | |
| FP32 parity against the provisional table | not started (blocked on 91) | Needs the kernel |
| Benchmark CLI | not started (blocked on 91) | |
| DGX receipt | hardware unverified | Nothing has been run on DGX hardware |
| Replay of a research result through the authoritative path | partly | The rule is stated in section 2; no tooling in this lane performs or enforces the replay |

### Open items

1. Replace this module's layout checks with issue 91's layout and validation helpers when they are published, and re-run the scalar parity tests unchanged.
2. Confirm the code 4 policy (hold and report a world whose result leaves the supported domain). It is a research-domain choice with no scalar counterpart.
3. Decide whether airflow, loads and generation should get declared upper bounds. Today they are unbounded and an extreme value surfaces as code 4.
4. Per-world plant parameters are excluded in this interface version. Adding them is an interface revision.
5. The adapter identifies the scalar plant's infeasibility reason by message fragment. If `physics.py` ever rewords those messages the adapter re-raises instead of misclassifying, but the mapping would need updating.
6. Set the Torch kernel's reproducibility tolerance and confirm or revise the FP32 table from measurements on real hardware, as a recorded interface revision if any value changes.

### Issue 91 airflow interface this lane is designed against

Source: the `INTERFACE (v1)` comment on issue 91, posted with draft PR 95. It is unmerged, so it may still change. Nothing from that lane is imported, copied or vendored here.

- Layout: per-world `[B]`, per-branch `[B, N]`, branch axis in ascending `zone_id` order, `float64` NumPy on the host. The zone order and dtype match this document. Issue 91 uses exactly one leading world axis; this module currently accepts any nonempty leading batch shape, which is wider. The Torch plant step will use the single `[B]` axis.
- Issue 91's layout dataclass and validator (`AirNetworkBatch`, `validate_air_network_batch`, error type `AirNetworkAnalyticError`) cover air-network fields only. They do not validate plant state, control or disturbance, so this module's checks for those stay; there is no overlapping definition to delete beyond the axis convention.
- Optional import: the Torch plant step will obtain Torch and the device through `air_network_batched_torch.require_torch()` and `require_device(device)`, which raise `AirNetworkBackendUnavailableError` and substitute no CPU fallback when CUDA is requested and missing.
- Batch independence stated by issue 91: bit-identical for NumPy FP64; tested to 1e-13 relative (FP64) and 1e-6 relative (FP32) for Torch on CPU, with no bitwise claim; CUDA untested. The Torch plant step therefore cannot declare bit-exact batch independence and will declare a tolerance.
- Everything is per world, including branch resistances. There is no resistance-multiplier input: the plant step will multiply `open_supply_resistance_pa_s2_m6` by its per-world `[B, N]` fault multiplier before the call, as `_air_network_spec` does.
- The fan speed passed in is the effective speed (achieved speed times the fault multiplier).
- Consumed outputs: `zone_flow_m3_s` `[B, N]` and `fan_electrical_power_w` `[B]`.
- Fan speed exactly `0.0` gives exact zeros in every output. For tiny positive speeds the batched solver is linear in speed, which differs from the scalar solver's underflow behaviour. Agreement with the scalar solver is to tolerance (stated as 1e-12 of physical scale in FP64), not bitwise.

- The scalar air-network solver is not ground truth at extreme parameter corners: where branch pressure is far below shutoff pressure its fixed 100-iteration bisection is not converged. Agreement at 1e-12 of scale is stated for fixture-band parameters only.

Rules the Torch plant step will follow as a result:

- Take the exact-zero branches from the commanded effective fan speed being exactly zero, never from two solvers happening to agree.
- Keep scalar-reference comparisons on fixture-band air-network parameters, or predeclare a wider tolerance for corners before comparing.
- Keep this lane's tolerances in this lane's own module; the issue 91 contract file is not edited.

Consequence for this lane: the plant's exact-zero branches (zero total extraction, zero total airflow) are guaranteed to match the scalar plant only at fan speed exactly zero, and the bit-level agreement the NumPy reference shows when it is fed the scalar solver's airflow will become tolerance-level agreement once the batched airflow feeds the step. The predeclared tables, not bit equality, are the standard, and threshold-adjacent tests for the Torch kernel must include tiny positive fan speeds.

### Version impact

Minor, for the release owner to decide. The lane adds two source modules, tests and this document. No version, lock or source-binding file is changed, and no scenario, contract, trace schema or checked-in artifact is touched.
