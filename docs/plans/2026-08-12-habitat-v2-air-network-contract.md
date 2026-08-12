# Habitat V2 Phase 2 Slice 2: explicit multizone air-network contract

Date: 2026-08-12
Status: corrected after one bounded independent review; approved for isolated local implementation
Repository: `C:\Users\Nxiss\code\aeolus-habitat-v2-air-network`
Branch: `ben/habitat-v2-air-network`
Base: `087b1e9a97bba726087638636a3e76662b9d4f5b`
Publication authority: implementation, focused verification, branch push, review, and merge are authorised. Learned-model implementation and training are not authorised in this phase.
Version impact: minor, candidate `0.5.0`

## What are we making?

A versioned Habitat V2 scenario and trace contract for an explicit eight-zone recirculating air network. The network turns fan speed and per-zone damper commands into physically derived pressure, airflow and electrical-power telemetry. Existing scenario-v1, scenario-v2, trace-v1 and trace-v2 bytes and behaviour remain unchanged.

## Why this slice exists

The accepted Habitat V2 conservation kernel uses direct per-zone airflow commands and a linear fan-power coefficient. That is suitable for proving mass, energy and lineage accounting but too shallow for the submission world and for action-conditioned forecasting.

This slice introduces causal competition between rooms. The fan, shared resistance, branch resistance and damper positions determine delivered airflow. Commands no longer equal physical outcomes.

## Scenario-v3 schema

`aeolus_habitat_v2_scenario_v3` extends scenario-v2 through a new closed schema. It requires:

- 2..16 arbitrary unique, non-empty zone IDs
- exactly one operating mode per timeline segment
- zone geometry in habitat metres
- explicit air-network definition
- initial fan and damper actuator state
- fan-speed and damper-position commands instead of direct room airflow commands
- stable physical component IDs and referentially valid placement metadata

Old parsers remain closed. Scenario-v1 and scenario-v2 reject v3-only fields.

Identity dispatch is closed and per-version:

- scenario-v1 -> trace-v1 + equations-v1
- scenario-v2 -> trace-v2 + equations-v1
- scenario-v3 -> trace-v3 + equations-v2

No global equation identity is reinterpreted. A forged or unsupported schema,
trace and equation combination must fail before initial-state construction or
any physical transition. Existing v1/v2 canonical bytes, parser paths, run IDs,
replay behavior and checked digest fixtures remain unchanged.

## Reference world

The checked-in reference contains:

- laboratory
- common area and galley
- two crew-quarter zones
- hygiene and medical zone
- airlock and suitport vestibule
- equipment and power bay
- air-processing bay

The geometry is notional and labelled as such.

## Air-network model

### Components

- one variable-speed fan between return and supply sides
- shared supply-trunk resistance
- shared return-trunk resistance
- filter/scrubber resistance
- exactly one parallel branch per declared zone, including service and processing zones
- one motorised supply damper per branch
- one return branch per zone
- supply diffuser and return grille placement per zone

### Branch law

For each zone `i`:

`branch_delta_p = effective_branch_resistance_i * q_i * abs(q_i)`

For nonnegative recirculation flow:

`q_i = sqrt(branch_delta_p / effective_branch_resistance_i)`

### Damper law

`area_fraction_i = leak_fraction_i + (1 - leak_fraction_i) * actual_position_i`

`effective_supply_resistance_i = open_supply_resistance_i / area_fraction_i**2`

The branch effective resistance is supply plus return resistance.
`leak_fraction_i` is finite and in `(0, 1]`, so the denominator is always
strictly positive. A hard-zero-leak branch is outside this slice rather than
being represented as an infinite floating-point resistance.

### Shared loss

`shared_delta_p = shared_resistance * total_q * abs(total_q)`

where shared resistance includes declared supply trunk, return trunk and filter values.

### Fan curve

At speed fraction `s`:

- `s = 0` returns exactly zero pressure, flow and power without evaluating a
  normalized fan curve
- for `s > 0`, `Q_free(s) = Q_free_rated * s`
- `P_shutoff(s) = P_shutoff_rated * s**2`
- `P_fan(Q, s) = P_shutoff(s) * max(0, 1 - (Q / Q_free(s))**2)`

The rated free-delivery flow and rated shutoff pressure are finite and strictly
positive. Total efficiency is finite and in `(0, 1]`.

The operating point satisfies:

For a common nonnegative parallel-branch pressure `p_b`:

- `q_i(p_b) = sqrt(p_b / R_i)`
- `Q(p_b) = sum_i(q_i(p_b))`
- `F(p_b) = P_fan(Q(p_b), s) - R_shared * Q(p_b)**2 - p_b`

The operating point satisfies `F(p_b) = 0`.

The solver brackets `p_b` on `[0, P_shutoff(s)]` and performs exactly 100
bisection iterations. If `F(mid) > 0`, the lower endpoint becomes `mid`;
otherwise the upper endpoint becomes `mid`. The emitted operating-point
residual is `P_fan - shared_delta_p - p_b` in Pa. Accounting accepts a residual
only when `abs(residual) <= max(1e-9 Pa, 1e-10 * pressure_scale)`.

### Fan power

`air_power_w = fan_pressure_pa * total_q_m3_s`

`fan_electrical_power_w = air_power_w / total_efficiency`

No hidden empirical multiplier is applied.

### Fixed-density reduced-order closure

The network is incompressible at one declared reference density
`air_density_kg_m3` from the equipment contract. It does not infer local branch
density from each zone's changing gas state. For each zone:

- the same `q_i` is the reduced-order supply and return volumetric flow
- `mass_flow_i = air_density_kg_m3 * q_i`
- `mass_balance_residual_i = mass_supply_i - mass_return_i`, therefore zero for
  this one-loop topology

The trace labels this as fixed-density reduced-order closure. It is not a claim
that distinct local-density supply and return streams were solved.

### Actuator dynamics

- requested fan speed and damper positions are dimensionless fractions in `[0, 1]`
- actual state is rate limited by explicit per-second slew
- the actual state used during an interval is traceable
- commands outside bounds fail closed

## Trace-v3 additions

The v3 trace exposes both truth and receipts:

- requested fan-speed fraction
- actual fan-speed fraction
- requested and actual damper positions by component ID
- fan pressure rise in Pa
- shared pressure loss in Pa
- branch pressure loss by zone in Pa
- delivered zone airflow in m³/s
- density-derived zone mass flow in kg/s
- total airflow in m³/s
- fan air power and electrical power in W
- declared total efficiency and reference air density used by accounting
- per-zone fixed-density supply-return mass-balance residual in kg/s
- fan/system operating-point residual

The initial row uses null interval receipts where no step has run.
All component definitions and placement metadata are already inside canonical
scenario bytes. `scenario_sha256` is the sole component/placement lineage
authority: SHA-256 over UTF-8 JSON with sorted keys, no insignificant spaces and
non-finite values forbidden. Trace replay recomputes this digest and all rows,
rather than accepting separate decorative component or placement digest strings.

## Closed v3 wire contract

The parser requires exact key equality at every object boundary.

- top level: `schema_version`, `name`, `dt_seconds`, `steps`, `zones`,
  `equipment`, `initial_utility`, `timeline`, `air_network`
- zone: legacy physical fields plus `geometry`; geometry is exactly `center_m`
  and `size_m`, each a finite three-number SI vector, with strictly positive size
- air network: `supply_plenum_position_m`, `return_plenum_position_m`, `fan`,
  `shared_resistance`, `branches`
- fan: `id`, `rated_free_delivery_m3_s`, `rated_shutoff_pressure_pa`,
  `total_efficiency`, `speed_slew_fraction_per_s`, `position_m`
- shared resistance: `supply_trunk_pa_s2_m6`, `return_trunk_pa_s2_m6`,
  `filter_pa_s2_m6`
- branch: `zone_id`, `damper_id`, `open_supply_resistance_pa_s2_m6`,
  `return_resistance_pa_s2_m6`, `damper_leak_fraction`,
  `damper_slew_fraction_per_s`, `supply_diffuser_position_m`,
  `return_grille_position_m`, `damper_position_m`, `duct_polyline_m`
- initial actuator state replaces legacy `actual_airflow_m3_s` with exactly
  `actual_fan_speed_fraction` and `actual_damper_position_by_id`
- v3 command replaces legacy `airflow_m3_s` with exactly
  `fan_speed_fraction` and `damper_position_by_id`; all other plant command
  fields remain present
- each branch zone-ID set equals the declared zone-ID set exactly; each damper
  ID is unique; initial and commanded damper maps equal that damper-ID set
- trace row: all v2 row keys plus exactly `air_network_receipt`
- v3 commanded action: scrubber, condenser, cooling and oxygen maps plus fan and
  damper requests, with no direct airflow command
- v3 actual action: the same physical action fields plus derived per-zone
  `airflow_m3_s`
- network receipt: requested/actual fan and dampers; fan, shared and branch
  pressures; total and per-zone volumetric flow; per-zone mass flow; air and
  electrical power; efficiency; density; operating residual; mass residuals

Unknown, missing, non-finite, wrong-type, out-of-range or topology-mismatched
values fail closed before state construction.

## Executable invariants

1. Same scenario bytes produce byte-identical trace bytes.
2. Scenario-v1 and scenario-v2 canonical digests, run IDs and trace digests remain unchanged.
3. Zone and component IDs are unique and referentially valid.
4. Zone geometry dimensions and all resistance, fan and efficiency values are finite and physically signed.
5. Every declared zone has exactly one supply diffuser, return grille, supply damper, supply branch and return branch.
6. Fixed-reference-density supply and return mass flow match per zone within declared tolerance.
7. Total fan flow equals the sum of branch flows within tolerance.
8. The fan/system operating-point residual is within tolerance.
9. Zero actual fan speed yields zero pressure, airflow and fan electrical power.
10. Closing a damper cannot increase that branch's airflow with all other actual states fixed.
11. Increasing a branch or shared resistance cannot increase total flow with all other actual states fixed.
12. Fan electrical power equals pressure times flow divided by declared efficiency within tolerance.
13. Delivered airflow, not requested command, drives thermodynamic recirculation.
14. Existing non-fan electrical receipts remain closed. V3 uses derived fan electrical power instead of the legacy linear coefficient.
15. Geometry placement metadata never silently changes physics values.
16. V1 and V2 remain bound to equations-v1; V3 is bound to equations-v2.
17. Mass flow equals measured volumetric flow times declared air density within tolerance.

## TDD and implementation sequence

1. RED then GREEN: standalone air-network value objects and closed validation.
2. RED then GREEN: zero-speed and one-branch analytical cases.
3. RED then GREEN: deterministic multibranch operating-point solve and conservation residuals.
4. RED then GREEN: fan and damper actuator slew.
5. RED then GREEN: scenario-v3 parser while v1/v2 remain closed and byte-preserved.
6. RED then GREEN: v3 plant integration uses delivered network flow and derived fan power.
7. RED then GREEN: trace-v3 receipts, replay and forgery rejection.
8. RED then GREEN: checked-in eight-zone reference scenario and CLI run.
9. Focused Habitat V2 tests throughout.
10. One complete suite at PR boundary, plus lint, compile, build and clean-wheel smoke.

## Non-goals for this PR

- physical and sensor fault injection
- automatic control or safety policy
- training-corpus generation
- learned model implementation or training
- model quantization
- frontend implementation
- arbitrary general nonlinear duct networks
- CFD
- exterior leakage, buoyancy or wind pressure
- manufacturer-certified fan and damper curves

## Review boundary

One bounded independent review will focus on:

- v1/v2 preservation
- solver monotonicity and convergence
- pressure, flow and power dimensions
- command-versus-actual actuator separation
- trace receipt integrity
- geometry-to-component referential consistency

Only finding-specific corrections and targeted retesting are allowed before the PR boundary suite.
