# Habitat V2 actuator-feedback contract

Date: 2026-08-12
Status: implemented locally, exact candidate and independent review pending
Base: `42e85e23bf065203b3d5b68d4b7398300bbb807a`
Version impact: minor, resulting package version `0.7.0`

## Purpose

This bounded PR adds an honest operational feedback boundary before HMC or any model-input work. It keeps requested commands, achieved actuator state, fault-effective delivery, physical response and measured feedback as separate layers.

It does not add an HMC, safety supervisor, model, training corpus or control authority.

## Closed V5 lineage

- scenario: `aeolus_habitat_v2_scenario_v5`
- trace: `aeolus_habitat_v2_trace_v5`
- equations: `aeolus_habitat_v2_equations_v4`
- actuator feedback: `aeolus_habitat_v2_actuator_feedback_v1`

V1 through V4 identities and canonical replay bytes remain frozen.

## Causal actuator path

Each completed V5 step follows this order:

1. validate the requested command
2. apply physical and policy slew limits to obtain achieved state
3. apply bounded physical-effectiveness faults
4. advance canonical plant physics
5. instrument the completed physical response
6. apply deterministic feedback noise and feedback-sensor faults
7. emit causal receipts and the measured operational row

The V5 trace exposes:

- `commanded_action`: requested command
- `actual_action`: achieved actuator state
- `actuator_receipt`: requested, achieved and fault-effective layers for fan, dampers, scrubber, condenser, cooling and oxygen
- `air_network_receipt`: physical airflow, pressure and power response
- `operational_feedback`: measured feedback after deterministic noise and sensor faults
- `telemetry` and `sensor_disagreement`: measured environmental response
- `fault_receipt`: evaluator-only truth that must never enter the later HMC or model input

## Actuator and fault semantics

Cooling and oxygen achieved state is immutable per completed step and obeys declared slew limits. Fan and damper achieved state remains distinct from fault-effective response.

Scrubber, condenser, cooling and oxygen effectiveness multipliers are bounded in `[0, 1]`. Same-authority overlapping effectiveness faults are rejected. All fault intervals are half-open `[start_step, end_step)`.

A fan degradation leaves its achieved setpoint intact while reducing the fault-effective speed used by physics. Measured fan-speed feedback is instrumented from that fault-effective physical response, not copied from the achieved setpoint.

## Operational feedback

Noise is deterministic and keyed by scenario sensor seed, resource or component ID, channel ID and completed step. Feedback bias and stuck profiles are applied after noise and before the final channel clamp.

The fan-current channel is a reduced-order DC-bus current computed as electrical fan power divided by declared bus voltage. It is not a hardware-qualified motor model.

## External command boundary

`advance_one_step_with_command` validates the full current-step command before mutation and binds its canonical SHA-256 digest into the step receipt. The caller may replace only the current command. The scenario remains authoritative for mode, load, generation, fault schedule, seed, topology and equipment limits.

## Safety and claims boundary

This is a deterministic reduced-order research analogue. It is not CFD, flight software, a certified spacecraft digital twin or physical validation. Exact effective values and evaluator truth may support evaluation, but the later HMC and model-input contracts must accept operational observations only.
