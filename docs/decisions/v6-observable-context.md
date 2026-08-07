# V6 observable-context boundary

**Status:** frozen before V6 specialist training.

`observable_context_v1` is a versioned `float32` projection for V6 diagnostic and specialist-policy features. It is separate from `model_input_v1`; no existing V5 artifact can be loaded against it.

## Included observable telemetry

For the current three non-processing-zone hub topology, the contract contains 46 ordered fields:

- 3 zone sensor CO₂ readings;
- 21 actuator fields: setpoint, actual position, tracking residual, moving flag, movement duration, power, and direction for each zone;
- 18 airflow fields: requested, delivered, and residual flow for both directions of every loop;
- 4 system fields: shared capacity, total requested flow, total delivered flow, and capacity scale.

The selector is canonical JSON and is bound to the same validated directed topology digest as `model_input_v1`. A missing, renamed, or extra zone/actuator/connection is rejected before projection. Values are checked after conversion to `float32` so finite source values cannot silently become infinities.

## Deliberate exclusions

The following trace values are **not** V6 candidate inputs in this contract:

- `occupancy_multiplier` — available only in simulator traces and allowed for historical forensics; not assumed observable in deployment;
- `co2_mass` — hidden plant state;
- `source_co2_mass` — hidden simulator source state.

A regression mutates all three and proves the context tensor is unchanged.

## Consequence

A later V6 specialist may use the context only after its own train/calibration/validation split and artifact metadata bind this contract. The context adds physical-control observables; it does not solve the causal-identifiability problem by leaking simulator truth.
