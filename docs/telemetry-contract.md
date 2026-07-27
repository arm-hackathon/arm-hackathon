# ICARUS telemetry contract

This contract separates observable replay data from simulator truth. It is
binding for trace writers, model-facing projections and visualisation changes.

## Fault target semantics

Schema-v7 keeps PR #9's `connection_id` for connection faults (the gradual
primary-fan and blocked-path profiles). The value is an **outbound loop
metering identifier**, not a claim that the JSON edge is a physical fan or
duct. Each non-processing zone has one stable outbound meter into the
processing bay and one paired return path. Applying the fault multiplier once
to that loop preserves the accepted PR #9 identifier while correctly reducing
both legs' delivered airflow equally.

The configuration validator only accepts an existing connection that ends at
the processing bay. It rejects return paths, unknown targets and duplicate
targets. The corresponding physical loop remains constrained by both outbound
and return static health.

The `frozen_sensor` profile targets a zone instead of a connection: from its
`start_tick`, the zone's sensor holds the reading it showed on the first
frozen tick. The held reading is ordinary observable telemetry — what changes
is that the controller can no longer see the true concentration. The
validator accepts only existing non-processing zones and rejects duplicate
zone targets.

## Trace schema

Each JSONL row has exactly these top-level fields:

```text
tick
zones
connections
actuators
system
```

Connection telemetry is intentionally narrow:

```text
requested_airflow
delivered_airflow
airflow_residual
```

`requested_airflow` describes controller demand at the measured actuator
position. `delivered_airflow` describes the physical loop result after static
health, deterministic degradation and shared-capacity allocation.
`airflow_residual` is request minus delivery.

System telemetry contains shared capacity, total requested delivery, total
delivered airflow and capacity scale. Zone records contain replay/presentation
values such as CO₂ mass, sensor concentration, generated source mass and
occupancy multiplier. Actuator records contain setpoint, measured position,
tracking residual, movement and power.

## Explicit model projection

No model is implemented in this slice. If a future consumer needs a feature
row, it must use `icarus.trace.model_feature_row()`, which returns only:

| Entity | Allowlisted values |
|---|---|
| Zone | `sensor_co2_concentration` |
| Actuator | `setpoint`, `actual_position`, `tracking_residual`, `power` |
| Connection | `requested_airflow`, `delivered_airflow`, `airflow_residual` |

The projection deliberately excludes occupancy and source mass even though
those presentation/debug fields remain in a trace. Visualisation requirements
do not change this projection.

## Forbidden hidden truth

The following must not enter model features or replay telemetry:

- injected fault type or label;
- fault start/end schedule;
- hidden effectiveness;
- static connection health;
- random seed;
- internal source-noise state;
- which zone or connection a declared fault targets;
- a zone's frozen-sensor state or stored freeze value (the held sensor reading
  itself is telemetry; the fact that it is held is not);
- future values or labels derived from them.

A trace writer validates the observable allowlist before serialising a row.
The visualiser independently rejects undeclared connection telemetry. Tests
assert that a degradation replay and its model projection contain none of the
forbidden fields.

## Corpus boundary

`icarus.corpus` builds the labelled window corpus for the future fault
classifier. Its leakage rules are strict:

- every corpus feature row is exactly `model_feature_row()` output for that
  tick — no derived statistics, no extra fields;
- labels come from the scenario's declared fault profiles evaluated at the
  window's final measured tick (configuration truth), never from telemetry;
- a window with no active fault is labelled `nominal`; a window where more
  than one fault is active is rejected, because corpus v1 ships single-fault
  scenarios only;
- corpus output (`corpus.jsonl`, `manifest.json`) is a generated artifact and
  belongs under `out/`, not in git;
- regenerating from the same scenarios, window and stride is byte-identical.

## Change rule

Any telemetry change must update this document, the trace validator, visualiser
validation, the corpus projection and tests in the same change. Adding a chart
is not permission to add a model feature.
