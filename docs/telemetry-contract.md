# ICARUS telemetry contract

This contract separates observable replay data from simulator truth. It is
binding for trace writers, model-facing projections and visualisation changes.

## Fault target semantics

Schema-v7 keeps PR #9's `connection_id` for the gradual primary-fan profile.
The value is an **outbound loop metering identifier**, not a claim that the
JSON edge is a physical fan or duct. Each non-processing zone has one stable
outbound meter into the processing bay and one paired return path. Applying the
fault multiplier once to that loop preserves the accepted PR #9 identifier
while correctly reducing both legs' delivered airflow equally.

The configuration validator only accepts an existing connection that ends at
the processing bay. It rejects return paths, unknown targets and duplicate
targets. The corresponding physical loop remains constrained by both outbound
and return static health.

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
- future values or labels derived from them.

A trace writer validates the observable allowlist before serialising a row.
The visualiser independently rejects undeclared connection telemetry. Tests
assert that a degradation replay and its model projection contain none of the
forbidden fields.

## Change rule

Any telemetry change must update this document, the trace validator, visualiser
validation and tests in the same change. Adding a chart is not permission to
add a model feature.
