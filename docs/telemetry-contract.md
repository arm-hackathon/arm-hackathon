# Project AEOLUS telemetry contract

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

## Gate 1: topology-bound model input

`aeolus.trace.model_feature_row()` remains the narrow observable projection. Gate
1 adds `aeolus.model_input.model_input_v1(record, contract)`: it validates the
record through that projection and returns an exact `numpy.float32` tensor with
shape `(24,)`. It does not accept fault labels, schedules, target metadata,
health, seeds, or any presentation/debug field.

Before selecting any scalar, `model_input_v1()` requires the record's exact
zone, actuator and connection ID sets to match the topology embedded in its
contract. Extra or missing entities, including unselected processing-zone and
return-leg telemetry, are rejected rather than silently ignored.

For the accepted three-zone habitat, the ordered values are:

```text
zones.cabin_a.sensor_co2_concentration
zones.cabin_b.sensor_co2_concentration
zones.lab.sensor_co2_concentration
actuators.cabin_a.setpoint
actuators.cabin_a.actual_position
actuators.cabin_a.tracking_residual
actuators.cabin_a.power
actuators.cabin_b.setpoint
actuators.cabin_b.actual_position
actuators.cabin_b.tracking_residual
actuators.cabin_b.power
actuators.lab.setpoint
actuators.lab.actual_position
actuators.lab.tracking_residual
actuators.lab.power
connections.cabin_a_to_processing.requested_airflow
connections.cabin_a_to_processing.delivered_airflow
connections.cabin_a_to_processing.airflow_residual
connections.cabin_b_to_processing.requested_airflow
connections.cabin_b_to_processing.delivered_airflow
connections.cabin_b_to_processing.airflow_residual
connections.lab_to_processing.requested_airflow
connections.lab_to_processing.delivered_airflow
connections.lab_to_processing.airflow_residual
```

The selector derives its connection IDs from the validated `HabitatConfig`
hub graph, not identifier suffixes. It uses only each non-processing zone's
outbound leg; processing-zone sensor telemetry and all return-leg telemetry are
therefore deliberately absent. A different graph may build a contract, but a
v1 selector with any count other than 24 fields is rejected.

`build_model_input_contract(config)` returns canonical JSON and SHA-256 hashes
for both boundaries:

- topology JSON contains `schema_version`, the processing-zone ID, ordered
  non-processing zone IDs, and ordered primary-loop outbound/return directed
  edges (`id`, `from_zone`, `to_zone`);
- selector JSON contains `schema_version: model_input_v1`, `dtype: float32`,
  shape `[24]`, the ordered field list, and the topology hash;
- `topology_hash` and `selector_hash` are the SHA-256 digests of their exact,
  sorted-key, compact UTF-8 JSON bytes.

`model_artifact_metadata(contract)` emits exactly `model_input_version`,
`selector_sha256`, and `topology_sha256`. No model artifact loader exists in
Gate 1. Every future artifact-load or inference-setup path must call
`assert_model_contract_compatible(metadata, contract)` before use; missing,
extra, non-string, stale, or mismatched metadata is rejected. The contract also
self-validates its canonical JSON and hashes before a tensor is built.

`RuleBaseline` receives the validated `HabitatConfig` at construction and
pairs outbound/return legs from graph direction. It fails closed when a feature
window's zone, actuator or connection IDs do not match that topology. Renaming
connection IDs therefore preserves loop behavior when the graph and telemetry
agree.

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

`aeolus.corpus` builds the labelled window corpus for the future fault
classifier. Its leakage rules are strict:

- corpus v1 feature rows remain exactly `model_feature_row()` output for each
  tick; corpus v2 is out of scope for Gate 1 and must use exact
  `model_input_v1()` float32 values plus persisted selector/topology hashes;
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
