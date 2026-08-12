# Project AEOLUS telemetry contract

This contract separates observable replay data from simulator truth. It binds
trace writers, model-facing projections, recovery traces, and visualisation
changes.

Schema-v9 describes the standard plant. Schema-v10 adds a separately validated
reserve topology and recovery trace envelope; it does not widen
`model_input_v1`. C4 recovery development is a reproducible negative result:
the reserve and authority fields are trace/audit data, not evidence of an
accepted controller or qualified model.

## Fault target semantics

Schema-v9 and schema-v10 retain PR #9's `connection_id` for connection faults
(the gradual-primary-fan and blocked-path profiles). The value is an **outbound
loop metering identifier**, not a claim that the JSON edge is a physical fan or
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

`requested_airflow` describes controller demand recomputed from measured
actuator position. `delivered_airflow` describes the bounded observation of the
physical loop after static health, fault effectiveness, shared-capacity
allocation, fixed meter bias and per-tick measurement noise.
`airflow_residual` is always measured request minus measured delivery.

System telemetry contains shared capacity, total measured requested airflow,
total measured delivered airflow and the physical capacity scale. Zone records
contain replay/presentation values such as CO₂ mass, sensor concentration,
generated source mass and occupancy multiplier. Actuator records contain
setpoint, measured position, tracking residual, movement and power.

## Schema-v10 recovery trace envelope

A recovery row is a distinct `aeolus_recovery_trace_v1` document with exactly
these top-level fields:

```text
plant
reserve
authority
schema_version
```

`plant` is the unchanged validated legacy projection. `reserve` holds reserve
connection, actuator, and aggregate-system telemetry. `authority` binds run and
epoch identity, causal observation/decision ticks, state, owner, target, reason,
dwell, and matching command digests. The writer rejects topology mismatches,
unknown fields, non-finite values, reserve delivery above capacity, inconsistent
request/delivery residuals, invalid state/owner combinations, and a command
that has not been acknowledged as applied.

A recovery trace makes the reserve mechanism observable for auditing. It does
not alter `model_feature_row()`: that function projects only the validated
`plant` fields, so reserve telemetry, authority state, target, reason, epoch,
and digests are not current model features. Any attempt to introduce them would
require a new versioned selector contract and a separate leakage review.

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
`selector_sha256`, and `topology_sha256`. The detector JSON and ONNX artifacts
embed those values. The prediction path builds the scenario's contract and
rejects missing, extra, non-string, stale, or mismatched metadata before
inference. The contract also self-validates its canonical JSON and hashes
before a tensor is built.

`RuleBaseline` receives the validated `HabitatConfig` at construction and
pairs outbound/return legs from graph direction. It fails closed when a feature
window's zone, actuator or connection IDs do not match that topology. Renaming
connection IDs therefore preserves loop behavior when the graph and telemetry
agree.

## Scenario-family manifest (Gate 2 accepted)

`scenarios/families.json` defines the unit that future training and evaluation
splits must keep independent. Each family names one fault-free reference
scenario, one exactly-one-fault scenario, a declared class, and exactly one of
`train`, `validation`, `test`, or `stress`. The current three checked-in fixture
families are test-only
contract fixtures; they do not claim to be a usable training corpus.

`load_family_manifest()` rejects unknown fields, duplicate family IDs or exact
reference/fault pairs, source scenarios reused across splits, stale or malformed
Gate-1 metadata, fault-bearing references, multi-fault paired runs, class
mismatches, and any reference/fault pair that differs outside `fault_profiles`.
The latter rule requires a true counterfactual: the
`frozen_sensor_healthy.json` control preserves the lab demand transition from
`frozen_sensor.json` and removes only the freeze profile.

The manifest is canonical sorted-key compact JSON with a SHA-256 identity. Its
family list is ordered by `family_id` for hashing, so source-file ordering does
not alter the contract.

Schema-v9 family equality includes the complete `telemetry` settings. A paired
reference and fault scenario must therefore use identical demand, seed,
measurement-noise and measurement-bias configuration and differ only in
`fault_profiles`.

`observable_onset(family, metadata)` replays the validated pair and finds the
first equal-tick difference between their `model_input_v1` float32 vectors. It
fails closed for metadata/topology mismatch, unequal trace lengths, inconsistent
tick numbering, or a pair that never differs. It persists the scenario hashes
and frozen contract metadata with the resulting tick.

Corpus v2 emits one healthy-reference stream and one fault stream per family.
Healthy and pre-onset fault windows are `nominal`; fully observable post-onset
windows receive the family fault class; an onset-straddling window is
`excluded_transition`. Evaluators feed every window to stateful labellers, but
exclude transition windows from training and from scored accuracy, confusion,
class support, and latency totals. They reject mixed, missing, or stale
model-input contract metadata.

## C4 schema-v10 recovery development

`scenarios/sweep-recovery-development.json` is an
`aeolus_sweep_v4` **development** specification. Every generated family has a
validated primary/reserve topology and declares the fixed four counterfactual
arms. `aeolus.recovery_evidence` requires a clean source checkout, refuses an
existing output directory, hashes the source/sweep/settings/scenarios/traces,
and writes an evidence receipt with separate safety and benefit predicates.

C4 at `74154956d64309f067ada7593e2ef8786d140b4e` produced a deterministic,
reproducible negative result. The transient physical-zero acknowledgement safety
predicate and physical-reserve-delivery benefit predicate are false. The runner
therefore does not authorise adviser training, model integration, export,
threshold tuning, or final-suite work. `docs/recovery-protocol-acceptance.md`
is the binding record of that outcome.

## Historical sweep v3 and frozen policy artifacts

`scenarios/sweep-v2.json` is historical experimental context. The v3 development
and final specifications remain archived with their historical protocol record;
they are not C4 inputs and must not be run, inspected, or used to revise the C4
recovery decision.

The development specification assigns disjoint scenario families to train and
validation only. `aeolus.protocol.select_development` trains both learned
candidates from train rows, selects by validation macro-F1, cross-entropy,
serialized size and name, calibrates rules on validation from the fixed 216-point
grid, exports FP32 ONNX and enforces validation parity. It persists the complete
selection and calibration receipts in a strict frozen policy.

The final specification contains only `final` families. The final evaluator
requires development and final corpora/manifests, a policy, detector JSON and
ONNX artifact plus expected hashes. It checks family disjointness and contract
metadata; reconstructs candidate selection, ONNX parity and rule calibration
from development rows; verifies the saved validation comparison; then applies
the frozen policy to final rows once. It does not select a candidate, tune a
threshold or revise the policy outcome from final evidence.

Both learned candidates consume exact ten-row `model_input_v1 float32[24]`
windows. Softmax flattens to 240 values. `TemporalMLPDetector` embeds
`temporal_summary_v1`: five summaries of each channel and three safe loop
residual/request ratios, producing 135 values for a 16-unit ReLU hidden layer.
Normalization is calculated from training rows only. Training is
class-balanced and excludes transition rows.

The robust `RuleBaseline` is selected independently on validation from a fixed
216-point grid. Classification excludes onset-straddling windows, while causal
latency uses stride-one rolling windows and permits a correct detection from
`end_tick >= observable_onset_tick`. Detection latency is simulator ticks, not
wall-clock inference performance.

The exact prediction classes are `nominal`,
`gradual_primary_fan_degradation`, `blocked_path`, and `frozen_sensor`.
Both detector classes implement `predict_window()` and return the selected
label, confidence, and named probabilities. The rolling CLI emits those values
with each `end_tick`.

The strict JSON loader rejects unknown artifact fields, wrong input shape,
class-vocabulary drift, non-finite parameters, invalid normalization scales,
and selector/topology mismatch. The FP32 ONNX graph embeds the same metadata.
See [protocol v3 acceptance](protocol-v3-acceptance.md) for the measured result,
independence boundary and reproduction commands.

## Forbidden hidden truth

The following must not enter model features. Standard plant telemetry excludes
these values entirely; a recovery trace may contain separately allowlisted
reserve/authority audit fields, but `model_feature_row()` still excludes them.

- injected fault type or label;
- fault start/end schedule;
- hidden effectiveness;
- static connection health;
- random seed;
- internal source-noise state;
- internal measurement-noise samples or bias state;
- which zone or connection a declared fault targets;
- a zone's frozen-sensor state or stored freeze value (the held sensor reading
  itself is telemetry; the fact that it is held is not);
- recovery authority state, reason, target, epoch, owner, or command digests;
- reserve-path telemetry, reserve command state, or reserve capacity values;
- future values or labels derived from them.

A trace writer validates the observable allowlist before serialising a row.
The visualiser independently rejects undeclared connection telemetry. Tests
assert that a degradation replay and its model projection contain none of the
forbidden fields.

## Corpus boundary

`aeolus.corpus` builds the labelled window corpus for the fault classifier. Its
leakage rules are strict:

- corpus v1 feature rows remain exactly `model_feature_row()` output for each
  tick; corpus v2 uses exact `model_input_v1()` float32 values plus persisted
  selector/topology hashes;
- corpus-v2 labels derive from paired observable onset, never from a declared
  fault-profile start tick; healthy and pre-onset rows are `nominal`,
  onset-straddling rows are `excluded_transition`, and only fully observable
  rows receive the single family fault class;
- a window where more than one fault is active is rejected; multi-fault
  taxonomy remains out of scope;
- corpus output (`corpus.jsonl`, `manifest.json`) is a generated artifact and
  belongs under `out/`, not in git;
- regenerating from the same scenarios, window and stride is byte-identical.

## Change rule

Any telemetry change must update this document, the trace validator, visualiser
validation, the corpus projection and tests in the same change. Adding a chart
is not permission to add a model feature.
