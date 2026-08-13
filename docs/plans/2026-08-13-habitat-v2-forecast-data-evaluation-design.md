# Habitat V2 forecast data/evaluation design

**Status:** proposed local implementation contract; non-normative and not team-frozen

**Design base:** `ec8a6e07cddcd97915398e5a84d348b30d850c86`

**Observed candidate:** `ed3fd5c949a382ec8ffdb060733990dd00803777`

**Moving HMC candidates inspected:** `6fd307317d7b8ccdb78ba2cf21f5bed3ffd084b5`, `0445af158edbbd7189dcbe7cad8600ca35deddb0`, `50d515a50480ba91a95ff8c8e3c6b65534d39665`

**HMC status:** moving; all `${FINAL_HMC_*}` values below are unresolved bindings

**Scope:** data and evaluation foundation only

This document recommends one bounded contract for converting verified Habitat
V2 operational histories plus complete proposed actions into leakage-resistant
multi-step forecast samples. It also defines deterministic splitting,
baselines and neutral evaluation. It does not select a learned architecture,
train a model, define direct control or approve any contract on the team's
behalf.

## 1. Inspected repository material

The review used a separate checkout and inspected the following read-only.

### Canonical foundation

- `src/aeolus/habitat_v2/physics.py`
- `src/aeolus/habitat_v2/state.py`
- `src/aeolus/habitat_v2/runner.py`
- `src/aeolus/habitat_v2/scenario.py`
- `src/aeolus/habitat_v2/instrumentation.py`
- `src/aeolus/habitat_v2/telemetry.py`
- Habitat V2 V5 scenarios and tests

### Operational observability candidate

- `contracts/habitat_v2_observability_v1.json`
- `src/aeolus/habitat_v2/observability.py`
- `src/aeolus/habitat_v2/qualification.py`
- observability qualification scenarios, evidence and tests

### Moving HMC candidates

- `contracts/habitat_v2_hmc_v1.json`
- `src/aeolus/habitat_v2/hmc_contract.py`
- `src/aeolus/habitat_v2/hmc.py`
- `src/aeolus/habitat_v2/snapshot.py`
- `src/aeolus/habitat_v2/health.py`
- `src/aeolus/habitat_v2/safety.py`
- `src/aeolus/habitat_v2/proposal.py`
- `src/aeolus/habitat_v2/control_trace.py`
- focused HMC lifecycle, proposal, snapshot, safety and trace tests

The older `6fd3073...` HMC restack was used to enumerate provisional fields.
The later HMC trace-authority remediation was inspected to identify API and
identity changes, but no moving head is treated as final.

### Historical reusable patterns

- `src/aeolus/corpus.py`
- `src/aeolus/families.py`
- `src/aeolus/protocol.py`
- `docs/corpus-v2-acceptance.md`
- `docs/protocol-v3-acceptance.md`
- their tests
- `docs/plans/2026-08-13-habitat-v2-model-admission-prd.md` at
  `cf089be77b0b2175b71223af851688c6b7bc22fd`

These supply parser, split and artifact-freeze patterns only. Their historical
four-class target, model input, corpus and conclusion are not reused.

## 2. Governing causal contract

Let `a` be an admitted completed observation step, `W` the history length and
`H` the future horizon.

```text
history snapshots:       a-W+1, ..., a
proposal decision:       after completed snapshot a
requested application:   transition a -> a+1
forecast targets:        completed states a+1, ..., a+H
continuation:            NO_PROPOSAL after the anchor
```

The history row at `a` contains the command that produced completed state `a`.
The candidate action is a separate input. It cannot influence the history.

HMC arbitrates the anchor proposal. Accepted, modified and rejected outcomes
remain in the corpus. After the anchor, HMC receives `NO_PROPOSAL` and retains
full authority over safe hold, emergency override and plant stepping.

The source run must have one constant operating mode. The complete interval
`[a-W+1, a+H]` must not cross a mode transition, fault onset or recovery
boundary. Reset step zero is not model input.

Observation cadence is exactly `60.0 s`. Rather than guessing the temporal
shape, a disjoint public timing pilot selects:

```text
W in {4, 8, 16}
H in {2, 4, 8}
```

The selected `W`, `H`, pilot manifest and selection receipt become part of the
final input and target contract identities.

## 3. Model-input contract

Proposed schema ID:

```text
aeolus_habitat_v2_forecast_input_v1
```

### 3.1 Topology binding

Version 1 supports only the inspected eight-zone topology. Zone order is
lexicographic:

```text
air_processing_bay
airlock_suitport
common_galley
crew_quarters_a
crew_quarters_b
equipment_power_bay
hygiene_medical
laboratory
```

Damper order follows HMC `ObservableTopology.branch_pairs` in the same zone
order. The fan is `primary_supply_fan` in the inspected scenarios.

The inspected candidate topology SHA-256 was:

```text
b0246a9dc8f847c3236068c8e1eeeddb31809a680e6133eaf038ea197d6e10e6
```

This value is evidence about the inspected candidate, not the final binding.
The implemented contract must require `${FINAL_HMC_OBSERVABLE_TOPOLOGY_SHA256}`
and an exact ordered topology manifest.

### 3.2 Operational sample order

Each completed snapshot contributes 167 operational samples.

Environmental channel order:

| Offset within a zone | Channel | Unit |
|---:|---|---|
| 0 | `temperature_k` | `K` |
| 1 | `pressure_pa` | `Pa` |
| 2 | `co2_ppm` | `ppm` |
| 3 | `o2_mole_fraction` | `mole_fraction` |
| 4 | `relative_humidity` | `fraction` |

Append, in order:

1. primary environmental telemetry, zone-major: 40 samples;
2. secondary environmental telemetry, zone-major: 40 samples;
3. primary-minus-secondary, zone-major: 40 samples;
4. operational feedback in the order below: 47 samples.

| Feedback channel | Expansion | Unit |
|---|---:|---|
| `fan_speed_fraction` | scalar | `fraction` |
| `fan_dc_bus_current_a` | scalar | `A` |
| `damper_position_by_id` | 8 dampers | `fraction` |
| `branch_airflow_m3_s` | 8 zones | `m3_s` |
| `branch_differential_pressure_pa` | 8 zones | `Pa` |
| `scrubber_capture_rate_mol_s` | scalar | `mol_s` |
| `condenser_removal_rate_mol_s` | scalar | `mol_s` |
| `cooling_delivery_w` | 8 zones | `W` |
| `oxygen_delivery_mol_s` | 8 zones | `mol_s` |
| `battery_state_of_charge` | scalar | `fraction` |
| `oxygen_store_fraction` | scalar | `fraction` |
| `sorbent_remaining_fraction` | scalar | `fraction` |

For every sample, persist its canonical binary64-or-null value and HMC status
in the replay witness. Project it to:

- `float32(value)` when available, otherwise exactly `0.0f`;
- a five-way float32 one-hot status in this order:

  ```text
  AVAILABLE
  MISSING
  NON_FINITE
  MALFORMED
  DEPENDENCY_UNAVAILABLE
  ```

Reject descriptor, unit, availability, reason and value inconsistencies.
Reject conversion overflow or a non-finite float32 result.

### 3.3 Previous authoritative command

Each history row must have command-reference kind `COMPLETED_FINAL_COMMAND`.
Append these 27 finite values:

```text
fan_speed_fraction                         fraction
damper_position_by_id[8]                  fraction
scrubber_duty                             fraction
condenser_duty                            fraction
cooling_removed_w[8]                      W
oxygen_injection_mol_s[8]                 mol_s
```

The 167 operational values plus 27 command values form 194 numeric values per
history row.

### 3.4 Mode and health

Operating-mode one-hot order:

```text
dormant, occupied, eva_transition, contingency
```

HMC health-state one-hot order:

```text
NOMINAL, DEGRADED, CRITICAL, UNKNOWN
```

Both are operational inputs because HMC arbitration depends on them. Omitting
them aliases identical sensor values that HMC treats differently.

### 3.5 Alarm lifecycle

Materialise a closed, ordered alarm-slot manifest from the final HMC health
policy and topology. For every slot, encode:

```text
ABSENT, RAISED, ACTIVE, CLEARED
```

The inspected candidate produces 287 possible alarm IDs for the fixed
topology. That count is provisional. The final contract must list every
`alarm_id`, expected family, target and severity and bind the ordered manifest
by SHA-256. A missing known alarm is `ABSENT`; an unknown or duplicate alarm ID
is invalid.

### 3.6 Duplicate resource gauges

The snapshot exposes the three resource gauges both through operational
feedback and a dedicated resource-gauge block. Require semantic equality, then
include them only through operational feedback.

### 3.7 Proposed action

The action is one complete canonical V5 external command using the same
27-value order as the previous authoritative command. It is an absolute command,
not a delta, partial update or action-class number.

Validate the canonical command against the bound topology, final external-
command contract and scenario capacities before projection. Action ID, source,
confidence, catalogue metadata and later HMC disposition are provenance only.

### 3.8 Tensor interface

```text
history_numeric_f32[W,194]
history_sample_status_f32[W,167,5]
history_mode_f32[W,4]
history_health_f32[W,4]
history_alarm_lifecycle_f32[W,A,4]
proposed_action_f32[27]
```

`A` is the exact final alarm-manifest length; the inspected candidate suggests
`A=287`.

### 3.9 Forbidden model-facing fields

The projector accepts only the contracted operational snapshot subobjects and
the complete proposed action. It must reject or be structurally unable to read:

- fault profiles, type, target, timing and effectiveness;
- family, split, scenario, run and sample identities as features;
- raw seeds and sensor-noise state;
- exact hidden gas moles, Wh/mol inventories or effective actuator values;
- exact realised loads or future schedules;
- future modes, observations, proposals, commands and loads;
- physical target truth, including the evaluator-only anchor truth;
- fault, accounting, air-network and actuator receipts;
- plant invariant residuals and hidden resource state;
- anchor or later proposal outcome, disposition, reason codes, accepted
  command, emergency flag, preflight result and plant receipt;
- hashes, IDs, timestamps and catalogue labels presented as numeric features.

Changing any forbidden metadata while retaining the same operational snapshot
and action must not change the projected tensor.

## 4. Forecast-target contract

Proposed schema ID:

```text
aeolus_habitat_v2_forecast_target_v1
```

For each future completed step, output 51 physical values. The canonical target
truth remains binary64; the model/evaluator tensor is `float32[H,51]`.

Zone-major field order:

| Offset within a zone | Field | Unit |
|---:|---|---|
| 0 | `temperature_k` | `K` |
| 1 | `pressure_pa` | `Pa` |
| 2 | `co2_ppm` | `ppm` |
| 3 | `o2_mole_fraction` | `mole_fraction` |
| 4 | `relative_humidity` | `fraction` |
| 5 | `branch_airflow_m3_s` | `m3_s` |

After 48 zone values, append:

```text
battery_state_of_charge       fraction
oxygen_store_fraction         fraction
sorbent_remaining_fraction    fraction
```

Targets are instantaneous completed-state values at `+60, ..., +60H` seconds.
There is no temporal averaging, interpolation, masking, shortening, padding or
imputation. Retain the evaluator-only physical anchor vector for crossing
derivation, but never expose it to model input.

Reject the complete sample if any target is missing, non-finite, outside the
closed topology, physically invariant-invalid or unavailable because the HMC
run terminated before the full horizon.

Physical truth is the target rather than later noisy sensor samples. This asks
the forecaster to predict the habitat response to an action, not an independent
future sensor-noise draw.

## 5. Envelope-crossing and candidate-output contracts

### 5.1 Crossings

Crossing polarity is **harmful-positive**. A new physical warning-entry crossing
occurs when the prior physical value is strictly safe and the next value meets
or exceeds a harmful high threshold, or meets or falls below a harmful low
threshold. Equality is unsafe.

Bind final thresholds to `${FINAL_HMC_HEALTH_POLICY_SHA256}`. The inspected
candidate defines:

| Envelope | Warning entry |
|---|---:|
| CO2 | `>= 2500 ppm` |
| O2 | `<= 0.285` |
| low temperature | `<= 291 K` |
| high temperature | `>= 300 K` |
| relative humidity | `>= 0.65` |
| battery reserve | `<= 0.20` |
| oxygen-store reserve | `<= 0.20` |
| sorbent reserve | `<= 0.20` |

Pressure and airflow have regression metrics only because the inspected HMC
defines no approved warning envelope for them.

An anchor already outside an envelope is `PREEXISTING_HARM`: retain it for
regression and unsafe-occupancy reporting, but exclude it from new-crossing
classification for that envelope.

### 5.2 Candidate output

Proposed schema ID:

```text
aeolus_habitat_v2_forecast_output_v1
```

The output binds sample, input, target, model and preprocessing identities and
has one of two statuses:

```text
PREDICTION
  values_f32: exact H x 51 finite values
  abstention_reason: null

ABSTAIN
  values_f32: null
  abstention_reason:
    INPUT_UNAVAILABLE | OUT_OF_SUPPORT | SELECTIVE_RISK_THRESHOLD
```

Malformed shape, timeout, exception, non-finite value or domain-invalid value
is `INVALID_OUTPUT`, never an abstention or silently repaired prediction.

## 6. Corpus, canonicalisation and provenance

Proposed schema IDs:

```text
aeolus_habitat_v2_forecast_corpus_v1
aeolus_habitat_v2_forecast_split_v1
aeolus_habitat_v2_forecast_evaluation_v1
```

### 6.1 Canonical persistence

Canonical UTF-8 JSONL is the sole corpus authority. Tensor files are
hash-recorded deterministic projections, never a second canonical dataset.

D1-owned JSON uses:

```python
json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
```

JSONL is exactly one canonical object followed by `LF` per row. Parsers reject
unknown/missing fields, duplicate keys, booleans in numeric positions,
non-finite numbers and noncanonical bytes. Embedded HMC artifacts retain the
HMC's own canonical bytes and self-hash rules.

Use domain-separated length-framed SHA-256 for public identities and
HMAC-SHA-256 for custodian-keyed universe/split identities. Every record has a
self-hash calculated with its `record_sha256` field omitted. Manifests also
record ordinary SHA-256 over complete artifact bytes.

### 6.2 Identity hierarchy

```text
family_cluster_id
  -> family_id
    -> scenario_member_id
      -> forecast_run_id
        -> witness_id
        -> sample_id
```

- `family_cluster_id`: semantic physical regime; excludes treatment,
  repetition/seed, action, anchor, `W`, `H`, result and split.
- `family_id`: cluster plus one noise repetition commitment.
- `scenario_member_id`: one healthy or treatment scenario.
- `forecast_run_id`: scenario member plus anchor, action and HMC control-run
  identity.
- `witness_id`: forecast run plus completed step.
- `sample_id`: run, timing contracts, ordered history witnesses, proposed action
  and ordered target witnesses.

Splits never enter identities.

### 6.3 Canonical tables

`family_clusters.jsonl` fields:

```text
schema_version, family_cluster_id, suite_role,
generator_contract_sha256, profile_packet_sha256, operating_mode,
load_regime_id, semantic_cluster_ordinal, initial_state_profile_id,
source_prefix_policy_id, topology_rotation_id, semantic_descriptor_sha256,
record_sha256
```

`families.jsonl` fields:

```text
schema_version, family_id, family_cluster_id, repetition_index,
repetition_seed_commitment_sha256, structural_baseline_sha256, record_sha256
```

`scenario_members.jsonl` fields:

```text
schema_version, scenario_member_id, family_id, member_ordinal,
member_profile_id, member_kind, treatment_profile_sha256, scenario_sha256,
plant_run_id, structural_baseline_sha256, record_sha256
```

`split_assignments.jsonl` fields:

```text
schema_version, family_cluster_id, stratum_id, split_rank_sha256,
stratum_rank_index, split, split_assignment_sha256, record_sha256
```

`control_runs.jsonl` fields:

```text
schema_version, forecast_run_id, scenario_member_id, anchor_completed_step,
action_id, proposed_action_sha256, reset_nonce_commitment_sha256,
scenario_sha256, plant_run_id, hmc_control_run_id, authority_epoch,
control_trace_sha256, control_trace_header_sha256,
control_trace_body_sha256, control_trace_footer_sha256,
hmc_implementation_git_sha, terminal_status, final_sequence,
pre_anchor_action_source_sha256, anchor_snapshot_sha256,
anchor_proposal_receipt_sha256, anchor_arbitration_receipt_sha256,
anchor_disposition, anchor_final_command_sha256,
anchor_step_receipt_sha256, record_sha256
```

`control_traces.jsonl` holds the exact parsed canonical HMC control trace once
per forecast run.

`replay_witnesses.jsonl` fields:

```text
schema_version, witness_id, forecast_run_id, control_trace_sha256, sequence,
completed_step, completed_time_s, operational_snapshot, snapshot_sha256,
snapshot_verification_receipt, snapshot_verification_receipt_sha256,
step_receipt, completed_step_receipt_sha256, final_command,
final_command_sha256, plant_receipt, completed_plant_receipt_digest,
physical_state_sha256, physical_target_values_f64,
physical_projection_sha256, step_event_ordinal, snapshot_event_ordinal,
source_binding_sha256, contract_bundle_sha256, record_sha256
```

`samples.jsonl` fields:

```text
schema_version, sample_id, family_cluster_id, family_id, scenario_member_id,
forecast_run_id, split, split_assignment_sha256, anchor_completed_step,
cadence_s, window_steps, horizon_steps, history_witness_ids, action_id,
proposed_action_sha256, target_witness_ids, observable_topology_sha256,
source_binding_sha256, contract_bundle_sha256, record_sha256
```

`manifest.json` binds the ordered artifact list, counts, units, generation
command/environment, source binding and contract bundle.

### 6.4 Required source and contract bundle

The source binding includes:

```text
source_commit_sha, source_tree_sha, clean_tree_required,
final_hmc_commit_sha, final_hmc_tree_sha,
hmc_source_file_manifest_sha256, generator_source_file_manifest_sha256,
uv_lock_sha256, pyproject_sha256, Python implementation/minor,
platform identity
```

The contract bundle includes at least:

```text
scenario and trace schema versions,
equation and actuator-feedback revisions,
observability and topology hashes,
HMC, snapshot and verification hashes,
external-command, preflight, health and safety hashes,
proposal/arbitration/step/terminal/control-trace schema hashes,
forecast input/target/corpus/split/evaluation/timing hashes,
normal action-catalogue and action-source hashes,
profile-packet hash
```

The HMC trace header's claimed implementation SHA is not trusted by itself.
Generation must compare HMC-owned Git blobs with an externally supplied final
HMC source manifest and require the header value to match that frozen component
identity. D1's later commit is not expected to equal the HMC component SHA.

### 6.5 Source-trace-to-sample lineage

The current HMC trace commits snapshot hashes and verification receipts but
does not persist every snapshot body or intermediate physical target. D1 must
therefore replay each complete run and persist normalized replay witnesses.

A sample is valid only if:

1. source and contract bindings match expected immutable identities;
2. the closed V5 scenario reparses and its canonical SHA matches;
3. the complete HMC control trace parses and independently replays;
4. scenario, plant run, control run, epoch and topology agree everywhere;
5. each snapshot body recomputes its snapshot hash and verification receipt;
6. each step/final command/plant receipt cross-links and independently replays;
7. each evaluator-only physical projection reproduces exactly;
8. the source-prefix policy is followed before the anchor;
9. exactly one catalogue proposal occurs at the anchor and later cycles use
   `NO_PROPOSAL`;
10. the run completes all declared 72 transitions; and
11. every history and target witness exists exactly once.

Terminal failure, incomplete truth or lineage drift invalidates the corpus.
Rows may not be silently dropped.

### 6.6 Deterministic generation

Generate into a new empty staging directory, validate completely, then publish
atomically to a new destination. Never overwrite or resume a canonical corpus.

Generate train/validation twice with the same retained keys in separate fresh
directories and compare all rows, counts, byte lengths, artifact hashes and
manifest hashes. Synthetic-final fixtures receive the same duplicate test. A
real withheld final corpus is generated only by its custodian.

## 7. Family roster, splits and leakage controls

### 7.1 Recommended bounded roster

The bounded main roster is:

```text
4 constant modes
x 3 load regimes
x 20 semantic clusters per mode/load stratum
= 240 clusters

x 2 noise repetitions
x 13 members (healthy + 12 treatment profiles)
x 3 anchors (completed steps 16, 40, 64)
x 4 normal complete proposed actions
= 74,880 forecast runs/samples
```

Each scenario has 72 transitions. Fault onset is completed step 25. A transient
treatment is active on `[25,49)` and a persistent treatment on `[25,73)`.
With maximum `W=16`, `H=8`, anchors 16, 40 and 64 yield pre-treatment,
active-treatment and recovery/late-active intervals without crossing a phase
boundary.

The exact initial profiles, loads, source-prefix policies, treatment values,
topology rotations and four actions must be persisted in separate closed
profile/action packets. They are recommended team-owned scenario choices, not
facts already frozen by the inspected HMC.

### 7.2 Split algorithm

Before any simulation or outcome inspection, the custodian computes a keyed
rank for each `family_cluster_id` inside every mode/load stratum. Sort by
`(rank, family_cluster_id)` and assign:

```text
rank 0..13   TRAIN
rank 14..16  VALIDATION
rank 17..19  FINAL
```

Totals:

| Unit | Train | Validation | Final | Total |
|---|---:|---:|---:|---:|
| clusters | 168 | 36 | 36 | 240 |
| families | 336 | 72 | 72 | 480 |
| members | 4,368 | 936 | 936 | 6,240 |
| samples | 52,416 | 11,232 | 11,232 | 74,880 |

Every repetition, healthy/treatment member, action, anchor and window inherits
its cluster's split. Coverage failure invalidates the suite; it never causes
outcome-based reassignment or a new split salt.

### 7.3 Pair isolation and leakage rejection

Healthy/treatment members in a family must have the same canonical scenario
after replacing only `fault_profiles` with the healthy value. Reject other
differences.

Reject cross-split intersections or semantic aliases across:

```text
family cluster, family, canonical scenario, structural physical profile,
scenario member, plant run, HMC control run, forecast run, witness and sample
```

Split before extracting any history/target windows. Fit normalisation and
fitted baselines on training clusters only.

Required adversarial leakage checks include:

- forbidden metadata permutations cannot change input tensors;
- changing one allowed operational value affects only its declared slot;
- future-witness permutation cannot affect input but must break lineage;
- future commands, dispositions, reasons and target truth are rejected at the
  input boundary;
- renamed/reformatted semantic scenario replicas cannot cross splits;
- validation/final records cannot enter preprocessing or training folds; and
- released development artifacts contain no final IDs, keys, traces or
  targets.

Coverage is reported by independent cluster, never by overlapping-row count.
Require the exact roster inventory, all target locations/treatments/actions,
complete action-to-target relationships and adequate supported metric cells.

### 7.4 Final-set boundary

With a public bounded roster, broad held-out scenario strata may be inferred by
elimination. The defensible v1 claim is therefore:

```text
distribution-transparent, realisation-blind
```

The custodian withholds exact final cluster identities, variants, seeds,
scenarios, traces, witnesses, targets, keys and corpus bytes. If the project
requires hidden semantic strata rather than hidden realisations, it must expand
or privately select the roster before implementation.

## 8. Deterministic baselines

### 8.1 Persistence

For each target slot, obtain the most recent causal operational estimate:

- environmental fields: mean primary and secondary when both are available;
  otherwise the available head;
- airflow and resource fields: operational feedback.

Repeat the estimate across all future offsets. Abstain for the complete sample
if any required target slot has no causal estimate.

### 8.2 Linear extrapolation

For each target slot, fit ordinary least squares against completed timestamps
using available causal history values. Require at least three observations.
With one or two, use persistence. With none, abstain for the complete sample.
Extrapolate independently to each future offset.

### 8.3 Fitted statistical baseline

Use direct multi-output ridge regression as the one fitted baseline:

- input: flattened contracted history and complete proposed action;
- output: flattened `H x 51` trajectory;
- float64 preprocessing, fit and inference reference;
- intercept included;
- training-only mean and scale;
- deterministic five-fold whole-cluster cross-validation;
- alpha grid `{1e-6, 1e-4, 1e-2, 1, 100}`;
- choose the lowest cluster-macro normalized error; smaller alpha breaks an
  exact tie; and
- use a deterministic NumPy solver under pinned NumPy/BLAS/thread settings.

Freeze an action-blinded ridge diagnostic which removes the 27 proposed-action
features from both fit and evaluation. If action-aware ridge cannot reliably
beat it, the corpus has not demonstrated usable action information.

Ridge is a baseline only. This contract makes no learned-candidate architecture
choice.

### 8.4 Unreachable physics oracle

Use exact simulator replay and evaluator truth to reproduce the target
trajectory. Label it **UNREACHABLE_PHYSICS_ORACLE**. It validates target
extraction and reports a numerical floor/headroom diagnostic. It cannot be a
production comparator and cannot by itself prove learnability.

## 9. Neutral evaluator

### 9.1 Regression metrics

Report native MAE and RMSE for every target field/zone and future offset.

Fit one scale per target slot using training targets only:

```text
scale = P95 - P5
```

A zero or non-finite scale marks that mandatory slot unsupported. Normalise
point errors by the frozen scale, then aggregate without allowing dense windows
to add statistical weight:

```text
point -> sample -> equal action/anchor/repetition contributions within cluster
      -> equal target/horizon cells -> equal cluster mean
```

Report all native cells alongside the aggregate.

### 9.2 Crossing metrics

For every envelope and declared headline horizon, report raw TP/FP/TN/FN plus
precision, recall, false-positive rate and false-negative rate. Harmful crossing
is positive. A prediction is positive if its physical trajectory newly enters
the envelope by that horizon.

Undefined denominators are reported as unsupported, never coerced to zero or
one.

### 9.3 Abstention and invalid outputs

Report whole-sample coverage overall and by action, regime and truth polarity,
plus selective MAE/RMSE on answered samples.

For the primary operational crossing metric, `ABSTAIN` means no crossing
prediction: a positive abstention is a false negative and a negative abstention
is a true negative. Also report covered-only crossing diagnostics.

Count malformed, timed-out, non-finite and domain-invalid outputs separately.
They fail the candidate interface and never become abstentions.

### 9.4 Runtime and artifact metrics

Measure preprocessing plus batch-one inference with one process and one thread:

```text
100 warm-up calls
1000 timed calls
p50, p95 and p99 latency
artifact bytes
peak process RSS
```

Pin runtime, Python, NumPy/BLAS, platform and thread environment. This is a
software evaluator contract, not an Arm performance claim.

### 9.5 Fail-closed identity behavior

Before reading predictions, recompute and compare expected source, corpus,
split, input, target, evaluator, action, model and preprocessing identities.
Reject dirty source, unknown fields, wrong topology, wrong shape, substituted
artifact, stale manifest, split overlap or prediction/sample mismatch.

## 10. Timing and baseline qualification before training

These procedures are evidence gates, not model training.

### 10.1 Timing pilot

Use a precommitted, semantically disjoint 60-cluster public pilot. Generate the
maximum history/future witnesses once and slice the same anchor/action examples
for all nine `(W,H)` pairs. Pilot families cannot enter the canonical corpus.

Use nested whole-cluster evaluation so a fitted baseline is never selected and
measured on the same clusters. For each horizon require:

- zero replay/identity failures;
- complete targets;
- adequate independent-cluster support for mandatory regression cells;
- adequate positive and negative cluster support for mandatory crossing cells;
- measurable paired action effect versus no-proposal continuation;
- action-aware ridge reliably better than action-blinded ridge; and
- no corrected critical-cell regression versus persistence/linear baselines.

For each viable horizon, select the shortest `W` for which no longer history
has a one-sided paired whole-cluster 95% improvement bound above zero. Select
the largest viable `H` and its chosen `W`. Exact ties choose shorter history.
If none is viable, record `STOP_NO_DEFENSIBLE_TIMING` and do not generate the
canonical corpus.

Freeze all nine results, support counts, excluded pilot identities, test
parameters and the selection proof in a non-overwritable timing receipt.

### 10.2 Baseline selection and margin derivation

Before learned training:

1. Freeze corpus, split, timing, baseline and evaluator bytes.
2. Fit every preprocessing value and ridge parameter on training clusters only.
3. Evaluate the frozen validation set once and retain every supported and
   unsupported result.
4. Select persistence versus linear using higher coverage, then lower
   cluster-macro normalized MAE; unresolved tie goes to persistence.
5. Ridge may replace that deterministic comparator only with zero invalid
   outputs, non-inferior coverage, reliable aggregate error improvement, no
   corrected critical-cell regression and positive action-information evidence
   over action-blinded ridge.
6. Use 10,000 deterministic whole-cluster bootstrap resamples. Store the seed
   derivation and quantile method in the baseline contract.
7. For every selected-baseline metric `m`, freeze:

   ```text
   radius[m] = max(
       abs(B[m] - bootstrap_q2.5[m]),
       abs(bootstrap_q97.5[m] - B[m])
   )
   ```

8. A future candidate must achieve point improvement at least `radius` and a
   one-sided paired 95% cluster-bootstrap lower bound above zero.
9. Freeze the critical-cell universe before training and use Holm-corrected
   familywise alpha `0.05` for non-regression.
10. Treat deterministic repeat-evaluation drift as technical failure, not as a
    quality tolerance.

This fixes the statistical procedure now while deriving numeric admission
margins only from baseline evidence. The output is one of:

```text
PROCEED_TO_EXPERIMENT_FREEZE
STOP_NO_ACTION_INFORMATION
STOP_NO_DEFENSIBLE_HEADROOM
STOP_UNDERPOWERED
```

A proceed result is evidence, not permission to train.

## 11. Proposed tests and acceptance checks

### Input and target

- exact schema closure, units, order, topology and tensor dimensions;
- all descriptor/status/value combinations and float32 overflow rejection;
- previous-command and duplicate-resource-gauge validation;
- exact mode, health and ordered alarm-manifest encoding;
- complete-action validation and duplicate command rejection;
- target extraction against independent plant replay;
- no target imputation, masking or partial horizon; and
- crossing equality, polarity, pre-existing harm and unsupported envelopes.

### Leakage and lineage

- hidden truth, seed metadata and future-field permutation invariance;
- rejection of fault truth, future schedule and later HMC outcomes;
- trace, snapshot, receipt, command, plant and physical-projection cross-links;
- forged, reordered, omitted, duplicated, truncated and substituted evidence;
- exact history/action/target timing and phase-boundary tests;
- healthy/treatment equality except `fault_profiles`;
- family/seed/treatment/action/window split leakage and semantic aliases;
- training-only preprocessing and deterministic cluster folds; and
- released-development scan for withheld final material.

### Evaluator and evidence

- persistence fallback and missing-input behavior;
- causal linear fit with one/two/three observation boundaries;
- ridge standardisation, alpha selection and action-blinded ablation;
- abstention, invalid output and zero-denominator behavior;
- cluster-macro aggregation and bootstrap golden vectors;
- exact repeated generation and repeated evaluation;
- non-overwrite, wrong-expected-hash and artifact-substitution rejection;
- synthetic final commitment, first-open, second-open and post-open failure; and
- focused tests, full suite, Ruff, compilation, sdist/wheel build, isolated
  install and external import/CLI smoke.

## 12. Rejected alternatives

- **Historical four-class classification:** optimises simulator fault names,
  not action-conditioned future physical outcomes.
- **HMC emergency templates as normal actions:** emergency fallbacks are not a
  representative forecast action set.
- **Partial commands or direct learned control:** violate the complete-command
  and HMC-authority boundary.
- **Omitting mode, health or alarms:** HMC arbitration consumes this operational
  state, creating causal target aliasing if the model cannot see it.
- **Forecasting across operating-mode transitions:** imminent mode is not a
  causal model input in the inspected snapshot contract.
- **Noisy measurements as targets:** rewards prediction of future random sensor
  error rather than physical response.
- **Seed, run or window splits:** allow near-identical physical regimes to leak
  across boundaries.
- **Two canonical corpus formats:** JSONL plus an independently authoritative
  tensor corpus creates avoidable identity drift.
- **Mean-only baseline comparison:** hides critical target and regime failures.
- **Preselecting an accuracy margin:** violates the requirement to derive
  margins from baseline evidence.
- **Treating the physics oracle as production:** it uses evaluator-only truth.

## 13. Assumptions requiring authorised team ratification

These are recommended project choices, not claims of Ben's personal approval:

- physical completed-state values are the targets;
- physical warning-entry is harmful-positive;
- cadence remains 60 seconds;
- version 1 is constant-mode and fixed to the current eight-zone topology;
- the bounded scenario/profile roster and four normal actions are suitable;
- accepted, modified and rejected HMC proposal outcomes all remain in scope;
- the final set is distribution-transparent but realisation-blind;
- the independent-cluster support and statistical procedure are acceptable;
  and
- an authorised custodian will retain withheld final identities and targets.

## 14. Unresolved final-HMC and external bindings

Do not guess these values:

```text
${FINAL_HMC_COMMIT_SHA}
${FINAL_HMC_TREE_SHA}
${FINAL_HMC_OBSERVABILITY_PARENT_SHA}
${FINAL_HMC_CONTRACT_SHA256}
${FINAL_HMC_SNAPSHOT_SCHEMA_SHA256}
${FINAL_HMC_SNAPSHOT_VERIFICATION_CONTRACT_SHA256}
${FINAL_HMC_OBSERVABLE_TOPOLOGY_SHA256}
${FINAL_HMC_EXTERNAL_COMMAND_CONTRACT_SHA256}
${FINAL_HMC_PREFLIGHT_CONTRACT_SHA256}
${FINAL_HMC_HEALTH_POLICY_SHA256}
${FINAL_HMC_SAFETY_POLICY_SHA256}
${FINAL_HMC_PROPOSAL_RECEIPT_SCHEMA_SHA256}
${FINAL_HMC_ARBITRATION_RECEIPT_SCHEMA_SHA256}
${FINAL_HMC_STEP_RECEIPT_SCHEMA_SHA256}
${FINAL_HMC_TERMINAL_RECEIPT_SCHEMA_SHA256}
${FINAL_HMC_CONTROL_TRACE_SCHEMA_SHA256}
${FINAL_HMC_ALARM_MANIFEST_SHA256}
${FINAL_HMC_PACKAGE_AND_TEST_RECEIPT_SHA256}
```

Also unresolved:

- a separate four-command normal forecast-action catalogue and hash;
- the pre-anchor action-source contract;
- final scenario/profile packet and hash;
- evidence-selected `W` and `H` plus timing receipt;
- custodian identities, split keys and final commitments; and
- baseline results, selected comparator and numeric margins.

## 15. Verdict

The recommended design is decision-complete for review: it defines what the
data/evaluation implementation must do and how evidence-dependent values are
selected.

The official data-foundation implementation is **blocked** until the final HMC
bytes and a separate normal forecast-action catalogue exist. Canonical corpus
generation additionally requires the scenario/profile packet and custody
arrangement. No learned-model architecture selection or training is authorised
by this document.
