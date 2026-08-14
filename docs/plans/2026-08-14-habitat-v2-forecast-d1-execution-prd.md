# Habitat V2 Forecast D1 Execution PRD

**Status:** executable local delivery contract

**Date:** 2026-08-14

**Integration branch:** `ben/habitat-v2-forecast-data-foundation`

**Final deterministic foundation:** `79d6a718e0d44122a763bb72f9c8ed929f39fd23`

**Final foundation tree:** `91cea3b4c2334a4ece140bd1bf7144353f52ec0d`

**Alex design source:** original commit `f21787771a3abd278625aacef2f2bad757b37523`, locally preserved as authored commit `a86b1bceb19761ce1fbe0331352f3e6a4f792f00`

**Intended version impact:** `minor`. Keep package version `0.8.0` while the candidate remains unpublished. Apply the consolidated bump only at the publication gate.

## 1. Outcome

Deliver the first executable Habitat V2 action-conditioned forecasting foundation.

D1 is complete only when the repository contains and exercises:

1. frozen final-HMC source, topology, alarm and action bindings;
2. a fail-closed operational-snapshot and proposed-action projector;
3. an evaluator-only physical target projector;
4. strict canonical corpus records and provenance;
5. whole-family deterministic splits and leakage rejection;
6. replay-bound development corpus generation through the final HMC;
7. persistence, linear extrapolation and direct multi-output ridge baselines;
8. a model-neutral evaluator;
9. a timing and baseline gate that emits an honest proceed or stop receipt; and
10. tests and an end-to-end verification command.

No learned Habitat V2 model is trained in D1.

## 2. Authority and safety invariants

- The learned forecast interface is advisory only.
- Deterministic HMC arbitration remains the sole final command authority.
- Model code never calls the plant step boundary.
- The proposed action is a complete V5 command, not a delta or action class.
- HMC disposition, accepted command and future HMC state are never model inputs.
- Physical target truth is evaluator-only.
- Fault identities, seeds, future schedules, hidden inventories and simulator receipts are never model inputs.
- Malformed evidence fails closed. Nothing is silently repaired, imputed or dropped.
- Reset step zero is not a model history row.
- History and target windows cannot cross operating-mode, treatment-onset or recovery boundaries.
- A terminal or incomplete HMC run cannot produce a corpus sample.
- `main` must remain unchanged.

## 3. Frozen final-HMC bindings

All bindings are recomputed through production parsers or Git object bytes. Do not copy an unverified prose value.

Expected final values:

```text
final_hmc_commit_sha                 79d6a718e0d44122a763bb72f9c8ed929f39fd23
final_hmc_tree_sha                   91cea3b4c2334a4ece140bd1bf7144353f52ec0d
hmc_contract_sha256                  9f4d269ad8d073d6370f5239d8a78f2541db3001097a460447a8feb84fee2414
snapshot_schema_sha256               85c500a6971fe01dff4b9789a0882ab75e7883c37e15a860adb1a87a46f39970
snapshot_verification_contract       c6154bb7bcf4ab4e86e0d1ada7f6be229a1ae3c805d9c948c907744ab3babee1
observable_topology_sha256            b0246a9dc8f847c3236068c8e1eeeddb31809a680e6133eaf038ea197d6e10e6
external_command_contract_sha256      be06976a01085585772527b883bb2a7539ac9f88a3653ad7f9b0e36845d73281
preflight_contract_sha256             45949d05c99e06d980920e2d2d8905315221d96c80e3ea3a2755f6fa1784ede5
health_policy_sha256                  db0352334caf512dd4f85cf2903012cf9756ccb0d7d8aa2146c28848f2087713
safety_policy_sha256                  198cabe62a079eb35bc40aba0c387c08aa009726f7d15f988ae481e5afe82f19
proposal_receipt_schema_sha256        ec3209b3f01abee275aec524c9959a58a3d231ed0cf129b03831793363bf1626
arbitration_receipt_schema_sha256     560ccc0fbb5cdeb8364a36dda5f87282e04bd85d8895a45449759712b335a804
step_receipt_schema_sha256            fe9cbc362b4262df9e310ed5803727f93312c6a30df4ca973714a3211866ca64
terminal_receipt_schema_sha256        265b6d595f4cad28d91bea79e6a28529303e3a9835d71575e0b1087a9f881a40
snapshot_receipt_schema_sha256        db54a6c1d78e082db88cdd8300437fe4f6f68998ea83c63c8f63519d4d84b039
control_trace_schema_sha256           6398cf76c013f1008efe980a0bbc40f9ba668a128191336aa29c04485e32e8b5
```

The binding artifact must also contain an ordered manifest of every existing Habitat V2 source file at the final HMC commit, with Git blob ID and ordinary SHA-256. Future D1 files are not part of that frozen component manifest.

Frozen parent-owned artifact oracles:

```text
contracts/habitat_v2_forecast_hmc_binding_v1.json
  release tier                       DEVELOPMENT_FIXTURE_ONLY
  source file count                  27
  source manifest SHA-256            93ac2189a914be1c62c6ba23e373d8abb5bdbb553c0e4ac0f3d87789a7bb21bd
  binding SHA-256                    f1890ca5813a98bb13bc628263c85c152fca28c0464843dc8304773b43a05bcc

contracts/habitat_v2_forecast_alarm_manifest_v1.json
  release tier                       DEVELOPMENT_FIXTURE_ONLY
  alarm slot count                   287
  alarm-slots SHA-256                075114456c8a7176c82ebbd0688115ba62d1ec3f835935239c16751410ca971b
  alarm-manifest SHA-256             f27db07c4b7d15a09ec625855d3131bb21734d8725ccb9644a78b982921d9aec

contracts/habitat_v2_forecast_action_catalogue_v1.json
  release tier                       DEVELOPMENT_FIXTURE_ONLY
  action count                       4
  catalogue SHA-256                  476df714510cc9435a4b82ebb23c8ebfab7d6953930c3b0481124a2af45521f9

contracts/habitat_v2_forecast_development_profile_v1.json
  release tier                       DEVELOPMENT_FIXTURE_ONLY
  profile-manifest SHA-256           e6748b21735b3fce668ffccc0b820ebf4df5ab61d204bffb540b3b4e612e3fed

contracts/habitat_v2_forecast_development_records_v1.json
  release tier                       DEVELOPMENT_FIXTURE_ONLY
  record-contract SHA-256            04fa1a8bad2220a6d800fd7ddbeb94646b044ef6f1c7005c45a8cae3f26bd3c7

scenarios/habitat_v2_forecast_development.json
  scenario SHA-256                   d321f86acddbdc3fb73df47f03367fc7acab0c8cfb6dbd66096d30bef5c0e3e8
  run ID                             c418923037a8d9df6fac72dd5a2b4ad11d001a07b85c5d883151f6f4ff10576c
```

The four expected command hashes, in catalogue order, are:

```text
d65c89b029d316a62b03b6c903d2a99d4afc3d40ec1beaad3a55ed8519fcc6e6
566b8cac580986279b93d9d693583b0b18cdc061f00bf53a835f34705407df8a
7577f354ddb3e4f85271e2f00ce3627d7eb9cba441d83242acef9a7545af7390
7c7b36dc871f32aad240c2fb05b05d683f739fb50a485f6c93599afe0df82bd4
```

## 4. Forecast contract

### 4.1 Fixed topology

Version 1 supports only the final eight-zone topology in lexicographic zone order.

The fan and branch order come from `derive_observable_topology`.

Contract generation must persist and hash the exact topology mapping.

### 4.2 History projection

One completed snapshot projects to:

```text
operational numeric values       167
previous authoritative command    27
history_numeric_f32 row           194
sample status one-hot         167 x 5
mode one-hot                         4
health one-hot                       4
alarm lifecycle one-hot        287 x 4
```

Status order:

```text
AVAILABLE
MISSING
NON_FINITE
MALFORMED
DEPENDENCY_UNAVAILABLE
```

Mode order:

```text
dormant
occupied
eva_transition
contingency
```

Health order:

```text
NOMINAL
DEGRADED
CRITICAL
UNKNOWN
```

Alarm lifecycle order:

```text
ABSENT
RAISED
ACTIVE
CLEARED
```

The 287 alarm slots must be generated from the final topology and health contract, persisted as exact entries and hashed. Expected families are:

```text
80 environmental threshold slots
80 sensor-disagreement slots
6 resource-gauge slots
34 actuator-tracking slots
87 telemetry-unknown slots
```

Unknown, duplicate, reordered or semantically inconsistent descriptors, resource gauges or alarms are invalid.

The command reference must be `COMPLETED_FINAL_COMMAND` for every admitted history row.

### 4.3 Proposed action

Project one complete canonical V5 command in this order:

```text
fan_speed_fraction
8 damper positions in topology branch order
scrubber_duty
condenser_duty
8 cooling_removed_w values in zone order
8 oxygen_injection_mol_s values in zone order
```

The normal action catalogue contains the four complete, production-validated commands from the four operating-mode rows in `scenarios/habitat_v2_actuator_feedback.json`.

They are forecast candidates, not emergency templates. Catalogue labels and IDs are provenance only.

### 4.4 Target projection

Each future completed physical state contains 51 binary64 truth values:

- per zone in zone order: temperature, pressure, CO2, O2 fraction, relative humidity and branch airflow;
- then battery state of charge, oxygen-store fraction and sorbent remaining fraction.

The candidate/evaluator tensor is finite `float32[H,51]`.

No target may be missing, imputed, padded or shortened.

## 5. Timing state

D1 supports candidate values:

```text
W in {4, 8, 16}
H in {2, 4, 8}
```

Do not label any pair as the final timing choice without the declared public timing pilot and a separately ratified statistical policy artifact.

The development fixture may exercise `W=4`, `H=2`, but its receipt must say `DEVELOPMENT_FIXTURE_ONLY`.

The D1 timing gate emits exactly:

```text
STOP_UNDERPOWERED
```

`SELECTED` and `STOP_NO_DEFENSIBLE_TIMING` are unsupported in D1 and must be rejected. A future timing policy artifact must separately hash every support threshold, aggregation rule, paired statistic, confidence procedure, correction method, stop-precedence rule and comparator margin before those outcomes can be implemented.

## 6. Corpus and provenance

### 6.1 Canonical serialization

The development fixture packet uses canonical UTF-8 JSONL serialization:

```python
json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
```

Every row ends in one LF.

Strict parsers reject:

- duplicate keys;
- unknown or missing fields;
- booleans in numeric positions;
- non-finite values;
- noncanonical bytes;
- bad self-hashes; and
- identity or lineage drift.

### 6.2 Normative development record contract

`contracts/habitat_v2_forecast_development_records_v1.json` is normative. It closes:

- `family_clusters`;
- `families`;
- `scenario_members`;
- `split_assignments`;
- `control_runs`;
- `control_traces`;
- `replay_witnesses`;
- `samples`; and
- `manifest`.

For each record type it fixes the exact path, schema version, allowed fields, identity fields, identity domain and self-hash field. It also fixes nested step-witness, tensor, provenance and artifact structures. Unknown fields, alternative table names or alternative identity derivations are forbidden.

Record identifiers are:

```text
SHA256(UTF8(identity_domain) || 0x00 || canonical_json(identity_body))
```

`identity_body` contains exactly the declared identity fields. Split labels are excluded from stable family, run and sample identities.

Every JSONL row uses `record_sha256` computed over canonical JSON with exactly that field omitted. The manifest uses `manifest_sha256` with exactly that field omitted. No other field may be omitted for self-hashing.

Every persisted row, manifest and CLI result must carry `release_tier = DEVELOPMENT_FIXTURE_ONLY`. Every referenced packet path must live under `development-fixture-only/`. Persisted D1 split labels are `DEVELOPMENT`. Custodian, final-release and final-score fields are rejected.

### 6.3 Split unit

The split unit is `family_cluster_id`.

All repetitions, healthy/treatment siblings, actions, anchors and timing windows inherit the cluster split.

Use HMAC-SHA-256 keyed ranking within declared strata. Splits never enter stable family or sample identities.

Reject intersections or semantic aliases across cluster, family, scenario, plant run, HMC run, forecast run, witness and sample identities.

Fit normalisation and fitted baselines on training clusters only.

### 6.4 HMC run and shadow replay

For each development forecast run:

1. parse one closed V5 scenario and final HMC contract;
2. reset HMC with an exact 32-byte nonce;
3. call `observe()`, capture the issued snapshot plus `SnapshotVerificationReceipt`, and obtain the capability with `verify_snapshot(snapshot, receipt)`;
4. before the anchor, call `propose(None, verification_handle)` and require `NO_PROPOSAL`;
5. at the anchor, construct one closed `aeolus_habitat_v2_control_proposal_v1` mapping from the current public HMC and snapshot identities plus the selected catalogue command;
6. include exactly `schema_version`, `control_run_id`, `authority_epoch`, non-empty `source_id`, non-empty `source_type`, `completed_observation_step`, `observation_snapshot_sha256`, `requested_application_step`, `observable_topology_sha256`, complete `proposed_command`, `confidence` and `proposal_sha256`;
7. set `completed_observation_step` and `requested_application_step` to the issued verification receipt's completed step, set `confidence` to `None`, then compute `proposal_sha256` over the canonical proposal body before appending the self-hash;
8. call `propose(proposal_mapping, verification_handle)` and require `attempt_class == CANONICAL_PROPOSAL` plus `validation_outcome == VALID` before admitting the run;
9. after the anchor, call `propose(None, verification_handle)` and require `NO_PROPOSAL`;
10. call `arbitrate()` and `step()` so HMC issues every final command and step receipt;
11. independently apply each final command to a shadow physics state;
12. require the shadow plant-receipt digest to equal the HMC step receipt;
13. store evaluator-only target truth from the shadow state;
14. call `export_control_trace(final_hmc_commit_sha)` with the exact frozen source SHA;
15. strictly parse and independently replay the trace; and
16. require final state and all bound identities to agree.

Never access `hmc._state` to obtain targets.

### 6.5 Publication safety

Generate into a new empty staging directory. Every persisted relative path must begin `development-fixture-only/`. Validate all artifacts, then atomically rename to a new destination.

Never overwrite or resume any packet. Reserve the term `canonical corpus` exclusively for the later custodian-produced campaign.

Every manifest, receipt and CLI result must state `release_tier = DEVELOPMENT_FIXTURE_ONLY`. Reject final-set and custody fields.

Generate the development fixture packet twice in separate directories and require byte-identical hashes and counts.

## 7. Development fixture

The development packet is bound to:

```text
source scenario
  scenarios/habitat_v2_actuator_feedback.json
  SHA-256 a9ee8eecdb4a952ef95347edcabb7dad614280eb496877cc9cddf8a5c9f77de7

development scenario
  scenarios/habitat_v2_forecast_development.json
  SHA-256 d321f86acddbdc3fb73df47f03367fc7acab0c8cfb6dbd66096d30bef5c0e3e8

allowed-difference manifest
  contracts/habitat_v2_forecast_development_profile_v1.json
  SHA-256 e6748b21735b3fce668ffccc0b820ebf4df5ab61d204bffb540b3b4e612e3fed
```

The scenario has exactly:

- constant `occupied` mode;
- no treatment faults;
- 24 transitions at 60 seconds;
- one anchor at completed step 16;
- the four normal catalogue actions;
- `W=4` and `H=2` for development plumbing only;
- no hidden-final claim; and
- `DEVELOPMENT_FIXTURE_ONLY` in every packet record, path, manifest, receipt and CLI result.

The allowed-difference manifest permits changes only to `name`, `steps`, `fault_profiles` and `timeline`. All other top-level scenario values must byte-canonically match the source scenario. Validate the scenario with `Scenario.from_mapping`, `HabitatManagementComputer.reset` and all four catalogue commands before Track B begins.

This fixture proves contracts, causality, replay, projection, baseline plumbing and deterministic persistence. It does not qualify timing, statistical support, a canonical corpus or model training.

## 8. Baselines

### 8.1 Persistence

Obtain the most recent causal operational estimate for every target slot:

- environmental fields use the mean of primary and secondary when both are available, otherwise the one available head;
- branch airflow uses operational feedback; and
- battery, oxygen-store and sorbent fractions use operational feedback after exact duplicate-resource validation.

Repeat that complete 51-slot estimate across the horizon.

Abstain for the complete sample when any target slot lacks a causal estimate.

### 8.2 Linear extrapolation

Build the same 51-slot causal operational estimate at each completed history timestamp. Fit each target slot against completed timestamps using available causal history.

- three or more points: ordinary least squares;
- one or two points: persistence;
- no points: whole-sample abstention.

### 8.3 Ridge

Implement direct multi-output ridge with NumPy only:

- flattened contracted history plus proposed action;
- flattened `H x 51` output;
- float64 fit and inference reference;
- training-only mean and scale;
- deterministic whole-cluster folds;
- alpha grid `{1e-6,1e-4,1e-2,1,100}`;
- lowest cluster-macro normalised validation error;
- smaller alpha breaks an exact tie; and
- action-blinded diagnostic fit with the proposed-action features removed.

No scikit-learn dependency.

## 9. Model-neutral evaluator

The evaluator validates all expected identities before reading predictions.

Candidate status is exactly:

```text
PREDICTION
ABSTAIN
```

Malformed shape, timeout, exception, non-finite value, wrong sample identity or domain-invalid value is `INVALID_OUTPUT`, not abstention.

Required outputs:

- native MAE and RMSE by target and horizon;
- training-only P95 minus P5 scales;
- cluster-macro normalised MAE;
- harmful-positive envelope TP, FP, TN and FN;
- precision, recall, false-positive rate and false-negative rate with unsupported denominators explicit;
- whole-sample coverage and selective error;
- invalid-output count; and
- action-aware versus action-blinded ridge comparison.

Harmful crossing is positive. Existing harm at the anchor is excluded from new-crossing classification and reported separately.

## 10. Baseline gate

The D1 baseline gate emits exactly `STOP_UNDERPOWERED`.

`PROCEED_TO_EXPERIMENT_FREEZE`, `STOP_NO_ACTION_INFORMATION` and `STOP_NO_DEFENSIBLE_HEADROOM` are rejected as unsupported until a separately hashed, ratified baseline policy artifact supplies every support threshold, aggregation rule, paired statistic, confidence procedure, correction method, stop-precedence rule and comparator margin.

This gate proves honest plumbing only. It is not permission to train.

## 11. File ownership

### Track A: contracts and projection

Own only:

- `src/aeolus/habitat_v2/forecast/contracts.py`
- `src/aeolus/habitat_v2/forecast/projection.py`
- `tests/habitat_v2/test_forecast_contracts.py`
- `tests/habitat_v2/test_forecast_projection.py`

### Track B: corpus and splits

Own only:

- `src/aeolus/habitat_v2/forecast/corpus.py`
- `src/aeolus/habitat_v2/forecast/pipeline.py`
- `tests/habitat_v2/test_forecast_corpus.py`
- `tests/habitat_v2/test_forecast_pipeline.py`

### Track C: baselines and evaluator

Own only:

- `src/aeolus/habitat_v2/forecast/baselines.py`
- `src/aeolus/habitat_v2/forecast/evaluation.py`
- `src/aeolus/habitat_v2/forecast/timing.py`
- `tests/habitat_v2/test_forecast_baselines.py`
- `tests/habitat_v2/test_forecast_evaluation.py`
- `tests/habitat_v2/test_forecast_timing.py`

### Parent integration owner

Own only:

- package exports;
- frozen JSON contract artifacts;
- development scenario/profile packet;
- end-to-end verification script;
- evidence receipts;
- changelog/version declaration;
- worker-commit integration; and
- final corrections.

Workers must not edit another track's files.

## 12. TDD and acceptance

Each track begins with failing tests, records the RED failure, implements the smallest solution, then records GREEN.

Minimum decisive checks:

```text
pytest focused forecast tests
pytest full suite
ruff check src tests scripts
python -m compileall src scripts
build sdist and wheel
install wheel into a clean environment
import and run external D1 smoke verification
generate development fixture twice and compare artifact hashes
strictly parse and replay every generated control trace
scan released development artifacts for withheld-final material and secrets
git diff --check
```

`main` SHA is checked before and after.

## 13. Stop conditions

Stop and report rather than weakening the contract if:

- final HMC source or contract identities do not match;
- the 287-slot alarm manifest cannot be reproduced;
- any normal catalogue action fails production command validation;
- shadow replay differs from HMC receipts;
- complete trace parse or replay fails;
- family leakage is detected;
- duplicate generation differs;
- baseline evaluation is underpowered; or
- satisfying a test would require exposing hidden truth to model input.

## 14. Explicit non-goals

D1 does not:

- train a neural network or any learned Habitat V2 candidate;
- claim a trained model exists;
- select final `W` or `H` from the compact fixture;
- generate the proposed 74,880-sample canonical campaign;
- claim withheld-final custody;
- claim Arm64 execution or performance;
- modify HMC authority semantics;
- merge, tag, release, deploy or submit; or
- modify `main`.
