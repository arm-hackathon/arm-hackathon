# V6 Conditional Specialists and Room-Family Development Plan

> **For Hermes:** Execute one task at a time with RED-GREEN-REFACTOR. Use a fresh implementation branch stacked on `ben/v5-nominal-counterfactuals`; do not alter, regenerate into, or reselect from V3/V4/V5 historical evidence.

**Goal:** Determine whether an observable, operating-state-aware two-specialist detector can meet the existing false-alert gate across previously unseen *room-physics families*, while preserving paired-reference evidence and leaving all response-layer integration disabled.

**Architecture:** V6 is a new development-only protocol. It first performs a historical forensic audit of the frozen V5 negative result. It then trains/evaluates a sensor-health specialist and a physical-airflow specialist, each using explicitly versioned observable residual features, behind a conservative policy that may abstain. The first V6 corpus varies room volume, per-loop capacity, shared-capacity headroom, occupancy regimes, and measurement conditions while retaining the current three-room hub graph so every candidate shares one strict input contract. A variable-number/variable-graph room model is a separate research spike after V6; it must not be smuggled into the V6 safety comparison.

**Tech stack:** Python 3.11, NumPy, ONNX opset 17 where a learned candidate is exported, pytest, Ruff, uv locked dependencies.

---

## 1. Verified starting point and decision record

### What V5 established

`docs/evidence/v5-development-outcome.md` records a reproducible development-only failure:

| Method | Macro F1 | False-alert episodes / 1,000 healthy ticks | Healthy streams alerted |
| --- | ---: | ---: | ---: |
| Rules baseline | 0.6045 | 63.81 | 8/8 |
| Best learned candidate: balanced gated CNN | 0.6632 | 121.25 | 8/8 |

The ceiling remains **10 false-alert episodes per 1,000 healthy ticks**. No learned model was selected. Rules are only the least-bad development baseline, not a safe operational method. V5 also established that changing the *timing* of declared nominal load while preserving per-zone total load does not fix the false-alert failure. Lowering alert burden through the existing threshold/persistence grid caused unacceptable fault-recall loss.

### Current code constraints that change the plan

1. `src/aeolus/config.py` supports a directed **hub**: one `air_processing` zone and exactly one outbound/return pair for each non-processing zone. It is not a general airflow/duct graph solver.
2. `src/aeolus/model_input.py` defines `model_input_v1` as a topology-bound `float32[24]` selector. The present selector requires exactly three non-processing zones.
3. `src/aeolus/families.py` requires every family in a manifest to use one identical model-input contract. This is correct: rows from different reordered vectors cannot be silently combined.
4. `src/aeolus/trace.py` already emits relevant observable telemetry: sensor CO2, occupancy multiplier, actuator state, request/delivery/residual airflow, and system capacity state. It does not expose fault effectiveness or connection health through `model_feature_row`.
5. `src/aeolus/error_analysis.py` groups historical classification errors by class/profile/family but does not yet preserve a stateful healthy-alert episode ledger or the operating-state values needed to test the false-positive hypothesis.

### Decision

V6 will **not**:

- add another generic four-way CNN/MLP to the existing 24-value input and call that a redesign;
- relax the `<= 10` false-alert ceiling, the baseline-comparison bound, or fault-recall constraints after seeing V5;
- use V3/V4/V5 validation results as a V6 selection or calibration input;
- create a V6 final suite, select a response action, or connect a detector to a response layer;
- claim that synthetic room diversity proves real-habitat performance.

V6 will use historical V5 evidence only to specify falsifiable hypotheses. It will make all fit/calibration/validation choices against fresh V6 development data, then require a separately predeclared final suite if and only if development gates pass.

---

## 2. The hypothesis V6 must test

### Primary hypothesis: sensor-health evidence

**Hypothesis H1:** Most healthy `frozen_sensor` alerts occur because current methods equate low observed sensor movement with a sensor fault even when the observable operating state does not imply meaningful sensor movement.

A frozen-sensor concern should require all of the following:

```text
observed sensor movement is persistently low
AND observable controller / airflow / demand context predicts meaningful movement
AND at least one independent observable corroborates the mismatch
AND the mismatch persists for a predeclared interval
```

A low-variance but settled room should therefore remain nominal. A deliberately frozen sensor during changing conditions should become a concern.

### Secondary hypothesis: physical-airflow evidence

**Hypothesis H2:** Current physical-fault false alerts arise when raw delivery residual is interpreted outside its operating context—for example during legitimate shared-capacity contention, actuator transients, or a settled low-demand state.

A physical-flow concern should require a persistent, isolated request-versus-delivery residual conditioned on controller state, demand/headroom context, and transient-versus-settled state. Abrupt versus gradual fault naming happens only after a physical concern exists; it is not a forced prediction for every unusual window.

### Falsification criteria

H1 is weakened if healthy frozen-sensor episodes have an expected-change proxy distribution indistinguishable from true frozen-sensor episodes within the same operating profile, or if adding the proxy does not reduce healthy episodes without unacceptable frozen-sensor recall loss.

H2 is weakened if the conditioned airflow residual does not separate healthy contention/transients from blocked/degraded paths, or if an explicit settling/saturation state fails to explain the healthy physical-alert episodes.

A negative V6 result is a valid outcome. It means the selected observables do not support this decision policy safely; it does not justify lowering the gate.

---

## 3. Scope and phased architecture

### V6-A: comparable room-physics development protocol — in scope

Keep the current three-zone hub graph and stable zone/connection identities. Vary the *physics and operating conditions* behind that graph:

- zone air-volume asymmetry;
- per-loop maximum-airflow asymmetry;
- shared-airflow capacity/headroom;
- low, high, staggered, and transition-heavy occupancy/load schedules;
- controller settling and actuator-saturation conditions;
- bounded, physically declared telemetry noise/bias/drift regimes already supported by the simulator.

This is genuinely broader room diversity while keeping one strict input contract. It tests whether a policy generalises across different room dynamics rather than memorising one standard habitat.

### V6-B: variable-topology research spike — explicitly out of the V6 comparison

Adding more rooms, series ducts, branching paths, or arbitrary room graphs requires all of the following:

- a new validated scenario/plant topology, not a presentation-only graph;
- physically defined routing and capacity allocation rules;
- variable-size trace/input representation with explicit topology semantics;
- a topology-aware baseline and model interface;
- family manifests that bind each row to a compatible topology/input contract;
- held-out graph-family evaluation.

This would confound plant-engine changes with detector-policy changes if bundled into V6-A. Implement it as a separately reviewed spike after V6-A’s error audit has established that the smaller experiment is worth running. Its success criterion is simulator/contract correctness, **not** detector selection.

---

## 4. Protocol freeze before implementation

### Task 1: Record V6-A boundaries and retire V5 from selection

**Objective:** Make the historical/development boundary executable before any new feature code exists.

**Files:**
- Create: `docs/decisions/v6-development-boundary.md`
- Create: `tests/test_model_cycle_v6.py`
- Modify: `src/aeolus/model_cycle_v6.py` (created in Task 8)

**Step 1 — RED:** Write tests requiring a V6 runner to reject V3/V4/V5 seed clusters, V5 manifests/reports, non-V6 sweep schemas, non-empty output directories, and any final/response authorization request.

**Step 2 — GREEN:** Add V6 constants only after the V6 design document fixes fresh seed ranges and prohibited historical seed sets. The runner must expose only `train`, `calibration`, and `validation` roles.

**Step 3 — Verify:** Run the focused tests, then `PYTHONPATH=src uv run --locked --python 3.11 --extra dev pytest tests/test_model_cycle_v6.py -q`.

**Acceptance:** A V5 family manifest cannot be passed through the V6 code path, even under a renamed filename.

**Commit:** `test: freeze v6 historical evidence boundary`

### Task 2: Predeclare V6-A room-physics families and split allocation

**Objective:** Specify a small, interpretable room-family matrix rather than a random-parameter soup.

**Files:**
- Create: `scenarios/v6/room-balanced.json`
- Create: `scenarios/v6/room-volume-asymmetric.json`
- Create: `scenarios/v6/room-capacity-constrained.json`
- Create: `scenarios/v6/room-transition-heavy.json`
- Create: `scenarios/sweep-v6-development.json`
- Create: `tests/test_sweep_v6.py`
- Modify: `src/aeolus/sweep.py`

**Predeclared family intent:**

| Family | Purpose | Split role |
| --- | --- | --- |
| `room-balanced` | Reference three-zone hub; balanced volumes/capacities | fit only |
| `room-volume-asymmetric` | Same graph, materially different zone volumes and loop maxima | fit only |
| `room-capacity-constrained` | Legitimate shared-capacity contention and saturation-prone operating profiles | calibration only |
| `room-transition-heavy` | Staggered, high-transition nominal schedules with different room response times | validation only |

The exact numeric values, seed clusters, fault onset grid, severities, family counts, and profile identifiers must be written in the versioned sweep before any canonical run. A room-family configuration, its seed/run cluster, and its paired fault/reference documents may appear in **one role only**. Validation must contain at least one complete room-physics family not seen in fit or calibration.

**Step 1 — RED:** Test that `aeolus_sweep_v6` requires a non-empty ordered `room_families` mapping, each family names a base scenario, all base scenarios have no fault profiles, and all resolve to the same declared V6 observable-input contract.

**Step 2 — RED:** Test rejection for a family repeated across split roles, reused seed/run cluster, missing reference/fault pairing, changed room configuration on only the fault side, or a scenario content collision with V3/V4/V5 manifests.

**Step 3 — GREEN:** Add a V6-only sweep parser/generator. It must create a reference from the selected room-family base scenario, apply the same named operating profile to that reference, deep-copy it for every fault, and change only `fault_profiles` in the paired fault document.

**Step 4 — Verify:** Generate a fixture sweep. Assert exact reference/fault equality after removing `fault_profiles`; assert room family and split are embedded in every generated family identity; validate every scenario with `load_scenario()`.

**Acceptance:** More complex conditions are declared source data with a stable family ID. They are not injected only into healthy scenarios and are never unnamed random noise.

**Commit:** `feat: add v6 room-family sweep contract`

---

## 5. Historical V5 episode forensics

### Task 3: Build an immutable healthy-alert episode ledger

**Objective:** Determine why methods alert before changing the model/corpus again.

**Files:**
- Create: `src/aeolus/alert_forensics.py`
- Create: `tests/test_alert_forensics.py`
- Modify: `src/aeolus/error_analysis.py` only if shared strict JSON/hash helpers avoid duplication
- Create: `docs/evidence/v5-alert-forensics.md` after the frozen historical replay

**Ledger requirements:**

For every **nominal/reference** V5 stream and each operational method (rules plus every learned candidate), replay every causal window in order and record each nominal-to-non-nominal episode. Every ledger record must bind:

```text
historical-forensic-only evidence role
source commit and source-manifest SHA-256
family-manifest SHA-256
model artifact SHA-256 or exact rule parameter record
family_id, room/profile identity, scenario role, stream identity
predicted label, episode start/end, duration, and window endpoints
sensor slopes/ranges by zone
actuator setpoint, position, tracking residual, and moving state
request, delivery, normalized residual, and cross-loop isolation context
system capacity scale / total request-delivery context where observable
```

Do not include hidden `connection.health`, fault effectiveness, scenario labels, random-source state, or any other simulator truth in a detector feature. The ledger may include the true nominal role only because it is a historical evaluation label, never an inference input.

**Step 1 — RED:** Write a fixture with consecutive false labels and prove the ledger emits one episode, not one row per overlapping window. Add cases for stream reset, excluded transition windows, exact source/model digest checks, and rejection of a non-reference stream in a healthy-alert report.

**Step 2 — GREEN:** Implement a streaming replay function that receives a validated manifest, immutable candidate/rule artifact, and rows in family/role/time order. It must derive episodes from the full unfiltered causal stream, not from a score-filtered error subset.

**Step 3 — RED:** Write a test that mutates one observable context value and proves the canonical ledger hash changes. Write a separate test proving a forged model or rule digest is rejected.

**Step 4 — GREEN:** Implement strict report validation and canonical serialization.

**Step 5 — Historical execution:** Use a separate clean worktree at V5 source commit `9e81011c93961a645a75d4cd7d61b2ef4ab6c9c2`. If the canonical V5 output is unavailable, reproduce it into a new ignored directory in that worktree; never write into the historical canonical output path. Verify regenerated source/sweep/family/report hashes before forensic generation. Mark the forensic report `historical_forensic_only` and prohibit it from every V6 selection API.

**Step 6 — Analysis:** Produce tables by method, predicted class, operating profile, healthy stream, and episode duration. For H1, compare the observed-sensor slope/range and the proposed expected-change proxy within the same profile. For H2, compare residual ratio/isolation and actuator/capacity state within the same profile. State whether each hypothesis is supported, contradicted, or unresolved.

**Acceptance:** We can point to the state at each healthy alert rather than inferring root cause from macro F1.

**Commit:** `feat: add stateful historical alert forensics`

---

## 6. Observable-context and residual contract

### Task 4: Decide the real observability boundary before adding a feature

**Objective:** Prevent a simulator-only convenience value from becoming an undeclared production sensor.

**Files:**
- Create: `docs/contracts/v6-observable-context.md`
- Create: `tests/test_model_input_v2.py`
- Modify: `src/aeolus/model_input.py`
- Modify: `src/aeolus/trace.py` only if an already-observable field needs a stricter projection

**Decision table to approve in the contract:**

| Candidate context | Current trace availability | May become V6 input? | Rule |
| --- | --- | --- | --- |
| Sensor CO2, actuator position/setpoint/tracking, request/delivery/residual | Existing model-visible fields | Yes | Retain strict projection and contract hashes |
| Actuator moving flag, system capacity scale, aggregate demand/delivery | Existing observable trace fields, not model_input_v1 | Yes, if declared in a new V6 projection | Must be emitted/validated identically in reference and fault traces |
| Occupancy multiplier | Trace field, but may not exist in a real deployment | Diagnostic-only until Ben/team explicitly confirms an operational workload/occupancy signal exists | Otherwise exclude it from deployed-candidate input |
| CO2 mass, source mass, connection health/effectiveness, fault start/label, simulator seed/noise state | Simulator truth or evaluation metadata | Never | Reject if present in a model/residual input |

**Design:** Add a new, immutable `model_input_v2` or named `v6_observable_context_v1` projection rather than changing v1. Preserve `model_input_v1` byte-for-byte for historical artifacts. The V6 projection remains fixed-width for the same three-zone hub and includes only approved observable fields. Its metadata must bind selector JSON, topology JSON/hash, projection version, dtype/shape, and any context-availability declaration.

**Step 1 — RED:** Test that v1 outputs and metadata do not change. Test the new projection’s stable order, exact finite dtype/shape, topology binding, rejection of unknown/missing fields, and rejection of injected hidden truth.

**Step 2 — GREEN:** Implement the new projection and strict metadata validator. Use topology-derived field selection, not positional magic numbers.

**Step 3 — RED:** Add a test where two valid room-physics base scenarios have different volumes/capacities but identical hub topology; they must share the V6 input contract. Add a test where zone/connection identities or graph shape differ; they must fail compatibility.

**Step 4 — GREEN:** Update V6 manifest/corpus validation to require the new contract and retain the old V1 validator untouched.

**Acceptance:** Every new signal has a named operational meaning and provenance. No inferred “expected change” is allowed to read simulator truth.

**Commit:** `feat: add versioned v6 observable context contract`

### Task 5: Implement transparent conditional residual features

**Objective:** Build the evidence the two specialists actually need, not another unstructured telemetry vector.

**Files:**
- Create: `src/aeolus/residual_features.py`
- Create: `tests/test_residual_features.py`
- Modify: `src/aeolus/corpus.py` or create `src/aeolus/corpus_v3.py`

**Required feature groups:**

1. **Sensor-health features, per zone**
   - observed sensor slope, range, and maximum delta over the causal window;
   - actuator setpoint/actual-position movement and tracking residual;
   - request/delivery change and residual change on the zone’s outbound leg;
   - independently observable sibling-zone movement summaries;
   - an explicit, documented expected-change proxy derived only from the approved context.

2. **Physical-airflow features, per outbound loop**
   - request, delivery, and normalized request-delivery residual;
   - residual slope/jump/persistence;
   - isolation versus the other outbound loops;
   - controller tracking/movement state;
   - shared capacity/headroom and transient-versus-settled proxy.

3. **No hidden labels**
   - never encode fault type, start time, effectiveness, health, config source multiplier, seed, or true class;
   - calculate every field from the same causal window presented to the candidate.

**Step 1 — RED:** For a hand-built ten-tick observable fixture, assert each residual exactly. Include zero-request division, all-zero movement, actuator transition, shared contention, and altered hidden truth with unchanged observables.

**Step 2 — GREEN:** Implement pure deterministic feature functions with finite-value checks and explicit denominator behaviour.

**Step 3 — RED:** Construct paired reference/fault traces where a frozen sensor occurs during meaningful expected movement; assert observed movement falls while expected-change proxy remains high. Construct a legitimate settled reference trace; assert low observed movement alone does not imply high expected change.

**Step 4 — GREEN:** Add the minimal feature logic needed to satisfy the tests.

**Acceptance:** Every feature has a physical reading and a test demonstrating the confound it is supposed to resolve.

**Commit:** `feat: add conditional residual feature projection`

---

## 7. Specialist detectors and conservative decision policy

### Task 6: Replace forced four-way prediction with two evidence specialists

**Objective:** Separate measurement-integrity evidence from physical-process evidence.

**Files:**
- Create: `src/aeolus/specialists.py`
- Create: `tests/test_specialists.py`
- Modify: `src/aeolus/baseline.py` only to add a separate V6 conditional-rule baseline; do not change the historical V1 `RuleBaseline` semantics

**Interfaces:**

```python
class SensorHealthSpecialist:
    def assess_window(self, window: list[dict]) -> SensorHealthAssessment: ...

class PhysicalFlowSpecialist:
    def assess_window(self, window: list[dict]) -> PhysicalFlowAssessment: ...

class V6DecisionPolicy:
    def label_window(self, window: list[dict]) -> str: ...
    def reset(self) -> None: ...
```

Use structured assessments that preserve score, threshold result, corroborating zone/loop identity, and reason code. The public policy label set is:

```text
nominal
uncertain                # logged; not an operational alert
sensor_health_concern    # operational concern; may later become frozen_sensor
physical_flow_concern    # operational concern; may later become blocked/degradation
frozen_sensor
blocked_path
gradual_primary_fan_degradation
```

The policy must prefer `uncertain` over a named fault when evidence is incomplete. It must never have a code path that upgrades `uncertain` to a named class based on hidden ground truth.

**Candidate set:**

1. V6 conditional-rules baseline: explicit residual, expected-change, corroboration, and persistence conditions.
2. Learned sensor-health specialist: a small, topology-bound binary scorer trained only on sensor-health labels/features.
3. Learned physical-flow specialist: a small, topology-bound scorer trained only on physical-flow labels/features, followed by abrupt-versus-gradual classification only when a physical concern is established.

Keep the first learned implementation deliberately small and auditable. A new generic CNN is out of scope unless V6 forensic evidence later falsifies the specialist representation and a separate plan justifies it.

**Step 1 — RED:** Test that a flat sensor with low expected change yields `nominal` or `uncertain`, while a flat sensor with meaningful expected change and corroboration yields `sensor_health_concern`. Test that a shared-capacity transient does not become a physical alert merely because residual is high.

**Step 2 — GREEN:** Implement the conditional rules. Keep threshold/persistence parameters immutable data records and enumerate a predeclared calibration grid.

**Step 3 — RED:** Test specialist reset boundaries, finite score/threshold validation, no cross-stream state leakage, and a policy case in which conflicting specialist evidence becomes `uncertain`.

**Step 4 — GREEN:** Implement the specialist/policy classes and strict serialization/export contracts for any learned scorer.

**Step 5 — RED/GREEN:** Add ONNX parity, operator allowlist, contract hash, and model/input-shape tests for each learned candidate before training it.

**Acceptance:** Sensor and physical concerns can be independently inspected. An uncertain window cannot improve fault recall or macro F1 by being silently remapped to the true class.

**Commit:** `feat: add v6 conditional specialist policy`

### Task 7: Define evaluation semantics that make abstention honest

**Objective:** Allow uncertainty without creating an escape hatch from operational safety.

**Files:**
- Create: `src/aeolus/evaluate_v6.py`
- Create: `tests/test_evaluate_v6.py`
- Modify: `src/aeolus/evaluate.py` only for shared, backwards-compatible utilities
- Create: `docs/contracts/v6-evaluation-semantics.md`

**Rules:**

- Run every causal window in time order and reset state at each `(family_id, scenario_role)` stream boundary.
- Count an operational false-alert episode on a nominal reference stream when the policy transitions from `nominal`/`uncertain` into an operational concern or named fault. Deduplicate contiguous alert windows into one episode.
- `uncertain` is not an operator alert, but it is reported as a separate nominal-uncertainty burden.
- On a fault stream, `uncertain` is **not** a correct class prediction and cannot establish detection latency. It therefore lowers fault recall / contributes to missed-detection burden rather than creating a free pass.
- Report specialist binary recall/precision, final named-fault macro F1, class-specific recall, healthy episode burden, healthy stream burden, uncertainty burden, and observable-onset latency.
- Exclude onset-spanning supervised windows exactly as the existing protocol does, but replay them for stateful policy/latency continuity.

**Acceptance gates:**

Reuse the existing V5 operational false-alert ceiling, baseline delta, fault-recall limits, and artifact/parity requirements unchanged. Add only stricter V6 requirements: no unresolved/uncertain post-onset fault episode may be treated as detected; the report must name the retained baseline when no method passes; all final/response authorizations remain false on a development failure.

**Step 1 — RED:** Build a fixture where a detector outputs `uncertain` forever after fault onset; assert zero named detection, failed recall/episode criteria, and no false claim of latency.

**Step 2 — RED:** Build a nominal fixture with many overlapping alert windows; assert it counts as one episode. Add a second separated episode and assert the denominator is healthy ticks, not windows.

**Step 3 — GREEN:** Implement strict evaluator/report schema and the explicit pass predicate.

**Acceptance:** Abstention reduces unsafe forced naming but cannot hide missed faults or turn correlated windows into inflated evidence.

**Commit:** `feat: add v6 stateful specialist evaluation`

---

## 8. Manifest, corpus, and V6 development runner

### Task 8: Extend evidence identity for room families and V6 inputs

**Objective:** Bind every training/evaluation row to its room family and V6 observable contract.

**Files:**
- Modify: `src/aeolus/families.py`
- Modify: `src/aeolus/corpus.py`
- Create: `tests/test_families_v6.py`
- Create: `tests/test_corpus_v6.py`

**Design:** Introduce versioned V6 manifest/corpus schemas rather than adding optional fields to historical V1/V2 records. Every V6 family/row must include a `room_family_id`; the canonical manifest must bind the family’s base-scenario identity/configuration digest as well as reference/fault digests. The corpus must bind the V6 projection metadata and the generated-input manifest digest.

**Step 1 — RED:** Test rejection for missing/unknown room family, room family appearing in multiple split roles, changed base configuration without changed digest, mixed input contracts, scenario-name-only identity reuse, or a generated corpus whose row room family disagrees with its family manifest.

**Step 2 — GREEN:** Implement versioned parsers/serializers and canonical hashing.

**Step 3 — Verify:** Deliberately alter one base-scenario physical value while preserving filenames and sweep layout; assert the input/family manifest identity changes and the runner rejects stale receipt bindings.

**Acceptance:** Scenario names, seed lists, and layout hashes cannot masquerade as evidence that the same generated room inputs were evaluated.

**Commit:** `feat: bind v6 room-family corpus identity`

### Task 9: Implement and freeze the V6 development runner

**Objective:** Train on fresh fit rooms, calibrate only on fresh calibration rooms, and evaluate once on an unseen validation room family.

**Files:**
- Create: `src/aeolus/model_cycle_v6.py`
- Create: `tests/test_model_cycle_v6.py`
- Create: `docs/evidence/v6-development-outcome.md` only after canonical execution

**Runner sequence:**

1. Validate exact V6 sweep hash, V6 schema, fresh seeds, room-family allocation, and clean/empty output directory.
2. Generate V6 scenarios and a strict room-family manifest.
3. Generate V6 corpus and validate every row against manifest and observable-onset evidence.
4. Fit only on fit-role room families.
5. Choose rule thresholds and learned-policy thresholds only on calibration-role room families using the complete predeclared feasibility grid.
6. Evaluate the frozen candidates/policy once on validation-role room families.
7. Write all candidates, full grid, eligible count, baseline result, retained method, rejection reasons, source/input manifests, model/artifact hashes, parity receipts, and authorization flags.
8. If no candidate is eligible, set `retained_method` to the V6 conditional baseline, mark every learned candidate ineligible, and set final/response authorization false.

**Step 1 — RED:** Test each role boundary, exact contract match, room-family split isolation, stale sweep hash, empty-output condition, all-candidate failure authority, and prohibition on model selection from validation data.

**Step 2 — GREEN:** Implement the minimum orchestration that passes each test.

**Step 3 — RED/GREEN:** Verify the report includes the complete feasibility grid—not only the preferred threshold—and fails validation if any artifact hash, source path, policy decision, or gate flag disagrees with recomputation.

**Acceptance:** The runner makes a failed gate machine-readable and impossible to narrate away.

**Commit:** `feat: add frozen v6 development protocol`

---

## 9. Canonical evidence execution and review

### Task 10: Run V6 only after source freeze

**Objective:** Produce real development evidence, not a plausible metric table.

**Files:**
- Generated ignored directory: `out/v6-conditional-specialists-<timestamp>/`
- Promoted, reviewed only after validation: `docs/evidence/v6-development-outcome.md`

**Steps:**

1. Ensure all V6 source/tests/specs are committed locally and the worktree is clean. Record exact source commit and source manifest at process entry.
2. Run complete gates from that clean commit:

```sh
PYTHONPATH=src uv run --locked --python 3.11 --extra dev pytest -q
uv run --locked --python 3.11 --extra dev ruff check src tests
python -m compileall -q src

git diff --check
```

3. Run the canonical V6 command into a new empty ignored directory using the declared locked Python/dependency configuration. For a long run, use a tracked background process; do not let a foreground timeout create a half-receipt that gets mistaken for evidence.
4. Parse and strictly validate the raw report, every artifact, policy, ONNX parity receipt, source manifest, sweep digest, room-family manifest, corpus manifest, and feasibility grid.
5. Re-run from the frozen source into a second fresh ignored directory. Compare report/model/input hashes when determinism is expected; otherwise classify any difference precisely and compare semantic metrics/contracts before promotion.
6. Generate the claim ledger: every public metric/table/caveat must point to a receipt path, computed derivation, or frozen-plan constraint.
7. Run a separate security/methodology review: hidden-truth leakage, topology/room-family split leakage, reference-fault pairing, policy authority, episode denominators, `uncertain` semantics, artifact parity, source completeness, and stale documentation.

**Decision rule:**

- **Gate fails:** document the negative result; retain only the least-bad development baseline; no final suite and no response integration.
- **Gate passes:** do not call the detector finished. Freeze the chosen policy/model/parameters; then write a separate plan/spec for a fresh, topology/room-family-disjoint final suite.

**Commit:** `docs: record v6 development outcome` only after all receipts/review are complete.

---

## 10. Variable-topology spike after V6-A

### Task 11: Establish whether a more complex graph is physically and evidentially justified

**Objective:** Decide whether “more rooms” requires a general graph simulator, without contaminating V6-A.

**Files:**
- Create: `docs/spikes/variable-topology-feasibility.md`
- Create: `tests/test_topology_spike.py`
- Optional throwaway implementation under: `spikes/variable_topology/` (not imported by `src/aeolus`)

**Questions to answer with a small throwaway prototype:**

1. Can the plant define mass/airflow allocation for a series/branching graph while preserving current physical invariants?
2. Which directed paths are controllable/observable, and how do shared bottlenecks differ from the hub’s proportional-capacity rule?
3. What is the smallest graph-aware input representation that preserves node/edge identity and is compatible across graph sizes?
4. Can the rules baseline distinguish a local fault from a network-wide resource constraint with only declared observables?
5. What new causal failure modes are introduced by a graph model that do not exist in the three-zone hub?

**Hard stop:** Do not import spike code into production simulator paths, train detectors on it, or claim graph generalisation. Destroy/abandon the spike if the plant invariants or observability contract cannot be made explicit.

**Follow-on plan required before production integration:** scenario schema version, plant solver and invariants, trace projection, graph-aware input contract, topology-held-out family protocol, baseline/model ports, and a fresh development/final sequence.

---

## 11. Work order, verification, and human gates

### Execution order

```text
1. Freeze V6 boundaries
2. Produce V5 historical healthy-alert episode forensics
3. Decide observability contract
4. Freeze room-family sweep and split allocation
5. Implement residual features
6. Implement conditional rules and specialist policy
7. Implement honest V6 evaluation and manifests
8. Implement V6 runner
9. Run canonical V6 development evidence
10. Review result
11. Only then consider a separate final-suite plan or a variable-topology spike
```

### Required review questions before canonical V6 execution

- Does every V6 room family use a source-controlled base configuration and a fresh, role-isolated seed/run cluster?
- Are fault/reference pairs byte-identical except for `fault_profiles` after operating-profile shaping?
- Is every feature sourced from a causal observable field that could exist outside this simulator?
- Does `uncertain` fail to inflate recall or latency while remaining visible as an operational burden?
- Does every healthy alert episode have an unambiguous denominator and stream identity?
- Can a changed room configuration or generated input alter a manifest/receipt without being detected?
- Does a no-winner report identify the baseline as retained and prohibit integration automatically?

### Explicit human decisions needed before implementation proceeds

1. Confirm whether an operational AEOLUS deployment would expose an occupancy/workload signal. If not, retain occupancy multiplier as V5/V6 forensic context only and exclude it from candidate inputs.
2. Approve the final numeric room-family matrix, fresh seed ranges, expected family count, and V6 candidate/grid scope before the sweep is frozen.
3. Approve any later expansion from V6-A’s fixed three-zone hub to variable-topology plant work as a separate project scope.
4. Approve any push, PR creation, merge, final-suite generation, response integration, deployment, or hardware claim separately.

---

## 12. Expected outcome boundaries

**[solid]** V5 rejects the existing generic learned detector approach under its declared timing-counterfactual development protocol. The current rules baseline is also operationally unsafe.

**[likely]** Conditional residual evidence and specialist policies will diagnose the dominant `frozen_sensor`/nominal confound more directly than a forced generic four-class predictor.

**[speculative]** Such a redesign will pass the existing safety gate. V6 exists to test this, not assume it.

The immediate success is a trustworthy answer: either the redesigned evidence/policy lowers healthy-alert burden without sacrificing fault detection across unseen room physics, or it fails visibly and tells us which observability/plant assumption must change next.
