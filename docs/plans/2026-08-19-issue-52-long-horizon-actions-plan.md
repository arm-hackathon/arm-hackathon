# Issue #52 Full Implementation, Training, and Qualification Plan

## 1. Document control

- Issue: https://github.com/arm-hackathon/arm-hackathon/issues/52
- Design date: 2026-08-19
- Audited base: `843a5c1485de841462cbb47e486c2185099b71a2`
- Required approver: Ben (`bbeennyy860-cyber`)
- Short note: `docs/plans/2026-08-19-issue-52-long-horizon-actions-design.md`
- Machine-readable preregistration: `contracts/habitat_v2_forecast_issue_52_preregistration_v1.json`
- Preregistration SHA-256: `E0A24B2FD9309ED551DCD6D4FB98EFF1FDDA6B364DE2DBE73584CCF1ADA7E61F`
- Status: `BEN_APPROVED_CONTRACT_IMPLEMENTATION_AND_BOUNDED_PILOT`
- Approval status is recorded below; the frozen preregistration bytes remain unchanged.
- `DATA_GENERATION_AUTHORISED=false`
- `TRAINING_AUTHORISED=false`
- `EXPERIMENTS_AUTHORISED=false`
- `DEPLOYMENT_AUTHORISED=false`
- `METRIC_AMENDMENT_APPROVED=false`

This appendix is normative. The short note is the plain-English review entry point. Ben approves the Git commit containing both documents and the preregistration, which avoids circular file-digest references. Creating this documentation branch and commit is planning work; it does not open any implementation or experiment gate.

## 2. Verified baseline and scope correction

The audited Habitat V2 code contains no eight-step forecaster, four-action planner, 32-step forecaster, action-conditioned rollout model, or candidate ranker. The existing four-output learned artifact classifies `nominal`, `gradual_primary_fan_degradation`, `blocked_path`, and `frozen_sensor`; those are fault classes, not actions. Its recorded result also did not demonstrate advantage over its rule baseline.

Issue #52 is therefore a new bounded forecast-and-rank lane. It must not be implemented as `8 -> 32` and `4 -> 12` constant changes. The explicit eight-step and four-candidate versions exist only as new experiment ablations.

## 3. Goals and non-goals

Goals:

1. Use 16 causal one-minute observations.
2. Forecast 32 one-minute future states conditioned on a complete candidate schedule.
3. Compare exactly 12 finite, versioned normal-operation schedules.
4. Quantify uncertainty and abstain on invalid, ambiguous, unsupported, OOD, or late inputs.
5. Score forecast trajectories in deterministic code.
6. Submit zero or one proposal for the current HMC application step.
7. Demonstrate practical forecast and ranking gains under frozen statistical rules.
8. Preserve or improve all hard safety, provenance, replay, and systems gates.

Non-goals:

1. No direct actuator, plant-step, capability-minting, override, or bypass authority for learned code.
2. No free-form commands or learned emergency templates.
3. No online or production reinforcement learning.
4. No tuning on the sealed final split.
5. No safety-limit, reserve-limit, emergency-threshold, or HMC-policy changes in this issue.
6. No inter-process model service in version 1.
7. No deployment authorization through design or experiment approval.

## 4. Exact causal and temporal contract

Let the latest completed, HMC-issued and runtime-verified state be `S_s` where `snapshot.completed_step == s`.

- The current valid proposal uses `requested_application_step=s`.
- Candidate schedule element zero is command `A_s`.
- Applying `A_s` transforms `S_s` into first target state `S_{s+1}`.
- Schedule element `k` is `A_{s+k}` for `k in [0,31]`.
- Forecast target element `k` is `S_{s+k+1}` for `k in [0,31]`.
- Only `A_s` may be proposed now; `A_{s+1}` through `A_{s+31}` carry no authority.
- The planner reevaluates after the next valid snapshot.

The history is 16 consecutive observations ending in `S_s`: `S_{s-15}` through `S_s`. Inputs may contain only information available at or before `S_s`, plus future values that are genuinely fixed and known at decision time. Realized future measurements, disturbances, HMC decisions, health reductions, and labels are prohibited.

## 5. Runtime history contract

Add an in-process `VerifiedHistoryBuffer` owned by the advisory candidate source and located outside HMC authority modules.

For each control cycle:

1. Receive the exact snapshot and receipt emitted by HMC.
2. Call `hmc.verify_snapshot(snapshot, receipt)` in the same process.
3. Accept the record only if verification succeeds and the returned handle matches snapshot digest, cycle, and sequence.
4. Project the snapshot to an immutable normalized observation record.
5. Store record identity, snapshot digest, verification receipt digest, sequence, completed step/time, run, epoch, HMC contract, snapshot schema, topology, and previous-chain identities.
6. Require sequence and completed step to increment by one and time to increment by exactly 60 seconds.
7. Clear the buffer on run, epoch, scenario, topology, HMC contract, snapshot schema, or cadence change; gap; duplicate; reversal; chain mismatch; verification failure; terminal transition; or manual reset.
8. Retain exactly the latest 16 accepted records.
9. Return `WARMUP_NO_PROPOSAL` until 16 records exist. This means no learned proposal for the first 15 observations.

The buffer stores projected records and evidence identities, not serializable `VerifiedSnapshotHandle` objects. Handles remain non-serializable. Version 1 has no IPC boundary and makes no claim of signatures, MACs, or external cryptographic authentication. Any later IPC design needs a signed transport contract and separate safety review.

## 6. Observation and target schemas

### 6.1 History input

The history manifest deterministically orders causal operational fields by stable descriptor ID. It may include primary telemetry, command reference, operational feedback, disagreement, health state, alarms, and missingness indicators only when each field is available in the runtime snapshot and declared in the schema. Duplicate semantic channels are forbidden.

Input shape is `float32[16,F_history(topology_manifest)]`. The artifact binds the exact observation manifest and topology digest. Unknown fields, missing required fields, duplicate descriptor IDs, ordering drift, or digest mismatch fail closed.

### 6.2 Forecast target

Targets are a minimal semantic projection, not every snapshot field:

- for every ordered habitable zone: CO2 concentration, dry-bulb temperature, and relative humidity;
- once globally: battery resource gauge, oxygen-reserve resource gauge, and sorbent-reserve resource gauge.

The unique width is `W(T) = 3 * habitable_zone_count(T) + 3`. The audited two-zone V5 topology has `W=9`. Resource gauges appear once even if another snapshot view references the same descriptors. The target tensor is `float32[32,W(T)]`.

Every target manifest entry records ID, source descriptor, unit, zone or global scope, physical range, transform, inverse transform, loss normalization scale, safety relevance, and crossing thresholds. The implementation stops if a required descriptor is absent or duplicated. A topology change requires a new manifest and model qualification.

Environmental targets come exclusively from `primary_telemetry`; secondary telemetry and disagreement may be inputs or diagnostics but never substitute labels. Resource targets come from the three unique operational resource gauge descriptors. A missing, duplicate, unavailable, or non-finite required target in any candidate invalidates the entire 12-candidate decision for training and every metric. The family is quarantined only if no complete decision remains. Hidden truth may be recorded in a separate evaluator namespace for observability diagnostics and simulator validation; it is excluded from model features, model targets, normalizers, and candidate scores.

## 7. Candidate catalogue

Version 1 contains exactly 12 canonical normal-operation schedules. Each candidate has a stable ID, purpose, applicable operating modes, and 32 complete commands. Commands use only the existing proposal command surface and declared bounds. Emergency safe-action templates remain HMC-owned and are excluded.

Construction:

1. Inventory required command fields, units, static bounds, mode constraints, reserve policy, and actuator-rate semantics.
2. Define `candidate_hold` from achieved-state/current-policy semantics.
3. Define 11 conservative bounded variants across ventilation, circulation, scrubbing, thermal, humidity, and resource-preserving intent without inventing unsupported controls.
4. Ensure every candidate is semantically distinct and deterministically ordered.
5. Validate all 32 commands and reject catalogue publication on malformed, non-finite, duplicate, out-of-bound, or unstable serialization.
6. Canonically serialize with UTF-8, sorted object keys, compact separators, and one trailing LF; bind SHA-256 to all data and model artifacts.

Three statuses are never conflated:

- `STATICALLY_VALID`: all commands satisfy schema and static bounds.
- `ROLLOUT_FEASIBLE`: all 32 transitions complete from one offline checkpoint.
- `RUNTIME_FIRST_STEP_FEASIBLE`: HMC preflight accepts `A_s` from current state.

Static validity does not promise dynamic feasibility. Offline future infeasibility is retained as a labelled hard violation and makes that candidate ranking-ineligible for the affected checkpoint. Runtime HMC preflight remains authoritative for the first command.

The four-candidate ablation is frozen after catalogue validation as `candidate_hold` plus the three non-hold candidates with lexicographically smallest stable IDs. No result may influence this subset.

## 8. Distinct runtime outcomes

- `SELECTED_HOLD`: the ranker intentionally selects `candidate_hold`; this is a valid proposal and is measured as a learned hold selection.
- `ABSTAINED`: validated model execution elects no candidate because uncertainty, OOD, ambiguity, or eligibility gates fail; submit no learned proposal.
- `WARMUP_NO_PROPOSAL`: fewer than 16 verified records; submit no learned proposal.
- `INVALID_OUTPUT`: malformed shape, NaN, infinity, identity mismatch, unknown candidate, or interval error; reject before proposal and submit no learned proposal.
- `TIMEOUT_NO_PROPOSAL`: the deadline expires; discard late output and submit no learned proposal.
- `HMC_REJECTED_TO_HOLD`: HMC receives a proposal and rejects it to deterministic hold without emergency override.
- `HMC_EMERGENCY_OVERRIDDEN`: HMC replaces a proposal under its emergency policy.

Each outcome has a separate count, rate, latency, and HMC receipt association. They must not share one generic fallback metric.

## 9. Long-horizon scenarios and decision eligibility

Existing checked-in Habitat V2 scenarios are at most 10 steps and cannot supply these groups. Add a long-horizon V5 family contract after approval.

For a decision at completed step `s`:

- history eligibility requires `s >= 15` and states `S_{s-15}` through `S_s`;
- cadence eligibility requires `scenario.dt_seconds == 60` and exact 60-second observation increments;
- future eligibility requires configured transitions through `A_{s+31}`, so `scenario.steps >= s+32`;
- the smallest scenario for one decision at `s=15` has 47 configured transitions and states through `S_47`;
- multiple eligible decisions require correspondingly longer scenarios;
- no end padding, wrapped timeline, repeated truth, or truncated target is allowed.

Scenario families cover operating mode, occupancy/load level, initial environmental state, resource state, actuator state, fault type/severity/onset, sensor condition, thermal/external conditions, and disturbance schedule. Family identity includes all semantic base conditions; variants of one family never cross data splits.

## 10. Offline counterfactual rollout kernel

The corpus generator uses an offline-only kernel that cannot mint HMC handles or capabilities and is not imported by runtime planning code.

One canonical checkpoint contains:

- scenario and topology identities;
- decision step `s`;
- exact `PlantState S_s`;
- sensor memory and health tracker state needed for causal instrumentation;
- exogenous and operating-mode timeline known to the scenario;
- fault schedule and sensor-fault state;
- deterministic seed and random-stream state;
- history observation and evidence digests.

For each candidate, clone the checkpoint by canonical reconstruction, then call the deterministic one-step physics seam for `A_s` through `A_{s+31}`. After every step, run the same instrumentation projection needed to produce operational target labels and advance sensor memory. Common random numbers and the same exogenous/fault timeline are used across candidates.

Each result records all 32 target states, hidden evaluator truth in a segregated namespace, command and state digests, per-step feasibility, termination reason, and replay receipts. A dynamic infeasibility or emergency condition ends that candidate rollout, marks its remaining horizon unavailable, and makes that candidate ranking-ineligible; it does not delete the decision group. Unavailable trajectories are masked from training and every forecast metric under one method-independent eligibility mask. A missing, duplicate, or non-finite required primary environmental or unique resource label in any candidate invalidates the entire 12-candidate decision. Generator defects, identity mismatch, wrong cadence, and insufficient scenario length also invalidate the decision. A semantic family is quarantined only when no complete decision remains.

The ranking-group digest covers checkpoint identity, all 12 candidate IDs, catalogue and schema digests, target rows or termination records, simulator commit, and kernel version.

## 11. Pilot, power, coverage, and split

The old fixed total of 240 families is removed because it lacked a cost or power justification.

After sign-off, run a deterministic feasibility pilot of at most 32 semantic families and at most 12,288 candidate transitions (`32 * 12 * 32`). The pilot may exercise the simulator and deterministic baselines but may not fit a learned model or inspect a final split. It estimates:

- per-family primary-metric variance and paired correlations;
- frequency and distribution of dynamic infeasibility;
- action identifiability and oracle headroom;
- coverage-cell feasibility;
- transition time, storage, and replay determinism.

Power uses paired family log-ratios with target magnitude `-log(0.90) = 0.10536051565782628`. Forecast power compares the regularized linear action-conditioned baseline with action-agnostic persistence on identical complete decisions. Ranking power is blocked until the metric amendment freezes the exact true score; it then compares the same frozen non-neural forecaster with 12 candidates versus the frozen four-candidate subset. Calculate `pilot_sd` with `ddof=1`. Fewer than two finite pairs, a non-finite result, or a cap violation stops work; zero SD uses 30 FINAL families. Required FINAL families and roster selection follow the exact JSON formula. Coverage cells are the Cartesian product of four operating modes (`occupied`, `eva_transition`, `contingency`, `dormant`) and fault presence (`absent`, `present`). Every cell requires at least 6 TRAIN, 3 VALIDATION, and 3 FINAL families.

Caps are 384 families and 2,000,000 candidate transitions. If power, minimum coverage, or complete paired rollouts exceed either cap, stop and return to Ben; do not lower the effect threshold after seeing pilot data.

Whole families are sorted by `SHA256("issue52-split-v1" || family_id)` and allocated 70% TRAIN, 15% VALIDATION, 15% FINAL using largest-remainder integer allocation with stable TRAIN, VALIDATION, FINAL tie order. Split manifests are frozen before fitting. FINAL remains access-controlled and is invoked once per approved model version after architecture, preprocessing, thresholds, catalogue, score, and artifact digests are frozen.

## 12. Baseline-first model development

Implement before neural training:

1. Persistence, action-agnostic.
2. Recent-delta linear continuation, action-agnostic.
3. Regularized linear multi-output action-conditioned model.
4. Linear autoregressive action-conditioned dynamics rolled for 32 steps.
5. A handwritten physics-informed predictor only if it uses no future truth or hidden runtime state.

All methods receive identical groups, inputs, target manifests, candidates, splits, and metrics. The best non-neural baseline is selected on VALIDATION by the primary forecast metric with stable complexity then name tie-break. It is frozen before learned FINAL evaluation.

Before learned fitting, observability and action-information tests must show that histories with materially different futures are either distinguishable or covered by uncertainty/abstention, and that at least one non-hold candidate pair produces target differences above instrumentation tolerance in each mandatory operating-mode coverage cell. Otherwise stop.

## 13. Learned forecast model and training

Interface:

- history: `float32[16,F_history(T)]`;
- candidate schedule: `float32[32,F_action(T)]`;
- optional known-future schedule: `float32[32,F_known(T)]`;
- output mean: `float32[32,W(T)]`;
- output calibrated 90% interval bounds: two tensors of the same shape;
- one shared action-conditioned model evaluated for all candidates.

Candidate model order:

1. Compact shared-step autoregressive MLP.
2. Compact direct multi-horizon MLP only if the first cannot meet validation gates.
3. A small temporal architecture only after a written validation-based failure analysis.

The smallest candidate passing all VALIDATION gates wins. No end-to-end policy network is in scope.

Training procedure:

1. Fit normalizers on TRAIN only and store physical inverse transforms.
2. Train on TRAIN families only with fixed seeds `520032`, `520033`, and `520034`.
3. Use weighted Huber trajectory loss normalized per manifest scale; horizons 1-8 and 9-32 are reported separately, with equal mean weight per horizon inside each band.
4. Freeze optimizer, learning rate, batch size, maximum epochs, clipping, and early-stopping rule before the first comparative learned run.
5. Select checkpoint on VALIDATION primary metric; tie-break by lower safety-critical error, smaller artifact, then earlier epoch.
6. Calibrate 90% intervals on VALIDATION only using conformal residual calibration grouped by family.
7. Freeze OOD, interval, ambiguity, and score thresholds before FINAL.

Artifact includes architecture, weights, normalizers, all schema/catalogue/split/preregistration digests, source commit, seeds, hyperparameters, runtime requirements, validation results, and qualification state. Any identity mismatch fails loading.

## 14. Deterministic ranking

The model predicts trajectories; deterministic code ranks them. Candidate score components may include normalized predicted safety exposure, dangerous crossing risk, environmental tracking error, resource depletion, command energy proxy, actuator movement/wear, reserve use, uncertainty, and intervention relative to hold. Before any comparative fitting, ranking power calculation, or experiment, a separate commit-bound amendment must freeze the exact true-trajectory score, predicted-score formula, component units, normalizers, weights, hard-infeasibility value, operational metric formulas and denominators, and target/catalogue/command manifest digests, and Ben must approve that commit. Validation may select a model under those frozen formulas but may not redefine the primary endpoint. A recorded rollout infeasibility is always a hard ineligibility, never a finite score penalty.

A predicted hard safety crossing, dynamic infeasibility record, exhausted resource interval, or uncertainty limit makes a candidate ineligible. If no candidate remains, abstain. Ties choose lower predicted safety exposure, then lower uncertainty, then lower intervention, then `candidate_hold`, then stable candidate ID.

The ranker may return only a candidate ID and score evidence. The adapter extracts `A_s` and creates at most one proposal bound to current snapshot and `requested_application_step=s`.

## 15. Advisory integration and HMC authority

The source is feature-disabled by default and in-process. It has no plant object, `HMC.step` reference, capability issuer, emergency template, or authority token.

HMC remains the sole authority and may:

- accept the valid proposal;
- policy-modify or clamp it under operating-mode or reserve policy;
- emergency-override it;
- reject invalid or infeasible input to safe hold.

HMC still reparses the proposal; validates run, epoch, topology, scenario, snapshot, application step, freshness, shape, and bounds; performs deterministic arbitration and current-step preflight; binds the exact final command to a capability; performs the plant step; and verifies causal replay before commit.

Any implementation diff in `hmc.py`, `safety.py`, proposal validation, capability issuance, physics application, or replay is isolated and requires Ben's separate safety-lane review. A model gain cannot waive that review.

## 16. Preregistered metrics and gates

The JSON preregistration is authoritative for numbers. Summary:

- Primary forecast: normalized MAE over horizons 9-32. Learned/best-baseline point ratio at most 0.90 and paired 95% bootstrap upper ratio below 0.98.
- Short horizon: horizons 1-8 point ratio at most 1.05 and upper ratio at most 1.08.
- Primary ranking, inactive before the approved metric amendment: normalized oracle regret for 12 candidates versus frozen four-candidate ablation; point ratio at most 0.90 and upper ratio below 0.98.
- Dangerous crossings: recall difference at least -0.02; false-crossing-rate difference at most +0.01.
- Total safety-bound exposure: paired mean difference at most 0 and upper 95% bound at most 0.
- Wear, reserve use, healthy false intervention, inactive before the approved metric amendment: ratio point and upper 95% bound at most 1.05.
- Authority, replay, provenance, split leakage, and non-finite committed-state violations: exactly zero.
- Inference all 12: p99 at most 250 ms, timeout rate zero, measured on the frozen qualification-host fingerprint with no other benchmark workload.
- Bootstrap: 10,000 paired whole-family resamples using the JSON's SHA-256 counter sampler, ascending UTF-8 family order, type-7 percentile interval, and non-finite failure rule; Holm uses the registered one-sided bootstrap p-values, UTF-8 tie-break, and step-down procedure.

Ratio edge cases are frozen: if the comparator is zero, the candidate must also be zero and the ratio gate passes as equality; a positive candidate value fails. A bootstrap resample where both values are zero contributes ratio 1.0. This rule applies to forecast, regret, wear, reserve, and intervention ratios.

Metric algorithms are frozen. For each complete candidate trajectory, NMAE is the mean over selected horizons and targets of `abs(prediction-target)/manifest_scale`; average candidates within decision, decisions within family, then families. Point ratios divide these aggregate means, never average per-family ratios. Ranking regret per decision is `(selected_true_score-oracle_true_score)/max(abs(oracle_true_score),1e-12)` and follows the same decision-then-family aggregation. Oracle ties use true score, safety exposure, intervention, `candidate_hold`, then stable ID. Crossing events are first directional threshold transitions per family/candidate/channel; matches require the same identity/direction within one step. Recall is matched/truth events, with no truth events scoring 1 only when there are also no predictions. False rate is false events/non-crossing truth opportunities, with zero opportunities scoring 0.

Bootstrap samples whole families with replacement using the preregistered SHA-256 counter sampler and recomputes the complete aggregate and ratio. Families are ordered by ascending UTF-8 ID. The 95% interval is the equal-tailed percentile interval from sorted bootstrap values using type-7 linear quantiles at 0.025 and 0.975; any non-finite family value, aggregate, or bootstrap statistic fails the gate. Holm correction is applied separately to the crossing family and operational-regression family using the preregistered one-sided bootstrap p-values, metric-name tie-break, step-down stopping rule, and adjusted-p formula. The three operational measures remain separate gates and stay inactive until the metric amendment freezes their formulas.

Hard authority and safety gates take precedence over forecast and utility gains.

## 17. Required tests

Contract tests:

- exact `S_s`, `A_s`, `S_{s+1}`, application-step semantics;
- topology-derived unique target order and audited V5 width 9;
- no duplicated resource descriptors;
- 16-record history continuity and all reset reasons;
- warm-up for first 15 observations;
- canonical catalogue, exactly 12 IDs, 32 commands each;
- separate static, rollout, and runtime feasibility states;
- long-scenario eligibility and truncated-group rejection.

Data tests:

- checkpoint reconstruction and paired initial identity;
- sensor memory, health, fault, and exogenous timeline preservation;
- no hidden truth or future leakage;
- family-disjoint deterministic split;
- common random numbers;
- byte-identical manifests and deterministic replay;
- explicit infeasibility and termination labels.

Model and ranker tests:

- artifact/schema/catalogue/topology compatibility;
- finite output, interval ordering, inverse transforms;
- candidate batch/order invariance;
- fixed-seed reproducibility;
- deterministic scores and tie-breaks;
- separate hold, abstention, invalid, timeout, `HMC_REJECTED_TO_HOLD`, and `HMC_EMERGENCY_OVERRIDDEN` counters, rates, and receipt associations;
- only `A_s` reaches proposal construction.

Authority and integration tests:

- zero/one proposal invariant;
- stale run/epoch/topology/snapshot/step rejection;
- emergency, mode, reserve, and preflight precedence;
- exact final-command capability binding;
- learned modules cannot import or call the plant-step seam;
- disabled source reproduces deterministic HMC traces;
- late output cannot affect a later cycle;
- tampering causes replay failure and prevents invalid commit;
- full demo completes with feature off, shadow forecast, and qualified advisory mode.

Performance tests report p50/p95/p99 latency, startup time, peak RSS, artifact size, and deterministic repeatability. The initial frozen host is `DESKTOP-069U89D`: Windows 11 Pro version 10.0.26200 build 26200, x64, AMD Ryzen 5 4600H with Radeon Graphics, 12 logical processors, 16,505,847,808 physical-memory bytes, and Python 3.14.0. Run 20 untimed warm-ups followed by 1,000 timed all-12 decisions with no competing benchmark workload. Measure with monotonic `time.perf_counter_ns` from immediately before current-history projection/validation through all 12 forecasts, output validation, scoring, selection, and proposal-adapter construction. Exactly 250 ms is timely; greater is timeout and its result is discarded even if completion is observed concurrently. p99 is nearest rank `ceil(0.99*N)` in sorted timings. Record runtime-library versions in the result artifact. A different deployment target requires a separate performance gate.

## 18. Evidence, failure handling, and rollback

Every decision trace records history, schema, topology, catalogue, model, normalizer, snapshot, candidate, forecast, interval, score, selected outcome, proposal, HMC receipt, final command, capability, replay, and latency identities. Large arrays may be content-addressed but their digest and retention path remain in the trace.

Invalid history, unavailable model, artifact mismatch, OOD, uncertainty, ambiguity, NaN, shape error, or timeout produces no learned proposal. HMC rejection or override follows existing deterministic semantics. Replay failure prevents invalid state commit.

Rollback disables the candidate source and removes the artifact reference without weakening HMC. Traces and failed artifact identities are preserved.

## 19. Phased execution after approval

Phase 1, contracts: freeze manifests, history, timing, long-scenario, catalogue, rollout-row, artifact, and trace schemas.

Phase 2, pilot: create up to 32 families and run raw deterministic rollout feasibility. Then commit the exact manifests, true and predicted score formulas, operational metric formulas, power result, roster, and split as a metric amendment and obtain Ben approval before comparative fitting.

Phase 3, data infrastructure: implement long scenarios, offline kernel, validators, leakage checks, and deterministic corpus generation.

Phase 4, baselines: implement all non-neural comparators, observability tests, oracle headroom, and action-identifiability gate.

Phase 5, model: train in increasing complexity, select and calibrate on VALIDATION, freeze the complete candidate.

Phase 6, ranking: freeze deterministic objective and compare 12 candidates with the four-candidate ablation and oracle.

Phase 7, integration: add disabled-by-default advisory source, VerifiedHistoryBuffer, evidence, and fail-closed behavior.

Phase 8, qualification: run focused tests, full existing suite, end-to-end demo, replay, systems benchmark, and one sealed FINAL invocation.

Phase 9, review: provide exact diff, artifacts, manifests, reports, limitations, and rollback. Ben separately reviews every safety-core change before merge. Deployment remains blocked.

## 20. Stop conditions

Stop and return to Ben if target descriptors are missing or duplicate; 12 semantically distinct bounded candidates cannot be defined; long scenarios cannot provide complete groups; offline replay is nondeterministic; hidden-state aliasing defeats calibrated abstention; action effects are unidentifiable; power or coverage exceeds caps; a split leak occurs; a baseline dominates learned candidates; latency fails; any authority, safety, provenance, replay, or committed-state gate fails; or scope changes after approval.

Precedence is authority, safety, leakage/provenance, replay/determinism, systems budget, forecast quality, then ranking utility.

## 21. Approval record

Ben's approval must identify the Git commit containing this appendix, the short note, and preregistration. Suggested statement:

> I, Ben (`bbeennyy860-cyber`), approve the Issue #52 design package at the identified commit. I authorize contract implementation and the bounded raw feasibility pilot under the attached HMC authority boundary. Comparative fitting, ranking power calculation, model training, experiments, and deployment remain blocked until I approve the required commit-bound metric amendment. Any safety-core change requires my separate review before merge.

- Approver: `Ben (bbeennyy860-cyber)`
- Approval link: `Repository-owner-provided approval record; no public URL`
- Approval timestamp: `2026-08-21T20:04:24+01:00`
- Approved Git commit: `9531acd44797bff2531c451d7609e8c0b8c6710b`
- Scope exceptions: `Implementation and bounded raw feasibility pilot only; safety-core diff remains subject to separate PR review`
- Test-fixture boundary: `Current Issue #52 tests use a repeated short V5 fixture for contract/replay coverage only; no qualified long-horizon scenario family or corpus is claimed`
- `BEN_SIGN_OFF=true`
- `CODE_AUTHORISED=true`
- `DATA_GENERATION_AUTHORISED=false`
- `TRAINING_AUTHORISED=false`
- `EXPERIMENTS_AUTHORISED=false`
- `DEPLOYMENT_AUTHORISED=false`
- `METRIC_AMENDMENT_APPROVED=false`

Initial approval authorizes contract implementation and the bounded raw feasibility pilot only. Comparative fitting, model training, experiments, sealed-final evaluation, and deployment remain unauthorized until the metric amendment is separately approved and their authorization flags are explicitly changed. Any `hmc.py` safety-core change remains subject to Ben's separate PR review before merge.
