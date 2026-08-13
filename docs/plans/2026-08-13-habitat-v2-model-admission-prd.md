# Habitat V2 Model Admission and Qualification PRD

**Status:** active execution tracker
**Created:** 2026-08-13
**Owner:** Ben Anokye-Davies
**Canonical integration branch:** `alex/ai-2`
**Frozen starting commit:** `ec8a6e07cddcd97915398e5a84d348b30d850c86`
**Starting tree:** `36ec4eae0aa02142d49be0e46796a8aa0045c7dc`
**Release mode:** parallel PRs with one later release decision
**Current integration version:** `0.7.0`, unreleased
**Publication rule:** draft PRs only until human review and comprehension gates pass

## 1. Plain-language objective

Build and test one bounded learned component for Habitat V2 without giving it control authority.

The learned component receives only operationally observable history plus one proposed safe action. It forecasts how selected habitat conditions will change over a fixed future horizon. The deterministic Habitat Management Computer (HMC) remains the sole authority that may accept a complete command and apply it to the plant.

The work succeeds only if the complete evidence chain is reproducible:

1. the corrected simulator produces causally valid traces;
2. operational telemetry excludes evaluator-only truth;
3. the HMC safely arbitrates complete commands;
4. a versioned generator creates leakage-safe whole-run datasets;
5. strong frozen baselines establish real headroom;
6. one bounded learned candidate is trained under a predeclared budget;
7. an untouched evaluation protocol compares it with the frozen baselines;
8. the result records either admission or an honest negative finding.

## 2. Current decision

### 2.1 Selected model role

The selected role is **action-conditioned multi-step forecasting**.

For each sample, the model consumes:

- a fixed window of completed operational observations;
- the operating mode visible in those observations;
- the complete proposed action selected from the deterministic safe-action catalogue;
- no hidden future loads, future commands, seeds, fault identities, exact plant truth, accounting residuals, or evaluator receipts.

It predicts selected operational outcomes over a fixed future horizon. The final tensor fields, units, ordering, window length, and horizon must be frozen in a closed versioned contract before corpus generation.

### 2.2 Authority boundary

The learned candidate is advisory-only.

```text
plant truth
  -> fallible operational telemetry and actuator feedback
  -> closed model-input projection
  -> action-conditioned forecaster
  -> predicted trajectories and uncertainty / abstention
  -> deterministic HMC arbitration and physics preflight
  -> authoritative complete command
  -> canonical plant step
  -> next operational observation
```

The model must never:

- issue a plant command directly;
- bypass HMC lifecycle or capability checks;
- change safety thresholds;
- consume evaluator-only truth at runtime;
- silently clip, repair, or replace an invalid action;
- become safety authority because its offline metric is better.

### 2.3 Why forecasting was selected

Forecasting is selected instead of exact fault classification because:

- the current operational-observability contract qualifies abnormality and subsystem localisation, not exact fault identity;
- exact simulator fault labels are evaluator truth and would encourage a hidden-label shortcut;
- the HMC already defines a finite complete-action boundary suitable for counterfactual action comparison;
- forecasting can be judged on physical trajectory accuracy and later on closed-loop outcomes;
- a deterministic baseline can remain authoritative if learning adds no value.

### 2.4 Explicitly rejected role

The historical four-class diagnosis model is not the Habitat V2 candidate. Its code may supply implementation patterns, but its corpus, topology identity, metrics, artifact, and qualification result are historical evidence only. The recorded historical verdict was `ai_advantage_demonstrated=false` with the rule baseline preferred.

## 3. Scope

### In scope

- restacking operational observability onto the corrected integration foundation;
- restacking and completing deterministic HMC replay validation;
- a closed Habitat V2 model-input and target contract;
- deterministic scenario-family generation;
- whole-run corpus and split manifests;
- provenance, leakage, replay, and duplicate-generation validation;
- deterministic forecasting baselines and headroom analysis;
- one frozen experiment protocol;
- one small learned forecasting candidate in shadow mode;
- development and untouched-final evaluation receipts;
- an honest admission or rejection verdict;
- focused draft PRs with version impact and clean-install evidence.

### Out of scope

- direct learned control or reinforcement learning;
- a learned safety governor;
- online learning or model updates during a run;
- weakening deterministic baselines after results are visible;
- target-device optimisation before model qualification;
- Arm latency, energy, power, or thermal claims;
- viewer or frontend implementation;
- merging into `main`;
- tagging, releasing, or publishing `0.8.0` or any later version;
- reconsidering rejected temporal early-risk PR #23;
- calling the reduced-order simulator a flight digital twin or validated physical habitat.

## 4. Protected invariants

Every stage must preserve these invariants:

1. **Causal command binding:** external-command validation recomputes the transition from the exact supplied command and rejects missing or mismatched command context.
2. **Atomic plant stepping:** rejected actions cannot partially mutate plant state.
3. **Deterministic replay:** canonical replay must reproduce every bound row, not only metadata or final state.
4. **Closed schemas:** unknown, missing, mistyped, non-finite, stale, copied, or identity-mismatched data fails closed.
5. **Truth separation:** fault identity, exact hidden state, resource truth, accounting residuals, future schedules, and seeds remain evaluator-only unless a separately modelled fallible meter makes a field operationally observable.
6. **HMC authority:** only a capability issued by the active HMC instance for the current authority epoch and application step may reach the external-command boundary.
7. **Complete commands:** proposals and accepted actions use the full topology-valid command schema. No partial implicit command inheritance is allowed across the authority boundary.
8. **Versioned provenance:** simulator, scenario, trace, observability, HMC, model-input, corpus, split, baseline, experiment, model, and evaluation identities are explicit and hash-bound.
9. **Whole-run isolation:** no overlapping windows or paired-family variants cross train, validation, or final boundaries.
10. **Final-set discipline:** final identities are frozen before training and are consumed once by the canonical final evaluator.
11. **Metric polarity:** healthy, wrong-target, frozen-sensor, and invariant-violating interventions are harmful and target zero. Earlier protection and lower harmful physical exposure are beneficial only when physical outcomes improve.
12. **Honest negative results:** if the learned candidate fails a frozen gate, the deterministic baseline remains preferred and the failure receipt is retained.

## 5. Branch and PR map

| Stage | Intended branch | Intended base | Review idea | Version impact |
|---|---|---|---|---|
| Tracker | `ben/habitat-v2-model-admission-prd` | `alex/ai-2` | This PRD and live evidence checklist | `none` |
| O1 | `ben/habitat-v2-observability-restack` | `alex/ai-2` | Operational observability on corrected plant bytes | `minor`, consolidated release line |
| H1 | `ben/habitat-v2-hmc-restack` | O1 | Deterministic HMC plus full control-trace replay | `minor`, consolidated release line |
| D1 | `ben/habitat-v2-forecast-data-foundation` | H1 | Model contract, corpus, splits, baselines, evaluator | `minor`, consolidated release line |
| E1 | `ben/habitat-v2-forecast-experiment-freeze` | D1 | Frozen experiment and final-access protocol | `none` unless executable behaviour changes |
| M1 | `ben/habitat-v2-forecast-candidate` | E1 | Bounded training, shadow evaluation, verdict | `minor`, consolidated release line |

All code PRs remain draft until the immutable diff, evidence packet, one bounded independent review, and Ben comprehension gate are complete. No PR in this sequence authorises merge, tag, release, registry publication, deployment, or a hardware claim.

## 6. Evidence rules

A checkbox may move to complete only when its row names immutable evidence. Acceptable evidence includes:

- full commit SHA and base SHA;
- binary range-diff SHA-256;
- exact test command and terminal result;
- artifact path plus SHA-256;
- canonical report with schema and provenance validation;
- GitHub draft PR URL and exact head/base identities;
- independent review verdict bound to the candidate SHA.

A green focused subset is not a green full suite. A historical receipt is not current evidence. A generated file from dirty or later-modified source is stale. A model warning is not an accepted intervention. Missing evidence is `not started` or `blocked`, never inferred as passed.

## 7. Execution gates and live checklist

### Gate F0: foundation freeze

- [x] GitHub live `alex/ai-2` bound to `ec8a6e07cddcd97915398e5a84d348b30d850c86`.
- [x] Starting tree bound to `36ec4eae0aa02142d49be0e46796a8aa0045c7dc`.
- [x] Starting tree independently matched correction commit `68f2e42a3c26ca798babebcea262875f101d5735`.
- [x] `main` confirmed unchanged at `5253176e0e4e498e2dcff34905ce08a257209506`.
- [x] No release or tag authorised.
- [x] PR #23 explicitly excluded.
- [x] Model role selected as advisory action-conditioned forecasting.

**Gate status:** complete.
**Evidence:** GitHub refs and PR #32 merge metadata queried 2026-08-13. No source mutation was used to establish this gate.

### Gate O1: operational observability convergence

- [ ] Restack the minimal PR #30 feature/fix commits onto F0 without retaining obsolete merge topology.
- [ ] Preserve the corrected external-command accounting boundary.
- [ ] Resolve version and changelog lineage under repository release-train policy.
- [ ] Rerun all observability qualification cases from final bytes.
- [ ] Confirm harmful concern activation for every declared harmful fixture.
- [ ] Confirm healthy false-concern count and denominator with explicit bad-is-positive polarity.
- [ ] Confirm subsystem localisation and explicit `UNKNOWN` abstention semantics.
- [ ] Revalidate earliest-divergence receipt and all aggregate hashes.
- [ ] Run focused observability tests.
- [ ] Run the complete locked suite and installed-wheel smoke.
- [ ] Freeze commit, diff hash, artifacts, and draft PR.

**Gate status:** not started on corrected base.
**Prior evidence:** draft PR #30 at `ce99b42e2631e3fb205fdf35ec6424671952af85`, 789-test CI pass. This is input evidence, not proof of the restacked candidate.

### Gate H1: deterministic HMC convergence

- [ ] Restack the minimal PR #31 HMC commits onto completed O1.
- [ ] Reconcile the executable contract with final implementation.
- [ ] Reparse supplied typed scenario objects through the closed plant schema at reset.
- [ ] Preserve lifecycle `reset -> observe -> propose -> arbitrate -> step`.
- [ ] Preserve instance, authority-epoch, and application-step capability binding.
- [ ] Preserve complete-command validation and canonical physics preflight.
- [ ] Preserve finite safe-action catalogue and explicit reason codes.
- [ ] Preserve terminal fail-safe on unexpected post-arbitration failure.
- [ ] Implement or prove the full control-trace parser and deterministic replay validator.
- [ ] Validate header, event order, predecessor chain, footer, terminal status, final state, and every referenced receipt.
- [ ] Replay the complete proposal sequence from the same scenario, contract, nonce, and authority epoch, then compare canonical rows byte-for-byte.
- [ ] Add adversarial rejection for copied/stale capability, wrong HMC, wrong epoch, wrong step, missing predecessor, reordered event, forged receipt, truncated trace, and final-state mismatch.
- [ ] Prove a no-proposal safe hold repeats the last authoritative final command rather than partially achieved actuator state.
- [ ] Run focused HMC tests.
- [ ] Run the complete locked suite and installed-wheel smoke.
- [ ] Freeze commit, diff hash, artifacts, and draft PR.

**Gate status:** not started on completed O1.
**Prior evidence:** draft PR #31 at `843a5c1485de841462cbb47e486c2185099b71a2`, 867-test CI pass and 78 focused HMC tests. Its public description is stale and does not close this gate.

### Gate D1: Habitat V2 forecast data and evaluation foundation

#### Contract

- [ ] Freeze `aeolus_habitat_v2_forecast_input_v1` with exact ordered fields, units, scalar types, topology identity, window length, cadence, action encoding, and forbidden fields.
- [ ] Freeze `aeolus_habitat_v2_forecast_target_v1` with exact ordered fields, units, forecast steps, aggregation semantics, and missing-data policy.
- [ ] Freeze canonical JSON and tensor conversion with pre-conversion and post-conversion semantic validation.
- [ ] Bind every sample to simulator, scenario, trace, observability, HMC, feature, target, and action-catalogue identities.
- [ ] Add hidden-truth leakage tests and a later-field permutation adversarial test.

#### Scenario families

- [ ] Define healthy, harmful, ambiguous, recovery, compound, actuator-delivery, sensor, operating-mode, load, and topology-location family dimensions.
- [ ] Define deterministic seed derivation from semantic family identity.
- [ ] Create structurally paired healthy and treatment variants where applicable.
- [ ] Prove paired scenarios differ only in the declared treatment fields.
- [ ] Require sufficient completed history and future horizon around every admitted sample.

#### Splits

- [ ] Freeze whole-family train, validation, and final assignments before training.
- [ ] Prove family, scenario, run, and overlapping-window disjointness.
- [ ] Prove class/regime/target coverage by independent run, not by row count alone.
- [ ] Store a forbidden final-family identity manifest outside training and calibration inputs.
- [ ] Reject any generator/evaluator attempt that crosses the split boundary.

#### Corpus and provenance

- [ ] Generate from clean committed bytes only.
- [ ] Store source and contract hashes, runtime versions, lock hash, command, counts, units, feature summaries, target summaries, and replay trace identities.
- [ ] Generate twice in the declared environment and compare canonical bytes.
- [ ] Reject schema-valid artifact substitution and wrong source-to-artifact lineage.
- [ ] Keep generated canonical artifacts write-once.

#### Baselines and evaluator

- [ ] Implement last-value persistence.
- [ ] Implement linear extrapolation.
- [ ] Implement one compact fitted statistical baseline using training data only.
- [ ] Implement a simulator/physics oracle diagnostic where evaluator truth makes it valid, and label it as unreachable deployment headroom rather than a production comparator.
- [ ] Define per-target MAE and RMSE in native units.
- [ ] Define horizon-wise and aggregate errors.
- [ ] Define safety-envelope crossing precision/recall only from forecasted operational outcomes, with explicit polarity.
- [ ] Define abstention coverage and selective risk.
- [ ] Define runtime, artifact-size, invalid-output, and non-finite-output measures.
- [ ] Make the evaluator neutral to candidate type and fail closed on identity drift.
- [ ] Run focused data-contract tests, full suite, package build, clean-wheel install, and external CLI/import smoke.

**Gate status:** not started.
**Model-training permission after D1:** still closed until B1 and E1 complete.

### Gate B1: frozen baseline and headroom result

- [ ] Run all frozen baselines over train-development diagnostics and untouched validation families.
- [ ] Run the oracle/headroom diagnostic without exposing evaluator truth to production inputs.
- [ ] Report every target, horizon, regime, family, and baseline, including failed or unsupported cells.
- [ ] Confirm observable inputs contain enough information for a learned candidate to have meaningful headroom.
- [ ] Confirm no baseline or metric was weakened after results were visible.
- [ ] Record selected production comparator and why.
- [ ] Record `PROCEED_TO_EXPERIMENT_FREEZE` or `STOP_NO_DEFENSIBLE_HEADROOM`.

**Gate status:** not started.

### Gate E1: immutable experiment freeze

- [ ] Freeze one primary learned candidate family and at most one declared ablation.
- [ ] Freeze architecture, parameter ceiling, seed list, optimiser, loss, regularisation, early-stopping rule, and epoch/time budget.
- [ ] Freeze training-only normalisation and its identity.
- [ ] Freeze validation-only model-selection and abstention calibration procedure.
- [ ] Freeze final-family one-access rule and exclusive lock behaviour.
- [ ] Freeze all quality thresholds before learned results are visible.
- [ ] Freeze tie-break and rejection rules.
- [ ] Freeze output paths as write-once.
- [ ] Prove dirty source, identity drift, prior final claim, missing prerequisite, or overlapping split fails before corpus/model/report creation.

#### Admission thresholds

The learned candidate may be admitted as a development candidate only if all declared conditions pass:

- [ ] aggregate validation error improves over the selected frozen production baseline by the predeclared margin;
- [ ] no safety-critical target or declared harmful regime regresses beyond its predeclared tolerance;
- [ ] all outputs are finite and within the target contract's representable bounds;
- [ ] abstention calibration satisfies the frozen selective-risk constraint;
- [ ] artifact identity, source identity, and inference payload fingerprints match;
- [ ] deterministic repeated inference produces equivalent outputs within the frozen numerical tolerance;
- [ ] shadow integration produces zero direct commands and zero HMC authority bypasses.

Exact numeric margins must be derived and frozen from B1 before training. They may not be selected after seeing learned-model results.

**Gate status:** not started.
**Training permission after E1:** open only for the bounded development experiment.

### Gate M1: bounded learned forecasting experiment

- [ ] Run fail-closed preflight from clean committed E1 bytes.
- [ ] Train only the frozen candidate set and seed budget.
- [ ] Preserve all candidate results, not only the best result.
- [ ] Store model bytes, canonical fingerprint, training receipt, normalisation receipt, and validation predictions.
- [ ] Select and calibrate using validation only.
- [ ] Export FP32 ONNX only if the candidate first passes the frozen development-quality gate.
- [ ] Verify framework-to-ONNX parity if export occurs.
- [ ] Integrate the candidate in shadow mode behind HMC with no command authority.
- [ ] Mutate runtime identity with valid scenario identity and require rejection.
- [ ] Mutate live model parameters while preserving the claimed artifact identity and require payload-fingerprint rejection before the first tick.
- [ ] Run focused experiment tests, full suite, package build, clean-wheel install, and external smoke.

**Gate status:** not started.

### Gate Q1: qualification and honest verdict

#### Offline qualification

- [ ] Compare learned candidate and every frozen baseline on the frozen validation matrix.
- [ ] Report per-target, per-horizon, per-regime, per-family, aggregate, abstention, invalid-output, runtime, and size results.
- [ ] Apply the frozen admission rule mechanically.

#### Final evaluation

- [ ] Require clean committed source and unresolved-review gate completion.
- [ ] Acquire an atomic exclusive final-evaluation lock.
- [ ] Prove final-family identities are disjoint and previously unconsumed.
- [ ] Consume the final set once through the canonical evaluator.
- [ ] Hash evaluator sources, model, normalisation, manifests, dependencies, commands, predictions, metrics, and final report.
- [ ] Refuse overwrite or a second canonical final claim.

#### Shadow closed-loop comparison

Compare at least these arms on identical paired families:

1. deterministic HMC with selected baseline forecast;
2. deterministic HMC with learned advisory forecast;
3. deterministic HMC without learned advice where the architecture permits a meaningful control.

Report separately:

- model warnings;
- HMC accepted proposals;
- HMC vetoes and reasons;
- healthy-reference interventions, where lower is better and target is zero;
- wrong-target interventions, where lower is better and target is zero;
- frozen-sensor interventions, where lower is better and target is zero;
- invariant violations, where lower is better and target is zero;
- harmful physical exposure, where lower is better;
- time to protection, where earlier is better only when exposure improves;
- resource and actuator usage;
- recovery time;
- missed harmful cases.

- [ ] Record `ADMITTED_DEVELOPMENT_CANDIDATE` only if every frozen gate passes.
- [ ] Otherwise record `REJECTED_RETAIN_BASELINE` with all failed gates and retain the negative result.
- [ ] State explicitly that neither verdict proves final hardware, deployment, flight, energy, power, or thermal readiness.

**Gate status:** not started.

### Gate R1: immutable review and publication

For each code candidate:

- [ ] Freeze base SHA, head SHA, clean tree, changed files, binary diff, binary diff SHA-256, tests, artifacts, and publication state.
- [ ] Run one bounded independent review against the accepted contract and exact bytes.
- [ ] If rejected, retain the rejected receipt and permit one finding-specific correction with a new regression and targeted retest.
- [ ] Teach Ben the changed behaviour and safety boundary.
- [ ] Complete five diff-grounded comprehension questions covering purpose, data flow, invariant, edge/failure case, and trade-off.
- [ ] Publish or update a draft PR with exact base/head, version impact, test commands, evidence links, limitations, and explicit non-claims.
- [ ] Verify the posted PR body and checks.
- [ ] Do not mark ready, merge, tag, release, or deploy without separate authorization.

**Gate status:** not started.

## 8. Frozen experiment design constraints

The detailed numeric contract will be produced in D1 and frozen in E1, but these constraints cannot change:

- prediction is action-conditioned, not direct policy learning;
- samples use completed observations only;
- future labels cannot appear in the input window;
- all splits occur by semantic scenario family and complete run;
- normalisation is fitted on training data only;
- final families do not influence architecture, loss, thresholds, calibration, or selection;
- deterministic baselines are implemented and frozen before learned training;
- the primary comparator cannot be selected after seeing candidate results;
- any learned gain must survive target-wise and harmful-regime checks, not only one aggregate mean;
- invalid or non-finite outputs are failures, not silently repaired predictions;
- abstention is explicit and measured;
- model artifact identity is recomputed from live payload bytes before use;
- changes to simulator, contract, corpus, split, evaluator, or model-loading bytes invalidate downstream receipts.

## 9. Required artifact set

The completed sequence must produce versioned equivalents of:

- `contracts/habitat_v2_observability_v1.json`
- `contracts/habitat_v2_hmc_v1.json`
- `contracts/habitat_v2_forecast_input_v1.json`
- `contracts/habitat_v2_forecast_target_v1.json`
- `contracts/habitat_v2_forecast_corpus_v1.json`
- `contracts/habitat_v2_forecast_split_v1.json`
- `contracts/habitat_v2_forecast_evaluation_v1.json`
- `contracts/habitat_v2_forecast_experiment_v1.json`
- scenario-family manifest
- corpus provenance receipt
- split manifest and forbidden-final manifest
- baseline metrics and headroom report
- experiment freeze receipt
- candidate training ledger
- model and normalisation fingerprints
- validation report
- optional FP32 ONNX parity receipt
- final-access claim and lock receipt
- final qualification report
- shadow closed-loop comparison report
- immutable review receipt per code PR

Exact paths may change during the design slice, but every artifact must have a closed schema, canonical identity, source binding, and parser test.

## 10. Stop conditions

Stop the sequence and report the exact blocker if any of these occurs:

- current foundation or parent branch moves after candidate work begins;
- observability or HMC restack loses a protected invariant;
- full control-trace replay cannot be made unambiguous from the contract;
- model-facing inputs require hidden evaluator truth to distinguish outcomes;
- split coverage cannot be achieved without family leakage;
- no baseline/headroom result justifies training;
- a canonical runner correctly refuses dirty or mismatched source;
- one-access final identities were consumed or contaminated;
- the bounded independent review rejects the permitted correction;
- a requested publication would require merge, tag, release, registry, deployment, or hardware claims without separate approval.

## 11. Final completion definition

This PRD is complete only when:

1. every gate has immutable evidence or an explicit stopped/rejected verdict;
2. all code PRs are focused draft PRs with verified posted bodies;
3. the learned candidate is either honestly admitted as a development-only advisory forecaster or rejected in favour of the frozen deterministic baseline;
4. no merge, tag, release, deployment, or hardware claim has occurred without separate authorization;
5. the final report distinguishes implementation, artifacts, review, comprehension, publication, merge, release, model, closed-loop, and hardware status.

A trained model by itself does not complete this PRD.
