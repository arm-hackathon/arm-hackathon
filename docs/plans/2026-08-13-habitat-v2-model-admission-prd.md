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
**Evidence refresh:** 2026-08-13, after publishing corrected O1 and restacked H1 heads and independently reviewing the O1 packet correction
**Current code status:** O1 and H1 are published draft PRs. The eight-file O1 packet correction has one bounded independent review with no blocking findings. Candidate-wide O1 review and H1 review remain pending. Neither candidate is approved, merged, tagged, released, deployed, or a learned-model qualification
**Project-submission boundary:** an early hackathon entry is separate from GitHub merge/release status and does not freeze later repository work

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
| Tracker | `ben/habitat-v2-model-admission-prd` | `alex/ai-2` | This PRD and live evidence checklist, draft PR #33 | `none` |
| O1 | `ben/habitat-v2-observability-restack` | `alex/ai-2` | Operational observability on corrected plant bytes, draft PR #34 at `9685c8ebe772bed88e86d7eaa57ab605c7c90dc0`; packet correction independently reviewed | `minor`, consolidated release line |
| H1 | `ben/habitat-v2-hmc-restack` | O1 | Deterministic HMC plus full control-trace replay, draft PR #35 at `0445af158edbbd7189dcbe7cad8600ca35deddb0` | `minor`, consolidated release line |
| D1 | `ben/habitat-v2-forecast-data-foundation` | H1 | Model contract, corpus, splits, baselines, evaluator | `minor`, consolidated release line |
| E1 | `ben/habitat-v2-forecast-experiment-freeze` | D1 | Frozen experiment and final-access protocol | `none` unless executable behaviour changes |
| M1 | `ben/habitat-v2-forecast-candidate` | E1 | Bounded training, shadow evaluation, verdict | `minor`, consolidated release line |

All code PRs remain draft until the immutable diff, evidence packet, one bounded independent review, and Ben comprehension gate are complete. Draft publication may happen earlier so the exact candidate can receive CI and teammate review, but publication alone is not readiness or approval. No PR in this sequence authorises merge, tag, release, registry publication, deployment, or a hardware claim.

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

- [x] Restack the minimal PR #30 feature/fix commits onto F0 without retaining obsolete merge topology.
- [x] Preserve the corrected external-command accounting boundary.
- [x] Resolve version and changelog lineage under repository release-train policy.
- [x] Rerun all observability qualification cases from final bytes.
- [x] Confirm harmful concern activation for every declared harmful fixture.
- [x] Confirm healthy false-concern count and denominator with explicit bad-is-positive polarity.
- [x] Confirm subsystem localisation and explicit `UNKNOWN` abstention semantics.
- [x] Revalidate earliest-divergence receipt and all aggregate hashes.
- [x] Run focused observability tests.
- [x] Run the complete locked suite and installed-wheel smoke.
- [x] Freeze commit, diff hash, artifacts, and external bundle.
- [x] Publish the corrected O1 draft PR with exact evidence and limitations.
- [x] Run one bounded independent review of the eight-file qualification-packet reproducibility correction at exact commit `9685c8ebe772bed88e86d7eaa57ab605c7c90dc0` against parent `ed3fd5c949a382ec8ffdb060733990dd00803777`.

**Gate status:** published draft, verification, and independent review of the packet correction are complete. Candidate-wide O1 review, Ben's comprehension gate, ready-for-review transition, merge and release remain pending.

**Frozen evidence:** draft PR #34, <https://github.com/arm-hackathon/arm-hackathon/pull/34>, targets `alex/ai-2` at `ec8a6e07cddcd97915398e5a84d348b30d850c86`. Exact head `9685c8ebe772bed88e86d7eaa57ab605c7c90dc0`, tree `aa28c9204afeeee48ea116360e4b710c7827d1d5`, three commits, 25 changed files and base-relative binary diff SHA-256 `eaf3f0d63b8055681324388dab17743da1f6063634fc1fa63f9aba9b0406b855`. Detached exact-head verification passed 810/810 full locked tests, 51/51 focused qualification-plus-packet tests, Ruff, focused formatting, compilation, lock validation and exact-range diff hygiene. The canonical packet is tracked at `docs/evidence/habitat-v2-operational-observability-qualification-packet.json`; its Git blob is `f232b9e4a54caaee1494779dbc113e88f23e55be`, and a tracked fail-closed producer plus CI byte-comparison gate reproduces SHA-256 `1afed658237fd62404094eac2d50a78b8db9ad19f9b612add9ff37d1b0e3866b`. Metrics are harmful concern coverage 6/6, healthy false concerns 0/1, eligible localisation 5/5, ambiguous abstention 1/1 and overclaims 0/6. Exact fault identification remains deliberately unclaimed. Isolated installed-wheel packet generation passed. Fresh GitHub Actions run `31718771048` passed the locked suite and installed-wheel smoke. The bounded implementation audit remains author self-review. A separate bounded reviewer inspected only the eight-file packet correction from `ed3fd5c949a382ec8ffdb060733990dd00803777` to `9685c8ebe772bed88e86d7eaa57ab605c7c90dc0` and found no Critical, High, Medium, or blocking findings. The reviewer personally confirmed the exact parent and scope, two byte-identical packet rebuilds, fail-closed no-write behaviour on digest mismatch, 3/3 focused tests, 810/810 full tests, packaging and isolated import, CI command suitability, no secret exposure, and no Alex/Yaro path overlap. The minor non-blocking note is that malformed `--expected-sha256` length or format is not explicitly tested. The public receipt is recorded in the PR #34 body and <https://github.com/arm-hackathon/arm-hackathon/pull/34#issuecomment-5283342918>. This clears the packet correction only, not candidate-wide independent approval of all O1 implementation bytes. No physical, hardware, deployment or learned-model qualification is claimed.

### Gate H1: deterministic HMC convergence

- [x] Restack the minimal PR #31 HMC commits onto completed O1.
- [x] Reconcile the executable contract with final implementation.
- [x] Reparse supplied typed scenario objects through the closed plant schema at reset.
- [x] Preserve lifecycle `reset -> observe -> propose -> arbitrate -> step`.
- [x] Preserve instance, authority-epoch, and application-step capability binding.
- [x] Preserve complete-command validation and canonical physics preflight.
- [x] Preserve finite safe-action catalogue and explicit reason codes.
- [x] Preserve terminal fail-safe on unexpected post-arbitration failure.
- [x] Implement and prove the full control-trace parser and deterministic replay validator.
- [x] Validate header, event order, predecessor chain, footer, terminal status, final state, and every referenced receipt.
- [x] Validate the complete recorded proposal/arbitration sequence from the same scenario, contract, nonce, and authority epoch. Canonical proposals are self-contained; opaque rejected inputs remain commitment-bound by exact input digest and allowlisted reason because their payloads are deliberately not retained.
- [x] Independently replay every committed authoritative final command through the deterministic plant, validate each causal plant-receipt digest, and compare the final hidden plant-state identity.
- [x] Add adversarial rejection for copied/stale capability, wrong HMC, wrong epoch, wrong step, missing predecessor, reordered event, forged receipt, truncated trace, partial completed cycle, and final-state mismatch.
- [x] Prove a no-proposal safe hold repeats the last authoritative final command rather than partially achieved actuator state.
- [x] Run focused HMC tests.
- [x] Run the complete locked suite and installed-wheel smoke.
- [x] Freeze commit, diff hash, artifacts, and draft PR.

**Gate status:** published draft and implementation verification complete. Independent review, Ben's comprehension gate, ready-for-review transition, merge and release remain pending.

**Frozen evidence:** draft PR #35, <https://github.com/arm-hackathon/arm-hackathon/pull/35>, is stacked on O1 exact parent `9685c8ebe772bed88e86d7eaa57ab605c7c90dc0`. Exact head `0445af158edbbd7189dcbe7cad8600ca35deddb0`, tree `a8399a89f8f014b64cde4e63a102e93cc97ced0a`, four HMC commits, 23 changed files and parent-relative binary diff SHA-256 `bc12831814175a0ea1e10924c21bbdfcece5ddf428aee986ad2ff1193fd33ede`. The HMC patch was content-preserved during restack: stable parent-relative patch diff SHA-256 `16bf395ac77fc5b6a7c70a9132d8e85d3d4980d1118fb695e33325f498a7acc1` matched before and after. Post-restack verification passed 919/919 full locked tests and 34/34 focused HMC-control-trace-plus-packet tests. Ruff over all stack-changed Python files, compilation, lock validation and exact-range diff hygiene passed. A `core.autocrlf=false` Git-archive build produced wheel SHA-256 `dd2b4ea43abaa123b8d1f1271469d72bef0e17d71c9e2a718c45a2e4f5595e86` and sdist SHA-256 `03535857b6c2b24a8d5e46a91f7714c6e601ab5bd225767d843f1fe4c31565f5`; `control_trace.py`, `hmc.py` and `qualification_packet.py` matched their submitted Git blobs in both artifacts. Isolated installed replay completed four committed steps, produced byte-identical repeated traces, and reproduced final state SHA-256 `9d8039a4e32a262947a9a3339de369284289466c4afec672944a5c56e5da42e6`. Fresh GitHub Actions run `31718773450` passed the locked suite and installed-wheel smoke. The author adversarial audit rejected all six fully rehashed authority forgeries at parse and replay, 12/12 checks, but is self-review rather than independent approval. No learned model, direct learned control, hardware, deployment, merge or release claim is made.

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

- [x] Freeze base SHA, head SHA, clean tree, changed files, binary diff, binary diff SHA-256, tests, artifacts, and publication state for O1 and H1.
- [x] Run one bounded independent review of the O1 eight-file packet reproducibility correction against exact commit `9685c8ebe772bed88e86d7eaa57ab605c7c90dc0` and parent `ed3fd5c949a382ec8ffdb060733990dd00803777`.
- [ ] Run candidate-wide bounded independent reviews against the accepted contract and exact bytes for O1 and H1.
- [ ] If rejected, retain the rejected receipt and permit one finding-specific correction with a new regression and targeted retest.
- [ ] Teach Ben the changed behaviour and safety boundary.
- [ ] Complete five diff-grounded comprehension questions covering purpose, data flow, invariant, edge/failure case, and trade-off.
- [x] Publish or update draft PRs #34 and #35 with exact base/head, version impact, test commands, evidence links, limitations, and explicit non-claims.
- [x] Verify the posted PR bodies, exact remote heads and fresh GitHub Actions checks for #34 and #35.
- [x] Keep both PRs in draft state. Do not mark ready, merge, tag, release or deploy without separate authorization.

**Gate status:** draft publication evidence is complete for O1 and H1, and the O1 packet correction has a bounded independent review with no blocking findings. Candidate-wide O1 and H1 reviews, teaching, Ben's five-question comprehension gate, ready-for-review decision, merge and release remain pending. This tracker update does not satisfy those remaining human gates.

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
