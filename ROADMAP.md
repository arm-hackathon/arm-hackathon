# AEOLUS research roadmap

- **Status:** proposed research direction; implementation is tracked through bounded GitHub issues
- **Last updated:** 2026-09-04
- **Planning baseline:** `main` at `6009d96a3daa8a6be982b6a3d0cbd72de847c91f` (`0.8.0`)
- **Authority invariant:** learned components may advise; the deterministic Habitat Management Computer (HMC) remains the sole final-command, plant-step, and replay authority

## Purpose

AEOLUS is moving from a sequence of bounded habitat-simulation studies toward a
falsifiable research platform for uncertainty-aware environmental decision
support under partial observability.

The primary research claim we want to test is:

> A compact model, trained from random initialization on AEOLUS-generated causal
> histories, can improve a preregistered HMC-filtered decision metric on genuinely
> unseen disturbance and configuration families relative to strong non-learned
> and learned baselines, without receiving actuator authority.

This is a research claim about advisory decision quality in a notional simulator.
It is not a claim of real-habitat performance, physical safety, flight
qualification, deployment readiness, or autonomous control.

## Current foundation

### Deterministic world

Habitat V2 currently provides a deterministic eight-zone, reduced-order
environmental analogue with:

- explicit CO2, O2, water-vapour, inert-gas, temperature, and pressure state;
- multizone recirculating airflow with a shared fan and dampered branches;
- CO2 sorbent, oxygen-store, condensed-water, battery, and thermal inventories;
- requested and achieved actuator state, slew limits, and effectiveness faults;
- physical and sensor faults with evaluator-only truth separated from operational
  observations;
- operating-mode context that does not silently alter loads, commands,
  capacities, thresholds, or physics;
- deterministic HMC proposal arbitration;
- schema-versioned scenarios, strict replay, and hash-bound evidence lineage.

The supported description is a **deterministic, source-grounded, notional
lunar-habitat environmental analogue with explicit engineering assumptions and
replayable multivariable dynamics**. It is not CFD, a complete ECLSS model, a
NASA/Artemis/Gateway digital twin, or flight-validated software.

### Learned-model evidence

The repository contains several different evidence levels. They must not be
collapsed into one claim:

1. The historical `action_aware_mlp_v1` is a roughly 2.1-million-parameter,
   action-aware point forecaster. Its archived results are useful historical
   evidence, but the original full training and run custody is incomplete, so
   current `main` cannot independently reproduce the complete campaign.
2. The Issue #53 dropout forecaster is qualified only for its narrow,
   forecast-only missingness contract. It does not establish closed-loop utility.
3. The Issue #54 distillation study showed that a smaller model can preserve
   aggregate forecast error while losing action-ranking quality. Forecast
   accuracy alone is therefore not a sufficient decision metric.
4. The Issue #55 controller race showed that HMC authority prevents bypasses but
   does not make harmful admitted advice beneficial. A point model improved the
   declared occupied-mode comfort metric while substantially worsening safety
   exposure versus rules.
5. The Issue #56 V4/V10 development study reported six family wins, no ties or
   losses, six admissions versus two, lower paired aggregate safety exposure,
   and zero HMC mismatches. Those six families represent three paired condition
   groups, not six independent environments. The winning `c9` system also mixes
   learned and deterministic/statistical screening logic, so its result does not
   yet isolate the learned component's contribution.

Issue #56 is **development evidence only**. It does not qualify a deployable
model or establish broad generalisation.

## Relationship to the historical working-model PRD

[`docs/plans/2026-08-09-aeolus-working-model-prd.md`](docs/plans/2026-08-09-aeolus-working-model-prd.md)
remains an important historical execution and decision record. Its deadline,
branch, PR, and model-result details describe an earlier project state and are
not the current roadmap.

The following boundaries remain active:

- simulator and evaluation contracts precede model claims;
- model inputs are causal and operationally observable at decision time;
- future measurements, fault schedules, hidden truth, seeds, and evaluator state
  are prohibited model inputs;
- related scenario variants and all counterfactual actions from one starting
  state remain in one split group;
- strong deterministic and linear baselines are mandatory;
- unavailable, malformed, stale, uncertain, or out-of-distribution advice fails
  closed;
- HMC owns all command and plant-step authority;
- negative results are retained;
- blind evaluation is not used for iterative tuning;
- model quality, runtime performance, and Arm-specific optimisation are separate
  evidence lanes.

## Target system

The intended research loop is:

```text
observe -> infer state -> predict consequences -> rank -> constrain -> act -> verify
```

### Observe

A future model may consume only information available through the operational
interface at the decision point:

- a causal window of completed telemetry;
- explicit observed-value masks and time-since-last-valid-observation features;
- requested commands and achieved actuator state;
- declared resource gauges;
- declared operating context;
- HMC dispositions that would genuinely be retained operationally;
- known topology and configuration descriptors;
- the candidate catalogue action or future action plan being evaluated;
- only future schedules that are explicitly declared and known at runtime.

It must not consume:

- hidden physical truth;
- fault labels or future fault schedules;
- simulator seeds or internal noise/bias state;
- future measurements;
- counterfactual outcomes;
- evaluator-only reserve or audit state;
- the future HMC arbitration result;
- undeclared future crew or environmental loads.

### Infer state

The next model should infer a compact belief state from temporal history rather
than depend solely on a large hand-engineered summary. Missingness must remain
visible through masks and staleness features; unmarked zero-imputation is
prohibited.

### Predict consequences

For each finite-catalogue action, the model should predict distributions over:

- per-zone environmental trajectories;
- safety-bound crossing probability;
- cumulative safety exposure and maximum boundary crossing;
- comfort deviation;
- oxygen, sorbent, battery, and thermal-resource consequences;
- action-minus-hold deltas;
- the existing short, medium, and longer forecast horizons.

Full trajectory heads provide diagnostic visibility. Action-minus-hold heads
answer the operational question: **what is likely to improve or worsen if this
action is proposed now?**

An optional residual formulation may be studied only if its reference rollout
uses the same causal, runtime-observable inputs and declared candidate action as
the learned model. A residual against privileged simulator or evaluator state
would invalidate the evidence.

### Rank

A deterministic scorer should rank admissible candidate actions
lexicographically:

1. hard admissibility;
2. upper-bound safety risk;
3. expected safety exposure;
4. resource depletion;
5. comfort deviation;
6. intervention cost and actuator wear.

A weighted summary may be reported, but comfort or resource savings must never
hide a safety regression.

### Constrain, act, and verify

- The model proposes an action or abstains.
- HMC independently checks every proposal.
- HMC owns the final command and plant transition.
- Missing context, numerical failure, uncertainty, OOD detection, or model
  disagreement causes abstention.
- Each decision retains model identity, corpus and feature identities, action
  scores, uncertainty, abstention reason, HMC disposition, achieved command,
  realised outcome, and replay lineage.

## Model-development ladder

### Required baselines

Every future model study should retain:

1. HMC/rules only;
2. hold/current-command;
3. the current V4 `c8` deterministic O2 guard;
4. the current V4 `c9` hybrid;
5. action-agnostic ridge;
6. action-conditioned ridge or a controlled linear state-space model.

A sparse controlled-dynamics method such as SINDYc is optional if pilot residuals
support an interpretable low-order structure. It is not required scope.

### BDM-v1: primary compact neural candidate

The proposed **AEOLUS Belief-Dynamics Model v1 (BDM-v1)** is a small causal,
dilated temporal convolutional network:

- trained from random initialization only on an AEOLUS-generated development
  corpus;
- causal bounded history with explicit masks and staleness features;
- separate candidate-action encoding;
- direct multi-horizon prediction instead of relying on recursive rollout;
- quantile outputs such as `q05`, `q50`, and `q95`;
- a small, preregistered parameter budget;
- several declared random initializations, all reported rather than selecting
  only the best seed.

A compact TCN is preferred before a Transformer because the corpus and context
are bounded, CPU determinism matters, and attention has not yet earned its added
complexity.

If missing observations dominate failure, GRU-D is the one justified comparator.
The first study should not become a wide TCN/GRU/LSTM/Transformer sweep.

### Uncertainty and abstention

Use separate development and calibration evidence for:

- conditional quantile spread;
- independently initialized ensemble disagreement if a pilot justifies the
  compute cost;
- horizon-specific conformal correction;
- reliability and coverage by scenario family and regime;
- risk-versus-coverage and useful-opportunity-versus-abstention curves;
- explicit OOD and invalid-input rejection.

These estimates support selective abstention. They are not a mathematical safety
guarantee, which is why the independent HMC boundary remains non-negotiable.

### Deferred model directions

The following do not enter the first BDM-v1 claim:

- reinforcement learning;
- continuous action optimisation;
- learned MPC;
- direct neural actuator authority;
- online learning;
- large architecture sweeps;
- graph-temporal topology generalisation;
- foundation or pretrained sequence models.

A graph-temporal model becomes eligible only if BDM-v1 fails a predeclared
network-configuration holdout in a way that a topology-aware representation can
plausibly address.

## Simulation-development direction

The next simulation work should increase decision relevance and falsifiability,
not realism for its own sake.

### 1. Machine-readable numerical provenance

Promote the current numerical ledger into a checked parameter manifest containing:

- value, unit, and valid range;
- classification as physical constant, public requirement/range,
  physics-derived quantity, engineering assumption, or stress-test range;
- citation where applicable;
- uncertainty distribution used by the family generator;
- affected systems and evaluation metrics.

NASA life-support and habitat sources support requirements and system context;
they do not calibrate AEOLUS to a flight design.

### 2. Independent reference checks

The production simulator must not be its own only oracle. Add a small independent
reference implementation or high-precision harness for selected cases:

- species and ideal-gas conservation;
- well-mixed exchange;
- humidity and condensation equilibrium;
- lumped thermal balance;
- fan/system-curve operating point;
- actuator slew;
- resource depletion.

Reference checks should not simply call the production helpers they are intended
to test.

### 3. Scenario Family Generator v2

Generate families by causal mechanism, not merely by random seed. Variation
should include:

- occupancy and explicit metabolic/thermal schedules;
- room volume and initial inventories;
- fan and branch resistance;
- scrubber, condenser, cooling, and oxygen effectiveness;
- oxygen, sorbent, battery, and thermal-resource reserves;
- fault onset, duration, severity, composition, and recovery shape;
- actuator delay, partial achievement, saturation, and intermittent failure;
- sensor bias, drift, quantisation, saturation, stale samples, and correlated
  outages;
- command/telemetry latency;
- compound resource, sensor, and actuator failures.

Operating-mode labels remain context only. A mode-specific scenario template may
explicitly declare different physical loads or commands, but no hidden mode
switch may silently change the plant.

### 4. Mechanism-held-out custody

Use four group-disjoint partitions:

1. `TRAIN` for fitting and scaling;
2. `DEV` for architecture and hyperparameter decisions;
3. `CALIBRATION` for intervals and abstention thresholds only;
4. `BLIND_FINAL` for one preregistered run after freezing the candidate.

The grouping key should include causal scenario template, physical parameter
band, fault mechanism/composition, operating schedule, action opportunity, and
sensor-failure bundle. Paired sensor variants and every counterfactual action
from one starting state remain in the same group and do not count as independent
samples.

The blind population size should be set by a pilot power analysis and materially
exceed the current three independent V4 evaluation condition groups.

### 5. Paired counterfactual action labels

At each decision opportunity:

- checkpoint one causal state;
- preserve identical prior observations and disturbances;
- roll out hold and every admissible catalogue action;
- label full trajectories and action-minus-hold outcomes;
- retain counterfactual traces only as training/evaluation labels;
- never expose future or counterfactual information to runtime inputs.

### Deferred simulator scope

Do not currently expand into water recovery, waste processing, fire chemistry,
biological closed loops, arbitrary duct networks, CFD, a claimed lunar exterior
model, or hardware-in-the-loop. Those are separate projects after the current
atmospheric decision-support question is resolved.

## Evidence and promotion gates

### Gate 0: isolate model credit

On fresh development families, compare:

- HMC/rules only;
- `c8` deterministic O2 guard only;
- the `c9` learned/statistical screen without the guard;
- full `c9` hybrid;
- action-conditioned linear baseline;
- BDM-v1.

If improvement disappears without the hand-written guard, report a deterministic
guard improvement rather than learned-model progress.

### Gate 1: corpus integrity

Required:

- zero hidden-truth inputs;
- zero group leakage;
- deterministic regeneration;
- immutable manifests and hashes;
- independently verified counterfactual traces;
- complete code/data/protocol lineage;
- blind-evaluation custody separated from model development.

### Gate 2: offline decision value

BDM-v1 must beat the action-conditioned baseline on decision-relevant metrics:

- counterfactual safety-exposure error;
- action-value/delta error;
- finite-catalogue regret;
- beneficial-action precision and recall;
- action-ranking quality;
- calibration and interval coverage;
- selected-action diversity.

Aggregate forecast error is secondary. A low-error model that ranks a harmful
action first fails.

### Gate 3: useful calibrated abstention

Required:

- coverage by horizon and scenario stratum;
- harmful-admission rate;
- useful-opportunity recall;
- explicit OOD/invalid-input abstention;
- a risk-versus-coverage curve;
- no success obtained by abstaining on nearly every useful opportunity.

Numerical thresholds must be frozen from pilot evidence before blind evaluation.

### Gate 4: HMC-filtered closed-loop development

Compare all arms on identical exogenous traces. Hard conditions include:

- zero authority, replay, provenance, and non-finite-decision violations;
- zero HMC final-command mismatches;
- model-path p99 latency within the current 250 ms ceiling;
- no hard-safety family loss outside a preregistered non-inferiority margin;
- paired aggregate safety benefit, or equal safety with a predeclared resource
  benefit;
- confidence intervals calculated at the independent condition-group level.

### Gate 5: blind confirmation

- Freeze architecture, weights, thresholds, scorer, action catalogue, and source.
- Open the blind population once.
- Publish negative results unchanged.
- A failed gate returns the model to development with a new hypothesis and a new
  future holdout.
- A failed final population cannot be reused as the blind set for a revised
  candidate.

### Gate 6: conditional runtime and release work

Only after blind confirmation may a later issue evaluate:

- native Arm64 performance;
- FP32/quantised numerical and action-selection equivalence;
- clean package installation;
- runtime resource use;
- an installable release candidate.

ONNX export, quantisation, green CI, or Arm execution are not substitutes for
model evidence.

## Suggested 3-6 month sequence

The phases are evidence gates rather than deadline promises.

### Phase 0: freeze the research contract

- successor benchmark/evidence contract;
- canonical current-status matrix;
- primary hypothesis and non-claims;
- baseline and ablation roster;
- metric hierarchy and split custody.

### Phase 1: strengthen the world and evaluation

- machine-readable parameter provenance;
- independent reference checks;
- sensitivity analysis;
- Scenario Family Generator v2;
- mechanism-held-out split contract.

### Phase 2: build the baseline pack

- paired counterfactual corpus;
- learning curves;
- HMC, hold, `c8`, `c9`, ridge, and state-space baselines;
- learned-only versus guard-only attribution.

### Phase 3: build BDM-v1

- causal TCN;
- direct trajectory and action-delta heads;
- masks and staleness inputs;
- quantile outputs;
- multi-seed and feature ablations.

### Phase 4: calibrated advisory study

- uncertainty calibration;
- OOD and abstention policy;
- deterministic action scorer;
- HMC-filtered closed-loop development comparison.

### Phase 5: blind confirmation

- frozen candidate packet;
- preregistered blind protocol;
- one blind run;
- immutable result and independent review.

### Phase 6: one harder challenge

Only after a successful blind result, test one new boundary such as correlated
sensor loss, delayed/stuck actuators, an unseen resource regime, a network
configuration shift, or compound-fault recovery.

### Phase 7: conditional runtime optimisation

Only after the model survives the challenge should AEOLUS reopen Arm64,
quantisation, packaging, and release work for this model generation.

## Stop or redirect criteria

Pause the model lane and preserve the result if:

- BDM-v1 cannot beat the action-conditioned linear baseline on action ranking;
- improvement disappears when the deterministic guard is removed;
- results depend mostly on one dormant/O2-boundary pattern;
- uncertainty is badly miscalibrated under family shift;
- apparent success comes from near-total abstention;
- one plausible simulator assumption reverses the result;
- leakage or family overlap is found;
- any HMC authority or replay inconsistency appears;
- the blind confirmation fails.

The correct response is to improve the question, corpus, or simulator—not to
silently add parameters or tune against the failed blind population.

## Work tracking and contribution boundaries

The initial programme is split into bounded issues with explicit dependencies:

| Phase | Issue | Initial owner | Dependency boundary |
|---|---|---|---|
| Contract | [#70 — Freeze the BDM-v1 benchmark and evidence contract](https://github.com/arm-hackathon/arm-hackathon/issues/70) | [Alex Kurkar (`akurkar07`)](https://github.com/akurkar07) | Foundation |
| World | [#71 — Add machine-readable Habitat V2 provenance and independent reference checks](https://github.com/arm-hackathon/arm-hackathon/issues/71) | [Alex Kurkar (`akurkar07`)](https://github.com/akurkar07) | Align with #70 |
| World | [#72 — Build Scenario Family Generator v2 with mechanism-held-out splits](https://github.com/arm-hackathon/arm-hackathon/issues/72) | [MS Mesh (`MS-Mesh`)](https://github.com/MS-Mesh) | Blocked by #70 and #71 |
| Attribution | [#73 — Isolate learned-model credit with c8/c9 ablations](https://github.com/arm-hackathon/arm-hackathon/issues/73) | [MS Mesh (`MS-Mesh`)](https://github.com/MS-Mesh) | Blocked by #70 and #72 |
| Baselines | [#74 — Implement action-conditioned ridge and controlled state-space baselines](https://github.com/arm-hackathon/arm-hackathon/issues/74) | [Alex Kurkar (`akurkar07`)](https://github.com/akurkar07) | Blocked by #70 and #72 |
| Model | [#75 — Prototype the BDM-v1 causal temporal-convolutional adviser](https://github.com/arm-hackathon/arm-hackathon/issues/75) | [MS Mesh (`MS-Mesh`)](https://github.com/MS-Mesh) | Blocked by #70, #72, #73, and #74 |
| Calibration | [#76 — Calibrate BDM-v1 uncertainty, OOD detection, and abstention](https://github.com/arm-hackathon/arm-hackathon/issues/76) | [Alex Kurkar (`akurkar07`)](https://github.com/akurkar07) | Blocked by #70 and #75 |
| Closed loop | [#77 — Run the HMC-filtered development study and prepare blind confirmation](https://github.com/arm-hackathon/arm-hackathon/issues/77) | [MS Mesh (`MS-Mesh`)](https://github.com/MS-Mesh) | Blocked by #73 and #76 |

The protected blind run is intentionally **not** an open implementation issue.
A separate one-shot issue may be created only if #77 passes, the complete
candidate is frozen, and repository-owner approval is recorded. Runtime,
quantisation, packaging, and release work remain downstream of that result.

Each issue states its observable acceptance checks, non-goals, and evidence
boundary. Contributors should keep implementation files single-owner, use
isolated branches, retain negative evidence, and open reviewable pull requests
rather than combine simulator, model, closed-loop, and runtime claims in one
change.

## References

Repository evidence and design records:

- [`README.md`](README.md)
- [`MODEL_CARD.md`](MODEL_CARD.md)
- [`CORPUS_DATASHEET.md`](CORPUS_DATASHEET.md)
- [`SAFETY_CASE.md`](SAFETY_CASE.md)
- [`DESIGN_TRADEOFFS.md`](DESIGN_TRADEOFFS.md)
- [`docs/provenance/habitat-v2-numerical-ledger.md`](docs/provenance/habitat-v2-numerical-ledger.md)
- [`docs/evidence/issue-54-distillation-card.md`](docs/evidence/issue-54-distillation-card.md)
- [`docs/evidence/issue-55-race-card.md`](docs/evidence/issue-55-race-card.md)
- [`docs/evidence/issue-56-action-risk-v4-model-v10.md`](docs/evidence/issue-56-action-risk-v4-model-v10.md)
- [`docs/evidence/closed-loop-advisory-historical-index.md`](docs/evidence/closed-loop-advisory-historical-index.md)

External technical context:

- [NASA Life Support Baseline Values and Assumptions Document](https://ntrs.nasa.gov/citations/20180001338)
- [NASA Gateway integrated ECLSS modelling](https://ntrs.nasa.gov/citations/20230009866)
- [Learning-based model predictive control](https://arxiv.org/abs/1906.12189)
- [Temporal convolutional sequence modelling](https://arxiv.org/abs/1803.01271)
- [GRU-D for missing multivariate time series](https://doi.org/10.1038/s41598-018-24271-9)
- [Sparse identification of nonlinear dynamics with control](https://arxiv.org/abs/1605.06682)
- [Probabilistic ensembles with trajectory sampling](https://arxiv.org/abs/1805.12114)
- [Runtime-assurance architectures](https://arxiv.org/abs/2110.03506)
