# AEOLUS development programme

- **Status:** proposed programme for team review; not authorization to execute all work packages.
- **Source snapshot:** 2026-09-06.
- **Team:** Ben, Alex and Yaroslav; the responsibility split below is proposed, not a reassignment of existing contributions.
- **Scope:** the complete simulation, model, planning, evaluation and research-platform project.
- **Entry point:** [research roadmap](../../ROADMAP.md).

> Build a habitat-focused environmental-intelligence platform that learns from observation histories, predicts the consequences of interventions, plans under uncertainty, and demonstrates when its recommendations improve outcomes.

The ambition is substantial. The route is successive complete research slices, not a promise that more agents or a larger model will produce a breakthrough. The immediate task is to establish a useful decision problem and understand the existing failures. A visualization is a way to inspect the result, not a substitute for it.

## Contents

1. [Starting point](#1-starting-point)
2. [Target capability and scope](#2-target-capability-and-scope)
3. [Three-person responsibility split](#3-three-person-responsibility-split)
4. [Architecture and shared contracts](#4-architecture-and-shared-contracts)
5. [Scientific questions and success criteria](#5-scientific-questions-and-success-criteria)
6. [Evidence-gated phases](#6-evidence-gated-phases)
7. [Work-package catalogue](#7-work-package-catalogue)
8. [First execution wave](#8-first-execution-wave)
9. [Agents, integration and review](#9-agents-integration-and-review)
10. [Compute, reproducibility and release](#10-compute-reproducibility-and-release)
11. [Turning the plan into GitHub issues](#11-turning-the-plan-into-github-issues)
12. [Decisions, risks and resume point](#12-decisions-risks-and-resume-point)
13. [Source map and verification boundary](#13-source-map-and-verification-boundary)

## 1. Starting point

### Merged foundation versus candidate work

The inspected remote `main` is [`25a8a9ac194472d05b13983d2f8078f3e0b67f21`](https://github.com/arm-hackathon/arm-hackathon/tree/25a8a9ac194472d05b13983d2f8078f3e0b67f21). Package metadata declares `aeolus` version `0.8.0`, Python >=3.10, and a NumPy core. Existing CI includes a locked Python 3.11 suite, lint, compilation, lockfile validation, package build and installed-wheel smoke. This is existing infrastructure to extend, not something to reinvent.

The deterministic eight-zone reduced-order plant already represents gas inventories, temperature, pressure, airflow, resources, actuator dynamics, faults and observations. The HMC owns final commands, plant stepping and replay authority. Numerical provenance and independent reference-check implementations are already present on main following the contract/provenance work associated with issues #70–71 and PRs #79–80.

The newer research is an **open PR stack**, not all merged into main:

| PR | Candidate capability | Head SHA | Base |
|---|---|---|---|
| [#84](https://github.com/arm-hackathon/arm-hackathon/pull/84) | Family generation and corpus | `784cb2683d08ce6e8c1ca0466989705875b5d1a8` | `main` |
| [#85](https://github.com/arm-hackathon/arm-hackathon/pull/85) | Guard/learned-component attribution | `35657a720b20aa5568894567a68541548cfa7494` | `research/scenario-family-generator-v2` |
| [#86](https://github.com/arm-hackathon/arm-hackathon/pull/86) | Linear baselines | `6d9436d8de1c8eb91bf5981666c1b07e72eca363` | `research/bdm-v1-guard-ablations` |
| [#87](https://github.com/arm-hackathon/arm-hackathon/pull/87) | BDM-v1 causal TCN | `148884cf526f87801bbf6999d0eea60a1925dae1` | `research/bdm-v1-linear-baselines` |
| [#88](https://github.com/arm-hackathon/arm-hackathon/pull/88) | Calibration and abstention | `7d598a9df35484e54b5b73662105255389bc0b52` | `research/bdm-v1-tcn-adviser` |
| [#89](https://github.com/arm-hackathon/arm-hackathon/pull/89) | Closed-loop development study | `142af4a194ee1eba6815d94bbc92e79073c50a6d` | `research/bdm-v1-calibrated-abstention` |

Issues #70–71 are closed; #72–77 remain open at this snapshot. Preserve their actual assignees and contributed work. The new responsibility split is forward-looking. PRs #81–83 are closed, unmerged precursors to the newer generator work; inspect their relationship before recovering anything, rather than recreating them as new tasks.

Before implementation, refresh these refs, review the stack, and ask contributors about unpushed work. Choosing the next exact foundation is WP01. Publication of this plan neither merges that stack nor certifies it.

### What the latest evidence changes

The [Issue #75 card](https://github.com/arm-hackathon/arm-hackathon/blob/142af4a194ee1eba6815d94bbc92e79073c50a6d/docs/evidence/issue-75-bdm-v1-card.md) reports a 53,240-parameter pure-NumPy causal TCN trained from random initialization across five seeds. Its promotion gate failed. Reported action ranking was `0.1776` versus `0.2839` for the action-conditioned ridge baseline, and reported trajectory MAE was `11183.752544` versus `263.267153`. These are the card's aggregate metrics, not a single physical-unit error or newly reproduced measurements.

The [Issue #77 card](https://github.com/arm-hackathon/arm-hackathon/blob/142af4a194ee1eba6815d94bbc92e79073c50a6d/docs/evidence/issue-77-closed-loop-card.md) reports six arms across 32 DEV families and 192 episodes. No arm admitted a proposal; the calibrated BDM abstained at every decision. All arms had zero measured safety exposure. The candidate was withheld, and the published record states that BLIND_FINAL remained sealed.

These are **reported development results inspected in source**, not independent reproduction or broad model-safety findings. They motivate three separate questions:

1. Why did the learned predictor perform so poorly? Investigate scaling, targets, gradients, optimization and train/runtime parity; flat action labels alone do not explain every trajectory error.
2. Does the current task contain useful permissible interventions on the chosen horizon and metrics?
3. Where do advice and physical action diverge: abstention, HMC rejection, command modification, actuator limits, or genuinely ineffective action?

Preserve the failed protocol and its evidence unchanged. A new metric, regime, implementation fix or model generation needs a new study identity. It must not retrospectively turn the old failure into a pass.

## 2. Target capability and scope

### A concrete complete-system target

A researcher loads an unseen, valid habitat configuration and a declared disturbance regime. Equipment degrades while sensors provide imperfect observations. AEOLUS:

1. receives only the information available at the decision time;
2. estimates plausible hidden environmental state;
3. predicts consequences of permitted candidate interventions;
4. recommends a bounded plan or explains why it abstains;
5. passes every proposal through deterministic HMC arbitration;
6. records the command actually issued and actuation actually achieved;
7. compares predictions with the resulting observations and evaluator-only outcomes;
8. reports benefit, harm, uncertainty and cost relative to competent conventional methods.

The first slice uses the existing layout and a small justified problem. Unseen layouts are a later generalization target, not a requirement to build a universal world editor before learning anything.

### Approach choice

- **Scale BDM-v1 immediately:** quick to launch, but could spend compute on uninformative labels or inaccessible interventions. Not the main programme.
- **Rewrite as a universal simulator and large world model:** expands too many unknowns simultaneously and discards working behavior. Defer.
- **Build progressively harder complete slices:** diagnose, establish opportunity, compare methods, widen the environment, independently confirm. Recommended.

Remain habitat-focused while making interfaces and topology representations reusable. Candidate later transfers include other multizone environmental systems; none is validated by a habitat-only result.

### Non-goals

No physical life-support control, real-building actuation, flight qualification, certified safety, CFD fidelity claim, online self-modification, or direct learned actuator authority. Reinforcement learning, pretrained foundation models and continuous unrestricted action search are not initial commitments. A model may remain compact or grow if controlled experiments justify it; parameter count is not success.

Agents developing the software, the trained AEOLUS model, and any future runtime agent are different things. This plan does not introduce an LLM into the command path.

## 3. Three-person responsibility split

This is a **proposed allocation for Ben, Alex and Yaroslav to agree**, not a statement that all three have accepted roles or that existing issues have changed owners.

### Ben — models and state estimation

**Question:** What can we infer and predict from the information available?

Own the learned-model workstream: feature/preprocessing semantics, training, hidden-state estimation, action-conditioned prediction, uncertainty and model diagnostics. Maintain serious non-neural prediction/estimation baselines rather than assuming a neural model must win.

Outputs include reproducible checkpoints, prediction interfaces, model cards, learning curves, ablations and documented failure modes. Candidate agent tasks can separate numerical diagnosis, baseline implementation, training infrastructure and controlled architecture experiments.

Ben does not independently alter physical truth, final evaluation criteria or HMC authority to improve scores.

### Alex — simulation world and data generation

**Question:** What happens in the environment, and why should we trust it?

Own physical mechanisms, numerical assumptions, configurations, sensors, actuators, faults, resource accounting and scenario generation. Own the simulator-side generation machinery with Ben co-reviewing learning-data semantics.

Outputs include tested physical behavior, justified parameter ranges, scenario definitions, numerical comparisons and versioned population manifests. Candidate agent tasks can separate physical mechanisms, sensor/fault regimes, reference calculations and simulation throughput.

Alex does not manufacture favorable scenarios after model selection. The world needs validation independent of a model's performance. Hidden truth used for labels must remain inaccessible to deployed predictors and observation-only baselines.

### Yaroslav — planning and system integration

**Question:** Given those predictions, what should the system do?

Own action catalogues and bounded planners, resource/comfort objectives within the shared contract, HMC adapters, proposal-to-achievement tracing, and end-to-end integration. Own the conventional control/planning baseline with Ben supplying its estimator or predictor where needed. Coordinate the research interface and replay experience.

Outputs include executable advisory plans, deterministic fallback behavior, a comparative closed-loop runner and a complete inspectable research path. Candidate agent tasks can separate classical planning, proposal adapters, integration tests and replay/interface work.

This is a substantive control and systems lane, not ownership of every miscellaneous task. It does not confer unilateral control of safety policy or evaluation conclusions.

### Shared work with explicit responsibility

- **Benchmark and public claims:** all three review. Nominate one editor, but model authors cannot change success criteria after seeing candidate results.
- **Independent evaluation:** rotate across lanes; a person/agent not selecting the candidate reproduces the decisive comparison. Separate final-population custody from candidate tuning. Same-model review is a second inspection, not independent scientific ground truth.
- **Integration coordinator:** Yaroslav is proposed for coordinating the next integration sequence, not sole scientific authority. Confirm this alongside lane preferences.
- **Tests, docs and packaging:** each lane covers its component; an explicitly scoped release issue coordinates the assembled product. Do not place all testing and infrastructure on one person.
- **Attribution:** credit follows actual contributions, including existing work, not the proposed future lane names.

## 4. Architecture and shared contracts

```text
frozen scenario + exogenous inputs
                 |
                 v
        deterministic plant --> permitted observation history
                 ^                         |
                 |                         v
        achieved actuation          belief/state estimator
                 ^                         |
                 |                         v
       HMC final command <--- proposal <-- predictor + planner
                 |
                 v
     authoritative trace, outcomes and replay --> evaluation / interface
```

The HMC remains the sole final-command, plant-step and replay authority. The drawing is conceptual; adapt existing modules instead of inventing a second stepping loop.

Before dispatching interface-dependent issues, the team freezes the following contracts. Names here describe responsibilities, not newly implemented Python types.

| Contract | Minimum content | Proposed editor and co-review |
|---|---|---|
| Scenario/configuration | units, topology, equipment, loads, fault/sensor definitions, version, run-frozen identity | Alex; Yaroslav checks runtime compatibility |
| Observation history | timebase, sensor IDs, values, units, masks, staleness, operational commands/gauges, known context | Ben; Alex checks physical/causal meaning |
| Candidate plan | admissible action vocabulary, timing/horizon, bounded search, requested commands, fallback/abstention | Yaroslav; Alex checks actuation semantics |
| Prediction/belief | targets, horizons, ordering, units, uncertainty semantics, artifact identity, invalid-input behavior | Ben; Yaroslav checks planner consumption |
| Decision trace | proposal, HMC disposition, final command, physical achievement, model identity, timing, outcomes | Yaroslav; Alex checks plant lineage |
| Dataset/experiment | group and partition IDs, generation/config/code identities, transforms, seeds, selection history | Alex and Ben; independent evaluator reviews |

Rules that apply to every interface:

- Requested commands, HMC-final commands and achieved actuation are different fields.
- Future measurements, undeclared future loads, fault schedules, seeds and hidden truth cannot leak into operational inputs. Known future schedules require an explicit runtime-availability contract.
- Privileged diagnostic oracles have separately labelled interfaces and results, never silently shared features.
- Masked observations remain visibly missing; zero is not a universal missing-value code.
- Training transforms are fit on TRAIN, serialized and applied identically at inference.
- Every prediction declares whether it forecasts requested, expected HMC-filtered or achieved action semantics; a planner cannot assume those are interchangeable.
- Historical contracts retain their identities. A breaking schema change needs a version/migration decision and compatibility tests.
- The UI reads authoritative snapshots/traces; it cannot run a second physics engine or bypass HMC.

Keep the monorepo initially. Reuse `src/aeolus/habitat_v2`, existing contracts, runners and tests. Introduce adapters where needed; do not rename the `forecast_issue*` family merely for aesthetic consistency. Core schemas, lockfiles and version files each have one editor per active change.

## 5. Scientific questions and success criteria

### Two distinct decision tracks

**Recovery:** justified disturbances create harmful exposure or delayed recovery which permissible action may reduce. Measure integrated exposure in declared units, peak excursions, recovery time, resource use, unnecessary interventions and worst-family outcomes.

**Efficient operation:** a competent baseline already maintains safety. Test resource use, comfort or actuator wear subject to predeclared hard constraints and safety non-inferiority. Do not require strict improvement on a safety metric already at its minimum, or hide a safety loss inside a weighted comfort score.

These are proposed new protocols. BDM-v1's existing gates remain unchanged. Choose one narrow primary first slice; do not promise simultaneous victory on both tracks.

### Decision-opportunity gate before broad neural search

The small DEV pilot must establish:

- distinct permissible commands whose **achieved** actions differ;
- a measurable outcome difference above declared numerical/noise tolerances;
- an observation-only method with enough information to exploit at least some opportunities;
- no-action and harmful-intervention controls;
- ambiguous cases in which permitted histories do not distinguish hidden causes;
- a privileged diagnostic opportunity estimate, explicitly separated from deployable performance.

If the oracle cannot improve outcomes, revisit the task/horizon/action set. If the oracle succeeds but observation-only baselines cannot, investigate partial observability and information availability. If conventional planning succeeds and neural methods do not, investigate the model rather than weakening the baseline. If all advice is rejected, investigate command semantics and policy compatibility without loosening constraints merely to admit advice.

### Comparison and statistical contract

WP05 must freeze the primary endpoint, units, metric direction, safety margins, baselines, independent sample unit, sample-size rationale, tuning budget and stopping criteria before candidate selection. Numerical thresholds are deliberately not invented in this document; later issues cannot start final evaluation with these choices still open.

Use matched exogenous scenarios across arms. Keep related variants and all counterfactual actions from a shared starting state in the same split group. Windows, decision steps and random seeds are not automatically independent experiments. Report paired group-level differences with uncertainty, distributions and worst-family cases, not only a pooled mean.

Retain HMC/rules, hold, relevant existing guard/hybrid methods, strong linear/state-space prediction, and competent conventional model-based control. Not every historical arm must be applicable to every new track; exclusions need a predeclared reason. Give comparators the same permitted information and fair tuning/compute treatment. Separate oracle upper bounds from realistic baselines.

A model result needs checkpoints and data lineage, learning curves, all declared seeds and attempts, causal-input checks, prediction/decision metrics, calibration and ablations. A planning result additionally needs changed achieved actions and improved realized outcomes after HMC filtering. Low prediction error, green CI, attractive graphics and fast inference are not substitutes.

## 6. Evidence-gated phases

No fixed completion date is asserted. These are dependency and evidence gates, not a calendar promising research success.

### Phase A — establish the foundation and diagnose

Review the existing stack and original artifacts; reproduce only authorized, available development material. Run separate model-failure, decision-opportunity and authority-path investigations. Choose a pinned foundation without erasing contributions or negative evidence.

**Exit:** exact base decision, documented reproduction limits, discriminating diagnoses or explicit unresolved hypotheses, and the new benchmark contract. A fresh rerun is labelled a rerun, not presented as the original missing receipt.

### Phase B — demonstrate a useful conventional decision

Use the current layout. Add only physical or observation mechanisms the diagnosis demonstrates are missing. Build a small causal-group-disjoint pilot population and observation-only baseline/controller with replayable useful, no-action and ambiguous cases.

**Exit:** one end-to-end permissible intervention improves the chosen outcome against the declared comparison, with control cases and checked physical semantics. This establishes opportunity, not learned-model benefit.

### Phase C — train and compare models

Build an optional accelerator-backed training path if justified. Prove target scaling, inverse transforms, tiny-set fit, gradients, checkpoint recovery and train/inference parity before a broad search. Compare temporal and hybrid candidates under shared data and budgets; graph models are conditional on the topology question.

**Exit:** reproducible model artifacts, controlled baseline comparisons and a defensible prediction/uncertainty result. A negative result is an acceptable finding and changes the next experiment.

### Phase D — demonstrate learned planning value

Start with finite actions, then bounded sequences or model-predictive planning. Compare the selected learned predictor against the same planner using conventional prediction where possible. Exercise abstention, malformed inputs, latency failures and command modifications.

**Exit:** useful closed-loop effect attributable to the learned component, or a precise null result, with constraints, worst cases, uncertainty and compute costs reported separately.

### Phase E — widen and falsify

Test new layouts, equipment, loads, sensor regimes and fault combinations. Use held-out mechanisms and structures rather than only new seeds. Challenge apparent gains with independent numerical references and counterexamples.

**Exit:** a transfer/failure map with narrower claims where needed. If a simulator quirk explains the gain, correct it and restart the affected development study.

### Phase F — independently confirm and release

Freeze a candidate and protocol; obtain specific permission for untouched confirmation with independent custody. Preserve the old BDM-v1 sealed population and its protocol boundary. A final set used to revise a candidate becomes development evidence; confirmation then needs fresh isolation.

**Exit:** a reproducible research release, including failures and limitations. General package/interface maintenance can proceed earlier; releasing a model as independently confirmed depends on the confirmation result. Tags, registry publication and physical deployment remain separate actions.

### Phase G — optional empirical validation

Seek a safe external dataset, independent reference simulator or supervised non-life-critical laboratory collaboration. Agree data rights, measurement uncertainty, safety and domain-shift questions first.

**Exit:** bounded external-validity evidence, not an inference that simulator success makes physical control safe. This phase depends on resources and domain collaboration not yet established.

## 7. Work-package catalogue

`WP01`–`WP24` are **planning identifiers, not GitHub issue numbers**. No new issues are created by this document. Each package below has an owner proposal, dependency boundary, deliverable and proof. Later packages are provisional and should be decomposed only when their inputs exist.

Dependencies require accepted artifacts or a recorded justified no-change outcome, not just a completed agent session. Optional paths are stated explicitly so a useful early slice is not blocked by every ambitious later experiment.

### WP01 — choose and preserve the next foundation

- **Lead:** Yaroslav, with Ben and Alex reviewing the choice. **Dependencies:** none.
- **Work:** inventory main, PRs #84–89, their ancestry/diffs/checks, original artifacts and contributor worktrees. Identify what is accepted, needs correction, is evidence-only, or should be deferred. Preserve attribution and old protocols.
- **Deliverable:** a foundation decision recording exact base/head SHAs, retained invariants, outstanding reviews and integration order.
- **Acceptance:** every overlapping capability has an explicit disposition; missing artifacts are named; contributors confirm ownership before edits; an independent reviewer can trace the choice to source and verification evidence.
- **Boundary:** recommend convergence; merging or taking over another branch requires separate agreement.

### WP02 — explain the BDM-v1 failure

- **Lead:** Ben. **Dependencies:** none; read/diagnose against the pinned research tip, with any later fix based on WP01.
- **Work:** trace units, feature assembly, targets, loss scaling, inverse transforms, gradients, initialization, early stopping and runtime/offline parity. Separate optimization defects from weak labels and insufficient observations.
- **Deliverable:** ranked hypotheses and minimal discriminating tests, with results or an exact reproduction blocker.
- **Acceptance:** each conclusion points to code and measured evidence; tiny-set and gradient checks distinguish numerical failure from task failure; original evidence remains unchanged.
- **Boundary:** no large architecture sweep or claim that flat labels explain all errors. Corrections become separate bounded issues with regression tests.

### WP03 — measure decision opportunity

- **Lead:** Alex. **Dependencies:** none; use pinned permitted DEV definitions.
- **Work:** compare hold and admissible actions at identical starting states and exogenous inputs. Inspect horizon, effect size, actuator achievement and information available when deciding. Include useful, neutral, harmful and ambiguous cases.
- **Deliverable:** an opportunity report and small development witness set, including a labelled privileged oracle comparison.
- **Acceptance:** proposed actions are distinguished from achieved actions; differences exceed declared tolerances; observational ambiguity is explicit; no final/hidden population is accessed.
- **Boundary:** parameter ranges need physical justification, not selection for a preferred model's advantage.

### WP04 — trace advice through HMC to actuation

- **Lead:** Yaroslav. **Dependencies:** none; use the pinned research tip, coordinating read-only diagnosis with WP03.
- **Work:** trace proposal, abstention, HMC validation, rejection/modification, issued command and physical achievement. Identify why arms collapse to hold and whether any interface mismatch is a defect.
- **Deliverable:** disposition/achievement traces and regression proposals for demonstrated mismatches.
- **Acceptance:** representative failed and successful paths are reproducible; all-rejected and all-abstaining outcomes remain distinguishable; no second plant-step authority is introduced.
- **Boundary:** do not relax safety/reserve/mode policy solely to admit model advice.

### WP05 — freeze the next success and interface contract

- **Lead:** joint; Yaroslav proposed as editor, with independent evaluation review. **Dependencies:** WP02, WP03, WP04.
- **Work:** choose the first track, endpoints/units, constraints, baseline roster, independent grouping, splits, numerical margins, sample-size rationale, budgets and stopping rules. Resolve the shared contracts in section 4.
- **Deliverable:** reviewed benchmark and interface specifications with explicit study identity and unresolved choices settled before execution dependent on them.
- **Acceptance:** useful-decision pilot evidence motivates the contract; no success criterion is selected from final candidate outcomes; hidden/future inputs and privileged controls are explicitly separated.
- **Boundary:** this does not reopen or revise BDM-v1's failed protocol.

### WP06 — reproduce the existing baseline evidence

- **Lead:** cross-lane reviewer not responsible for the run being checked. **Dependencies:** none for the pinned research source; execution and artifact-access scope must be approved.
- **Work:** inventory required corpus/checkpoints/receipts, run relevant existing checks and bounded reproductions, compare to published claims.
- **Deliverable:** source/environment/artifact-bound reproduction report including missing original material.
- **Acceptance:** real commands and exit results are retained; original versus newly generated evidence is distinguished; failures and unavailable artifacts are not replaced with inferred output.
- **Boundary:** no protected evaluation, hidden refitting cost or unapproved long experiment.

### WP07 — add only necessary physical mechanisms

- **Lead:** Alex. **Dependencies:** WP01, WP03, WP04, WP05.
- **Work:** implement the smallest justified mechanism needed for the first meaningful decision, such as a supported degradation, resource or response-delay regime. If the current plant is sufficient, record a no-change result.
- **Deliverable:** a tested physical slice or explicit proof that no plant change is necessary.
- **Acceptance:** relevant conservation/balance, bounds, units and replay checks pass; independent numerical references exercise the changed mechanism; historical behavior is preserved or explicitly versioned.
- **Boundary:** no general simulator rewrite or new realism claims from extra parameters alone.

### WP08 — support configurable layouts

- **Lead:** Alex. **Dependencies:** WP07; a reviewed topology schema is required.
- **Work:** separate zone/connection configuration from the initial layout; define valid graph, capacities and ordering without hard-coding room names into model inputs.
- **Deliverable:** multiple valid configurations and compatibility/migration documentation.
- **Acceptance:** topology/flow conservation, invalid-graph rejection, ordering/permutation behavior and legacy-layout equivalence are tested.
- **Boundary:** optional for the first fixed-layout result; not a full arbitrary-network editor or CFD engine.

### WP09 — diversify faults and observations

- **Lead:** Alex; Ben co-reviews observability. **Dependencies:** WP05, WP07.
- **Work:** introduce justified degradation, compound faults and missing/biased/delayed sensing. Include healthy lookalikes and hidden causes that are indistinguishable from available history.
- **Deliverable:** versioned scenario/sensor regimes with explicit treatment and observation timing.
- **Acceptance:** paired cases differ only in intended treatment; onset labels agree with observability; physical truth never becomes an undeclared feature; healthy and no-action controls are present.
- **Boundary:** support a small pilot subset first; broad diversity is conditional on measured coverage gaps.

### WP10 — build the training population and corpus

- **Lead:** Alex, with Ben owning model-facing serialization review. **Dependencies:** WP05, WP07, WP09; WP08 only for topology expansion.
- **Work:** extend existing family/corpus machinery, first producing a small pilot artifact, then a frozen training population after pilot review. Preserve counterfactual support and causal-group disjointness.
- **Deliverable:** manifests, generation code, shards and a coverage/leakage report with regeneration instructions.
- **Acceptance:** related variants/counterfactuals stay together; preprocessing uses TRAIN only; coverage is measured rather than cherry-picked for model success; runtime features exclude evaluator-only state; cost/throughput are recorded.
- **Boundary:** the accepted pilot artifact may unblock WP12; larger generation needs measured resource caps and population approval.

### WP11 — establish a reliable training path

- **Lead:** Ben. **Dependencies:** WP01, WP02, WP05; a reviewed small input fixture suffices before the broad WP10 corpus.
- **Work:** add an optional PyTorch or justified alternative lane without imposing GPU dependencies on the simulator core. Implement data loading, explicit transforms, checkpoint/resume and controlled CPU/GPU operation.
- **Deliverable:** a reproducible training/sanity path with checkpoint metadata.
- **Acceptance:** tiny-set fit, finite-difference/autograd comparison where applicable, forward/target-inverse parity, non-finite rejection and interrupted-run recovery are exercised; hardware/framework details and budgets are explicit.
- **Boundary:** training plumbing is not evidence of a better model; dependency/lock changes have a single owner.

### WP12 — establish serious classical comparisons

- **Lead:** Ben for estimators/predictors; Yaroslav for the controller integration. Assign one accountable owner per resulting issue. **Dependencies:** WP05, WP07, accepted WP10 pilot.
- **Work:** reuse ridge/state-space work; add a justified observer, system-identification method or physics-based predictor where needed. Build a competent observation-only controller through the same HMC path.
- **Deliverable:** tuned conventional baselines and the first useful conventional closed-loop slice.
- **Acceptance:** identical permitted information, constraints and declared tuning budgets; no-action and ambiguity controls; replay shows actual benefit or a well-explained failure; independent inspection checks baseline competence.
- **Boundary:** do not cripple conventional methods to make neural improvement easier.

### WP13 — compare temporal learned candidates

- **Lead:** Ben. **Dependencies:** WP10 frozen training population, WP11, WP12.
- **Work:** compare a small justified set of temporal representations, starting from diagnosed limitations rather than a universal architecture sweep. Train from random initialization and measure data/model/optimization scaling.
- **Deliverable:** saved checkpoints, learning curves, seed/config ledger, ablations and baseline comparisons.
- **Acceptance:** all declared attempts are accounted for; artifact identities bind code/data/transforms; prediction units and decision metrics are correct; missingness/action-history ablations test substantive dependence.
- **Boundary:** no pretrained-model or size claims without a separately declared study; a failed comparison remains a result.

### WP14 — test hybrid dynamics

- **Lead:** Ben; Alex reviews physical assumptions. **Dependencies:** WP10, WP11, WP12.
- **Work:** compare learned residuals or hidden-parameter estimation around known dynamics to pure conventional and pure learned methods. Audit what state initializes the reference model.
- **Deliverable:** a hybrid candidate and controlled ablation on the same population/interface.
- **Acceptance:** operational input parity; no privileged rollout leak; errors and decision outcomes are compared under declared budgets.
- **Boundary:** an optional competing hypothesis, not a mandatory prerequisite for every temporal-model experiment.

### WP15 — test graph-based generalization

- **Lead:** Ben; Alex owns topology semantics. **Dependencies:** WP08, WP10 topology population, WP11, WP12, WP13.
- **Work:** investigate topology-aware prediction on genuinely held-out layouts/equipment structures with matched non-graph comparisons.
- **Deliverable:** a graph-temporal candidate and structural-transfer report.
- **Acceptance:** permutation consistency, held-out structure identities, matched information/budgets and failure-by-layout reporting; no counting renamed copies as new layouts.
- **Boundary:** optional until topology transfer is a selected claim; not required for initial fixed-layout utility.

### WP16 — estimate belief and calibrate uncertainty

- **Lead:** Ben, with cross-lane calibration review. **Dependencies:** WP05, WP12, WP13; include WP14 or WP15 only if selecting those candidates.
- **Work:** represent ambiguous hidden state, calibrate intervals/disagreement on the permitted partition, and define selective advice for stale, invalid or shifted inputs.
- **Deliverable:** frozen calibration artifacts, abstention policy and coverage/risk analysis.
- **Acceptance:** ambiguity, useful-opportunity recall, harmful confidence and OOD behavior are tested; every withheld advice path is inspectable; calibration data is not used as unrestricted model-selection data.
- **Boundary:** calibrated uncertainty is empirical evidence, not a general safety guarantee or a win through universal abstention.

### WP17 — build bounded planning

- **Lead:** Yaroslav. **Dependencies:** WP04, WP05, WP12; learned-planner evaluation also requires the selected predictor and WP16.
- **Work:** begin with finite admissible actions and a conventional predictor, then test bounded sequences/MPC. Define costs, horizons, replanning, resource constraints, timeout and fallback behavior.
- **Deliverable:** planner interface and executable HMC-governed proposals with conventional and learned predictor adapters.
- **Acceptance:** search respects declared compute/latency limits, invalid advice fails closed, plan semantics match achieved actuation, and model/planner contributions can be ablated separately.
- **Boundary:** conventional scaffolding can proceed before a neural winner; no unrestricted neural actuator control.

### WP18 — measure closed-loop attribution

- **Lead:** cross-lane evaluator; Yaroslav supplies integration support, not sole verdict. **Dependencies:** WP13, WP16, WP17; selected optional candidates must also be frozen.
- **Work:** reuse/extend the comparative runner with matched exogenous traces, outcome units and proposal-to-achievement lineage. Compare conventional and learned systems under the frozen contract.
- **Deliverable:** paired group-level result, traces, all gate outcomes and failure analysis.
- **Acceptance:** admitted/modified/rejected/abstained/achieved counts are distinct; actual outcome benefit and worst-group losses are measured; improvements from deterministic guards are not credited to learning.
- **Boundary:** development evidence only; no independent confirmation claim or retroactive metric change.

### WP19 — falsify the result

- **Lead:** a reviewer outside the candidate-selection lane. **Dependencies:** WP07 for early numerical checks; WP18 for the final challenge report.
- **Work:** attack leakage, sensitivity to assumptions, simulator exploitation, weak baselines, mechanism shifts and misleading pooled improvements. Use independent references, not calls to the same production helpers.
- **Deliverable:** reproducible counterexamples or bounded evidence that named challenges were survived.
- **Acceptance:** tests can genuinely fail the candidate; mitigations are rerun on changed bytes; remaining simulator-validity limits narrow the claim.
- **Boundary:** agent agreement is not external validation; no exploratory access to the final confirmation population.

### WP20 — independently confirm a frozen candidate

- **Lead:** a designated custodian/evaluator not tuning the candidate. **Dependencies:** WP18, WP19; WP15 only for a topology-transfer claim.
- **Work:** freeze code, model, transforms, thresholds, action semantics, protocol and analysis. Obtain explicit one-shot approval for a fresh untouched population with separated custody.
- **Deliverable:** immutable confirmation results, including failures, and an independent reproduction report.
- **Acceptance:** custody/selection history is auditable; the approved candidate is exactly evaluated; all declared outcomes are reported; any adaptive reuse ends the population's untouched status.
- **Boundary:** the existing BDM-v1 blind boundary remains intact; a failed candidate is not renamed as qualified.

### WP21 — measure and improve performance

- **Lead:** Alex for simulation, Ben for inference; split into separate scoped issues. **Dependencies:** WP10 for generation; selected model artifact for inference.
- **Work:** profile throughput and memory first, then justify vectorization, multiprocessing, caching, export or quantization. Choose target hardware before target-specific promises.
- **Deliverable:** baseline/optimized artifacts and matched-workload measurements.
- **Acceptance:** simulator semantics and model quality/action choices retain declared tolerances; report actual device, p50/p99, throughput and resource cost; include unsuccessful optimizations.
- **Boundary:** no claiming an Arm/NPU result from another machine or treating speed as learned decision benefit.

### WP22 — build the research interface and replay experience

- **Lead:** Yaroslav coordinates; component ownership may be delegated. **Dependencies:** WP04, WP05 trace contract; use WP18 evidence when presenting learned outcomes.
- **Work:** expose scenario configuration, observations versus hidden truth, predictions, uncertainty, proposals, HMC dispositions and achieved outcomes. Begin with a simple trace inspector; add 3D/Blender views only when they improve inspection.
- **Deliverable:** a research interface backed by versioned real traces, with future/live/predicted views visibly distinct.
- **Acceptance:** displayed values and timing match source traces; replay is repeatable; stale/absent forecasts are not invented; the UI has no second physics or command authority.
- **Boundary:** interface progress can happen earlier, but synthetic fixtures must be labelled and cannot be shown as learned-model evidence.

### WP23 — make the platform reproducible and contributable

- **Lead:** Yaroslav coordinates; each lane supplies its install/run and artifact requirements. **Dependencies:** accepted interfaces; confirmed-model claims require WP20; optional performance/UI claims require their evidence.
- **Work:** extend existing wheel/CI checks, package model metadata, document data availability, provide minimal real-user examples and a contributor path. Add tag-triggered release verification where justified.
- **Deliverable:** installable artifacts, tested reproduction instructions, release notes and a technical report including limitations/negatives.
- **Acceptance:** clean installation outside the checkout exercises the intended user path; version/tag/artifact agree; users can obtain required artifacts under declared licenses; claims match evidence.
- **Boundary:** maintenance can proceed before confirmation. Releases, registries, signing and deployment require their own approval; do not label an unconfirmed model confirmed.

### WP24 — propose empirical validation

- **Lead:** joint, with an appropriate domain collaborator if available. **Dependencies:** mature simulation findings and explicit human direction.
- **Work:** identify a safe data/reference/lab route, intended transfer question, data rights, measurement uncertainty, supervision and costs.
- **Deliverable:** an actionable external-validation proposal; actual measurements only under separately approved scope.
- **Acceptance:** consent, data rights and safety responsibilities are explicit; observed versus simulated quantities are distinguished; conclusions remain bounded to the measured setting.
- **Boundary:** no autonomous procurement, physical control or claim that a partner/facility already exists.

## 8. First execution wave

After team agreement, create only the immediate diagnostic/foundation issues and the success-contract follow-on. These are more specific than asking each person to build an entire subsystem.

| Person | Initial focus | Starting output |
|---|---|---|
| Ben | WP02: why the model failed | numerical/feature-path hypothesis report and minimal probes |
| Alex | WP03: where useful decisions exist | admissible-action opportunity and observation-ambiguity witnesses |
| Yaroslav | WP04: why advice does not become useful actuation | proposal-to-HMC-to-achievement trace and mismatch findings |
| Joint, coordinated by Yaroslav | WP01 and bounded WP06 | foundation decision and honest reproduction/artifact inventory |
| Joint follow-on | WP05 | first track, interfaces, units, baselines, budgets and frozen acceptance |

These initial investigations can read the same immutable source without competing edits. Regression implementation starts only in an assigned file scope. Contract drafting can begin alongside diagnostics, but contract acceptance depends on their findings.

### Existing source targets

At the research tip, begin with:

- Model: `src/aeolus/habitat_v2/forecast_issue75_bdm.py`, `src/aeolus/habitat_v2/forecast_issue74_baselines.py`, `scripts/train_issue75_bdm.py`.
- World/data: `src/aeolus/habitat_v2/bdm_v1_families.py`, `src/aeolus/habitat_v2/bdm_v1_corpus.py`, `src/aeolus/habitat_v2/bdm_v1_evaluation.py`.
- Command path: `src/aeolus/habitat_v2/hmc.py`, `src/aeolus/habitat_v2/hmc_contract.py`, `src/aeolus/habitat_v2/physics.py`, `src/aeolus/habitat_v2/forecast_issue77_closed_loop.py`.
- Evidence/contract: `docs/research/2026-09-05-bdm-v1-benchmark-contract.md`, `contracts/habitat_v2_bdm_v1_tcn_preregistration_v1.json`, `contracts/habitat_v2_bdm_v1_closed_loop_preregistration_v1.json`, the #73–77 evidence cards.

Paths are starting points, not authorization to rewrite all of them. The pinned source map below remains necessary because the research files are not all on main or this documentation branch.

### Verification commands to reuse after execution approval

From the agreed research checkout with approved dependencies:

```bash
uv run --locked --python 3.11 --extra dev python -m pytest -q tests/habitat_v2/test_issue75_bdm.py tests/habitat_v2/test_issue74_baselines.py tests/habitat_v2/test_issue77_closed_loop.py
uv run --locked --python 3.11 --extra dev python -m pytest -q tests/habitat_v2/test_bdm_v1_families.py tests/habitat_v2/test_bdm_v1_corpus.py tests/habitat_v2/test_hmc_physics_boundary.py tests/habitat_v2/test_physics_reference_oracles.py
uv run --locked --python 3.11 --extra dev python -m pytest -q
uv run --locked --python 3.11 --extra dev ruff check .
uv run --locked --python 3.11 --extra dev python -m compileall -q src tests scripts
uv lock --check
git diff --check
```

These are future implementation/reproduction commands, **not runs performed to write this plan**. Existing study scripts may refit several models; inspect their cost and output paths before execution. A docs link check does not reproduce an experiment.

## 9. Agents, integration and review

### Elastic capacity, not a headcount target

With three people each operating four or five simultaneous sessions, the planning assumption is **12–15 active sessions total**, including leads and reviewers. A larger 20–30-agent pool can be staged or expanded only if actual limits and useful throughput support it. Subscription allowances are not GPU-training resources or a guarantee of sustained concurrency.

A possible normal pod is one lead, one reviewer and two or three workers. Start with a small handoff/integration pilot, then fill only independent ready tasks. Reserve review capacity. More agent writers to the same core files can reduce throughput.

Within each lane, agents may investigate, implement, run authorized experiments, review or document. They do not make final scientific/safety decisions merely because a task completed. Track model/harness/effort and meaningful AI contribution; do not assume provider labels imply identical behavior or statistically independent judgments.

### Working rules

1. GitHub is the shared project record. Local queues reference the same issue IDs rather than creating a competing status system.
2. Every issue has one accountable human, a bounded file/artifact scope, a pinned base and explicit permitted side effects.
3. Use isolated branches/worktrees. Shared contracts, HMC, physics, lockfiles and versions have one writer at a time. Read-only review can be parallel.
4. Claim work visibly and leave a last-update/recovery note. A stopped agent leaves the exact commit, remaining blocker and first resume action.
5. Each handoff includes source/base/head identities, changed files, commands/exit results, artifact hashes, failed attempts, limits and the next decision.
6. Review another worker's actual diff and decisive artifacts. One bounded review/correction cycle is the default; scientific disagreements need a discriminating experiment or human decision, not an endless reviewer chain.
7. Integrate accepted slices through one agreed sequence. Avoid stacking new studies indefinitely on moving, unreviewed foundations.
8. Collaborative PRs begin as drafts without automatic reviewer requests. Human approval governs readiness, reviewer requests, merges and publication claims.
9. Account access remains individual. Share code and sanitized artifacts, never tokens or private session contents.
10. Idle workers do not invent scope. Stop at a failed scientific gate and record the redirection.

A useful review cadence is event-driven: shared-interface acceptance, first complete slice, candidate freeze, and release/claim review. A short regular coordination meeting is optional; no meeting schedule is imposed here.

## 10. Compute, reproducibility and release

### Resource decisions before scaling

Inventory available CPU/GPU, RAM, disk, CI capacity and measured simulation throughput. Keep separate budgets for agent reasoning, simulation, training, storage/egress and human review. No specific machine, rental budget or training duration is assumed.

Each expensive run needs a cap, expected output, stop/checkpoint rules and a hypothesis. Measure pilot throughput and extrapolate in recorded calculations before broad generation. More correlated windows do not create more independent evidence.

Start with repository-native tools: Python/NumPy, uv, existing runners and CI. Select optional training, storage or experiment services only when measured needs justify them. Favor immutable manifests/shards and simple machine-readable run records before installing a distributed platform.

### Evidence and artifact contract

A run records code/config/data/transform/checkpoint identities, seeds, framework/device versions, hyperparameters, training/selection history, wall time and failures. Checkpoints have tested resume and inference paths. Training repeatability, deterministic plant replay, cross-platform numerical equivalence and cross-platform byte identity are separate claims.

Maintain TRAIN, DEV, CALIBRATION and genuinely untouched confirmation boundaries. Custody applies to people and agents as well as code paths. Candidate authors do not inspect final outcomes to decide what to submit.

Preserve old receipts, negatives and fixtures. New bytes or semantics invalidate affected old proof; rerun the relevant experiment under a new identity. Missing original weights or inaccessible corpora are explicit reproduction limits, not a reason to invent a plausible result.

### Release contract

Every code PR declares `major`, `minor`, `patch` or `none`. In parallel work, record release intent per slice and let one release owner consolidate the numeric version/changelog change. Do not have every worker edit version files.

The eventual release should include simulator/training tools, compatible model/data artifacts where licensable, clean-install instructions, a minimal real user path, comparison results, failure cases and contribution guidance. Actual artifact availability matters as much as a README command. Use existing installed-wheel CI and extend it as necessary.

Model learning, learned planning benefit, inference speed, package readiness, pushed commits, CI, release publication and deployment are separate claims. A simulation-only research release is legitimate; it must not imply physical readiness.

## 11. Turning the plan into GitHub issues

### Conversion sequence

1. Review this draft's direction, lane preferences and immediate wave together.
2. Refresh existing issues #70–77 and PRs #84–89. Link or amend existing work only with its owner; do not duplicate completed/candidate capabilities under new names.
3. Convert WP01–04 and scoped WP06 first; create WP05 as their dependent contract task. Do not create all later packages as ready work merely to occupy agents.
4. Record actual issue-number mappings back in this document or a single linked tracking issue. Keep WP IDs stable; GitHub numbers are assigned by GitHub, never guessed.
5. Once WP05 is accepted, split the next phase into implementation-sized issues with exact schemas, thresholds, commands and budgets. Later research hypotheses remain provisional until evidence supports them.

### Issue template

```markdown
## Outcome
One plain-language sentence describing what is being made or established.

## Starting point
- Programme: WPxx
- Repository/base SHA:
- Existing issue/PR relationship:
- Accountable human / implementation owner / independent reviewer:

## Behavior or experiment
1. Inputs and current behavior.
2. The bounded change or hypothesis test.
3. Expected observable result, including negative outcomes.

## Scope
- Files/interfaces/artifacts this issue may change:
- Non-goals and protected evidence:
- Allowed execution, installation and compute budget:
- Publication/merge boundary:

## Dependencies
- Actual issue links and the accepted artifact/interface each supplies.
- No unresolved schema or metric choice delegated to competing workers.

## Acceptance
- [ ] Observable behavior or experiment result demonstrated.
- [ ] Exact focused checks and regression proof where relevant.
- [ ] Data/causality/authority invariants checked.
- [ ] Decisive evidence reproduced by the named reviewer.
- [ ] Failures, limitations and version impact recorded.

## Handoff
Base/head SHA; changed files; commands and exit results; artifact identities;
failed/aborted attempts; unresolved risks; exact next action.
```

An issue is ready when its owner can execute without choosing a shared interface or success criterion on another lane's behalf. A work-package paragraph is not, by itself, permission for a worker to fill in consequential missing choices.

## 12. Decisions, risks and resume point

### Decisions for the team

- Confirm or revise Ben/models, Alex/world, Yaroslav/planning and integration; retain existing contributor ownership.
- Confirm habitat-first scope and select the first narrow recovery or efficient-operation question from diagnostic evidence.
- Choose the foundation/convergence reviewer and independent evidence custodian.
- Inventory resources and authorize a bounded first execution wave, including whether installs, diagnostic simulations and refitting are allowed.
- Agree interface editors, review/integration responsibilities and the publication path for future changes.

No numeric compute budget, final architecture, real-world partner or team commitment is fabricated to make this plan appear complete. These are explicit decisions required at the relevant gates.

### Stop and redirect

- **No meaningful permissible action:** improve the question/horizon/regime, not just the neural model.
- **Poor model numerics:** fix and verify the implementation before increasing size.
- **Unidentifiable hidden state:** improve legitimate sensing/history or represent ambiguity; do not leak truth.
- **Strong conventional method already solves the task:** retain it and investigate a genuinely harder justified boundary.
- **Benefit disappears after HMC or actuator limits:** fix the model/planner action semantics, not the authority boundary.
- **Simulator assumption reverses the result:** narrow the claim and validate that assumption independently.
- **Leakage or final-population contamination:** invalidate the affected claim and re-establish honest isolation.
- **Compute/review costs dominate useful progress:** reduce concurrency/search breadth and checkpoint rather than inventing more work.

### Exact next step after this publication

Review sections 3, 5 and 8 with Ben, Alex and Yaroslav. Resolve lane preferences and the first-wave execution scope, then create the first bounded issues using section 11. This draft publication alone does not launch agents, train models, generate a corpus, merge research PRs, open a blind population or deploy anything.

## 13. Source map and verification boundary

- [Pinned main](https://github.com/arm-hackathon/arm-hackathon/tree/25a8a9ac194472d05b13983d2f8078f3e0b67f21).
- [Pinned research source](https://github.com/arm-hackathon/arm-hackathon/tree/142af4a194ee1eba6815d94bbc92e79073c50a6d).
- [Main package metadata](https://github.com/arm-hackathon/arm-hackathon/blob/25a8a9ac194472d05b13983d2f8078f3e0b67f21/pyproject.toml).
- [Main CI definition](https://github.com/arm-hackathon/arm-hackathon/blob/25a8a9ac194472d05b13983d2f8078f3e0b67f21/.github/workflows/ci.yml).
- [Existing physics provenance](https://github.com/arm-hackathon/arm-hackathon/blob/25a8a9ac194472d05b13983d2f8078f3e0b67f21/src/aeolus/habitat_v2/physics_provenance.py).
- [Original BDM-v1 contract](https://github.com/arm-hackathon/arm-hackathon/blob/25a8a9ac194472d05b13983d2f8078f3e0b67f21/docs/research/2026-09-05-bdm-v1-benchmark-contract.md).
- [Issue #73 attribution card](https://github.com/arm-hackathon/arm-hackathon/blob/142af4a194ee1eba6815d94bbc92e79073c50a6d/docs/evidence/issue-73-guard-attribution-card.md).
- [Issue #74 baseline card](https://github.com/arm-hackathon/arm-hackathon/blob/142af4a194ee1eba6815d94bbc92e79073c50a6d/docs/evidence/issue-74-linear-baselines-card.md).
- [Issue #75 model card](https://github.com/arm-hackathon/arm-hackathon/blob/142af4a194ee1eba6815d94bbc92e79073c50a6d/docs/evidence/issue-75-bdm-v1-card.md).
- [Issue #76 calibration card](https://github.com/arm-hackathon/arm-hackathon/blob/142af4a194ee1eba6815d94bbc92e79073c50a6d/docs/evidence/issue-76-calibration-card.md).
- [Issue #77 closed-loop card](https://github.com/arm-hackathon/arm-hackathon/blob/142af4a194ee1eba6815d94bbc92e79073c50a6d/docs/evidence/issue-77-closed-loop-card.md).
- [Earlier roadmap snapshot](https://github.com/arm-hackathon/arm-hackathon/blob/61341d3fa9fcdd17524d19dedac38c350666f244/ROADMAP.md), preserved in Git history; its BDM-v1-first sequence is historical, not an instruction to repeat completed experiments.

This document is an AI-assisted planning synthesis for human review. Its preparation inspected live refs, issue/PR state, source paths, package/CI definitions and published evidence. Documentation checks do not independently reproduce the simulator, training runs or numerical results. Source refs and observed PR states are a dated snapshot; refresh them before dispatching implementation.
