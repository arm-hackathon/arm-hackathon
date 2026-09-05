# BDM-v1 Benchmark And Evidence Contract — Design Record

Date: 2026-09-05
Issue: [#70](https://github.com/arm-hackathon/arm-hackathon/issues/70)
Status: **ACCEPTED RESEARCH CONTRACT — freezes evaluation terms before any BDM-v1 implementation or training**
Machine-readable contract: [`contracts/habitat_v2_bdm_v1_benchmark_contract_v1.json`](../../contracts/habitat_v2_bdm_v1_benchmark_contract_v1.json)
Enforcement module: `src/aeolus/habitat_v2/bdm_v1_benchmark_contract.py`

## Purpose

This record and its companion JSON contract freeze the benchmark, evidence,
and custody terms for the first Belief-Dynamics Model (BDM-v1) study line
before any model code or training exists. Every downstream issue (#71–#77)
conforms to this contract rather than silently redefining it. Consistent with
repository policy, no protected blind data is opened and no model is trained
in this change.

## Primary hypothesis and non-claims

Hypothesis (frozen in the contract): a compact causal belief-dynamics model
consuming only runtime-observable causal history plus an explicit
candidate-action encoding can improve HMC-filtered closed-loop decision
outcomes over both the frozen deterministic O2-excess guard (`c8`) and an
action-conditioned linear baseline, on fresh mechanism-held-out development
families, with calibrated selective abstention — scored on physical safety
outcomes first at the independent causal-group level.

Non-claims: no digital-twin/CFD/flight claims; no qualification,
certification, deployment, hardware, or real-world-safety claims; no
learned-superiority claim before Gates 0/2/3/4 pass; no Arm/ONNX/quantisation
claim before Gate 6; no blind-population access before its separate one-shot
authorization.

## Evidence status matrix

The contract carries a four-category reconciliation of repository evidence:

- **implemented** — deterministic world, HMC authority, observability
  qualification, Issue #52 long-horizon lane, Issue #53 dropout lane
  (QUALIFIED forecast lane; one earlier sealed version remains immutably
  NOT QUALIFIED), Issue #54 distillation (development evidence only),
  Issue #55 race and Issue #56 V3 frozen baseline, the concluded Issue #56
  V4 line (protocol revision v10: per-family superiority, development
  evidence only), the FP32 Arm benchmark receipt, and the judge demo.
- **historical** — the closed-loop advisory campaign (78/24/0) retained with
  its documented custody limits, and the deterministic recovery line.
- **proposed** — issues #71–#77 and the future blind-confirmation issue.
- **not_claimed** — the ledger's unsupported-claim list plus qualification,
  deployment, physical-validity, and extended Arm claims.

## Causal input schema

Nine field classes are frozen with per-field `dtype`, `shape`, `unit`,
`timing`, `missingness`, `observability`, and `provenance` rules: zone
environment history; observation masks and staleness (unmarked
zero-imputation prohibited); command and actuator feedback; declared resource
gauges; operating context; retained HMC dispositions; topology and
configuration descriptors; candidate-action encoding; and explicitly declared
known-future schedules (empty unless a scenario declares them).

The causal window is at most 16 completed plant steps at or before the
decision step. Nine prohibited input classes are frozen verbatim (hidden
physical truth, fault labels/schedules, seeds, internal noise/bias state,
future measurements, counterfactual outcomes, evaluator-only reserve/audit
state, future HMC arbitration results, undeclared future loads). The
enforcement API rejects any study feature manifest containing a prohibited or
undeclared field before fitting.

## Actions, horizons, labels

The action catalogue is the four frozen `normal-*` commands plus abstention,
bound to the forecast-contract catalogue and immutable within a study.
Horizons are the existing short/medium/long keys (4, 16, 32) plus the
episode-remaining convention. Labels use the 51-target projection vocabulary
with the five decision targets (crossing event, safety exposure, maximum
crossing, comfort deviation, resource composite) and action-minus-hold
deltas. Counterfactual rollouts are label material only — checkpointed
causal state, identical prior observations and disturbances, hold plus every
admissible catalogue action — and are never runtime inputs.

## Comparison roster

Seven frozen arms: HMC/rules only, hold, `c8_o2_excess_guard`,
`c9_o2_guard_statistical`, action-agnostic ridge, action-conditioned linear
(ridge or controlled state-space, choice frozen inside #74), and BDM-v1. All
learned and hybrid arms are advisory behind HMC. Issue #73's ablation arms
(guard-only, learned-screen-without-guard) are referenced so Gate 0
attribution remains visible in every verdict.

## Metrics and statistics

Metric polarity is frozen per metric; the decision-metric hierarchy is
lexicographic: zero hard-admissibility violations, per-family hard-safety
non-inferiority, paired aggregate safety benefit (or equal safety with a
predeclared resource benefit), then action-value/regret errors, beneficial
precision/recall, resource depletion, comfort, and intervention cost.
Aggregate forecast error is explicitly secondary. The independent statistical
unit is the causal group; paired sensor variants, counterfactual action
branches, and within-family decision steps are declared non-independent.
Arms are compared on identical exogenous traces; bootstraps resample causal
groups with per-study preregistered seeds.

## Split custody

Four group-disjoint partitions with exclusive uses: TRAIN (fitting/scaling),
DEV (design decisions), CALIBRATION (intervals and abstention thresholds
only), BLIND_FINAL (one preregistered run after candidate freeze). Grouping
keys: causal scenario template, physical parameter band, fault
mechanism/composition, operating schedule, action opportunity, and
sensor-failure bundle. The blind manifest is sealed by hash without outcomes;
its size is set by a pilot power analysis materially exceeding the three
independent evaluation condition groups of the V4 line.

## Thresholds

The model-path latency ceiling (p99 ≤ 250 ms) is the only frozen numeric
threshold, carried over from the Issue #56 protocols. Every decision
threshold is recorded as `TBD_FROM_PILOT`: each must be frozen in a study
preregistration derived from DEV-partition pilot evidence before any
closed-loop verdict and before any blind access. No result-derived value may
ever be presented as frozen.

## Stop criteria

The contract freezes nine stop-or-redirect criteria verbatim from the
research roadmap (failure to beat the linear baseline on action ranking;
improvement vanishing without the deterministic guard; dependence on a single
dormant/O2 pattern; miscalibration under family shift; success via near-total
abstention; reversal under one plausible simulator assumption; leakage or
family overlap; any HMC authority or replay inconsistency; blind failure).
The required response is to improve the question, corpus, or simulator —
never to tune against a failed blind population.

## Enforcement API

The contract module ships executable checks that downstream issues consume:
`validate_model_input_fields` (prohibited/undeclared fields),
`validate_causal_window` (future steps and window bounds),
`validate_group_disjointness` (partition overlap and duplicated groups),
`validate_targets_declared` and `validate_metrics_declared` (undeclared
target/metric drift), and `threshold_is_frozen`. Tests exercise every
rejection path plus drift mutations of the contract itself (polarity flips,
horizon edits, custody weakening, threshold tampering, roster changes,
field-record degradation, duplicate JSON keys).

## Issue acceptance checklist mapping

- Design record + machine-readable contract committed together: this file and
  the JSON contract land in one commit.
- Per-field type/shape/unit/timing/missingness/observability/provenance
  rules: `input_schema.field_classes` plus validator enforcement and the
  field-record tests.
- Tests reject prohibited inputs, non-causal windows, group overlap, and
  undeclared target/metric changes: enforcement tests in
  `tests/habitat_v2/test_bdm_v1_benchmark_contract.py`.
- Thresholds preregistered or marked `TBD_FROM_PILOT`: thresholds section and
  `test_thresholds_are_preregistered_or_tbd`.
- Condition-group/family units, no inflated independence: metrics section and
  `test_statistical_unit_is_causal_group`.
- No protected data opened, no model trained: this change adds a contract,
  a validator, tests, and documentation only.

## What this does not prove

Accepting this contract proves only that the evaluation terms are frozen and
machine-checkable. It proves nothing about BDM-v1 quality; every claim still
requires the downstream gates, and negative or null results will be published
unchanged. HMC remains the sole final-command, plant-step, and replay
authority in every study governed by this contract.
