# Temporal early-risk predictor development result

Status: **development candidate retained, final claim blocked**

Date: 2026-08-09

Version impact for a future code PR: **minor**

## Decision

Retain the compact temporal early-risk predictor as an opt-in development candidate behind the deterministic recovery supervisor.

Do not enable it by default, evaluate it on the opened deterministic final suite, export it to Arm, claim hardware readiness, or grant it direct actuator authority.

The result demonstrates bounded closed-loop value on the fresh development validation split while preserving the tested safety boundary. It does not establish untouched-final generalisation. Benefit was concentrated in two cabin-B gradual-degradation families, and cabin-A physical benefit remains unproven.

## Lineage and scope

The work was branched from local recovery-candidate tip `89ff124` in isolated worktree `ben/temporal-early-risk-v1`.

Yaro's `yarofix2` branch and PR #18 are ancestors of this work. That contribution established the governed response path, command validation, scenario runner and response evidence mechanics used here. The later recovery candidate added the blind-verified deterministic policy and was therefore the correct successor base. Restarting from the old `yarofix2` tip would have discarded later hardening.

No source change was made on Yaro's branch.

## Frozen development contract

The first-candidate contract is recorded in [`temporal-early-risk-development-contract.md`](temporal-early-risk-development-contract.md).

- Input: exactly ten completed `model_input_v1 float32[24]` telemetry ticks.
- Positive label: a persistent gradual-degradation fault will make exactly one crew cabin cross the declared physical CO2 ceiling within the next twelve ticks.
- Output classes: `no_early_risk`, `risk:cabin_a`, `risk:cabin_b`.
- Healthy references, abrupt faults, transient faults and frozen sensors are negative lookalikes.
- Lab faults are outside the learned target set and remain under deterministic recovery only.
- Hidden fault type, target, schedule, seed and future physical state are forbidden model inputs.
- The predictor emits warnings only. It never emits reserve commands.

## Fresh corpus

A new version-4 development sweep was generated from [`scenarios/sweep-early-risk-development.json`](../../scenarios/sweep-early-risk-development.json).

- Fresh families: **216**
- Train families: **144**
- Validation families: **72**
- Causal stride-1 windows: **33,925**
- Training rows:
  - `no_early_risk`: **21,947**
  - `risk:cabin_a`: **47**
  - `risk:cabin_b`: **94**
- Validation rows:
  - `no_early_risk`: **11,684**
  - `risk:cabin_a`: **12**
  - `risk:cabin_b`: **141**

Corpus provenance:

- source family manifest SHA-256: `961aa0dc1ae0bc2fe97f4405181195da7e7bc6fa8345303c2b56f8d58e74646c`
- corpus manifest SHA-256: `fe09a5cc6fe8c94b9887c509b5be514c03467f045df5e489109925d424fa1dc8`
- corpus bytes SHA-256: `5cfaa01c922e5364d7af72a640e5acf4135f6f8ecdd9c12b888cfc0929d028b3`
- forbidden opened-final manifest SHA-256: `26b46fa0f78e70cc1a2449ef3c84215c39f6afa3a00616ff52b3ca2fb471685c`

Canonical identity disjointness passed before corpus admission. The opened deterministic final families and traces were not used for training, calibration or predictor-plus-governor evaluation.

## Model and calibration

The baseline is deterministic class-balanced softmax regression over the existing 135-value temporal summary. It is intentionally smaller and easier to inspect than a GRU or transformer.

Tracked candidate:

- path: [`models/early-risk-softmax-v1-candidate.json`](../../models/early-risk-softmax-v1-candidate.json)
- artifact internal SHA-256: `6eeaf089f70f7da97e82983921b1ad20956fa67c4d7fda2001576209efb40063`
- artifact bytes SHA-256: `2f88fac553f3dba6abd3c6f0a4793aa921fbeeb8682b4de740eca88a490b5139`
- best epoch: **180/200**
- validation macro-F1 before abstention: **0.4295693685**
- probability threshold: **0.5666666667**
- top-two margin threshold: **0.1666666667**

Calibrated validation-window behaviour:

- minimum target recall: **50.0%**
- cabin-A recall: **50.0%**
- cabin-B recall: **73.76%**
- healthy-reference warning fraction: **4.55%**
- negative-fault warning fraction: **4.36%**
- model-level wrong-target warnings: **8**

These are advisory-warning metrics. They are not intervention metrics. The deterministic physical gate remains responsible for rejecting unsupported warnings.

Two complete training and calibration runs from the same corpus produced byte-identical artifacts with identical internal and bytes hashes.

## Deterministic acceptance boundary

An advisory can affect recovery only when all of these conditions hold:

1. artifact, selector and topology hashes match;
2. run identity, authority epoch, completed tick and sequence match;
3. target is `cabin_a` or `cabin_b`;
4. probability and margin meet the artifact's frozen thresholds;
5. current measured airflow shortfall for that target is at least 4%;
6. the same zone is the unique physical shortfall leader by the existing five-percentage-point isolation margin;
7. accepted evidence persists for the existing two-tick entry requirement.

The predictor cannot bypass actuator bounds, reserve delivery checks, slew limits, minimum dwell, recovery clear, handback or physical-zero acknowledgement.

With no predictor supplied, `run_recovery_scenario` instantiates no advisory settings and preserves the deterministic governor path.

## Held-out development closed-loop comparison

The tracked predictor was compared on all **72 validation families** using:

1. reserve off;
2. deterministic governor alone;
3. predictor advisory plus deterministic governor.

The validation split was held out from weight training but was used for abstention calibration. It is therefore development evidence, not untouched final evidence.

Metric polarity:

- A model warning is advisory only and cannot command reserve flow.
- An accepted advisory observation means the deterministic physical gate accepted that warning as supporting evidence.
- A healthy or wrong-target intervention is undesired. Zero is good.
- Positive excess reduction versus governor alone is improvement. Negative is worsening.

Results:

- validation families: **72**
- physically harmful families: **22**
- harmful persistent-gradual families: **5**
- fault-arm model warning windows: **726**
- healthy-reference model warning windows: **19**
- fault-arm observations accepted by the physical gate: **17**
- healthy-reference observations accepted by the physical gate: **0**
- healthy-reference interventions: **0**
- frozen-sensor interventions: **0**
- wrong-target interventions: **0**
- invariant violations: **0**

Closed-loop value:

- harmful gradual families protected earlier: **3/5**
- median lead when earlier: **17 ticks**
- harmful gradual families with positive excess reduction versus governor alone: **2/5**
- harmful gradual families worsened: **0/5**
- median excess reduction versus governor alone among applicable harmful gradual cases: **34.66%**

The two measurable gains were cabin-B persistent gradual-degradation families:

- 51 ticks earlier with **96.99%** less integrated physical CO2 excess;
- 17 ticks earlier with **69.32%** less integrated physical CO2 excess.

One cabin-A family entered four ticks earlier, but both governor variants already held integrated excess at zero. This improves timing but is not measurable physical benefit.

## Lifecycle checks

Across **24 transient families**:

- advisory-triggered protection episodes: **3**
- repeated protection episodes: **0**
- handback recurrences: **0**
- handback timeouts: **0**
- non-zero final reserve states: **0**

Persistent faults correctly remain in protection at the end of the finite run, so `final_physical_zero=false` is not a defect for those cases.

## Verification completed so far

- early-risk unit tests: **7 passed**
- recovery boundary tests: **37 passed**
- model-input, early-risk, recovery and recovery-scenario focused set: **82 passed**
- tracked-artifact recovery-scenario smoke: **9 passed**
- complete isolated Python 3.11 repository suite after the final
  artifact-binding correction: **487 passed in 125.04 seconds**
- real training reproducibility: byte-identical
- fresh validation closed-loop comparison: completed
- adversarial artifact-substitution regression: altered live weights with the
  original claimed artifact identity are rejected before simulation
- bounded independent review: timed out after producing the artifact-binding
  finding above; the finding was corrected, targeted regression-tested and
  followed by the complete repository suite. The reviewer did not return a
  formal post-correction verdict, and no second review swarm was launched
- default no-predictor trace parity against `89ff124`: byte-identical for a
  healthy governed run and a blocked-fault governed run

Default-path parity hashes:

- healthy governed trace: `9b584312de06eb5dbfbc1807acc762469515b62e32dc211aea15fb673f9ca59b`
- blocked-fault governed trace: `6fd5eec87317e9dc0a7b3b1a05fff62b9e6ce7fcedcbc292a06dd2a06f423433`

The development comparison receipt is stored outside the repository at:

`C:/Users/Nxiss/state/aeolus-research/early-risk-development-v1/closed-loop-comparison-v3.json`

Receipt SHA-256: `796bcad64ac17d1df8bb988e9011d0cced0b6cf82db3e47b327587eebaf34369`

## Limitations and next gate

This first candidate is evidence of a useful learned warning path, not evidence of a qualified safety model.

- Validation positives are sparse and asymmetric, especially cabin A.
- Benefit is concentrated in cabin B.
- Warning volume is high relative to accepted physical evidence.
- The validation split participated in threshold calibration.
- No untouched predictor final corpus has been generated or opened.
- No Arm export, quantisation, latency measurement or hardware benchmark has been attempted.

Before any final claim, freeze the source, artifact and acceptance policy, generate a separate untouched predictor final corpus, predeclare intervention-safety and physical-benefit gates, and run it once. Arm optimisation remains blocked until that result passes.