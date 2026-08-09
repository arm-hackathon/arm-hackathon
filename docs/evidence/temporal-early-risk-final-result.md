# Temporal early-risk final evaluation result

## Decision

**Final verdict: `REJECT_SAFETY`.**

The opt-in temporal early-risk candidate is **not admitted**. Do not promote it,
merge it into a release path, enable it by default, export it to Arm, quantise it,
benchmark it, or optimise it. Preserve the candidate code, artifact, corpus, and
receipt as negative/limited research evidence.

This verdict does not reopen or weaken the separate deterministic-recovery final
result. The deterministic governor remains the only safety and actuator
authority.

## Frozen evaluation

The one-shot evaluation ran from clean source commit
`f6b7742c026b7580dd6794147360c50afbff5a40` after the final contract,
evaluator, tests, sweep, thresholds, candidate identities, and forbidden-manifest
identities were committed.

Before the run:

- the complete Python 3.11 suite passed: **513 tests**;
- the final evaluator's focused suite passed: **26 tests**;
- the final sweep canonical SHA-256 was
  `d70da5bcad631b2d29b8f801e6679ffefad6bdeb4dc0bb647efc67a3892d7077`;
- the candidate artifact bytes SHA-256 was
  `2f88fac553f3dba6abd3c6f0a4793aa921fbeeb8682b4de740eca88a490b5139`;
- the generated final manifest canonical SHA-256 was
  `5510419fcc00325bc2011f7936aa228563b638d558ea23e40b5d14d9e1fef7ed`;
- the complete generated corpus contained **150 files** and matched a fresh
  deterministic regeneration byte for byte, with payload-tree SHA-256
  `ce76ea285d3a11f98504ff16a1089e31cbda6dd5d807c0f3bc8a0e49883b1b6c`;
- all **144** families were final-only and canonically disjoint from the exact
  pinned temporal-predictor development corpus and deterministic-recovery final
  corpus; and
- neither a prior result nor a consumption lock existed.

The evaluator claimed the corpus before the first simulation and ran the frozen
three-arm comparison once. The lock now exists. This corpus must not be replayed,
relabelled as untouched, or used for tuning.

## Gate result

### Safety gate: failed

Desired intervention means a physically justified reserve-protection action for
the correct target. A false intervention means physical reserve activation in a
healthy reference, frozen-sensor case, or wrong target. Zero is good for all
false-intervention and violation metrics below.

Observed counts:

- healthy-reference physical interventions: **0**;
- frozen-sensor physical interventions: **0**;
- wrong-target physical interventions: **0**;
- invariant violations: **0**;
- missed harmful physical families: **0**;
- worsened harmful physical families: **0**;
- repeated transient protection episodes: **0**;
- transient handback recurrences: **0**;
- transient handback timeouts: **0**;
- transient families ending without acknowledged physical reserve zero: **9**.

The last count had a frozen requirement of zero, so the combined candidate failed
the safety/lifecycle gate.

All nine failures were late-start `t79` transient-gradual families. The immutable
receipt records **zero accepted advisory observations** in all nine, equal
predictor-plus-governor and governor-alone first-protection ticks, equal
protection-entry counts, and equal integrated physical excess. This is evidence
that the model did not change those recorded outcomes. However, the receipt did
not preserve the governor-alone final-physical-zero field, so differential
lifecycle attribution cannot be proved after the fact. The honest conclusion is
therefore not "the model caused nine failures". It is "the combined candidate
failed the predeclared final lifecycle gate, and the receipt cannot formally
attribute the failures between arms." No replay is allowed to fill that evidence
gap.

### Benefit gate: failed on sample sufficiency

Only families with positive reserve-off integrated physical CO2 excess count as
harmful gradual evidence. Positive reduction is good. Negative reduction is
worsening, and the evaluator treats any value below zero as worsening without an
epsilon loophole.

Observed harmful-gradual evidence:

- eligible harmful-gradual families: **7**, below the frozen minimum of **8**;
- earlier protection: **4/7** (**57.14%**), above the 40% fraction threshold;
- positive physical-excess reduction: **3/7** (**42.86%**), above the 25%
  fraction threshold;
- median fractional physical-excess reduction: **13.35%**, above the 10%
  threshold;
- worsened harmful physical families: **0**.

The observed proportions and median were positive, but the evidence set did not
meet the predeclared minimum eligible-family count. It is therefore insufficient
for admission.

## Advisory integrity

The model emitted **66** warning windows on healthy references and **2,472** on
fault arms. The deterministic governor accepted **0** healthy-reference advisory
observations and **32** fault-arm advisory observations. Warnings are model-level
signals, not physical interventions. The zero accepted healthy advisories and
zero healthy physical interventions confirm that deterministic authority remained
in force.

## Receipt

Machine-readable evidence:

`C:/Users/Nxiss/state/aeolus-research/early-risk-final-v1/early-risk-final-evidence.json`

Verified identities:

- canonical report self-hash:
  `4f4312ba8f6962e9fd6576f9bb4691e7c88e4ef266b1aa379053ebeb38e60c63`;
- receipt-file bytes SHA-256:
  `dc7fcc3d4691281216b56c9e883fb7d6febb65103e833b9ebde387b8f8e9e722`;
- consumption-lock bytes SHA-256:
  `fb9810e7fcb99f105c10c55b2780bda6b583314d7d9e90415125c98e5ceb316a`.

The report self-hash was recomputed independently and matched. The receipt records
144 family rows, four healthy references, the clean frozen Git head, all source
hashes, both forbidden-manifest proofs, the complete corpus-tree hash, thresholds,
per-family metrics, diagnostics, and the final `REJECT_SAFETY` assessment.

## Disposition

1. Keep the deterministic recovery policy and its separate passing evidence.
2. Do not ship or actively retain this predictor candidate in a release path.
3. Preserve this branch's candidate and final evidence for audit and learning.
4. Do not recalibrate against, rerun, or train on this consumed final corpus.
5. Do not begin Arm export, quantisation, benchmarking, or optimisation.
6. Any future predictor must be a new version and a new study. Before generating
   new untouched families, that study would need to resolve the baseline
   transient-lifecycle evidence gap, record final-zero status for both comparison
   arms, and predeclare enough physically harmful gradual cases. This result is
   not permission to tune the rejected candidate against the final failures.

Version impact: **none**. This is research evidence and documentation, not a
shipped interface or release.
