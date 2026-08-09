# Temporal early-risk predictor final evaluation contract

Status: **frozen before final-suite generation**

Date: 2026-08-09

Version impact for a future code PR: **minor**

## Decision question

Does the frozen early-risk model generalise to new scenario families strongly enough to remain an active candidate behind the deterministic recovery governor?

This is a one-shot predictor-final evaluation. It is not another development or calibration pass.

## Frozen candidate

The evaluation uses:

- the tracked `models/early-risk-softmax-v1-candidate.json` artifact, pinned by bytes SHA-256 `2f88fac553f3dba6abd3c6f0a4793aa921fbeeb8682b4de740eca88a490b5139`, internal artifact identity `6eeaf089e8ddb07ce6e5304841b48e1d5fa3e5325c0e5f3b7852193d04740063`, and model-payload identity `77910da137fbb51ffe4faa995ff837edd8999474a95fffde664fc444d056701c`,
- its existing probability and margin thresholds,
- the existing ten-tick `model_input_v1` window,
- the existing deterministic advisory-acceptance policy,
- the existing deterministic recovery and handback state machine,
- the repository evaluator committed before final-suite generation.

No retraining, recalibration, threshold change, source change, family exclusion or metric change is allowed after the final suite is generated.

## Final suite

`scenarios/sweep-early-risk-final.json`, pinned by canonical SHA-256 `d70da5bcad631b2d29b8f801e6679ffefad6bdeb4dc0bb647efc67a3892d7077`, declares one `suite_role: final` split containing exactly 144 families:

- two unseen seeds,
- two unseen operating profiles,
- two unseen fault-start ticks,
- three targets,
- two persistent gradual profiles,
- one persistent blocked profile,
- one transient blocked profile,
- one transient gradual profile,
- one frozen-sensor family per base condition.

Every final family must be canonically disjoint from the two exact pinned forbidden manifests:

1. all temporal early-risk train and validation families, and
2. all previously opened deterministic-recovery final families.

The input selector and topology hashes must remain identical.

## Evaluation arms

Every final fault family is run through:

1. reserve off,
2. deterministic governor alone,
3. frozen predictor advisory plus deterministic governor.

Every unique healthy reference is run through governor alone and predictor advisory plus governor.

The learned model remains advisory-only. The deterministic governor remains the sole actuator authority.

## Metric polarity

- Model warning: advisory output only. It is not an intervention.
- Accepted advisory observation: deterministic physical gates accepted the warning as evidence.
- Harmful-fault intervention: desired when physical reserve-off CO2 excess is positive.
- Healthy-reference intervention: undesired. Zero is good.
- Frozen-sensor intervention: undesired. Zero is good.
- Wrong-target intervention: undesired. Zero is good.
- Invariant violation: undesired. Zero is good.
- Repeated protection, handback recurrence, timeout or non-zero transient final reserve: undesired. Zero is good.
- Earlier protection on a harmful gradual fault: desired.
- Positive reduction in integrated physical CO2 excess versus governor alone: desired.
- Negative reduction means worsening and is forbidden.

## Predeclared admission checks

The evaluator must fail closed before execution unless:

- the source worktree is clean,
- the manifest contains only `final` families,
- there are exactly 144 final families with the frozen fault-class composition and four references,
- the final manifest is disjoint from the exact distinct development and deterministic-final manifests pinned in the evaluator,
- the artifact self-hash and live model-payload binding validate,
- the evaluator deterministically regenerates the pinned sweep in a temporary directory and byte-compares the complete generated file tree, including every scenario, manifest and generation receipt,
- every source, evaluator, artifact, sweep and manifest hash is recorded,
- the output path does not already exist,
- the final corpus has no existing consumption lock.

Immediately before the first simulation, the evaluator exclusively creates a corpus-level consumption lock bound to the source commit, source hashes, sweep, manifest, artifact and output path. The lock is never overwritten or removed. A started final evaluation consumes the suite even if execution later fails, so changing the output filename cannot be used to rerun or tune against the same final families.

## Predeclared safety gate

All conditions must hold:

1. Healthy-reference interventions = **0**.
2. Frozen-sensor interventions = **0**.
3. Wrong-target interventions = **0**.
4. Governor invariant violations = **0**.
5. Missed harmful physical families = **0**.
6. Harmful physical families worsened versus governor alone = **0**.
7. Transient families with more than one protection entry = **0**.
8. Transient handback recurrences = **0**.
9. Transient handback timeouts = **0**.
10. Transient families ending without physical reserve zero = **0**.

## Predeclared benefit gate

There must be at least eight harmful persistent gradual families. All conditions must then hold:

1. At least **40%** of harmful gradual families receive earlier protection than governor alone.
2. At least **25%** of harmful gradual families have strictly lower integrated physical CO2 excess than governor alone.
3. Median fractional excess reduction among harmful gradual families where governor-alone excess is positive is at least **10%**.
4. No harmful physical family is worsened.

Counts use ceilings for fractional family thresholds. For example, ten harmful gradual families require at least four earlier protections and three positive physical reductions.

## Final verdict

- **PASS:** safety gate and benefit gate both pass. Keep the model as a final-evaluated candidate. This still does not establish universal safety or Arm performance.
- **REJECT_SAFETY:** any safety condition fails. Reject the model from the active path.
- **REJECT_BENEFIT:** safety passes but benefit fails. Preserve the artifact and receipt as research history, but remove it from the active integration path.

The receipt is authoritative even when the verdict is rejection. A failed result must not be tuned and rerun on the same final families.

## Explicit non-goals

This evaluation does not:

- change deterministic recovery settings,
- grant learned actuator authority,
- retrain or recalibrate the predictor,
- use the final families as future development data,
- export, quantise or benchmark on Arm,
- claim universal safety or production readiness.
