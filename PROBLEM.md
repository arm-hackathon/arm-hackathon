# Problem assessment: AEOLUS against the Arm review brief

## Bottom line

The schema-v9 / protocol-v3 experiment is a strong engineering milestone, but
AEOLUS is not ready to submit against the Arm brief unchanged.

The final temporal model losing to the rule baseline is a measured limitation,
not a reason to reselect against the final suite. The more serious gaps are that
the project still has no INT8 artifact, no declared Arm target, and no measured
Arm-specific optimization. A bounded simulated response now exists as
development-stage, evidence-backed design on `ben/bounded-response` (see
`docs/bounded-response.md`), but it is not yet final-suite evidence.

This assessment was made against:

- `AEOLUS_ARM_Project_Review_Transcript_and_Technical_Notes.pdf`
- `AEOLUS_ARM_Project_Review_Raw_Timestamped_Captions.pdf`

The raw captions corroborate the cleaned review notes: simulation is
acceptable, judges will inspect the code, the largest benchmark percentage is
not automatically the winner, and standard optimization can score highly when
it is real, appropriate, measured, documented, and reproducible. The reviewers
also repeatedly emphasize that merely running an unchanged model on Arm is not
enough.

## Assessment against reviewer expectations

| Reviewer expectation | Current position | Assessment |
|---|---|---|
| Simulation is acceptable | Deterministic schema-v9 simulator with imperfect measurements | Strong |
| Judges can inspect a real implementation | Strict schemas, corpus integrity, tests, artifacts | Strong |
| Make the project easy to run | Documented staged, hash-bound reproduction | Strong |
| Compare honestly with a baseline | Softmax, temporal MLP and calibrated rules; development-only selection and frozen final evaluation | Strong |
| AI identifies faults | MLP loses overall and produces too many false alarms on final families | Weak |
| Diagnosis leads to simulated action | Bounded governor with parity evidence on 129 development families; redundant fan absent | Partial |
| Perform a deliberate Arm optimization | Only an FP32 ONNX baseline exists | Missing |
| Match runtime to a declared Arm target | No target-device benchmark exists | Missing |
| Compare FP32 and INT8 | No INT8 artifact or quality/performance comparison exists | Missing |

The negative model result is not disqualifying by itself. The reviewers
explicitly said they are not judging this as “the best benchmark percentage
wins.” It becomes a serious product problem when combined with the original
claim that AI will identify faults and trigger actions.

## What the current experiment fixed

Earlier benchmarks favoured deterministic rules because faults had near-exact
simulator signatures. Schema v9 introduces downstream sensor noise, bias and
drift, subtle fault families, and controller-facing imperfect CO2 sensing.
Protocol v3 then separates development selection from final evaluation by
family, rather than treating an inspected v2 test partition as reusable.

On the 180-family final suite:

| Evidence | Temporal MLP | Calibrated rules |
|---|---:|---:|
| Macro-F1 | 0.5754744477098027 | 0.642588422763726 |
| Nominal false-alarm rate | 38.5698% | 0.5631% |
| Median detection latency | 9 ticks | 10 ticks |

Rules are no longer effectively perfect, which shows that the measurement model
removed the original shortcut. The model's 11.1% median-latency reduction misses
the fixed 20% threshold and comes with a severe false-alarm regression. The
recorded `ai_advantage_demonstrated` result is correctly `false`.

The previous v2 IID/stress numbers remain historical development context only.
They are not comparable with, and must not be presented as, protocol-v3 final
or OOD evidence.

## Where the model adds value

The per-class evidence reveals a complementary signal rather than a wholly
failed model:

| Final class | MLP recall | Rule recall |
|---|---:|---:|
| Gradual degradation | 61.79% | 50.29% |
| Blocked path | 72.41% | 92.78% |
| Frozen sensor | 73.89% | 4.44% |

The realistic readout model defeats the old exact-constant frozen-sensor rule,
while the temporal model retains substantial frozen-sensor recall. This is a
credible AI contribution.

However, frozen-sensor precision is only `20.72%`, and the overall false-alarm
rate is too high for the model to trigger ventilation actions safely. The
current model should therefore remain experimental rather than being described
as the preferred fault detector.

## Largest conflict with the Arm brief

The reviewers identify INT8 as the obvious optimization baseline and accept
hardware-aware runtime/backend work as a legitimate alternative. They want to
see what changed for the target hardware and want measured evidence such as:

- model size;
- inference latency;
- throughput;
- peak memory;
- detection-quality change;
- exact device, CPU features, runtime and backend.

AEOLUS currently has a reproducible FP32 ONNX model. That is a suitable
baseline, not yet an Arm optimization. There is no evidence that the model was
quantized, that an Arm-aware backend was selected, or that any result was
measured on a declared Arm target.

This missing optimization evidence is more damaging to brief alignment than
the rule baseline winning.

The original proposal also states that diagnosis causes actions during the
simulation. The current experiment stops at prediction. The review notes say
simulation can prove orchestration logic and untreated-versus-controlled
outcomes, but those results do not exist yet.

## Recommended recovery path

### 1. Freeze the current result

Keep schema v9, the protocol-v3 development suite, the frozen final suite and
the negative final evidence as the baseline. Do not weaken rules or retune using
the final report.

Because the current final suite has now been inspected, subsequent model changes
must use a new predeclared development protocol and a separate unseen final
suite. Reusing the current final families as new final evidence would undermine
the held-out claim.

### 2. Use AI as a bounded complement

The highest-evidence direction is a hybrid detector:

- retain rules for blockage and degradation;
- use a compact learned detector for noisy frozen-sensor evidence;
- calibrate confidence and require persistence;
- permit abstention rather than forcing a fault class;
- retain deterministic rules as the safety guardrail.

This uses the model where the measured evidence shows value instead of trying
to force it to replace stronger rules.

### 3. Optimize for operational constraints

The class-balanced loss and validation macro-F1 selection reward fault recall
but tolerate excessive nominal alarms. Future selection should predeclare a
false-alarm constraint and fault-recall requirements, then optimize quality or
latency within those constraints.

Candidate techniques include confidence calibration, persistent decisions,
class-specific thresholds, explicit abstention, and a smaller frozen-sensor
specialist. These choices must be made using training and validation evidence,
not the inspected test partition.

### 4. Implement the bounded response

Landed in development evidence (`docs/bounded-response.md`): a deterministic,
causal governor that emits bounded commands with structured rationale. It
- acts only on sufficiently persistent, valid evidence;
- keeps commands within declared limits and never under-drives a hot zone;
- records the reason for every command;
- is compared on identical fault runs with response enabled and disabled;
- reports time above ceiling, response latency, energy and invariant
  violations on a 129-family response sweep.

Still to demonstrate as final-suite evidence: a redundant-fan model, and a
fault run where healthy capacity visibly recovers airflow (the governor's
parity result shows the single-loop failure is delivery-bound, so the recovery
claim requires the redundancy model).

### 5. Complete the Arm optimization evidence

Choose one primary Arm target before choosing the final runtime. A Raspberry
Pi 5 or similar Cortex-A edge device matches the reviewers’ interpretation of
the application more naturally than an unexplained cloud-only target.

For the final compact model:

1. establish the accepted FP32 baseline;
2. create a post-training INT8 model;
3. measure FP32 versus INT8 quality, size, latency, throughput and peak memory;
4. compare ONNX Runtime with an Arm-aware alternative only where the final
   operators and target support it;
5. document the exact device, CPU, OS, runtime, backend and commands;
6. attempt quantization-aware training only if post-training INT8 loses too
   much quality.

## Recommended submission narrative

The strongest defensible narrative is:

> Deterministic rules handle obvious mechanical faults, while a compact
> temporal AI detector recovers frozen-sensor visibility lost under realistic
> measurement noise. A bounded governor acts only when combined evidence is
> safe, and the final learned component is quantized and measured on a declared
> Arm target.

That narrative preserves the honest negative result, gives AI a demonstrated
role, keeps deterministic safety logic in control, and directly addresses the
reviewers’ requirements for real optimization, measurements, documentation,
and reproducibility.

## Current severity

- As an engineering and benchmark milestone: strong.
- As evidence that a standalone AI detector should replace rules: weak.
- As a completed response to the Arm optimization brief: incomplete.
- If submitted unchanged: high risk.
- If used as the baseline for a hybrid detector, bounded response and measured
  Arm optimization: recoverable and technically credible.
