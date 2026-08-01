# Problem assessment: AEOLUS against the Arm review brief

## Bottom line

The schema-v9 experiment is a strong engineering milestone, but AEOLUS is not
ready to submit against the Arm brief unchanged.

The temporal model losing to the rule baseline is moderately problematic and
recoverable. The more serious gaps are that the project still has no bounded
simulated response, no INT8 artifact, no declared Arm target, and no measured
Arm-specific optimization.

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
| Make the project easy to run | One-command experiment and documented reproduction | Strong |
| Compare honestly with a baseline | Softmax, temporal MLP, calibrated rules, IID and OOD evidence | Strong |
| AI identifies faults | MLP loses overall and produces too many false alarms | Weak |
| Diagnosis leads to simulated action | Governor and redundant response remain absent | Missing |
| Perform a deliberate Arm optimization | Only an FP32 ONNX baseline exists | Missing |
| Match runtime to a declared Arm target | No target-device benchmark exists | Missing |
| Compare FP32 and INT8 | No INT8 artifact or quality/performance comparison exists | Missing |

The negative model result is not disqualifying by itself. The reviewers
explicitly said they are not judging this as “the best benchmark percentage
wins.” It becomes a serious product problem when combined with the original
claim that AI will identify faults and trigger actions.

## What the new experiment fixed

The earlier benchmark strongly favoured deterministic rules: the rule baseline
scored approximately `0.987` macro-F1 because faults had near-exact simulator
signatures. Schema v9 introduces downstream sensor noise, bias and drift,
subtle fault families, IID family-held-out testing, and separate OOD stress.

On the new IID benchmark:

| Evidence | Temporal MLP | Calibrated rules |
|---|---:|---:|
| Macro-F1 | 0.5765 | 0.6410 |
| Nominal false-alarm rate | 35.36% | 2.53% |
| Median causal latency | 9 ticks | 11 ticks |

Rules are no longer effectively perfect. That confirms that the realism work
removed the original shortcuts and produced a materially harder benchmark.
The model is faster, but its `18.2%` median latency improvement misses the
fixed `20%` threshold and is accompanied by an unacceptable false-alarm
regression. The recorded `ai_advantage_demonstrated` result is therefore
correctly `false`.

OOD stress is worse for both approaches and still favours the rule baseline:

| Evidence | Temporal MLP | Calibrated rules |
|---|---:|---:|
| Macro-F1 | 0.3085 | 0.4386 |
| Nominal false-alarm rate | 66.38% | 0.51% |

## Where the model adds value

The per-class evidence reveals a complementary signal rather than a wholly
failed model:

| IID class | MLP recall | Rule recall |
|---|---:|---:|
| Gradual degradation | 59.4% | 58.2% |
| Blocked path | 69.5% | 88.0% |
| Frozen sensor | 75.0% | 2.6% |

The realistic readout model defeats the old exact-constant frozen-sensor rule,
while the temporal model retains substantial frozen-sensor recall. This is a
credible AI contribution.

However, frozen-sensor precision is only `24.1%`, and the overall false-alarm
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

Keep schema v9, sweep v2, the IID test, the stress partition, and the negative
evidence as the development baseline. Do not weaken rules or retune using the
current test result.

Because the current IID test has now been inspected, subsequent model changes
must use new predeclared, unseen final-test families. Reusing the current test
as final evidence would undermine the held-out claim.

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

Add deterministic response logic that:

- acts only on sufficiently persistent, valid evidence;
- keeps commands within declared limits;
- hands control back when inference or telemetry is invalid;
- records the reason for every decision;
- compares identical fault runs with response enabled and disabled;
- reports airflow recovery and CO2 outcome changes.

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
