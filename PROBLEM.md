# Problem assessment: AEOLUS recovery closeout

## Bottom line

AEOLUS now has a deterministic schema-v10 recovery implementation and a
reproducible four-arm development harness, but the C4 gate result is negative.
It must not be represented as a working recovery controller, an AI-qualified
system, a physical simulation result, or an Arm-optimised submission.

The C4 run at `74154956d64309f067ada7593e2ef8786d140b4e` generated 756
families and 3,024 traces. Its evidence receipt is
`1cbb9d428824f57c500b4a1ac3859b4ea6ef0a0dd4e70012b2e6c35d230a1730`.
Duplicate and clean-checkout reproductions matched it byte-for-byte. The
negative result is reproducible, not inconclusive.

## Blocking evidence

| Gate | Observed result | Status | Why it blocks the stronger claim |
|---|---|---|---|
| Transient handback acknowledgement | The required physical-zero acknowledgement was not established within the frozen bound for transient families. | Failed | Command zero is not proof that reserve position and delivered flow reached zero. |
| Physical reserve delivery for benefit | Required non-zero governed reserve delivery was absent across the eligible defined benefit families. | Failed | Improved aggregate CO₂/delivery submetrics cannot demonstrate recovery without the claimed physical mechanism. |
| Safety aggregate | Other safety invariants passed, including no reserve-off delivery and no failed-reserve rearm. | Failed overall | A required false subgate makes the aggregate false. |
| Benefit aggregate | Median improvement, improvement fraction, delivery non-regression, and healthy non-regression subgates passed. | Failed overall | A required false physical-delivery subgate makes the aggregate false. |

The most important trap is treating aggregate improvement as recovery. The
protocol requires a mechanism-to-outcome chain: reserve command, observed
physical reserve delivery, and predeclared target-zone benefit. C4 does not
close that chain.

## What improved the engineering baseline

- The recovery topology models primary and reserve paths separately in a
  schema-v10 simulation.
- The authority consumes completed-tick observable input, validates ownership
  and topology, and records its state transitions in a strict trace schema.
- Four exact counterfactual arms isolate healthy/fault and reserve-off/governed
  effects for each family.
- The evidence runner records input/source hashes, rejects reused output paths,
  and supports byte-identical relocated reproduction.
- Eleven stress paths passed, including malformed authority, reserve delivery
  failure, recurrence, dropout, ambiguity, saturation, noise/drift, and
  denominator handling.

These are defensible implementation and reproducibility improvements. They are
not proof of accepted recovery efficacy.

## Explicitly not established

- an accepted deterministic recovery layer;
- any new trained adviser, model selection, ONNX artifact, or AI advantage;
- a final-suite result or final-suite validation;
- INT8 quantisation, Arm64 performance, energy performance, cloud execution,
  hardware-in-the-loop testing, deployment, or physical control.

Historical protocol-v3 model material remains archived background only. C4/C6
neither ran nor inspected final-suite data, and it does not promote a historical
model into a recovery authority.

## Required next gate

No automatic retry is authorised. A future recovery effort requires a new
predeclared protocol and a human scope decision. It must retain the C4 failed
result, state how it will measure physical-zero acknowledgement and delivered
reserve airflow, and use new evidence before claiming a changed conclusion.

Until that happens, the correct submission posture is: deterministic simulation
with a reproducible negative recovery-development result and no hardware or AI
qualification claim.
