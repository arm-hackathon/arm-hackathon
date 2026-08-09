# Project AEOLUS plan

## Current status

**Outcome B — reproducible negative recovery-development result.**

C4 is frozen at source commit
`74154956d64309f067ada7593e2ef8786d140b4e`. The four-arm development runner
completed deterministically, but two required gates are negative:

| Required gate | Outcome | Consequence |
|---|---|---|
| Transient handback physical-zero acknowledgement | Failed | Recovery authority is not accepted as a safe completed handback mechanism. |
| Physical reserve delivery for benefit | Failed | The simulation does not demonstrate physical recovery benefit. |

A duplicate A/B run and a clean-checkout reproduction matched the C4 receipt
and every trace. Reproducibility establishes the result; it does not repair the
failed gates.

C5 is closed. No new adviser training, tuning, export, model integration, or
final-suite operation is authorised by the C4 result.

## Project goal — not yet met

The intended demonstration is:

```text
observable primary-path concern
→ deterministic authority validates identity, topology, persistence and bounds
→ simulated independent reserve path is commanded
→ delivered airflow and environmental outcome improve over the exact reserve-off arm
→ authority returns reserve-off only after physical-zero acknowledgement
```

A learned component, if a later protocol independently qualifies one, may advise
only. It must never own reserve commands or write plant state.

This goal is not established by the current source or evidence. The C4 negative
result prevents describing the project as a working recovery controller or an
AI-driven recovery system.

## What is implemented and verified in source

- Deterministic, seeded abstract habitat simulation with strict scenario and
  trace validation.
- Schema-v9 standard plant scenarios and schema-v10 recovery topology.
- Independent primary and reserve directed path pairs per non-processing zone
  in the recovery schema.
- Observable-only `model_input_v1` projection; hidden fault truth, schedules,
  health, seeds, and internal noise state remain outside model features.
- A deterministic reserve authority with bounded transitions, causal
  completed-tick observations, command ownership checks, failure latching, and
  write-once trace output.
- A four-arm recovery development runner with source, sweep, settings, trace,
  and evidence hashes.
- C4 canonical execution: 756 families, 3,024 traces, receipt
  `1cbb9d428824f57c500b4a1ac3859b4ea6ef0a0dd4e70012b2e6c35d230a1730`.
- Eleven targeted stress/falsification tests and byte-identical duplicate and
  clean-checkout reproductions.

## What the evidence does not establish

- accepted safe recovery or successful handback;
- measured physical reserve delivery or recovery benefit;
- a qualified adviser, AI advantage, model integration, or ONNX export for this
  recovery protocol;
- any final-suite result;
- INT8 quantisation, Arm64 performance, hardware validation, deployment, or
  real environmental control.

Historical model experiments remain historical. They must not be reused to
claim a C4-qualified recovery adviser.

## Frozen boundaries

1. The C4 safety and benefit predicates are frozen. Green submetrics cannot
   override a false required predicate.
2. The C4 development corpus is not a licence to tune thresholds, architecture,
   or physical assumptions after inspecting its result.
3. Final-suite inputs remain outside this closeout and require a separate human
   gate.
4. Generated traces, corpora, package artifacts, and closeout receipts remain
   under ignored `out/` paths.
5. The simulation uses abstract units only. No hardware or real-safety claim is
   valid without separately scoped evidence.

## Next decision

The safe next step is a human decision on whether to open a new, explicitly
predeclared recovery protocol. That protocol would need to diagnose the two C4
blockers before it runs: transient physical-zero acknowledgement and actual
reserve delivery in eligible benefit families. It must use new evidence rather
than relabel this negative run as acceptance.

Until then, the project is a deterministic simulation and a documented negative
recovery-development result.
