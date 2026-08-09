# AEOLUS

**A**irflow and **E**nvironmental **O**bservation **L**aboratory for
**U**ser-defined **S**cenarios

Version `0.2.2` is an integration-only local package of the closed
recovery-development work. It is not a published or hardware-qualified release;
the `0.2.0` C4/C11 artifacts remain source-pinned historical evidence.

AEOLUS is a deterministic habitat simulation in abstract CO₂ and airflow units.
It is not a spacecraft, life-support, building-control, or safety-critical
system, and it must not control physical equipment.

## Current status: Outcome B — reproducible negative development result

The C4 recovery-development gate was run from immutable source commit
`74154956d64309f067ada7593e2ef8786d140b4e` on branch
`ben/independent-recovery`.

- The development sweep generated **756** independent scenario families and
  **3,024** four-arm recovery traces.
- The canonical receipt self-hash is
  `1cbb9d428824f57c500b4a1ac3859b4ea6ef0a0dd4e70012b2e6c35d230a1730`.
- A duplicate run compared 3,801 files and all 3,024 traces byte-for-byte;
  a clean-checkout reproduction agreed with the same evidence identity.
- Eleven targeted stress/falsification tests passed. They exercise delivery
  failure, recurrence, observation dropout, ambiguity, malformed authority,
  saturation, high noise/drift, and zero-denominator paths.
- The recovery safety gate is **false** because transient scenarios did not
  establish physical-zero acknowledgement inside the frozen handback bound.
- The benefit gate is **false** because the physical-reserve-delivery criterion
  failed for eligible, defined validation families. Other aggregate submetrics
  do not override that required criterion.

The gate is therefore a completed negative result. The deterministic recovery
implementation is not accepted as demonstrated recovery, and C5 is closed:
no new adviser corpus, training, tuning, ONNX export, integration, or final-suite
operation is authorised by this result.

See [`docs/recovery-protocol-acceptance.md`](docs/recovery-protocol-acceptance.md)
for the frozen contract, receipts, and exact scope boundary.

## Implemented simulation boundary

The repository currently provides:

- deterministic, seeded JSONL replay of a validated abstract habitat;
- schema-v9 standard scenarios and schema-v10 recovery scenarios;
- a topology-bound `model_input_v1 float32[24]` projection that excludes fault
  truth, schedules, health, seeds, and simulator-only state;
- a v10 simulated primary/reserve topology with one paired reserve path per
  non-processing zone;
- a deterministic authority state machine (`NOMINAL`, `DEGRADED`, `PROTECT`,
  `HANDBACK`) that owns the reserve command channel only while active;
- strict, write-once recovery traces that retain the legacy plant projection
  and add separate reserve and authority telemetry; and
- a four-arm development evidence runner:
  `reference_reserve_off`, `reference_governed`, `fault_reserve_off`, and
  `fault_governed`.

These are software and simulation contracts. They are not evidence of physical
recovery, a qualified model, AI advantage, hardware performance, deployment
readiness, or Arm optimisation.

## Historical model work

The repository retains historical protocol-v3 model and FP32 ONNX code and its
archived documentation. It is not the basis of this recovery closeout. No
final-suite data was run, inspected, or used for C4/C6 decisions, and no
historical model is qualified to control the reserve path.

## Source-checkout verification

```bash
uv sync --locked --python 3.11 --extra dev
uv run --locked --python 3.11 --extra dev python -m pytest -q
uv run --locked --python 3.11 --extra dev ruff check .
python -m compileall -q src tests
```

Generate a standard deterministic plant replay from a source checkout:

```bash
uv run --locked --python 3.11 python -m aeolus \
  scenarios/standard_habitat.json out/standard.jsonl
```

The command writes a new file only; generated traces belong under ignored
`out/` paths.

## Installed-package use

The installed module accepts a scenario path and an output path; it does not
require `PYTHONPATH=src`:

```bash
python -I -m aeolus /absolute/path/to/scenario.json /absolute/path/to/trace.jsonl
```

Scenario JSON remains an explicit input rather than a hidden packaged fixture.
For the deterministic recovery API, use the schema-v10
`scenarios/recovery_habitat.json` input as shown in the recovery acceptance
record.

## Frozen recovery evidence command

The following is the C4 development command. It must run from a clean detached
worktree at the exact C4 source commit and write to a new output directory. It
is a reproduction command, not permission to reopen C5 or tune the failed
gates.

```bash
git worktree add --detach /absolute/c4-source \
  74154956d64309f067ada7593e2ef8786d140b4e
cd /absolute/c4-source
test "$(git rev-parse HEAD)" = \
  74154956d64309f067ada7593e2ef8786d140b4e
test -z "$(git status --porcelain)"
uv run --locked --python 3.11 --extra dev python -m aeolus.recovery_evidence \
  scenarios/sweep-recovery-development.json /absolute/new-output-directory
```

It produces a large, ignored development corpus. Do not replace historical
outputs, use final-suite inputs, or interpret a deterministic rerun as a passed
safety or benefit gate. Later runner versions bind `uv.lock`, runtime package
versions, and pre/post source provenance, and compare the complete relocated
output trees. Those later checks improve future reproduction tooling; they do
not rewrite the frozen C4 receipt.

## Project boundaries

No C4/C6 claim is made about INT8 quantisation, Arm64 benchmarks, cloud
provisioning, hardware-in-the-loop testing, physical deployment, real-world
CO₂ limits, production control, or a final result. No push, merge, deploy,
cloud action, or final-suite operation is part of this closeout.

See the [simulation rules](docs/simulation-rules.md),
[telemetry contract](docs/telemetry-contract.md),
[recovery acceptance record](docs/recovery-protocol-acceptance.md), and
[project plan](PLAN.md).
