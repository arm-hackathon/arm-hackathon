# AEOLUS

**A**irflow and **E**nvironmental **O**bservation **L**aboratory for
**U**ser-defined **S**cenarios

Version `0.4.0` adds a versioned Habitat Plant V2 operating-mode context while
preserving the `0.3.0` deterministic foundation and closed recovery-development
evidence. It is not a published or hardware-qualified release. The `0.2.0`
C4/C11 artifacts remain source-pinned historical evidence.

AEOLUS contains a legacy deterministic simulator in abstract units and a
separate Habitat Plant V2 grey-box research analogue with explicit SI
accounting. Neither is a spacecraft, life-support, building-control or
safety-critical system, and neither must control physical equipment.

## Current status: deterministic recovery passes blind final verification

The repaired deterministic recovery policy was frozen at source commit
`d1d39d04d5c2bb2c8a7d7c32eb2a77faa518df26` and evaluated once against the
untouched version-4 final suite.

- **252** final scenario families produced **1,008** four-arm traces.
- All **79/79** harmful physical airflow families entered protection.
- Healthy activations, frozen-sensor activations, wrong-zone actions, repeated
  protection episodes, handback recurrences, handback timeouts, and invariant
  violations were all **zero**.
- All **72/72** transient families handed back with acknowledged physical zero.
- Median integrated physical CO2-excess reduction was **80.396%**; **72/79**
  eligible families improved and seven were unchanged. None were worsened.
- The safety and physical-benefit gates both passed.

The accepted architecture retains deterministic actuator authority. This result
does not justify a learned recovery controller, and Arm model export or
optimisation remains blocked until a bounded learned component demonstrates
reproducible closed-loop value on separate untouched evidence.

See
[`docs/evidence/recovery-final-verification-result.md`](docs/evidence/recovery-final-verification-result.md)
for the receipt, hashes, limitations, and next-step gate. The earlier negative C4
result remains preserved as historical development evidence in
[`docs/recovery-protocol-acceptance.md`](docs/recovery-protocol-acceptance.md).

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
archived documentation. It is not the basis of the deterministic recovery
result, and no historical model is qualified to control the reserve path. The
final suite was used only for the frozen deterministic policy's one-time
verification, not for model training, selection, or tuning.

## Source-checkout verification

```bash
uv sync --locked --python 3.11 --extra dev
uv run --locked --python 3.11 --extra dev python -m pytest -q
uv run --locked --python 3.11 --extra dev ruff check .
python -m compileall -q src tests
```

Run the checked-in Habitat Plant V2 reference scenario from a source checkout:

```bash
uv run --locked --python 3.11 --extra dev python -m aeolus.habitat_v2 \
  scenarios/habitat_v2_reference.json out/habitat-v2-reference.jsonl
```

Run the checked-in scenario-v2 operating-mode example:

```bash
uv run --locked --python 3.11 --extra dev python -m aeolus.habitat_v2 \
  scenarios/habitat_v2_operating_modes.json out/habitat-v2-modes.jsonl
```

Scenario-v2 requires every timeline segment to declare exactly one of
`occupied`, `eva_transition`, `contingency`, or `dormant`. Trace-v2 records the
mode applied to each completed interval. The initial row records `null` because
no interval has produced it yet. Modes are context only: they do not select
loads, commands, capacities, thresholds, or physics. The checked-in example
intentionally holds physical inputs constant across all four labels so this
boundary remains visible.

The V2 command validates the strict scenario schema, executes the deterministic
plant, validates every trace row against the parsed scenario and refuses to
overwrite an existing output file.

Generate a standard deterministic legacy-plant replay from a source checkout:

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
python -I -m aeolus.habitat_v2 \
  /absolute/path/to/habitat-v2-scenario.json \
  /absolute/path/to/habitat-v2-trace.jsonl
```

Both commands require an explicit scenario file and refuse to overwrite an
existing trace. The checked-in reference scenarios are source examples, not
hidden package fixtures.
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
