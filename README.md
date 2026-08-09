# AEOLUS

**A**irflow and **E**nvironmental **O**bservation **L**aboratory for
**U**ser-defined **S**cenarios

Version `0.2.3` is an integration-only local package of the closed
recovery-development work. It is not a published or hardware-qualified release;
the `0.2.0` C4/C11 artifacts remain source-pinned historical evidence.

AEOLUS is a deterministic habitat simulation in abstract CO₂ and airflow units.
It is not a spacecraft, life-support, building-control, or safety-critical
system, and it must not control physical equipment.

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

## Rejected research candidate: advisory temporal early risk

The repository preserves an opt-in compact temporal predictor that estimates
whether `cabin_a` or `cabin_b` will cross the declared physical CO2 ceiling
within twelve ticks. It uses ten completed `model_input_v1` telemetry ticks and
cannot issue actuator commands. It is research evidence only and is not admitted
to a release path.

The candidate was frozen and evaluated once on 144 untouched, final-only
families. It produced zero healthy, frozen-sensor, or wrong-target physical
interventions, zero invariant violations, and no worsened harmful physical
families. However, nine transient families ended without acknowledged physical
reserve zero, violating the predeclared lifecycle gate. Only seven families
qualified as harmful-gradual physical evidence, below the frozen minimum of
eight. The binding verdict is **`REJECT_SAFETY`**.

The receipt does not show that accepted model advice caused the nine lifecycle
failures: all nine recorded zero accepted advisory observations and matched the
governor-alone arm on the stored timing, entry-count, and excess metrics. It did
not preserve governor-alone final-zero status, so formal differential
attribution is unavailable and replay is forbidden. This limitation does not
change the rejection verdict.

The deterministic governor remains the only safety and actuator authority.
Omitting the predictor keeps its separately verified path unchanged. No Arm
export, quantisation, benchmarking, optimisation, calibration, or final-corpus
reuse is permitted.

See the final
[`docs/evidence/temporal-early-risk-final-result.md`](docs/evidence/temporal-early-risk-final-result.md),
the frozen
[`docs/evidence/temporal-early-risk-final-contract.md`](docs/evidence/temporal-early-risk-final-contract.md),
and the earlier development-only
[`docs/evidence/temporal-early-risk-development-result.md`](docs/evidence/temporal-early-risk-development-result.md).

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
- an opt-in, hash-bound temporal early-risk advisory path whose warnings remain
  behind deterministic physical evidence and lifecycle gates; and
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
