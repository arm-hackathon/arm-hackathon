# AEOLUS

**A**irflow and **E**nvironmental **O**bservation **L**aboratory for
**U**ser-defined **S**cenarios

Version `0.8.0` adds a closed V5 operational-observability qualification layer on
the actuator-feedback world: explicit fixture identities, a SHA-256-bound ordered
feature manifest, matched treatment-pair/report identity bindings, finite
completed-row decision windows, provenance-bound trace/pair matching, separate
abnormality/localisation/exact-identification answers, explicit treatment IDs,
temporally complete clearance/recovery fixtures, hard-negative checks, and
denominator-explicit aggregate metrics. See
[`docs/evidence/habitat-v2-operational-observability-qualification.md`](docs/evidence/habitat-v2-operational-observability-qualification.md)
for the bounded contract, tracked packet, and qualification receipt. Rebuild and
verify that packet from a source checkout with:

```bash
uv run --locked --python 3.11 --extra dev python \
  scripts/build_habitat_v2_observability_packet.py \
  --source-root . \
  --output out/habitat-v2-observability-qualification-packet.json \
  --expected-sha256 \
  1afed658237fd62404094eac2d50a78b8db9ad19f9b612add9ff37d1b0e3866b
```

Version `0.7.0` added the
closed scenario-v5 actuator-feedback layer on
the corrected scenario-v3 multizone air network. It provides fan degradation,
branch-resistance increase, damper jam, sensor bias/drift and stuck-sensor
profiles, plus redundant primary/secondary observations and evaluator-only
truth receipts. A checked-in eight-zone compound-fault scenario demonstrates
all five mechanisms. Scenario-v1 through scenario-v3 remain separate frozen
contracts. This is not a published or hardware-qualified release. The `0.2.0`
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

The accepted architecture retains deterministic actuator authority. A separate
frozen `DEMO_ONLY_PERMANENTLY_EXCLUDED` FP32 advisory candidate is SHA-256-bound
to its FP64 source and reduces raw model-array bytes exactly 50%, from
`28,759,024` to `14,379,512`. Three native `ubuntu-24.04-arm` repetitions on a
Neoverse-N2 runner passed the predeclared `1e-4` prediction-parity threshold and
recorded a median-of-run speedup of about `1.73x`; see authoritative exact-head
[run 31941351824](https://github.com/arm-hackathon/arm-hackathon/actions/runs/31941351824).
This is bounded benchmark evidence, not an INT8, Arm-specific kernel/operator,
NEON, deployment, physical-board, NPU, energy, thermal, qualification,
production, learned-control, or actuator-authority claim.

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

## Judge demo: one reproducible local command

**New here? Take the guided tour instead** — an interactive walkthrough that
explains the project in plain language and lets you run the pieces yourself:

```bash
uv run --locked --python 3.11 --extra dev python scripts/aeolus_tour.py
```

The tour offers: a live run of the trained forecaster (you pick the action),
a step-by-step replay of the recorded paired experiment, independent receipt
verification, and a plain-English explanation of the architecture.

From a clean source checkout with [uv](https://docs.astral.sh/uv/) available:

```bash
uv run --locked --python 3.11 --extra dev python scripts/run_habitat_v2_forecast_judge_demo.py
```

The command creates a new ignored receipt directory, runs the local simulated
forecast, and independently verifies a fresh deterministic HMC replay before it
prints the `file:` URL for the self-contained report. It never overwrites an
older receipt. The report is advisory-only: it does not execute model inference
in the browser, does not control hardware, and does not qualify or validate a
model. The deterministic HMC remains the sole command authority.

For a separate browser-local fixture explorer, open
[`demo/browser-simulator/index.html`](demo/browser-simulator/index.html) locally.
It operates offline, exposes four fixed scenarios, makes no hardware/controller
claim, and every row has `actionAuthority: "none"`.

## Judge demo: the trained closed-loop MLP (development evidence)

The repository also carries the action-aware MLP trained on the Historical V2
pilot archive (training run `full-v1-20260818-a`; 23,400 simulator examples;
held-out normalized MAE 0.1146 on 17 unseen scenario clusters). From a source
checkout:

```bash
uv run --locked --python 3.11 --extra dev python scripts/run_habitat_v2_mlp_forecast_demo.py
```

The command forecasts all four catalogue actions at step 16 with the trained
model, lets HMC execute one operator-selected action, and prints each
candidate's forecast error against the realized simulator truth alongside the
trace and replay identities. The model is pure NumPy at inference (no torch
install needed) and advisory-only; deterministic HMC remains the sole command
authority. The full closed-loop paired campaign (238 runs; 78 better / 24
equal / 0 worse on pre-registered safety-exceedance metrics) is documented in
merged PR
[#40](https://github.com/arm-hackathon/arm-hackathon/pull/40). This is
development evidence only — not qualification, not deployment.

## What the experiment found

Model documentation and assurance:

- [`MODEL_CARD.md`](MODEL_CARD.md) — identity, training, held-out metrics,
  limitations, and failure modes for `action_aware_mlp_v1`.
- [`CORPUS_DATASHEET.md`](CORPUS_DATASHEET.md) — composition, collection,
  split discipline, and known biases of the training archive.
- [`SAFETY_CASE.md`](SAFETY_CASE.md) — the claims–argument–evidence case
  for the learned advisory path, including its residual risks.
- `scripts/check_habitat_v2_mlp_drift.py` — scores live telemetry against
  the training distribution and flags drift (diagnostic only).

Headline results from the paired closed-loop campaign (each scenario run
twice with identical scenario, noise, and reset — once with canonical HMC
alone, once with the model advising):

- **The demo scenario:** canonical HMC drifted past the CO2 warning threshold
  and stayed there for 29 steps (integrated exceedance 19.94); the
  model-advised arm never crossed it (0.0). The adviser acted at step 37 of
  72 — before the first violation at step 43.
- **Across 102 fault pairs:** 78 safer, 24 equal, 0 worse; 72 advised runs
  finished with zero threshold exceedance.
- **The authority boundary is exercised, not just asserted:** HMC overrode
  81 of 793 model proposals — the model advises, it never commands.
- **Broken sensors:** the adviser verifiably abstains whenever any telemetry
  is missing and HMC continues alone (merged PR
  [#41](https://github.com/arm-hackathon/arm-hackathon/pull/41)).
  Forecasting *with* missing sensors is future work, not claimed here.
- **The honest cost:** advised runs consumed more resources (median
  +757 Wh battery, +1.97 mol oxygen, +6.04 mol sorbent) — the safety margin
  is bought with consumables, and all runs stayed above resource floors.
- **Reproducibility:** every run replays bit-for-bit; the same commands
  produce the same numbers and trace hashes on any machine.

## Source-checkout verification

```bash
uv sync --locked --python 3.11 --extra dev
uv run --locked --python 3.11 --extra dev python -m pytest -q
uvx ruff@0.14.10 check .
uv run --locked --python 3.11 --extra dev python -m compileall -q src tests
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

Run the checked-in scenario-v3 multizone air-network example:

```bash
uv run --locked --python 3.11 --extra dev python -m aeolus.habitat_v2 \
  scenarios/habitat_v2_air_network.json out/habitat-v2-air-network.jsonl
```

Scenario-v3 replaces direct per-zone airflow commands with a fan-speed command
and one damper command per declared zone. The deterministic solver derives a
single fan/system operating point, per-zone volumetric flow in `m³/s`, fixed-
reference-density mass flow in `kg/s`, pressure losses in Pa, and fan power in
Trace-v3 records commanded and achieved actuator positions plus an explicit
network receipt. The validator recomputes the canonical transition from the
parsed scenario and exact pre-step plant state, cross-checks fan electrical
power against the electrical bus receipt, and then replays the full scenario
byte-for-byte.

The checked-in eight-zone habitat and its dimensions, resistances, schedules,
and loads are declared research assumptions for deterministic software testing.
They are not a NASA floor plan, calibrated CFD, a certified digital twin, or
evidence about flight hardware.

Run the checked-in scenario-v4 compound-fault example:

```bash
uv run --locked --python 3.11 --extra dev python -m aeolus.habitat_v2 \
  scenarios/habitat_v2_compound_faults.json \
  out/habitat-v2-compound-faults.jsonl
```

Scenario-v4 keeps physical truth separate from operational observations.
`telemetry` is the primary observed feed. `sensor_disagreement` contains the
secondary feed and signed primary-minus-secondary residuals. The evaluator-only
`fault_receipt` contains physical truth, sensor residuals and deterministically
ordered active-fault effects. These truth fields are not a future model-input
contract. Sensor faults never alter plant state, and no learned component owns
actuator or recovery authority.

The V2 command validates the strict scenario schema, executes the deterministic
plant, validates every trace row against the parsed scenario and refuses to
overwrite an existing output file.

Run the checked-in scenario-v5 actuator-feedback example:

```bash
uv run --locked --python 3.11 --extra dev python -m aeolus.habitat_v2 \
  scenarios/habitat_v2_actuator_feedback.json \
  out/habitat-v2-actuator-feedback.jsonl
```

Scenario-v5 is a new closed lineage. It retains V4's eight-zone topology,
physical fault receipts and independent primary/secondary telemetry, then adds
rate-limited achieved cooling and oxygen state, effectiveness faults, and
stateless deterministic operational feedback. Its trace lineage carries the
separate `aeolus_habitat_v2_actuator_feedback_v1` identity. V5 commands and
traces are causally replayed; V1–V4 contracts are preserved unchanged.

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
