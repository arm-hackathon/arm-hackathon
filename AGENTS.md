# Contributor and Agent Guide

## Purpose and safety boundary

AEOLUS is a deterministic environmental-simulation research repository. It contains:

- a legacy abstract habitat simulator with seeded JSONL replay;
- Habitat Plant V2, a separate grey-box research analogue with explicit SI accounting;
- versioned scenario, telemetry, trace, and forecast-contract tooling; and
- bounded, offline demonstrations and historical evidence artifacts.

It is **not** a spacecraft, life-support, building-control, hardware-control, safety-critical, or production-control system. Do not represent simulation output as hardware validation, qualification, certification, deployment readiness, or physical performance. No code path in this repository should operate real equipment.

The deterministic Habitat Management Controller (HMC) is the sole actuator authority for the local forecast demo. Forecast/model output is advisory-only. The browser simulator has no authority at all (`actionAuthority: "none"`). Preserve these boundaries in code, tests, generated reports, and documentation.

## Repository map

| Path | Role |
| --- | --- |
| `src/aeolus/` | Legacy deterministic simulator, recovery/evidence code, CLI modules, trace handling, and visualization. |
| `src/aeolus/habitat_v2/` | Habitat Plant V2 schemas, physics, air network, telemetry, HMC, safety, traces, qualification, and scenario runner. |
| `src/aeolus/habitat_v2/forecast/` | Forecast-only fixture contracts, projection, corpus/evaluation helpers, bounded live-demo generation, and verification. |
| `scenarios/` | Checked-in closed-schema simulator inputs. Do not treat them as real-hardware specifications. |
| `contracts/` | Versioned HMC/forecast/observability contracts and the reviewed-HMC source snapshot package. |
| `artifacts/` | Checked-in historical and demo-only artifacts; see each artifact README/receipt before changing an artifact or its provenance. |
| `tests/` | Unit, schema, replay, contract, forecast, evidence, and installed-package tests. Habitat V2 tests live in `tests/habitat_v2/`. |
| `scripts/` | Reproducible packet, demo, benchmark, installed-wheel, and verification entry points. |
| `docs/` | Simulation/telemetry contracts, evidence records, plans, research, and provenance documentation. |
| `demo/browser-simulator/` | Self-contained browser-local fixture explorer and Node test. It is deliberately separate from the Python judge receipt. |
| `.github/workflows/` | CI, judge-demo, and narrowly-scoped evidence workflows. |
| `out/` | Ignored local output only: generated traces, receipts, corpora, build/evidence outputs, and demo reports. |

Read `README.md`, `CONTRIBUTING.md`, `docs/simulation-rules.md`, and `docs/telemetry-contract.md` before changing simulator behavior, telemetry, scenarios, or model-facing projections.

## Architecture and data flow

### Simulator lineage

1. A checked-in scenario is parsed and validated as a closed schema.
2. The deterministic plant/Habitat V2 runner evolves the simulated state using declared inputs and seeded deterministic behavior.
3. Strict trace writers emit the allowed observable projection as JSONL; validators reject schema, topology, timing, bounds, and ownership drift.
4. Downstream visualization, corpus, evaluation, and evidence code consume validated output rather than hidden simulator truth.

Keep hidden truth out of model features: fault labels and schedules, fault effectiveness, static health, seeds, internal noise/bias state, reserve/authority audit state, reserve telemetry, and future-derived values must not become model inputs. Adding presentation fields or charts does not authorize widening a trace or model projection.

### Habitat V2 forecast/judge demo

The judge command creates a fresh ignored receipt directory, runs the bounded Python forecast demo, and independently verifies a fresh deterministic HMC replay. The generated `index.html` embeds report data for local viewing; the browser does not perform model inference. The command prints `file:` URLs only after verification.

The browser-local simulator is a separate static fixture explorer. Open its `index.html` locally; it makes no network requests and exposes four fixed scenarios. Its deterministic JavaScript trace rows must retain `actionAuthority: "none"`.

## Source, contract, and snapshot boundaries

### Editable source of behavior

The active implementation is under `src/`, with checked-in inputs under `scenarios/`, authoritative active contracts under `contracts/`, and tests under `tests/`. Modify these only with corresponding contract-aware tests and documentation where applicable.

### Reviewed-contract source snapshot

`contracts/habitat-v2-forecast-reviewed-hmc-v1/` is a copied, reviewed-contract artifact, not an alternate implementation tree:

- `manifest.json` binds its content, source paths, byte lengths, and hashes.
- `sources/` contains reviewed copies of selected HMC source, a scenario, a contract, packaging metadata, and lockfile bytes.
- Forecast contract loading verifies the snapshot and validates active source bytes against the reviewed binding.

Do **not** casually edit files below that directory, regenerate them opportunistically, or make the snapshot diverge from its manifest. A legitimate reviewed-contract update is a coordinated versioned-contract change: update the active source/contract first, deliberately create a new reviewed package or manifest as the governing contract requires, preserve provenance, and add tests that reject substitution or drift.

Likewise, do not hand-edit generated receipts, JSONL traces, corpus output, build outputs, or `out/` artifacts to obtain a desired result. Regenerate them through the documented command into a new ignored output path and retain the command/provenance needed to reproduce them.

## Environment and setup

- Project metadata requires Python `>=3.10`; the locked development/CI workflow uses Python 3.11.
- Use [uv](https://docs.astral.sh/uv/) and the committed `uv.lock`. There is no `requirements.txt`.
- Runtime dependency: NumPy. The `dev` extra supplies pytest, ONNX/ONNX Runtime, and Ruff.
- Python packages use a `src/` layout; do not solve imports by setting `PYTHONPATH` for normal source-checkout work. Pytest is configured to use `src`.
- The browser demo uses Node's built-in test runner and has no install step.

From a clean source checkout:

```bash
uv sync --locked --python 3.11 --extra dev
uv run --locked --python 3.11 --extra dev python -m pytest -q
uv run --locked --python 3.11 --extra dev ruff check .
uv run --locked --python 3.11 --extra dev python -m compileall -q src tests scripts
uv lock --check
git diff --check
```

Use `--locked` for reproducible development and CI-equivalent commands. If changing declared dependencies, intentionally update `pyproject.toml` and `uv.lock` together, then run `uv lock --check`; do not silently resolve a different dependency set.

## Development and verification commands

### Full source-checkout gate

Run this before a normal Python change, then add focused tests for the touched behavior:

```bash
uv run --locked --python 3.11 --extra dev python -m pytest -q
uv run --locked --python 3.11 --extra dev ruff check .
uv run --locked --python 3.11 --extra dev python -m compileall -q src tests scripts
uv lock --check
git diff --check
```

CI also rebuilds the tracked Habitat V2 observability packet, compares it byte-for-byte to the tracked packet, builds a wheel and sdist, then smoke-tests the built wheel against the compound-fault scenario. For changes that affect those surfaces, reproduce the relevant CI command or workflow logic locally where practical instead of assuming unit tests are enough.

### Core local simulator examples

All generated output belongs in an ignored new `out/` path and commands refuse to overwrite existing trace paths.

```bash
mkdir -p out
uv run --locked --python 3.11 python -m aeolus \
  scenarios/standard_habitat.json out/standard.jsonl
uv run --locked --python 3.11 python -m aeolus.visualise \
  out/standard.jsonl out/standard.html

uv run --locked --python 3.11 --extra dev python -m aeolus.habitat_v2 \
  scenarios/habitat_v2_reference.json out/habitat-v2-reference.jsonl
```

Use `python -I -m aeolus ...` or `python -I -m aeolus.habitat_v2 ...` only when testing an installed package with explicit absolute scenario and output paths.

### Judge demo and browser demo

Run the local judge receipt from the repository root:

```bash
uv run --locked --python 3.11 --extra dev python \
  scripts/run_habitat_v2_forecast_judge_demo.py
```

The wrapper allocates a new directory below `out/judge-demo-runs/`, runs the forecast, and invokes independent report verification. It accepts only the listed local action choices and requires the output parent to remain inside the repository. Do not redirect it to a tracked location or reuse a previous receipt directory.

Verify browser fixtures without starting a server:

```bash
node --test demo/browser-simulator/simulator.test.mjs
```

For manual exploration, open `demo/browser-simulator/index.html` directly in a modern browser. Do not add network calls, external dependencies, model inference, controller integration, or authority semantics to this demo.

### Focused forecast checks

When changing forecast, receipt, or judge-demo code, run the focused suite as used by the judge workflow in addition to relevant broader tests:

```bash
uv run --locked --python 3.11 --extra dev python -m pytest -q \
  tests/habitat_v2/test_forecast_*.py
uv run --locked --python 3.11 --extra dev ruff check \
  scripts/run_habitat_v2_forecast_judge_demo.py
uv run --locked --python 3.11 --extra dev python -m py_compile \
  scripts/run_habitat_v2_forecast_judge_demo.py
node --test demo/browser-simulator/simulator.test.mjs
```

## Determinism, provenance, and safety rules

- Keep runs deterministic: do not introduce wall-clock values, unseeded randomness, implicit environment-dependent inputs, or unordered serialization into replayed output.
- Canonical JSON, explicit ordering, exact schema field sets, byte/hash checks, and write-once outputs are intentional safety/provenance controls. Preserve them rather than loosening validators for convenience.
- A same-code, same-runtime, same-platform run is expected to produce byte-identical JSONL where the simulator contract says it is deterministic.
- Forecast reports bind generated artifacts and listed source files by length and SHA-256, then re-run deterministic forecast/HMC behavior during verification. Update verification and tests with any intentional report-schema or provenance change.
- Keep simulated plant truth separate from observable telemetry. A field visible in a trace is not automatically eligible for a forecast/model projection.
- Do not claim that historical artifacts, benchmark evidence, a local demo, or a passing replay establishes qualification, production deployment, hardware compatibility, or real-world control.

## Contracts, scenarios, telemetry, and test expectations

### Changing scenarios or plant behavior

Scenarios are closed-schema inputs; reject unknown fields and invalid values rather than accepting partial or permissive input. Preserve schema lineage: a new capability must not silently reinterpret older scenario or trace versions.

For changes to physics, scenario parsing, air-network behavior, fault behavior, HMC/safety logic, trace schema, or replay validation:

1. Identify the controlling contract and relevant tests before implementation.
2. Update validation, deterministic replay behavior, and fixtures together.
3. Add or update focused tests under `tests/` (normally `tests/habitat_v2/` for V2 work).
4. Re-run the full locked suite and any scenario/replay command affected.
5. Update `docs/simulation-rules.md` and/or `docs/telemetry-contract.md` when the externally documented contract changes.

### Changing telemetry or model projections

A telemetry change is a cross-cutting contract change. Update the trace writer, trace validator, visualizer validation, relevant corpus/projection code, documentation, and tests in the same change. Explicitly review leakage: model-facing projections must continue to exclude hidden fault truth, future data, authority state, and reserve-specific audit fields.

### Changing forecast contracts or demo artifacts

`src/aeolus/habitat_v2/forecast/contracts.py` fails closed against the frozen development fixture and reviewed HMC provenance. Contract loaders, report verification, and tests intentionally enforce exact fields, canonical JSON, hashes, file sets, and action-conditioned predictions. Treat any failure as evidence of a boundary mismatch, not something to bypass by weakening assertions.

Demo-only model/artifact files and their receipts are provenance-bound. If a deliberate artifact replacement is in scope, update the corresponding code constants, receipts/manifests, report verification, and tests as one reviewed change; otherwise leave them untouched.

## Protected data and operational prohibitions

- Do not inspect, run, copy, modify, or use final-suite/validation inputs or resulting data unless a separately authorized, documented human gate explicitly scopes that work.
- Do not tune thresholds, train/export/integrate a model, or reclassify historical negative evidence based on development/final data without an approved new protocol.
- Do not push, merge, deploy, provision cloud resources, access credentials, trigger unneeded remote workflows, or connect the project to hardware without explicit repository-owner approval.
- Never commit secrets, tokens, API keys, private URLs, local machine paths, usernames, environment dumps, or generated receipts that embed them. Use environment variables or secret stores only where an explicitly approved integration requires them.
- Keep generated outputs in ignored `out/`; inspect `git status` before staging so that caches, virtual environments, traces, evidence, and local reports are not committed accidentally.

## Code and documentation conventions

- Follow the existing Python style and formatter/linter behavior; use Ruff as the checked lint authority.
- Prefer small, typed, explicit functions and immutable/validated data boundaries consistent with existing Habitat V2 code.
- Preserve fail-closed validation, explicit error messages, canonical serialization, and path-containment checks.
- Keep JavaScript browser-fixture code dependency-free, deterministic, and compatible with the Node test runner used in `demo/browser-simulator/`.
- Document what a change proves and, equally importantly, what it does **not** prove. Avoid claims beyond the simulator/replay contract.
- When adding a command to documentation, run it from a clean source checkout if feasible and state required tools/working directory/output behavior accurately.

## Git and pull-request workflow

- Treat the configured `origin` as authoritative; verify the owner/repository and target branch before any delivery action.
- Branch from an up-to-date intended target using `name/short-description`.
- Keep each change focused on one simulator, scenario, trace, evidence, contract, or documentation concern.
- Use concise conventional commits when the repository's surrounding history does so.
- Before requesting review, run the applicable local gate, inspect `git diff`, and run `git diff --check`.
- Describe changed contracts, deterministic/provenance implications, safety-boundary effects, and exact verification commands/results in a PR. Do not state that tests or evidence passed unless they were actually run for that commit.
- Do not push, alter remotes, request review, merge, or deploy without repository-owner approval.

## Definition of done

A change is ready for review only when all applicable items are true:

- scope and non-goals are preserved or explicitly documented;
- source, scenario, contract, snapshot, and generated-artifact boundaries are respected;
- targeted tests plus the full locked suite (when feasible for the change) pass;
- Ruff, compilation, lockfile check, and `git diff --check` pass as applicable;
- deterministic/replay/provenance verification has been run for any affected path;
- affected documentation and contract records accurately match behavior;
- no generated output, secrets, local paths, credentials, or unauthorized final/validation data are staged;
- the diff is small, reviewable, and explains any deliberate artifact/contract update.

## Troubleshooting

| Symptom | Grounded response |
| --- | --- |
| `uv` reports a lock mismatch or resolves unexpected versions | Use `uv sync --locked --python 3.11 --extra dev`; if dependencies changed intentionally, update `pyproject.toml` and `uv.lock` together, then run `uv lock --check`. |
| Python cannot import `aeolus` during normal development | Run through `uv run` after `uv sync`; the package uses the `src/` layout and pytest is configured accordingly. Do not permanently add ad hoc `PYTHONPATH` workarounds. |
| A simulator/demo command refuses an existing output | This is expected write-once behavior. Choose a new path under ignored `out/`; do not delete or overwrite evidence to make a rerun succeed. |
| Forecast contract loading rejects source or snapshot drift | Inspect the active source, controlling binding, and reviewed-package manifest. Do not edit copied snapshot files or loosen hash checks as a shortcut. |
| Receipt verification fails | Treat it as an identity, canonical-JSON, file-set, provenance, report-embedding, forecast, or replay mismatch. Regenerate a new receipt and investigate the first failing boundary. |
| Browser demo tests fail because `crypto.subtle` is absent | Run the documented Node test command; the test initializes Node Web Crypto before importing the simulator. Avoid browser-only globals without a tested fallback. |
| A proposed change needs data or a result outside the checked-in development fixture | Stop and obtain explicit authorization and a separately scoped protocol; do not access protected validation/final material by default. |
