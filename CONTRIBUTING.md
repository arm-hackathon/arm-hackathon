# Contributing

## Setup

The authoritative clone URL is the configured `origin` remote:

```bash
git clone https://github.com/arm-hackathon/arm-hackathon.git
cd arm-hackathon
uv sync --locked --python 3.11 --extra dev
uv run --locked --python 3.11 --extra dev python -m pytest -q
```

The project has no `requirements.txt`. `uv.lock` is committed so the declared
Python dependencies resolve reproducibly.

## Local workflow

1. Branch from an up-to-date target branch using `name/short-description`.
2. Keep a change scoped to one simulator, scenario, trace, evidence, or
   documentation concern.
3. Run the locked verification suite before commit:

   ```bash
   uv run --locked --python 3.11 --extra dev python -m pytest -q
   uv run --locked --python 3.11 --extra dev ruff check .
   uv run --locked --python 3.11 --extra dev python -m compileall -q src tests scripts
   uv lock --check
   git diff --check
   ```

4. Generate source-checkout replay output only under a new ignored `out/` path:

   ```bash
   uv run --locked --python 3.11 python -m aeolus \
     scenarios/standard_habitat.json out/standard.jsonl
   ```

5. Do not push a branch, alter a remote, request review, merge, deploy, or
   provision cloud resources without the repository owner's approval.

## Repository map

| Path | Purpose |
|---|---|
| `scenarios/` | Closed-schema legacy and Habitat V2 simulator inputs. |
| `src/aeolus/` | Legacy simulator, recovery/evidence code, CLI, and trace handling. |
| `src/aeolus/habitat_v2/` | Habitat V2 physics, telemetry, HMC, safety, replay, and forecast fixtures. |
| `contracts/` | Versioned HMC, forecast, observability, and reviewed-source contracts. |
| `artifacts/` | Historical or demo-only artifacts; read the local artifact README before use. |
| `tests/` | Unit, scenario, replay, evidence, forecast, and contract tests. |
| `scripts/` | Reproducible source, package, benchmark, demo, and verification entry points. |
| `docs/` | Current simulation, telemetry, evidence, plan, and historical records. |
| `demo/browser-simulator/` | Offline fixture explorer with no controller authority. |
| `out/` | Ignored generated traces, corpora, packages, and local receipts. |

## Current boundaries

C4 recovery development remains a documented negative result, not an accepted
recovery result. Later Habitat V2 work includes bounded forecast-only research,
demo artifacts, and native Arm64 development measurements; none rewrites C4 or
authorizes final-suite access, model training/export/integration, deployment,
hardware integration, production control, or a physical-performance claim.
HMC remains the sole actuator authority for the local forecast demo; the browser
fixture has no authority.
