# Contributing

## Setup

The authoritative clone URL is the configured `origin` remote:

```bash
git clone https://github.com/arm-hackathon/arm-hackathon.git
cd arm-hackathon
uv run --extra dev python -m pytest
```

The project has no `requirements.txt`. `uv` resolves the project and the
`dev` extra from `pyproject.toml`; `uv.lock` is committed to make that
resolution reproducible. `uv` creates its local environment as needed.

## Local workflow

1. Branch from an up-to-date target branch using `name/short-description`.
2. Keep a change scoped to one simulator, scenario, trace or documentation
   concern.
3. Run the full suite before commit:

   ```bash
   uv run --extra dev python -m pytest
   ```

4. For changed simulation behaviour, generate the affected replay locally and
   inspect its visualisation:

   ```bash
   mkdir -p out
   PYTHONPATH=src uv run python -m icarus \
     scenarios/standard_habitat.json out/standard.jsonl
   PYTHONPATH=src uv run python -m icarus.visualise \
     out/standard.jsonl out/standard.html
   ```

5. Run `git diff --check`, commit with a short imperative subject, then open a
   pull request for review. Do not push a branch or alter a remote without the
   repository owner's approval.

## Repository map

| Path | Purpose |
|---|---|
| `scenarios/` | Closed-schema v7 scenario inputs. |
| `src/icarus/config.py` | Scenario parsing and validation. |
| `src/icarus/plant.py` | Deterministic plant, airflow and CO₂ mass transfer. |
| `src/icarus/scenario.py` | Warm-up, measured runs and fault-profile scheduling. |
| `src/icarus/trace.py` | JSONL writer and allowlisted model-feature projection. |
| `src/icarus/visualise.py` | Dependency-free local HTML replay visualiser. |
| `tests/` | Unit, scenario, replay and visualisation tests. |
| `docs/` | Current simulation and telemetry contracts. |
| `out/` | Ignored generated local traces and reports. |

## Current boundaries

ICARUS currently implements deterministic simulation and replay only. It does
not include model training or inference, ONNX, quantisation, a governor,
redundant fan, Arm measurements, a dashboard, API, database, cloud service or
hardware integration. Do not represent future work as current behaviour.
