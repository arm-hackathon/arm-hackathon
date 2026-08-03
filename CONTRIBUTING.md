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
   PYTHONPATH=src uv run python -m aeolus \
     scenarios/standard_habitat.json out/standard.jsonl
   PYTHONPATH=src uv run python -m aeolus.visualise \
     out/standard.jsonl out/standard.html
   ```

5. Run `git diff --check`, commit with a short imperative subject, then open a
   pull request for review. Do not push a branch or alter a remote without the
   repository owner's approval.

## Repository map

| Path | Purpose |
|---|---|
| `scenarios/` | Closed-schema v9 scenarios and sweep-v1/v2/v3 specifications. |
| `src/aeolus/config.py` | Scenario parsing and validation. |
| `src/aeolus/plant.py` | Deterministic plant, airflow and CO₂ mass transfer. |
| `src/aeolus/scenario.py` | Warm-up, measured runs and fault-profile scheduling. |
| `src/aeolus/sweep.py` | Deterministic family-held-out scenario generation. |
| `src/aeolus/detector.py` | Softmax/temporal-MLP training, strict inference and FP32 ONNX export. |
| `src/aeolus/protocol.py` | Validation-only selection, frozen policy and final-only evaluation. |
| `src/aeolus/trace.py` | JSONL writer and allowlisted model-feature projection. |
| `src/aeolus/visualise.py` | Dependency-free local HTML replay visualiser. |
| `tests/` | Unit, scenario, replay and visualisation tests. |
| `docs/` | Current simulation and telemetry contracts. |
| `out/` | Ignored generated local traces and reports. |

## Current boundaries

AEOLUS currently includes deterministic simulation/replay plus an experimental
softmax and temporal-MLP training/inference paths with FP32 ONNX export. It does not include
INT8 quantisation, a governor, redundant fan, Arm measurements, a dashboard,
API, database, cloud service or hardware integration. The accepted held-out
evidence does not show an AI advantage; do not represent future work or the
negative model result as completed deployment evidence.
