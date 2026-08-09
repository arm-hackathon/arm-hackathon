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
   python -m compileall -q src tests
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
| `scenarios/` | Closed-schema v9 standard and v10 recovery scenarios; historical and development sweep specifications. |
| `src/aeolus/config.py` | Scenario parsing and validation. |
| `src/aeolus/plant.py` | Deterministic plant, primary/reserve airflow, and CO₂ mass transfer. |
| `src/aeolus/scenario.py` | Warm-up, measured runs, fault scheduling, and recovery-arm execution. |
| `src/aeolus/recovery.py` | Deterministic reserve authority, settings, validation, and state machine. |
| `src/aeolus/recovery_evidence.py` | Write-once four-arm development evidence runner and gates. |
| `src/aeolus/trace.py` | JSONL writers and strict telemetry/model-feature boundaries. |
| `tests/` | Unit, scenario, replay, evidence, and contract tests. |
| `docs/` | Current simulation, telemetry, recovery, and historical protocol records. |
| `out/` | Ignored generated traces, corpora, packages, and local closeout receipts. |

## Current boundaries

C4 recovery development is a documented negative result, not an accepted
recovery result. The mandatory transient handback acknowledgement and physical
reserve-delivery benefit gates are false. C5 adviser work is closed: do not train,
tune, export, integrate, or inspect final-suite data on the strength of C4.

AEOLUS includes deterministic simulation/replay and historical model/FP32 ONNX
code, but no C4-qualified adviser, INT8 result, Arm benchmark, hardware
integration, cloud service, physical deployment, or production control claim.
