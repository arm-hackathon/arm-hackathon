# AGENTS.md — AEOLUS

Deterministic habitat environmental simulation with replayable traces.
Airflow and Environmental Observation Laboratory for User-defined Scenarios.

**Current status (2026-08-07):** development handoff. The V6 cycle is a
documented honest negative (learned candidate lost to rules). The V7 candidate
(escalated rules + gated residual centroid) has **no canonical verdict yet** —
the first canonical run was aborted, a Windows attempt wedged in pool startup,
and the runner was hardened. Next owner: run the V7 canonical evaluation (see
"Next steps"). All PRs remain drafts; nothing is merged.

## Repository layout

- `src/aeolus/` — simulator, features, detectors, evaluation, model cycles.
- `tests/` — pytest suite; `test_model_cycle_v7.py` covers runner observability.
- `scenarios/` — sweep specs and scenario JSON (authoritative family splits).
- `scripts/run_v7_canonical.py` — the one canonical V7 runner entry point.
- `docs/plans/` — predeclared cycle plans; read before changing a cycle.
- `out/` — generated evidence, gitignored, never committed.

## Commands

```bash
uv sync --extra dev
uv run --extra dev python -m pytest -q        # full suite (~60s)
uv run ruff check .                           # lint
PYTHONPATH=src .venv/bin/python scripts/run_v7_canonical.py \
    scenarios/sweep-v7-development.json out/v7-<strategy>-canonical-YYYY-MM-DD-<letter>
```

## Evidence and evaluation discipline (non-negotiable)

- Fit, calibration, and validation families are **disjoint by design** — never
  mix roles. Fit = train transformations/normalisation; calibration = threshold
  selection; validation = one untouched final evaluation.
- Never retune against validation or a frozen final suite. A result that was
  tuned after seeing validation is not evidence.
- Canonical runs require a **clean Git worktree** (provenance hashes are
  recorded at launch) and a **fresh, empty output directory**. Never reuse a
  partial or failed output dir; preserve it for debugging and use a new one.
- The runner writes `calibration-progress.log` (per-eval lines) and
  `v7-development-report.json` (the full receipt: source hashes, grid, selected
  parameters, baseline vs candidate metrics, acceptance gate).
- A **negative result is a completed gate**, not unfinished work. Document the
  loss with receipts; do not relabel old output as final.
- The acceptance gate is frozen: candidate must beat the rule baseline on
  named-fault macro-F1 AND keep healthy alert episodes <= 10 per 1,000 ticks
  (and <= 2 above rules) AND not regress detection recall.
- Deployment/optimisation claims require measured evidence on a declared
  target. ONNX export or small model size alone is not an Arm-optimisation
  claim.

## Current state and next steps

1. Checkout `ben/v6-conditional-specialists` (`52128eb`), clean worktree.
2. Launch the canonical V7 run on **Linux** (the calibration pool works there:
   ~2–4h on 2 cores, ~30–70 min on 8+ cores) or **Windows** (runs sequentially,
   ~6h; the pool is disabled on win32).
   - Worker cap: default 12, override with `AEOLUS_CALIBRATION_WORKERS`.
3. Watch `calibration-progress.log`; when it finishes, extract the verdict from
   `v7-development-report.json` (`development_gate_passed`, `selected_candidate`).
4. If the gate passes: freeze candidate + thresholds, generate an untouched
   final suite, evaluate once, commit evidence, draft PR.
5. If it fails: error analysis by family/fault class, predeclare ONE strategy
   change, implement test-first, new canonical run. Levers, in order:
   architecture → simulation complexity → data volume.
6. The AEOLUS cron loop worker is **paused** (provider outage + handoff).
   Re-enable only deliberately; do not create new cron jobs from this repo.

## Conventions

- Python 3.11+, type hints, docstrings on public surfaces.
- Test-first: RED → GREEN → REFACTOR. New behaviour ships with tests.
- Ruff-clean and full pytest green before any push.
- `scripts/` entry points must work on Linux and Windows (no fork assumptions).
- Do not touch `alex/*` or `yarofix*` branches. PRs stay drafts until the
  owner requests review. No merges without explicit approval.
- Generated artifacts go in ignored `out/`; never commit evidence blobs.
