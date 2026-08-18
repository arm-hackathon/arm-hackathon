# Closed-Loop Forecast Advisory — Development Evidence (2026-08-18)

**Status: development evidence only. Not qualification. Not deployment. No
learned actuator authority.**

This directory contains a closed-loop experiment that integrates the
Historical V2 development forecaster (MLP, action-aware, trained on the
`habitat-v2-forecast-pilot-v1` archive) into the real HMC lifecycle as an
**adviser**. The model scores each frozen catalogue action by predicted
safety-threshold exceedance over the next 8 steps and submits the best option
as a canonical proposal. HMC arbitration remains the sole authority and
overrode 81 of 793 proposals across the campaign.

## Contents

- `aeolus_closed_loop.py` — harness: risk functional, model wrapper, lifecycle runner
- `run_paired.py` — paired experiment runner (control vs advised, identical seeds)
- `preregistration.json` / `preregistration-v2.json` — scoring rules, roster and
  success criteria frozen before outcomes (self-hashed)
- `CLOSED_LOOP_REPORT.md` — v1: 8-pair contingency-only result
- `CLOSED_LOOP_REPORT_V2.md` — v2: full 17-cluster held-out roster (238 runs)
- `results-summary.json` — compact per-run metrics with SHA-256 bindings to the
  full local result files (per repo convention, raw evidence blobs are not committed)

## Headline (v2, pre-registered)

- 102 fault pairs: **78 better, 24 equal, 0 worse** on integrated safety-threshold
  exceedance; 72 wins driven to exactly zero exceedance
- 17 healthy pairs: 16/17 exactly zero; the one remainder reduced exceedance
  8.354 → 0.038 vs canonical HMC
- Zero terminal regressions; strict trace replay and shadow-physics digest
  equality held in all 238 runs
- Resource cost: median +757 Wh battery, +1.97 mol O2, +6.04 mol sorbent per
  intervening pair (the adviser buys safety with consumables)

## Limitations

Development model without operational availability masks; advisory path only;
canonical HMC baseline is deliberately passive; no sealed-corpus qualification
has been performed for this adviser.

Implementation was agent-assisted (Hermes/Atlas) under human direction;
preregistrations, invariants and claim boundaries are recorded in-file.
