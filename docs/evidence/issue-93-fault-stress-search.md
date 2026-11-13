# Issue #93: budgeted fault stress-search with authoritative replay

Status: CPU harness delivered with a measured matched-budget result. Bounds, fixture and discovery tiers are **proposed, not yet agreed** (see the PROPOSAL comment on issue #93). GPU screening is **open and not run**.

Version impact: `minor` (new optional module and CLI). The numeric version, changelog and lockfile are left to the release owner; no existing file is changed.

## Result in brief

On the proposed development fixture, with the proposed bounds, **neither search method found a fault scenario that pushes plant truth past an HMC warning threshold**. Across 10 runs (2 methods, 5 seeds, 120 simulation calls each, 1190 fully evaluated candidates) both declared discovery tiers recorded zero discoveries. This is a negative result and is reported as one.

The adaptive method was **not** measurably better than the random baseline: random search reached the higher best objective on 3 of 5 seeds, cross-entropy on 2, with a mean difference of -0.028 scale units. No claim of an adaptive advantage is made.

All 50 selected cases (5 per run) were confirmed on the full authoritative pipeline, and two manifests replayed with every case reproduced within the declared tolerance.

## What this is and is not

The harness proposes fault scenarios inside declared bounds, runs each on the authoritative CPU plant under the deterministic HMC with no adviser, and scores a pre-declared objective from privileged simulator truth against a healthy twin.

- Outputs are biased development discoveries selected with privileged truth. They are not IID samples, not a validation, final or BLIND_FINAL population, and not evidence of generalisation.
- Adding any case to a training corpus needs a separately reviewed split and provenance decision. Every episode of one `causal_family_id` must stay in one partition.
- The harness never builds or runs a BDM-v1 roster family and never opens the custody registry. Candidates use their own fixture, names, sensor seed and HMC nonce namespace.
- Simulator truth is used only by the evaluator. There is no adviser interface: the HMC receives `propose(None)` on every cycle.
- It is a simulation research instrument. Nothing here is hardware validation or control of real equipment.

## Files

| Path | Role |
| --- | --- |
| `src/aeolus/habitat_v2/stress_search.py` | Specification loader, candidate generation and identity, both search methods, budgets, audit log, manifest, replay, comparison. |
| `scripts/run_fault_stress_search.py` | CLI: `search`, `resume`, `replay`, `compare`, `run-all`. |
| `scripts/stress_search_spec_v1.json` | The search specification as data. |
| `tests/habitat_v2/test_stress_search_unit.py` | Bookkeeping tests with a fabricated evaluator (not integration evidence). |
| `tests/habitat_v2/test_stress_search_hardening.py` | Regression tests for review fixes (fabricated evaluator). |
| `tests/habitat_v2/test_stress_search_integration.py` | Both methods, resume, confirmation, replay and comparison on the real CPU plant and HMC. |

## Specification (proposed)

Committed in its own commit (`d014e72`) before any reported run.

**Fixture `stress-search-nominal-hold-v1`.** The checked-in development base fixture `scenarios/habitat_v2_forecast_development.json`, copied in memory, with one timeline segment stretched to 96 steps of 60 s, the initial achieved actuation set to that segment's scheduled command, its own name prefix and sensor seed `93000001`. Sensor noise amplitudes stay at the reviewed HMC values.

Why the change to initial actuation is needed: on the unmodified base fixture, and on every BDM-v1 family derived from it, the equipment starts off. The HMC hold rule repeats the initial achieved state, so with no adviser it stays off and every physical fault multiplies zero. An integration test pins this: a fault on the unmodified base gives a maximum truth difference of exactly 0.0 against its healthy twin, while the same fault on the nominal-hold fixture moves truth.

**Policy `hmc_hold_no_adviser_v1`.** The deterministic HMC with no proposal on any step. No controller or model tuning.

**Fault space.** Ranges are imported from the reviewed BDM-v1 generator vocabulary and the spec loader rejects anything wider.

| Fault type | Multiplier range | Target |
| --- | --- | --- |
| `fan_speed_degradation` | 0.80 to 0.98 | the fan |
| `branch_resistance_increase` | 1.30 to 3.00 | one zone |
| `cooling_delivery_degradation` | 0.65 to 0.95 | one zone |
| `oxygen_delivery_degradation` | 0.65 to 0.95 | one zone |
| `scrubber_effectiveness_degradation` | 0.50 to 0.90 | scrubber |
| `condenser_effectiveness_degradation` | 0.50 to 0.90 | condenser |
| `damper_jam` | none | one damper |

Onset step 8 to 72; duration 16 to 64 steps; end step at most 95 (a draw that runs past it is rejected and recorded, never clipped). Ramp shapes `worsening`, `sustained`, `recovery`. One to three faults per candidate. Same-target overlaps are rejected by the existing scenario validator. Sensor and feedback defects, and any change to topology, loads, equipment or noise, are unsupported and rejected.

**Objective `peak_hazard_shift_v1`** (maximise). Hazard thresholds are the warning-enter levels in the HMC contract (`high_co2` 2500 ppm, `high_temperature` 300 K, `low_temperature` 291 K, `high_humidity` 0.65, `low_oxygen` 0.285), normalised by the Issue 55 per-channel scales (CO2 800 ppm, temperature 10 K, humidity 0.25, O2 0.005). The objective is the largest amount, over steps, zones and hazard families, by which the fault moves truth toward a warning threshold relative to the healthy twin. Units: scale units, dimensionless.

Recorded for every candidate but not optimised: `hazard_exposure_delta` (truth excess beyond the warning thresholds summed over zone-steps, fault minus twin), `residual_shift_at_end` (the shift at the final step, a recovery measure), the Issue 55 `safety_exposure` and `resource_composite`, HMC emergency-action steps and terminal status.

**Discovery tiers.** Tier A `hazard_exposure_delta > 0`: the fault causes a real warning-threshold crossing. Tier B `peak_hazard_shift >= 0.5`.

**Methods and budget.** Seeded random search, and cross-entropy search with fixed settings (population 12, elite 3, smoothing 0.7, minimum sigma 0.05); no hyperparameter sweep. Each run: 120 simulation calls (the healthy twin counts as one), 480 candidate attempts, 1800 s wall time. Seeds `issue93-s1` to `issue93-s5`.

**Selection and replay.** Up to 5 cases per run, at most one per causal family, objective above 0. Replay tolerance 1e-9 absolute on every metric, with identical control-trace hash required.

### What was pilot-informed

- A short in-memory probe of the plant (single faults at band edges, one three-fault combination) was run before the spec was written. It showed movements of up to about 340 ppm CO2, 0.09 humidity and 0.1 K, none near a threshold. Tier B's 0.5 was chosen after seeing those numbers, so tier B is pilot-informed, not blind. Tier A needs no tuned number.
- One CLI smoke run used a separate spec (`pilot-smoke`, 14 calls, seeds `pilot-a` and `pilot-b`). Candidate draws are keyed by the spec digest and seed, so no pilot draw is shared with the five evidence seeds. Pilot output is not part of the reported result.
- Test runs use their own specs and seeds (`test-s1`, `test-s2`).

## Evidence run

Command, from a clean checkout at commit `2dd32a4`:

```bash
uv run --locked --python 3.11 --extra dev python \
  scripts/run_fault_stress_search.py run-all --output out/issue-93-evidence-v1
```

Runtime: Python 3.11.16 (CPython), NumPy 2.4.6, Windows 10 build 26200 on AMD64, not containerised (no image digest). Source identity digest `da015453b8c2ac98366c720480674066eda018db806e15506ad57578aff986e7` (line-ending-normalised hashes of the harness and the simulated-path sources). Comparison digest `3756cdfe1c7e8d2c47fd6f498ca7b8d14223a2a65e5579c480bff5ac49b7efb9`.

Wall time was 69 to 112 s per run, 822 s in total. It is **provisional**: the full test suite and other work shared the machine. No run came near the wall budget.

### Per run

| Method | Seed | Sim calls | Attempts | Evaluated | Families evaluated | Best objective | Tier A families | Tier B families |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_search | issue93-s1 | 120 | 259 | 119 | 89 | 0.2232 | 0 | 0 |
| random_search | issue93-s2 | 120 | 217 | 119 | 82 | 0.3114 | 0 | 0 |
| random_search | issue93-s3 | 120 | 227 | 119 | 86 | 0.2476 | 0 | 0 |
| random_search | issue93-s4 | 120 | 227 | 119 | 76 | 0.2687 | 0 | 0 |
| random_search | issue93-s5 | 120 | 236 | 119 | 94 | 0.3589 | 0 | 0 |
| cross_entropy | issue93-s1 | 120 | 147 | 119 | 27 | 0.2805 | 0 | 0 |
| cross_entropy | issue93-s2 | 120 | 127 | 119 | 23 | 0.2257 | 0 | 0 |
| cross_entropy | issue93-s3 | 120 | 134 | 119 | 41 | 0.2196 | 0 | 0 |
| cross_entropy | issue93-s4 | 120 | 133 | 119 | 39 | 0.2745 | 0 | 0 |
| cross_entropy | issue93-s5 | 120 | 135 | 119 | 61 | 0.2679 | 0 | 0 |

Every run ended on the simulation-call budget, so budgets are matched exactly. Discovery rate is 0 families per simulation call for both tiers and both methods.

### Per method and paired

| Method | Best objective, mean (min to max) | Seeds with a tier A discovery | Seeds with a tier B discovery |
| --- | --- | --- | --- |
| random_search | 0.2820 (0.2232 to 0.3589) | 0 of 5 | 0 of 5 |
| cross_entropy | 0.2537 (0.2196 to 0.2805) | 0 of 5 | 0 of 5 |

| Seed | Random | Cross-entropy | Difference |
| --- | --- | --- | --- |
| issue93-s1 | 0.2232 | 0.2805 | 0.0573 |
| issue93-s2 | 0.3114 | 0.2257 | -0.0857 |
| issue93-s3 | 0.2476 | 0.2196 | -0.0280 |
| issue93-s4 | 0.2687 | 0.2745 | 0.0059 |
| issue93-s5 | 0.3589 | 0.2679 | -0.0910 |

These figures are descriptive. No significance test was run. Seeds are repeated searches on one fixture, not independent populations. Best objective is the quantity the adaptive method optimises, so it is the comparison most favourable to it, and it still did not come out ahead.

### All attempts

Across the 10 runs: 1842 attempts, of which 1190 were evaluated, 581 were rejected for parameters outside the declared space (in every case a fault running past the latest end step) and 71 were rejected by the existing scenario validator (same-target overlap). There were no duplicates, HMC terminal failures, non-finite outputs or simulator errors. Rejections cost an attempt but no simulation call. Random search loses about half its attempts to the end-step rule because onset and duration are drawn independently; cross-entropy loses far fewer once its distribution moves.

Cross-entropy concentrated on fewer causal families (23 to 61 per run against 76 to 94 for random search). That is the expected cost of adapting, and here it bought no higher objective.

### Hardest cases found

The largest shift on any seed was 0.359 scale units (about 287 ppm CO2 toward the 2500 ppm warning level), from random search on `issue93-s5`: a sustained scrubber degradation to 0.514 from step 10 for 17 steps combined with a later worsening fan degradation. It peaked at step 26 in `air_processing_bay` and had almost fully recovered by the end of the horizon (residual 0.013). The next cases were CO2 and humidity shifts of 0.28 to 0.31 from scrubber or condenser degradation combined with fan or branch faults. The humidity cases had not recovered by the end (residual 0.25 to 0.27).

For scale, the healthy twin sits about 2.1 scale units below the CO2 warning level and about 0.7 below the humidity warning level, so the hardest case found closes roughly a sixth of the CO2 margin.

### Confirmation and replay

All 50 selected cases were re-run with control-trace export, strict parse and replay, and all 50 are `CONFIRMED`: metrics equal to the search-time values within 1e-9 and identical trace hashes.

Two manifests were then replayed in a separate process:

```bash
uv run --locked --python 3.11 --extra dev python scripts/run_fault_stress_search.py \
  replay --manifest out/issue-93-evidence-v1/cross_entropy-issue93-s1/manifest.json \
  --output out/issue-93-evidence-v1-replay-cross_entropy-issue93-s1
```

Both `cross_entropy-issue93-s1` and `random_search-issue93-s1` returned `PASS`, 5 of 5 cases reproduced. The other eight manifests were not replayed separately; their cases were confirmed at finalisation only.

Each manifest carries the exact replay command, and each run directory holds the spec, the audit log, the manifest, an environment record and, per selected case, the scenario, the control trace and a per-step actuation record (adviser request, HMC final command, plant requested, achieved and effective actuation, active faults).

## Reproducibility, resume and integrity

- **Deterministic sequence.** Candidate draws are a pure function of the spec digest, method, seed and attempt index. Tests run the same real search twice and require identical audit records and identical manifest bytes.
- **Audit.** Append-only JSONL in which each record is hash-chained to the previous one. Every attempt gets exactly one record with its disposition and reason. Wall-clock values sit outside the chain so the chained records stay deterministic; they are still validated on load.
- **Budgets.** Simulation calls, candidate attempts and wall time. The first two are counters inside the chain and are re-derived and cross-checked on resume.
- **Interrupt and resume.** Ctrl-C stops between plant steps. An episode cut mid-run is recorded as incomplete and not charged. `resume` re-derives the recorded attempts, checks each against the audit, then continues; nothing recorded is re-simulated and nothing is skipped. A run that ended on a budget cannot be resumed. A resumed run reproduces the uninterrupted run's candidate records and selected results; its manifest hash differs because the manifest records the number of sessions.
- **Single writer.** A run directory takes an operating-system lock; a second concurrent writer is refused.
- **Manifest binding.** Replay and comparison recompute every audit-determined section of a manifest from the audit beside it. Dropping a case, changing a metric, relabelling the method or editing budget figures and re-sealing the hash fails closed. Stored case files are verified against their digests and against regenerated bytes.
- **Comparison.** Refuses a missing or duplicate (method, seed), mixed identities, and any run that ended on wall time or a failed baseline. An empty selection replays as `NO_CASES`, never `PASS`.

## Limitations

- **The integrity checks are not tamper-proof.** Hashes are unkeyed. They catch accidental edits and inconsistent ones, not a coordinated rewrite of both audit and manifest. The wall-time figures sit outside the hash chain; a consistent rewrite of them is not detected.
- **A wall-time stop is final and machine-dependent.** Such a run cannot be resumed and is refused by the comparison. None occurred in the evidence run.
- **A process killed while appending an audit line** leaves a partial record; the log is not repaired and that run must be discarded.
- **Replay tolerance was exercised on one platform only.** Trace hashes are required to be identical, which holds on the same code, runtime and platform. Cross-platform replay was not tested.
- **The fixture over-scrubs.** Under the nominal command the healthy twin drives CO2 from about 800 ppm to near zero in several zones and raises O2 slightly above 0.30. Faults that weaken scrubbing therefore move CO2 back toward its starting level, not toward a hazard, which is part of why no threshold is reachable. This is a property of the base fixture's schedule and loads, not something the harness changes.
- **Issue 55 `safety_exposure` is recorded but is not a useful objective here.** Its bounds count CO2 below 300 ppm and O2 above 0.30 as exposure, both of which the healthy twin already shows.
- **Narrow scope.** One fixture, one fixed policy, physical faults only, eight-zone topology, 96-step horizon.
- **Non-finite simulator output could not be produced by the real plant.** That path is tested by making the truth projection refuse a state at a chosen step on the real pipeline, and by fabricated evaluators.
- **Timing is provisional** and no throughput claim is made.

## Open items

1. **Bounds, fixture and tiers are not yet agreed.** Three decisions are with Alex or Ben: accept the nominal-hold fixture; keep faults-only bounds or also vary load scale (0.75 to 1.3), which would make thresholds reachable but doubles cost because the twin must be rerun per load; accept the objective and tiers. A change is a new spec version and a rerun, not an edit.
2. **GPU screening is not implemented and no DGX run exists.** It depends on issues #91 and #92. The spec loader refuses any screening backend other than `none`, and every manifest records `OPEN_NOT_RUN`.
3. **What the null means for the programme.** Inside the reviewed BDM-v1 fault bands, under hold on this fixture, faults alone do not create a decision opportunity. That agrees with the zero-exposure finding of the Issue #77 study and points at operating conditions, not fault magnitude, as the lever to examine next.

## Verification

Run on the branch at commit `cc5c28e` (the commit before this document was added):

| Check | Result |
| --- | --- |
| `pytest -q` on the three stress-search test files | 578 passed (unit and integration) plus 49 passed (hardening) |
| `pytest -q` on the rest of the suite | 1329 passed, 4 skipped (same as `main` before this change) |
| `ruff check .` | clean |
| `python -m compileall -q src tests scripts` | clean |
| `uv lock --check` | clean |
| `git diff --check` against `main` | clean |

The pre-existing suite and the stress-search files were run as separate invocations, not as one combined run.
