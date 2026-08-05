# V5 Nominal Counterfactuals Implementation Plan

> **For Hermes:** Execute each task with RED-GREEN-REFACTOR. Keep the v3 final suite and v4 development evidence immutable historical evidence.

**Goal:** Create a fresh v5 development-only corpus with physically declared benign occupancy/load-shape variation, then re-evaluate the unchanged candidate set under the existing false-alert acceptance gates.

**Architecture:** Extend only a new `aeolus_sweep_v5` schema with per-zone occupancy schedules that are applied to the fault-free reference and copied unchanged into each paired fault scenario. Retain the v4 detector, gate, metrics and acceptance rules, but use fresh v5 seed clusters and a v5-specific canonical runner. This isolates the question: whether richer legitimate nominal variation exposes a safer feature boundary. It does not retune against v3 final evidence, connect any response layer, or claim a final result.

**Tech Stack:** Python 3.11, NumPy, ONNX opset 17, pytest, Ruff.

---

## Frozen design constraints

- V3 final and v4 development results remain historical; neither may become a selection input.
- V5 is development-only. It has no final split, no response integration and no deployment authorization unless its development gate passes.
- The learned candidates, gate grid and safety thresholds are copied unchanged from v4. The corpus change is the experimental variable.
- Every changed occupancy schedule is applied before generating the fault pair. Reference/fault documents may differ only in `fault_profiles`.
- V5 seed clusters are fresh and disjoint from v3/v4: fit `1100..1103`, train-internal calibration `1104..1105`, validation `1300..1305`.
- Test and validation families must prove scenario-content disjointness from v3 development, v3 final and v4 development manifests.
- The result is a data-quality diagnostic. It cannot establish operational safety, OOD robustness, hardware behaviour, or a final selection result.

## Predeclared v5 operating profiles

All use existing schema-v9 telemetry fields and the same fault severities/classes as v4.

1. `v5-primary-low-baseline`: global source multiplier `0.8`, capacity `24.0`, base occupancy schedules.
2. `v5-primary-high-baseline`: global source multiplier `1.2`, capacity `36.0`, base occupancy schedules.
3. `v5-staggered-load`: global source multiplier `1.0`, capacity `30.0`; crew and lab occupancy peaks occur at separate periods.
4. `v5-lab-peak-transition`: global source multiplier `1.05`, capacity `32.0`; a substantial but nominal lab peak and counter-phased crew demand create actuator and airflow transitions.

The exact per-zone periods are represented in the versioned JSON spec, not embedded in Python. They must span ticks 1 through 120 without overlaps or gaps, target only non-processing zones, and be accepted by the existing strict scenario parser.

## Task 1: Add a test-first v5 sweep schema

**Objective:** Add `aeolus_sweep_v5` without changing v1–v4 parsing or their canonical outputs.

**Files:**
- Modify: `src/aeolus/sweep.py`
- Modify: `tests/test_sweep.py`
- Create: `tests/test_v5_sweep.py`

**Step 1 — RED:** Test that v5 requires development role, exactly `train`/`validation`, and a complete `occupancy_profiles` mapping for all non-processing zones.

**Step 2 — RED:** Test rejection for missing zones, processing-zone schedules, malformed/non-finite periods, and v4 profiles that add the v5-only field.

**Step 3 — GREEN:** Add `SWEEP_V5_VERSION`, a v5 operating-profile representation, strict parsing, and copy validated profiles to reference documents before fault construction.

**Step 4 — Verify:** Generate a small v5 fixture and assert reference/fault equality after removing `fault_profiles`; run old sweep tests unchanged.

## Task 2: Freeze the v5 development corpus specification

**Objective:** Add a new, reviewable source-of-truth JSON specification whose normal variation directly targets the measured high-demand false-alert geometry.

**Files:**
- Create: `scenarios/sweep-v5-development.json`
- Modify: `tests/test_v5_sweep.py`

**Step 1 — RED:** Test exact v5 split roles, seed clusters, profile IDs, family count and disjointness from generated v3/v4 manifests.

**Step 2 — GREEN:** Declare all four profiles, seed groups, fault start ticks, existing fault severities and exact zone occupancy schedules.

**Step 3 — Verify:** Generate into an empty ignored directory, validate pair equality, check profile-specific reference identities are distinct, and validate manifest content disjointness.

## Task 3: Add a v5 development runner without modifying v4 policy

**Objective:** Reuse the approved v4 training/evaluation mechanics under a separate v5 identity and provenance receipt.

**Files:**
- Create: `src/aeolus/model_cycle_v5.py`
- Create: `tests/test_model_cycle_v5.py`

**Step 1 — RED:** Test v5 runner rejects non-v5 specs, wrong seed groups, historical v3/v4 seed reuse, non-empty output directories, and altered canonical specs.

**Step 2 — GREEN:** Implement `run_v5_development()` using unchanged candidates, calibration grid and acceptance bounds; use v5-only seed constants and output report schema/identity.

**Step 3 — RED/GREEN:** Test report retains rules when all candidates fail, never authorizes a response/final suite on failure, and records source provenance.

## Task 4: Run the v5 development diagnostic

**Objective:** Produce a fresh ignored development receipt after source freezes.

**Files:**
- Generated only: `out/v5-nominal-counterfactuals-*/`

**Steps:**
1. Confirm a clean worktree and capture source hashes.
2. Run the declared command with CPython 3.11, locked dependencies, empty output, and `mlp_epochs=300`, `cnn_epochs=300`.
3. Parse strict report/artifact metadata, compare candidate/rule false-alert episode burden per healthy profile, and record whether the gate passed.
4. Re-run into a second empty ignored output path; compare deterministic receipt/model/report hashes.

## Task 5: Review and close out

**Objective:** Verify no false claim has crossed the evidence boundary.

**Files:**
- Modify only after results and review: `docs/evidence/...` or `README.md`

**Steps:**
1. Run full pytest, Ruff, diff whitespace and secret scans.
2. Run a focused methodology/security review of schema validation, split isolation, scenario pairing and receipt provenance.
3. Commit only reviewed source/tests/docs locally; do not push, request review, merge or alter PR #19.
4. State whether v5 is a negative diagnostic, a development candidate, or has earned a separately predeclared final-suite proposal.
