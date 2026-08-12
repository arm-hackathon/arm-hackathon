# AEOLUS progression

Last updated: 2026-08-12
Current pre-model objective: complete the Habitat V2 deterministic world, data and judge-facing runtime, then stop before learned-model implementation or training.

## Canonical lineage

- public base: `origin/main` at `5253176`
- accepted recovery foundation: `89ff124`
- Habitat V2 conservation kernel: `8712ea0`
- versioned operating modes: `087b1e9`
- active stack: `ben/habitat-v2-air-network`

## Verified foundations

### Deterministic recovery

- blind verification accepted
- recovery authority remains deterministic
- learned output cannot own emergency recovery

### Habitat V2 conservation kernel

- SI gas inventories
- CO₂ and O₂ accounting
- humidity and condensation
- thermal receipts
- electrical and battery receipts
- deterministic lineage and trace replay

### Operating modes

- `occupied`
- `eva_transition`
- `contingency`
- `dormant`
- mode is explicit context and does not silently alter physics
- 571 project-environment tests passed on 2026-08-12
- Ruff 0.14.10 and compile checks passed

## Active stacked slices

### 1. Explicit multizone air network

Status: candidate frozen for bounded review
Branch: `ben/habitat-v2-air-network`
Target version: `0.5.0`

Deliverables:

- eight-zone notional habitat
- supply and return topology
- fan curve and deterministic operating-point solver
- per-zone motorised dampers
- pressure, airflow and fan-power telemetry
- physical placement metadata
- scenario-v3 and trace-v3
- old-contract byte preservation

Verification on the candidate bytes:

- repository suite: `595 passed in 118.86s`
- Ruff 0.14.10: passed
- Python 3.11 compilation: passed
- wheel and source distribution built as `0.5.0`
- clean external wheel import under `python -I`: passed
- installed eight-zone CLI run repeated byte-for-byte
- installed trace: 5 rows, 8 zones, final step 4
- installed trace SHA-256:
  `dd3b3a579f5eaa8b08b0ffa5a230f5ef833f39233dcefc07a12e2ad4d6b3bd8d`
- wheel SHA-256:
  `6610f6bb7f67ad445b90630444204e10cd9fb8a89aadd306714e1844e1be5e95`
- source distribution SHA-256:
  `9b77e023308ebe0d2aedc30bc0dc9d505b0099db8e584e09f54bd9d2c16ec882`
- maximum installed-run residuals:
  - species: `9.094947017729282e-13 mol`
  - zone thermal: `2.0942752598784864e-07 J`
  - system thermal: `6.511545507237315e-07 J`
  - electrical: `1.4210854715202004e-14 Wh`
  - fan/system operating point: `1.4210854715202004e-13 Pa`
  - fixed-density supply-return mass: `0.0 kg/s`
- build warning: the existing `project.license` TOML-table form is deprecated by
  setuptools and should be corrected in a separate packaging-maintenance slice

### 2. Fault and observation layer

Status: pending

Deliverables:

- duct blockage
- filter loading
- fan degradation
- stuck and lagging dampers
- sensor bias, drift and freeze
- command versus delivered disagreement
- truth versus observed telemetry
- deterministic fault manifests and receipts

### 3. Scenario families and corpus-v3

Status: pending

Deliverables:

- mission-mode and occupancy schedules
- equipment and metabolic load schedules
- single and compound fault families
- whole-scenario train, validation and blind-test separation
- candidate-action rollout records
- simulator, scenario, feature and target provenance
- raw deterministic traces suitable for later model work

### 4. Judge-facing viewer and packaging

Status: pending

Deliverables:

- browser-based Three.js habitat cutaway
- components and airflow bound to real IDs
- replay timeline and fault visualization
- baseline and deterministic-control comparison
- offline assets and CPU-only path
- one-command local demo
- clean installation and replay evidence

### 5. Native Arm64 evidence

Status: pending

Deliverables before model training:

- Azure Arm64 machine receipt
- native simulator and viewer smoke check
- reproducible environment metadata
- benchmark harness ready for later model candidates

## Explicit stop boundary

Do not implement a learned model, select model architecture in code, quantize a model, or begin training without Ben participating. The pre-model phase may freeze dataset, feature, target, evaluation and optimization contracts.

## Verification policy

- focused tests during each slice
- one complete suite per meaningful PR boundary
- no repeated broad test runs without new evidence
- environment-bound commands use `python -m pytest`, not a global `pytest` executable
- one bounded independent review per slice
- code PR publication requires Ben's diff-grounded comprehension gate

## Submission claims boundary

AEOLUS is a deterministic, reduced-order, physics-informed research simulator and AI-control evaluation environment. It is not CFD, a certified spacecraft digital twin, flight software or real-world validation.
