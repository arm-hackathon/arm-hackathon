# AEOLUS progression

Last updated: 2026-08-27

Audited snapshot: GitHub `main` through `056170f` (PR #67) contains the
deterministic Habitat V2 world and replay boundary, the Issue #52/#53 forecast
integration, the Issue #54 distillation study, the Issue #55 controller race,
and the Issue #56 V1 and V3 action-risk evidence plus the V2/V3 development
lineage and V4 diagnostic groundwork. Every learned lane remains simulation-only
and advisory; HMC remains the sole final-command and plant-step authority.

## Canonical lineage

- audited public-main baseline: `056170f` (through PR #67)
- Issue #52/#53 semantic integration: PR #62, with the reviewed HMC V2 source
  package and full development lineage recorded by PR #63
- corrected Issue #54 distillation evidence: PR #64
- Issue #55 controller-race evidence: PR #65
- Issue #56 V1 and V3 evidence plus V2/V3 development lineage: PR #66
- Issue #56 V4 diagnostics-only groundwork: PR #67
- historical 2026-08-18/19 closed-loop advisory disposition: PR #68 preserves
  the off-main PR #50/#59 identities and evidence boundary without integrating
  their obsolete ensemble stack

## Current research findings

- Issue #53 qualified a separate forecast-only model for the frozen
  independent-dropout contract. Correlated or mixed dropout, resource-gauge dropout,
  adversarial channel loss, deployment, and actuator authority remain outside
  that result.
- Issue #54 found that smaller students can retain forecast accuracy, while the
  tiny MLP can lose action-ranking quality despite passing the accuracy gate.
- Issue #55 found that the point-model arm improved mean comfort but worsened
  mean normalized safety exposure sharply; HMC authority prevented a command
  bypass but did not prevent those admitted proposals from worsening safety.
- Issue #56 V3 passed its bounded six-family risk-filtered safety gate but made
  only two proposals and abstained 76 times. V4 is diagnostic groundwork only:
  model training, export, quantization, integration, and threshold changes are
  not authorized.
- The older five-model V3 ensemble is historical development evidence from a
  former forecast stack. Its retained file records 119 runs whose aggregate
  threshold-exceedance and terminal-status fields match V2; resource, trace,
  final-state, proposal, and override fields often differ. Current `main`
  neither carries that ensemble nor supports rerunning the campaign. See the
  [historical evidence
  index](docs/evidence/closed-loop-advisory-historical-index.md).

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
- historical verification snapshot: 571 project-environment tests passed on
  2026-08-12; Ruff 0.14.10 and compile checks also passed

## Historical Habitat V2 build record (2026-08-12 snapshot)

The statuses and branch names below preserve the development state recorded on
2026-08-12. They are not the current `main` status; later merged evidence is
summarized above.

### 1. Explicit multizone air network

Status at 2026-08-12: corrected and independently approved local candidate
Branch at 2026-08-12: `ben/habitat-v2-receipt-authority-fix`
Target version at 2026-08-12: `0.5.0`

Deliverables:

- eight-zone notional habitat
- supply and return topology
- fan curve and deterministic operating-point solver
- per-zone motorised dampers
- pressure, airflow and fan-power telemetry
- physical placement metadata
- scenario-v3 and trace-v3
- old-contract byte preservation

The first candidate at `5df56c0` was rejected because standalone accounting
validation accepted a causally false but internally coherent zero-flow receipt.
The correction at `6cbb8a4` requires exact pre-step state, recomputes the
canonical transition and rejects omitted or forged network evidence. Its
bounded independent rereview returned `APPROVE` with no release-blocking
regression.

Verification on the corrected candidate bytes:

- repository suite: `601 passed in 121.14s`
- Ruff 0.14.10: passed
- Python 3.11 compilation: passed
- wheel and source distribution built as `0.5.0`
- clean external wheel import under `python -I`: passed
- installed eight-zone CLI run repeated byte-for-byte
- installed trace: 5 rows, 8 zones, final step 4
- installed trace SHA-256:
  `dd3b3a579f5eaa8b08b0ffa5a230f5ef833f39233dcefc07a12e2ad4d6b3bd8d`
- wheel SHA-256:
  `a1fd00a63f0026afc01b827ab795d072e92a64e2c2d8b3aa362a7262d4f06eb4`
- source distribution SHA-256:
  `f94bc229cbcf55b9316a22229e9c4dc663464e2d36fdf5e92cf5c43810bf05f1`
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

Status at 2026-08-12: finding-specific local correction complete; replacement
freeze and bounded correction rereview gate publication
Branch at 2026-08-12: `ben/habitat-v2-fault-sensors`
Target version at 2026-08-12: `0.6.0`

Deliverables:

- scenario-v4, trace-v4 and equations-v3 identities
- fan degradation and per-zone supply-resistance increase
- damper jam with previous-achieved-position hold and post-fault slew resumption
- primary and secondary deterministic sensor observations
- sensor bias/drift and stuck-observation memory
- command, achieved actuator and effective-performance separation
- observed telemetry, disagreement and evaluator-truth separation
- deterministic fault manifests, receipts and compound replay
- checked-in eight-zone compound-fault scenario

The immutable candidate at `940608b` was rejected because it clamped
truth-plus-noise before applying sensor bias. This violated the frozen order of
truth plus noise, then active sensor fault, then one final channel clamp. The
replacement defers clamping until after fault application while retaining stuck
sensor memory as the previous completed final observation.

Correction evidence before replacement freeze:

- RED lower-bound case: observed `1000.0`, required `0.0`
- RED upper-bound case: observed `999000.0`, required `1000000.0`
- focused bias-boundary, ordinary bias, stuck-memory and compound replay family:
  `5 passed`
- complete fault/sensor plus CLI boundary: `34 passed`
- version contract: `1 passed`
- full repository suite: `630 passed in 117.72s`
- Ruff 0.14.10, Python compilation, `uv lock --check` and
  `git diff --check`: passed

Rejected-candidate source verification retained for historical identity:

- focused fault/sensor, CLI and version tests: `33 passed`
- complete Habitat V2 suite: `157 passed`
- full repository suite: `628 passed in 118.98s`
- Ruff 0.14.10, compilation, lock and `git diff --check`: passed
- checked-in compound trace: 5 rows, 8 zones, final step 4
- active-fault counts by row: `0,5,5,4,0`
- compound trace SHA-256:
  `7151a62b5db6c001d4131d1711c53a63f2fc3d57444b46c823f1c1bda70e0ded`
- replacement package artifacts, full-repository verification and rereview are
  required before the corrected candidate can be approved

### 3. Scenario families and corpus-v3

Status at 2026-08-12: pending

Deliverables:

- mission-mode and occupancy schedules
- equipment and metabolic load schedules
- single and compound fault families
- whole-scenario train, validation and blind-test separation
- candidate-action rollout records
- simulator, scenario, feature and target provenance
- raw deterministic traces suitable for later model work

### 4. Judge-facing viewer and packaging

Status at 2026-08-12: pending

Deliverables:

- browser-based Three.js habitat cutaway
- components and airflow bound to real IDs
- replay timeline and fault visualization
- baseline and deterministic-control comparison
- offline assets and CPU-only path
- one-command local demo
- clean installation and replay evidence

### 5. Native Arm64 evidence

Status at 2026-08-12: pending

Deliverables before model training:

- Azure Arm64 machine receipt
- native simulator and viewer smoke check
- reproducible environment metadata
- benchmark harness ready for later model candidates

## Current stop boundary

The repository subsequently merged issue-specific protocols and research work;
that history does not create blanket authorization for further experiments.
The Issue #56 V4 draft and code provide diagnostic groundwork for the frozen
V3 receipt, but the draft is explicitly pending authorization. It does not
authorize learned-model training, export, quantization, integration, or
threshold changes; any such work requires a new explicit gate.

## Verification policy

- focused tests during each slice
- one complete suite per meaningful PR boundary
- no repeated broad test runs without new evidence
- environment-bound commands use locked `uv run ... python -m pytest`, not a
  global `pytest` executable
- one bounded independent review per slice
- PR publication and merge require explicit repository-owner authorization and
  diff-grounded review

## Submission claims boundary

AEOLUS is a deterministic, reduced-order, physics-informed research simulator and AI-control evaluation environment. It is not CFD, a certified spacecraft digital twin, flight software or real-world validation.

## Habitat V2 operational observability qualification (inserted evidence gate)

The bounded V5 observability slice is implemented at package version `0.8.0`.
It qualifies matched open-loop operational projections without evaluator-truth
leakage; provenance-binds scenario/run/source-trace/fixture identities and
rejects manifest/trace substitution before scoring. It uses explicit treatment
IDs, an immutable ordered manifest with declared scoring treatment, decision-time
completed-row latency, and separate abnormality/localisation/exact-identification
answers. Fixtures include clearance, bounded recovery, and a post-recovery tail;
hard-negative and aggregate artifacts bind their exact operational provenance,
ordered report/grading inputs, and explicit denominators. The checked V1–V5
simulator lineage remains unchanged.

## Habitat V2 actuator feedback (PR E)

The V5 bounded slice is implemented at package version `0.7.0`. It keeps the
V1–V4 lineage closed, models achieved cooling/oxygen response and effective
delivery faults, emits deterministic operational feedback, and exposes the
validated `advance_one_step_with_command` boundary.
