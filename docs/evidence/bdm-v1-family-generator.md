# BDM-v1 Scenario Family Generator v2 — Evidence Record

Date: 2026-09-05
Issues: #72 (parts 1–3); contract: #70
(`contracts/habitat_v2_bdm_v1_benchmark_contract_v1.json`); provenance: #71
(`contracts/habitat_v2_physics_provenance_v1.json`)
Status: development evidence only. This record proves deterministic roster
generation, custody, and blind-size justification. It proves nothing about
model quality, hardware, qualification, or real-world control.

## What was built

1. **Generator core** (`src/aeolus/habitat_v2/bdm_v1_families.py`):
   deterministic generator for causally distinct Habitat V2 development
   families. Twelve causal templates span the five contract strata; draws come
   only from Issue #71 manifest bands and the explicitly declared generator
   vocabularies below. Families are pure functions of (generator version,
   registry seed, group index, variant index, feasibility attempt).
2. **Roster + custody registry** (`scripts/generate_bdm_v1_families.py` →
   `contracts/habitat_v2_bdm_v1_family_custody_v1.json`): 80 causal groups /
   160 families; partition quotas TRAIN 40, DEV 16, CALIBRATION 12,
   BLIND_FINAL 12 groups; group-disjoint custody; BLIND_FINAL sealed with
   digests only.
3. **Blind-size power pilot** (`scripts/run_bdm_v1_blind_power_pilot.py`):
   DEV-only pilot replacing the contract's `TBD_FROM_PILOT` blind-size
   justification with a frozen, receipt-bound rule.

## Template and strata structure

| Template | Stratum | Groups |
| --- | --- | --- |
| t01_healthy_steady | no_action | 7 |
| t02_o2_regime_stress | useful_opportunity | 7 |
| t03_fan_degradation | useful_opportunity | 7 |
| t04_branch_resistance | useful_opportunity | 7 |
| t05_cooling_delivery_loss | useful_opportunity | 7 |
| t06_oxygen_delivery_loss | useful_opportunity | 7 |
| t07_scrubber_condenser_stress | useful_opportunity | 7 |
| t08_sensor_defect_bundle | sensor_failure | 7 |
| t09_actuator_failure | actuator_failure | 6 |
| t10_compound_physical_sensor | compound | 6 |
| t11_low_reserve_depletion | useful_opportunity | 6 |
| t12_schedule_transition | useful_opportunity | 6 |

Strata counts by partition (groups): TRAIN 27/4/3/3/3, DEV 11/1/2/1/1,
CALIBRATION 8/1/1/1/1, BLIND_FINAL 8/1/1/1/1 in the order
useful_opportunity / no_action / sensor_failure / actuator_failure / compound.
Every partition covers all five strata.

Each group carries exactly two paired sensor variants (a: healthy sensor
projection; b: declared sensor-defect projection). Paired variants share one
causal group key and one true plant trajectory seed structure; they differ
only in the declared observation-layer treatment. The pilot confirmed this:
matched pairs produced identical true-plant comfort values (e.g.
`bdm-v1-f0080-a` / `bdm-v1-f0081-b`, both 139.639932), because the sensor
defect perturbs observations only, never the plant.

## Declared generator vocabularies

Frozen in the registry (`declared_generator_vocabularies`) and mirrored by
tests: initial-oxygen regime bands (deficient/nominal/high with the
`GENERATOR_O2_EXCESS_THRESHOLD`), fault multiplier bands per fault class,
fault onset classes, fault duration range [16, 64] steps, ramp shapes, sensor
bias bands, and sensor bundle kinds. Explicitly **excluded from drawing**:

- sensor noise amplitudes — the reviewed HMC contract's reset rejects any
  scenario whose environmental sensor noise differs from the contract, so no
  HMC-replayable generator may draw them (see manifest revision below);
- Issue 55/56 race operating-condition bands — bound to those fixtures;
- per-zone load records — scaled by the group-level `load_scale` draw and
  explicit schedule segments instead.

Decision convention: 96-step episodes, 13 decision steps (16..64, stride 4),
risk horizon 32 steps (the V2/Issue #56 convention, not the Issue #55 race
horizon-8 convention).

## Provenance manifest revision (Issue #71 records)

During generator development two manifest defects were found and fixed
(revision note in `docs/evidence/habitat-v2-parameter-provenance.md`):

1. The four `sensor_primary_noise_*` records were reclassified to
   `generator_variable: false` (HMC reset pins them; drawing them yields
   unreplayable scenarios).
2. `sink_temperature_k` and `initial_temperature_k` valid ranges were widened
   so their declared relative bands fit inside them, and
   `load_physics_provenance_manifest` now fails closed on any relative or
   absolute band exceeding its record's valid range.

Post-revision manifest sha256:
`dec91ce7796c8b33cfc317cb95f13f015e22683027a7347e2c2bec7256eb6668`.

## Generation and checks

Command (write-once output; roster regenerated once to embed the frozen
justification — both runs byte-identical apart from the registry
`blind_seal`):

```bash
uv run --locked --python 3.11 --extra dev python \
  scripts/generate_bdm_v1_families.py --output out/bdm-v1-families-v2 \
  --blind-size-justification "<justification string below>"
```

Results:

- 160/160 families passed closed v5 schema validation and the bounded 8-step
  no-proposal HMC feasibility smoke replay at attempt 0 (no redraws);
- group disjointness validated through the Issue #70 contract API;
- regeneration check: every family rebuilt from its recorded attempt with
  byte-identical scenario digests;
- `families_sha256`:
  `16d0a33838a2b4b197371b5526e4cbe3503b043afbd6a0c9b8b4cf48c6b881b8`;
- `scenarios_sha256`:
  `de90dccba1f9c3baea204ecf3a651d64816919b87916e080403e85929dbc86b6`;
- BLIND_FINAL definitions digest (sealed, outcomes `NOT_COMPUTED`):
  `451d97c80887617a0994efd0f7b7ac4c0514ccdf1ddffbbcef27c532ef972f29`;
- registry sha256 before justification freeze (`TBD_FROM_PILOT`):
  `18d39a3dfdf82d1d411c24b9c55340842ff7cc84cfb7139cc6c2a55c9ab3ebb5`;
- final registry sha256:
  `cac69c3c4caf6b3f35172194083b5b08c5499feaba2b2108ba42af372a628074`;
- base scenario sha256:
  `9de0325fbbf99099e02683adb650187876ea7ff41a3dc9525b2042574dd0c588`.

## Blind-size power pilot

Command:

```bash
uv run --locked --python 3.11 --extra dev python \
  scripts/run_bdm_v1_blind_power_pilot.py --output out/bdm-v1-blind-power-pilot-v1
```

Population (declared before execution): the first 8 DEV causal groups in
registry order, both paired variants (16 families). Arm: Issue #55
`rules_only` (HMC hold, zero proposals) with full 96-step shadow-receipt
verification and the preregistered Issue #55 metrics.

Observation — **hold-arm floor effect**: all 16 families produced
`safety_exposure = 0.0`, `safety_violation_steps = 0`, and
`resource_composite = 0.0`. The deterministic HMC hold keeps the true plant
inside all 51 race target bounds on every roster family tested. Group-mean
comfort deviation: mean 119.40912938405414, SD 44.92471247646109.

Because the primary metric floors at zero under hold, a continuous power
analysis on safety exposure differences versus hold is degenerate. The pilot
therefore sizes the blind with a declared zero-event rule on the primary
safety gate plus a continuous proxy diagnostic:

- **R1 (primary, zero-event binomial)**: with N blind groups and zero harmful
  groups observed, the one-sided 95% upper bound on the harmful-group rate is
  `1 - 0.05^(1/N)` = 0.2209 at N=12; a true harmful-group rate of 0.25 is
  rejected at alpha=0.05 because `0.75^12 = 0.0317 <= 0.05` (N=10 would not:
  0.0563). This matches the contract's decision hierarchy, whose first levels
  are zero hard violations and per-family safety non-inferiority.
- **R2 (secondary continuous proxy, diagnostic only)**: detecting a 50%
  relative shift in group-mean comfort deviation with 80% power at two-sided
  alpha=0.05, under the conservative independent-arm difference SD
  `sqrt(2) * 44.92`, requires 9 groups <= 12. Comfort is a declared secondary
  metric; this rule never gates the primary safety verdict. A 25% comfort MDE
  would require about 36 groups and is explicitly not powered by this blind.
- **R3 (contract floor)**: 12 groups materially exceed the contract's
  "more than three independent evaluation condition groups" requirement (4x).

Pilot receipt sha256 (written once, canonical JSON):
`87ad2d33b3b821d119b2771f300174b57392a8f555aff5701c2c2ec0392aef6d`.

Frozen justification string embedded in the registry `blind_seal`:

```text
pilot_receipt:87ad2d33b3b821d119b2771f300174b57392a8f555aff5701c2c2ec0392aef6d:declared_blind_groups=12:harmful_rate_detectable=0.25:alpha=0.05:zero_event_upper_bound=0.2209:comfort_proxy_required_groups=9:contract_floor=3
```

The pilot script fails closed if a re-run ever observes non-zero hold
exposure or resource depletion (the declared rule would be invalid and a
revised preregistered pilot would be required), if the blind size cannot
reject the declared harmful rate, or if R2 exceeds the declared blind size.

## Verification

Run for these commits from a clean checkout:

```bash
uv run --locked --python 3.11 --extra dev python -m pytest -q
uv run --locked --python 3.11 --extra dev ruff check .
uv run --locked --python 3.11 --extra dev python -m compileall -q src tests scripts
uv lock --check
git diff --check
```

Focused suites: `tests/habitat_v2/test_bdm_v1_families.py` (17 tests),
`tests/habitat_v2/test_bdm_v1_custody.py` (9 tests),
`tests/habitat_v2/test_bdm_v1_power_pilot.py` (3 tests),
`tests/habitat_v2/test_physics_provenance.py` (9 tests).

## Boundaries and non-claims

- The BLIND_FINAL partition stays sealed: digests only, no outcome was
  computed anywhere in this work, and opening requires the separate one-shot
  authorization per the Issue #70 contract.
- Blind per-stratum coverage is thin for failure strata (1 group each of
  sensor_failure, actuator_failure, compound, no_action; 8
  useful_opportunity). Blind verdicts are roster-level and per-family; blind
  per-stratum subgroup claims are not supported by this sizing.
- The hold-arm floor effect means zero-exposure holds cannot discriminate
  arms on the primary metric; arm discrimination comes from advisory-arm
  behavior under HMC filtering, which this pilot does not measure.
- The comfort-proxy power figure is a diagnostic bound, not a safety claim.
- Latency, quantization, packaging, and Arm optimization remain deferred to
  Gate 6 per the contract; nothing here exercises them.
- The roster generation corpus builder (feature/label corpus over these
  families) is intentionally not part of Issue #72; it lands as the shared
  Wave 3 deliverable.
- All results are simulator development evidence only; learned components are
  advisory-only; HMC remains the sole final-command, plant-step, and replay
  authority.
