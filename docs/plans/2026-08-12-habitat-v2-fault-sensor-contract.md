# Habitat V2 deterministic fault and sensor contract

Date: 2026-08-12
Status: approved for isolated local implementation
Branch: `ben/habitat-v2-fault-sensors`
Base: `5df56c0327a1557a99cfca085ce620f042cccb88`

## Plain-language goal

Build a habitat run in which hidden ventilation faults change the real world,
redundant sensors can disagree with each other and with truth, and every effect
can be replayed exactly. The operating trace must make the primary and secondary
sensor feeds visible while keeping simulator truth in a separate evaluator
receipt. No learned model is built or trained in this slice.

## Options considered

1. **Add optional fault fields to scenario-v3 and trace-v3.** Rejected. The
   air-network commit froze those closed schemas and their bytes.
2. **Inject faults from a demo-only script.** Rejected. The scenario digest,
   replay validator and future corpus provenance would no longer own the full
   experiment.
3. **Add scenario-v4 and trace-v4 alongside v1-v3.** Chosen. It preserves every
   older identity and puts fault schedules, observations and evaluator truth
   under one deterministic contract.

## Version dispatch

| Scenario | Trace | Equations | Meaning |
|---|---|---|---|
| v1 | v1 | equations-v1 | frozen two-zone SI reference |
| v2 | v2 | equations-v1 | frozen operating-mode context |
| v3 | v3 | equations-v2 | frozen multizone fan/path/damper network |
| v4 | v4 | equations-v3 | v3 world plus deterministic faults and redundant sensors |

The v4 scenario adds exactly two top-level fields to the v3 shape:

- `sensor_model`
- `fault_profiles`

Unknown and missing fields fail closed at every level. V1-v3 parsers, identity
derivation, trace fields and known bytes remain unchanged.

## Step and interval semantics

A profile interval is half-open: `[start_step, end_step)`.

- `start_step >= 1`
- `end_step <= steps + 1`
- `start_step < end_step`
- a profile active at step `n` affects the transition whose emitted row is
  `step == n`
- row zero is always a healthy initial observation

Linear profiles use

```text
progress = (step - start_step) / (end_step - start_step - 1)
```

when the active interval has at least two steps. A one-step profile applies its
`end_*` value. Endpoints are therefore exact and deterministic.

## Physical fault types

### `fan_speed_degradation`

Fields:

- `id`, `type`, `start_step`, `end_step`
- `start_multiplier`, `end_multiplier`

Both multipliers are in `(0, 1]`. The active multiplier scales achieved fan
speed only for the fan-curve solve:

```text
effective_fan_speed = actual_fan_speed * active_multiplier
```

The commanded and slewed actuator positions remain visible as
`actual_fan_speed_fraction`. Airflow and fan power use effective speed, which is
recorded separately as `effective_fan_speed_fraction` in v4 network receipts.
This is a reduced-order drive-effectiveness fault, not a motor thermal or
bearing model.

### `branch_resistance_increase`

Fields:

- `id`, `type`, `zone_id`, `start_step`, `end_step`
- `start_multiplier`, `end_multiplier`

Multipliers are finite and at least `1`. The active multiplier scales that
zone's open supply-path resistance before the operating-point solve. Return and
shared resistance remain unchanged.

### `damper_jam`

Fields:

- `id`, `type`, `damper_id`, `start_step`, `end_step`

During the active interval the damper holds its previous achieved position. It
does not jump to a fabricated jam angle. When the profile clears, normal slew
resumes from the held position.

## Sensor model

Each zone has two deterministic sensor heads: `primary` and `secondary`. Both
measure the five existing channels:

- `temperature_k`
- `pressure_pa`
- `co2_ppm`
- `o2_mole_fraction`
- `relative_humidity`

`sensor_model` contains:

- integer `random_seed`
- `primary_noise_amplitude`
- `secondary_noise_amplitude`

Each amplitude object has exactly the five channel names and finite,
non-negative values in the channel's own units. A stateless SHA-256 sample keyed
by scenario seed, zone, head, channel and emitted step produces a stable value in
`[-1, 1)`. Observations are truth plus amplitude times sample, then the active
sensor fault, then clamped to the channel's physical output range.

## Sensor fault types

### `sensor_bias_drift`

Fields:

- `id`, `type`, `zone_id`, `sensor_head`, `channel`
- `start_step`, `end_step`, `start_bias`, `end_bias`

The interpolated bias is additive in the channel's declared unit.

### `sensor_stuck`

Fields:

- `id`, `type`, `zone_id`, `sensor_head`, `channel`
- `start_step`, `end_step`

The active head/channel emits its previous completed-row value. A profile that
starts at step one therefore holds the healthy row-zero value. Sensor memory is
runner-local observation state and never changes physical plant state.

## Composition and conflicts

Simultaneous faults on different targets are allowed and form a compound
scenario. Profiles targeting the same authority may not overlap:

- one fan degradation at a time
- one resistance profile per zone at a time
- one jam per damper at a time
- one sensor fault per `(zone, head, channel)` at a time

A branch resistance increase and damper jam on the same branch may overlap,
because they act on different physical mechanisms. A physical fault and any
sensor fault may overlap. Profile IDs are unique.

## Trace-v4 boundary

Trace-v4 retains all trace-v3 fields and adds:

### `sensor_disagreement`

For every zone:

- `secondary`: the five secondary observed channels
- `primary_minus_secondary`: the five signed residuals

The existing `telemetry` field becomes the primary observed feed. It no longer
represents truth on v4.

### `fault_receipt`

For row zero, `fault_receipt` is `null`. Every completed row contains exactly:

- `truth_telemetry`: five physical truth channels per zone
- `primary_residual`: primary observed minus truth per zone/channel
- `secondary_residual`: secondary observed minus truth per zone/channel
- `active_faults`: a deterministically ordered list of generic entries with
  `fault_id`, `fault_type`, `target_id`, `effect_name`, `effect_value`

`fault_receipt` is evaluator and demo truth. It is not part of the future model
feature projection. The model-facing data contract must select fields
explicitly and may use only operational telemetry, disagreement, completed
actions and completed network receipts.

## Data flow

For emitted step `n`:

1. Select the scenario timeline segment and active v4 profiles for `n`.
2. Slew fan and dampers from prior achieved actuator state.
3. Hold jammed dampers, then apply fan and branch effectiveness to the network
   solve.
4. Advance recirculation, species, thermal and electrical state.
5. Validate accounting and physical invariants against truth.
6. Compute healthy primary and secondary sensor observations from truth.
7. Apply active sensor bias/stuck effects using only completed sensor memory.
8. Emit primary telemetry, secondary/disagreement telemetry, evaluator truth
   and the active-fault receipt.
9. Canonicalise and validate the whole trace by byte-for-byte replay.

## Acceptance tests

1. V4 has distinct scenario, trace and equation identities.
2. V1-v3 known scenario digests and trace bytes do not change.
3. Unknown fields, IDs, channels, heads, bad intervals, non-finite values and
   overlapping same-target profiles fail closed.
4. Fan degradation lowers effective speed and delivered flow without changing
   the commanded fan position.
5. Branch blockage raises branch resistance and changes the shared operating
   point.
6. A jammed damper holds its previous achieved position and resumes slew after
   clearing.
7. Primary and secondary healthy observations are deterministic and disagree
   only within declared amplitudes.
8. Bias/drift affects only the targeted head/channel, not physical truth.
9. A stuck sensor holds its prior observation while truth continues changing.
10. Simultaneous physical and sensor faults replay byte-for-byte.
11. `fault_receipt` truth and residual arithmetic is recomputed and validated.
12. Any finite mutation of observation, truth, active-fault effect or physical
    output is rejected by full deterministic replay.
13. Full repository tests, Ruff, compilation and `git diff --check` pass.

## Explicit non-goals

- no stochastic mutable RNG
- no CFD or component failure-probability claim
- no leak/depressurisation or combustion model
- no controller, planner or safety-governor changes
- no corpus generation
- no model architecture, training, tuning, selection or quantisation
- no Arm benchmark and no viewer in this branch

## Execution gate

Approved for isolated local implementation on
`ben/habitat-v2-fault-sensors`. Stop after deterministic v4 scenario/trace,
compound-fault replay, tests, documentation, versioned package evidence and one
bounded review. Do not push, merge, publish, train a model or provision cloud
hardware without the later gate.
