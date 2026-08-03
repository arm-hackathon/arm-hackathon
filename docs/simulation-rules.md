# Project AEOLUS simulation rules

This document defines the current AEOLUS simulator. It is a deterministic,
abstract hub-layout CO₂ and airflow model, not a spacecraft or life-support
model.

## Scope and units

The model contains circulation through one shared air-processing bay and a CO₂
scrubber. Oxygen, pressure, temperature, humidity, contaminants, leaks, fire,
hardware protocols and external environment are out of scope.

All values are abstract units:

- `co2_units` — amount of CO₂, not ppm or kilograms;
- `airflow_units_per_second` — circulation amount per tick, not a real flow
  rate;
- actuator power — an abstract reporting value, not an electrical measurement.

## Schema v9

A scenario is closed-schema JSON with exactly these top-level keys:

| Key | Meaning |
|---|---|
| `version` | Must be `9`. |
| `zones` | Zone specifications. Exactly one must be `air_processing`. |
| `connections` | Directed paths in the hub layout. |
| `control` | CO₂ thresholds and actuator command bounds. |
| `actuator` | Full-stroke duration and abstract power values. |
| `simulation` | Deterministic source-noise seed. |
| `telemetry` | Deterministic measurement-noise and bias fractions. |
| `air_system` | Shared delivery capacity and scrubber removal fraction. |
| `fault_profiles` | Explicit list of deterministic fault profiles; `[]` means healthy. |

Every zone contains `id`, `label`, `preset`,
`co2_generation_per_second`, `co2_generation_epsilon`,
`co2_noise_correlation`, `occupancy_profile` and `air_volume`.

Every connection contains `id`, `from`, `to`, `max_airflow` and `health`.
Connection health is a hidden static physical constraint in `0.0..1.0`; it is
not emitted in replay telemetry.

`telemetry` contains exactly `airflow_noise_fraction`,
`airflow_bias_fraction`, `airflow_drift_fraction`,
`actuator_position_noise_fraction`, `co2_sensor_noise_fraction`,
`co2_sensor_bias_fraction`, and `co2_sensor_drift_fraction`. Each is a finite
number in `0.0..1.0`. Missing or unknown settings are rejected. Zero for all
seven preserves the earlier numerical behaviour. Schema-v8 input is rejected.

The layout is intentionally narrow:

```text
non-processing zone ── outbound meter ──> processing bay
non-processing zone <── paired return ─── processing bay
```

Each non-processing zone requires exactly one path in each direction. No
self-loops, bypass paths or multiple directed paths between the same zone pair
are accepted.

## Fault profiles

`fault_profiles` is required even for a healthy scenario (`[]`). Three
deterministic profile types are accepted. Profile type and fields are
allowlisted per type; unknown fields are rejected.

### `gradual_primary_fan_degradation`

```json
{
  "type": "gradual_primary_fan_degradation",
  "connection_id": "cabin_a_to_processing",
  "start_tick": 20,
  "end_tick": 80,
  "end_effectiveness": 0.4
}
```

For measured tick `t`, effectiveness is deterministic:

```text
t <= start_tick: 1.0
t >= end_tick:   end_effectiveness
otherwise:       linear interpolation from 1.0 to end_effectiveness
```

### `blocked_path`

```json
{
  "type": "blocked_path",
  "connection_id": "cabin_b_to_processing",
  "start_tick": 30,
  "blocked_effectiveness": 0.05
}
```

A sudden blockage: effectiveness is `1.0` before `start_tick` and
`blocked_effectiveness` (finite, in `[0.0, 1.0)`) from `start_tick` onward.

### `frozen_sensor`

```json
{
  "type": "frozen_sensor",
  "zone_id": "lab",
  "start_tick": 30
}
```

From `start_tick` onward the target zone's sensor holds its latent physical
reading from the first frozen tick. The true concentration keeps evolving
underneath. Fixed bias, bounded drift and per-tick readout noise are then
applied downstream of the held value, and that measured value drives the
controller and appears in telemetry. The target must be an existing
non-processing zone.

Shared rules:

- connection faults (`gradual_primary_fan_degradation`, `blocked_path`) must
  target an existing outbound path ending at the processing bay; only one
  connection fault can target a connection;
- only one sensor fault can target a zone;
- tick fields are positive integers; for the degradation profile
  `end_tick > start_tick`;
- profile type and fields are allowlisted per type; unknown fields are
  rejected.

Profiles are not accumulated into mutable state. Connection effectiveness and
sensor-freeze membership are calculated from the measured tick every replay.
Warm-up has no profile injection, so `start_tick` always refers to the
visible trace tick.

## One simulation tick

Time advances in fixed one-second ticks.

1. Each zone adds its occupancy-scaled, seeded and correlated CO₂ source.
2. A latent sensor reads `co2_mass / air_volume` after source addition. An
   active `frozen_sensor` holds that latent value from its first frozen tick.
3. CO2 measurement adds fixed per-zone bias, bounded piecewise-linear drift
   and per-tick readout noise. Effects are scaled by the controller upper
   threshold and the result is clamped non-negative. The measured value—not
   latent truth—drives each non-processing-zone controller.
4. Its actuator moves towards that setpoint by at most
   `1 / full_stroke_seconds` of a stroke.
5. Each circulation loop computes **physical requested airflow** from only
   nominal physical capacity and physical actuator position:

   ```text
   requested = min(outbound.max_airflow, inbound.max_airflow) * actual_position
   ```

6. Both loop legs constrain physical delivery. AEOLUS derives a hidden static
   health factor from the two healthy path capacities:

   ```text
   static_health = min(outbound.max_airflow * outbound.health,
                       inbound.max_airflow * inbound.health)
                   / min(outbound.max_airflow, inbound.max_airflow)
   ```

7. The target profile, if any, provides hidden fault effectiveness. Before the
   shared fan is considered:

   ```text
   provisional_delivered = requested * static_health * fault_effectiveness
   ```

8. If provisional delivery exceeds `shared_airflow_capacity`, all loops receive
   the same deterministic proportional capacity scale. The final value is:

   ```text
   delivered = provisional_delivered * capacity_scale
   airflow_residual = requested - delivered
   ```

   A loop's outbound and return paths report identical delivered airflow.
   Delivery never exceeds request or shared capacity.

9. All zones extract CO₂ from the same pre-transfer state. Extraction mixes in
   the processing bay; the scrubber captures its configured fraction and the
   rest returns in proportion to delivered loop airflow.

10. After the physical step, actuator/airflow projection creates replay telemetry.
    Samples are SHA-256-derived uniform values in `[-1, 1)`, keyed by scenario
    seed, entity, channel, and tick. Bias uses tick zero. Drift linearly
    interpolates independently keyed samples at 20-tick anchors. Actuator
    position receives bounded per-tick noise. Each airflow meter receives a
    fixed connection bias, bounded drift and independent per-tick noise.
    Observable request is recomputed from observable position;
    observable delivery is clamped to `0..request`; residual is recomputed; and
    saturated observations are proportionally scaled to shared capacity. This
    this projection never mutates mass, actuator, controller, physical airflow,
    or fault state; independently metered paired legs may report different
    noisy observations. CO2 measurement occurs earlier because it is a genuine
    controller input; it can therefore change subsequent control and physical
    evolution without directly rewriting latent mass.

CO₂ is conserved: generated mass is split only between airborne zone mass and
the processing bay's captured store.

## Standard run and determinism

`STANDARD_RUN` has 120 measured ticks and a 60-tick unrecorded warm-up. The
warm-up uses the declared source-noise seed with separate historical ticks,
holds initial occupancy conditions, then resets visible tick and captured CO₂
while retaining settled physical state.

No wall clock or unseeded random source participates in a run. The same
scenario produces byte-identical JSONL output when run with the same code,
Python runtime and platform.

## Trace and model projection

Each trace row contains `tick`, `zones`, `connections`, `actuators` and
`system`. Connection telemetry has exactly:

```text
requested_airflow
delivered_airflow
airflow_residual
```

System telemetry has exactly:

```text
shared_airflow_capacity
total_requested_airflow
total_delivered_airflow
capacity_scale
```

Trace writers validate the observable allowlist before serialising a row. The
visualiser independently rejects undeclared connection telemetry. The visualiser
can consume generated traces from all five shipped scenarios and plots
requested/delivered airflow plus residuals.

Fault state, fault effectiveness, static health, random seed, source-noise
state and measurement-noise state are deliberately absent from trace
telemetry. `model_feature_row()` has its own strict projection:

- zone sensor CO₂ concentration;
- actuator setpoint, actual position, tracking residual and power;
- requested, delivered and residual airflow.

Presentation needs never add model features implicitly.

## Validation

`config.py` rejects scenarios with clear `ValueError`s for unsupported versions,
unknown fields, invalid numeric values, missing hub paths, bad occupancy ranges,
invalid connection health, invalid control/actuator settings and malformed fault
profiles. `visualise.py` independently validates replay shape and tick sequence
before it writes a report.

## Local commands

```bash
uv run --extra dev python -m pytest
mkdir -p out
uv run python -m aeolus scenarios/standard_habitat.json out/standard.jsonl
uv run python -m aeolus.visualise out/standard.jsonl out/standard.html
uv run python -m aeolus.corpus out/corpus \
  scenarios/standard_habitat.json scenarios/high_demand_healthy.json \
  scenarios/primary_fan_degradation.json scenarios/blocked_path.json \
  scenarios/frozen_sensor.json
uv run python -m aeolus.evaluate out/corpus/corpus.jsonl \
  scenarios/standard_habitat.json scenarios/high_demand_healthy.json \
  scenarios/primary_fan_degradation.json scenarios/blocked_path.json \
  scenarios/frozen_sensor.json
uv run python -m aeolus.corpus --v2 out/corpus-v2 scenarios/families.json
uv run python -m aeolus.evaluate --v2 out/corpus-v2/corpus.jsonl \
  scenarios/families.json \
  --expected-family-manifest-sha256 828880e3257036ff2897a6cc2668c25b87734f8c57004ed36e62b2b6d66f6541 \
  --split test
# Historical sweep-v2 commands are intentionally omitted here. For the current
# development-selection → final-evaluation protocol, use:
# docs/protocol-v3-acceptance.md
uv run python -m aeolus.detector predict \
  artifacts/aeolus_fault_detector.json scenarios/standard_habitat.json
```

`aeolus.corpus` writes a labelled window corpus (`corpus.jsonl`) and a
`manifest.json` into the given output directory. `aeolus.evaluate` grades the
rule baseline against that corpus and prints accuracy, confusion and
detection-latency metrics as JSON. See
`docs/telemetry-contract.md` for the corpus leakage boundary.

## Deliberately absent

AEOLUS has experimental softmax and temporal-MLP training plus FP32 ONNX export. It has no
INT8 quantisation, governor, redundant fan, recovery controller, Arm benchmark,
dashboard, API, cloud service or hardware connection.
