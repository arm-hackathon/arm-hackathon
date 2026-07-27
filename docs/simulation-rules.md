# ICARUS simulation rules

This document defines the current ICARUS simulator. It is a deterministic,
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

## Schema v7

A scenario is closed-schema JSON with exactly these top-level keys:

| Key | Meaning |
|---|---|
| `version` | Must be `7`. |
| `zones` | Zone specifications. Exactly one must be `air_processing`. |
| `connections` | Directed paths in the hub layout. |
| `control` | CO₂ thresholds and actuator command bounds. |
| `actuator` | Full-stroke duration and abstract power values. |
| `simulation` | Deterministic source-noise seed. |
| `air_system` | Shared delivery capacity and scrubber removal fraction. |
| `fault_profiles` | Explicit list of deterministic fault profiles; `[]` means healthy. |

Every zone contains `id`, `label`, `preset`,
`co2_generation_per_second`, `co2_generation_epsilon`,
`co2_noise_correlation`, `occupancy_profile` and `air_volume`.

Every connection contains `id`, `from`, `to`, `max_airflow` and `health`.
Connection health is a hidden static physical constraint in `0.0..1.0`; it is
not emitted in replay telemetry.

The layout is intentionally narrow:

```text
non-processing zone ── outbound meter ──> processing bay
non-processing zone <── paired return ─── processing bay
```

Each non-processing zone requires exactly one path in each direction. No
self-loops, bypass paths or multiple directed paths between the same zone pair
are accepted.

## Gradual primary-fan degradation

`fault_profiles` is required even for a healthy scenario. The only accepted
profile currently is:

```json
{
  "type": "gradual_primary_fan_degradation",
  "connection_id": "cabin_a_to_processing",
  "start_tick": 20,
  "end_tick": 80,
  "end_effectiveness": 0.4
}
```

Rules:

- the target must be an existing outbound path ending at the processing bay;
- only one profile can target a connection;
- `start_tick` and `end_tick` are positive integers, with `end_tick > start_tick`;
- `end_effectiveness` is finite and in `[0.0, 1.0)`;
- profile type and fields are allowlisted; unknown fields are rejected.

For measured tick `t`, effectiveness is deterministic:

```text
t <= start_tick: 1.0
t >= end_tick:   end_effectiveness
otherwise:       linear interpolation from 1.0 to end_effectiveness
```

The profile is not accumulated into mutable state. It is calculated from the
measured tick every replay. Warm-up has no profile injection, so `start_tick`
always refers to the visible trace tick.

## One simulation tick

Time advances in fixed one-second ticks.

1. Each zone adds its occupancy-scaled, seeded and correlated CO₂ source.
2. Sensors read `co2_mass / air_volume` after source addition.
3. Each non-processing-zone controller maps its sensor value to a bounded
   setpoint.
4. Its actuator moves towards that setpoint by at most
   `1 / full_stroke_seconds` of a stroke.
5. Each circulation loop computes **requested airflow** from only nominal
   physical capacity and measured actuator position:

   ```text
   requested = min(outbound.max_airflow, inbound.max_airflow) * actual_position
   ```

6. Both loop legs constrain physical delivery. ICARUS derives a hidden static
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
can consume generated traces from all three shipped scenarios and plots
requested/delivered airflow plus residuals.

Fault state, fault effectiveness, static health, random seed and source-noise
state are deliberately absent from trace telemetry. `model_feature_row()` has
its own strict projection:

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
uv run python -m icarus scenarios/standard_habitat.json out/standard.jsonl
uv run python -m icarus.visualise out/standard.jsonl out/standard.html
```

## Deliberately absent

ICARUS currently has no model or ONNX path, quantisation, governor, redundant
fan, recovery controller, Arm benchmark, dashboard, API, cloud service or
hardware connection.
