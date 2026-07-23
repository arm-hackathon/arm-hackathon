# ICARUS simulation rules

This document is the single source of truth for what the current simulation
slice does, in plain English. It is a simulation of ideas, not of real
spacecraft hardware.

## Scope

NASA ECLSS material lists cabin-air circulation, CO₂ removal, atmosphere
management, and thermal control as life-support functions. This slice models
only **circulation** and **CO₂ removal**. Oxygen, total pressure,
temperature, humidity, trace contaminants, lunar dust, fire, leaks, and the
external vacuum are held constant and stay out of scope.

The habitat is a **user-editable scenario graph**, not a hard-coded pair of
rooms. The current format supports the simple hub layout only: one
air-processing bay in the middle, every other zone connected to it by one
directed path each way. It is not a general fluid solver.

## Units

All values are abstract simulation units:

- `co2_units` — amount of CO₂. Not ppm, not kilograms, not a safety limit.
- `airflow_units_per_second` — air moved along a path. Not a real flow rate.

No number in this repository is a real spacecraft measurement or a real
safety threshold.

## The scenario file

A scenario is versioned JSON (see `scenarios/standard_habitat.json`) with
seven top-level keys:

| Key | Meaning |
|---|---|
| `version` | Format version. Only `5` is supported. |
| `zones` | List of rooms. |
| `connections` | List of directed air paths between rooms. |
| `control` | CO₂ thresholds and actuator command bounds. |
| `actuator` | Stroke time and abstract power characteristics. |
| `simulation` | Seed used for deterministic source variation. |
| `air_system` | Shared fan capacity and scrubber removal fraction. |

Every zone has exactly these fields:

| Field | Meaning |
|---|---|
| `id` | Unique zone id. |
| `label` | Human-readable name. |
| `preset` | One of `crew_cabin`, `lab`, `air_processing`, `storage`. |
| `co2_generation_per_second` | CO₂ source added each tick (`>= 0`). |
| `co2_generation_epsilon` | Maximum seeded variation above or below the source. |
| `co2_noise_correlation` | Persistence of source variation from one tick to the next. |
| `occupancy_profile` | Scheduled tick ranges that scale the baseline source. |
| `air_volume` | Total air in the zone (`> 0`), used to size the moved share. |

Every connection has exactly these fields:

| Field | Meaning |
|---|---|
| `id` | Unique connection id. |
| `from` | Source zone id. |
| `to` | Target zone id. |
| `max_airflow` | Air the path moves per tick at health 1.0 (`> 0`). |
| `health` | Path health in `0.0..1.0`. `1.0` is fully healthy. |

The control block has four fields:

| Field | Meaning |
|---|---|
| `co2_lower_threshold` | A reading at or below this value requests the minimum command. |
| `co2_upper_threshold` | A reading at or above this value requests the maximum command. |
| `minimum_command` | Lowest permitted actuator command in `0.0..1.0`. |
| `maximum_command` | Highest permitted actuator command in `0.0..1.0`. |

Readings between the thresholds are mapped linearly onto the command range.

The actuator block has three fields:

| Field | Meaning |
|---|---|
| `full_stroke_seconds` | Time required to move from fully closed to fully open. |
| `moving_power` | Abstract power reported while the actuator is travelling. |
| `holding_power` | Abstract power reported while holding position. |

The standard scenario uses seed `7`. Each tick and zone receives an
occupancy-scaled source plus bounded, correlated variation. The same scenario
and seed always produce the same sequence.

The air-system block declares the total airflow available to all room loops
and the scrubber's removal fraction. When total local demand exceeds shared
capacity, every request is scaled proportionally. The standard scenario has
24 airflow units of shared capacity.

Exactly one zone must use the `air_processing` preset; it is the common
mixing and scrubber bay.

## The standard habitat

`scenarios/standard_habitat.json` is the reference hub layout: two crew
cabins, a lab with a smaller activity-driven source, and the air-processing
bay.

```text
                cabin_a (Crew Cabin A, CO₂ source)
                    |  ^
       max out 10.0 |  | max return 10.0
                    v  |
   cabin_b  <====  processing  ====>  lab
   (Crew Cabin B,  (Air Processing   (Lab)
    CO₂ source)     Bay, scrubber)
   max 10.0 each way  max 8.0 each way
```

Every non-processing zone has one directed path to the processing bay and
one directed path back. All six connections start at health `1.0`. The room
loops draw from the same 24-unit fan capacity, and their extracted air mixes
before being scrubbed and returned. This couples all room conditions.

## One tick, in order

Time advances in fixed 1-second ticks. Each tick, exactly this happens:

1. Every zone adds its occupancy-scaled, correlated CO₂ source mass.
2. An idealised sensor reads `CO₂ mass / air volume` after the source is added.
3. For each non-processing zone, the controller maps that reading linearly
   between the configured minimum and maximum actuator setpoints.
4. Each actuator moves towards its setpoint by no more than
   `1 / full_stroke_seconds` of its stroke per tick.
5. Each loop requests
   `max_airflow * health * actual_position`, limited by its weaker path.
6. If total demand exceeds shared capacity, every loop is scaled by the same
   allocation factor.
7. Every zone simultaneously sends CO₂ mass into the common processing flow.
8. The scrubber captures its configured fraction of the combined mass.
9. Remaining CO₂ is mixed and returned to rooms in proportion to allocated
   airflow. This allows one room's source to affect the others.
10. The trace records sources, occupancy, mass, concentration, actuator
    behaviour, requested and allocated airflow, and shared-capacity use.

A loop's return path reports the same actual airflow as its outbound path.
The weaker leg governs the loop, so a path at health `0.0` moves no air and
scrubs nothing.

CO₂ is conserved: what the zones generate is always split between airborne
CO₂ and the processing bay's captured store. Nothing appears or vanishes.

## The standard run

`STANDARD_RUN` in `src/icarus/scenario.py` declares the run constants:

| Constant | Value | Meaning |
|---|---|---|
| `total_ticks` | 120 | Length of a run. |
| `warmup_ticks` | 60 | Unrecorded pre-roll under initial occupancy conditions. |
| `crew_cabin_co2_concentration_ceiling` | 0.30 | Declared crew-cabin concentration ceiling. |

The warm-up settles concentration, actuator position, source correlation and
shared airflow before the measured trace begins. It uses a separate seeded
noise sequence, holds each zone at its tick-1 occupancy multiplier, then resets
the visible tick and captured-CO₂ counter. Across the subsequent scheduled
occupancy changes, healthy cabin concentration remains below `0.30`.

## Validation

`src/icarus/config.py` rejects a scenario with a clear `ValueError` (the
CLI prints it and exits non-zero) when it has:

- an unsupported or missing version;
- an unexpected field at the scenario, block, zone, connection or occupancy
  period level;
- missing or invalid simulation seed;
- missing or invalid shared airflow or scrubber settings;
- missing, non-finite or inconsistent CO₂ control settings;
- missing, non-finite or invalid actuator settings;
- no zones, or no `air_processing` zone;
- more than one `air_processing` zone;
- a duplicate zone id or connection id;
- an unsupported preset;
- a non-positive air volume, negative CO₂ source or invalid noise setting;
- malformed or overlapping occupancy periods;
- a connection referencing an unknown source or target zone;
- a connection that loops from a zone back to itself;
- a non-positive max airflow;
- a health outside `0.0..1.0`;
- a non-finite number (`NaN` or `Infinity`, which Python's JSON parser
  accepts) in any numeric field;
- a connection that does not touch the air-processing bay (hub layout
  only), or more than one directed path between the same zone pair;
- a non-processing zone missing either its path to processing or its
  return path from processing.

The HTML visualiser also rejects a malformed replay trace. A trace must use
positive, consecutive integer ticks beginning at `1`, keep the same zone,
connection and actuator ids on every row, and contain non-negative requested
and allocated airflow. Connection health must remain in `0.0..1.0`.

## Determinism

Fixed ticks, a declared seed and no wall clock. Seeded source variation is
reproducible. Running the same scenario file twice produces identical records,
and trace files are byte-identical between runs (floats serialise through
`repr` with sorted JSON keys). This is what makes a trace a replay.

## How to run

```bash
# tests (pytest reads src/ via pyproject.toml)
python3 -m pytest

# produce a replay trace from an explicit scenario file
PYTHONPATH=src python3 -m icarus scenarios/standard_habitat.json traces/standard_habitat.jsonl

# produce a standalone HTML visualisation
PYTHONPATH=src python3 -m icarus.visualise traces/standard_habitat.jsonl out/standard_habitat.html
```

Runtime code uses only the Python standard library. `pytest` is needed for
the test suite only.

## Deliberately not here yet

- Gradual fault profiles (connection health changing over time).
- AI/ML fault detection (ONNX model, quantisation, benchmarks).
- Safety governor or backup fan response to detected faults.
- Dashboard, API, MQTT, database, hardware integration, Docker, cloud.
- Any claim about real lunar ECLSS safety or engineering accuracy.
