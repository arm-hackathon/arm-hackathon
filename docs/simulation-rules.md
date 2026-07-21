# ICARUS simulation rules (scenario graph, PR #6)

This document is the single source of truth for what the current simulation
slice does, in plain English. It is a simulation of ideas, not of real
spacecraft hardware.

## Scope

NASA ECLSS material lists cabin-air circulation, CO2 removal, atmosphere
management, and thermal control as life-support functions. This slice models
only **circulation** and **CO2 removal**. Oxygen, total pressure,
temperature, humidity, trace contaminants, lunar dust, fire, leaks, and the
external vacuum are held constant and stay out of scope.

The habitat is a **user-editable scenario graph**, not a hard-coded pair of
rooms. This PR supports the simple hub layout only: one air-processing bay
in the middle, every other zone connected to it by one directed path each
way. It is not a general fluid solver.

## Units

All values are abstract simulation units:

- `co2_units` — amount of CO2. Not ppm, not kilograms, not a safety limit.
- `airflow_units_per_second` — air moved along a path. Not a real flow rate.

No number in this repository is a real spacecraft measurement or a real
safety threshold.

## The scenario file

A scenario is versioned JSON (see `scenarios/standard_habitat.json`) with
three top-level keys:

| Key | Meaning |
|---|---|
| `version` | Format version. Only `1` is supported. |
| `zones` | List of rooms. |
| `connections` | List of directed air paths between rooms. |

Every zone has exactly these fields:

| Field | Meaning |
|---|---|
| `id` | Unique zone id. |
| `label` | Human-readable name. |
| `preset` | One of `crew_cabin`, `lab`, `air_processing`, `storage`. |
| `co2_generation_per_second` | CO2 source added each tick (`>= 0`). |
| `air_volume` | Total air in the zone (`> 0`), used to size the moved share. |

Every connection has exactly these fields:

| Field | Meaning |
|---|---|
| `id` | Unique connection id. |
| `from` | Source zone id. |
| `to` | Target zone id. |
| `max_airflow` | Air the path moves per tick at health 1.0 (`> 0`). |
| `health` | Path health in `0.0..1.0`. `1.0` is fully healthy. |

Exactly one zone must use the `air_processing` preset; it is the scrubber
bay. The scrubber's removal fraction is a declared constant
(`AIR_PROCESSING_SCRUBBER_REMOVAL_FRACTION = 0.5` in `src/icarus/config.py`),
not a per-zone field, because the graph format fixes the zone fields.

## The standard habitat

`scenarios/standard_habitat.json` is the reference hub layout: two crew
cabins with a positive CO2 source each, a lab with no source, and the
air-processing bay.

```text
                cabin_a (Crew Cabin A, CO2 source)
                    |  ^
          out 10.0  |  |  return 10.0
                    v  |
   cabin_b  <====  processing  ====>  lab
   (Crew Cabin B,  (Air Processing   (Lab, no source)
    CO2 source)     Bay, scrubber)
     10.0 each way     8.0 each way
```

Every non-processing zone has one directed path to the processing bay and
one directed path back. All six connections start at health `1.0`.

## One tick, in order

Time advances in fixed 1-second ticks. Each tick, exactly this happens:

1. Every zone adds its configured `co2_generation_per_second` to its own
   airborne CO2.
2. For each non-processing zone, the loop airflow is calculated from its
   path to the processing bay: `airflow = max_airflow * health`.
3. The scrubber captures `zone_co2 * (airflow / air_volume) *
   scrubber_removal_fraction` CO2 from the air that moved through that
   path. The zone keeps the rest.
4. All CO2 captured this tick is added to the processing bay's cumulative
   captured counter, which only ever grows.
5. A trace row is appended to the JSONL trace: every zone's CO2 (plus the
   processing bay's captured counter) and every connection's actual
   airflow and health.

A loop's return path reports the same actual airflow as its outbound path:
the outbound leg meters the loop, and the cleaned air comes back along the
return leg. A path at health `0.0` moves no air and scrubs nothing.

CO2 is conserved: what the zones generate is always split between airborne
CO2 and the processing bay's captured store. Nothing appears or vanishes.

## The standard run

`STANDARD_RUN` in `src/icarus/scenario.py` declares the run constants:

| Constant | Value | Meaning |
|---|---|---|
| `total_ticks` | 120 | Length of a run. |
| `warmup_ticks` | 60 | Ticks excluded from the ceiling check. |
| `crew_cabin_co2_ceiling` | 30.0 | Declared crew-cabin CO2 ceiling for the standard scenario. |

Healthy, both crew cabins settle around 20 CO2 units and stay below the
declared ceiling after warm-up.

## Validation

`src/icarus/config.py` rejects a scenario with a clear `ValueError` (the
CLI prints it and exits non-zero) when it has:

- an unsupported or missing version;
- no zones, or no `air_processing` zone;
- more than one `air_processing` zone;
- a duplicate zone id or connection id;
- an unsupported preset;
- a non-positive air volume or a negative CO2 source;
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

## Determinism

Fixed ticks, fixed inputs, no randomness, no wall clock. Running the same
scenario file twice produces identical records, and trace files are
byte-identical between runs (floats serialise through `repr` with sorted
JSON keys). This is what makes a trace a replay.

## How to run

```bash
# tests (pytest reads src/ via pyproject.toml)
python3 -m pytest

# produce a replay trace from an explicit scenario file
PYTHONPATH=src python3 -m icarus scenarios/standard_habitat.json traces/standard_habitat.jsonl
```

Runtime code uses only the Python standard library. `pytest` is needed for
the test suite only.

## Deliberately not here yet

- Gradual fault profiles (connection health changing over time).
- AI/ML fault detection (ONNX model, quantization, benchmarks).
- Safety governor, backup fan, or any automatic command.
- Dashboard, API, MQTT, database, hardware integration, Docker, cloud.
- Any claim about real lunar ECLSS safety or engineering accuracy.
