# ICARUS simulation rules (proof loop, PR #1)

This document is the single source of truth for what the first simulation
slice does, in plain English. It is a simulation of ideas, not of real
spacecraft hardware.

## Scope

NASA ECLSS material lists cabin-air circulation, CO2 removal, atmosphere
management, and thermal control as life-support functions. This slice models
only **circulation** and **CO2 removal**. Oxygen, total pressure,
temperature, humidity, trace contaminants, lunar dust, fire, leaks, and the
external vacuum are held constant and stay out of scope.

## Units

All values are abstract simulation units:

- `co2_units` — amount of CO2. Not ppm, not kilograms, not a safety limit.
- `airflow_units_per_second` — air moved by the fan. Not a real flow rate.

No number in this repository is a real spacecraft measurement or a real
safety threshold.

## The plant

Two sealed rooms:

- **Room A** — the crew cabin. One crew member adds CO2 at a fixed rate.
- **Room B** — the air-processing bay. The main fan pushes Room A air
  through a simple scrubber here. `room_b_co2` is the cumulative CO2 the
  scrubber has captured so far; it only ever grows.

Default knobs (`PlantConfig`):

| Knob | Default | Meaning |
|---|---|---|
| `crew_co2_per_second` | 1.0 | CO2 the crew adds to Room A each tick |
| `fan_base_airflow` | 10.0 | Air moved per tick at effectiveness 1.0 |
| `scrubber_removal_fraction` | 0.5 | Fraction of CO2 captured from moved air |
| `room_a_air_volume` | 100.0 | Total Room A air, used to size the moved share |

## One tick, in order

Time advances in fixed 1-second ticks. Each tick, exactly this happens:

1. Crew adds `crew_co2_per_second` CO2 to Room A.
2. The main fan moves `airflow = fan_base_airflow * main_fan_effectiveness`
   units of Room A air through the scrubber.
3. The scrubber captures `room_a_co2 * (airflow / room_a_air_volume) *
   scrubber_removal_fraction` CO2 from that moved air.
4. The cleaned air returns to Room A; the captured CO2 is added to Room B.
5. A trace row with the new state is appended to the JSONL trace.

CO2 is conserved: what the crew adds is always split between Room A's air
and Room B's captured store. Nothing appears or vanishes.

## Scenarios

| Scenario | Length | Behaviour |
|---|---|---|
| `nominal` | 120 ticks | Fan effectiveness stays 1.0. Room A CO2 settles around 19 units and stays below the declared ceiling of 30 units after the 60-tick warm-up. |
| `primary_fan_degradation` | 120 ticks | Fan effectiveness drops from 1.0 to 0.35 at tick 20. The trace row for tick 20 is the first to show the weaker fan. Airflow falls from 10.0 to 3.5 units per tick; less CO2 is captured per tick, so Room A CO2 climbs well above nominal (about 49 units by tick 120). |

`main_fan_effectiveness` is the only fault knob. There is no backup fan, no
automatic command, and no control decision in this slice.

## Determinism

Fixed ticks, fixed inputs, no randomness, no wall clock. Running the same
scenario name twice produces identical records, and trace files are
byte-identical between runs (floats serialise through `repr` with sorted
JSON keys). This is what makes a trace a replay.

## How to run

```bash
# tests (pytest reads src/ via pyproject.toml)
python3 -m pytest

# produce a replay trace
PYTHONPATH=src python3 -m icarus nominal traces/nominal.jsonl
PYTHONPATH=src python3 -m icarus primary_fan_degradation traces/primary_fan_degradation.jsonl
```

Runtime code uses only the Python standard library. `pytest` is needed for
the test suite only.

## Deliberately not here yet

- AI/ML fault detection (ONNX model, quantization, benchmarks).
- Safety governor and redundant-fan command.
- Dashboard, API, MQTT, database, hardware integration, Docker, cloud.
- Any claim about real lunar ECLSS safety or engineering accuracy.
