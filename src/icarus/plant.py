"""Two-room ventilation plant for the ICARUS proof loop.

Room A is the crew cabin. Room B is the air-processing bay holding a simple
CO2 scrubber. The main fan moves cabin air through the scrubber once per
1-second tick. See docs/simulation-rules.md for the full tick order and the
meaning of each unit.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PlantConfig:
    """Fixed knobs of the proof-loop plant, in abstract simulation units."""

    crew_co2_per_second: float = 1.0
    fan_base_airflow: float = 10.0
    scrubber_removal_fraction: float = 0.5
    room_a_air_volume: float = 100.0


@dataclass(frozen=True)
class PlantState:
    """Complete plant state at the end of a tick."""

    tick: int = 0
    room_a_co2: float = 0.0
    room_b_co2: float = 0.0
    main_fan_effectiveness: float = 1.0


def main_fan_airflow(config: PlantConfig, state: PlantState) -> float:
    """Air the main fan moves this tick, in airflow_units_per_second."""
    return config.fan_base_airflow * state.main_fan_effectiveness


def with_main_fan_effectiveness(state: PlantState, effectiveness: float) -> PlantState:
    """Return a copy of ``state`` with a new main-fan effectiveness (the fault knob)."""
    return replace(state, main_fan_effectiveness=effectiveness)


def step_plant(config: PlantConfig, state: PlantState) -> PlantState:
    """Advance the plant by one 1-second tick and return the new state.

    Tick order: crew adds CO2 to Room A; the main fan moves a share of Room A
    air through the scrubber; the scrubber captures a fixed fraction of the
    CO2 in that moved air; the cleaned air returns to Room A while the
    captured CO2 accumulates in Room B.
    """
    room_a_co2 = state.room_a_co2 + config.crew_co2_per_second
    moved_fraction = main_fan_airflow(config, state) / config.room_a_air_volume
    captured = room_a_co2 * moved_fraction * config.scrubber_removal_fraction
    return replace(
        state,
        tick=state.tick + 1,
        room_a_co2=room_a_co2 - captured,
        room_b_co2=state.room_b_co2 + captured,
    )
