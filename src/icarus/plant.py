"""Hub-layout ventilation plant for the ICARUS scenario graph.

Each non-processing zone pushes air through its own directed path to the
air-processing bay; the scrubber there captures a declared fraction of the
CO2 in the moved air; the cleaned air returns along the paired return path.
This is the simple hub layout of docs/simulation-rules.md, not a general
fluid solver.

All quantities are abstract simulation units (``co2_units``,
``airflow_units_per_second``). They are not real spacecraft ppm, kilograms,
or safety limits.
"""

from dataclasses import dataclass, field

from icarus.config import (
    AIR_PROCESSING_SCRUBBER_REMOVAL_FRACTION,
    ConnectionSpec,
    HabitatConfig,
)


@dataclass(frozen=True)
class HabitatState:
    """Complete habitat state at the end of a tick.

    ``zone_co2`` maps zone id to airborne CO2. ``captured_co2`` is the
    processing bay's cumulative captured counter; it only ever grows. The
    mapping is rebuilt every tick and treated as immutable between ticks.
    """

    tick: int = 0
    zone_co2: dict[str, float] = field(default_factory=dict)
    captured_co2: float = 0.0


def initial_state(config: HabitatConfig) -> HabitatState:
    """Fresh state: every zone empty, nothing captured yet."""
    return HabitatState(
        tick=0,
        zone_co2={zone.id: 0.0 for zone in config.zones},
        captured_co2=0.0,
    )


def path_airflow(connection: ConnectionSpec) -> float:
    """Actual air a path moves this tick, in airflow_units_per_second."""
    return connection.max_airflow * connection.health


def step_habitat(
    config: HabitatConfig, state: HabitatState
) -> tuple[HabitatState, dict[str, float]]:
    """Advance the habitat by one 1-second tick.

    Tick order (see docs/simulation-rules.md):

    1. Every zone adds its configured CO2 source.
    2. Each non-processing zone's loop airflow is computed from its path to
       the processing bay: ``max_airflow * health``.
    3. The scrubber captures its declared fraction of that zone's CO2 from
       the moved air; the zone keeps the rest.
    4. The captured CO2 is added to the processing bay's cumulative counter.

    Returns the new state plus the actual airflow of every connection. A
    loop's return path reports the same actual airflow as its outbound path:
    the outbound leg meters the loop, and the cleaned air comes back along
    the return leg.
    """
    # 1. Sources.
    zone_co2 = {
        zone.id: state.zone_co2[zone.id] + zone.co2_generation_per_second
        for zone in config.zones
    }

    captured_this_tick = 0.0
    airflows: dict[str, float] = {}
    for zone in config.non_processing_zones():
        # 2. Loop airflow from the zone's path to the processing bay.
        outbound = config.path_to_processing(zone.id)
        inbound = config.path_from_processing(zone.id)
        airflow = path_airflow(outbound)
        # 3. Capture the declared fraction of this zone's CO2 from moved air.
        moved_fraction = airflow / zone.air_volume
        captured = (
            zone_co2[zone.id] * moved_fraction * AIR_PROCESSING_SCRUBBER_REMOVAL_FRACTION
        )
        zone_co2[zone.id] -= captured
        # 4. Accumulate in the processing bay's captured counter.
        captured_this_tick += captured
        airflows[outbound.id] = airflow
        airflows[inbound.id] = airflow

    new_state = HabitatState(
        tick=state.tick + 1,
        zone_co2=zone_co2,
        captured_co2=state.captured_co2 + captured_this_tick,
    )
    return new_state, airflows
