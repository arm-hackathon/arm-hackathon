"""Hub-layout ventilation plant for the ICARUS scenario graph.

Each non-processing zone has an idealised CO2 sensor. Its reading drives a
bounded actuator command for the zone's ventilation loop. Air passes through
the air-processing bay, where the scrubber captures a declared fraction of
the CO2 in the moved air, then returns along the paired path.
This is the simple hub layout of docs/simulation-rules.md, not a general
fluid solver.

All quantities are abstract simulation units (``co2_units``,
``airflow_units_per_second``). They are not real spacecraft ppm, kilograms,
or safety limits.
"""

import math
import random
from dataclasses import dataclass, field

from icarus.actuator import ActuatorState, RateLimitedActuator
from icarus.config import ConnectionSpec, HabitatConfig
from icarus.control import CO2SensorReading, ProportionalCO2Controller


@dataclass(frozen=True)
class HabitatState:
    """Complete habitat state at the end of a tick.

    ``zone_co2_mass`` maps zone id to airborne CO2 mass. ``captured_co2`` is the
    processing bay's cumulative captured counter; it only ever grows. The
    mapping is rebuilt every tick and treated as immutable between ticks.
    """

    tick: int = 0
    zone_co2_mass: dict[str, float] = field(default_factory=dict)
    captured_co2: float = 0.0
    sensor_co2_concentration: dict[str, float] = field(default_factory=dict)
    source_co2_mass: dict[str, float] = field(default_factory=dict)
    source_noise: dict[str, float] = field(default_factory=dict)
    occupancy_multiplier: dict[str, float] = field(default_factory=dict)
    actuators: dict[str, ActuatorState] = field(default_factory=dict)
    requested_airflows: dict[str, float] = field(default_factory=dict)
    capacity_scale: float = 1.0


def initial_state(config: HabitatConfig) -> HabitatState:
    """Fresh state: every zone empty, nothing captured yet."""
    return HabitatState(
        tick=0,
        zone_co2_mass={zone.id: 0.0 for zone in config.zones},
        captured_co2=0.0,
        sensor_co2_concentration={zone.id: 0.0 for zone in config.zones},
        source_co2_mass={zone.id: 0.0 for zone in config.zones},
        source_noise={zone.id: 0.0 for zone in config.zones},
        occupancy_multiplier={zone.id: 1.0 for zone in config.zones},
        actuators={
            zone.id: ActuatorState() for zone in config.non_processing_zones()
        },
        requested_airflows={connection.id: 0.0 for connection in config.connections},
        capacity_scale=1.0,
    )


def path_airflow(connection: ConnectionSpec, actuator_position: float = 1.0) -> float:
    """Requested path airflow at a bounded, normalised actuator position."""
    position = max(0.0, min(1.0, actuator_position))
    return connection.max_airflow * connection.health * position


def _occupancy_multiplier(zone, tick: int) -> float:
    for period in zone.occupancy_profile:
        if period.start_tick <= tick <= period.end_tick:
            return period.multiplier
    return 1.0


def _co2_source_for_tick(
    config: HabitatConfig,
    zone,
    source_tick: int,
    previous_noise: float,
    occupancy_tick: int,
) -> tuple[float, float, float]:
    """Return seeded, correlated source mass and its component values."""
    occupancy = _occupancy_multiplier(zone, occupancy_tick)
    epsilon = zone.co2_generation_epsilon
    if epsilon == 0.0:
        return zone.co2_generation_per_second * occupancy, 0.0, occupancy
    # String seeding avoids dependence on Python's process-randomised hash and
    # makes a zone/tick sample independent of graph iteration order.
    generator = random.Random(
        f"icarus:{config.simulation.random_seed}:{source_tick}:{zone.id}"
    )
    innovation = generator.uniform(-epsilon, epsilon)
    correlation = zone.co2_noise_correlation
    innovation_scale = math.sqrt(1.0 - correlation * correlation)
    noise = correlation * previous_noise + innovation_scale * innovation
    noise = max(-epsilon, min(epsilon, noise))
    baseline = zone.co2_generation_per_second * occupancy
    return max(0.0, baseline + noise), noise, occupancy


def step_habitat(
    config: HabitatConfig,
    state: HabitatState,
    *,
    source_tick: int | None = None,
    occupancy_tick: int | None = None,
) -> tuple[HabitatState, dict[str, float]]:
    """Advance the habitat by one 1-second tick.

    Tick order (see docs/simulation-rules.md):

    1. Every zone adds its occupancy-scaled, seeded CO2 source sample.
    2. Ideal sensors measure concentration and local controllers set demand.
    3. Each actuator moves towards its setpoint at its declared stroke rate.
    4. Requested loop flows share the central fan's bounded capacity.
    5. All zones simultaneously send CO2 mass into the shared processing flow.
    6. The scrubber captures a fraction; the common return flow redistributes
       the remaining mass across connected zones.

    Returns the new state plus the actual airflow of every connection. A
    loop's return path reports the same actual airflow as its outbound path;
    the weaker leg limits the loop and cleaned air returns along the other.
    """
    # 1. Occupancy-scaled sources with correlated, replayable variation.
    source_co2_mass: dict[str, float] = {}
    source_noise: dict[str, float] = {}
    occupancy_multiplier: dict[str, float] = {}
    next_tick = state.tick + 1
    source_tick = next_tick if source_tick is None else source_tick
    occupancy_tick = next_tick if occupancy_tick is None else occupancy_tick
    for zone in config.zones:
        source, noise, occupancy = _co2_source_for_tick(
            config,
            zone,
            source_tick,
            state.source_noise[zone.id],
            occupancy_tick,
        )
        source_co2_mass[zone.id] = source
        source_noise[zone.id] = noise
        occupancy_multiplier[zone.id] = occupancy

    zone_co2_mass = {
        zone.id: state.zone_co2_mass[zone.id] + source_co2_mass[zone.id]
        for zone in config.zones
    }
    # 2. Sensors report concentration, not absolute mass.
    sensor_co2_concentration = {
        zone.id: zone_co2_mass[zone.id] / zone.air_volume
        for zone in config.zones
    }
    controller = ProportionalCO2Controller(config.control)
    actuator_model = RateLimitedActuator(config.actuator)

    # 3. Actuator movement and unconstrained loop requests.
    actuators: dict[str, ActuatorState] = {}
    requested_by_zone: dict[str, float] = {}
    for zone in config.non_processing_zones():
        outbound = config.path_to_processing(zone.id)
        inbound = config.path_from_processing(zone.id)
        setpoint = controller.command_for(
            CO2SensorReading(
                zone_id=zone.id,
                value=sensor_co2_concentration[zone.id],
            )
        )
        actuator = actuator_model.step(state.actuators[zone.id], setpoint)
        actuators[zone.id] = actuator
        requested_by_zone[zone.id] = min(
            path_airflow(outbound, actuator.actual_position),
            path_airflow(inbound, actuator.actual_position),
        )

    # 4. All loops share one fan capacity. Proportional scaling preserves each
    # local controller's relative demand without exceeding the system limit.
    total_requested = sum(requested_by_zone.values())
    shared_capacity = config.air_system.shared_airflow_capacity
    capacity_scale = (
        min(1.0, shared_capacity / total_requested)
        if total_requested > 0.0
        else 1.0
    )
    actual_by_zone = {
        zone_id: requested * capacity_scale
        for zone_id, requested in requested_by_zone.items()
    }

    requested_airflows: dict[str, float] = {}
    airflows: dict[str, float] = {}
    for zone in config.non_processing_zones():
        outbound = config.path_to_processing(zone.id)
        inbound = config.path_from_processing(zone.id)
        for connection in (outbound, inbound):
            requested_airflows[connection.id] = requested_by_zone[zone.id]
            airflows[connection.id] = actual_by_zone[zone.id]

    # 5. Calculate every extraction from the same pre-transfer state.
    retained_mass = dict(zone_co2_mass)
    extracted_mass: dict[str, float] = {}
    for zone in config.non_processing_zones():
        moved_fraction = min(actual_by_zone[zone.id] / zone.air_volume, 1.0)
        extracted = zone_co2_mass[zone.id] * moved_fraction
        extracted_mass[zone.id] = extracted
        retained_mass[zone.id] -= extracted

    # 6. The shared return stream mixes zones after scrubbing, which couples
    # their environmental state even though each retains a local controller.
    total_extracted_mass = sum(extracted_mass.values())
    captured_this_tick = (
        total_extracted_mass * config.air_system.scrubber_removal_fraction
    )
    returned_mass = total_extracted_mass - captured_this_tick
    total_actual_airflow = sum(actual_by_zone.values())
    if total_actual_airflow > 0.0:
        for zone in config.non_processing_zones():
            return_share = actual_by_zone[zone.id] / total_actual_airflow
            retained_mass[zone.id] += returned_mass * return_share

    new_state = HabitatState(
        tick=next_tick,
        zone_co2_mass=retained_mass,
        captured_co2=state.captured_co2 + captured_this_tick,
        sensor_co2_concentration=sensor_co2_concentration,
        source_co2_mass=source_co2_mass,
        source_noise=source_noise,
        occupancy_multiplier=occupancy_multiplier,
        actuators=actuators,
        requested_airflows=requested_airflows,
        capacity_scale=capacity_scale,
    )
    return new_state, airflows
