"""Hub-layout ventilation plant for the ICARUS scenario graph.

Each non-processing zone has an idealised CO₂ sensor. Its reading drives a
bounded actuator command for that zone's circulation loop. Air passes through
the shared processing bay, where a scrubber captures a declared fraction of
CO₂ before the mixed return stream reaches each room.

All quantities are abstract simulation units. This is a hub-layout simulator,
not a general fluid solver.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Mapping

from icarus.actuator import ActuatorState, RateLimitedActuator
from icarus.config import ConnectionSpec, HabitatConfig
from icarus.control import CO2SensorReading, ProportionalCO2Controller


@dataclass(frozen=True)
class HabitatState:
    """Complete habitat state at the end of one fixed-duration tick."""

    tick: int = 0
    zone_co2_mass: dict[str, float] = field(default_factory=dict)
    captured_co2: float = 0.0
    sensor_co2_concentration: dict[str, float] = field(default_factory=dict)
    source_co2_mass: dict[str, float] = field(default_factory=dict)
    source_noise: dict[str, float] = field(default_factory=dict)
    occupancy_multiplier: dict[str, float] = field(default_factory=dict)
    actuators: dict[str, ActuatorState] = field(default_factory=dict)
    requested_airflows: dict[str, float] = field(default_factory=dict)
    delivered_airflows: dict[str, float] = field(default_factory=dict)
    airflow_residuals: dict[str, float] = field(default_factory=dict)
    capacity_scale: float = 1.0


def initial_state(config: HabitatConfig) -> HabitatState:
    """Create a fresh, empty and fully deterministic plant state."""
    zero_airflow = {connection.id: 0.0 for connection in config.connections}
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
        requested_airflows=zero_airflow,
        delivered_airflows=dict(zero_airflow),
        airflow_residuals=dict(zero_airflow),
        capacity_scale=1.0,
    )


def path_airflow(connection: ConnectionSpec, actuator_position: float = 1.0) -> float:
    """Return one path's nominal requested flow at a measured actuator position."""
    position = max(0.0, min(1.0, actuator_position))
    return connection.max_airflow * position


def requested_loop_airflow(
    outbound: ConnectionSpec,
    inbound: ConnectionSpec,
    actuator_position: float,
) -> float:
    """Return the loop's commanded physical capacity before health or faults."""
    return min(
        path_airflow(outbound, actuator_position),
        path_airflow(inbound, actuator_position),
    )


def _loop_static_health(outbound: ConnectionSpec, inbound: ConnectionSpec) -> float:
    """Return the loop's hidden physical-capacity fraction from both path legs."""
    nominal_capacity = min(outbound.max_airflow, inbound.max_airflow)
    healthy_capacity = min(
        outbound.max_airflow * outbound.health,
        inbound.max_airflow * inbound.health,
    )
    return healthy_capacity / nominal_capacity


def _loop_fault_effectiveness(
    outbound: ConnectionSpec,
    connection_effectiveness: Mapping[str, float],
) -> float:
    """Validate and return the hidden fault multiplier for the loop meter."""
    effectiveness = connection_effectiveness.get(outbound.id, 1.0)
    if not isinstance(effectiveness, (int, float)) or isinstance(effectiveness, bool):
        raise ValueError("connection effectiveness must be a finite number in 0.0..1.0")
    effectiveness = float(effectiveness)
    if not math.isfinite(effectiveness) or not 0.0 <= effectiveness <= 1.0:
        raise ValueError("connection effectiveness must be a finite number in 0.0..1.0")
    return effectiveness


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
    connection_effectiveness: Mapping[str, float] | None = None,
    source_tick: int | None = None,
    occupancy_tick: int | None = None,
) -> tuple[HabitatState, dict[str, float]]:
    """Advance one deterministic tick and return state plus delivered path flows.

    Requested flow is derived only from nominal loop capacity and measured
    actuator position. Static connection health and an optional scenario fault
    reduce physical delivery later; shared capacity then allocates the resulting
    provisional delivery proportionally. The distinction makes demand and
    degraded delivery observable without exposing hidden fault truth.
    """
    connection_effectiveness = connection_effectiveness or {}

    # 1. Occupancy-scaled, replayable CO₂ sources.
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

    # 2. Sensors, controllers and measured actuator movement.
    sensor_co2_concentration = {
        zone.id: zone_co2_mass[zone.id] / zone.air_volume
        for zone in config.zones
    }
    controller = ProportionalCO2Controller(config.control)
    actuator_model = RateLimitedActuator(config.actuator)
    actuators: dict[str, ActuatorState] = {}
    requested_by_zone: dict[str, float] = {}
    provisional_delivered_by_zone: dict[str, float] = {}
    for zone in config.non_processing_zones():
        outbound = config.path_to_processing(zone.id)
        inbound = config.path_from_processing(zone.id)
        setpoint = controller.command_for(
            CO2SensorReading(zone_id=zone.id, value=sensor_co2_concentration[zone.id])
        )
        actuator = actuator_model.step(state.actuators[zone.id], setpoint)
        actuators[zone.id] = actuator

        requested = requested_loop_airflow(
            outbound,
            inbound,
            actuator.actual_position,
        )
        static_health = _loop_static_health(outbound, inbound)
        fault_effectiveness = _loop_fault_effectiveness(
            outbound,
            connection_effectiveness,
        )
        requested_by_zone[zone.id] = requested
        provisional_delivered_by_zone[zone.id] = (
            requested * static_health * fault_effectiveness
        )

    # 3. Proportional shared-capacity allocation acts on physical delivery, not
    # controller request. It cannot manufacture capacity lost to health/faults.
    total_provisional_delivery = sum(provisional_delivered_by_zone.values())
    shared_capacity = config.air_system.shared_airflow_capacity
    capacity_scale = (
        math.nextafter(shared_capacity / total_provisional_delivery, 0.0)
        if total_provisional_delivery > shared_capacity
        else 1.0
    )
    delivered_by_zone = {
        zone_id: provisional * capacity_scale
        for zone_id, provisional in provisional_delivered_by_zone.items()
    }

    requested_airflows: dict[str, float] = {}
    delivered_airflows: dict[str, float] = {}
    airflow_residuals: dict[str, float] = {}
    for zone in config.non_processing_zones():
        outbound = config.path_to_processing(zone.id)
        inbound = config.path_from_processing(zone.id)
        requested = requested_by_zone[zone.id]
        delivered = delivered_by_zone[zone.id]
        residual = requested - delivered
        for connection in (outbound, inbound):
            requested_airflows[connection.id] = requested
            delivered_airflows[connection.id] = delivered
            airflow_residuals[connection.id] = residual

    # 4. Calculate every extraction from the same pre-transfer state.
    retained_mass = dict(zone_co2_mass)
    extracted_mass: dict[str, float] = {}
    for zone in config.non_processing_zones():
        moved_fraction = min(delivered_by_zone[zone.id] / zone.air_volume, 1.0)
        extracted = zone_co2_mass[zone.id] * moved_fraction
        extracted_mass[zone.id] = extracted
        retained_mass[zone.id] -= extracted

    # 5. Mix, scrub and return all transferred mass simultaneously.
    total_extracted_mass = sum(extracted_mass.values())
    captured_this_tick = (
        total_extracted_mass * config.air_system.scrubber_removal_fraction
    )
    returned_mass = total_extracted_mass - captured_this_tick
    total_delivered_airflow = sum(delivered_by_zone.values())
    if total_delivered_airflow > 0.0:
        for zone in config.non_processing_zones():
            return_share = delivered_by_zone[zone.id] / total_delivered_airflow
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
        delivered_airflows=delivered_airflows,
        airflow_residuals=airflow_residuals,
        capacity_scale=capacity_scale,
    )
    return new_state, delivered_airflows
