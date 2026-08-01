"""Deterministic runs of a validated scenario graph.

A run is an unrecorded warm-up followed by fixed measured ticks. Fault
profiles are evaluated from the measured tick, never accumulated into mutable
state, so the same scenario and seed always reproduce the same trace.
"""

import math
from contextlib import nullcontext
from dataclasses import dataclass, replace

from aeolus.config import HabitatConfig
from aeolus.measurement import (
    deterministic_measurement_drift,
    deterministic_measurement_sample,
)
from aeolus.plant import initial_state, requested_loop_airflow, step_habitat
from aeolus.trace import TickRecord, TraceWriter


@dataclass(frozen=True)
class RunSpec:
    """Declared constants of a run: length, warm-up and cabin concentration."""

    total_ticks: int
    warmup_ticks: int
    crew_cabin_co2_concentration_ceiling: float

    def __post_init__(self) -> None:
        if self.total_ticks < 1:
            raise ValueError("total_ticks must be positive")
        if self.warmup_ticks < 0:
            raise ValueError("warmup_ticks must not be negative")
        if self.crew_cabin_co2_concentration_ceiling <= 0.0:
            raise ValueError("crew-cabin CO2 ceiling must be positive")


# Declared run constants for the standard habitat. The ceiling is the
# declared crew-cabin concentration ceiling for the measured run.
STANDARD_RUN = RunSpec(
    total_ticks=120,
    warmup_ticks=60,
    crew_cabin_co2_concentration_ceiling=0.30,
)


def run_scenario(
    config: HabitatConfig,
    *,
    run: RunSpec = STANDARD_RUN,
    trace_path=None,
) -> list[TickRecord]:
    """Run a validated scenario graph and return one record per tick, in order.

    When ``trace_path`` is given, each record is also appended to that file
    as one JSONL row, immediately after its tick is computed.
    """
    state = initial_state(config)
    # Warm up under the first declared occupancy conditions. Negative source
    # ticks give the pre-roll its own deterministic noise sequence, while the
    # measured run still begins with scenario tick 1.
    for warmup_index in range(run.warmup_ticks):
        state, _ = step_habitat(
            config,
            state,
            source_tick=warmup_index - run.warmup_ticks,
            occupancy_tick=1,
        )
    # The warm-up establishes physical state but is not part of measured time
    # or captured-CO2 accounting in the replay.
    state = replace(state, tick=0, captured_co2=0.0)
    records: list[TickRecord] = []

    writer_context = (
        TraceWriter(trace_path) if trace_path is not None else nullcontext(None)
    )
    with writer_context as writer:
        while state.tick < run.total_ticks:
            next_tick = state.tick + 1
            state, airflows = step_habitat(
                config,
                state,
                connection_effectiveness=_connection_effectiveness(config, next_tick),
                frozen_zones=_frozen_sensor_zones(config, next_tick),
            )
            record = _tick_record(config, state, airflows)
            records.append(record)
            if writer is not None:
                writer.write(record)
    return records


def _connection_effectiveness(config: HabitatConfig, tick: int) -> dict[str, float]:
    """Return hidden fault multipliers for one measured tick."""
    return {
        profile.connection_id: profile.effectiveness_at(tick)
        for profile in config.connection_faults()
    }


def _frozen_sensor_zones(config: HabitatConfig, tick: int) -> frozenset[str]:
    """Return zone ids whose sensors are frozen at one measured tick."""
    return frozenset(
        profile.zone_id
        for profile in config.sensor_faults()
        if profile.is_frozen_at(tick)
    )


def _tick_record(
    config: HabitatConfig, state, airflows: dict[str, float]
) -> TickRecord:
    """Snapshot sensor, plant and actuator telemetry for every zone and path."""
    processing_id = config.processing_zone().id
    observed_positions, observed_connections = _observed_loop_telemetry(
        config, state, airflows
    )
    zones: dict[str, dict[str, float]] = {}
    for zone in config.zones:
        entry = {
            "co2_mass": state.zone_co2_mass[zone.id],
            "co2_concentration": state.zone_co2_mass[zone.id] / zone.air_volume,
            "sensor_co2_concentration": state.sensor_co2_concentration[zone.id],
            "source_co2_mass": state.source_co2_mass[zone.id],
            "occupancy_multiplier": state.occupancy_multiplier[zone.id],
        }
        if zone.id == processing_id:
            entry["captured_co2"] = state.captured_co2
        zones[zone.id] = entry
    connections = observed_connections
    actuators = {
        zone_id: {
            "setpoint": actuator.setpoint,
            "actual_position": observed_positions[zone_id],
            "tracking_residual": actuator.setpoint - observed_positions[zone_id],
            "moving": float(actuator.moving),
            "movement_seconds": actuator.movement_seconds,
            "power": actuator.power,
            "direction": float(actuator.direction),
        }
        for zone_id, actuator in state.actuators.items()
    }
    return TickRecord(
        tick=state.tick,
        zones=zones,
        connections=connections,
        actuators=actuators,
        system={
            "shared_airflow_capacity": config.air_system.shared_airflow_capacity,
            "total_requested_airflow": sum(
                observed_connections[connection.id]["requested_airflow"]
                for connection in config.connections
                if connection.to_zone == processing_id
            ),
            "total_delivered_airflow": sum(
                observed_connections[connection.id]["delivered_airflow"]
                for connection in config.connections
                if connection.to_zone == processing_id
            ),
            "capacity_scale": state.capacity_scale,
        },
    )


def _observed_loop_telemetry(
    config: HabitatConfig, state, physical_airflows: dict[str, float]
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Project hidden plant state into bounded, noisy observable measurements."""
    settings = config.telemetry
    observed_positions: dict[str, float] = {}
    observed_connections: dict[str, dict[str, float]] = {}

    for zone in config.non_processing_zones():
        actuator = state.actuators[zone.id]
        position_noise = (
            settings.actuator_position_noise_fraction
            * deterministic_measurement_sample(
                config, zone.id, "actuator_position", state.tick
            )
        )
        observed_position = max(
            0.0, min(1.0, actuator.actual_position + position_noise)
        )
        observed_positions[zone.id] = observed_position

        outbound = config.path_to_processing(zone.id)
        inbound = config.path_from_processing(zone.id)
        observed_requested = requested_loop_airflow(
            outbound, inbound, observed_position
        )
        for connection in (outbound, inbound):
            bias = (
                connection.max_airflow
                * settings.airflow_bias_fraction
                * deterministic_measurement_sample(
                    config, connection.id, "airflow_bias", 0
                )
            )
            noise = (
                connection.max_airflow
                * settings.airflow_noise_fraction
                * deterministic_measurement_sample(
                    config, connection.id, "airflow_noise", state.tick
                )
            )
            drift = (
                connection.max_airflow
                * settings.airflow_drift_fraction
                * deterministic_measurement_drift(
                    config, connection.id, "airflow", state.tick
                )
            )
            observed_delivered = max(
                0.0,
                min(
                    observed_requested,
                    physical_airflows[connection.id] + bias + drift + noise,
                ),
            )
            observed_connections[connection.id] = {
                "requested_airflow": observed_requested,
                "delivered_airflow": observed_delivered,
                "airflow_residual": observed_requested - observed_delivered,
            }

    processing_id = config.processing_zone().id
    outbound_ids = [
        connection.id
        for connection in config.connections
        if connection.to_zone == processing_id
    ]
    observed_total = sum(
        observed_connections[connection_id]["delivered_airflow"]
        for connection_id in outbound_ids
    )
    capacity = config.air_system.shared_airflow_capacity
    if observed_total > capacity:
        scale = capacity / observed_total
        if sum(
            observed_connections[connection_id]["delivered_airflow"] * scale
            for connection_id in outbound_ids
        ) > capacity:
            scale = math.nextafter(scale, 0.0)
        for zone in config.non_processing_zones():
            for connection in (
                config.path_to_processing(zone.id),
                config.path_from_processing(zone.id),
            ):
                entry = observed_connections[connection.id]
                entry["delivered_airflow"] *= scale
                entry["airflow_residual"] = (
                    entry["requested_airflow"] - entry["delivered_airflow"]
                )
    return observed_positions, observed_connections
