"""Deterministic runs of a validated scenario graph.

A run is an unrecorded warm-up followed by a fixed number of measured,
1-second ticks over a :class:`HabitatConfig`. Seeded variation, fixed inputs
and no wall clock ensure the same scenario produces a byte-identical trace.
"""

from contextlib import nullcontext
from dataclasses import dataclass, replace

from icarus.config import HabitatConfig
from icarus.plant import initial_state, step_habitat
from icarus.trace import TickRecord, TraceWriter


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
            state, airflows = step_habitat(config, state)
            record = _tick_record(config, state, airflows)
            records.append(record)
            if writer is not None:
                writer.write(record)
    return records


def _tick_record(
    config: HabitatConfig, state, airflows: dict[str, float]
) -> TickRecord:
    """Snapshot sensor, plant and actuator telemetry for every zone and path."""
    processing_id = config.processing_zone().id
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
    connections = {
        connection.id: {
            "requested_airflow": state.requested_airflows[connection.id],
            "airflow": airflows[connection.id],
            "health": connection.health,
        }
        for connection in config.connections
    }
    actuators = {
        zone_id: {
            "setpoint": actuator.setpoint,
            "actual_position": actuator.actual_position,
            "tracking_residual": actuator.tracking_residual,
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
                state.requested_airflows[connection.id]
                for connection in config.connections
                if connection.to_zone == processing_id
            ),
            "total_actual_airflow": sum(
                airflows[connection.id]
                for connection in config.connections
                if connection.to_zone == processing_id
            ),
            "capacity_scale": state.capacity_scale,
        },
    )
