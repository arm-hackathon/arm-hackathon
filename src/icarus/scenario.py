"""Deterministic runs of a validated scenario graph.

A run is a fixed number of 1-second ticks over a :class:`HabitatConfig`.
No randomness, no wall clock, and no fault profile yet: the same scenario
file always produces the same records and a byte-identical trace.
"""

from contextlib import nullcontext
from dataclasses import dataclass

from icarus.config import HabitatConfig
from icarus.plant import initial_state, step_habitat
from icarus.trace import TickRecord, TraceWriter


@dataclass(frozen=True)
class RunSpec:
    """Declared constants of a run: length, warm-up window, cabin ceiling."""

    total_ticks: int
    warmup_ticks: int
    crew_cabin_co2_ceiling: float


# Declared run constants for the standard habitat. The ceiling is the
# scenario's declared crew-cabin CO2 ceiling, checked after the warm-up
# window; healthy cabins settle well below it.
STANDARD_RUN = RunSpec(
    total_ticks=120,
    warmup_ticks=60,
    crew_cabin_co2_ceiling=30.0,
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
    """Snapshot every zone's CO2 and every connection's airflow and health."""
    processing_id = config.processing_zone().id
    zones: dict[str, dict[str, float]] = {}
    for zone in config.zones:
        entry = {"co2": state.zone_co2[zone.id]}
        if zone.id == processing_id:
            entry["captured_co2"] = state.captured_co2
        zones[zone.id] = entry
    connections = {
        connection.id: {
            "airflow": airflows[connection.id],
            "health": connection.health,
        }
        for connection in config.connections
    }
    return TickRecord(tick=state.tick, zones=zones, connections=connections)
