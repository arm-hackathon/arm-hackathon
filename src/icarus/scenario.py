"""Named, deterministic scenarios for the ICARUS proof loop.

A scenario is a fixed number of 1-second ticks plus an optional main-fan
fault applied at a declared tick. Running the same scenario name always
produces the same records: no randomness, no wall clock, fixed inputs.
"""

from contextlib import nullcontext
from dataclasses import dataclass

from icarus.plant import (
    PlantConfig,
    PlantState,
    main_fan_airflow,
    step_plant,
    with_main_fan_effectiveness,
)
from icarus.trace import TickRecord, TraceWriter


@dataclass(frozen=True)
class ScenarioSpec:
    """Declared constants of a named scenario."""

    name: str
    total_ticks: int
    fault_tick: int | None
    degraded_main_fan_effectiveness: float
    warmup_ticks: int
    nominal_co2_ceiling: float | None


SCENARIOS: dict[str, ScenarioSpec] = {
    "nominal": ScenarioSpec(
        name="nominal",
        total_ticks=120,
        fault_tick=None,
        degraded_main_fan_effectiveness=1.0,
        warmup_ticks=60,
        nominal_co2_ceiling=30.0,
    ),
    "primary_fan_degradation": ScenarioSpec(
        name="primary_fan_degradation",
        total_ticks=120,
        fault_tick=20,
        degraded_main_fan_effectiveness=0.35,
        warmup_ticks=60,
        nominal_co2_ceiling=None,
    ),
}


def run_scenario(
    name: str,
    *,
    config: PlantConfig | None = None,
    trace_path=None,
) -> list[TickRecord]:
    """Run a named scenario and return one record per tick, in tick order.

    When ``trace_path`` is given, each record is also appended to that file
    as one JSONL row, immediately after its tick is computed.
    """
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}")
    spec = SCENARIOS[name]
    config = config or PlantConfig()
    state = PlantState()
    records: list[TickRecord] = []

    writer_context = TraceWriter(trace_path) if trace_path is not None else nullcontext(None)
    with writer_context as writer:
        while state.tick < spec.total_ticks:
            # The fault takes effect for the tick that starts at fault_tick,
            # so the record written for that tick already shows the weakness.
            if spec.fault_tick is not None and state.tick + 1 == spec.fault_tick:
                state = with_main_fan_effectiveness(state, spec.degraded_main_fan_effectiveness)
            airflow = main_fan_airflow(config, state)
            state = step_plant(config, state)
            record = TickRecord(
                tick=state.tick,
                room_a_co2=state.room_a_co2,
                room_b_co2=state.room_b_co2,
                main_fan_effectiveness=state.main_fan_effectiveness,
                airflow=airflow,
            )
            records.append(record)
            if writer is not None:
                writer.write(record)
    return records
