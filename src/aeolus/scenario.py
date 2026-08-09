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
from aeolus.model_input import build_model_input_contract, model_input_v1
from aeolus.plant import (
    HabitatState,
    initial_state,
    requested_loop_airflow,
    step_habitat,
)
from aeolus.recovery import (
    AuthorityEvent,
    AuthorityState,
    DeterministicRecoverySupervisor,
    RecoveryDecision,
    RecoveryObservation,
    ReserveCommandOwner,
    _decision_digest,
    validate_recovery_decision,
)
from aeolus.trace import (
    RecoveryTickRecord,
    RecoveryTraceWriter,
    TickRecord,
    TraceWriter,
)


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


@dataclass(frozen=True)
class RecoveryRunResult:
    """Observable recovery rows plus evaluator-only physical replay state."""

    records: tuple[RecoveryTickRecord, ...]
    states: tuple[HabitatState, ...]
    decisions: tuple[RecoveryDecision, ...]
    events: tuple[AuthorityEvent, ...]


# Declared run constants for the standard habitat. The ceiling is the
# declared crew-cabin concentration ceiling for the measured run.
STANDARD_RUN = RunSpec(
    total_ticks=120,
    warmup_ticks=60,
    crew_cabin_co2_concentration_ceiling=0.30,
)
RECOVERY_RUN = RunSpec(
    total_ticks=180,
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
    return _run_scenario(config, run=run, trace_path=trace_path, governor=None)


def run_governed_scenario(
    config: HabitatConfig,
    governor,
    *,
    run: RunSpec = STANDARD_RUN,
    trace_path=None,
) -> list[TickRecord]:
    """Run one scenario under an external bounded-response governor.

    The governor is a causal decision maker: before each measured tick it
    returns bounded per-zone commands via :meth:`next_commands` using only
    ``model_input_v1`` vectors of completed ticks (fed through
    :meth:`observe`). It never sees hidden fault truth, so its behaviour is
    observable-only by construction. ``governor.reset()`` is called before the
    run, mirroring the streaming evaluator contract.
    """
    reset = getattr(governor, "reset", None)
    if callable(reset):
        reset()
    return _run_scenario(config, run=run, trace_path=trace_path, governor=governor)


def run_recovery_scenario(
    config: HabitatConfig,
    *,
    run_id: str,
    governed: bool,
    run: RunSpec = RECOVERY_RUN,
    trace_path=None,
) -> RecoveryRunResult:
    """Run one causal reserve-recovery arm through the shared plant."""
    if config.version != 10 or not config.reserve_connections:
        raise ValueError("recovery runs require a validated version-10 topology")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("recovery run_id must be non-empty")

    state = initial_state(config)
    for warmup_index in range(run.warmup_ticks):
        state, _ = step_habitat(
            config,
            state,
            source_tick=warmup_index - run.warmup_ticks,
            occupancy_tick=1,
        )
    state = replace(state, tick=0, captured_co2=0.0)

    contract = build_model_input_contract(config)
    supervisor = (
        DeterministicRecoverySupervisor(config, run_id=run_id, contract=contract)
        if governed
        else None
    )
    decision = (
        supervisor.cold_start(decision_tick=1)
        if supervisor is not None
        else _reserve_off_decision(config, run_id=run_id, decision_tick=1)
    )
    records: list[RecoveryTickRecord] = []
    states: list[HabitatState] = []
    decisions: list[RecoveryDecision] = []
    writer_context = (
        RecoveryTraceWriter(trace_path, config) if trace_path is not None else nullcontext()
    )
    with writer_context as writer:
        for tick in range(1, run.total_ticks + 1):
            expected_state = (
                supervisor.state if supervisor is not None else AuthorityState.NOMINAL
            )
            expected_owner = (
                ReserveCommandOwner.DETERMINISTIC_RECOVERY_SUPERVISOR
                if expected_state in (AuthorityState.PROTECT, AuthorityState.HANDBACK)
                else ReserveCommandOwner.RESERVE_OFF
            )
            reserve_commands = validate_recovery_decision(
                decision,
                config=config,
                expected_run_id=run_id,
                expected_authority_epoch=(
                    supervisor.authority_epoch if supervisor is not None else 0
                ),
                expected_decision_tick=tick,
                expected_observation_tick=tick - 1,
                expected_sequence=tick,
                expected_state=expected_state,
                expected_owner=expected_owner,
            )
            effectiveness = _connection_effectiveness(config, tick)
            frozen_zones = _frozen_sensor_zones(config, tick)
            state, airflows = step_habitat(
                config,
                state,
                connection_effectiveness=effectiveness,
                frozen_zones=frozen_zones,
                reserve_commands=reserve_commands,
                source_tick=run.warmup_ticks + tick,
                occupancy_tick=tick,
            )
            plant_record = _tick_record(config, state, airflows)
            record = RecoveryTickRecord(
                plant=plant_record,
                reserve=_reserve_trace_telemetry(config, state),
                authority=_authority_telemetry(decision),
            )
            if writer is not None:
                writer.write(record)
            records.append(record)
            states.append(state)
            decisions.append(decision)

            if tick == run.total_ticks:
                continue
            decision = (
                supervisor.decide(
                    _recovery_observation(config, contract, supervisor, record)
                )
                if supervisor is not None
                else _reserve_off_decision(
                    config, run_id=run_id, decision_tick=tick + 1
                )
            )

    events = tuple(supervisor.event_history) if supervisor is not None else ()
    return RecoveryRunResult(
        records=tuple(records),
        states=tuple(states),
        decisions=tuple(decisions),
        events=events,
    )


def _run_scenario(
    config: HabitatConfig,
    *,
    run: RunSpec,
    trace_path,
    governor,
) -> list[TickRecord]:
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
    contract = build_model_input_contract(config) if governor is not None else None
    if governor is not None:
        # Pre-seed the governor's causal window with the warm-up observations
        # so its first measured command uses the same observation basis as the
        # baseline controller at measured tick 1 (no artificial edge lag).
        warmup_state = initial_state(config)
        warmup_window: list[list[float]] = []
        for warmup_index in range(run.warmup_ticks):
            warmup_state, warmup_airflows = step_habitat(
                config,
                warmup_state,
                source_tick=warmup_index - run.warmup_ticks,
                occupancy_tick=1,
            )
            warmup_window.append(
                model_input_v1(
                    _tick_record(config, warmup_state, warmup_airflows), contract
                ).tolist()
            )
        try:
            window_ticks = getattr(
                getattr(governor, "settings", None), "window_ticks", None
            )
        except Exception:
            window_ticks = None
        if (
            isinstance(window_ticks, bool)
            or not isinstance(window_ticks, int)
            or window_ticks < 1
        ):
            keep = run.warmup_ticks
        else:
            keep = min(window_ticks, run.warmup_ticks)
        for vector in warmup_window[-keep:]:
            governor.observe(vector)

    writer_context = (
        TraceWriter(trace_path) if trace_path is not None else nullcontext(None)
    )
    with writer_context as writer:
        while state.tick < run.total_ticks:
            next_tick = state.tick + 1
            override_commands = None
            if governor is not None:
                override_commands, _ = governor.next_commands()
            state, airflows = step_habitat(
                config,
                state,
                connection_effectiveness=_connection_effectiveness(config, next_tick),
                frozen_zones=_frozen_sensor_zones(config, next_tick),
                override_commands=override_commands,
            )
            record = _tick_record(config, state, airflows)
            records.append(record)
            if governor is not None:
                governor.observe(model_input_v1(record, contract).tolist())
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
        while sum(
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


def _reserve_off_decision(
    config: HabitatConfig, *, run_id: str, decision_tick: int
) -> RecoveryDecision:
    commands = {zone.id: 0.0 for zone in config.non_processing_zones()}
    reason = "cold_start" if decision_tick == 1 else "no_concern"
    digest = _decision_digest(
        run_id=run_id,
        authority_epoch=0,
        decision_tick=decision_tick,
        observation_tick=decision_tick - 1,
        sequence=decision_tick,
        state=AuthorityState.NOMINAL,
        owner=ReserveCommandOwner.RESERVE_OFF,
        target_zone_id=None,
        reserve_commands=commands,
        reason=reason,
        dwell_ticks=0,
    )
    return RecoveryDecision(
        run_id=run_id,
        authority_epoch=0,
        decision_tick=decision_tick,
        observation_tick=decision_tick - 1,
        sequence=decision_tick,
        state=AuthorityState.NOMINAL,
        reserve_command_owner=ReserveCommandOwner.RESERVE_OFF,
        target_zone_id=None,
        reserve_commands=commands,
        reason=reason,
        command_digest=digest,
        dwell_ticks=0,
    )


def _authority_telemetry(decision: RecoveryDecision) -> dict[str, object]:
    return {
        "run_id": decision.run_id,
        "authority_epoch": decision.authority_epoch,
        "decision_tick": decision.decision_tick,
        "sequence": decision.sequence,
        "state": decision.state.value,
        "reserve_command_owner": decision.reserve_command_owner.value,
        "target_zone_id": decision.target_zone_id,
        "reason": decision.reason,
        "dwell_ticks": decision.dwell_ticks,
        "observation_tick": decision.observation_tick,
        "command_digest": decision.command_digest,
        "applied_command_digest": decision.command_digest,
    }


def _reserve_trace_telemetry(
    config: HabitatConfig, state: HabitatState
) -> dict[str, object]:
    settings = config.telemetry
    connections: dict[str, dict[str, float]] = {}
    actuators: dict[str, dict[str, float | bool]] = {}
    requested_total = 0.0
    delivered_total = 0.0
    for zone in config.non_processing_zones():
        actuator = state.reserve.actuators[zone.id]
        position_factor = 1.0 + (
            settings.actuator_position_noise_fraction
            * deterministic_measurement_sample(
                config, zone.id, "reserve_actuator_position", state.tick
            )
        )
        observed_position = max(
            0.0, min(1.0, actuator.actual_position * position_factor)
        )
        outbound = config.reserve_path_to_processing(zone.id)
        inbound = config.reserve_path_from_processing(zone.id)
        observed_requested = requested_loop_airflow(
            outbound, inbound, observed_position
        )
        flow_factor = 1.0 + (
            settings.airflow_bias_fraction
            * deterministic_measurement_sample(
                config, outbound.id, "reserve_airflow_bias", 0
            )
            + settings.airflow_drift_fraction
            * deterministic_measurement_drift(
                config, outbound.id, "reserve_airflow", state.tick
            )
            + settings.airflow_noise_fraction
            * deterministic_measurement_sample(
                config, outbound.id, "reserve_airflow_noise", state.tick
            )
        )
        observed_delivered = max(
            0.0,
            min(
                observed_requested,
                state.reserve.delivered_airflows[outbound.id] * max(0.0, flow_factor),
            ),
        )
        entry = {
            "requested_airflow": observed_requested,
            "delivered_airflow": observed_delivered,
            "airflow_residual": observed_requested - observed_delivered,
        }
        connections[outbound.id] = dict(entry)
        connections[inbound.id] = dict(entry)
        actuators[zone.id] = {
            "setpoint": actuator.setpoint,
            "actual_position": observed_position,
            "tracking_residual": actuator.setpoint - observed_position,
            "moving": actuator.moving,
            "movement_seconds": actuator.movement_seconds,
            "power": actuator.power,
        }
        requested_total += observed_requested
        delivered_total += observed_delivered

    return {
        "connections": connections,
        "actuators": actuators,
        "system": {
            "reserve_airflow_capacity": config.air_system.reserve_airflow_capacity,
            "total_requested_airflow": requested_total,
            "total_delivered_airflow": delivered_total,
            "capacity_scale": state.reserve.capacity_scale,
            "total_power": sum(
                actuator.power for actuator in state.reserve.actuators.values()
            ),
        },
    }


def _recovery_observation(
    config: HabitatConfig,
    contract,
    supervisor: DeterministicRecoverySupervisor,
    record: RecoveryTickRecord,
) -> RecoveryObservation:
    zone_ids = tuple(zone.id for zone in config.non_processing_zones())
    primary_outbound = {
        zone_id: config.path_to_processing(zone_id).id for zone_id in zone_ids
    }
    reserve_outbound = {
        zone_id: config.reserve_path_to_processing(zone_id).id for zone_id in zone_ids
    }
    reserve_return = {
        zone_id: config.reserve_path_from_processing(zone_id).id for zone_id in zone_ids
    }
    return RecoveryObservation(
        run_id=supervisor.run_id,
        authority_epoch=supervisor.authority_epoch,
        completed_tick=record.plant.tick,
        sequence=int(record.authority["sequence"]),
        model_input_v1=tuple(float(value) for value in model_input_v1(record, contract)),
        selector_sha256=contract.selector_hash,
        topology_sha256=contract.topology_hash,
        zone_ids=zone_ids,
        primary_outbound_ids=primary_outbound,
        reserve_outbound_ids=reserve_outbound,
        reserve_return_ids=reserve_return,
        co2_concentration={
            zone_id: float(record.plant.zones[zone_id]["sensor_co2_concentration"])
            for zone_id in zone_ids
        },
        primary_requested_airflow={
            zone_id: float(
                record.plant.connections[primary_outbound[zone_id]][
                    "requested_airflow"
                ]
            )
            for zone_id in zone_ids
        },
        primary_delivered_airflow={
            zone_id: float(
                record.plant.connections[primary_outbound[zone_id]][
                    "delivered_airflow"
                ]
            )
            for zone_id in zone_ids
        },
        reserve_actual_position={
            zone_id: float(
                record.reserve["actuators"][zone_id]["actual_position"]
            )
            for zone_id in zone_ids
        },
        reserve_delivered_airflow={
            zone_id: float(
                record.reserve["connections"][reserve_outbound[zone_id]][
                    "delivered_airflow"
                ]
            )
            for zone_id in zone_ids
        },
        applied_command_digest=str(record.authority["applied_command_digest"]),
    )
