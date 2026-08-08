"""Tests for the hub-layout ventilation plant model."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from aeolus.actuator import ActuatorState
from aeolus.config import ConnectionSpec, load_scenario, parse_scenario
from aeolus.plant import initial_state, path_airflow, step_habitat

RECOVERY_SCENARIO_PATH = Path(__file__).resolve().parents[1] / "scenarios" / "recovery_habitat.json"


def test_initial_state_has_empty_zones_and_zero_capture(standard_scenario_path):
    config = load_scenario(standard_scenario_path)
    state = initial_state(config)

    assert state.tick == 0
    assert state.zone_co2_mass == {
        "cabin_a": 0.0,
        "cabin_b": 0.0,
        "lab": 0.0,
        "processing": 0.0,
    }
    assert state.captured_co2 == 0.0
    assert all(
        reading == 0.0 for reading in state.sensor_co2_concentration.values()
    )
    assert all(source == 0.0 for source in state.source_co2_mass.values())
    assert set(state.actuators) == {"cabin_a", "cabin_b", "lab"}
    assert all(actuator.actual_position == 0.0 for actuator in state.actuators.values())


def test_path_airflow_is_requested_capacity_times_measured_position():
    connection = ConnectionSpec(
        id="c", from_zone="a", to_zone="b", max_airflow=10.0, health=0.35
    )

    assert path_airflow(connection) == pytest.approx(10.0)
    assert path_airflow(connection, actuator_position=0.5) == pytest.approx(5.0)


def test_one_tick_uses_concentration_and_shared_return(standard_scenario_path):
    config = load_scenario(standard_scenario_path)
    state, _ = step_habitat(config, initial_state(config))

    assert state.tick == 1
    source_a = state.source_co2_mass["cabin_a"]
    source_b = state.source_co2_mass["cabin_b"]
    assert 0.82 <= source_a <= 1.18
    assert 0.62 <= source_b <= 0.98
    assert state.sensor_co2_concentration["cabin_a"] == pytest.approx(
        source_a / 100.0
    )
    assert state.sensor_co2_concentration["cabin_b"] == pytest.approx(
        source_b / 100.0
    )
    assert all(
        actuator.actual_position == pytest.approx(1.0 / 30.0)
        for actuator in state.actuators.values()
    )
    # The lab receives mixed return air even when its own source is small.
    assert state.zone_co2_mass["lab"] > 0.0
    assert state.zone_co2_mass["processing"] == 0.0
    assert state.captured_co2 > 0.0


def test_one_tick_conserves_co2(standard_scenario_path):
    config = load_scenario(standard_scenario_path)
    state, _ = step_habitat(config, initial_state(config))

    generated = sum(state.source_co2_mass.values())
    assert sum(state.zone_co2_mass.values()) + state.captured_co2 == pytest.approx(
        generated
    )


def test_every_connection_reports_actual_airflow(standard_scenario_path):
    config = load_scenario(standard_scenario_path)
    _, airflows = step_habitat(config, initial_state(config))

    assert set(airflows) == {c.id for c in config.connections}
    assert airflows["cabin_a_to_processing"] == pytest.approx(1.0 / 3.0)
    # The cleaned air returns along the paired path at the same actual airflow.
    assert airflows["processing_to_cabin_a"] == pytest.approx(1.0 / 3.0)
    assert airflows["lab_to_processing"] == pytest.approx(8.0 / 30.0)


def test_zero_health_path_moves_no_air_and_scrubs_nothing(standard_doc):
    for connection in standard_doc["connections"]:
        if connection["id"] == "cabin_a_to_processing":
            connection["health"] = 0.0
    config = parse_scenario(standard_doc)

    state, airflows = step_habitat(config, initial_state(config))

    assert airflows["cabin_a_to_processing"] == 0.0
    assert airflows["processing_to_cabin_a"] == 0.0
    # cabin_a keeps its whole tick of CO2; cabin_b is still scrubbed.
    source_a = state.source_co2_mass["cabin_a"]
    assert state.zone_co2_mass["cabin_a"] == pytest.approx(source_a)
    assert state.captured_co2 > 0.0


def test_rising_sensor_co2_increases_actuator_command(standard_scenario_path):
    config = load_scenario(standard_scenario_path)
    state = initial_state(config)
    commands = []

    for _ in range(10):
        state, _ = step_habitat(config, state)
        commands.append(state.actuators["cabin_a"].setpoint)

    assert commands[-1] > commands[0]


def test_weaker_return_path_limits_controlled_loop_airflow(standard_doc):
    for connection in standard_doc["connections"]:
        if connection["id"] == "processing_to_cabin_a":
            connection["health"] = 0.5
    config = parse_scenario(standard_doc)

    _, airflows = step_habitat(config, initial_state(config))

    assert airflows["cabin_a_to_processing"] == pytest.approx(1.0 / 6.0)
    assert airflows["processing_to_cabin_a"] == pytest.approx(1.0 / 6.0)


def test_shared_capacity_limits_total_zone_airflow(standard_scenario_path):
    config = load_scenario(standard_scenario_path)
    state = initial_state(config)
    constrained_totals = []

    for _ in range(120):
        state, airflows = step_habitat(config, state)
        if state.capacity_scale < 1.0:
            constrained_totals.append(
                sum(
                    airflows[config.path_to_processing(zone.id).id]
                    for zone in config.non_processing_zones()
                )
            )

    assert constrained_totals
    assert all(
        total == pytest.approx(config.air_system.shared_airflow_capacity)
        for total in constrained_totals
    )


def test_occupancy_profile_changes_source_baseline(standard_scenario_path):
    config = load_scenario(standard_scenario_path)
    state = initial_state(config)
    sources = []

    for _ in range(50):
        state, _ = step_habitat(config, state)
        sources.append(state.source_co2_mass["cabin_a"])

    assert max(sources[40:]) > max(sources[:40])


def _primary_state_projection(state):
    return (
        state.tick,
        state.zone_co2_mass,
        state.captured_co2,
        state.sensor_co2_concentration,
        state.source_co2_mass,
        state.source_noise,
        state.occupancy_multiplier,
        state.actuators,
        state.requested_airflows,
        state.delivered_airflows,
        state.airflow_residuals,
        state.capacity_scale,
        state.frozen_sensor_readings,
    )


def _full_position_state(config):
    state = initial_state(config)
    full = {zone_id: ActuatorState(setpoint=1.0, actual_position=1.0) for zone_id in state.actuators}
    reserve = replace(
        state.reserve,
        actuators={
            zone_id: ActuatorState(setpoint=1.0, actual_position=1.0)
            for zone_id in state.reserve.actuators
        },
    )
    return replace(state, actuators=full, reserve=reserve)


def test_recovery_initial_state_has_independent_zeroed_reserve_plane():
    config = load_scenario(RECOVERY_SCENARIO_PATH)
    state = initial_state(config)

    assert set(state.reserve.actuators) == {"cabin_a", "cabin_b", "lab"}
    assert all(actuator.actual_position == 0.0 for actuator in state.reserve.actuators.values())
    expected_paths = {connection.id for connection in config.reserve_connections}
    assert set(state.reserve.requested_airflows) == expected_paths
    assert set(state.reserve.delivered_airflows) == expected_paths
    assert set(state.reserve.airflow_residuals) == expected_paths
    assert state.reserve.capacity_scale == 1.0
    assert state.reserve.total_power == 0.0


def test_zero_reserve_is_exactly_equivalent_to_legacy_physics(standard_scenario_path):
    legacy = load_scenario(standard_scenario_path)
    recovery = load_scenario(RECOVERY_SCENARIO_PATH)
    legacy_state = initial_state(legacy)
    recovery_state = initial_state(recovery)

    for _ in range(20):
        legacy_state, legacy_flows = step_habitat(legacy, legacy_state)
        recovery_state, recovery_flows = step_habitat(recovery, recovery_state)
        assert _primary_state_projection(recovery_state) == _primary_state_projection(legacy_state)
        assert recovery_flows == legacy_flows
        assert all(value == 0.0 for value in recovery_state.reserve.delivered_airflows.values())


@pytest.mark.parametrize(
    "commands",
    [
        {"cabin_a": 1.0},
        {"cabin_a": 1.0, "cabin_b": 0.0, "lab": 0.0, "extra": 0.0},
        {"cabin_a": True, "cabin_b": 0.0, "lab": 0.0},
        {"cabin_a": float("nan"), "cabin_b": 0.0, "lab": 0.0},
        {"cabin_a": -0.1, "cabin_b": 0.0, "lab": 0.0},
        {"cabin_a": 1.1, "cabin_b": 0.0, "lab": 0.0},
    ],
)
def test_reserve_command_boundary_rejects_malformed_mappings(commands):
    config = load_scenario(RECOVERY_SCENARIO_PATH)

    with pytest.raises(ValueError, match="reserve commands"):
        step_habitat(config, initial_state(config), reserve_commands=commands)


def test_one_reserve_command_uses_paired_paths_and_rate_limited_actuator():
    config = load_scenario(RECOVERY_SCENARIO_PATH)
    commands = {"cabin_a": 1.0, "cabin_b": 0.0, "lab": 0.0}

    state, _ = step_habitat(
        config, initial_state(config), reserve_commands=commands
    )

    assert state.reserve.actuators["cabin_a"].actual_position == pytest.approx(1.0 / 30.0)
    outbound = config.reserve_path_to_processing("cabin_a").id
    inbound = config.reserve_path_from_processing("cabin_a").id
    assert state.reserve.requested_airflows[outbound] == pytest.approx(4.0 / 30.0)
    assert state.reserve.delivered_airflows[outbound] == pytest.approx(4.0 / 30.0)
    assert state.reserve.delivered_airflows[inbound] == pytest.approx(4.0 / 30.0)


def test_reserve_shared_capacity_scales_multi_zone_delivery_proportionally():
    config = load_scenario(RECOVERY_SCENARIO_PATH)
    state = _full_position_state(config)

    next_state, _ = step_habitat(
        config,
        state,
        reserve_commands={"cabin_a": 1.0, "cabin_b": 1.0, "lab": 1.0},
    )

    outbound_flows = [
        next_state.reserve.delivered_airflows[
            config.reserve_path_to_processing(zone.id).id
        ]
        for zone in config.non_processing_zones()
    ]
    assert sum(outbound_flows) == pytest.approx(4.0)
    assert outbound_flows == pytest.approx([4.0 / 3.0] * 3)
    assert next_state.reserve.capacity_scale == pytest.approx(1.0 / 3.0)


def test_primary_fault_effectiveness_cannot_reduce_reserve_delivery():
    config = load_scenario(RECOVERY_SCENARIO_PATH)
    state = _full_position_state(config)
    commands = {"cabin_a": 1.0, "cabin_b": 0.0, "lab": 0.0}

    healthy, _ = step_habitat(config, state, reserve_commands=commands)
    faulted, _ = step_habitat(
        config,
        state,
        reserve_commands=commands,
        connection_effectiveness={"cabin_a_to_processing": 0.05},
    )

    assert faulted.reserve == healthy.reserve
    assert faulted.delivered_airflows["cabin_a_to_processing"] < healthy.delivered_airflows[
        "cabin_a_to_processing"
    ]


def test_reserve_demand_cannot_reduce_primary_delivery():
    config = load_scenario(RECOVERY_SCENARIO_PATH)
    state = _full_position_state(config)

    reserve_off, off_primary = step_habitat(
        config,
        state,
        reserve_commands={"cabin_a": 0.0, "cabin_b": 0.0, "lab": 0.0},
    )
    reserve_on, on_primary = step_habitat(
        config,
        state,
        reserve_commands={"cabin_a": 1.0, "cabin_b": 1.0, "lab": 1.0},
    )

    assert on_primary == off_primary
    assert reserve_on.requested_airflows == reserve_off.requested_airflows
    assert reserve_on.delivered_airflows == reserve_off.delivered_airflows
    assert reserve_on.capacity_scale == reserve_off.capacity_scale


def test_zero_health_on_one_reserve_leg_blocks_only_that_loop():
    document = json.loads(RECOVERY_SCENARIO_PATH.read_text(encoding="utf-8"))
    for connection in document["reserve_connections"]:
        if connection["id"] == "reserve_processing_to_cabin_a":
            connection["health"] = 0.0
    config = parse_scenario(document)
    state = _full_position_state(config)

    next_state, _ = step_habitat(
        config,
        state,
        reserve_commands={"cabin_a": 1.0, "cabin_b": 1.0, "lab": 0.0},
    )

    a_outbound = config.reserve_path_to_processing("cabin_a").id
    b_outbound = config.reserve_path_to_processing("cabin_b").id
    assert next_state.reserve.delivered_airflows[a_outbound] == 0.0
    assert next_state.reserve.delivered_airflows[b_outbound] > 0.0


def test_primary_and_reserve_flow_are_scrubbed_once_with_mass_conservation():
    config = load_scenario(RECOVERY_SCENARIO_PATH)
    state = _full_position_state(config)
    state = replace(
        state,
        zone_co2_mass={"cabin_a": 40.0, "cabin_b": 30.0, "lab": 10.0, "processing": 0.0},
    )
    before_total = sum(state.zone_co2_mass.values()) + state.captured_co2

    next_state, _ = step_habitat(
        config,
        state,
        reserve_commands={"cabin_a": 1.0, "cabin_b": 0.0, "lab": 0.0},
    )
    generated = sum(next_state.source_co2_mass.values())
    pre_transfer = {
        zone_id: state.zone_co2_mass[zone_id] + next_state.source_co2_mass[zone_id]
        for zone_id in state.zone_co2_mass
    }
    expected_extracted = 0.0
    for zone in config.non_processing_zones():
        primary = next_state.delivered_airflows[config.path_to_processing(zone.id).id]
        reserve = next_state.reserve.delivered_airflows[
            config.reserve_path_to_processing(zone.id).id
        ]
        expected_extracted += pre_transfer[zone.id] * min(
            (primary + reserve) / zone.air_volume, 1.0
        )
    captured_delta = next_state.captured_co2 - state.captured_co2

    assert captured_delta == pytest.approx(
        expected_extracted * config.air_system.scrubber_removal_fraction
    )
    assert sum(next_state.zone_co2_mass.values()) + next_state.captured_co2 == pytest.approx(
        before_total + generated, abs=1e-9
    )


def test_ninety_five_percent_primary_loss_saturates_without_claiming_full_restoration():
    config = load_scenario(RECOVERY_SCENARIO_PATH)
    state = _full_position_state(config)
    state = replace(
        state,
        actuators={
            zone_id: (
                ActuatorState(setpoint=1.0, actual_position=1.0)
                if zone_id == "cabin_a"
                else ActuatorState()
            )
            for zone_id in state.actuators
        },
        reserve=replace(
            state.reserve,
            actuators={
                zone_id: (
                    ActuatorState(setpoint=1.0, actual_position=1.0)
                    if zone_id == "cabin_a"
                    else ActuatorState()
                )
                for zone_id in state.reserve.actuators
            },
        ),
    )
    commands = {"cabin_a": 1.0, "cabin_b": 0.0, "lab": 0.0}
    primary_commands = {"cabin_a": 1.0, "cabin_b": 0.0, "lab": 0.0}

    healthy, _ = step_habitat(
        config,
        state,
        override_commands=primary_commands,
        reserve_commands={zone: 0.0 for zone in commands},
    )
    faulted, _ = step_habitat(
        config,
        state,
        override_commands=primary_commands,
        reserve_commands=commands,
        connection_effectiveness={"cabin_a_to_processing": 0.05},
    )
    primary_id = config.path_to_processing("cabin_a").id
    reserve_id = config.reserve_path_to_processing("cabin_a").id
    lost = healthy.delivered_airflows[primary_id] - faulted.delivered_airflows[primary_id]
    restored = faulted.reserve.delivered_airflows[reserve_id]

    assert restored == pytest.approx(4.0)
    assert restored / lost == pytest.approx(4.0 / 9.5)
    assert restored / lost < 0.5
