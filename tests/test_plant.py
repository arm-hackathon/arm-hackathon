"""Tests for the hub-layout ventilation plant model."""

import pytest

from icarus.config import ConnectionSpec, load_scenario, parse_scenario
from icarus.plant import initial_state, path_airflow, step_habitat


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


def test_path_airflow_is_max_airflow_times_health():
    connection = ConnectionSpec(
        id="c", from_zone="a", to_zone="b", max_airflow=10.0, health=0.35
    )

    assert path_airflow(connection) == pytest.approx(3.5)
    assert path_airflow(connection, actuator_position=0.5) == pytest.approx(1.75)


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
