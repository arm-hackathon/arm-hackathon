"""Tests for the hub-layout ventilation plant model."""

import pytest

from icarus.config import ConnectionSpec, load_scenario, parse_scenario
from icarus.plant import initial_state, path_airflow, step_habitat


def test_initial_state_has_empty_zones_and_zero_capture(standard_scenario_path):
    config = load_scenario(standard_scenario_path)
    state = initial_state(config)

    assert state.tick == 0
    assert state.zone_co2 == {
        "cabin_a": 0.0,
        "cabin_b": 0.0,
        "lab": 0.0,
        "processing": 0.0,
    }
    assert state.captured_co2 == 0.0


def test_path_airflow_is_max_airflow_times_health():
    connection = ConnectionSpec(
        id="c", from_zone="a", to_zone="b", max_airflow=10.0, health=0.35
    )

    assert path_airflow(connection) == pytest.approx(3.5)


def test_one_tick_adds_sources_then_captures_declared_fraction(standard_scenario_path):
    config = load_scenario(standard_scenario_path)
    state, _ = step_habitat(config, initial_state(config))

    assert state.tick == 1
    # Each cabin: 1.0 unit added, then 0.5 * (10/100) of it captured.
    assert state.zone_co2["cabin_a"] == pytest.approx(0.95)
    assert state.zone_co2["cabin_b"] == pytest.approx(0.95)
    assert state.zone_co2["lab"] == 0.0
    assert state.zone_co2["processing"] == 0.0
    # The processing bay's counter accumulates both cabins' captures.
    assert state.captured_co2 == pytest.approx(0.10)


def test_one_tick_conserves_co2(standard_scenario_path):
    config = load_scenario(standard_scenario_path)
    state, _ = step_habitat(config, initial_state(config))

    generated = sum(z.co2_generation_per_second for z in config.zones)
    assert sum(state.zone_co2.values()) + state.captured_co2 == pytest.approx(generated)


def test_every_connection_reports_actual_airflow(standard_scenario_path):
    config = load_scenario(standard_scenario_path)
    _, airflows = step_habitat(config, initial_state(config))

    assert set(airflows) == {c.id for c in config.connections}
    assert airflows["cabin_a_to_processing"] == pytest.approx(10.0)
    # The cleaned air returns along the paired path at the same actual airflow.
    assert airflows["processing_to_cabin_a"] == pytest.approx(10.0)
    assert airflows["lab_to_processing"] == pytest.approx(8.0)


def test_zero_health_path_moves_no_air_and_scrubs_nothing(standard_doc):
    for connection in standard_doc["connections"]:
        if connection["id"] == "cabin_a_to_processing":
            connection["health"] = 0.0
    config = parse_scenario(standard_doc)

    state, airflows = step_habitat(config, initial_state(config))

    assert airflows["cabin_a_to_processing"] == 0.0
    assert airflows["processing_to_cabin_a"] == 0.0
    # cabin_a keeps its whole tick of CO2; cabin_b is still scrubbed.
    assert state.zone_co2["cabin_a"] == pytest.approx(1.0)
    assert state.zone_co2["cabin_b"] == pytest.approx(0.95)
    assert state.captured_co2 == pytest.approx(0.05)
