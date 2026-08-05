"""Causal V6 residual feature contracts."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from aeolus.config import load_scenario
from aeolus.residual_features import ResidualFeatureProjector
from aeolus.scenario import run_scenario

ROOT = Path(__file__).resolve().parents[1]
STANDARD = ROOT / "scenarios" / "standard_habitat.json"


def _projector() -> ResidualFeatureProjector:
    return ResidualFeatureProjector(load_scenario(STANDARD))


def _window() -> list:
    records = copy.deepcopy(run_scenario(load_scenario(STANDARD))[:10])
    for tick, record in enumerate(records):
        actuator = record.actuators["cabin_a"]
        actuator["setpoint"] = tick / 10
        actuator["actual_position"] = tick / 10
        actuator["tracking_residual"] = 0.0
        actuator["moving"] = 1.0 if tick else 0.0
        actuator["movement_seconds"] = float(tick)
        actuator["direction"] = 1.0 if tick else 0.0
        record.zones["cabin_a"]["sensor_co2_concentration"] = 0.10 + 0.01 * tick
        outbound = record.connections["cabin_a_to_processing"]
        outbound["requested_airflow"] = 10.0
        outbound["delivered_airflow"] = 10.0 if tick < 9 else 7.0
        outbound["airflow_residual"] = 0.0 if tick < 9 else 3.0
        for connection_id in ("cabin_b_to_processing", "lab_to_processing"):
            connection = record.connections[connection_id]
            connection["requested_airflow"] = 10.0
            connection["delivered_airflow"] = 10.0
            connection["airflow_residual"] = 0.0
        record.system["shared_airflow_capacity"] = 40.0
        record.system["total_requested_airflow"] = 30.0
        record.system["total_delivered_airflow"] = 27.0 if tick == 9 else 30.0
        record.system["capacity_scale"] = 0.9 if tick == 9 else 1.0
    return records


def test_sensor_and_physical_residuals_are_exact_and_causal():
    projector = _projector()
    window = _window()

    sensor = projector.sensor_features(window, "cabin_a")
    physical = projector.physical_features(window, "cabin_a")

    assert sensor.sensor_slope == pytest.approx(0.01)
    assert sensor.sensor_range == pytest.approx(0.09)
    assert sensor.sensor_max_delta == pytest.approx(0.01)
    assert sensor.actuator_actual_span == pytest.approx(0.9)
    assert sensor.outbound_delivery_change == pytest.approx(-3.0)
    assert sensor.expected_change_proxy == pytest.approx(0.9)
    assert physical.request == pytest.approx(10.0)
    assert physical.delivery == pytest.approx(7.0)
    assert physical.residual == pytest.approx(3.0)
    assert physical.normalized_residual == pytest.approx(0.3)
    assert physical.residual_slope == pytest.approx(3.0 / 9.0)
    assert physical.residual_max_jump == pytest.approx(3.0)
    assert physical.residual_persistence == pytest.approx(0.1)
    assert physical.isolation_ratio == pytest.approx(1.0)
    assert physical.capacity_headroom == pytest.approx(13.0 / 40.0)
    assert physical.transient_proxy == pytest.approx(0.9)


def test_zero_request_and_settled_sensor_are_not_false_expected_change():
    projector = _projector()
    window = _window()
    for record in window:
        record.zones["cabin_a"]["sensor_co2_concentration"] = 0.2
        for actuator_id in ("cabin_a", "cabin_b", "lab"):
            actuator = record.actuators[actuator_id]
            actuator["setpoint"] = actuator["actual_position"] = 0.4
            actuator["moving"] = actuator["movement_seconds"] = actuator["direction"] = 0.0
        for connection_id in ("cabin_a_to_processing", "cabin_b_to_processing", "lab_to_processing"):
            connection = record.connections[connection_id]
            connection["requested_airflow"] = 0.0
            connection["delivered_airflow"] = 0.0
            connection["airflow_residual"] = 0.0
        record.system["total_requested_airflow"] = 0.0
        record.system["total_delivered_airflow"] = 0.0
        record.system["capacity_scale"] = 1.0

    sensor = projector.sensor_features(window, "cabin_a")
    physical = projector.physical_features(window, "cabin_a")

    assert sensor.sensor_max_delta == 0.0
    assert sensor.expected_change_proxy == 0.0
    assert physical.normalized_residual == 0.0
    assert physical.isolation_ratio == 0.0
    assert physical.residual_persistence == 0.0


def test_shared_capacity_contention_is_explicit_even_without_local_residual():
    projector = _projector()
    window = _window()
    for record in window:
        for actuator_id in ("cabin_a", "cabin_b", "lab"):
            actuator = record.actuators[actuator_id]
            actuator["setpoint"] = actuator["actual_position"] = 0.4
            actuator["moving"] = actuator["movement_seconds"] = actuator["direction"] = 0.0
        record.system["capacity_scale"] = 0.5
        record.system["shared_airflow_capacity"] = 40.0
        record.system["total_requested_airflow"] = 40.0
        record.system["total_delivered_airflow"] = 20.0
        for connection_id in ("cabin_a_to_processing", "cabin_b_to_processing", "lab_to_processing"):
            connection = record.connections[connection_id]
            connection["requested_airflow"] = 10.0
            connection["delivered_airflow"] = 10.0
            connection["airflow_residual"] = 0.0

    physical = projector.physical_features(window, "cabin_a")

    assert physical.normalized_residual == 0.0
    assert physical.capacity_contention == pytest.approx(0.5)
    assert physical.transient_proxy == pytest.approx(0.5)
    assert physical.settled_proxy == pytest.approx(0.5)


def test_flat_sensor_with_observable_reconfiguration_has_expected_change():
    projector = _projector()
    window = _window()
    for record in window:
        record.zones["cabin_a"]["sensor_co2_concentration"] = 0.2

    sensor = projector.sensor_features(window, "cabin_a")

    assert sensor.sensor_max_delta == 0.0
    assert sensor.expected_change_proxy == pytest.approx(0.9)


def test_hidden_truth_changes_cannot_change_residual_features():
    projector = _projector()
    observable = _window()
    changed_truth = copy.deepcopy(observable)
    for record in changed_truth:
        zone = record.zones["cabin_a"]
        zone["co2_mass"] = 999.0
        zone["source_co2_mass"] = 777.0
        zone["occupancy_multiplier"] = 42.0

    assert projector.sensor_features(observable, "cabin_a") == projector.sensor_features(changed_truth, "cabin_a")
    assert projector.physical_features(observable, "cabin_a") == projector.physical_features(changed_truth, "cabin_a")
