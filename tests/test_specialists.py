"""V6 conditional specialist and policy contracts."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from aeolus.config import load_scenario
from aeolus.residual_features import ResidualFeatureProjector
from aeolus.scenario import run_scenario
from aeolus.specialists import (
    ConditionalRuleParameters,
    PhysicalFlowSpecialist,
    SensorHealthSpecialist,
    V6DecisionPolicy,
)

ROOT = Path(__file__).resolve().parents[1]
STANDARD = ROOT / "scenarios" / "standard_habitat.json"


def _config():
    return load_scenario(STANDARD)


def _window() -> list:
    records = copy.deepcopy(run_scenario(_config())[:10])
    for index, record in enumerate(records):
        record.zones["cabin_a"]["sensor_co2_concentration"] = 0.2
        actuator = record.actuators["cabin_a"]
        actuator["setpoint"] = actuator["actual_position"] = index / 10
        actuator["tracking_residual"] = 0.0
        actuator["moving"] = 1.0 if index else 0.0
        actuator["movement_seconds"] = float(index)
        actuator["direction"] = 1.0 if index else 0.0
        for connection_id in ("cabin_a_to_processing", "cabin_b_to_processing", "lab_to_processing"):
            connection = record.connections[connection_id]
            connection["requested_airflow"] = 10.0
            connection["delivered_airflow"] = 10.0
            connection["airflow_residual"] = 0.0
        record.system["shared_airflow_capacity"] = 40.0
        record.system["total_requested_airflow"] = 30.0
        record.system["total_delivered_airflow"] = 30.0
        record.system["capacity_scale"] = 1.0
    return records


def _settle(window: list) -> None:
    for record in window:
        for zone_id in ("cabin_a", "cabin_b", "lab"):
            actuator = record.actuators[zone_id]
            actuator["setpoint"] = actuator["actual_position"] = 0.4
            actuator["moving"] = actuator["movement_seconds"] = actuator["direction"] = 0.0


def test_flat_sensor_needs_observable_response_opportunity_before_concern():
    config = _config()
    settled = _window()
    _settle(settled)
    active = _window()

    settled_assessment = SensorHealthSpecialist(ResidualFeatureProjector(config)).assess_window(settled)
    active_assessment = SensorHealthSpecialist(ResidualFeatureProjector(config)).assess_window(active)

    assert settled_assessment.zone_id is None
    assert settled_assessment.reason_code == "no_expected_change"
    assert active_assessment.zone_id == "cabin_a"
    assert active_assessment.reason_code == "flat_sensor_with_corroboration"
    assert V6DecisionPolicy(config).label_window(settled) == "nominal"
    assert V6DecisionPolicy(config).label_window(active) == "sensor_health_concern"


def test_shared_capacity_transient_cannot_become_physical_concern():
    window = _window()
    for record in window:
        _settle([record])
        for connection_id in ("cabin_a_to_processing", "cabin_b_to_processing", "lab_to_processing"):
            connection = record.connections[connection_id]
            connection["requested_airflow"] = 10.0
            connection["delivered_airflow"] = 6.0
            connection["airflow_residual"] = 4.0
        record.system["total_requested_airflow"] = 30.0
        record.system["total_delivered_airflow"] = 18.0
        record.system["capacity_scale"] = 0.6

    assessment = PhysicalFlowSpecialist(ResidualFeatureProjector(_config())).assess_window(window)

    assert assessment.loop_zone_id is None
    assert assessment.reason_code == "shared_capacity_transient"
    assert V6DecisionPolicy(_config()).label_window(window) == "nominal"


def test_conflicting_sensor_and_physical_evidence_is_uncertain():
    window = _window()
    for index, record in enumerate(window):
        local = record.actuators["cabin_a"]
        local["setpoint"] = local["actual_position"] = 0.4
        local["moving"] = local["movement_seconds"] = local["direction"] = 0.0
        sibling = record.actuators["cabin_b"]
        sibling["setpoint"] = sibling["actual_position"] = index / 10
        sibling["moving"] = 1.0 if index else 0.0
        sibling["movement_seconds"] = float(index)
        sibling["direction"] = 1.0 if index else 0.0
        connection = record.connections["cabin_a_to_processing"]
        connection["requested_airflow"] = 10.0
        connection["delivered_airflow"] = 5.0
        connection["airflow_residual"] = 5.0

    policy = V6DecisionPolicy(_config())

    assert policy.label_window(window) == "uncertain"


def test_policy_reset_cannot_carry_a_concern_across_streams():
    policy = V6DecisionPolicy(_config())
    active = _window()
    settled = _window()
    _settle(settled)

    assert policy.label_window(active) == "sensor_health_concern"
    policy.reset()
    assert policy.label_window(settled) == "nominal"


@pytest.mark.parametrize("field", ("sensor_max_delta", "expected_change_proxy", "residual_threshold"))
def test_specialist_thresholds_reject_nonfinite_values(field: str):
    values = ConditionalRuleParameters().__dict__.copy()
    values[field] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        ConditionalRuleParameters(**values)
