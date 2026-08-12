from __future__ import annotations

from copy import deepcopy

import pytest

from aeolus.habitat_v2.runner import run_scenario
from aeolus.habitat_v2.scenario import (
    EQUATION_CONTRACT_REVISION_V3,
    SCENARIO_SCHEMA_VERSION_V4,
    TRACE_SCHEMA_VERSION_V4,
    Scenario,
    ScenarioValidationError,
)
from aeolus.habitat_v2.trace import validate_trace_bytes

from .test_air_network_scenario import scenario_v3_mapping


CHANNELS = {
    "temperature_k": 0.02,
    "pressure_pa": 2.0,
    "co2_ppm": 1.5,
    "o2_mole_fraction": 0.00001,
    "relative_humidity": 0.0005,
}


def scenario_v4_mapping() -> dict:
    mapping = deepcopy(scenario_v3_mapping())
    mapping["schema_version"] = "aeolus_habitat_v2_scenario_v4"
    mapping["sensor_model"] = {
        "random_seed": 20260812,
        "primary_noise_amplitude": dict(CHANNELS),
        "secondary_noise_amplitude": {
            channel: amplitude * 1.5 for channel, amplitude in CHANNELS.items()
        },
    }
    mapping["fault_profiles"] = []
    return mapping


def test_scenario_v4_has_distinct_schema_trace_and_equation_identity() -> None:
    scenario = Scenario.from_mapping(scenario_v4_mapping())

    assert scenario.scenario_schema_version == SCENARIO_SCHEMA_VERSION_V4
    assert scenario.trace_schema_version == TRACE_SCHEMA_VERSION_V4
    assert scenario.equation_contract_revision == EQUATION_CONTRACT_REVISION_V3


def test_healthy_v4_trace_exposes_redundant_sensors_and_truth_receipt() -> None:
    scenario = Scenario.from_mapping(scenario_v4_mapping())
    first = run_scenario(scenario)
    second = run_scenario(scenario)

    assert first.trace_bytes == second.trace_bytes
    assert validate_trace_bytes(first.trace_bytes, scenario=scenario) == first.rows

    for row in first.rows:
        assert set(row["sensor_disagreement"]) == {"crew_cabin", "work_airlock"}
        fault_receipt = row["fault_receipt"]
        if row["step"] == 0:
            assert fault_receipt is None
            continue
        assert fault_receipt["active_faults"] == []
        for zone_id in ("crew_cabin", "work_airlock"):
            primary = row["telemetry"][zone_id]
            secondary = row["sensor_disagreement"][zone_id]["secondary"]
            difference = row["sensor_disagreement"][zone_id][
                "primary_minus_secondary"
            ]
            truth = fault_receipt["truth_telemetry"][zone_id]
            for channel in CHANNELS:
                assert difference[channel] == pytest.approx(
                    primary[channel] - secondary[channel]
                )
                assert fault_receipt["primary_residual"][zone_id][channel] == (
                    pytest.approx(primary[channel] - truth[channel])
                )
                assert fault_receipt["secondary_residual"][zone_id][channel] == (
                    pytest.approx(secondary[channel] - truth[channel])
                )
                assert abs(primary[channel] - truth[channel]) <= CHANNELS[channel]
                assert abs(secondary[channel] - truth[channel]) <= (
                    CHANNELS[channel] * 1.5
                )


def test_fan_degradation_reduces_effective_speed_and_flow_not_actuator_position() -> None:
    healthy_scenario = Scenario.from_mapping(scenario_v4_mapping())
    fault_mapping = scenario_v4_mapping()
    fault_mapping["fault_profiles"] = [
        {
            "id": "fan-drive-loss",
            "type": "fan_speed_degradation",
            "start_step": 1,
            "end_step": 3,
            "start_multiplier": 0.5,
            "end_multiplier": 0.5,
        }
    ]
    fault_scenario = Scenario.from_mapping(fault_mapping)

    healthy = run_scenario(healthy_scenario)
    faulted = run_scenario(fault_scenario)

    healthy_row = healthy.rows[1]
    fault_row = faulted.rows[1]
    assert fault_row["actual_action"]["fan_speed_fraction"] == pytest.approx(
        healthy_row["actual_action"]["fan_speed_fraction"]
    )
    assert fault_row["air_network_receipt"]["effective_fan_speed_fraction"] == (
        pytest.approx(
            fault_row["actual_action"]["fan_speed_fraction"] * 0.5
        )
    )
    assert fault_row["air_network_receipt"]["total_flow_m3_s"] < (
        healthy_row["air_network_receipt"]["total_flow_m3_s"]
    )
    assert fault_row["fault_receipt"]["active_faults"] == [
        {
            "fault_id": "fan-drive-loss",
            "fault_type": "fan_speed_degradation",
            "target_id": "supply_fan",
            "effect_name": "fan_speed_multiplier",
            "effect_value": 0.5,
        }
    ]
    assert faulted.rows[3]["fault_receipt"]["active_faults"] == []


def test_branch_resistance_fault_reduces_target_zone_flow() -> None:
    healthy_scenario = Scenario.from_mapping(scenario_v4_mapping())
    fault_mapping = scenario_v4_mapping()
    fault_mapping["fault_profiles"] = [
        {
            "id": "cabin-duct-blockage",
            "type": "branch_resistance_increase",
            "zone_id": "crew_cabin",
            "start_step": 1,
            "end_step": 3,
            "start_multiplier": 4.0,
            "end_multiplier": 4.0,
        }
    ]
    fault_scenario = Scenario.from_mapping(fault_mapping)

    healthy_row = run_scenario(healthy_scenario).rows[1]
    fault_row = run_scenario(fault_scenario).rows[1]

    assert fault_row["commanded_action"] == healthy_row["commanded_action"]
    assert fault_row["actual_action"]["damper_position_by_id"] == (
        healthy_row["actual_action"]["damper_position_by_id"]
    )
    assert fault_row["actual_action"]["airflow_m3_s"]["crew_cabin"] < (
        healthy_row["actual_action"]["airflow_m3_s"]["crew_cabin"]
    )
    assert fault_row["fault_receipt"]["active_faults"] == [
        {
            "fault_id": "cabin-duct-blockage",
            "fault_type": "branch_resistance_increase",
            "target_id": "crew_cabin",
            "effect_name": "open_supply_resistance_multiplier",
            "effect_value": 4.0,
        }
    ]


def test_branch_resistance_fault_rejects_multiplier_below_one() -> None:
    mapping = scenario_v4_mapping()
    mapping["fault_profiles"] = [
        {
            "id": "invalid-cabin-resistance",
            "type": "branch_resistance_increase",
            "zone_id": "crew_cabin",
            "start_step": 1,
            "end_step": 3,
            "start_multiplier": 0.5,
            "end_multiplier": 1.0,
        }
    ]

    with pytest.raises(
        ScenarioValidationError,
        match=r"fault_profiles\[0\]\.start_multiplier: must be at least 1",
    ):
        Scenario.from_mapping(mapping)


def test_branch_resistance_fault_rejects_unknown_target_zone() -> None:
    mapping = scenario_v4_mapping()
    mapping["fault_profiles"] = [
        {
            "id": "unknown-duct-blockage",
            "type": "branch_resistance_increase",
            "zone_id": "missing_zone",
            "start_step": 1,
            "end_step": 3,
            "start_multiplier": 2.0,
            "end_multiplier": 2.0,
        }
    ]

    with pytest.raises(
        ScenarioValidationError,
        match=r"fault_profiles\[0\]\.zone_id: must identify a declared zone",
    ):
        Scenario.from_mapping(mapping)


def test_branch_resistance_faults_may_overlap_on_different_zones() -> None:
    mapping = scenario_v4_mapping()
    mapping["fault_profiles"] = [
        {
            "id": "cabin-duct-blockage",
            "type": "branch_resistance_increase",
            "zone_id": "crew_cabin",
            "start_step": 1,
            "end_step": 3,
            "start_multiplier": 2.0,
            "end_multiplier": 2.0,
        },
        {
            "id": "airlock-duct-blockage",
            "type": "branch_resistance_increase",
            "zone_id": "work_airlock",
            "start_step": 1,
            "end_step": 3,
            "start_multiplier": 3.0,
            "end_multiplier": 3.0,
        },
    ]

    scenario = Scenario.from_mapping(mapping)

    assert [profile["id"] for profile in scenario.data["fault_profiles"]] == [
        "cabin-duct-blockage",
        "airlock-duct-blockage",
    ]


def test_branch_resistance_faults_reject_overlap_on_same_zone() -> None:
    mapping = scenario_v4_mapping()
    mapping["fault_profiles"] = [
        {
            "id": "cabin-duct-blockage-a",
            "type": "branch_resistance_increase",
            "zone_id": "crew_cabin",
            "start_step": 1,
            "end_step": 3,
            "start_multiplier": 2.0,
            "end_multiplier": 2.0,
        },
        {
            "id": "cabin-duct-blockage-b",
            "type": "branch_resistance_increase",
            "zone_id": "crew_cabin",
            "start_step": 2,
            "end_step": 4,
            "start_multiplier": 3.0,
            "end_multiplier": 3.0,
        },
    ]

    with pytest.raises(
        ScenarioValidationError,
        match="branch resistance profiles for crew_cabin may not overlap",
    ):
        Scenario.from_mapping(mapping)
