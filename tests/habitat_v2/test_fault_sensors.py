from __future__ import annotations

from copy import deepcopy
import json
import re

import pytest

from aeolus.habitat_v2.physics import advance_one_step, initial_state
from aeolus.habitat_v2.runner import (
    AccountingInvariantError,
    run_scenario,
    validate_accounting_receipt,
)
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


def zero_sensor_noise(mapping: dict) -> None:
    for head in ("primary_noise_amplitude", "secondary_noise_amplitude"):
        mapping["sensor_model"][head] = {channel: 0.0 for channel in CHANNELS}


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


def test_scenario_v4_accounting_requires_air_network_receipt() -> None:
    scenario = Scenario.from_mapping(scenario_v4_mapping())
    pre_step_state = initial_state(scenario)
    receipt = deepcopy(advance_one_step(scenario, pre_step_state).receipt)
    receipt.pop("air_network")

    with pytest.raises(
        AccountingInvariantError,
        match="scenario-v3/v4 accounting requires an air-network receipt",
    ):
        validate_accounting_receipt(
            receipt,
            scenario=scenario,
            pre_step_state=pre_step_state,
        )


@pytest.mark.parametrize(
    ("field_path", "profile"),
    (
        (
            "sensor_model.primary_noise_amplitude.co2_ppm",
            None,
        ),
        (
            "fault_profiles[0].start_bias",
            {
                "id": "sensor-drift",
                "type": "sensor_bias_drift",
                "zone_id": "crew_cabin",
                "sensor_head": "primary",
                "channel": "co2_ppm",
                "start_step": 1,
                "end_step": 3,
                "start_bias": float("nan"),
                "end_bias": 10.0,
            },
        ),
        (
            "fault_profiles[0].start_multiplier",
            {
                "id": "branch-blockage",
                "type": "branch_resistance_increase",
                "zone_id": "crew_cabin",
                "start_step": 1,
                "end_step": 3,
                "start_multiplier": float("nan"),
                "end_multiplier": 2.0,
            },
        ),
    ),
)
def test_scenario_v4_rejects_non_finite_in_memory_numbers(
    field_path: str,
    profile: dict | None,
) -> None:
    mapping = scenario_v4_mapping()
    if profile is None:
        mapping["sensor_model"]["primary_noise_amplitude"]["co2_ppm"] = float(
            "nan"
        )
    else:
        mapping["fault_profiles"] = [profile]

    with pytest.raises(
        ScenarioValidationError,
        match=re.escape(f"$.{field_path} must be finite"),
    ):
        Scenario.from_mapping(mapping)


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


def test_damper_jam_holds_previous_position_then_resumes_slew() -> None:
    healthy_scenario = Scenario.from_mapping(scenario_v4_mapping())
    jam_mapping = scenario_v4_mapping()
    jam_mapping["fault_profiles"] = [
        {
            "id": "cabin-damper-jam",
            "type": "damper_jam",
            "damper_id": "crew_cabin_supply_damper",
            "start_step": 1,
            "end_step": 3,
        }
    ]
    jam_scenario = Scenario.from_mapping(jam_mapping)

    healthy = run_scenario(healthy_scenario)
    jammed = run_scenario(jam_scenario)

    damper_id = "crew_cabin_supply_damper"
    assert healthy.rows[1]["actual_action"]["damper_position_by_id"][
        damper_id
    ] == pytest.approx(0.8)
    assert jammed.rows[1]["actual_action"]["damper_position_by_id"][
        damper_id
    ] == pytest.approx(1.0)
    assert jammed.rows[2]["actual_action"]["damper_position_by_id"][
        damper_id
    ] == pytest.approx(1.0)
    assert jammed.rows[3]["actual_action"]["damper_position_by_id"][
        damper_id
    ] == pytest.approx(0.8)
    assert jammed.rows[1]["fault_receipt"]["active_faults"] == [
        {
            "fault_id": "cabin-damper-jam",
            "fault_type": "damper_jam",
            "target_id": damper_id,
            "effect_name": "held_damper_position",
            "effect_value": 1.0,
        }
    ]
    assert jammed.rows[3]["fault_receipt"]["active_faults"] == []


def test_damper_jam_rejects_unknown_target() -> None:
    mapping = scenario_v4_mapping()
    mapping["fault_profiles"] = [
        {
            "id": "unknown-damper-jam",
            "type": "damper_jam",
            "damper_id": "missing_damper",
            "start_step": 1,
            "end_step": 3,
        }
    ]

    with pytest.raises(
        ScenarioValidationError,
        match=r"fault_profiles\[0\]\.damper_id: must identify a declared damper",
    ):
        Scenario.from_mapping(mapping)


def test_damper_jams_may_overlap_on_different_dampers() -> None:
    mapping = scenario_v4_mapping()
    mapping["fault_profiles"] = [
        {
            "id": "cabin-damper-jam",
            "type": "damper_jam",
            "damper_id": "crew_cabin_supply_damper",
            "start_step": 1,
            "end_step": 3,
        },
        {
            "id": "airlock-damper-jam",
            "type": "damper_jam",
            "damper_id": "work_airlock_supply_damper",
            "start_step": 1,
            "end_step": 3,
        },
    ]

    scenario = Scenario.from_mapping(mapping)

    assert {profile["id"] for profile in scenario.data["fault_profiles"]} == {
        "cabin-damper-jam",
        "airlock-damper-jam",
    }


def test_damper_jams_reject_overlap_on_same_damper() -> None:
    mapping = scenario_v4_mapping()
    mapping["fault_profiles"] = [
        {
            "id": "cabin-damper-jam-a",
            "type": "damper_jam",
            "damper_id": "crew_cabin_supply_damper",
            "start_step": 1,
            "end_step": 3,
        },
        {
            "id": "cabin-damper-jam-b",
            "type": "damper_jam",
            "damper_id": "crew_cabin_supply_damper",
            "start_step": 2,
            "end_step": 4,
        },
    ]

    with pytest.raises(
        ScenarioValidationError,
        match="damper jam profiles for crew_cabin_supply_damper may not overlap",
    ):
        Scenario.from_mapping(mapping)


def test_sensor_bias_drift_targets_one_head_channel_without_changing_truth() -> None:
    healthy_mapping = scenario_v4_mapping()
    zero_sensor_noise(healthy_mapping)
    fault_mapping = deepcopy(healthy_mapping)
    fault_mapping["fault_profiles"] = [
        {
            "id": "cabin-primary-co2-drift",
            "type": "sensor_bias_drift",
            "zone_id": "crew_cabin",
            "sensor_head": "primary",
            "channel": "co2_ppm",
            "start_step": 1,
            "end_step": 3,
            "start_bias": 10.0,
            "end_bias": 20.0,
        }
    ]
    healthy = run_scenario(Scenario.from_mapping(healthy_mapping))
    faulted = run_scenario(Scenario.from_mapping(fault_mapping))

    target_id = "crew_cabin/primary/co2_ppm"
    for step, expected_bias in ((1, 10.0), (2, 20.0)):
        healthy_row = healthy.rows[step]
        fault_row = faulted.rows[step]
        healthy_truth = healthy_row["fault_receipt"]["truth_telemetry"]
        fault_truth = fault_row["fault_receipt"]["truth_telemetry"]
        assert fault_truth == healthy_truth
        assert fault_row["sensor_disagreement"]["crew_cabin"]["secondary"] == (
            healthy_row["sensor_disagreement"]["crew_cabin"]["secondary"]
        )
        assert fault_row["telemetry"]["crew_cabin"]["co2_ppm"] == pytest.approx(
            fault_truth["crew_cabin"]["co2_ppm"] + expected_bias
        )
        assert fault_row["fault_receipt"]["active_faults"] == [
            {
                "fault_id": "cabin-primary-co2-drift",
                "fault_type": "sensor_bias_drift",
                "target_id": target_id,
                "effect_name": "additive_sensor_bias",
                "effect_value": expected_bias,
            }
        ]

    assert faulted.rows[3]["telemetry"] == healthy.rows[3]["telemetry"]
    assert faulted.rows[3]["fault_receipt"]["active_faults"] == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("zone_id", "missing_zone", "must identify a declared zone"),
        ("sensor_head", "tertiary", "must be primary or secondary"),
        ("channel", "methane_ppm", "must identify a declared sensor channel"),
    ],
)
def test_sensor_bias_drift_rejects_unknown_target_parts(
    field: str, value: str, message: str
) -> None:
    mapping = scenario_v4_mapping()
    profile = {
        "id": "invalid-sensor-drift",
        "type": "sensor_bias_drift",
        "zone_id": "crew_cabin",
        "sensor_head": "primary",
        "channel": "co2_ppm",
        "start_step": 1,
        "end_step": 3,
        "start_bias": 10.0,
        "end_bias": 20.0,
    }
    profile[field] = value
    mapping["fault_profiles"] = [profile]

    with pytest.raises(ScenarioValidationError, match=message):
        Scenario.from_mapping(mapping)


def test_sensor_faults_reject_overlap_on_same_head_channel() -> None:
    mapping = scenario_v4_mapping()
    mapping["fault_profiles"] = [
        {
            "id": "sensor-drift-a",
            "type": "sensor_bias_drift",
            "zone_id": "crew_cabin",
            "sensor_head": "primary",
            "channel": "co2_ppm",
            "start_step": 1,
            "end_step": 3,
            "start_bias": 10.0,
            "end_bias": 20.0,
        },
        {
            "id": "sensor-drift-b",
            "type": "sensor_bias_drift",
            "zone_id": "crew_cabin",
            "sensor_head": "primary",
            "channel": "co2_ppm",
            "start_step": 2,
            "end_step": 4,
            "start_bias": -5.0,
            "end_bias": -10.0,
        },
    ]

    with pytest.raises(
        ScenarioValidationError,
        match="sensor fault profiles for crew_cabin/primary/co2_ppm may not overlap",
    ):
        Scenario.from_mapping(mapping)


def test_sensor_stuck_holds_previous_completed_observation_then_releases() -> None:
    healthy_mapping = scenario_v4_mapping()
    zero_sensor_noise(healthy_mapping)
    fault_mapping = deepcopy(healthy_mapping)
    fault_mapping["fault_profiles"] = [
        {
            "id": "cabin-primary-co2-stuck",
            "type": "sensor_stuck",
            "zone_id": "crew_cabin",
            "sensor_head": "primary",
            "channel": "co2_ppm",
            "start_step": 1,
            "end_step": 3,
        }
    ]
    healthy = run_scenario(Scenario.from_mapping(healthy_mapping))
    fault_scenario = Scenario.from_mapping(fault_mapping)
    faulted = run_scenario(fault_scenario)

    row_zero_value = healthy.rows[0]["telemetry"]["crew_cabin"]["co2_ppm"]
    assert healthy.rows[1]["telemetry"]["crew_cabin"]["co2_ppm"] != pytest.approx(
        row_zero_value
    )
    for step in (1, 2):
        fault_row = faulted.rows[step]
        truth_value = fault_row["fault_receipt"]["truth_telemetry"]["crew_cabin"][
            "co2_ppm"
        ]
        assert fault_row["telemetry"]["crew_cabin"]["co2_ppm"] == pytest.approx(
            row_zero_value
        )
        assert truth_value != pytest.approx(row_zero_value)
        assert fault_row["fault_receipt"]["active_faults"] == [
            {
                "fault_id": "cabin-primary-co2-stuck",
                "fault_type": "sensor_stuck",
                "target_id": "crew_cabin/primary/co2_ppm",
                "effect_name": "held_sensor_observation",
                "effect_value": row_zero_value,
            }
        ]

    assert faulted.rows[3]["telemetry"] == healthy.rows[3]["telemetry"]
    assert faulted.rows[3]["fault_receipt"]["active_faults"] == []
    assert validate_trace_bytes(faulted.trace_bytes, scenario=fault_scenario) == (
        faulted.rows
    )


@pytest.mark.parametrize(
    ("seed", "bias", "expected_co2_ppm"),
    (
        (3, 1_000.0, 0.0),
        (5, -1_000.0, 1_000_000.0),
    ),
)
def test_sensor_bias_is_applied_before_the_single_final_channel_clamp(
    seed: int,
    bias: float,
    expected_co2_ppm: float,
) -> None:
    mapping = scenario_v4_mapping()
    mapping["sensor_model"]["random_seed"] = seed
    mapping["sensor_model"]["primary_noise_amplitude"][
        "co2_ppm"
    ] = 1_000_000_000.0
    mapping["fault_profiles"] = [
        {
            "id": "cabin-primary-co2-bias",
            "type": "sensor_bias_drift",
            "zone_id": "crew_cabin",
            "sensor_head": "primary",
            "channel": "co2_ppm",
            "start_step": 1,
            "end_step": 2,
            "start_bias": bias,
            "end_bias": bias,
        }
    ]
    scenario = Scenario.from_mapping(mapping)

    run = run_scenario(scenario)

    assert run.rows[1]["telemetry"]["crew_cabin"]["co2_ppm"] == expected_co2_ppm
    assert validate_trace_bytes(run.trace_bytes, scenario=scenario) == run.rows


def test_sensor_stuck_conflicts_with_bias_on_same_head_channel() -> None:
    mapping = scenario_v4_mapping()
    mapping["fault_profiles"] = [
        {
            "id": "sensor-drift",
            "type": "sensor_bias_drift",
            "zone_id": "crew_cabin",
            "sensor_head": "primary",
            "channel": "co2_ppm",
            "start_step": 1,
            "end_step": 3,
            "start_bias": 10.0,
            "end_bias": 20.0,
        },
        {
            "id": "sensor-stuck",
            "type": "sensor_stuck",
            "zone_id": "crew_cabin",
            "sensor_head": "primary",
            "channel": "co2_ppm",
            "start_step": 2,
            "end_step": 4,
        },
    ]

    with pytest.raises(
        ScenarioValidationError,
        match="sensor fault profiles for crew_cabin/primary/co2_ppm may not overlap",
    ):
        Scenario.from_mapping(mapping)


def test_compound_physical_and_sensor_faults_replay_byte_identically() -> None:
    mapping = scenario_v4_mapping()
    zero_sensor_noise(mapping)
    mapping["fault_profiles"] = [
        {
            "id": "a-fan-loss",
            "type": "fan_speed_degradation",
            "start_step": 1,
            "end_step": 4,
            "start_multiplier": 0.7,
            "end_multiplier": 0.5,
        },
        {
            "id": "b-cabin-blockage",
            "type": "branch_resistance_increase",
            "zone_id": "crew_cabin",
            "start_step": 1,
            "end_step": 4,
            "start_multiplier": 2.0,
            "end_multiplier": 4.0,
        },
        {
            "id": "c-airlock-damper-jam",
            "type": "damper_jam",
            "damper_id": "work_airlock_supply_damper",
            "start_step": 1,
            "end_step": 3,
        },
        {
            "id": "d-cabin-primary-co2-drift",
            "type": "sensor_bias_drift",
            "zone_id": "crew_cabin",
            "sensor_head": "primary",
            "channel": "co2_ppm",
            "start_step": 1,
            "end_step": 4,
            "start_bias": 5.0,
            "end_bias": 15.0,
        },
        {
            "id": "e-airlock-secondary-pressure-stuck",
            "type": "sensor_stuck",
            "zone_id": "work_airlock",
            "sensor_head": "secondary",
            "channel": "pressure_pa",
            "start_step": 1,
            "end_step": 4,
        },
    ]
    scenario = Scenario.from_mapping(mapping)

    first = run_scenario(scenario)
    second = run_scenario(scenario)

    assert first.trace_bytes == second.trace_bytes
    assert validate_trace_bytes(first.trace_bytes, scenario=scenario) == first.rows
    assert [
        fault["fault_id"] for fault in first.rows[1]["fault_receipt"]["active_faults"]
    ] == [
        "a-fan-loss",
        "b-cabin-blockage",
        "c-airlock-damper-jam",
        "d-cabin-primary-co2-drift",
        "e-airlock-secondary-pressure-stuck",
    ]
    assert first.rows[1]["air_network_receipt"][
        "effective_fan_speed_fraction"
    ] < first.rows[1]["actual_action"]["fan_speed_fraction"]
    assert first.rows[1]["actual_action"]["damper_position_by_id"][
        "work_airlock_supply_damper"
    ] == pytest.approx(1.0)
    assert first.rows[1]["sensor_disagreement"]["work_airlock"]["secondary"][
        "pressure_pa"
    ] == pytest.approx(
        first.rows[0]["sensor_disagreement"]["work_airlock"]["secondary"][
            "pressure_pa"
        ]
    )


@pytest.mark.parametrize("mutation", ("telemetry", "fault_effect"))
def test_v4_trace_rejects_finite_observation_and_fault_receipt_mutations(
    mutation: str,
) -> None:
    scenario = Scenario.from_mapping(scenario_v4_mapping())
    run = run_scenario(scenario)
    rows = [json.loads(line) for line in run.trace_bytes.splitlines()]
    if mutation == "telemetry":
        rows[1]["telemetry"]["crew_cabin"]["co2_ppm"] += 1.0
    else:
        fault_mapping = scenario_v4_mapping()
        fault_mapping["fault_profiles"] = [
            {
                "id": "fan-drive-loss",
                "type": "fan_speed_degradation",
                "start_step": 1,
                "end_step": 3,
                "start_multiplier": 0.8,
                "end_multiplier": 0.6,
            }
        ]
        scenario = Scenario.from_mapping(fault_mapping)
        rows = [
            json.loads(line) for line in run_scenario(scenario).trace_bytes.splitlines()
        ]
        rows[1]["fault_receipt"]["active_faults"][0]["effect_value"] += 0.01
    mutated = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )

    with pytest.raises(ValueError):
        validate_trace_bytes(mutated, scenario=scenario)
