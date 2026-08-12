from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from aeolus.habitat_v2.physics import advance_one_step, initial_state
from aeolus.habitat_v2.runner import (
    AccountingInvariantError,
    run_scenario,
    validate_accounting_receipt,
)
from aeolus.habitat_v2.scenario import Scenario, ScenarioValidationError
from aeolus.habitat_v2.trace import validate_trace_bytes


def test_v5_rejects_overlapping_effectiveness_faults() -> None:
    mapping = v5_mapping()
    mapping["fault_profiles"].extend(
        [
            {
                "id": "first-cooling-loss",
                "type": "cooling_delivery_degradation",
                "zone_id": "air_processing_bay",
                "start_step": 1,
                "end_step": 3,
                "start_multiplier": 0.8,
                "end_multiplier": 0.8,
            },
            {
                "id": "overlapping-cooling-loss",
                "type": "cooling_delivery_degradation",
                "zone_id": "air_processing_bay",
                "start_step": 2,
                "end_step": 3,
                "start_multiplier": 0.7,
                "end_multiplier": 0.7,
            },
        ]
    )

    with pytest.raises(ValueError, match="effectiveness profiles"):
        Scenario.from_mapping(mapping)


def v5_mapping() -> dict[str, object]:
    path = Path(__file__).parents[2] / "scenarios" / "habitat_v2_compound_faults.json"
    mapping = json.loads(path.read_text(encoding="utf-8"))
    mapping["schema_version"] = "aeolus_habitat_v2_scenario_v5"
    mapping["actuator_feedback"] = {
        "dc_bus_voltage_v": 120.0,
        "cooling_slew_w_per_s": 5.0,
        "oxygen_slew_mol_s2": 0.00001,
        "feedback_sensor_noise_amplitude": 0.0,
    }
    mapping["initial_utility"]["actual_cooling_removed_w"] = {
        zone["id"]: 0.0 for zone in mapping["zones"]
    }
    mapping["initial_utility"]["actual_oxygen_injection_mol_s"] = {
        zone["id"]: 0.0 for zone in mapping["zones"]
    }
    return mapping


def test_v5_is_closed_lineage_with_feedback_contract_identity() -> None:
    scenario = Scenario.from_mapping(v5_mapping())

    assert scenario.scenario_schema_version == "aeolus_habitat_v2_scenario_v5"
    assert scenario.trace_schema_version == "aeolus_habitat_v2_trace_v5"
    assert scenario.equation_contract_revision == "aeolus_habitat_v2_equations_v4"
    assert scenario.actuator_feedback_contract_revision == "aeolus_habitat_v2_actuator_feedback_v1"


def test_v5_trace_lineage_carries_feedback_contract_identity() -> None:
    scenario = Scenario.from_mapping(v5_mapping())
    run = run_scenario(scenario)

    assert (
        run.rows[0]["lineage"]["actuator_feedback_contract_revision"]
        == "aeolus_habitat_v2_actuator_feedback_v1"
    )

    tampered = [json.loads(line) for line in run.trace_bytes.splitlines()]
    tampered[1]["lineage"]["actuator_feedback_contract_revision"] = "substituted"
    tampered_bytes = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in tampered
    )
    with pytest.raises(ValueError, match="actuator feedback contract"):
        validate_trace_bytes(tampered_bytes, scenario=scenario)


def test_v4_rejects_v5_only_fields() -> None:
    mapping = v5_mapping()
    mapping["schema_version"] = "aeolus_habitat_v2_scenario_v4"

    with pytest.raises(ValueError, match="unknown top-level fields"):
        Scenario.from_mapping(mapping)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda mapping: mapping["actuator_feedback"].pop("dc_bus_voltage_v"),
        lambda mapping: mapping["actuator_feedback"].update({"unreviewed": 1.0}),
        lambda mapping: mapping["actuator_feedback"].update(
            {"cooling_slew_w_per_s": float("nan")}
        ),
    ],
)
def test_v5_feedback_configuration_is_strict(mutation) -> None:
    mapping = v5_mapping()
    mutation(mapping)

    with pytest.raises(ValueError):
        Scenario.from_mapping(mapping)


def test_v5_achieved_cooling_and_oxygen_are_rate_limited() -> None:
    scenario = Scenario.from_mapping(v5_mapping())
    result = advance_one_step(scenario, initial_state(scenario))

    assert result.state.utility.actual_cooling_removed_w["air_processing_bay"] == 300.0
    assert result.state.utility.actual_oxygen_injection_mol_s["laboratory"] == 0.0006
    assert result.receipt["actuators"]["cooling"]["achieved_w"]["air_processing_bay"] == 300.0
    assert result.receipt["actuators"]["oxygen"]["achieved_mol_s"]["laboratory"] == 0.0006


def test_v5_effectiveness_fault_changes_delivery_not_achieved_state() -> None:
    healthy_mapping = v5_mapping()
    faulted_mapping = deepcopy(healthy_mapping)
    faulted_mapping["fault_profiles"].append(
        {
            "id": "cooling-loss",
            "type": "cooling_delivery_degradation",
            "zone_id": "air_processing_bay",
            "start_step": 1,
            "end_step": 2,
            "start_multiplier": 0.25,
            "end_multiplier": 0.25,
        }
    )
    healthy = advance_one_step(
        Scenario.from_mapping(healthy_mapping), initial_state(Scenario.from_mapping(healthy_mapping))
    )
    faulted_scenario = Scenario.from_mapping(faulted_mapping)
    faulted = advance_one_step(faulted_scenario, initial_state(faulted_scenario))

    assert faulted.state.utility.actual_cooling_removed_w == healthy.state.utility.actual_cooling_removed_w
    assert faulted.state.utility.effective_cooling_delivery_by_zone["air_processing_bay"] < healthy.state.utility.effective_cooling_delivery_by_zone["air_processing_bay"]


def test_v5_effectiveness_fault_does_not_contaminate_fan_multiplier() -> None:
    mapping = v5_mapping()
    mapping["fault_profiles"].append(
        {
            "id": "cooling-only-loss",
            "type": "cooling_delivery_degradation",
            "zone_id": "air_processing_bay",
            "start_step": 1,
            "end_step": 2,
            "start_multiplier": 0.25,
            "end_multiplier": 0.25,
        }
    )
    scenario = Scenario.from_mapping(mapping)
    result = advance_one_step(scenario, initial_state(scenario))
    healthy_scenario = Scenario.from_mapping(v5_mapping())
    healthy = advance_one_step(healthy_scenario, initial_state(healthy_scenario))

    assert result.receipt["air_network"]["effective_fan_speed_fraction"] == healthy.receipt[
        "air_network"
    ]["effective_fan_speed_fraction"]


@pytest.mark.parametrize(
    ("fault_type", "target", "state_field"),
    [
        ("scrubber_capture_degradation", None, "effective_scrubber_capture_ability"),
        ("condenser_removal_degradation", None, "effective_condenser_removal_ability"),
        ("oxygen_delivery_degradation", "laboratory", "effective_oxygen_delivery_by_zone"),
    ],
)
def test_v5_all_delivery_effectiveness_faults_are_physical(
    fault_type: str, target: str | None, state_field: str
) -> None:
    healthy_mapping = v5_mapping()
    fault = {
        "id": f"{fault_type}-loss",
        "type": fault_type,
        "start_step": 1,
        "end_step": 2,
        "start_multiplier": 0.25,
        "end_multiplier": 0.25,
    }
    if target is not None:
        fault["zone_id"] = target
    faulted_mapping = deepcopy(healthy_mapping)
    faulted_mapping["fault_profiles"].append(fault)
    healthy_scenario = Scenario.from_mapping(healthy_mapping)
    faulted_scenario = Scenario.from_mapping(faulted_mapping)
    healthy = advance_one_step(healthy_scenario, initial_state(healthy_scenario))
    faulted = advance_one_step(faulted_scenario, initial_state(faulted_scenario))

    if target is None:
        assert getattr(faulted.state.utility, state_field) < getattr(
            healthy.state.utility, state_field
        )
    else:
        assert faulted.state.utility.actual_oxygen_injection_mol_s[target] == healthy.state.utility.actual_oxygen_injection_mol_s[target]
        assert faulted.state.utility.effective_oxygen_delivery_by_zone[target] < healthy.state.utility.effective_oxygen_delivery_by_zone[target]


def test_v5_operational_feedback_is_deterministic_and_complete() -> None:
    scenario = Scenario.from_mapping(v5_mapping())
    first = advance_one_step(scenario, initial_state(scenario))
    second = advance_one_step(scenario, initial_state(scenario))
    feedback = first.receipt["operational_feedback"]

    assert feedback == second.receipt["operational_feedback"]
    assert set(feedback) == {
        "fan_speed_fraction",
        "fan_dc_bus_current_a",
        "damper_position_by_id",
        "branch_airflow_m3_s",
        "branch_differential_pressure_pa",
        "scrubber_capture_rate_mol_s",
        "condenser_removal_rate_mol_s",
        "cooling_delivery_w",
        "oxygen_delivery_mol_s",
        "battery_state_of_charge",
        "oxygen_store_fraction",
        "sorbent_remaining_fraction",
    }


def test_v5_fan_feedback_measures_fault_effective_response() -> None:
    healthy_mapping = v5_mapping()
    faulted_mapping = deepcopy(healthy_mapping)
    faulted_mapping["fault_profiles"] = [
        profile
        for profile in faulted_mapping["fault_profiles"]
        if profile["type"] != "fan_speed_degradation"
    ]
    faulted_mapping["fault_profiles"].append(
        {
            "id": "fan-feedback-loss",
            "type": "fan_speed_degradation",
            "start_step": 1,
            "end_step": 2,
            "start_multiplier": 0.25,
            "end_multiplier": 0.25,
        }
    )
    healthy_mapping["fault_profiles"] = [
        profile
        for profile in healthy_mapping["fault_profiles"]
        if profile["type"] != "fan_speed_degradation"
    ]
    healthy_scenario = Scenario.from_mapping(healthy_mapping)
    faulted_scenario = Scenario.from_mapping(faulted_mapping)

    healthy = advance_one_step(healthy_scenario, initial_state(healthy_scenario))
    faulted = advance_one_step(faulted_scenario, initial_state(faulted_scenario))

    healthy_air = healthy.receipt["air_network"]
    faulted_air = faulted.receipt["air_network"]
    assert faulted_air["actual_fan_speed_fraction"] == healthy_air[
        "actual_fan_speed_fraction"
    ]
    assert faulted_air["effective_fan_speed_fraction"] < faulted_air[
        "actual_fan_speed_fraction"
    ]
    assert faulted.receipt["operational_feedback"]["fan_speed_fraction"] == (
        faulted_air["effective_fan_speed_fraction"]
    )
    assert faulted.receipt["operational_feedback"]["fan_speed_fraction"] < (
        healthy.receipt["operational_feedback"]["fan_speed_fraction"]
    )
    assert faulted_air["total_flow_m3_s"] < healthy_air["total_flow_m3_s"]


def test_v5_trace_binds_requested_achieved_effective_and_measured_layers() -> None:
    scenario = Scenario.from_mapping(v5_mapping())
    run = run_scenario(scenario)
    row = run.rows[1]
    receipt = row["actuator_receipt"]

    assert row["commanded_action"]["fan_speed_fraction"] == receipt["fan"][
        "requested_fraction"
    ]
    assert row["actual_action"]["fan_speed_fraction"] == receipt["fan"][
        "achieved_fraction"
    ]
    assert row["air_network_receipt"]["effective_fan_speed_fraction"] == receipt[
        "fan"
    ]["effective_fraction"]
    assert row["operational_feedback"]["fan_speed_fraction"] == receipt["fan"][
        "effective_fraction"
    ]

    assert set(receipt) == {
        "fan",
        "dampers",
        "scrubber",
        "condenser",
        "cooling",
        "oxygen",
    }
    assert receipt["cooling"]["requested_w"] == row["commanded_action"][
        "cooling_removed_w"
    ]
    assert receipt["cooling"]["achieved_w"] == row["actual_action"][
        "cooling_removed_w"
    ]
    assert receipt["cooling"]["effective_w"] == row["operational_feedback"][
        "cooling_delivery_w"
    ]
    assert receipt["oxygen"]["requested_mol_s"] == row["commanded_action"][
        "oxygen_injection_mol_s"
    ]
    assert receipt["oxygen"]["achieved_mol_s"] == row["actual_action"][
        "oxygen_injection_mol_s"
    ]
    assert receipt["oxygen"]["effective_mol_s"] == row["operational_feedback"][
        "oxygen_delivery_mol_s"
    ]

    tampered = [json.loads(line) for line in run.trace_bytes.splitlines()]
    tampered[1]["actuator_receipt"]["cooling"]["effective_w"][
        "air_processing_bay"
    ] += 1.0
    tampered_bytes = b"".join(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for value in tampered
    )
    with pytest.raises(ValueError):
        validate_trace_bytes(tampered_bytes, scenario=scenario)


def test_v5_external_command_is_validated_before_mutation_and_bound_to_receipt() -> None:
    from aeolus.habitat_v2.physics import advance_one_step_with_command

    scenario = Scenario.from_mapping(v5_mapping())
    state = initial_state(scenario)
    command = deepcopy(scenario.data["timeline"][0]["command"])
    command["fan_speed_fraction"] = 0.4
    result = advance_one_step_with_command(scenario, state, command)

    assert result.receipt["external_command_digest"]
    assert result.receipt["realised_loads"] == scenario.data["timeline"][0]["loads"]
    validate_accounting_receipt(
        result.receipt,
        scenario=scenario,
        pre_step_state=state,
        command=command,
    )

    bad_command = deepcopy(command)
    bad_command["damper_position_by_id"].popitem()
    with pytest.raises(ValueError):
        advance_one_step_with_command(scenario, state, bad_command)
    assert state == initial_state(scenario)


@pytest.mark.parametrize(
    "scenario_filename",
    [
        "habitat_v2_reference.json",
        "habitat_v2_operating_modes.json",
        "habitat_v2_air_network.json",
        "habitat_v2_compound_faults.json",
        "habitat_v2_actuator_feedback.json",
    ],
)
def test_external_receipt_binds_command_for_every_public_schema(
    scenario_filename: str,
) -> None:
    from aeolus.habitat_v2.physics import advance_one_step_with_command

    scenario_path = Path(__file__).parents[2] / "scenarios" / scenario_filename
    scenario = Scenario.from_mapping(
        json.loads(scenario_path.read_text(encoding="utf-8"))
    )
    state = initial_state(scenario)
    command = deepcopy(scenario.data["timeline"][0]["command"])
    command["scrubber_duty"] = 0.4
    result = advance_one_step_with_command(scenario, state, command)
    assert result.receipt["external_command_digest"]
    validate_accounting_receipt(
        result.receipt,
        scenario=scenario,
        pre_step_state=state,
        command=command,
    )

    wrong_command = deepcopy(command)
    wrong_command["scrubber_duty"] = 0.5

    with pytest.raises(AccountingInvariantError, match="causal recomputation"):
        validate_accounting_receipt(
            result.receipt,
            scenario=scenario,
            pre_step_state=state,
            command=wrong_command,
        )


@pytest.mark.parametrize(
    "scenario_filename",
    [
        "habitat_v2_reference.json",
        "habitat_v2_operating_modes.json",
        "habitat_v2_air_network.json",
        "habitat_v2_compound_faults.json",
        "habitat_v2_actuator_feedback.json",
    ],
)
def test_external_receipt_rejects_null_digest_without_command_context(
    scenario_filename: str,
) -> None:
    from aeolus.habitat_v2.physics import advance_one_step_with_command

    scenario_path = Path(__file__).parents[2] / "scenarios" / scenario_filename
    scenario = Scenario.from_mapping(
        json.loads(scenario_path.read_text(encoding="utf-8"))
    )
    state = initial_state(scenario)
    command = deepcopy(scenario.data["timeline"][0]["command"])
    receipt = deepcopy(advance_one_step_with_command(scenario, state, command).receipt)
    receipt["external_command_digest"] = None

    with pytest.raises(
        AccountingInvariantError, match="requires the supplied command"
    ):
        validate_accounting_receipt(
            receipt,
            scenario=scenario,
            pre_step_state=state,
        )


@pytest.mark.parametrize(
    "scenario_filename",
    [
        "habitat_v2_reference.json",
        "habitat_v2_operating_modes.json",
        "habitat_v2_air_network.json",
        "habitat_v2_compound_faults.json",
        "habitat_v2_actuator_feedback.json",
    ],
)
def test_external_receipt_cannot_hide_changed_command_by_deleting_digest(
    scenario_filename: str,
) -> None:
    from aeolus.habitat_v2.physics import advance_one_step_with_command

    scenario_path = Path(__file__).parents[2] / "scenarios" / scenario_filename
    scenario = Scenario.from_mapping(
        json.loads(scenario_path.read_text(encoding="utf-8"))
    )
    state = initial_state(scenario)
    command = deepcopy(scenario.data["timeline"][0]["command"])
    command["scrubber_duty"] = 0.4
    receipt = deepcopy(advance_one_step_with_command(scenario, state, command).receipt)
    del receipt["external_command_digest"]

    with pytest.raises(AccountingInvariantError, match="causal recomputation"):
        validate_accounting_receipt(
            receipt,
            scenario=scenario,
            pre_step_state=state,
        )


def test_v5_external_receipt_accepts_timeline_command_with_explicit_context() -> None:
    from aeolus.habitat_v2.physics import advance_one_step_with_command

    scenario = Scenario.from_mapping(v5_mapping())
    state = initial_state(scenario)
    command = deepcopy(scenario.data["timeline"][0]["command"])
    result = advance_one_step_with_command(scenario, state, command)

    validate_accounting_receipt(
        result.receipt,
        scenario=scenario,
        pre_step_state=state,
        command=command,
    )


def test_v5_external_receipt_rejects_wrong_validation_command() -> None:
    from aeolus.habitat_v2.physics import advance_one_step_with_command

    scenario = Scenario.from_mapping(v5_mapping())
    state = initial_state(scenario)
    command = deepcopy(scenario.data["timeline"][0]["command"])
    command["fan_speed_fraction"] = 0.4
    result = advance_one_step_with_command(scenario, state, command)
    wrong_command = deepcopy(command)
    wrong_command["fan_speed_fraction"] = 0.5

    with pytest.raises(AccountingInvariantError, match="causal recomputation"):
        validate_accounting_receipt(
            result.receipt,
            scenario=scenario,
            pre_step_state=state,
            command=wrong_command,
        )


def test_v5_external_receipt_requires_command_validation_context() -> None:
    from aeolus.habitat_v2.physics import advance_one_step_with_command

    scenario = Scenario.from_mapping(v5_mapping())
    state = initial_state(scenario)
    command = deepcopy(scenario.data["timeline"][0]["command"])
    result = advance_one_step_with_command(scenario, state, command)

    with pytest.raises(
        AccountingInvariantError, match="requires the supplied command"
    ):
        validate_accounting_receipt(
            result.receipt,
            scenario=scenario,
            pre_step_state=state,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda command: command.update({"fan_speed_fraction": True}),
        lambda command: command.update({"fan_speed_fraction": "0.4"}),
        lambda command: command["damper_position_by_id"].update(
            {next(iter(command["damper_position_by_id"])): False}
        ),
        lambda command: command.update({"scrubber_duty": "0.5"}),
        lambda command: command.update({"condenser_duty": True}),
        lambda command: command["cooling_removed_w"].update(
            {next(iter(command["cooling_removed_w"])): False}
        ),
        lambda command: command["oxygen_injection_mol_s"].update(
            {next(iter(command["oxygen_injection_mol_s"])): "0.0001"}
        ),
    ],
)
def test_v5_external_command_rejects_non_numeric_json_types(mutation) -> None:
    from aeolus.habitat_v2.physics import advance_one_step_with_command

    scenario = Scenario.from_mapping(v5_mapping())
    state = initial_state(scenario)
    command = deepcopy(scenario.data["timeline"][0]["command"])
    mutation(command)

    with pytest.raises(ValueError, match="finite numeric data"):
        advance_one_step_with_command(scenario, state, command)
    assert state == initial_state(scenario)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    ("field", "nested"),
    [
        ("fan_speed_fraction", False),
        ("damper_position_by_id", True),
        ("scrubber_duty", False),
        ("condenser_duty", False),
        ("cooling_removed_w", True),
        ("oxygen_injection_mol_s", True),
    ],
)
def test_v5_external_command_rejects_non_finite_values_with_field_path(
    value: float, field: str, nested: bool
) -> None:
    from aeolus.habitat_v2.physics import advance_one_step_with_command

    scenario = Scenario.from_mapping(v5_mapping())
    state = initial_state(scenario)
    command = deepcopy(scenario.data["timeline"][0]["command"])
    expected_path = field
    if nested:
        child = next(iter(command[field]))
        command[field][child] = value
        expected_path = f"{field}.{child}"
    else:
        command[field] = value

    with pytest.raises(
        ScenarioValidationError,
        match=rf"external command {expected_path} must be finite numeric data",
    ):
        advance_one_step_with_command(scenario, state, command)
    assert state == initial_state(scenario)


def test_v5_run_and_trace_validation_bind_feedback_rows() -> None:
    scenario = Scenario.from_mapping(v5_mapping())
    run = run_scenario(scenario)

    rows = validate_trace_bytes(run.trace_bytes, scenario=scenario)
    assert len(rows) == scenario.data["steps"] + 1
    assert rows[0]["operational_feedback"] == run.rows[0]["operational_feedback"]
    assert rows[1]["actual_action"]["cooling_removed_w"]["air_processing_bay"] == 300.0


def test_checked_in_v5_eight_zone_scenario_replays_byte_identically() -> None:
    path = Path(__file__).parents[2] / "scenarios" / "habitat_v2_actuator_feedback.json"
    scenario = Scenario.from_mapping(json.loads(path.read_text(encoding="utf-8")))
    first = run_scenario(scenario)
    second = run_scenario(scenario)

    assert len(scenario.data["zones"]) == 8
    assert first.trace_bytes == second.trace_bytes
    assert validate_trace_bytes(first.trace_bytes, scenario=scenario) == first.rows


def test_v5_feedback_bias_and_stuck_faults_target_operational_channels() -> None:
    mapping = v5_mapping()
    mapping["fault_profiles"].extend(
        [
            {
                "id": "battery-feedback-bias",
                "type": "feedback_sensor_bias_drift",
                "resource_id": "battery",
                "channel": "battery_state_of_charge",
                "start_step": 1,
                "end_step": 2,
                "start_bias": 0.1,
                "end_bias": 0.1,
            },
            {
                "id": "oxygen-store-feedback-stuck",
                "type": "feedback_sensor_stuck",
                "resource_id": "oxygen_store",
                "channel": "oxygen_store_fraction",
                "start_step": 1,
                "end_step": 3,
            },
        ]
    )
    scenario = Scenario.from_mapping(mapping)
    run = run_scenario(scenario)
    healthy = run_scenario(Scenario.from_mapping(v5_mapping()))

    assert run.rows[1]["operational_feedback"]["battery_state_of_charge"] == pytest.approx(
        healthy.rows[1]["operational_feedback"]["battery_state_of_charge"] + 0.1
    )
    assert (
        run.rows[2]["operational_feedback"]["oxygen_store_fraction"]
        == run.rows[1]["operational_feedback"]["oxygen_store_fraction"]
    )
