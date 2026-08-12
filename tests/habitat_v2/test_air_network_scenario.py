from __future__ import annotations

from copy import deepcopy

import pytest

import aeolus.habitat_v2.runner as runner_module
from aeolus.habitat_v2.physics import StepResult, advance_one_step, initial_state
from aeolus.habitat_v2.runner import (
    AccountingInvariantError,
    run_scenario,
    validate_accounting_receipt,
)
from aeolus.habitat_v2.scenario import (
    EQUATION_CONTRACT_REVISION_V2,
    SCENARIO_SCHEMA_VERSION_V3,
    TRACE_SCHEMA_VERSION_V3,
    Scenario,
)
from aeolus.habitat_v2.trace import validate_trace_bytes

from ._helpers import reference_scenario_mapping


def scenario_v3_mapping() -> dict:
    mapping = deepcopy(reference_scenario_mapping())
    mapping["schema_version"] = "aeolus_habitat_v2_scenario_v3"

    geometry_by_zone = {
        "crew_cabin": {
            "center_m": [-2.0, 0.0, 1.5],
            "size_m": [4.0, 5.0, 3.0],
        },
        "work_airlock": {
            "center_m": [2.0, 0.0, 1.5],
            "size_m": [4.0, 5.0, 3.0],
        },
    }
    for zone in mapping["zones"]:
        zone["geometry"] = geometry_by_zone[zone["id"]]

    mapping["air_network"] = {
        "supply_plenum_position_m": [0.0, -2.0, 2.7],
        "return_plenum_position_m": [0.0, 2.0, 2.7],
        "fan": {
            "id": "supply_fan",
            "rated_free_delivery_m3_s": 0.40,
            "rated_shutoff_pressure_pa": 500.0,
            "total_efficiency": 0.70,
            "speed_slew_fraction_per_s": 0.01,
            "position_m": [0.0, -2.0, 1.0],
        },
        "shared_resistance": {
            "supply_trunk_pa_s2_m6": 400.0,
            "return_trunk_pa_s2_m6": 300.0,
            "filter_pa_s2_m6": 300.0,
        },
        "branches": [
            {
                "zone_id": zone_id,
                "damper_id": f"{zone_id}_supply_damper",
                "open_supply_resistance_pa_s2_m6": 2_000.0,
                "return_resistance_pa_s2_m6": 1_000.0,
                "damper_leak_fraction": 0.05,
                "damper_slew_fraction_per_s": 0.02,
                "supply_diffuser_position_m": [center_x, -1.5, 2.6],
                "return_grille_position_m": [center_x, 1.5, 2.4],
                "damper_position_m": [center_x, -2.0, 2.4],
                "duct_polyline_m": [
                    [0.0, -2.0, 2.7],
                    [center_x, -2.0, 2.7],
                    [center_x, -1.5, 2.6],
                ],
            }
            for zone_id, center_x in (("crew_cabin", -2.0), ("work_airlock", 2.0))
        ],
    }

    for field in (
        "max_total_airflow_m3_s",
        "max_zone_airflow_m3_s",
        "airflow_slew_m3_s2",
        "fan_power_w_per_m3_s",
    ):
        del mapping["equipment"][field]

    del mapping["initial_utility"]["actual_airflow_m3_s"]
    mapping["initial_utility"]["actual_fan_speed_fraction"] = 0.0
    mapping["initial_utility"]["actual_damper_position_by_id"] = {
        "crew_cabin_supply_damper": 1.0,
        "work_airlock_supply_damper": 1.0,
    }

    for segment in mapping["timeline"]:
        segment["operating_mode"] = "occupied"
        del segment["command"]["airflow_m3_s"]
        segment["command"]["fan_speed_fraction"] = 0.75
        segment["command"]["damper_position_by_id"] = {
            "crew_cabin_supply_damper": 0.80,
            "work_airlock_supply_damper": 0.60,
        }
    return mapping


def _run_with_first_receipt_mutation(
    monkeypatch: pytest.MonkeyPatch,
    scenario: Scenario,
    mutate_receipt,
) -> None:
    canonical_advance = advance_one_step

    def advance_with_forged_receipt(
        active_scenario: Scenario, state
    ) -> StepResult:
        result = canonical_advance(active_scenario, state)
        if state.step != 0:
            return result
        forged = deepcopy(result.receipt)
        mutate_receipt(forged)
        return StepResult(state=result.state, receipt=forged)

    monkeypatch.setattr(
        runner_module, "advance_one_step", advance_with_forged_receipt
    )
    runner_module.run_scenario(scenario)


def _replace_with_coherent_zero_flow_receipt(
    receipt: dict, scenario: Scenario
) -> None:
    network = receipt["air_network"]
    zone_ids = sorted(network["zone_flow_m3_s"])
    damper_ids = sorted(network["actual_damper_position_by_id"])
    network.update(
        {
            "requested_fan_speed_fraction": 0.0,
            "actual_fan_speed_fraction": 0.0,
            "requested_damper_position_by_id": {
                damper_id: 0.0 for damper_id in damper_ids
            },
            "actual_damper_position_by_id": {
                damper_id: 0.0 for damper_id in damper_ids
            },
            "fan_pressure_rise_pa": 0.0,
            "shared_pressure_loss_pa": 0.0,
            "branch_pressure_loss_pa": {zone_id: 0.0 for zone_id in zone_ids},
            "total_flow_m3_s": 0.0,
            "zone_flow_m3_s": {zone_id: 0.0 for zone_id in zone_ids},
            "zone_mass_flow_kg_s": {zone_id: 0.0 for zone_id in zone_ids},
            "fan_air_power_w": 0.0,
            "fan_electrical_power_w": 0.0,
            "operating_point_residual_pa": 0.0,
            "mass_balance_residual_kg_s": {
                zone_id: 0.0 for zone_id in zone_ids
            },
        }
    )

    electrical = receipt["electrical"]
    original_fan_load_wh = float(electrical["fan_load_wh"])
    electrical["fan_load_wh"] = 0.0
    electrical["served_load_wh"] -= original_fan_load_wh
    electrical["battery_charge_input_wh"] = (
        electrical["generation_wh"] - electrical["served_load_wh"]
    )
    charge_efficiency = float(
        scenario.data["equipment"]["battery_charge_efficiency"]
    )
    electrical["battery_charge_stored_wh"] = (
        electrical["battery_charge_input_wh"] * charge_efficiency
    )
    electrical["charge_conversion_loss_wh"] = (
        electrical["battery_charge_input_wh"]
        - electrical["battery_charge_stored_wh"]
    )
    electrical["battery_energy_delta_wh"] = electrical[
        "battery_charge_stored_wh"
    ]
    electrical["residual_wh"] = 0.0

    receipt["thermal"]["external_heat_rejected_j"] -= (
        original_fan_load_wh * 3600.0
    )


def test_scenario_v3_has_distinct_schema_trace_and_equation_identity() -> None:
    scenario = Scenario.from_mapping(scenario_v3_mapping())

    assert scenario.scenario_schema_version == SCENARIO_SCHEMA_VERSION_V3
    assert scenario.trace_schema_version == TRACE_SCHEMA_VERSION_V3
    assert scenario.equation_contract_revision == EQUATION_CONTRACT_REVISION_V2
    assert {zone["id"] for zone in scenario.data["zones"]} == {
        "crew_cabin",
        "work_airlock",
    }


def test_scenario_v3_slews_actuators_and_derives_delivered_airflow() -> None:
    scenario = Scenario.from_mapping(scenario_v3_mapping())
    state = initial_state(scenario)

    assert state.utility.actual_fan_speed_fraction == 0.0
    assert state.utility.actual_airflow_m3_s == {
        "crew_cabin": 0.0,
        "work_airlock": 0.0,
    }

    result = advance_one_step(scenario, state)
    network = result.receipt["air_network"]

    assert result.state.utility.actual_fan_speed_fraction == pytest.approx(0.60)
    assert result.state.utility.actual_damper_position_by_id == {
        "crew_cabin_supply_damper": pytest.approx(0.80),
        "work_airlock_supply_damper": pytest.approx(0.60),
    }
    assert result.state.utility.actual_airflow_m3_s == network["zone_flow_m3_s"]
    assert network["total_flow_m3_s"] == pytest.approx(
        sum(network["zone_flow_m3_s"].values())
    )
    assert network["fan_electrical_power_w"] > 0.0
    assert abs(network["operating_point_residual_pa"]) <= 1e-9
    assert result.receipt["electrical"]["fan_load_wh"] == pytest.approx(
        network["fan_electrical_power_w"] * 60.0 / 3600.0
    )


def test_scenario_v3_trace_replays_with_network_command_and_receipts() -> None:
    scenario = Scenario.from_mapping(scenario_v3_mapping())
    first = run_scenario(scenario)
    second = run_scenario(scenario)

    assert first.trace_bytes == second.trace_bytes
    assert validate_trace_bytes(first.trace_bytes, scenario=scenario) == first.rows
    assert first.rows[0]["air_network_receipt"] is None
    assert first.rows[1]["commanded_action"] == {
        "fan_speed_fraction": 0.75,
        "damper_position_by_id": {
            "crew_cabin_supply_damper": 0.80,
            "work_airlock_supply_damper": 0.60,
        },
        "scrubber_duty": 0.60,
        "condenser_duty": 0.50,
        "cooling_removed_w": {
            "crew_cabin": 120.0,
            "work_airlock": 120.0,
        },
        "oxygen_injection_mol_s": {
            "crew_cabin": 0.00030,
            "work_airlock": 0.00030,
        },
    }
    assert first.rows[1]["actual_action"]["fan_speed_fraction"] == pytest.approx(
        0.60
    )
    assert first.rows[1]["air_network_receipt"]["total_flow_m3_s"] == pytest.approx(
        sum(first.rows[1]["actual_action"]["airflow_m3_s"].values())
    )


def test_network_accounting_rejects_corrupted_fan_power() -> None:
    scenario = Scenario.from_mapping(scenario_v3_mapping())
    pre_step_state = initial_state(scenario)
    result = advance_one_step(scenario, pre_step_state)
    corrupted = deepcopy(result.receipt)
    corrupted["air_network"]["fan_electrical_power_w"] += 1.0

    with pytest.raises(AccountingInvariantError, match="fan electrical power"):
        validate_accounting_receipt(
            corrupted,
            scenario=scenario,
            pre_step_state=pre_step_state,
        )


def test_network_accounting_binds_reference_density_to_scenario() -> None:
    scenario = Scenario.from_mapping(scenario_v3_mapping())
    pre_step_state = initial_state(scenario)
    result = advance_one_step(scenario, pre_step_state)
    corrupted = deepcopy(result.receipt)
    corrupted["air_network"]["air_density_kg_m3"] = 0.90
    corrupted["air_network"]["zone_mass_flow_kg_s"] = {
        zone_id: flow * 0.90
        for zone_id, flow in corrupted["air_network"]["zone_flow_m3_s"].items()
    }

    with pytest.raises(AccountingInvariantError, match="declared reference density"):
        validate_accounting_receipt(
            corrupted,
            scenario=scenario,
            pre_step_state=pre_step_state,
        )


def test_network_accounting_binds_fan_power_to_electrical_bus() -> None:
    scenario = Scenario.from_mapping(scenario_v3_mapping())
    pre_step_state = initial_state(scenario)
    result = advance_one_step(scenario, pre_step_state)
    corrupted = deepcopy(result.receipt)
    network = corrupted["air_network"]

    pressure_scale = 1.25
    network["fan_pressure_rise_pa"] *= pressure_scale
    network["shared_pressure_loss_pa"] *= pressure_scale
    network["branch_pressure_loss_pa"] = {
        zone_id: pressure * pressure_scale
        for zone_id, pressure in network["branch_pressure_loss_pa"].items()
    }
    network["fan_air_power_w"] = (
        network["fan_pressure_rise_pa"] * network["total_flow_m3_s"]
    )
    network["fan_electrical_power_w"] = (
        network["fan_air_power_w"] / network["total_efficiency"]
    )
    network["operating_point_residual_pa"] = 0.0

    with pytest.raises(AccountingInvariantError, match="electrical fan load"):
        validate_accounting_receipt(
            corrupted,
            scenario=scenario,
            pre_step_state=pre_step_state,
        )


def test_network_accounting_rejects_coherent_alternative_operating_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = Scenario.from_mapping(scenario_v3_mapping())

    with pytest.raises(AccountingInvariantError, match="causal recomputation"):
        _run_with_first_receipt_mutation(
            monkeypatch,
            scenario,
            lambda receipt: _replace_with_coherent_zero_flow_receipt(
                receipt, scenario
            ),
        )


def test_network_accounting_rejects_forged_non_network_receipt() -> None:
    scenario = Scenario.from_mapping(scenario_v3_mapping())
    pre_step_state = initial_state(scenario)
    receipt = deepcopy(advance_one_step(scenario, pre_step_state).receipt)
    receipt["electrical"]["generation_wh"] += 1000.0
    receipt["electrical"]["curtailed_generation_wh"] += 1000.0

    with pytest.raises(AccountingInvariantError, match="causal recomputation"):
        validate_accounting_receipt(
            receipt,
            scenario=scenario,
            pre_step_state=pre_step_state,
        )


def test_network_accounting_requires_v3_network_receipt() -> None:
    scenario = Scenario.from_mapping(scenario_v3_mapping())
    pre_step_state = initial_state(scenario)
    receipt = deepcopy(advance_one_step(scenario, pre_step_state).receipt)
    del receipt["air_network"]

    with pytest.raises(AccountingInvariantError, match="air-network receipt"):
        validate_accounting_receipt(
            receipt,
            scenario=scenario,
            pre_step_state=pre_step_state,
        )


@pytest.mark.parametrize(
    "field",
    (
        "requested_fan_speed_fraction",
        "actual_fan_speed_fraction",
        "requested_damper_position_by_id",
        "actual_damper_position_by_id",
    ),
)
def test_network_accounting_rejects_forged_actuator_receipt(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    scenario = Scenario.from_mapping(scenario_v3_mapping())

    def mutate(receipt: dict) -> None:
        network = receipt["air_network"]
        if field.endswith("damper_position_by_id"):
            damper_id = sorted(network[field])[0]
            network[field][damper_id] = 0.0
        else:
            network[field] = 0.0

    with pytest.raises(AccountingInvariantError, match="causal recomputation"):
        _run_with_first_receipt_mutation(monkeypatch, scenario, mutate)
