from __future__ import annotations

from copy import deepcopy

import pytest

from aeolus.habitat_v2.physics import advance_one_step, initial_state
from aeolus.habitat_v2.scenario import Scenario

from ._helpers import reference_scenario_mapping


def thermal_only_scenario() -> Scenario:
    mapping = deepcopy(reference_scenario_mapping())
    mapping["steps"] = 1
    mapping["timeline"][0]["end_step"] = 1
    mapping["equipment"]["base_load_w"] = 0.0
    for zone in mapping["zones"]:
        zone["passive_thermal_conductance_w_per_k"] = 0.0
    crew = mapping["zones"][0]
    crew["initial"]["temperature_k"] = 300.0
    crew["thermal_capacity_j_per_k"] = 5_000_000.0
    crew["passive_thermal_conductance_w_per_k"] = 100.0
    crew["sink_temperature_k"] = 280.0

    segment = mapping["timeline"][0]
    segment["generation_w"] = 1_000.0
    for load in segment["loads"].values():
        for field in load:
            load[field] = 0.0
    segment["loads"]["crew_cabin"]["sensible_heat_w"] = 500.0
    command = segment["command"]
    command["airflow_m3_s"] = {"crew_cabin": 0.0, "work_airlock": 0.0}
    command["scrubber_duty"] = 0.0
    command["condenser_duty"] = 0.0
    command["cooling_removed_w"] = {
        "crew_cabin": 100.0,
        "work_airlock": 0.0,
    }
    command["oxygen_injection_mol_s"] = {
        "crew_cabin": 0.0,
        "work_airlock": 0.0,
    }
    return Scenario.from_mapping(mapping)


def test_zone_thermal_receipt_closes_in_joules() -> None:
    scenario = thermal_only_scenario()
    before = initial_state(scenario)

    result = advance_one_step(scenario, before)
    after = result.state

    sensible_j = 500.0 * 60.0
    passive_j = 100.0 * (280.0 - 300.0) * 60.0
    cooling_j = 100.0 * 60.0
    expected_delta_j = sensible_j + passive_j - cooling_j
    expected_temperature_k = 300.0 + expected_delta_j / 5_000_000.0

    assert after.zones["crew_cabin"].temperature_k == pytest.approx(
        expected_temperature_k
    )
    receipt = result.receipt["thermal"]["zones"]["crew_cabin"]
    assert receipt["metabolic_heat_added_j"] == pytest.approx(sensible_j)
    assert receipt["recirculation_heat_added_j"] == pytest.approx(0.0)
    assert receipt["passive_heat_received_j"] == pytest.approx(0.0)
    assert receipt["passive_heat_rejected_j"] == pytest.approx(-passive_j)
    assert receipt["cooling_heat_removed_j"] == pytest.approx(cooling_j)
    assert receipt["zone_thermal_energy_delta_j"] == pytest.approx(expected_delta_j)
    scale_j = max(
        1.0,
        sensible_j,
        abs(passive_j),
        cooling_j,
        abs(expected_delta_j),
    )
    tolerance_j = max(1e-6, 1e-10 * scale_j)
    assert abs(receipt["zone_thermal_residual_j"]) <= tolerance_j
    assert abs(result.receipt["thermal"]["system_residual_j"]) <= tolerance_j


def test_recirculation_heat_is_equal_and_opposite_across_zones() -> None:
    mapping = deepcopy(reference_scenario_mapping())
    mapping["steps"] = 1
    mapping["timeline"][0]["end_step"] = 1
    mapping["equipment"]["base_load_w"] = 0.0
    mapping["equipment"]["airflow_slew_m3_s2"] = 1.0
    mapping["zones"][0]["initial"]["temperature_k"] = 300.0
    mapping["zones"][1]["initial"]["temperature_k"] = 290.0
    for zone in mapping["zones"]:
        zone["passive_thermal_conductance_w_per_k"] = 0.0
    segment = mapping["timeline"][0]
    segment["generation_w"] = 1_000.0
    for load in segment["loads"].values():
        for field in load:
            load[field] = 0.0
    command = segment["command"]
    command["airflow_m3_s"] = {"crew_cabin": 0.10, "work_airlock": 0.10}
    command["scrubber_duty"] = 0.0
    command["condenser_duty"] = 0.0
    command["cooling_removed_w"] = {"crew_cabin": 0.0, "work_airlock": 0.0}
    command["oxygen_injection_mol_s"] = {
        "crew_cabin": 0.0,
        "work_airlock": 0.0,
    }
    scenario = Scenario.from_mapping(mapping)
    before = initial_state(scenario)

    result = advance_one_step(scenario, before)
    after = result.state

    mixed_temperature_k = 295.0
    expected_crew_heat_j = 0.85 * 1005.0 * 0.10 * (mixed_temperature_k - 300.0) * 60.0
    expected_work_heat_j = -expected_crew_heat_j
    crew_receipt = result.receipt["thermal"]["zones"]["crew_cabin"]
    work_receipt = result.receipt["thermal"]["zones"]["work_airlock"]
    assert crew_receipt["recirculation_heat_added_j"] == pytest.approx(
        expected_crew_heat_j
    )
    assert work_receipt["recirculation_heat_added_j"] == pytest.approx(
        expected_work_heat_j
    )
    assert (
        crew_receipt["recirculation_heat_added_j"]
        + work_receipt["recirculation_heat_added_j"]
    ) == pytest.approx(0.0, abs=1e-9)
    assert after.zones["crew_cabin"].temperature_k == pytest.approx(
        300.0 + expected_crew_heat_j / 5_000_000.0
    )
    assert after.zones["work_airlock"].temperature_k == pytest.approx(
        290.0 + expected_work_heat_j / 4_000_000.0
    )


def test_electrical_deficit_uses_efficiency_adjusted_battery_energy() -> None:
    mapping = deepcopy(reference_scenario_mapping())
    mapping["steps"] = 1
    mapping["timeline"][0]["end_step"] = 1
    mapping["equipment"]["base_load_w"] = 600.0
    mapping["equipment"]["battery_capacity_wh"] = 100.0
    mapping["equipment"]["battery_discharge_efficiency"] = 0.80
    mapping["equipment"]["battery_max_discharge_w"] = 1_000.0
    mapping["initial_utility"]["battery_energy_wh"] = 100.0
    for zone in mapping["zones"]:
        zone["passive_thermal_conductance_w_per_k"] = 0.0
    segment = mapping["timeline"][0]
    segment["generation_w"] = 300.0
    for load in segment["loads"].values():
        for field in load:
            load[field] = 0.0
    command = segment["command"]
    command["airflow_m3_s"] = {"crew_cabin": 0.0, "work_airlock": 0.0}
    command["scrubber_duty"] = 0.0
    command["condenser_duty"] = 0.0
    command["cooling_removed_w"] = {"crew_cabin": 0.0, "work_airlock": 0.0}
    command["oxygen_injection_mol_s"] = {
        "crew_cabin": 0.0,
        "work_airlock": 0.0,
    }
    scenario = Scenario.from_mapping(mapping)
    before = initial_state(scenario)

    result = advance_one_step(scenario, before)
    after = result.state
    receipt = result.receipt["electrical"]

    assert receipt["generation_wh"] == pytest.approx(5.0)
    assert receipt["served_load_wh"] == pytest.approx(10.0)
    assert receipt["battery_withdrawn_wh"] == pytest.approx(6.25)
    assert receipt["battery_bus_output_wh"] == pytest.approx(5.0)
    assert receipt["discharge_conversion_loss_wh"] == pytest.approx(1.25)
    assert receipt["battery_charge_input_wh"] == pytest.approx(0.0)
    assert receipt["battery_charge_stored_wh"] == pytest.approx(0.0)
    assert receipt["curtailed_generation_wh"] == pytest.approx(0.0)
    assert receipt["battery_energy_delta_wh"] == pytest.approx(-6.25)
    assert after.utility.battery_energy_wh == pytest.approx(93.75)
    assert abs(receipt["residual_wh"]) <= 1e-10


def test_electrical_surplus_charges_to_capacity_then_records_curtailment() -> None:
    mapping = deepcopy(reference_scenario_mapping())
    mapping["steps"] = 1
    mapping["timeline"][0]["end_step"] = 1
    mapping["equipment"]["base_load_w"] = 0.0
    mapping["equipment"]["battery_capacity_wh"] = 100.0
    mapping["equipment"]["battery_charge_efficiency"] = 0.80
    mapping["equipment"]["battery_max_charge_w"] = 1_000.0
    mapping["initial_utility"]["battery_energy_wh"] = 95.0
    for zone in mapping["zones"]:
        zone["passive_thermal_conductance_w_per_k"] = 0.0
    segment = mapping["timeline"][0]
    segment["generation_w"] = 600.0
    for load in segment["loads"].values():
        for field in load:
            load[field] = 0.0
    command = segment["command"]
    command["airflow_m3_s"] = {"crew_cabin": 0.0, "work_airlock": 0.0}
    command["scrubber_duty"] = 0.0
    command["condenser_duty"] = 0.0
    command["cooling_removed_w"] = {"crew_cabin": 0.0, "work_airlock": 0.0}
    command["oxygen_injection_mol_s"] = {
        "crew_cabin": 0.0,
        "work_airlock": 0.0,
    }
    scenario = Scenario.from_mapping(mapping)

    result = advance_one_step(scenario, initial_state(scenario))
    receipt = result.receipt["electrical"]

    assert receipt["generation_wh"] == pytest.approx(10.0)
    assert receipt["battery_charge_input_wh"] == pytest.approx(6.25)
    assert receipt["battery_charge_stored_wh"] == pytest.approx(5.0)
    assert receipt["charge_conversion_loss_wh"] == pytest.approx(1.25)
    assert receipt["curtailed_generation_wh"] == pytest.approx(3.75)
    assert receipt["battery_withdrawn_wh"] == pytest.approx(0.0)
    assert result.state.utility.battery_energy_wh == pytest.approx(100.0)
    assert abs(receipt["residual_wh"]) <= 1e-10


def test_infeasible_electrical_command_returns_no_advanced_state() -> None:
    mapping = deepcopy(reference_scenario_mapping())
    mapping["steps"] = 1
    mapping["timeline"][0]["end_step"] = 1
    mapping["equipment"]["base_load_w"] = 2_000.0
    mapping["equipment"]["battery_capacity_wh"] = 100.0
    mapping["equipment"]["battery_max_discharge_w"] = 10.0
    mapping["initial_utility"]["battery_energy_wh"] = 1.0
    mapping["timeline"][0]["generation_w"] = 0.0
    scenario = Scenario.from_mapping(mapping)
    before = initial_state(scenario)

    from aeolus.habitat_v2.physics import InfeasibleActionError

    with pytest.raises(InfeasibleActionError, match="electrical demand"):
        advance_one_step(scenario, before)

    assert before.step == 0
    assert before.utility.battery_energy_wh == pytest.approx(1.0)
