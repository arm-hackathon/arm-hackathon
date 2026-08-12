from __future__ import annotations

from copy import deepcopy

import pytest

from aeolus.habitat_v2.physics import advance_one_step, initial_state
from aeolus.habitat_v2.scenario import Scenario

from ._helpers import reference_scenario_mapping


def source_only_scenario() -> Scenario:
    mapping = reference_scenario_mapping()
    mapping["steps"] = 1
    mapping["timeline"][0]["end_step"] = 1
    mapping["equipment"]["base_load_w"] = 0.0
    for zone in mapping["zones"]:
        zone["passive_thermal_conductance_w_per_k"] = 0.0
    command = mapping["timeline"][0]["command"]
    command["airflow_m3_s"] = {"crew_cabin": 0.0, "work_airlock": 0.0}
    command["scrubber_duty"] = 0.0
    command["condenser_duty"] = 0.0
    command["cooling_removed_w"] = {"crew_cabin": 0.0, "work_airlock": 0.0}
    command["oxygen_injection_mol_s"] = {
        "crew_cabin": 0.0,
        "work_airlock": 0.0,
    }
    mapping["timeline"][0]["generation_w"] = 0.0
    mapping["timeline"][0]["loads"] = {
        "crew_cabin": {
            "co2_generation_mol_s": 0.001,
            "o2_consumption_mol_s": 0.002,
            "water_vapor_generation_mol_s": 0.003,
            "sensible_heat_w": 0.0,
        },
        "work_airlock": {
            "co2_generation_mol_s": 0.0,
            "o2_consumption_mol_s": 0.0,
            "water_vapor_generation_mol_s": 0.0,
            "sensible_heat_w": 0.0,
        },
    }
    return Scenario.from_mapping(mapping)


def test_one_step_applies_declared_species_sources_without_hidden_sink() -> None:
    scenario = source_only_scenario()
    before = initial_state(scenario)

    result = advance_one_step(scenario, before)
    after = result.state
    dt_seconds = float(scenario.data["dt_seconds"])

    assert after.step == 1
    assert after.zones["crew_cabin"].co2_mol - before.zones[
        "crew_cabin"
    ].co2_mol == pytest.approx(0.001 * dt_seconds)
    assert before.zones["crew_cabin"].o2_mol - after.zones[
        "crew_cabin"
    ].o2_mol == pytest.approx(0.002 * dt_seconds)
    assert after.zones["crew_cabin"].water_vapor_mol - before.zones[
        "crew_cabin"
    ].water_vapor_mol == pytest.approx(0.003 * dt_seconds)
    assert after.zones["crew_cabin"].inert_mol == before.zones["crew_cabin"].inert_mol
    assert after.zones["work_airlock"] == before.zones["work_airlock"]
    assert after.utility.captured_co2_mol == before.utility.captured_co2_mol
    assert after.utility.condensed_water_mol == before.utility.condensed_water_mol


def test_rate_limited_recirculation_mixes_zones_and_conserves_species() -> None:
    mapping = reference_scenario_mapping()
    mapping["steps"] = 1
    mapping["timeline"][0]["end_step"] = 1
    mapping["equipment"]["base_load_w"] = 0.0
    mapping["equipment"]["airflow_slew_m3_s2"] = 0.001
    mapping["zones"][0]["initial"]["co2_ppm"] = 2_000.0
    mapping["zones"][1]["initial"]["co2_ppm"] = 400.0
    for zone in mapping["zones"]:
        zone["passive_thermal_conductance_w_per_k"] = 0.0
    segment = mapping["timeline"][0]
    segment["generation_w"] = 0.0
    for load in segment["loads"].values():
        for field in load:
            load[field] = 0.0
    command = segment["command"]
    command["airflow_m3_s"] = {"crew_cabin": 0.08, "work_airlock": 0.08}
    command["scrubber_duty"] = 0.0
    command["condenser_duty"] = 0.0
    command["cooling_removed_w"] = {"crew_cabin": 0.0, "work_airlock": 0.0}
    command["oxygen_injection_mol_s"] = {
        "crew_cabin": 0.0,
        "work_airlock": 0.0,
    }
    scenario = Scenario.from_mapping(mapping)
    before = initial_state(scenario)

    after = advance_one_step(scenario, before).state

    assert after.utility.actual_airflow_m3_s == pytest.approx(
        {"crew_cabin": 0.06, "work_airlock": 0.06}
    )
    for species in ("co2_mol", "o2_mol", "water_vapor_mol", "inert_mol"):
        before_total = sum(getattr(zone, species) for zone in before.zones.values())
        after_total = sum(getattr(zone, species) for zone in after.zones.values())
        assert after_total == pytest.approx(before_total, rel=1e-12, abs=1e-12)

    zone_config = {zone["id"]: zone for zone in mapping["zones"]}
    before_crew = before.zones["crew_cabin"].telemetry(
        volume_m3=zone_config["crew_cabin"]["volume_m3"]
    )
    after_crew = after.zones["crew_cabin"].telemetry(
        volume_m3=zone_config["crew_cabin"]["volume_m3"]
    )
    before_work = before.zones["work_airlock"].telemetry(
        volume_m3=zone_config["work_airlock"]["volume_m3"]
    )
    after_work = after.zones["work_airlock"].telemetry(
        volume_m3=zone_config["work_airlock"]["volume_m3"]
    )
    assert after_crew["co2_ppm"] < before_crew["co2_ppm"]
    assert after_work["co2_ppm"] > before_work["co2_ppm"]


def test_processing_is_rate_limited_capacity_bounded_and_conservative() -> None:
    mapping = reference_scenario_mapping()
    mapping["steps"] = 1
    mapping["timeline"][0]["end_step"] = 1
    mapping["equipment"]["base_load_w"] = 0.0
    mapping["initial_utility"]["co2_sorbent_remaining_mol"] = 0.03
    for zone in mapping["zones"]:
        zone["passive_thermal_conductance_w_per_k"] = 0.0
        zone["initial"]["co2_ppm"] = 5_000.0
        zone["initial"]["relative_humidity"] = 0.80
    segment = mapping["timeline"][0]
    segment["generation_w"] = 0.0
    for load in segment["loads"].values():
        for field in load:
            load[field] = 0.0
    command = segment["command"]
    command["airflow_m3_s"] = {"crew_cabin": 0.08, "work_airlock": 0.06}
    command["scrubber_duty"] = 1.0
    command["condenser_duty"] = 1.0
    command["cooling_removed_w"] = {"crew_cabin": 0.0, "work_airlock": 0.0}
    command["oxygen_injection_mol_s"] = {
        "crew_cabin": 0.0,
        "work_airlock": 0.0,
    }
    scenario = Scenario.from_mapping(mapping)
    before = initial_state(scenario)

    after = advance_one_step(scenario, before).state

    assert after.utility.actual_scrubber_duty == pytest.approx(0.60)
    assert after.utility.actual_condenser_duty == pytest.approx(0.60)
    captured_delta = after.utility.captured_co2_mol - before.utility.captured_co2_mol
    condensed_delta = (
        after.utility.condensed_water_mol - before.utility.condensed_water_mol
    )
    airborne_co2_delta = sum(zone.co2_mol for zone in before.zones.values()) - sum(
        zone.co2_mol for zone in after.zones.values()
    )
    airborne_water_delta = sum(
        zone.water_vapor_mol for zone in before.zones.values()
    ) - sum(zone.water_vapor_mol for zone in after.zones.values())

    assert captured_delta == pytest.approx(0.03)
    assert after.utility.co2_sorbent_remaining_mol == pytest.approx(0.0)
    assert airborne_co2_delta == pytest.approx(captured_delta, abs=1e-12)
    assert condensed_delta > 0.0
    assert condensed_delta <= (0.003 * 0.60 * 60.0) + 1e-12
    assert airborne_water_delta == pytest.approx(condensed_delta, abs=1e-12)


def test_oxygen_injection_moves_inventory_without_creating_mass() -> None:
    mapping = deepcopy(source_only_scenario().data)
    for load in mapping["timeline"][0]["loads"].values():
        for field in load:
            load[field] = 0.0
    mapping["timeline"][0]["command"]["oxygen_injection_mol_s"] = {
        "crew_cabin": 0.00030,
        "work_airlock": 0.00020,
    }
    scenario = Scenario.from_mapping(mapping)
    before = initial_state(scenario)

    after = advance_one_step(scenario, before).state

    injected_mol = (0.00030 + 0.00020) * 60.0
    airborne_o2_delta = sum(zone.o2_mol for zone in after.zones.values()) - sum(
        zone.o2_mol for zone in before.zones.values()
    )
    store_delta = before.utility.oxygen_store_mol - after.utility.oxygen_store_mol
    assert airborne_o2_delta == pytest.approx(injected_mol)
    assert store_delta == pytest.approx(injected_mol)


def test_post_cooling_supersaturation_condenses_and_conserves_water() -> None:
    mapping = deepcopy(reference_scenario_mapping())
    mapping["steps"] = 1
    mapping["timeline"][0]["end_step"] = 1
    mapping["equipment"]["base_load_w"] = 0.0
    crew = mapping["zones"][0]
    crew["initial"]["temperature_k"] = 300.0
    crew["initial"]["relative_humidity"] = 0.99
    crew["thermal_capacity_j_per_k"] = 10_000.0
    for zone in mapping["zones"]:
        zone["passive_thermal_conductance_w_per_k"] = 0.0
    segment = mapping["timeline"][0]
    segment["generation_w"] = 1_000.0
    for load in segment["loads"].values():
        for field in load:
            load[field] = 0.0
    command = segment["command"]
    command["airflow_m3_s"] = {"crew_cabin": 0.0, "work_airlock": 0.0}
    command["scrubber_duty"] = 0.0
    command["condenser_duty"] = 0.0
    command["cooling_removed_w"] = {
        "crew_cabin": 1_000.0,
        "work_airlock": 0.0,
    }
    command["oxygen_injection_mol_s"] = {
        "crew_cabin": 0.0,
        "work_airlock": 0.0,
    }
    scenario = Scenario.from_mapping(mapping)
    before = initial_state(scenario)

    result = advance_one_step(scenario, before)
    after = result.state
    crew_config = {zone["id"]: zone for zone in mapping["zones"]}["crew_cabin"]
    telemetry = after.zones["crew_cabin"].telemetry(volume_m3=crew_config["volume_m3"])
    vapour_removed_mol = (
        before.zones["crew_cabin"].water_vapor_mol
        - after.zones["crew_cabin"].water_vapor_mol
    )
    condensate_added_mol = (
        after.utility.condensed_water_mol - before.utility.condensed_water_mol
    )

    assert telemetry["relative_humidity"] == pytest.approx(1.0, abs=1e-12)
    assert vapour_removed_mol > 0.0
    assert condensate_added_mol == pytest.approx(vapour_removed_mol)
    assert result.receipt["passive_condensation_mol"]["crew_cabin"] == pytest.approx(
        vapour_removed_mol
    )


def test_step_fails_before_return_when_crew_load_exceeds_zone_oxygen() -> None:
    mapping = deepcopy(source_only_scenario().data)
    mapping["timeline"][0]["loads"]["crew_cabin"]["o2_consumption_mol_s"] = 1_000.0
    scenario = Scenario.from_mapping(mapping)
    before = initial_state(scenario)

    from aeolus.habitat_v2.physics import InfeasibleActionError

    with pytest.raises(InfeasibleActionError, match="oxygen load"):
        advance_one_step(scenario, before)

    assert before.step == 0
    assert before.zones["crew_cabin"].o2_mol > 0.0


def test_global_species_and_water_receipt_closes() -> None:
    scenario = Scenario.from_mapping(reference_scenario_mapping())
    result = advance_one_step(scenario, initial_state(scenario))

    accounting = result.receipt["species_accounting"]
    tolerance_mol = accounting["tolerance_mol"]
    for field in (
        "co2_residual_mol",
        "o2_residual_mol",
        "water_residual_mol",
        "inert_residual_mol",
    ):
        assert abs(accounting[field]) <= tolerance_mol


def test_one_step_physics_is_independent_of_zone_iteration_order() -> None:
    forward_mapping = reference_scenario_mapping()
    reversed_mapping = deepcopy(forward_mapping)
    reversed_mapping["zones"].reverse()
    forward_scenario = Scenario.from_mapping(forward_mapping)
    reversed_scenario = Scenario.from_mapping(reversed_mapping)

    forward = advance_one_step(forward_scenario, initial_state(forward_scenario))
    reversed_result = advance_one_step(
        reversed_scenario, initial_state(reversed_scenario)
    )

    assert forward.state == reversed_result.state
    assert forward.receipt == reversed_result.receipt
