from __future__ import annotations

import pytest

from aeolus.habitat_v2.scenario import Scenario, ScenarioValidationError

from ._helpers import reference_scenario_mapping


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_zone_id",
        "missing_actual_airflow_zone",
        "extra_load_zone",
        "missing_cooling_zone",
        "extra_oxygen_zone",
    ],
)
def test_zone_indexed_fields_must_match_exact_topology(mutation: str) -> None:
    mapping = reference_scenario_mapping()

    if mutation == "duplicate_zone_id":
        mapping["zones"][1]["id"] = "crew_cabin"
    elif mutation == "missing_actual_airflow_zone":
        del mapping["initial_utility"]["actual_airflow_m3_s"]["work_airlock"]
    elif mutation == "extra_load_zone":
        mapping["timeline"][0]["loads"]["ghost_zone"] = dict(
            mapping["timeline"][0]["loads"]["crew_cabin"]
        )
    elif mutation == "missing_cooling_zone":
        del mapping["timeline"][0]["command"]["cooling_removed_w"]["work_airlock"]
    elif mutation == "extra_oxygen_zone":
        mapping["timeline"][0]["command"]["oxygen_injection_mol_s"]["ghost_zone"] = 0.0
    else:  # pragma: no cover - protects the test table itself
        raise AssertionError(mutation)

    with pytest.raises(ScenarioValidationError, match="zone topology"):
        Scenario.from_mapping(mapping)


@pytest.mark.parametrize(
    "mutation",
    [
        "zero_dt",
        "fractional_steps",
        "zero_zone_volume",
        "invalid_relative_humidity",
        "excess_oxygen_fraction",
        "zero_cooling_cop",
        "invalid_battery_efficiency",
        "battery_above_capacity",
        "duty_above_one",
        "negative_zone_flow",
        "total_flow_above_capacity",
        "timeline_gap",
    ],
)
def test_invalid_physical_or_timeline_value_is_rejected(mutation: str) -> None:
    mapping = reference_scenario_mapping()

    if mutation == "zero_dt":
        mapping["dt_seconds"] = 0.0
    elif mutation == "fractional_steps":
        mapping["steps"] = 4.5
    elif mutation == "zero_zone_volume":
        mapping["zones"][0]["volume_m3"] = 0.0
    elif mutation == "invalid_relative_humidity":
        mapping["zones"][0]["initial"]["relative_humidity"] = 1.01
    elif mutation == "excess_oxygen_fraction":
        mapping["zones"][0]["initial"]["o2_mole_fraction"] = 0.71
    elif mutation == "zero_cooling_cop":
        mapping["equipment"]["cooling_coefficient_of_performance"] = 0.0
    elif mutation == "invalid_battery_efficiency":
        mapping["equipment"]["battery_charge_efficiency"] = 1.01
    elif mutation == "battery_above_capacity":
        mapping["initial_utility"]["battery_energy_wh"] = 10_001.0
    elif mutation == "duty_above_one":
        mapping["timeline"][0]["command"]["scrubber_duty"] = 1.01
    elif mutation == "negative_zone_flow":
        mapping["timeline"][0]["command"]["airflow_m3_s"]["crew_cabin"] = -0.01
    elif mutation == "total_flow_above_capacity":
        mapping["timeline"][0]["command"]["airflow_m3_s"] = {
            "crew_cabin": 0.11,
            "work_airlock": 0.10,
        }
    elif mutation == "timeline_gap":
        mapping["timeline"][0]["start_step"] = 1
    else:  # pragma: no cover - protects the test table itself
        raise AssertionError(mutation)

    with pytest.raises(ScenarioValidationError, match="invalid scenario value"):
        Scenario.from_mapping(mapping)
