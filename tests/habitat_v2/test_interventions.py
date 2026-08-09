from __future__ import annotations

from copy import deepcopy

from aeolus.habitat_v2.runner import run_scenario
from aeolus.habitat_v2.scenario import Scenario

from ._helpers import reference_scenario_mapping


def _run(mapping):
    return run_scenario(Scenario.from_mapping(mapping))


def test_increasing_scrubber_duty_lowers_future_co2() -> None:
    low = reference_scenario_mapping()
    high = deepcopy(low)
    low["timeline"][0]["command"]["scrubber_duty"] = 0.0
    high["timeline"][0]["command"]["scrubber_duty"] = 0.8

    low_final = _run(low).rows[-1]["telemetry"]
    high_final = _run(high).rows[-1]["telemetry"]

    for zone_id in low_final:
        assert high_final[zone_id]["co2_ppm"] < low_final[zone_id]["co2_ppm"]


def test_increasing_condenser_duty_lowers_future_humidity() -> None:
    low = reference_scenario_mapping()
    high = deepcopy(low)
    low["timeline"][0]["command"]["condenser_duty"] = 0.0
    high["timeline"][0]["command"]["condenser_duty"] = 0.8

    low_final = _run(low).rows[-1]["telemetry"]
    high_final = _run(high).rows[-1]["telemetry"]

    for zone_id in low_final:
        assert (
            high_final[zone_id]["relative_humidity"]
            < low_final[zone_id]["relative_humidity"]
        )


def test_allocating_more_cooling_lowers_future_temperature() -> None:
    low = reference_scenario_mapping()
    high = deepcopy(low)
    low["timeline"][0]["command"]["cooling_removed_w"] = {
        "crew_cabin": 0.0,
        "work_airlock": 0.0,
    }
    high["timeline"][0]["command"]["cooling_removed_w"] = {
        "crew_cabin": 300.0,
        "work_airlock": 300.0,
    }

    low_final = _run(low).rows[-1]["telemetry"]
    high_final = _run(high).rows[-1]["telemetry"]

    for zone_id in low_final:
        assert (
            high_final[zone_id]["temperature_k"] < low_final[zone_id]["temperature_k"]
        )


def test_increased_airflow_changes_cross_zone_environmental_coupling() -> None:
    isolated = reference_scenario_mapping()
    coupled = deepcopy(isolated)
    for mapping in (isolated, coupled):
        mapping["steps"] = 1
        mapping["timeline"][0]["end_step"] = 1
        mapping["zones"][0]["initial"]["co2_ppm"] = 2_000.0
        mapping["zones"][1]["initial"]["co2_ppm"] = 400.0
        mapping["timeline"][0]["command"]["scrubber_duty"] = 0.0
        for load in mapping["timeline"][0]["loads"].values():
            load["co2_generation_mol_s"] = 0.0
            load["o2_consumption_mol_s"] = 0.0
            load["water_vapor_generation_mol_s"] = 0.0
            load["sensible_heat_w"] = 0.0
    isolated["timeline"][0]["command"]["airflow_m3_s"] = {
        "crew_cabin": 0.0,
        "work_airlock": 0.0,
    }
    coupled["timeline"][0]["command"]["airflow_m3_s"] = {
        "crew_cabin": 0.10,
        "work_airlock": 0.10,
    }

    isolated_final = _run(isolated).rows[-1]["telemetry"]
    coupled_final = _run(coupled).rows[-1]["telemetry"]

    assert (
        coupled_final["crew_cabin"]["co2_ppm"] < isolated_final["crew_cabin"]["co2_ppm"]
    )
    assert (
        coupled_final["work_airlock"]["co2_ppm"]
        > isolated_final["work_airlock"]["co2_ppm"]
    )
