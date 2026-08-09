from __future__ import annotations

import pytest

from aeolus.habitat_v2.physics import initial_state
from aeolus.habitat_v2.scenario import Scenario

from ._helpers import reference_scenario_mapping


def test_initial_species_inventories_reproduce_declared_atmosphere() -> None:
    mapping = reference_scenario_mapping()
    scenario = Scenario.from_mapping(mapping)

    state = initial_state(scenario)

    assert state.step == 0
    for zone_config in mapping["zones"]:
        zone_id = zone_config["id"]
        initial = zone_config["initial"]
        zone = state.zones[zone_id]
        telemetry = zone.telemetry(volume_m3=zone_config["volume_m3"])

        assert zone.inert_mol > 0.0
        assert telemetry["pressure_pa"] == pytest.approx(
            initial["pressure_pa"], rel=1e-12
        )
        assert telemetry["co2_ppm"] == pytest.approx(initial["co2_ppm"], rel=1e-12)
        assert telemetry["o2_mole_fraction"] == pytest.approx(
            initial["o2_mole_fraction"], rel=1e-12
        )
        assert telemetry["relative_humidity"] == pytest.approx(
            initial["relative_humidity"], rel=1e-12
        )
