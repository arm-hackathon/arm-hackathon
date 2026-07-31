"""Frozen-sensor fault behaviour in the plant and scenario runner."""

from __future__ import annotations

import pytest

from aeolus.config import load_scenario, parse_scenario
from aeolus.plant import initial_state, step_habitat
from aeolus.scenario import run_scenario


def _frozen_lab_doc(standard_doc: dict, start_tick: int) -> dict:
    standard_doc["version"] = 7
    standard_doc["fault_profiles"] = [
        {"type": "frozen_sensor", "zone_id": "lab", "start_tick": start_tick}
    ]
    return standard_doc


def test_step_habitat_freezes_sensor_reading_but_not_truth(standard_scenario_path):
    config = load_scenario(standard_scenario_path)
    state = initial_state(config)

    state, _ = step_habitat(config, state, frozen_zones={"lab"})
    freeze_value = state.sensor_co2_concentration["lab"]
    truth_at_freeze = state.zone_co2_mass["lab"]
    state, _ = step_habitat(config, state, frozen_zones={"lab"})
    state, _ = step_habitat(config, state, frozen_zones={"lab"})

    assert state.sensor_co2_concentration["lab"] == pytest.approx(freeze_value)
    assert state.zone_co2_mass["lab"] != pytest.approx(truth_at_freeze)


def test_frozen_sensor_holds_controller_setpoint(standard_scenario_path):
    config = load_scenario(standard_scenario_path)
    state = initial_state(config)

    setpoints = []
    for _ in range(3):
        state, _ = step_habitat(config, state, frozen_zones={"lab"})
        setpoints.append(state.actuators["lab"].setpoint)

    assert setpoints[1] == pytest.approx(setpoints[0])
    assert setpoints[2] == pytest.approx(setpoints[0])


def test_step_habitat_rejects_frozen_sensor_for_unknown_zone(standard_scenario_path):
    config = load_scenario(standard_scenario_path)

    with pytest.raises(ValueError, match="frozen"):
        step_habitat(config, initial_state(config), frozen_zones={"missing"})


def test_scenario_runner_applies_frozen_sensor_only_from_start_tick(
    standard_doc, tmp_path
):
    config = parse_scenario(_frozen_lab_doc(standard_doc, start_tick=40))
    trace_path = tmp_path / "frozen.jsonl"

    records = run_scenario(config, trace_path=trace_path)

    assert len(records) == 120
    assert len(trace_path.read_text(encoding="utf-8").splitlines()) == 120
    pre = [r.zones["lab"]["sensor_co2_concentration"] for r in records[:39]]
    post = [r.zones["lab"]["sensor_co2_concentration"] for r in records[39:]]
    assert len(set(pre)) > 1
    assert len(set(post)) == 1
    assert records[119].zones["lab"]["co2_concentration"] != pytest.approx(post[-1])
    assert set(records[60].zones["lab"]) == {
        "co2_mass",
        "co2_concentration",
        "sensor_co2_concentration",
        "source_co2_mass",
        "occupancy_multiplier",
    }
