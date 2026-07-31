"""Behavioural acceptance for the checked-in blocked-path and frozen-sensor scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aeolus.config import BlockedPath, FrozenSensor, load_scenario
from aeolus.scenario import run_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCENARIO = REPO_ROOT / "scenarios" / "high_demand_healthy.json"
BLOCKED_SCENARIO = REPO_ROOT / "scenarios" / "blocked_path.json"
FROZEN_SCENARIO = REPO_ROOT / "scenarios" / "frozen_sensor.json"


def test_fault_scenarios_are_paired_with_high_demand_baseline():
    base = json.loads(BASELINE_SCENARIO.read_text(encoding="utf-8"))
    base.pop("fault_profiles")

    blocked_doc = json.loads(BLOCKED_SCENARIO.read_text(encoding="utf-8"))
    assert blocked_doc.pop("fault_profiles"), "blocked_path must declare a fault"
    assert blocked_doc == base, "blocked_path.json differs from the baseline habitat"

    frozen_doc = json.loads(FROZEN_SCENARIO.read_text(encoding="utf-8"))
    assert frozen_doc.pop("fault_profiles"), "frozen_sensor must declare a fault"
    frozen_lab = next(zone for zone in frozen_doc["zones"] if zone["id"] == "lab")
    assert frozen_lab.pop("occupancy_profile"), (
        "frozen_sensor.json must step lab demand so the frozen reading diverges"
    )
    base_lab = next(zone for zone in base["zones"] if zone["id"] == "lab")
    base_lab.pop("occupancy_profile")
    assert frozen_doc == base, (
        "frozen_sensor.json differs from the baseline beyond lab occupancy"
    )


def test_blocked_path_scenario_declares_block_on_cabin_b_outbound():
    config = load_scenario(BLOCKED_SCENARIO)

    assert config.fault_profiles == (
        BlockedPath(
            connection_id="cabin_b_to_processing",
            start_tick=30,
            blocked_effectiveness=0.05,
        ),
    )


def test_blocked_path_scenario_collapses_cabin_b_delivery_at_block_tick():
    records = run_scenario(load_scenario(BLOCKED_SCENARIO))

    assert len(records) == 120
    by_tick = {record.tick: record for record in records}
    before = by_tick[29].connections["cabin_b_to_processing"]
    after = by_tick[31].connections["cabin_b_to_processing"]
    assert before["airflow_residual"] == pytest.approx(0.0, abs=1e-12)
    assert after["delivered_airflow"] == pytest.approx(
        0.05 * after["requested_airflow"], rel=1e-9
    )
    assert after["airflow_residual"] == pytest.approx(
        0.95 * after["requested_airflow"], rel=1e-9
    )
    assert (
        by_tick[120].zones["cabin_b"]["co2_concentration"]
        > by_tick[120].zones["cabin_a"]["co2_concentration"]
    )


def test_frozen_sensor_scenario_declares_lab_freeze():
    config = load_scenario(FROZEN_SCENARIO)

    assert config.fault_profiles == (FrozenSensor(zone_id="lab", start_tick=30),)


def test_frozen_sensor_scenario_holds_lab_reading_from_freeze_tick():
    records = run_scenario(load_scenario(FROZEN_SCENARIO))

    assert len(records) == 120
    pre = [r.zones["lab"]["sensor_co2_concentration"] for r in records[:29]]
    post = [r.zones["lab"]["sensor_co2_concentration"] for r in records[29:]]
    assert len(set(pre)) > 1
    assert len(set(post)) == 1
    # Lab demand drops at tick 41; the frozen reading cannot follow the truth down.
    assert records[119].zones["lab"]["co2_concentration"] < post[-1]
    lab_setpoints = [r.actuators["lab"]["setpoint"] for r in records[29:]]
    assert len(set(lab_setpoints)) == 1
    cabin_a_readings = {
        r.zones["cabin_a"]["sensor_co2_concentration"] for r in records[29:]
    }
    assert len(cabin_a_readings) > 1
