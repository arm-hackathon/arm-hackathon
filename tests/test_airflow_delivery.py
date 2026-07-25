"""Observable request-versus-delivery contracts for schema-v7 scenarios."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from icarus.actuator import ActuatorState
from icarus.config import ConnectionSpec, load_scenario, parse_scenario
from icarus.plant import initial_state, requested_loop_airflow, step_habitat
from icarus.scenario import STANDARD_RUN, run_scenario


REPO_ROOT = Path(__file__).resolve().parents[1]
STANDARD_PATH = REPO_ROOT / "scenarios" / "standard_habitat.json"
HIGH_DEMAND_PATH = REPO_ROOT / "scenarios" / "high_demand_healthy.json"
DEGRADATION_PATH = REPO_ROOT / "scenarios" / "primary_fan_degradation.json"


def _fault_profile(**overrides):
    profile = {
        "type": "gradual_primary_fan_degradation",
        "connection_id": "cabin_a_to_processing",
        "start_tick": 20,
        "end_tick": 80,
        "end_effectiveness": 0.4,
    }
    profile.update(overrides)
    return profile


def _config_doc() -> dict:
    return json.loads(STANDARD_PATH.read_text(encoding="utf-8"))


def test_requested_loop_airflow_uses_capacity_and_measured_position_not_health():
    outbound = ConnectionSpec(
        id="outbound", from_zone="cabin", to_zone="processing", max_airflow=10.0, health=0.2
    )
    inbound = ConnectionSpec(
        id="inbound", from_zone="processing", to_zone="cabin", max_airflow=8.0, health=0.9
    )

    assert requested_loop_airflow(outbound, inbound, actuator_position=0.5) == pytest.approx(4.0)


def test_fault_only_directly_reduces_the_target_loop_without_shared_capacity_coupling():
    healthy_doc = _config_doc()
    healthy_doc["version"] = 7
    healthy_doc["fault_profiles"] = []
    healthy_doc["air_system"]["shared_airflow_capacity"] = 40.0
    degraded_doc = copy.deepcopy(healthy_doc)
    degraded_doc["fault_profiles"] = [_fault_profile()]

    healthy_config = parse_scenario(healthy_doc)
    degraded_config = parse_scenario(degraded_doc)
    starting_state = initial_state(healthy_config)
    positioned_state = replace(
        starting_state,
        actuators={zone_id: ActuatorState(actual_position=1.0) for zone_id in starting_state.actuators},
    )

    healthy_state, healthy_flows = step_habitat(healthy_config, positioned_state)
    degraded_state, degraded_flows = step_habitat(
        degraded_config,
        positioned_state,
        connection_effectiveness={"cabin_a_to_processing": 0.4},
    )

    assert degraded_state.requested_airflows["cabin_a_to_processing"] == pytest.approx(
        healthy_state.requested_airflows["cabin_a_to_processing"]
    )
    assert degraded_flows["cabin_a_to_processing"] < healthy_flows["cabin_a_to_processing"]
    assert degraded_state.airflow_residuals["cabin_a_to_processing"] > 0.0
    for connection_id in (
        "cabin_b_to_processing",
        "processing_to_cabin_b",
        "lab_to_processing",
        "processing_to_lab",
    ):
        assert degraded_state.requested_airflows[connection_id] == pytest.approx(
            healthy_state.requested_airflows[connection_id]
        )
        assert degraded_flows[connection_id] == pytest.approx(healthy_flows[connection_id])


def test_standard_scenario_is_v7_and_declares_no_fault_profile():
    config = load_scenario(STANDARD_PATH)

    assert config.version == 7
    assert config.fault_profiles == ()


def test_paired_high_demand_scenarios_differ_only_by_fault_profiles():
    healthy = json.loads(HIGH_DEMAND_PATH.read_text(encoding="utf-8"))
    degraded = json.loads(DEGRADATION_PATH.read_text(encoding="utf-8"))

    healthy.pop("fault_profiles")
    degraded.pop("fault_profiles")

    assert healthy == degraded


def test_same_v7_scenario_produces_byte_identical_records_and_traces(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    first_records = run_scenario(load_scenario(DEGRADATION_PATH), trace_path=first)
    second_records = run_scenario(load_scenario(DEGRADATION_PATH), trace_path=second)

    assert first_records == second_records
    assert first.read_bytes() == second.read_bytes()


def test_healthy_high_demand_raises_request_and_delivery_tracks_without_fault_residual():
    standard = run_scenario(load_scenario(STANDARD_PATH))
    high_demand = run_scenario(load_scenario(HIGH_DEMAND_PATH))

    standard_request_total = sum(
        record.connections["cabin_a_to_processing"]["requested_airflow"] for record in standard
    )
    high_demand_request_total = sum(
        record.connections["cabin_a_to_processing"]["requested_airflow"]
        for record in high_demand
    )
    assert high_demand_request_total > standard_request_total

    for record in high_demand:
        for connection in record.connections.values():
            assert connection["delivered_airflow"] == pytest.approx(
                connection["requested_airflow"]
            )
            assert connection["airflow_residual"] == pytest.approx(0.0)


def test_degradation_separates_requested_and_delivered_airflow_at_profile_end():
    records = run_scenario(load_scenario(DEGRADATION_PATH))
    at_start = records[19].connections["cabin_a_to_processing"]
    at_end = records[79].connections["cabin_a_to_processing"]

    assert at_start["airflow_residual"] == pytest.approx(0.0)
    assert at_end["requested_airflow"] > 0.0
    assert at_end["delivered_airflow"] < at_end["requested_airflow"]
    assert at_end["airflow_residual"] == pytest.approx(
        at_end["requested_airflow"] - at_end["delivered_airflow"]
    )


def test_loop_return_delivery_matches_outbound_and_delivery_never_exceeds_request_or_capacity():
    config = load_scenario(DEGRADATION_PATH)

    for record in run_scenario(config):
        outbound_total = 0.0
        for zone in config.non_processing_zones():
            outbound = config.path_to_processing(zone.id).id
            inbound = config.path_from_processing(zone.id).id
            outbound_entry = record.connections[outbound]
            inbound_entry = record.connections[inbound]
            assert inbound_entry["delivered_airflow"] == pytest.approx(
                outbound_entry["delivered_airflow"]
            )
            assert outbound_entry["delivered_airflow"] <= outbound_entry["requested_airflow"]
            assert inbound_entry["delivered_airflow"] <= inbound_entry["requested_airflow"]
            outbound_total += outbound_entry["delivered_airflow"]
        assert outbound_total <= config.air_system.shared_airflow_capacity


def test_degradation_run_preserves_mass_conservation_and_measured_tick_numbering():
    config = load_scenario(DEGRADATION_PATH)
    records = run_scenario(config)

    assert len(records) == STANDARD_RUN.total_ticks
    assert [record.tick for record in records] == list(range(1, STANDARD_RUN.total_ticks + 1))
    assert records[0].connections["cabin_a_to_processing"]["airflow_residual"] == pytest.approx(0.0)

    generated = sum(
        zone["source_co2_mass"]
        for record in records
        for zone in record.zones.values()
    )
    final_airborne = sum(zone["co2_mass"] for zone in records[-1].zones.values())
    captured = records[-1].zones[config.processing_zone().id]["captured_co2"]
    initial_airborne = sum(
        records[0].zones[zone.id]["sensor_co2_concentration"] * zone.air_volume
        - records[0].zones[zone.id]["source_co2_mass"]
        for zone in config.zones
    )

    assert final_airborne + captured == pytest.approx(initial_airborne + generated)
