"""Causal recovery-runner integration tests."""

from __future__ import annotations

import json
from pathlib import Path

from aeolus.config import parse_scenario
from aeolus.scenario import RunSpec, run_recovery_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_SCENARIO = REPO_ROOT / "scenarios" / "recovery_habitat.json"


def _document() -> dict:
    return json.loads(RECOVERY_SCENARIO.read_text(encoding="utf-8"))


def _config_with_fault(profile: dict):
    document = _document()
    document["fault_profiles"] = [profile]
    return parse_scenario(document)


def _blocked(*, transient: bool = False):
    profile = {
        "type": "blocked_path",
        "connection_id": "cabin_a_to_processing",
        "start_tick": 5,
        "blocked_effectiveness": 0.65,
    }
    if transient:
        profile.update(type="transient_blocked_path", end_tick=20)
    return profile


def _reserve_delivered(record) -> float:
    return float(record.reserve["system"]["total_delivered_airflow"])


def _authority_states(result) -> list[str]:
    return [record.authority["state"] for record in result.records]


def test_healthy_runner_is_causal_byte_stable_and_reserve_off(tmp_path):
    config = parse_scenario(_document())
    run = RunSpec(
        total_ticks=20,
        warmup_ticks=60,
        crew_cabin_co2_concentration_ceiling=0.30,
    )
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    first = run_recovery_scenario(
        config,
        run_id="healthy",
        governed=True,
        run=run,
        trace_path=first_path,
    )
    second = run_recovery_scenario(
        config,
        run_id="healthy",
        governed=True,
        run=run,
        trace_path=second_path,
    )

    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert len(first.records) == len(first.states) == len(first.decisions) == 20
    for tick, (record, decision) in enumerate(
        zip(first.records, first.decisions, strict=True), start=1
    ):
        assert record.plant.tick == tick
        assert decision.decision_tick == tick
        assert decision.observation_tick == tick - 1
        assert decision.sequence == tick
        assert record.authority["command_digest"] == decision.command_digest
        assert record.authority["applied_command_digest"] == decision.command_digest
        assert _reserve_delivered(record) == 0.0
    assert "PROTECT" not in _authority_states(first)


def test_persistent_fault_activates_only_governed_reserve_plane():
    config = _config_with_fault(_blocked())
    run = RunSpec(
        total_ticks=45,
        warmup_ticks=60,
        crew_cabin_co2_concentration_ceiling=0.30,
    )

    baseline = run_recovery_scenario(
        config, run_id="fault-baseline", governed=False, run=run
    )
    governed = run_recovery_scenario(
        config, run_id="fault-governed", governed=True, run=run
    )

    assert all(_reserve_delivered(record) == 0.0 for record in baseline.records)
    assert "PROTECT" in _authority_states(governed)
    assert any(_reserve_delivered(record) > 0.0 for record in governed.records)
    assert all(
        set(decision.reserve_commands) == {"cabin_a", "cabin_b", "lab"}
        for decision in governed.decisions
    )
    first_protect = _authority_states(governed).index("PROTECT")
    assert all(
        baseline.records[index].plant == governed.records[index].plant
        for index in range(first_protect)
    )


def test_frozen_sensor_fault_never_receives_reserve_authority():
    config = _config_with_fault(
        {
            "type": "frozen_sensor",
            "zone_id": "cabin_a",
            "start_tick": 5,
        }
    )
    result = run_recovery_scenario(
        config,
        run_id="frozen",
        governed=True,
        run=RunSpec(
            total_ticks=45,
            warmup_ticks=60,
            crew_cabin_co2_concentration_ceiling=0.30,
        ),
    )

    assert "PROTECT" not in _authority_states(result)
    assert all(_reserve_delivered(record) == 0.0 for record in result.records)


def test_transient_fault_handback_reaches_acknowledged_physical_zero():
    config = _config_with_fault(_blocked(transient=True))
    result = run_recovery_scenario(
        config,
        run_id="transient",
        governed=True,
        run=RunSpec(
            total_ticks=100,
            warmup_ticks=60,
            crew_cabin_co2_concentration_ceiling=0.30,
        ),
    )

    states = _authority_states(result)
    assert "PROTECT" in states
    assert "HANDBACK" in states
    last_handback = max(index for index, state in enumerate(states) if state == "HANDBACK")
    assert any(state == "NOMINAL" for state in states[last_handback + 1 :])
    for zone_id in ("cabin_a", "cabin_b", "lab"):
        values = [decision.reserve_commands[zone_id] for decision in result.decisions]
        assert all(abs(right - left) <= 0.1 + 1e-12 for left, right in zip(values, values[1:]))
    for record in result.records[-5:]:
        assert record.reserve["system"]["total_delivered_airflow"] == 0.0
        assert all(
            actuator["actual_position"] == 0.0
            for actuator in record.reserve["actuators"].values()
        )


def test_reserve_delivery_failure_latches_and_shuts_down_within_bound():
    document = _document()
    document["fault_profiles"] = [_blocked()]
    for connection in document["reserve_connections"]:
        if connection["id"] in {
            "reserve_cabin_a_to_processing",
            "reserve_processing_to_cabin_a",
        }:
            connection["health"] = 0.0
    config = parse_scenario(document)
    result = run_recovery_scenario(
        config,
        run_id="reserve-failure",
        governed=True,
        run=RunSpec(
            total_ticks=80,
            warmup_ticks=60,
            crew_cabin_co2_concentration_ceiling=0.30,
        ),
    )

    failure_events = [
        event for event in result.events if event.reason == "reserve_delivery_failure"
    ]
    assert len(failure_events) == 1
    failure_tick = failure_events[0].decision_tick
    after = result.decisions[failure_tick - 1 :]
    assert all(decision.state.value != "PROTECT" for decision in after)
    zero = next(
        decision
        for decision in after
        if all(value == 0.0 for value in decision.reserve_commands.values())
    )
    assert zero.decision_tick - failure_tick <= 36
    assert all(
        decision.state.value != "PROTECT"
        for decision in result.decisions[zero.decision_tick :]
    )


def test_recovery_runner_preserves_valid_scenario_zone_order():
    document = _document()
    non_processing = [
        zone for zone in document["zones"] if zone["preset"] != "air_processing"
    ]
    processing = [
        zone for zone in document["zones"] if zone["preset"] == "air_processing"
    ]
    document["zones"] = [*reversed(non_processing), *processing]
    config = parse_scenario(document)

    result = run_recovery_scenario(
        config,
        run_id="reordered-topology",
        governed=True,
        run=RunSpec(
            total_ticks=5,
            warmup_ticks=60,
            crew_cabin_co2_concentration_ceiling=0.30,
        ),
    )

    assert tuple(zone.id for zone in config.non_processing_zones()) == (
        "lab",
        "cabin_b",
        "cabin_a",
    )
    assert len(result.records) == len(result.decisions) == 5
