from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aeolus.habitat_v2.hmc import HabitatManagementComputer
from aeolus.habitat_v2.hmc_contract import load_hmc_contract
from aeolus.habitat_v2.physics import command_from_achieved_state
from aeolus.habitat_v2.safety import ArbitrationReceipt
from aeolus.habitat_v2.scenario import Scenario


def _scenario() -> Scenario:
    path = Path(__file__).parents[2] / "scenarios" / "habitat_v2_actuator_feedback.json"
    return Scenario.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def _scenario_with_first_mode(mode: str) -> Scenario:
    path = Path(__file__).parents[2] / "scenarios" / "habitat_v2_actuator_feedback.json"
    mapping = json.loads(path.read_text(encoding="utf-8"))
    mapping["timeline"][0]["operating_mode"] = mode
    return Scenario.from_mapping(mapping)


def _scenario_with_low_battery() -> Scenario:
    path = Path(__file__).parents[2] / "scenarios" / "habitat_v2_actuator_feedback.json"
    mapping = json.loads(path.read_text(encoding="utf-8"))
    mapping["initial_utility"]["battery_energy_wh"] = 3800.0
    return Scenario.from_mapping(mapping)


def _contract():
    path = Path(__file__).parents[2] / "contracts" / "habitat_v2_hmc_v1.json"
    return load_hmc_contract(path)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _without(mapping: dict[str, object], field: str) -> dict[str, object]:
    result = dict(mapping)
    result.pop(field)
    return result


def _chain_hash(
    previous: str,
    ordinal: int,
    event_kind: str,
    receipt_sha256: str,
) -> str:
    kind = event_kind.encode("utf-8")
    return hashlib.sha256(
        b"aeolus-habitat-v2-hmc-control-chain-v1"
        + bytes.fromhex(previous)
        + ordinal.to_bytes(8, "big")
        + len(kind).to_bytes(8, "big")
        + kind
        + bytes.fromhex(receipt_sha256)
    ).hexdigest()


def test_no_proposal_arbitration_selects_initial_achieved_hold_without_mutation() -> (
    None
):
    scenario = _scenario()
    contract = _contract()
    hmc = HabitatManagementComputer.reset(scenario, contract, b"a" * 32)
    snapshot, verification = hmc.observe()
    handle = hmc.verify_snapshot(snapshot, verification)
    proposal_receipt = hmc.propose(None, handle)
    previous_chain = hmc.current_control_chain_sha256
    state_before = hmc._state
    hold = command_from_achieved_state(scenario, state_before).command

    receipt = hmc.arbitrate()

    assert type(receipt) is ArbitrationReceipt
    assert hmc.lifecycle_phase == "ARBITRATED"
    assert hmc._state is state_before
    mapping = receipt.to_mapping()
    assert mapping == {
        "receipt_schema_sha256": contract.arbitration_receipt_schema_sha256,
        "hmc_contract_sha256": contract.hmc_contract_sha256,
        "safety_policy_sha256": contract.safety_policy_sha256,
        "safe_action_catalogue_sha256": contract.safe_action_catalogue_sha256,
        "preflight_contract_sha256": contract.preflight_contract_sha256,
        "observable_topology_sha256": hmc.observable_topology_sha256,
        "control_run_id": hmc.control_run_id,
        "authority_epoch": hmc.authority_epoch,
        "sequence": 0,
        "observation_snapshot_sha256": snapshot.snapshot_sha256,
        "proposal_receipt_sha256": proposal_receipt.proposal_receipt_sha256,
        "accepted_proposal_sha256": None,
        "requested_command": None,
        "requested_command_sha256": None,
        "final_command": hold.to_mapping(),
        "final_command_sha256": hold.sha256,
        "disposition": "REJECTED",
        "reason_codes": ["safe_hold_no_proposal"],
        "command_owner": "baseline_hold",
        "emergency_override": False,
        "emergency_reserve_use": False,
        "imminent_application_mode": "occupied",
        "preflight_result": receipt.preflight_result,
        "decision_step": 0,
        "application_step": 0,
        "event_ordinal": 2,
        "previous_control_chain_sha256": previous_chain,
        "arbitration_receipt_sha256": receipt.arbitration_receipt_sha256,
    }
    preflight = mapping["preflight_result"]
    assert preflight == {
        "classification": "FEASIBLE",
        "application_step": 0,
        "command_sha256": hold.sha256,
        "preflight_contract_sha256": contract.preflight_contract_sha256,
        "preflight_result_sha256": preflight["preflight_result_sha256"],
    }
    expected_preflight_digest = hashlib.sha256(
        _canonical_bytes(_without(preflight, "preflight_result_sha256"))
    ).hexdigest()
    assert preflight["preflight_result_sha256"] == expected_preflight_digest
    assert (
        receipt.arbitration_receipt_sha256
        == hashlib.sha256(
            _canonical_bytes(_without(mapping, "arbitration_receipt_sha256"))
        ).hexdigest()
    )
    assert len(hmc.control_events) == 3
    event = hmc.control_events[-1].to_mapping()
    assert event["event_kind"] == "ARBITRATION"
    assert event["receipt"] == mapping
    assert event["control_chain_sha256"] == _chain_hash(
        previous_chain,
        2,
        "ARBITRATION",
        receipt.arbitration_receipt_sha256,
    )


def test_occupied_valid_proposal_is_accepted_but_external_source_never_owns_command() -> (
    None
):
    scenario = _scenario()
    hmc = HabitatManagementComputer.reset(scenario, _contract(), b"o" * 32)
    snapshot, verification = hmc.observe()
    handle = hmc.verify_snapshot(snapshot, verification)
    proposed_command = json.loads(json.dumps(scenario.data["timeline"][0]["command"]))
    proposal_body = {
        "schema_version": "aeolus_habitat_v2_control_proposal_v1",
        "control_run_id": hmc.control_run_id,
        "authority_epoch": hmc.authority_epoch,
        "source_id": "external-adviser-v1",
        "source_type": "external_adviser",
        "completed_observation_step": 0,
        "observation_snapshot_sha256": snapshot.snapshot_sha256,
        "requested_application_step": 0,
        "observable_topology_sha256": hmc.observable_topology_sha256,
        "proposed_command": proposed_command,
        "confidence": None,
    }
    proposal_sha256 = hashlib.sha256(_canonical_bytes(proposal_body)).hexdigest()
    proposal_receipt = hmc.propose(
        {**proposal_body, "proposal_sha256": proposal_sha256}, handle
    )
    state_before = hmc._state

    receipt = hmc.arbitrate()

    mapping = receipt.to_mapping()
    assert hmc._state is state_before
    assert mapping["accepted_proposal_sha256"] == proposal_sha256
    assert mapping["requested_command"] == proposed_command
    assert (
        mapping["requested_command_sha256"] == proposal_receipt.requested_command_sha256
    )
    assert mapping["final_command"] == proposed_command
    assert mapping["final_command_sha256"] == proposal_receipt.requested_command_sha256
    assert mapping["disposition"] == "ACCEPTED"
    assert mapping["reason_codes"] == ["accepted_as_proposed"]
    assert mapping["command_owner"] == "supervisor_modified"
    assert mapping["command_owner"] != "external_adviser"
    assert mapping["emergency_override"] is False
    assert mapping["emergency_reserve_use"] is False
    assert mapping["imminent_application_mode"] == "occupied"
    assert mapping["preflight_result"]["classification"] == "FEASIBLE"
    assert (
        mapping["preflight_result"]["command_sha256"] == mapping["final_command_sha256"]
    )


def test_rejected_proposal_arbitration_falls_back_to_complete_safe_hold() -> None:
    scenario = _scenario()
    hmc = HabitatManagementComputer.reset(scenario, _contract(), b"r" * 32)
    snapshot, verification = hmc.observe()
    handle = hmc.verify_snapshot(snapshot, verification)
    proposal_receipt = hmc.propose({}, handle)
    hold = command_from_achieved_state(scenario, hmc._state).command

    receipt = hmc.arbitrate()

    mapping = receipt.to_mapping()
    assert proposal_receipt.reason_code == "rejected_malformed"
    assert mapping["accepted_proposal_sha256"] is None
    assert mapping["requested_command"] is None
    assert mapping["requested_command_sha256"] is None
    assert mapping["final_command"] == hold.to_mapping()
    assert mapping["final_command_sha256"] == hold.sha256
    assert mapping["disposition"] == "REJECTED"
    assert mapping["reason_codes"] == ["rejected_malformed"]
    assert mapping["command_owner"] == "baseline_hold"
    assert mapping["emergency_override"] is False
    assert mapping["preflight_result"]["classification"] == "FEASIBLE"


def test_dormant_mode_clamps_only_oxygen_and_cooling_increases_to_safe_hold() -> None:
    scenario = _scenario_with_first_mode("dormant")
    hmc = HabitatManagementComputer.reset(scenario, _contract(), b"d" * 32)
    snapshot, verification = hmc.observe()
    handle = hmc.verify_snapshot(snapshot, verification)
    hold = command_from_achieved_state(scenario, hmc._state).command.to_mapping()
    proposed = json.loads(json.dumps(hold))
    proposed["fan_speed_fraction"] = min(1.0, hold["fan_speed_fraction"] + 0.1)
    first_zone = min(proposed["cooling_removed_w"])
    proposed["cooling_removed_w"][first_zone] = (
        hold["cooling_removed_w"][first_zone] + 100.0
    )
    proposed["oxygen_injection_mol_s"][first_zone] = (
        hold["oxygen_injection_mol_s"][first_zone] + 0.0003
    )
    proposal_body = {
        "schema_version": "aeolus_habitat_v2_control_proposal_v1",
        "control_run_id": hmc.control_run_id,
        "authority_epoch": hmc.authority_epoch,
        "source_id": "baseline-v1",
        "source_type": "deterministic_baseline",
        "completed_observation_step": 0,
        "observation_snapshot_sha256": snapshot.snapshot_sha256,
        "requested_application_step": 0,
        "observable_topology_sha256": hmc.observable_topology_sha256,
        "proposed_command": proposed,
        "confidence": 1.0,
    }
    proposal_sha256 = hashlib.sha256(_canonical_bytes(proposal_body)).hexdigest()
    hmc.propose({**proposal_body, "proposal_sha256": proposal_sha256}, handle)

    receipt = hmc.arbitrate()

    mapping = receipt.to_mapping()
    final = mapping["final_command"]
    assert mapping["imminent_application_mode"] == "dormant"
    assert mapping["accepted_proposal_sha256"] == proposal_sha256
    assert mapping["requested_command"] == proposed
    assert mapping["disposition"] == "MODIFIED"
    assert mapping["reason_codes"] == ["modified_operating_mode_rule"]
    assert mapping["command_owner"] == "supervisor_modified"
    assert final["fan_speed_fraction"] == proposed["fan_speed_fraction"]
    assert final["cooling_removed_w"] == hold["cooling_removed_w"]
    assert final["oxygen_injection_mol_s"] == hold["oxygen_injection_mol_s"]
    assert final["scrubber_duty"] == proposed["scrubber_duty"]
    assert final["condenser_duty"] == proposed["condenser_duty"]
    assert mapping["preflight_result"]["classification"] == "FEASIBLE"


def test_low_battery_gauge_clamps_all_normal_proposal_increases_to_hold() -> None:
    scenario = _scenario_with_low_battery()
    hmc = HabitatManagementComputer.reset(scenario, _contract(), b"b" * 32)
    snapshot, verification = hmc.observe()
    handle = hmc.verify_snapshot(snapshot, verification)
    snapshot_mapping = snapshot.to_mapping()
    battery_sample = next(
        sample
        for sample in snapshot_mapping["operational_resource_gauges"]["samples"]
        if sample["descriptor_id"] == "battery_state_of_charge"
    )
    assert battery_sample["value"] < 0.20
    hold = command_from_achieved_state(scenario, hmc._state).command.to_mapping()
    proposed = json.loads(json.dumps(hold))
    proposed["fan_speed_fraction"] = min(1.0, hold["fan_speed_fraction"] + 0.1)
    proposed["scrubber_duty"] = min(1.0, hold["scrubber_duty"] + 0.1)
    proposed["condenser_duty"] = min(1.0, hold["condenser_duty"] + 0.1)
    for damper_id in proposed["damper_position_by_id"]:
        proposed["damper_position_by_id"][damper_id] = min(
            1.0, hold["damper_position_by_id"][damper_id] + 0.1
        )
    for zone_id in proposed["cooling_removed_w"]:
        proposed["cooling_removed_w"][zone_id] = (
            hold["cooling_removed_w"][zone_id] + 50.0
        )
        proposed["oxygen_injection_mol_s"][zone_id] = (
            hold["oxygen_injection_mol_s"][zone_id] + 0.0001
        )
    proposal_body = {
        "schema_version": "aeolus_habitat_v2_control_proposal_v1",
        "control_run_id": hmc.control_run_id,
        "authority_epoch": hmc.authority_epoch,
        "source_id": "baseline-v1",
        "source_type": "deterministic_baseline",
        "completed_observation_step": 0,
        "observation_snapshot_sha256": snapshot.snapshot_sha256,
        "requested_application_step": 0,
        "observable_topology_sha256": hmc.observable_topology_sha256,
        "proposed_command": proposed,
        "confidence": None,
    }
    proposal_sha256 = hashlib.sha256(_canonical_bytes(proposal_body)).hexdigest()
    hmc.propose({**proposal_body, "proposal_sha256": proposal_sha256}, handle)

    receipt = hmc.arbitrate()

    mapping = receipt.to_mapping()
    assert mapping["requested_command"] == proposed
    assert mapping["final_command"] == hold
    assert mapping["disposition"] == "MODIFIED"
    assert mapping["reason_codes"] == ["modified_resource_reserve"]
    assert mapping["command_owner"] == "supervisor_modified"
    assert mapping["emergency_reserve_use"] is False
    assert "battery_energy_wh" not in receipt.canonical_bytes.decode("utf-8")
    assert mapping["preflight_result"]["classification"] == "FEASIBLE"


def test_arbitration_receipt_public_construction_and_subclassing_are_disabled() -> None:
    with pytest.raises(TypeError, match="issued by"):
        ArbitrationReceipt()  # type: ignore[call-arg]

    with pytest.raises(TypeError, match="cannot subclass"):

        class ForgedArbitrationReceipt(ArbitrationReceipt):
            pass
