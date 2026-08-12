from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pytest

from aeolus.habitat_v2.control_trace import StepReceipt
from aeolus.habitat_v2.hmc import HabitatManagementComputer
from aeolus.habitat_v2.hmc_contract import load_hmc_contract
from aeolus.habitat_v2.scenario import Scenario


def _scenario() -> Scenario:
    path = Path(__file__).parents[2] / "scenarios" / "habitat_v2_actuator_feedback.json"
    return Scenario.from_mapping(json.loads(path.read_text(encoding="utf-8")))


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


def test_no_proposal_full_cycle_commits_step_and_next_observation_atomically() -> None:
    scenario = _scenario()
    contract = _contract()
    hmc = HabitatManagementComputer.reset(scenario, contract, b"s" * 32)
    snapshot_zero, verification_zero = hmc.observe()
    hmc.verify_snapshot(snapshot_zero, verification_zero)
    proposal = hmc.propose(None)
    arbitration = hmc.arbitrate()
    state_before = hmc._state
    arbitration_chain = hmc.current_control_chain_sha256
    capability = hmc._step_capability
    assert capability is not None
    with pytest.raises(TypeError, match="not serialisable"):
        pickle.dumps(capability)

    step_receipt = hmc.step()

    assert type(step_receipt) is StepReceipt
    assert hmc.lifecycle_phase == "STEPPED"
    assert hmc._state is not state_before
    assert hmc._state.step == 1
    assert hmc._step_capability is None
    step_mapping = step_receipt.to_mapping()
    assert step_mapping == {
        "receipt_schema_sha256": contract.step_receipt_schema_sha256,
        "hmc_contract_sha256": contract.hmc_contract_sha256,
        "external_command_contract_sha256": contract.external_command_contract_sha256,
        "observable_topology_sha256": hmc.observable_topology_sha256,
        "control_run_id": hmc.control_run_id,
        "authority_epoch": hmc.authority_epoch,
        "observation_sequence": 0,
        "application_step": 0,
        "proposal_receipt_sha256": proposal.proposal_receipt_sha256,
        "arbitration_receipt_sha256": arbitration.arbitration_receipt_sha256,
        "final_command_sha256": arbitration.final_command_sha256,
        "returned_external_command_digest": arbitration.final_command_sha256,
        "plant_receipt_digest": step_receipt.plant_receipt_digest,
        "application_outcome": "APPLIED",
        "previous_step_receipt_digest": contract.data["null_roots"]["step_receipt"][
            "sha256"
        ],
        "event_ordinal": 3,
        "previous_control_chain_sha256": arbitration_chain,
        "step_receipt_sha256": step_receipt.step_receipt_sha256,
    }
    assert (
        step_receipt.step_receipt_sha256
        == hashlib.sha256(
            _canonical_bytes(_without(step_mapping, "step_receipt_sha256"))
        ).hexdigest()
    )
    assert len(step_receipt.plant_receipt_digest) == 64
    assert step_receipt.plant_receipt_digest == hmc._last_plant_receipt_digest

    assert [event.event_kind for event in hmc.control_events] == [
        "SNAPSHOT_VERIFICATION",
        "PROPOSAL",
        "ARBITRATION",
        "STEP",
        "SNAPSHOT_VERIFICATION",
    ]
    step_event = hmc.control_events[3]
    next_snapshot_event = hmc.control_events[4]
    assert step_event.receipt == step_mapping
    assert step_event.previous_control_chain_sha256 == arbitration_chain
    assert next_snapshot_event.previous_control_chain_sha256 == (
        step_event.control_chain_sha256
    )

    committed_state = hmc._state
    next_snapshot, next_verification = hmc.observe()

    assert hmc.lifecycle_phase == "OBSERVED"
    assert hmc._state is committed_state
    assert (next_snapshot, next_verification) == hmc.observe()
    snapshot_mapping = next_snapshot.to_mapping()
    assert snapshot_mapping["sequence"] == 1
    assert snapshot_mapping["completed_step"] == 1
    assert snapshot_mapping["completed_time_s"] == 60.0
    assert snapshot_mapping["completed_application_step"] == 0
    assert snapshot_mapping["completed_operating_mode"] == "occupied"
    assert snapshot_mapping["command_reference"] == {
        "source_kind": "authoritative_command_reference",
        "command_reference_kind": "COMPLETED_FINAL_COMMAND",
        "command": arbitration.final_command,
    }
    assert snapshot_mapping["completed_plant_receipt_digest"] == (
        step_receipt.plant_receipt_digest
    )
    assert snapshot_mapping["completed_step_receipt_digest"] == (
        step_receipt.step_receipt_sha256
    )
    verification_mapping = next_verification.to_mapping()
    assert verification_mapping["sequence"] == 1
    assert verification_mapping["completed_step"] == 1
    assert verification_mapping["snapshot_sha256"] == next_snapshot.snapshot_sha256
    assert verification_mapping["completed_plant_receipt_digest"] == (
        step_receipt.plant_receipt_digest
    )
    assert verification_mapping["completed_step_receipt_digest"] == (
        step_receipt.step_receipt_sha256
    )
    assert verification_mapping["previous_verification_receipt_digest"] == (
        verification_zero.snapshot_verification_receipt_sha256
    )
    assert verification_mapping["event_ordinal"] == 4
    assert verification_mapping["previous_control_chain_sha256"] == (
        step_event.control_chain_sha256
    )
    assert next_snapshot_event.receipt == verification_mapping
    serialised_snapshot = next_snapshot.canonical_bytes.decode("utf-8")
    for forbidden in (
        "active_faults",
        "fault_id",
        "realised_loads",
        "air_network",
        "species_accounting",
        "external_command_digest",
    ):
        assert forbidden not in serialised_snapshot


def test_second_cycle_uses_current_sequence_and_repeats_last_final_command() -> None:
    scenario = _scenario()
    hmc = HabitatManagementComputer.reset(scenario, _contract(), b"t" * 32)
    snapshot_zero, verification_zero = hmc.observe()
    hmc.verify_snapshot(snapshot_zero, verification_zero)
    first_command = snapshot_zero.to_mapping()["command_reference"]["command"]
    first_command["fan_speed_fraction"] = 1.0
    proposal_body = {
        "schema_version": "aeolus_habitat_v2_control_proposal_v1",
        "control_run_id": hmc.control_run_id,
        "authority_epoch": hmc.authority_epoch,
        "source_id": "baseline-v1",
        "source_type": "deterministic_baseline",
        "completed_observation_step": 0,
        "observation_snapshot_sha256": snapshot_zero.snapshot_sha256,
        "requested_application_step": 0,
        "observable_topology_sha256": hmc.observable_topology_sha256,
        "proposed_command": first_command,
        "confidence": None,
    }
    proposal_sha256 = hashlib.sha256(_canonical_bytes(proposal_body)).hexdigest()
    hmc.propose({**proposal_body, "proposal_sha256": proposal_sha256})
    arbitration_zero = hmc.arbitrate()
    step_zero = hmc.step()
    snapshot_one, verification_one = hmc.observe()
    hmc.verify_snapshot(snapshot_one, verification_one)
    achieved_fan = next(
        sample["value"]
        for sample in snapshot_one.to_mapping()["operational_feedback"]["samples"]
        if sample["descriptor_id"] == "fan_speed_fraction"
    )
    assert achieved_fan != arbitration_zero.final_command["fan_speed_fraction"]

    proposal_one = hmc.propose(None)
    arbitration_one = hmc.arbitrate()
    step_one = hmc.step()
    snapshot_two, verification_two = hmc.observe()

    assert proposal_one.sequence == 1
    assert proposal_one.requested_application_step == 1
    assert proposal_one.observation_snapshot_sha256 == snapshot_one.snapshot_sha256
    arbitration_mapping = arbitration_one.to_mapping()
    assert arbitration_mapping["sequence"] == 1
    assert arbitration_mapping["decision_step"] == 1
    assert arbitration_mapping["application_step"] == 1
    assert arbitration_mapping["final_command"] == arbitration_zero.final_command
    assert arbitration_mapping["final_command_sha256"] == (
        arbitration_zero.final_command_sha256
    )
    step_mapping = step_one.to_mapping()
    assert step_mapping["observation_sequence"] == 1
    assert step_mapping["application_step"] == 1
    assert step_mapping["previous_step_receipt_digest"] == step_zero.step_receipt_sha256
    assert snapshot_two.to_mapping()["sequence"] == 2
    assert snapshot_two.to_mapping()["completed_application_step"] == 1
    assert snapshot_two.to_mapping()["command_reference"]["command"] == (
        arbitration_one.final_command
    )
    assert verification_two.to_mapping()["previous_verification_receipt_digest"] == (
        verification_one.snapshot_verification_receipt_sha256
    )


def test_step_receipt_public_construction_and_subclassing_are_disabled() -> None:
    with pytest.raises(TypeError, match="issued by"):
        StepReceipt()  # type: ignore[call-arg]

    with pytest.raises(TypeError, match="cannot subclass"):

        class ForgedStepReceipt(StepReceipt):
            pass
