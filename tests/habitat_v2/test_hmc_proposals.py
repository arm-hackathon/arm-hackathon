from __future__ import annotations

import hashlib
import json
from copy import copy
from pathlib import Path

import pytest

from aeolus.habitat_v2.hmc import HabitatManagementComputer
from aeolus.habitat_v2.hmc_contract import load_hmc_contract
from aeolus.habitat_v2.physics import validate_external_command
from aeolus.habitat_v2.proposal import ProposalReceipt
from aeolus.habitat_v2.scenario import Scenario
from aeolus.habitat_v2.snapshot import (
    SnapshotVerificationError,
    _issue_receipt_control_event,
)


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


def test_no_proposal_issues_one_exact_receipt_without_plant_or_health_mutation() -> (
    None
):
    contract = _contract()
    hmc = HabitatManagementComputer.reset(_scenario(), contract, b"p" * 32)
    snapshot, verification = hmc.observe()
    handle = hmc.verify_snapshot(snapshot, verification)
    state_before = hmc._state
    sensor_memory_before = hmc._sensor_memory
    health_tracker_before = hmc._health_tracker
    measurement_before = hmc._last_operational_measurement
    previous_chain = hmc.current_control_chain_sha256

    receipt = hmc.propose(None, handle)

    assert type(receipt) is ProposalReceipt
    assert hmc.lifecycle_phase == "PROPOSED"
    assert hmc._state is state_before
    assert hmc._sensor_memory is sensor_memory_before
    assert hmc._health_tracker is health_tracker_before
    assert hmc._last_operational_measurement is measurement_before
    mapping = receipt.to_mapping()
    assert mapping == {
        "receipt_schema_sha256": contract.proposal_receipt_schema_sha256,
        "hmc_contract_sha256": contract.hmc_contract_sha256,
        "observable_topology_sha256": hmc.observable_topology_sha256,
        "control_run_id": hmc.control_run_id,
        "authority_epoch": hmc.authority_epoch,
        "sequence": 0,
        "observation_snapshot_sha256": snapshot.snapshot_sha256,
        "requested_application_step": 0,
        "attempt_class": "NONE",
        "attempt_evidence_sha256": contract.data["null_roots"]["proposal_receipt"][
            "sha256"
        ],
        "source_id": None,
        "source_type": None,
        "proposal": None,
        "proposal_sha256": None,
        "requested_command_sha256": None,
        "validation_outcome": "NO_PROPOSAL",
        "reason_code": "no_proposal",
        "event_ordinal": 1,
        "previous_control_chain_sha256": previous_chain,
        "proposal_receipt_sha256": receipt.proposal_receipt_sha256,
    }
    assert (
        receipt.proposal_receipt_sha256
        == hashlib.sha256(
            _canonical_bytes(_without(mapping, "proposal_receipt_sha256"))
        ).hexdigest()
    )
    assert len(hmc.control_events) == 2
    event = hmc.control_events[-1].to_mapping()
    assert event["event_ordinal"] == 1
    assert event["event_kind"] == "PROPOSAL"
    assert event["receipt"] == mapping
    assert event["receipt_sha256"] == receipt.proposal_receipt_sha256
    assert event["previous_control_chain_sha256"] == previous_chain
    assert event["control_chain_sha256"] == _chain_hash(
        previous_chain,
        1,
        "PROPOSAL",
        receipt.proposal_receipt_sha256,
    )
    assert hmc.current_control_chain_sha256 == event["control_chain_sha256"]


def test_opaque_malformed_proposal_emits_closed_rejection_without_retaining_input() -> (
    None
):
    contract = _contract()
    hmc = HabitatManagementComputer.reset(_scenario(), contract, b"m" * 32)
    snapshot, verification = hmc.observe()
    handle = hmc.verify_snapshot(snapshot, verification)
    state_before = hmc._state
    sensor_memory_before = hmc._sensor_memory
    health_tracker_before = hmc._health_tracker
    measurement_before = hmc._last_operational_measurement
    previous_chain = hmc.current_control_chain_sha256
    reason_code = "rejected_malformed"
    reason_bytes = reason_code.encode("utf-8")
    expected_evidence = hashlib.sha256(
        b"aeolus-habitat-v2-hmc-rejected-proposal-v1"
        + len(reason_bytes).to_bytes(8, "big")
        + reason_bytes
    ).hexdigest()

    receipt = hmc.propose(object(), handle)

    assert hmc.lifecycle_phase == "PROPOSED"
    assert hmc._state is state_before
    assert hmc._sensor_memory is sensor_memory_before
    assert hmc._health_tracker is health_tracker_before
    assert hmc._last_operational_measurement is measurement_before
    mapping = receipt.to_mapping()
    assert mapping["attempt_class"] == "REJECTED_INPUT"
    assert mapping["attempt_evidence_sha256"] == expected_evidence
    assert mapping["validation_outcome"] == "REJECTED"
    assert mapping["reason_code"] == reason_code
    assert mapping["source_id"] is None
    assert mapping["source_type"] is None
    assert mapping["proposal"] is None
    assert mapping["proposal_sha256"] is None
    assert mapping["requested_command_sha256"] is None
    assert mapping["observation_snapshot_sha256"] == snapshot.snapshot_sha256
    assert mapping["requested_application_step"] == 0
    assert mapping["event_ordinal"] == 1
    assert mapping["previous_control_chain_sha256"] == previous_chain
    assert "object at" not in receipt.canonical_bytes.decode("utf-8")
    assert (
        receipt.proposal_receipt_sha256
        == hashlib.sha256(
            _canonical_bytes(_without(mapping, "proposal_receipt_sha256"))
        ).hexdigest()
    )
    assert len(hmc.control_events) == 2
    assert hmc.control_events[-1].receipt == mapping


def test_canonicalisable_malformed_proposal_hashes_exact_input_without_retaining_it() -> (
    None
):
    hmc = HabitatManagementComputer.reset(_scenario(), _contract(), b"c" * 32)
    snapshot, verification = hmc.observe()
    handle = hmc.verify_snapshot(snapshot, verification)
    proposal: dict[str, object] = {}
    expected_evidence = hashlib.sha256(_canonical_bytes(proposal)).hexdigest()

    receipt = hmc.propose(proposal, handle)

    mapping = receipt.to_mapping()
    assert mapping["attempt_class"] == "REJECTED_INPUT"
    assert mapping["attempt_evidence_sha256"] == expected_evidence
    assert mapping["validation_outcome"] == "REJECTED"
    assert mapping["reason_code"] == "rejected_malformed"
    assert mapping["proposal"] is None
    assert mapping["proposal_sha256"] is None
    assert mapping["requested_command_sha256"] is None
    assert proposal == {}
    assert (
        receipt.proposal_receipt_sha256
        == hashlib.sha256(
            _canonical_bytes(_without(mapping, "proposal_receipt_sha256"))
        ).hexdigest()
    )


def test_valid_control_proposal_is_reparsed_and_bound_to_exact_command_identity() -> (
    None
):
    scenario = _scenario()
    hmc = HabitatManagementComputer.reset(scenario, _contract(), b"v" * 32)
    snapshot, verification = hmc.observe()
    handle = hmc.verify_snapshot(snapshot, verification)
    command = validate_external_command(
        scenario,
        scenario.data["timeline"][0]["command"],
    )
    body = {
        "schema_version": "aeolus_habitat_v2_control_proposal_v1",
        "control_run_id": hmc.control_run_id,
        "authority_epoch": hmc.authority_epoch,
        "source_id": "baseline-v1",
        "source_type": "deterministic_baseline",
        "completed_observation_step": 0,
        "observation_snapshot_sha256": snapshot.snapshot_sha256,
        "requested_application_step": 0,
        "observable_topology_sha256": hmc.observable_topology_sha256,
        "proposed_command": command.to_mapping(),
        "confidence": 0.5,
    }
    proposal_sha256 = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    proposal = {**body, "proposal_sha256": proposal_sha256}

    receipt = hmc.propose(proposal, handle)

    mapping = receipt.to_mapping()
    assert mapping["attempt_class"] == "CANONICAL_PROPOSAL"
    assert mapping["attempt_evidence_sha256"] == proposal_sha256
    assert mapping["source_id"] == "baseline-v1"
    assert mapping["source_type"] == "deterministic_baseline"
    assert mapping["proposal"] == proposal
    assert mapping["proposal_sha256"] == proposal_sha256
    assert mapping["requested_command_sha256"] == command.sha256
    assert mapping["validation_outcome"] == "VALID"
    assert mapping["reason_code"] == "valid"
    assert (
        receipt.proposal_receipt_sha256
        == hashlib.sha256(
            _canonical_bytes(_without(mapping, "proposal_receipt_sha256"))
        ).hexdigest()
    )


def test_control_event_rejects_minimal_arbitrary_self_digest_mapping() -> None:
    content = {"proposal_receipt_sha256": "22" * 32, "decision": "safe_hold"}
    receipt_sha256 = hashlib.sha256(_canonical_bytes(content)).hexdigest()

    with pytest.raises(ValueError, match="closed schema"):
        _issue_receipt_control_event(
            event_ordinal=2,
            event_kind="ARBITRATION",
            receipt_mapping={**content, "arbitration_receipt_sha256": receipt_sha256},
            receipt_sha256=receipt_sha256,
            previous_control_chain_sha256="11" * 32,
        )


def _issued_arbitration_mapping() -> tuple[dict[str, object], str, int, str]:
    hmc = HabitatManagementComputer.reset(_scenario(), _contract(), b"e" * 32)
    snapshot, verification = hmc.observe()
    handle = hmc.verify_snapshot(snapshot, verification)
    hmc.propose(None, handle)
    receipt = hmc.arbitrate().to_mapping()
    return (
        receipt,
        str(receipt["arbitration_receipt_sha256"]),
        int(receipt["event_ordinal"]),
        str(receipt["previous_control_chain_sha256"]),
    )


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_control_event_rejects_non_exact_receipt_fields(mutation: str) -> None:
    receipt, digest, ordinal, previous = _issued_arbitration_mapping()
    if mutation == "extra":
        receipt["extra"] = True
    else:
        receipt.pop("reason_codes")

    with pytest.raises(ValueError, match="closed schema"):
        _issue_receipt_control_event(
            event_ordinal=ordinal,
            event_kind="ARBITRATION",
            receipt_mapping=receipt,
            receipt_sha256=digest,
            previous_control_chain_sha256=previous,
        )


def test_control_event_rejects_wrong_receipt_schema_identity() -> None:
    receipt, digest, ordinal, previous = _issued_arbitration_mapping()
    receipt["receipt_schema_sha256"] = "00" * 32

    with pytest.raises(ValueError, match="schema identity"):
        _issue_receipt_control_event(
            event_ordinal=ordinal,
            event_kind="ARBITRATION",
            receipt_mapping=receipt,
            receipt_sha256=digest,
            previous_control_chain_sha256=previous,
        )


@pytest.mark.parametrize("mismatch", ["ordinal", "previous"])
def test_control_event_rejects_receipt_causal_mismatch(mismatch: str) -> None:
    receipt, digest, ordinal, previous = _issued_arbitration_mapping()

    with pytest.raises(ValueError, match="ordinal|previous chain"):
        _issue_receipt_control_event(
            event_ordinal=ordinal + (mismatch == "ordinal"),
            event_kind="ARBITRATION",
            receipt_mapping=receipt,
            receipt_sha256=digest,
            previous_control_chain_sha256=(
                "00" * 32 if mismatch == "previous" else previous
            ),
        )


def test_control_event_rejects_receipt_for_wrong_event_kind() -> None:
    receipt, digest, ordinal, previous = _issued_arbitration_mapping()

    with pytest.raises(ValueError, match="closed schema"):
        _issue_receipt_control_event(
            event_ordinal=ordinal,
            event_kind="PROPOSAL",
            receipt_mapping=receipt,
            receipt_sha256=digest,
            previous_control_chain_sha256=previous,
        )


def test_proposal_requires_exact_current_runtime_capability_and_fails_closed() -> None:
    scenario = _scenario()
    contract = _contract()
    hmc = HabitatManagementComputer.reset(scenario, contract, b"x" * 32)
    foreign_hmc = HabitatManagementComputer.reset(scenario, contract, b"x" * 32)
    snapshot, verification = hmc.observe()
    foreign_snapshot, foreign_verification = foreign_hmc.observe()
    handle = hmc.verify_snapshot(snapshot, verification)
    foreign_handle = foreign_hmc.verify_snapshot(foreign_snapshot, foreign_verification)
    chain_before = hmc.current_control_chain_sha256
    events_before = hmc.control_events

    with pytest.raises(TypeError):
        hmc.propose(None)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="not serialisable"):
        copy(handle)
    cloned_handle = object.__new__(type(handle))
    for field in (
        "owner_identity",
        "control_run_id",
        "authority_epoch",
        "snapshot_sha256",
        "verification_receipt_sha256",
        "cycle_id",
        "sequence",
        "snapshot_identity",
        "receipt_identity",
    ):
        object.__setattr__(cloned_handle, field, getattr(handle, field))
    for invalid in (None, object(), foreign_handle, cloned_handle):
        with pytest.raises(SnapshotVerificationError, match="exact current"):
            hmc.propose(None, invalid)  # type: ignore[arg-type]

    assert hmc.lifecycle_phase == "OBSERVED"
    assert hmc.current_control_chain_sha256 == chain_before
    assert hmc.control_events == events_before
    receipt = hmc.propose(None, handle)
    assert receipt.observation_snapshot_sha256 == snapshot.snapshot_sha256

    with pytest.raises(RuntimeError, match="PROPOSED"):
        hmc.propose(None, handle)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("owner_identity", 0),
        ("control_run_id", "00" * 32),
        ("authority_epoch", "00" * 32),
        ("cycle_id", "00" * 32),
        ("sequence", 1),
        ("snapshot_sha256", "00" * 32),
        ("verification_receipt_sha256", "00" * 32),
        ("snapshot_identity", 0),
        ("receipt_identity", 0),
    ),
)
def test_proposal_rejects_every_mismatched_capability_binding(
    field: str,
    replacement: object,
) -> None:
    hmc = HabitatManagementComputer.reset(_scenario(), _contract(), b"b" * 32)
    snapshot, verification = hmc.observe()
    handle = hmc.verify_snapshot(snapshot, verification)
    object.__setattr__(handle, field, replacement)

    with pytest.raises(SnapshotVerificationError, match="exact current"):
        hmc.propose(None, handle)

    assert hmc.lifecycle_phase == "OBSERVED"
    assert len(hmc.control_events) == 1


def test_proposal_receipt_public_construction_and_subclassing_are_disabled() -> None:
    with pytest.raises(TypeError, match="issued by"):
        ProposalReceipt()  # type: ignore[call-arg]

    with pytest.raises(TypeError, match="cannot subclass"):

        class ForgedProposalReceipt(ProposalReceipt):
            pass
