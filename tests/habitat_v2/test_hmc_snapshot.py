from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from aeolus.habitat_v2.hmc import HabitatManagementComputer
from aeolus.habitat_v2.hmc_contract import load_hmc_contract
from aeolus.habitat_v2.physics import command_from_achieved_state
from aeolus.habitat_v2.scenario import Scenario
from aeolus.habitat_v2.snapshot import (
    OperationalSnapshot,
    SnapshotVerificationError,
    SnapshotVerificationReceipt,
    VerifiedSnapshotHandle,
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


def _domain_hash(label: str, *parts: bytes) -> str:
    return hashlib.sha256(label.encode("utf-8") + b"".join(parts)).hexdigest()


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


def _all_mapping_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(str(key) for key in value)
        for nested in value.values():
            keys.update(_all_mapping_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(_all_mapping_keys(nested))
    return keys


def test_reset_observe_issues_one_cached_causally_bound_snapshot_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aeolus.habitat_v2.physics as physics_module

    scenario = _scenario()
    contract = _contract()

    def timeline_must_not_be_read(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("row-zero observe reached the scenario timeline")

    monkeypatch.setattr(physics_module, "_segment_for_step", timeline_must_not_be_read)

    hmc = HabitatManagementComputer.reset(scenario, contract, b"s" * 32)
    snapshot, receipt = hmc.observe()
    cached_snapshot, cached_receipt = hmc.observe()

    assert type(snapshot) is OperationalSnapshot
    assert type(receipt) is SnapshotVerificationReceipt
    assert cached_snapshot is snapshot
    assert cached_receipt is receipt
    assert hmc.lifecycle_phase == "OBSERVED"

    hold = command_from_achieved_state(hmc._scenario, hmc._state)
    snapshot_mapping = snapshot.to_mapping()
    assert snapshot_mapping == json.loads(snapshot.canonical_bytes)
    assert snapshot_mapping["schema_version"] == contract.snapshot_schema_version
    assert snapshot_mapping["control_run_id"] == hmc.control_run_id
    assert snapshot_mapping["authority_epoch"] == hmc.authority_epoch
    assert snapshot_mapping["sequence"] == 0
    assert snapshot_mapping["completed_step"] == 0
    assert snapshot_mapping["completed_time_s"] == 0.0
    assert snapshot_mapping["completed_application_step"] is None
    assert snapshot_mapping["completed_operating_mode"] is None
    assert snapshot_mapping["command_reference"] == {
        "source_kind": "authoritative_command_reference",
        "command_reference_kind": "INITIAL_ACHIEVED_STATE_HOLD",
        "command": hold.command.to_mapping(),
    }
    assert snapshot_mapping["derived_health"] == {
        "source_kind": "derived_health",
        "health_state": "NOMINAL",
    }
    assert snapshot_mapping["active_operational_alarms"] == {
        "source_kind": "alarm_receipt",
        "alarms": [],
    }
    assert (
        snapshot_mapping["completed_plant_receipt_digest"]
        == contract.data["null_roots"]["plant_receipt"]["sha256"]
    )
    assert (
        snapshot_mapping["completed_step_receipt_digest"]
        == contract.data["null_roots"]["step_receipt"]["sha256"]
    )
    assert (
        snapshot.snapshot_sha256
        == hashlib.sha256(
            _canonical_bytes(_without(snapshot_mapping, "snapshot_sha256"))
        ).hexdigest()
    )

    resource_samples = snapshot_mapping["operational_resource_gauges"]["samples"]
    feedback_samples = {
        sample["descriptor_id"]: sample
        for sample in snapshot_mapping["operational_feedback"]["samples"]
    }
    assert resource_samples == [
        feedback_samples[channel]
        for channel in (
            "battery_state_of_charge",
            "oxygen_store_fraction",
            "sorbent_remaining_fraction",
        )
    ]

    forbidden_keys = {
        "fault_receipt",
        "fault_id",
        "fault_type",
        "truth_telemetry",
        "primary_residual",
        "secondary_residual",
        "realised_loads",
        "random_seed",
        "actual_action",
        "actuator_receipt",
        "active_faults",
    }
    assert _all_mapping_keys(snapshot_mapping).isdisjoint(forbidden_keys)

    receipt_mapping = receipt.to_mapping()
    expected_issuer = _domain_hash(
        "aeolus-habitat-v2-hmc-issuer-v1",
        bytes.fromhex(hmc.control_run_id),
        bytes.fromhex(hmc.authority_epoch),
        bytes.fromhex(contract.hmc_contract_sha256),
    )
    expected_cycle = _domain_hash(
        "aeolus-habitat-v2-hmc-cycle-v1",
        bytes.fromhex(hmc.control_run_id),
        bytes.fromhex(hmc.authority_epoch),
        (0).to_bytes(8, "big"),
        bytes.fromhex(snapshot.snapshot_sha256),
    )
    assert receipt_mapping == json.loads(receipt.canonical_bytes)
    assert receipt_mapping["receipt_schema_sha256"] == (
        contract.snapshot_verification_receipt_schema_sha256
    )
    assert receipt_mapping["snapshot_verification_contract_sha256"] == (
        contract.snapshot_verification_contract_sha256
    )
    assert receipt_mapping["issuer_id"] == expected_issuer
    assert receipt_mapping["cycle_id"] == expected_cycle
    assert receipt_mapping["sequence"] == 0
    assert receipt_mapping["completed_step"] == 0
    assert receipt_mapping["completed_time_s"] == 0.0
    assert receipt_mapping["snapshot_sha256"] == snapshot.snapshot_sha256
    assert (
        receipt_mapping["completed_plant_receipt_digest"]
        == contract.data["null_roots"]["plant_receipt"]["sha256"]
    )
    assert (
        receipt_mapping["completed_step_receipt_digest"]
        == contract.data["null_roots"]["step_receipt"]["sha256"]
    )
    assert (
        receipt_mapping["previous_verification_receipt_digest"]
        == contract.data["null_roots"]["verification_receipt"]["sha256"]
    )
    assert receipt_mapping["event_ordinal"] == 0
    assert (
        receipt_mapping["previous_control_chain_sha256"]
        == contract.data["null_roots"]["control_chain"]["sha256"]
    )
    assert (
        receipt.snapshot_verification_receipt_sha256
        == hashlib.sha256(
            _canonical_bytes(
                _without(
                    receipt_mapping,
                    "snapshot_verification_receipt_sha256",
                )
            )
        ).hexdigest()
    )
    assert hmc.current_control_chain_sha256 == _chain_hash(
        receipt.previous_control_chain_sha256,
        receipt.event_ordinal,
        "SNAPSHOT_VERIFICATION",
        receipt.snapshot_verification_receipt_sha256,
    )
    assert len(hmc.control_events) == 1
    event = hmc.control_events[0].to_mapping()
    assert event["event_kind"] == "SNAPSHOT_VERIFICATION"
    assert event["receipt"] == receipt_mapping
    assert event["control_chain_sha256"] == hmc.current_control_chain_sha256


def test_row_zero_snapshot_serialises_health_reducer_output() -> None:
    source = _scenario()
    mapping = deepcopy(dict(source.data))
    mapping["initial_utility"]["battery_energy_wh"] = 3800.0
    scenario = Scenario.from_mapping(mapping)
    hmc = HabitatManagementComputer.reset(scenario, _contract(), b"h" * 32)

    snapshot, _ = hmc.observe()
    snapshot_mapping = snapshot.to_mapping()

    assert snapshot_mapping["derived_health"] == {
        "source_kind": "derived_health",
        "health_state": "DEGRADED",
    }
    assert snapshot_mapping["active_operational_alarms"] == {
        "source_kind": "alarm_receipt",
        "alarms": [
            {
                "alarm_id": "low_battery_gauge/battery_state_of_charge/warning",
                "family": "low_battery_gauge",
                "target": "battery_state_of_charge",
                "severity": "WARNING",
                "lifecycle": "RAISED",
            }
        ],
    }


def test_snapshot_and_verification_public_construction_is_disabled() -> None:
    with pytest.raises(TypeError, match="issued by"):
        OperationalSnapshot()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="issued by"):
        SnapshotVerificationReceipt()  # type: ignore[call-arg]

    with pytest.raises(TypeError, match="cannot subclass"):

        class ForgedSnapshot(OperationalSnapshot):
            pass

    with pytest.raises(TypeError, match="cannot subclass"):

        class ForgedReceipt(SnapshotVerificationReceipt):
            pass


def test_snapshot_runtime_verification_requires_exact_issued_objects_and_hmc() -> None:
    scenario = _scenario()
    contract = _contract()
    first_hmc = HabitatManagementComputer.reset(scenario, contract, b"v" * 32)
    replay_hmc = HabitatManagementComputer.reset(scenario, contract, b"v" * 32)
    snapshot, receipt = first_hmc.observe()
    replay_snapshot, replay_receipt = replay_hmc.observe()

    handle = first_hmc.verify_snapshot(snapshot, receipt)

    assert type(handle) is VerifiedSnapshotHandle
    assert handle.snapshot_sha256 == snapshot.snapshot_sha256
    assert handle.cycle_id == receipt.cycle_id
    assert handle.sequence == 0
    assert first_hmc.verify_snapshot(snapshot, receipt) is handle
    assert snapshot.canonical_bytes == replay_snapshot.canonical_bytes
    assert receipt.canonical_bytes == replay_receipt.canonical_bytes

    with pytest.raises(SnapshotVerificationError, match="exact issued"):
        first_hmc.verify_snapshot(replay_snapshot, receipt)
    with pytest.raises(SnapshotVerificationError, match="exact issued"):
        first_hmc.verify_snapshot(snapshot, replay_receipt)
    with pytest.raises(SnapshotVerificationError, match="exact issued"):
        replay_hmc.verify_snapshot(snapshot, receipt)

    with pytest.raises(TypeError, match="issued by"):
        VerifiedSnapshotHandle()  # type: ignore[call-arg]

    with pytest.raises(TypeError, match="cannot subclass"):

        class ForgedHandle(VerifiedSnapshotHandle):
            pass
