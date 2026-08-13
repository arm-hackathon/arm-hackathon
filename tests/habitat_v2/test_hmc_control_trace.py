from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

import aeolus.habitat_v2.hmc as hmc_module
from aeolus.habitat_v2.control_trace import (
    ControlTrace,
    ControlTraceError,
    ControlTraceIssuanceError,
    ControlTraceReplayResult,
    parse_control_trace,
    replay_control_trace,
)
from aeolus.habitat_v2.hmc import HabitatManagementComputer
from aeolus.habitat_v2.hmc_contract import canonical_json_bytes, load_hmc_contract
from aeolus.habitat_v2.scenario import Scenario


def _scenario() -> Scenario:
    path = Path(__file__).parents[2] / "scenarios" / "habitat_v2_actuator_feedback.json"
    return Scenario.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def _contract():
    path = Path(__file__).parents[2] / "contracts" / "habitat_v2_hmc_v1.json"
    return load_hmc_contract(path)


def _complete_hmc(reset_nonce: bytes = b"t" * 32) -> tuple[Scenario, object]:
    scenario = _scenario()
    hmc = HabitatManagementComputer.reset(scenario, _contract(), reset_nonce)
    for _ in range(int(scenario.data["steps"])):
        snapshot, verification = hmc.observe()
        hmc.verify_snapshot(snapshot, verification)
        hmc.propose(None)
        hmc.arbitrate()
        hmc.step()
    return scenario, hmc


def test_completed_control_trace_is_deterministic() -> None:
    scenario_a, hmc_a = _complete_hmc()
    scenario_b, hmc_b = _complete_hmc()

    trace_a = hmc_a.export_control_trace("a" * 40)
    trace_b = hmc_b.export_control_trace("a" * 40)

    assert scenario_a.scenario_sha256 == scenario_b.scenario_sha256
    assert trace_a.canonical_bytes == trace_b.canonical_bytes


def test_control_trace_strict_parse_round_trip() -> None:
    scenario, hmc = _complete_hmc()
    contract = _contract()
    trace = hmc.export_control_trace("a" * 40)

    parsed = parse_control_trace(trace.canonical_bytes, scenario, contract)

    assert type(parsed) is ControlTrace
    assert parsed.canonical_bytes == trace.canonical_bytes
    assert (
        parsed.header["control_trace_schema_sha256"]
        == contract.control_trace_schema_sha256
    )
    assert parsed.footer["terminal_status"] == "COMPLETED"
    assert len(parsed.events) == 1 + 4 * int(scenario.data["steps"])


def test_strict_parser_requires_the_closed_scenario_context() -> None:
    _, hmc = _complete_hmc()
    trace = hmc.export_control_trace("a" * 40)

    with pytest.raises(ControlTraceError, match="requires a Scenario"):
        parse_control_trace(trace.canonical_bytes, contract=_contract())


def test_premature_export_and_malformed_git_sha_are_rejected() -> None:
    scenario = _scenario()
    hmc = HabitatManagementComputer.reset(scenario, _contract(), b"p" * 32)

    with pytest.raises(ControlTraceIssuanceError, match="all configured steps"):
        hmc.export_control_trace("a" * 40)

    with pytest.raises(ControlTraceIssuanceError, match="40-character"):
        _complete_hmc()[1].export_control_trace("A" * 40)


def test_completed_steps_with_a_pending_proposal_cannot_export_as_completed() -> None:
    _, hmc = _complete_hmc()
    snapshot, verification = hmc.observe()
    hmc.verify_snapshot(snapshot, verification)
    hmc.propose(None)

    with pytest.raises(ControlTraceIssuanceError, match="stable completed lifecycle"):
        hmc.export_control_trace("a" * 40)


def _framed_digest(domain: str, payloads: list[bytes]) -> str:
    framed = bytearray(domain.encode("utf-8"))
    for payload in payloads:
        framed.extend(len(payload).to_bytes(8, "big"))
        framed.extend(payload)
    return hashlib.sha256(bytes(framed)).hexdigest()


def test_parser_rejects_a_rehashed_completed_trace_with_a_dangling_proposal() -> None:
    scenario, completed = _complete_hmc()
    contract = _contract()
    valid = _mapping(completed.export_control_trace("a" * 40))

    _, pending = _complete_hmc()
    snapshot, verification = pending.observe()
    pending.verify_snapshot(snapshot, verification)
    pending.propose(None)
    valid["events"].append(pending.control_events[-1].to_mapping())  # type: ignore[union-attr]
    footer = valid["footer"]  # type: ignore[assignment]
    footer["event_count"] = len(valid["events"])  # type: ignore[arg-type,index]
    footer["final_control_chain_sha256"] = valid["events"][-1][  # type: ignore[index]
        "control_chain_sha256"
    ]
    footer["control_trace_body_sha256"] = _framed_digest(  # type: ignore[index]
        str(contract.data["control_trace"]["domains"]["body"]),
        [canonical_json_bytes(event) for event in valid["events"]],  # type: ignore[union-attr]
    )
    _reself(footer, "control_trace_footer_sha256")  # type: ignore[arg-type]

    with pytest.raises(ControlTraceError, match="completed trace has a partial cycle"):
        parse_control_trace(_trace_bytes(valid), scenario, contract)


def test_control_trace_replays_without_the_source_hmc() -> None:
    scenario, hmc = _complete_hmc()
    trace = hmc.export_control_trace("a" * 40)

    replay = replay_control_trace(trace.canonical_bytes, scenario, _contract())

    assert replay.committed_step_count == scenario.data["steps"]
    assert replay.final_state.step == scenario.data["steps"]
    assert replay.final_state_sha256 == trace.footer["final_state_sha256"]


def test_terminal_failure_control_trace_is_parseable_and_replays_committed_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario()
    contract = _contract()
    hmc = HabitatManagementComputer.reset(scenario, contract, b"f" * 32)
    snapshot, verification = hmc.observe()
    hmc.verify_snapshot(snapshot, verification)
    hmc.propose(None)
    hmc.arbitrate()

    def fail_physics(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("hidden physics failure")

    monkeypatch.setattr(hmc_module, "advance_one_step_with_command", fail_physics)
    hmc.step()
    trace = hmc.export_control_trace("a" * 40)

    parsed = parse_control_trace(trace.canonical_bytes, scenario, contract)
    replay = replay_control_trace(trace.canonical_bytes, scenario, contract)

    assert parsed.footer["terminal_status"] == "TERMINAL_FAILURE"
    assert replay.committed_step_count == 0
    assert replay.final_state.step == 0


def test_terminal_trace_after_a_committed_cycle_replays_only_last_good_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario()
    contract = _contract()
    hmc = HabitatManagementComputer.reset(scenario, contract, b"g" * 32)
    for _ in range(2):
        snapshot, verification = hmc.observe()
        hmc.verify_snapshot(snapshot, verification)
        hmc.propose(None)
        hmc.arbitrate()
        hmc.step()

    snapshot, verification = hmc.observe()
    hmc.verify_snapshot(snapshot, verification)
    hmc.propose(None)
    hmc.arbitrate()

    def fail_physics(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("hidden later physics failure")

    monkeypatch.setattr(hmc_module, "advance_one_step_with_command", fail_physics)
    hmc.step()
    trace = hmc.export_control_trace("a" * 40)

    replay = replay_control_trace(trace.canonical_bytes, scenario, contract)

    assert trace.footer["terminal_status"] == "TERMINAL_FAILURE"
    assert replay.committed_step_count == 2
    assert replay.final_state.step == 2
    assert replay.final_state_sha256 == trace.footer["final_state_sha256"]


def _mapping(trace: ControlTrace) -> dict[str, object]:
    return deepcopy(trace.to_mapping())


def _reself(mapping: dict[str, object], field: str) -> None:
    content = dict(mapping)
    content.pop(field)
    mapping[field] = hashlib.sha256(canonical_json_bytes(content)).hexdigest()


def _trace_bytes(mapping: dict[str, object]) -> bytes:
    return canonical_json_bytes(mapping)


_RECEIPT_SELF_FIELD = {
    "SNAPSHOT_VERIFICATION": "snapshot_verification_receipt_sha256",
    "PROPOSAL": "proposal_receipt_sha256",
    "ARBITRATION": "arbitration_receipt_sha256",
    "STEP": "step_receipt_sha256",
    "TERMINAL": "terminal_failure_receipt_sha256",
}


def _replace_rehashed_receipt_references(
    value: object, remap: dict[str, str]
) -> object:
    if type(value) is dict:
        return {
            key: _replace_rehashed_receipt_references(item, remap)
            for key, item in value.items()
        }
    if type(value) is list:
        return [_replace_rehashed_receipt_references(item, remap) for item in value]
    if type(value) is str:
        return remap.get(value, value)
    return value


def _control_chain_digest(
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


def _rehash_trace_from_event(
    mapping: dict[str, object],
    start_index: int,
    contract: object,
) -> None:
    events = mapping["events"]
    assert type(events) is list
    header = mapping["header"]
    assert type(header) is dict
    previous_chain = (
        header["null_control_chain_sha256"]
        if start_index == 0
        else events[start_index - 1]["control_chain_sha256"]
    )
    assert type(previous_chain) is str
    remap: dict[str, str] = {}
    for event in events[start_index:]:
        assert type(event) is dict
        old_receipt_sha256 = event["receipt_sha256"]
        assert type(old_receipt_sha256) is str
        event["receipt"] = _replace_rehashed_receipt_references(event["receipt"], remap)
        receipt = event["receipt"]
        assert type(receipt) is dict
        event_kind = event["event_kind"]
        assert type(event_kind) is str
        event_ordinal = event["event_ordinal"]
        assert type(event_ordinal) is int
        event["previous_control_chain_sha256"] = previous_chain
        receipt["previous_control_chain_sha256"] = previous_chain
        self_field = _RECEIPT_SELF_FIELD[event_kind]
        _reself(receipt, self_field)
        receipt_sha256 = receipt[self_field]
        assert type(receipt_sha256) is str
        remap[old_receipt_sha256] = receipt_sha256
        event["receipt_sha256"] = receipt_sha256
        event["control_chain_sha256"] = _control_chain_digest(
            previous_chain,
            event_ordinal,
            event_kind,
            receipt_sha256,
        )
        previous_chain = event["control_chain_sha256"]

    footer = mapping["footer"]
    assert type(footer) is dict
    updated_footer = _replace_rehashed_receipt_references(footer, remap)
    assert type(updated_footer) is dict
    mapping["footer"] = updated_footer
    updated_footer["final_control_chain_sha256"] = previous_chain
    updated_footer["control_trace_body_sha256"] = _framed_digest(
        str(contract.data["control_trace"]["domains"]["body"]),
        [canonical_json_bytes(event) for event in events],
    )
    _reself(updated_footer, "control_trace_footer_sha256")


def test_parser_and_replay_reject_fully_rehashed_forged_arbitration_semantics() -> None:
    scenario, hmc = _complete_hmc()
    contract = _contract()
    mapping = _mapping(hmc.export_control_trace("a" * 40))
    events = mapping["events"]
    assert type(events) is list
    arbitration_index = max(
        index
        for index, event in enumerate(events)
        if event["event_kind"] == "ARBITRATION"
    )
    proposal = events[arbitration_index - 1]["receipt"]
    arbitration = events[arbitration_index]["receipt"]
    assert proposal["attempt_class"] == "NONE"
    assert proposal["validation_outcome"] == "NO_PROPOSAL"
    assert arbitration["disposition"] == "REJECTED"

    arbitration["disposition"] = "ACCEPTED"
    _rehash_trace_from_event(mapping, arbitration_index, contract)
    forged = _trace_bytes(mapping)

    with pytest.raises(ControlTraceError, match="arbitration semantics"):
        parse_control_trace(forged, scenario, contract)
    with pytest.raises(ControlTraceError, match="arbitration semantics"):
        replay_control_trace(forged, scenario, contract)


@pytest.mark.parametrize(
    "mutation",
    ("unknown", "missing", "noncanonical", "malformed_sha"),
)
def test_strict_parser_rejects_closed_shape_and_encoding_errors(mutation: str) -> None:
    scenario, hmc = _complete_hmc()
    contract = _contract()
    trace = hmc.export_control_trace("a" * 40)
    mapping = _mapping(trace)
    if mutation == "unknown":
        mapping["header"]["unexpected"] = True  # type: ignore[index]
        data = _trace_bytes(mapping)
    elif mutation == "missing":
        del mapping["footer"]["event_count"]  # type: ignore[index]
        data = _trace_bytes(mapping)
    elif mutation == "noncanonical":
        data = trace.canonical_bytes + b"\n"
    else:
        mapping["header"]["scenario_sha256"] = "A" * 64  # type: ignore[index]
        data = _trace_bytes(mapping)

    with pytest.raises(ControlTraceError):
        parse_control_trace(data, scenario, contract)


@pytest.mark.parametrize(
    "tamper",
    ("header", "embedded_receipt", "event_chain", "body", "footer"),
)
def test_strict_parser_rejects_tampering_at_each_trace_layer(tamper: str) -> None:
    scenario, hmc = _complete_hmc()
    contract = _contract()
    trace = hmc.export_control_trace("a" * 40)
    mapping = _mapping(trace)
    if tamper == "header":
        mapping["header"]["scenario_sha256"] = "0" * 64  # type: ignore[index]
        _reself(mapping["header"], "control_trace_header_sha256")  # type: ignore[arg-type,index]
        mapping["footer"]["control_trace_header_sha256"] = mapping["header"][  # type: ignore[index]
            "control_trace_header_sha256"
        ]
        _reself(mapping["footer"], "control_trace_footer_sha256")  # type: ignore[arg-type,index]
    elif tamper == "embedded_receipt":
        mapping["events"][0]["receipt"]["sequence"] = 99  # type: ignore[index]
    elif tamper == "event_chain":
        mapping["events"][1]["control_chain_sha256"] = "0" * 64  # type: ignore[index]
    elif tamper == "body":
        mapping["footer"]["control_trace_body_sha256"] = "0" * 64  # type: ignore[index]
        _reself(mapping["footer"], "control_trace_footer_sha256")  # type: ignore[arg-type,index]
    else:
        mapping["footer"]["event_count"] = 999  # type: ignore[index]
        _reself(mapping["footer"], "control_trace_footer_sha256")  # type: ignore[arg-type,index]

    with pytest.raises(ControlTraceError):
        parse_control_trace(_trace_bytes(mapping), scenario, contract)


@pytest.mark.parametrize("operation", ("reorder", "delete", "duplicate"))
def test_strict_parser_rejects_reordered_deleted_or_duplicated_events(
    operation: str,
) -> None:
    scenario, hmc = _complete_hmc()
    contract = _contract()
    trace = hmc.export_control_trace("a" * 40)
    mapping = _mapping(trace)
    events = mapping["events"]  # type: ignore[assignment]
    if operation == "reorder":
        events[1], events[2] = events[2], events[1]  # type: ignore[index]
    elif operation == "delete":
        del events[2]  # type: ignore[index]
    else:
        events.insert(2, deepcopy(events[2]))  # type: ignore[attr-defined,index]

    with pytest.raises(ControlTraceError):
        parse_control_trace(_trace_bytes(mapping), scenario, contract)


def test_replay_rejects_command_plant_receipt_and_final_state_tampering() -> None:
    scenario, hmc = _complete_hmc()
    contract = _contract()
    trace = hmc.export_control_trace("a" * 40)
    for tamper in ("command", "plant", "final_state"):
        mapping = _mapping(trace)
        if tamper == "command":
            mapping["events"][2]["receipt"]["final_command"][  # type: ignore[index]
                "scrubber_duty"
            ] = 0.123
        elif tamper == "plant":
            mapping["events"][3]["receipt"]["plant_receipt_digest"] = "0" * 64  # type: ignore[index]
        else:
            mapping["footer"]["final_state_sha256"] = "0" * 64  # type: ignore[index]
            _reself(mapping["footer"], "control_trace_footer_sha256")  # type: ignore[arg-type,index]
        with pytest.raises(ControlTraceError):
            replay_control_trace(_trace_bytes(mapping), scenario, contract)


def test_public_trace_construction_and_subclassing_are_disabled() -> None:
    with pytest.raises(TypeError, match="issued"):
        ControlTrace()  # type: ignore[call-arg]

    with pytest.raises(TypeError, match="cannot subclass"):

        class ForgedControlTrace(ControlTrace):
            pass

    with pytest.raises(TypeError, match="issued"):
        ControlTraceReplayResult()  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "field",
    (
        "control_trace_schema_sha256",
        "hmc_contract_sha256",
        "scenario_sha256",
        "control_run_id",
        "authority_epoch",
        "observable_topology_sha256",
    ),
)
def test_parser_rejects_wrong_header_identities(field: str) -> None:
    scenario, hmc = _complete_hmc()
    contract = _contract()
    mapping = _mapping(hmc.export_control_trace("a" * 40))
    mapping["header"][field] = "0" * 64  # type: ignore[index]
    _reself(mapping["header"], "control_trace_header_sha256")  # type: ignore[arg-type,index]
    mapping["footer"]["control_trace_header_sha256"] = mapping["header"][  # type: ignore[index]
        "control_trace_header_sha256"
    ]
    _reself(mapping["footer"], "control_trace_footer_sha256")  # type: ignore[arg-type,index]

    with pytest.raises(ControlTraceError):
        parse_control_trace(_trace_bytes(mapping), scenario, contract)


def test_parser_rejects_impossible_status_and_inconsistent_last_good_pointer() -> None:
    scenario, hmc = _complete_hmc()
    contract = _contract()
    trace = hmc.export_control_trace("a" * 40)
    for mutation in ("status", "last_good"):
        mapping = _mapping(trace)
        if mutation == "status":
            mapping["footer"]["terminal_status"] = "TERMINAL_FAILURE"  # type: ignore[index]
        else:
            mapping["footer"]["last_good_step_receipt_sha256"] = "0" * 64  # type: ignore[index]
        _reself(mapping["footer"], "control_trace_footer_sha256")  # type: ignore[arg-type,index]
        with pytest.raises(ControlTraceError):
            parse_control_trace(_trace_bytes(mapping), scenario, contract)
