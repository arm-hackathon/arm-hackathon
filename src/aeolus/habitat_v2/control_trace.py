from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .hmc_contract import HMCContract, canonical_json_bytes
from .scenario import Scenario
from .state import PlantState
from .snapshot import ControlEvent, _FinalIssuedType

_STEP_RECEIPT_ISSUANCE_TOKEN = object()
_TERMINAL_RECEIPT_ISSUANCE_TOKEN = object()
_STEP_FIELDS = (
    "receipt_schema_sha256",
    "hmc_contract_sha256",
    "external_command_contract_sha256",
    "observable_topology_sha256",
    "control_run_id",
    "authority_epoch",
    "observation_sequence",
    "application_step",
    "proposal_receipt_sha256",
    "arbitration_receipt_sha256",
    "final_command_sha256",
    "returned_external_command_digest",
    "plant_receipt_digest",
    "application_outcome",
    "previous_step_receipt_digest",
    "event_ordinal",
    "previous_control_chain_sha256",
    "step_receipt_sha256",
)
_TERMINAL_FIELDS = (
    "receipt_schema_sha256",
    "terminal_contract_sha256",
    "hmc_contract_sha256",
    "observable_topology_sha256",
    "control_run_id",
    "authority_epoch",
    "sequence",
    "application_step",
    "lifecycle_phase",
    "last_good_snapshot_sha256",
    "last_good_verification_receipt_sha256",
    "last_good_step_receipt_sha256",
    "proposal_receipt_sha256",
    "arbitration_receipt_sha256",
    "final_command_sha256",
    "candidate_plant_receipt_digest",
    "plant_state_committed",
    "reason_code",
    "event_ordinal",
    "previous_control_chain_sha256",
    "terminal_failure_receipt_sha256",
)


class StepReceiptIssuanceError(ValueError):
    """Raised when a closed authoritative step receipt cannot be issued."""


class TerminalReceiptIssuanceError(ValueError):
    """Raised when a closed terminal receipt cannot be issued."""


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_object_id(value: object) -> bool:
    return (
        type(value) is str
        and len(value) in {40, 64}
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_nullable_sha256(value: object) -> bool:
    return value is None or _is_sha256(value)


def _is_non_negative_integer(value: object) -> bool:
    return type(value) is int and value >= 0


@dataclass(frozen=True, init=False, slots=True)
class StepReceipt(_FinalIssuedType):
    canonical_bytes: bytes
    step_receipt_sha256: str
    plant_receipt_digest: str
    final_command_sha256: str
    returned_external_command_digest: str
    application_step: int

    def __init__(
        self,
        *,
        mapping: Mapping[str, Any] | None = None,
        canonical_bytes: bytes | None = None,
        _token: object | None = None,
    ) -> None:
        if _token is not _STEP_RECEIPT_ISSUANCE_TOKEN:
            raise TypeError("StepReceipt must be issued by an HMC")
        if type(mapping) is not dict or type(canonical_bytes) is not bytes:
            raise StepReceiptIssuanceError("step receipt issuance data is malformed")
        for field in (
            "step_receipt_sha256",
            "plant_receipt_digest",
            "final_command_sha256",
            "returned_external_command_digest",
            "application_step",
        ):
            object.__setattr__(self, field, mapping[field])
        object.__setattr__(self, "canonical_bytes", canonical_bytes)

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)


@dataclass(frozen=True, init=False, slots=True)
class TerminalFailureReceipt(_FinalIssuedType):
    canonical_bytes: bytes
    terminal_failure_receipt_sha256: str
    reason_code: str
    application_step: int | None
    candidate_plant_receipt_digest: str | None

    def __init__(
        self,
        *,
        mapping: Mapping[str, Any] | None = None,
        canonical_bytes: bytes | None = None,
        _token: object | None = None,
    ) -> None:
        if _token is not _TERMINAL_RECEIPT_ISSUANCE_TOKEN:
            raise TypeError("TerminalFailureReceipt must be issued by an HMC")
        if type(mapping) is not dict or type(canonical_bytes) is not bytes:
            raise TerminalReceiptIssuanceError(
                "terminal receipt issuance data is malformed"
            )
        for field in (
            "terminal_failure_receipt_sha256",
            "reason_code",
            "application_step",
            "candidate_plant_receipt_digest",
        ):
            object.__setattr__(self, field, mapping[field])
        object.__setattr__(self, "canonical_bytes", canonical_bytes)

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)


def _validate_common_identities(
    content: Mapping[str, Any],
    *,
    contract: HMCContract,
) -> None:
    if type(contract) is not HMCContract:
        raise TypeError("receipt issuance requires the exact HMCContract type")
    for field in (
        "observable_topology_sha256",
        "control_run_id",
        "authority_epoch",
        "previous_control_chain_sha256",
    ):
        if not _is_sha256(content[field]):
            raise ValueError(f"{field} must be lowercase SHA-256 hex")
    if content["hmc_contract_sha256"] != contract.hmc_contract_sha256:
        raise ValueError("receipt HMC contract identity is inconsistent")
    if not _is_non_negative_integer(content["event_ordinal"]):
        raise ValueError("receipt event ordinal is malformed")


def _issue_step_receipt(
    content: Mapping[str, Any],
    *,
    contract: HMCContract,
) -> StepReceipt:
    self_field = "step_receipt_sha256"
    if type(content) is not dict or set(content) != set(_STEP_FIELDS) - {self_field}:
        raise StepReceiptIssuanceError("step receipt content is malformed")
    try:
        _validate_common_identities(content, contract=contract)
    except (TypeError, ValueError) as error:
        raise StepReceiptIssuanceError(str(error)) from None
    expected_identities = {
        "receipt_schema_sha256": contract.step_receipt_schema_sha256,
        "external_command_contract_sha256": contract.external_command_contract_sha256,
    }
    if any(content[field] != value for field, value in expected_identities.items()):
        raise StepReceiptIssuanceError("step receipt contract identity is inconsistent")
    for field in (
        "proposal_receipt_sha256",
        "arbitration_receipt_sha256",
        "final_command_sha256",
        "returned_external_command_digest",
        "plant_receipt_digest",
        "previous_step_receipt_digest",
    ):
        if not _is_sha256(content[field]):
            raise StepReceiptIssuanceError("step receipt digest is malformed")
    if (
        not _is_non_negative_integer(content["observation_sequence"])
        or not _is_non_negative_integer(content["application_step"])
        or content["observation_sequence"] != content["application_step"]
        or content["application_outcome"] != "APPLIED"
        or content["final_command_sha256"]
        != content["returned_external_command_digest"]
    ):
        raise StepReceiptIssuanceError("step receipt causal fields are inconsistent")
    mapping = dict(content)
    mapping[self_field] = hashlib.sha256(canonical_json_bytes(mapping)).hexdigest()
    canonical = canonical_json_bytes(mapping)
    return StepReceipt(
        mapping=mapping,
        canonical_bytes=canonical,
        _token=_STEP_RECEIPT_ISSUANCE_TOKEN,
    )


def _issue_terminal_failure_receipt(
    content: Mapping[str, Any],
    *,
    contract: HMCContract,
) -> TerminalFailureReceipt:
    self_field = "terminal_failure_receipt_sha256"
    if type(content) is not dict or set(content) != set(_TERMINAL_FIELDS) - {
        self_field
    }:
        raise TerminalReceiptIssuanceError("terminal receipt content is malformed")
    try:
        _validate_common_identities(content, contract=contract)
    except (TypeError, ValueError) as error:
        raise TerminalReceiptIssuanceError(str(error)) from None
    if (
        content["receipt_schema_sha256"] != contract.terminal_receipt_schema_sha256
        or content["terminal_contract_sha256"]
        != contract.terminal_receipt_schema_sha256
    ):
        raise TerminalReceiptIssuanceError(
            "terminal schema/contract identity alias is inconsistent"
        )
    if (
        not _is_non_negative_integer(content["sequence"])
        or (
            content["application_step"] is not None
            and not _is_non_negative_integer(content["application_step"])
        )
        or type(content["lifecycle_phase"]) is not str
        or content["lifecycle_phase"]
        not in {"RESET", "OBSERVED", "PROPOSED", "ARBITRATED", "STEPPED"}
        or content["plant_state_committed"] is not False
    ):
        raise TerminalReceiptIssuanceError("terminal lifecycle fields are malformed")
    for field in (
        "last_good_snapshot_sha256",
        "last_good_verification_receipt_sha256",
        "last_good_step_receipt_sha256",
    ):
        if not _is_sha256(content[field]):
            raise TerminalReceiptIssuanceError("terminal last-good digest is malformed")
    for field in (
        "proposal_receipt_sha256",
        "arbitration_receipt_sha256",
        "final_command_sha256",
        "candidate_plant_receipt_digest",
    ):
        if not _is_nullable_sha256(content[field]):
            raise TerminalReceiptIssuanceError("terminal nullable digest is malformed")
    if (
        content["reason_code"]
        not in contract.data["safety_policy"]["terminal_reason_codes"]
    ):
        raise TerminalReceiptIssuanceError("terminal reason code is unsupported")
    mapping = dict(content)
    mapping[self_field] = hashlib.sha256(canonical_json_bytes(mapping)).hexdigest()
    canonical = canonical_json_bytes(mapping)
    return TerminalFailureReceipt(
        mapping=mapping,
        canonical_bytes=canonical,
        _token=_TERMINAL_RECEIPT_ISSUANCE_TOKEN,
    )


_RECEIPT_NAME_BY_KIND = {
    "SNAPSHOT_VERIFICATION": "snapshot_verification",
    "PROPOSAL": "proposal",
    "ARBITRATION": "arbitration",
    "STEP": "step",
    "TERMINAL": "terminal",
}
_CONTRACT_IDENTITY_FIELDS = {
    "hmc_contract_sha256": "hmc_contract_sha256",
    "snapshot_schema_sha256": "snapshot_schema_sha256",
    "snapshot_verification_contract_sha256": ("snapshot_verification_contract_sha256"),
    "external_command_contract_sha256": "external_command_contract_sha256",
    "preflight_contract_sha256": "preflight_contract_sha256",
    "health_policy_sha256": "health_policy_sha256",
    "safety_policy_sha256": "safety_policy_sha256",
    "safe_action_catalogue_sha256": "safe_action_catalogue_sha256",
    "proposal_receipt_schema_sha256": "proposal_receipt_schema_sha256",
    "arbitration_receipt_schema_sha256": "arbitration_receipt_schema_sha256",
    "step_receipt_schema_sha256": "step_receipt_schema_sha256",
    "terminal_receipt_schema_sha256": "terminal_receipt_schema_sha256",
    "control_trace_schema_sha256": "control_trace_schema_sha256",
}


@dataclass(frozen=True, slots=True)
class ControlTraceValidationResult:
    canonical_bytes: bytes
    header: Mapping[str, Any]
    events: tuple[Mapping[str, Any], ...]
    footer: Mapping[str, Any]
    terminal_status: str
    final_sequence: int
    final_control_chain_sha256: str
    final_state_sha256: str


def _fail(message: str) -> None:
    raise ControlTraceValidationError(message)


def _self_digest(mapping: Mapping[str, Any], field: str) -> str:
    body = dict(mapping)
    body.pop(field, None)
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _domain_digest(domain: str, records: Sequence[Mapping[str, Any]]) -> str:
    payload = bytearray(domain.encode("utf-8"))
    for record in records:
        encoded = canonical_json_bytes(record)
        payload.extend(len(encoded).to_bytes(8, "big"))
        payload.extend(encoded)
    return hashlib.sha256(payload).hexdigest()


def _chain_digest(
    domain: str,
    previous: str,
    ordinal: int,
    kind: str,
    receipt_digest: str,
) -> str:
    kind_bytes = kind.encode("utf-8")
    try:
        payload = (
            domain.encode("utf-8")
            + bytes.fromhex(previous)
            + ordinal.to_bytes(8, "big")
            + len(kind_bytes).to_bytes(8, "big")
            + kind_bytes
            + bytes.fromhex(receipt_digest)
        )
    except (ValueError, OverflowError) as error:
        raise ControlTraceValidationError("control-chain input is malformed") from error
    return hashlib.sha256(payload).hexdigest()


def _exact_fields(value: object, fields: Sequence[str], label: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{label} must be a JSON object")
    mapping = value
    unknown = sorted(set(mapping) - set(fields))
    missing = sorted(set(fields) - set(mapping))
    if unknown or missing:
        _fail(f"invalid {label} fields; unknown={unknown}, missing={missing}")
    return mapping


def _header_identity_values(contract: HMCContract) -> dict[str, str]:
    return {
        field: str(getattr(contract, attribute))
        for field, attribute in _CONTRACT_IDENTITY_FIELDS.items()
    }


def create_control_trace_header(
    *,
    contract: HMCContract,
    hmc_implementation_git_sha: str,
    scenario_sha256: str,
    plant_run_id: str,
    observable_topology_sha256: str,
    control_run_id: str,
    authority_epoch: str,
    reset_nonce: bytes,
) -> dict[str, Any]:
    """Create the exact self-digested V1 header from HMC reset metadata."""
    if type(contract) is not HMCContract:
        raise TypeError("control trace requires the exact HMCContract type")
    if type(reset_nonce) is not bytes or len(reset_nonce) != 32:
        raise ControlTraceValidationError("reset nonce must be exactly 32 bytes")
    if not _is_git_object_id(hmc_implementation_git_sha):
        raise ControlTraceValidationError(
            "HMC implementation git SHA must be a lowercase full Git object ID"
        )
    for label, value in (
        ("scenario identity", scenario_sha256),
        ("plant run identity", plant_run_id),
        ("observable topology identity", observable_topology_sha256),
        ("control run identity", control_run_id),
        ("authority epoch", authority_epoch),
    ):
        if not _is_sha256(value):
            raise ControlTraceValidationError(f"{label} must be lowercase SHA-256 hex")
    header: dict[str, Any] = {
        "record_type": "CONTROL_TRACE_HEADER",
        **_header_identity_values(contract),
        "hmc_implementation_git_sha": hmc_implementation_git_sha,
        "scenario_sha256": scenario_sha256,
        "plant_run_id": plant_run_id,
        "observable_topology_sha256": observable_topology_sha256,
        "control_run_id": control_run_id,
        "authority_epoch": authority_epoch,
        "reset_nonce_hex": reset_nonce.hex(),
        "null_control_chain_sha256": str(
            contract.data["null_roots"]["control_chain"]["sha256"]
        ),
    }
    header["control_trace_header_sha256"] = _self_digest(
        header, "control_trace_header_sha256"
    )
    return header


def _event_mapping(value: object) -> dict[str, Any]:
    if type(value) is dict:
        return json.loads(canonical_json_bytes(value))
    to_mapping = getattr(value, "to_mapping", None)
    if not callable(to_mapping):
        raise TypeError("control events must be exact mappings or issued event objects")
    mapping = to_mapping()
    if type(mapping) is not dict:
        raise TypeError("issued control event returned a non-mapping")
    return json.loads(canonical_json_bytes(mapping))


def _final_state_content(
    *,
    terminal_status: str,
    final_sequence: int,
    last_snapshot: str,
    last_verification: str,
    last_step: str,
    last_plant: str,
    terminal_receipt: str,
    final_chain: str,
) -> dict[str, Any]:
    return {
        "terminal_status": terminal_status,
        "final_sequence": final_sequence,
        "last_good_snapshot_sha256": last_snapshot,
        "last_good_verification_receipt_sha256": last_verification,
        "last_good_step_receipt_sha256": last_step,
        "last_good_plant_receipt_digest": last_plant,
        "terminal_failure_receipt_sha256": terminal_receipt,
        "final_control_chain_sha256": final_chain,
    }


def serialize_control_trace(
    *,
    header: Mapping[str, Any],
    events: Sequence[object],
    contract: HMCContract,
    terminal_status: str,
) -> bytes:
    """Close HMC events into canonical JSONL and validate the produced artifact."""
    if type(header) is not dict:
        raise TypeError("control trace header must be an exact mapping")
    event_mappings = tuple(_event_mapping(event) for event in events)
    if not event_mappings:
        raise ControlTraceValidationError("control trace must contain events")
    roots = contract.data["null_roots"]
    snapshots = [
        event["receipt"]
        for event in event_mappings
        if event.get("event_kind") == "SNAPSHOT_VERIFICATION"
    ]
    if not snapshots:
        raise ControlTraceValidationError("control trace has no verified snapshot")
    last_snapshot_receipt = snapshots[-1]
    final_sequence = last_snapshot_receipt["sequence"]
    last_snapshot = last_snapshot_receipt["snapshot_sha256"]
    last_verification = last_snapshot_receipt["snapshot_verification_receipt_sha256"]
    last_step = last_snapshot_receipt["completed_step_receipt_digest"]
    last_plant = last_snapshot_receipt["completed_plant_receipt_digest"]
    terminal_receipt = str(roots["terminal_receipt"]["sha256"])
    if terminal_status == "TERMINAL_FAILURE":
        last_event_receipt = event_mappings[-1].get("receipt", {})
        terminal_receipt = last_event_receipt.get(
            "terminal_failure_receipt_sha256", terminal_receipt
        )
        final_sequence = last_event_receipt.get("sequence", final_sequence)
        last_snapshot = last_event_receipt.get(
            "last_good_snapshot_sha256", last_snapshot
        )
        last_verification = last_event_receipt.get(
            "last_good_verification_receipt_sha256", last_verification
        )
        last_step = last_event_receipt.get("last_good_step_receipt_sha256", last_step)
    final_chain = event_mappings[-1]["control_chain_sha256"]
    trace_contract = contract.data["control_trace"]
    body_digest = _domain_digest(str(trace_contract["domains"]["body"]), event_mappings)
    state = _final_state_content(
        terminal_status=terminal_status,
        final_sequence=final_sequence,
        last_snapshot=last_snapshot,
        last_verification=last_verification,
        last_step=last_step,
        last_plant=last_plant,
        terminal_receipt=terminal_receipt,
        final_chain=final_chain,
    )
    footer: dict[str, Any] = {
        "record_type": "CONTROL_TRACE_FOOTER",
        "control_trace_schema_sha256": contract.control_trace_schema_sha256,
        "control_trace_header_sha256": header.get("control_trace_header_sha256"),
        "control_run_id": header.get("control_run_id"),
        "authority_epoch": header.get("authority_epoch"),
        **state,
        "event_count": len(event_mappings),
        "control_trace_body_sha256": body_digest,
        "final_state_sha256": _domain_digest(
            str(trace_contract["domains"]["final_state"]), (state,)
        ),
    }
    footer["control_trace_footer_sha256"] = _self_digest(
        footer, "control_trace_footer_sha256"
    )
    rows = (dict(header), *event_mappings, footer)
    encoded = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    validate_control_trace_bytes(encoded, contract=contract)
    return encoded


def _validate_header(header: object, contract: HMCContract) -> dict[str, Any]:
    fields = tuple(contract.data["control_trace"]["header_fields"])
    result = _exact_fields(header, fields, "control trace header")
    if result["record_type"] != "CONTROL_TRACE_HEADER":
        _fail("first record is not the control trace header")
    if result["control_trace_header_sha256"] != _self_digest(
        result, "control_trace_header_sha256"
    ):
        _fail("control trace header self digest is inconsistent")
    for field, expected in _header_identity_values(contract).items():
        if result[field] != expected:
            _fail(f"header {field} is inconsistent with the HMC contract")
    roots = contract.data["null_roots"]
    if result["null_control_chain_sha256"] != roots["control_chain"]["sha256"]:
        _fail("header null control-chain identity is inconsistent")
    if not _is_git_object_id(result["hmc_implementation_git_sha"]):
        _fail("header hmc_implementation_git_sha is not a lowercase full Git object ID")
    for field in (
        "scenario_sha256",
        "plant_run_id",
        "observable_topology_sha256",
        "control_run_id",
        "authority_epoch",
    ):
        if not _is_sha256(result[field]):
            _fail(f"header {field} is not lowercase SHA-256 hex")
    nonce = result["reset_nonce_hex"]
    if type(nonce) is not str or len(nonce) != 64:
        _fail("header reset nonce is malformed")
    try:
        nonce_bytes = bytes.fromhex(nonce)
    except ValueError:
        _fail("header reset nonce is malformed")
    expected_run = hashlib.sha256(
        b"aeolus-habitat-v2-hmc-run-v1"
        + bytes.fromhex(result["scenario_sha256"])
        + bytes.fromhex(result["hmc_contract_sha256"])
        + bytes.fromhex(result["snapshot_schema_sha256"])
        + bytes.fromhex(result["observable_topology_sha256"])
        + nonce_bytes
    ).hexdigest()
    if result["control_run_id"] != expected_run:
        _fail("header control run identity is not derived from reset inputs")
    expected_epoch = hashlib.sha256(
        b"aeolus-habitat-v2-hmc-epoch-v1" + bytes.fromhex(expected_run) + nonce_bytes
    ).hexdigest()
    if result["authority_epoch"] != expected_epoch:
        _fail("header authority epoch is not derived from reset inputs")
    return result


def _validate_receipt(
    event: dict[str, Any],
    *,
    header: dict[str, Any],
    contract: HMCContract,
) -> dict[str, Any]:
    kind = event["event_kind"]
    name = _RECEIPT_NAME_BY_KIND.get(kind)
    if name is None:
        _fail("control event kind is unsupported")
    schema = contract.data["receipt_schemas"][name]
    receipt = _exact_fields(
        event["receipt"], tuple(schema["fields"]), f"{kind} receipt"
    )
    self_field = str(schema["self_digest_field"])
    if not _is_sha256(receipt[self_field]) or receipt[self_field] != _self_digest(
        receipt, self_field
    ):
        _fail(f"{kind} receipt self digest is inconsistent")
    if event["receipt_sha256"] != receipt[self_field]:
        _fail(f"{kind} event and receipt identities differ")
    if receipt["receipt_schema_sha256"] != getattr(
        contract, f"{name}_receipt_schema_sha256"
    ):
        _fail(f"{kind} receipt schema identity is inconsistent")
    for field in (
        "hmc_contract_sha256",
        "observable_topology_sha256",
        "control_run_id",
        "authority_epoch",
    ):
        if receipt[field] != header[field]:
            _fail(f"{kind} receipt {field} is inconsistent with the header")
    if receipt["event_ordinal"] != event["event_ordinal"]:
        _fail(f"{kind} receipt event ordinal is inconsistent")
    if (
        receipt["previous_control_chain_sha256"]
        != event["previous_control_chain_sha256"]
    ):
        _fail(f"{kind} receipt control-chain predecessor is inconsistent")
    return receipt


def _validate_event_envelope(
    event: object,
    *,
    ordinal: int,
    previous_chain: str,
    header: dict[str, Any],
    contract: HMCContract,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fields = tuple(contract.data["control_trace"]["event_fields"])
    result = _exact_fields(event, fields, f"control event {ordinal}")
    if result["record_type"] != "CONTROL_EVENT":
        _fail(f"record {ordinal + 1} is not a control event")
    if result["event_ordinal"] != ordinal:
        _fail("control event ordinals are not continuous")
    if result["previous_control_chain_sha256"] != previous_chain:
        _fail("control-chain predecessor link is inconsistent")
    receipt = _validate_receipt(result, header=header, contract=contract)
    expected_chain = _chain_digest(
        str(contract.data["control_trace"]["domains"]["chain"]),
        previous_chain,
        ordinal,
        result["event_kind"],
        result["receipt_sha256"],
    )
    if result["control_chain_sha256"] != expected_chain:
        _fail("control-chain digest is inconsistent")
    return result, receipt


def _validate_lifecycle(
    events: Sequence[dict[str, Any]],
    receipts: Sequence[dict[str, Any]],
    *,
    contract: HMCContract,
) -> None:
    kinds = [event["event_kind"] for event in events]
    if not kinds or kinds[0] != "SNAPSHOT_VERIFICATION":
        _fail("control trace must begin with snapshot verification")
    terminal_positions = [
        index for index, kind in enumerate(kinds) if kind == "TERMINAL"
    ]
    if terminal_positions and terminal_positions != [len(kinds) - 1]:
        _fail("terminal event must occur exactly once and last")
    expected = "PROPOSAL"
    for index, kind in enumerate(kinds[1:], start=1):
        if kind == "TERMINAL":
            break
        if kind != expected:
            _fail(f"lifecycle grammar is invalid at event {index}")
        expected = {
            "PROPOSAL": "ARBITRATION",
            "ARBITRATION": "STEP",
            "STEP": "SNAPSHOT_VERIFICATION",
            "SNAPSHOT_VERIFICATION": "PROPOSAL",
        }[kind]

    roots = contract.data["null_roots"]
    last_snapshot = receipts[0]
    last_proposal: dict[str, Any] | None = None
    last_arbitration: dict[str, Any] | None = None
    last_step: dict[str, Any] | None = None
    expected_sequence = 0
    for event, receipt in zip(events, receipts, strict=True):
        kind = event["event_kind"]
        if kind == "SNAPSHOT_VERIFICATION":
            if receipt["snapshot_schema_sha256"] != contract.snapshot_schema_sha256:
                _fail("snapshot receipt schema identity is inconsistent")
            if (
                receipt["snapshot_verification_contract_sha256"]
                != contract.snapshot_verification_contract_sha256
            ):
                _fail("snapshot verification contract identity is inconsistent")
            if (
                receipt["sequence"] != expected_sequence
                or receipt["completed_step"] != expected_sequence
            ):
                _fail("snapshot sequence continuity is inconsistent")
            if expected_sequence == 0:
                if (
                    receipt["completed_plant_receipt_digest"]
                    != roots["plant_receipt"]["sha256"]
                    or receipt["completed_step_receipt_digest"]
                    != roots["step_receipt"]["sha256"]
                    or receipt["previous_verification_receipt_digest"]
                    != roots["verification_receipt"]["sha256"]
                ):
                    _fail("initial snapshot null-root continuity is inconsistent")
            else:
                if last_step is None or (
                    receipt["completed_plant_receipt_digest"]
                    != last_step["plant_receipt_digest"]
                    or receipt["completed_step_receipt_digest"]
                    != last_step["step_receipt_sha256"]
                    or receipt["previous_verification_receipt_digest"]
                    != last_snapshot["snapshot_verification_receipt_sha256"]
                ):
                    _fail("snapshot/plant/step continuity is inconsistent")
            last_snapshot = receipt
        elif kind == "PROPOSAL":
            if (
                receipt["sequence"] != expected_sequence
                or receipt["requested_application_step"] != expected_sequence
                or receipt["observation_snapshot_sha256"]
                != last_snapshot["snapshot_sha256"]
            ):
                _fail("proposal causal reference is inconsistent")
            last_proposal = receipt
        elif kind == "ARBITRATION":
            if last_proposal is None or (
                receipt["sequence"] != expected_sequence
                or receipt["decision_step"] != expected_sequence
                or receipt["application_step"] != expected_sequence
                or receipt["observation_snapshot_sha256"]
                != last_snapshot["snapshot_sha256"]
                or receipt["proposal_receipt_sha256"]
                != last_proposal["proposal_receipt_sha256"]
            ):
                _fail("arbitration causal reference is inconsistent")
            for field, expected_value in (
                ("safety_policy_sha256", contract.safety_policy_sha256),
                ("safe_action_catalogue_sha256", contract.safe_action_catalogue_sha256),
                ("preflight_contract_sha256", contract.preflight_contract_sha256),
            ):
                if receipt[field] != expected_value:
                    _fail(f"arbitration {field} is inconsistent")
            last_arbitration = receipt
        elif kind == "STEP":
            if (
                last_proposal is None
                or last_arbitration is None
                or (
                    receipt["observation_sequence"] != expected_sequence
                    or receipt["application_step"] != expected_sequence
                    or receipt["proposal_receipt_sha256"]
                    != last_proposal["proposal_receipt_sha256"]
                    or receipt["arbitration_receipt_sha256"]
                    != last_arbitration["arbitration_receipt_sha256"]
                    or receipt["final_command_sha256"]
                    != last_arbitration["final_command_sha256"]
                    or receipt["returned_external_command_digest"]
                    != last_arbitration["final_command_sha256"]
                    or receipt["application_outcome"] != "APPLIED"
                )
            ):
                _fail("step causal reference is inconsistent")
            expected_previous_step = (
                roots["step_receipt"]["sha256"]
                if last_step is None
                else last_step["step_receipt_sha256"]
            )
            if receipt["previous_step_receipt_digest"] != expected_previous_step:
                _fail("step receipt predecessor is inconsistent")
            if (
                receipt["external_command_contract_sha256"]
                != contract.external_command_contract_sha256
            ):
                _fail("step external command contract identity is inconsistent")
            last_step = receipt
            expected_sequence += 1
        elif kind == "TERMINAL":
            phase_for_expected = {
                "PROPOSAL": "OBSERVED",
                "ARBITRATION": "PROPOSED",
                "STEP": "ARBITRATED",
                "SNAPSHOT_VERIFICATION": "STEPPED",
            }[expected]
            if receipt["lifecycle_phase"] != phase_for_expected:
                _fail("terminal lifecycle phase is inconsistent with its prefix")
            if (
                receipt["sequence"] != expected_sequence
                or receipt["plant_state_committed"] is not False
            ):
                _fail("terminal sequence or commit semantics are inconsistent")
            expected_step = (
                roots["step_receipt"]["sha256"]
                if last_step is None
                else last_step["step_receipt_sha256"]
            )
            if (
                receipt["last_good_snapshot_sha256"] != last_snapshot["snapshot_sha256"]
                or receipt["last_good_verification_receipt_sha256"]
                != last_snapshot["snapshot_verification_receipt_sha256"]
                or receipt["last_good_step_receipt_sha256"] != expected_step
            ):
                _fail("terminal last-good references are inconsistent")


def validate_control_trace_bytes(
    data: bytes,
    *,
    contract: HMCContract,
    expected_header: Mapping[str, Any] | None = None,
) -> ControlTraceValidationResult:
    """Parse and deterministically replay one complete canonical V1 JSONL artifact."""
    if type(contract) is not HMCContract:
        raise TypeError("control trace validation requires the exact HMCContract type")
    if type(data) is not bytes or not data or not data.endswith(b"\n"):
        raise ControlTraceValidationError(
            "control trace must be complete newline-terminated bytes"
        )
    raw_lines = data.split(b"\n")
    if raw_lines[-1] != b"" or any(not line for line in raw_lines[:-1]):
        _fail("control trace contains empty or trailing records")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw_lines[:-1]):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ControlTraceValidationError(
                f"control trace record {index} is not valid UTF-8 JSON"
            ) from error
        if type(row) is not dict or canonical_json_bytes(row) != line:
            _fail(f"control trace record {index} is not canonical JSON")
        rows.append(row)
    if len(rows) < 3:
        _fail("control trace is truncated")
    if rows[-1].get("record_type") != "CONTROL_TRACE_FOOTER":
        _fail("control trace is missing its final footer")
    if any(row.get("record_type") == "CONTROL_TRACE_FOOTER" for row in rows[:-1]):
        _fail("control trace contains a non-final or duplicate footer")
    if any(row.get("record_type") == "CONTROL_TRACE_HEADER" for row in rows[1:]):
        _fail("control trace contains a duplicate header")
    header = _validate_header(rows[0], contract)
    if expected_header is not None and dict(expected_header) != header:
        _fail("control trace header does not match expected run metadata")
    events: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    previous = header["null_control_chain_sha256"]
    for ordinal, row in enumerate(rows[1:-1]):
        event, receipt = _validate_event_envelope(
            row,
            ordinal=ordinal,
            previous_chain=previous,
            header=header,
            contract=contract,
        )
        events.append(event)
        receipts.append(receipt)
        previous = event["control_chain_sha256"]
    _validate_lifecycle(events, receipts, contract=contract)

    footer_fields = tuple(contract.data["control_trace"]["footer_fields"])
    footer = _exact_fields(rows[-1], footer_fields, "control trace footer")
    if footer["control_trace_footer_sha256"] != _self_digest(
        footer, "control_trace_footer_sha256"
    ):
        _fail("control trace footer self digest is inconsistent")
    for field in (
        "control_trace_schema_sha256",
        "control_trace_header_sha256",
        "control_run_id",
        "authority_epoch",
    ):
        if footer[field] != header[field]:
            _fail(f"footer {field} is inconsistent with the header")
    if footer["event_count"] != len(events):
        _fail("footer event count is inconsistent")
    body_digest = _domain_digest(
        str(contract.data["control_trace"]["domains"]["body"]), events
    )
    if footer["control_trace_body_sha256"] != body_digest:
        _fail("control trace body digest is inconsistent")
    if footer["final_control_chain_sha256"] != previous:
        _fail("footer final control-chain digest is inconsistent")
    terminal_status = footer["terminal_status"]
    has_terminal = events[-1]["event_kind"] == "TERMINAL"
    if terminal_status == "COMPLETED":
        if has_terminal or events[-1]["event_kind"] != "SNAPSHOT_VERIFICATION":
            _fail("COMPLETED trace must end at a verified snapshot")
        final_receipt = receipts[-1]
        expected_terminal = contract.data["null_roots"]["terminal_receipt"]["sha256"]
        expected_values = {
            "final_sequence": final_receipt["sequence"],
            "last_good_snapshot_sha256": final_receipt["snapshot_sha256"],
            "last_good_verification_receipt_sha256": final_receipt[
                "snapshot_verification_receipt_sha256"
            ],
            "last_good_step_receipt_sha256": final_receipt[
                "completed_step_receipt_digest"
            ],
            "last_good_plant_receipt_digest": final_receipt[
                "completed_plant_receipt_digest"
            ],
            "terminal_failure_receipt_sha256": expected_terminal,
        }
    elif terminal_status == "TERMINAL_FAILURE":
        if not has_terminal:
            _fail("TERMINAL_FAILURE trace must end in a terminal event")
        terminal = receipts[-1]
        last_snapshot_receipt = next(
            receipt
            for event, receipt in reversed(tuple(zip(events, receipts, strict=True)))
            if event["event_kind"] == "SNAPSHOT_VERIFICATION"
        )
        expected_values = {
            "final_sequence": terminal["sequence"],
            "last_good_snapshot_sha256": terminal["last_good_snapshot_sha256"],
            "last_good_verification_receipt_sha256": terminal[
                "last_good_verification_receipt_sha256"
            ],
            "last_good_step_receipt_sha256": terminal["last_good_step_receipt_sha256"],
            "last_good_plant_receipt_digest": last_snapshot_receipt[
                "completed_plant_receipt_digest"
            ],
            "terminal_failure_receipt_sha256": terminal[
                "terminal_failure_receipt_sha256"
            ],
        }
    else:
        _fail("footer terminal status is unsupported")
    for field, expected in expected_values.items():
        if footer[field] != expected:
            _fail(f"footer {field} is inconsistent with replayed state")
    state = _final_state_content(
        terminal_status=terminal_status,
        final_sequence=footer["final_sequence"],
        last_snapshot=footer["last_good_snapshot_sha256"],
        last_verification=footer["last_good_verification_receipt_sha256"],
        last_step=footer["last_good_step_receipt_sha256"],
        last_plant=footer["last_good_plant_receipt_digest"],
        terminal_receipt=footer["terminal_failure_receipt_sha256"],
        final_chain=footer["final_control_chain_sha256"],
    )
    final_state = _domain_digest(
        str(contract.data["control_trace"]["domains"]["final_state"]), (state,)
    )
    if footer["final_state_sha256"] != final_state:
        _fail("footer final-state digest is inconsistent")
    canonical = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    return ControlTraceValidationResult(
        canonical_bytes=canonical,
        header=json.loads(canonical_json_bytes(header)),
        events=tuple(json.loads(canonical_json_bytes(event)) for event in events),
        footer=json.loads(canonical_json_bytes(footer)),
        terminal_status=terminal_status,
        final_sequence=footer["final_sequence"],
        final_control_chain_sha256=previous,
        final_state_sha256=final_state,
    )


__all__ = (
    "ControlTraceValidationError",
    "ControlTraceValidationResult",
    "StepReceipt",
    "StepReceiptIssuanceError",
    "TerminalFailureReceipt",
    "TerminalReceiptIssuanceError",
    "create_control_trace_header",
    "serialize_control_trace",
    "validate_control_trace_bytes",
)


_CONTROL_TRACE_ISSUANCE_TOKEN = object()
_CONTROL_TRACE_REPLAY_TOKEN = object()
_TRACE_HEADER_RECORD_TYPE = "CONTROL_TRACE_HEADER"
_TRACE_FOOTER_RECORD_TYPE = "CONTROL_TRACE_FOOTER"
_NULLABLE_RECEIPT_DIGEST_FIELDS = {
    "accepted_proposal_sha256",
    "candidate_plant_receipt_digest",
    "final_command_sha256",
    "proposal_sha256",
    "requested_command_sha256",
    "arbitration_receipt_sha256",
    "proposal_receipt_sha256",
}
_RECEIPT_KIND_NAMES = {
    "SNAPSHOT_VERIFICATION": "snapshot_verification",
    "PROPOSAL": "proposal",
    "ARBITRATION": "arbitration",
    "STEP": "step",
    "TERMINAL": "terminal",
}
_RECEIPT_SELF_FIELDS = {
    "SNAPSHOT_VERIFICATION": "snapshot_verification_receipt_sha256",
    "PROPOSAL": "proposal_receipt_sha256",
    "ARBITRATION": "arbitration_receipt_sha256",
    "STEP": "step_receipt_sha256",
    "TERMINAL": "terminal_failure_receipt_sha256",
}


class ControlTraceError(ValueError):
    """Base class for whole-control-trace failures."""


class ControlTraceIssuanceError(ControlTraceError):
    """Raised when a whole-control-trace artifact cannot be issued."""


class ControlTraceParseError(ControlTraceError):
    """Raised when trace bytes are not an exact closed control trace."""


class ControlTraceReplayError(ControlTraceError):
    """Raised when a parsed trace does not replay to its final identity."""


ControlTraceValidationError = ControlTraceParseError
ControlTraceExportError = ControlTraceIssuanceError


def _trace_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ControlTraceParseError(f"{label} must be lowercase SHA-256 hex")
    return value


def _trace_git_sha(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ControlTraceIssuanceError(
            "hmc_implementation_git_sha must be exact lowercase 40-character hex"
        )
    return value


def _trace_non_negative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ControlTraceParseError(f"{label} must be a non-negative integer")
    return value


def _trace_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ControlTraceParseError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _trace_reject_constant(value: str) -> None:
    raise ControlTraceParseError(f"non-finite JSON constant is forbidden: {value}")


def _trace_decode(data: bytes) -> dict[str, Any]:
    if type(data) is not bytes:
        raise ControlTraceParseError("control trace must be bytes")
    try:
        decoded = data.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=_trace_no_duplicates,
            parse_constant=_trace_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ControlTraceError) as error:
        if isinstance(error, ControlTraceError):
            raise
        raise ControlTraceParseError(
            "control trace must be valid UTF-8 JSON"
        ) from error
    if type(value) is not dict:
        raise ControlTraceParseError("control trace root must be a JSON object")
    try:
        canonical = canonical_json_bytes(value)
    except Exception as error:  # noqa: BLE001 - closed parser boundary
        raise ControlTraceParseError(
            "control trace contains non-finite JSON"
        ) from error
    if data != canonical:
        raise ControlTraceParseError("control trace JSON is not canonical")
    return value


def _trace_exact_fields(
    value: object, expected: tuple[str, ...], *, label: str
) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ControlTraceParseError(f"{label} must be a JSON object")
    actual = set(value)
    expected_set = set(expected)
    if actual != expected_set:
        unknown = sorted(actual - expected_set)
        missing = sorted(expected_set - actual)
        raise ControlTraceParseError(
            f"invalid {label} fields; unknown={unknown}, missing={missing}"
        )
    return value


def _trace_self_digest(mapping: Mapping[str, Any], field: str) -> str:
    content = dict(mapping)
    content.pop(field, None)
    return hashlib.sha256(canonical_json_bytes(content)).hexdigest()


def _trace_framed_digest(domain: str, payloads: tuple[bytes, ...]) -> str:
    domain_bytes = domain.encode("utf-8")
    framed = bytearray(domain_bytes)
    for payload in payloads:
        framed.extend(len(payload).to_bytes(8, "big"))
        framed.extend(payload)
    return hashlib.sha256(bytes(framed)).hexdigest()


def _trace_state_mapping(state: PlantState) -> dict[str, Any]:
    """Return the complete deterministic JSON projection of a PlantState."""
    zones = {
        str(zone_id): {
            "co2_mol": float(zone.co2_mol),
            "o2_mol": float(zone.o2_mol),
            "water_vapor_mol": float(zone.water_vapor_mol),
            "inert_mol": float(zone.inert_mol),
            "temperature_k": float(zone.temperature_k),
        }
        for zone_id, zone in sorted(state.zones.items(), key=lambda item: str(item[0]))
    }
    utility = state.utility
    utility_mapping = {
        "co2_sorbent_remaining_mol": float(utility.co2_sorbent_remaining_mol),
        "captured_co2_mol": float(utility.captured_co2_mol),
        "condensed_water_mol": float(utility.condensed_water_mol),
        "oxygen_store_mol": float(utility.oxygen_store_mol),
        "battery_energy_wh": float(utility.battery_energy_wh),
        "actual_airflow_m3_s": {
            str(key): float(value)
            for key, value in sorted(
                utility.actual_airflow_m3_s.items(), key=lambda item: str(item[0])
            )
        },
        "actual_scrubber_duty": float(utility.actual_scrubber_duty),
        "actual_condenser_duty": float(utility.actual_condenser_duty),
        "external_heat_rejected_j": float(utility.external_heat_rejected_j),
        "external_heat_received_j": float(utility.external_heat_received_j),
        "actual_fan_speed_fraction": utility.actual_fan_speed_fraction,
        "actual_damper_position_by_id": {
            str(key): float(value)
            for key, value in sorted(
                utility.actual_damper_position_by_id.items(),
                key=lambda item: str(item[0]),
            )
        },
        "actual_cooling_removed_w": {
            str(key): float(value)
            for key, value in sorted(
                utility.actual_cooling_removed_w.items(), key=lambda item: str(item[0])
            )
        },
        "actual_oxygen_injection_mol_s": {
            str(key): float(value)
            for key, value in sorted(
                utility.actual_oxygen_injection_mol_s.items(),
                key=lambda item: str(item[0]),
            )
        },
        "effective_scrubber_capture_ability": float(
            utility.effective_scrubber_capture_ability
        ),
        "effective_condenser_removal_ability": float(
            utility.effective_condenser_removal_ability
        ),
        "effective_cooling_delivery_by_zone": {
            str(key): float(value)
            for key, value in sorted(
                utility.effective_cooling_delivery_by_zone.items(),
                key=lambda item: str(item[0]),
            )
        },
        "effective_oxygen_delivery_by_zone": {
            str(key): float(value)
            for key, value in sorted(
                utility.effective_oxygen_delivery_by_zone.items(),
                key=lambda item: str(item[0]),
            )
        },
        "last_operational_feedback": utility.last_operational_feedback,
    }
    return {"step": int(state.step), "zones": zones, "utility": utility_mapping}


def _trace_final_state_digest(state: PlantState, domain: str) -> str:
    return _trace_framed_digest(
        domain, (canonical_json_bytes(_trace_state_mapping(state)),)
    )


def _trace_expected_identities(
    scenario: Scenario, contract: HMCContract, reset_nonce: bytes
) -> dict[str, str]:
    from .telemetry import derive_observable_topology

    topology = derive_observable_topology(scenario).sha256
    control_run_id = hashlib.sha256(
        b"aeolus-habitat-v2-hmc-run-v1"
        + bytes.fromhex(scenario.scenario_sha256)
        + bytes.fromhex(contract.hmc_contract_sha256)
        + bytes.fromhex(contract.snapshot_schema_sha256)
        + bytes.fromhex(topology)
        + reset_nonce
    ).hexdigest()
    authority_epoch = hashlib.sha256(
        b"aeolus-habitat-v2-hmc-epoch-v1" + bytes.fromhex(control_run_id) + reset_nonce
    ).hexdigest()
    return {
        "scenario_sha256": scenario.scenario_sha256,
        "plant_run_id": scenario.run_id,
        "hmc_contract_sha256": contract.hmc_contract_sha256,
        "snapshot_schema_sha256": contract.snapshot_schema_sha256,
        "snapshot_verification_contract_sha256": contract.snapshot_verification_contract_sha256,
        "observable_topology_sha256": topology,
        "external_command_contract_sha256": contract.external_command_contract_sha256,
        "preflight_contract_sha256": contract.preflight_contract_sha256,
        "health_policy_sha256": contract.health_policy_sha256,
        "safety_policy_sha256": contract.safety_policy_sha256,
        "safe_action_catalogue_sha256": contract.safe_action_catalogue_sha256,
        "proposal_receipt_schema_sha256": contract.proposal_receipt_schema_sha256,
        "arbitration_receipt_schema_sha256": contract.arbitration_receipt_schema_sha256,
        "step_receipt_schema_sha256": contract.step_receipt_schema_sha256,
        "terminal_receipt_schema_sha256": contract.terminal_receipt_schema_sha256,
        "control_trace_schema_sha256": contract.control_trace_schema_sha256,
        "control_run_id": control_run_id,
        "authority_epoch": authority_epoch,
        "null_control_chain_sha256": str(
            contract.data["null_roots"]["control_chain"]["sha256"]
        ),
    }


def _trace_issue_artifact(mapping: dict[str, Any]) -> ControlTrace:
    canonical = canonical_json_bytes(mapping)
    return ControlTrace(
        canonical_bytes=canonical,
        _token=_CONTROL_TRACE_ISSUANCE_TOKEN,
    )


@dataclass(frozen=True, init=False, slots=True)
class ControlTrace(_FinalIssuedType):
    """Exact immutable canonical whole-control-trace artifact."""

    canonical_bytes: bytes

    def __init__(
        self,
        *,
        canonical_bytes: bytes | None = None,
        _token: object | None = None,
    ) -> None:
        if _token is not _CONTROL_TRACE_ISSUANCE_TOKEN:
            raise TypeError("ControlTrace must be issued by an HMC or parser")
        if type(canonical_bytes) is not bytes:
            raise ControlTraceIssuanceError("control trace bytes are malformed")
        object.__setattr__(self, "canonical_bytes", canonical_bytes)

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)

    @property
    def header(self) -> Mapping[str, Any]:
        return MappingProxyType(self.to_mapping()["header"])

    @property
    def events(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(MappingProxyType(event) for event in self.to_mapping()["events"])

    @property
    def footer(self) -> Mapping[str, Any]:
        return MappingProxyType(self.to_mapping()["footer"])


def _issue_control_trace(
    *,
    scenario: Scenario,
    contract: HMCContract,
    reset_nonce: bytes,
    state: PlantState,
    lifecycle_phase: str,
    events: tuple[ControlEvent, ...],
    hmc_implementation_git_sha: str,
) -> ControlTrace:
    if type(scenario) is not Scenario or type(contract) is not HMCContract:
        raise ControlTraceIssuanceError(
            "trace issuance requires exact scenario and contract"
        )
    if type(reset_nonce) is not bytes or len(reset_nonce) != 32:
        raise ControlTraceIssuanceError("trace reset nonce is malformed")
    if type(state) is not PlantState:
        raise ControlTraceIssuanceError("trace final state is malformed")
    if lifecycle_phase != "TERMINAL" and state.step < int(scenario.data["steps"]):
        raise ControlTraceIssuanceError(
            "control trace export requires all configured steps or TERMINAL lifecycle"
        )
    if lifecycle_phase not in {"STEPPED", "OBSERVED", "TERMINAL"}:
        raise ControlTraceIssuanceError(
            "completed control trace export requires a stable completed lifecycle"
        )
    git_sha = _trace_git_sha(hmc_implementation_git_sha)
    mappings: list[dict[str, Any]] = []
    for event in events:
        if type(event) is not ControlEvent:
            raise ControlTraceIssuanceError(
                "trace events must be exact issued ControlEvent values"
            )
        mappings.append(event.to_mapping())
    identities = _trace_expected_identities(scenario, contract, reset_nonce)
    header_content = {
        "record_type": _TRACE_HEADER_RECORD_TYPE,
        **identities,
        "hmc_implementation_git_sha": git_sha,
        "reset_nonce_hex": reset_nonce.hex(),
        "control_trace_header_sha256": "",
    }
    header_content["control_trace_header_sha256"] = _trace_self_digest(
        header_content, "control_trace_header_sha256"
    )
    terminal = bool(mappings and mappings[-1]["event_kind"] == "TERMINAL")
    if not terminal and state.step < int(scenario.data["steps"]):
        raise ControlTraceIssuanceError(
            "completed trace has not committed all configured steps"
        )
    latest_snapshot = None
    latest_verification = None
    latest_step = None
    latest_plant = None
    terminal_receipt_sha = str(
        contract.data["null_roots"]["terminal_receipt"]["sha256"]
    )
    for event in mappings:
        if event["event_kind"] == "SNAPSHOT_VERIFICATION":
            latest_snapshot = event["receipt"]["snapshot_sha256"]
            latest_verification = event["receipt"][
                "snapshot_verification_receipt_sha256"
            ]
        elif event["event_kind"] == "STEP":
            latest_step = event["receipt"]["step_receipt_sha256"]
            latest_plant = event["receipt"]["plant_receipt_digest"]
        elif event["event_kind"] == "TERMINAL":
            terminal_receipt_sha = event["receipt"]["terminal_failure_receipt_sha256"]
    null_roots = contract.data["null_roots"]
    latest_snapshot = latest_snapshot or str(null_roots["snapshot"]["sha256"])
    latest_verification = latest_verification or str(
        null_roots["verification_receipt"]["sha256"]
    )
    latest_step = latest_step or str(null_roots["step_receipt"]["sha256"])
    latest_plant = latest_plant or str(null_roots["plant_receipt"]["sha256"])
    event_bytes = tuple(canonical_json_bytes(event) for event in mappings)
    body_domain = str(contract.data["control_trace"]["domains"]["body"])
    final_domain = str(contract.data["control_trace"]["domains"]["final_state"])
    footer_content = {
        "record_type": _TRACE_FOOTER_RECORD_TYPE,
        "control_trace_schema_sha256": contract.control_trace_schema_sha256,
        "control_trace_header_sha256": header_content["control_trace_header_sha256"],
        "control_run_id": identities["control_run_id"],
        "authority_epoch": identities["authority_epoch"],
        "terminal_status": "TERMINAL_FAILURE" if terminal else "COMPLETED",
        "final_sequence": int(state.step),
        "last_good_snapshot_sha256": latest_snapshot,
        "last_good_verification_receipt_sha256": latest_verification,
        "last_good_step_receipt_sha256": latest_step,
        "last_good_plant_receipt_digest": latest_plant,
        "terminal_failure_receipt_sha256": terminal_receipt_sha,
        "final_control_chain_sha256": (
            mappings[-1]["control_chain_sha256"]
            if mappings
            else identities["null_control_chain_sha256"]
        ),
        "event_count": len(mappings),
        "control_trace_body_sha256": _trace_framed_digest(body_domain, event_bytes),
        "final_state_sha256": _trace_final_state_digest(state, final_domain),
        "control_trace_footer_sha256": "",
    }
    footer_content["control_trace_footer_sha256"] = _trace_self_digest(
        footer_content, "control_trace_footer_sha256"
    )
    return _trace_issue_artifact(
        {"header": header_content, "events": mappings, "footer": footer_content}
    )


def _trace_validate_context(
    header: dict[str, Any], scenario: Scenario | None, contract: HMCContract | None
) -> tuple[Scenario | None, HMCContract | None, dict[str, str] | None]:
    if contract is not None and type(contract) is not HMCContract:
        raise ControlTraceParseError("parser requires the exact HMCContract type")
    if scenario is not None and type(scenario) is not Scenario:
        raise ControlTraceParseError("parser requires the exact Scenario type")
    if contract is not None:
        try:
            contract_mapping = json.loads(contract.canonical_bytes)
            parsed_contract = HMCContract.from_mapping(contract_mapping)
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
            raise ControlTraceParseError(
                "contract identity is not a closed contract"
            ) from error
        if (
            parsed_contract.canonical_bytes != contract.canonical_bytes
            or parsed_contract.hmc_contract_sha256 != contract.hmc_contract_sha256
        ):
            raise ControlTraceParseError("contract identity is not a closed contract")
    if scenario is not None:
        try:
            parsed_scenario = Scenario.from_mapping(scenario.data)
        except (TypeError, ValueError) as error:
            raise ControlTraceParseError(
                "scenario identity is not a closed scenario"
            ) from error
        if any(
            getattr(parsed_scenario, field) != getattr(scenario, field)
            for field in (
                "canonical_bytes",
                "scenario_sha256",
                "scenario_schema_version",
                "trace_schema_version",
                "equation_contract_revision",
                "actuator_feedback_contract_revision",
                "run_id",
            )
        ):
            raise ControlTraceParseError("scenario identity is not a closed scenario")
    identities = None
    if scenario is not None and contract is not None:
        try:
            scenario.validate_contract_identities()
            identities = _trace_expected_identities(
                scenario, contract, bytes.fromhex(header["reset_nonce_hex"])
            )
        except (AttributeError, ValueError, TypeError) as error:
            raise ControlTraceParseError(
                "scenario or contract identities are invalid"
            ) from error
        for field, expected in identities.items():
            if header[field] != expected:
                raise ControlTraceParseError(f"header {field} identity is inconsistent")
    return scenario, contract, identities


def _trace_validate_header(
    header: object, scenario: Scenario | None, contract: HMCContract | None
) -> tuple[dict[str, Any], Scenario | None, HMCContract | None]:
    if contract is None:
        raise ControlTraceParseError("strict trace parsing requires an HMCContract")
    expected = tuple(contract.data["control_trace"]["header_fields"])
    parsed = _trace_exact_fields(header, expected, label="trace header")
    if parsed["record_type"] != _TRACE_HEADER_RECORD_TYPE:
        raise ControlTraceParseError("trace header record_type is invalid")
    for field in expected:
        if field.endswith("_sha256"):
            _trace_sha256(parsed[field], label=f"header {field}")
    _trace_git_sha_for_parse(parsed["hmc_implementation_git_sha"])
    reset_nonce_hex = parsed["reset_nonce_hex"]
    if (
        type(reset_nonce_hex) is not str
        or len(reset_nonce_hex) != 64
        or reset_nonce_hex != reset_nonce_hex.lower()
        or any(character not in "0123456789abcdef" for character in reset_nonce_hex)
    ):
        raise ControlTraceParseError("header reset_nonce_hex is malformed")
    if parsed["control_trace_header_sha256"] != _trace_self_digest(
        parsed, "control_trace_header_sha256"
    ):
        raise ControlTraceParseError("header self digest is inconsistent")
    if parsed["control_trace_schema_sha256"] != contract.control_trace_schema_sha256:
        raise ControlTraceParseError(
            "header control-trace schema identity is inconsistent"
        )
    for field in (
        "hmc_contract_sha256",
        "snapshot_schema_sha256",
        "snapshot_verification_contract_sha256",
        "external_command_contract_sha256",
        "preflight_contract_sha256",
        "health_policy_sha256",
        "safety_policy_sha256",
        "safe_action_catalogue_sha256",
        "proposal_receipt_schema_sha256",
        "arbitration_receipt_schema_sha256",
        "step_receipt_schema_sha256",
        "terminal_receipt_schema_sha256",
        "control_run_id",
        "authority_epoch",
        "null_control_chain_sha256",
    ):
        if parsed[field] != getattr(contract, field, parsed[field]):
            if field == "null_control_chain_sha256":
                expected_value = str(
                    contract.data["null_roots"]["control_chain"]["sha256"]
                )
            else:
                expected_value = getattr(contract, field, None)
            if parsed[field] != expected_value:
                # Run/epoch/topology are checked against Scenario below; contract-only
                # fields must always match the closed contract.
                if field not in {"control_run_id", "authority_epoch"}:
                    raise ControlTraceParseError(
                        f"header {field} identity is inconsistent"
                    )
    _trace_validate_context(parsed, scenario, contract)
    return parsed, scenario, contract


def _trace_git_sha_for_parse(value: object) -> None:
    if (
        type(value) is not str
        or len(value) != 40
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ControlTraceParseError("hmc_implementation_git_sha is malformed")


def _trace_validate_receipt(
    event_kind: str,
    receipt: object,
    *,
    contract: HMCContract,
    scenario: Scenario | None,
    run_id: str,
    authority_epoch: str,
    topology_sha256: str | None,
) -> dict[str, Any]:
    name = _RECEIPT_KIND_NAMES[event_kind]
    schema = contract.data["receipt_schemas"][name]
    parsed = _trace_exact_fields(
        receipt, tuple(schema["fields"]), label=f"{name} receipt"
    )
    self_field = str(schema["self_digest_field"])
    if parsed[self_field] != _trace_self_digest(parsed, self_field):
        raise ControlTraceParseError(f"{name} receipt self digest is inconsistent")
    expected_schema = getattr(
        contract,
        {
            "snapshot_verification": "snapshot_verification_receipt_schema_sha256",
            "proposal": "proposal_receipt_schema_sha256",
            "arbitration": "arbitration_receipt_schema_sha256",
            "step": "step_receipt_schema_sha256",
            "terminal": "terminal_receipt_schema_sha256",
        }[name],
    )
    if parsed["receipt_schema_sha256"] != expected_schema:
        raise ControlTraceParseError(f"{name} receipt schema identity is inconsistent")
    for field in parsed:
        if field.endswith("_sha256"):
            if parsed[field] is None and field in _NULLABLE_RECEIPT_DIGEST_FIELDS:
                continue
            _trace_sha256(parsed[field], label=f"{name} receipt {field}")
    if (
        parsed["control_run_id"] != run_id
        or parsed["authority_epoch"] != authority_epoch
    ):
        raise ControlTraceParseError(
            f"{name} receipt run or epoch identity is inconsistent"
        )
    if (
        topology_sha256 is not None
        and parsed["observable_topology_sha256"] != topology_sha256
    ):
        raise ControlTraceParseError(
            f"{name} receipt topology identity is inconsistent"
        )
    if parsed["hmc_contract_sha256"] != contract.hmc_contract_sha256:
        raise ControlTraceParseError(
            f"{name} receipt HMC contract identity is inconsistent"
        )
    _trace_non_negative_int(
        parsed["event_ordinal"], label=f"{name} receipt event_ordinal"
    )
    if name == "snapshot_verification":
        if (
            parsed["snapshot_verification_contract_sha256"]
            != contract.snapshot_verification_contract_sha256
        ):
            raise ControlTraceParseError(
                "snapshot verification contract identity is inconsistent"
            )
        if parsed["snapshot_schema_sha256"] != contract.snapshot_schema_sha256:
            raise ControlTraceParseError("snapshot schema identity is inconsistent")
    elif name == "proposal":
        if parsed["attempt_class"] not in {
            "NONE",
            "CANONICAL_PROPOSAL",
            "REJECTED_INPUT",
        }:
            raise ControlTraceParseError("proposal attempt class is invalid")
        if parsed["validation_outcome"] not in {"NO_PROPOSAL", "VALID", "REJECTED"}:
            raise ControlTraceParseError("proposal validation outcome is invalid")
    elif name == "arbitration":
        if parsed["disposition"] not in {"ACCEPTED", "MODIFIED", "REJECTED"}:
            raise ControlTraceParseError("arbitration disposition is invalid")
        if type(parsed["requested_command"]) not in {dict, type(None)}:
            raise ControlTraceParseError("arbitration requested command is malformed")
        if type(parsed["final_command"]) is not dict:
            raise ControlTraceParseError("arbitration final command is malformed")
        if scenario is not None:
            from .physics import validate_external_command

            try:
                command = validate_external_command(scenario, parsed["final_command"])
            except Exception as error:  # noqa: BLE001 - parser boundary
                raise ControlTraceParseError(
                    "arbitration final command is invalid"
                ) from error
            if command.sha256 != parsed["final_command_sha256"]:
                raise ControlTraceParseError(
                    "arbitration final command digest is inconsistent"
                )
    elif name == "step":
        if parsed["application_outcome"] != "APPLIED":
            raise ControlTraceParseError("step application outcome is impossible")
        if parsed["final_command_sha256"] != parsed["returned_external_command_digest"]:
            raise ControlTraceParseError("step command digests are inconsistent")
        if parsed["observation_sequence"] != parsed["application_step"]:
            raise ControlTraceParseError("step sequence and application step differ")
    else:
        if (
            parsed["terminal_contract_sha256"]
            != contract.terminal_receipt_schema_sha256
        ):
            raise ControlTraceParseError("terminal contract identity is inconsistent")
        if parsed["plant_state_committed"] is not False:
            raise ControlTraceParseError(
                "terminal receipt commits impossible plant state"
            )
        if parsed["lifecycle_phase"] not in {"RESET", "PROPOSED", "ARBITRATED"}:
            raise ControlTraceParseError("terminal lifecycle phase is impossible")
        if (
            parsed["reason_code"]
            not in contract.data["safety_policy"]["terminal_reason_codes"]
        ):
            raise ControlTraceParseError("terminal reason code is unsupported")
    return parsed


def _trace_validate_event(
    event: object,
    *,
    ordinal: int,
    previous_chain: str,
    contract: HMCContract,
    scenario: Scenario | None,
    run_id: str,
    authority_epoch: str,
    topology_sha256: str | None,
) -> dict[str, Any]:
    expected_fields = tuple(contract.data["control_trace"]["event_fields"])
    parsed = _trace_exact_fields(event, expected_fields, label="control event")
    if parsed["record_type"] != "CONTROL_EVENT":
        raise ControlTraceParseError("control event record_type is invalid")
    if parsed["event_ordinal"] != ordinal:
        raise ControlTraceParseError("control event ordinals are not contiguous")
    if parsed["event_kind"] not in _RECEIPT_KIND_NAMES:
        raise ControlTraceParseError("control event kind is unsupported")
    _trace_sha256(parsed["receipt_sha256"], label="control event receipt_sha256")
    _trace_sha256(
        parsed["previous_control_chain_sha256"], label="control event previous chain"
    )
    _trace_sha256(parsed["control_chain_sha256"], label="control event chain")
    if parsed["previous_control_chain_sha256"] != previous_chain:
        raise ControlTraceParseError("control event chain is discontinuous")
    receipt = _trace_validate_receipt(
        parsed["event_kind"],
        parsed["receipt"],
        contract=contract,
        scenario=scenario,
        run_id=run_id,
        authority_epoch=authority_epoch,
        topology_sha256=topology_sha256,
    )
    self_field = _RECEIPT_SELF_FIELDS[parsed["event_kind"]]
    if parsed["receipt_sha256"] != receipt[self_field]:
        raise ControlTraceParseError("control event receipt and kind do not match")
    expected_chain = _chain_digest(
        str(contract.data["control_trace"]["domains"]["chain"]),
        previous_chain,
        ordinal,
        parsed["event_kind"],
        parsed["receipt_sha256"],
    )
    if parsed["control_chain_sha256"] != expected_chain:
        raise ControlTraceParseError("control event chain digest is inconsistent")
    if receipt["event_ordinal"] != ordinal:
        raise ControlTraceParseError("receipt and event ordinals differ")
    if receipt["previous_control_chain_sha256"] != previous_chain:
        raise ControlTraceParseError("receipt and event previous chains differ")
    return parsed


def _trace_validate_sequence(
    events: list[dict[str, Any]],
    *,
    scenario: Scenario | None,
    contract: HMCContract,
    footer: dict[str, Any],
) -> None:
    kinds = [event["event_kind"] for event in events]
    terminal = bool(kinds and kinds[-1] == "TERMINAL")
    prefix = kinds[:-1] if terminal else kinds
    if not prefix and not terminal:
        raise ControlTraceParseError("control trace must contain events")
    if prefix:
        if prefix[0] != "SNAPSHOT_VERIFICATION":
            raise ControlTraceParseError("control event lifecycle ordering is invalid")
        complete_steps, remainder = divmod(len(prefix) - 1, 4)
        expected_prefix = ["SNAPSHOT_VERIFICATION"]
        for _ in range(complete_steps):
            expected_prefix.extend(
                ["PROPOSAL", "ARBITRATION", "STEP", "SNAPSHOT_VERIFICATION"]
            )
        if remainder:
            expected_prefix.extend(["PROPOSAL", "ARBITRATION"][:remainder])
        if prefix != expected_prefix:
            raise ControlTraceParseError("control event lifecycle ordering is invalid")
        if terminal and remainder not in {1, 2}:
            raise ControlTraceParseError("terminal event is not causally reachable")
    elif terminal:
        if len(events) != 1:
            raise ControlTraceParseError("terminal event ordering is invalid")
    if not terminal:
        if remainder:
            raise ControlTraceParseError("completed trace has a partial cycle")
        if scenario is not None and len(
            [kind for kind in kinds if kind == "STEP"]
        ) != int(scenario.data["steps"]):
            raise ControlTraceParseError(
                "completed trace does not contain all configured steps"
            )
        if footer["terminal_status"] != "COMPLETED":
            raise ControlTraceParseError("completed event sequence has terminal status")
    else:
        if footer["terminal_status"] != "TERMINAL_FAILURE":
            raise ControlTraceParseError("terminal event has completed status")
        if len(events) == 1:
            if events[0]["receipt"]["lifecycle_phase"] != "RESET":
                raise ControlTraceParseError(
                    "initial terminal lifecycle phase is impossible"
                )
        elif events[-2]["event_kind"] not in {"PROPOSAL", "ARBITRATION"}:
            raise ControlTraceParseError("terminal event is not causally reachable")


def _trace_validate_causality(
    events: list[dict[str, Any]],
    *,
    contract: HMCContract,
    footer: dict[str, Any],
) -> None:
    null_roots = contract.data["null_roots"]
    previous_verification = str(null_roots["verification_receipt"]["sha256"])
    previous_step = str(null_roots["step_receipt"]["sha256"])
    current_snapshot = None
    current_proposal = None
    current_arbitration = None
    latest_step = previous_step
    latest_plant = str(null_roots["plant_receipt"]["sha256"])
    latest_snapshot = str(null_roots["snapshot"]["sha256"])
    latest_verification = previous_verification
    seen_receipts: set[str] = set()
    event_by_receipt: dict[str, dict[str, Any]] = {}
    pending_step: dict[str, Any] | None = None
    for event in events:
        kind = event["event_kind"]
        receipt = event["receipt"]
        receipt_sha = event["receipt_sha256"]
        if receipt_sha in seen_receipts:
            raise ControlTraceParseError("duplicate receipt identity in trace")
        seen_receipts.add(receipt_sha)
        event_by_receipt[receipt_sha] = event
        if kind == "SNAPSHOT_VERIFICATION":
            if receipt["previous_verification_receipt_digest"] != previous_verification:
                raise ControlTraceParseError(
                    "snapshot verification lineage is inconsistent"
                )
            if receipt["sequence"] != receipt["completed_step"]:
                raise ControlTraceParseError("snapshot sequence is inconsistent")
            if pending_step is None:
                if receipt["completed_step_receipt_digest"] != str(
                    null_roots["step_receipt"]["sha256"]
                ) or receipt["completed_plant_receipt_digest"] != str(
                    null_roots["plant_receipt"]["sha256"]
                ):
                    raise ControlTraceParseError(
                        "initial snapshot completed identities are inconsistent"
                    )
            else:
                if (
                    receipt["completed_step_receipt_digest"]
                    != pending_step["step_receipt_sha256"]
                    or receipt["completed_plant_receipt_digest"]
                    != pending_step["plant_receipt_digest"]
                ):
                    raise ControlTraceParseError(
                        "snapshot completed identities are not linked to STEP"
                    )
                pending_step = None
            current_snapshot = receipt["snapshot_sha256"]
            latest_snapshot = current_snapshot
            latest_verification = receipt["snapshot_verification_receipt_sha256"]
            previous_verification = latest_verification
            current_proposal = None
            current_arbitration = None
        elif kind == "PROPOSAL":
            if receipt["observation_snapshot_sha256"] != current_snapshot:
                raise ControlTraceParseError(
                    "proposal is not linked to current snapshot"
                )
            if receipt["sequence"] != receipt["requested_application_step"]:
                raise ControlTraceParseError("proposal sequence is inconsistent")
            current_proposal = receipt["proposal_receipt_sha256"]
        elif kind == "ARBITRATION":
            if receipt["observation_snapshot_sha256"] != current_snapshot:
                raise ControlTraceParseError(
                    "arbitration is not linked to current snapshot"
                )
            if receipt["proposal_receipt_sha256"] != current_proposal:
                raise ControlTraceParseError(
                    "arbitration is not linked to current proposal"
                )
            if receipt["decision_step"] != receipt["application_step"]:
                raise ControlTraceParseError(
                    "arbitration application step is inconsistent"
                )
            current_arbitration = receipt["arbitration_receipt_sha256"]
        elif kind == "STEP":
            if receipt["proposal_receipt_sha256"] != current_proposal:
                raise ControlTraceParseError("step is not linked to current proposal")
            if receipt["arbitration_receipt_sha256"] != current_arbitration:
                raise ControlTraceParseError(
                    "step is not linked to current arbitration"
                )
            if receipt["previous_step_receipt_digest"] != previous_step:
                raise ControlTraceParseError("step receipt lineage is inconsistent")
            arbitration_event = event_by_receipt.get(
                receipt["arbitration_receipt_sha256"]
            )
            if (
                arbitration_event is None
                or receipt["final_command_sha256"]
                != arbitration_event["receipt"]["final_command_sha256"]
            ):
                raise ControlTraceParseError(
                    "step command identity is not linked to ARBITRATION"
                )
            previous_step = receipt["step_receipt_sha256"]
            latest_step = previous_step
            latest_plant = receipt["plant_receipt_digest"]
            pending_step = receipt
        elif kind == "TERMINAL":
            if receipt["last_good_snapshot_sha256"] != latest_snapshot:
                raise ControlTraceParseError(
                    "terminal last-good snapshot is inconsistent"
                )
            if receipt["last_good_verification_receipt_sha256"] != latest_verification:
                raise ControlTraceParseError(
                    "terminal last-good verification is inconsistent"
                )
            if receipt["last_good_step_receipt_sha256"] != latest_step:
                raise ControlTraceParseError("terminal last-good step is inconsistent")
            if receipt["proposal_receipt_sha256"] != current_proposal:
                raise ControlTraceParseError(
                    "terminal proposal linkage is inconsistent"
                )
            expected_arb = current_arbitration
            if receipt["arbitration_receipt_sha256"] != expected_arb:
                raise ControlTraceParseError(
                    "terminal arbitration linkage is inconsistent"
                )
            if expected_arb is None and receipt["final_command_sha256"] is not None:
                raise ControlTraceParseError(
                    "terminal final command linkage is inconsistent"
                )
            if (
                expected_arb is not None
                and receipt["final_command_sha256"]
                != event_by_receipt[expected_arb]["receipt"]["final_command_sha256"]
            ):
                raise ControlTraceParseError(
                    "terminal final command is not linked to ARBITRATION"
                )
            if receipt["sequence"] != footer["final_sequence"]:
                raise ControlTraceParseError("terminal sequence is inconsistent")
            if receipt["application_step"] != (
                None
                if expected_arb is None and current_proposal is None
                else footer["final_sequence"]
            ):
                raise ControlTraceParseError(
                    "terminal application step is inconsistent"
                )
            if (
                footer["terminal_failure_receipt_sha256"]
                != receipt["terminal_failure_receipt_sha256"]
            ):
                raise ControlTraceParseError(
                    "footer terminal receipt linkage is inconsistent"
                )
    if footer["last_good_snapshot_sha256"] != latest_snapshot:
        raise ControlTraceParseError("footer last-good snapshot is inconsistent")
    if footer["last_good_verification_receipt_sha256"] != latest_verification:
        raise ControlTraceParseError("footer last-good verification is inconsistent")
    if footer["last_good_step_receipt_sha256"] != latest_step:
        raise ControlTraceParseError("footer last-good step is inconsistent")
    if footer["last_good_plant_receipt_digest"] != latest_plant:
        raise ControlTraceParseError("footer last-good plant receipt is inconsistent")


def _trace_validate_proposal_semantics(
    receipt: dict[str, Any],
    *,
    scenario: Scenario,
    contract: HMCContract,
    snapshot_sha256: str,
) -> None:
    policy = contract.data["safety_policy"]
    attempt_class = receipt["attempt_class"]
    outcome = receipt["validation_outcome"]
    reason = receipt["reason_code"]
    nullable_fields = (
        "source_id",
        "source_type",
        "proposal",
        "proposal_sha256",
        "requested_command_sha256",
    )
    if reason not in policy["proposal_reason_codes"]:
        raise ControlTraceParseError("proposal semantics contain an unsupported reason")
    if attempt_class == "NONE":
        if (
            outcome != "NO_PROPOSAL"
            or reason != "no_proposal"
            or any(receipt[field] is not None for field in nullable_fields)
            or receipt["attempt_evidence_sha256"]
            != str(contract.data["null_roots"]["proposal_receipt"]["sha256"])
        ):
            raise ControlTraceParseError("proposal semantics are inconsistent")
        return
    if attempt_class == "REJECTED_INPUT":
        if (
            outcome != "REJECTED"
            or reason in {"no_proposal", "valid"}
            or any(receipt[field] is not None for field in nullable_fields)
        ):
            raise ControlTraceParseError("proposal semantics are inconsistent")
        return
    if attempt_class != "CANONICAL_PROPOSAL":
        raise ControlTraceParseError("proposal semantics are inconsistent")

    from .proposal import ProposalValidationError, parse_control_proposal

    proposal = receipt["proposal"]
    if type(proposal) is not dict:
        raise ControlTraceParseError("canonical proposal payload is missing")
    try:
        parsed = parse_control_proposal(
            proposal,
            scenario=scenario,
            control_run_id=receipt["control_run_id"],
            authority_epoch=receipt["authority_epoch"],
            observable_topology_sha256=receipt["observable_topology_sha256"],
            completed_observation_step=receipt["sequence"],
            observation_snapshot_sha256=snapshot_sha256,
            requested_application_step=receipt["requested_application_step"],
        )
    except ProposalValidationError as error:
        raise ControlTraceParseError(
            "canonical proposal semantics are invalid"
        ) from error
    if (
        outcome != "VALID"
        or reason != "valid"
        or receipt["attempt_evidence_sha256"] != parsed.proposal_sha256
        or receipt["proposal_sha256"] != parsed.proposal_sha256
        or receipt["source_id"] != parsed.source_id
        or receipt["source_type"] != parsed.source_type
        or receipt["requested_command_sha256"] != parsed.requested_command.sha256
        or canonical_json_bytes(proposal) != parsed.canonical_bytes
    ):
        raise ControlTraceParseError("canonical proposal semantics are inconsistent")


def _trace_same_mapping(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return canonical_json_bytes(actual) == canonical_json_bytes(expected)


def _trace_hydrate_rejected_proposal(
    verifier: Any,
    event: dict[str, Any],
) -> Any:
    """Hydrate one validated payload-redacted proposal for policy replay only."""
    from .hmc import LifecyclePhase
    from .proposal import _issue_proposal_receipt
    from .snapshot import _issue_receipt_control_event

    if verifier.lifecycle_phase != "OBSERVED":
        raise ControlTraceParseError("proposal replay lifecycle is inconsistent")
    receipt_mapping = event["receipt"]
    content = dict(receipt_mapping)
    content.pop("proposal_receipt_sha256")
    issued = _issue_proposal_receipt(content)
    replay_event = _issue_receipt_control_event(
        event_ordinal=event["event_ordinal"],
        event_kind="PROPOSAL",
        receipt_mapping=issued.to_mapping(),
        receipt_sha256=issued.proposal_receipt_sha256,
        previous_control_chain_sha256=event["previous_control_chain_sha256"],
    )
    if not _trace_same_mapping(replay_event.to_mapping(), event):
        raise ControlTraceParseError("proposal replay event is inconsistent")
    verifier._cached_proposal_receipt = issued
    verifier._cached_control_proposal = None
    verifier._cached_arbitration_receipt = None
    verifier._step_capability = None
    verifier._control_events.append(replay_event)
    verifier._current_control_chain_sha256 = replay_event.control_chain_sha256
    verifier._phase = LifecyclePhase.PROPOSED
    return issued


def _trace_validate_authority_semantics(
    events: list[dict[str, Any]],
    *,
    scenario: Scenario,
    contract: HMCContract,
    reset_nonce: bytes,
) -> None:
    """Re-derive HMC authority decisions with the closed implementation."""
    from .hmc import HabitatManagementComputer
    from .physics import advance_one_step_with_command as replay_step_executor
    from .safety import ArbitrationReceipt

    verifier = HabitatManagementComputer.reset(scenario, contract, reset_nonce)
    current_snapshot_sha256: str | None = None
    current_verified_snapshot = None
    for event in events:
        kind = event["event_kind"]
        receipt = event["receipt"]
        if kind == "TERMINAL":
            break
        if kind == "SNAPSHOT_VERIFICATION":
            observed = verifier.observe()
            if type(observed) is TerminalFailureReceipt:
                raise ControlTraceParseError("snapshot semantics do not replay")
            snapshot, verification = observed
            if (
                snapshot.snapshot_sha256 != receipt["snapshot_sha256"]
                or not _trace_same_mapping(verification.to_mapping(), receipt)
                or not _trace_same_mapping(
                    verifier.control_events[-1].to_mapping(), event
                )
            ):
                raise ControlTraceParseError("snapshot semantics do not replay")
            current_verified_snapshot = verifier.verify_snapshot(snapshot, verification)
            current_snapshot_sha256 = snapshot.snapshot_sha256
            continue
        if kind == "PROPOSAL":
            if current_snapshot_sha256 is None or current_verified_snapshot is None:
                raise ControlTraceParseError("proposal replay lacks a current snapshot")
            _trace_validate_proposal_semantics(
                receipt,
                scenario=scenario,
                contract=contract,
                snapshot_sha256=current_snapshot_sha256,
            )
            if receipt["attempt_class"] == "NONE":
                issued = verifier.propose(None, current_verified_snapshot)
            elif receipt["attempt_class"] == "CANONICAL_PROPOSAL":
                issued = verifier.propose(receipt["proposal"], current_verified_snapshot)
            else:
                issued = _trace_hydrate_rejected_proposal(verifier, event)
            current_verified_snapshot = None
            if not _trace_same_mapping(
                issued.to_mapping(), receipt
            ) or not _trace_same_mapping(
                verifier.control_events[-1].to_mapping(), event
            ):
                raise ControlTraceParseError("proposal semantics do not replay")
            continue
        if kind == "ARBITRATION":
            issued = verifier.arbitrate()
            if type(issued) is not ArbitrationReceipt or (
                not _trace_same_mapping(issued.to_mapping(), receipt)
                or not _trace_same_mapping(
                    verifier.control_events[-1].to_mapping(), event
                )
            ):
                raise ControlTraceParseError(
                    "arbitration semantics do not match deterministic HMC policy"
                )
            continue
        if kind == "STEP":
            issued = verifier._step_with_executor(replay_step_executor)
            if type(issued) is not StepReceipt or (
                not _trace_same_mapping(issued.to_mapping(), receipt)
                or len(verifier.control_events) < 2
                or not _trace_same_mapping(
                    verifier.control_events[-2].to_mapping(), event
                )
            ):
                raise ControlTraceParseError("step semantics do not replay")
            continue
        raise ControlTraceParseError("control event kind is unsupported")


def parse_control_trace(
    data: bytes,
    scenario: Scenario | None = None,
    contract: HMCContract | None = None,
) -> ControlTrace:
    """Parse and strictly validate canonical whole-control-trace bytes."""
    root = _trace_decode(data)
    if contract is None:
        raise ControlTraceParseError("strict trace parsing requires an HMCContract")
    if scenario is None:
        raise ControlTraceParseError("strict trace parsing requires a Scenario")
    expected_top = {"header", "events", "footer"}
    if set(root) != expected_top:
        raise ControlTraceParseError(
            "trace root must contain exactly header, events, and footer"
        )
    header, scenario, contract = _trace_validate_header(
        root["header"], scenario, contract
    )
    footer_fields = tuple(contract.data["control_trace"]["footer_fields"])
    footer = _trace_exact_fields(root["footer"], footer_fields, label="trace footer")
    if footer["record_type"] != _TRACE_FOOTER_RECORD_TYPE:
        raise ControlTraceParseError("trace footer record_type is invalid")
    for field in footer_fields:
        if field.endswith("_sha256"):
            _trace_sha256(footer[field], label=f"footer {field}")
    for field in ("event_count", "final_sequence"):
        _trace_non_negative_int(footer[field], label=f"footer {field}")
    if footer["terminal_status"] not in {"COMPLETED", "TERMINAL_FAILURE"}:
        raise ControlTraceParseError("footer terminal status is unsupported")
    if footer["control_trace_footer_sha256"] != _trace_self_digest(
        footer, "control_trace_footer_sha256"
    ):
        raise ControlTraceParseError("footer self digest is inconsistent")
    if footer["control_trace_schema_sha256"] != contract.control_trace_schema_sha256:
        raise ControlTraceParseError("footer schema identity is inconsistent")
    if footer["control_trace_header_sha256"] != header["control_trace_header_sha256"]:
        raise ControlTraceParseError("footer header identity is inconsistent")
    if (
        footer["control_run_id"] != header["control_run_id"]
        or footer["authority_epoch"] != header["authority_epoch"]
    ):
        raise ControlTraceParseError("footer run or epoch identity is inconsistent")
    if type(root["events"]) is not list:
        raise ControlTraceParseError("trace events must be an ordered array")
    events: list[dict[str, Any]] = []
    previous_chain = header["null_control_chain_sha256"]
    for ordinal, event in enumerate(root["events"]):
        parsed_event = _trace_validate_event(
            event,
            ordinal=ordinal,
            previous_chain=previous_chain,
            contract=contract,
            scenario=scenario,
            run_id=header["control_run_id"],
            authority_epoch=header["authority_epoch"],
            topology_sha256=(
                None if scenario is None else header["observable_topology_sha256"]
            ),
        )
        events.append(parsed_event)
        previous_chain = parsed_event["control_chain_sha256"]
    if footer["event_count"] != len(events):
        raise ControlTraceParseError("footer event count is inconsistent")
    if footer["final_control_chain_sha256"] != previous_chain:
        raise ControlTraceParseError("footer final chain identity is inconsistent")
    body_domain = str(contract.data["control_trace"]["domains"]["body"])
    expected_body = _trace_framed_digest(
        body_domain, tuple(canonical_json_bytes(event) for event in events)
    )
    if footer["control_trace_body_sha256"] != expected_body:
        raise ControlTraceParseError("footer body digest is inconsistent")
    if footer["final_sequence"] != len(
        [event for event in events if event["event_kind"] == "STEP"]
    ):
        raise ControlTraceParseError("footer final sequence is inconsistent")
    _trace_validate_sequence(
        events, scenario=scenario, contract=contract, footer=footer
    )
    _trace_validate_causality(events, contract=contract, footer=footer)
    _trace_validate_authority_semantics(
        events,
        scenario=scenario,
        contract=contract,
        reset_nonce=bytes.fromhex(header["reset_nonce_hex"]),
    )
    return _trace_issue_artifact(root)


@dataclass(frozen=True, init=False, slots=True)
class ControlTraceReplayResult:
    """Immutable result of independent whole-trace replay."""

    trace: ControlTrace
    final_state: PlantState
    final_state_sha256: str
    committed_step_count: int

    def __init__(
        self,
        *,
        trace: ControlTrace | None = None,
        final_state: PlantState | None = None,
        final_state_sha256: str | None = None,
        committed_step_count: int | None = None,
        _token: object | None = None,
    ) -> None:
        if _token is not _CONTROL_TRACE_REPLAY_TOKEN:
            raise TypeError("ControlTraceReplayResult must be issued by replay")
        object.__setattr__(self, "trace", trace)
        object.__setattr__(self, "final_state", final_state)
        object.__setattr__(self, "final_state_sha256", final_state_sha256)
        object.__setattr__(self, "committed_step_count", committed_step_count)


def replay_control_trace(
    data: bytes, scenario: Scenario, contract: HMCContract
) -> ControlTraceReplayResult:
    """Validate then replay committed external commands from a whole trace."""
    try:
        trace = parse_control_trace(data, scenario, contract)
    except ControlTraceError:
        raise
    from .physics import (
        advance_one_step_with_command,
        validate_external_command,
        validate_external_step_result,
        initial_state,
    )

    state = initial_state(scenario)
    events = list(trace.to_mapping()["events"])
    by_receipt = {event["receipt_sha256"]: event for event in events}
    step_count = 0
    for event in events:
        if event["event_kind"] != "STEP":
            continue
        step = event["receipt"]
        arbitration_event = by_receipt.get(step["arbitration_receipt_sha256"])
        if (
            arbitration_event is None
            or arbitration_event["event_kind"] != "ARBITRATION"
        ):
            raise ControlTraceReplayError("step arbitration linkage is not replayable")
        arbitration = arbitration_event["receipt"]
        command = arbitration["final_command"]
        try:
            canonical_command = validate_external_command(scenario, command)
            if canonical_command.sha256 != arbitration["final_command_sha256"]:
                raise ValueError("arbitration command digest mismatch")
            candidate = advance_one_step_with_command(scenario, state, command)
            validate_external_step_result(scenario, state, command, candidate)
        except Exception as error:  # noqa: BLE001 - replay failure boundary
            raise ControlTraceReplayError(
                "trace command or plant replay failed"
            ) from error
        returned_digest = candidate.receipt.get("external_command_digest")
        plant_digest = hashlib.sha256(
            canonical_json_bytes(candidate.receipt)
        ).hexdigest()
        if (
            returned_digest != step["returned_external_command_digest"]
            or returned_digest != step["final_command_sha256"]
            or plant_digest != step["plant_receipt_digest"]
        ):
            raise ControlTraceReplayError("trace STEP receipt does not match replay")
        state = candidate.state
        step_count += 1
    final_domain = str(contract.data["control_trace"]["domains"]["final_state"])
    final_digest = _trace_final_state_digest(state, final_domain)
    if final_digest != trace.footer["final_state_sha256"]:
        raise ControlTraceReplayError(
            "trace final state identity does not match replay"
        )
    return ControlTraceReplayResult(
        trace=trace,
        final_state=state,
        final_state_sha256=final_digest,
        committed_step_count=step_count,
        _token=_CONTROL_TRACE_REPLAY_TOKEN,
    )


ControlTraceReplay = ControlTraceReplayResult


__all__ += (
    "ControlTrace",
    "ControlTraceError",
    "ControlTraceIssuanceError",
    "ControlTraceParseError",
    "ControlTraceReplayError",
    "ControlTraceReplayResult",
    "ControlTraceReplay",
    "ControlTraceValidationError",
    "ControlTraceExportError",
    "parse_control_trace",
    "replay_control_trace",
    "_issue_control_trace",
    "_trace_state_mapping",
)
