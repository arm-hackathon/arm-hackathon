from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .hmc_contract import canonical_json_bytes
from .physics import CanonicalExternalCommand, validate_external_command
from .scenario import Scenario, ScenarioValidationError
from .snapshot import _FinalIssuedType

_PROPOSAL_RECEIPT_ISSUANCE_TOKEN = object()
_CONTROL_PROPOSAL_ISSUANCE_TOKEN = object()

CONTROL_PROPOSAL_SCHEMA_V1 = "aeolus_habitat_v2_control_proposal_v1"
_CONTROL_PROPOSAL_FIELDS = {
    "schema_version",
    "control_run_id",
    "authority_epoch",
    "source_id",
    "source_type",
    "completed_observation_step",
    "observation_snapshot_sha256",
    "requested_application_step",
    "observable_topology_sha256",
    "proposed_command",
    "confidence",
    "proposal_sha256",
}


class ProposalIssuanceError(ValueError):
    """Raised when closed proposal evidence cannot be issued."""


class ProposalValidationError(ValueError):
    """Closed proposal rejection carrying one allowlisted reason code."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_non_negative_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


@dataclass(frozen=True, init=False, slots=True)
class ControlProposal(_FinalIssuedType):
    canonical_bytes: bytes
    proposal_sha256: str
    source_id: str
    source_type: str
    completed_observation_step: int
    observation_snapshot_sha256: str
    requested_application_step: int
    requested_command: CanonicalExternalCommand

    def __init__(
        self,
        *,
        mapping: Mapping[str, Any] | None = None,
        canonical_bytes: bytes | None = None,
        requested_command: CanonicalExternalCommand | None = None,
        _token: object | None = None,
    ) -> None:
        if _token is not _CONTROL_PROPOSAL_ISSUANCE_TOKEN:
            raise TypeError("ControlProposal must be issued by an HMC parser")
        if (
            type(mapping) is not dict
            or type(canonical_bytes) is not bytes
            or type(requested_command) is not CanonicalExternalCommand
        ):
            raise ProposalIssuanceError("control proposal issuance data is malformed")
        for field in (
            "proposal_sha256",
            "source_id",
            "source_type",
            "completed_observation_step",
            "observation_snapshot_sha256",
            "requested_application_step",
        ):
            object.__setattr__(self, field, mapping[field])
        object.__setattr__(self, "canonical_bytes", canonical_bytes)
        object.__setattr__(self, "requested_command", requested_command)

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)


def parse_control_proposal(
    mapping: Mapping[str, Any],
    *,
    scenario: Scenario,
    control_run_id: str,
    authority_epoch: str,
    observable_topology_sha256: str,
    completed_observation_step: int,
    observation_snapshot_sha256: str,
    requested_application_step: int,
) -> ControlProposal:
    if type(mapping) is not dict or any(type(key) is not str for key in mapping):
        raise ProposalValidationError("rejected_malformed")
    if set(mapping) != _CONTROL_PROPOSAL_FIELDS:
        raise ProposalValidationError("rejected_malformed")
    if mapping["schema_version"] != CONTROL_PROPOSAL_SCHEMA_V1:
        raise ProposalValidationError("rejected_malformed")
    if not _is_sha256(mapping["proposal_sha256"]):
        raise ProposalValidationError("rejected_malformed")
    body = dict(mapping)
    claimed_proposal_sha256 = str(body.pop("proposal_sha256"))
    try:
        canonical_body = canonical_json_bytes(body)
        canonical = canonical_json_bytes(mapping)
    except (TypeError, ValueError):
        raise ProposalValidationError("rejected_malformed") from None
    if hashlib.sha256(canonical_body).hexdigest() != claimed_proposal_sha256:
        raise ProposalValidationError("rejected_malformed")
    if mapping["control_run_id"] != control_run_id:
        raise ProposalValidationError("rejected_wrong_run")
    if mapping["authority_epoch"] != authority_epoch:
        raise ProposalValidationError("rejected_wrong_epoch")
    if mapping["observable_topology_sha256"] != observable_topology_sha256:
        raise ProposalValidationError("rejected_wrong_topology")
    observation_step = mapping["completed_observation_step"]
    application_step = mapping["requested_application_step"]
    if not _is_non_negative_integer(observation_step) or not _is_non_negative_integer(
        application_step
    ):
        raise ProposalValidationError("rejected_malformed")
    if observation_step < completed_observation_step:
        raise ProposalValidationError("rejected_stale_observation")
    if observation_step > completed_observation_step:
        raise ProposalValidationError("rejected_future_observation")
    if mapping["observation_snapshot_sha256"] != observation_snapshot_sha256:
        raise ProposalValidationError("rejected_stale_observation")
    if application_step < requested_application_step:
        raise ProposalValidationError("rejected_stale_observation")
    if application_step > requested_application_step:
        raise ProposalValidationError("rejected_future_observation")
    source_id = mapping["source_id"]
    source_type = mapping["source_type"]
    if (
        type(source_id) is not str
        or not source_id
        or type(source_type) is not str
        or not source_type
    ):
        raise ProposalValidationError("rejected_malformed")
    confidence = mapping["confidence"]
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ProposalValidationError("rejected_malformed")
    proposed_command = mapping["proposed_command"]
    if type(proposed_command) is not dict:
        raise ProposalValidationError("rejected_malformed")
    try:
        requested_command = validate_external_command(scenario, proposed_command)
    except ScenarioValidationError as error:
        message = str(error)
        if "topology mismatch" in message:
            reason_code = "rejected_wrong_topology"
        elif "out of bounds" in message or "exceeds capacity" in message:
            reason_code = "rejected_out_of_bounds"
        else:
            reason_code = "rejected_malformed"
        raise ProposalValidationError(reason_code) from None
    return ControlProposal(
        mapping=dict(mapping),
        canonical_bytes=canonical,
        requested_command=requested_command,
        _token=_CONTROL_PROPOSAL_ISSUANCE_TOKEN,
    )


@dataclass(frozen=True, init=False, slots=True)
class ProposalReceipt(_FinalIssuedType):
    canonical_bytes: bytes
    receipt_schema_sha256: str
    hmc_contract_sha256: str
    observable_topology_sha256: str
    control_run_id: str
    authority_epoch: str
    sequence: int
    observation_snapshot_sha256: str
    requested_application_step: int
    attempt_class: str
    attempt_evidence_sha256: str
    source_id: str | None
    source_type: str | None
    proposal: Mapping[str, Any] | None
    proposal_sha256: str | None
    requested_command_sha256: str | None
    validation_outcome: str
    reason_code: str
    event_ordinal: int
    previous_control_chain_sha256: str
    proposal_receipt_sha256: str

    def __init__(
        self,
        *,
        mapping: Mapping[str, Any] | None = None,
        canonical_bytes: bytes | None = None,
        _token: object | None = None,
    ) -> None:
        if _token is not _PROPOSAL_RECEIPT_ISSUANCE_TOKEN:
            raise TypeError("ProposalReceipt must be issued by an HMC")
        if type(mapping) is not dict or type(canonical_bytes) is not bytes:
            raise ProposalIssuanceError("proposal receipt issuance data is malformed")
        for field in (
            "receipt_schema_sha256",
            "hmc_contract_sha256",
            "observable_topology_sha256",
            "control_run_id",
            "authority_epoch",
            "sequence",
            "observation_snapshot_sha256",
            "requested_application_step",
            "attempt_class",
            "attempt_evidence_sha256",
            "source_id",
            "source_type",
            "proposal",
            "proposal_sha256",
            "requested_command_sha256",
            "validation_outcome",
            "reason_code",
            "event_ordinal",
            "previous_control_chain_sha256",
            "proposal_receipt_sha256",
        ):
            object.__setattr__(self, field, mapping[field])
        object.__setattr__(self, "canonical_bytes", canonical_bytes)

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)


def _issue_proposal_receipt(content: Mapping[str, Any]) -> ProposalReceipt:
    self_field = "proposal_receipt_sha256"
    if type(content) is not dict or self_field in content:
        raise ProposalIssuanceError("proposal receipt content is malformed")
    mapping = dict(content)
    mapping[self_field] = hashlib.sha256(canonical_json_bytes(mapping)).hexdigest()
    canonical = canonical_json_bytes(mapping)
    return ProposalReceipt(
        mapping=mapping,
        canonical_bytes=canonical,
        _token=_PROPOSAL_RECEIPT_ISSUANCE_TOKEN,
    )
