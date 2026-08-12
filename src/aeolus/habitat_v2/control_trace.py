from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .hmc_contract import HMCContract, canonical_json_bytes
from .snapshot import _FinalIssuedType

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


__all__ = (
    "StepReceipt",
    "StepReceiptIssuanceError",
    "TerminalFailureReceipt",
    "TerminalReceiptIssuanceError",
)
