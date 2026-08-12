from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .hmc_contract import canonical_json_bytes

_SNAPSHOT_ISSUANCE_TOKEN = object()
_RECEIPT_ISSUANCE_TOKEN = object()
_EVENT_ISSUANCE_TOKEN = object()
_VERIFIED_HANDLE_TOKEN = object()


class SnapshotIssuanceError(ValueError):
    """Raised when immutable operational evidence cannot be issued."""


class SnapshotVerificationError(ValueError):
    """Raised when runtime issuance capability or canonical evidence is invalid."""


class _FinalIssuedType:
    _sealed = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if any(getattr(base, "_sealed", False) for base in cls.__bases__):
            raise TypeError(f"{cls.__name__} cannot subclass a final issued type")
        cls._sealed = True


def _mapping_bytes(mapping: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(mapping)


def _self_digest(mapping: Mapping[str, Any], field: str) -> str:
    content = dict(mapping)
    content.pop(field)
    return hashlib.sha256(_mapping_bytes(content)).hexdigest()


def _chain_digest(
    previous_control_chain_sha256: str,
    event_ordinal: int,
    event_kind: str,
    receipt_sha256: str,
) -> str:
    event_kind_bytes = event_kind.encode("utf-8")
    return hashlib.sha256(
        b"aeolus-habitat-v2-hmc-control-chain-v1"
        + bytes.fromhex(previous_control_chain_sha256)
        + event_ordinal.to_bytes(8, "big")
        + len(event_kind_bytes).to_bytes(8, "big")
        + event_kind_bytes
        + bytes.fromhex(receipt_sha256)
    ).hexdigest()


@dataclass(frozen=True, init=False, slots=True)
class OperationalSnapshot(_FinalIssuedType):
    canonical_bytes: bytes
    snapshot_sha256: str

    def __init__(
        self,
        *,
        canonical_bytes: bytes | None = None,
        snapshot_sha256: str | None = None,
        _token: object | None = None,
    ) -> None:
        if _token is not _SNAPSHOT_ISSUANCE_TOKEN:
            raise TypeError("OperationalSnapshot must be issued by an HMC")
        if type(canonical_bytes) is not bytes or type(snapshot_sha256) is not str:
            raise SnapshotIssuanceError("snapshot issuance data is malformed")
        object.__setattr__(self, "canonical_bytes", canonical_bytes)
        object.__setattr__(self, "snapshot_sha256", snapshot_sha256)

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)


@dataclass(frozen=True, init=False, slots=True)
class SnapshotVerificationReceipt(_FinalIssuedType):
    canonical_bytes: bytes
    receipt_schema_sha256: str
    snapshot_verification_contract_sha256: str
    hmc_contract_sha256: str
    snapshot_schema_sha256: str
    observable_topology_sha256: str
    control_run_id: str
    authority_epoch: str
    issuer_id: str
    cycle_id: str
    sequence: int
    completed_step: int
    completed_time_s: float
    snapshot_sha256: str
    completed_plant_receipt_digest: str
    completed_step_receipt_digest: str
    previous_verification_receipt_digest: str
    event_ordinal: int
    previous_control_chain_sha256: str
    snapshot_verification_receipt_sha256: str

    def __init__(
        self,
        *,
        mapping: Mapping[str, Any] | None = None,
        canonical_bytes: bytes | None = None,
        _token: object | None = None,
    ) -> None:
        if _token is not _RECEIPT_ISSUANCE_TOKEN:
            raise TypeError("SnapshotVerificationReceipt must be issued by an HMC")
        if mapping is None or type(canonical_bytes) is not bytes:
            raise SnapshotIssuanceError(
                "verification receipt issuance data is malformed"
            )
        for field in (
            "receipt_schema_sha256",
            "snapshot_verification_contract_sha256",
            "hmc_contract_sha256",
            "snapshot_schema_sha256",
            "observable_topology_sha256",
            "control_run_id",
            "authority_epoch",
            "issuer_id",
            "cycle_id",
            "sequence",
            "completed_step",
            "completed_time_s",
            "snapshot_sha256",
            "completed_plant_receipt_digest",
            "completed_step_receipt_digest",
            "previous_verification_receipt_digest",
            "event_ordinal",
            "previous_control_chain_sha256",
            "snapshot_verification_receipt_sha256",
        ):
            object.__setattr__(self, field, mapping[field])
        object.__setattr__(self, "canonical_bytes", canonical_bytes)

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)


@dataclass(frozen=True, init=False, slots=True)
class ControlEvent(_FinalIssuedType):
    canonical_bytes: bytes
    record_type: str
    event_ordinal: int
    event_kind: str
    receipt_sha256: str
    previous_control_chain_sha256: str
    control_chain_sha256: str
    receipt: Mapping[str, Any]

    def __init__(
        self,
        *,
        mapping: Mapping[str, Any] | None = None,
        canonical_bytes: bytes | None = None,
        _token: object | None = None,
    ) -> None:
        if _token is not _EVENT_ISSUANCE_TOKEN:
            raise TypeError("ControlEvent must be issued by an HMC")
        if mapping is None or type(canonical_bytes) is not bytes:
            raise SnapshotIssuanceError("control event issuance data is malformed")
        for field in (
            "record_type",
            "event_ordinal",
            "event_kind",
            "receipt_sha256",
            "previous_control_chain_sha256",
            "control_chain_sha256",
            "receipt",
        ):
            object.__setattr__(self, field, mapping[field])
        object.__setattr__(self, "canonical_bytes", canonical_bytes)

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)


@dataclass(frozen=True, init=False, slots=True)
class VerifiedSnapshotHandle(_FinalIssuedType):
    snapshot_sha256: str
    cycle_id: str
    sequence: int

    def __init__(
        self,
        *,
        snapshot_sha256: str | None = None,
        cycle_id: str | None = None,
        sequence: int | None = None,
        _token: object | None = None,
    ) -> None:
        if _token is not _VERIFIED_HANDLE_TOKEN:
            raise TypeError("VerifiedSnapshotHandle must be issued by an HMC")
        if (
            type(snapshot_sha256) is not str
            or type(cycle_id) is not str
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
        ):
            raise SnapshotVerificationError("verified snapshot handle is malformed")
        object.__setattr__(self, "snapshot_sha256", snapshot_sha256)
        object.__setattr__(self, "cycle_id", cycle_id)
        object.__setattr__(self, "sequence", sequence)

    def __reduce__(self) -> object:
        raise TypeError("VerifiedSnapshotHandle is not serialisable")


def _issue_operational_snapshot(content: Mapping[str, Any]) -> OperationalSnapshot:
    if "snapshot_sha256" in content:
        raise SnapshotIssuanceError("snapshot content already contains its self digest")
    digest = hashlib.sha256(_mapping_bytes(content)).hexdigest()
    mapping = {**content, "snapshot_sha256": digest}
    canonical = _mapping_bytes(mapping)
    return OperationalSnapshot(
        canonical_bytes=canonical,
        snapshot_sha256=digest,
        _token=_SNAPSHOT_ISSUANCE_TOKEN,
    )


def _issue_snapshot_verification_receipt(
    content: Mapping[str, Any],
) -> SnapshotVerificationReceipt:
    self_field = "snapshot_verification_receipt_sha256"
    if self_field in content:
        raise SnapshotIssuanceError(
            "verification content already contains its self digest"
        )
    mapping = dict(content)
    mapping[self_field] = _self_digest({**mapping, self_field: ""}, self_field)
    canonical = _mapping_bytes(mapping)
    return SnapshotVerificationReceipt(
        mapping=mapping,
        canonical_bytes=canonical,
        _token=_RECEIPT_ISSUANCE_TOKEN,
    )


def _issue_control_event(
    *,
    event_ordinal: int,
    event_kind: str,
    receipt: SnapshotVerificationReceipt,
) -> ControlEvent:
    if type(receipt) is not SnapshotVerificationReceipt:
        raise SnapshotIssuanceError(
            "snapshot control event requires the exact verification receipt"
        )
    return _issue_receipt_control_event(
        event_ordinal=event_ordinal,
        event_kind=event_kind,
        receipt_mapping=receipt.to_mapping(),
        receipt_sha256=receipt.snapshot_verification_receipt_sha256,
        previous_control_chain_sha256=receipt.previous_control_chain_sha256,
    )


def _issue_receipt_control_event(
    *,
    event_ordinal: int,
    event_kind: str,
    receipt_mapping: Mapping[str, Any],
    receipt_sha256: str,
    previous_control_chain_sha256: str,
) -> ControlEvent:
    self_digest_fields = {
        "SNAPSHOT_VERIFICATION": "snapshot_verification_receipt_sha256",
        "PROPOSAL": "proposal_receipt_sha256",
        "ARBITRATION": "arbitration_receipt_sha256",
        "STEP": "step_receipt_sha256",
        "TERMINAL": "terminal_failure_receipt_sha256",
    }
    if (
        isinstance(event_ordinal, bool)
        or not isinstance(event_ordinal, int)
        or event_ordinal < 0
        or type(event_kind) is not str
        or event_kind not in self_digest_fields
        or type(receipt_mapping) is not dict
        or type(receipt_sha256) is not str
        or type(previous_control_chain_sha256) is not str
    ):
        raise SnapshotIssuanceError("control event issuance data is malformed")
    self_digest_field = self_digest_fields[event_kind]
    if (
        self_digest_field not in receipt_mapping
        or receipt_mapping[self_digest_field] != receipt_sha256
    ):
        raise SnapshotIssuanceError("control event receipt self digest is inconsistent")
    canonical_receipt = _mapping_bytes(receipt_mapping)
    if (
        hashlib.sha256(
            _mapping_bytes(
                {
                    key: value
                    for key, value in receipt_mapping.items()
                    if key != self_digest_field
                }
            )
        ).hexdigest()
        != receipt_sha256
    ):
        raise SnapshotIssuanceError("control event receipt digest is inconsistent")
    if len(canonical_receipt) == 0:
        raise SnapshotIssuanceError("control event receipt is empty")
    try:
        chain = _chain_digest(
            previous_control_chain_sha256,
            event_ordinal,
            event_kind,
            receipt_sha256,
        )
    except (TypeError, ValueError) as error:
        raise SnapshotIssuanceError(
            "control event digest input is malformed"
        ) from error
    mapping = {
        "record_type": "CONTROL_EVENT",
        "event_ordinal": event_ordinal,
        "event_kind": event_kind,
        "receipt_sha256": receipt_sha256,
        "previous_control_chain_sha256": previous_control_chain_sha256,
        "control_chain_sha256": chain,
        "receipt": dict(receipt_mapping),
    }
    return ControlEvent(
        mapping=mapping,
        canonical_bytes=_mapping_bytes(mapping),
        _token=_EVENT_ISSUANCE_TOKEN,
    )


def verify_issued_snapshot_pair(
    *,
    snapshot: OperationalSnapshot,
    receipt: SnapshotVerificationReceipt,
    event: ControlEvent,
) -> VerifiedSnapshotHandle:
    if (
        type(snapshot) is not OperationalSnapshot
        or type(receipt) is not SnapshotVerificationReceipt
    ):
        raise SnapshotVerificationError(
            "runtime verification requires exact issued types"
        )
    if type(event) is not ControlEvent:
        raise SnapshotVerificationError(
            "runtime verification requires exact issued event"
        )
    snapshot_mapping = snapshot.to_mapping()
    receipt_mapping = receipt.to_mapping()
    event_mapping = event.to_mapping()
    if snapshot.canonical_bytes != _mapping_bytes(snapshot_mapping):
        raise SnapshotVerificationError("snapshot canonical bytes are inconsistent")
    if receipt.canonical_bytes != _mapping_bytes(receipt_mapping):
        raise SnapshotVerificationError("verification canonical bytes are inconsistent")
    if event.canonical_bytes != _mapping_bytes(event_mapping):
        raise SnapshotVerificationError("event canonical bytes are inconsistent")
    expected_snapshot = _self_digest(snapshot_mapping, "snapshot_sha256")
    if snapshot.snapshot_sha256 != expected_snapshot:
        raise SnapshotVerificationError("snapshot self digest is inconsistent")
    expected_receipt = _self_digest(
        receipt_mapping, "snapshot_verification_receipt_sha256"
    )
    if receipt.snapshot_verification_receipt_sha256 != expected_receipt:
        raise SnapshotVerificationError("verification self digest is inconsistent")
    if receipt.snapshot_sha256 != snapshot.snapshot_sha256:
        raise SnapshotVerificationError("snapshot and verification identities differ")
    expected_chain = _chain_digest(
        receipt.previous_control_chain_sha256,
        receipt.event_ordinal,
        "SNAPSHOT_VERIFICATION",
        receipt.snapshot_verification_receipt_sha256,
    )
    if (
        event.event_kind != "SNAPSHOT_VERIFICATION"
        or event.event_ordinal != receipt.event_ordinal
        or event.receipt_sha256 != receipt.snapshot_verification_receipt_sha256
        or event.previous_control_chain_sha256 != receipt.previous_control_chain_sha256
        or event.control_chain_sha256 != expected_chain
        or event.receipt != receipt_mapping
    ):
        raise SnapshotVerificationError("snapshot control event is inconsistent")
    return VerifiedSnapshotHandle(
        snapshot_sha256=snapshot.snapshot_sha256,
        cycle_id=receipt.cycle_id,
        sequence=receipt.sequence,
        _token=_VERIFIED_HANDLE_TOKEN,
    )
