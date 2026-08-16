"""Closed FIT/CAL cluster selection for the qualified V2 corpus.

The split is selected from the approved roster before any continuation is
materialized or any HMC runner is called.  Validation clusters are intentionally
not returned by ``fit_cal_cluster_ids`` and remain closed.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import canonical_json_bytes
from .pilot import PilotDesign

PROTOCOL_RELATIVE = Path("docs/plans/2026-08-16-habitat-v2-qualified-model-protocol-v1.json")
_PACKET_COUNT_PER_CLUSTER = 2 * 13 * 3
_EXAMPLES_PER_PACKET = 5


class QualificationSplitError(ValueError):
    """The frozen V2 protocol or approved roster cannot define a closed split."""


@dataclass(frozen=True, slots=True)
class QualificationSplit:
    protocol_sha256: str
    fit_cluster_ids: tuple[str, ...]
    cal_cluster_ids: tuple[str, ...]
    validation_cluster_ids: tuple[str, ...]

    @property
    def authorized_cluster_ids(self) -> frozenset[str]:
        return frozenset((*self.fit_cluster_ids, *self.cal_cluster_ids))

    @property
    def expected_packet_count(self) -> int:
        return len(self.authorized_cluster_ids) * _PACKET_COUNT_PER_CLUSTER

    @property
    def expected_example_count(self) -> int:
        return self.expected_packet_count * _EXAMPLES_PER_PACKET

    def split_for(self, cluster_id: str) -> str:
        """Return the frozen split for an exact approved cluster ID."""
        if cluster_id in self.fit_cluster_ids:
            return "fit"
        if cluster_id in self.cal_cluster_ids:
            return "cal"
        if cluster_id in self.validation_cluster_ids:
            return "validation"
        raise QualificationSplitError(f"unknown approved cluster ID: {cluster_id}")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_qualified_protocol(root: str | Path) -> dict[str, Any]:
    """Load the exact self-hashed qualification protocol without opening data."""
    path = Path(root).resolve() / PROTOCOL_RELATIVE
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationSplitError("qualified protocol cannot be read") from error
    if type(value) is not dict:
        raise QualificationSplitError("qualified protocol must be an object")
    declared = value.pop("protocol_sha256", None)
    if type(declared) is not str or declared != _sha256(canonical_json_bytes(value)):
        raise QualificationSplitError("qualified protocol self-hash mismatch")
    value["protocol_sha256"] = declared
    return value


def _cluster_id(mode: str, load: str, role: str) -> str:
    return f"pilot-v1/{mode}/{load.lower()}/{role}"


def _rank(protocol_sha256: str, mode: str, load: str, role: str) -> bytes:
    cluster_id = _cluster_id(mode, load, role)
    message = f"{mode}\0{load}\0{cluster_id}".encode()
    return hmac.new(bytes.fromhex(protocol_sha256), message, hashlib.sha256).digest()


def build_qualification_split(design: PilotDesign, protocol: dict[str, Any]) -> QualificationSplit:
    """Build and verify the whole-cluster 3/1/1 FIT/CAL/validation allocation."""
    if type(design) is not PilotDesign:
        raise QualificationSplitError("exact approved PilotDesign is required")
    try:
        families = protocol["scenario_families"]
        split = protocol["split"]
        protocol_sha256 = protocol["protocol_sha256"]
        modes = tuple(families["operating_modes"])
        loads = tuple(families["load_regimes"])
        roles = tuple(families["semantic_profile_roles"])
        per = split["clusters_per_stratum"]
    except (KeyError, TypeError) as error:
        raise QualificationSplitError("qualified protocol split is malformed") from error
    if (
        type(protocol_sha256) is not str
        or len(protocol_sha256) != 64
        or modes != design.operating_modes
        or loads != design.load_regimes
        or roles != design.semantic_profile_roles
        or per != {"fit": 3, "cal": 1, "val": 1}
    ):
        raise QualificationSplitError("qualified protocol/design split binding drifts")
    expected_ids = {_cluster_id(mode, load, role) for mode in modes for load in loads for role in roles}
    actual_ids = {cluster.cluster_id for cluster in design.clusters}
    if actual_ids != expected_ids or len(actual_ids) != 60:
        raise QualificationSplitError("approved roster cluster identities drift")

    by: dict[str, list[str]] = {"fit": [], "cal": [], "validation": []}
    for mode in modes:
        for load in loads:
            ordered = sorted(roles, key=lambda role: _rank(protocol_sha256, mode, load, role))
            for role in ordered[:3]:
                by["fit"].append(_cluster_id(mode, load, role))
            by["cal"].append(_cluster_id(mode, load, ordered[3]))
            by["validation"].append(_cluster_id(mode, load, ordered[4]))
    result = QualificationSplit(
        protocol_sha256=protocol_sha256,
        fit_cluster_ids=tuple(sorted(by["fit"])),
        cal_cluster_ids=tuple(sorted(by["cal"])),
        validation_cluster_ids=tuple(sorted(by["validation"])),
    )
    if (
        len(result.fit_cluster_ids) != 36
        or len(result.cal_cluster_ids) != 12
        or len(result.validation_cluster_ids) != 12
        or len(result.authorized_cluster_ids) != 48
        or result.authorized_cluster_ids & frozenset(result.validation_cluster_ids)
        or result.expected_packet_count != 3744
        or result.expected_example_count != 18720
    ):
        raise QualificationSplitError("qualified split counts or isolation drift")
    return result


def fit_cal_cluster_ids(root: str | Path, design: PilotDesign) -> QualificationSplit:
    """Return the only clusters authorized for this worker run; never validation."""
    return build_qualification_split(design, load_qualified_protocol(root))
