"""Machine-readable Habitat V2 physics parameter provenance (Issue #71).

This module loads and fail-closed validates
`contracts/habitat_v2_physics_provenance_v1.json`, the checked form of
`docs/provenance/habitat-v2-numerical-ledger.md`. Every high-impact parameter
carries value, unit, valid range, classification, citation, source path,
generator variability, uncertainty distribution, and affected systems and
metrics, so the Scenario Family Generator (Issue #72) samples only declared
distributions and no parameter silently lacks provenance.

Classifications follow the ledger: physical_constant, public_requirement,
physics_derived, engineering_assumption, stress_test_range. Engineering
assumptions and stress-test ranges are never presented as NASA, ESA, Artemis,
Gateway, or flight-qualified data.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

PHYSICS_PROVENANCE_FILENAME = "habitat_v2_physics_provenance_v1.json"
PHYSICS_PROVENANCE_MANIFEST_ID = "habitat_v2_physics_provenance_v1"
PHYSICS_PROVENANCE_SCHEMA_VERSION = "aeolus_habitat_v2_physics_provenance_v1"
PHYSICS_PROVENANCE_STATUS = "ACCEPTED_PROVENANCE_MANIFEST"

PROVENANCE_CLASSIFICATIONS = (
    "physical_constant",
    "public_requirement",
    "physics_derived",
    "engineering_assumption",
    "stress_test_range",
)

_PARAMETER_RECORD_KEYS = frozenset(
    {
        "parameter_id",
        "value",
        "unit",
        "valid_range",
        "classification",
        "citation",
        "source_path",
        "equation",
        "generator_variable",
        "uncertainty_distribution",
        "affected_systems",
        "affected_metrics",
    }
)

_DISTRIBUTION_KINDS = ("uniform_relative", "uniform_absolute")


class PhysicsProvenanceError(ValueError):
    """Raised when the physics provenance manifest or a query against it is invalid."""


def _strict_json(raw: bytes) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: set[str] = set()
        for key, _ in pairs:
            if key in seen:
                raise PhysicsProvenanceError(f"duplicate manifest key {key!r}")
            seen.add(key)
        return dict(pairs)

    try:
        return json.loads(raw, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as error:
        raise PhysicsProvenanceError("physics provenance manifest is not valid JSON") from error


def _exact(mapping: Any, fields: set[str], label: str) -> dict[str, Any]:
    if type(mapping) is not dict:
        raise PhysicsProvenanceError(f"provenance {label} must be an object")
    if set(mapping) != fields:
        raise PhysicsProvenanceError(f"provenance {label} field set drifted")
    return mapping


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_record(record: Any, systems: frozenset[str], metrics: frozenset[str], seen: set[str]) -> None:
    fields = _exact(record, _PARAMETER_RECORD_KEYS, "parameter record")
    pid = fields["parameter_id"]
    if type(pid) is not str or not pid or pid in seen:
        raise PhysicsProvenanceError(f"provenance parameter id {pid!r} is invalid or duplicated")
    seen.add(pid)

    value = fields["value"]
    equation = fields["equation"]
    if value is None and (type(equation) is not str or not equation):
        raise PhysicsProvenanceError(f"parameter {pid} has null value without an equation")
    if value is not None and not _is_number(value):
        raise PhysicsProvenanceError(f"parameter {pid} value must be numeric or null")

    if type(fields["unit"]) is not str or not fields["unit"]:
        raise PhysicsProvenanceError(f"parameter {pid} unit is invalid")
    if type(fields["source_path"]) is not str or not fields["source_path"]:
        raise PhysicsProvenanceError(f"parameter {pid} source path is invalid")
    if equation is not None and type(equation) is not str:
        raise PhysicsProvenanceError(f"parameter {pid} equation must be a string or null")

    classification = fields["classification"]
    if classification not in PROVENANCE_CLASSIFICATIONS:
        raise PhysicsProvenanceError(f"parameter {pid} has unknown classification")

    citation = fields["citation"]
    if classification in ("physical_constant", "public_requirement"):
        if type(citation) is not str or not citation:
            raise PhysicsProvenanceError(
                f"parameter {pid} classification {classification} requires a citation"
            )
    elif citation is not None and type(citation) is not str:
        raise PhysicsProvenanceError(f"parameter {pid} citation must be a string or null")

    valid_range = fields["valid_range"]
    if valid_range is not None:
        if (
            type(valid_range) is not list
            or len(valid_range) != 2
            or not all(_is_number(bound) for bound in valid_range)
            or valid_range[0] > valid_range[1]
        ):
            raise PhysicsProvenanceError(f"parameter {pid} valid_range is invalid")
        if _is_number(value) and not (valid_range[0] <= value <= valid_range[1]):
            raise PhysicsProvenanceError(f"parameter {pid} value lies outside its valid range")

    generator_variable = fields["generator_variable"]
    if type(generator_variable) is not bool:
        raise PhysicsProvenanceError(f"parameter {pid} generator_variable must be boolean")
    distribution = fields["uncertainty_distribution"]
    if generator_variable:
        dist = _exact(
            distribution, {"kind", "low", "high"}, f"{pid} uncertainty distribution"
        )
        if dist["kind"] not in _DISTRIBUTION_KINDS:
            raise PhysicsProvenanceError(f"parameter {pid} distribution kind is unknown")
        if not _is_number(dist["low"]) or not _is_number(dist["high"]):
            raise PhysicsProvenanceError(f"parameter {pid} distribution bounds must be numeric")
        if dist["kind"] == "uniform_relative" and not (dist["low"] <= 0.0 <= dist["high"]):
            raise PhysicsProvenanceError(
                f"parameter {pid} relative band must straddle zero"
            )
        if dist["kind"] == "uniform_absolute" and dist["low"] > dist["high"]:
            raise PhysicsProvenanceError(f"parameter {pid} absolute band is inverted")
        if valid_range is not None and dist["kind"] == "uniform_absolute":
            if dist["low"] < valid_range[0] or dist["high"] > valid_range[1]:
                raise PhysicsProvenanceError(
                    f"parameter {pid} absolute band exceeds its valid range"
                )
        if (
            valid_range is not None
            and dist["kind"] == "uniform_relative"
            and _is_number(value)
        ):
            low_edge = value * (1.0 + float(dist["low"]))
            high_edge = value * (1.0 + float(dist["high"]))
            if low_edge < valid_range[0] - 1e-9 or high_edge > valid_range[1] + 1e-9:
                raise PhysicsProvenanceError(
                    f"parameter {pid} relative band exceeds its valid range"
                )
    elif distribution is not None:
        raise PhysicsProvenanceError(
            f"parameter {pid} declares an uncertainty distribution without generator variability"
        )

    for key, vocabulary in (("affected_systems", systems), ("affected_metrics", metrics)):
        entries = fields[key]
        if type(entries) is not list or not entries:
            raise PhysicsProvenanceError(f"parameter {pid} {key} must be a non-empty list")
        unknown = set(entries) - vocabulary
        if unknown:
            raise PhysicsProvenanceError(
                f"parameter {pid} {key} has undeclared entries {sorted(unknown)}"
            )


def validate_physics_provenance_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on the physics provenance manifest."""

    root = _exact(
        manifest,
        {
            "schema_version",
            "manifest_id",
            "status",
            "authorization",
            "narrative_source",
            "primary_fixture",
            "classifications",
            "affected_systems_vocabulary",
            "affected_metrics_vocabulary",
            "parameters",
            "conservation_tolerances",
            "sensitivity",
        },
        "manifest root",
    )
    if (
        root["schema_version"] != PHYSICS_PROVENANCE_SCHEMA_VERSION
        or root["manifest_id"] != PHYSICS_PROVENANCE_MANIFEST_ID
        or root["status"] != PHYSICS_PROVENANCE_STATUS
        or type(root["narrative_source"]) is not str
        or type(root["primary_fixture"]) is not str
    ):
        raise PhysicsProvenanceError("provenance manifest identity drifted")

    authorization = _exact(
        root["authorization"],
        {"authorized_by", "authorized_via_issue", "authorized_at"},
        "authorization",
    )
    if (
        authorization["authorized_by"] != "repository_owner"
        or authorization["authorized_via_issue"] != 71
    ):
        raise PhysicsProvenanceError("provenance manifest authorization drifted")

    if tuple(root["classifications"]) != PROVENANCE_CLASSIFICATIONS:
        raise PhysicsProvenanceError("provenance classifications drifted")
    systems = frozenset(root["affected_systems_vocabulary"])
    metrics = frozenset(root["affected_metrics_vocabulary"])
    if not systems or not metrics:
        raise PhysicsProvenanceError("provenance vocabularies are empty")

    parameters = root["parameters"]
    if type(parameters) is not list or not parameters:
        raise PhysicsProvenanceError("provenance parameters are empty")
    seen: set[str] = set()
    for record in parameters:
        _validate_record(record, systems, metrics, seen)

    tolerances = root["conservation_tolerances"]
    if type(tolerances) is not dict or not tolerances:
        raise PhysicsProvenanceError("conservation tolerances are invalid")
    for name, bound in tolerances.items():
        if name == "rationale":
            if type(bound) is not str or not bound:
                raise PhysicsProvenanceError("conservation tolerance rationale is invalid")
            continue
        if not _is_number(bound) or bound <= 0.0:
            raise PhysicsProvenanceError(f"conservation tolerance {name} must be positive")

    sensitivity = _exact(
        root["sensitivity"],
        {"script", "evidence", "method", "decision_outputs", "reversal_flag_rule"},
        "sensitivity",
    )
    if (
        type(sensitivity["script"]) is not str
        or type(sensitivity["method"]) is not str
        or type(sensitivity["reversal_flag_rule"]) is not str
        or type(sensitivity["decision_outputs"]) is not list
        or not sensitivity["decision_outputs"]
    ):
        raise PhysicsProvenanceError("provenance sensitivity section drifted")
    return dict(root)


def load_physics_provenance_manifest(root: str | Path) -> tuple[dict[str, Any], str]:
    """Load, validate, and hash the frozen physics provenance manifest bytes."""

    path = Path(root).resolve() / "contracts" / PHYSICS_PROVENANCE_FILENAME
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise PhysicsProvenanceError("physics provenance manifest is unreadable") from error
    return validate_physics_provenance_manifest(_strict_json(raw)), hashlib.sha256(raw).hexdigest()


def parameter_by_id(manifest: Mapping[str, Any], parameter_id: str) -> dict[str, Any]:
    for record in manifest["parameters"]:
        if record["parameter_id"] == parameter_id:
            return record
    raise PhysicsProvenanceError(f"unknown provenance parameter {parameter_id!r}")


def parameters_for_system(manifest: Mapping[str, Any], system: str) -> list[dict[str, Any]]:
    if system not in manifest["affected_systems_vocabulary"]:
        raise PhysicsProvenanceError(f"undeclared affected system {system!r}")
    return [
        record
        for record in manifest["parameters"]
        if system in record["affected_systems"]
    ]


def generator_variable_parameters(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        record for record in manifest["parameters"] if record["generator_variable"]
    ]


def sample_band(record: Mapping[str, Any], fraction: float) -> float:
    """Deterministically map a fraction in [-1, 1] onto the declared band.

    Used by the bounded sensitivity run and available to the family generator;
    no randomness is introduced here.
    """

    if not -1.0 <= fraction <= 1.0:
        raise PhysicsProvenanceError("band fraction must lie within [-1, 1]")
    distribution = record["uncertainty_distribution"]
    if distribution is None:
        raise PhysicsProvenanceError(
            f"parameter {record['parameter_id']!r} has no declared band"
        )
    value = float(record["value"])
    if distribution["kind"] == "uniform_relative":
        low = value * (1.0 + float(distribution["low"]))
        high = value * (1.0 + float(distribution["high"]))
    else:
        low = float(distribution["low"])
        high = float(distribution["high"])
    if fraction <= 0.0:
        return low + (value - low) * (fraction + 1.0)
    return value + (high - value) * fraction


__all__ = [
    "PHYSICS_PROVENANCE_FILENAME",
    "PHYSICS_PROVENANCE_MANIFEST_ID",
    "PHYSICS_PROVENANCE_SCHEMA_VERSION",
    "PHYSICS_PROVENANCE_STATUS",
    "PROVENANCE_CLASSIFICATIONS",
    "PhysicsProvenanceError",
    "generator_variable_parameters",
    "load_physics_provenance_manifest",
    "parameter_by_id",
    "parameters_for_system",
    "sample_band",
    "validate_physics_provenance_manifest",
]
