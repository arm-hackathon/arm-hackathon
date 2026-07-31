"""Strict scenario-family manifests for leakage-safe AEOLUS corpus v2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from aeolus.config import (
    BlockedPath,
    FrozenSensor,
    GradualPrimaryFanDegradation,
    HabitatConfig,
    load_scenario,
)
from aeolus.model_input import (
    build_model_input_contract,
    model_artifact_metadata,
    model_input_v1,
)
from aeolus.scenario import run_scenario


FAMILY_MANIFEST_VERSION = "aeolus_family_manifest_v1"
_SPLITS = frozenset({"train", "validation", "test"})
_TOP_LEVEL_FIELDS = frozenset(
    {
        "families",
        "model_input_version",
        "schema_version",
        "selector_sha256",
        "topology_sha256",
    }
)
_FAMILY_FIELDS = frozenset(
    {
        "family_id",
        "fault_class",
        "fault_scenario",
        "reference_scenario",
        "split",
    }
)
_FAULT_CLASSES = {
    GradualPrimaryFanDegradation: "gradual_primary_fan_degradation",
    BlockedPath: "blocked_path",
    FrozenSensor: "frozen_sensor",
}


@dataclass(frozen=True)
class ScenarioFamily:
    """One reference/fault pair assigned to a single evaluation split."""

    family_id: str
    split: str
    fault_class: str
    reference_path: Path
    fault_path: Path

    def canonical_representation(self) -> dict[str, str]:
        """Return the canonical, path-independent manifest representation."""
        return {
            "family_id": self.family_id,
            "fault_class": self.fault_class,
            "fault_scenario": self.fault_path.name,
            "reference_scenario": self.reference_path.name,
            "split": self.split,
        }


@dataclass(frozen=True)
class FamilyManifest:
    """Validated family split and Gate-1 contract binding."""

    families: tuple[ScenarioFamily, ...]
    contract_metadata: dict[str, str]
    canonical_json: str
    manifest_sha256: str


def load_family_manifest(path: Path) -> FamilyManifest:
    """Load and validate a strict family manifest relative to its directory."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"family manifest file not found: {path}") from None
    except OSError as exc:
        raise ValueError(f"cannot read family manifest {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"family manifest is not valid JSON: {exc}") from None
    return parse_family_manifest(document, base_dir=path.parent)


def parse_family_manifest(document: object, *, base_dir: Path) -> FamilyManifest:
    """Validate a manifest and bind every pair to the actual Gate-1 contract."""
    if not isinstance(document, dict):
        raise ValueError("family manifest must be a JSON object")
    _reject_unknown_fields(document, _TOP_LEVEL_FIELDS, "family manifest")
    if document.get("schema_version") != FAMILY_MANIFEST_VERSION:
        raise ValueError("family manifest schema_version is unsupported")

    model_input_version = document.get("model_input_version")
    selector_sha256 = document.get("selector_sha256")
    topology_sha256 = document.get("topology_sha256")
    if (
        not isinstance(model_input_version, str)
        or not isinstance(selector_sha256, str)
        or not isinstance(topology_sha256, str)
    ):
        raise ValueError("family manifest contract metadata values must be strings")
    declared_metadata: dict[str, str] = {
        "model_input_version": model_input_version,
        "selector_sha256": selector_sha256,
        "topology_sha256": topology_sha256,
    }
    if not _is_sha256(selector_sha256) or not _is_sha256(topology_sha256):
        raise ValueError("family manifest contract hashes must be lowercase SHA-256")

    raw_families = document.get("families")
    if not isinstance(raw_families, list) or not raw_families:
        raise ValueError("family manifest families must be a non-empty list")

    families: list[ScenarioFamily] = []
    seen_ids: set[str] = set()
    scenario_splits: dict[Path, str] = {}
    for raw_family in raw_families:
        family = _parse_family(raw_family, base_dir=base_dir)
        if family.family_id in seen_ids:
            raise ValueError(f"duplicate family_id {family.family_id!r}")
        seen_ids.add(family.family_id)
        _validate_family_pair(family, declared_metadata)
        for scenario_path in (family.reference_path, family.fault_path):
            existing_split = scenario_splits.setdefault(scenario_path, family.split)
            if existing_split != family.split:
                raise ValueError("scenario is assigned to more than one split")
        families.append(family)

    ordered_families = tuple(sorted(families, key=lambda family: family.family_id))
    canonical_document = {
        "families": [family.canonical_representation() for family in ordered_families],
        "model_input_version": declared_metadata["model_input_version"],
        "schema_version": FAMILY_MANIFEST_VERSION,
        "selector_sha256": declared_metadata["selector_sha256"],
        "topology_sha256": declared_metadata["topology_sha256"],
    }
    canonical_json = _canonical_json(canonical_document)
    return FamilyManifest(
        families=ordered_families,
        contract_metadata=dict(declared_metadata),
        canonical_json=canonical_json,
        manifest_sha256=_sha256(canonical_json),
    )


@dataclass(frozen=True)
class ObservableOnset:
    """Auditable first divergence of a fault replay from its paired reference."""

    family_id: str
    tick: int
    contract_metadata: dict[str, str]
    reference_scenario_sha256: str
    fault_scenario_sha256: str


def observable_onset(
    family: ScenarioFamily, contract_metadata: Mapping[str, str]
) -> ObservableOnset:
    """Return the first paired tick whose ``model_input_v1`` vector differs."""
    reference_config = load_scenario(family.reference_path)
    fault_config = load_scenario(family.fault_path)
    contract = build_model_input_contract(reference_config)
    expected_metadata = model_artifact_metadata(contract)
    if dict(contract_metadata) != expected_metadata:
        raise ValueError("observable onset contract metadata does not match reference")
    if model_artifact_metadata(build_model_input_contract(fault_config)) != expected_metadata:
        raise ValueError("observable onset scenarios do not share one model input contract")

    reference_records = run_scenario(reference_config)
    fault_records = run_scenario(fault_config)
    if len(reference_records) != len(fault_records):
        raise ValueError("observable onset scenarios have different trace lengths")
    for reference_record, fault_record in zip(reference_records, fault_records):
        if reference_record.tick != fault_record.tick:
            raise ValueError("observable onset scenarios have different tick numbering")
        reference_input = model_input_v1(reference_record, contract)
        fault_input = model_input_v1(fault_record, contract)
        if not np.array_equal(reference_input, fault_input):
            return ObservableOnset(
                family_id=family.family_id,
                tick=reference_record.tick,
                contract_metadata=expected_metadata,
                reference_scenario_sha256=hashlib.sha256(
                    family.reference_path.read_bytes()
                ).hexdigest(),
                fault_scenario_sha256=hashlib.sha256(
                    family.fault_path.read_bytes()
                ).hexdigest(),
            )
    raise ValueError(f"family {family.family_id!r} never becomes model-input observable")


def _parse_family(raw_family: object, *, base_dir: Path) -> ScenarioFamily:
    if not isinstance(raw_family, dict):
        raise ValueError("family entry must be a JSON object")
    _reject_unknown_fields(raw_family, _FAMILY_FIELDS, "family entry")
    required = _FAMILY_FIELDS - set(raw_family)
    if required:
        raise ValueError(f"family entry is missing required fields {sorted(required)!r}")

    family_id = raw_family["family_id"]
    split = raw_family["split"]
    fault_class = raw_family["fault_class"]
    if not isinstance(family_id, str) or not family_id:
        raise ValueError("family_id must be a non-empty string")
    if split not in _SPLITS:
        raise ValueError(f"family split must be one of {sorted(_SPLITS)!r}")
    if fault_class not in set(_FAULT_CLASSES.values()):
        raise ValueError(f"unsupported family fault_class {fault_class!r}")

    return ScenarioFamily(
        family_id=family_id,
        split=split,
        fault_class=fault_class,
        reference_path=_scenario_path(base_dir, raw_family["reference_scenario"]),
        fault_path=_scenario_path(base_dir, raw_family["fault_scenario"]),
    )


def _scenario_path(base_dir: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path.endswith(".json"):
        raise ValueError("family scenario paths must be JSON file names")
    path = Path(raw_path)
    if path.name != raw_path:
        raise ValueError("family scenario paths must not contain directories")
    return base_dir / path


def _validate_family_pair(
    family: ScenarioFamily, declared_metadata: Mapping[str, str]
) -> None:
    reference_config = load_scenario(family.reference_path)
    fault_config = load_scenario(family.fault_path)
    if reference_config.fault_profiles:
        raise ValueError("family reference scenario must declare no fault profiles")
    if len(fault_config.fault_profiles) != 1:
        raise ValueError("family fault scenario must declare exactly one fault profile")

    expected_class = _FAULT_CLASSES[type(fault_config.fault_profiles[0])]
    if family.fault_class != expected_class:
        raise ValueError("family fault_class does not match its paired fault scenario")
    if not _same_non_fault_configuration(reference_config, fault_config):
        raise ValueError("family reference and fault scenarios differ outside fault profiles")

    for config in (reference_config, fault_config):
        actual_metadata = model_artifact_metadata(build_model_input_contract(config))
        if actual_metadata != dict(declared_metadata):
            raise ValueError("family manifest contract metadata does not match scenario topology")


def _same_non_fault_configuration(
    reference: HabitatConfig, fault: HabitatConfig
) -> bool:
    """Compare every validated scenario field except fault profiles."""
    return (
        reference.version == fault.version
        and reference.zones == fault.zones
        and reference.connections == fault.connections
        and reference.control == fault.control
        and reference.actuator == fault.actuator
        and reference.simulation == fault.simulation
        and reference.air_system == fault.air_system
    )


def _reject_unknown_fields(
    value: Mapping[str, Any], allowed: frozenset[str], subject: str
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{subject} has unknown fields {sorted(unknown)!r}")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
