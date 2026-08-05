"""Strict V6 generated-room-family manifest validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from aeolus.config import load_scenario
from aeolus.observable_context import OBSERVABLE_CONTEXT_VERSION
from aeolus.sweep_v6 import V6RoomFamily, V6SweepSpec

V6_GENERATED_MANIFEST_VERSION = "aeolus_family_manifest_v6"
_ROLES = frozenset({"fit", "calibration", "validation"})
_FAULT_CLASSES = frozenset(
    {"frozen_sensor", "blocked_path", "gradual_primary_fan_degradation"}
)
_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "sweep_spec_sha256", "observable_context", "families"}
)
_FAMILY_KEYS = frozenset(
    {
        "family_id",
        "room_family_id",
        "role",
        "fault_class",
        "reference_scenario",
        "fault_scenario",
        "base_scenario_sha256",
        "reference_scenario_sha256",
        "fault_scenario_sha256",
    }
)
_CONTEXT_KEYS = frozenset(
    {"observable_context_version", "selector_sha256", "topology_sha256"}
)


@dataclass(frozen=True)
class V6ScenarioFamily:
    """One validated generated V6 reference/fault pair."""

    family_id: str
    room_family_id: str
    role: str
    fault_class: str
    reference_path: Path
    fault_path: Path
    base_scenario_sha256: str
    reference_scenario_sha256: str
    fault_scenario_sha256: str

    def canonical_representation(self) -> dict[str, str]:
        """Return a path-independent identity record."""
        return {
            "family_id": self.family_id,
            "room_family_id": self.room_family_id,
            "role": self.role,
            "fault_class": self.fault_class,
            "reference_scenario": self.reference_path.name,
            "fault_scenario": self.fault_path.name,
            "base_scenario_sha256": self.base_scenario_sha256,
            "reference_scenario_sha256": self.reference_scenario_sha256,
            "fault_scenario_sha256": self.fault_scenario_sha256,
        }


@dataclass(frozen=True)
class V6FamilyManifest:
    """V6 room-family manifest bound to a verified sweep and exact files."""

    sweep_spec_sha256: str
    observable_context: dict[str, str]
    families: tuple[V6ScenarioFamily, ...]
    canonical_json: str
    manifest_sha256: str

    @property
    def family_count(self) -> int:
        return len(self.families)

    @property
    def families_by_role(self) -> Mapping[str, int]:
        return {
            role: sum(family.role == role for family in self.families)
            for role in sorted(_ROLES)
        }


def load_v6_family_manifest(path: str | Path, *, expected_sweep: V6SweepSpec) -> V6FamilyManifest:
    """Load and fail closed on V6 room/sweep/context/scenario identity drift."""
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"V6 family manifest not found: {source}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read V6 family manifest: {exc}") from None
    return parse_v6_family_manifest(document, base_dir=source.parent, expected_sweep=expected_sweep)


def parse_v6_family_manifest(
    document: object, *, base_dir: Path, expected_sweep: V6SweepSpec
) -> V6FamilyManifest:
    """Validate generated V6 pairs against their source-controlled sweep."""
    if not isinstance(expected_sweep, V6SweepSpec):
        raise ValueError("V6 family manifest requires a validated expected sweep")
    if not isinstance(document, dict) or set(document) != _TOP_LEVEL_KEYS:
        raise ValueError("V6 family manifest has unknown or missing fields")
    if document.get("schema_version") != V6_GENERATED_MANIFEST_VERSION:
        raise ValueError("V6 family manifest schema_version is unsupported")
    if document.get("sweep_spec_sha256") != expected_sweep.sha256:
        raise ValueError("V6 family manifest sweep digest does not match expected sweep")
    context = _validate_context(document.get("observable_context"), expected_sweep)
    raw_families = document.get("families")
    if not isinstance(raw_families, list) or not raw_families:
        raise ValueError("V6 family manifest families must be a non-empty list")

    expected_rooms = {
        family.room_family_id: family for family in expected_sweep.room_families
    }
    families: list[V6ScenarioFamily] = []
    seen_family_ids: set[str] = set()
    seen_fault_paths: set[str] = set()
    for raw in raw_families:
        family = _parse_family(raw, base_dir=base_dir, expected_rooms=expected_rooms)
        if family.family_id in seen_family_ids:
            raise ValueError("V6 family manifest has duplicate family_id")
        if family.fault_path.name in seen_fault_paths:
            raise ValueError("V6 family manifest reuses a fault scenario")
        seen_family_ids.add(family.family_id)
        seen_fault_paths.add(family.fault_path.name)
        families.append(family)

    roles = {family.role for family in families}
    if roles != _ROLES:
        raise ValueError("V6 family manifest is missing a development role")
    ordered = tuple(sorted(families, key=lambda family: family.family_id))
    canonical_json = _canonical_json(
        {
            "schema_version": V6_GENERATED_MANIFEST_VERSION,
            "sweep_spec_sha256": expected_sweep.sha256,
            "observable_context": context,
            "families": [family.canonical_representation() for family in ordered],
        }
    )
    return V6FamilyManifest(
        sweep_spec_sha256=expected_sweep.sha256,
        observable_context=context,
        families=ordered,
        canonical_json=canonical_json,
        manifest_sha256=_sha256(canonical_json),
    )


def _validate_context(value: object, expected_sweep: V6SweepSpec) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != _CONTEXT_KEYS:
        raise ValueError("V6 family manifest observable context is malformed")
    if any(not isinstance(item, str) for item in value.values()):
        raise ValueError("V6 family manifest observable context values must be strings")
    expected = expected_sweep.room_families[0].context_metadata
    if value != expected or value["observable_context_version"] != OBSERVABLE_CONTEXT_VERSION:
        raise ValueError("V6 family manifest observable context does not match sweep")
    return dict(value)


def _parse_family(
    raw: object, *, base_dir: Path, expected_rooms: Mapping[str, V6RoomFamily]
) -> V6ScenarioFamily:
    if not isinstance(raw, dict) or set(raw) != _FAMILY_KEYS:
        raise ValueError("V6 family manifest entry has unknown or missing fields")
    values = {key: raw[key] for key in _FAMILY_KEYS}
    for key in ("family_id", "room_family_id", "role", "fault_class"):
        if not isinstance(values[key], str) or not values[key]:
            raise ValueError(f"V6 family manifest {key} is malformed")
    if values["role"] not in _ROLES or values["fault_class"] not in _FAULT_CLASSES:
        raise ValueError("V6 family manifest role or fault class is unsupported")
    room = expected_rooms.get(values["room_family_id"])
    if room is None or room.role != values["role"]:
        raise ValueError("V6 family manifest room family does not match sweep role")
    if values["base_scenario_sha256"] != room.base_scenario_sha256:
        raise ValueError("V6 family manifest base scenario digest does not match sweep")
    for key in (
        "base_scenario_sha256",
        "reference_scenario_sha256",
        "fault_scenario_sha256",
    ):
        if not _is_sha256(values[key]):
            raise ValueError(f"V6 family manifest {key} is not a lowercase SHA-256")
    reference_path = _scenario_path(base_dir, values["reference_scenario"])
    fault_path = _scenario_path(base_dir, values["fault_scenario"])
    if _file_sha256(reference_path) != values["reference_scenario_sha256"]:
        raise ValueError("V6 family manifest reference scenario digest does not match bytes")
    if _file_sha256(fault_path) != values["fault_scenario_sha256"]:
        raise ValueError("V6 family manifest fault scenario digest does not match bytes")
    _validate_pair(reference_path, fault_path, values["fault_class"])
    return V6ScenarioFamily(
        family_id=values["family_id"],
        room_family_id=values["room_family_id"],
        role=values["role"],
        fault_class=values["fault_class"],
        reference_path=reference_path,
        fault_path=fault_path,
        base_scenario_sha256=values["base_scenario_sha256"],
        reference_scenario_sha256=values["reference_scenario_sha256"],
        fault_scenario_sha256=values["fault_scenario_sha256"],
    )


def _scenario_path(base_dir: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.endswith(".json"):
        raise ValueError("V6 family manifest scenario path is malformed")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative.name != value:
        raise ValueError("V6 family manifest scenario path must be a file name")
    path = base_dir / relative
    if not path.is_file():
        raise ValueError(f"V6 family manifest scenario is missing: {relative}")
    return path


def _validate_pair(reference_path: Path, fault_path: Path, fault_class: str) -> None:
    try:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        fault = json.loads(fault_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot validate V6 scenario pair: {exc}") from None
    if not isinstance(reference, dict) or not isinstance(fault, dict):
        raise ValueError("V6 scenario pair documents must be objects")
    reference_without_profiles = dict(reference)
    reference_profiles = reference_without_profiles.pop("fault_profiles", None)
    if reference_profiles != []:
        raise ValueError("V6 reference scenario must not declare faults")
    fault_without_profiles = dict(fault)
    fault_without_profiles.pop("fault_profiles", None)
    if reference_without_profiles != fault_without_profiles:
        raise ValueError("V6 fault/reference pair differs outside fault_profiles")
    fault_config = load_scenario(fault_path)
    if len(fault_config.fault_profiles) != 1:
        raise ValueError("V6 fault scenario must have exactly one fault profile")
    profile = fault_config.fault_profiles[0]
    actual = type(profile).__name__
    expected_type = {
        "frozen_sensor": "FrozenSensor",
        "blocked_path": "BlockedPath",
        "gradual_primary_fan_degradation": "GradualPrimaryFanDegradation",
    }[fault_class]
    if actual != expected_type:
        raise ValueError("V6 manifest fault_class does not match fault scenario")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
