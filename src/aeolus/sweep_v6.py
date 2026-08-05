"""Strict V6 room-physics family sweep generation."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aeolus.config import load_scenario, parse_scenario
from aeolus.observable_context import (
    build_observable_context_contract,
    observable_context_metadata,
)

V6_SWEEP_VERSION = "aeolus_sweep_v6"
V6_FAMILY_MANIFEST_VERSION = "aeolus_family_manifest_v6"
_V6_TOP_LEVEL_KEYS = frozenset({"schema_version", "suite_role", "targets", "room_families"})
_V6_FAMILY_KEYS = frozenset(
    {
        "id",
        "role",
        "base_scenario",
        "seeds",
        "fault_start_ticks",
        "operating_profiles",
        "gradual_profiles",
        "blocked_effectiveness",
    }
)
_PROFILE_KEYS = frozenset({"id", "source_multiplier", "shared_airflow_capacity", "telemetry"})
_TELEMETRY_KEYS = frozenset(
    {
        "airflow_noise_fraction",
        "airflow_bias_fraction",
        "airflow_drift_fraction",
        "actuator_position_noise_fraction",
        "co2_sensor_noise_fraction",
        "co2_sensor_bias_fraction",
        "co2_sensor_drift_fraction",
    }
)
_GRADUAL_KEYS = frozenset({"duration_ticks", "end_effectiveness"})
_ROLES = frozenset({"fit", "calibration", "validation"})


@dataclass(frozen=True)
class V6OperatingProfile:
    """A predeclared observable operating regime for one room family."""

    profile_id: str
    source_multiplier: float
    shared_airflow_capacity: float
    telemetry: dict[str, float]


@dataclass(frozen=True)
class V6GradualProfile:
    """A deterministic gradual-fault severity declaration."""

    duration_ticks: int
    end_effectiveness: float


@dataclass(frozen=True)
class V6RoomFamily:
    """One room-physics base scenario allocated to one development role."""

    room_family_id: str
    role: str
    base_scenario_path: Path
    base_scenario_sha256: str
    context_metadata: dict[str, str]
    seeds: tuple[int, ...]
    fault_start_ticks: tuple[int, ...]
    operating_profiles: tuple[V6OperatingProfile, ...]
    gradual_profiles: tuple[V6GradualProfile, ...]
    blocked_effectiveness: tuple[float, ...]


@dataclass(frozen=True)
class V6SweepSpec:
    """Validated V6 development sweep, isolated by room-physics family."""

    source_path: Path
    targets: tuple[str, ...]
    room_families: tuple[V6RoomFamily, ...]
    canonical_json: str
    sha256: str


def load_v6_sweep_spec(path: str | Path) -> V6SweepSpec:
    """Load one strict V6 sweep specification."""
    source_path = Path(path)
    try:
        document = json.loads(source_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"V6 sweep specification not found: {source_path}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read V6 sweep specification: {exc}") from None
    return parse_v6_sweep_spec(document, source_path=source_path)


def parse_v6_sweep_spec(document: object, *, source_path: Path) -> V6SweepSpec:
    """Fail closed on V6 room identity, role, seed, or context leakage."""
    if not isinstance(document, dict) or set(document) != _V6_TOP_LEVEL_KEYS:
        raise ValueError("V6 sweep specification has unknown or missing fields")
    if document.get("schema_version") != V6_SWEEP_VERSION:
        raise ValueError("V6 sweep schema_version is unsupported")
    if document.get("suite_role") != "development":
        raise ValueError("V6 sweep suite_role must be development")
    targets = _parse_targets(document.get("targets"))
    raw_families = document.get("room_families")
    if not isinstance(raw_families, list) or not raw_families:
        raise ValueError("V6 room_families must be a non-empty list")

    room_families: list[V6RoomFamily] = []
    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    for raw_family in raw_families:
        family = _parse_room_family(raw_family, source_path.parent, targets)
        if family.room_family_id in seen_ids:
            raise ValueError("duplicate room family identity")
        seen_ids.add(family.room_family_id)
        overlap = seen_seeds.intersection(family.seeds)
        if overlap:
            raise ValueError("V6 seed cluster is reused across room families")
        seen_seeds.update(family.seeds)
        room_families.append(family)
    roles = {family.role for family in room_families}
    if roles != _ROLES:
        raise ValueError("V6 room families must include fit, calibration, and validation roles")
    metadata = {tuple(sorted(family.context_metadata.items())) for family in room_families}
    if len(metadata) != 1:
        raise ValueError("V6 room families do not share one observable context contract")

    ordered = tuple(sorted(room_families, key=lambda item: item.room_family_id))
    canonical_json = _canonical_json(document)
    return V6SweepSpec(
        source_path=source_path,
        targets=targets,
        room_families=ordered,
        canonical_json=canonical_json,
        sha256=_sha256(canonical_json),
    )


def generate_v6_sweep(spec_path: str | Path, output_dir: str | Path) -> dict[str, object]:
    """Generate paired V6 scenarios while preserving each room's base physics."""
    spec = load_v6_sweep_spec(spec_path)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"V6 sweep output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    families: list[dict[str, str]] = []
    counts = {role: 0 for role in sorted(_ROLES)}
    names: set[str] = set()
    for room in spec.room_families:
        base_document = _load_document(room.base_scenario_path)
        base_config = parse_scenario(base_document)
        for seed in room.seeds:
            for profile in room.operating_profiles:
                reference = _reference_document(base_document, seed, profile)
                reference_name = f"{room.role}-{room.room_family_id}-s{seed}-{profile.profile_id}-reference.json"
                _write_scenario(destination, reference_name, reference, names)
                for start_tick in room.fault_start_ticks:
                    for target in spec.targets:
                        connection_id = base_config.path_to_processing(target).id
                        for gradual in room.gradual_profiles:
                            fault = copy.deepcopy(reference)
                            fault["fault_profiles"] = [{
                                "type": "gradual_primary_fan_degradation",
                                "connection_id": connection_id,
                                "start_tick": start_tick,
                                "end_tick": start_tick + gradual.duration_ticks,
                                "end_effectiveness": gradual.end_effectiveness,
                            }]
                            _append_family(destination, families, names, room, reference_name, fault, start_tick, target, "gradual", "gradual_primary_fan_degradation")
                            counts[room.role] += 1
                        for effectiveness in room.blocked_effectiveness:
                            fault = copy.deepcopy(reference)
                            fault["fault_profiles"] = [{
                                "type": "blocked_path",
                                "connection_id": connection_id,
                                "start_tick": start_tick,
                                "blocked_effectiveness": effectiveness,
                            }]
                            _append_family(destination, families, names, room, reference_name, fault, start_tick, target, "blocked", "blocked_path")
                            counts[room.role] += 1
                        fault = copy.deepcopy(reference)
                        fault["fault_profiles"] = [{"type": "frozen_sensor", "zone_id": target, "start_tick": start_tick}]
                        _append_family(destination, families, names, room, reference_name, fault, start_tick, target, "frozen", "frozen_sensor")
                        counts[room.role] += 1

    context_metadata = spec.room_families[0].context_metadata
    manifest = {
        "schema_version": V6_FAMILY_MANIFEST_VERSION,
        "sweep_spec_sha256": spec.sha256,
        "observable_context": context_metadata,
        "families": sorted(families, key=lambda item: item["family_id"]),
    }
    manifest_json = _canonical_json(manifest)
    _write_json(destination / "families-v6.json", manifest)
    receipt = {
        "schema_version": V6_SWEEP_VERSION,
        "sweep_spec_sha256": spec.sha256,
        "family_manifest_sha256": _sha256(manifest_json),
        "families_by_role": counts,
        "total_families": sum(counts.values()),
        "generated_scenario_files": len(names),
        "observable_context": context_metadata,
    }
    _write_json(destination / "sweep-v6-receipt.json", receipt)
    return receipt


def _parse_room_family(raw: object, base_dir: Path, targets: tuple[str, ...]) -> V6RoomFamily:
    if not isinstance(raw, dict) or set(raw) != _V6_FAMILY_KEYS:
        raise ValueError("V6 room family has unknown or missing fields")
    family_id = raw["id"]
    role = raw["role"]
    base_name = raw["base_scenario"]
    if not isinstance(family_id, str) or not family_id:
        raise ValueError("V6 room family id must be a non-empty string")
    if role not in _ROLES:
        raise ValueError("V6 room family role is unsupported")
    if not isinstance(base_name, str) or not base_name.endswith(".json"):
        raise ValueError("V6 room family base_scenario must be a relative JSON path")
    relative_path = Path(base_name)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("V6 room family base_scenario must be a safe relative JSON path")
    base_path = base_dir / relative_path
    config = load_scenario(base_path)
    if config.fault_profiles:
        raise ValueError("V6 room family base scenario must not declare faults")
    zone_ids = {zone.id for zone in config.non_processing_zones()}
    if not set(targets).issubset(zone_ids):
        raise ValueError("V6 sweep targets are absent from a room-family base scenario")
    context_metadata = observable_context_metadata(build_observable_context_contract(config))
    return V6RoomFamily(
        room_family_id=family_id,
        role=role,
        base_scenario_path=base_path,
        base_scenario_sha256=_sha256(_canonical_json(_load_document(base_path))),
        context_metadata=context_metadata,
        seeds=_positive_unique_ints(raw["seeds"], "seeds"),
        fault_start_ticks=_positive_unique_ints(raw["fault_start_ticks"], "fault_start_ticks"),
        operating_profiles=_parse_profiles(raw["operating_profiles"]),
        gradual_profiles=_parse_gradual(raw["gradual_profiles"]),
        blocked_effectiveness=_positive_unique_floats(raw["blocked_effectiveness"], "blocked_effectiveness"),
    )


def _parse_targets(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ValueError("V6 targets must be a non-empty list of identifiers")
    if len(set(value)) != len(value):
        raise ValueError("V6 targets must be unique")
    return tuple(value)


def _parse_profiles(value: object) -> tuple[V6OperatingProfile, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("V6 operating_profiles must be non-empty")
    profiles: list[V6OperatingProfile] = []
    ids: set[str] = set()
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != _PROFILE_KEYS:
            raise ValueError("V6 operating profile has unknown or missing fields")
        profile_id = raw["id"]
        telemetry = raw["telemetry"]
        if not isinstance(profile_id, str) or not profile_id or profile_id in ids:
            raise ValueError("V6 operating profile ids must be unique non-empty strings")
        if not isinstance(telemetry, dict) or set(telemetry) != _TELEMETRY_KEYS:
            raise ValueError("V6 operating profile telemetry is malformed")
        ids.add(profile_id)
        profiles.append(V6OperatingProfile(profile_id, _positive_float(raw["source_multiplier"], "source_multiplier"), _positive_float(raw["shared_airflow_capacity"], "shared_airflow_capacity"), {key: _nonnegative_float(number, key) for key, number in telemetry.items()}))
    return tuple(profiles)


def _parse_gradual(value: object) -> tuple[V6GradualProfile, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("V6 gradual_profiles must be non-empty")
    result: list[V6GradualProfile] = []
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != _GRADUAL_KEYS:
            raise ValueError("V6 gradual profile is malformed")
        duration = raw["duration_ticks"]
        if not isinstance(duration, int) or isinstance(duration, bool) or duration < 1:
            raise ValueError("V6 gradual duration_ticks must be a positive integer")
        result.append(V6GradualProfile(duration, _effectiveness(raw["end_effectiveness"], "end_effectiveness")))
    return tuple(result)


def _positive_unique_ints(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in value) or len(set(value)) != len(value):
        raise ValueError(f"V6 {name} must be unique positive integers")
    return tuple(value)


def _positive_unique_floats(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"V6 {name} must be non-empty")
    parsed = tuple(_effectiveness(item, name) for item in value)
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"V6 {name} must be unique")
    return parsed


def _positive_float(value: object, name: str) -> float:
    if not isinstance(value, (float, int)) or isinstance(value, bool) or float(value) <= 0:
        raise ValueError(f"V6 {name} must be positive")
    return float(value)


def _nonnegative_float(value: object, name: str) -> float:
    if not isinstance(value, (float, int)) or isinstance(value, bool) or float(value) < 0:
        raise ValueError(f"V6 {name} must be non-negative")
    return float(value)


def _effectiveness(value: object, name: str) -> float:
    number = _positive_float(value, name)
    if number > 1:
        raise ValueError(f"V6 {name} must be at most one")
    return number


def _reference_document(base: dict[str, Any], seed: int, profile: V6OperatingProfile) -> dict[str, Any]:
    document = copy.deepcopy(base)
    document["simulation"]["random_seed"] = seed
    document["telemetry"] = dict(profile.telemetry)
    document["air_system"]["shared_airflow_capacity"] = profile.shared_airflow_capacity
    document["fault_profiles"] = []
    for zone in document["zones"]:
        if zone["preset"] != "air_processing":
            zone["co2_generation_per_second"] *= profile.source_multiplier
            zone["co2_generation_epsilon"] *= profile.source_multiplier
    parse_scenario(document)
    return document


def _append_family(destination: Path, families: list[dict[str, str]], names: set[str], room: V6RoomFamily, reference_name: str, fault: dict[str, Any], start_tick: int, target: str, fault_suffix: str, fault_class: str) -> None:
    family_id = f"{room.role}-{room.room_family_id}-s{fault['simulation']['random_seed']}-{start_tick}-{target}-{fault_suffix}"
    fault_name = f"{family_id}.json"
    _write_scenario(destination, fault_name, fault, names)
    families.append(
        {
            "family_id": family_id,
            "room_family_id": room.room_family_id,
            "role": room.role,
            "fault_class": fault_class,
            "reference_scenario": reference_name,
            "fault_scenario": fault_name,
            "base_scenario_sha256": room.base_scenario_sha256,
            "reference_scenario_sha256": hashlib.sha256(
                (destination / reference_name).read_bytes()
            ).hexdigest(),
            "fault_scenario_sha256": hashlib.sha256(
                (destination / fault_name).read_bytes()
            ).hexdigest(),
        }
    )


def _write_scenario(destination: Path, name: str, document: dict[str, Any], names: set[str]) -> None:
    if name in names:
        return
    parse_scenario(document)
    _write_json(destination / name, document)
    names.add(name)


def _load_document(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read scenario {path}: {exc}") from None
    if not isinstance(value, dict):
        raise ValueError("scenario must be a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical_json(value) + "\n", encoding="utf-8")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
