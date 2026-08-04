"""Deterministic scenario-family sweeps for AEOLUS model experiments."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from aeolus.config import HabitatConfig, load_scenario, parse_scenario
from aeolus.families import FAMILY_MANIFEST_VERSION, load_family_manifest
from aeolus.model_input import build_model_input_contract, model_artifact_metadata

SWEEP_VERSION = "aeolus_sweep_v1"
SWEEP_V2_VERSION = "aeolus_sweep_v2"
SWEEP_V3_VERSION = "aeolus_sweep_v3"
SWEEP_V4_VERSION = "aeolus_sweep_v4"
SUPPORTED_SWEEP_VERSIONS = frozenset(
    {SWEEP_VERSION, SWEEP_V2_VERSION, SWEEP_V3_VERSION, SWEEP_V4_VERSION}
)
USAGE = "Usage: PYTHONPATH=src python -m aeolus.sweep <sweep.json> <output-dir>"
_PRIMARY_SPLITS = ("train", "validation", "test")
_ALL_SPLITS = (*_PRIMARY_SPLITS, "stress")
_TOP_LEVEL_KEYS = frozenset({"schema_version", "base_scenario", "targets", "splits"})
_V3_TOP_LEVEL_KEYS = _TOP_LEVEL_KEYS | {"suite_role"}
_V3_SPLITS_BY_ROLE = {
    "development": ("train", "validation"),
    "final": ("final",),
}
_V4_SPLITS_BY_ROLE = {"development": ("train", "validation")}
_SPLIT_V1_KEYS = frozenset(
    {
        "seeds",
        "fault_start_ticks",
        "operating_profiles",
        "gradual_end_effectiveness",
        "blocked_effectiveness",
    }
)
_SPLIT_V2_KEYS = frozenset(
    {
        "seeds",
        "fault_start_ticks",
        "operating_profiles",
        "gradual_profiles",
        "blocked_effectiveness",
    }
)
_GRADUAL_PROFILE_KEYS = frozenset({"duration_ticks", "end_effectiveness"})
_PROFILE_KEYS = frozenset(
    {"id", "source_multiplier", "shared_airflow_capacity", "telemetry"}
)
_TELEMETRY_V1_KEYS = frozenset(
    {
        "airflow_noise_fraction",
        "airflow_bias_fraction",
        "actuator_position_noise_fraction",
    }
)
_TELEMETRY_V2_KEYS = frozenset(
    {
        *_TELEMETRY_V1_KEYS,
        "airflow_drift_fraction",
        "co2_sensor_noise_fraction",
        "co2_sensor_bias_fraction",
        "co2_sensor_drift_fraction",
    }
)
_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class OperatingProfile:
    """One observable operating regime used across every fault class."""

    profile_id: str
    source_multiplier: float
    shared_airflow_capacity: float
    telemetry: dict[str, float]


@dataclass(frozen=True)
class GradualSweepProfile:
    """One degradation duration and terminal effectiveness pair."""

    duration_ticks: int
    end_effectiveness: float


@dataclass(frozen=True)
class SplitSweep:
    """The independent parameters assigned to one corpus split."""

    seeds: tuple[int, ...]
    fault_start_ticks: tuple[int, ...]
    operating_profiles: tuple[OperatingProfile, ...]
    gradual_profiles: tuple[GradualSweepProfile, ...]
    blocked_effectiveness: tuple[float, ...]


@dataclass(frozen=True)
class SweepSpec:
    """Validated sweep definition bound to a fault-free base scenario."""

    source_path: Path
    base_scenario_path: Path
    schema_version: str
    suite_role: str | None
    targets: tuple[str, ...]
    splits: dict[str, SplitSweep]
    canonical_json: str
    sha256: str


def load_sweep_spec(path: str | Path) -> SweepSpec:
    """Load and strictly validate a sweep specification."""
    source_path = Path(path)
    try:
        document = json.loads(source_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"sweep specification not found: {source_path}") from None
    except OSError as exc:
        raise ValueError(f"cannot read sweep specification {source_path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"sweep specification is not valid JSON: {exc}") from None
    return parse_sweep_spec(document, source_path=source_path)


def parse_sweep_spec(document: object, *, source_path: Path) -> SweepSpec:
    """Validate an already parsed sweep specification."""
    raw = _require_mapping(document, "sweep specification")
    schema_version = raw.get("schema_version")
    if schema_version not in SUPPORTED_SWEEP_VERSIONS:
        raise ValueError("sweep specification schema_version is unsupported")
    has_suite_role = schema_version in (SWEEP_V3_VERSION, SWEEP_V4_VERSION)
    _require_exact_keys(
        raw,
        _V3_TOP_LEVEL_KEYS if has_suite_role else _TOP_LEVEL_KEYS,
        "sweep specification",
    )
    suite_role: str | None = None
    if schema_version == SWEEP_V3_VERSION:
        suite_role = raw["suite_role"]
        if suite_role not in _V3_SPLITS_BY_ROLE:
            raise ValueError("sweep v3 suite_role must be development or final")
    elif schema_version == SWEEP_V4_VERSION:
        suite_role = raw["suite_role"]
        if suite_role not in _V4_SPLITS_BY_ROLE:
            raise ValueError("sweep v4 suite_role must be development")

    base_name = raw["base_scenario"]
    if not isinstance(base_name, str) or Path(base_name).name != base_name:
        raise ValueError("sweep base_scenario must be a JSON file name")
    if not base_name.endswith(".json"):
        raise ValueError("sweep base_scenario must be a JSON file name")
    base_path = source_path.parent / base_name
    base_config = load_scenario(base_path)
    if base_config.fault_profiles:
        raise ValueError("sweep base scenario must not declare fault profiles")

    targets = _parse_targets(raw["targets"], base_config)
    raw_splits = _require_mapping(raw["splits"], "sweep splits")
    if schema_version == SWEEP_V3_VERSION:
        assert suite_role is not None
        split_names = _V3_SPLITS_BY_ROLE[suite_role]
    elif schema_version == SWEEP_V4_VERSION:
        assert suite_role is not None
        split_names = _V4_SPLITS_BY_ROLE[suite_role]
    else:
        split_names = _ALL_SPLITS if schema_version == SWEEP_V2_VERSION else _PRIMARY_SPLITS
    _require_exact_keys(raw_splits, frozenset(split_names), "sweep splits")
    splits = {
        split: _parse_split(
            raw_splits[split], split=split, schema_version=schema_version
        )
        for split in split_names
    }
    _validate_seed_independence(splits)

    canonical_json = json.dumps(
        raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return SweepSpec(
        source_path=source_path,
        base_scenario_path=base_path,
        schema_version=schema_version,
        suite_role=suite_role,
        targets=targets,
        splits=splits,
        canonical_json=canonical_json,
        sha256=hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
    )


def generate_sweep(spec_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Generate byte-stable paired scenarios and their family manifest."""
    spec = load_sweep_spec(spec_path)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"sweep output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    base_document = json.loads(spec.base_scenario_path.read_text(encoding="utf-8"))
    base_config = parse_scenario(base_document)
    contract_metadata = model_artifact_metadata(build_model_input_contract(base_config))
    families: list[dict[str, str]] = []
    scenario_names: set[str] = set()
    split_counts = {split: 0 for split in spec.splits}

    for split in spec.splits:
        split_spec = spec.splits[split]
        for seed in split_spec.seeds:
            for operating in split_spec.operating_profiles:
                reference = _reference_document(base_document, seed, operating)
                reference_name = (
                    f"{split}-s{seed}-{operating.profile_id}-reference.json"
                )
                _write_scenario(destination, reference_name, reference, scenario_names)

                for start_tick in split_spec.fault_start_ticks:
                    for target in spec.targets:
                        connection_id = base_config.path_to_processing(target).id
                        for gradual in split_spec.gradual_profiles:
                            family_id = _family_id(
                                split,
                                seed,
                                operating.profile_id,
                                start_tick,
                                target,
                                f"degradation-d{gradual.duration_ticks:03d}",
                                gradual.end_effectiveness,
                            )
                            fault = copy.deepcopy(reference)
                            fault["fault_profiles"] = [
                                {
                                    "type": "gradual_primary_fan_degradation",
                                    "connection_id": connection_id,
                                    "start_tick": start_tick,
                                    "end_tick": start_tick + gradual.duration_ticks,
                                    "end_effectiveness": gradual.end_effectiveness,
                                }
                            ]
                            _append_family(
                                destination,
                                families,
                                scenario_names,
                                family_id=family_id,
                                split=split,
                                fault_class="gradual_primary_fan_degradation",
                                reference_name=reference_name,
                                fault=fault,
                            )
                            split_counts[split] += 1

                        for effectiveness in split_spec.blocked_effectiveness:
                            family_id = _family_id(
                                split,
                                seed,
                                operating.profile_id,
                                start_tick,
                                target,
                                "blocked",
                                effectiveness,
                            )
                            fault = copy.deepcopy(reference)
                            fault["fault_profiles"] = [
                                {
                                    "type": "blocked_path",
                                    "connection_id": connection_id,
                                    "start_tick": start_tick,
                                    "blocked_effectiveness": effectiveness,
                                }
                            ]
                            _append_family(
                                destination,
                                families,
                                scenario_names,
                                family_id=family_id,
                                split=split,
                                fault_class="blocked_path",
                                reference_name=reference_name,
                                fault=fault,
                            )
                            split_counts[split] += 1

                        family_id = _family_id(
                            split,
                            seed,
                            operating.profile_id,
                            start_tick,
                            target,
                            "frozen",
                            None,
                        )
                        fault = copy.deepcopy(reference)
                        fault["fault_profiles"] = [
                            {
                                "type": "frozen_sensor",
                                "zone_id": target,
                                "start_tick": start_tick,
                            }
                        ]
                        _append_family(
                            destination,
                            families,
                            scenario_names,
                            family_id=family_id,
                            split=split,
                            fault_class="frozen_sensor",
                            reference_name=reference_name,
                            fault=fault,
                        )
                        split_counts[split] += 1

    family_document = {
        "families": sorted(families, key=lambda family: family["family_id"]),
        **contract_metadata,
        "schema_version": FAMILY_MANIFEST_VERSION,
    }
    family_path = destination / "families.json"
    _write_json(family_path, family_document)
    manifest = load_family_manifest(family_path)
    receipt = {
        "schema_version": spec.schema_version,
        "sweep_spec_sha256": spec.sha256,
        "family_manifest_sha256": manifest.manifest_sha256,
        "families_by_split": split_counts,
        "total_families": sum(split_counts.values()),
        "generated_scenario_files": len(scenario_names),
        **contract_metadata,
    }
    _write_json(destination / "sweep-receipt.json", receipt)
    return receipt


def _reference_document(
    base_document: dict[str, Any], seed: int, operating: OperatingProfile
) -> dict[str, Any]:
    document = copy.deepcopy(base_document)
    document["simulation"]["random_seed"] = seed
    document["telemetry"] = dict(operating.telemetry)
    document["air_system"]["shared_airflow_capacity"] = (
        operating.shared_airflow_capacity
    )
    document["fault_profiles"] = []
    for zone in document["zones"]:
        if zone["preset"] == "air_processing":
            continue
        zone["co2_generation_per_second"] *= operating.source_multiplier
        zone["co2_generation_epsilon"] *= operating.source_multiplier
    parse_scenario(document)
    return document


def _append_family(
    destination: Path,
    families: list[dict[str, str]],
    scenario_names: set[str],
    *,
    family_id: str,
    split: str,
    fault_class: str,
    reference_name: str,
    fault: dict[str, Any],
) -> None:
    fault_name = f"{family_id}.json"
    _write_scenario(destination, fault_name, fault, scenario_names)
    families.append(
        {
            "family_id": family_id,
            "fault_class": fault_class,
            "fault_scenario": fault_name,
            "reference_scenario": reference_name,
            "split": split,
        }
    )


def _write_scenario(
    destination: Path,
    name: str,
    document: dict[str, Any],
    scenario_names: set[str],
) -> None:
    if name in scenario_names:
        return
    parse_scenario(document)
    _write_json(destination / name, document)
    scenario_names.add(name)


def _write_json(path: Path, document: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        )


def _parse_targets(value: object, base: HabitatConfig) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("sweep targets must be a non-empty list")
    if any(not isinstance(target, str) or not target for target in value):
        raise ValueError("sweep targets must contain non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError("sweep targets must not contain duplicates")
    valid = {zone.id for zone in base.non_processing_zones()}
    unexpected = sorted(set(value) - valid)
    if unexpected:
        raise ValueError(f"sweep target is not a controllable zone: {unexpected[0]!r}")
    return tuple(sorted(value))


def _parse_split(
    value: object, *, split: str, schema_version: str
) -> SplitSweep:
    raw = _require_mapping(value, f"sweep split {split!r}")
    split_keys = (
        _SPLIT_V2_KEYS
        if schema_version in (SWEEP_V2_VERSION, SWEEP_V3_VERSION, SWEEP_V4_VERSION)
        else _SPLIT_V1_KEYS
    )
    _require_exact_keys(raw, split_keys, f"sweep split {split!r}")
    seeds = _integer_list(raw["seeds"], f"sweep split {split!r} seeds", minimum=0)
    start_ticks = _integer_list(
        raw["fault_start_ticks"],
        f"sweep split {split!r} fault_start_ticks",
        minimum=1,
        maximum=80,
    )
    profiles_raw = raw["operating_profiles"]
    if not isinstance(profiles_raw, list) or not profiles_raw:
        raise ValueError(f"sweep split {split!r} operating_profiles must be non-empty")
    profiles = tuple(
        _parse_operating_profile(
            profile, split=split, schema_version=schema_version
        )
        for profile in profiles_raw
    )
    if len({profile.profile_id for profile in profiles}) != len(profiles):
        raise ValueError(f"sweep split {split!r} operating profile ids must be unique")
    if schema_version in (SWEEP_V2_VERSION, SWEEP_V3_VERSION, SWEEP_V4_VERSION):
        gradual_raw = raw["gradual_profiles"]
        if not isinstance(gradual_raw, list) or not gradual_raw:
            raise ValueError(
                f"sweep split {split!r} gradual_profiles must be non-empty"
            )
        gradual = tuple(
            sorted(
                (_parse_gradual_profile(item, split=split) for item in gradual_raw),
                key=lambda item: (item.duration_ticks, item.end_effectiveness),
            )
        )
        if len(set(gradual)) != len(gradual):
            raise ValueError(
                f"sweep split {split!r} gradual_profiles must not contain duplicates"
            )
    else:
        gradual = tuple(
            GradualSweepProfile(40, effectiveness)
            for effectiveness in _fraction_list(
                raw["gradual_end_effectiveness"],
                f"sweep split {split!r} gradual_end_effectiveness",
                strict=True,
            )
        )
    blocked = _fraction_list(
        raw["blocked_effectiveness"],
        f"sweep split {split!r} blocked_effectiveness",
        strict=True,
    )
    return SplitSweep(seeds, start_ticks, profiles, gradual, blocked)


def _parse_operating_profile(
    value: object, *, split: str, schema_version: str
) -> OperatingProfile:
    raw = _require_mapping(value, f"sweep split {split!r} operating profile")
    _require_exact_keys(raw, _PROFILE_KEYS, f"sweep split {split!r} operating profile")
    profile_id = raw["id"]
    if not isinstance(profile_id, str) or not _ID_PATTERN.fullmatch(profile_id):
        raise ValueError("sweep operating profile id must be a lowercase slug")
    source_multiplier = _finite_number(raw["source_multiplier"], "source_multiplier")
    capacity = _finite_number(raw["shared_airflow_capacity"], "shared_airflow_capacity")
    if source_multiplier <= 0.0 or capacity <= 0.0:
        raise ValueError("sweep operating profile source and capacity must be positive")
    telemetry_raw = _require_mapping(raw["telemetry"], "sweep telemetry profile")
    telemetry_keys = (
        _TELEMETRY_V2_KEYS
        if schema_version in (SWEEP_V2_VERSION, SWEEP_V3_VERSION, SWEEP_V4_VERSION)
        else _TELEMETRY_V1_KEYS
    )
    _require_exact_keys(telemetry_raw, telemetry_keys, "sweep telemetry profile")
    telemetry = {
        key: _finite_number(telemetry_raw[key], f"sweep telemetry {key}")
        for key in sorted(telemetry_keys)
    }
    if schema_version == SWEEP_VERSION:
        telemetry.update(
            {
                key: 0.0
                for key in sorted(_TELEMETRY_V2_KEYS - _TELEMETRY_V1_KEYS)
            }
        )
    if any(not 0.0 <= value <= 1.0 for value in telemetry.values()):
        raise ValueError("sweep telemetry fractions must be in 0.0..1.0")
    return OperatingProfile(profile_id, source_multiplier, capacity, telemetry)


def _parse_gradual_profile(value: object, *, split: str) -> GradualSweepProfile:
    raw = _require_mapping(value, f"sweep split {split!r} gradual profile")
    _require_exact_keys(
        raw, _GRADUAL_PROFILE_KEYS, f"sweep split {split!r} gradual profile"
    )
    duration = raw["duration_ticks"]
    if isinstance(duration, bool) or not isinstance(duration, int) or duration < 1:
        raise ValueError("sweep gradual duration_ticks must be a positive integer")
    effectiveness = _finite_number(
        raw["end_effectiveness"], "sweep gradual end_effectiveness"
    )
    if not 0.0 < effectiveness < 1.0:
        raise ValueError("sweep gradual end_effectiveness must be between zero and one")
    return GradualSweepProfile(duration, effectiveness)


def _validate_seed_independence(splits: Mapping[str, SplitSweep]) -> None:
    owners: dict[int, str] = {}
    for split, split_spec in splits.items():
        for seed in split_spec.seeds:
            previous = owners.setdefault(seed, split)
            if previous != split:
                raise ValueError("sweep seed is assigned to more than one split")


def _integer_list(
    value: object, description: str, *, minimum: int, maximum: int | None = None
) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{description} must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{description} must contain integers")
    if any(item < minimum or (maximum is not None and item > maximum) for item in value):
        raise ValueError(f"{description} contains an out-of-range value")
    if len(set(value)) != len(value):
        raise ValueError(f"{description} must not contain duplicates")
    return tuple(sorted(value))


def _fraction_list(value: object, description: str, *, strict: bool) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{description} must be a non-empty list")
    values = tuple(_finite_number(item, description) for item in value)
    invalid = (
        any(not 0.0 < item < 1.0 for item in values)
        if strict
        else any(not 0.0 <= item <= 1.0 for item in values)
    )
    if invalid:
        raise ValueError(f"{description} values must be between zero and one")
    if len(set(values)) != len(values):
        raise ValueError(f"{description} must not contain duplicates")
    return tuple(sorted(values))


def _finite_number(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{description} must be a finite number")
    return number


def _require_mapping(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], description: str
) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing:
        raise ValueError(f"{description} is missing required field {missing[0]!r}")
    if unexpected:
        raise ValueError(f"{description} has unexpected field {unexpected[0]!r}")


def _family_id(
    split: str,
    seed: int,
    profile_id: str,
    start_tick: int,
    target: str,
    fault: str,
    effectiveness: float | None,
) -> str:
    suffix = "" if effectiveness is None else f"-e{round(effectiveness * 1000):03d}"
    return (
        f"{split}-s{seed}-{profile_id}-t{start_tick}-{target}-{fault}{suffix}"
    )


def main(argv: Sequence[str]) -> int:
    if len(argv) != 2:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        receipt = generate_sweep(argv[0], argv[1])
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"cannot generate sweep: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main(sys.argv[1:]))
