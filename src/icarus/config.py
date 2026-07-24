"""User-editable scenario graph loading and validation for ICARUS.

A scenario file is versioned JSON describing the habitat as a directed
graph: ``zones`` (rooms) and ``connections`` (directed air paths). This
module is the only place that parses that format; the rest of the
simulation works on the validated :class:`HabitatConfig` it returns.

This PR supports the simple hub layout only: every connection links one
non-processing zone to the single ``air_processing`` bay, and every
non-processing zone has exactly one path each way. It is not a general
fluid solver.

All quantities remain abstract simulation units. See
docs/simulation-rules.md for the full rules.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_SCENARIO_VERSION = 1

ALLOWED_PRESETS = frozenset({"crew_cabin", "lab", "air_processing", "storage"})

# Declared fraction of moved-air CO2 the air_processing scrubber captures
# each tick. Declared here because the scenario format fixes the zone
# fields; a future PR may move it onto the air_processing preset.
AIR_PROCESSING_SCRUBBER_REMOVAL_FRACTION = 0.5

_ZONE_FIELDS = ("id", "label", "preset", "co2_generation_per_second", "air_volume")
_CONNECTION_FIELDS = ("id", "from", "to", "max_airflow", "health")
_FAULT_PROFILE_FIELDS = (
    "type",
    "connection_id",
    "start_tick",
    "end_tick",
    "end_effectiveness",
)
_GRADUAL_PRIMARY_FAN_DEGRADATION = "gradual_primary_fan_degradation"


@dataclass(frozen=True)
class ZoneSpec:
    """One validated zone from the scenario graph."""

    id: str
    label: str
    preset: str
    co2_generation_per_second: float
    air_volume: float


@dataclass(frozen=True)
class ConnectionSpec:
    """One validated directed connection from the scenario graph."""

    id: str
    from_zone: str
    to_zone: str
    max_airflow: float
    health: float


@dataclass(frozen=True)
class GradualPrimaryFanDegradation:
    """A deterministic linear loss of effectiveness on one primary fan."""

    connection_id: str
    start_tick: int
    end_tick: int
    end_effectiveness: float

    def effectiveness_at(self, tick: int) -> float:
        """Return the hidden effectiveness multiplier for ``tick``."""
        if tick <= self.start_tick:
            return 1.0
        if tick >= self.end_tick:
            return self.end_effectiveness
        progress = (tick - self.start_tick) / (self.end_tick - self.start_tick)
        return 1.0 + progress * (self.end_effectiveness - 1.0)


@dataclass(frozen=True)
class HabitatConfig:
    """A validated scenario graph: zones plus directed hub connections."""

    version: int
    zones: tuple[ZoneSpec, ...]
    connections: tuple[ConnectionSpec, ...]
    fault_profiles: tuple[GradualPrimaryFanDegradation, ...] = ()

    def processing_zone(self) -> ZoneSpec:
        """The single air_processing zone (validation guarantees exactly one)."""
        for zone in self.zones:
            if zone.preset == "air_processing":
                return zone
        raise LookupError("no air_processing zone")  # unreachable post-validation

    def non_processing_zones(self) -> tuple[ZoneSpec, ...]:
        """Every zone that is not the air_processing bay, in file order."""
        return tuple(z for z in self.zones if z.preset != "air_processing")

    def path_to_processing(self, zone_id: str) -> ConnectionSpec:
        """The zone's outbound path to the air_processing bay."""
        processing_id = self.processing_zone().id
        for connection in self.connections:
            if connection.from_zone == zone_id and connection.to_zone == processing_id:
                return connection
        raise LookupError(f"no path to processing from {zone_id!r}")

    def path_from_processing(self, zone_id: str) -> ConnectionSpec:
        """The zone's return path from the air_processing bay."""
        processing_id = self.processing_zone().id
        for connection in self.connections:
            if connection.from_zone == processing_id and connection.to_zone == zone_id:
                return connection
        raise LookupError(f"no return path from processing to {zone_id!r}")


def load_scenario(path) -> HabitatConfig:
    """Read a scenario JSON file and return its validated graph.

    Raises :class:`ValueError` with a clear message for a missing file,
    unparseable JSON, or any validation failure.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ValueError(f"scenario file not found: {path}") from None
    except OSError as exc:
        raise ValueError(f"cannot read scenario file {path}: {exc}") from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"scenario file {path} is not valid JSON: {exc}") from None
    return parse_scenario(data)


def parse_scenario(data: Any) -> HabitatConfig:
    """Validate a parsed scenario document and return its graph.

    Raises :class:`ValueError` with a clear message on the first problem.
    """
    if not isinstance(data, dict):
        raise ValueError("scenario document must be a JSON object")

    version = _parse_version(data)
    zones = _parse_zones(data)
    processing_count = sum(1 for z in zones if z.preset == "air_processing")
    if processing_count == 0:
        raise ValueError("scenario has no air_processing zone")
    if processing_count > 1:
        raise ValueError("scenario has more than one air_processing zone")
    connections = _parse_connections(data, {z.id for z in zones})
    _enforce_hub_pairing(zones, connections)
    fault_profiles = _parse_fault_profiles(data, zones, connections)
    return HabitatConfig(
        version=version,
        zones=zones,
        connections=connections,
        fault_profiles=fault_profiles,
    )


def _parse_version(data: dict) -> int:
    if "version" not in data:
        raise ValueError("scenario must declare a version")
    version = data["version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError(f"scenario version must be an integer, got {version!r}")
    if version != SUPPORTED_SCENARIO_VERSION:
        raise ValueError(
            f"unsupported version {version}; expected {SUPPORTED_SCENARIO_VERSION}"
        )
    return version


def _require_number(value: Any, what: str) -> float:
    # bool is a subclass of int; JSON true/false is never a valid number here.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number, got {value!r}")
    result = float(value)
    # Python's json accepts NaN/Infinity tokens. Neither is a usable simulation
    # value, and NaN comparisons are always False, so range checks below would
    # silently wave them through.
    if not math.isfinite(result):
        raise ValueError(f"{what} must be a finite number, got {value!r}")
    return result


def _require_positive_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"{what} must be positive")
    return value


def _parse_zones(data: dict) -> tuple[ZoneSpec, ...]:
    if "zones" not in data:
        raise ValueError("scenario must define 'zones'")
    raw_zones = data["zones"]
    if not isinstance(raw_zones, list):
        raise ValueError("'zones' must be a list")
    if not raw_zones:
        raise ValueError("scenario has no zones")

    zones: list[ZoneSpec] = []
    seen_ids: set[str] = set()
    for raw in raw_zones:
        if not isinstance(raw, dict):
            raise ValueError(f"zone entry must be an object, got {raw!r}")
        for field_name in _ZONE_FIELDS:
            if field_name not in raw:
                raise ValueError(f"zone {raw.get('id')!r} is missing required field {field_name!r}")
        zone_id = raw["id"]
        if not isinstance(zone_id, str) or not zone_id:
            raise ValueError(f"zone id must be a non-empty string, got {zone_id!r}")
        if not isinstance(raw["label"], str) or not raw["label"]:
            raise ValueError(f"zone {zone_id!r}: label must be a non-empty string")
        preset = raw["preset"]
        if preset not in ALLOWED_PRESETS:
            raise ValueError(
                f"zone {zone_id!r} has unsupported preset {preset!r}; "
                f"choose from {sorted(ALLOWED_PRESETS)}"
            )
        air_volume = _require_number(raw["air_volume"], f"zone {zone_id!r}: air_volume")
        if air_volume <= 0.0:
            raise ValueError(f"zone {zone_id!r}: air_volume must be positive")
        co2_generation = _require_number(
            raw["co2_generation_per_second"],
            f"zone {zone_id!r}: co2_generation_per_second",
        )
        if co2_generation < 0.0:
            raise ValueError(
                f"zone {zone_id!r}: co2_generation_per_second must not be negative"
            )
        if zone_id in seen_ids:
            raise ValueError(f"duplicate zone id {zone_id!r}")
        seen_ids.add(zone_id)
        zones.append(
            ZoneSpec(
                id=zone_id,
                label=raw["label"],
                preset=preset,
                co2_generation_per_second=co2_generation,
                air_volume=air_volume,
            )
        )
    return tuple(zones)


def _parse_connections(data: dict, zone_ids: set[str]) -> tuple[ConnectionSpec, ...]:
    if "connections" not in data:
        raise ValueError("scenario must define 'connections'")
    raw_connections = data["connections"]
    if not isinstance(raw_connections, list):
        raise ValueError("'connections' must be a list")

    connections: list[ConnectionSpec] = []
    seen_ids: set[str] = set()
    for raw in raw_connections:
        if not isinstance(raw, dict):
            raise ValueError(f"connection entry must be an object, got {raw!r}")
        for field_name in _CONNECTION_FIELDS:
            if field_name not in raw:
                raise ValueError(
                    f"connection {raw.get('id')!r} is missing required field {field_name!r}"
                )
        connection_id = raw["id"]
        if not isinstance(connection_id, str) or not connection_id:
            raise ValueError(
                f"connection id must be a non-empty string, got {connection_id!r}"
            )
        from_zone = raw["from"]
        to_zone = raw["to"]
        for endpoint, zone_ref in (("from", from_zone), ("to", to_zone)):
            if zone_ref not in zone_ids:
                raise ValueError(
                    f"connection {connection_id!r} references unknown zone {zone_ref!r} "
                    f"in field {endpoint!r}"
                )
        # A self-loop moves air nowhere. The air_processing variant otherwise
        # satisfies "touches the bay" and evades the per-zone pairing counts,
        # then crashes the run with a raw KeyError.
        if from_zone == to_zone:
            raise ValueError(
                f"connection {connection_id!r} loops from zone {from_zone!r} to itself"
            )
        max_airflow = _require_number(
            raw["max_airflow"], f"connection {connection_id!r}: max_airflow"
        )
        if max_airflow <= 0.0:
            raise ValueError(
                f"connection {connection_id!r}: max_airflow must be positive"
            )
        health = _require_number(raw["health"], f"connection {connection_id!r}: health")
        if not 0.0 <= health <= 1.0:
            raise ValueError(
                f"connection {connection_id!r}: health must be in 0.0..1.0"
            )
        if connection_id in seen_ids:
            raise ValueError(f"duplicate connection id {connection_id!r}")
        seen_ids.add(connection_id)
        connections.append(
            ConnectionSpec(
                id=connection_id,
                from_zone=from_zone,
                to_zone=to_zone,
                max_airflow=max_airflow,
                health=health,
            )
        )
    return tuple(connections)


def _parse_fault_profiles(
    data: dict,
    zones: tuple[ZoneSpec, ...],
    connections: tuple[ConnectionSpec, ...],
) -> tuple[GradualPrimaryFanDegradation, ...]:
    raw_profiles = data.get("fault_profiles", [])
    if not isinstance(raw_profiles, list):
        raise ValueError("'fault_profiles' must be a list")

    by_id = {connection.id: connection for connection in connections}
    processing_id = next(zone.id for zone in zones if zone.preset == "air_processing")
    profiles: list[GradualPrimaryFanDegradation] = []
    seen_connections: set[str] = set()

    for raw in raw_profiles:
        if not isinstance(raw, dict):
            raise ValueError(f"fault profile entry must be an object, got {raw!r}")
        for field_name in _FAULT_PROFILE_FIELDS:
            if field_name not in raw:
                raise ValueError(
                    f"fault profile is missing required field {field_name!r}"
                )

        profile_type = raw["type"]
        if profile_type != _GRADUAL_PRIMARY_FAN_DEGRADATION:
            raise ValueError(f"unsupported fault profile type {profile_type!r}")

        connection_id = raw["connection_id"]
        if not isinstance(connection_id, str) or not connection_id:
            raise ValueError(
                f"fault profile connection_id must be a non-empty string, "
                f"got {connection_id!r}"
            )
        if connection_id not in by_id:
            raise ValueError(
                f"fault profile references unknown connection {connection_id!r}"
            )
        connection = by_id[connection_id]
        if connection.to_zone != processing_id:
            raise ValueError(
                f"fault profile connection {connection_id!r} is not a primary fan "
                f"path to the air_processing bay"
            )
        if connection_id in seen_connections:
            raise ValueError(
                f"more than one fault profile targets connection {connection_id!r}"
            )

        start_tick = _require_positive_int(
            raw["start_tick"], f"fault profile {connection_id!r}: start_tick"
        )
        end_tick = _require_positive_int(
            raw["end_tick"], f"fault profile {connection_id!r}: end_tick"
        )
        if end_tick <= start_tick:
            raise ValueError(
                f"fault profile {connection_id!r}: end_tick must be after start_tick"
            )

        end_effectiveness = _require_number(
            raw["end_effectiveness"],
            f"fault profile {connection_id!r}: end_effectiveness",
        )
        if not 0.0 <= end_effectiveness < 1.0:
            raise ValueError(
                f"fault profile {connection_id!r}: end_effectiveness must be "
                f"at least 0.0 and below 1.0"
            )

        seen_connections.add(connection_id)
        profiles.append(
            GradualPrimaryFanDegradation(
                connection_id=connection_id,
                start_tick=start_tick,
                end_tick=end_tick,
                end_effectiveness=end_effectiveness,
            )
        )

    return tuple(profiles)


def _enforce_hub_pairing(
    zones: tuple[ZoneSpec, ...], connections: tuple[ConnectionSpec, ...]
) -> None:
    """Require the simple hub layout: one path each way per non-processing zone."""
    processing_id = next(z.id for z in zones if z.preset == "air_processing")
    for connection in connections:
        if processing_id not in (connection.from_zone, connection.to_zone):
            raise ValueError(
                f"connection {connection.id!r} does not touch the air_processing bay; "
                f"only hub layout connections are supported"
            )
    for zone in zones:
        if zone.preset == "air_processing":
            continue
        outbound = [
            c
            for c in connections
            if c.from_zone == zone.id and c.to_zone == processing_id
        ]
        inbound = [
            c
            for c in connections
            if c.from_zone == processing_id and c.to_zone == zone.id
        ]
        if len(outbound) > 1:
            raise ValueError(
                f"zone {zone.id!r} has more than one path to the air_processing bay"
            )
        if len(inbound) > 1:
            raise ValueError(
                f"zone {zone.id!r} has more than one return path from the "
                f"air_processing bay"
            )
        if not outbound:
            raise ValueError(
                f"zone {zone.id!r} has no path to the air_processing bay"
            )
        if not inbound:
            raise ValueError(
                f"zone {zone.id!r} has no return path from the air_processing bay"
            )
