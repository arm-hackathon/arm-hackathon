"""JSONL replay traces and an allowlisted model-feature projection for AEOLUS."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aeolus.config import HabitatConfig

_ZONE_FIELDS = frozenset(
    {
        "co2_mass",
        "co2_concentration",
        "sensor_co2_concentration",
        "source_co2_mass",
        "occupancy_multiplier",
    }
)
_PROCESSING_ZONE_FIELDS = _ZONE_FIELDS | {"captured_co2"}
_CONNECTION_FIELDS = frozenset(
    {"requested_airflow", "delivered_airflow", "airflow_residual"}
)
_ACTUATOR_FIELDS = frozenset(
    {
        "setpoint",
        "actual_position",
        "tracking_residual",
        "moving",
        "movement_seconds",
        "power",
        "direction",
    }
)
_SYSTEM_FIELDS = frozenset(
    {
        "shared_airflow_capacity",
        "total_requested_airflow",
        "total_delivered_airflow",
        "capacity_scale",
    }
)
_MODEL_ACTUATOR_FIELDS = (
    "setpoint",
    "actual_position",
    "tracking_residual",
    "power",
)
RECOVERY_TRACE_VERSION = "aeolus_recovery_trace_v1"
_RESERVE_TOP_LEVEL_FIELDS = frozenset({"connections", "actuators", "system"})
_RESERVE_ACTUATOR_FIELDS = _ACTUATOR_FIELDS - {"direction"}
_RESERVE_SYSTEM_FIELDS = frozenset(
    {
        "reserve_airflow_capacity",
        "total_requested_airflow",
        "total_delivered_airflow",
        "capacity_scale",
        "total_power",
    }
)
_AUTHORITY_FIELDS = frozenset(
    {
        "run_id",
        "authority_epoch",
        "decision_tick",
        "sequence",
        "state",
        "reserve_command_owner",
        "target_zone_id",
        "reason",
        "dwell_ticks",
        "observation_tick",
        "command_digest",
        "applied_command_digest",
    }
)
RECOVERY_AUTHORITY_REASONS = frozenset(
    {
        "advisory_entry_persistence_met",
        "advisory_unique_concern",
        "ambiguous_concern",
        "cold_start",
        "degraded_clear",
        "entry_persistence_met",
        "failure_latched",
        "handback_abort",
        "handback_begin",
        "handback_complete",
        "handback_ramp",
        "handback_ramp_down",
        "handback_recurrence",
        "handback_start",
        "handback_timeout",
        "handback_wait",
        "no_concern",
        "observation_unavailable",
        "protect_hold",
        "protect_increase",
        "recovery_clear",
        "reserve_delivery_failure",
        "reserve_failure_handback_complete",
        "reserve_failure_shutdown",
        "target_changed",
        "unique_concern",
    }
)


@dataclass(frozen=True)
class TickRecord:
    """The persisted observable state of one simulator tick."""

    tick: int
    zones: dict[str, dict[str, float]] = field(default_factory=dict)
    connections: dict[str, dict[str, float]] = field(default_factory=dict)
    actuators: dict[str, dict[str, float]] = field(default_factory=dict)
    system: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveryTickRecord:
    """Versioned recovery row with an unchanged legacy plant projection."""

    plant: TickRecord
    reserve: dict[str, Any]
    authority: dict[str, Any]
    schema_version: str = RECOVERY_TRACE_VERSION


class TraceWriter:
    """Write validated, deterministic JSONL replay rows."""

    def __init__(self, path) -> None:
        self._path = Path(path)
        self._handle = None

    def __enter__(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._handle = self._path.open("x", encoding="utf-8")
        except FileExistsError as exc:
            raise FileExistsError(f"trace output already exists: {self._path}") from exc
        return self

    def write(self, record: TickRecord) -> None:
        if self._handle is None:
            raise RuntimeError("TraceWriter.write() called outside a 'with' block")
        _validate_observable_telemetry(record)
        self._handle.write(json.dumps(asdict(record), sort_keys=True, allow_nan=False) + "\n")

    def __exit__(self, *exc_info):
        if self._handle is not None:
            self._handle.close()
        return False


class RecoveryTraceWriter:
    """Write strict versioned recovery rows without widening legacy traces."""

    def __init__(self, path, config: HabitatConfig) -> None:
        if not isinstance(config, HabitatConfig) or config.version != 10:
            raise ValueError("RecoveryTraceWriter requires a validated v10 habitat config")
        self._path = Path(path)
        self._config = config
        self._handle = None

    def __enter__(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._handle = self._path.open("x", encoding="utf-8")
        except FileExistsError as exc:
            raise FileExistsError(f"trace output already exists: {self._path}") from exc
        return self

    def write(self, record: RecoveryTickRecord) -> None:
        if self._handle is None:
            raise RuntimeError(
                "RecoveryTraceWriter.write() called outside a 'with' block"
            )
        _validate_recovery_telemetry(record, self._config)
        self._handle.write(
            json.dumps(asdict(record), sort_keys=True, allow_nan=False) + "\n"
        )

    def __exit__(self, *exc_info):
        if self._handle is not None:
            self._handle.close()
        return False


def model_feature_row(
    record: TickRecord | RecoveryTickRecord,
) -> dict[str, dict[str, dict[str, float]]]:
    """Return only the explicit model-observable projection of a trace record."""
    plant = record.plant if isinstance(record, RecoveryTickRecord) else record
    _validate_observable_telemetry(plant)
    return {
        "zones": {
            zone_id: {"sensor_co2_concentration": values["sensor_co2_concentration"]}
            for zone_id, values in sorted(plant.zones.items())
        },
        "actuators": {
            actuator_id: {
                field_name: values[field_name] for field_name in _MODEL_ACTUATOR_FIELDS
            }
            for actuator_id, values in sorted(plant.actuators.items())
        },
        "connections": {
            connection_id: {
                field_name: values[field_name]
                for field_name in (
                    "requested_airflow",
                    "delivered_airflow",
                    "airflow_residual",
                )
            }
            for connection_id, values in sorted(plant.connections.items())
        },
    }


def _validate_recovery_telemetry(
    record: RecoveryTickRecord, config: HabitatConfig
) -> None:
    if record.schema_version != RECOVERY_TRACE_VERSION:
        raise ValueError("recovery trace schema_version is unsupported")
    _validate_observable_telemetry(record.plant)
    expected_zone_ids = {zone.id for zone in config.non_processing_zones()}
    expected_all_zone_ids = {zone.id for zone in config.zones}
    expected_primary_ids = {connection.id for connection in config.connections}
    expected_reserve_ids = {
        connection.id for connection in config.reserve_connections
    }
    if (
        set(record.plant.zones) != expected_all_zone_ids
        or set(record.plant.connections) != expected_primary_ids
        or set(record.plant.actuators) != expected_zone_ids
    ):
        raise ValueError("recovery plant telemetry does not match config topology")
    _validate_mapping(record.reserve, "reserve")
    _validate_mapping(record.authority, "authority")
    if set(record.reserve) != _RESERVE_TOP_LEVEL_FIELDS:
        raise ValueError("reserve telemetry has unexpected fields")

    connections = record.reserve["connections"]
    actuators = record.reserve["actuators"]
    system = record.reserve["system"]
    _validate_mapping(connections, "reserve connections")
    _validate_mapping(actuators, "reserve actuators")
    _validate_mapping(system, "reserve system")
    if set(connections) != expected_reserve_ids or set(actuators) != expected_zone_ids:
        raise ValueError("reserve telemetry does not match config topology")
    for connection_id, values in connections.items():
        _validate_mapping(values, f"reserve connection {connection_id!r}")
        if set(values) != _CONNECTION_FIELDS:
            raise ValueError(
                f"reserve connection {connection_id!r} has unexpected telemetry fields"
            )
        for field_name, value in values.items():
            _require_finite_non_negative(
                value, f"reserve connection {connection_id!r} {field_name}"
            )
        if values["delivered_airflow"] > values["requested_airflow"]:
            raise ValueError("reserve connection delivery exceeds request")
        expected = values["requested_airflow"] - values["delivered_airflow"]
        if not math.isclose(values["airflow_residual"], expected, abs_tol=1e-12):
            raise ValueError("reserve connection residual is inconsistent")

    for zone_id in sorted(expected_zone_ids):
        outbound = connections[config.reserve_path_to_processing(zone_id).id]
        inbound = connections[config.reserve_path_from_processing(zone_id).id]
        if outbound != inbound:
            raise ValueError("reserve path pair telemetry is inconsistent")

    for actuator_id, values in actuators.items():
        _validate_mapping(values, f"reserve actuator {actuator_id!r}")
        if set(values) != _RESERVE_ACTUATOR_FIELDS:
            raise ValueError(
                f"reserve actuator {actuator_id!r} has unexpected telemetry fields"
            )
        moving = values["moving"]
        if not isinstance(moving, bool):
            raise ValueError(f"reserve actuator {actuator_id!r} moving must be boolean")
        for field_name, value in values.items():
            if field_name == "moving":
                continue
            _require_finite(value, f"reserve actuator {actuator_id!r} {field_name}")
        for field_name in ("setpoint", "actual_position"):
            if not 0.0 <= values[field_name] <= 1.0:
                raise ValueError(
                    f"reserve actuator {actuator_id!r} {field_name} must be in 0.0..1.0"
                )
        for field_name in ("movement_seconds", "power"):
            if values[field_name] < 0.0:
                raise ValueError(
                    f"reserve actuator {actuator_id!r} {field_name} must not be negative"
                )
        expected_tracking = values["setpoint"] - values["actual_position"]
        if not math.isclose(
            values["tracking_residual"], expected_tracking, abs_tol=1e-12
        ):
            raise ValueError("reserve actuator tracking residual is inconsistent")
        if values["moving"] != (values["movement_seconds"] > 0.0):
            raise ValueError("reserve actuator moving state is inconsistent")

    if set(system) != _RESERVE_SYSTEM_FIELDS:
        raise ValueError("reserve system has unexpected telemetry fields")
    for field_name, value in system.items():
        _require_finite_non_negative(value, f"reserve system {field_name}")
    if system["reserve_airflow_capacity"] <= 0.0:
        raise ValueError("reserve system capacity must be positive")
    if not 0.0 <= system["capacity_scale"] <= 1.0:
        raise ValueError("reserve system capacity_scale must be in 0.0..1.0")
    if system["total_delivered_airflow"] > system["reserve_airflow_capacity"]:
        raise ValueError("reserve system delivered airflow exceeds capacity")
    if not math.isclose(
        system["reserve_airflow_capacity"],
        config.air_system.reserve_airflow_capacity,
        abs_tol=1e-12,
    ):
        raise ValueError("reserve system capacity does not match config")
    outbound_rows = [
        connections[config.reserve_path_to_processing(zone_id).id]
        for zone_id in sorted(expected_zone_ids)
    ]
    expected_requested = math.fsum(row["requested_airflow"] for row in outbound_rows)
    expected_delivered = math.fsum(row["delivered_airflow"] for row in outbound_rows)
    expected_power = math.fsum(values["power"] for values in actuators.values())
    if not math.isclose(
        system["total_requested_airflow"], expected_requested, abs_tol=1e-12
    ):
        raise ValueError("reserve system total requested airflow is inconsistent")
    if not math.isclose(
        system["total_delivered_airflow"], expected_delivered, abs_tol=1e-12
    ):
        raise ValueError("reserve system total delivered airflow is inconsistent")
    if not math.isclose(system["total_power"], expected_power, abs_tol=1e-12):
        raise ValueError("reserve system total power is inconsistent")

    authority = record.authority
    if set(authority) != _AUTHORITY_FIELDS:
        raise ValueError("authority telemetry has unexpected fields")
    for field_name in ("run_id", "state", "reserve_command_owner", "reason"):
        if not isinstance(authority[field_name], str) or not authority[field_name]:
            raise ValueError(f"authority {field_name} must be a non-empty string")
    for field_name, minimum in (
        ("authority_epoch", 0),
        ("decision_tick", 1),
        ("sequence", 1),
        ("dwell_ticks", 0),
        ("observation_tick", 0),
    ):
        value = authority[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"authority {field_name} is invalid")
    if authority["observation_tick"] >= authority["decision_tick"]:
        raise ValueError("authority observation_tick must precede decision_tick")
    target = authority["target_zone_id"]
    if target is not None and (
        not isinstance(target, str) or target not in expected_zone_ids
    ):
        raise ValueError("authority target_zone_id is invalid")
    if authority["decision_tick"] != record.plant.tick:
        raise ValueError("authority decision_tick does not match plant tick")
    if authority["reason"] not in RECOVERY_AUTHORITY_REASONS:
        raise ValueError("authority reason is invalid")
    if authority["state"] not in {"NOMINAL", "DEGRADED", "PROTECT", "HANDBACK"}:
        raise ValueError("authority state is invalid")
    if authority["reserve_command_owner"] not in {
        "reserve_off",
        "deterministic_recovery_supervisor",
    }:
        raise ValueError("authority reserve_command_owner is invalid")
    off_state = authority["state"] in {"NOMINAL", "DEGRADED"}
    expected_owner = (
        "reserve_off" if off_state else "deterministic_recovery_supervisor"
    )
    if authority["reserve_command_owner"] != expected_owner:
        raise ValueError("authority state and reserve command owner are inconsistent")
    if authority["state"] == "NOMINAL" and target is not None:
        raise ValueError("authority NOMINAL state cannot retain a target")
    if not off_state and target is None:
        raise ValueError("authority active reserve state requires a target")
    for field_name in ("command_digest", "applied_command_digest"):
        value = authority[field_name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"authority {field_name} must be lowercase SHA-256")
    if authority["command_digest"] != authority["applied_command_digest"]:
        raise ValueError("authority applied command digest does not acknowledge decision")


def _validate_observable_telemetry(record: TickRecord) -> None:
    if isinstance(record.tick, bool) or not isinstance(record.tick, int) or record.tick < 1:
        raise ValueError("trace tick must be a positive integer")
    _validate_mapping(record.zones, "zones")
    _validate_mapping(record.connections, "connections")
    _validate_mapping(record.actuators, "actuators")
    _validate_mapping(record.system, "system")

    for zone_id, values in record.zones.items():
        _validate_mapping(values, f"zone {zone_id!r}")
        fields = set(values)
        if fields not in (_ZONE_FIELDS, _PROCESSING_ZONE_FIELDS):
            raise ValueError(
                f"zone {zone_id!r} has unexpected telemetry fields {sorted(fields - _PROCESSING_ZONE_FIELDS)!r}"
            )
        for field_name, value in values.items():
            _require_finite_non_negative(value, f"zone {zone_id!r} {field_name}")

    for connection_id, values in record.connections.items():
        _validate_mapping(values, f"connection {connection_id!r}")
        if set(values) != _CONNECTION_FIELDS:
            raise ValueError(
                f"connection {connection_id!r} has unexpected telemetry fields "
                f"{sorted(set(values) - _CONNECTION_FIELDS)!r}"
            )
        for field_name, value in values.items():
            _require_finite_non_negative(value, f"connection {connection_id!r} {field_name}")
        if values["delivered_airflow"] > values["requested_airflow"]:
            raise ValueError(
                f"connection {connection_id!r} delivered airflow exceeds requested airflow"
            )
        expected_residual = values["requested_airflow"] - values["delivered_airflow"]
        if not math.isclose(values["airflow_residual"], expected_residual, abs_tol=1e-12):
            raise ValueError(
                f"connection {connection_id!r} airflow residual does not match request minus delivery"
            )

    for actuator_id, values in record.actuators.items():
        _validate_mapping(values, f"actuator {actuator_id!r}")
        if set(values) != _ACTUATOR_FIELDS:
            raise ValueError(
                f"actuator {actuator_id!r} has unexpected telemetry fields "
                f"{sorted(set(values) - _ACTUATOR_FIELDS)!r}"
            )
        for field_name, value in values.items():
            _require_finite(value, f"actuator {actuator_id!r} {field_name}")
        for field_name in ("setpoint", "actual_position"):
            if not 0.0 <= values[field_name] <= 1.0:
                raise ValueError(f"actuator {actuator_id!r} {field_name} must be in 0.0..1.0")
        for field_name in ("moving", "movement_seconds", "power"):
            if values[field_name] < 0.0:
                raise ValueError(f"actuator {actuator_id!r} {field_name} must not be negative")

    if set(record.system) != _SYSTEM_FIELDS:
        raise ValueError(
            f"system has unexpected telemetry fields {sorted(set(record.system) - _SYSTEM_FIELDS)!r}"
        )
    for field_name, value in record.system.items():
        _require_finite_non_negative(value, f"system {field_name}")
    if record.system["shared_airflow_capacity"] <= 0.0:
        raise ValueError("system shared_airflow_capacity must be positive")
    if not 0.0 <= record.system["capacity_scale"] <= 1.0:
        raise ValueError("system capacity_scale must be in 0.0..1.0")
    if record.system["total_delivered_airflow"] > record.system["shared_airflow_capacity"]:
        raise ValueError("system delivered airflow exceeds shared capacity")


def _validate_mapping(value: Any, description: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{description} telemetry must be an object")


def _require_finite(value: Any, description: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{description} must be finite")


def _require_finite_non_negative(value: Any, description: str) -> None:
    _require_finite(value, description)
    if value < 0.0:
        raise ValueError(f"{description} must not be negative")
