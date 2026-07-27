"""JSONL replay traces and an allowlisted model-feature projection for ICARUS."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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


@dataclass(frozen=True)
class TickRecord:
    """The persisted observable state of one simulator tick."""

    tick: int
    zones: dict[str, dict[str, float]] = field(default_factory=dict)
    connections: dict[str, dict[str, float]] = field(default_factory=dict)
    actuators: dict[str, dict[str, float]] = field(default_factory=dict)
    system: dict[str, float] = field(default_factory=dict)


class TraceWriter:
    """Write validated, deterministic JSONL replay rows."""

    def __init__(self, path) -> None:
        self._path = Path(path)
        self._handle = None

    def __enter__(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("w", encoding="utf-8")
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


def model_feature_row(record: TickRecord) -> dict[str, dict[str, dict[str, float]]]:
    """Return only the explicit model-observable projection of a trace record."""
    _validate_observable_telemetry(record)
    return {
        "zones": {
            zone_id: {"sensor_co2_concentration": values["sensor_co2_concentration"]}
            for zone_id, values in sorted(record.zones.items())
        },
        "actuators": {
            actuator_id: {
                field_name: values[field_name] for field_name in _MODEL_ACTUATOR_FIELDS
            }
            for actuator_id, values in sorted(record.actuators.items())
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
            for connection_id, values in sorted(record.connections.items())
        },
    }


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
