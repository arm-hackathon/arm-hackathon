"""Versioned observable context for V6 specialist diagnostics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aeolus.config import HabitatConfig
from aeolus.model_input import (
    ModelInputContract,
    SelectorField,
    build_model_input_contract,
    model_artifact_metadata,
    model_input_v1,
)
from aeolus.trace import TickRecord

OBSERVABLE_CONTEXT_VERSION = "observable_context_v1"
OBSERVABLE_CONTEXT_DTYPE = "float32"
_ARTIFACT_KEYS = frozenset(
    {"observable_context_version", "selector_sha256", "topology_sha256"}
)
_ACTUATOR_FIELDS = (
    "setpoint",
    "actual_position",
    "tracking_residual",
    "moving",
    "movement_seconds",
    "power",
    "direction",
)
_CONNECTION_FIELDS = ("requested_airflow", "delivered_airflow", "airflow_residual")
_SYSTEM_FIELDS = (
    "shared_airflow_capacity",
    "total_requested_airflow",
    "total_delivered_airflow",
    "capacity_scale",
)


@dataclass(frozen=True)
class ObservableContextContract:
    """Ordered, observable-only V6 specialist context bound to topology."""

    model_input_contract: ModelInputContract
    fields: tuple[SelectorField, ...]
    selector_json: str
    selector_hash: str


def build_observable_context_contract(config: HabitatConfig) -> ObservableContextContract:
    """Build the V6 context selector from the validated current hub topology."""
    model_contract = build_model_input_contract(config)
    fields = _context_fields(model_contract)
    selector_json = _canonical_json(
        {
            "dtype": OBSERVABLE_CONTEXT_DTYPE,
            "fields": [field.as_dict() for field in fields],
            "shape": [len(fields)],
            "topology_hash": model_contract.topology_hash,
            "schema_version": OBSERVABLE_CONTEXT_VERSION,
        }
    )
    return ObservableContextContract(
        model_input_contract=model_contract,
        fields=fields,
        selector_json=selector_json,
        selector_hash=_sha256(selector_json),
    )


def observable_context_v1(
    record: TickRecord, contract: ObservableContextContract
) -> NDArray[np.float32]:
    """Project one validated record into the ordered V6 observable context."""
    _validate_contract(contract)
    model_input_v1(record, contract.model_input_contract)
    groups: Mapping[str, Mapping[str, Mapping[str, float | bool]]] = {
        "zones": record.zones,
        "actuators": record.actuators,
        "connections": record.connections,
        "system": {"system": record.system},
    }
    try:
        values = [groups[field.group][field.entity_id][field.field] for field in contract.fields]
    except (KeyError, TypeError) as exc:
        raise ValueError("record does not satisfy the observable-context contract") from exc
    with np.errstate(over="ignore", invalid="ignore"):
        tensor = np.asarray(values, dtype=np.float32)
    if tensor.shape != (len(contract.fields),):
        raise ValueError("observable context has an unexpected shape")
    if not np.isfinite(tensor).all():
        raise ValueError("observable context contains non-finite float32 values")
    return tensor


def observable_context_metadata(contract: ObservableContextContract) -> dict[str, str]:
    """Return exact compatibility metadata for a V6 context consumer."""
    _validate_contract(contract)
    return {
        "observable_context_version": OBSERVABLE_CONTEXT_VERSION,
        "selector_sha256": contract.selector_hash,
        "topology_sha256": contract.model_input_contract.topology_hash,
    }


def assert_observable_context_compatible(
    metadata: object, contract: ObservableContextContract
) -> None:
    """Reject malformed or incompatible specialist-context metadata."""
    expected = observable_context_metadata(contract)
    if not isinstance(metadata, Mapping) or set(metadata) != _ARTIFACT_KEYS:
        raise ValueError("observable context metadata is malformed")
    if any(not isinstance(value, str) for value in metadata.values()):
        raise ValueError("observable context metadata values must be strings")
    if dict(metadata) != expected:
        raise ValueError("observable context metadata does not match this inference setup")


def _context_fields(model_contract: ModelInputContract) -> tuple[SelectorField, ...]:
    model_artifact_metadata(model_contract)
    topology = json.loads(model_contract.topology_json)
    zone_ids = topology["non_processing_zone_ids"]
    fields: list[SelectorField] = [
        SelectorField("zones", zone_id, "sensor_co2_concentration") for zone_id in zone_ids
    ]
    for zone_id in zone_ids:
        fields.extend(SelectorField("actuators", zone_id, field) for field in _ACTUATOR_FIELDS)
    for loop in topology["primary_loops"]:
        for edge_name in ("outbound", "return"):
            connection_id = loop[edge_name]["id"]
            fields.extend(
                SelectorField("connections", connection_id, field)
                for field in _CONNECTION_FIELDS
            )
    fields.extend(SelectorField("system", "system", field) for field in _SYSTEM_FIELDS)
    return tuple(fields)


def _validate_contract(contract: ObservableContextContract) -> None:
    if not isinstance(contract, ObservableContextContract):
        raise ValueError("observable context requires an ObservableContextContract")
    expected_fields = _context_fields(contract.model_input_contract)
    if contract.fields != expected_fields:
        raise ValueError("observable context selector does not match its topology")
    expected_json = _canonical_json(
        {
            "dtype": OBSERVABLE_CONTEXT_DTYPE,
            "fields": [field.as_dict() for field in contract.fields],
            "shape": [len(contract.fields)],
            "topology_hash": contract.model_input_contract.topology_hash,
            "schema_version": OBSERVABLE_CONTEXT_VERSION,
        }
    )
    if contract.selector_json != expected_json:
        raise ValueError("observable context selector representation does not match its fields")
    if _sha256(contract.selector_json) != contract.selector_hash:
        raise ValueError("observable context selector hash does not match its JSON")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
