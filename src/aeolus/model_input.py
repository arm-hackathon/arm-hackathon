"""Versioned, topology-derived model-input selection for AEOLUS."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from aeolus.config import HabitatConfig
from aeolus.trace import TickRecord, model_feature_row

MODEL_INPUT_VERSION = "model_input_v1"
TOPOLOGY_VERSION = "aeolus_topology_v1"
MODEL_INPUT_DTYPE = "float32"
MODEL_INPUT_SHAPE = (24,)
_ARTIFACT_METADATA_KEYS = frozenset(
    {"model_input_version", "selector_sha256", "topology_sha256"}
)
_TOPOLOGY_KEYS = frozenset(
    {"schema_version", "processing_zone_id", "non_processing_zone_ids", "primary_loops"}
)
_LOOP_KEYS = frozenset({"zone_id", "outbound", "return"})
_EDGE_KEYS = frozenset({"id", "from_zone", "to_zone"})


@dataclass(frozen=True)
class SelectorField:
    """One ordered observable scalar selected into a model input tensor."""

    group: str
    entity_id: str
    field: str

    def as_dict(self) -> dict[str, str]:
        """Return the canonical JSON-ready representation."""
        return {"entity_id": self.entity_id, "field": self.field, "group": self.group}


@dataclass(frozen=True)
class ModelInputContract:
    """The ordered v1 observable fields bound to validated topology."""

    fields: tuple[SelectorField, ...]
    selector_json: str
    selector_hash: str
    topology_json: str
    topology_hash: str


def build_model_input_contract(config: HabitatConfig) -> ModelInputContract:
    """Build the fixed v1 selector from validated hub topology."""
    topology_json = _canonical_json(_topology_representation(config))
    topology_hash = _sha256(topology_json)
    fields = _selector_fields(config)
    if len(fields) != MODEL_INPUT_SHAPE[0]:
        raise ValueError(
            f"{MODEL_INPUT_VERSION} requires exactly {MODEL_INPUT_SHAPE[0]} fields, "
            f"got {len(fields)}"
        )
    selector_json = _canonical_json(
        {
            "dtype": MODEL_INPUT_DTYPE,
            "fields": [field.as_dict() for field in fields],
            "shape": list(MODEL_INPUT_SHAPE),
            "topology_hash": topology_hash,
            "schema_version": MODEL_INPUT_VERSION,
        }
    )
    return ModelInputContract(
        fields=fields,
        selector_json=selector_json,
        selector_hash=_sha256(selector_json),
        topology_json=topology_json,
        topology_hash=topology_hash,
    )


def model_input_v1(
    record: TickRecord, contract: ModelInputContract
) -> NDArray[np.float32]:
    """Return the exact ordered float32 v1 tensor from observable telemetry."""
    topology = _validate_contract(contract)
    features = model_feature_row(record)
    _validate_record_topology(features, topology)
    try:
        values = [
            features[field.group][field.entity_id][field.field]
            for field in contract.fields
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError("record does not satisfy the model-input contract") from exc
    with np.errstate(over="ignore", invalid="ignore"):
        tensor = np.asarray(values, dtype=np.float32)
    if tensor.shape != MODEL_INPUT_SHAPE:
        raise ValueError("model input has an unexpected shape")
    if not np.isfinite(tensor).all():
        raise ValueError("model input contains non-finite float32 values")
    return tensor


def model_artifact_metadata(contract: ModelInputContract) -> dict[str, str]:
    """Return the exact metadata a compatible model artifact must declare."""
    _validate_contract(contract)
    return {
        "model_input_version": MODEL_INPUT_VERSION,
        "selector_sha256": contract.selector_hash,
        "topology_sha256": contract.topology_hash,
    }


def assert_model_contract_compatible(
    metadata: object, contract: ModelInputContract
) -> None:
    """Reject artifact metadata that is malformed or mismatched for ``contract``."""
    expected = model_artifact_metadata(contract)
    if not isinstance(metadata, Mapping) or set(metadata) != _ARTIFACT_METADATA_KEYS:
        raise ValueError("model artifact contract metadata is malformed")
    if any(not isinstance(value, str) for value in metadata.values()):
        raise ValueError("model artifact contract metadata values must be strings")
    if dict(metadata) != expected:
        raise ValueError("model artifact contract does not match this inference setup")


def _selector_fields(config: HabitatConfig) -> tuple[SelectorField, ...]:
    zones = config.non_processing_zones()
    fields: list[SelectorField] = [
        SelectorField("zones", zone.id, "sensor_co2_concentration") for zone in zones
    ]
    for zone in zones:
        fields.extend(
            SelectorField("actuators", zone.id, name)
            for name in ("setpoint", "actual_position", "tracking_residual", "power")
        )
    for zone in zones:
        outbound = config.path_to_processing(zone.id)
        fields.extend(
            SelectorField("connections", outbound.id, name)
            for name in ("requested_airflow", "delivered_airflow", "airflow_residual")
        )
    return tuple(fields)


@lru_cache(maxsize=256)
def _validate_contract(contract: ModelInputContract) -> dict[str, Any]:
    if not isinstance(contract, ModelInputContract):
        raise ValueError("model input requires a ModelInputContract")
    if len(contract.fields) != MODEL_INPUT_SHAPE[0]:
        raise ValueError("model input contract has an unexpected field count")
    if not all(isinstance(field, SelectorField) for field in contract.fields):
        raise ValueError("model input contract has malformed selector fields")
    try:
        topology = json.loads(contract.topology_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("model input contract has malformed topology JSON") from exc
    if not isinstance(topology, dict) or _canonical_json(topology) != contract.topology_json:
        raise ValueError("model input contract topology JSON is not canonical")
    if _sha256(contract.topology_json) != contract.topology_hash:
        raise ValueError("model input contract topology hash does not match its JSON")
    if contract.fields != _fields_from_topology(topology):
        raise ValueError("model input contract selector does not match its topology")
    expected_selector_json = _canonical_json(
        {
            "dtype": MODEL_INPUT_DTYPE,
            "fields": [field.as_dict() for field in contract.fields],
            "shape": list(MODEL_INPUT_SHAPE),
            "topology_hash": contract.topology_hash,
            "schema_version": MODEL_INPUT_VERSION,
        }
    )
    if contract.selector_json != expected_selector_json:
        raise ValueError("model input contract selector representation does not match its fields")
    if _sha256(contract.selector_json) != contract.selector_hash:
        raise ValueError("model input contract selector hash does not match its JSON")
    return topology


def _validate_record_topology(
    features: Mapping[str, Mapping[str, Mapping[str, float]]],
    topology: Mapping[str, Any],
) -> None:
    non_processing_zone_ids = set(topology["non_processing_zone_ids"])
    expected_by_group = {
        "zones": non_processing_zone_ids | {topology["processing_zone_id"]},
        "actuators": non_processing_zone_ids,
        "connections": {
        edge["id"]
        for loop in topology["primary_loops"]
        for edge in (loop["outbound"], loop["return"])
        },
    }
    for group, expected in expected_by_group.items():
        actual = set(features[group])
        if actual == expected:
            continue
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected, key=repr)
        raise ValueError(
            f"record topology does not match model input contract at {group}: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )


def _fields_from_topology(topology: Mapping[str, Any]) -> tuple[SelectorField, ...]:
    try:
        if set(topology) != _TOPOLOGY_KEYS:
            raise ValueError
        processing_id = topology["processing_zone_id"]
        zone_ids = topology["non_processing_zone_ids"]
        loops = topology["primary_loops"]
        if (
            topology["schema_version"] != TOPOLOGY_VERSION
            or not _is_non_empty_id(processing_id)
            or not isinstance(zone_ids, list)
            or not isinstance(loops, list)
            or len(zone_ids) != len(loops)
            or not all(_is_non_empty_id(zone_id) for zone_id in zone_ids)
            or processing_id in zone_ids
            or len(set(zone_ids)) != len(zone_ids)
        ):
            raise ValueError
        outbound_ids: list[str] = []
        return_ids: list[str] = []
        for zone_id, loop in zip(zone_ids, loops):
            if (
                not isinstance(loop, Mapping)
                or set(loop) != _LOOP_KEYS
                or loop["zone_id"] != zone_id
            ):
                raise ValueError
            outbound = loop["outbound"]
            inbound = loop["return"]
            if (
                not isinstance(outbound, Mapping)
                or set(outbound) != _EDGE_KEYS
                or not isinstance(inbound, Mapping)
                or set(inbound) != _EDGE_KEYS
            ):
                raise ValueError
            outbound_id = outbound["id"]
            return_id = inbound["id"]
            if not all(
                _is_non_empty_id(identifier)
                for identifier in (
                    outbound_id,
                    return_id,
                    outbound["from_zone"],
                    outbound["to_zone"],
                    inbound["from_zone"],
                    inbound["to_zone"],
                )
            ):
                raise ValueError
            if (outbound["from_zone"], outbound["to_zone"]) != (
                zone_id,
                processing_id,
            ):
                raise ValueError
            if (inbound["from_zone"], inbound["to_zone"]) != (
                processing_id,
                zone_id,
            ):
                raise ValueError
            outbound_ids.append(outbound_id)
            return_ids.append(return_id)
        edge_ids = outbound_ids + return_ids
        if (
            len(set(outbound_ids)) != len(outbound_ids)
            or len(set(return_ids)) != len(return_ids)
            or len(set(edge_ids)) != len(edge_ids)
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("model input contract topology is malformed") from exc

    fields: list[SelectorField] = [
        SelectorField("zones", zone_id, "sensor_co2_concentration")
        for zone_id in zone_ids
    ]
    for zone_id in zone_ids:
        fields.extend(
            SelectorField("actuators", zone_id, field_name)
            for field_name in ("setpoint", "actual_position", "tracking_residual", "power")
        )
    for outbound_id in outbound_ids:
        fields.extend(
            SelectorField("connections", outbound_id, field_name)
            for field_name in (
                "requested_airflow",
                "delivered_airflow",
                "airflow_residual",
            )
        )
    return tuple(fields)


def _is_non_empty_id(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _topology_representation(config: HabitatConfig) -> dict[str, Any]:
    processing = config.processing_zone()
    loops = []
    for zone in config.non_processing_zones():
        outbound = config.path_to_processing(zone.id)
        inbound = config.path_from_processing(zone.id)
        loops.append(
            {
                "outbound": _edge_representation(outbound),
                "return": _edge_representation(inbound),
                "zone_id": zone.id,
            }
        )
    return {
        "non_processing_zone_ids": [zone.id for zone in config.non_processing_zones()],
        "primary_loops": loops,
        "processing_zone_id": processing.id,
        "schema_version": TOPOLOGY_VERSION,
    }


def _edge_representation(connection) -> dict[str, str]:
    return {
        "from_zone": connection.from_zone,
        "id": connection.id,
        "to_zone": connection.to_zone,
    }


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
