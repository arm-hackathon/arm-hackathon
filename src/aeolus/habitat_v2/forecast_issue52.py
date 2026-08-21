"""Bounded long-horizon forecast and action ranking for Issue #52.

This module is deliberately outside the HMC authority core.  It owns only
causal observations, finite candidate schedules, offline rollouts, forecast
quality, and proposal construction.  HMC remains responsible for parsing,
preflight, policy, capability issuance, plant stepping, and replay.

The implementation is useful without a trained artifact: the default
forecaster is a deterministic action-conditioned baseline.  A fitted linear
forecaster can be created from offline rollout samples once the data and
metric approvals are in place.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import time
from types import MappingProxyType
from typing import Any

import numpy as np

from .hmc_contract import canonical_json_bytes
from .instrumentation import (
    OperationalMeasurement,
)
from .physics import (
    CanonicalExternalCommand,
    command_from_achieved_state,
    initial_state,
    validate_external_command,
)
from .proposal import ProposalReceipt
from .scenario import Scenario, ScenarioValidationError
from .snapshot import (
    OperationalSnapshot,
    SnapshotVerificationReceipt,
)
from .telemetry import derive_observable_topology


ISSUE52_SCHEMA_VERSION = "aeolus_habitat_v2_forecast_issue_52_v1"
HISTORY_STEPS = 16
HORIZON_STEPS = 32
CATALOGUE_SIZE = 12
CADENCE_SECONDS = 60.0
MIN_SCENARIO_STEPS = HISTORY_STEPS + HORIZON_STEPS - 1
INFERENCE_DEADLINE_MS = 250.0
MAX_NORMALIZED_UNCERTAINTY = 4.0
AMBIGUITY_MARGIN = 0.01
ALL_OPERATING_MODES = (
    "occupied",
    "eva_transition",
    "contingency",
    "dormant",
)
TARGET_ENVIRONMENTAL_FIELDS = (
    ("co2_ppm", "ppm", 800.0, 1_000.0, 300.0, 5_000.0, "safety_critical"),
    ("temperature_k", "K", 295.15, 10.0, 250.0, 330.0, "safety_critical"),
    (
        "relative_humidity",
        "fraction",
        0.45,
        0.25,
        0.0,
        1.0,
        "operational",
    ),
)
TARGET_RESOURCE_FIELDS = (
    (
        "battery_state_of_charge",
        "fraction",
        0.75,
        0.25,
        0.0,
        1.0,
        "safety_critical",
    ),
    (
        "oxygen_store_fraction",
        "fraction",
        0.75,
        0.25,
        0.0,
        1.0,
        "safety_critical",
    ),
    (
        "sorbent_remaining_fraction",
        "fraction",
        0.75,
        0.25,
        0.0,
        1.0,
        "safety_critical",
    ),
)
OUTCOMES = (
    "SELECTED_HOLD",
    "SELECTED_CANDIDATE",
    "ABSTAINED",
    "WARMUP_NO_PROPOSAL",
    "INVALID_OUTPUT",
    "TIMEOUT_NO_PROPOSAL",
    "HMC_REJECTED_TO_HOLD",
    "HMC_EMERGENCY_OVERRIDDEN",
    "DISABLED",
)


class Issue52ContractError(ValueError):
    """Raised when a long-horizon contract is malformed or inconsistent."""


class Issue52HistoryError(ValueError):
    """Raised when a verified causal history cannot be accepted."""


class Issue52ForecastError(ValueError):
    """Raised when a forecast or score cannot be produced safely."""


class Issue52RolloutError(ValueError):
    """Raised when an offline counterfactual checkpoint is malformed."""


def _canonical(value: Any) -> bytes:
    try:
        return canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise Issue52ContractError("value is not finite canonical JSON") from error


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, label: str) -> str:
    if not _is_sha256(value):
        raise Issue52ContractError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _stable_damper_factor(damper_id: str) -> float:
    digest = hashlib.sha256(damper_id.encode("utf-8")).digest()
    return 1.0 if digest[0] & 1 else 0.75


def _validate_live_hmc_binding(hmc: Any, scenario: Scenario) -> None:
    """Refuse an advisory/HMC pairing unless immutable identities agree."""

    binding = getattr(hmc, "advisory_binding", lambda: None)()
    if not isinstance(binding, Mapping):
        raise Issue52HistoryError("HMC does not expose advisory identities")
    if binding.get("scenario_sha256") != scenario.scenario_sha256:
        raise Issue52HistoryError("advisory scenario does not bind the issuing HMC")
    hmc_contract_sha256 = binding.get("hmc_contract_sha256")
    if not _is_sha256(hmc_contract_sha256):
        raise Issue52HistoryError("HMC advisory contract identity is invalid")


def _validate_adviser_input_types(
    hmc: Any,
    snapshot: OperationalSnapshot,
    verification: SnapshotVerificationReceipt,
) -> None:
    if type(snapshot) is not OperationalSnapshot or type(
        verification
    ) is not SnapshotVerificationReceipt:
        raise Issue52HistoryError("adviser requires exact issued snapshot evidence")
    if not hasattr(hmc, "verify_snapshot") or not hasattr(hmc, "propose"):
        raise Issue52HistoryError("adviser requires the HMC authority interface")


def _safety_bounds_for_descriptor(
    descriptor: "TargetDescriptor", health_policy: Any | None
) -> tuple[float, float]:
    lower = descriptor.crossing_lower
    upper = descriptor.crossing_upper
    if health_policy is None:
        return (
            descriptor.lower if lower is None else float(lower),
            descriptor.upper if upper is None else float(upper),
        )
    if not isinstance(health_policy, Mapping):
        raise Issue52ContractError("ranker contract has no health policy")
    environmental = health_policy.get("environmental")
    if not isinstance(environmental, Mapping):
        raise Issue52ContractError("ranker health policy is malformed")
    field_name = descriptor.descriptor_id.rsplit("/", 1)[-1]
    for rule in environmental.values():
        if not isinstance(rule, Mapping) or rule.get("channel") != field_name:
            continue
        direction = rule.get("direction")
        critical = rule.get("critical_enter")
        if not isinstance(critical, (int, float)) or isinstance(critical, bool):
            raise Issue52ContractError("ranker health critical threshold is malformed")
        if direction == "LOW":
            lower = max(descriptor.lower, float(critical))
        elif direction == "HIGH":
            upper = min(descriptor.upper, float(critical))
        else:
            raise Issue52ContractError("ranker health threshold direction is malformed")
    if descriptor.scope == "global":
        gauges = health_policy.get("resource_gauges")
        critical = None if not isinstance(gauges, Mapping) else gauges.get("critical_enter")
        if isinstance(critical, (int, float)) and not isinstance(critical, bool):
            lower = max(descriptor.lower, float(critical))
    if lower is None or upper is None or not lower < upper:
        raise Issue52ContractError("ranker safety bounds are invalid")
    return float(lower), float(upper)


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Issue52ContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise Issue52ContractError(f"{label} must be finite")
    return result


def _readonly(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32).copy()
    value.setflags(write=False)
    return value


def _copy_json(value: Any) -> Any:
    return json.loads(
        json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(nested) for key, nested in sorted(value.items())}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _unfreeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _unfreeze(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_unfreeze(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class TargetDescriptor:
    descriptor_id: str
    source_descriptor: str
    source: str
    unit: str
    nominal: float
    scale: float
    lower: float
    upper: float
    scope: str = "global"
    transform: str = "identity"
    inverse_transform: str = "identity"
    safety_relevance: str = "operational"
    crossing_lower: float | None = None
    crossing_upper: float | None = None

    def __post_init__(self) -> None:
        if (
            type(self.descriptor_id) is not str
            or not self.descriptor_id
            or type(self.source_descriptor) is not str
            or not self.source_descriptor
            or type(self.source) is not str
            or not self.source
            or type(self.unit) is not str
            or not self.unit
            or self.scope not in {"zone", "global"}
            or type(self.transform) is not str
            or type(self.inverse_transform) is not str
            or type(self.safety_relevance) is not str
        ):
            raise Issue52ContractError("target descriptor identity is invalid")
        for label, value in (
            ("nominal", self.nominal),
            ("scale", self.scale),
            ("lower", self.lower),
            ("upper", self.upper),
        ):
            if not math.isfinite(float(value)):
                raise Issue52ContractError(f"target descriptor {label} is not finite")
        if self.scale <= 0.0 or self.lower >= self.upper:
            raise Issue52ContractError("target descriptor scale or bounds are invalid")
        for label, value in (
            ("crossing_lower", self.crossing_lower),
            ("crossing_upper", self.crossing_upper),
        ):
            if value is not None and not math.isfinite(float(value)):
                raise Issue52ContractError(f"target descriptor {label} is not finite")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "descriptor_id": self.descriptor_id,
            "source_descriptor": self.source_descriptor,
            "source": self.source,
            "unit": self.unit,
            "nominal": self.nominal,
            "scale": self.scale,
            "lower": self.lower,
            "upper": self.upper,
            "scope": self.scope,
            "transform": self.transform,
            "inverse_transform": self.inverse_transform,
            "safety_relevance": self.safety_relevance,
            "crossing_lower": self.crossing_lower,
            "crossing_upper": self.crossing_upper,
        }


@dataclass(frozen=True, slots=True)
class TargetManifest:
    scenario_sha256: str
    topology_sha256: str
    descriptors: tuple[TargetDescriptor, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.scenario_sha256, label="target scenario identity")
        _require_sha256(self.topology_sha256, label="target topology identity")
        if not self.descriptors:
            raise Issue52ContractError("target manifest cannot be empty")
        identifiers = [descriptor.descriptor_id for descriptor in self.descriptors]
        if len(identifiers) != len(set(identifiers)):
            raise Issue52ContractError("target manifest contains duplicate descriptors")
        if tuple(identifiers[-3:]) != (
            "battery_state_of_charge",
            "oxygen_store_fraction",
            "sorbent_remaining_fraction",
        ):
            raise Issue52ContractError("target manifest resource order is not canonical")
        _require_sha256(self.manifest_sha256, label="target manifest identity")
        payload = {
            "schema_version": f"{ISSUE52_SCHEMA_VERSION}.target_manifest",
            "scenario_sha256": self.scenario_sha256,
            "topology_sha256": self.topology_sha256,
            "descriptors": [item.to_mapping() for item in self.descriptors],
        }
        if self.manifest_sha256 != _sha(payload):
            raise Issue52ContractError("target manifest digest is inconsistent")

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "TargetManifest":
        if type(scenario) is not Scenario:
            raise Issue52ContractError("target manifest requires Scenario")
        topology = derive_observable_topology(scenario)
        descriptors: list[TargetDescriptor] = []
        for zone_id in topology.zone_ids:
            for (
                field_name,
                unit,
                nominal,
                scale,
                lower,
                upper,
                safety_relevance,
            ) in (
                TARGET_ENVIRONMENTAL_FIELDS
            ):
                descriptors.append(
                    TargetDescriptor(
                        descriptor_id=f"{zone_id}/{field_name}",
                        source="primary_telemetry",
                        unit=unit,
                        nominal=nominal,
                        scale=scale,
                        lower=lower,
                        upper=upper,
                        scope="zone",
                        source_descriptor=f"primary_telemetry.{zone_id}/{field_name}",
                        safety_relevance=safety_relevance,
                        crossing_lower=lower,
                        crossing_upper=upper,
                    )
                )
        for (
            field_name,
            unit,
            nominal,
            scale,
            lower,
            upper,
            safety_relevance,
        ) in TARGET_RESOURCE_FIELDS:
            descriptors.append(
                TargetDescriptor(
                    descriptor_id=field_name,
                    source="operational_resource_gauges",
                    unit=unit,
                    nominal=nominal,
                    scale=scale,
                    lower=lower,
                    upper=upper,
                    scope="global",
                    source_descriptor=f"operational_resource_gauges.{field_name}",
                    safety_relevance=safety_relevance,
                    crossing_lower=lower,
                    crossing_upper=upper,
                )
            )
        payload = {
            "schema_version": f"{ISSUE52_SCHEMA_VERSION}.target_manifest",
            "scenario_sha256": scenario.scenario_sha256,
            "topology_sha256": topology.sha256,
            "descriptors": [item.to_mapping() for item in descriptors],
        }
        return cls(
            scenario_sha256=scenario.scenario_sha256,
            topology_sha256=topology.sha256,
            descriptors=tuple(descriptors),
            manifest_sha256=_sha(payload),
        )

    @property
    def width(self) -> int:
        return len(self.descriptors)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": f"{ISSUE52_SCHEMA_VERSION}.target_manifest",
            "scenario_sha256": self.scenario_sha256,
            "topology_sha256": self.topology_sha256,
            "descriptors": [item.to_mapping() for item in self.descriptors],
            "manifest_sha256": self.manifest_sha256,
        }


def _sample_map(block: Any, *, label: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(block, Mapping) or not isinstance(block.get("samples"), list):
        raise Issue52HistoryError(f"{label} must contain a sample list")
    result: dict[str, Mapping[str, Any]] = {}
    for sample in block["samples"]:
        if not isinstance(sample, Mapping) or type(sample.get("descriptor_id")) is not str:
            raise Issue52HistoryError(f"{label} contains a malformed sample")
        descriptor_id = str(sample["descriptor_id"])
        if descriptor_id in result:
            raise Issue52HistoryError(f"{label} contains duplicate {descriptor_id}")
        result[descriptor_id] = sample
    return result


def _sample_value(sample: Mapping[str, Any], *, label: str) -> tuple[float, bool]:
    availability = sample.get("availability")
    if availability == "AVAILABLE":
        value = _finite(sample.get("value"), label=label)
        if sample.get("unavailable_reason") is not None:
            raise Issue52HistoryError(f"{label} has an availability reason")
        return value, True
    if availability == "UNAVAILABLE" and sample.get("value") is None:
        return math.nan, False
    raise Issue52HistoryError(f"{label} availability is malformed")


def target_from_snapshot(
    snapshot_mapping: Mapping[str, Any], manifest: TargetManifest
) -> tuple[np.ndarray, np.ndarray]:
    """Project only primary environmental telemetry and unique resource gauges."""

    primary_block = snapshot_mapping.get("primary_telemetry")
    resource_block = snapshot_mapping.get("operational_resource_gauges")
    if (
        not isinstance(primary_block, Mapping)
        or primary_block.get("source_kind") != "primary_sensor_head"
        or not isinstance(resource_block, Mapping)
        or resource_block.get("source_kind") != "operational_resource_gauge"
    ):
        raise Issue52HistoryError("snapshot target source identities are invalid")
    primary = _sample_map(primary_block, label="primary")
    resources = _sample_map(
        resource_block, label="resources"
    )
    values = np.full(manifest.width, np.nan, dtype=np.float32)
    available = np.zeros(manifest.width, dtype=bool)
    for index, descriptor in enumerate(manifest.descriptors):
        source = primary if descriptor.source == "primary_telemetry" else resources
        sample = source.get(descriptor.descriptor_id)
        if sample is None:
            raise Issue52HistoryError(
                f"required target descriptor {descriptor.descriptor_id} is missing"
            )
        value, is_available = _sample_value(
            sample, label=f"target {descriptor.descriptor_id}"
        )
        values[index] = np.float32(value)
        available[index] = is_available
    values.setflags(write=False)
    available.setflags(write=False)
    return values, available


def _target_from_measurement(
    measurement: OperationalMeasurement, manifest: TargetManifest
) -> tuple[np.ndarray, np.ndarray]:
    primary = {sample.descriptor_id: sample.to_mapping() for sample in measurement.primary}
    feedback = {
        sample.descriptor_id: sample.to_mapping()
        for sample in measurement.operational_feedback
    }
    mapping = {
        "primary_telemetry": {
            "source_kind": "primary_sensor_head",
            "samples": list(primary.values()),
        },
        "operational_resource_gauges": {
            "source_kind": "operational_resource_gauge",
            "samples": [
                feedback[name].copy()
                for name in (
                    "battery_state_of_charge",
                    "oxygen_store_fraction",
                    "sorbent_remaining_fraction",
                )
                if name in feedback
            ]
        },
    }
    return target_from_snapshot(mapping, manifest)


def targets_from_measurement(
    measurement: OperationalMeasurement, manifest: TargetManifest
) -> tuple[np.ndarray, np.ndarray]:
    """Project the approved target channels from an issued measurement."""

    if type(measurement) is not OperationalMeasurement:
        raise Issue52HistoryError("target projection requires an issued measurement")
    return _target_from_measurement(measurement, manifest)


@dataclass(frozen=True, slots=True)
class ObservationRecord:
    snapshot_sha256: str
    verification_receipt_sha256: str
    control_run_id: str
    authority_epoch: str
    topology_sha256: str
    hmc_contract_sha256: str
    snapshot_schema_sha256: str
    scenario_sha256: str
    previous_verification_receipt_sha256: str
    previous_control_chain_sha256: str
    control_chain_sha256: str
    sequence: int
    completed_step: int
    completed_time_s: float
    mode: str | None
    command: Mapping[str, Any]
    command_sha256: str
    target_values: np.ndarray = field(compare=False)
    available_mask: np.ndarray = field(compare=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
            or isinstance(self.completed_step, bool)
            or not isinstance(self.completed_step, int)
            or self.completed_step < 0
        ):
            raise Issue52HistoryError("observation identity is negative")
        if not math.isfinite(float(self.completed_time_s)):
            raise Issue52HistoryError("observation time is not finite")
        for label, value in (
            ("snapshot", self.snapshot_sha256),
            ("verification receipt", self.verification_receipt_sha256),
            ("control run", self.control_run_id),
            ("authority epoch", self.authority_epoch),
            ("topology", self.topology_sha256),
            ("HMC contract", self.hmc_contract_sha256),
            ("snapshot schema", self.snapshot_schema_sha256),
            ("scenario", self.scenario_sha256),
            ("previous verification receipt", self.previous_verification_receipt_sha256),
            ("previous control chain", self.previous_control_chain_sha256),
            ("control chain", self.control_chain_sha256),
            ("command", self.command_sha256),
        ):
            if label in {"control run", "authority epoch"}:
                if type(value) is not str or not value:
                    raise Issue52HistoryError(f"observation {label} identity is invalid")
            else:
                _require_sha256(value, label=f"observation {label} identity")
        if self.mode is not None and (type(self.mode) is not str or not self.mode):
            raise Issue52HistoryError("observation operating mode is invalid")
        values = np.asarray(self.target_values, dtype=np.float32)
        mask = np.asarray(self.available_mask, dtype=bool)
        if values.ndim != 1 or mask.shape != values.shape:
            raise Issue52HistoryError("observation target shape is invalid")
        if not np.isfinite(values[mask]).all():
            raise Issue52HistoryError("available observation targets are non-finite")
        values = values.copy()
        mask = mask.copy()
        values.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "target_values", values)
        object.__setattr__(self, "available_mask", mask)
        if type(self.command) is not dict:
            raise Issue52HistoryError("observation command must be an object")
        frozen_command = _freeze(dict(self.command))
        object.__setattr__(self, "command", frozen_command)
        if _sha(_unfreeze(frozen_command)) != self.command_sha256:
            raise Issue52HistoryError("observation command identity is inconsistent")

    @classmethod
    def from_snapshot(
        cls,
        snapshot: OperationalSnapshot,
        verification: SnapshotVerificationReceipt,
        manifest: TargetManifest,
        scenario: Scenario,
        *,
        control_chain_sha256: str,
    ) -> "ObservationRecord":
        if type(snapshot) is not OperationalSnapshot or type(
            verification
        ) is not SnapshotVerificationReceipt:
            raise Issue52HistoryError("history requires exact issued snapshot types")
        snapshot_mapping = snapshot.to_mapping()
        receipt_mapping = verification.to_mapping()
        if snapshot_mapping.get("snapshot_sha256") != snapshot.snapshot_sha256:
            raise Issue52HistoryError("snapshot self identity is inconsistent")
        if receipt_mapping.get("snapshot_sha256") != snapshot.snapshot_sha256:
            raise Issue52HistoryError("verification does not bind snapshot")
        if receipt_mapping.get("completed_step") != snapshot_mapping.get("completed_step"):
            raise Issue52HistoryError("verification step does not bind snapshot")
        if receipt_mapping.get("sequence") != snapshot_mapping.get("sequence"):
            raise Issue52HistoryError("verification sequence does not bind snapshot")
        if receipt_mapping.get("control_run_id") != snapshot_mapping.get("control_run_id"):
            raise Issue52HistoryError("verification run does not bind snapshot")
        if receipt_mapping.get("authority_epoch") != snapshot_mapping.get("authority_epoch"):
            raise Issue52HistoryError("verification epoch does not bind snapshot")
        for field_name, label in (
            ("hmc_contract_sha256", "HMC contract"),
            ("snapshot_schema_sha256", "snapshot schema"),
            ("observable_topology_sha256", "topology"),
        ):
            if receipt_mapping.get(field_name) != snapshot_mapping.get(field_name):
                raise Issue52HistoryError(f"verification {label} does not bind snapshot")
        _require_sha256(control_chain_sha256, label="current control chain")
        if (
            receipt_mapping.get("snapshot_verification_receipt_sha256")
            != verification.snapshot_verification_receipt_sha256
        ):
            raise Issue52HistoryError("verification receipt self identity is inconsistent")
        if manifest.scenario_sha256 != scenario.scenario_sha256:
            raise Issue52HistoryError("target manifest does not bind scenario")
        if manifest.topology_sha256 != receipt_mapping.get("observable_topology_sha256"):
            raise Issue52HistoryError("target manifest does not bind snapshot topology")
        command_reference = snapshot_mapping.get("command_reference")
        if not isinstance(command_reference, Mapping):
            raise Issue52HistoryError("snapshot command reference is missing")
        command = command_reference.get("command")
        try:
            canonical = validate_external_command(scenario, command)
        except Exception as error:
            raise Issue52HistoryError("snapshot command reference is invalid") from error
        values, available = target_from_snapshot(snapshot_mapping, manifest)
        return cls(
            snapshot_sha256=snapshot.snapshot_sha256,
            verification_receipt_sha256=str(
                verification.snapshot_verification_receipt_sha256
            ),
            control_run_id=str(receipt_mapping["control_run_id"]),
            authority_epoch=str(receipt_mapping["authority_epoch"]),
            topology_sha256=str(receipt_mapping["observable_topology_sha256"]),
            hmc_contract_sha256=str(receipt_mapping["hmc_contract_sha256"]),
            snapshot_schema_sha256=str(receipt_mapping["snapshot_schema_sha256"]),
            scenario_sha256=manifest.scenario_sha256,
            previous_verification_receipt_sha256=str(
                receipt_mapping["previous_verification_receipt_digest"]
            ),
            previous_control_chain_sha256=str(
                receipt_mapping["previous_control_chain_sha256"]
            ),
            control_chain_sha256=control_chain_sha256,
            sequence=int(receipt_mapping["sequence"]),
            completed_step=int(receipt_mapping["completed_step"]),
            completed_time_s=float(receipt_mapping["completed_time_s"]),
            mode=snapshot_mapping.get("completed_operating_mode"),
            command=canonical.to_mapping(),
            command_sha256=canonical.sha256,
            target_values=values,
            available_mask=available,
        )


@dataclass(frozen=True, slots=True)
class HistoryAppend:
    accepted: bool
    status: str
    size: int
    reset_reason: str | None = None


class VerifiedHistoryBuffer:
    """In-process verified observation buffer with explicit reset semantics."""

    def __init__(self, manifest: TargetManifest, *, window_steps: int = HISTORY_STEPS):
        if (
            type(manifest) is not TargetManifest
            or isinstance(window_steps, bool)
            or not isinstance(window_steps, int)
            or window_steps != HISTORY_STEPS
        ):
            raise Issue52HistoryError("history buffer configuration is invalid")
        self.manifest = manifest
        self.window_steps = int(window_steps)
        self._records: deque[ObservationRecord] = deque(maxlen=self.window_steps)

    @property
    def records(self) -> tuple[ObservationRecord, ...]:
        return tuple(self._records)

    @property
    def ready(self) -> bool:
        return len(self._records) == self.window_steps

    def clear(self) -> None:
        self._records.clear()

    def append(
        self,
        hmc: Any,
        snapshot: OperationalSnapshot,
        verification: SnapshotVerificationReceipt,
        *,
        cadence_seconds: float = CADENCE_SECONDS,
        scenario: Scenario,
    ) -> HistoryAppend:
        _validate_live_hmc_binding(hmc, scenario)
        if (
            float(cadence_seconds) != CADENCE_SECONDS
            or float(scenario.data["dt_seconds"]) != CADENCE_SECONDS
        ):
            self.clear()
            raise Issue52HistoryError("Issue 52 history requires exact 60-second cadence")
        if getattr(hmc, "lifecycle_phase", None) == "TERMINAL":
            self.clear()
            raise Issue52HistoryError("terminal HMC lifecycle cannot feed history")
        try:
            handle = hmc.verify_snapshot(snapshot, verification)
            record = ObservationRecord.from_snapshot(
                snapshot,
                verification,
                self.manifest,
                scenario,
                control_chain_sha256=str(hmc.current_control_chain_sha256),
            )
        except Exception as error:
            self.clear()
            raise Issue52HistoryError("snapshot verification failed") from error
        if (
            handle.snapshot_sha256 != record.snapshot_sha256
            or handle.sequence != record.sequence
        ):
            self.clear()
            raise Issue52HistoryError("verified handle does not match observation")
        events = tuple(getattr(hmc, "control_events", ()))
        if not events:
            self.clear()
            raise Issue52HistoryError("verified snapshot has no control-chain event")
        current_event = events[-1]
        if (
            current_event.event_kind != "SNAPSHOT_VERIFICATION"
            or current_event.receipt_sha256 != record.verification_receipt_sha256
            or current_event.control_chain_sha256 != record.control_chain_sha256
            or current_event.previous_control_chain_sha256
            != record.previous_control_chain_sha256
        ):
            self.clear()
            raise Issue52HistoryError("verified snapshot control chain does not bind record")
        if self._records:
            previous = self._records[-1]
            if record.snapshot_sha256 == previous.snapshot_sha256:
                self.clear()
                raise Issue52HistoryError("duplicate snapshot was supplied")
            identity_matches = (
                previous.control_run_id == record.control_run_id
                and previous.authority_epoch == record.authority_epoch
                and previous.topology_sha256 == record.topology_sha256
                and previous.hmc_contract_sha256 == record.hmc_contract_sha256
                and previous.snapshot_schema_sha256 == record.snapshot_schema_sha256
                and previous.scenario_sha256 == record.scenario_sha256
                and record.previous_verification_receipt_sha256
                == previous.verification_receipt_sha256
            )
            continuity_matches = (
                record.sequence == previous.sequence + 1
                and record.completed_step == previous.completed_step + 1
                and math.isclose(
                    record.completed_time_s,
                    previous.completed_time_s + float(cadence_seconds),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            )
            if not identity_matches or not continuity_matches:
                if not identity_matches:
                    reason = "identity_change"
                elif (
                    record.previous_verification_receipt_sha256
                    != previous.verification_receipt_sha256
                ):
                    reason = "chain_mismatch"
                else:
                    reason = "cadence_or_sequence_gap"
                self.clear()
                self._records.append(record)
                return HistoryAppend(True, "RESET", 1, reason)
        self._records.append(record)
        return HistoryAppend(True, "ACCEPTED", len(self._records))

    def forecast_history(self) -> "ForecastHistory":
        if not self.ready:
            raise Issue52HistoryError("history is still warming up")
        return ForecastHistory.from_records(self.records)


@dataclass(frozen=True, slots=True)
class ForecastHistory:
    records: tuple[ObservationRecord, ...]
    target_values: np.ndarray = field(compare=False)
    available_mask: np.ndarray = field(compare=False)
    completed_times_s: np.ndarray = field(compare=False)

    @classmethod
    def from_records(cls, records: Sequence[ObservationRecord]) -> "ForecastHistory":
        items = tuple(records)
        if len(items) != HISTORY_STEPS:
            raise Issue52ForecastError("forecast history must contain 16 records")
        width = items[0].target_values.shape[0]
        if any(item.target_values.shape != (width,) for item in items):
            raise Issue52ForecastError("forecast history target widths differ")
        values = np.stack([item.target_values for item in items]).astype(np.float32)
        available = np.stack([item.available_mask for item in items]).astype(bool)
        times = np.asarray([item.completed_time_s for item in items], dtype=np.float64)
        if not np.all(np.diff(times) > 0.0):
            raise Issue52ForecastError("forecast history times are not increasing")
        if any(
            current.sequence != previous.sequence + 1
            or current.completed_step != previous.completed_step + 1
            or current.control_run_id != previous.control_run_id
            or current.authority_epoch != previous.authority_epoch
            or current.topology_sha256 != previous.topology_sha256
            or current.hmc_contract_sha256 != previous.hmc_contract_sha256
            or current.snapshot_schema_sha256 != previous.snapshot_schema_sha256
            or current.scenario_sha256 != previous.scenario_sha256
            or current.previous_verification_receipt_sha256
            != previous.verification_receipt_sha256
            or not math.isclose(
                current.completed_time_s,
                previous.completed_time_s + CADENCE_SECONDS,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for previous, current in zip(items, items[1:])
        ):
            raise Issue52ForecastError("forecast history is not exactly causal")
        values.setflags(write=False)
        available.setflags(write=False)
        times.setflags(write=False)
        return cls(items, values, available, times)

    @property
    def latest(self) -> np.ndarray:
        return self.target_values[-1]

    @property
    def latest_record(self) -> ObservationRecord:
        return self.records[-1]

    @property
    def slope(self) -> np.ndarray:
        result = np.zeros(self.target_values.shape[1], dtype=np.float32)
        x = self.completed_times_s
        for column in range(result.shape[0]):
            mask = self.available_mask[:, column]
            if int(mask.sum()) < 2:
                continue
            centered = x[mask] - x[mask].mean()
            denominator = float(np.dot(centered, centered))
            if denominator > 0.0:
                result[column] = np.float32(
                    np.dot(centered, self.target_values[mask, column] - self.target_values[mask, column].mean())
                    / denominator
                )
        result.setflags(write=False)
        return result


def _command_vector(scenario: Scenario, command: Mapping[str, Any]) -> np.ndarray:
    canonical = validate_external_command(scenario, command)
    value = canonical.to_mapping()
    zones = tuple(sorted(str(zone["id"]) for zone in scenario.data["zones"]))
    dampers = tuple(
        sorted(str(branch["damper_id"]) for branch in scenario.data["air_network"]["branches"])
    )
    vector = [float(value["fan_speed_fraction"])]
    vector.extend(float(value["damper_position_by_id"][damper]) for damper in dampers)
    vector.extend((float(value["scrubber_duty"]), float(value["condenser_duty"])))
    vector.extend(float(value["cooling_removed_w"][zone]) for zone in zones)
    vector.extend(float(value["oxygen_injection_mol_s"][zone]) for zone in zones)
    result = np.asarray(vector, dtype=np.float32)
    result.setflags(write=False)
    return result


def _blend_command(
    base: Mapping[str, Any], goal: Mapping[str, Any], alpha: float
) -> dict[str, Any]:
    result = dict(base)
    for field_name in ("scrubber_duty", "condenser_duty", "fan_speed_fraction"):
        result[field_name] = float(
            base[field_name] + (goal[field_name] - base[field_name]) * alpha
        )
    for field_name in (
        "damper_position_by_id",
        "cooling_removed_w",
        "oxygen_injection_mol_s",
    ):
        result[field_name] = {
            key: float(base[field_name][key] + (goal[field_name][key] - base[field_name][key]) * alpha)
            for key in sorted(base[field_name])
        }
    return result


def _validate_candidate_command(
    scenario: Scenario, command: Mapping[str, Any]
) -> CanonicalExternalCommand:
    """Canonicalize generated commands without rejecting round-off at a shared cap."""

    adjusted = _copy_json(command)
    oxygen = adjusted["oxygen_injection_mol_s"]
    capacity = float(scenario.data["equipment"]["oxygen_injection_max_total_mol_s"])
    total = sum(float(value) for value in oxygen.values())
    if total > capacity:
        excess = total - capacity
        for zone_id in sorted(oxygen, reverse=True):
            value = float(oxygen[zone_id])
            reduction = min(value, excess)
            oxygen[zone_id] = value - reduction
            excess -= reduction
            if excess <= 0.0:
                break
    return validate_external_command(scenario, adjusted)


def _goal_command(base: Mapping[str, Any], spec: Mapping[str, float]) -> dict[str, Any]:
    goal = _copy_json(base)
    goal["fan_speed_fraction"] = min(1.0, max(0.0, float(base["fan_speed_fraction"]) + spec["fan"]))
    goal["scrubber_duty"] = min(1.0, max(0.0, float(base["scrubber_duty"]) + spec["scrubber"]))
    goal["condenser_duty"] = min(1.0, max(0.0, float(base["condenser_duty"]) + spec["condenser"]))
    for damper_id in goal["damper_position_by_id"]:
        offset = spec["damper"] * _stable_damper_factor(damper_id)
        goal["damper_position_by_id"][damper_id] = min(
            1.0, max(0.0, float(goal["damper_position_by_id"][damper_id]) + offset)
        )
    for zone_id in goal["cooling_removed_w"]:
        goal["cooling_removed_w"][zone_id] = min(
            1_000.0,
            max(0.0, float(goal["cooling_removed_w"][zone_id]) + spec["cooling"]),
        )
        goal["oxygen_injection_mol_s"][zone_id] = max(
            0.0,
            float(goal["oxygen_injection_mol_s"][zone_id]) + spec["oxygen"],
        )
    if spec["oxygen"] > 0.0:
        total = sum(float(value) for value in goal["oxygen_injection_mol_s"].values())
        maximum = float(base.get("_issue52_oxygen_capacity", math.inf))
        if total > maximum:
            excess = total - maximum
            adjustable = sorted(
                goal["oxygen_injection_mol_s"],
                key=lambda zone_id: (
                    -float(goal["oxygen_injection_mol_s"][zone_id]),
                    zone_id,
                ),
            )
            for zone_id in adjustable:
                reduction = min(excess, float(goal["oxygen_injection_mol_s"][zone_id]))
                goal["oxygen_injection_mol_s"][zone_id] = float(
                    goal["oxygen_injection_mol_s"][zone_id]
                ) - reduction
                excess -= reduction
                if excess <= 0.0:
                    break
        remaining = maximum - sum(
            float(value) for value in goal["oxygen_injection_mol_s"].values()
        )
        if remaining < 0.0:
            final_zone = sorted(goal["oxygen_injection_mol_s"])[-1]
            goal["oxygen_injection_mol_s"][final_zone] = max(
                0.0,
                float(goal["oxygen_injection_mol_s"][final_zone]) + remaining,
            )
    return goal


@dataclass(frozen=True, slots=True)
class CandidateSchedule:
    candidate_id: str
    purpose: str
    commands: tuple[CanonicalExternalCommand, ...]
    schedule_sha256: str
    applicable_modes: tuple[str, ...] = ALL_OPERATING_MODES

    def __post_init__(self) -> None:
        if type(self.candidate_id) is not str or not self.candidate_id:
            raise Issue52ContractError("candidate ID must be non-empty")
        if type(self.purpose) is not str or not self.purpose:
            raise Issue52ContractError("candidate purpose must be non-empty")
        if type(self.commands) is not tuple or len(self.commands) != HORIZON_STEPS:
            raise Issue52ContractError("candidate schedule must contain 32 commands")
        if any(type(command) is not CanonicalExternalCommand for command in self.commands):
            raise Issue52ContractError("candidate schedule contains a noncanonical command")
        if (
            type(self.applicable_modes) is not tuple
            or not self.applicable_modes
            or any(mode not in ALL_OPERATING_MODES for mode in self.applicable_modes)
            or len(set(self.applicable_modes)) != len(self.applicable_modes)
            or tuple(
                mode for mode in ALL_OPERATING_MODES if mode in self.applicable_modes
            )
            != self.applicable_modes
        ):
            raise Issue52ContractError("candidate schedule operating modes are invalid")
        _require_sha256(self.schedule_sha256, label="candidate schedule identity")
        payload = {
            "candidate_id": self.candidate_id,
            "purpose": self.purpose,
            "applicable_modes": list(self.applicable_modes),
            "commands": [command.to_mapping() for command in self.commands],
            "command_sha256": [command.sha256 for command in self.commands],
        }
        if self.schedule_sha256 != _sha(payload):
            raise Issue52ContractError("candidate schedule digest is inconsistent")

    @property
    def first_command(self) -> CanonicalExternalCommand:
        return self.commands[0]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "purpose": self.purpose,
            "applicable_modes": list(self.applicable_modes),
            "commands": [command.to_mapping() for command in self.commands],
            "command_sha256": [command.sha256 for command in self.commands],
            "schedule_sha256": self.schedule_sha256,
        }


@dataclass(frozen=True, slots=True)
class CandidateCatalogue:
    scenario_sha256: str
    topology_sha256: str
    candidates: tuple[CandidateSchedule, ...]
    catalogue_sha256: str
    base_command_sha256: str

    @classmethod
    def from_scenario(
        cls,
        scenario: Scenario,
        base_command: Mapping[str, Any] | CanonicalExternalCommand | None = None,
    ) -> "CandidateCatalogue":
        if type(scenario) is not Scenario:
            raise Issue52ContractError("candidate catalogue requires Scenario")
        topology = derive_observable_topology(scenario)
        if base_command is None:
            base = command_from_achieved_state(scenario, initial_state(scenario)).command
        elif type(base_command) is CanonicalExternalCommand:
            base = validate_external_command(scenario, base_command.to_mapping())
        else:
            try:
                base = validate_external_command(scenario, base_command)
            except (TypeError, ScenarioValidationError) as error:
                raise Issue52ContractError("candidate base command is invalid") from error
        base_mapping = base.to_mapping()
        goal_base = {
            **base_mapping,
            "_issue52_oxygen_capacity": float(
                scenario.data["equipment"]["oxygen_injection_max_total_mol_s"]
            ),
        }
        specs: tuple[tuple[str, str, dict[str, float]], ...] = (
            ("candidate_balanced", "balanced ventilation and resource use", {"fan": 0.10, "damper": 0.08, "scrubber": 0.08, "condenser": 0.06, "cooling": 80.0, "oxygen": 0.00005}),
            ("candidate_ventilation", "higher circulation for environmental mixing", {"fan": 0.16, "damper": 0.12, "scrubber": 0.02, "condenser": 0.0, "cooling": 30.0, "oxygen": 0.0}),
            ("candidate_scrubbing", "higher CO2 removal", {"fan": 0.04, "damper": 0.04, "scrubber": 0.18, "condenser": 0.04, "cooling": 20.0, "oxygen": 0.0}),
            ("candidate_cooling", "thermal protection", {"fan": 0.06, "damper": 0.04, "scrubber": 0.02, "condenser": 0.08, "cooling": 220.0, "oxygen": 0.0}),
            ("candidate_dehumidifying", "humidity control", {"fan": 0.05, "damper": 0.03, "scrubber": 0.02, "condenser": 0.16, "cooling": 100.0, "oxygen": 0.0}),
            ("candidate_oxygen", "oxygen reserve support", {"fan": 0.05, "damper": 0.04, "scrubber": 0.02, "condenser": 0.02, "cooling": 20.0, "oxygen": 0.00018}),
            ("candidate_resource_preserve", "lower resource expenditure", {"fan": -0.08, "damper": -0.06, "scrubber": -0.08, "condenser": -0.08, "cooling": -40.0, "oxygen": 0.0}),
            ("candidate_laboratory", "laboratory branch support", {"fan": 0.08, "damper": 0.16, "scrubber": 0.05, "condenser": 0.03, "cooling": 70.0, "oxygen": 0.00004}),
            ("candidate_crew", "crew-zone support", {"fan": 0.08, "damper": 0.10, "scrubber": 0.07, "condenser": 0.04, "cooling": 90.0, "oxygen": 0.00008}),
            ("candidate_low_intervention", "small bounded adjustment", {"fan": 0.03, "damper": 0.03, "scrubber": 0.03, "condenser": 0.03, "cooling": 20.0, "oxygen": 0.00002}),
            ("candidate_high_protection", "combined safety-margin protection", {"fan": 0.12, "damper": 0.10, "scrubber": 0.12, "condenser": 0.12, "cooling": 180.0, "oxygen": 0.00010}),
        )
        candidates: list[CandidateSchedule] = []
        for candidate_id, purpose, spec in (("candidate_hold", "retain achieved actuator state", {"fan": 0.0, "damper": 0.0, "scrubber": 0.0, "condenser": 0.0, "cooling": 0.0, "oxygen": 0.0}), *specs):
            goal = _goal_command(goal_base, spec)
            commands = tuple(
                _validate_candidate_command(
                    scenario,
                    _blend_command(
                        {key: value for key, value in base_mapping.items() if key != "_issue52_oxygen_capacity"},
                        goal,
                        min(1.0, (step + 1) / float(HORIZON_STEPS)),
                    ),
                )
                for step in range(HORIZON_STEPS)
            )
            payload = {
                "candidate_id": candidate_id,
                "purpose": purpose,
                "applicable_modes": list(ALL_OPERATING_MODES),
                "commands": [command.to_mapping() for command in commands],
                "command_sha256": [command.sha256 for command in commands],
            }
            candidates.append(
                CandidateSchedule(
                    candidate_id=candidate_id,
                    purpose=purpose,
                    commands=commands,
                    schedule_sha256=_sha(payload),
                    applicable_modes=ALL_OPERATING_MODES,
                )
            )
        if len(candidates) != CATALOGUE_SIZE:
            raise AssertionError("issue 52 candidate catalogue size drift")
        command_sequences = {
            tuple(command.sha256 for command in candidate.commands)
            for candidate in candidates
        }
        if len(command_sequences) != CATALOGUE_SIZE:
            raise Issue52ContractError("candidate catalogue contains duplicate schedules")
        catalogue_payload = {
            "schema_version": f"{ISSUE52_SCHEMA_VERSION}.candidate_catalogue",
            "scenario_sha256": scenario.scenario_sha256,
            "topology_sha256": topology.sha256,
            "base_command_sha256": base.sha256,
            "candidates": [candidate.to_mapping() for candidate in candidates],
        }
        return cls(
            scenario_sha256=scenario.scenario_sha256,
            topology_sha256=topology.sha256,
            candidates=tuple(candidates),
            catalogue_sha256=_sha(catalogue_payload),
            base_command_sha256=base.sha256,
        )

    def __post_init__(self) -> None:
        if len(self.candidates) != CATALOGUE_SIZE:
            raise Issue52ContractError("catalogue must contain exactly 12 candidates")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(ids)) != len(ids) or "candidate_hold" not in ids:
            raise Issue52ContractError("catalogue candidate IDs are not unique")
        if any(len(candidate.commands) != HORIZON_STEPS for candidate in self.candidates):
            raise Issue52ContractError("every candidate must contain 32 commands")
        if len({tuple(command.sha256 for command in candidate.commands) for candidate in self.candidates}) != CATALOGUE_SIZE:
            raise Issue52ContractError("catalogue candidate schedules are not unique")
        _require_sha256(self.scenario_sha256, label="catalogue scenario identity")
        _require_sha256(self.topology_sha256, label="catalogue topology identity")
        _require_sha256(self.base_command_sha256, label="catalogue base command identity")
        payload = {
            "schema_version": f"{ISSUE52_SCHEMA_VERSION}.candidate_catalogue",
            "scenario_sha256": self.scenario_sha256,
            "topology_sha256": self.topology_sha256,
            "base_command_sha256": self.base_command_sha256,
            "candidates": [candidate.to_mapping() for candidate in self.candidates],
        }
        if self.catalogue_sha256 != _sha(payload):
            raise Issue52ContractError("candidate catalogue digest is inconsistent")


@dataclass(frozen=True, slots=True)
class CandidateFeasibility:
    candidate_id: str
    static_status: str
    rollout_status: str
    runtime_status: str
    rollout_reason: str | None = None
    runtime_reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.candidate_id) is not str or not self.candidate_id:
            raise Issue52ContractError("feasibility candidate ID is invalid")
        if self.static_status not in {"STATICALLY_VALID", "STATICALLY_INVALID"}:
            raise Issue52ContractError("static feasibility status is invalid")
        if self.rollout_status not in {"ROLLOUT_FEASIBLE", "ROLLOUT_INFEASIBLE", "NOT_EVALUATED"}:
            raise Issue52ContractError("rollout feasibility status is invalid")
        if self.runtime_status not in {
            "RUNTIME_FIRST_STEP_FEASIBLE",
            "RUNTIME_FIRST_STEP_INFEASIBLE",
            "NOT_EVALUATED",
        }:
            raise Issue52ContractError("runtime feasibility status is invalid")


def assess_static_feasibility(
    scenario: Scenario, catalogue: CandidateCatalogue
) -> tuple[CandidateFeasibility, ...]:
    """Validate every candidate command independently of a current plant state."""

    results: list[CandidateFeasibility] = []
    for candidate in catalogue.candidates:
        try:
            for command in candidate.commands:
                validate_external_command(scenario, command.to_mapping())
        except (ScenarioValidationError, ValueError) as error:
            results.append(
                CandidateFeasibility(
                    candidate.candidate_id,
                    "STATICALLY_INVALID",
                    "NOT_EVALUATED",
                    "NOT_EVALUATED",
                    rollout_reason=type(error).__name__,
                )
            )
        else:
            results.append(
                CandidateFeasibility(
                    candidate.candidate_id,
                    "STATICALLY_VALID",
                    "NOT_EVALUATED",
                    "NOT_EVALUATED",
                )
            )
    return tuple(results)


def extend_scenario_for_issue52(
    scenario: Scenario, *, minimum_steps: int = HISTORY_STEPS + HORIZON_STEPS - 1
) -> Scenario:
    """Repeat the existing deterministic timeline into a causal V5 horizon.

    This creates an experiment scenario from an already validated scenario.  It
    does not alter the runtime scenario or HMC policy and is intentionally
    separate from the runtime advisory source.
    """

    if type(scenario) is not Scenario:
        raise Issue52ContractError("scenario extension requires Scenario")
    if minimum_steps < MIN_SCENARIO_STEPS:
        raise Issue52ContractError("minimum horizon is not long enough")
    if float(scenario.data["dt_seconds"]) != CADENCE_SECONDS:
        raise Issue52ContractError("Issue 52 scenarios require a 60-second cadence")
    if int(scenario.data["steps"]) >= minimum_steps:
        return scenario
    data = _copy_json(scenario.data)
    original = list(data["timeline"])
    if not original:
        raise Issue52ContractError("scenario has no timeline")
    repeated: list[dict[str, Any]] = []
    cursor = 0
    while cursor < minimum_steps:
        for segment in original:
            duration = int(segment["end_step"]) - int(segment["start_step"])
            if duration <= 0:
                raise Issue52ContractError("scenario timeline contains empty segment")
            end = min(minimum_steps, cursor + duration)
            item = _copy_json(segment)
            item["start_step"] = cursor
            item["end_step"] = end
            repeated.append(item)
            cursor = end
            if cursor >= minimum_steps:
                break
    data["steps"] = minimum_steps
    data["timeline"] = repeated
    try:
        return Scenario.from_mapping(data)
    except ScenarioValidationError as error:
        raise Issue52ContractError("extended scenario failed V5 validation") from error


@dataclass(frozen=True, slots=True)
class TrainingSample:
    family_id: str
    split: str
    scenario_sha256: str
    manifest_sha256: str
    checkpoint_sha256: str
    schedule_sha256: str
    history: ForecastHistory
    schedule: CandidateSchedule
    targets: np.ndarray = field(compare=False)

    def __post_init__(self) -> None:
        if type(self.family_id) is not str or not self.family_id:
            raise Issue52ForecastError("training family identity is invalid")
        if self.split not in {"TRAIN", "VALIDATION", "FINAL"}:
            raise Issue52ForecastError("training split is invalid")
        for label, value in (
            ("scenario", self.scenario_sha256),
            ("manifest", self.manifest_sha256),
            ("checkpoint", self.checkpoint_sha256),
            ("schedule", self.schedule_sha256),
        ):
            _require_sha256(value, label=f"training {label} identity")
        if self.schedule_sha256 != self.schedule.schedule_sha256:
            raise Issue52ForecastError("training schedule identity is inconsistent")
        if any(
            record.scenario_sha256 != self.scenario_sha256
            for record in self.history.records
        ):
            raise Issue52ForecastError("training history scenario identity is inconsistent")
        values = np.asarray(self.targets, dtype=np.float32)
        if values.shape != (HORIZON_STEPS, self.history.target_values.shape[1]):
            raise Issue52ForecastError("training target shape is invalid")
        if not np.isfinite(values).all():
            raise Issue52ForecastError("training targets must be finite")
        values = values.copy()
        values.setflags(write=False)
        object.__setattr__(self, "targets", values)


@dataclass(frozen=True, slots=True)
class ForecastTrajectory:
    status: str
    mean: np.ndarray | None = field(compare=False)
    lower: np.ndarray | None = field(compare=False)
    upper: np.ndarray | None = field(compare=False)
    model_id: str
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"PREDICTION", "ABSTAIN", "INVALID_OUTPUT"}:
            raise Issue52ForecastError("forecast trajectory status is invalid")
        if type(self.model_id) is not str or not self.model_id:
            raise Issue52ForecastError("forecast trajectory model identity is invalid")
        present = (self.mean, self.lower, self.upper)
        if self.status == "PREDICTION" and any(value is None for value in present):
            raise Issue52ForecastError("prediction trajectory is missing an interval")
        if self.status != "PREDICTION" and any(value is not None for value in present):
            raise Issue52ForecastError("non-prediction trajectory contains output data")
        if self.status == "PREDICTION":
            mean = np.asarray(self.mean, dtype=np.float32)
            lower = np.asarray(self.lower, dtype=np.float32)
            upper = np.asarray(self.upper, dtype=np.float32)
            if (
                mean.ndim != 2
                or lower.shape != mean.shape
                or upper.shape != mean.shape
                or not np.isfinite(mean).all()
                or not np.isfinite(lower).all()
                or not np.isfinite(upper).all()
                or np.any(lower > upper)
            ):
                raise Issue52ForecastError("prediction trajectory arrays are invalid")
            object.__setattr__(self, "mean", _readonly(mean))
            object.__setattr__(self, "lower", _readonly(lower))
            object.__setattr__(self, "upper", _readonly(upper))


def _history_uncertainty(history: ForecastHistory) -> np.ndarray:
    values = history.target_values.astype(np.float64).copy()
    values[~history.available_mask] = np.nan
    differences = np.diff(values, axis=0)
    result = np.nanstd(differences, axis=0).astype(np.float32)
    result[~np.isfinite(result)] = 0.0
    result = np.maximum(result, np.float32(0.01))
    result.setflags(write=False)
    return result


def _feature_matrix(
    history: ForecastHistory,
    schedule: CandidateSchedule,
    scenario: Scenario,
) -> np.ndarray:
    if not np.all(history.available_mask[-1]):
        raise Issue52ForecastError("latest history row contains unavailable targets")
    latest = history.latest.astype(np.float64)
    slope = history.slope.astype(np.float64) / CADENCE_SECONDS
    rows: list[np.ndarray] = []
    for horizon, command in enumerate(schedule.commands):
        action = _command_vector(scenario, command.to_mapping()).astype(np.float64)
        rows.append(
            np.concatenate(
                ([1.0], latest, slope, action, [horizon / float(HORIZON_STEPS - 1)])
            )
        )
    return np.stack(rows)


@dataclass(frozen=True, slots=True)
class ActionConditionedLinearForecaster:
    scenario: Scenario
    manifest: TargetManifest
    coefficients: np.ndarray | None = field(default=None, compare=False)
    model_id: str = "issue52-action-conditioned-linear-v1"

    def __post_init__(self) -> None:
        if type(self.scenario) is not Scenario or type(self.manifest) is not TargetManifest:
            raise Issue52ForecastError("forecaster requires exact scenario and manifest")
        if self.manifest.scenario_sha256 != self.scenario.scenario_sha256:
            raise Issue52ForecastError("forecaster manifest does not bind scenario")
        if type(self.model_id) is not str or not self.model_id:
            raise Issue52ForecastError("forecaster model identity is invalid")
        if self.coefficients is not None:
            values = np.asarray(self.coefficients, dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != self.manifest.width or not np.isfinite(
                values
            ).all():
                raise Issue52ForecastError("forecaster coefficients are invalid")
            object.__setattr__(self, "coefficients", _readonly(values))

    @classmethod
    def fit(
        cls,
        samples: Sequence[TrainingSample],
        *,
        scenario: Scenario | None = None,
        manifest: TargetManifest | None = None,
        alpha: float = 1e-6,
    ) -> "ActionConditionedLinearForecaster":
        if type(scenario) is not Scenario:
            raise Issue52ForecastError("fit requires an exact Scenario")
        bound_manifest = (
            TargetManifest.from_scenario(scenario) if manifest is None else manifest
        )
        return cls.fit_for_scenario(
            scenario,
            bound_manifest,
            samples,
            alpha=alpha,
        )

    @classmethod
    def fit_for_scenario(
        cls,
        scenario: Scenario,
        manifest: TargetManifest,
        samples: Sequence[TrainingSample],
        *,
        alpha: float = 1e-6,
    ) -> "ActionConditionedLinearForecaster":
        items = tuple(samples)
        if len(items) < 3 or len({item.family_id for item in items}) < 2:
            raise Issue52ForecastError("fitting requires at least three samples and two families")
        if any(item.split != "TRAIN" for item in items):
            raise Issue52ForecastError("fitting accepts TRAIN samples only")
        if type(scenario) is not Scenario or type(manifest) is not TargetManifest:
            raise Issue52ForecastError("fit inputs are not exact contract types")
        if manifest.scenario_sha256 != scenario.scenario_sha256:
            raise Issue52ForecastError("fit manifest does not bind scenario")
        if not math.isfinite(float(alpha)) or alpha <= 0.0:
            raise Issue52ForecastError("fit regularization must be positive and finite")
        if any(
            item.scenario_sha256 != scenario.scenario_sha256
            or item.manifest_sha256 != manifest.manifest_sha256
            or item.schedule_sha256 != item.schedule.schedule_sha256
            or item.history.target_values.shape[1] != manifest.width
            for item in items
        ):
            raise Issue52ForecastError("training history does not match manifest width")
        features = np.concatenate(
            [_feature_matrix(item.history, item.schedule, scenario) for item in items],
            axis=0,
        )
        targets = np.concatenate([item.targets for item in items], axis=0).astype(np.float64)
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise Issue52ForecastError("training data contains non-finite values")
        gram = features.T @ features
        regularizer = np.eye(gram.shape[0], dtype=np.float64) * float(alpha)
        try:
            coefficients = np.linalg.solve(gram + regularizer, features.T @ targets)
        except np.linalg.LinAlgError as error:
            raise Issue52ForecastError("action-conditioned fit is singular") from error
        coefficients = coefficients.astype(np.float32)
        coefficients.setflags(write=False)
        model_id = "issue52-action-conditioned-linear-" + hashlib.sha256(coefficients.tobytes()).hexdigest()[:16]
        return cls(scenario, manifest, coefficients, model_id)

    def _heuristic(self, history: ForecastHistory, schedule: CandidateSchedule) -> np.ndarray:
        latest = history.latest.astype(np.float64)
        slope = history.slope.astype(np.float64) / CADENCE_SECONDS
        previous_action = _command_vector(self.scenario, history.latest_record.command).astype(np.float64)
        width = self.manifest.width
        branch_count = len(self.scenario.data["air_network"]["branches"])
        zone_count = len(self.scenario.data["zones"])
        result = np.empty((HORIZON_STEPS, width), dtype=np.float64)
        for horizon, command in enumerate(schedule.commands):
            action = _command_vector(self.scenario, command.to_mapping()).astype(np.float64)
            delta = action - previous_action
            fan_delta = delta[0]
            damper_delta = float(np.mean(delta[1 : 1 + branch_count]))
            scrubber_delta = delta[1 + branch_count]
            condenser_delta = delta[2 + branch_count]
            cooling_start = 3 + branch_count
            cooling_delta = float(np.mean(delta[cooling_start : cooling_start + zone_count])) / 1000.0
            oxygen_delta = float(np.sum(delta[cooling_start + zone_count:]))
            row = latest + slope * float(horizon + 1)
            for index, descriptor in enumerate(self.manifest.descriptors):
                if descriptor.descriptor_id.endswith("/co2_ppm"):
                    effect = -90.0 * scrubber_delta - 20.0 * fan_delta - 10.0 * damper_delta
                elif descriptor.descriptor_id.endswith("/temperature_k"):
                    effect = -18.0 * cooling_delta - 2.0 * fan_delta
                elif descriptor.descriptor_id.endswith("/relative_humidity"):
                    effect = -0.22 * condenser_delta - 0.04 * cooling_delta
                elif descriptor.descriptor_id == "battery_state_of_charge":
                    effect = -0.03 * (abs(fan_delta) + abs(scrubber_delta) + abs(condenser_delta) + abs(cooling_delta))
                elif descriptor.descriptor_id == "oxygen_store_fraction":
                    effect = -8.0 * oxygen_delta
                else:
                    effect = -0.02 * max(0.0, scrubber_delta)
                row[index] += effect * float(horizon + 1) / HORIZON_STEPS
            result[horizon] = row
        return result

    def forecast(
        self, history: ForecastHistory, schedule: CandidateSchedule
    ) -> ForecastTrajectory:
        if (
            schedule.applicable_modes
            and history.latest_record.mode is not None
            and history.latest_record.mode not in schedule.applicable_modes
        ):
            return ForecastTrajectory(
                "ABSTAIN", None, None, None, self.model_id, "candidate_mode_inapplicable"
            )
        if history.target_values.shape[1] != self.manifest.width:
            return ForecastTrajectory("ABSTAIN", None, None, None, self.model_id, "manifest_width_mismatch")
        if not np.all(history.available_mask[-1]):
            return ForecastTrajectory("ABSTAIN", None, None, None, self.model_id, "latest_targets_unavailable")
        try:
            if self.coefficients is None:
                mean = self._heuristic(history, schedule)
            else:
                mean = _feature_matrix(history, schedule, self.scenario) @ self.coefficients
        except (Issue52ForecastError, ScenarioValidationError, ValueError):
            return ForecastTrajectory("INVALID_OUTPUT", None, None, None, self.model_id, "forecast_features_invalid")
        if not np.isfinite(mean).all():
            return ForecastTrajectory("INVALID_OUTPUT", None, None, None, self.model_id, "forecast_non_finite")
        uncertainty = _history_uncertainty(history).astype(np.float64)
        lower = np.empty_like(mean)
        upper = np.empty_like(mean)
        for index, descriptor in enumerate(self.manifest.descriptors):
            width = max(float(uncertainty[index]), descriptor.scale * 0.02)
            spread = width * np.sqrt(np.arange(1, HORIZON_STEPS + 1, dtype=np.float64))
            lower[:, index] = mean[:, index] - spread
            upper[:, index] = mean[:, index] + spread
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            return ForecastTrajectory(
                "INVALID_OUTPUT", None, None, None, self.model_id, "interval_non_finite"
            )
        if np.any(lower > upper):
            return ForecastTrajectory(
                "INVALID_OUTPUT", None, None, None, self.model_id, "interval_order_invalid"
            )
        normalized_width = np.asarray(
            [descriptor.scale for descriptor in self.manifest.descriptors],
            dtype=np.float64,
        )
        if float(
            np.max((upper - lower) / normalized_width[None, :])
        ) > MAX_NORMALIZED_UNCERTAINTY:
            return ForecastTrajectory(
                "ABSTAIN", None, None, None, self.model_id, "uncertainty_limit"
            )
        return ForecastTrajectory(
            "PREDICTION",
            _readonly(mean),
            _readonly(lower),
            _readonly(upper),
            self.model_id,
        )


@dataclass(frozen=True, slots=True)
class PersistenceForecaster:
    manifest: TargetManifest
    model_id: str = "issue52-persistence-v1"

    def forecast(
        self, history: ForecastHistory, schedule: CandidateSchedule
    ) -> ForecastTrajectory:
        if (
            schedule.applicable_modes
            and history.latest_record.mode is not None
            and history.latest_record.mode not in schedule.applicable_modes
        ):
            return ForecastTrajectory(
                "ABSTAIN", None, None, None, self.model_id, "candidate_mode_inapplicable"
            )
        if history.target_values.shape[1] != self.manifest.width or not np.all(history.available_mask[-1]):
            return ForecastTrajectory("ABSTAIN", None, None, None, self.model_id, "latest_targets_unavailable")
        mean = np.repeat(history.latest[None, :], HORIZON_STEPS, axis=0)
        uncertainty = _history_uncertainty(history)
        lower = np.empty_like(mean)
        upper = np.empty_like(mean)
        for index, descriptor in enumerate(self.manifest.descriptors):
            spread = max(float(uncertainty[index]), descriptor.scale * 0.02)
            lower[:, index] = float(mean[0, index] - spread)
            upper[:, index] = float(mean[0, index] + spread)
        if float(
            np.max(
                (upper - lower)
                / np.asarray(
                    [descriptor.scale for descriptor in self.manifest.descriptors],
                    dtype=np.float64,
                )[None, :]
            )
        ) > MAX_NORMALIZED_UNCERTAINTY:
            return ForecastTrajectory(
                "ABSTAIN", None, None, None, self.model_id, "uncertainty_limit"
            )
        return ForecastTrajectory("PREDICTION", _readonly(mean), _readonly(lower), _readonly(upper), self.model_id)


@dataclass(frozen=True, slots=True)
class CandidateScore:
    candidate_id: str
    score: float
    hard_ineligible: bool
    safety_exposure: float
    tracking_error: float
    uncertainty: float
    intervention: float
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RankedDecision:
    outcome: str
    selected_candidate_id: str | None
    scores: tuple[CandidateScore, ...]
    reason: str | None = None


def score_trajectory(
    manifest: TargetManifest,
    history: ForecastHistory,
    candidate: CandidateSchedule,
    trajectory: ForecastTrajectory,
    scenario: Scenario,
    health_policy: Any | None = None,
) -> CandidateScore:
    if trajectory.status != "PREDICTION" or trajectory.mean is None or trajectory.lower is None or trajectory.upper is None:
        return CandidateScore(candidate.candidate_id, math.inf, True, math.inf, math.inf, math.inf, math.inf, trajectory.reason)
    mean = np.asarray(trajectory.mean, dtype=np.float64)
    lower = np.asarray(trajectory.lower, dtype=np.float64)
    upper = np.asarray(trajectory.upper, dtype=np.float64)
    if (
        mean.shape != (HORIZON_STEPS, manifest.width)
        or lower.shape != mean.shape
        or upper.shape != mean.shape
        or not np.isfinite(mean).all()
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower > upper)
    ):
        return CandidateScore(candidate.candidate_id, math.inf, True, math.inf, math.inf, math.inf, math.inf, "shape_or_finite_failure")
    safety = 0.0
    tracking = 0.0
    uncertainty = 0.0
    hard = False
    for index, descriptor in enumerate(manifest.descriptors):
        scale = max(descriptor.scale, 1e-9)
        safety_lower, safety_upper = _safety_bounds_for_descriptor(
            descriptor, health_policy
        )
        tracking += float(np.mean(np.abs(mean[:, index] - descriptor.nominal) / scale))
        lower_crossing = np.maximum(0.0, safety_lower - lower[:, index]) / scale
        upper_crossing = np.maximum(0.0, upper[:, index] - safety_upper) / scale
        safety += float(np.mean(lower_crossing + upper_crossing))
        uncertainty += float(np.mean((upper[:, index] - lower[:, index]) / scale))
        hard = hard or bool(
            np.any(lower[:, index] < safety_lower)
            or np.any(upper[:, index] > safety_upper)
        )
    hard = hard or uncertainty > MAX_NORMALIZED_UNCERTAINTY
    current = _command_vector(scenario, history.latest_record.command).astype(np.float64)
    first = _command_vector(scenario, candidate.first_command.to_mapping()).astype(np.float64)
    intervention = float(np.mean(np.abs(first - current)))
    score = tracking + 0.5 * safety + 0.1 * uncertainty + 0.05 * intervention
    return CandidateScore(
        candidate_id=candidate.candidate_id,
        score=score if not hard else math.inf,
        hard_ineligible=hard,
        safety_exposure=safety,
        tracking_error=tracking,
        uncertainty=uncertainty,
        intervention=intervention,
        reason=(
            "forecast_uncertainty_limit"
            if uncertainty > MAX_NORMALIZED_UNCERTAINTY
            else "predicted_hard_bound_crossing"
            if hard
            else None
        ),
    )


def rank_candidates(
    catalogue: CandidateCatalogue,
    manifest: TargetManifest,
    history: ForecastHistory,
    trajectories: Mapping[str, ForecastTrajectory],
    scenario: Scenario,
    *,
    ambiguity_margin: float = AMBIGUITY_MARGIN,
    health_policy: Any | None = None,
) -> RankedDecision:
    if not math.isfinite(float(ambiguity_margin)) or ambiguity_margin < 0.0:
        raise Issue52ForecastError("candidate ambiguity margin is invalid")
    expected_ids = {candidate.candidate_id for candidate in catalogue.candidates}
    if set(trajectories) != expected_ids:
        return RankedDecision("INVALID_OUTPUT", None, (), "forecast_candidate_set_mismatch")
    if any(
        trajectory.status == "INVALID_OUTPUT" for trajectory in trajectories.values()
    ):
        return RankedDecision("INVALID_OUTPUT", None, (), "forecast_invalid_output")
    if any(trajectory.status != "PREDICTION" for trajectory in trajectories.values()):
        return RankedDecision("ABSTAINED", None, (), "forecast_abstention")
    current_mode = history.latest_record.mode
    application_mode = None
    if current_mode is not None:
        try:
            from .physics import operating_mode_for_application_step

            application_mode = operating_mode_for_application_step(
                scenario, history.latest_record.completed_step
            )
        except ScenarioValidationError:
            application_mode = current_mode
    if application_mode is not None and any(
        application_mode not in candidate.applicable_modes for candidate in catalogue.candidates
    ):
        return RankedDecision("INVALID_OUTPUT", None, (), "candidate_mode_catalogue_mismatch")
    scores = tuple(
        score_trajectory(
            manifest,
            history,
            candidate,
            trajectories[candidate.candidate_id],
            scenario,
            health_policy,
        )
        for candidate in catalogue.candidates
    )
    eligible = [score for score in scores if not score.hard_ineligible and math.isfinite(score.score)]
    if not eligible:
        return RankedDecision("ABSTAINED", None, scores, "no_candidate_passed_forecast_gates")
    selected = min(
        eligible,
        key=lambda item: (
            item.score,
            item.safety_exposure,
            item.uncertainty,
            item.intervention,
            0 if item.candidate_id == "candidate_hold" else 1,
            item.candidate_id,
        ),
    )
    ordered = sorted(
        eligible,
        key=lambda item: (
            item.score,
            item.safety_exposure,
            item.uncertainty,
            item.intervention,
            0 if item.candidate_id == "candidate_hold" else 1,
            item.candidate_id,
        ),
    )
    if len(ordered) > 1 and ordered[1].score - ordered[0].score <= float(
        ambiguity_margin
    ):
        return RankedDecision("ABSTAINED", None, scores, "candidate_margin_ambiguous")
    outcome = "SELECTED_HOLD" if selected.candidate_id == "candidate_hold" else "SELECTED_CANDIDATE"
    return RankedDecision(outcome, selected.candidate_id, scores)


def build_control_proposal(
    hmc: Any,
    snapshot: OperationalSnapshot,
    decision: RankedDecision,
    catalogue: CandidateCatalogue,
) -> dict[str, Any] | None:
    if decision.selected_candidate_id is None:
        return None
    candidate = next(
        (item for item in catalogue.candidates if item.candidate_id == decision.selected_candidate_id),
        None,
    )
    if candidate is None:
        raise Issue52ForecastError("ranker selected an unknown candidate")
    snapshot_mapping = snapshot.to_mapping()
    completed_step = snapshot_mapping.get("completed_step")
    if isinstance(completed_step, bool) or not isinstance(completed_step, int):
        raise Issue52ForecastError("snapshot completed step is invalid")
    score = next(item.score for item in decision.scores if item.candidate_id == candidate.candidate_id)
    confidence = 1.0 / (1.0 + max(0.0, score)) if math.isfinite(score) else None
    body = {
        "schema_version": "aeolus_habitat_v2_control_proposal_v1",
        "control_run_id": hmc.control_run_id,
        "authority_epoch": hmc.authority_epoch,
        "source_id": "issue52-advisory-v1",
        "source_type": "forecast_ranker",
        "completed_observation_step": completed_step,
        "observation_snapshot_sha256": snapshot.snapshot_sha256,
        "requested_application_step": completed_step,
        "observable_topology_sha256": hmc.observable_topology_sha256,
        "proposed_command": candidate.first_command.to_mapping(),
        "confidence": confidence,
    }
    return {**body, "proposal_sha256": _sha(body)}


@dataclass(frozen=True, slots=True)
class AdvisoryDecision:
    outcome: str
    candidate_id: str | None
    proposal: Mapping[str, Any] | None
    ranked: RankedDecision | None
    history_status: str
    reason: str | None = None
    latency_ms: float | None = None
    hmc_receipt_sha256: str | None = None


@dataclass(slots=True)
class Issue52AdvisorySource:
    scenario: Scenario
    manifest: TargetManifest
    catalogue: CandidateCatalogue
    forecaster: ActionConditionedLinearForecaster
    enabled: bool = False
    inference_deadline_ms: float = INFERENCE_DEADLINE_MS
    ambiguity_margin: float = AMBIGUITY_MARGIN
    history: VerifiedHistoryBuffer = field(init=False)
    _outcome_counts: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.manifest.scenario_sha256 != self.scenario.scenario_sha256:
            raise Issue52ContractError("advisory manifest does not bind scenario")
        if self.catalogue.scenario_sha256 != self.scenario.scenario_sha256:
            raise Issue52ContractError("advisory catalogue does not bind scenario")
        if self.forecaster.scenario.scenario_sha256 != self.scenario.scenario_sha256:
            raise Issue52ContractError("advisory forecaster does not bind scenario")
        if (
            not math.isfinite(float(self.inference_deadline_ms))
            or float(self.inference_deadline_ms) <= 0.0
            or not math.isfinite(float(self.ambiguity_margin))
            or float(self.ambiguity_margin) < 0.0
        ):
            raise Issue52ContractError("advisory runtime thresholds are invalid")
        self.history = VerifiedHistoryBuffer(self.manifest)
        self._outcome_counts = {outcome: 0 for outcome in OUTCOMES}

    @classmethod
    def create(
        cls,
        scenario: Scenario,
        *,
        enabled: bool = False,
        forecaster: ActionConditionedLinearForecaster | None = None,
        inference_deadline_ms: float = INFERENCE_DEADLINE_MS,
        ambiguity_margin: float = AMBIGUITY_MARGIN,
    ) -> "Issue52AdvisorySource":
        manifest = TargetManifest.from_scenario(scenario)
        catalogue = CandidateCatalogue.from_scenario(scenario)
        model = forecaster or ActionConditionedLinearForecaster(scenario, manifest)
        return cls(
            scenario,
            manifest,
            catalogue,
            model,
            enabled,
            inference_deadline_ms,
            ambiguity_margin,
        )

    @property
    def outcome_counts(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._outcome_counts))

    def _decision(
        self,
        outcome: str,
        candidate_id: str | None,
        proposal: Mapping[str, Any] | None,
        ranked: RankedDecision | None,
        history_status: str,
        reason: str | None = None,
        latency_ms: float | None = None,
        hmc_receipt_sha256: str | None = None,
    ) -> AdvisoryDecision:
        self._outcome_counts.setdefault(outcome, 0)
        self._outcome_counts[outcome] += 1
        return AdvisoryDecision(
            outcome,
            candidate_id,
            proposal,
            ranked,
            history_status,
            reason,
            latency_ms,
            hmc_receipt_sha256,
        )

    def decide(
        self,
        hmc: Any,
        snapshot: OperationalSnapshot,
        verification: SnapshotVerificationReceipt,
    ) -> AdvisoryDecision:
        started_ns = time.perf_counter_ns()
        if not self.enabled:
            return self._decision("DISABLED", None, None, None, "DISABLED")
        try:
            appended = self.history.append(
                hmc,
                snapshot,
                verification,
                cadence_seconds=float(self.scenario.data["dt_seconds"]),
                scenario=self.scenario,
            )
        except Issue52HistoryError as error:
            return self._decision("INVALID_OUTPUT", None, None, None, "RESET", str(error))
        if not self.history.ready:
            return self._decision("WARMUP_NO_PROPOSAL", None, None, None, appended.status)
        try:
            _validate_live_hmc_binding(hmc, self.scenario)
            if getattr(hmc, "lifecycle_phase", None) != "OBSERVED":
                raise Issue52HistoryError("advisory decision requires OBSERVED HMC phase")
            history = self.history.forecast_history()
            self.catalogue = CandidateCatalogue.from_scenario(
                self.scenario,
                base_command=history.latest_record.command,
            )
            trajectories = {
                candidate.candidate_id: self.forecaster.forecast(history, candidate)
                for candidate in self.catalogue.candidates
            }
            ranked = rank_candidates(
                self.catalogue,
                self.manifest,
                history,
                trajectories,
                self.scenario,
                ambiguity_margin=self.ambiguity_margin,
                health_policy=getattr(hmc, "advisory_safety_policy", lambda: None)(),
            )
            proposal = build_control_proposal(hmc, snapshot, ranked, self.catalogue)
            if proposal is not None:
                completed_step = snapshot.to_mapping().get("completed_step")
                try:
                    preflight = hmc.preflight_advisory_command(
                        proposal["proposed_command"],
                        int(completed_step),
                    )
                except Exception as error:  # noqa: BLE001 - preflight is advisory only
                    raise Issue52ForecastError("runtime first-step preflight failed") from error
                if preflight.classification != "FEASIBLE":
                    return self._decision(
                        "ABSTAINED",
                        None,
                        None,
                        ranked,
                        appended.status,
                        "runtime_first_step_infeasible",
                    )
        except Exception as error:  # noqa: BLE001 - advisory failures must not block HMC
            return self._decision("INVALID_OUTPUT", None, None, None, appended.status, str(error))
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        if elapsed_ms > float(self.inference_deadline_ms):
            return self._decision(
                "TIMEOUT_NO_PROPOSAL",
                None,
                None,
                ranked,
                appended.status,
                "inference_deadline_exceeded",
            )
        if ranked.selected_candidate_id is None or proposal is None:
            return self._decision(ranked.outcome, None, None, ranked, appended.status, ranked.reason)
        return self._decision(
            ranked.outcome,
            ranked.selected_candidate_id,
            proposal,
            ranked,
            appended.status,
        )

    def submit_decision(
        self, hmc: Any, decision: AdvisoryDecision
    ) -> ProposalReceipt:
        """Submit exactly the proposal already computed for the current cycle."""

        receipt = hmc.propose(None if decision.proposal is None else dict(decision.proposal))
        return receipt

    def submit(
        self,
        hmc: Any,
        snapshot: OperationalSnapshot,
        verification: SnapshotVerificationReceipt,
    ) -> tuple[AdvisoryDecision, ProposalReceipt]:
        decision = self.decide(hmc, snapshot, verification)
        return decision, self.submit_decision(hmc, decision)

    def _reclassify(
        self,
        decision: AdvisoryDecision,
        outcome: str,
        hmc_receipt_sha256: str,
        reason: str | None,
    ) -> AdvisoryDecision:
        """Reclassify one already-counted decision without double counting."""

        self._outcome_counts[decision.outcome] = max(
            0, self._outcome_counts[decision.outcome] - 1
        )
        self._outcome_counts.setdefault(outcome, 0)
        self._outcome_counts[outcome] += 1
        return AdvisoryDecision(
            outcome,
            decision.candidate_id,
            decision.proposal,
            decision.ranked,
            decision.history_status,
            reason,
            decision.latency_ms,
            hmc_receipt_sha256,
        )

    def reconcile_arbitration(
        self, decision: AdvisoryDecision, arbitration: Any
    ) -> AdvisoryDecision:
        """Record HMC's authoritative disposition after a submitted proposal."""

        if decision.proposal is None or not hasattr(arbitration, "to_mapping"):
            return decision
        try:
            mapping = arbitration.to_mapping()
            receipt_sha256 = str(mapping["arbitration_receipt_sha256"])
            emergency = mapping["emergency_override"] is True
            rejected_to_hold = (
                mapping["disposition"] == "REJECTED"
                and mapping["command_owner"] == "baseline_hold"
            )
        except Exception:  # noqa: BLE001 - receipt is authoritative only when intact
            return decision
        if emergency:
            return self._reclassify(
                decision,
                "HMC_EMERGENCY_OVERRIDDEN",
                receipt_sha256,
                decision.reason,
            )
        if rejected_to_hold:
            return self._reclassify(
                decision,
                "HMC_REJECTED_TO_HOLD",
                receipt_sha256,
                decision.reason,
            )
        return AdvisoryDecision(
            decision.outcome,
            decision.candidate_id,
            decision.proposal,
            decision.ranked,
            decision.history_status,
            decision.reason,
            decision.latency_ms,
            receipt_sha256,
        )


def normalized_mae(
    prediction: np.ndarray,
    truth: np.ndarray,
    manifest: TargetManifest,
    *,
    start_horizon: int = 1,
    end_horizon: int = HORIZON_STEPS,
    eligibility_mask: np.ndarray | None = None,
) -> float:
    predicted = np.asarray(prediction, dtype=np.float64)
    actual = np.asarray(truth, dtype=np.float64)
    if predicted.shape != actual.shape or predicted.ndim != 2 or predicted.shape[1] != manifest.width:
        raise Issue52ForecastError("metric arrays do not match target manifest")
    if not 1 <= start_horizon <= end_horizon <= predicted.shape[0]:
        raise Issue52ForecastError("metric horizon bounds are invalid")
    if eligibility_mask is None:
        mask = np.ones(predicted.shape, dtype=bool)
    else:
        mask = np.asarray(eligibility_mask, dtype=bool)
        if mask.shape != predicted.shape:
            raise Issue52ForecastError("metric eligibility mask does not match arrays")
    selected_mask = mask[start_horizon - 1 : end_horizon]
    selected_prediction = predicted[start_horizon - 1 : end_horizon]
    selected_actual = actual[start_horizon - 1 : end_horizon]
    if not selected_mask.any():
        raise Issue52ForecastError("metric eligibility mask contains no targets")
    if not np.isfinite(selected_prediction[selected_mask]).all() or not np.isfinite(
        selected_actual[selected_mask]
    ).all():
        raise Issue52ForecastError("eligible metric arrays contain non-finite values")
    scales = np.asarray([descriptor.scale for descriptor in manifest.descriptors], dtype=np.float64)
    errors = np.abs(selected_prediction - selected_actual) / scales[None, :]
    return float(np.mean(errors[selected_mask]))


__all__ = [
    "ActionConditionedLinearForecaster",
    "AdvisoryDecision",
    "CandidateCatalogue",
    "CandidateSchedule",
    "ForecastHistory",
    "ForecastTrajectory",
    "HistoryAppend",
    "Issue52AdvisorySource",
    "Issue52ContractError",
    "Issue52ForecastError",
    "Issue52HistoryError",
    "Issue52RolloutError",
    "ObservationRecord",
    "PersistenceForecaster",
    "RankedDecision",
    "TargetDescriptor",
    "TargetManifest",
    "TrainingSample",
    "VerifiedHistoryBuffer",
    "build_control_proposal",
    "extend_scenario_for_issue52",
    "normalized_mae",
    "rank_candidates",
    "score_trajectory",
    "target_from_snapshot",
    "targets_from_measurement",
]
